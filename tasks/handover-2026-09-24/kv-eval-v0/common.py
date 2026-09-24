"""(Copied from ornith-quant-job/validation/common.py; kv_eval.py overrides render_markdown and
routes upload_path through its batched Hub commits.)

Shared logging/upload/KL-report machinery for the Ornith-1.5-35B-A3B validation run.

Extracted and generalized (N arms, not a fixed BF16-vs-one-candidate pair) from the K2-Horizon
job's validation/validate.py, which is the harness that already worked on HF Jobs
(/tmp/.../scratchpad/k2h-job/validation/validate.py and toolcal/k2h_eval.py). The KL/top-1
methodology (truncated KL(ref||pack) over the reference's top-64 support, pooled — not
per-chunk-averaged — top-1 %, mean KL, p99 KL) is copied verbatim: it is what makes these
numbers comparable across arms and to prior art computed the same way.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F

_LOG_FILE = [None]
_ARGS = [None]


def set_log_file(path: str | None):
    _LOG_FILE[0] = path


def set_args(args):
    _ARGS[0] = args


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _LOG_FILE[0] is not None:
        with open(_LOG_FILE[0], "a") as f:
            f.write(line + "\n")


def upload_path(local_path: str, repo_rel_path: str):
    """Upload a file to validation/<repo_rel_path> in the output repo, or mirror it locally
    under --no-upload. Never raises: a failed upload must not lose local progress."""
    args = _ARGS[0]
    if args is None or getattr(args, "no_upload", True):
        out_dir = getattr(args, "out_dir", ".") if args else "."
        dest = os.path.join(out_dir, "upload_mirror", repo_rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)
        log(f"[no-upload] mirrored {local_path} -> {dest}")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{args.upload_prefix}/{repo_rel_path}",
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"validation: {repo_rel_path}",
        )
        log(f"uploaded {args.upload_prefix}/{repo_rel_path}")
    except Exception as e:
        log(f"WARNING: upload of {repo_rel_path} failed: {e!r} (local copy kept at {local_path})")


# =================================================================================================
# incremental summary (written + uploaded after every section, so a kill loses nothing done)
# =================================================================================================

SUMMARY = {"meta": {}, "sections": {}}


def save_summary(out_dir: str):
    json_path = os.path.join(out_dir, "summary.json")
    md_path = os.path.join(out_dir, "summary.md")
    with open(json_path, "w") as f:
        json.dump(SUMMARY, f, indent=1, default=str)
    with open(md_path, "w") as f:
        f.write(render_markdown(SUMMARY))
    upload_path(json_path, "summary.json")
    upload_path(md_path, "summary.md")


def render_markdown(summary: dict) -> str:
    lines = ["# Ornith-1.5-35B-A3B quant validation: full-precision vs EXL3 / NVFP4 / Q6_K", ""]
    meta = summary.get("meta", {})
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    for name, sec in summary.get("sections", {}).items():
        lines.append(f"## {name}")
        lines.append("")
        if isinstance(sec, dict) and "by_arm" in sec:
            lines.append("| arm | top-1 % | mean KL | p99 KL | ppl | n positions |")
            lines.append("|---|---|---|---|---|---|")
            for arm, r in sec["by_arm"].items():
                if not isinstance(r, dict) or "error" in r:
                    lines.append(f"| {arm} | - | - | - | - | ERROR: {r.get('error') if isinstance(r, dict) else r} |")
                    continue
                lines.append(f"| {arm} | {r.get('top1_pct', float('nan')):.2f} | {r.get('kl_mean', float('nan')):.4e} | "
                              f"{r.get('kl_p99', float('nan')):.4e} | {r.get('ppl', float('nan')):.4f} | {r.get('n_positions','-')} |")
            lines.append("")
        if isinstance(sec, dict) and "sizes" in sec:
            for arm, disk in sec.get("sizes", {}).items():
                lines.append(f"- **{arm}**: {disk}")
            lines.append("")
        lines.append("```json")
        lines.append(json.dumps(sec, indent=1, default=str)[:20000])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def record_section(name: str, data: dict, out_dir: str):
    SUMMARY["sections"][name] = data
    save_summary(out_dir)
    log(f"section '{name}' recorded and summary saved/uploaded")


# =================================================================================================
# shared prompt / token-id construction
# =================================================================================================

def get_perplexity_chunks(tokenizer, n_chunks: int, chunk_len: int):
    """wikitext-2 test, tokenized, split into n_chunks non-overlapping windows of chunk_len
    tokens each (the SAME chunks are reused for every arm, computed once from the tokenizer,
    which is shared across arms since all 4 arms are the same architecture/vocab)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    ids = tokenizer(text, add_special_tokens=False).input_ids
    need = n_chunks * chunk_len
    if len(ids) < need:
        reps = need // len(ids) + 1
        ids = ids * reps
    ids = ids[:need]
    return [ids[i * chunk_len:(i + 1) * chunk_len] for i in range(n_chunks)]


def render_chat_prompt(tokenizer, chat_dict: dict) -> str:
    messages = chat_dict["messages"]
    kwargs = dict(chat_dict.get("chat_template_kwargs", {}))
    tools = chat_dict.get("tools")
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs,
    )


# =================================================================================================
# truncated KL(ref || pack) over the reference's top-64 support + pooled top-1 / KL / ppl
# =================================================================================================

def kl_top1(ref_vals, ref_idx, ref_lse, pack_logits):
    ref_logp = ref_vals.float() - ref_lse[:, None]
    ref_p = ref_logp.exp()
    q_at_idx = pack_logits.gather(-1, ref_idx.long())
    q_lse = torch.logsumexp(pack_logits, dim=-1)
    pack_logp_at_idx = q_at_idx - q_lse[:, None]
    kl = (ref_p * (ref_logp - pack_logp_at_idx)).sum(-1)
    top1 = (pack_logits.argmax(-1) == ref_idx[:, 0]).float()
    return kl, top1


def wikitext_kl_report(logits_pass_fn, ppl_chunks: list[list[int]], ref_top64: list[dict]) -> dict:
    """Runs `logits_pass_fn(token_id_chunk) -> full-vocab fp32 logits [T,V]` over every wikitext-2
    chunk and reports top-1 %, mean KL and p99 KL POOLED over every position across every chunk
    (matching prior art's methodology: top-1%, mean/p99 KL(ref||pack) over >=10,240 positions).
    Also reports the pack's own next-token perplexity over the same chunks."""
    all_kl, all_top1 = [], []
    per_chunk = []
    nll_sum, nll_count = 0.0, 0
    for ci, chunk in enumerate(ppl_chunks):
        pack_logits = logits_pass_fn(chunk)
        top64 = ref_top64[ci]
        kl, top1 = kl_top1(top64["top64_vals"], top64["top64_idx"], top64["logsumexp"], pack_logits)
        all_kl.append(kl)
        all_top1.append(top1)
        per_chunk.append({"kl_mean": kl.mean().item(), "kl_p99": kl.quantile(0.99).item(), "top1": top1.mean().item()})
        lse = torch.logsumexp(pack_logits, dim=-1)
        logp = pack_logits[:-1] - lse[:-1, None]
        tgt = torch.tensor(chunk[1:], dtype=torch.long)
        nll = -logp.gather(-1, tgt[:, None]).squeeze(-1)
        nll_sum += nll.sum().item()
        nll_count += nll.numel()
        log(f"  ppl chunk {ci+1}: KL mean {kl.mean().item():.4e} p99 {kl.quantile(0.99).item():.4e} top1 {top1.mean().item()*100:.2f}%")
    pooled_kl = torch.cat(all_kl)
    pooled_top1 = torch.cat(all_top1)
    n_positions = pooled_kl.numel()
    report = {
        "n_positions": n_positions,
        "top1_pct": pooled_top1.mean().item() * 100.0,
        "kl_mean": pooled_kl.mean().item(),
        "kl_p99": pooled_kl.quantile(0.99).item(),
        "ppl": math.exp(nll_sum / max(nll_count, 1)),
        "per_chunk": per_chunk,
    }
    log(f"  pooled over {n_positions} positions: top-1 {report['top1_pct']:.2f}%  "
        f"mean KL {report['kl_mean']:.4f}  p99 KL {report['kl_p99']:.4f}  ppl {report['ppl']:.4f}")
    if n_positions < 10240:
        log(f"  WARNING: only {n_positions} positions (<10,240) — not directly comparable to a "
            f"result computed over the full corpus. Use full (non --small) mode for that.")
    return report


def chat_prompt_kl_report(logits_pass_fn, tokenizer, chat_prompts, ref_top64_by_id: dict) -> dict:
    """Same KL/top-1 methodology as wikitext_kl_report but over the (short) teacher-forced
    prompt token streams from validation/prompts.py's CHAT_PROMPTS, categorized (code / math /
    reasoning / tool_call) so a candidate that is fine on prose but bad on tool-call formatting
    shows up distinctly rather than being averaged away."""
    by_id = {}
    for cp in chat_prompts:
        ref = ref_top64_by_id[cp["id"]]
        ids = ref["ids"]
        pack_logits = logits_pass_fn(ids.tolist())
        kl, top1 = kl_top1(ref["top64_vals"], ref["top64_idx"], ref["logsumexp"], pack_logits)
        by_id[cp["id"]] = {
            "category": cp["category"],
            "prompt_tokens": len(ids),
            "top1_pct": top1.mean().item() * 100.0,
            "kl_mean": kl.mean().item(),
            "kl_p99": kl.quantile(0.99).item(),
        }
    return by_id
