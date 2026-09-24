#!/usr/bin/env python3
"""Ornith-1.5-35B-A3B: what FreeToken's KV-cache formats cost, per lane and per context position.

BF16 weights (ornith-ai/Ornith-1.5-35B-A3B), bf16 activations (FreeToken's compute dtype),
resident on one 96 GB GPU (~69 GB of weights), run through transformers 5.15.1's own
Qwen3.5-MoE modules, driven here chunk by chunk as FreeToken's scheduler drives them.

WHAT FreeToken quantizes, and WHEN (FreeToken-wt/reorg 33dc23e; file:line)
---------------------------------------------------------------------------
* Only the full-attention layers' K and V (10 of 40 layers). models/qwen3_5_moe/attention.py
  _project (70-84): q/k get their per-head RMSNorm (81-82) and then partial NeoX/mRoPE (83);
  v is the raw v_proj output (80). forward (91-95) hands (q, k, v) to the attention backend.
  => K is quantized AFTER k_norm AND AFTER RoPE; V after v_proj. Emulated at the same point.
* Quantized KV requires the triton backend (engine/engine.py:311-317). TritonAttentionBackend.
  forward (attention/triton.py:214 ``self.kvcache.store_kv(k, v, batch.out_loc, layer_id)``)
  quantizes the new tokens at WRITE time (kvcache/mha_pool.py:569 store_kv ->
  kernel/triton/kv_quant.py store kernel), BEFORE attention runs.
* Prefill/extend (triton.py:269-286): ``extend_paged_attention(..., k_extend=k, v_extend=v)``
  -> ``_extend_attention_split_kernel`` (kernel/triton/attention.py:1481): the PREFIX (tokens
  of earlier chunks / cached requests) is read from the quantized pool through ``_load_kv``
  (1568-1625, dequantized to the query dtype), but the CURRENT chunk's own keys/values are
  read from ``k_extend``/``v_extend`` = the fresh unquantized bf16 tensors (1634-1680).
  => within one prefill chunk, attention to same-chunk tokens is exact bf16; to all earlier
  tokens it is quantize->dequantize(->bf16). Chunk = --max-prefill-length (8192 in the
  Ornith and serve-default.sh launch lines).
* Decode (triton.py:241-263 ``decode_paged_attention``): everything, including the token
  just stored, is read back from the quantized pool.
* Dequant arithmetic (_load_kv, attention.py:267-497): (code as fp32) * (fp16 scale as fp32),
  cast to the query dtype (bf16). kvq.fake_quant reproduces it bit-exactly (dryrun/01).
* Linear-attention (GatedDeltaNet) state is NOT touched by --kv-cache-dtype: it lives in
  kvcache/linear_state_pool.py with the recurrent state in fp32 (ssm_state_dtype, 18-19, 89-93)
  and the conv state in the model dtype (79-86). Left untouched here (HF GDN, fp32 state).

EMULATION (per full-attention layer, per chunk [s, e)):  cache[s:e] <- exact k, v (prefill)
or fake_quant(k, v) (decode); attention of the chunk's queries over cache[0:e] with
bottom-right-aligned causal masking; then cache[s:e] <- fake_quant(k, v). Cache prefix
[0, s) always holds fake-quantized values. Two semantics are measured on the long documents:
  prefill  = FreeToken processing the document as ONE prompt in 8192-token chunks (the served
             prompt numerics, exactly);
  decode   = every position computed as a decode step (all keys, own included, quantized) --
             what the tokens FreeToken GENERATES see; the conservative bound.
Needles use the exact served sequence: prompt prefilled in 8192-chunks from position 0 (the
haystack is a whole number of chunks, so the question starts a fresh chunk exactly as it would
when FreeToken chunks the full prompt), then the answer tokens teacher-forced as decode steps.

EFFICIENCY: each lane runs every document ONCE per semantics. The reference (bf16 KV) keeps
only its final-normed hidden states (bf16, L x 2048: 1 GB at 256K); every lane's KL is computed
EXACTLY over the full 248,320-token vocab by re-applying lm_head to them block by block (the
same bf16 matmul the model does), so no logits are stored. Layers 0..2 (GatedDeltaNet, before
the first full-attention layer) are identical for every lane; their output is stored by the
reference pass and the lanes start at layer 3 (~7% saved). Running lanes side by side inside
one pass was rejected: the lanes diverge from layer 3 on, so it would need every lane's KV cache
resident at once (9 x 5.4 GB at 256K next to 69 GB of weights) for no compute saving (the
weights are resident, there is no weight streaming to amortize).

Stages: smoke (tiny slice of everything, throughput) | all (docs-prefill, needles, yarn,
docs-decode, report) | report. See job.sh.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

import common
from common import log, set_log_file, set_args, SUMMARY
import kvq
from needles_kv import build as build_haystack, TF_QUESTIONS

torch.set_grad_enabled(False)

# ---- batched uploads: one Hub commit per flush (<= one per FLUSH_S seconds, plus forced
# flushes at the end / on failure) instead of one commit per file per section.
_QUEUE: dict = {}
_LAST_FLUSH = [0.0]
FLUSH_S = 300
_mirror_upload = common.upload_path


def upload_path(local_path, repo_rel_path):
    _QUEUE[repo_rel_path] = local_path


def record_section(name, data, out_dir):
    common.record_section(name, data, out_dir)
    flush()


def flush(force=False):
    a = common._ARGS[0]
    if not _QUEUE:
        return
    if a is None or getattr(a, "no_upload", True) or not getattr(a, "repo", None):
        for rel, loc in list(_QUEUE.items()):
            _mirror_upload(loc, rel)
        _QUEUE.clear()
        return
    if not force and time.time() - _LAST_FLUSH[0] < FLUSH_S:
        return
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        items = dict(_QUEUE)
        logf = os.path.join(a.out_dir, "kv_eval.log")
        if os.path.exists(logf):
            items["kv_eval.log"] = logf
        ops = [CommitOperationAdd(path_in_repo=f"{a.upload_prefix}/{rel}", path_or_fileobj=loc)
               for rel, loc in items.items() if os.path.exists(loc)]
        HfApi().create_commit(repo_id=a.repo, repo_type="model", operations=ops,
                              commit_message=f"kv-validation: {len(ops)} files")
        log(f"uploaded {len(ops)} files to {a.repo}/{a.upload_prefix}/")
        _QUEUE.clear()
        _LAST_FLUSH[0] = time.time()
    except Exception as e:  # never lose local progress over an upload
        log(f"WARNING: upload failed ({e!r}); will retry at the next flush")


common.upload_path = upload_path

KI = 1024
PG19_LONG = "The Ragged Trousered Philanthropists by Robert Tressell"
PG19_SHORT = ["Carmen Ariza by Charles Francis Stocking", "Coningsby by Benjamin Disraeli",
              "The Crisis Complete by Winston Churchill"]


# =================================================================================================
# model
# =================================================================================================

def load_model(model_dir, device, dtype):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, device_map=None if device == "cpu" else device, attn_implementation="sdpa",
        output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if not any(t in k for t in ("visual", "vision", "mtp"))]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} language-model weights missing, e.g. {missing[:5]}")
    model.eval()
    inner = model.model.language_model if hasattr(model.model, "language_model") else model.model
    log(f"loaded {type(model).__name__} from {model_dir} in {time.time()-t0:.0f}s: "
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params, dtype {dtype}, device {device}, "
        f"experts impl {getattr(inner.config, '_experts_implementation', None)}")
    return model, inner


class GDNCache:
    """The slice of transformers' Cache API that Qwen3_5MoeGatedDeltaNet.forward uses
    (modeling_qwen3_5_moe.py:441-532), backed by transformers' own LinearAttentionLayer so the
    conv/recurrent state handling across chunks is the library's own."""

    def __init__(self, layer_ids):
        from transformers.cache_utils import LinearAttentionLayer
        self.layers = {i: LinearAttentionLayer() for i in layer_ids}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return bool(self.layers[layer_idx].has_previous_state[0])

    def update_conv_state(self, conv_states, layer_idx, **kw):
        return self.layers[layer_idx].update_conv_state(conv_states, 0, **kw)

    def update_recurrent_state(self, recurrent_states, layer_idx, **kw):
        return self.layers[layer_idx].update_recurrent_state(recurrent_states, 0)

    def snapshot(self):
        return {i: (l.conv_states[0].clone() if l.conv_states[0] is not None else None,
                    l.recurrent_states[0].clone() if l.recurrent_states[0] is not None else None,
                    l.has_previous_state[0]) for i, l in self.layers.items()}

    def restore(self, snap):
        for i, (c, r, h) in snap.items():
            l = self.layers[i]
            if c is not None:
                l.conv_states[0].copy_(c)
            if r is not None:
                l.recurrent_states[0].copy_(r)
            l.has_previous_state[0] = h


def attend_lower_right(q, K, V, e):
    """Causal attention of the chunk's queries q [1, Hq, C, D] (absolute positions e-C..e-1)
    over keys/values K, V [Hkv, >=e, D] positions 0..e-1, bottom-right aligned (query i sees
    keys <= e-C+i) -- the extend kernel's mask. SDPA with torch's CausalBias LOWER_RIGHT
    (flash / mem-efficient kernels, no materialized mask on GPU); GQA by repeating each KV head
    for its query group (group by group to bound memory)."""
    from torch.nn.attention.bias import causal_lower_right
    Hq, C = q.shape[1], q.shape[2]
    Hkv, D = K.shape[0], K.shape[-1]
    rep = Hq // Hkv
    bias = causal_lower_right(C, e)
    outs = []
    for g in range(Hkv):
        kk = K[g, :e].unsqueeze(0).unsqueeze(0).expand(1, rep, e, D).contiguous()
        vv = V[g, :e].unsqueeze(0).unsqueeze(0).expand(1, rep, e, D).contiguous()
        outs.append(F.scaled_dot_product_attention(q[:, g * rep:(g + 1) * rep], kk, vv, attn_mask=bias))
        del kk, vv
    return torch.cat(outs, dim=1)


class LaneState:
    def __init__(self, eng, lane, lmax, rotary):
        kname, vname, _ = kvq.LANES[lane] if lane in kvq.LANES else (lane.split("/")[0], lane.split("/")[1], "")
        self.lane = lane
        self.kspec, self.vspec = kvq.BY_NAME[kname], kvq.BY_NAME[vname]
        Hkv, D = eng.cfg.num_key_value_heads, eng.cfg.head_dim
        nf = len(eng.full_ids)
        self.K = torch.empty((nf, Hkv, lmax, D), dtype=eng.dtype, device=eng.device)
        self.V = torch.empty_like(self.K)
        self.gdn = GDNCache(eng.linear_ids)
        self.rotary = rotary
        self.err = torch.zeros((nf, 4), dtype=torch.float64)  # |kq-k|^2, |k|^2, |vq-v|^2, |v|^2


class Engine:
    def __init__(self, model, inner, device, dtype):
        self.model, self.inner, self.device, self.dtype = model, inner, device, dtype
        self.cfg = inner.config
        self.W = model.lm_head.weight
        types = self.cfg.layer_types
        self.full_ids = [i for i, t in enumerate(types) if t == "full_attention"]
        self.linear_ids = [i for i, t in enumerate(types) if t == "linear_attention"]
        self.slot = {li: j for j, li in enumerate(self.full_ids)}
        self.first_full = self.full_ids[0]
        self.rotary_default = inner.rotary_emb

    def make_rotary(self, yarn_factor=None, orig=None):
        if not yarn_factor:
            return self.rotary_default
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        cfg = copy.deepcopy(self.cfg)
        rp = dict(cfg.rope_parameters)
        orig = orig or cfg.max_position_embeddings
        # The override FreeToken's --rope-yarn-factor applies (engine/config.py:280-303), in the
        # form transformers takes (FreeToken's own parity test builds the HF side the same way:
        # tests/layers/test_mrope_yarn_engine_path.py:31-45).
        rp.update(rope_type="yarn", factor=float(yarn_factor), original_max_position_embeddings=int(orig))
        cfg.rope_parameters = rp
        cfg.max_position_embeddings = int(round(orig * yarn_factor))
        rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(self.device)
        log(f"yarn rotary: factor {yarn_factor} over {orig}: attention_scaling {rot.attention_scaling:.6f} "
            f"(0.1*ln(f)+1 = {0.1*math.log(yarn_factor)+1:.6f})")
        return rot

    def _full_attn(self, layer, x, pe, st, li, s, e, decode):
        a = layer.self_attn
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb
        B, C, _ = x.shape
        D = a.head_dim
        hs = (B, C, -1, D)
        qx, gate = torch.chunk(a.q_proj(x).view(B, C, -1, D * 2), 2, dim=-1)
        gate = gate.reshape(B, C, -1)
        q = a.q_norm(qx.view(hs)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(hs)).transpose(1, 2)
        v = a.v_proj(x).view(hs).transpose(1, 2)
        cos, sin = pe
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = k[0], v[0]                                   # [Hkv, C, D]
        kq, vq = kvq.fake_quant(st.kspec, k), kvq.fake_quant(st.vspec, v)
        j = self.slot[li]
        if st.kspec.enabled or st.vspec.enabled:
            kd, vd = k.float(), v.float()
            st.err[j, 0] += (kq.float() - kd).pow(2).sum().item(); st.err[j, 1] += kd.pow(2).sum().item()
            st.err[j, 2] += (vq.float() - vd).pow(2).sum().item(); st.err[j, 3] += vd.pow(2).sum().item()
        Kc, Vc = st.K[j], st.V[j]
        Kc[:, s:e] = kq if decode else k                    # extend: own chunk exact (k_extend)
        Vc[:, s:e] = vq if decode else v
        o = attend_lower_right(q, Kc, Vc, e)
        Kc[:, s:e] = kq                                     # what the pool holds from now on
        Vc[:, s:e] = vq
        o = o.transpose(1, 2).reshape(B, C, -1)
        o = o * torch.sigmoid(gate)
        return a.o_proj(o)

    def run_chunk(self, st, s, e, *, ids=None, h=None, start_layer=0, decode=False, h3_out=None):
        """One chunk [s, e) through layers start_layer..N-1 (mirrors Qwen3_5MoeDecoderLayer.
        forward, modeling_qwen3_5_moe.py:834-877). Returns final-normed hidden [e-s, H]."""
        dev = self.device
        if start_layer == 0:
            h = self.inner.embed_tokens(ids.view(1, -1).to(dev))
        else:
            h = h.view(1, e - s, -1).to(dev, self.dtype)
        C = e - s
        pos = torch.arange(s, e, device=dev).view(1, 1, C).expand(3, 1, C)
        pe = st.rotary(h, pos)
        for li in range(start_layer, len(self.inner.layers)):
            if li == self.first_full and h3_out is not None:
                h3_out[s:e] = h[0].to("cpu")
            layer = self.inner.layers[li]
            res = h
            x = layer.input_layernorm(h)
            if layer.block_type == "linear_attention":
                x = layer.linear_attn(hidden_states=x, cache_params=st.gdn, attention_mask=None)
            else:
                x = self._full_attn(layer, x, pe, st, li, s, e, decode)
            h = res + x
            res = h
            x = layer.mlp(layer.post_attention_layernorm(h))
            if isinstance(x, tuple):
                x = x[0]
            h = res + x
        return self.inner.norm(h)[0]

    def run_seq(self, st, ids, s0, chunk, *, h3=None, h3_out=None, decode=False, collect=None, from_h3=False):
        """Positions s0..s0+len(ids)-1 in FreeToken prefill chunks: absolute boundaries at
        multiples of ``chunk`` (the scheduler chunks the prompt from position 0). ``collect``
        receives (s, e, hidden) per chunk."""
        L = len(ids)
        s = s0
        while s < s0 + L:
            e = min((s // chunk + 1) * chunk, s0 + L)
            if from_h3:
                hid = self.run_chunk(st, s, e, h=h3[s - s0:e - s0], start_layer=self.first_full, decode=decode)
            else:
                cap = None
                if h3_out is not None:
                    cap = _Offset(h3_out, s0)
                hid = self.run_chunk(st, s, e, ids=ids[s - s0:e - s0], decode=decode, h3_out=cap)
            if collect is not None:
                collect(s, e, hid)
            s = e


class _Offset:
    """h3_out view addressed by absolute positions."""
    def __init__(self, t, off):
        self.t, self.off = t, off

    def __setitem__(self, sl, val):
        self.t[sl.start - self.off:sl.stop - self.off] = val


def logits_of(eng, hid):
    return F.linear(hid.to(eng.device, eng.dtype), eng.W).float()


def compare_block(eng, h_ref, h_lane, tgt):
    """Exact full-vocab KL(ref||lane), top-1 agreement, both NLLs of the true next token."""
    lr = torch.log_softmax(logits_of(eng, h_ref), -1)
    ll = torch.log_softmax(logits_of(eng, h_lane), -1)
    kl = (lr.exp() * (lr - ll)).sum(-1)
    agree = lr.argmax(-1) == ll.argmax(-1)
    t = tgt.to(eng.device)
    valid = t >= 0
    tt = t.clamp(min=0)[:, None]
    nll_l = torch.where(valid, -ll.gather(-1, tt)[:, 0], torch.nan)
    nll_r = torch.where(valid, -lr.gather(-1, tt)[:, 0], torch.nan)
    return kl.cpu(), agree.cpu(), nll_l.cpu(), nll_r.cpu()


# =================================================================================================
# data
# =================================================================================================

def load_docs(args, tok):
    """-> list of (name, token ids). Deterministic: pinned PG19-test titles (fallbacks by length),
    first N tokens of each."""
    need = [("long", args.long_len)] + [(f"short{i}", args.short_len) for i in range(args.n_short)]
    texts = {}
    if args.doc_files:
        files = args.doc_files.split(",")
        texts = {f"file:{os.path.basename(f)}": open(f).read() for f in files}
        order = list(texts)
        src = "local files (dry run)"
    else:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            import pyarrow.parquet as pq
            repo = "emozilla/pg19-test"
            files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                           if f.startswith("data/test") and f.endswith(".parquet"))
            d = {}
            for f in files:
                t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pydict()
                d.update(zip(t["short_book_title"], t["text"]))
            texts = d
            by_len = sorted(d, key=lambda k: -len(d[k]))
            order = [PG19_LONG] + PG19_SHORT + [k for k in by_len if k != PG19_LONG and k not in PG19_SHORT]
            order = [k for k in order if k in d]
            src = f"PG19 test ({repo}, {len(d)} books)"
        except Exception as ex:  # fall back to one long wikitext-103 stream
            log(f"WARNING: PG19 unavailable ({ex!r}); falling back to wikitext-103-raw-v1 train")
            from huggingface_hub import hf_hub_download, HfApi
            import pyarrow.parquet as pq
            repo = "Salesforce/wikitext"
            fs = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                        if f.startswith("wikitext-103-raw-v1/train"))
            rows = pq.read_table(hf_hub_download(repo, fs[0], repo_type="dataset")).to_pydict()["text"]
            text = "".join(rows)
            per = len(text) // (1 + args.n_short)
            texts = {f"wikitext103_part{i}": text[i * per:(i + 1) * per] for i in range(1 + args.n_short)}
            order = list(texts)
            src = "wikitext-103-raw-v1 train (concatenated articles; PG19 fallback)"
    docs, used = [], set()
    for role, n in need:
        for title in order:
            if title in used:
                continue
            ids = tok(texts[title], add_special_tokens=False).input_ids
            if len(ids) >= n:
                docs.append((role, title, ids[:n]))
                used.add(title)
                break
        else:
            raise SystemExit(f"no document with >= {n} tokens for {role}")
    SUMMARY["meta"]["documents"] = {"source": src, **{r: f"{t} (first {len(i)} tokens)" for r, t, i in docs}}
    for r, t, i in docs:
        log(f"doc {r}: {t!r}, {len(i)} tokens")
    return docs


def chat_parts(tok):
    """(head, tail) of the chat template around a single user message, thinking off."""
    SPLIT = "⁣SPLIT⁣"
    s = tok.apply_chat_template([{"role": "user", "content": SPLIT}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    a, b = s.split(SPLIT)
    return a, b


def build_needle(tok, n_pre, seed, chunk):
    """Token ids: prefix = chat head + haystack, trimmed to exactly n_pre (a multiple of the
    chunk); per question: question+tail ids and answer ids."""
    assert n_pre % chunk == 0
    head, tail = chat_parts(tok)
    # Size the haystack so that at most ~1% of it is trimmed off the end (the deepest needle
    # sits at 97%): tokens = overhead (template + planted sentences) + rate * filler budget.
    ntok = lambda g: len(tok(head + build_haystack(g, random.Random(seed))[0], add_special_tokens=False).input_ids)
    overhead = ntok(0)
    want_lo, want_hi = n_pre, n_pre + max(4, n_pre // 200)          # trim <= 0.5%
    target = (want_lo + want_hi) // 2
    guess = max(64, n_pre - overhead)
    for _ in range(25):
        n1 = ntok(guess)
        if want_lo <= n1 <= want_hi:
            break
        rate = max((n1 - overhead) / guess, 1e-3)          # filler tokens per unit of guess
        guess = max(1, guess + int(round((target - n1) / rate)))
    hay, truth = build_haystack(guess, random.Random(seed))
    ids = tok(head + hay, add_special_tokens=False).input_ids
    if len(ids) < n_pre:
        raise SystemExit(f"haystack sizing failed: {len(ids)} < {n_pre}")
    ids = ids[:n_pre]
    kept = tok.decode(ids)
    for c in truth["codes"]:
        assert c in kept, f"needle {c} trimmed away"
    qs = {}
    for name, (question, ans) in TF_QUESTIONS.items():
        q_ids = tok("\n\n" + question + tail, add_special_tokens=False).input_ids
        a_ids = tok(ans(truth), add_special_tokens=False).input_ids
        qs[name] = (q_ids, a_ids, ans(truth))
    return ids, qs, truth


# =================================================================================================
# runner
# =================================================================================================

class Runner:
    def __init__(self, args, eng, tok):
        self.a, self.eng, self.tok = args, eng, tok
        self.t_start = time.time()
        self.href, self.h3 = {}, {}
        self.skipped = []
        self.raw_dir = os.path.join(args.out_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        edges = [int(x) for x in args.bucket_edges.split(",")]
        self.buckets = list(zip(edges[:-1], edges[1:]))

    def minutes(self):
        return (time.time() - self.t_start) / 60

    def budget_ok(self, what):
        if self.a.budget_min and self.minutes() > self.a.budget_min:
            self.skipped.append(what)
            log(f"BUDGET: {self.minutes():.1f} min > {self.a.budget_min} -- skipping {what}")
            SUMMARY["meta"]["skipped_for_budget"] = self.skipped
            return False
        return True

    def bname(self, lo, hi):
        f = lambda x: f"{x//KI}K" if x >= KI else str(x)
        return f"{f(lo)}-{f(hi)}"

    def bucketize(self, per_doc, max_pos=None):
        """per_doc: list of dicts with kl, agree, nll_l, nll_r (position-indexed)."""
        out = {}
        for lo, hi in self.buckets + [(0, 1 << 40)]:
            kl, ag, nl, nr = [], [], [], []
            for d in per_doc:
                sl = slice(lo, min(hi, len(d["kl"])))
                if sl.start >= sl.stop:
                    continue
                kl.append(d["kl"][sl]); ag.append(d["agree"][sl]); nl.append(d["nll_l"][sl]); nr.append(d["nll_r"][sl])
            if not kl:
                continue
            kl = torch.cat(kl).double(); ag = torch.cat(ag).double()
            nl = torch.cat(nl).double(); nr = torch.cat(nr).double()
            m = ~nl.isnan()
            key = "all" if hi == 1 << 40 else self.bname(lo, hi)
            q = torch.quantile(kl.float(), torch.tensor([0.5, 0.9, 0.99, 0.999])).tolist() if kl.numel() < 16_000_000 else [float("nan")] * 4
            out[key] = {
                "n": int(kl.numel()), "kl_mean": kl.mean().item(), "kl_p50": q[0], "kl_p90": q[1],
                "kl_p99": q[2], "kl_p999": q[3], "kl_max": kl.max().item(),
                "frac_kl_gt_0.1": (kl > 0.1).double().mean().item(),
                "top1_agree_pct": 100 * ag.mean().item(),
                "ppl_lane": math.exp(nl[m].mean().item()) if m.any() else float("nan"),
                "ppl_ref": math.exp(nr[m].mean().item()) if m.any() else float("nan"),
            }
            r = out[key]
            r["dppl_pct"] = 100 * (r["ppl_lane"] / r["ppl_ref"] - 1)
        return out

    # ---------------------------------------------------------------- documents
    def doc_ref(self, docs, rotary):
        eng, a = self.eng, self.a
        for role, title, ids in docs:
            L = len(ids)
            href = torch.empty((L, eng.cfg.hidden_size), dtype=eng.dtype)
            h3 = torch.empty((L, eng.cfg.hidden_size), dtype=eng.dtype)
            st = LaneState(eng, "bf16", L, rotary)
            t0 = time.time()

            def put(s, e, hid):
                href[s:e] = hid.to("cpu")
            eng.run_seq(st, torch.tensor(ids), 0, a.chunk, h3_out=h3, collect=put)
            del st
            self.href[role], self.h3[role] = href, h3
            dt = time.time() - t0
            log(f"  ref {role}: {L} tokens in {dt:.1f}s ({L/dt:.0f} tok/s, all {len(eng.inner.layers)} layers)")
            SUMMARY["meta"].setdefault("throughput", {})[f"ref_{role}"] = f"{L/dt:.0f} tok/s"
            self._free()

    def doc_lane(self, docs, lane, mode, rotary=None, max_len=None, tag=None):
        eng, a = self.eng, self.a
        rotary = rotary or eng.rotary_default
        per_doc, t0, ntok = [], time.time(), 0
        err = None
        for role, title, ids in docs:
            L = min(len(ids), max_len or len(ids))
            ids_t = torch.tensor(ids[:L])
            tgt = torch.cat([ids_t[1:], torch.tensor([-1])])
            if max_len and L < len(ids):
                tgt[L - 1] = ids[L]
            st = LaneState(eng, lane, L, rotary)
            rec = {"kl": torch.empty(L), "agree": torch.empty(L, dtype=torch.bool),
                   "nll_l": torch.empty(L), "nll_r": torch.empty(L)}
            href = self.href[role]

            def put(s, e, hid):
                for b0 in range(s, e, a.logit_block):
                    b1 = min(b0 + a.logit_block, e)
                    kl, ag, nl, nr = compare_block(eng, href[b0:b1], hid[b0 - s:b1 - s], tgt[b0:b1])
                    rec["kl"][b0:b1], rec["agree"][b0:b1], rec["nll_l"][b0:b1], rec["nll_r"][b0:b1] = kl, ag, nl, nr
            eng.run_seq(st, ids_t, 0, a.chunk, h3=self.h3[role][:L], from_h3=True,
                        decode=(mode == "decode"), collect=put)
            err = st.err if err is None else err + st.err
            del st
            self._free()
            per_doc.append(rec)
            ntok += L
        dt = time.time() - t0
        res = self.bucketize(per_doc)
        res["seconds"] = round(dt, 1)
        res["tok_per_s"] = round(ntok / dt)
        if err is not None and err[:, 1].sum() > 0:
            res["kv_rel_rmse_per_full_layer"] = {
                str(li): {"K": math.sqrt(err[j, 0] / max(err[j, 1], 1e-30)),
                          "V": math.sqrt(err[j, 2] / max(err[j, 3], 1e-30))}
                for j, li in enumerate(eng.full_ids)}
        name = tag or f"docs_{mode}"
        torch.save({"lane": lane, "mode": mode, "docs": [(r, t, len(i)) for r, t, i in docs],
                    "per_doc": [{k: (v.half() if v.dtype == torch.float32 else v) for k, v in d.items()} for d in per_doc]},
                   os.path.join(self.raw_dir, f"{name}__{lane}.pt"))
        upload_path(os.path.join(self.raw_dir, f"{name}__{lane}.pt"), f"raw/{name}__{lane}.pt")
        al = res.get("all", {})
        log(f"  {name} {lane}: KL mean {al.get('kl_mean', float('nan')):.4e} p99 {al.get('kl_p99', float('nan')):.4e} "
            f"top1 {al.get('top1_agree_pct', float('nan')):.2f}% dppl {al.get('dppl_pct', float('nan')):+.3f}% "
            f"({dt:.0f}s, {ntok/dt:.0f} tok/s)")
        return res

    # ---------------------------------------------------------------- needles
    def needle_lane(self, nd, lane, rotary=None, ref=None, keep_h3=False):
        """nd: dict(ids, qs). Returns per-question scores and (for the ref) answer logprobs."""
        eng, a = self.eng, self.a
        rotary = rotary or eng.rotary_default
        ids, qs = nd["ids"], nd["qs"]
        P = len(ids)
        lmax = P + max(len(q) + len(an) for q, an, _ in qs.values())
        st = LaneState(eng, lane, lmax, rotary)
        use_h3 = "h3" in nd and not keep_h3
        t0 = time.time()
        h3pre = None
        if not use_h3:
            h3pre = torch.empty((P, eng.cfg.hidden_size), dtype=eng.dtype)
        eng.run_seq(st, torch.tensor(ids), 0, a.chunk, h3=nd.get("h3", {}).get("pre") if use_h3 else None,
                    from_h3=use_h3, h3_out=h3pre)
        t_pre = time.time() - t0
        snap = st.gdn.snapshot()
        out, lps, h3q = {}, {}, {}
        for name, (q_ids, a_ids, ans_text) in qs.items():
            st.gdn.restore(snap)
            got = {}
            hq = None if use_h3 else torch.empty((len(q_ids), eng.cfg.hidden_size), dtype=eng.dtype)
            ha = None if use_h3 else torch.empty((max(len(a_ids) - 1, 1), eng.cfg.hidden_size), dtype=eng.dtype)

            def put_q(s, e, hid):
                if e == P + len(q_ids):
                    got["first"] = hid[-1:]
            eng.run_seq(st, torch.tensor(q_ids), P, a.chunk, h3=nd["h3"]["q"][name] if use_h3 else None,
                        from_h3=use_h3, h3_out=hq, collect=put_q)
            hids = [got["first"]]
            if len(a_ids) > 1:
                s = P + len(q_ids)
                e = s + len(a_ids) - 1
                if use_h3:
                    hid = eng.run_chunk(st, s, e, h=nd["h3"]["a"][name], start_layer=eng.first_full, decode=True)
                else:
                    hid = eng.run_chunk(st, s, e, ids=torch.tensor(a_ids[:-1]), decode=True, h3_out=_Offset(ha, s))
                hids.append(hid)
            lg = logits_of(eng, torch.cat(hids))                      # [n_ans, V]
            lp = torch.log_softmax(lg, -1)
            tgt = torch.tensor(a_ids, device=lg.device)
            tok_lp = lp.gather(-1, tgt[:, None])[:, 0]
            ranks = (lg > lg.gather(-1, tgt[:, None])).sum(-1)
            r = {"answer": ans_text, "n_tokens": len(a_ids),
                 "logprob": tok_lp.sum().item(), "token_logprobs": [round(x, 4) for x in tok_lp.tolist()],
                 "ranks": ranks.tolist(), "first_token_rank": int(ranks[0]),
                 "greedy_match": bool((ranks == 0).all()),
                 "p_answer": math.exp(tok_lp.sum().item())}
            if ref is not None:
                rlp = ref["lps"][name].to(lp.device)
                r["kl_at_answer_mean"] = (rlp.exp() * (rlp - lp)).sum(-1).mean().item()
                r["dlogprob_vs_ref"] = r["logprob"] - ref["scores"][name]["logprob"]
            out[name] = r
            lps[name] = lp.cpu()
            if not use_h3:
                h3q[name] = (hq, ha)
        del st
        self._free()
        dt = time.time() - t0
        log(f"  needle {nd['label']} {lane}: prefix {P} tok in {t_pre:.1f}s ({P/max(t_pre,1e-9):.0f} tok/s); "
            + " ".join(f"{k}={'OK' if v['greedy_match'] else 'x'}({v['logprob']:.2f})" for k, v in out.items()))
        extra = {}
        if not use_h3:
            extra = {"pre": h3pre, "q": {k: v[0] for k, v in h3q.items()}, "a": {k: v[1] for k, v in h3q.items()}}
        return {"scores": out, "lps": lps, "seconds": round(dt, 1), "prefix_tokens": P}, extra

    def _free(self):
        if self.eng.device != "cpu":
            torch.cuda.empty_cache()


# =================================================================================================
# stages
# =================================================================================================

def lanes_from(args):
    lanes = [l for l in args.lanes.split(",") if l]
    for l in lanes:
        assert l in kvq.LANES, l
    return lanes


def run_all(args, eng, tok, smoke=False):
    R = Runner(args, eng, tok)
    lanes = lanes_from(args)
    qlanes = [l for l in lanes if l != "bf16"]
    SUMMARY["meta"].update({
        "model": args.model_dir, "dtype": args.dtype, "prefill_chunk": args.chunk,
        "lanes": {l: kvq.LANES[l] for l in lanes},
        "kl": "exact KL(ref||lane) over the full vocabulary, fp32, per position; ref = same weights, bf16 KV",
        "buckets": [R.bname(lo, hi) for lo, hi in R.buckets],
    })
    docs = load_docs(args, tok)

    # ---- 1. documents, prefill semantics (FreeToken's served prompt path) ----
    log("=== docs: reference (bf16 KV) ===")
    R.doc_ref(docs, eng.rotary_default)
    sec = {}
    if args.determinism_check:
        short = [d for d in docs if d[0] != "long"] or docs
        sec["bf16_rerun(control)"] = R.doc_lane(short, "bf16", "prefill", tag="docs_prefill_control")
        record_section("docs_prefill", sec, args.out_dir)
    for lane in qlanes:
        if not R.budget_ok(f"docs_prefill {lane}"):
            break
        sec[lane] = R.doc_lane(docs, lane, "prefill")
        record_section("docs_prefill", sec, args.out_dir)

    # ---- 2. needles ----
    if args.needle_sizes:
        nsec = {}
        needle_data = {}
        for n_pre in [int(x) for x in args.needle_sizes.split(",")]:
            ids, qs, truth = build_needle(tok, n_pre, 20260922 + n_pre, args.chunk)
            nd = {"ids": ids, "qs": qs, "label": f"{n_pre}"}
            log(f"=== needle haystack {n_pre} prefix tokens (+{max(len(q) for q, _, _ in qs.values())} question) ===")
            ref, extra = R.needle_lane(nd, "bf16")
            nd["h3"] = extra
            needle_data[n_pre] = (nd, ref)
            entry = {"prompt_tokens": n_pre + max(len(q) + len(an) for q, an, _ in qs.values()),
                     "truth": truth, "bf16": ref["scores"]}
            nsec[str(n_pre)] = entry
            record_section("needles", nsec, args.out_dir)
            for lane in qlanes:
                if not R.budget_ok(f"needle {n_pre} {lane}"):
                    break
                r, _ = R.needle_lane(nd, lane, ref=ref)
                entry[lane] = r["scores"]
                record_section("needles", nsec, args.out_dir)
            nd.pop("h3", None) if n_pre != min(int(x) for x in args.needle_sizes.split(",")) else None
        R.needle_data = needle_data

    # ---- 3. YaRN (bf16 KV): factor vs none at everyday lengths; one needle past 262K ----
    if args.yarn_factor:
        ysec = {}
        rot = eng.make_rotary(args.yarn_factor, args.yarn_orig)
        if R.budget_ok("yarn short docs"):
            ysec[f"docs_first_{args.yarn_short_len}_tokens"] = R.doc_lane(
                docs, "bf16", "prefill", rotary=rot, max_len=args.yarn_short_len, tag="yarn_docs")
            ysec["meta"] = {"factor": args.yarn_factor, "original_max_position_embeddings": args.yarn_orig,
                            "reference": "same model, same bf16 KV, no rope scaling (the checkpoint's own)"}
            record_section("yarn", ysec, args.out_dir)
        small_needle = min(int(x) for x in args.needle_sizes.split(",")) if args.needle_sizes else None
        if small_needle and R.budget_ok("yarn needle small"):
            nd, ref = R.needle_data[small_needle]
            nd["label"] = f"{small_needle}-yarn"
            r, _ = R.needle_lane(nd, "bf16", rotary=rot, ref=ref)
            ysec[f"needle_{small_needle}"] = {"yarn": r["scores"], "no_scaling": ref["scores"]}
            record_section("yarn", ysec, args.out_dir)
        if args.yarn_needle_size and R.budget_ok("yarn long needle"):
            ids, qs, truth = build_needle(tok, args.yarn_needle_size, 20260922 + args.yarn_needle_size, args.chunk)
            nd = {"ids": ids, "qs": qs, "label": f"{args.yarn_needle_size}-yarn"}
            ry, extra = R.needle_lane(nd, "bf16", rotary=rot)
            nd["h3"] = extra
            e = {"prompt_tokens": args.yarn_needle_size + max(len(q) + len(an) for q, an, _ in qs.values()),
                 "truth": truth, "yarn": ry["scores"]}
            ysec[f"needle_{args.yarn_needle_size}"] = e
            record_section("yarn", ysec, args.out_dir)
            if args.yarn_long_contrast and R.budget_ok("no-scaling long needle"):
                nd["label"] = f"{args.yarn_needle_size}-noscale"
                rn, _ = R.needle_lane(nd, "bf16", ref=ry)
                e["no_scaling(beyond native 262144)"] = rn["scores"]
                record_section("yarn", ysec, args.out_dir)
            del nd
        R._free()

    # ---- 4. documents, decode semantics (every position a decode step: conservative bound) ----
    if "decode" in args.doc_modes:
        dsec = {}
        for lane in qlanes:
            if not R.budget_ok(f"docs_decode {lane}"):
                break
            dsec[lane] = R.doc_lane(docs, lane, "decode")
            record_section("docs_decode", dsec, args.out_dir)

    SUMMARY["meta"]["wall_minutes"] = round(R.minutes(), 1)
    SUMMARY["meta"]["skipped_for_budget"] = R.skipped
    common.save_summary(args.out_dir)
    log(f"done in {R.minutes():.1f} min; skipped: {R.skipped or 'nothing'}")
    flush(force=True)


def probe_and_project(eng, args, full_defaults):
    """Smoke stage: time one full chunk (lanes' path, layers 3..N-1, q4_0 lane) at depth 0 and
    at the deepest position the full run reaches (the YaRN needle), check peak VRAM there, and
    project the full run's wall time from a + b*depth per chunk."""
    dev = eng.device
    C = args.chunk
    deep = max(full_defaults["yarn_needle_size"], full_defaults["long_len"]) + 128
    res = {}
    for depth in (0, deep - C):
        st = LaneState(eng, "q4_0", depth + C, eng.rotary_default)
        st.K.zero_(); st.V.zero_()
        h = torch.randn(C, eng.cfg.hidden_size, device=dev, dtype=eng.dtype)
        for _ in range(2):
            if dev != "cpu":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            eng.run_chunk(st, depth, depth + C, h=h, start_layer=eng.first_full)
            if dev != "cpu":
                torch.cuda.synchronize()
            dt = time.time() - t0
        res[depth] = dt
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev != "cpu" else float("nan")
        log(f"  probe: chunk of {C} at depth {depth}: {dt:.2f}s, peak VRAM {peak:.1f} GiB")
        SUMMARY["meta"].setdefault("probe", {})[f"depth_{depth}"] = {"seconds": round(dt, 3), "peak_vram_gib": round(peak, 2)}
        del st
        torch.cuda.empty_cache() if dev != "cpu" else None
    a = res[0]; b = (res[deep - C] - a) / max(deep - C, 1)

    def seq(L):
        return sum(a + b * (c * C + C / 2) for c in range(math.ceil(L / C)))
    f = full_defaults
    docs = [f["long_len"]] + [f["short_len"]] * f["n_short"]
    needles = f["needle_sizes"]
    nq = len(kvq.LANES) - 1
    t = seq(f["long_len"]) * 1.1 * 1 + sum(seq(x) for x in docs[1:]) * 1.1        # ref (all layers)
    t += sum(seq(x) for x in docs[1:])                                               # bf16 control
    t += nq * sum(seq(x) for x in docs) * 2                                          # prefill + decode
    t += (nq + 1) * sum(seq(x) for x in needles) * 1.05                              # needles (+ref)
    t += sum(seq(min(x, f["yarn_short_len"])) for x in docs) + seq(needles[0]) + 2 * seq(f["yarn_needle_size"])
    proj = {"per_chunk_s_at_depth0": round(a, 3), "per_chunk_s_per_1M_depth": round(b * 1e6, 3),
            "projected_main_minutes_excl_load": round(t / 60, 1)}
    log(f"  PROJECTION for --stage all with defaults: {proj}")
    SUMMARY["meta"]["projection"] = proj
    return proj


def attention_selftest(device, dtype):
    """GPU: the lower-right causal SDPA used for extend attention vs an explicit masked
    reference; also say which SDPA kernel path is taken. Refuses to run on mismatch."""
    from torch.nn.attention.bias import causal_lower_right
    torch.manual_seed(1)
    Hq, Hkv, C, e, D = 16, 2, 300, 1000, 256
    q = torch.randn(1, Hq, C, D, device=device, dtype=dtype)
    K = torch.randn(Hkv, e, D, device=device, dtype=dtype)
    V = torch.randn(Hkv, e, D, device=device, dtype=dtype)
    got = attend_lower_right(q, K, V, e).float()
    Kr = K.float().repeat_interleave(Hq // Hkv, 0)[None]; Vr = V.float().repeat_interleave(Hq // Hkv, 0)[None]
    s = (q.float() @ Kr.transpose(-1, -2)) / math.sqrt(D)
    qpos = torch.arange(e - C, e, device=device)[:, None]; kpos = torch.arange(e, device=device)[None]
    s = s.masked_fill(kpos > qpos, float("-inf"))
    want = torch.softmax(s, -1) @ Vr
    d = (got - want).abs().max().item()
    info = {"max_abs_diff_vs_fp32_reference": d}
    if device != "cpu":
        from torch.backends.cuda import SDPAParams, can_use_flash_attention, can_use_efficient_attention
        kk = K[0:1, :e].unsqueeze(0).expand(1, 8, e, D).contiguous()
        p = SDPAParams(q[:, :8], kk, kk, None, 0.0, False, False)
        info.update(flash=bool(can_use_flash_attention(p)), mem_efficient=bool(can_use_efficient_attention(p)),
                    device=torch.cuda.get_device_name(), capability=str(torch.cuda.get_device_capability()))
    log(f"attention self-test: {info}")
    if device != "cpu" and not (info.get("flash") or info.get("mem_efficient")):
        raise SystemExit("no fused SDPA kernel for lower-right causal: would materialize masks at 256K; refusing")
    if d > (2e-2 if dtype != torch.float32 else 1e-4):
        raise SystemExit(f"lower-right causal attention mismatch {d}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "all", "report"])
    ap.add_argument("--model-dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lanes", default=",".join(kvq.LANES))
    ap.add_argument("--chunk", type=int, default=8192, help="FreeToken --max-prefill-length")
    ap.add_argument("--long-len", type=int, default=256 * KI)
    ap.add_argument("--short-len", type=int, default=32 * KI)
    ap.add_argument("--n-short", type=int, default=3)
    ap.add_argument("--doc-files", default="", help="dry run: local text files instead of PG19")
    ap.add_argument("--bucket-edges", default=",".join(str(x * KI) for x in (0, 8, 32, 64, 128, 256)))
    ap.add_argument("--doc-modes", default="prefill,decode")
    ap.add_argument("--determinism-check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--needle-sizes", default=f"{4*8192},{16*8192},{31*8192}")
    ap.add_argument("--yarn-factor", type=float, default=1.5)
    ap.add_argument("--yarn-orig", type=int, default=262144)
    ap.add_argument("--yarn-short-len", type=int, default=32 * KI)
    ap.add_argument("--yarn-needle-size", type=int, default=43 * 8192)
    ap.add_argument("--yarn-long-contrast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--logit-block", type=int, default=1024)
    ap.add_argument("--budget-min", type=float, default=0.0, help="soft wall budget; later work is skipped")
    ap.add_argument("--out-dir", default="./kv-validation")
    ap.add_argument("--repo")
    ap.add_argument("--upload-prefix", default="kv-validation")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_log_file(os.path.join(args.out_dir, "kv_eval.log"))
    set_args(args)
    common.render_markdown = render_markdown
    if args.stage == "report":
        s = json.load(open(os.path.join(args.out_dir, "summary.json")))
        SUMMARY.update(s)
        common.save_summary(args.out_dir)
        return

    dtype = getattr(torch, args.dtype)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    SUMMARY["meta"]["attention_selftest"] = attention_selftest(args.device, dtype)
    model, inner = load_model(args.model_dir, args.device, dtype)
    eng = Engine(model, inner, args.device, dtype)
    log(f"full-attention layers {eng.full_ids}; lanes start at layer {eng.first_full}")
    if args.stage == "smoke":
        args.upload_prefix = args.upload_prefix + "/smoke"
    run_all(args, eng, tok)
    if args.stage == "smoke":
        d = {a.dest: a.default for a in ap._actions}
        full = {"yarn_needle_size": d["yarn_needle_size"], "long_len": d["long_len"], "short_len": d["short_len"],
                "n_short": d["n_short"], "yarn_short_len": d["yarn_short_len"],
                "needle_sizes": [int(x) for x in d["needle_sizes"].split(",")]}
        probe_and_project(eng, args, full)
        common.save_summary(args.out_dir)
        flush(force=True)


# =================================================================================================
# report
# =================================================================================================

def render_markdown(summary):
    L = ["# Ornith-1.5-35B-A3B: FreeToken KV-cache formats vs bf16 KV (BF16 weights)", ""]
    meta = summary.get("meta", {})
    for k in ("model", "dtype", "prefill_chunk", "kl", "documents", "wall_minutes", "skipped_for_budget", "throughput"):
        if k in meta:
            L.append(f"- **{k}**: {meta[k]}")
    if "lanes" in meta:
        L += ["", "| lane | K | V | FreeToken |", "|---|---|---|---|"]
        L += [f"| {l} | {k} | {v} | {n} |" for l, (k, v, n) in meta["lanes"].items()]
    secs = summary.get("sections", {})
    for sname, title in (("docs_prefill", "Long documents, prefill semantics (FreeToken's prompt path, 8192-token chunks)"),
                         ("docs_decode", "Long documents, decode semantics (every position a decode step)")):
        sec = secs.get(sname)
        if not sec:
            continue
        buckets = [b for b in meta.get("buckets", [])] + ["all"]
        if sname == "docs_prefill":
            L += ["", f"Note: under prefill semantics the FIRST {meta.get('prefill_chunk')}-token chunk of every "
                  "document attends only to its own chunk, which FreeToken reads unquantized (k_extend), so KL "
                  "there is exactly 0 by construction; the decode-semantics tables below cover that range."]
        for metric, fmt in (("kl_mean", "{:.2e}"), ("kl_p99", "{:.2e}"), ("top1_agree_pct", "{:.2f}"), ("dppl_pct", "{:+.2f}")):
            L += ["", f"## {title}: {metric}", "", "| lane | " + " | ".join(buckets) + " |", "|---" * (len(buckets) + 1) + "|"]
            for lane, r in sec.items():
                L.append(f"| {lane} | " + " | ".join(fmt.format(r[b][metric]) if b in r else "-" for b in buckets) + " |")
        kvs = {l: r.get("kv_rel_rmse_per_full_layer") for l, r in sec.items() if r.get("kv_rel_rmse_per_full_layer")}
        if kvs:
            layers = list(next(iter(kvs.values())))
            L += ["", "KV relative RMS error per full-attention layer (K / V):", "",
                  "| lane | " + " | ".join(layers) + " |", "|---" * (len(layers) + 1) + "|"]
            for l, d in kvs.items():
                L.append(f"| {l} | " + " | ".join(f"{d[x]['K']:.3f} / {d[x]['V']:.3f}" for x in layers) + " |")
    nsec = secs.get("needles")
    if nsec:
        L += ["", "## Needles (teacher-forced answer: log-prob, greedy would reproduce it = OK)", ""]
        for size, e in nsec.items():
            lanes = [k for k in e if k not in ("prompt_tokens", "truth")]
            qs = list(e["bf16"])
            L += [f"### {e['prompt_tokens']} prompt tokens", "", "| lane | " + " | ".join(qs) + " |", "|---" * (len(qs) + 1) + "|"]
            for lane in lanes:
                L.append(f"| {lane} | " + " | ".join(
                    f"{'OK' if e[lane][q]['greedy_match'] else 'x'} {e[lane][q]['logprob']:.2f}" +
                    (f" (KL {e[lane][q]['kl_at_answer_mean']:.1e})" if "kl_at_answer_mean" in e[lane][q] else "")
                    for q in qs) + " |")
            L.append("")
    ysec = secs.get("yarn")
    if ysec:
        L += ["", "## YaRN (bf16 KV) vs the checkpoint's unscaled RoPE", "", f"{ysec.get('meta', {})}", ""]
        for k, v in ysec.items():
            if k.startswith("docs_"):
                L += [f"{k}:", "", "| bucket | KL mean | KL p99 | top-1 agree % | dppl % |", "|---|---|---|---|---|"]
                L += [f"| {b} | {r['kl_mean']:.2e} | {r['kl_p99']:.2e} | {r['top1_agree_pct']:.2f} | {r['dppl_pct']:+.2f} |"
                      for b, r in v.items() if isinstance(r, dict) and "kl_mean" in r]
                L.append("")
            elif k.startswith("needle_"):
                for arm, sc in v.items():
                    if isinstance(sc, dict) and sc and isinstance(next(iter(sc.values())), dict):
                        L.append(f"- {k} {arm}: " + ", ".join(
                            f"{q}={'OK' if s['greedy_match'] else 'x'} {s['logprob']:.2f}" for q, s in sc.items()))
                L.append("")
    L += ["", "## raw", "", "```json", json.dumps(summary, indent=1, default=str)[:60000], "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        main()
    finally:
        flush(force=True)
