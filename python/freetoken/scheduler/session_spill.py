"""Bounded cold storage for reclaimable agent-session inference state.

The cache is deliberately process-local.  A checkpoint is useful only while the exact
model, quantized KV layout, and radix-cache generation remain alive; server restart files
are therefore cleaned with their owning :class:`SessionSpillStore` and never trusted.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class SpillChunk:
    family: str
    layer: int
    start: int
    value: torch.Tensor | None = None
    file: Path | None = None


@dataclass
class SessionSpillRecord:
    token_ids: torch.Tensor
    num_pages: int
    byte_size: int
    fingerprint: tuple
    tier: str
    chunks: list[SpillChunk]
    valid: bool = True
    created_at: float = 0.0


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _auto_root() -> Path:
    explicit = os.environ.get("FREETOKEN_CACHE_DIR")
    if explicit:
        return Path(explicit) / "session-spill"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "freetoken" / "session-spill"


class SessionSpillStore:
    """RAM-first, disk-overflow checkpoint store with hard per-process budgets."""

    def __init__(
        self,
        kv_pool,
        linear_state_pool,
        *,
        directory: str,
        ram_budget_bytes: int,
        disk_budget_bytes: int,
        host_reserve_bytes: int,
    ) -> None:
        self.kv_pool = kv_pool
        self.linear_state_pool = linear_state_pool
        self.ram_budget_bytes = max(0, int(ram_budget_bytes))
        self.disk_budget_bytes = max(0, int(disk_budget_bytes))
        self.host_reserve_bytes = max(0, int(host_reserve_bytes))
        self.ram_bytes = 0
        self.disk_bytes = 0
        self._records: list[SessionSpillRecord] = []
        base = _auto_root() if directory == "auto" else Path(directory).expanduser()
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="server-", dir=base))
        self.root.chmod(0o700)

    @classmethod
    def create_if_supported(cls, engine, config) -> SessionSpillStore | None:
        directory = getattr(config, "session_spill_dir", None)
        pool = getattr(engine, "kv_cache", None)
        linear = getattr(engine, "linear_state_pool", None)
        if (
            not directory
            or config.tp_info.size != 1
            or config.page_size != 1
            or linear is None
            or not getattr(pool, "growable", False)
            or not hasattr(pool, "iter_session_spill_tensors")
        ):
            return None
        return cls(
            pool,
            linear,
            directory=directory,
            ram_budget_bytes=int(config.session_spill_ram_gb * (1 << 30)),
            disk_budget_bytes=int(config.session_spill_disk_gb * (1 << 30)),
            host_reserve_bytes=int(config.host_ram_reserve_gb * (1 << 30)),
        )

    def _payload_bytes(self, num_pages: int, token_ids: torch.Tensor) -> int:
        return (
            self.kv_pool.session_spill_bytes(num_pages)
            + self.linear_state_pool.bytes_per_slot()
            + token_ids.numel() * 4
        )

    def _choose_tier(self, byte_size: int) -> str | None:
        # The extra 256 MiB covers one bounded D2H gather and Python/torch metadata.
        ram_headroom = _mem_available_bytes() - self.host_reserve_bytes - (256 << 20)
        if byte_size <= self.ram_budget_bytes - self.ram_bytes and byte_size <= ram_headroom:
            return "ram"
        try:
            free_disk = shutil.disk_usage(self.root).free
        except OSError:
            free_disk = 0
        if byte_size <= self.disk_budget_bytes - self.disk_bytes and byte_size + (1 << 30) <= free_disk:
            return "disk"
        return None

    @torch.inference_mode()
    def spill(
        self,
        token_ids: torch.Tensor,
        page_indices: torch.Tensor,
        linear_slot: int,
    ) -> SessionSpillRecord | None:
        tokens = token_ids.detach().to(device="cpu", dtype=torch.int32).clone()
        num_pages = int(page_indices.numel())
        byte_size = self._payload_bytes(num_pages, tokens)
        tier = self._choose_tier(byte_size)
        if tier is None:
            return None

        chunks: list[SpillChunk] = []
        record = SessionSpillRecord(
            token_ids=tokens,
            num_pages=num_pages,
            byte_size=byte_size,
            fingerprint=self.kv_pool.session_spill_fingerprint(),
            tier=tier,
            chunks=chunks,
            created_at=time.monotonic(),
        )
        target = None
        try:
            # Snapshot writes are enqueued on the engine stream. Session release is a safe
            # scheduler boundary, and this barrier makes the D2H checkpoint exact.
            if page_indices.device.type == "cuda":
                torch.cuda.synchronize(page_indices.device)
            sources = self.kv_pool.iter_session_spill_tensors(page_indices, chunk_pages=16_384)
            if tier == "disk":
                target = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=self.root))
                target.chmod(0o700)
            for ordinal, (family, layer, start, value) in enumerate(sources):
                value = value.contiguous()
                if tier == "ram":
                    chunks.append(SpillChunk(family, layer, start, value=value))
                else:
                    path = target / f"{ordinal:06d}.pt"
                    torch.save(value, path)
                    path.chmod(0o600)
                    chunks.append(SpillChunk(family, layer, start, file=path))

            for family, value in (
                (
                    "gdn_conv",
                    self.linear_state_pool.conv_states[:, linear_slot].cpu().clone(),
                ),
                (
                    "gdn_recurrent",
                    self.linear_state_pool.recurrent_states[:, linear_slot].cpu().clone(),
                ),
            ):
                value = value.contiguous()
                if tier == "ram":
                    chunks.append(SpillChunk(family, -1, 0, value=value))
                else:
                    path = target / f"{len(chunks):06d}.pt"
                    torch.save(value, path)
                    path.chmod(0o600)
                    chunks.append(SpillChunk(family, -1, 0, file=path))
        except Exception:
            if target is not None:
                shutil.rmtree(target, ignore_errors=True)
            return None

        if tier == "ram":
            self.ram_bytes += byte_size
        else:
            self.disk_bytes += byte_size
        self._records.append(record)
        return record

    def _disk_has_room(self, byte_size: int) -> bool:
        try:
            free_disk = shutil.disk_usage(self.root).free
        except OSError:
            return False
        return byte_size <= self.disk_budget_bytes - self.disk_bytes and byte_size + (1 << 30) <= free_disk

    def _demote_to_disk(self, record: SessionSpillRecord) -> bool:
        if not record.valid or record.tier != "ram" or not self._disk_has_room(record.byte_size):
            return False
        target = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=self.root))
        target.chmod(0o700)
        replacements: list[Path] = []
        try:
            for ordinal, chunk in enumerate(record.chunks):
                if chunk.value is None:
                    raise ValueError("RAM checkpoint chunk has no tensor")
                path = target / f"{ordinal:06d}.pt"
                torch.save(chunk.value, path)
                path.chmod(0o600)
                replacements.append(path)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            return False
        for chunk, path in zip(record.chunks, replacements, strict=True):
            chunk.value = None
            chunk.file = path
        record.tier = "disk"
        self.ram_bytes = max(0, self.ram_bytes - record.byte_size)
        self.disk_bytes += record.byte_size
        return True

    def enforce_host_reserve(self) -> tuple[int, int]:
        """Demote oldest RAM records until the live host reserve is healthy.

        If disk cannot accept a record, invalidate it instead. Recomputing one cold
        session is preferable to letting host pressure kill the serving process.
        """
        demoted = dropped = 0
        floor = self.host_reserve_bytes + (256 << 20)
        candidates = sorted(
            (r for r in self._records if r.valid and r.tier == "ram"),
            key=lambda r: r.created_at,
        )
        for record in candidates:
            if _mem_available_bytes() >= floor:
                break
            if self._demote_to_disk(record):
                demoted += 1
            else:
                self.discard(record)
                dropped += 1
        return demoted, dropped

    def iter_chunks(self, record: SessionSpillRecord) -> Iterator[tuple[SpillChunk, torch.Tensor]]:
        if not record.valid or record.fingerprint != self.kv_pool.session_spill_fingerprint():
            raise ValueError("stale or incompatible session checkpoint")
        for chunk in record.chunks:
            if chunk.value is not None:
                yield chunk, chunk.value
            elif chunk.file is not None:
                yield (
                    chunk,
                    torch.load(chunk.file, map_location="cpu", weights_only=True),
                )
            else:
                raise ValueError("session checkpoint chunk has no payload")

    def discard(self, record: SessionSpillRecord | None) -> None:
        if record is None or not record.valid:
            return
        record.valid = False
        self._records = [candidate for candidate in self._records if candidate is not record]
        if record.tier == "ram":
            self.ram_bytes = max(0, self.ram_bytes - record.byte_size)
        else:
            self.disk_bytes = max(0, self.disk_bytes - record.byte_size)
            parents = {chunk.file.parent for chunk in record.chunks if chunk.file is not None}
            for parent in parents:
                shutil.rmtree(parent, ignore_errors=True)
        record.chunks.clear()

    def shutdown(self) -> None:
        for record in self._records:
            record.valid = False
            record.chunks.clear()
        self._records.clear()
        shutil.rmtree(self.root, ignore_errors=True)
        self.ram_bytes = 0
        self.disk_bytes = 0


__all__ = ["SessionSpillRecord", "SessionSpillStore", "SpillChunk"]
