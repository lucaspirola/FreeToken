"""Bounded cold storage for reclaimable agent-session inference state.

A checkpoint is useful only while the exact model and quantized KV layout stay the same,
so every record carries a manifest with the model id, the K/V layout fingerprint, and the
sha256 of its prompt prefix.  Records live in deterministic, session-keyed directories under
a stable root: a restarted server adopts the ones whose manifest still matches the live pool
and deletes everything else (which also collects directories leaked by a crash).

Retention is by capacity and age, never by lease lifetime: a spill that would exceed the
total byte cap (or the filesystem guard) evicts least-recently-used checkpoints until it
fits, and only a single record larger than the whole cap is refused.

A queued session's disk record can be promoted to the RAM tier ahead of its admission
(:meth:`start_prefetch`): the read runs on one background thread, and the main thread
installs the result, so the look-ahead never races the store's accounting.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

MANIFEST_NAME = "manifest.json"
TOKENS_NAME = "tokens.pt"
MANIFEST_VERSION = 1


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
    session_id: str = ""
    # Wall-clock so that ages survive a restart; ``last_used_at`` is the LRU key.
    created_at: float = 0.0
    last_used_at: float = 0.0
    directory: Path | None = field(default=None)


@dataclass
class _Prefetch:
    """One in-flight disk->RAM promotion. Only the reader loop touches ``values``."""

    session_id: str
    record: SessionSpillRecord
    cancel: threading.Event
    started: float
    thread: threading.Thread | None = None
    values: list[torch.Tensor] | None = None


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


def _jsonable(value):
    """Canonical JSON form of a fingerprint tuple (tuples and lists compare equal)."""
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_ours(entry: Path) -> bool:
    """Only entries this store's layouts can produce are eligible for startup GC."""
    name = entry.name
    if entry.is_dir():
        # Current layout: sha256(session id). Legacy layout: one mkdtemp root per server.
        return (len(name) == 64 and all(c in "0123456789abcdef" for c in name)) or (
            name.startswith("server-") or name.startswith("checkpoint-")
        )
    return entry.suffix in {".pt", ".tmp", ".json"}


def _prefix_hash(token_ids: torch.Tensor) -> str:
    tokens = token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


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
        limit_bytes: int | None = None,
        persist: bool = True,
        model_id: str = "",
    ) -> None:
        self.kv_pool = kv_pool
        self.linear_state_pool = linear_state_pool
        self.ram_budget_bytes = max(0, int(ram_budget_bytes))
        self.disk_budget_bytes = max(0, int(disk_budget_bytes))
        self.host_reserve_bytes = max(0, int(host_reserve_bytes))
        # Total RAM+disk retention cap. ``None`` keeps the per-tier budgets as the only bound.
        self.limit_bytes = (
            self.ram_budget_bytes + self.disk_budget_bytes
            if limit_bytes is None
            else max(0, int(limit_bytes))
        )
        self.persist = bool(persist)
        self.model_id = str(model_id)
        self._prefetch: _Prefetch | None = None
        self.ram_bytes = 0
        self.disk_bytes = 0
        self._records: list[SessionSpillRecord] = []
        self._by_session: dict[str, SessionSpillRecord] = {}
        base = _auto_root() if directory == "auto" else Path(directory).expanduser()
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Stable, non-random root: checkpoints are addressable across restarts, and the
        # startup scan below is what reclaims anything a crash left behind.
        self.root = base
        self._adopt_root()

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
            limit_bytes=int(config.session_spill_limit_gb * (1 << 30)),
            persist=bool(getattr(config, "session_spill_persist", True)),
            model_id=str(getattr(config, "model_path", "")),
        )

    # ------------------------------------------------------------------ layout

    def _record_dir(self, session_id: str) -> Path:
        return self.root / hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _payload_bytes(self, num_pages: int, token_ids: torch.Tensor) -> int:
        return (
            self.kv_pool.session_spill_bytes(num_pages)
            + self.linear_state_pool.bytes_per_slot()
            + token_ids.numel() * 4
        )

    # ------------------------------------------------------------- persistence

    def _write_manifest(self, record: SessionSpillRecord) -> None:
        directory = record.directory
        if directory is None or record.tier != "disk":
            return
        manifest = {
            "version": MANIFEST_VERSION,
            "session_id": record.session_id,
            "model_id": self.model_id,
            "prefix_sha256": _prefix_hash(record.token_ids),
            "num_tokens": int(record.token_ids.numel()),
            "num_pages": int(record.num_pages),
            "byte_size": int(record.byte_size),
            "fingerprint": _jsonable(record.fingerprint),
            "created_at": record.created_at,
            "last_used_at": record.last_used_at,
            "chunks": [
                [chunk.family, chunk.layer, chunk.start, chunk.file.name]
                for chunk in record.chunks
                if chunk.file is not None
            ],
        }
        path = directory / MANIFEST_NAME
        tmp = directory / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)

    def _load_record(self, directory: Path) -> SessionSpillRecord | None:
        """Rebuild one on-disk record, or return None when it must be deleted."""
        try:
            manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
            return None
        session_id = manifest.get("session_id")
        if not isinstance(session_id, str) or self._record_dir(session_id) != directory:
            return None
        if manifest.get("model_id") != self.model_id:
            return None
        live = self.kv_pool.session_spill_fingerprint()
        if manifest.get("fingerprint") != _jsonable(live):
            return None
        try:
            tokens = torch.load(directory / TOKENS_NAME, map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int32:
            return None
        if tokens.numel() != manifest.get("num_tokens"):
            return None
        if _prefix_hash(tokens) != manifest.get("prefix_sha256"):
            return None
        chunks: list[SpillChunk] = []
        for entry in manifest.get("chunks", ()):
            try:
                family, layer, start, name = entry
                path = directory / str(name)
            except (TypeError, ValueError):
                return None
            if path.parent != directory or not path.is_file():
                return None
            chunks.append(SpillChunk(str(family), int(layer), int(start), file=path))
        if not chunks:
            return None
        return SessionSpillRecord(
            token_ids=tokens,
            num_pages=int(manifest.get("num_pages", 0)),
            byte_size=int(manifest.get("byte_size", 0)),
            fingerprint=live,
            tier="disk",
            chunks=chunks,
            session_id=session_id,
            created_at=float(manifest.get("created_at", 0.0)),
            last_used_at=float(manifest.get("last_used_at", 0.0)),
            directory=directory,
        )

    def _adopt_root(self) -> None:
        """Adopt still-valid checkpoints under the root; delete stale or foreign ones."""
        adopted = deleted = 0
        try:
            entries = sorted(self.root.iterdir())
        except OSError:
            return
        for entry in entries:
            record = None
            if self.persist and entry.is_dir():
                try:
                    record = self._load_record(entry)
                except Exception:  # a corrupt record is never an admission gate
                    record = None
            if record is None:
                if not _is_ours(entry):
                    # Never delete something this store could not have written: a spill
                    # directory the operator shares with other content stays intact.
                    logger.warning(
                        "Ignoring unrecognized entry %s in the session spill root", entry
                    )
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                deleted += 1
                continue
            self._track(record)
            adopted += 1
        while (self.ram_bytes + self.disk_bytes) > self.limit_bytes and self._evict_one_lru():
            pass
        if adopted or deleted:
            logger.info_rank0(
                "Session spill root %s: adopted %d checkpoint(s), removed %d stale entr(ies)",
                self.root,
                adopted,
                deleted,
            )

    # -------------------------------------------------------------- accounting

    def _track(self, record: SessionSpillRecord) -> None:
        self.discard(self._by_session.get(record.session_id))
        self._records.append(record)
        if record.session_id:
            self._by_session[record.session_id] = record
        if record.tier == "ram":
            self.ram_bytes += record.byte_size
        else:
            self.disk_bytes += record.byte_size

    @property
    def num_records(self) -> int:
        return len(self._records)

    def get(self, session_id: str) -> SessionSpillRecord | None:
        """Find a checkpoint by session id, even after its lease object is gone."""
        record = self._by_session.get(session_id)
        return record if record is not None and record.valid else None

    def touch(self, record: SessionSpillRecord | None) -> None:
        if record is None or not record.valid:
            return
        record.last_used_at = time.time()
        try:
            self._write_manifest(record)
        except OSError:
            pass

    def _evict_one_lru(self, exclude_session: str | None = None) -> bool:
        candidates = [
            record
            for record in self._records
            if record.valid and record.session_id != exclude_session
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda record: (record.last_used_at, record.created_at))
        logger.info_rank0(
            "Evicting cold session checkpoint %s (%.2f GiB, %s) to stay inside the "
            "session spill cap",
            victim.session_id,
            victim.byte_size / (1 << 30),
            victim.tier,
        )
        self.discard(victim)
        return True

    def _ram_has_room(self, byte_size: int) -> bool:
        # The extra 256 MiB covers one bounded D2H gather and Python/torch metadata.
        ram_headroom = _mem_available_bytes() - self.host_reserve_bytes - (256 << 20)
        return byte_size <= self.ram_budget_bytes - self.ram_bytes and byte_size <= ram_headroom

    def _choose_tier(self, byte_size: int) -> str | None:
        if self._ram_has_room(byte_size):
            return "ram"
        if self._disk_has_room(byte_size):
            return "disk"
        return None

    def _reserve_tier(self, byte_size: int, session_id: str) -> str | None:
        """Pick a tier for ``byte_size``, evicting least-recently-used records to fit."""
        if byte_size > self.limit_bytes:
            return None  # a single record larger than the whole cap is refused
        while True:
            if self.ram_bytes + self.disk_bytes + byte_size <= self.limit_bytes:
                tier = self._choose_tier(byte_size)
                if tier is not None:
                    return tier
            if not self._evict_one_lru(session_id):
                return None

    # ------------------------------------------------------------------ spill

    @torch.inference_mode()
    def spill(
        self,
        session_id: str,
        token_ids: torch.Tensor,
        page_indices: torch.Tensor,
        linear_slot: int,
    ) -> SessionSpillRecord | None:
        tokens = token_ids.detach().to(device="cpu", dtype=torch.int32).clone()
        num_pages = int(len(page_indices))
        byte_size = self._payload_bytes(num_pages, tokens)
        # Drop any previous checkpoint for this session first: its directory is the same
        # deterministic path this spill is about to write.
        self.discard(self.get(session_id))
        tier = self._reserve_tier(byte_size, session_id)
        if tier is None:
            return None

        now = time.time()
        chunks: list[SpillChunk] = []
        record = SessionSpillRecord(
            token_ids=tokens,
            num_pages=num_pages,
            byte_size=byte_size,
            fingerprint=self.kv_pool.session_spill_fingerprint(),
            tier=tier,
            chunks=chunks,
            session_id=session_id,
            created_at=now,
            last_used_at=now,
        )
        target = None
        try:
            # Snapshot writes are enqueued on the engine stream. Session release is a safe
            # scheduler boundary, and this barrier makes the D2H checkpoint exact.
            if page_indices.device.type == "cuda":
                torch.cuda.synchronize(page_indices.device)
            sources = self.kv_pool.iter_session_spill_tensors(page_indices, chunk_pages=16_384)
            if tier == "disk":
                target = self._prepare_dir(session_id)
                record.directory = target
                torch.save(tokens, target / TOKENS_NAME)
                (target / TOKENS_NAME).chmod(0o600)
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
            if tier == "disk":
                self._write_manifest(record)
        except Exception:
            if target is not None:
                shutil.rmtree(target, ignore_errors=True)
            return None

        self._track(record)
        return record

    def _prepare_dir(self, session_id: str) -> Path:
        target = self._record_dir(session_id)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(mode=0o700, parents=True)
        target.chmod(0o700)
        return target

    def _disk_has_room(self, byte_size: int) -> bool:
        try:
            free_disk = shutil.disk_usage(self.root).free
        except OSError:
            return False
        return byte_size <= self.disk_budget_bytes - self.disk_bytes and byte_size + (1 << 30) <= free_disk

    def _demote_to_disk(self, record: SessionSpillRecord) -> bool:
        if not record.valid or record.tier != "ram" or not self._disk_has_room(record.byte_size):
            return False
        target = self._prepare_dir(record.session_id)
        replacements: list[Path] = []
        try:
            torch.save(record.token_ids, target / TOKENS_NAME)
            (target / TOKENS_NAME).chmod(0o600)
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
        record.directory = target
        self.ram_bytes = max(0, self.ram_bytes - record.byte_size)
        self.disk_bytes += record.byte_size
        try:
            self._write_manifest(record)
        except OSError:
            pass
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
        if record.tier == "ram":
            for chunk in record.chunks:
                if chunk.value is None:
                    raise ValueError("RAM session checkpoint chunk has no payload")
                yield chunk, chunk.value
            return

        disk_chunks = list(record.chunks)
        if not disk_chunks:
            return

        def load(chunk: SpillChunk) -> torch.Tensor:
            if chunk.file is None:
                raise ValueError("disk session checkpoint chunk has no file")
            return torch.load(chunk.file, map_location="cpu", weights_only=True)

        # One bounded look-ahead overlaps NVMe/deserialization of chunk N+1 with the
        # caller's GPU install of chunk N. At most two host chunks are live, preserving
        # the same fixed-memory streaming contract as the original sequential reader.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-spill") as pool:
            pending = pool.submit(load, disk_chunks[0])
            for index, chunk in enumerate(disk_chunks):
                value = pending.result()
                if index + 1 < len(disk_chunks):
                    pending = pool.submit(load, disk_chunks[index + 1])
                yield chunk, value

    # ---------------------------------------------------------------- prefetch

    def _make_ram_room(self, byte_size: int, protect: set[str]) -> bool:
        """Demote least-recently-used RAM records to disk until ``byte_size`` fits.

        ``protect`` is never demoted: the resident and restoring sessions' checkpoints are
        the ones about to be needed, so paying to move them would defeat the look-ahead.
        """
        if self._ram_has_room(byte_size):
            return True
        victims = sorted(
            (
                record
                for record in self._records
                if record.valid and record.tier == "ram" and record.session_id not in protect
            ),
            key=lambda record: (record.last_used_at, record.created_at),
        )
        for victim in victims:
            if self._demote_to_disk(victim) and self._ram_has_room(byte_size):
                return True
        return self._ram_has_room(byte_size)

    def start_prefetch(self, session_id: str, *, protect: Iterable[str] = ()) -> bool:
        """Begin promoting one queued session's disk checkpoint to RAM. One at a time.

        Returns False (never raises) when there is nothing to promote or the RAM budget
        cannot hold it: the look-ahead is an optimization, never an admission gate.
        """
        self.collect_prefetch()  # reap a finished or cancelled predecessor
        if self._prefetch is not None:
            return False
        record = self.get(session_id)
        if record is None or record.tier != "disk":
            return False
        files = [chunk.file for chunk in record.chunks]
        if any(path is None for path in files):
            return False
        if not self._make_ram_room(record.byte_size, set(protect) | {session_id}):
            return False
        cancel = threading.Event()
        state = _Prefetch(session_id, record, cancel, time.perf_counter())

        def _read() -> None:
            values: list[torch.Tensor] = []
            try:
                for path in files:
                    if cancel.is_set():
                        return
                    values.append(torch.load(path, map_location="cpu", weights_only=True))
            except Exception:  # a torn read just means the restore reads it from disk
                return
            state.values = values

        state.thread = threading.Thread(target=_read, name="session-prefetch", daemon=True)
        self._prefetch = state
        state.thread.start()
        return True

    def collect_prefetch(
        self, session_id: str | None = None, *, wait: bool = False
    ) -> str | None:
        """Install a finished promotion (main thread only). Returns the session promoted."""
        state = self._prefetch
        if state is None or (session_id is not None and state.session_id != session_id):
            return None
        if state.thread is not None and state.thread.is_alive():
            if not wait:
                return None
            state.thread.join()
        self._prefetch = None
        record, values = state.record, state.values
        if (
            state.cancel.is_set()
            or values is None
            or not record.valid
            or record.tier != "disk"
            or len(values) != len(record.chunks)
            or not self._ram_has_room(record.byte_size)
        ):
            return None
        self._promote_to_ram(record, values)
        logger.info_rank0(
            "Prefetched cold session %s to RAM (%.2f GiB in %.2f s)",
            record.session_id,
            record.byte_size / (1 << 30),
            time.perf_counter() - state.started,
        )
        return record.session_id

    def cancel_prefetch(self, session_id: str | None = None) -> bool:
        """Abandon an in-flight promotion; its reader drops the bytes it already read."""
        state = self._prefetch
        if state is None or (session_id is not None and state.session_id != session_id):
            return False
        state.cancel.set()
        return True

    def _promote_to_ram(self, record: SessionSpillRecord, values: list[torch.Tensor]) -> None:
        directory = record.directory
        for chunk, value in zip(record.chunks, values, strict=True):
            chunk.value = value
            chunk.file = None
        record.tier = "ram"
        record.directory = None
        self.disk_bytes = max(0, self.disk_bytes - record.byte_size)
        self.ram_bytes += record.byte_size
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

    def discard(self, record: SessionSpillRecord | None) -> None:
        if record is None or not record.valid:
            return
        if self._prefetch is not None and self._prefetch.record is record:
            self.cancel_prefetch()
        record.valid = False
        self._records = [candidate for candidate in self._records if candidate is not record]
        if self._by_session.get(record.session_id) is record:
            self._by_session.pop(record.session_id, None)
        if record.tier == "ram":
            self.ram_bytes = max(0, self.ram_bytes - record.byte_size)
        else:
            self.disk_bytes = max(0, self.disk_bytes - record.byte_size)
            parents = {chunk.file.parent for chunk in record.chunks if chunk.file is not None}
            if record.directory is not None:
                parents.add(record.directory)
            for parent in parents:
                shutil.rmtree(parent, ignore_errors=True)
        record.chunks.clear()

    def shutdown(self) -> None:
        """Persisting shutdown only flushes manifests; the root survives for the next run."""
        self.cancel_prefetch()
        if not self.persist:
            for record in list(self._records):
                self.discard(record)
            self._records.clear()
            self._by_session.clear()
            self.ram_bytes = 0
            self.disk_bytes = 0
            return
        for record in list(self._records):
            if record.tier == "ram":
                # RAM payloads die with the process; move what still fits so the next
                # run can adopt them, and drop the rest.
                try:
                    demoted = self._demote_to_disk(record)
                except Exception:
                    demoted = False
                if not demoted:
                    self.discard(record)
                    continue
            try:
                self._write_manifest(record)
            except OSError:
                pass


__all__ = ["SessionSpillRecord", "SessionSpillStore", "SpillChunk"]
