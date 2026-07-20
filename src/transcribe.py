"""
Verbatim transcription pipeline (the new primary path).

    video.mov
      → select_frames    (dense sample, drop static dupes)        frame_select.py
      → VerbatimOCR       (vision model, char-faithful)            ocr.py
      → stitch_frames     (text-space dedup of scroll overlap)     stitch.py
      → LLM cleanup       (seams + indentation ONLY, optional)     this file
      → output/transcript.md

Everything is local: vision OCR + text cleanup both use the selected
OpenAI-compatible server (vLLM on DGX Spark by default).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .config import ScreenLensConfig
from .frame_select import select_frames
from .ocr import OCRFrameResult, VerbatimOCR
from .omlx_client import (
    InferenceClient,
    resolve_llm_model,
    resolve_ocr_model_revision,
    resolve_role_api_key,
    resolve_role_backend,
    resolve_role_base_url,
    resolve_role_context,
)
from .stitch import stitch_frames

logger = logging.getLogger("screenlens.transcribe")

# Cleanup is seam/indent repair ONLY — it must never drop content. An LLM
# (especially a reasoning model) tends to "improve" by condensing, silently
# dropping code blocks or lists. After each chunk we check what fraction of its
# distinct non-blank input lines survived; below this we discard the LLM output
# and keep the raw stitched chunk. The small slack tolerates legitimate edits
# (stray header/footer removal, rejoining a line split across a frame seam).
MIN_CHUNK_COVERAGE = 0.97
RESUME_SCHEMA_VERSION = 1
STRICT_OCR_PIPELINE_VERSION = 1


def _chunk_coverage(src: str, repaired: str) -> float:
    """Fraction of distinct non-blank input lines (whitespace-normalized) that
    still appear in the repaired output. 1.0 means nothing was dropped."""
    def norm_lines(t: str) -> set[str]:
        return {re.sub(r"\s+", "", l) for l in t.splitlines() if l.strip()}

    src_lines = norm_lines(src)
    if not src_lines:
        return 1.0
    out_lines = norm_lines(repaired)
    return sum(1 for l in src_lines if l in out_lines) / len(src_lines)


CLEANUP_SYSTEM = (
    "You repair a transcript that was OCR'd frame-by-frame from a scrolling "
    "screen recording and then stitched together. Your edits are STRICTLY "
    "limited:\n"
    "1. Fix obvious stitch seams: remove a duplicated line where two frames "
    "overlapped, or rejoin a line that was split across the overlap.\n"
    "2. Restore consistent indentation for code blocks.\n"
    "3. Remove stray page headers/footers that slipped through (e.g. 'Page 3 of "
    "16', running titles).\n\n"
    "You must NOT paraphrase, summarize, translate, complete, or 'improve' any "
    "content. Do not invent text. Do not add commentary. If a word is garbled "
    "and you cannot be certain, leave it exactly as-is. Output ONLY the repaired "
    "transcript."
)


def _llm_client(cfg) -> InferenceClient:
    rc = cfg.reconstruction
    return InferenceClient.from_endpoint(
        base_url=resolve_role_base_url(rc),
        model=resolve_llm_model(rc),
        api_key=resolve_role_api_key(rc, "VLLM_LLM_API_KEY", "LLM_API_KEY"),
        backend=resolve_role_backend(rc),
        timeout=rc.timeout_seconds,
        context_size=resolve_role_context(rc),
        default_max_tokens=rc.max_tokens,
        default_temperature=rc.temperature,
    )


def _cleanup_transcript(text: str, cfg) -> str:
    """LLM seam/indent cleanup, chunked by blank-line boundaries to fit context."""
    client = _llm_client(cfg)
    extra = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if cfg.reconstruction.disable_thinking
        else None
    )
    # Cleanup is near-verbatim, so the repaired output is ~the same size as the
    # input. The binding limit is therefore the OUTPUT cap (max_tokens), not just
    # the context window: a chunk larger than max_tokens can emit guarantees
    # mid-chunk truncation and silent content loss. Bound chunk input by BOTH the
    # output cap and the context window (input+output+prompt must co-fit), with a
    # safety margin. (chars ≈ tokens*4)
    chars_per_token = 4
    max_out_chars = cfg.reconstruction.max_tokens * chars_per_token
    max_ctx_chars = int(cfg.reconstruction.model_context * 0.45) * chars_per_token
    budget_chars = int(min(max_out_chars, max_ctx_chars) * 0.85)
    paras = text.split("\n\n")
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur and cur_len + len(p) > budget_chars:
            chunks.append("\n\n".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))

    out = []
    for i, ch in enumerate(chunks):
        logger.info("LLM cleanup chunk %d/%d", i + 1, len(chunks))
        repaired = client.chat(
            CLEANUP_SYSTEM,
            "Repair this stitched transcript segment. Output only the repaired text:\n\n" + ch,
            max_tokens=cfg.reconstruction.max_tokens,
            temperature=0.0,
            extra=extra,
        ).strip()
        coverage = _chunk_coverage(ch, repaired)
        if coverage < MIN_CHUNK_COVERAGE:
            logger.warning(
                "Cleanup chunk %d/%d dropped content (line coverage %.0f%% < %.0f%%); "
                "keeping the raw stitched chunk to preserve fidelity.",
                i + 1, len(chunks), coverage * 100, MIN_CHUNK_COVERAGE * 100,
            )
            out.append(ch.strip())
        else:
            out.append(repaired)
    return "\n\n".join(out).strip() + "\n"


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for directory-entry changes."""
    try:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            # Some otherwise-atomic filesystems reject directory fsync with
            # EINVAL. Callers still flush file contents before replacement.
            logger.debug("Directory fsync unavailable for %s: %s", path, exc)
    finally:
        os.close(directory_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace ``path`` without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ocr_contract(ocr: VerbatimOCR) -> dict:
    """Return fidelity-affecting OCR settings, deliberately excluding secrets.

    Timeout and concurrency are operational knobs and may safely change on a
    resumed run. Model, request template, token/context limits, tiling, and the
    deterministic backstop affect the transcription and therefore may not.
    """
    cfg = ocr.config
    profile = ocr.request_profile
    return {
        "strict_ocr_pipeline_version": STRICT_OCR_PIPELINE_VERSION,
        "backend": ocr.client.backend.value,
        "base_url": ocr.client.base_url,
        "model": ocr.model,
        "model_revision": resolve_ocr_model_revision(cfg),
        "model_context": ocr.client.context_size,
        "max_tokens": cfg.max_tokens,
        "request_profile": {
            "family": profile.family,
            "system_prompt": profile.system_prompt,
            "user_prompt": profile.user_prompt,
            "image_first": profile.image_first,
            "temperature": profile.temperature,
            "extra": profile.extra,
        },
        "tile_fallback_enabled": cfg.tile_fallback_enabled,
        "tile_rows": cfg.tile_rows,
        "tile_overlap_ratio": cfg.tile_overlap_ratio,
        "tile_min_overlap_pixels": cfg.tile_min_overlap_pixels,
        "tile_max_tokens": cfg.tile_max_tokens,
        "tile_retry_max_tokens": cfg.tile_retry_max_tokens,
        "tile_max_depth": cfg.tile_max_depth,
        "tile_min_core_height": cfg.tile_min_core_height,
        "tile_max_requests": cfg.tile_max_requests,
        "deterministic_backstop": cfg.deterministic_backstop,
    }


@contextmanager
def _transcription_lock(data_dir: Path) -> Iterator[None]:
    """Prevent two writers from corrupting the same resumable run directory."""
    lock_path = data_dir / "transcribe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another transcription process is already using {data_dir}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_frames(frames: list[dict], data_dir: Path) -> list[dict]:
    documented: list[dict] = []
    for frame in frames:
        path = Path(str(frame["path"])).resolve()
        try:
            relative_path = str(path.relative_to(data_dir.resolve()))
        except ValueError as exc:
            raise RuntimeError(
                f"Selected frame is outside the transcription run directory: {path}"
            ) from exc
        documented.append(
            {
                **frame,
                "path": str(path),
                "relative_path": relative_path,
                "sha256": _sha256_file(path),
            }
        )
    return documented


def _build_resume_manifest(
    *,
    video_path: str,
    frames: list[dict],
    config: ScreenLensConfig,
    ocr_contract: dict,
    data_dir: Path,
) -> dict:
    video = Path(video_path).resolve()
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "created_at_epoch": time.time(),
        "video": {
            "path": str(video),
            "size_bytes": video.stat().st_size,
            "sha256": _sha256_file(video),
        },
        "frame_selection": config.frame_selection.model_dump(mode="json"),
        "frames": _manifest_frames(frames, data_dir),
        "ocr_contract": ocr_contract,
        "ocr_contract_fingerprint": _canonical_fingerprint(ocr_contract),
    }


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read resume metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Resume metadata {path} is not a JSON object.")
    return value


def _validate_resume_manifest(
    manifest: dict,
    *,
    video_path: str,
    ocr_contract: dict,
    data_dir: Path,
) -> list[dict]:
    if manifest.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise RuntimeError(
            "Resume manifest schema is unsupported; start a new transcription."
        )

    expected_contract = manifest.get("ocr_contract")
    expected_fingerprint = manifest.get("ocr_contract_fingerprint")
    actual_fingerprint = _canonical_fingerprint(ocr_contract)
    if (
        expected_contract != ocr_contract
        or expected_fingerprint != actual_fingerprint
    ):
        expected_model = (
            expected_contract.get("model")
            if isinstance(expected_contract, dict)
            else "unknown"
        )
        raise RuntimeError(
            "Resume OCR contract mismatch: this run was created for "
            f"{expected_model} ({str(expected_fingerprint)[:12]}), but the "
            f"current model/settings resolve to {ocr_contract.get('model')} "
            f"({actual_fingerprint[:12]}). Reuse the original OCR settings or "
            "start a new transcription directory."
        )

    video_meta = manifest.get("video")
    video = Path(video_path).resolve()
    if not isinstance(video_meta, dict):
        raise RuntimeError("Resume manifest is missing video identity metadata.")
    if (
        video.stat().st_size != video_meta.get("size_bytes")
        or _sha256_file(video) != video_meta.get("sha256")
    ):
        raise RuntimeError(
            f"Resume video mismatch: {video} is not the recording captured in "
            f"{data_dir / 'ocr' / 'resume_manifest.json'}."
        )

    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise RuntimeError("Resume manifest contains no selected frames.")
    frames: list[dict] = []
    run_root = data_dir.resolve()
    seen_frame_ids: set[int] = set()
    for position, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise RuntimeError("Resume manifest contains malformed frame metadata.")
        frame_id = item.get("frame_id")
        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
            or frame_id in seen_frame_ids
            or frame_id != position
        ):
            raise RuntimeError(
                "Resume manifest frame IDs must be unique ordered integers "
                "starting at zero."
            )
        seen_frame_ids.add(frame_id)
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(
                f"Resume frame {frame_id} has no run-relative path."
            )
        path = (run_root / relative).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Resume frame {frame_id} escapes the run directory: {relative}"
            ) from exc
        expected_hash = item.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise RuntimeError(
                f"Resume frame changed or is missing: {path}. Start a new "
                "transcription so frame selection and OCR stay consistent."
            )
        frames.append({**item, "path": str(path)})
    return frames


def _result_record(
    frame: dict,
    result: OCRFrameResult,
    *,
    contract_fingerprint: str,
) -> dict:
    diagnostics = asdict(result)
    return {
        **frame,
        "ocr": diagnostics.pop("text"),
        "ocr_status": diagnostics.pop("status"),
        "ocr_complete": diagnostics.pop("complete"),
        "ocr_tiled": diagnostics.pop("tiled"),
        "ocr_full_frame_attempted": diagnostics.pop("full_frame_attempted"),
        "ocr_tile_requests": diagnostics.pop("tile_requests"),
        "ocr_tile_leaves": diagnostics.pop("tile_leaves"),
        "ocr_tile_max_depth": diagnostics.pop("tile_max_depth"),
        "ocr_unmatched_seams": diagnostics.pop("unmatched_seams"),
        "ocr_error": diagnostics.pop("error"),
        "ocr_contract_fingerprint": contract_fingerprint,
    }


def _coerce_result(value: object) -> OCRFrameResult:
    """Keep lightweight third-party/test OCR adapters source-compatible."""
    if isinstance(value, OCRFrameResult):
        return value
    if isinstance(value, str):
        return OCRFrameResult(
            text=value,
            status="complete" if value else "empty",
            complete=True,
        )
    raise RuntimeError(f"Strict OCR returned unsupported result type: {type(value)!r}")


def _partial_transcript(records: list[dict | None]) -> str:
    """Build an explicitly gapped preview without joining nonadjacent frames."""
    parts: list[str] = []
    contiguous: list[str] = []

    def flush_complete_run() -> None:
        if not contiguous:
            return
        parts.append(
            stitch_frames(
                [text.split("\n") for text in contiguous],
                fuzzy=0.85,
                strip_boilerplate=True,
                preserve_unmatched_overlap=True,
                preserve_whitespace=True,
            ).text()
        )
        contiguous.clear()

    missing: list[str] = []
    for index, record in enumerate(records):
        if record is not None and record.get("ocr_complete") is True:
            if missing:
                flush_complete_run()
                parts.append(
                    "<!-- OCR-INCOMPLETE: missing frame position(s) "
                    + ", ".join(missing)
                    + "; transcript continuity is unknown -->"
                )
                missing.clear()
            contiguous.append(str(record.get("ocr", "")))
            continue
        frame_id = record.get("frame_id") if record is not None else None
        missing.append(str(frame_id if frame_id is not None else index))

    flush_complete_run()
    if missing:
        parts.append(
            "<!-- OCR-INCOMPLETE: missing frame position(s) "
            + ", ".join(missing)
            + "; transcript continuity is unknown -->"
        )
    return "\n\n".join(part for part in parts if part)


def transcribe_video(
    video_path: str,
    config: ScreenLensConfig,
    data_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    """Run the full verbatim pipeline for one video. Returns a result dict."""
    t0 = time.time()
    data_dir = Path(data_dir)
    if resume and not data_dir.is_dir():
        raise RuntimeError(f"Resume directory does not exist: {data_dir}")
    frames_dir = data_dir / "frames"
    ocr_dir = data_dir / "ocr"
    out_dir = data_dir / "output"
    data_dir.mkdir(parents=True, exist_ok=True)

    with _transcription_lock(data_dir):
        for directory in (frames_dir, ocr_dir, out_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if not resume and any(
            any(directory.iterdir()) for directory in (frames_dir, ocr_dir, out_dir)
        ):
            raise RuntimeError(
                f"Refusing a fresh transcription in nonempty run directory "
                f"{data_dir}. Use --resume-dir for that exact run or choose a "
                "new output directory."
            )

        ocr = VerbatimOCR(config.ocr)
        contract = _ocr_contract(ocr)
        contract_fingerprint = _canonical_fingerprint(contract)
        manifest_path = ocr_dir / "resume_manifest.json"

        # 1. Select or validate the immutable frame set ──────────────────────
        if resume and manifest_path.exists():
            manifest = _load_json(manifest_path)
            frames = _validate_resume_manifest(
                manifest,
                video_path=video_path,
                ocr_contract=contract,
                data_dir=data_dir,
            )
            logger.info("Validated %d resumable frames", len(frames))
        else:
            if resume and any(ocr_dir.glob("ocr_*.json")):
                raise RuntimeError(
                    f"Cannot safely resume {data_dir}: per-frame OCR exists but "
                    "resume_manifest.json does not. Start a new transcription."
                )
            # This also bootstraps a pre-checkpoint run whose frame directory
            # exists but contains no persisted OCR (the July 15 transition).
            frames = select_frames(video_path, str(frames_dir), config.frame_selection)
            if not frames:
                return {"error": "No frames extracted", "stage": "select"}
            manifest = _build_resume_manifest(
                video_path=video_path,
                frames=frames,
                config=config,
                ocr_contract=contract,
                data_dir=data_dir,
            )
            _atomic_write_json(manifest_path, manifest)
            frames = manifest["frames"]
            logger.info("Selected %d frames and saved resume manifest", len(frames))

        # 2. Reuse only audited complete records; retry missing/failed frames ─
        records: list[dict | None] = [None] * len(frames)
        pending_indices: list[int] = []
        tile_first_proven = False
        for index, frame in enumerate(frames):
            record_path = ocr_dir / f"ocr_{int(frame['frame_id']):06d}.json"
            if record_path.exists():
                try:
                    record = _load_json(record_path)
                except RuntimeError as exc:
                    logger.warning("Ignoring unreadable OCR checkpoint: %s", exc)
                else:
                    identity_matches = (
                        record.get("ocr_contract_fingerprint")
                        == contract_fingerprint
                        and record.get("sha256") == frame.get("sha256")
                        and record.get("frame_id") == frame.get("frame_id")
                    )
                    if identity_matches and record.get("ocr_tiled") is True:
                        tile_first_proven = True
                    reusable = (
                        record.get("ocr_complete") is True
                        and identity_matches
                    )
                    if reusable:
                        records[index] = record
                        continue
            pending_indices.append(index)

        if tile_first_proven:
            # A previous process already proved that full frames exhaust this
            # run's exact OCR contract. Preserve that circuit-breaker decision
            # across an explicit resume instead of spending minutes reproving it.
            ocr.prefer_tiles()

        if pending_indices:
            # Never leave a stale canonical transcript visible while this run is
            # known to be incomplete. Per-frame checkpoints remain untouched.
            for stale in (
                ocr_dir / "all_ocr.json",
                out_dir / "transcript.raw.md",
                out_dir / "transcript.md",
                out_dir / "transcript.partial.md",
                out_dir / "transcribe_meta.json",
            ):
                stale.unlink(missing_ok=True)
            # Make the absence of stale canonical outputs durable before the
            # first potentially multi-minute model request. Atomic progress
            # writes fsync ``ocr_dir``; ``out_dir`` needs its own barrier.
            _fsync_directory(ocr_dir)
            _fsync_directory(out_dir)

        progress_path = ocr_dir / "progress.json"

        def write_progress() -> None:
            complete = sum(
                1
                for record in records
                if record is not None and record.get("ocr_complete") is True
            )
            failed = sum(
                1
                for record in records
                if record is not None and record.get("ocr_status") == "failed"
            )
            _atomic_write_json(
                progress_path,
                {
                    "schema_version": RESUME_SCHEMA_VERSION,
                    "frames_total": len(frames),
                    "frames_complete": complete,
                    "frames_failed": failed,
                    "frames_pending": len(frames) - complete,
                    "ocr_contract_fingerprint": contract_fingerprint,
                    "updated_at_epoch": time.time(),
                },
            )

        # Invalidate any stale 100%-complete progress state before the first
        # potentially multi-minute model request, not only after its callback.
        write_progress()

        batch_error: BaseException | None = None
        if pending_indices:
            logger.info(
                "OCR pending: %d frame(s); reusing %d complete checkpoint(s)",
                len(pending_indices),
                len(frames) - len(pending_indices),
            )
            pending_paths = [str(frames[index]["path"]) for index in pending_indices]

            def persist(local_index: int, raw_result: OCRFrameResult) -> None:
                global_index = pending_indices[local_index]
                result = _coerce_result(raw_result)
                frame = frames[global_index]
                record = _result_record(
                    frame,
                    result,
                    contract_fingerprint=contract_fingerprint,
                )
                records[global_index] = record
                record_path = ocr_dir / f"ocr_{int(frame['frame_id']):06d}.json"
                _atomic_write_json(record_path, record)
                write_progress()

            try:
                batch_results = ocr.ocr_frames_verbatim(
                    pending_paths,
                    on_result=persist,
                )
                # Adapters are encouraged to call on_result progressively. This
                # final pass covers simple implementations that only return.
                for local_index, raw_result in enumerate(batch_results):
                    global_index = pending_indices[local_index]
                    if records[global_index] is None:
                        persist(local_index, _coerce_result(raw_result))
            except Exception as exc:
                batch_error = exc
                logger.error("Strict OCR batch stopped early: %s", exc)
            except KeyboardInterrupt as exc:
                batch_error = exc
                logger.warning(
                    "Transcription interrupted; finalizing resumable checkpoints"
                )
        else:
            logger.info("All %d OCR frames were already complete", len(frames))

        incomplete_indices = [
            index
            for index, record in enumerate(records)
            if record is None or record.get("ocr_complete") is not True
        ]
        if incomplete_indices:
            partial_path = out_dir / "transcript.partial.md"
            _atomic_write_text(partial_path, _partial_transcript(records))
            failure_meta = {
                "stage": "ocr_incomplete",
                "video": str(Path(video_path).resolve()),
                "frames_selected": len(frames),
                "frames_complete": len(frames) - len(incomplete_indices),
                "frames_incomplete": len(incomplete_indices),
                "incomplete_frame_ids": [
                    frames[index].get("frame_id") for index in incomplete_indices
                ],
                "ocr_model": ocr.model,
                "ocr_max_tokens": config.ocr.max_tokens,
                "ocr_contract_fingerprint": contract_fingerprint,
                "partial_transcript_path": str(partial_path),
                "resume_dir": str(data_dir.resolve()),
                "error": str(batch_error) if batch_error else None,
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            _atomic_write_json(out_dir / "transcribe_meta.json", failure_meta)
            if isinstance(batch_error, KeyboardInterrupt):
                raise batch_error
            detail = f" ({batch_error})" if batch_error else ""
            raise RuntimeError(
                f"Verbatim OCR is incomplete for {len(incomplete_indices)} of "
                f"{len(frames)} frames{detail}. Complete frame checkpoints and "
                f"a partial transcript were preserved. Resume with "
                f"--resume-dir {data_dir}."
            )

        if isinstance(batch_error, KeyboardInterrupt):
            # Honor Ctrl+C even in the narrow race where the last completion
            # callback was persisted immediately before the signal arrived.
            raise batch_error

        ocr_records = [record for record in records if record is not None]
        texts = [str(record.get("ocr", "")) for record in ocr_records]
        _atomic_write_json(ocr_dir / "all_ocr.json", ocr_records)
        non_empty = sum(1 for text in texts if text.strip())
        logger.info("OCR done: %d/%d frames had text", non_empty, len(texts))

        # 3. Stitch (text-space dedup) ───────────────────────────────────────
        frames_lines = [text.split("\n") for text in texts if text]
        stitched = stitch_frames(
            frames_lines,
            fuzzy=0.85,
            strip_boilerplate=True,
            preserve_unmatched_overlap=True,
            preserve_whitespace=True,
        )
        transcript = stitched.text()
        raw_path = out_dir / "transcript.raw.md"
        _atomic_write_text(raw_path, transcript)
        logger.info("Stitched transcript: %d lines", len(stitched.lines))

        # 4. Optional LLM seam/indent cleanup ────────────────────────────────
        clean_path = out_dir / "transcript.md"
        if config.reconstruction.enabled and transcript.strip():
            try:
                cleaned = _cleanup_transcript(transcript, config)
                _atomic_write_text(clean_path, cleaned)
            except Exception as exc:
                logger.error("LLM cleanup failed (%s); raw stitched transcript kept", exc)
                _atomic_write_text(clean_path, transcript)
        else:
            _atomic_write_text(clean_path, transcript)
        (out_dir / "transcript.partial.md").unlink(missing_ok=True)

        tiled_frames = sum(1 for record in ocr_records if record.get("ocr_tiled"))
        uncertain_tiled_frames = sum(
            1
            for record in ocr_records
            if record.get("ocr_status") == "complete_tiled_uncertain"
        )
        uncertain_tiled_frame_ids = [
            record.get("frame_id")
            for record in ocr_records
            if record.get("ocr_status") == "complete_tiled_uncertain"
        ]
        tile_requests = sum(
            int(record.get("ocr_tile_requests") or 0) for record in ocr_records
        )
        tile_unmatched = sum(
            int(record.get("ocr_unmatched_seams") or 0) for record in ocr_records
        )
        meta = {
            "video": str(Path(video_path).resolve()),
            "frames_selected": len(frames),
            "frames_with_text": non_empty,
            "frames_tiled": tiled_frames,
            "frames_tiled_uncertain": uncertain_tiled_frames,
            "uncertain_tiled_frame_ids": uncertain_tiled_frame_ids,
            "tile_requests": tile_requests,
            "tile_unmatched_seams": tile_unmatched,
            "stitch_unmatched_seams": stitched.unmatched_seams,
            "ocr_model": ocr.model,
            "ocr_max_tokens": config.ocr.max_tokens,
            "ocr_contract_fingerprint": contract_fingerprint,
            "llm_model": (
                resolve_llm_model(config.reconstruction)
                if config.reconstruction.enabled
                else None
            ),
            "transcript_path": str(clean_path),
            "raw_transcript_path": str(raw_path),
            "resume_dir": str(data_dir.resolve()),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        _atomic_write_json(out_dir / "transcribe_meta.json", meta)
        write_progress()
        return {"stage": "done", **meta}
