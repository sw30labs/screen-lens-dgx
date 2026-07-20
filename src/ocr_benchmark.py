"""Read-only benchmark harness for already-served OCR models."""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import InferenceBackend, OCRConfig
from .ocr import VerbatimOCR, _NO_IMAGE_RE
from .omlx_client import normalize_api_base_url


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class OCRBenchmarkTarget:
    """One already-running OpenAI-compatible OCR endpoint/model pair."""

    model: str
    base_url: str
    backend: InferenceBackend = InferenceBackend.vllm
    api_key: str | None = None


def select_representative_frames(
    sources: Iterable[str | Path],
    *,
    limit: int = 12,
) -> list[Path]:
    """Discover images and uniformly sample them in stable path order."""
    if limit < 1:
        raise ValueError("frame limit must be at least 1")

    discovered: dict[Path, None] = {}
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            candidates = (
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates = iter((path,))
        else:
            candidates = iter(())
        for candidate in candidates:
            discovered[candidate.resolve()] = None

    frames = sorted(discovered, key=lambda item: str(item))
    if len(frames) <= limit:
        return frames
    if limit == 1:
        return [frames[len(frames) // 2]]

    # Include both ends and spread the remaining samples across the recording.
    indices = [round(i * (len(frames) - 1) / (limit - 1)) for i in range(limit)]
    return [frames[index] for index in indices]


def output_sanity(text: str) -> dict[str, object]:
    """Compute cheap, ground-truth-free warning signals for one OCR output."""
    stripped = text.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    normalized_lines = [re.sub(r"\s+", " ", line.strip()).lower() for line in lines]
    unique_line_ratio = (
        len(set(normalized_lines)) / len(normalized_lines)
        if normalized_lines
        else 0.0
    )
    repeated_line_ratio = 1.0 - unique_line_ratio if normalized_lines else 0.0
    looks_repetitive = len(normalized_lines) >= 4 and repeated_line_ratio >= 0.5
    no_image_refusal = bool(_NO_IMAGE_RE.search(stripped))
    nonempty = bool(stripped)
    warnings = []
    if not nonempty:
        warnings.append("empty")
    if no_image_refusal:
        warnings.append("no-image-refusal")
    if looks_repetitive:
        warnings.append("repetitive-lines")
    if "\ufffd" in stripped:
        warnings.append("replacement-characters")

    return {
        "status": "ok" if not warnings else "warning",
        "warnings": warnings,
        "nonempty": nonempty,
        "characters": len(stripped),
        "lines": len(lines),
        "unique_line_ratio": round(unique_line_ratio, 4),
        "no_image_refusal": no_image_refusal,
        "replacement_characters": stripped.count("\ufffd"),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def benchmark_ocr_targets(
    frames: Iterable[str | Path],
    targets: Iterable[OCRBenchmarkTarget],
    *,
    output_path: str | Path,
    timeout_seconds: float = 600.0,
    max_tokens: int = 4096,
    concurrency: int = 2,
    warmup: bool = True,
) -> dict[str, object]:
    """Benchmark served targets sequentially and write a detailed JSON report.

    This intentionally performs no downloads and starts/stops no services.
    Targets remain sequential to avoid cross-model GPU contention, while frame
    concurrency reflects the actual ScreenLens pipeline load. By default, one
    unmeasured first-frame request warms each already-served target so lazy
    compilation and cache initialization do not dominate its latency.
    """
    selected = [Path(frame).resolve() for frame in frames]
    if not selected:
        raise ValueError("no image frames were selected for the OCR benchmark")
    if concurrency < 1:
        raise ValueError("OCR benchmark concurrency must be at least 1")
    target_list = list(targets)
    if not target_list:
        raise ValueError("at least one OCR benchmark target is required")

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_available": False,
        "quality_note": (
            "Sanity fields detect empty, blind, or repetitive output only; "
            "they are not OCR accuracy scores."
        ),
        "warmup_enabled": warmup,
        "frames": [str(frame) for frame in selected],
        "targets": [],
    }
    target_reports: list[dict[str, object]] = []

    for target in target_list:
        workers = min(concurrency, len(selected))
        warmup_seconds = 0.0
        setup_started = time.perf_counter()
        try:
            config = OCRConfig(
                backend=target.backend,
                base_url=target.base_url,
                model=target.model,
                api_key=target.api_key,
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                concurrency=concurrency,
                deterministic_backstop=False,
            )
            ocr = VerbatimOCR(config)
            ocr.assert_vision_capable()
            if warmup:
                warmup_started = time.perf_counter()
                # Warm-up establishes connectivity/vision and initializes lazy
                # kernels. A dense first frame may legitimately hit the output
                # cap; truncation is scored on its measured request below.
                ocr.ocr_frame(str(selected[0]))
                warmup_seconds = time.perf_counter() - warmup_started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            frame_reports = [
                {
                    "frame": str(frame),
                    "latency_seconds": 0.0,
                    "characters_per_second": 0.0,
                    "success": False,
                    "error": f"target setup failed: {error}",
                    "sanity": output_sanity(""),
                    "output": "",
                }
                for frame in selected
            ]
            target_reports.append(
                {
                    "model": target.model,
                    "base_url": normalize_api_base_url(target.base_url),
                    "backend": target.backend.value,
                    "request_family": "unavailable",
                    "concurrency": concurrency,
                    "workers_used": workers,
                    "warmup_seconds": round(warmup_seconds, 6),
                    "setup_error": error,
                    "summary": {
                        "frames": len(selected),
                        "successes": 0,
                        "failures": len(selected),
                        "nonempty_outputs": 0,
                        "wall_seconds": round(
                            time.perf_counter() - setup_started, 6
                        ),
                        "frames_per_second": 0.0,
                        "successful_frames_per_second": 0.0,
                        "mean_latency_seconds": 0.0,
                        "median_latency_seconds": 0.0,
                        "p95_latency_seconds": 0.0,
                        "output_characters": 0,
                    },
                    "frames": frame_reports,
                }
            )
            continue

        target_started = time.perf_counter()

        def benchmark_frame(frame: Path) -> dict[str, object]:
            started = time.perf_counter()
            output = ""
            error: str | None = None
            try:
                output = ocr.ocr_frame(str(frame), require_complete=True)
            except Exception as exc:  # report failures without aborting comparison
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            sanity = output_sanity(output)
            return {
                "frame": str(frame),
                "latency_seconds": round(latency, 6),
                "characters_per_second": round(len(output) / latency, 3)
                if latency > 0
                else 0.0,
                "success": error is None,
                "error": error,
                "sanity": sanity,
                "output": output,
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            frame_reports = list(pool.map(benchmark_frame, selected))

        wall_seconds = time.perf_counter() - target_started
        successes = [item for item in frame_reports if item["success"]]
        latencies = [float(item["latency_seconds"]) for item in successes]
        nonempty = sum(
            bool(item["sanity"]["nonempty"])  # type: ignore[index]
            for item in successes
        )
        target_reports.append(
            {
                "model": target.model,
                "base_url": normalize_api_base_url(target.base_url),
                "backend": target.backend.value,
                "request_family": ocr.request_profile.family,
                "concurrency": concurrency,
                "workers_used": workers,
                "warmup_seconds": round(warmup_seconds, 6),
                "setup_error": None,
                "summary": {
                    "frames": len(selected),
                    "successes": len(successes),
                    "failures": len(selected) - len(successes),
                    "nonempty_outputs": nonempty,
                    "wall_seconds": round(wall_seconds, 6),
                    "frames_per_second": round(len(selected) / wall_seconds, 4)
                    if wall_seconds > 0
                    else 0.0,
                    "successful_frames_per_second": round(
                        len(successes) / wall_seconds, 4
                    )
                    if wall_seconds > 0
                    else 0.0,
                    "mean_latency_seconds": round(statistics.fmean(latencies), 6)
                    if latencies
                    else 0.0,
                    "median_latency_seconds": round(statistics.median(latencies), 6)
                    if latencies
                    else 0.0,
                    "p95_latency_seconds": round(_percentile(latencies, 0.95), 6),
                    "output_characters": sum(len(str(item["output"])) for item in successes),
                },
                "frames": frame_reports,
            }
        )

    report["targets"] = target_reports
    destination = Path(output_path).expanduser().resolve()
    report["output_path"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
