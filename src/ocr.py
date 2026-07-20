"""
Verbatim OCR pass.

Transcribes each frame character-for-character with a VISION model through a
vLLM or oMLX OpenAI-compatible server. Unlike captioning, this never describes
— it copies.

Defenses baked in after the original failure (a text-only model was used for
vision and silently returned "no image provided" for all 173 frames):

  * Hard capability guard — refuse to run if the configured model isn't
    vision-capable, and a live probe that catches "no image" refusals.
  * Anti-loop sampler controls (repetition_penalty / no_repeat_ngram_size).
  * Optional Apple Vision deterministic backstop (ocrmac) for code, where VLMs
    are known to hallucinate tokens.
"""
from __future__ import annotations

import logging
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from typing import Any, Callable

from PIL import Image

from .config import OCRConfig
from .omlx_client import (
    InferenceClient,
    InferenceTruncatedError,
    normalized_model_id,
    resolve_ocr_api_key,
    resolve_ocr_base_url,
    resolve_ocr_context,
    resolve_ocr_model,
    resolve_role_backend,
)
from .stitch import stitch_frames

logger = logging.getLogger("screenlens.ocr")

# Sentinels a blind/text-only model emits when it can't see the image.
_NO_IMAGE_RE = re.compile(
    r"\b(no (image|video frame|picture)\b.*(provided|attached|been))"
    r"|(please (attach|provide|upload).{0,40}(image|frame))"
    r"|(i (cannot|can't|am unable to) see (an|the|any) image)",
    re.IGNORECASE | re.DOTALL,
)

_EMPTY_MARKERS = {"[no text]", "[notext]", "no text", ""}


@dataclass(frozen=True)
class OCRRequestProfile:
    """Model-specific OpenAI chat shape and sampling controls for OCR."""

    family: str
    system_prompt: str | None
    user_prompt: str | None
    image_first: bool
    temperature: float
    extra: dict[str, Any]


@dataclass(frozen=True)
class OCRFrameResult:
    """Auditable result of one strict verbatim frame request.

    ``complete`` means every accepted generation terminated normally: callers
    must never treat a failed or length-truncated prefix as transcription. Seam
    certainty is reported separately by ``status`` and ``unmatched_seams``.
    Empty text can still be complete when no legible text is visible.
    """

    text: str
    status: str
    complete: bool
    tiled: bool = False
    full_frame_attempted: bool = True
    tile_requests: int = 0
    tile_leaves: int = 0
    tile_max_depth: int = 0
    unmatched_seams: int = 0
    error: str | None = None


@dataclass
class _TileStats:
    requests: int = 0
    leaves: int = 0
    max_depth: int = 0


class OCRTileRequestError(RuntimeError):
    """A tile-stage failure carrying the work completed before it stopped."""

    def __init__(self, message: str, stats: _TileStats):
        self.tile_requests = stats.requests
        self.tile_leaves = stats.leaves
        self.tile_max_depth = stats.max_depth
        super().__init__(message)


class OCRTileExhaustedError(OCRTileRequestError):
    """Raised when bounded tile subdivision still cannot return complete OCR."""


def _annotate_frame_failure(
    exc: Exception,
    *,
    full_frame_attempted: bool,
    tiled: bool,
) -> None:
    """Attach auditable stage facts without changing the public exception type."""
    try:
        setattr(exc, "screenlens_full_frame_attempted", full_frame_attempted)
        setattr(exc, "screenlens_tiled", tiled)
    except (AttributeError, TypeError):
        pass


def ocr_request_profile(model: str, config: OCRConfig) -> OCRRequestProfile:
    """Return the vendor-recommended request contract for an OCR model.

    Purpose-built OCR models use narrow chat templates and can reject or lose
    accuracy with Qwen-specific sampler fields. Unknown/general vision models
    retain ScreenLens' existing verbatim prompt and anti-loop defenses.
    """
    compact = normalized_model_id(model).replace("-", "")
    if "lightonocr" in compact:
        # LightOnOCR-2 is trained for an image-only user turn. Its model card
        # recommends light sampling rather than deterministic decoding.
        return OCRRequestProfile(
            family="lightonocr-2",
            system_prompt=None,
            user_prompt=None,
            image_first=True,
            temperature=0.2,
            extra={"top_p": 0.9},
        )
    if "glmocr" in compact:
        return OCRRequestProfile(
            family="glm-ocr",
            system_prompt=None,
            user_prompt="Text Recognition:",
            image_first=True,
            temperature=0.0,
            extra={},
        )
    if "paddleocrvl" in compact:
        return OCRRequestProfile(
            family="paddleocr-vl",
            system_prompt=None,
            user_prompt="OCR:",
            image_first=True,
            temperature=0.0,
            extra={},
        )

    extra: dict[str, Any] = {
        "repetition_penalty": config.repetition_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size or None,
    }
    if config.disable_thinking:
        # Qwen-style reasoning models otherwise spend the OCR budget thinking.
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    return OCRRequestProfile(
        family="generic-vision",
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        image_first=False,
        temperature=config.temperature,
        extra=extra,
    )


class VerbatimOCR:
    """Transcribe frames verbatim through the selected vision server."""

    def __init__(self, config: OCRConfig):
        self.config = config
        if config.deterministic_backstop and platform.system() != "Darwin":
            raise RuntimeError(
                "The deterministic OCR backstop uses Apple Vision and is only "
                "available on macOS; omit --deterministic on DGX Spark."
            )
        self.model = resolve_ocr_model(config)
        self.client = InferenceClient.from_endpoint(
            base_url=resolve_ocr_base_url(config),
            model=self.model,
            api_key=resolve_ocr_api_key(config),
            backend=resolve_role_backend(config),
            timeout=config.timeout_seconds,
            context_size=resolve_ocr_context(config),
            default_max_tokens=config.max_tokens,
            default_temperature=config.temperature,
        )
        self.request_profile = ocr_request_profile(self.model, config)
        self._extra = self.request_profile.extra
        self._probed = False
        self._probe_path: str | None = None
        self._probe_raw: str | None = None
        self._probe_lock = Lock()
        # Once a full frame consumes the model context, avoid making the same
        # multi-minute doomed request for every later frame in this process.
        self._tile_first = Event()

    def _chat(
        self,
        image_path: str,
        *,
        max_tokens: int,
        require_complete: bool = False,
    ) -> str:
        """Send one frame using this model family's native OCR contract."""
        profile = self.request_profile
        return self.client.chat(
            profile.system_prompt,
            profile.user_prompt,
            images=[image_path],
            image_first=profile.image_first,
            max_tokens=max_tokens,
            temperature=profile.temperature,
            extra=profile.extra,
            require_complete=require_complete,
            preserve_whitespace=require_complete,
        )

    # ── Capability checks ────────────────────────────────────────────────────

    def assert_vision_capable(self) -> None:
        """Refuse to run against a text-only model (the original bug)."""
        supports = self.client.model_supports_vision()
        if supports is False and self.config.require_vision_model:
            raise RuntimeError(
                f"OCR model '{self.model}' looks text-only. Verbatim OCR sends "
                f"images, so it would read every frame blind (this is exactly the "
                f"bug that produced 173 empty captions). Set ocr.model or the "
                f"selected provider's OCR/model environment variable to a vision "
                f"model (VL / vision / omni / *-OCR). To override, set "
                f"ocr.require_vision_model=false."
            )
        if supports is None:
            logger.warning(
                "Could not verify '%s' is vision-capable from its name; relying on "
                "the live probe. If OCR returns '[NO TEXT]' for every frame, the "
                "model is text-only.", self.model,
            )

    def probe(self, sample_image_path: str) -> None:
        """Validate vision and cache the full first-frame transcription."""
        if self._probed:
            return
        try:
            raw = self._chat(sample_image_path, max_tokens=self.config.max_tokens)
        except Exception as exc:
            raise RuntimeError(f"OCR probe call failed: {exc}") from exc
        if _NO_IMAGE_RE.search(raw or ""):
            raise RuntimeError(
                f"OCR model '{self.model}' responded as if no image was sent "
                f"(\"{(raw or '').strip()[:80]}…\"). It is not actually seeing "
                f"frames. Pick a vision-capable model on your "
                f"{self.client.backend.value} server."
            )
        with self._probe_lock:
            self._probe_path = sample_image_path
            self._probe_raw = raw
            self._probed = True

    # ── Per-frame OCR ────────────────────────────────────────────────────────

    def ocr_frame(self, image_path: str, *, require_complete: bool = False) -> str:
        """Transcribe one image, optionally rejecting length-truncated output."""
        raw: str | None = None
        with self._probe_lock:
            if self._probe_path == image_path:
                if not require_complete:
                    raw = self._probe_raw
                # A completeness-sensitive caller must issue its own checked
                # request instead of consuming an unchecked probe response.
                self._probe_path = None
                self._probe_raw = None
        if raw is None:
            raw = self._chat(
                image_path,
                max_tokens=self.config.max_tokens,
                require_complete=require_complete,
            )
        return self._normalize_text(image_path, raw, reconcile=True)

    def _normalize_text(
        self,
        image_path: str,
        raw: str | None,
        *,
        reconcile: bool,
        preserve_whitespace: bool = False,
    ) -> str:
        """Normalize one complete model response without changing its content."""
        text = raw or ""
        if not preserve_whitespace:
            text = text.strip()
        if _NO_IMAGE_RE.search(text):
            raise RuntimeError(
                f"OCR model returned a 'no image' refusal mid-run — it is not "
                f"vision-capable: {text[:80]}"
            )
        # Strip an accidental outer ``` fence wrapping the whole answer.
        fenced = text.strip()
        if (
            not preserve_whitespace
            and fenced.startswith("```")
            and fenced.endswith("```")
            and fenced.count("```") == 2
        ):
            match = re.fullmatch(
                r"```[a-zA-Z0-9_+-]*\r?\n?(.*?)\r?\n?```",
                fenced,
                flags=re.DOTALL,
            )
            if match is not None:
                text = match.group(1)
                if not preserve_whitespace:
                    text = text.strip()
        if text.strip().lower() in _EMPTY_MARKERS:
            return ""
        if reconcile and self.config.deterministic_backstop:
            text = self._reconcile_with_apple_vision(image_path, text)
        return text

    def ocr_frame_verbatim(self, image_path: str) -> OCRFrameResult:
        """OCR one frame without ever accepting a truncated response.

        A normal full-frame call gets the configured (deliberately large)
        budget. If the server reports ``finish_reason=length``, the incomplete
        prefix is discarded and the original image is transcribed as bounded,
        overlapping horizontal bands. Only truncation activates tiling; auth,
        transport, model, and malformed-response errors remain visible.
        """
        full_frame_attempted = not (
            self.config.tile_fallback_enabled and self._tile_first.is_set()
        )
        if full_frame_attempted:
            try:
                raw = self._chat(
                    image_path,
                    max_tokens=self.config.max_tokens,
                    require_complete=True,
                )
                text = self._normalize_text(
                    image_path,
                    raw,
                    reconcile=True,
                    preserve_whitespace=True,
                )
                return OCRFrameResult(
                    text=text,
                    status="complete" if text else "empty",
                    complete=True,
                )
            except InferenceTruncatedError as exc:
                if not self.config.tile_fallback_enabled:
                    _annotate_frame_failure(
                        exc,
                        full_frame_attempted=True,
                        tiled=False,
                    )
                    raise
                self._tile_first.set()
                logger.warning(
                    "Full-frame OCR exhausted the context for %s; discarding "
                    "the prefix and retrying with overlapping horizontal tiles.",
                    image_path,
                )
            except Exception as exc:
                _annotate_frame_failure(
                    exc,
                    full_frame_attempted=True,
                    tiled=False,
                )
                raise

        try:
            text, stats, unmatched = self._ocr_tiled(image_path)
        except Exception as exc:
            _annotate_frame_failure(
                exc,
                full_frame_attempted=full_frame_attempted,
                tiled=True,
            )
            raise
        if self.config.deterministic_backstop:
            # Reconcile once against the original image. Reconciling individual
            # overlapping crops would multiply deterministic OCR and seam noise.
            text = self._reconcile_with_apple_vision(image_path, text)
        return OCRFrameResult(
            text=text,
            status=(
                "complete_tiled_uncertain"
                if text and unmatched
                else "complete_tiled" if text else "empty"
            ),
            complete=True,
            tiled=True,
            full_frame_attempted=full_frame_attempted,
            tile_requests=stats.requests,
            tile_leaves=stats.leaves,
            tile_max_depth=stats.max_depth,
            unmatched_seams=unmatched,
        )

    def prefer_tiles(self) -> None:
        """Skip a known-doomed full-frame request on later/resumed frames."""
        if self.config.tile_fallback_enabled:
            self._tile_first.set()

    def _ocr_tiled(self, image_path: str) -> tuple[str, _TileStats, int]:
        """Return complete OCR assembled from bounded horizontal bands."""
        source_path = Path(image_path)
        stats = _TileStats()
        with Image.open(source_path) as opened:
            image = opened.copy()
        width, height = image.size
        if width < 1 or height < 1:
            raise RuntimeError(f"OCR image has invalid dimensions: {source_path}")

        # Keep initial logical cores reasonably tall, while retaining at least
        # two bands for ordinary images. The minimum-height guard applies to
        # recursive subdivision, where an endlessly shrinking crop is unsafe.
        rows_by_height = max(2, height // self.config.tile_min_core_height)
        rows = min(self.config.tile_rows, rows_by_height)
        boundaries = [round(i * height / rows) for i in range(rows + 1)]

        with TemporaryDirectory(prefix="screenlens-ocr-tiles-") as tmp:
            tmp_dir = Path(tmp)
            leaf_texts: list[str] = []
            for row in range(rows):
                leaf_texts.extend(
                    self._ocr_tile_region(
                        image=image,
                        original_path=source_path,
                        tmp_dir=tmp_dir,
                        core_top=boundaries[row],
                        core_bottom=boundaries[row + 1],
                        depth=1,
                        stats=stats,
                    )
                )

        if not any(text.strip() for text in leaf_texts):
            return "", stats, 0
        stitched = stitch_frames(
            [text.split("\n") for text in leaf_texts],
            fuzzy=0.80,
            strip_boilerplate=False,
            preserve_unmatched_overlap=True,
            preserve_whitespace=True,
        )
        if stitched.unmatched_seams:
            logger.warning(
                "Tile OCR for %s had %d unmatched seam(s); both sides were "
                "preserved for verbatim safety.",
                image_path,
                stitched.unmatched_seams,
            )
        return stitched.text() if stitched.lines else "", stats, stitched.unmatched_seams

    def _ocr_tile_region(
        self,
        *,
        image: Image.Image,
        original_path: Path,
        tmp_dir: Path,
        core_top: int,
        core_bottom: int,
        depth: int,
        stats: _TileStats,
    ) -> list[str]:
        """OCR one logical band, recursively bisecting it once if necessary."""
        if stats.requests >= self.config.tile_max_requests:
            raise OCRTileExhaustedError(
                f"Tile OCR exhausted its {self.config.tile_max_requests}-request "
                f"limit for {original_path}; no truncated prefix was kept.",
                stats,
            )

        core_height = max(1, core_bottom - core_top)
        overlap = max(
            self.config.tile_min_overlap_pixels,
            round(core_height * self.config.tile_overlap_ratio),
        )
        crop_top = max(0, core_top - overlap)
        crop_bottom = min(image.height, core_bottom + overlap)
        tile_path = tmp_dir / (
            f"tile_{stats.requests:02d}_d{depth}_"
            f"{core_top:06d}-{core_bottom:06d}.png"
        )
        # PNG is intentional: JPEG recompression can change small glyphs and
        # makes a retry less faithful than the selected source frame.
        image.crop((0, crop_top, image.width, crop_bottom)).save(tile_path, "PNG")

        stats.requests += 1
        stats.max_depth = max(stats.max_depth, depth)
        cap = (
            self.config.tile_max_tokens
            if depth == 1
            else self.config.tile_retry_max_tokens
        )
        cap = min(self.config.max_tokens, cap)
        try:
            raw = self._chat(
                str(tile_path),
                max_tokens=cap,
                require_complete=True,
            )
        except InferenceTruncatedError as exc:
            can_split = (
                depth < self.config.tile_max_depth
                and core_height >= self.config.tile_min_core_height * 2
                and stats.requests + 2 <= self.config.tile_max_requests
            )
            if not can_split:
                raise OCRTileExhaustedError(
                    f"Tile OCR still truncated at depth {depth} for rows "
                    f"{core_top}:{core_bottom} of {original_path}; incomplete "
                    "output was discarded.",
                    stats,
                ) from exc
            midpoint = core_top + core_height // 2
            return self._ocr_tile_region(
                image=image,
                original_path=original_path,
                tmp_dir=tmp_dir,
                core_top=core_top,
                core_bottom=midpoint,
                depth=depth + 1,
                stats=stats,
            ) + self._ocr_tile_region(
                image=image,
                original_path=original_path,
                tmp_dir=tmp_dir,
                core_top=midpoint,
                core_bottom=core_bottom,
                depth=depth + 1,
                stats=stats,
            )
        except Exception as exc:
            raise OCRTileRequestError(
                f"Tile OCR request failed at depth {depth} for rows "
                f"{core_top}:{core_bottom} of {original_path}: {exc}",
                stats,
            ) from exc

        try:
            text = self._normalize_text(
                str(tile_path),
                raw,
                reconcile=False,
                preserve_whitespace=True,
            )
        except Exception as exc:
            raise OCRTileRequestError(
                f"Tile OCR returned an unusable response at depth {depth} for "
                f"rows {core_top}:{core_bottom} of {original_path}: {exc}",
                stats,
            ) from exc
        stats.leaves += 1
        return [text]

    def ocr_frames_verbatim(
        self,
        image_paths: list[str],
        *,
        on_result: Callable[[int, OCRFrameResult], None] | None = None,
    ) -> list[OCRFrameResult]:
        """Strict OCR batch with ordered results and completion callbacks.

        The first frame runs synchronously so a dense-frame truncation trips the
        tile-first circuit before the executor launches the rest. Remaining
        callbacks run in this coordinator thread as futures complete, making it
        safe for callers to atomically persist progress without worker races.
        """
        if not image_paths:
            return []
        self.assert_vision_capable()
        results: list[OCRFrameResult | None] = [None] * len(image_paths)

        def failed_result(exc: Exception, *, full_frame_attempted: bool) -> OCRFrameResult:
            actual_full_frame_attempted = bool(
                getattr(
                    exc,
                    "screenlens_full_frame_attempted",
                    full_frame_attempted,
                )
            )
            tiled = bool(
                getattr(
                    exc,
                    "screenlens_tiled",
                    isinstance(exc, OCRTileRequestError)
                    or not actual_full_frame_attempted,
                )
            )
            return OCRFrameResult(
                text="",
                status="failed",
                complete=False,
                tiled=tiled,
                full_frame_attempted=actual_full_frame_attempted,
                tile_requests=int(getattr(exc, "tile_requests", 0)),
                tile_leaves=int(getattr(exc, "tile_leaves", 0)),
                tile_max_depth=int(getattr(exc, "tile_max_depth", 0)),
                error=str(exc),
            )

        def capture(index: int) -> tuple[OCRFrameResult, Exception | None]:
            path = image_paths[index]
            full_frame_attempted = not (
                self.config.tile_fallback_enabled and self._tile_first.is_set()
            )
            try:
                return self.ocr_frame_verbatim(path), None
            except Exception as exc:
                logger.error("Strict OCR failed on %s: %s", path, exc)
                return (
                    failed_result(
                        exc,
                        full_frame_attempted=full_frame_attempted,
                    ),
                    None if isinstance(exc, OCRTileExhaustedError) else exc,
                )

        first_full_frame_attempted = not (
            self.config.tile_fallback_enabled and self._tile_first.is_set()
        )
        try:
            first = self.ocr_frame_verbatim(image_paths[0])
        except Exception as exc:
            logger.error("Strict OCR failed on %s: %s", image_paths[0], exc)
            first = failed_result(
                exc,
                full_frame_attempted=first_full_frame_attempted,
            )
            results[0] = first
            if on_result is not None:
                on_result(0, first)
            # Tile exhaustion is frame-specific: keep checkpointing subsequent
            # frames. A blind model, authentication/transport failure, or a
            # disabled fallback would otherwise repeat the same doomed call for
            # the entire recording, so stop after persisting the first failure.
            if not isinstance(exc, OCRTileExhaustedError):
                raise
        results[0] = first
        if on_result is not None and first.complete:
            on_result(0, first)

        workers = max(1, min(self.config.concurrency, len(image_paths) - 1))
        if len(image_paths) > 1:
            # Daemon workers matter for a local 600-second HTTP timeout: the
            # stdlib ThreadPoolExecutor uses non-daemon threads and its context
            # manager waits for every in-flight/queued call after Ctrl+C. A
            # bounded set of daemon workers lets the CLI stop promptly while
            # still delivering callbacks from this coordinator during normal
            # operation. Workers only claim another index after finishing one.
            work: Queue[int] = Queue()
            completed: Queue[tuple[int, OCRFrameResult, Exception | None]] = Queue()
            stop = Event()
            initial_end = min(len(image_paths), workers + 1)
            for index in range(1, initial_end):
                work.put(index)
            next_index = initial_end

            def worker() -> None:
                while not stop.is_set():
                    try:
                        index = work.get(timeout=0.25)
                    except Empty:
                        continue
                    if stop.is_set():
                        return
                    result, terminal_error = capture(index)
                    completed.put((index, result, terminal_error))

            threads = [
                Thread(
                    target=worker,
                    name=f"screenlens-ocr-{number + 1}",
                    daemon=True,
                )
                for number in range(workers)
            ]
            for thread in threads:
                thread.start()

            received = 0
            try:
                while received < len(image_paths) - 1:
                    try:
                        index, result, terminal_error = completed.get(timeout=0.25)
                    except Empty:
                        continue
                    results[index] = result
                    received += 1
                    if on_result is not None:
                        on_result(index, result)
                    if terminal_error is not None:
                        stop.set()
                        raise terminal_error
                    if next_index < len(image_paths):
                        work.put(next_index)
                        next_index += 1
            except BaseException:
                stop.set()
                # Persist results that reached the coordinator queue just before
                # the interrupt; never make the user repay already-finished OCR.
                while True:
                    try:
                        index, result, _terminal_error = completed.get_nowait()
                    except Empty:
                        break
                    if results[index] is None:
                        results[index] = result
                        if on_result is not None:
                            on_result(index, result)
                raise
            finally:
                stop.set()
            for thread in threads:
                thread.join()

        # All slots are assigned above; the assertion documents that invariant
        # for type checkers without changing caller-visible ordering.
        assert all(result is not None for result in results)
        return [result for result in results if result is not None]

    def ocr_frames(self, image_paths: list[str]) -> list[str]:
        if not image_paths:
            return []
        self.assert_vision_capable()
        self.probe(image_paths[0])
        workers = max(1, min(self.config.concurrency, len(image_paths)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._safe_ocr, image_paths))

    def _safe_ocr(self, path: str) -> str:
        try:
            return self.ocr_frame(path)
        except Exception as exc:  # keep ordering; one bad frame shouldn't kill the run
            logger.error("OCR failed on %s: %s", path, exc)
            return ""

    # ── Deterministic backstop (Apple Vision) ────────────────────────────────

    def _reconcile_with_apple_vision(self, image_path: str, vlm_text: str) -> str:
        """Flag lines where the VLM disagrees with Apple Vision's literal read.

        Apple Vision never hallucinates a fake line (but mangles indentation), so
        on a strong character-level disagreement we trust the deterministic read
        and annotate the seam for human review. Best-effort: silently skip if
        ocrmac/Apple Vision isn't available.
        """
        try:
            lines = apple_vision_lines(image_path)
        except Exception as exc:
            logger.debug("Apple Vision backstop unavailable (%s); keeping VLM text", exc)
            return vlm_text
        if not lines:
            return vlm_text
        from difflib import SequenceMatcher

        det = "\n".join(lines)
        ratio = SequenceMatcher(None, vlm_text, det).ratio()
        if ratio < 0.55:
            # Large divergence — surface both so nothing is silently invented.
            return (
                vlm_text
                + "\n\n<!-- OCR-DISAGREEMENT: Apple Vision read this region "
                "differently; verify -->\n"
                + det
            )
        return vlm_text


def apple_vision_lines(image_path: str) -> list[str]:
    """Deterministic OCR via Apple Vision (ocrmac), language correction OFF.

    Correction OFF is mandatory: the default corrects toward dictionary words,
    which is wrong for code/identifiers. Returns lines in top-to-bottom order.
    Raises if ocrmac/pyobjc isn't installed (macOS only).
    """
    from ocrmac import ocrmac  # type: ignore

    annotations = ocrmac.OCR(
        image_path,
        recognition_level="accurate",
        language_preference=["en-US"],
    ).recognize()
    # annotations: list of (text, confidence, bbox[x,y,w,h]) with y from bottom.
    items = []
    for text, conf, bbox in annotations:
        x, y, w, h = bbox
        items.append((round(1.0 - y, 4), round(x, 4), text))  # sort top→bottom, left→right
    items.sort()
    return [t for _, _, t in items]
