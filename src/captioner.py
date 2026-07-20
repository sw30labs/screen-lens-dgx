"""
Frame Captioning Module.

Backends:
  1. **vllm** — DGX Spark's local OpenAI-compatible multimodal server.
  2. **omlx** — Optional OpenAI-compatible alias (legacy name; not a product path).
  3. **ollama** (fallback) — Any Ollama vision model.
     Works on any platform with Ollama installed.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .config import CaptioningConfig, CaptionBackend
from .omlx_client import InferenceClient, resolve_inference_model, validate_vision_model

logger = logging.getLogger("screenlens.captioner")


# ── OpenAI-compatible vision backends (vLLM / oMLX) ───────────────

class OpenAICompatibleCaptioner:
    """Caption frames through the selected direct inference server."""

    def __init__(self, config: CaptioningConfig):
        self.config = config
        validate_vision_model(resolve_inference_model(config))
        self._client = InferenceClient(config)

    def caption(
        self,
        image_path: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        require_complete: bool = False,
    ) -> str:
        """Generate a caption for a single frame."""
        extra = {
            "repetition_penalty": self.config.repetition_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size or None,
        }
        if self.config.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return self._client.chat(
            self.config.system_prompt,
            self.config.user_prompt,
            images=[image_path],
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            temperature=(
                temperature if temperature is not None else self.config.temperature
            ),
            extra=extra,
            require_complete=require_complete,
        )

    def _caption_with_retry(self, image_path: str) -> str:
        """Caption one frame without allowing its failure to poison a batch."""
        try:
            return self.caption(image_path)
        except Exception as exc:
            last_error = exc

        retry_max_tokens = min(
            self.config.max_tokens,
            self.config.retry_max_tokens,
        )
        for attempt in range(1, self.config.retry_attempts + 1):
            logger.warning(
                "Caption request failed for %s (%s); retrying %s/%s with "
                "max_tokens=%s",
                Path(image_path).name,
                last_error,
                attempt,
                self.config.retry_attempts,
                retry_max_tokens,
            )
            try:
                return self.caption(
                    image_path,
                    max_tokens=retry_max_tokens,
                    temperature=0.0,
                    require_complete=True,
                )
            except Exception as exc:
                last_error = exc

        logger.error(
            "Caption request failed for %s after %s retries: %s",
            Path(image_path).name,
            self.config.retry_attempts,
            last_error,
        )
        return f"[Error captioning frame: {last_error}]"

    def caption_batch(self, image_paths: list[str]) -> list[str]:
        """Submit isolated concurrent requests and preserve input order."""
        if not image_paths:
            return []
        max_workers = max(1, min(self.config.batch_size, len(image_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self._caption_with_retry, image_paths))


# ── Ollama Backend ──────────────────────────────────────────────────────────

class OllamaCaptioner:
    """Caption frames using any Ollama vision model."""

    def __init__(self, config: CaptioningConfig):
        self.config = config

    def caption(self, image_path: str) -> str:
        """Generate a caption for a single frame."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
        )

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            ("system", self.config.system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": self.config.user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            ),
        ]

        response = llm.invoke(messages)
        return response.content

    def caption_batch(self, image_paths: list[str]) -> list[str]:
        """Sequential fallback: Ollama has no batch API, so we loop.

        Exists for call-site uniformity with OpenAICompatibleCaptioner.
        """
        return [self.caption(p) for p in image_paths]


# ── Factory + Batch Processing ──────────────────────────────────────────────

def _get_captioner(config: CaptioningConfig):
    """Return the appropriate captioner backend."""
    if config.backend in (CaptionBackend.vllm, CaptionBackend.omlx):
        return OpenAICompatibleCaptioner(config)

    return OllamaCaptioner(config)


def save_caption_record(frame: dict, output_dir: str) -> None:
    """Persist one caption immediately so interrupted runs retain their progress."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    caption_file = destination / f"caption_{frame['frame_id']:06d}.json"
    with open(caption_file, "w", encoding="utf-8") as f:
        json.dump(frame, f, indent=2)


def _save_combined_caption_records(results: list[dict], output_dir: str) -> None:
    """Persist the combined caption artifact after every frame has completed."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_file = destination / "all_captions.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_caption_records(results: list[dict], output_dir: str) -> None:
    """Persist per-frame and combined caption records using the standard schema."""
    for frame in results:
        save_caption_record(frame, output_dir)

    _save_combined_caption_records(results, output_dir)


def caption_frames(
    frames_meta: list[dict],
    config: Optional[CaptioningConfig] = None,
    output_dir: Optional[str] = None,
    record_transform: Optional[Callable[[dict], dict]] = None,
) -> list[dict]:
    """
    Generate captions for all extracted frames.

    Drives the captioner via ``caption_batch`` in chunks of ``config.batch_size``.
    For vLLM/oMLX each chunk becomes concurrent OpenAI-compatible requests.
    For Ollama it falls back to sequential per-image calls.

    Adds a 'caption' field to each frame metadata dict. ``record_transform``
    can attach derived fields before each result is appended or persisted; this
    keeps composed pipelines progressively restart-safe without duplicating the
    caption batching implementation.
    Optionally saves per-frame and combined caption JSON files to output_dir.
    """
    if config is None:
        config = CaptioningConfig()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    captioner = _get_captioner(config)
    backend_name = config.backend.value
    if config.backend in (CaptionBackend.vllm, CaptionBackend.omlx):
        model_name = resolve_inference_model(config).split("/")[-1]
    else:
        model_name = config.ollama_model

    batch_size = max(1, int(config.batch_size))
    print(f"Captioning with {backend_name} ({model_name}) — batch_size={batch_size}")

    results: list[dict] = []
    pbar = tqdm(total=len(frames_meta), desc="Captioning frames")

    for chunk_start in range(0, len(frames_meta), batch_size):
        chunk = frames_meta[chunk_start : chunk_start + batch_size]
        image_paths = [f["path"] for f in chunk]

        try:
            captions = captioner.caption_batch(image_paths)
        except Exception as e:
            logger.error(
                f"Batch caption failed (frames "
                f"{chunk[0]['frame_id']}–{chunk[-1]['frame_id']}): {e}"
            )
            captions = [f"[Error captioning frame: {e}]"] * len(chunk)

        # Defensive: pad/truncate if the backend returned the wrong count
        if len(captions) != len(chunk):
            logger.warning(
                f"caption_batch returned {len(captions)} results for {len(chunk)} frames; "
                f"padding with error markers"
            )
            captions = (captions + ["[Error: missing caption]"] * len(chunk))[: len(chunk)]

        for frame, caption in zip(chunk, captions):
            enriched = {**frame, "caption": caption}
            if record_transform is not None:
                enriched = record_transform(enriched)
            results.append(enriched)
            if output_dir:
                save_caption_record(enriched, output_dir)

        pbar.update(len(chunk))

    pbar.close()

    if output_dir:
        _save_combined_caption_records(results, output_dir)

    return results


# Compatibility names retained for callers from oMLX-only releases.
OMLXCaptioner = OpenAICompatibleCaptioner
VLLMCaptioner = OpenAICompatibleCaptioner
