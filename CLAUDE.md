# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ScreenLens-DGX** is a local video scene intelligence pipeline for **NVIDIA DGX Spark only**. It ingests screen recordings, extracts keyframes, generates dense captions with a vision-language model on **vLLM**, embeds them with OpenCLIP on **CUDA**, stores them in ChromaDB, and answers natural-language queries. Defaults are always vLLM + CUDA with two concurrent image requests. A second pipeline (`reconstruct`) rebuilds visible Python files, Markdown, PDFs, and GUI walkthroughs. A third pipeline (`transcribe`) OCRs frames character-for-character and stitches scrolling overlap in text space.

This is the DGX-only fork of dual-platform `screen-lens`. Do not reintroduce Apple Silicon / oMLX product defaults or a macOS launcher.

## Common Commands

```bash
# Install (prefer the DGX helper for CUDA 13 wheels)
./setup_and_run_dgx.sh setup
# or, inside .venv-dgx:
pip install -e ".[dev,tui]"

# Ingest with DGX defaults (vLLM/CUDA)
python -m src.cli ingest "video.mov"

# Optional Ollama captioning fallback
python -m src.cli ingest "video.mov" --backend ollama --strategy fixed_fps --fps 1.0

python -m src.cli ingest "video.mov" --backend vllm \
  --inference-model Qwen/Qwen3.6-27B-FP8 --device cuda --batch-size 2

python -m src.cli batch "/path/to/recordings/"
python -m src.cli search "What application is being demonstrated?"
python -m src.cli run "video.mov" "Summarize what happens"
python -m src.cli summarize
python -m src.cli reconstruct

# Verbatim transcription (cleanup OFF by default)
python -m src.cli transcribe "video.mov"
python -m src.cli transcribe "doc.mov" --cleanup

python -m src.cli models
python -m src.cli info

# Tests
pytest tests/ -v

# DGX Spark bootstrap, validation, and launcher
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run ingest input-videos/demo.mov
```

`ffmpeg` must be on PATH. Use `setup_and_run_dgx.sh` and `docs/DGX_SPARK.md`; the helper creates `.venv-dgx` with Python 3.12 and the checked CUDA 13 torch/torchvision wheels, then starts or reuses the exact NVIDIA Qwen model. Ollama is optional.

## Architecture

The codebase has two LangGraph `StateGraph` pipelines plus the straight-line transcribe path. They share one `ScreenLensConfig` and the provider-neutral `InferenceClient` in the legacy-named `omlx_client.py` module.

### Pipeline 1 — Ingest / Search (`src/pipeline.py`)

`StateGraph` over a `ScreenLensState` TypedDict. Three graph builders:

- `build_ingest_graph()`  — `ingest → caption → embed`
- `build_search_graph()`  — `search → summarize`
- `build_full_graph()`    — both chained end-to-end

| Node | Module | Purpose |
|---|---|---|
| `ingest_node` | `frame_extractor.py` | Hybrid keyframe detection (SSIM + pHash + HSV) or fixed-FPS fallback |
| `caption_node` | `captioner.py` | `OpenAICompatibleCaptioner` (vLLM default); `OllamaCaptioner` optional |
| `embed_node` | `embedder.py`, `vector_store.py` | OpenCLIP `ViT-B-32` on CUDA (or CPU in tests) → ChromaDB |
| `search_node` | `vector_store.py` | CLIP query encode + ChromaDB cosine search |
| `summarize_node` / `summarize_all_node` | `pipeline.py` | Summaries through the configured direct client |

### Pipeline 2 — Reconstruct (`src/reconstruct.py`)

Graph: `classify → plan → (parallel workers | sequential) → qa_reflect → save`, with a retry edge from `qa_reflect → plan` up to `MAX_QA_ITERATIONS = 3`.

- **Parallel fan-out via `Send`.** Parallel only when `parallel_safe=True` AND more than one task.
- **Reducer:** `artifacts: Annotated[list[dict], operator.add]`
- **Client cache** keyed by provider, endpoint, and model
- **JSON parsing** via `parse_json_response` (never raw `json.loads` on model output)

### Pipeline 3 — Transcribe (`src/transcribe.py`)

Straight pipeline: `select_frames → VerbatimOCR → stitch_frames → (optional) LLM cleanup → output/transcript.md`.

- **Thinking disabled** for OCR/cleanup (`enable_thinking: false`)
- **Cleanup OFF by default**; coverage guard keeps raw when LLM drops content
- OCR/cleanup default to `VLLM_MODEL` (`Qwen/Qwen3.6-27B-FP8`)

### Configuration (`src/config.py`)

`ScreenLensConfig` composes frame, captioning, embedding, vector DB, search, OCR, frame selection, and reconstruction configs. Defaults: **vLLM, CUDA, concurrency 2**. `CaptionBackend` is `vllm | omlx | ollama` (`omlx` is a legacy optional alias only). Use `--inference-url` / `--inference-model` / `--inference-api-key`; `--vllm-*` and `--omlx-*` are aliases. Caption `max_tokens` defaults to 32K.

### Data Layout

```
data/<slug>/
  frames/
  captions/
  ocr/
  chromadb/
  output/
    transcript.raw.md
    transcript.md
    reconstruction_meta.json
    transcribe_meta.json
```

## Things to Watch For

- **CLIP device** defaults to CUDA. Tests pin CPU.
- **`data/` is gitignored** along with video formats.
- **Context planning** uses `vllm_model_context`; keep it aligned with the serving limit.
- **Transcribe OCR must use a vision model.** Text-only aborts via capability guard + live probe.
- **`disable_thinking` stays on for OCR/cleanup.**
- **Don't size cleanup chunks past the output token cap.**
- **DGX concurrency stays at two.** Do not raise casually.
- **Do not start two port-8000 stacks.** Helper reuses DigitalTwin's exact-model endpoint. See `docs/DGX_SPARK.md`.

## Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs. If unclear, stop and ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess. Every changed line should trace to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified. For multi-step tasks, state a brief plan with verify checks.
