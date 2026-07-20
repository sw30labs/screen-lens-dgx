# Repository Guidelines

## Project Structure & Module Organization

**ScreenLens-DGX** is a Python 3.12 package for local video scene intelligence on **NVIDIA DGX Spark only** (vLLM + CUDA). Core code lives in `src/`: `cli.py` exposes the Typer CLI, `pipeline.py` wires LangGraph flows, and modules such as `frame_extractor.py`, `captioner.py`, `embedder.py`, `vector_store.py`, `reconstruct.py`, and `config.py` own individual stages. The verbatim path adds `frame_select.py`, `ocr.py`, `stitch.py`, and `transcribe.py`. The provider-neutral `InferenceClient` lives in the legacy-named `omlx_client.py`; Ollama is optional. The platform launcher is `setup_and_run_dgx.sh`; deployment also uses `compose.dgx-spark.yaml` and `docs/DGX_SPARK.md`. Tests live in `tests/`, with end-to-end scenarios in `tests/test_cases.yaml`. Generated outputs, model caches, videos, virtual environments, and databases belong in ignored paths such as `.local-models/`, `data/`, `OUTPUT/`, `ratita/`, and `input-videos/`.

This is a DGX-only fork of the dual-platform `screen-lens` repo. There is no macOS/oMLX product path.

## Build, Test, and Development Commands

Install the package locally (prefer the DGX helper so CUDA 13 wheels are correct):

```bash
./setup_and_run_dgx.sh setup
# or, inside an already-correct CUDA venv:
pip install -e ".[dev,tui]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run the CLI directly during development:

```bash
python -m src.cli info
python -m src.cli ingest "video.mov"
python -m src.cli search "What application is shown?"
python -m src.cli transcribe "video.mov"   # verbatim OCR path; cleanup off by default (--cleanup to enable)
```

Use provider-neutral direct-inference flags; `--vllm-*` (and legacy `--omlx-*`) are aliases:

```bash
python -m src.cli ingest "video.mov" --backend vllm \
  --inference-model Qwen/Qwen3.6-27B-FP8 \
  --device cuda --batch-size 2
```

On DGX Spark:

```bash
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run
```

The DGX helper creates `.venv-dgx` with Python 3.12 and pinned CUDA 13 torch/torchvision wheels. It reuses an exact-model service already owned by DigitalTwin and never stops that external stack. Do not install a generic PyPI torch wheel on Spark.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, clear type hints where useful, and concise docstrings for public classes, functions, and non-obvious helpers. Prefer `Path` for filesystem work and Pydantic config models from `src/config.py` instead of scattered constants. Keep module names lowercase with underscores, test functions named `test_*`, and classes named with `CamelCase`. No formatter or linter is currently configured in `pyproject.toml`; keep imports organized and run tests before submitting.

## Testing Guidelines

Tests use `pytest`, `pytest-asyncio`, and `pyyaml` from the `dev` extra. Add focused tests beside related coverage in `tests/test_pipeline.py` or split into `tests/test_<module>.py` files as coverage grows. Mock ChromaDB, OpenCLIP, vLLM, Ollama, and video processing so unit tests stay local and repeatable. Keep CPU explicit in portable embedding tests. `./setup_and_run_dgx.sh smoke` is the intentional live multimodal check: it must read `test.mov` from `assets/ingest-demo.png`. Avoid committing generated frames, captions, embeddings, videos, model caches, or `.venv-dgx`.

## Commit & Pull Request Guidelines

Recent history mostly follows Conventional Commit style, for example `feat(assemble): ...`, `fix(reconstruct): ...`, and `refactor(reconstruct): ...`. Use short, imperative subjects with a scope when helpful. Pull requests should include a summary, test results, linked issues when applicable, and screenshots or terminal output for user-visible changes. Note DGX/GB10, CUDA 13, and large-model assumptions.

## Agent-Specific Instructions

Keep edits narrowly scoped and preserve local artifacts, especially `.env`, `.local-models/`, and existing DigitalTwin-owned services. Do not move generated data into version control.

**Defaults (always):** vLLM, CUDA, concurrency **two**. Caption output has a 32K requested ceiling. At a matching 32K vLLM context, ScreenLens omits the literal `max_tokens` field so vLLM allocates the context remaining after prompt and image tokens instead of rejecting a zero-input reservation. Caption and reconstruction batching must remain size-budgeted: model repetition can make one caption much larger than the global average.

Do not reintroduce Apple Silicon / oMLX product defaults or a macOS launcher. When changing pipeline behavior, update README examples and `docs/DGX_SPARK.md`.
