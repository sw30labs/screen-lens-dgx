# NVIDIA DGX Spark deployment

**ScreenLens-DGX** targets DGX Spark only, through local OpenAI-compatible vLLM
services. The canonical multimodal endpoint is shared by captioning, OCR, and
reconstruction. There is no Apple Silicon / oMLX product path in this fork.

## Architecture

- The shared vLLM endpoint binds only to `http://127.0.0.1:8000/v1`.
- ScreenLens runs in `.venv-dgx` with Python 3.12, CUDA 13 PyTorch, and CUDA
  OpenCLIP embeddings.
- The default `Qwen/Qwen3.6-27B-FP8` checkpoint handles semantic frame
  captioning, verbatim OCR, and text reconstruction.
- Hugging Face downloads and vLLM compilation caches persist outside the
  container and can be shared with another project.

When ScreenLens starts its own service, the model revision is pinned to
`e89b16ebf1988b3d6befa7de50abc2d76f26eb09` and the validated vLLM image is
digest-pinned. The server exposes the model's native 262,144-token context,
admits at most two sequences, and targets 45% GPU memory so the long FP8 KV
cache fits while leaving unified memory for image inference, OpenCLIP, the
operating system, and other local workloads.

ScreenLens requests captions with a 32K output ceiling. Prompt, chat-template,
image, and completion tokens share the server's 262K context, so the caption
limit leaves substantial input headroom. If `VLLM_MAX_MODEL_LEN` is deliberately
reduced to the same 32K value, the client omits `max_tokens`; vLLM then assigns
the exact context remaining after the input instead of making an impossible
zero-input reservation.
Direct captions also use light repetition controls so a malformed generation
cannot consume that entire ceiling by looping. Later reconstruction stages
greedily pack captions by serialized size and split an individually oversized
caption; they do not assume every frame caption is near the global average.
Caption failures are isolated per frame, so a timed-out request cannot discard
the successful result beside it in the same two-request chunk. ScreenLens
retries only the failed frame once with a 2,048-token ceiling (configurable as
`captioning.retry_attempts` and `captioning.retry_max_tokens`) before recording
an error marker for that frame and continuing. Retries use deterministic
decoding and must terminate naturally within the bounded ceiling; a truncated
loop is not accepted as a caption.
The shared extraction pass requests at most 1,400 output tokens while retaining
the server's entire context as completion headroom. Recursive synthesis filters
notes to the current file or artifact and uses the same full ceiling. Long
reconstruction calls have an independent 1,800-second HTTP timeout, configurable
as `reconstruction.timeout_seconds`, so they do not inherit the shorter caption
request budget. A group
that still ends with `finish_reason=length` is discarded and retried with less
input; incomplete prefixes never flow into later passes. If the endpoint and
`VLLM_MAX_MODEL_LEN` are upgraded together, reconstruction automatically uses a
larger served context such as Qwen3.6's native 262K window.

The service uses Qwen's built-in `mtp` speculative method with two draft
tokens. MTP is lossless speculative decoding: it changes throughput, not the
model's answer or context limit. DFlash is intentionally absent because it
requires an additional draft checkpoint and does not solve completion-length
failures.

## Shared OCR service

`./setup_and_run_dgx.sh ocr-up`, `run transcribe`, and hybrid OCR reuse the same
`Qwen/Qwen3.6-27B-FP8` endpoint as captioning. If the service is not running,
the helper starts the canonical port-8000 service once. It never loads a second
default checkpoint or allocates a duplicate vLLM worker.

The strict transcribe path still retries context-truncated frames as overlapping
horizontal tiles and persists completed frames atomically. Resume an interrupted
run by passing its exact timestamped directory:

```bash
./setup_and_run_dgx.sh run transcribe input/part1.mov \
  --ocr-max-tokens 16384 \
  --resume-dir data/part1_20260715_225443
```

Explicit `--ocr-url`, `--ocr-model`, `--ocr-model-revision`, or `--config-file`
values remain caller-owned overrides. The helper will not download or replace a
custom endpoint.

## Sharing vLLM with DigitalTwin

DigitalTwin uses the same model and loopback port. Only one Compose project can
own port 8000, so `./setup_and_run_dgx.sh llm-up` checks `/v1/models` first. If an
already-running service exposes the configured model id and at least the
configured context length, ScreenLens reuses it and does not start or recreate
a container. The subsequent image smoke proves that the shared endpoint really
processes vision input.

The standard models response does not expose a Hugging Face revision or every
vLLM launch flag. When reusing a service, its owner remains responsible for the
revision and runtime settings; ScreenLens verifies the observable model id,
context length, and multimodal behavior. The pinned revision and image apply
directly when this repository owns the service.

Likewise, `run` reuses a ready service. `llm-down` and `llm-logs` operate only on
the container created by this repository; they never stop or attach to a
DigitalTwin-owned service. If port 8000 serves a different model, the helper
fails instead of replacing it.

To avoid duplicate model storage when ScreenLens owns the service, point its
caches at an existing compatible cache before startup:

```bash
export DGX_HF_CACHE="$HOME/models/huggingface"
export DGX_VLLM_CACHE="$HOME/models/.cache/vllm"
```

These centralized paths are also the repository defaults.

## Prerequisites

The checked configuration expects:

- NVIDIA DGX Spark / GB10 running a current DGX OS on ARM64
- CUDA 13 and a working `nvidia-smi`
- Docker Engine, the Compose plugin, and NVIDIA Container Toolkit
- Python 3.12 with `venv` support
- ffmpeg/ffprobe recommended for complete video metadata (a missing binary uses the quiet OpenCV fallback)
- approximately 70 GiB of free disk for a first container/model download
- outbound HTTPS to Docker Hub, Hugging Face, and Python package indexes
- a Hugging Face read token when this repository must start vLLM

Verify Docker GPU passthrough independently if the NVIDIA runtime was recently
installed:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If Docker does not list an NVIDIA runtime, perform the one-time administrator
configuration and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If the user cannot access `/var/run/docker.sock`, add that user to the Docker
group and start a fresh login session. Do not run ScreenLens itself as root.

## One-time setup

From the repository root:

```bash
(umask 077; touch .env)
chmod 600 .env
${EDITOR:-nano} .env
```

Add `HF_TOKEN=hf_...` to `.env` if ScreenLens may need to start its own server.
The helper reads only its documented variable whitelist as literal values; it
does not source or execute `.env`.

Then run:

```bash
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh llm-wait
./setup_and_run_dgx.sh smoke
# Optional now; automatic for `run transcribe` and `run ... --hybrid-ocr`:
./setup_and_run_dgx.sh ocr-up
./setup_and_run_dgx.sh ocr-smoke
```

`setup` creates `.venv-dgx`, installs `torch==2.11.0+cu130` and
`torchvision==0.26.0+cu130` from PyTorch's CUDA 13 index, installs ScreenLens,
and executes real CUDA matrix and OpenCLIP image-encoder operations. It does not
install vLLM into the application environment; the server's CUDA/Triton stack
stays isolated in its container.

The first dense-server start may download roughly 24 GB of model data and build
FlashInfer/Torch caches. The default readiness timeout is 1,800 seconds. The OCR
checkpoint is approximately 2 GB and has its own 900-second readiness budget.

## Validation

Readiness is not based only on container state. `llm-wait` requires
`/v1/models` to contain the configured model id and advertise at least
`VLLM_MAX_MODEL_LEN`. `smoke` then uses ScreenLens's own inference client to
send `assets/ingest-demo.png` as an image and passes only when the model reads
the visible filename `test.mov`.

```bash
./setup_and_run_dgx.sh llm-wait
./setup_and_run_dgx.sh smoke
```

This catches text-only deployments and OpenAI-compatible servers that are alive
but not actually processing vision content.

The OCR checks validate the same port-8000 endpoint through the OCR client:

```bash
./setup_and_run_dgx.sh ocr-wait
./setup_and_run_dgx.sh ocr-smoke
```

`ocr-wait` requires the exact `OCR_MODEL` and at least `OCR_MAX_MODEL_LEN`.
`ocr-smoke` calls `VerbatimOCR` with the production `OCRConfig` and passes only
when Qwen transcribes `test.mov` from the same fixture. This exercises the
model-aware image request adapter, not a separate curl-only probe.

## Running ScreenLens

With no additional arguments, `run` launches the TUI:

```bash
./setup_and_run_dgx.sh run
```

CLI arguments pass through to `python -m src.cli`:

```bash
./setup_and_run_dgx.sh run ingest input-videos/demo.mov
./setup_and_run_dgx.sh run ingest input-videos/demo.mov --hybrid-ocr
./setup_and_run_dgx.sh run transcribe input-videos/demo.mov --ocr-max-tokens 16384
./setup_and_run_dgx.sh run models
```

The launcher inspects only explicit CLI arguments when deciding which services
to start. `transcribe` without `--cleanup` starts only OCR; `transcribe
--cleanup` starts OCR plus the dense text service; hybrid ingest starts both.
`benchmark-ocr` starts neither because it compares endpoints that must already
be served. Explicit/config-file OCR targets are never started by the helper.
Other commands preserve the normal dense-service behavior.

The helper exports the selected `VLLM_*` and `OCR_*` connections, sets
`SCREENLENS_BACKEND=vllm` and `SCREENLENS_DEVICE=cuda`, and bounds ScreenLens
caption and OCR client concurrency to two. The OCR server's eight-sequence
limit is headroom rather than the normal client fan-out. The helper also
points `HF_HOME` at `DGX_HF_CACHE` so OpenCLIP weights share the persistent
application/model cache; placeholder tokens are removed before launching the
application.

## Commands

| Command | Effect |
|---|---|
| `doctor` | Read-only host, GPU, Docker, Compose, disk, Python, token, and reuse checks |
| `setup` | Build or repair `.venv-dgx` and run CUDA/OpenCLIP preflight |
| `llm-up` | Reuse an exact-model service or start this repository's Compose service |
| `llm-wait` | Wait for exact model discovery through `/v1/models` |
| `llm-logs` | Follow only the ScreenLens-owned container logs |
| `llm-down` | Stop/remove only the ScreenLens-owned `llm` service and preserve its sibling/caches |
| `ocr-up` | Reuse an exact OCR endpoint or start the ScreenLens-owned small-model service |
| `ocr-wait` | Wait for exact OCR model/context discovery through `/v1/models` |
| `ocr-smoke` | Run the production verbatim OCR path against `ingest-demo.png` |
| `ocr-logs` | Follow only the ScreenLens-owned OCR container logs |
| `ocr-down` | Stop/remove only the ScreenLens-owned OCR service; never touch the LLM |
| `smoke` | Run the real `ingest-demo.png` vision assertion |
| `run` | Ensure only the endpoint(s) needed by the CLI arguments, export DGX defaults, and invoke ScreenLens |
| `help` | Show command and environment help |

## Configuration

Settings may be exported or stored in the private `.env`; exported nonempty
values take precedence.

Older ScreenLens `.env` files may contain the former repository default
`OCR_MODEL=Qwen3.6-27B-bf16`. When that exact value came from `.env` and has no
paired OCR endpoint or revision, the DGX helper treats it as a legacy default
and uses the canonical Qwen FP8 model for the current invocation without editing the file.
Explicitly exported values and custom endpoint/revision combinations are never
rewritten.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | required for a new local start | Hugging Face read token; never printed |
| `HF_HUB_DISABLE_XET` | `1` | Use resumable standard Hub HTTP and avoid observed Xet CAS 401 failures |
| `DGX_VENV_DIR` | `.venv-dgx` | Python 3.12 application environment |
| `DGX_PYTHON_BIN` | `python3.12` | Interpreter used to create the environment |
| `DGX_HF_CACHE` | `$HOME/models/huggingface` | Centralized vLLM and OpenCLIP Hugging Face cache |
| `DGX_VLLM_CACHE` | `$HOME/models/.cache/vllm` | Centralized compilation/runtime cache |
| `VLLM_IMAGE` | validated `vllm/vllm-openai@sha256:…` digest | ARM64 vLLM image; override deliberately when validating an upgrade |
| `VLLM_MODEL` | `Qwen/Qwen3.6-27B-FP8` | Served and requested dense multimodal model id |
| `VLLM_MODEL_REVISION` | pinned SHA above | Reproducible model contents |
| `VLLM_QUANTIZATION` | `fp8` | vLLM quantization backend for the canonical checkpoint |
| `VLLM_KV_CACHE_DTYPE` | `auto` | Use the model/runtime-selected KV-cache precision |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible API root |
| `VLLM_API_KEY` | `local` | Placeholder for loopback, or bearer token for an externally authenticated endpoint |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.45` | vLLM allocator target sized for the long FP8 KV cache |
| `VLLM_MAX_MODEL_LEN` | `262144` | Native serving limit and Python prompt-planning context |
| `VLLM_START_TIMEOUT` | `1800` | Readiness timeout in seconds |
| `VLLM_LOG_TAIL` | `200` | Initial line count for `llm-logs` |
| `OCR_VLLM_IMAGE` | same validated vLLM digest | ARM64 image for the independent OCR container |
| `OCR_MODEL` | `Qwen/Qwen3.6-27B-FP8` | Provider-neutral OCR model id shared with captioning |
| `OCR_MODEL_REVISION` | same pinned SHA as `VLLM_MODEL_REVISION` | Pinned contents for the canonical checkpoint |
| `OCR_BASE_URL` | `http://127.0.0.1:8000/v1` | Shared provider-neutral OCR API root |
| `OCR_API_KEY` | `local` | OCR bearer token placeholder or external endpoint credential |
| `OCR_GPU_MEMORY_UTILIZATION` | `0.45` | Same allocator target as the shared Qwen service |
| `OCR_MAX_MODEL_LEN` | `262144` | Shared serving limit and readiness requirement |
| `OCR_MAX_NUM_SEQS` | `2` | Shared server sequence ceiling |
| `OCR_MAX_NUM_BATCHED_TOKENS` | `8192` | Scheduler token budget per OCR iteration |
| `OCR_START_TIMEOUT` | `900` | OCR readiness timeout in seconds |
| `OCR_LOG_TAIL` | `200` | Initial line count for `ocr-logs` |
| `VLLM_OCR_MODEL`, `VLLM_OCR_BASE_URL`, `VLLM_OCR_API_KEY` | unset | Lower-priority compatibility aliases for the provider-neutral OCR variables |

The included loopback services do not enable API authentication. `local` only
satisfies clients that require a nonempty key. Do not expose ports 8000 or 8001
to the LAN or Internet; use an authenticated TLS reverse proxy for remote access.

When selecting another OCR checkpoint for the bundled Compose service, set
`OCR_MODEL_REVISION` to the matching immutable revision as well. The generic
serving flags require one-image OpenAI-compatible chat support; model-specific
quantization or processor flags may require a separate validation pass.
For a caller-owned endpoint, pass that identity with
`--ocr-model-revision <immutable-revision>` (or set `ocr.model_revision` in the
config file) so interrupted runs can verify the exact checkpoint on resume.

## Memory and performance notes

DGX Spark memory is unified, not 128 GB of VRAM plus separate system RAM. The
dense FP8 service uses a `0.45` allocator target because its 262K window needs
a substantial FP8 KV cache. Keep dense caption concurrency at two, and lower
that allocator or context together if other sustained GPU workloads must share
the host.

OCR shares that allocation and does not start a second worker. The server
admits two sequences across captioning and OCR. The allocator target is a
ceiling, not measured steady-state consumption; leave headroom for model
weights, OpenCLIP, Docker, and the operating system. If the host swaps, lower
request concurrency or the model context before reducing the allocator below
what vLLM needs for the model and KV cache.

OCR decoding time also scales with output length. Hybrid indexing and the
benchmark use a 4,096-token ceiling by default. A
full-screen dense document can spend far longer decoding than an ordinary UI or
terminal frame and may fill that cap. Use `--ocr-max-tokens 1024` to put a
tighter latency ceiling on hybrid indexing. Verbatim transcription deliberately
uses the full 16,384-token server context for the initial full-frame request; at
that matching ceiling the vLLM client requests all space remaining after image
tokens instead of reserving an impossible zero-input completion. If it
truncates, ScreenLens can make up to 12 bounded tile requests at 4,096 tokens
each, so the 16K value is not a total-work ceiling for a tiled frame.
`benchmark-ocr` rejects cap-filled output so its successful latency statistics
never hide truncation.

For JSON configuration, hybrid OCR uses the lower of `ocr.max_tokens` and
`hybrid_ingest.ocr_max_tokens`; raise both to lift the 4,096-token hybrid bound.
An older serialized config containing `ocr.max_tokens: 4096` remains an explicit
choice—remove that field or change it to `16384` for full-context verbatim vLLM.
External vLLM endpoints must set `OCR_MAX_MODEL_LEN` to their actual served
context so the client does not infer the managed 16K limit incorrectly. oMLX
keeps a 4,096-token default unless raised for a verified larger context.

Integrated-GPU memory fields may appear unavailable in `nvidia-smi`. Use
`free -h`, the DGX Dashboard, and per-process measurements for the shared pool.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Docker permission denied | Add the user to the Docker group, log out/in, and rerun `doctor` |
| NVIDIA runtime missing | Configure it with `nvidia-ctk`, restart Docker, and validate GPU passthrough |
| `HF_TOKEN` missing | Add a read token to mode-600 `.env`; it is unnecessary only when reusing a ready service |
| Public OpenCLIP weights load without `HF_TOKEN` | Supported; ScreenLens hides only the unauthenticated-download advisory while surfacing real Hub failures |
| Reconstruction exhausts the served context | The incomplete prefix is discarded automatically and the input group is split; a final failure means one minimum-size group still cannot fit even with the full served context |
| Hugging Face Xet/CAS returns 401 | Keep `HF_HUB_DISABLE_XET=1`; standard Hub HTTP resumes partial downloads |
| Port 8000 has another model | Stop/reconfigure its owner or set `VLLM_BASE_URL`; ScreenLens will not replace it |
| Port 8001 is occupied or serves another model | Stop/reconfigure its owner or set `OCR_BASE_URL`; `ocr-up` will not replace it |
| First startup appears stalled | Follow the owning stack's logs and allow the 1,800-second model/compile timeout |
| PyTorch reports no CUDA | Rerun `setup`; do not replace the `+cu130` ARM64 wheels with PyPI CPU wheels |
| `pip check` reports the known cuSPARSELt SBSA tag | The helper accepts only that exact warning after successful real CUDA/OpenCLIP operations |
| Vision smoke omits `test.mov` | Confirm the exact multimodal model and inspect vLLM logs; a text-only or mismatched service is not valid |
| OCR smoke omits `test.mov` | Confirm the pinned OCR model, inspect `ocr-logs`, and verify the model-aware image-only request profile |
| `llm-down` appears to leave OCR running | Expected: service lifecycle is isolated; use `ocr-down` explicitly |
| Out of memory or heavy swap | Stop competing workloads first; lower OCR concurrency, or reduce the corresponding service's memory/context settings |

## Scope of this fork

This repository is DGX Spark only. For the dual-platform tree (including Apple
Silicon / oMLX), use the original `screen-lens` project. Use this helper only on
Linux/ARM64 DGX Spark.
