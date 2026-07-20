# Quickstart (DGX Spark)

```bash
cd ~/Desktop/screen-lens-dgx
(umask 077; cp -n .env.example .env)
chmod 600 .env
# Edit HF_TOKEN if this repo must start vLLM (not needed to reuse DigitalTwin)

./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run                 # TUI
./setup_and_run_dgx.sh run ingest video.mov
./setup_and_run_dgx.sh run search "What app is shown?"
./setup_and_run_dgx.sh run transcribe video.mov
```

Defaults: **vLLM** · **CUDA** · concurrency **2** · model **Qwen/Qwen3.6-27B-FP8** on `:8000`.

See [README.md](README.md) and [docs/DGX_SPARK.md](docs/DGX_SPARK.md).
