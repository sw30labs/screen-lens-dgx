"""Unit-level coverage for the DGX Spark integration surface."""

from pathlib import Path
import os
import re
import subprocess
import sys

from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_cli_exposes_vllm_and_provider_neutral_aliases():
    from src.cli import app
    from typer.main import get_command

    result = CliRunner().invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "vllm" in result.output
    assert "--inference-url" in result.output
    ingest = get_command(app).commands["ingest"]
    model_option = next(param for param in ingest.params if param.name == "omlx_model")
    assert set(model_option.opts) == {
        "--inference-model",
        "--vllm-model",
        "--omlx-model",
    }


def test_direct_only_commands_never_default_to_ollama():
    from src.cli import DEFAULT_INFERENCE_BACKEND, app
    from typer.main import get_command

    commands = get_command(app).commands
    assert DEFAULT_INFERENCE_BACKEND in {"vllm", "omlx"}
    for name in ("summarize", "reconstruct", "assemble", "transcribe", "models"):
        backend = next(param for param in commands[name].params if param.name == "backend")
        assert backend.default == DEFAULT_INFERENCE_BACKEND

    env = os.environ.copy()
    env["SCREENLENS_BACKEND"] = "ollama"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.cli import DEFAULT_BACKEND, DEFAULT_INFERENCE_BACKEND; "
                "print(DEFAULT_BACKEND, DEFAULT_INFERENCE_BACKEND)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    caption_default, direct_default = probe.stdout.strip().split()
    assert caption_default == "ollama"
    assert direct_default in {"vllm", "omlx"}


def test_caption_options_route_to_provider_specific_config():
    from src.cli import _apply_captioning_options
    from src.config import CaptionBackend, ScreenLensConfig

    vllm = ScreenLensConfig()
    _apply_captioning_options(
        vllm,
        backend="vllm",
        omlx_url="http://spark.local:9000/v1",
        omlx_model="org/spark-vlm",
        omlx_api_key="spark-key",
    )
    assert vllm.captioning.backend == CaptionBackend.vllm
    assert vllm.captioning.vllm_base_url == "http://spark.local:9000/v1"
    assert vllm.captioning.vllm_model == "org/spark-vlm"
    assert vllm.captioning.vllm_api_key == "spark-key"

    omlx = ScreenLensConfig()
    _apply_captioning_options(
        omlx,
        backend="omlx",
        omlx_url="http://mac.local:8000/v1",
        omlx_model="mlx-community/vision-model",
        omlx_api_key="mlx-key",
    )
    assert omlx.captioning.backend == CaptionBackend.omlx
    assert omlx.captioning.omlx_model == "mlx-community/vision-model"
    assert omlx.captioning.omlx_api_key == "mlx-key"


def test_platform_launchers_have_valid_shell_syntax():
    # DGX-only fork: the Spark helper is the sole platform launcher.
    path = ROOT / "setup_and_run_dgx.sh"
    assert path.exists()
    assert not (ROOT / "setup_and_run_macos.sh").exists()
    assert path.stat().st_mode & 0o111
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_compose_recipe_is_loopback_only_and_bounded():
    compose = (ROOT / "compose.dgx-spark.yaml").read_text(encoding="utf-8")

    assert '"127.0.0.1:8000:8000"' in compose
    assert "platform: linux/arm64" in compose
    assert "vllm/vllm-openai@sha256:" in compose
    assert "Qwen/Qwen3.6-27B-FP8" in compose
    assert '"${VLLM_QUANTIZATION:-fp8}"' in compose
    assert '"${VLLM_KV_CACHE_DTYPE:-auto}"' in compose
    assert '"${VLLM_MAX_MODEL_LEN:-262144}"' in compose
    assert '"method":"mtp"' in compose
    assert "--moe-backend" not in compose
    assert "--max-num-seqs" in compose
    assert '      - "2"' in compose


def test_compose_has_isolated_bounded_ocr_service_with_pinned_revision():
    compose = (ROOT / "compose.dgx-spark.yaml").read_text(encoding="utf-8")

    assert "  ocr:\n" in compose
    assert '"127.0.0.1:8001:8001"' in compose
    assert "${OCR_MODEL:-Qwen/Qwen3.6-27B-FP8}" in compose
    revision = re.search(
        r"OCR_MODEL_REVISION:-([0-9a-f]{40})",
        compose,
    )
    assert revision is not None
    assert revision.group(1) == "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert revision.group(1) in helper
    assert revision.group(1) in example
    assert '"${OCR_GPU_MEMORY_UTILIZATION:-0.45}"' in compose
    assert '"${OCR_MAX_MODEL_LEN:-262144}"' in compose
    assert '"${OCR_MAX_NUM_SEQS:-2}"' in compose
    assert '"${OCR_MAX_NUM_BATCHED_TOKENS:-8192}"' in compose
    assert "--limit-mm-per-prompt" in compose
    assert "--enable-prefix-caching" in compose


def test_dgx_helper_exposes_ownership_safe_ocr_lifecycle():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    for command in ("ocr-up", "ocr-wait", "ocr-smoke", "ocr-logs", "ocr-down"):
        assert f"  {command})" in helper
    assert "VerbatimOCR(config).ocr_frames([image_path])" in helper
    assert "compose_with_token up --detach ocr" in helper
    assert "compose_control stop ocr" in helper
    assert "compose_control rm --force ocr" in helper
    assert "compose_control stop llm" in helper
    assert "compose_control rm --force llm" in helper
    assert "compose_control down" not in helper
    assert "--remove-orphans" not in helper


def test_dgx_helper_routes_only_commands_that_need_ocr():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert 'if [[ "${command}" == "transcribe" ]]' in helper
    assert 'elif [[ "${command}" == "benchmark-ocr" ]]' in helper
    assert '"${arg}" == "--hybrid-ocr"' in helper
    assert '"${arg}" == "--cleanup"' in helper
    assert '"${arg}" == "--ocr-url"' in helper
    assert '"${arg}" == "--ocr-model"' in helper
    assert '"${arg}" == "--ocr-model-revision"' in helper
    assert '"${arg}" == "--config-file"' in helper
    assert "--inference-url|--inference-url=*|--vllm-url|--vllm-url=*" in helper
    assert "--omlx-url|--omlx-url=*" in helper
    assert "explicit_ocr_target=0" in helper
    assert "if (( start_llm == 1 ))" in helper
    assert "if (( start_ocr == 1 ))" in helper


def test_dgx_helper_rejects_a_custom_model_with_default_revision():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert '"${OCR_MODEL}" != "${DEFAULT_OCR_MODEL}"' in helper
    assert '"${OCR_MODEL_REVISION}" == "${DEFAULT_OCR_MODEL_REVISION}"' in helper
    assert "cannot use the default model's pinned revision" in helper
    assert "starting custom OCR_MODEL=" in helper
    assert "requires OCR_MODEL_REVISION" in helper


def test_dgx_helper_migrates_only_the_old_repository_ocr_default(tmp_path):
    helper = tmp_path / "setup_and_run_dgx.sh"
    helper.write_text(
        (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    legacy_setting = "OCR_MODEL=Qwen3.6-27B-bf16\n"
    env_file.write_text(legacy_setting, encoding="utf-8")
    env = os.environ.copy()
    for key in (
        "OCR_MODEL",
        "OCR_MODEL_REVISION",
        "OCR_BASE_URL",
        "VLLM_OCR_MODEL",
        "VLLM_OCR_BASE_URL",
    ):
        env.pop(key, None)

    migrated = subprocess.run(
        ["bash", str(helper), "help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert migrated.returncode == 0, migrated.stderr
    assert "legacy OCR_MODEL=Qwen3.6-27B-bf16 found in .env" in migrated.stderr
    assert "using Qwen/Qwen3.6-27B-FP8 for this run" in migrated.stderr
    assert env_file.read_text(encoding="utf-8") == legacy_setting

    env["OCR_MODEL"] = "Qwen3.6-27B-bf16"
    explicit = subprocess.run(
        ["bash", str(helper), "help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert explicit.returncode == 0, explicit.stderr
    assert "legacy OCR_MODEL" not in explicit.stderr


def test_env_example_keeps_cross_platform_ocr_defaults_comment_only():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "#OCR_MODEL=Qwen/Qwen3.6-27B-FP8" in example
    assert "#OCR_BASE_URL=http://127.0.0.1:8000/v1" in example
    assert "#OCR_API_KEY=local" in example
    assert not re.search(r"^OCR_(?:MODEL|BASE_URL|API_KEY)=", example, re.MULTILINE)


def test_dgx_helper_accepts_a_model_specific_quantization_override():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert "[VLLM_QUANTIZATION]=1" in helper
    assert "[VLLM_KV_CACHE_DTYPE]=1" in helper
    assert 'VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"' in helper
    assert 'VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"' in helper
    assert "export VLLM_MODEL VLLM_MODEL_REVISION VLLM_QUANTIZATION" in helper


def test_dgx_smoke_uses_a_real_screen_image():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert "assets/ingest-demo.png" in helper
    assert "InferenceClient.from_endpoint" in helper
    assert "test.mov" in helper
