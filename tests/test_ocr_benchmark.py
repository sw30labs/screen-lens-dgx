"""Tests for the read-only served-model OCR benchmark harness."""
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from src.cli import app
from src.config import InferenceBackend
from src.ocr_benchmark import (
    OCRBenchmarkTarget,
    benchmark_ocr_targets,
    output_sanity,
    select_representative_frames,
)


def test_select_representative_frames_is_uniform_and_stable(tmp_path):
    for index in range(10):
        (tmp_path / f"frame_{index:02d}.png").write_bytes(b"image")
    (tmp_path / "ignore.txt").write_text("not an image")

    selected = select_representative_frames([tmp_path], limit=3)
    assert [path.name for path in selected] == [
        "frame_00.png",
        "frame_04.png",
        "frame_09.png",
    ]


def test_output_sanity_flags_empty_blind_and_repetitive_output():
    assert output_sanity("")["warnings"] == ["empty"]
    assert "no-image-refusal" in output_sanity(
        "No image has been provided."
    )["warnings"]
    repetitive = output_sanity("same\nsame\nsame\nsame")
    assert "repetitive-lines" in repetitive["warnings"]


def test_benchmark_writes_json_and_reports_throughput(tmp_path, monkeypatch):
    import json
    import src.ocr_benchmark as benchmark

    frames = []
    for index in range(4):
        frame = tmp_path / f"frame_{index}.png"
        frame.write_bytes(b"image")
        frames.append(frame)

    class FakeOCR:
        def __init__(self, config):
            self.model = config.model
            self.request_profile = SimpleNamespace(family="fake-ocr")

        def assert_vision_capable(self):
            return None

        def ocr_frame(self, frame, *, require_complete=False):
            return f"{self.model}: {Path(frame).stem}"

    monkeypatch.setattr(benchmark, "VerbatimOCR", FakeOCR)
    destination = tmp_path / "report.json"
    targets = [
        OCRBenchmarkTarget(
            model="vendor/ocr-a",
            base_url="http://127.0.0.1:8001/v1",
            backend=InferenceBackend.vllm,
            api_key="local",
        ),
        OCRBenchmarkTarget(
            model="vendor/ocr-b",
            base_url="http://127.0.0.1:8002/v1",
            backend=InferenceBackend.vllm,
            api_key="local",
        ),
    ]
    report = benchmark_ocr_targets(
        frames,
        targets,
        output_path=destination,
        concurrency=2,
    )

    assert destination.exists()
    saved = json.loads(destination.read_text())
    assert saved["ground_truth_available"] is False
    assert saved["warmup_enabled"] is True
    assert len(saved["targets"]) == 2
    for target in saved["targets"]:
        assert target["concurrency"] == 2
        assert target["setup_error"] is None
        assert target["warmup_seconds"] >= 0
        assert target["summary"]["successes"] == 4
        assert target["summary"]["failures"] == 0
        assert target["summary"]["frames_per_second"] > 0
        assert len(target["frames"]) == 4
        assert all(frame["sanity"]["status"] == "ok" for frame in target["frames"])
    assert report["output_path"] == str(destination.resolve())


def test_benchmark_cli_compares_repeatable_targets(tmp_path, monkeypatch):
    import json
    import src.ocr_benchmark as benchmark

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"image")
    destination = tmp_path / "cli-report.json"

    class FakeOCR:
        def __init__(self, config):
            self.model = config.model
            self.request_profile = SimpleNamespace(family="fake-ocr")

        def assert_vision_capable(self):
            return None

        def ocr_frame(self, frame, *, require_complete=False):
            return f"text from {self.model}"

    monkeypatch.setattr(benchmark, "VerbatimOCR", FakeOCR)
    result = CliRunner().invoke(
        app,
        [
            "benchmark-ocr",
            str(frame),
            "--model", "lightonai/LightOnOCR-2-1B",
            "--model", "zai-org/GLM-OCR",
            "--endpoint", "http://127.0.0.1:8001/v1",
            "--endpoint", "http://127.0.0.1:8002/v1",
            "--concurrency", "1",
            "--output", str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No ground truth" in result.output
    saved = json.loads(destination.read_text())
    assert [target["model"] for target in saved["targets"]] == [
        "lightonai/LightOnOCR-2-1B",
        "zai-org/GLM-OCR",
    ]
    assert all(target["concurrency"] == 1 for target in saved["targets"])


def test_benchmark_records_target_setup_failure_and_continues(tmp_path, monkeypatch):
    import src.ocr_benchmark as benchmark

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"image")

    class FakeOCR:
        def __init__(self, config):
            if config.model == "vendor/broken":
                raise RuntimeError("unsupported architecture")
            self.model = config.model
            self.request_profile = SimpleNamespace(family="fake-ocr")

        def assert_vision_capable(self):
            return None

        def ocr_frame(self, frame, *, require_complete=False):
            return "visible text"

    monkeypatch.setattr(benchmark, "VerbatimOCR", FakeOCR)
    report = benchmark_ocr_targets(
        [frame],
        [
            OCRBenchmarkTarget("vendor/broken", "http://127.0.0.1:8001/v1"),
            OCRBenchmarkTarget("vendor/good", "http://127.0.0.1:8002/v1"),
        ],
        output_path=tmp_path / "report.json",
    )

    broken, good = report["targets"]
    assert "unsupported architecture" in broken["setup_error"]
    assert broken["summary"]["failures"] == 1
    assert good["setup_error"] is None
    assert good["summary"]["successes"] == 1


def test_ingest_cli_applies_hybrid_ocr_flags(tmp_path, monkeypatch):
    import src.cli as cli

    video = tmp_path / "clip.mov"
    video.write_bytes(b"video")
    captured = {}

    class FakeGraph:
        def invoke(self, state):
            captured.update(state)
            return {"elapsed_seconds": {}, "embeddings_shape": []}

    monkeypatch.setattr(cli, "build_ingest_graph", lambda: FakeGraph())
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(video),
            "--hybrid-ocr",
            "--ocr-url", "http://127.0.0.1:8123",
            "--ocr-model", "zai-org/GLM-OCR",
            "--ocr-api-key", "ocr-secret",
            "--ocr-concurrency", "3",
            "--ocr-max-tokens", "1024",
        ],
    )
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config["hybrid_ingest"]["enabled"] is True
    assert config["ocr"]["base_url"] == "http://127.0.0.1:8123/v1"
    assert config["ocr"]["model"] == "zai-org/GLM-OCR"
    assert config["ocr"]["api_key"] == "ocr-secret"
    assert config["ocr"]["concurrency"] == 3
    assert config["ocr"]["max_tokens"] == 1024
    assert config["hybrid_ingest"]["ocr_max_tokens"] == 1024
