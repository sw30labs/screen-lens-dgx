"""Fast persistence and resume tests for strict verbatim transcription.

These tests deliberately use the real OCR configuration/contract builder while
replacing only frame extraction and the model calls.  That keeps resume
fingerprint coverage meaningful without requiring a running inference server.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _install_fake_frames(monkeypatch, transcribe_module, count: int = 3):
    """Replace video decoding with deterministic on-disk frame fixtures."""
    calls: list[Path] = []

    def fake_select_frames(video_path, output_dir, config):
        frames_dir = Path(output_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        calls.append(frames_dir)
        frames = []
        for index in range(count):
            path = frames_dir / f"frame_{index:06d}.png"
            path.write_bytes(f"fake-frame-{index}".encode())
            frames.append(
                {
                    "frame_id": index,
                    "frame_index": index * 10,
                    "timestamp": float(index),
                    "timestamp_str": f"00:00:{index:06.3f}",
                    "path": str(path),
                    "width": 100,
                    "height": 100,
                }
            )
        return frames

    monkeypatch.setattr(transcribe_module, "select_frames", fake_select_frames)
    return calls


def _config():
    from src.config import ScreenLensConfig

    config = ScreenLensConfig()
    config.ocr.model = "lightonai/LightOnOCR-2-1B"
    config.ocr.base_url = "http://127.0.0.1:8001/v1"
    config.ocr.max_tokens = 16384
    config.reconstruction.enabled = False
    return config


def _result(
    text: str,
    *,
    complete: bool = True,
    error: str | None = None,
    tiled: bool = False,
):
    from src.ocr import OCRFrameResult

    return OCRFrameResult(
        text=text,
        status="complete_tiled" if complete and tiled else "complete" if complete else "failed",
        complete=complete,
        tiled=tiled,
        error=error,
    )


def _record(data_dir: Path, frame_id: int) -> dict:
    path = data_dir / "ocr" / f"ocr_{frame_id:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_frame_is_atomically_persisted_before_later_ocr_failure(
    tmp_path, monkeypatch,
):
    """A process failure must not discard already completed model work."""
    import src.transcribe as transcribe_module

    video = tmp_path / "video.mov"
    video.write_bytes(b"real-enough-video-identity")
    data_dir = tmp_path / "run"
    _install_fake_frames(monkeypatch, transcribe_module)

    real_ocr = transcribe_module.VerbatimOCR
    observed_during_callback = {}

    class FailingOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            assert on_result is not None
            results = [
                _result("alpha"),
                _result("", complete=False, error="synthetic failure"),
                _result("", complete=False, error="not attempted"),
            ]
            on_result(0, results[0])

            # Persistence is required synchronously at callback time, not in a
            # final loop that is skipped when a later request raises.
            first_path = data_dir / "ocr" / "ocr_000000.json"
            observed_during_callback["exists"] = first_path.exists()
            if first_path.exists():
                observed_during_callback["record"] = json.loads(
                    first_path.read_text(encoding="utf-8")
                )

            on_result(1, results[1])
            on_result(2, results[2])
            return results

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", FailingOCR)

    with pytest.raises(RuntimeError, match=r"(?i)(OCR|incomplete|failed)"):
        transcribe_module.transcribe_video(str(video), _config(), data_dir)

    assert observed_during_callback["exists"] is True
    assert observed_during_callback["record"]["ocr"] == "alpha"
    assert observed_during_callback["record"]["ocr_complete"] is True
    assert _record(data_dir, 1)["ocr_complete"] is False
    assert not (data_dir / "output" / "transcript.md").exists()

    # An incomplete run may expose an explicitly named best-effort preview,
    # but it must never masquerade as the canonical completed transcript.
    partial = data_dir / "output" / "transcript.partial.md"
    if partial.exists():
        partial_text = partial.read_text(encoding="utf-8")
        assert "alpha" in partial_text
        assert "OCR-INCOMPLETE" in partial_text

    # Same-directory atomic replacement must not leave write temporaries behind.
    assert not list((data_dir / "ocr").glob("*.tmp"))
    assert not list((data_dir / "ocr").glob(".*.tmp"))


def test_explicit_resume_reuses_only_complete_records_and_preserves_frame_order(
    tmp_path, monkeypatch,
):
    """Failed/unattempted frames are retried; complete frames are not."""
    import src.transcribe as transcribe_module

    video = tmp_path / "video.mov"
    video.write_bytes(b"stable-video")
    data_dir = tmp_path / "run"
    select_calls = _install_fake_frames(monkeypatch, transcribe_module)
    real_ocr = transcribe_module.VerbatimOCR
    invocations: list[list[str]] = []

    class FirstRunOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            invocations.append([Path(path).name for path in paths])
            assert on_result is not None
            results = [
                _result("alpha"),
                _result("", complete=False, error="retry me", tiled=True),
                _result("", complete=False, error="not attempted"),
            ]
            for index, result in enumerate(results):
                on_result(index, result)
            return results

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", FirstRunOCR)
    with pytest.raises(RuntimeError, match=r"(?i)(OCR|incomplete|failed)"):
        transcribe_module.transcribe_video(str(video), _config(), data_dir)

    class ResumeOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            assert self._tile_first.is_set()
            invocations.append([Path(path).name for path in paths])
            assert on_result is not None
            results = [_result("bravo"), _result("charlie")]

            # Model requests can finish out of order.  The callback/persistence
            # order must not determine the final transcript order.
            on_result(1, results[1])
            on_result(0, results[0])
            return results

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", ResumeOCR)
    result = transcribe_module.transcribe_video(
        str(video), _config(), data_dir, resume=True
    )

    assert result["stage"] == "done"
    assert invocations == [
        ["frame_000000.png", "frame_000001.png", "frame_000002.png"],
        ["frame_000001.png", "frame_000002.png"],
    ]
    # Resume loads the immutable manifest rather than re-extracting frames.
    assert len(select_calls) == 1

    records = json.loads(
        (data_dir / "ocr" / "all_ocr.json").read_text(encoding="utf-8")
    )
    assert [record["frame_id"] for record in records] == [0, 1, 2]
    assert [record["ocr"] for record in records] == ["alpha", "bravo", "charlie"]
    assert all(record["ocr_complete"] for record in records)

    transcript = (data_dir / "output" / "transcript.md").read_text(
        encoding="utf-8"
    )
    assert transcript.index("alpha") < transcript.index("bravo") < transcript.index("charlie")


def test_resume_refuses_changed_ocr_contract_before_model_calls(tmp_path, monkeypatch):
    """A record may only be reused under the exact OCR fidelity contract."""
    import src.transcribe as transcribe_module

    video = tmp_path / "video.mov"
    video.write_bytes(b"stable-video")
    data_dir = tmp_path / "run"
    _install_fake_frames(monkeypatch, transcribe_module, count=1)
    real_ocr = transcribe_module.VerbatimOCR

    class CompleteOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            result = _result("alpha")
            if on_result is not None:
                on_result(0, result)
            return [result]

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", CompleteOCR)
    transcribe_module.transcribe_video(str(video), _config(), data_dir)

    model_was_called = False

    class MustNotRunOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            nonlocal model_was_called
            model_was_called = True
            raise AssertionError("contract mismatch should fail before OCR")

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", MustNotRunOCR)
    changed = _config()
    changed.ocr.max_tokens = 8192

    with pytest.raises(RuntimeError, match=r"(?i)(contract|fingerprint|configuration).*(mismatch|changed|match)"):
        transcribe_module.transcribe_video(
            str(video), changed, data_dir, resume=True
        )

    assert model_was_called is False

    revision_changed = _config()
    revision_changed.ocr.model_revision = "different-checkpoint-revision"
    with pytest.raises(RuntimeError, match=r"(?i)contract.*mismatch"):
        transcribe_module.transcribe_video(
            str(video), revision_changed, data_dir, resume=True
        )


def test_fresh_run_refuses_existing_generated_state(tmp_path):
    import src.transcribe as transcribe_module

    video = tmp_path / "video.mov"
    video.write_bytes(b"video")
    data_dir = tmp_path / "run"
    (data_dir / "ocr").mkdir(parents=True)
    (data_dir / "ocr" / "stale.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"(?i)(fresh|nonempty|resume)"):
        transcribe_module.transcribe_video(str(video), _config(), data_dir)


def test_keyboard_interrupt_finalizes_gapped_partial_before_reraising(
    tmp_path, monkeypatch,
):
    import src.transcribe as transcribe_module

    video = tmp_path / "video.mov"
    video.write_bytes(b"video")
    data_dir = tmp_path / "run"
    _install_fake_frames(monkeypatch, transcribe_module, count=2)
    real_ocr = transcribe_module.VerbatimOCR

    class InterruptingOCR(real_ocr):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            assert on_result is not None
            on_result(0, _result("    completed before interrupt  \n\n"))
            raise KeyboardInterrupt

    monkeypatch.setattr(transcribe_module, "VerbatimOCR", InterruptingOCR)
    with pytest.raises(KeyboardInterrupt):
        transcribe_module.transcribe_video(str(video), _config(), data_dir)

    assert _record(data_dir, 0)["ocr_complete"] is True
    partial = (data_dir / "output" / "transcript.partial.md").read_text(
        encoding="utf-8"
    )
    assert partial.startswith("    completed before interrupt  \n\n")
    assert "OCR-INCOMPLETE" in partial
    meta = json.loads(
        (data_dir / "output" / "transcribe_meta.json").read_text(encoding="utf-8")
    )
    assert meta["stage"] == "ocr_incomplete"
