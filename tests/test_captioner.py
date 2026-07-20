"""Focused tests for per-frame caption request isolation and retries."""
import json
from collections import defaultdict


def test_concurrent_batch_preserves_peer_and_retries_only_failed_frame(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            batch_size=2,
            retry_attempts=1,
            retry_max_tokens=1234,
        )
    )
    calls = defaultdict(list)

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls[image_path].append((max_tokens, temperature, require_complete))
        if image_path == "slow.jpg" and max_tokens is None:
            raise RuntimeError("request timed out")
        return f"caption:{image_path}:{max_tokens}"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["slow.jpg", "good.jpg"]) == [
        "caption:slow.jpg:1234",
        "caption:good.jpg:None",
    ]
    assert calls == {
        "slow.jpg": [(None, None, False), (1234, 0.0, True)],
        "good.jpg": [(None, None, False)],
    }


def test_concurrent_batch_marks_only_frame_that_exhausts_retries(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            batch_size=2,
            retry_attempts=2,
            retry_max_tokens=512,
        )
    )
    calls = defaultdict(list)

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls[image_path].append((max_tokens, temperature, require_complete))
        if image_path == "bad.jpg":
            raise RuntimeError("still broken")
        return "valid peer caption"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["bad.jpg", "good.jpg"]) == [
        "[Error captioning frame: still broken]",
        "valid peer caption",
    ]
    assert calls == {
        "bad.jpg": [
            (None, None, False),
            (512, 0.0, True),
            (512, 0.0, True),
        ],
        "good.jpg": [(None, None, False)],
    }


def test_retry_ceiling_never_exceeds_normal_caption_ceiling(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            max_tokens=256,
            retry_attempts=1,
            retry_max_tokens=2048,
        )
    )
    calls = []

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls.append((max_tokens, temperature, require_complete))
        if max_tokens is None:
            raise RuntimeError("first attempt failed")
        return "recovered"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["frame.jpg"]) == ["recovered"]
    assert calls == [(None, None, False), (256, 0.0, True)]


def test_caption_frames_persists_each_frame_before_requesting_the_next(
    monkeypatch, tmp_path,
):
    import src.captioner as captioner_module
    from src.config import CaptionBackend, CaptioningConfig

    output_dir = tmp_path / "captions"
    calls = 0

    class FakeCaptioner:
        def caption_batch(self, image_paths):
            nonlocal calls
            calls += 1
            if calls == 2:
                first_record = output_dir / "caption_000010.json"
                assert first_record.exists()
                assert json.loads(first_record.read_text())["caption"] == "caption:first.jpg"
            return [f"caption:{image_paths[0]}"]

    monkeypatch.setattr(
        captioner_module, "_get_captioner", lambda config: FakeCaptioner()
    )
    frames = [
        {"frame_id": 10, "path": "first.jpg"},
        {"frame_id": 11, "path": "second.jpg"},
    ]

    result = captioner_module.caption_frames(
        frames,
        CaptioningConfig(backend=CaptionBackend.ollama, batch_size=1),
        output_dir=str(output_dir),
        record_transform=lambda record: {
            **record,
            "semantic_caption": record["caption"],
            "ocr": f"ocr:{record['frame_id']}",
        },
    )

    assert calls == 2
    first = json.loads((output_dir / "caption_000010.json").read_text())
    assert first["semantic_caption"] == "caption:first.jpg"
    assert first["ocr"] == "ocr:10"
    assert json.loads((output_dir / "all_captions.json").read_text()) == result
