"""
Tests for the verbatim transcription path: text-space stitching, scroll-safe
frame selection, and the capability guard that prevents the original
blind-model regression (text-only model used for vision).

Run:  pytest tests/test_transcribe.py -v
"""
import random
from pathlib import Path
from threading import Event

import pytest

from src.stitch import stitch_frames, detect_boilerplate, line_ratio
from src.config import OCRConfig, FrameSelectionConfig
from src.ocr import (
    OCRFrameResult,
    OCRTileExhaustedError,
    VerbatimOCR,
    _NO_IMAGE_RE,
)
from src.omlx_client import InferenceTruncatedError


# ── Stitching ────────────────────────────────────────────────────────────────

def _make_scroll_frames(doc, view=20, step=3, header=None, footer=None, noise=0.0, seed=1):
    rng = random.Random(seed)
    frames, top = [], 0
    while top < len(doc):
        view_lines = doc[top:top + view]
        if noise:
            view_lines = [_noisy(x, rng, noise) for x in view_lines]
        page = 1 + top // view
        rendered = (header or []) + view_lines + ([f.format(page=page) for f in (footer or [])])
        frames.append(rendered)
        top += step
    return frames


def _noisy(line, rng, p):
    if rng.random() < p and line:
        i = rng.randrange(len(line))
        line = line[:i] + rng.choice("aeior ") + line[i + 1:]
    return line


def test_stitch_recovers_document_in_order():
    doc = [f"line {i:02d} content {i*7 % 13}" for i in range(60)]
    frames = _make_scroll_frames(doc, view=20, step=3)
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    # every doc line present, in order
    j = 0
    for d in doc:
        while j < len(out) and line_ratio(d, out[j]) < 0.9:
            j += 1
        assert j < len(out), f"missing line: {d}"
        j += 1


def test_stitch_no_duplication():
    doc = [f"unique row number {i}" for i in range(40)]
    frames = _make_scroll_frames(doc, view=15, step=2)
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    # length must be ~document length, not frames*view (no overlap leak)
    assert len(out) <= len(doc) + 2


def test_stitch_absorbs_exact_duplicate_frames():
    doc = [f"row {i}" for i in range(30)]
    frames = _make_scroll_frames(doc, view=12, step=3)
    frames.insert(3, list(frames[3]))   # static pause
    frames.insert(7, list(frames[7]))
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    assert len(out) <= len(doc) + 2


def test_stitch_tolerates_ocr_noise():
    doc = [f"the model risk validation step {i} requires approval" for i in range(50)]
    frames = _make_scroll_frames(doc, view=18, step=3, noise=0.25, seed=4)
    out = [l for l in stitch_frames(frames, fuzzy=0.8).lines if l.strip()]
    recovered = sum(1 for d in doc if any(line_ratio(d, o) >= 0.75 for o in out))
    assert recovered / len(doc) >= 0.9


def test_stitch_tolerates_dropped_lines():
    # OCR sometimes drops a line inside the overlap; difflib matching blocks
    # must still align around the indel without scrambling or duplicating.
    rng = random.Random(11)
    doc = [f"section {i}: the validation requires model approval step {i}" for i in range(50)]
    frames = _make_scroll_frames(doc, view=18, step=3, noise=0.15, seed=3)
    for fr in frames:                       # randomly drop one mid line per frame
        if len(fr) > 6 and rng.random() < 0.5:
            del fr[rng.randrange(2, len(fr) - 2)]
    out = [l for l in stitch_frames(frames, fuzzy=0.8).lines if l.strip()]
    recovered = sum(1 for d in doc if any(line_ratio(d, o) >= 0.75 for o in out))
    assert recovered / len(doc) >= 0.9
    assert len(out) <= len(doc) * 1.3       # no duplication blow-up


def test_conservative_tile_stitch_never_drops_an_overlap_insertion():
    stitched = stitch_frames(
        [["A", "B", "C"], ["A", "X only visible in later crop", "B", "C", "D"]],
        fuzzy=0.8,
        strip_boilerplate=False,
        preserve_unmatched_overlap=True,
    )
    assert stitched.lines == ["A", "X only visible in later crop", "B", "C", "D"]
    assert stitched.unmatched_seams == 1


def test_conservative_tile_stitch_preserves_fuzzy_numbered_variants():
    tail = [f"section {number} alpha" for number in range(100, 104)]
    head = [
        "section 100 alpha",
        "section 100 alphx",
        "section 101 alpha",
        "section 102 alpha",
        "section 103 alpha",
        "tail",
    ]
    stitched = stitch_frames(
        [tail, head],
        fuzzy=0.8,
        strip_boilerplate=False,
        preserve_unmatched_overlap=True,
    )
    assert "section 100 alphx" in stitched.lines
    for line in set(tail + head):
        assert line in stitched.lines
    assert stitched.unmatched_seams >= 1


def test_strict_stitch_preserves_outer_and_line_whitespace():
    raw = "    first\nsecond  \n\n"
    stitched = stitch_frames(
        [raw.split("\n")],
        strip_boilerplate=False,
        preserve_unmatched_overlap=True,
        preserve_whitespace=True,
    )
    assert stitched.text() == raw


def test_boilerplate_stripped():
    doc = [f"body line {i}" for i in range(40)]
    header = ["UBS MRM Guidelines", "Internal"]
    footer = ["Page {page} of 16", "Published: 30 April 2026"]
    frames = _make_scroll_frames(doc, header=header, footer=footer, view=15, step=3)
    boiler = detect_boilerplate(frames)
    assert any("mrm guidelines" in b for b in boiler)
    out = stitch_frames(frames).lines
    assert not any("of 16" in l for l in out)
    assert not any("MRM Guidelines" in l for l in out)


# ── Capability guard (prevents the blind-model regression) ───────────────────

def test_text_only_model_is_rejected():
    cfg = OCRConfig(model="MiniMax-M2.7")  # text-only — the original bug
    ocr = VerbatimOCR(cfg)
    with pytest.raises(RuntimeError, match="text-only"):
        ocr.assert_vision_capable()


def test_vision_model_passes_guard():
    cfg = OCRConfig(model="mlx-community/olmOCR-2-7B-1025-8bit")
    ocr = VerbatimOCR(cfg)
    ocr.assert_vision_capable()  # must not raise


def test_ocr_frames_reuses_full_budget_probe_for_first_frame(monkeypatch):
    cfg = OCRConfig(
        backend="vllm",
        model="lightonai/LightOnOCR-2-1B",
        max_tokens=777,
        concurrency=1,
    )
    ocr = VerbatimOCR(cfg)
    calls = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        calls.append((path, max_tokens, require_complete))
        return f"text from {path}"

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    assert ocr.ocr_frames(["first.png", "second.png"]) == [
        "text from first.png",
        "text from second.png",
    ]
    assert calls == [
        ("first.png", 777, False),
        ("second.png", 777, False),
    ]


# ── Strict tiled OCR fallback ────────────────────────────────────────────

def _strict_ocr(**overrides):
    values = {
        "backend": "vllm",
        "model": "lightonai/LightOnOCR-2-1B",
        "max_tokens": 16384,
        "concurrency": 3,
    }
    values.update(overrides)
    return VerbatimOCR(OCRConfig(**values))


def _png(path: Path, *, width: int = 320, height: int = 400) -> str:
    from PIL import Image

    Image.new("RGB", (width, height), color="white").save(path)
    return str(path)


def test_strict_ocr_complete_frame_requires_complete_generation(tmp_path, monkeypatch):
    image = _png(tmp_path / "frame.png")
    ocr = _strict_ocr()
    calls = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        calls.append((path, max_tokens, require_complete))
        return "literal line one\nliteral line two"

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    result = ocr.ocr_frame_verbatim(image)

    assert result == OCRFrameResult(
        text="literal line one\nliteral line two",
        status="complete",
        complete=True,
    )
    assert calls == [(image, 16384, True)]


def test_strict_ocr_preserves_model_whitespace_verbatim(tmp_path, monkeypatch):
    image = _png(tmp_path / "frame.png")
    ocr = _strict_ocr()
    raw = "    first\nsecond  \n\n"
    monkeypatch.setattr(ocr, "_chat", lambda *args, **kwargs: raw)
    assert ocr.ocr_frame_verbatim(image).text == raw


def test_strict_ocr_does_not_remove_literal_wrapper_like_visible_text(
    tmp_path, monkeypatch,
):
    image = _png(tmp_path / "frame.png")
    ocr = _strict_ocr()
    raw = "```python\nprint('visible fences')\n```\n<think>visible tag</think>"
    monkeypatch.setattr(ocr, "_chat", lambda *args, **kwargs: raw)
    assert ocr.ocr_frame_verbatim(image).text == raw


def test_strict_ocr_client_does_not_strip_response_whitespace(tmp_path, monkeypatch):
    import json
    import src.omlx_client as omlx_client

    image = _png(tmp_path / "frame.png")
    raw = "    first\nsecond  \n\n<think>literal visible tag</think>"

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": raw}, "finish_reason": "stop"}
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        omlx_client,
        "_urlopen",
        lambda request, timeout: FakeResponse(),
    )
    assert _strict_ocr().ocr_frame_verbatim(image).text == raw


def test_strict_ocr_truncation_uses_ordered_overlapping_png_bands(
    tmp_path, monkeypatch,
):
    from PIL import Image

    image = _png(tmp_path / "dense.png")
    ocr = _strict_ocr()
    band_text = iter(
        [
            "line A body 000\nalpha seam exact 10aa\nalpha seam exact 91bb",
            "alpha seam exact 10aa\nalpha seam exact 91bb\nline B body 111\n"
            "bravo junction exact 20cc\nbravo junction exact 82dd",
            "bravo junction exact 20cc\nbravo junction exact 82dd\nline C body 222\n"
            "charlie overlap exact 30ee\ncharlie overlap exact 73ff",
            "charlie overlap exact 30ee\ncharlie overlap exact 73ff\nline D body 333",
        ]
    )
    calls = []
    band_shapes = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        calls.append((path, max_tokens, require_complete))
        if path == image:
            raise InferenceTruncatedError("vllm", 16384)
        with Image.open(path) as crop:
            band_shapes.append((crop.format, crop.size))
        return next(band_text)

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    result = ocr.ocr_frame_verbatim(image)

    assert result.complete is True
    assert result.status == "complete_tiled"
    assert result.tiled is True
    assert result.tile_requests == 4
    assert result.tile_leaves == 4
    assert result.text.strip().splitlines() == [
        "line A body 000",
        "alpha seam exact 10aa",
        "alpha seam exact 91bb",
        "line B body 111",
        "bravo junction exact 20cc",
        "bravo junction exact 82dd",
        "line C body 222",
        "charlie overlap exact 30ee",
        "charlie overlap exact 73ff",
        "line D body 333",
    ]
    assert result.unmatched_seams == 0
    assert calls[0] == (image, 16384, True)
    assert all(call[1:] == (4096, True) for call in calls[1:])
    # Four 100px logical cores, expanded 48px at each available edge.
    assert band_shapes == [
        ("PNG", (320, 148)),
        ("PNG", (320, 196)),
        ("PNG", (320, 196)),
        ("PNG", (320, 148)),
    ]


def test_strict_ocr_subdivides_only_the_truncated_band_with_bounds(
    tmp_path, monkeypatch,
):
    from PIL import Image

    image = _png(tmp_path / "dense.png")
    ocr = _strict_ocr(
        tile_overlap_ratio=0.10,
        tile_min_overlap_pixels=4,
        tile_max_depth=2,
        tile_min_core_height=20,
        tile_max_requests=12,
    )
    tile_calls = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        if path == image:
            raise InferenceTruncatedError("vllm", 16384)
        with Image.open(path) as crop:
            tile_calls.append((max_tokens, crop.size))
        if len(tile_calls) == 1:
            raise InferenceTruncatedError("vllm", max_tokens)
        return f"leaf {len(tile_calls)}"

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    result = ocr.ocr_frame_verbatim(image)

    assert result.complete is True
    assert result.status == "complete_tiled_uncertain"
    assert result.tile_requests == 6  # one failed band + two children + three peers
    assert result.tile_leaves == 5
    assert result.tile_max_depth == 2
    # Children keep the full band budget: overlap makes each crop larger than
    # half its parent, so halving to 2048 would make recursive fallback weaker.
    assert [limit for limit, _ in tile_calls] == [4096, 4096, 4096, 4096, 4096, 4096]
    assert tile_calls[0][1] == (320, 110)
    assert all(height < 110 for _, (_, height) in tile_calls[1:3])


def test_strict_ocr_exhaustion_is_explicit_and_discards_truncated_prefix(
    tmp_path, monkeypatch,
):
    image = _png(tmp_path / "dense.png")
    ocr = _strict_ocr(tile_max_depth=1, tile_max_requests=4)
    attempted_prefixes = []

    def always_truncated(path, *, max_tokens, require_complete=False):
        attempted_prefixes.append(f"INCOMPLETE PREFIX FROM {path}")
        raise InferenceTruncatedError("vllm", max_tokens)

    monkeypatch.setattr(ocr, "_chat", always_truncated)

    with pytest.raises(OCRTileExhaustedError, match="(?i)(tile|band|truncat)"):
        ocr.ocr_frame_verbatim(image)

    # A failed strict call has no return value through which a generated prefix
    # could leak into the transcript. Only the full frame and first band ran.
    assert len(attempted_prefixes) == 2


def test_strict_batch_persists_tile_exhaustion_diagnostics(tmp_path, monkeypatch):
    image = _png(tmp_path / "dense.png")
    ocr = _strict_ocr(tile_max_depth=1, tile_max_requests=4)

    def always_truncated(path, *, max_tokens, require_complete=False):
        raise InferenceTruncatedError("vllm", max_tokens)

    monkeypatch.setattr(ocr, "_chat", always_truncated)
    result = ocr.ocr_frames_verbatim([image])[0]
    assert result.complete is False
    assert result.status == "failed"
    assert result.tiled is True
    assert result.full_frame_attempted is True
    assert result.tile_requests == 1
    assert result.tile_max_depth == 1


def test_strict_batch_stops_scheduling_after_systemic_worker_error(
    tmp_path, monkeypatch,
):
    paths = [_png(tmp_path / f"frame-{index}.png") for index in range(3)]
    ocr = _strict_ocr(concurrency=1)
    calls = []
    callbacks = []

    def fake_frame(path):
        calls.append(path)
        if path == paths[1]:
            raise RuntimeError("synthetic authentication failure")
        return OCRFrameResult(text=Path(path).stem, status="complete", complete=True)

    monkeypatch.setattr(ocr, "ocr_frame_verbatim", fake_frame)
    with pytest.raises(RuntimeError, match="authentication"):
        ocr.ocr_frames_verbatim(
            paths,
            on_result=lambda index, result: callbacks.append((index, result)),
        )
    assert calls == paths[:2]
    assert [index for index, _ in callbacks] == [0, 1]
    assert callbacks[-1][1].complete is False


def test_strict_ocr_circuit_breaker_makes_later_frames_tile_first(
    tmp_path, monkeypatch,
):
    first = _png(tmp_path / "first.png")
    second = _png(tmp_path / "second.png")
    ocr = _strict_ocr()
    originals_seen = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        if path in {first, second}:
            originals_seen.append(path)
            if path == first:
                raise InferenceTruncatedError("vllm", 16384)
            return "full-frame response should not be requested"
        return "tile text"

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    first_result = ocr.ocr_frame_verbatim(first)
    second_result = ocr.ocr_frame_verbatim(second)

    assert originals_seen == [first]
    assert first_result.tiled is True
    assert first_result.full_frame_attempted is True
    assert second_result.tiled is True
    assert second_result.full_frame_attempted is False


def test_strict_batch_callbacks_follow_completion_order_but_results_do_not(
    tmp_path, monkeypatch,
):
    first = _png(tmp_path / "first.png")
    slow = _png(tmp_path / "slow.png")
    fast = _png(tmp_path / "fast.png")
    paths = [first, slow, fast]
    ocr = _strict_ocr(concurrency=3)
    slow_started = Event()
    release_slow = Event()
    callback_order = []

    def fake_chat(path, *, max_tokens, require_complete=False):
        assert require_complete is True
        if path == slow:
            slow_started.set()
            assert release_slow.wait(timeout=2)
        elif path == fast:
            assert slow_started.wait(timeout=2)
        return f"text from {Path(path).stem}"

    def on_result(index, result):
        callback_order.append(index)
        if index == 2:
            release_slow.set()

    monkeypatch.setattr(ocr, "_chat", fake_chat)

    results = ocr.ocr_frames_verbatim(paths, on_result=on_result)

    assert callback_order == [0, 2, 1]
    assert [result.text for result in results] == [
        "text from first", "text from slow", "text from fast",
    ]
    assert all(result.complete for result in results)


def test_no_image_sentinel_regex():
    assert _NO_IMAGE_RE.search("No image or video frame has been provided.")
    assert _NO_IMAGE_RE.search("Please attach the image you'd like me to analyze.")
    assert not _NO_IMAGE_RE.search("def main():\n    return 0")


def test_tile_request_budget_must_cover_initial_rows():
    with pytest.raises(ValueError, match="tile_max_requests"):
        OCRConfig(tile_rows=4, tile_max_requests=3)


# ── End-to-end glue (mocked OCR server) ──────────────────────────────────────

def test_transcribe_end_to_end_with_mock_ocr(tmp_path, monkeypatch):
    """Full pipeline glue: select → OCR → stitch → write, with no real server."""
    import src.transcribe as T
    from src.config import ScreenLensConfig

    video = tmp_path / "video.mov"
    video.write_bytes(b"fixture-video")
    doc = [f"def step_{i}(x):  # row {i}" for i in range(40)]
    frames = _make_scroll_frames(doc, view=16, step=3)

    def fake_select_frames(video_path, output_dir, config):
        frame_dir = Path(output_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        fake_meta = []
        for i in range(len(frames)):
            path = frame_dir / f"frame_{i:06d}.png"
            path.write_bytes(f"fixture-frame-{i}".encode())
            fake_meta.append(
                {
                    "frame_id": i,
                    "frame_index": i,
                    "timestamp": float(i),
                    "timestamp_str": f"00:00:{i:02d}.000",
                    "path": str(path),
                    "width": 100,
                    "height": 100,
                }
            )
        return fake_meta

    monkeypatch.setattr(T, "select_frames", fake_select_frames)

    class _MockOCR(T.VerbatimOCR):
        def ocr_frames_verbatim(self, paths, *, on_result=None):
            results = [
                OCRFrameResult(
                    text="\n".join(frame),
                    status="complete",
                    complete=True,
                )
                for frame in frames
            ]
            for index, result in enumerate(results):
                if on_result is not None:
                    on_result(index, result)
            return results

    monkeypatch.setattr(T, "VerbatimOCR", _MockOCR)

    cfg = ScreenLensConfig()
    cfg.ocr.model = "lightonai/LightOnOCR-2-1B"
    cfg.ocr.max_tokens = 12345
    cfg.reconstruction.enabled = False      # skip the LLM cleanup (needs server)

    result = T.transcribe_video(str(video), cfg, tmp_path / "run")
    assert result["stage"] == "done"
    assert result["ocr_max_tokens"] == 12345
    transcript = (tmp_path / "run" / "output" / "transcript.md").read_text()
    out = [l for l in transcript.splitlines() if l.strip()]
    # def lines reconstructed without duplication blow-up (glue check, not a
    # precision re-test — see the dedicated stitch tests for that)
    assert sum(1 for d in doc if any(line_ratio(d, o) >= 0.85 for o in out)) >= 34
    assert len(out) <= len(doc) + 3


def test_transcribe_cli_only_routes_an_explicit_shared_url_to_ocr(
    tmp_path, monkeypatch,
):
    import src.transcribe as transcribe_module
    from src.cli import app
    from typer.testing import CliRunner

    video = tmp_path / "clip.mov"
    video.write_bytes(b"video")
    captured = []
    resume_flags = []

    def fake_transcribe(video_path, config, output_dir, *, resume=False):
        captured.append(config)
        resume_flags.append(resume)
        return {
            "stage": "done",
            "frames_selected": 0,
            "frames_with_text": 0,
            "ocr_model": "fake",
            "transcript_path": "fake.md",
        }

    monkeypatch.setattr(transcribe_module, "transcribe_video", fake_transcribe)
    runner = CliRunner()

    implicit = runner.invoke(app, ["transcribe", str(video)])
    assert implicit.exit_code == 0, implicit.output
    assert captured[-1].ocr.base_url is None
    assert captured[-1].ocr.max_tokens == 16384

    explicit = runner.invoke(
        app,
        [
            "transcribe",
            str(video),
            "--inference-url",
            "http://127.0.0.1:8000/v1",
            "--ocr-concurrency",
            "1",
            "--ocr-max-tokens",
            "8192",
        ],
    )
    assert explicit.exit_code == 0, explicit.output
    assert captured[-1].ocr.base_url == "http://127.0.0.1:8000/v1"
    assert captured[-1].ocr.concurrency == 1
    assert captured[-1].ocr.max_tokens == 8192

    config_file = tmp_path / "screenlens.json"
    config_file.write_text(
        '{"ocr":{"max_tokens":12288}}',
        encoding="utf-8",
    )
    configured = runner.invoke(
        app,
        ["transcribe", str(video), "--config-file", str(config_file)],
    )
    assert configured.exit_code == 0, configured.output
    assert captured[-1].ocr.max_tokens == 12288

    revised = runner.invoke(
        app,
        [
            "transcribe",
            str(video),
            "--ocr-model",
            "example/custom-ocr",
            "--ocr-model-revision",
            "abc123",
        ],
    )
    assert revised.exit_code == 0, revised.output
    assert captured[-1].ocr.model == "example/custom-ocr"
    assert captured[-1].ocr.model_revision == "abc123"

    resume_dir = tmp_path / "existing-run"
    resume_dir.mkdir()
    resumed = runner.invoke(
        app,
        ["transcribe", str(video), "--resume-dir", str(resume_dir)],
    )
    assert resumed.exit_code == 0, resumed.output
    assert resume_flags[-1] is True
    assert captured[-1].data_dir == resume_dir.resolve()


# ── Thinking leak regression ─────────────────────────────────────────────────
#
# A reasoning OCR model (e.g. Qwen3.x) emitted chain-of-thought instead of the
# transcription and exhausted max_tokens before closing </think>, so the whole
# response was untagged/truncated reasoning that leaked into transcript.md.

def test_strip_thinking_handles_truncated_open_tag():
    from src.omlx_client import strip_thinking
    # complete block
    assert strip_thinking("<think>reasoning</think>\n\nANSWER") == "ANSWER"
    # dangling close (opening tag was a prompt prefix) — keep the answer
    assert strip_thinking("reasoning</think>\n\nANSWER") == "ANSWER"
    # dangling open, generation truncated mid-thought — no answer survives
    assert strip_thinking("prefix<think>truncated reasoning forever") == "prefix"
    # clean text untouched
    assert strip_thinking("just an answer") == "just an answer"
    assert strip_thinking(
        "    indented\ntrailing  \n",
        preserve_outer_whitespace=True,
    ) == "    indented\ntrailing  \n"


def test_ocr_disables_thinking_in_request_payload(monkeypatch, tmp_path):
    """OCR must send chat_template_kwargs.enable_thinking=false so a reasoning
    model produces the transcription instead of burning the budget on CoT."""
    import json
    from PIL import Image
    import src.omlx_client as omlx_client

    img_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="white").save(img_path)

    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)

    ocr = VerbatimOCR(OCRConfig(model="Qwen3-VL-test", disable_thinking=True))
    assert ocr.ocr_frame(str(img_path)) == "hello"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}

    # When disabled, the knob must NOT be sent.
    captured.clear()
    ocr2 = VerbatimOCR(OCRConfig(model="Qwen3-VL-test", disable_thinking=False))
    ocr2.ocr_frame(str(img_path))
    assert "chat_template_kwargs" not in captured["payload"]


@pytest.mark.parametrize(
    ("model", "family", "prompt", "temperature", "top_p"),
    [
        ("lightonai/LightOnOCR-2-1B", "lightonocr-2", None, 0.2, 0.9),
        ("zai-org/GLM-OCR", "glm-ocr", "Text Recognition:", 0.0, None),
        ("PaddlePaddle/PaddleOCR-VL-1.5", "paddleocr-vl", "OCR:", 0.0, None),
    ],
)
def test_specialized_ocr_models_use_native_request_contract(
    monkeypatch,
    tmp_path,
    model,
    family,
    prompt,
    temperature,
    top_p,
):
    """Small OCR models need their own chat template, not the Qwen prompt."""
    import json
    from PIL import Image
    import src.omlx_client as omlx_client

    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="white").save(image)
    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "visible text"}}]}
            ).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode())
        return FakeResponse()

    monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)
    ocr = VerbatimOCR(OCRConfig(backend="vllm", model=model))
    assert ocr.ocr_frame(str(image)) == "visible text"
    assert ocr.request_profile.family == family

    payload = captured["payload"]
    assert [message["role"] for message in payload["messages"]] == ["user"]
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    if prompt is None:
        assert [item["type"] for item in content] == ["image_url"]
    else:
        assert content[1] == {"type": "text", "text": prompt}
    assert payload["temperature"] == temperature
    assert payload.get("top_p") == top_p
    assert "max_tokens" not in payload
    assert "repetition_penalty" not in payload
    assert "no_repeat_ngram_size" not in payload
    assert "chat_template_kwargs" not in payload


def test_generic_qwen_ocr_keeps_verbatim_prompt_and_sampler(monkeypatch, tmp_path):
    import json
    from PIL import Image
    import src.omlx_client as omlx_client

    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="white").save(image)
    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode())
        return FakeResponse()

    monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)
    ocr = VerbatimOCR(OCRConfig(backend="vllm", model="Qwen3-VL-test"))
    ocr.ocr_frame(str(image))
    payload = captured["payload"]
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"][0]["type"] == "text"
    assert payload["repetition_penalty"] == 1.15
    assert payload["no_repeat_ngram_size"] == 6
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_dedicated_ocr_environment_precedence(monkeypatch):
    from src.omlx_client import (
        DEFAULT_OCR_BASE_URL,
        DEFAULT_OCR_CONTEXT,
        DEFAULT_OCR_MODEL,
        DEFAULT_OMLX_BASE_URL,
        resolve_ocr_api_key,
        resolve_ocr_base_url,
        resolve_ocr_context,
        resolve_ocr_model,
        resolve_ocr_model_revision,
    )

    for name in (
        "OCR_BASE_URL", "VLLM_OCR_BASE_URL", "VLLM_BASE_URL",
        "OCR_MODEL", "VLLM_OCR_MODEL", "VLLM_MODEL",
        "OCR_MODEL_REVISION", "VLLM_OCR_MODEL_REVISION",
        "OCR_API_KEY", "VLLM_OCR_API_KEY", "VLLM_API_KEY",
        "OCR_MAX_MODEL_LEN", "VLLM_OCR_MAX_MODEL_LEN", "VLLM_MAX_MODEL_LEN",
    ):
        # An existing empty variable also prevents a local .env from filling it.
        monkeypatch.setenv(name, "")

    vllm = OCRConfig(backend="vllm")
    assert resolve_ocr_base_url(vllm) == DEFAULT_OCR_BASE_URL
    assert resolve_ocr_model(vllm) == DEFAULT_OCR_MODEL
    assert resolve_ocr_api_key(vllm) == "local"
    assert resolve_ocr_context(vllm) == DEFAULT_OCR_CONTEXT
    assert resolve_ocr_base_url(OCRConfig(backend="omlx")) == DEFAULT_OMLX_BASE_URL

    monkeypatch.setenv("VLLM_BASE_URL", "http://legacy:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "legacy/model")
    monkeypatch.setenv("VLLM_API_KEY", "legacy-key")
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "262144")
    assert resolve_ocr_base_url(vllm) == DEFAULT_OCR_BASE_URL
    assert resolve_ocr_model(vllm) == DEFAULT_OCR_MODEL
    assert resolve_ocr_api_key(vllm) == "local"
    assert resolve_ocr_context(vllm) == DEFAULT_OCR_CONTEXT

    monkeypatch.setenv("VLLM_OCR_BASE_URL", "http://alias:8001/v1")
    monkeypatch.setenv("VLLM_OCR_MODEL", "alias/model")
    monkeypatch.setenv("VLLM_OCR_API_KEY", "alias-key")
    monkeypatch.setenv("VLLM_OCR_MAX_MODEL_LEN", "14336")
    assert resolve_ocr_base_url(vllm) == "http://alias:8001/v1"
    assert resolve_ocr_model(vllm) == "alias/model"
    assert resolve_ocr_api_key(vllm) == "alias-key"
    assert resolve_ocr_context(vllm) == 14336

    monkeypatch.setenv("OCR_BASE_URL", "http://neutral:8002/v1/")
    monkeypatch.setenv("OCR_MODEL", "neutral/model")
    monkeypatch.setenv("OCR_API_KEY", "neutral-key")
    monkeypatch.setenv("OCR_MAX_MODEL_LEN", "12288")
    assert resolve_ocr_base_url(vllm) == "http://neutral:8002/v1"
    assert resolve_ocr_model(vllm) == "neutral/model"
    assert resolve_ocr_api_key(vllm) == "neutral-key"
    assert resolve_ocr_context(vllm) == 12288

    monkeypatch.setenv("OCR_MODEL_REVISION", "neutral-revision")
    assert resolve_ocr_model_revision(vllm) == "neutral-revision"

    explicit = OCRConfig(
        backend="vllm",
        base_url="http://explicit:9000",
        model="explicit/model",
        api_key="explicit-key",
    )
    assert resolve_ocr_base_url(explicit) == "http://explicit:9000/v1"
    assert resolve_ocr_model(explicit) == "explicit/model"
    assert resolve_ocr_api_key(explicit) == "explicit-key"
    # An explicit model must not inherit the revision of a different model
    # selected through the environment.
    assert resolve_ocr_model_revision(explicit) is None
    explicit.model_revision = "explicit-revision"
    assert resolve_ocr_model_revision(explicit) == "explicit-revision"


# ── Cleanup never drops content (coverage guard) ─────────────────────────────
#
# An LLM (esp. a reasoning model) silently condenses — dropping code blocks /
# lists despite "never remove content". The guard falls back to the raw stitched
# chunk whenever the repaired output dropped too many input lines.

def test_chunk_coverage_metric():
    from src.transcribe import _chunk_coverage
    src = "line one\n    line two\nline three"
    assert _chunk_coverage(src, "line one\nline two\n  line three") == 1.0  # reindent ok
    assert round(_chunk_coverage(src, "line one\nline three"), 2) == 0.67   # dropped a line
    assert _chunk_coverage(src, "") == 0.0


def test_cleanup_falls_back_to_raw_when_llm_drops_content(monkeypatch):
    import src.omlx_client as omlx_client
    import src.transcribe as T
    from src.config import ScreenLensConfig

    raw = "\n\n".join(f"keep_line_{i} = {i}" for i in range(20))

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            # The "LLM" returns only the first line — a gross content drop.
            import json
            return json.dumps(
                {"choices": [{"message": {"content": "keep_line_0 = 0"}}]}
            ).encode()

    monkeypatch.setattr(omlx_client, "_urlopen", lambda req, timeout: FakeResponse())

    cfg = ScreenLensConfig()
    out = T._cleanup_transcript(raw, cfg)
    # All original lines survive because the guard discarded the lossy LLM output.
    for i in range(20):
        assert f"keep_line_{i} = {i}" in out


# ── Scroll-safe frame selection on a REAL recording ──────────────────────────

REAL_VIDEO = Path(__file__).resolve().parents[1] / "input" / "policies.mov"


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="sample recording not present")
def test_select_frames_on_real_video(tmp_path):
    from src.frame_select import select_frames
    meta = select_frames(str(REAL_VIDEO), str(tmp_path), FrameSelectionConfig(sample_fps=2.0))
    assert len(meta) > 10                       # got real frames
    assert all(Path(m["path"]).exists() for m in meta)
    # timestamps strictly increasing
    ts = [m["timestamp"] for m in meta]
    assert ts == sorted(ts)
