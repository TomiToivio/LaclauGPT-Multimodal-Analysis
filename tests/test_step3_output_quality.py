from experiments.vllm_video_test import bounded_video_context, validate_video_analysis
import pandas as pd


def test_prioritizes_record_evidence_not_cross_record_rag(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_VIDEO_CONTEXT_MAX_CHARS", "7000")
    row = pd.Series({"video_id": "abc", "asr_transcript": "interview " * 500,
                     "rag_context_json": "irrelevant " * 10000,
                     "memory_context_json": "irrelevant other video " * 5000})
    prompt = bounded_video_context(row)
    assert "abc" in prompt and "interview" in prompt
    assert "irrelevant" not in prompt
    assert len(prompt) < 7600


def test_rejects_minimal_ok_output():
    ok, reason = validate_video_analysis('OK\\n\\nUncertainty: unknown.\\n{"SCROLL":false,"SCROLL_SECONDS":[]}')
    assert not ok and reason == "analysis_too_short"


def test_rejects_repetition_loop():
    stem = "This is not a podcast nor a radio show nor a tutorial or any other show. "
    ok, reason = validate_video_analysis("It begins with a political speech. " + stem * 30)
    assert not ok and reason == "degenerate_repetition"


def test_accepts_substantive_unique_analysis():
    body = "The opening scene features a person addressing a camera in a public square. "
    body += "A flag appears beside the speaker while campaign text moves on screen. "
    body += "The camera then shifts to a larger crowd and returns to a close view. "
    body += "A timestamp is present on the interface, but the exact date is unclear. "
    ok, reason = validate_video_analysis(body + '{"SCROLL":false,"SCROLL_SECONDS":[]}')
    assert ok, reason


def test_step3_32b_default_and_recommended_sampling():
    from experiments.vllm_video_test import DEFAULT_MODEL, SYSTEM_PROMPT, parse_args
    assert DEFAULT_MODEL == "Qwen/Qwen3-VL-32B-Instruct"
    args = parse_args([])
    assert (args.temperature, args.top_p, args.top_k) == (0.7, 0.8, 20)
    assert (args.repetition_penalty, args.presence_penalty) == (1.0, 1.5)
    assert "European Parliament" in SYSTEM_PROMPT
    assert "temporal" in SYSTEM_PROMPT
