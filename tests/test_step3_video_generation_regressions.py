"""Step 3 regression examples from Poland failed-batch review."""
from experiments.vllm_video_test import validate_video_analysis, parse_args


def test_repeated_nonclassification_loop_is_rejected():
    preface = "A political video shows a speaker and an outdoor campaign poster. "
    loop = "The video is not a documentary, but rather a political commentary. " * 30
    assert validate_video_analysis(preface + loop)[0] is False


def test_empty_verdict_with_uncertainty_is_not_valid_video_description():
    response = ("OK\n\nUncertainty:\n"
                "- The exact identity is not confirmed.\n"
                "- No party affiliation can be established from the frames.\n"
                "- The political message is inferred from metadata.\n")
    valid, reason = validate_video_analysis(response)
    assert valid is False
    assert reason == "missing_video_description"


def test_concise_specific_video_description_is_accepted():
    response = ("A Polish TikTok video shows a politician standing beside a campaign poster "
                "with a PiS logo and a border-defence slogan. The speaker gestures with a "
                "paper bag. The camera remains fixed; visible greenery and a fence are "
                "behind the poster. The candidate's identity is supported by poster text. "
                "The video is OK.\n\n"
                '{"SCROLL": false, "SCROLL_SECONDS": []}')
    assert validate_video_analysis(response)[0] is True


def test_step3_conservative_generation_defaults():
    args = parse_args([])
    assert args.temperature == 0.1
    assert args.presence_penalty == 0.0
    assert args.repetition_penalty == 1.12
