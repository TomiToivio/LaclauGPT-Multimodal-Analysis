"""Step 2 Poland review log: frames captured before feed content loaded."""
from laclaugpt_quality import frame_quality_decision


def test_phone_home_screen_is_recoverable():
    text = ("The frame shows an Android / GrapheneOS phone home screen "
            "(app launcher), not a social-media feed or any political scene. "
            "First row of app icons labelled Aurora Store, TikTok, Instagram.")
    status, reason = frame_quality_decision(text)
    assert status == "REPROCESS"
    assert "launcher" in reason


def test_instagram_loading_placeholder_is_recoverable():
    text = ("Instagram Reels shows an empty video canvas with no photographic "
            "content and skeleton/loading placeholders. (5) Image usability: OK.")
    assert frame_quality_decision(text)[0] == "REPROCESS"


def test_normal_political_frame_is_ok():
    text = ("A candidate is speaking in front of party flags. "
            "Poster says LEWICA. (5) Image usability: OK.")
    assert frame_quality_decision(text)[0] == "OK"


def test_regular_mentions_of_loading_are_not_overflagged():
    text = ("The scene shows a political rally with a crowd; a person is "
            "loading campaign leaflets into a car. (5) Image usability: OK.")
    assert frame_quality_decision(text)[0] == "OK"
