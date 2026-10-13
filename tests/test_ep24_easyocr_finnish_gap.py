#!/usr/bin/env python3
"""EasyOCR covers nine of the ten EP24 languages — record that Finnish is missing.

Main already fixed the *crash*: `EASYOCR_SCRIPT_GROUPS` splits Latin and Cyrillic
so `easyocr.Reader` can be constructed (see #128 / #139). It does not record that
`fi` is absent from every group, or why — so a Finnish overlay silently OCRs to
nothing, and the natural future edit ("EP24 has ten languages, add the missing
one") would reintroduce the unloadable-list error.

These tests make the gap explicit and check the support claim against the library
itself rather than trusting a comment.

Public, synthetic fixtures only.

Run:  python -m pytest tests/test_ep24_easyocr_finnish_gap.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ocr_backend as ob  # noqa: E402


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
    except Exception:
        return False
    return True


def test_finnish_is_recorded_as_unsupported() -> None:
    """The gap must be named, not merely implied by absence from a list."""
    assert "fi" in ob.EASYOCR_UNSUPPORTED_LANGS


def test_no_script_group_requests_the_missing_finnish_model() -> None:
    """Asking for `fi` raises and takes the whole Reader down, so it must not appear."""
    for name, members in ob.EASYOCR_SCRIPT_GROUPS.items():
        assert "fi" not in members, f"group {name!r} requests the missing Finnish model"


def test_the_other_nine_ep24_languages_are_all_present() -> None:
    """Recording one gap must not hide a second: nine languages, nine present.

    `en` is included because every EP24 run needs it for the shared Latin reader.
    """
    present = {lang for members in ob.EASYOCR_SCRIPT_GROUPS.values() for lang in members}
    for lang in ("en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "bg"):
        assert lang in present, lang
    assert len(present) == 10, f"expected 10 covered languages, got {sorted(present)}"


def test_requesting_finnish_raises_rather_than_silently_dropping_it() -> None:
    """A caller who asks for `fi` must be told, not handed a Latin-only reader.

    `easyocr_script_groups` already raises on unknown languages; this pins that
    Finnish specifically goes through that path, so its absence is loud.
    """
    with pytest.raises(ValueError):
        ob.easyocr_script_groups(["fi"])


def test_finnish_is_not_quietly_mapped_onto_a_latin_reader() -> None:
    """Regression guard for the tempting wrong fix.

    Substituting Swedish for Finnish (both Nordic, both Latin script) would make the
    crash go away while producing wrong text with no signal. Finnish must be
    rejected, not replaced.
    """
    with pytest.raises(ValueError):
        ob.easyocr_script_groups(["fi", "en"])


@pytest.mark.skipif(not _easyocr_available(), reason="easyocr not installed; cannot probe support")
def test_easyocr_supports_the_nine_and_rejects_finnish() -> None:
    """Check the claim against easyocr in a subprocess, not against a comment.

    Run out-of-process with `CUDA_VISIBLE_DEVICES=""`: easyocr and torch bind their
    device at import time, so an in-process override would not take effect, and the
    host GPU here (GTX 1060, sm_61) is rejected by the installed PyTorch (sm_75+).

    If easyocr ever ships a Finnish model this fails loudly — close the gap
    deliberately (add `fi` to the Latin group, drop it from
    EASYOCR_UNSUPPORTED_LANGS) rather than deleting the test.
    """
    import subprocess

    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import easyocr\n"
        "for code in ['en','fr','pl','sv','pt','de','es','hu','hr','bg']:\n"
        "    easyocr.Reader([code] if code == 'en' else [code, 'en'], gpu=False, verbose=False)\n"
        "print('nine-ok')\n"
        "try:\n"
        "    easyocr.Reader(['fi', 'en'], gpu=False, verbose=False)\n"
        "    print('finnish-supported')\n"
        "except ValueError:\n"
        "    print('finnish-unsupported')\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=540
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "nine-ok" in result.stdout, result.stdout[-800:]
    assert "finnish-unsupported" in result.stdout, (
        "easyocr now constructs a Finnish Reader; close the recorded gap in "
        "EASYOCR_UNSUPPORTED_LANGS and EASYOCR_SCRIPT_GROUPS deliberately"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
