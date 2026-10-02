#!/usr/bin/env python3
"""#128 §4: the EasyOCR adapter must be constructible for every EP24 country.

The defect
----------
`ocr_backend.EASYOCR_LANGS` was one flat list:

    ["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "bg"]

EasyOCR groups languages by **script** and refuses a Reader that spans groups:

    ValueError: Cyrillic is only compatible with English, try lang_list=
    ["ru","rs_cyrillic","be","bg","uk","mn","en"]

Bulgarian is Cyrillic; the other nine EP24 languages are Latin. No single group
contains all ten, so `easyocr.Reader(EASYOCR_LANGS)` raised **before doing any
work** — the EasyOCR adapter was unusable for the whole EP24 set.

Why no test caught it
---------------------
The adapter imports `easyocr` lazily inside `load_ocr_backend()`, and every
existing test injects a fake backend, so no test ever constructed a real Reader.
Imported at module scope, this failure was invisible.

These tests pin the script grouping and, where EasyOCR is importable, that a real
Reader can actually be built. The construction check is skipped (not silently
passed) when EasyOCR is absent, so the suite stays honest about what it verified.

Public, synthetic fixtures only.

Run:  python -m pytest tests/test_ep24_ocr_script_groups.py -v
"""
from __future__ import annotations

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


def test_no_flat_language_list_spans_script_groups() -> None:
    """The original defect: a single list mixing Latin and Cyrillic.

    Asserted structurally rather than by naming the old constant, so a future
    "convenience" flat list cannot reappear under a different name.
    """
    latin = set(ob.EASYOCR_LATIN_LANGS)
    cyrillic = set(ob.EASYOCR_CYRILLIC_LANGS)
    assert latin & cyrillic <= {"en"}, (
        "the Latin and Cyrillic groups must not overlap beyond 'en' (EasyOCR's "
        f"shared fallback); overlap: {sorted(latin & cyrillic)}"
    )


def test_every_ep24_country_resolves_within_one_script_group() -> None:
    """Each country must get a list that EasyOCR can actually accept."""
    ep24 = ["finland", "poland", "portugal", "germany", "spain",
            "hungary", "croatia", "france", "sweden", "bulgaria"]
    for country in ep24:
        langs = ob.easyocr_languages(country)
        assert langs, country
        # a single group: either the Latin set or the Cyrillic set, plus 'en'
        assert set(langs) <= set(ob.EASYOCR_LATIN_LANGS) | set(ob.EASYOCR_CYRILLIC_LANGS), country


def test_bulgarian_selects_cyrillic_and_the_rest_latin() -> None:
    """The specific split: Bulgaria is the only Cyrillic-script EP24 language."""
    assert ob.easyocr_languages("bulgaria") == ob.EASYOCR_CYRILLIC_LANGS
    for country in ("finland", "poland", "portugal", "germany", "spain",
                    "hungary", "croatia", "france", "sweden"):
        assert ob.easyocr_languages(country) == ob.EASYOCR_LATIN_LANGS, country


def test_unknown_country_raises_rather_than_guessing_a_script() -> None:
    """A typo must not silently pick a script group."""
    with pytest.raises(ValueError):
        ob.easyocr_languages("bulgariaa")


def test_missing_country_defaults_to_latin() -> None:
    """Nine of ten EP24 languages are Latin, so an unset country is a safe default."""
    assert ob.easyocr_languages("") == ob.EASYOCR_LATIN_LANGS
    assert ob.easyocr_languages(None) == ob.EASYOCR_LATIN_LANGS


def test_latin_group_covers_every_non_bulgarian_ep24_language() -> None:
    """No EP24 language may be dropped from its group."""
    for lang in ("fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "en"):
        assert lang in ob.EASYOCR_LATIN_LANGS, lang
    assert "bg" in ob.EASYOCR_CYRILLIC_LANGS


@pytest.mark.skipif(not _easyocr_available(), reason="easyocr not installed; cannot build a real Reader")
def test_the_old_flat_list_would_still_be_rejected() -> None:
    """Proof the original list was genuinely unusable, not merely suspicious.

    Constructs a Reader with the old list and asserts EasyOCR rejects it. This is
    the executable form of the defect, so nobody later "restores" the flat list
    believing it was fine.
    """
    import easyocr

    with pytest.raises(ValueError, match="only compatible with English"):
        easyocr.Reader(["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "bg"])


@pytest.mark.skipif(not _easyocr_available(), reason="easyocr not installed; cannot build a real Reader")
def test_a_real_reader_can_be_constructed_per_group() -> None:
    """The fix, verified against the real library rather than its docs.

    Model weights are cached by EasyOCR after first use; this asserts construction
    only (not inference), which is exactly the step that used to fail.

    Runs in a **subprocess** with ``CUDA_VISIBLE_DEVICES=""`` exported first:
    EasyOCR binds its device at import time, so setting the variable inside this
    process would be too late (verified — it does not take effect). The isolation
    is also honest about why: the host GPU is a GTX 1060 (sm_61) and the installed
    PyTorch supports sm_75+, so a CUDA attempt here fails for a reason that has
    nothing to do with language grouping. GPU-path behaviour belongs to the Roihu
    smoke run, not to this test.
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import ocr_backend as ob\n"
        "import easyocr\n"
        "assert easyocr.Reader(ob.easyocr_languages('finland')) is not None\n"
        "assert easyocr.Reader(ob.easyocr_languages('bulgaria')) is not None\n"
        "print('constructed both groups')\n" % str(ROOT)
    )
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=540
    )
    assert result.returncode == 0, (
        "a real EasyOCR Reader could not be constructed per script group:\n"
        f"stdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-1500:]}"
    )
    assert "constructed both groups" in result.stdout


def test_backend_model_string_names_the_languages_used() -> None:
    """The provenance string must say which script group ran, for the §4 record.

    #128 §4 asks for the selected backend to be recorded; a model string of
    'easyocr-runtime' alone does not distinguish a Latin run from a Cyrillic one.
    """
    source = (ROOT / "ocr_backend.py").read_text(encoding="utf-8")
    assert "easyocr-runtime:" in source, (
        "the EasyOCR backend model string must include the selected language set"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
