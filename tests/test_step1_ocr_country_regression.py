"""Step 1 log regressions: duplicate script OCR and empty ASR restart cache."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from ocr_backend import describe_ocr_backend, load_ocr_backend


def test_finland_easyocr_does_not_run_cyrillic_pass(monkeypatch):
    calls = []
    class Reader:
        def __init__(self, languages):
            calls.append(tuple(languages))
        def readtext(self, path):
            return [([0, 0], "Finnish on-screen text", 0.9)]
    monkeypatch.setenv("LACLAUGPT_OCR_ENGINE", "easyocr")
    monkeypatch.delenv("LACLAUGPT_OCR_LANGS", raising=False)
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    ocr = load_ocr_backend("Finland")
    assert len(calls) == 1
    assert "bg" not in calls[0]
    assert ocr.read("frame.jpg") == ("Finnish on-screen text", 1)
    assert describe_ocr_backend("Finland")["model"] == "easyocr-runtime+latin"


def test_bulgaria_easyocr_uses_cyrillic_once(monkeypatch):
    calls = []
    class Reader:
        def __init__(self, languages):
            calls.append(tuple(languages))
        def readtext(self, path):
            return [([0, 0], "Bulgarian text", 0.9)]
    monkeypatch.setenv("LACLAUGPT_OCR_ENGINE", "easyocr")
    monkeypatch.delenv("LACLAUGPT_OCR_LANGS", raising=False)
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    backend = load_ocr_backend("Bulgaria")
    assert len(calls) == 1
    assert calls[0] == ("bg", "en")
    assert backend.read("frame.jpg")[1] == 1
    assert describe_ocr_backend("Bulgaria")["model"] == "easyocr-runtime+cyrillic"


def test_explicit_ocr_languages_keep_user_override(monkeypatch):
    calls = []
    class Reader:
        def __init__(self, languages):
            calls.append(tuple(languages))
        def readtext(self, path):
            return []
    monkeypatch.setenv("LACLAUGPT_OCR_ENGINE", "easyocr")
    monkeypatch.setenv("LACLAUGPT_OCR_LANGS", "en,bg")
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    load_ocr_backend("Finland")
    assert len(calls) == 2
