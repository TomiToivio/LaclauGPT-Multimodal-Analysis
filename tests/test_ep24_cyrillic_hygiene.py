"""Tests for the Latin/Cyrillic script-hygiene detector (issue #91, BG pass).

Bulgaria is the only Cyrillic-script EP24 country. Mixing writing systems inside
one token yields characters that RENDER correctly and therefore survive human
review while defeating exact matching permanently. These tests pin that the
detector finds those, and only those.

The critical property is the NEGATIVE one: a legitimate bilingual label such as
`DPS (ДПС)` pairs the scripts in separate tokens and must NOT be reported. A
detector that flags it is noise, and a noisy gate gets ignored.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "cyrillic_hygiene", ROOT / "scripts" / "ep24" / "cyrillic_hygiene.py"
)
assert _SPEC and _SPEC.loader
hygiene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hygiene)

CYR_EM = "\u041c"  # CYRILLIC CAPITAL LETTER EM — looks like Latin M
CYR_ES = "\u0421"  # CYRILLIC CAPITAL LETTER ES — looks like Latin C


# --------------------------------------------------------------------------
# The defect: a script boundary INSIDE one token
# --------------------------------------------------------------------------


def test_single_token_mixing_scripts_is_a_defect() -> None:
    label = f"European Parliament (SE{CYR_EM})"
    assert hygiene.is_mixed_script(label), "'SEМ' mixes Latin S,E with Cyrillic М"

    finding = hygiene.audit_text(label)
    assert finding is not None
    assert finding["homoglyphs"], "the Cyrillic lookalike must be reported"
    hom = finding["homoglyphs"][0]
    assert hom["codepoint"] == "U+041C"
    assert hom["latin_lookalike"] == "M"
    assert hom["token"] == f"SE{CYR_EM}"


def test_nfc_does_not_repair_a_homoglyph() -> None:
    """NFC cannot fix it: the characters are canonically unrelated.

    This is why the defect is silent — normalisation cannot know which writing
    system the author intended, so the mismatch survives every normalising pass.
    """
    label = f"European Parliament (SE{CYR_EM})"
    assert hygiene.unicodedata.normalize("NFC", label) == label
    assert hygiene.audit_text(label)["nfc_changes"] is False


# --------------------------------------------------------------------------
# The negative controls: legitimate bilingual forms must NOT be reported
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "DPS (ДПС)",                                    # Latin token + Cyrillic token
        "ДПС (DPS)",
        "Солидарна България (Solidarity Bulgaria)",      # Cyrillic + Latin tokens
        "ГЕРБ – СДС",                                    # wholly Cyrillic
        "GERB-SDS",                                      # wholly Latin
        "Бойко Борисов",                                 # wholly Cyrillic person
        "Boyko Borisov",                                 # wholly Latin person
    ],
)
def test_legitimate_bilingual_and_single_script_labels_are_clean(label: str) -> None:
    assert not hygiene.is_mixed_script(label), f"{label!r} is a legitimate form"
    assert hygiene.audit_text(label) is None


def test_correct_cyrillic_is_not_reported_as_a_homoglyph() -> None:
    """Inside a Cyrillic word, `о`/`е` are correct Cyrillic, not lookalikes.

    Reporting them would bury the real finding in noise (this was the first
    version's bug: it flagged 9 labels when only 1 was defective). The assertion
    counts ALL reported characters, not just the mixed token's, so a scan that
    reports every lookalike anywhere in the label fails here.
    """
    finding = hygiene.audit_text(f"ПП (PP{CYR_ES})")
    assert finding is not None
    assert len(finding["homoglyphs"]) == 1, "only the mixed token is reported"
    assert finding["homoglyphs"][0]["token"] == f"PP{CYR_ES}"

    # A wholly Cyrillic label has no defect token at all, so nothing is reported
    # even though it contains the lookalike characters о/е/р/с.
    assert hygiene.audit_text("Движение за права и свободи") is None
    assert hygiene.find_homoglyphs("Движение за права и свободи") == []


# --------------------------------------------------------------------------
# File-level behaviour
# --------------------------------------------------------------------------


def _book(root: Path, name: str, labels: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(
            {
                "schema": "t",
                "country_code": "BG",
                "entries": [
                    {"kind": "entity", "label": lab, "aliases": []} for lab in labels
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_audit_scans_labels_and_aliases(tmp_path: Path) -> None:
    root = tmp_path / "codebooks"
    _book(root, "ep24_bg_private.json", [f"SE{CYR_EM}", "DPS (ДПС)"])
    report = hygiene.audit(root, "BG")
    assert report["mixed_script_count"] == 1
    assert report["findings"][0]["field"] == "label"


def test_audit_does_not_modify_the_codebook(tmp_path: Path) -> None:
    """The detector must never rewrite a researcher-grounded label."""
    root = tmp_path / "codebooks"
    _book(root, "ep24_bg_private.json", [f"SE{CYR_EM}"])
    path = root / "ep24_bg_private.json"
    before = path.read_bytes()
    hygiene.audit(root, "BG")
    assert path.read_bytes() == before


def test_lfs_pointer_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    root = tmp_path / "codebooks"
    _book(root, "ep24_bg_private.json", ["Clean Label"])
    (root / "countries").mkdir(parents=True, exist_ok=True)
    (root / "countries" / "bg.json").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:53fe3cfbb213d80e31b9621fac679fa06f2f7a9990126e2ec02ff8fed2e84775\n"
        "size 2664\n",
        encoding="utf-8",
    )
    report = hygiene.audit(root, "BG")
    assert "bg.json" in report["skipped_unreadable"]
    assert report["mixed_script_count"] == 0


def test_resolves_the_repo_country_map_not_the_slug(tmp_path: Path) -> None:
    """Finland's file is `ep24_finland_private.json`, not `ep24_fi_private.json`."""
    names = [p.name for p in hygiene._candidate_files(tmp_path, "FI")]
    assert "ep24_finland_private.json" in names
