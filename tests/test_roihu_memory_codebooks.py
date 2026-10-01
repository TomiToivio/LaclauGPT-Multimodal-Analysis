import json
from pathlib import Path

from roihu_codebooks import context_block, load_profile
from roihu_memory import EP24Memory, stable_id, upstream_stable_id


def test_context_aware_ids_preserve_unicode_and_country_collisions():
    assert stable_id("actor", "Åsa Romson", country="SE", language="sv") != stable_id("actor", "Asa Romson", country="SE", language="sv")
    assert stable_id("actor", "Progress", country="FI") != stable_id("actor", "Progress", country="PL")
    assert upstream_stable_id("actor", "Åsa (MEP)") == upstream_stable_id("actor", "Asa")


def test_memory_acceptance_ambiguity_lock_and_snapshot(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    fi = memory.add_object("actor", "Example Party", country="FI", language="fi", state="CANONICAL", origin="researcher_private", locked=True)
    pl = memory.add_object("actor", "Example Party", country="PL", language="pl", state="CANONICAL", origin="researcher_private", locked=True)
    assert memory.resolve("Example Party", "actor").decision == "AMBIGUOUS"
    assert memory.resolve("Example Party", "actor", country="FI").obj_id == fi
    assert memory.resolve("Example Party", "actor", country="PL").obj_id == pl

    proposal = memory.propose("topic", "new model guess", country="FI", reason="model-discovered", source_record_id="row-1")
    assert proposal.startswith("P-")
    assert memory.resolve("new model guess", "topic", country="FI").decision == "NEW"

    snap = memory.snapshot(tmp_path / "frozen.sqlite3")
    frozen = EP24Memory(snap)
    assert frozen.resolve("Example Party", "actor", country="FI").obj_id == fi


def _write_book(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_private_profile_merge_keeps_locked_human_entry_and_bilingual_context(tmp_path):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    _write_book(codebooks / "ep24_common_private.json", {
        "schema": "fixture", "country_code": "COMMON", "entries": [
            {"id": "common-1", "kind": "topic", "label": "ilmasto", "english_label": "climate", "language": "fi", "definition": "Ilmastopolitiikka", "english_definition": "Climate policy", "status": "researcher-grounded"}
        ]
    })
    _write_book(codebooks / "ep24_finland_private.json", {
        "schema": "fixture", "country_code": "FI", "language": "fi", "entries": [
            {"id": "fi-locked", "kind": "actor", "label": "Esimerkkipuolue", "english_label": "Example Party", "aliases": ["EP"], "status": "researcher-grounded", "locked": True}
        ]
    })
    entries, meta = load_profile(tmp_path, "FI", language="fi")
    block, provenance = context_block("EP puhuu ilmastosta", entries, country="FI", language="fi")
    assert meta["entry_count"] == 2
    assert "Esimerkkipuolue / Example Party" in block
    assert "ilmasto / climate" in block
    assert provenance["evidence_role"] == "background_context_not_source_evidence"
