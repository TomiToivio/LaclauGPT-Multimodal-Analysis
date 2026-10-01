import json
from pathlib import Path

from roihu_codebooks import context_block, load_profile
from roihu_memory import SCHEMA_VERSION, EP24Memory, stable_id, upstream_stable_id


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
    assert memory.resolve("Example Party", "actor", country="SE").decision == "NEW"

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


def test_locked_human_entry_wins_regardless_of_layer_merge_order(tmp_path, monkeypatch):
    import roihu_codebooks

    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    common = codebooks / "ep24_common_private.json"
    country = codebooks / "ep24_finland_private.json"
    _write_book(common, {
        "schema": "fixture", "country_code": "FI", "entries": [
            {"id": "human-1", "kind": "actor", "label": "Example Party",
             "definition": "Researcher-coded definition", "status": "researcher-grounded",
             "locked": True}
        ]
    })
    _write_book(country, {
        "schema": "fixture", "country_code": "FI", "entries": [
            {"id": "external-1", "kind": "actor", "label": "Example Party",
             "definition": "External-source definition", "status": "PROVISIONAL"}
        ]
    })
    monkeypatch.setattr(
        roihu_codebooks,
        "profile_paths",
        lambda _root, _country: [(country, "country"), (common, "common")],
    )

    entries, meta = load_profile(tmp_path, "FI")

    assert len(entries) == 1
    assert entries[0].entry_id == "human-1"
    assert entries[0].definition == "Researcher-coded definition"
    assert meta["conflicts"] == [
        {"kept": "human-1", "rejected": "external-1", "reason": "human_lock"}
    ]

    monkeypatch.setattr(
        roihu_codebooks,
        "profile_paths",
        lambda _root, _country: [(common, "common"), (country, "country")],
    )
    entries, _meta = load_profile(tmp_path, "FI")

    assert entries[0].entry_id == "human-1"
    assert entries[0].definition == "Researcher-coded definition"




def test_enrichment_appends_columns_without_rewriting_legacy_values(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("pandas")
    from roihu_enrich import enrich_file
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    _write_book(codebooks / "ep24_common_private.json", {"schema": "fixture", "country_code": "COMMON", "entries": []})
    _write_book(codebooks / "ep24_finland_private.json", {"schema": "fixture", "country_code": "FI", "language": "fi", "entries": [{"id": "fi-actor", "kind": "entity", "label": "Puolue", "english_label": "Party", "status": "researcher-grounded"}]})
    csv_path = tmp_path / "ep24_fi.csv"
    pd = __import__("pandas")
    original = pd.DataFrame([{"video_filename": "a/1", "summary_analysis": "Puolue esiintyy videolla", "entities": "Puolue", "topics": "demokratia"}])
    original.to_csv(csv_path, index=False)
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    obj_id = memory.add_object("entity", "Puolue", country="FI", language="fi", state="CANONICAL", locked=True)
    enrich_file(csv_path, country="FI", language="fi", private_root=tmp_path, memory=memory)
    out = pd.read_csv(csv_path)
    assert out.loc[0, "entities"] == "Puolue"
    assert out.loc[0, "topics"] == "demokratia"
    assert obj_id in out.loc[0, "ep24_memory_entity_ids"]
    assert "ep24_codebook_context_json" in out.columns


def test_seed_memory_does_not_attach_conflicting_alias(tmp_path):
    from roihu_enrich import seed_memory

    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    _write_book(codebooks / "ep24_common_private.json", {"schema": "fixture", "country_code": "COMMON", "entries": []})
    for code, filename in {
        "FI": "ep24_finland_private.json", "SE": "ep24_se_private.json", "PL": "ep24_poland_private.json",
        "PT": "ep24_pt_private.json", "DE": "ep24_de_private.json", "ES": "ep24_es_private.json",
        "HU": "ep24_hu_private.json", "HR": "ep24_hr_private.json", "FR": "ep24_fr_private.json",
        "BG": "ep24_bg_private.json",
    }.items():
        entries = []
        if code == "FI":
            entries = [{"id": "cb-fi-new", "kind": "actor", "label": "New Actor", "aliases": ["Taken Alias"], "status": "researcher-grounded", "locked": True}]
        _write_book(codebooks / filename, {"schema": "fixture", "country_code": code, "entries": entries})

    memory = EP24Memory(tmp_path / "memory.sqlite3")
    existing = memory.add_object("actor", "Existing Actor", country="FI", state="CANONICAL", locked=True)
    memory.add_alias(existing, "Taken Alias", country="FI")
    result = seed_memory(tmp_path, memory)
    assert result["alias_conflicts"] == 1
    assert memory.resolve("Taken Alias", "actor", country="FI").obj_id == existing


def test_populism_context_hook_is_opt_in_and_explicit():
    source = Path("roihu_populism.py").read_text(encoding="utf-8")
    assert "LACLAUGPT_ENRICHMENT_ENABLED" in source
    assert "add_codebook_context(country, user_prompt)" in source
    assert "formula_of_populism_codebook_context_json" in source
    assert "legacy_cached_result" in source


def test_memory_schema_migration_creates_backup_and_temporal_columns(tmp_path):
    import sqlite3

    path = tmp_path / "legacy-memory.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO meta VALUES('schema_version','1')")
        db.execute(
            """CREATE TABLE objects(
                obj_id TEXT PRIMARY KEY, kind TEXT NOT NULL, canonical_label TEXT NOT NULL,
                original_label TEXT NOT NULL, english_label TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT '', disambiguation TEXT NOT NULL DEFAULT '',
                definition TEXT NOT NULL DEFAULT '', state TEXT NOT NULL DEFAULT 'PROVISIONAL',
                origin TEXT NOT NULL DEFAULT '', locked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        db.execute("PRAGMA user_version=1")

    memory = EP24Memory(path)
    assert path.with_suffix(".sqlite3.schema-v1.bak").exists()
    with memory.connect() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(objects)")}
        assert {"valid_from", "valid_to"}.issubset(columns)
        # Compare against the module's SCHEMA_VERSION, never a literal: a hardcoded
        # number goes stale the moment the schema is bumped (it did, when #74
        # added the `relations` table and raised the version to 3).
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        # The migration must carry the newer schema across, not just bump a number.
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "relations" in tables, "a migrated db must gain the tables added after v1"


def test_proposal_shard_merge_is_deterministic_and_idempotent(tmp_path):
    canonical = EP24Memory(tmp_path / "canonical.sqlite3")
    shard_b = EP24Memory(tmp_path / "b.sqlite3")
    shard_a = EP24Memory(tmp_path / "a.sqlite3")
    shard_b.propose("topic", "beta", country="FI", reason="model", source_record_id="2", run_id="run-b", stage="enrich")
    shard_a.propose("topic", "alpha", country="FI", reason="model", source_record_id="1", run_id="run-a", stage="enrich")

    first = canonical.merge_proposal_shards([shard_b.path, shard_a.path])
    second = canonical.merge_proposal_shards([shard_a.path, shard_b.path])
    assert first == {"inserted": 2, "duplicates": 0, "conflicts": 0}
    assert second == {"inserted": 0, "duplicates": 2, "conflicts": 0}
    with canonical.connect() as db:
        labels = [row[0] for row in db.execute("SELECT raw_label FROM proposals ORDER BY proposal_id")]
    assert sorted(labels) == ["alpha", "beta"]


def test_disabled_enrichment_is_noop_without_runtime_dependencies(tmp_path, monkeypatch, capsys):
    from roihu_enrich import main

    monkeypatch.delenv("LACLAUGPT_ENRICHMENT_ENABLED", raising=False)
    result = main([
        "--private-root", str(tmp_path),
        "--memory-db", str(tmp_path / "disabled.sqlite3"),
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["enabled"] is False
    assert payload["status"] == "no-op"


def test_review_states_locks_redirects_and_crosswalk_survive_rerun(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    locked = memory.add_object(
        "entity",
        "Locked Name",
        country="FI",
        definition="human definition",
        state="CANONICAL",
        origin="researcher_private",
        locked=True,
        preserve_upstream_id="legacy-E-1",
    )
    memory.add_object(
        "entity",
        "Locked Name",
        obj_id=locked,
        country="FI",
        definition="model overwrite attempt",
        state="CANONICAL",
        origin="model",
        locked=False,
    )
    provisional = memory.add_object("topic", "Candidate Topic", country="FI")
    memory.set_state(provisional, "REJECTED")
    replacement = memory.add_object("entity", "Replacement", country="FI", state="CANONICAL")
    memory.redirect(locked, replacement, reason="researcher split/merge correction")

    with memory.connect() as db:
        row = db.execute("SELECT definition,locked FROM objects WHERE obj_id=?", (locked,)).fetchone()
        assert row["definition"] == "human definition"
        assert row["locked"] == 1
        assert db.execute("SELECT ep24_id FROM id_crosswalk WHERE upstream_id='legacy-E-1'").fetchone()[0] == locked
        assert db.execute("SELECT new_id FROM redirects WHERE old_id=?", (locked,)).fetchone()[0] == replacement
        assert db.execute("SELECT state FROM objects WHERE obj_id=?", (provisional,)).fetchone()[0] == "REJECTED"
    assert memory.resolve("Candidate Topic", "topic", country="FI").decision == "NEW"
