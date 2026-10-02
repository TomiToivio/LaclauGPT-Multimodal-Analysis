"""Guard: ep24_schema's public contract must stay importable.

`ep24_schema.py` is the ONE shared EP24 schema module (issue #57). Other modules
and tests import named symbols from it, so a refactor that makes a helper
"dynamic" and drops a public constant breaks every importer at collection time --
which is exactly what happened when `EP24_REPROCESS_COLUMNS` was removed while
`roihu_rdf.py` still imported it. The whole RDF stage and 24 of its tests went
red, and the failure surfaced only as a collection ImportError.

These tests pin the module's PUBLIC SURFACE rather than its internals, so the
schema can keep evolving (the pipeline is deliberately dynamic and preserves
whatever columns a country's CSV actually has) without silently deleting a name
that something else already depends on.
"""
from __future__ import annotations

import importlib

from ep24_schema import (
    EP24_REPROCESS_COLUMNS,
    LEGACY_ALIASES,
    REQUIRED_MEDIA_COLUMNS,
    source_metadata,
    stable_source_id,
    value,
)


def test_canonical_input_columns_are_defined_here():
    """Issue #57 requires the shared module to define the canonical input columns.

    This is the KEEP-SCHEMA as a contract constant. It must not be dropped just
    because the pipeline reads rows dynamically.
    """
    assert isinstance(EP24_REPROCESS_COLUMNS, tuple)
    assert len(EP24_REPROCESS_COLUMNS) == 13, "the canonical issue #21 keep-schema is 13 columns"
    for column in ("country", "author_username", "allas_filename", "video_id"):
        assert column in EP24_REPROCESS_COLUMNS


def test_required_media_columns_are_defined_here():
    assert tuple(REQUIRED_MEDIA_COLUMNS) == ("video_id", "allas_filename")


def test_legacy_aliases_are_read_only_mappings():
    assert isinstance(LEGACY_ALIASES, dict)
    assert all(isinstance(v, tuple) for v in LEGACY_ALIASES.values())


def test_every_importer_of_the_schema_still_resolves():
    """The real regression: an importer must not be left dangling.

    `roihu_rdf` imports `EP24_REPROCESS_COLUMNS` at module scope. If the constant
    disappears, that module fails to import and its entire test file errors out
    at collection, taking the RDF stage with it.
    """
    for module_name in ("roihu_rdf", "ep24_pipeline"):
        module = importlib.import_module(module_name)
        assert module is not None, module_name


def test_helpers_are_callable_and_dynamic():
    row = {
        "video_id": "vid-1",
        "allas_filename": "allas/vid-1.mp4",
        "country": "Finland",
        "some_future_column": "value",
        "empty": "",
    }
    # `value` reads a field, falling back to a legacy alias.
    assert value(row, "video_id") == "vid-1"
    assert value({"videoId": "legacy-1"}, "video_id") == "legacy-1"
    # `source_metadata` returns every non-empty field, including unknown ones,
    # so a country with extra columns still contributes prompt context.
    keys = [key for key, _ in source_metadata(row)]
    assert "some_future_column" in keys
    assert "empty" not in keys
    assert isinstance(stable_source_id(row), str)


def test_dynamic_read_does_not_require_the_canonical_constant():
    """A row missing canonical columns must still be readable.

    The constant documents the contract; it must NOT be used to filter rows, or
    the "preserve every incoming column" requirement would be violated.
    """
    minimal = {"video_id": "v", "allas_filename": "a.mp4"}
    assert value(minimal, "author_username", default="<absent>") == "<absent>"
    assert source_metadata(minimal) == [("video_id", "v"), ("allas_filename", "a.mp4")]
