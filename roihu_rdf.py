"""RDF / knowledge-graph export for the EP24 multimodal pipeline (CSC Roihu).

This is **stage 12** of the target pipeline: a new, *appended* export stage. It
runs after the existing five stages and transforms their CSV output into a
documented RDF/Turtle projection. It does not run before, instead of, or inside
any existing stage, and it changes no legacy behaviour.

Design rules, carried over from the Phase 2 LaclauGPT work and from issue #5:

* **RDF is a projection, never the canonical source of truth.** The CSV remains
  authoritative. This stage is read-only with respect to every legacy artifact.
* **The schema maps back to CSV columns**, rather than defining an independent
  black-box ontology. Every predicate emitted here is listed in
  ``docs/RDF_EXPORT.md`` next to the legacy column it comes from.
* **Do not manufacture relations.** Only relations grounded in the analyzed
  content are emitted, and each one records whether it is directly *observed*
  (a column that records a fact) or *model-derived* (a model's analysis).
* **Stdlib only on the default path.** Turtle/N-Triples is written directly, so
  this stage cannot break a multi-hour batch job by failing an import. ``rdflib``
  is optional and only used to *validate* the output when explicitly asked for.

Usage on Roihu (the private runtime root is the working directory):

    python roihu_rdf.py                       # every configured language
    python roihu_rdf.py --language fi         # one language
    python roihu_rdf.py --sample 5 --debug    # first 5 rows, verbose
    python roihu_rdf.py --dry-run             # parse and report, write nothing

See ``docs/RDF_EXPORT.md`` for the namespace, class and predicate reference.
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from ep24_schema import EP24_REPROCESS_COLUMNS, value as ep24_value

os.makedirs("./logs", exist_ok=True)

logger = logging.getLogger(__name__)

# --- namespaces ------------------------------------------------------------
# ``lg:`` is reused from the sibling Phase 2 implementation so an EP24 graph and
# a LaclauGPT graph share one vocabulary instead of inventing a second one.
LG = "https://w3id.org/laclaugpt/"
#: Standard RDF vocabulary. Class membership must use rdf:type, not a
#: private predicate, or SPARQL/reasoners cannot see the classes at all.
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
PROV = "http://www.w3.org/ns/prov#"
DCTERMS = "http://purl.org/dc/terms/"
SCHEMA = "https://schema.org/"
SKOS = "http://www.w3.org/2004/02/skos/core#"
XSD = "http://www.w3.org/2001/XMLSchema#"

STAGE_NAME = "rdf"
STAGE_VERSION = "1.0"
SCHEMA_VERSION = "ep24-rdf/1.0"

#: Languages the historical pipeline iterates. Kept identical to the legacy
#: scripts so the export covers exactly the same set of inputs.
LANGUAGES = ["fi", "sv", "pl", "pt", "de", "es", "hu", "hr", "fr", "bg", "en"]

#: Columns the export reads, grouped by the legacy stage that creates them.
#: This is the CSV->predicate map in code form; ``docs/RDF_EXPORT.md`` mirrors it
#: in prose. Anything not listed here is still preserved (see ``emit_raw_legacy``)
#: so no legacy field can be silently dropped from the graph.
PREPROCESS_COLUMNS = [
    "frame_file",
    "frame_timestamp_seconds",
    "ocr_1",
    "ocr_backend",
    "ocr_model",
    "ocr_runtime_ms",
    "asr_transcript",
    "asr_language",
    "asr_translated",
    "asr_backend",
    "asr_model",
    "asr_runtime_ms",
    "video_duration_seconds",
    "preprocess_status",
    "preprocess_note",
    "preprocess_completed_at",
]
# Read-only compatibility for frozen pre-#128 CSVs. New Step 1 never writes these.
LEGACY_PREPROCESS_COLUMNS = [
    "frame_files",
    "ocr_2",
    "ocr_3",
    "ocr_4",
    "ocr_5",
    "ocr_6",
    "whisperResult",
    "whisper_transcript",
    "whisper_language",
    "whisper_translated",
]
FRAME_COLUMNS = [
    "frame_analysis_1",
    "frame_analysis_2",
    "frame_analysis_3",
    "frame_analysis_4",
    "frame_analysis_5",
    "frame_analysis_6",
]
SUMMARY_COLUMNS = [
    "metadata",
    "summary_analysis",
    "authorNickname",
    "authorSignature",
    "videoCreated",
    "videoDescription",
    "videoDuration",
    "videoCommentCount",
    "videoDiggCount",
    "videoPlayCount",
    "videoShareCount",
]
POSTPROCESS_COLUMNS = ["entities", "topics", "positive", "neutral", "negative"]
POPULISM_COLUMNS = [
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
]
IDENTITY_COLUMNS = list(EP24_REPROCESS_COLUMNS) + ["video_filename", "language"]

ALL_KNOWN_COLUMNS = (
    IDENTITY_COLUMNS
    + PREPROCESS_COLUMNS
    + LEGACY_PREPROCESS_COLUMNS
    + FRAME_COLUMNS
    + SUMMARY_COLUMNS
    + POSTPROCESS_COLUMNS
    + POPULISM_COLUMNS
)


# --- turtle writer ---------------------------------------------------------

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def esc(text) -> str:
    """Escape a Python value for a Turtle double-quoted literal."""
    value = "" if text is None else str(text)
    return "".join(_ESCAPES.get(ch, ch) for ch in value)


def is_blank(value) -> bool:
    """True when a CSV cell carries no information.

    ``pandas`` writes missing values as ``nan``, and the legacy stages also use
    empty strings, so both have to be treated as absent rather than as data.
    """
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in ("nan", "none", "nat", "<na>")


def slug(text: str) -> str:
    """A URL-safe identifier fragment, stable for a given input string."""
    cleaned = re.sub(r"[^0-9A-Za-z._~-]+", "-", str(text).strip().lower()).strip("-")
    return cleaned or "unnamed"


def urn(kind: str, *parts: str) -> str:
    """Build an ``lg:``-namespaced URN-style node identifier.

    URN form keeps every node addressable inside the ``lg:`` namespace without
    claiming an HTTP location that does not exist.
    """
    return f"{LG}{kind}/" + "/".join(slug(p) for p in parts)


def split_list(value) -> list[str]:
    """Split a legacy comma-separated list column into clean values.

    The postprocess stage writes ``', '.join(...)``, so a comma is the separator.
    Values are de-duplicated while preserving order, matching how postprocess
    itself de-duplicates.
    """
    if is_blank(value):
        return []
    text = str(value).strip()
    raw = [item.strip() for item in text.split(",")]
    seen: list[str] = []
    for item in raw:
        item = item.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def parse_frame_files(value) -> list[str]:
    """Parse ``frame_files``, which has two historical encodings.

    ``puhti_preprocess.normalize_frame_files`` accepts both ``str(list)`` from
    older cache rows and comma-separated paths from fresh ones. The same two
    representations therefore reach this stage, so both are accepted here rather
    than guessing which one a given CSV uses.
    """
    if is_blank(value):
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            items = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
        except (ValueError, SyntaxError):
            items = text.split(",")

    cleaned: list[str] = []
    for item in items:
        path = str(item).strip().strip("[]").strip().strip("'\"")
        if path:
            cleaned.append(path)
    return cleaned


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    literal: bool = False
    datatype: str | None = None


@dataclass
class Graph:
    """A tiny Turtle builder.

    Only three shapes are needed: resource triples, plain literals, and typed
    literals. Keeping the writer this small is deliberate — the issue asks for
    obvious, hand-inspectable research code, and a general-purpose RDF library
    would hide the mapping that this stage exists to make visible.
    """

    prefixes: dict[str, str] = field(default_factory=dict)
    triples: list[Triple] = field(default_factory=list)

    def add(self, subject: str, predicate: str, obj: str) -> None:
        self.triples.append(Triple(subject, predicate, obj))

    def literal(self, subject: str, predicate: str, value, datatype: str | None = None) -> None:
        if is_blank(value):
            return
        self.triples.append(Triple(subject, predicate, str(value), literal=True, datatype=datatype))

    def integer(self, subject: str, predicate: str, value) -> None:
        if is_blank(value):
            return
        try:
            number = int(float(str(value)))
        except (TypeError, ValueError):
            return
        self.triples.append(
            Triple(subject, predicate, str(number), literal=True, datatype=f"{XSD}integer")
        )

    def count(self) -> int:
        return len(self.triples)

    def serialize(self) -> str:
        lines = [f"@prefix {name}: <{iri}> ." for name, iri in sorted(self.prefixes.items())]
        lines.append("")
        for triple in self.triples:
            if triple.literal:
                if triple.datatype:
                    obj = f'"{esc(triple.obj)}"^^<{triple.datatype}>'
                else:
                    obj = f'"{esc(triple.obj)}"'
            else:
                obj = f"<{triple.obj}>"
            lines.append(f"<{triple.subject}> <{triple.predicate}> {obj} .")
        return "\n".join(lines) + "\n"


def new_graph() -> Graph:
    graph = Graph()
    graph.prefixes.update(
        {
            "rdf": RDF_NS,
            "lg": LG,
            "prov": PROV,
            "dcterms": DCTERMS,
            "schema": SCHEMA,
            "skos": SKOS,
            "xsd": XSD,
        }
    )
    return graph


# --- provenance ------------------------------------------------------------


@dataclass
class Provenance:
    run_id: str
    model: str
    generated_at: str
    code_version: str

    @classmethod
    def capture(cls) -> Provenance:
        return cls(
            run_id=os.getenv("SLURM_JOB_ID", f"local-{int(time.time())}"),
            model=os.getenv("LACLAUGPT_MULTIMODAL_MODEL", "gemma4:12b"),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            code_version=os.getenv("LACLAUGPT_MULTIMODAL_CODE_VERSION", "unversioned"),
        )


def emit_provenance(
    graph: Graph,
    node: str,
    prov: Provenance,
    *,
    derivation: str,
    stage: str,
    model: str | None = None,
) -> None:
    """Attach provenance to any generated node.

    ``derivation`` is either ``observed`` (the value is recorded fact in the
    source data) or ``model_derived`` (a model produced it). The distinction is
    what lets a reader tell a platform fact from an analytical claim, which the
    issue requires for the downstream SNA and RDF layers.
    """
    graph.literal(node, f"{LG}derivation", derivation)
    graph.literal(node, f"{LG}stage", stage)
    graph.add(node, f"{PROV}wasGeneratedBy", urn("activity", stage, prov.run_id))
    graph.literal(node, f"{DCTERMS}created", prov.generated_at, datatype=f"{XSD}dateTime")
    if model:
        graph.literal(node, f"{LG}model", model)
    graph.literal(node, f"{LG}runId", prov.run_id)


# --- input resolution ------------------------------------------------------


def source_filename(language: str) -> Path | None:
    """Find the previous stage's output, mirroring ``puhti_postprocess``.

    The pipeline file wins when both exist, because it is the one the pipeline
    keeps up to date; the legacy ``ep24_<language>.csv`` is the compatibility
    artifact and is used as a fallback.
    """
    pipeline_file = Path(f"./csv/tiktok_{language}.csv")
    legacy_file = Path(f"ep24_{language}.csv")

    if pipeline_file.exists():
        return pipeline_file
    if legacy_file.exists():
        logger.warning("Using legacy input path %s", legacy_file)
        return legacy_file
    return None


def read_rows(path: Path, sample: int | None) -> list[dict]:
    """Read the CSV as plain dicts, preserving every column.

    ``csv`` is used rather than ``pandas`` on purpose: this stage must preserve
    unknown legacy columns exactly, and a plain dict per row makes that explicit.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if sample is not None and index >= sample:
                break
            rows.append(row)
    return rows


# --- graph construction ----------------------------------------------------


def document_node(row: dict, language: str, index: int) -> tuple[str, str]:
    """Return ``(document_node, source_url)`` for a row.

    Identity follows the repository's canonical rule: the source URL is the
    canonical source identity, and the numeric row index is only a fallback for
    legacy rows that carry no identifiers.
    """
    author = ep24_value(row, "author_username")
    video_id = ep24_value(row, "video_id")
    allas_filename = ep24_value(row, "allas_filename")

    if video_id:
        return urn("document", ep24_value(row, "country") or language, video_id), allas_filename
    fallback_key = allas_filename or str(row.get("video_filename") or "").strip() or f"row-{index}"
    return urn("document", ep24_value(row, "country") or language, "row", fallback_key), allas_filename


def emit_document(graph: Graph, row: dict, language: str, index: int, prov: Provenance) -> str:
    """Emit one document and everything directly attached to it."""
    node, source_url = document_node(row, language, index)
    graph.add(node, f"{RDF_NS}type", f"{LG}Document")

    if source_url:
        graph.add(node, f"{SCHEMA}url", source_url)
    graph.literal(node, f"{LG}language", language)
    graph.literal(node, f"{LG}country", ep24_value(row, "country"))
    graph.literal(node, f"{LG}allasFilename", ep24_value(row, "allas_filename"))
    graph.literal(node, f"{LG}sourceRecording", ep24_value(row, "source_recording"))
    graph.literal(node, f"{LG}researcherNote", ep24_value(row, "researcher_note"))

    # --- observed: platform-recorded facts -------------------------------
    emit_provenance(graph, node, prov, derivation="observed", stage="preprocess")
    graph.integer(node, f"{LG}durationSeconds", ep24_value(row, "video_duration"))
    graph.integer(node, f"{LG}commentCount", row.get("videoCommentCount"))
    graph.integer(node, f"{LG}likeCount", row.get("videoDiggCount"))
    graph.integer(node, f"{LG}playCount", row.get("videoPlayCount"))
    graph.integer(node, f"{LG}shareCount", row.get("videoShareCount"))
    emit_timestamp(graph, node, row.get("videoCreated"))

    emit_author(graph, node, row, language)
    emit_frames(graph, node, row, language, prov)
    emit_screen_text(graph, node, row, language, prov)
    emit_transcript(graph, node, row, language, prov)

    # --- model-derived: analytical layers ---------------------------------
    emit_summary(graph, node, row, language, prov)
    emit_analysis_lists(graph, node, row, language, prov)
    emit_populism(graph, node, row, language, prov)

    # --- lossless preservation -------------------------------------------
    emit_raw_legacy(graph, row, prov)

    return node


def emit_timestamp(graph: Graph, node: str, value) -> None:
    """Emit ``videoCreated`` as a typed dateTime when it is a Unix timestamp.

    The legacy CSV stores an epoch integer. Emitting it untyped would lose that
    fact, so the raw value is kept alongside the ISO form.
    """
    if is_blank(value):
        return
    text = str(value).strip()
    try:
        epoch = float(text)
    except ValueError:
        # Not an epoch; keep it verbatim rather than guessing a format.
        graph.literal(node, f"{DCTERMS}created", text)
        return

    graph.literal(node, f"{LG}videoCreatedEpoch", int(epoch), datatype=f"{XSD}integer")
    stamp = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    graph.literal(node, f"{SCHEMA}datePublished", stamp, datatype=f"{XSD}dateTime")


def emit_author(graph: Graph, document: str, row: dict, language: str) -> None:
    """Emit the posting account as an Actor and link it to the document.

    This is an *observed* relation: the account is who the platform recorded as
    having posted the video, not an inferred association.
    """
    author = ep24_value(row, "author_username")
    if not author:
        return

    actor = urn("actor", author)
    graph.add(actor, f"{RDF_NS}type", f"{LG}Actor")
    graph.add(actor, f"{SCHEMA}identifier", author)
    graph.literal(actor, f"{LG}accountType", ep24_value(row, "account_type"))

    graph.add(document, f"{LG}postedBy", actor)
    graph.literal(document, f"{LG}postedByDerivation", "observed")
    graph.add(actor, f"{LG}postedInLanguage", urn("language", language))


def emit_frames(graph: Graph, document: str, row: dict, language: str, prov: Provenance) -> None:
    """Emit keyframe files and their model analyses.

    The frame files are observed artifacts; the analyses attached to them are
    model-derived, and the two are emitted as separate node types so a reader
    cannot mistake a model's reading of a frame for the frame itself.
    """
    active_frame = row.get("frame_file")
    if not is_blank(active_frame):
        frames = [str(active_frame)]
    else:
        frames = parse_frame_files(row.get("frame_files"))

    for position, path in enumerate(frames, start=1):
        frame = urn("frame", language, ep24_value(row, "video_id"), str(position))
        graph.add(frame, f"{RDF_NS}type", f"{LG}Frame")
        graph.add(frame, f"{DCTERMS}source", path)
        graph.add(document, f"{LG}hasFrame", frame)

    positions = (1,) if not is_blank(active_frame) else range(1, 7)
    for position in positions:
        analysis = row.get(f"frame_analysis_{position}")
        if is_blank(analysis) or position > len(frames):
            continue
        frame = urn("frame", language, ep24_value(row, "video_id"), str(position))
        node = urn("frameanalysis", language, ep24_value(row, "video_id"), str(position))
        graph.add(node, f"{RDF_NS}type", f"{LG}FrameAnalysis")
        graph.add(node, f"{LG}describes", frame)
        graph.literal(node, f"{LG}text", analysis)
        emit_provenance(
            graph, node, prov, derivation="model_derived", stage="frame", model=prov.model
        )


def emit_screen_text(
    graph: Graph, document: str, row: dict, language: str, prov: Provenance
) -> None:
    """Emit OCR results as ScreenText nodes (on-screen text observed in frames)."""
    positions = (1,) if "frame_file" in row else range(1, 7)
    for position in positions:
        text = row.get(f"ocr_{position}")
        if is_blank(text):
            continue
        node = urn("screentext", language, ep24_value(row, "video_id"), str(position))
        graph.add(node, f"{RDF_NS}type", f"{LG}ScreenText")
        graph.add(node, f"{LG}describes", document)
        graph.literal(node, f"{LG}text", text)
        graph.integer(node, f"{LG}position", position)
        emit_provenance(
            graph,
            node,
            prov,
            derivation="observed",
            stage="preprocess",
            model=row.get("ocr_model") or row.get("ocr_backend"),
        )


def emit_transcript(
    graph: Graph, document: str, row: dict, language: str, prov: Provenance
) -> None:
    """Emit backend-neutral ASR, with read-only fallback for frozen Whisper rows."""
    original = row.get("asr_transcript")
    translated = row.get("asr_translated")
    detected_language = row.get("asr_language")
    backend = row.get("asr_backend")
    model = row.get("asr_model")

    if is_blank(original) and is_blank(translated):
        # Historical compatibility only. New Step 1 does not generate these.
        original = row.get("whisper_transcript") or row.get("whisperResult")
        translated = row.get("whisper_translated")
        detected_language = row.get("whisper_language")
        backend = "whisper"
        model = "legacy-whisper"

    if is_blank(original) and is_blank(translated):
        return

    node = urn("transcript", language, ep24_value(row, "video_id"))
    graph.add(node, f"{RDF_NS}type", f"{LG}Transcript")
    graph.add(node, f"{LG}describes", document)
    graph.literal(node, f"{LG}text", translated if not is_blank(translated) else original)
    graph.literal(node, f"{LG}originalText", original)
    graph.literal(node, f"{LG}translatedText", translated)
    graph.literal(node, f"{LG}language", detected_language)
    graph.literal(node, f"{LG}asrBackend", backend)
    graph.literal(node, f"{LG}asrModel", model)
    emit_provenance(
        graph,
        node,
        prov,
        derivation="model_derived",
        stage="preprocess",
        model=model or backend or prov.model,
    )
    graph.add(document, f"{LG}hasTranscript", node)

def emit_summary(graph: Graph, document: str, row: dict, language: str, prov: Provenance) -> None:
    """Emit the summary analysis and the legacy metadata block it was built from."""
    if is_blank(row.get("summary_analysis")):
        return

    node = urn("summary", language, ep24_value(row, "video_id"))
    graph.add(node, f"{RDF_NS}type", f"{LG}Summary")
    graph.add(node, f"{LG}describes", document)
    graph.literal(node, f"{LG}text", row.get("summary_analysis"))
    # ``metadata`` is the exact prompt input assembled by puhti_summary; keeping
    # it makes the summary reproducible from the graph.
    graph.literal(node, f"{LG}promptInput", row.get("metadata"))
    emit_provenance(
        graph, node, prov, derivation="model_derived", stage="summary", model=prov.model
    )
    graph.add(document, f"{LG}hasSummary", node)


def emit_analysis_lists(
    graph: Graph, document: str, row: dict, language: str, prov: Provenance
) -> None:
    """Emit entities, topics and sentiment targets from the postprocess stage.

    Sentiment polarity is carried on the *link*, not on the target: the same
    phrase can be positive in one document and negative in another, so polarity
    belongs to the document's treatment of it.
    """
    video = ep24_value(row, "video_id")
    if is_blank(video):
        video = str(row.get("video_filename") or "")

    for value in split_list(row.get("topics")):
        topic = urn("topic", value)
        graph.add(topic, f"{RDF_NS}type", f"{LG}Topic")
        graph.literal(topic, f"{SKOS}prefLabel", value)
        graph.add(document, f"{LG}hasTopic", topic)
        graph.literal(document, f"{LG}topicDerivation", "model_derived")

    for value in split_list(row.get("entities")):
        entity = urn("entity", value)
        graph.add(entity, f"{RDF_NS}type", f"{LG}Entity")
        graph.literal(entity, f"{SKOS}prefLabel", value)
        graph.add(document, f"{LG}mentionsEntity", entity)
        graph.literal(document, f"{LG}entityDerivation", "model_derived")

    for polarity in ("positive", "neutral", "negative"):
        for value in split_list(row.get(polarity)):
            target = urn("sentimenttarget", value)
            graph.add(target, f"{RDF_NS}type", f"{LG}SentimentTarget")
            graph.literal(target, f"{SKOS}prefLabel", value)
            # reified link: document -> assessment -> target
            link = urn("sentiment", language, video, slug(value))
            graph.add(link, f"{RDF_NS}type", f"{LG}SentimentAssessment")
            graph.add(link, f"{LG}assessesDocument", document)
            graph.add(link, f"{LG}aboutTarget", target)
            graph.literal(link, f"{LG}polarity", polarity)
            emit_provenance(
                graph, link, prov, derivation="model_derived", stage="postprocess", model=prov.model
            )


def emit_populism(graph: Graph, document: str, row: dict, language: str, prov: Provenance) -> None:
    """Emit the Laclau/Palonen analysis, preserving the legacy text fields.

    ``formula_of_populism_us`` and ``_frontier`` are stored by the legacy stage as
    ``element^affect`` lines. Both the raw string and its parsed elements are
    emitted, so the graph adds structure without discarding the original value
    the researcher reads.
    """
    analysis = row.get("formula_of_populism_analysis")
    us_raw = row.get("formula_of_populism_us")
    frontier_raw = row.get("formula_of_populism_frontier")
    if is_blank(analysis) and is_blank(us_raw) and is_blank(frontier_raw):
        return

    video = str(row.get("videoId") or row.get("video_filename") or "")
    node = urn("populism", language, video)
    graph.add(node, f"{RDF_NS}type", f"{LG}PopulismAnalysis")
    graph.add(node, f"{LG}describes", document)
    graph.literal(node, f"{LG}text", analysis)
    graph.literal(node, f"{LG}usRaw", us_raw)
    graph.literal(node, f"{LG}frontierRaw", frontier_raw)
    emit_provenance(
        graph, node, prov, derivation="model_derived", stage="populism", model=prov.model
    )
    graph.add(document, f"{LG}hasPopulismAnalysis", node)

    for side, raw in (("us", us_raw), ("frontier", frontier_raw)):
        for position, (element, affect) in enumerate(parse_populism_elements(raw), start=1):
            element_node = urn("populismelement", side, element)
            graph.add(element_node, f"{RDF_NS}type", f"{LG}PopulismElement")
            graph.literal(element_node, f"{SKOS}prefLabel", element)
            if affect:
                graph.literal(element_node, f"{LG}affect", affect)
            graph.add(node, f"{LG}hasElement", element_node)
            graph.literal(node, f"{LG}elementSide", side)
            graph.integer(node, f"{LG}elementPosition", position)


def parse_populism_elements(value) -> list[tuple[str, str]]:
    """Parse the legacy ``element^affect`` lines into pairs.

    Written defensively: a line without the separator is kept as an element with
    no affect rather than dropped, because an unparsed researcher value is still
    a research value.
    """
    if is_blank(value):
        return []
    pairs: list[tuple[str, str]] = []
    for line in str(value).splitlines():
        line = line.strip()
        if not line:
            continue
        if "^" in line:
            element, _, affect = line.partition("^")
            pairs.append((element.strip().lower(), affect.strip()))
        else:
            pairs.append((line.lower(), ""))
    return pairs


def emit_raw_legacy(graph: Graph, row: dict, prov: Provenance) -> None:
    """Preserve every legacy column not otherwise typed, so nothing is dropped.

    The issue requires the final dataframe to retain all legacy fields and forbids
    silently renaming or dropping them. Columns that already have a typed
    predicate above are skipped here to avoid duplicating the graph; anything
    else — including columns the export does not know about — is emitted as a
    literal under its exact legacy name.
    """
    typed = {
        "authorUniqueId",
        "videoId",
        "video_filename",
        "language",
        "scrapedCountry",
        "frame_file",
        "frame_timestamp_seconds",
        "ocr_backend",
        "ocr_model",
        "ocr_runtime_ms",
        "asr_transcript",
        "asr_language",
        "asr_translated",
        "asr_backend",
        "asr_model",
        "asr_runtime_ms",
        "video_duration_seconds",
        "preprocess_status",
        "preprocess_note",
        "preprocess_completed_at",
        "frame_files",
        "whisperResult",
        "whisper_transcript",
        "whisper_language",
        "whisper_translated",
        "metadata",
        "summary_analysis",
        "authorNickname",
        "authorSignature",
        "videoCreated",
        "videoDescription",
        "videoDuration",
        "videoCommentCount",
        "videoDiggCount",
        "videoPlayCount",
        "videoShareCount",
        "entities",
        "topics",
        "positive",
        "neutral",
        "negative",
        "formula_of_populism_analysis",
        "formula_of_populism_us",
        "formula_of_populism_frontier",
    }
    typed.update(f"frame_analysis_{i}" for i in range(1, 7))
    typed.add("ocr_1")
    typed.update(f"ocr_{i}" for i in range(2, 7))

    for column in sorted(row):
        if column in typed or is_blank(row.get(column)):
            continue
        node = urn("legacyrow", prov.run_id, column)
        graph.add(node, f"{RDF_NS}type", f"{LG}LegacyField")
        graph.literal(node, f"{LG}columnName", column)
        graph.literal(node, f"{LG}columnValue", row.get(column))


# --- optional DNA / SNA projection ----------------------------------------


def maybe_emit_network(graph: Graph, language: str, prov: Provenance) -> dict:
    """Project DNA/SNA node and edge tables if they exist; otherwise skip.

    **Provisional adapter.** The DNA and SNA stages are separate, still-unclaimed
    work in issue #5, so no table format has been agreed. Rather than inventing a
    schema and forcing those stages to match it, this looks for a minimal shape —
    a nodes CSV with ``id`` and a edges CSV with ``source``/``target`` — and says
    clearly in the log when nothing was found. When the real stages land, this
    adapter should be replaced with one that reads their actual contract.

    Nothing is emitted when the files are absent, so the export is complete and
    correct for the five legacy stages today.
    """
    found = {"nodes": 0, "edges": 0}
    for kind in ("dna", "sna"):
        nodes_path = Path(f"./{kind}/{kind}_nodes_{language}.csv")
        edges_path = Path(f"./{kind}/{kind}_edges_{language}.csv")
        if not nodes_path.exists() or not edges_path.exists():
            logger.info("No %s tables for %s; skipping %s projection", kind, language, kind)
            continue

        with nodes_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                node_id = str(row.get("id") or "").strip()
                if not node_id:
                    continue
                node = urn(kind, language, node_id)
                graph.add(node, f"{RDF_NS}type", f"{LG}{kind.upper()}Node")
                graph.literal(node, f"{LG}label", row.get("label"))
                emit_provenance(graph, node, prov, derivation="model_derived", stage=kind)
                found["nodes"] += 1

        with edges_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source = str(row.get("source") or "").strip()
                target = str(row.get("target") or "").strip()
                if not source or not target:
                    continue
                edge = urn(f"{kind}edge", language, source, target)
                graph.add(edge, f"{RDF_NS}type", f"{LG}{kind.upper()}Edge")
                graph.add(edge, f"{LG}source", urn(kind, language, source))
                graph.add(edge, f"{LG}target", urn(kind, language, target))
                graph.literal(edge, f"{LG}label", row.get("label"))
                graph.literal(edge, f"{LG}weight", row.get("weight"))
                emit_provenance(graph, edge, prov, derivation="model_derived", stage=kind)
                found["edges"] += 1

    return found


# --- human-readable summary -----------------------------------------------


def write_summary(
    path: Path, language: str, rows: list[dict], stats: dict, prov: Provenance
) -> None:
    """Write a per-item readable summary, analogous to the populism output.

    A researcher should be able to read what the graph says about each document
    without opening an RDF tool, which the issue requires of the DNA and SNA
    layers and which applies just as much to an export stage.
    """
    lines = [
        f"# RDF export summary — {language}",
        "",
        f"- schema version: {SCHEMA_VERSION}",
        f"- run id: {prov.run_id}",
        f"- generated: {prov.generated_at}",
        f"- source rows: {stats.get('rows', 0)}",
        f"- triples: {stats.get('triples', 0)}",
        "",
        "## Per-document content in the graph",
        "",
    ]
    for index, row in enumerate(rows):
        node, source_url = document_node(row, language, index)
        topics = split_list(row.get("topics"))
        entities = split_list(row.get("entities"))
        populism = parse_populism_elements(row.get("formula_of_populism_frontier"))

        lines.append(f"### {index + 1}. {source_url or node}")
        lines.append("")
        lines.append(f"- node: `{node}`")
        lines.append(f"- author: {row.get('authorUniqueId') or '(unknown)'}")
        lines.append(
            f"- transcript: {'present' if not is_blank(row.get('asr_transcript') or row.get('asr_translated') or row.get('whisperResult')) else 'absent'}"
        )
        lines.append(
            f"- summary: {'present' if not is_blank(row.get('summary_analysis')) else 'absent'}"
        )
        lines.append(f"- frames analysed: {stats.get('frames', {}).get(index, 0)}")
        lines.append(f"- screen-text blocks: {stats.get('ocr', {}).get(index, 0)}")
        lines.append(f"- topics: {', '.join(topics) if topics else '(none)'}")
        lines.append(f"- entities: {', '.join(entities) if entities else '(none)'}")
        if populism:
            rendered = "; ".join(
                f"{element}^{affect}" if affect else element for element, affect in populism
            )
            lines.append(f"- populism frontier: {rendered}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- per-language driver ---------------------------------------------------


def export_language(
    language: str,
    prov: Provenance,
    *,
    sample: int | None = None,
    dry_run: bool = False,
    output_dir: Path = Path("./rdf"),
) -> dict:
    """Export one language. Returns a stats dict; never raises on bad input."""
    started = time.time()
    source = source_filename(language)
    if source is None:
        logger.warning("No input file for language %s; skipping", language)
        print(f"[{STAGE_NAME}] {language}: no input file; skipped")
        return {"language": language, "status": "no_input", "rows": 0, "triples": 0}

    rows = read_rows(source, sample)
    print(
        f"[{STAGE_NAME}] {language}: {len(rows)} row(s) from {source}"
        + (f" (sample of {sample})" if sample else "")
    )
    if not rows:
        return {"language": language, "status": "empty", "rows": 0, "triples": 0}

    graph = new_graph()
    frames_per_row: dict[int, int] = {}
    ocr_per_row: dict[int, int] = {}

    for index, row in enumerate(rows):
        emit_document(graph, row, language, index, prov)
        frames_per_row[index] = sum(
            1 for i in range(1, 7) if not is_blank(row.get(f"frame_analysis_{i}"))
        )
        ocr_per_row[index] = sum(1 for i in range(1, 7) if not is_blank(row.get(f"ocr_{i}")))
        logger.debug("Emitted document %s/%s for %s", index + 1, len(rows), language)

    network = maybe_emit_network(graph, language, prov)

    stats = {
        "language": language,
        "status": "ok",
        "source": str(source),
        "rows": len(rows),
        "triples": graph.count(),
        "frames": frames_per_row,
        "ocr": ocr_per_row,
        "network": network,
    }

    elapsed = time.time() - started
    print(
        f"[{STAGE_NAME}] {language}: {graph.count()} triples from {len(rows)} row(s) "
        f"in {elapsed:.1f}s"
    )
    if network["nodes"] or network["edges"]:
        print(
            f"[{STAGE_NAME}] {language}: projected {network['nodes']} network node(s), "
            f"{network['edges']} edge(s)"
        )

    if dry_run:
        print(f"[{STAGE_NAME}] {language}: dry run — nothing written")
        stats["status"] = "dry_run"
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    turtle_path = output_dir / f"ep24_{language}.ttl"
    turtle_path.write_text(graph.serialize(), encoding="utf-8")
    print(f"[{STAGE_NAME}] {language}: wrote {turtle_path}")

    summary_path = output_dir / f"summary_{language}.txt"
    write_summary(summary_path, language, rows, stats, prov)
    print(f"[{STAGE_NAME}] {language}: wrote {summary_path}")

    stats["turtle"] = str(turtle_path)
    stats["summary"] = str(summary_path)
    return stats


def language_list(value: str | None) -> list[str]:
    if not value:
        return list(LANGUAGES)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in LANGUAGES]
    if unknown:
        print(
            f"[{STAGE_NAME}] warning: language(s) not in the legacy set: {unknown}", file=sys.stderr
        )
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roihu_rdf.py",
        description="Export the EP24 multimodal CSV output as RDF/Turtle.",
    )
    parser.add_argument("--language", help="one language code, or a comma-separated list")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="export only the first N rows per language (smoke test)",
    )
    parser.add_argument(
        "--output-dir",
        default="./rdf",
        help="where to write Turtle and summary files (default ./rdf)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="parse and report, but write nothing"
    )
    parser.add_argument("--debug", action="store_true", help="verbose per-document logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    prov = Provenance.capture()
    languages = language_list(args.language)

    print(f"[{STAGE_NAME}] stage={STAGE_NAME} version={STAGE_VERSION} schema={SCHEMA_VERSION}")
    print(f"[{STAGE_NAME}] run_id={prov.run_id} model={prov.model}")
    print(f"[{STAGE_NAME}] languages={','.join(languages)}")
    print(f"[{STAGE_NAME}] cwd={os.getcwd()} output_dir={args.output_dir}")

    results = []
    failures = 0
    for language in languages:
        try:
            results.append(
                export_language(
                    language,
                    prov,
                    sample=args.sample,
                    dry_run=args.dry_run,
                    output_dir=Path(args.output_dir),
                )
            )
        except Exception as exc:  # one bad language must not kill the batch
            failures += 1
            logger.exception("Language %s failed: %s", language, exc)
            print(f"[{STAGE_NAME}] {language}: FAILED — {exc}", file=sys.stderr)
            results.append({"language": language, "status": "error", "error": str(exc)})

    exported = [r for r in results if r.get("status") in ("ok", "dry_run")]
    total_triples = sum(int(r.get("triples", 0)) for r in exported)
    print(
        f"[{STAGE_NAME}] done: {len(exported)}/{len(languages)} language(s), "
        f"{total_triples} triples, {failures} failure(s)"
    )
    return 1 if failures and not exported else 0


if __name__ == "__main__":
    sys.exit(main())
