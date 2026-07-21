"""Document metadata schema — Workstream 0 (implements docs/plan.md §5.2).

One overloaded "provenance" string can't reproduce or legally audit a corpus
build, so every document carries structured, separately-typed metadata. This is
the lightweight Pydantic schema §5.2 calls for (not a database): the corpus
pipeline validates *every* record against `Document` at ingest.

§5.2 field groups, kept as distinct fields here:
  * origin  — how the text was produced
  * genre   — what kind of text it is
  * source/acquisition — where it came from + the reproducibility anchors
                         (url, download timestamp, stable doc id, content hashes)
  * rights  — license id/url + the per-use rights matrix (from rights.yaml)
  * split   — assignment from splits.yaml

Two enum values are ADDED to §5.2's written lists to represent the data we
actually have without making a false provenance claim (both flagged in the
corpus report):
  * Origin.web_crawl  — crawled web text of unknown authorship (MADLAD). Forcing
                        it into `authored_kreyol` or `machine_translation` would
                        assert provenance we cannot verify.
  * Genre.web         — crawl / mixed-domain web text (MADLAD). §5.2's 8 genres
                        are all authored-content genres; crawl needs its own bucket.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Origin(str, enum.Enum):
    """How the text was produced (§5.2 origin list + `web_crawl`)."""

    authored_kreyol = "authored_kreyol"
    human_translation = "human_translation"
    machine_translation = "machine_translation"
    machine_translation_reviewed = "machine_translation_reviewed"
    oral_transcription = "oral_transcription"
    synthetic_reviewed = "synthetic_reviewed"
    synthetic_unreviewed = "synthetic_unreviewed"
    web_crawl = "web_crawl"  # ADDED (see module docstring)


class Genre(str, enum.Enum):
    """What kind of text it is (§5.2 genre list + `web`)."""

    encyclopedic = "encyclopedic"
    conversational = "conversational"
    educational = "educational"
    religious = "religious"
    news = "news"
    dictionary = "dictionary"
    historical = "historical"
    proverb = "proverb"
    web = "web"  # ADDED (see module docstring)


class Split(str, enum.Enum):
    """Assignment from splits.yaml."""

    train = "train"
    tokenizer_eval = "tokenizer_eval"
    model_selection_dev = "model_selection_dev"
    final_devtest = "final_devtest"
    exhibit_examples = "exhibit_examples"


class RightsVerdict(str, enum.Enum):
    """Per-use verdict, mirroring rights.yaml."""

    allowed = "allowed"
    denied = "denied"
    unresolved = "unresolved"
    eval_only = "eval_only"


class Rights(BaseModel):
    """The per-use rights matrix copied from the source's rights.yaml entry.

    Carrying it on every document makes each record self-auditing: a training
    job can refuse any doc whose `model_training` is not `allowed`.
    """

    model_config = ConfigDict(extra="forbid")

    license_id: str
    license_url: str | None = None
    analysis: RightsVerdict
    tokenizer_training: RightsVerdict
    model_training: RightsVerdict
    redistribution: RightsVerdict


class Acquisition(BaseModel):
    """Where the document came from + the reproducibility anchors (§5.2)."""

    model_config = ConfigDict(extra="forbid")

    source: str                     # rights.yaml key, e.g. "madlad_400_ht_clean"
    source_name: str                # human-readable
    url: str | None = None          # per-doc URL if the source provides one
    revision: str | None = None     # HF sha / dump date — the immutable pin
    download_timestamp: str         # ISO-8601, when the source file was acquired
    doc_id: str                     # stable, deterministic across re-runs
    raw_content_hash: str           # sha256 of the text as ingested
    cleaned_content_hash: str | None = None  # sha256 of the normalized text


class Document(BaseModel):
    """One corpus document + its full §5.2 metadata.

    `extra="allow"`: downstream stages attach their own fields (normalization
    stats, `wiki_bot_stub`, filter flags, dedup `cluster_id`, …) without
    breaking validation of the core metadata, which is what we guarantee.
    """

    model_config = ConfigDict(extra="allow")

    text: str
    origin: Origin
    genre: Genre
    acquisition: Acquisition
    rights: Rights
    split: Split

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document text is empty")
        return v


def validate_record(record: dict) -> Document:
    """Validate one raw dict against the schema (raises on failure)."""
    return Document.model_validate(record)
