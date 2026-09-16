"""The shapes the MCP tools return, without the protocol that carries them.

Extracted from :mod:`model_doctor.app.mcp_server`, which now calls these and
adds nothing but argument validation and transport. The split exists because a
second surface needs the same answers: Ramanujan vendors the engine as a library
and serves its own MCP endpoint, and two copies of a comparison would drift
apart within a release. There is one definition of what a run looks like and one
of what a comparison contains, here.

Two rules make that possible, and both are load-bearing:

* **Nothing here imports ``mcp``.** No ``ToolError``, no ``Field``, no
  ``ToolAnnotations``. A caller that cannot have that dependency — Ramanujan
  forbids it in ``requirements.txt``, and a test enforces the ban — still gets
  the payloads. Refusals are raised as :class:`PayloadError` and each caller
  maps it into its own vocabulary.
* **Nothing here opens a database.** Every function takes a connection. The
  engine's own server opens one read-only; Ramanujan opens it through the
  service layer it already uses for the panel. Neither has to know how the
  other does it, and no second connection appears in a process that already
  has one.

What this module does *not* decide is which runs a caller may see. It answers
about the runs it is given. The engine's server passes every run in the
database, because its reader is the operator. Ramanujan passes only runs it can
resolve back to a model record it created, because its readers are many. That
boundary belongs to the caller and is deliberately not expressible here.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from model_doctor import config
from model_doctor.app import comparison, storage
from model_doctor.app.clustering import DISCRIMINATING_METHOD, FACTOR_SIGNATURE_METHOD
from model_doctor.app.diagnosis import Outcome
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Which optional table carries which kind of evidence. Checked by name before
#: reading, because a database written by an older schema simply lacks some.
EVIDENCE_TABLES: dict[str, str] = {
    "run_evaluations": "evaluation",
    "factor_rates": "factor_rates",
    "clusters": "groups",
    "recommendations": "recommendations",
    "mask_findings": "mask_findings",
    "run_benchmarks": "benchmarks",
    "heatmaps": "heatmaps",
    "image_diagnoses": "image_diagnoses",
    "finding_relations": "relations",
}


class PayloadError(Exception):
    """A refusal a caller can show its own reader.

    Deliberately not ``ToolError``: that lives in the ``mcp`` package, and one
    of the two callers cannot depend on it. The message is written to be read
    by whoever asked, so both callers can pass it through unaltered.
    """


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def resolve_run_ids(run_ids: Sequence[int]) -> list[int]:
    """Validate a caller's run ids before anything is read.

    Duplicates are dropped rather than rejected, because ``[4, 4, 5]`` is a
    harmless slip; an empty list, a non-integer, or more than the configured
    maximum is refused with the reason, because each would produce a response
    that is either empty or unreadable.
    """
    if not run_ids:
        raise PayloadError("run_ids must name at least one run.")
    cleaned: list[int] = []
    for value in run_ids:
        # `bool` is a subclass of `int`, and `True` as a run id is a mistake
        # worth naming rather than silently reading run 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise PayloadError(f"run_ids must be integers; got {value!r}.")
        if value not in cleaned:
            cleaned.append(value)
    if len(cleaned) > config.MCP_MAX_RUNS:
        raise PayloadError(
            f"get_analysis compares at most {config.MCP_MAX_RUNS} runs at once; "
            f"{len(cleaned)} were given. Split them across calls."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Pieces of a run
# ---------------------------------------------------------------------------

def evidence_present(connection: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Say which optional evidence a run carries, checking tables first."""
    present: dict[str, Any] = {}
    for table, kind in EVIDENCE_TABLES.items():
        if not storage.has_table(connection, table):
            present[kind] = False
            continue
        if kind == "evaluation":
            rows = storage.load_evaluations(connection, run_id)
            present[kind] = [r.task for r in rows] or False
        elif kind == "factor_rates":
            present[kind] = bool(storage.load_factor_rates(connection, run_id))
        elif kind == "groups":
            present[kind] = bool(
                storage.load_clusters(connection, run_id, DISCRIMINATING_METHOD)
            )
        elif kind == "recommendations":
            present[kind] = len(storage.load_recommendations(connection, run_id))
        elif kind == "mask_findings":
            present[kind] = bool(storage.load_mask_findings(connection, run_id))
        elif kind == "benchmarks":
            present[kind] = sorted(
                {b.device for b in storage.load_benchmarks(connection, run_id)}
            )
        elif kind == "heatmaps":
            present[kind] = len(storage.load_heatmaps(connection, run_id))
        elif kind == "image_diagnoses":
            present[kind] = bool(storage.load_image_diagnoses(connection, run_id))
        elif kind == "relations":
            present[kind] = bool(storage.load_finding_relations(connection, run_id))
    return present


def run_row(
    connection: sqlite3.Connection,
    run: storage.RunRecord,
    *,
    include_paths: bool = False,
) -> dict[str, Any]:
    """One run as a listing reports it.

    ``include_paths`` defaults to off and Ramanujan never turns it on: its
    readers are not the operator of this machine, and a checkpoint path
    describes the machine rather than the run.
    """
    counts = storage.outcome_counts(connection, run.id)
    findings = storage.load_findings(connection, run.id)
    classes = sorted({f.class_name for f in findings if f.class_name})
    job = storage.load_job_for_run(connection, run.id)
    evaluations = (
        storage.load_evaluations(connection, run.id)
        if storage.has_table(connection, "run_evaluations")
        else []
    )
    bbox = next((e for e in evaluations if e.task == "bbox"), None)

    row = comparison.run_identity(run, include_paths=include_paths)
    # Family comes from the job that produced the run, and from nowhere else:
    # a CLI run has no job, and detecting the family means loading the
    # checkpoint, which a read-only surface must never do.
    row["model"]["family"] = job.detector if job else None
    row["dataset"]["classes"] = classes
    row["dataset"]["classes_source"] = "findings"
    row["counts"] = {
        "images": len(storage.load_images(connection, run.id)),
        **{
            k: v
            for k, v in comparison.outcome_summary(counts).items()
            if k != "derived"
        },
    }

    def headline(key: str) -> float | None:
        value = comparison.measured(bbox.metrics.get(key)) if bbox else None
        return None if value is None else round(value, 4)

    row["headline"] = {
        "map50_box": headline("map50"),
        "map50_95_box": headline("map50_95"),
    }
    row["evidence"] = evidence_present(connection, run.id)
    # Replication only means something within one checkpoint: a run of
    # different weights that disagrees is a different observation, not a failed
    # replication.
    row["same_model_runs"] = storage.load_runs_for_model(connection, run.model_sha256)
    return row


def groups_for(
    connection: sqlite3.Connection, run_id: int, method: str
) -> list[comparison.GroupRow]:
    """Assemble each group with its members' outcome and class mix."""
    if not storage.has_table(connection, "clusters"):
        return []
    rows: list[comparison.GroupRow] = []
    for cluster in storage.load_clusters(connection, run_id, method):
        members = storage.load_cluster_members(connection, cluster.id)
        outcomes = Counter(m["outcome"] for m in members)
        classes = Counter(m["class_name"] for m in members if m["class_name"])
        rows.append(
            comparison.GroupRow(
                cluster_id=cluster.id,
                label=cluster.label,
                size=cluster.size,
                outcomes=dict(outcomes),
                classes=dict(classes),
            )
        )
    return rows


def mask_summary(
    connection: sqlite3.Connection, run_id: int
) -> dict[str, int] | None:
    """Outline-level outcomes, with unmeasured pairs named as such."""
    if not storage.has_table(connection, "mask_findings"):
        return None
    rows = storage.load_mask_findings(connection, run_id)
    if not rows:
        return None
    tally: Counter[str] = Counter()
    for row in rows:
        tally[row.mask_outcome or "not_measured"] += 1
    return dict(tally)


# ---------------------------------------------------------------------------
# The two payloads
# ---------------------------------------------------------------------------

def list_runs_payload(
    connection: sqlite3.Connection,
    *,
    runs: Sequence[storage.RunRecord] | None = None,
    include_paths: bool = False,
) -> dict[str, Any]:
    """The listing, for whichever runs the caller decides to show.

    ``runs`` is the caller's boundary, not this module's. Left ``None`` it
    lists every run in the database, which is what the engine's own server
    wants because its reader is the operator. Ramanujan passes the subset it
    can resolve back to a model record it created, because its readers are
    many and the database may also hold runs made from a shell on the host.
    """
    rows = storage.list_runs(connection) if runs is None else list(runs)
    return {
        "schema_version": storage.SCHEMA_VERSION,
        "runs": [run_row(connection, run, include_paths=include_paths) for run in rows],
    }


def get_analysis_payload(
    connection: sqlite3.Connection,
    run_ids: Sequence[int],
    *,
    baseline: int | None = None,
    include_paths: bool = False,
    include_descriptive_groups: bool = False,
    listing_tool: str = "list_runs",
) -> dict[str, Any]:
    """The stored evidence for one or more runs, in one comparable structure.

    Six sections, unchanged from the design this was extracted from: ``runs``
    is identity and configuration; ``comparability`` says whether these numbers
    may sit side by side and names every configuration difference;
    ``per_run`` carries outcomes with their derived metrics, evaluation,
    factors with their control rates, groups, recommendations and the image and
    relation lenses; ``cross_run`` gives deltas against a baseline and what
    replicates; ``evidence_gaps`` names what each run lacks and the command
    that fills it; and ``caveats`` is fixed text present in every response,
    because a reader who takes outcome counts for mAP will reach a wrong
    conclusion from correct numbers.
    """
    ids = resolve_run_ids(run_ids)
    base = ids[0] if baseline is None else baseline
    if base not in ids:
        raise PayloadError(f"baseline {base} is not among run_ids {ids}.")

    runs: dict[int, storage.RunRecord] = {}
    missing: list[int] = []
    for rid in ids:
        run = storage.load_run(connection, rid)
        if run is None:
            missing.append(rid)
        else:
            runs[rid] = run
    if missing:
        # The listing tool is named by the caller: this module serves two
        # surfaces and they do not expose the same tool names, so telling a
        # reader to call one that does not exist would be worse than silence.
        raise PayloadError(
            f"No such run: {', '.join(str(m) for m in missing)}. "
            f"Call {listing_tool} to see what exists."
        )

    outcomes = {rid: storage.outcome_counts(connection, rid) for rid in ids}
    evaluations = {
        rid: (
            storage.load_evaluations(connection, rid)
            if storage.has_table(connection, "run_evaluations")
            else []
        )
        for rid in ids
    }
    benchmarks = {
        rid: (
            storage.load_benchmarks(connection, rid)
            if storage.has_table(connection, "run_benchmarks")
            else []
        )
        for rid in ids
    }
    factors = {
        rid: (
            storage.load_factor_rates(connection, rid)
            if storage.has_table(connection, "factor_rates")
            else []
        )
        for rid in ids
    }
    groups = {rid: groups_for(connection, rid, DISCRIMINATING_METHOD) for rid in ids}
    recommendations = {
        rid: (
            storage.load_recommendations(connection, rid)
            if storage.has_table(connection, "recommendations")
            else []
        )
        for rid in ids
    }

    per_run: dict[int, dict[str, Any]] = {}
    present: dict[int, dict[str, Any]] = {}
    for rid in ids:
        present[rid] = evidence_present(connection, rid)
        block: dict[str, Any] = {
            "outcomes": comparison.outcome_summary(outcomes[rid]),
            "evaluation": comparison.evaluation_summary(evaluations[rid]),
            "mask_summary": mask_summary(connection, rid),
            "factors": comparison.factor_summary(factors[rid]),
            "groups": comparison.group_summary(groups[rid]),
            "recommendations": comparison.recommendation_summary(recommendations[rid]),
            "benchmarks": comparison.benchmark_summary(benchmarks[rid]),
            # A second lens over the same findings: how many whole photographs
            # were handled correctly, and what shape the mistakes took. None
            # when the pass has not been run.
            "images": comparison.image_summary(
                storage.load_image_diagnoses(connection, rid)
                if storage.has_table(connection, "image_diagnoses")
                else []
            ),
            # A third lens: not which object failed, nor what shape the image's
            # mistake took, but how two findings relate — which prediction
            # covers which object. None when unmeasured.
            "relations": comparison.relation_summary(
                storage.load_finding_relations(connection, rid)
                if storage.has_table(connection, "finding_relations")
                else []
            ),
        }
        if include_descriptive_groups:
            block["descriptive_groups"] = comparison.group_summary(
                groups_for(connection, rid, FACTOR_SIGNATURE_METHOD)
            )
        per_run[rid] = block

    return {
        "schema_version": storage.SCHEMA_VERSION,
        "runs": [
            run_row(connection, runs[rid], include_paths=include_paths) for rid in ids
        ],
        "comparability": comparison.comparability(
            [runs[rid] for rid in ids], evaluations, benchmarks
        ),
        "per_run": per_run,
        # Fingerprints are passed so replication is judged only between
        # differently configured runs. Without them a group present in three
        # re-executions of one command reads as replicated three times over.
        "cross_run": comparison.cross_run(
            base,
            outcomes,
            evaluations,
            factors,
            groups,
            recommendations,
            fingerprints={rid: comparison.run_fingerprint(runs[rid]) for rid in ids},
        ),
        "evidence_gaps": comparison.evidence_gaps(ids, present),
        "caveats": list(comparison.CAVEATS),
    }


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

#: How many findings one page returns. A run holds tens of thousands, and a
#: reader choosing which failures to look at does not need them all at once.
DEFAULT_FINDING_LIMIT = 50
MAX_FINDING_LIMIT = 200

#: The outcomes a caller may filter on, as stored.
OUTCOMES: tuple[str, ...] = tuple(o.value for o in Outcome)

#: What a reader must not conclude from a single finding. The run-level caveats
#: answer a different question, so these are additional rather than a subset.
FINDING_CAVEATS: tuple[str, ...] = (
    (
        "An outcome is the matcher's verdict at this run's IoU and confidence "
        "thresholds, not a statement about the model in general. The same "
        "prediction can be correct in one run and a failure in another."
    ),
    (
        "Factors are attributed conditions, not causes. A finding carrying "
        "`small_object` was small; whether that is why it failed is answered "
        "by the factor's rate against its control rate in get_analysis, never "
        "by one finding."
    ),
    (
        "One finding is an anecdote. Counts and rates come from get_analysis; "
        "reading a handful of findings and generalising is the mistake this "
        "surface is easiest to make."
    ),
)


def resolve_finding_page(
    limit: Any, offset: Any, outcome: Any
) -> tuple[int, int, str | None]:
    """Validate paging and filter arguments before anything is read.

    Refused with the reason rather than silently corrected: a caller who asked
    for outcome ``"fasle_positive"`` and received every finding would draw
    conclusions from a population they did not ask for.
    """
    if limit is None:
        limit = DEFAULT_FINDING_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise PayloadError("`limit` must be an integer.")
    if limit < 1:
        raise PayloadError("`limit` must be at least 1.")
    limit = min(limit, MAX_FINDING_LIMIT)

    if offset is None:
        offset = 0
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise PayloadError("`offset` must be an integer.")
    if offset < 0:
        raise PayloadError("`offset` must not be negative.")

    if outcome is not None and (
        not isinstance(outcome, str) or outcome not in OUTCOMES
    ):
        raise PayloadError(f"`outcome` must be one of: {', '.join(OUTCOMES)}.")
    return limit, offset, outcome


def _finding_row(record: storage.FindingRecord, filename: str | None) -> dict[str, Any]:
    """One finding as a listing reports it.

    Geometry is included because a box is what the finding *is*; the image is
    named by filename alone. :class:`~model_doctor.app.storage.ImageRow` also
    carries an absolute path on the host that ran the analysis, which no reader
    of this surface has any use for.
    """
    return {
        "finding_id": record.id,
        "run_id": record.run_id,
        "image_id": record.image_id,
        "image_filename": filename,
        "outcome": record.outcome,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "confidence": record.confidence,
        "iou": record.iou,
        "pred_box": list(record.pred_box) if record.pred_box else None,
        "truth_box": list(record.truth_box) if record.truth_box else None,
    }


def list_findings_payload(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    outcome: str | None = None,
    image_id: int | None = None,
    limit: int = DEFAULT_FINDING_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """A page of one run's findings, with the totals needed to page it.

    The caller has already decided this run may be read; this function does not
    re-decide it, for the same reason the listing does not — see the module
    docstring. ``run_id`` is taken as given.

    ``outcome_counts`` is the whole run regardless of the filter, because a
    reader looking at 50 false positives needs to know whether the run had 60
    or 6000, and that number does not change with the page.
    """
    if not isinstance(run_id, int) or isinstance(run_id, bool):
        raise PayloadError(f"run_id must be an integer; got {run_id!r}.")
    if storage.load_run(connection, run_id) is None:
        raise PayloadError(f"No such run: {run_id}.")

    resolved = Outcome(outcome) if outcome else None
    total = storage.count_findings(
        connection, run_id, outcome=resolved, image_id=image_id
    )
    records = storage.load_findings_page(
        connection, run_id, limit, offset, outcome=resolved, image_id=image_id
    )
    names = storage.load_image_names(
        connection, run_id, [r.image_id for r in records]
    )
    return {
        "schema_version": storage.SCHEMA_VERSION,
        "run_id": run_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {"outcome": outcome, "image_id": image_id},
        "outcome_counts": storage.outcome_counts(connection, run_id),
        "findings": [_finding_row(r, names.get(r.image_id)) for r in records],
        "caveats": list(FINDING_CAVEATS),
    }


def get_finding_payload(
    connection: sqlite3.Connection,
    run_id: int,
    finding_id: int,
    *,
    listing_tool: str = "list_findings",
) -> dict[str, Any]:
    """One finding with the evidence attached to it, and nothing beyond it.

    Both ids are required. See :func:`~model_doctor.app.storage.load_finding`
    for why: a finding id alone is a key into every run at once.

    The evidence here is what the stored passes attributed to *this* finding —
    its factors, its outline result, its measured relationships, the group it
    was placed in. Whether a heatmap exists is reported; where it lives is not.
    """
    for name, value in (("run_id", run_id), ("finding_id", finding_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PayloadError(f"{name} must be an integer; got {value!r}.")

    record = storage.load_finding(connection, run_id, finding_id)
    if record is None:
        # One message for "no such run", "no such finding" and "that finding is
        # in another run". Distinguishing them would let a caller map the id
        # space by reading the differences.
        raise PayloadError(
            f"No such finding: {finding_id} in run {run_id}. "
            f"Call {listing_tool} to see what exists."
        )

    names = storage.load_image_names(connection, run_id, [record.image_id])
    payload = _finding_row(record, names.get(record.image_id))
    payload["truth_polygon"] = record.truth_polygon

    factors = storage.load_factors_by_finding(connection, run_id)
    payload["factors"] = factors.get(finding_id, [])

    payload["mask"] = None
    if storage.has_table(connection, "mask_findings"):
        for row in storage.load_mask_findings(connection, run_id):
            if row.finding_id == finding_id:
                payload["mask"] = {
                    "box_iou": row.box_iou,
                    "mask_iou": row.mask_iou,
                    "mask_outcome": row.mask_outcome,
                    "pred_polygon": row.pred_polygon,
                }
                break

    relations: list[dict[str, Any]] = []
    if storage.has_table(connection, "finding_relations"):
        for row in storage.load_finding_relations(connection, run_id):
            if row.finding_id != finding_id:
                continue
            relations.append(
                {
                    "relation": row.relation,
                    "direction": row.direction,
                    "partner_finding_id": row.partner_finding_id,
                    "value": row.value,
                    "best_coverage": row.best_coverage,
                    "span": row.span,
                    "threshold": row.threshold,
                    "coverage_floor": row.coverage_floor,
                    "qualifies": row.qualifies,
                    "method": row.method,
                }
            )
    payload["relations"] = relations

    payload["groups"] = _groups_containing(connection, run_id, finding_id)

    # Presence only. `HeatmapRow.path` is a location on the host that ran the
    # analysis, and this surface serves no images.
    payload["heatmap"] = None
    if storage.has_table(connection, "heatmaps"):
        for row in storage.load_heatmaps(connection, run_id):
            if row.finding_id == finding_id:
                payload["heatmap"] = {
                    "available": True,
                    "method": row.method,
                    "target_layers": row.target_layers,
                }
                break

    return {
        "schema_version": storage.SCHEMA_VERSION,
        "finding": payload,
        "caveats": list(FINDING_CAVEATS),
    }


def _groups_containing(
    connection: sqlite3.Connection, run_id: int, finding_id: int
) -> list[dict[str, Any]]:
    """Which failure groups this finding was placed in.

    Both partitions are reported, each named, because a finding can sit in one
    and not the other and a reader comparing two findings needs to know which
    grouping they are being compared under.
    """
    if not storage.has_table(connection, "clusters") or not storage.has_table(
        connection, "cluster_members"
    ):
        return []
    found: list[dict[str, Any]] = []
    for method in (DISCRIMINATING_METHOD, FACTOR_SIGNATURE_METHOD):
        for cluster in storage.load_clusters(connection, run_id, method):
            members = connection.execute(
                "SELECT 1 FROM cluster_members WHERE cluster_id = ? "
                "AND finding_id = ? LIMIT 1",
                (cluster.id, finding_id),
            ).fetchone()
            if members:
                found.append(
                    {
                        "cluster_id": cluster.id,
                        "method": method,
                        "label": cluster.label,
                        "size": cluster.size,
                    }
                )
    return found


# ---------------------------------------------------------------------------
# Visual evidence
# ---------------------------------------------------------------------------
#
# Everything above returns numbers and text. These two return bytes, and that
# difference is worth stating plainly rather than leaving to the reader.
#
# A caller asks by id. The id is looked up *within a run*, the stored path is
# read from the row, and that path is handed to the engine's own
# `verified_file`, which remaps it, resolves symlinks and refuses anything
# outside the configured roots. No caller-supplied string ever reaches the
# filesystem: there is no argument here that a path, a URL or a bucket key
# could be smuggled through, because the only inputs are integers.
#
# The bytes are returned to the caller. What the caller then does with them is
# the caller's boundary to document — see the surface's own notes — and is not
# a decision this module can make on its behalf.

#: Largest file this surface will hand back, chosen from the limit downstream
#: rather than picked round.
#:
#: Base64 expands bytes by 4/3: ``len(b64encode(n)) == 4 * ceil(n / 3)``. The
#: binding constraint is the receiving model's per-image limit of 5 MB on the
#: *encoded* payload, so the largest raw file that can survive encoding is
#: ``5_000_000 * 3 / 4 = 3_750_000`` bytes. 3.5 MiB sits just under that:
#:
#:     3.5 MiB = 3,670,016 raw  ->  4,893,356 base64  =  4.89 MB
#:     3.75 MiB                 ->  5,242,880         =  5.24 MB   (over)
#:     4 MiB                    ->  5,592,408         =  5.59 MB   (over)
#:
#: The ~107 KB of headroom under 5 MB carries the JSON envelope around the
#: image block. For comparison, this application's own image path for LLM calls
#: (``services.imageCompression.resize_and_compress_safe``) caps raw bytes at
#: 4.5 MB, which is *above* the encoded limit once base64 is applied; this
#: surface does not copy that number.
#:
#: Refused with the size rather than truncated, because half an image is not
#: evidence.
MAX_IMAGE_BYTES = 3584 * 1024  # 3.5 MiB

#: Suffix to media type. A suffix not listed here is refused rather than
#: guessed: the media type is what tells a client how to decode the bytes, and
#: a wrong guess is worse than a refusal.
IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _read_verified(stored_path: str, *, what: str) -> tuple[bytes, str]:
    """Read a stored path through the engine's root check, or refuse safely.

    Every refusal here is deliberately vague. ``verified_file`` distinguishes
    "outside the permitted roots" from "no longer on disk", and its message for
    the second names the resolved location — useful in a server log, and
    exactly what must not travel to a reader. So the distinction is kept in the
    log and flattened here.
    """
    # Imported at call time: the engine's service layer pulls in the job runner
    # and the workspace helpers, and a caller that only ever asks for numbers
    # should not carry them.
    from model_doctor.app.service import ServiceError, verified_file

    try:
        resolved = verified_file(stored_path)
    except ServiceError as error:
        logger.warning(
            "mcp_payloads: refused %s (status %s)", what, getattr(error, "status", "?")
        )
        raise PayloadError(
            f"The {what} is not available. It may have been moved or removed "
            "since the analysis ran."
        ) from None

    size = resolved.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise PayloadError(
            f"That {what} is {size // 1024} KB, above the "
            f"{MAX_IMAGE_BYTES // 1024} KB this surface returns. "
            "It cannot be sent in one response."
        )
    media = IMAGE_MEDIA_TYPES.get(resolved.suffix.lower())
    if media is None:
        raise PayloadError(f"The {what} is in a format this surface cannot return.")
    return resolved.read_bytes(), media


def image_for_run(
    connection: sqlite3.Connection, run_id: int, image_id: int
) -> dict[str, Any]:
    """The photograph one of a run's findings was measured on.

    Scoped by run, so an image belonging to another run is absent rather than
    refused differently — the same property the finding lookup has, for the
    same reason.

    Returns the bytes, their media type, and the identity a reader needs to
    say which photograph this is. No path, in the result or in any refusal.
    """
    for name, value in (("run_id", run_id), ("image_id", image_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PayloadError(f"{name} must be an integer; got {value!r}.")

    row = connection.execute(
        "SELECT id, path, filename, width, height FROM images "
        "WHERE id = ? AND run_id = ?",
        (image_id, run_id),
    ).fetchone()
    if row is None:
        raise PayloadError(
            f"No such image: {image_id} in run {run_id}. "
            "Call list_findings to see which images a run's findings are on."
        )

    data, media = _read_verified(str(row["path"]), what="image")
    return {
        "image_id": row["id"],
        "run_id": run_id,
        "filename": row["filename"],
        "width": row["width"],
        "height": row["height"],
        "media_type": media,
        "bytes": len(data),
        "data": data,
    }


def heatmap_for_finding(
    connection: sqlite3.Connection,
    run_id: int,
    finding_id: int,
    *,
    method: str = "grad-cam",
    prefer_preview: bool = True,
) -> dict[str, Any]:
    """The overlay explaining one of a run's findings, if one was generated.

    Joined through ``findings`` so the run scopes the lookup, exactly as
    :func:`image_for_run` is scoped.

    A run can be complete and carry no heatmap at all — explanation is
    implemented for some detector families and not others — so absence is
    reported as an ordinary answer with a reason, never as a fault.

    ``prefer_preview`` returns the downscaled companion when one exists. It is
    on by default: the companion is a few hundred kilobytes where the original
    can be several megabytes, and a response carrying the original may not
    survive the journey to a reader at all.
    """
    for name, value in (("run_id", run_id), ("finding_id", finding_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PayloadError(f"{name} must be an integer; got {value!r}.")
    if not isinstance(method, str) or not method:
        raise PayloadError("`method` must be a non-empty string.")

    if not storage.has_table(connection, "heatmaps"):
        raise PayloadError(
            f"No {method} heatmap for finding {finding_id} in run {run_id}. "
            "Call get_finding to see whether one exists."
        )
    row = connection.execute(
        "SELECT h.path AS path, h.method AS method, h.target_layers AS target_layers "
        "FROM heatmaps h JOIN findings f ON f.id = h.finding_id "
        "WHERE h.finding_id = ? AND f.run_id = ? AND h.method = ?",
        (finding_id, run_id, method),
    ).fetchone()
    if row is None:
        # One message whether the finding is in another run, the run is
        # unknown, or the finding simply was never explained.
        raise PayloadError(
            f"No {method} heatmap for finding {finding_id} in run {run_id}. "
            "Call get_finding to see whether one exists."
        )

    stored = str(row["path"])
    served = "original"
    if prefer_preview:
        from model_doctor.app.explainability import preview_path

        companion = preview_path(Path(config.remap_path(stored)))
        if companion.is_file():
            stored, served = str(companion), "preview"
        else:
            # A run explained before previews existed has no companion. Falling
            # back to the original is right, but the original is the thing the
            # preview exists to avoid sending, so the caller is told what
            # happened rather than receiving a refusal about a size it did not
            # ask for.
            served = "original (no downscaled companion exists)"

    data, media = _read_verified(stored, what="heatmap")
    return {
        "finding_id": finding_id,
        "run_id": run_id,
        "method": row["method"],
        "target_layers": row["target_layers"],
        "resolution": served,
        "media_type": media,
        "bytes": len(data),
        "data": data,
    }
