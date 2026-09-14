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
from typing import Any

from model_doctor import config
from model_doctor.app import comparison, storage
from model_doctor.app.clustering import DISCRIMINATING_METHOD, FACTOR_SIGNATURE_METHOD

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
