"""The payload layer, which two MCP surfaces now share.

`tests/test_mcp_server.py` drives these same shapes through the engine's own
stdio server and is the contract for what the payloads *mean*. These tests pin
the properties that matter once a second caller exists: that the module needs
neither the `mcp` package nor a database of its own, that a caller chooses which
runs are visible, and that the six sections and the control-rate rule survive the
extraction unchanged.

The seeding helpers are the ones the server tests use, imported rather than
copied, so the two suites cannot describe different databases.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from model_doctor import config
from model_doctor.app import mcp_payloads, storage
from model_doctor.app.mcp_payloads import PayloadError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mcp_server import (  # noqa: E402
    OTHER_MODEL,
    SAME_MODEL,
    _seed_evidence,
    _seed_run,
)


@dataclass(frozen=True)
class _Seeded:
    """An open connection and the ids the fixture put in it.

    `sqlite3.Connection` refuses attribute assignment, so the two travel
    together rather than the ids riding on the connection.
    """

    conn: sqlite3.Connection
    ids: tuple[int, int, int]


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Three runs: two of one model at different sizes, one of another model."""
    database = tmp_path / "payloads.db"
    with storage.connect(database) as conn:
        run4, ids4 = _seed_run(
            conn, sha=SAME_MODEL, image_size=640,
            outcomes={"correct": 10, "false_positive": 4, "false_negative": 1},
        )
        run5, ids5 = _seed_run(
            conn, sha=SAME_MODEL, image_size=448,
            outcomes={"correct": 12, "false_positive": 2, "false_negative": 1},
        )
        run7, _ = _seed_run(
            conn, sha=OTHER_MODEL, image_size=448,
            outcomes={"correct": 5, "false_negative": 5},
        )
        _seed_evidence(conn, run4, ids4[10:], map50=0.69)
        _seed_evidence(conn, run5, ids5[12:], map50=0.80)
    monkeypatch.setattr(config, "DB_PATH", database)
    with storage.connect(database) as conn:
        yield _Seeded(conn, (run4, run5, run7))


# ---------------------------------------------------------------------------
# Why this module exists at all
# ---------------------------------------------------------------------------

class TestItCarriesNoProtocolAndOpensNoDatabase:
    """The two rules that make a second caller possible at all."""

    def test_importing_it_pulls_in_no_mcp_package(self):
        """Ramanujan forbids `mcp` in requirements and a test enforces the ban.

        If this module reached for `ToolError` or `Field`, the vendored copy
        would be unimportable there and the second surface could not exist.
        """
        import ast

        source = Path(mcp_payloads.__file__).read_text()
        imported: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [m for m in imported if m.split(".")[0] in {"mcp", "pydantic"}]

    def test_it_never_opens_a_database_of_its_own(self):
        """Every entry point takes a connection.

        One caller opens read-only through the engine's config; the other goes
        through Ramanujan's service layer. Neither should gain a second
        connection in a process that already has one.
        """
        import inspect

        for name in ("list_runs_payload", "get_analysis_payload", "run_row",
                     "evidence_present", "groups_for", "mask_summary"):
            first = list(inspect.signature(getattr(mcp_payloads, name)).parameters)[0]
            assert first == "connection", f"{name} takes {first!r} first"

        source = Path(mcp_payloads.__file__).read_text()
        assert "connect_read_only" not in source
        assert "storage.connect(" not in source

    def test_a_refusal_is_a_plain_exception(self):
        """A refusal is a plain exception, so any protocol layer can map it."""
        assert issubclass(PayloadError, Exception)
        assert PayloadError.__mro__[1] is Exception


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

class TestValidatingRunIds:
    """What a caller may ask for, and how a bad ask is refused."""

    def test_it_keeps_the_order_asked_for(self):
        """The order asked for is the order reported."""
        assert mcp_payloads.resolve_run_ids([5, 4, 7]) == [5, 4, 7]

    def test_a_repeated_id_is_a_slip_not_a_refusal(self):
        """A repeated id is a slip to absorb, not a mistake to refuse."""
        assert mcp_payloads.resolve_run_ids([4, 4, 5]) == [4, 5]

    def test_an_empty_list_is_refused(self):
        """Naming no run at all is a mistake worth saying out loud."""
        with pytest.raises(PayloadError, match="at least one run"):
            mcp_payloads.resolve_run_ids([])

    @pytest.mark.parametrize("bad", ["4", 4.0, None, [4]])
    def test_anything_that_is_not_an_integer_is_refused(self, bad):
        """A run id that is not an integer is refused."""
        with pytest.raises(PayloadError, match="must be integers"):
            mcp_payloads.resolve_run_ids([bad])

    def test_a_boolean_is_refused_rather_than_read_as_run_one(self):
        """`bool` subclasses `int`, so `True` would otherwise be run 1."""
        with pytest.raises(PayloadError, match="must be integers"):
            mcp_payloads.resolve_run_ids([True])

    def test_more_runs_than_the_configured_maximum_are_refused(self):
        """The configured ceiling is enforced."""
        too_many = list(range(config.MCP_MAX_RUNS + 1))
        with pytest.raises(PayloadError, match="at most"):
            mcp_payloads.resolve_run_ids(too_many)

    def test_exactly_the_maximum_is_allowed(self):
        """The ceiling is inclusive."""
        assert len(mcp_payloads.resolve_run_ids(list(range(config.MCP_MAX_RUNS)))) \
            == config.MCP_MAX_RUNS


# ---------------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------------

class TestTheListing:
    """What `list_runs_payload` says about a run, and what it withholds."""

    def test_it_lists_everything_when_the_caller_names_nothing(self, db):
        """Naming nothing lists every run the database holds."""
        payload = mcp_payloads.list_runs_payload(db.conn)
        assert len(payload["runs"]) == 3

    def test_the_caller_decides_which_runs_are_visible(self, db):
        """The boundary is the caller's, deliberately.

        Ramanujan passes only runs it can resolve back to a model record it
        created; the engine's own server passes everything.
        """
        run4, _run5, _run7 = db.ids
        only = [storage.load_run(db.conn, run4)]
        payload = mcp_payloads.list_runs_payload(db.conn, runs=only)
        assert [r["run_id"] for r in payload["runs"]] == [run4]

    def test_an_empty_selection_lists_nothing_rather_than_everything(self, db):
        """An empty selection lists nothing rather than everything.

        The difference between "show me these none" and "show me all" is
        the difference between a scoped surface and an open one.
        """
        payload = mcp_payloads.list_runs_payload(db.conn, runs=[])
        assert payload["runs"] == []

    def test_it_carries_no_filesystem_path_by_default(self, db):
        """A listing carries no filesystem path."""
        import json

        raw = json.dumps(mcp_payloads.list_runs_payload(db.conn))
        assert "/ws/" not in raw
        assert "model_path" not in raw

    def test_a_run_says_which_others_share_its_checkpoint(self, db):
        """Replication only means something within one set of weights."""
        run4, run5, run7 = db.ids
        rows = {r["run_id"]: r for r in mcp_payloads.list_runs_payload(db.conn)["runs"]}
        assert sorted(rows[run4]["same_model_runs"]) == sorted([run4, run5])
        assert rows[run7]["same_model_runs"] == [run7]

    def test_it_names_the_family_only_when_a_job_produced_the_run(self, db):
        """The family is named only when a job recorded it.

        Detecting it otherwise means loading the checkpoint, which a
        read-only surface must never do.
        """
        _run4, run5, run7 = db.ids
        rows = {r["run_id"]: r for r in mcp_payloads.list_runs_payload(db.conn)["runs"]}
        assert rows[run7]["model"]["family"] is None

    def test_it_reports_the_schema_it_was_read_from(self, db):
        """The payload names the schema version it was read from."""
        assert mcp_payloads.list_runs_payload(db.conn)["schema_version"] \
            == storage.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

class TestTheComparison:
    """The six sections, the baseline, and how a missing run is reported."""

    def test_a_single_run_gives_the_same_six_sections(self, db):
        """One run gives the same six sections as several."""
        run4, _, _ = db.ids
        payload = mcp_payloads.get_analysis_payload(db.conn, [run4])
        for section in ("runs", "comparability", "per_run", "cross_run",
                        "evidence_gaps", "caveats"):
            assert section in payload, section

    def test_several_runs_compare_against_the_first_by_default(self, db):
        """Without a baseline, the first id given is the baseline."""
        run4, run5, _ = db.ids
        payload = mcp_payloads.get_analysis_payload(db.conn, [run4, run5])
        assert payload["cross_run"]["baseline"] == run4

    def test_the_baseline_can_be_chosen(self, db):
        """A caller may choose which run the deltas are measured against."""
        run4, run5, _ = db.ids
        payload = mcp_payloads.get_analysis_payload(
            db.conn, [run4, run5], baseline=run5
        )
        assert payload["cross_run"]["baseline"] == run5

    def test_a_baseline_outside_the_set_is_refused(self, db):
        """A baseline that is not among the runs asked for is refused."""
        run4, run5, _ = db.ids
        with pytest.raises(PayloadError, match="not among run_ids"):
            mcp_payloads.get_analysis_payload(db.conn, [run4], baseline=run5)

    def test_an_unknown_run_names_the_missing_id(self, db):
        """A refusal names the id that was not found."""
        with pytest.raises(PayloadError, match="No such run: 999"):
            mcp_payloads.get_analysis_payload(db.conn, [999])

    def test_the_refusal_names_whichever_listing_tool_the_caller_exposes(
        self, db
    ):
        """The refusal names whichever listing tool the caller exposes.

        Two surfaces, two tool names. Telling a reader to call one that
        does not exist on their surface would be worse than saying nothing.
        """
        with pytest.raises(PayloadError, match="Call list_runs"):
            mcp_payloads.get_analysis_payload(db.conn, [999])
        with pytest.raises(PayloadError, match="Call list_analyses"):
            mcp_payloads.get_analysis_payload(
                db.conn, [999], listing_tool="list_analyses"
            )

    def test_one_missing_run_among_several_is_still_refused(self, db):
        """One unknown id refuses the whole request rather than half-answering."""
        run4, _, _ = db.ids
        with pytest.raises(PayloadError, match="No such run"):
            mcp_payloads.get_analysis_payload(db.conn, [run4, 999])


class TestFactorsCarryTheirControlRate:
    """A factor's rate on failures is meaningless without its rate on successes."""

    def test_a_count_is_never_reported_without_the_rate_it_is_measured_against(
        self, db
    ):
        """A count is never reported without the rate it is measured against.

        D-031. A factor on 80% of failures means nothing until you know it
        is on 90% of successes.
        """
        run4, _, _ = db.ids
        factors = mcp_payloads.get_analysis_payload(
            db.conn, [run4]
        )["per_run"][run4]["factors"]
        assert factors
        for entry in factors:
            for field in ("failure_rate", "correct_rate", "failure_count",
                          "failure_total", "correct_count", "correct_total"):
                assert field in entry, f"{entry['factor']} lacks {field}"

    def test_a_factor_commoner_on_successes_does_not_qualify(self, db):
        """A factor commoner where the model was right explains nothing."""
        run4, _, _ = db.ids
        factors = {
            f["factor"]: f
            for f in mcp_payloads.get_analysis_payload(
                db.conn, [run4]
            )["per_run"][run4]["factors"]
        }
        # Seeded at 4/5 on failures and 9/10 on successes: commoner where the
        # model was right, so it explains nothing.
        assert factors["edge_truncation"]["qualifies"] is False
        assert factors["small_object"]["qualifies"] is True

    def test_an_undefined_lift_is_null_and_never_infinity(self, db):
        """An undefined lift is reported as null, never as infinity."""
        run4, _, _ = db.ids
        factors = {
            f["factor"]: f
            for f in mcp_payloads.get_analysis_payload(
                db.conn, [run4]
            )["per_run"][run4]["factors"]
        }
        assert factors["blur"]["lift"] is None


class TestEvidenceGapsAndCaveats:
    """What a run lacks, and what a reader must not conclude from it."""

    def test_a_run_with_nothing_but_findings_is_told_what_it_lacks(self, db):
        """A run carrying only findings is told what evidence it lacks."""
        _, _, run7 = db.ids
        gaps = mcp_payloads.get_analysis_payload(db.conn, [run7])["evidence_gaps"]
        assert gaps[run7] if isinstance(gaps, dict) else gaps

    def test_every_response_carries_the_caveats(self, db):
        """Every response carries the caveats a reader needs."""
        run4, _, _ = db.ids
        caveats = mcp_payloads.get_analysis_payload(db.conn, [run4])["caveats"]
        assert caveats
        assert any("not COCO mAP" in c for c in caveats)
        assert any("correlations" in c for c in caveats)

    def test_the_caveats_are_the_engine_s_own_text(self, db):
        """The caveats are the engine's own text, not a paraphrase."""
        from model_doctor.app import comparison

        run4, _, _ = db.ids
        assert mcp_payloads.get_analysis_payload(db.conn, [run4])["caveats"] \
            == list(comparison.CAVEATS)


class TestDescriptiveGroups:
    """The full grouping is opt-in; the discriminating one is the default."""

    def test_they_are_absent_unless_asked_for(self, db):
        """Descriptive groups are absent unless asked for."""
        run4, _, _ = db.ids
        block = mcp_payloads.get_analysis_payload(db.conn, [run4])["per_run"][run4]
        assert "descriptive_groups" not in block

    def test_asking_adds_them_beside_the_discriminating_grouping(self, db):
        """Asking adds them beside the discriminating grouping, not instead of it."""
        run4, _, _ = db.ids
        block = mcp_payloads.get_analysis_payload(
            db.conn, [run4], include_descriptive_groups=True
        )["per_run"][run4]
        assert "descriptive_groups" in block
        assert "groups" in block


class TestItChangesNothing:
    """Read-only by construction, asserted rather than assumed."""

    def test_reading_twice_gives_the_same_answer(self, db):
        """Reading twice gives the same answer."""
        run4, _, _ = db.ids
        first = mcp_payloads.get_analysis_payload(db.conn, [run4])
        second = mcp_payloads.get_analysis_payload(db.conn, [run4])
        assert first == second

    def test_it_writes_nothing_to_the_database(self, db):
        """Read-only by construction, asserted rather than assumed."""
        def count(table: str) -> int:
            return db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        tables = ("runs", "findings", "images", "clusters", "recommendations")
        before = {t: count(t) for t in tables}
        run4, run5, _ = db.ids
        mcp_payloads.list_runs_payload(db.conn)
        mcp_payloads.get_analysis_payload(
            db.conn, [run4, run5], include_descriptive_groups=True
        )
        assert {t: count(t) for t in tables} == before

    def test_it_can_read_a_connection_opened_read_only(self, db):
        """It can read through a connection opened read-only.

        Ramanujan opens the database that way. If anything here wrote, that
        connection would raise rather than silently succeed.
        """
        database = Path(config.DB_PATH)
        readonly = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        readonly.row_factory = sqlite3.Row
        try:
            run4, _, _ = db.ids
            payload = mcp_payloads.get_analysis_payload(readonly, [run4])
            assert payload["runs"][0]["run_id"] == run4
        finally:
            readonly.close()
