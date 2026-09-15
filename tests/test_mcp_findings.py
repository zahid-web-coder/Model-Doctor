"""The findings payloads, which Phase 3 adds to both MCP surfaces.

`tests/test_mcp_payloads.py` covers the run-level shapes; these cover the two
that answer about individual findings. What matters here beyond the shape is
the pairing of ``run_id`` with ``finding_id``: finding ids come from one
sequence shared by every run, so the pairing is the only thing standing between
a caller holding an integer and a run they were never shown.

Seeding is imported from the server tests rather than copied, so the two suites
cannot describe different databases.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from model_doctor.app import mcp_payloads, storage
from model_doctor.app.mcp_payloads import PayloadError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mcp_server import (  # noqa: E402
    OTHER_MODEL,
    SAME_MODEL,
    _seed_run,
)


@dataclass(frozen=True)
class _Seeded:
    """An open connection and what the fixture put in it."""

    conn: sqlite3.Connection
    run_a: int
    run_b: int
    ids_a: list[int]
    ids_b: list[int]


@pytest.fixture()
def db(tmp_path: Path):
    """Two runs of different models, the first carrying per-finding evidence."""
    database = tmp_path / "findings.db"
    with storage.connect(database) as conn:
        run_a, ids_a = _seed_run(
            conn, sha=SAME_MODEL, image_size=640,
            outcomes={"correct": 3, "false_positive": 2, "false_negative": 1},
        )
        run_b, ids_b = _seed_run(
            conn, sha=OTHER_MODEL, image_size=448,
            outcomes={"correct": 1, "false_negative": 1},
        )
        failures = ids_a[3:]
        storage.save_root_causes(
            conn, run_a, [(failures[0], "small_object", 0.9, "area 120px")]
        )
        storage.save_mask_findings(
            conn, run_a, [(failures[0], 0.41, "poor_mask", None)]
        )
        storage.save_heatmaps(
            conn, run_a, "gradcam", "model.layer4",
            [(failures[0], "/ws/heat/1.png")],
        )
    with storage.connect(database) as conn:
        yield _Seeded(conn, run_a, run_b, ids_a, ids_b)


class TestArgumentValidation:
    """What a caller may ask for, and how a bad ask is refused."""

    def test_defaults_are_applied_when_nothing_is_given(self):
        """Omitting paging gives the documented defaults."""
        assert mcp_payloads.resolve_finding_page(None, None, None) == (
            mcp_payloads.DEFAULT_FINDING_LIMIT, 0, None
        )

    def test_the_limit_is_capped_rather_than_refused(self):
        """An over-large limit is clamped, not rejected."""
        limit, _, _ = mcp_payloads.resolve_finding_page(10_000, None, None)
        assert limit == mcp_payloads.MAX_FINDING_LIMIT

    @pytest.mark.parametrize("bad", ["5", 5.0, True, [5]])
    def test_a_non_integer_limit_is_refused(self, bad):
        """A limit that is not an integer is refused; `True` included."""
        with pytest.raises(PayloadError, match="`limit` must be an integer"):
            mcp_payloads.resolve_finding_page(bad, 0, None)

    def test_a_limit_below_one_is_refused(self):
        """A page of zero findings is a mistake worth naming."""
        with pytest.raises(PayloadError, match="at least 1"):
            mcp_payloads.resolve_finding_page(0, 0, None)

    def test_a_negative_offset_is_refused(self):
        """Paging backwards past the start is refused."""
        with pytest.raises(PayloadError, match="must not be negative"):
            mcp_payloads.resolve_finding_page(10, -1, None)

    def test_a_boolean_offset_is_refused_rather_than_read_as_zero(self):
        """`bool` subclasses `int`, so `False` would otherwise be offset 0."""
        with pytest.raises(PayloadError, match="`offset` must be an integer"):
            mcp_payloads.resolve_finding_page(10, False, None)

    def test_an_unknown_outcome_is_refused_and_names_the_valid_ones(self):
        """A misspelled outcome must not silently return everything."""
        with pytest.raises(PayloadError, match="must be one of"):
            mcp_payloads.resolve_finding_page(10, 0, "fasle_positive")

    def test_every_stored_outcome_is_accepted(self):
        """The accepted set is the stored set, not a hand-copied list."""
        for value in mcp_payloads.OUTCOMES:
            assert mcp_payloads.resolve_finding_page(10, 0, value)[2] == value


class TestTheListing:
    """What `list_findings_payload` reports, and what it withholds."""

    def test_it_lists_a_run_s_findings(self, db):
        """A run's findings come back with the totals needed to page them."""
        p = mcp_payloads.list_findings_payload(db.conn, db.run_a)
        assert p["run_id"] == db.run_a
        assert p["total"] == 6
        assert len(p["findings"]) == 6

    def test_a_finding_names_its_image_by_filename_only(self, db):
        """The image is named; its location on the host is not."""
        row = mcp_payloads.list_findings_payload(db.conn, db.run_a)["findings"][0]
        assert row["image_filename"] == "a.jpg"
        assert "path" not in row
        assert "/ws/" not in json.dumps(row)

    def test_filtering_by_outcome_narrows_both_page_and_total(self, db):
        """The total counts the filtered population, not the whole run."""
        p = mcp_payloads.list_findings_payload(
            db.conn, db.run_a, outcome="false_positive"
        )
        assert p["total"] == 2
        assert {f["outcome"] for f in p["findings"]} == {"false_positive"}

    def test_filtering_by_image_narrows_the_total_too(self, db):
        """An image filter applies to the count as well as the page."""
        image_id = mcp_payloads.list_findings_payload(
            db.conn, db.run_a
        )["findings"][0]["image_id"]
        p = mcp_payloads.list_findings_payload(db.conn, db.run_a, image_id=image_id)
        assert p["total"] == 6

    def test_paging_walks_the_run_without_repeating_a_finding(self, db):
        """Successive pages are disjoint and cover everything."""
        seen: list[int] = []
        for offset in (0, 2, 4):
            page = mcp_payloads.list_findings_payload(
                db.conn, db.run_a, limit=2, offset=offset
            )
            seen += [f["finding_id"] for f in page["findings"]]
        assert len(seen) == len(set(seen)) == 6

    def test_an_offset_past_the_end_is_empty_rather_than_an_error(self, db):
        """Running off the end is a normal outcome of paging."""
        p = mcp_payloads.list_findings_payload(db.conn, db.run_a, offset=500)
        assert p["findings"] == []
        assert p["total"] == 6

    def test_the_whole_run_s_outcome_counts_travel_with_every_page(self, db):
        """A page of failures must not be mistaken for the whole run."""
        p = mcp_payloads.list_findings_payload(
            db.conn, db.run_a, outcome="false_positive", limit=1
        )
        assert p["outcome_counts"]["correct"] == 3
        assert len(p["findings"]) == 1

    def test_an_unknown_run_is_refused(self, db):
        """A run that does not exist is named as such."""
        with pytest.raises(PayloadError, match="No such run: 999"):
            mcp_payloads.list_findings_payload(db.conn, 999)

    def test_a_non_integer_run_id_is_refused(self, db):
        """A run id that is not an integer is refused."""
        with pytest.raises(PayloadError, match="run_id must be an integer"):
            mcp_payloads.list_findings_payload(db.conn, "7")

    def test_every_page_carries_the_caveats(self, db):
        """What a reader must not conclude travels with the data."""
        p = mcp_payloads.list_findings_payload(db.conn, db.run_a)
        assert p["caveats"] == list(mcp_payloads.FINDING_CAVEATS)
        assert any("anecdote" in c for c in p["caveats"])


class TestOneFinding:
    """What `get_finding_payload` attaches to a single finding."""

    def test_it_returns_the_finding_asked_for(self, db):
        """The finding comes back under the run it was asked for."""
        fid = db.ids_a[0]
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, fid)
        assert p["finding"]["finding_id"] == fid
        assert p["finding"]["run_id"] == db.run_a

    def test_it_carries_the_factors_attributed_to_that_finding(self, db):
        """Factors come from the stored attribution, per finding."""
        failure = db.ids_a[3]
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, failure)
        assert p["finding"]["factors"] == ["small_object"]

    def test_a_finding_with_no_attributed_factor_says_so_with_an_empty_list(self, db):
        """Absence is reported as empty rather than omitted."""
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[0])
        assert p["finding"]["factors"] == []

    def test_it_carries_the_outline_result_when_masks_were_measured(self, db):
        """The mask pass is reported when it ran."""
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[3])
        assert p["finding"]["mask"]["mask_outcome"] == "poor_mask"

    def test_a_heatmap_is_reported_as_available_and_never_as_a_path(self, db):
        """Presence is useful; the file's location on the host is not."""
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[3])
        heat = p["finding"]["heatmap"]
        assert heat["available"] is True
        assert heat["method"] == "gradcam"
        assert "path" not in heat
        assert "/ws/" not in json.dumps(p)

    def test_relations_and_groups_are_present_as_lists(self, db):
        """Both are always lists, so a reader need not handle absence twice."""
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[3])
        assert isinstance(p["finding"]["relations"], list)
        assert isinstance(p["finding"]["groups"], list)

    def test_every_finding_carries_the_caveats(self, db):
        """One finding is an anecdote, and the payload says so."""
        p = mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[0])
        assert p["caveats"] == list(mcp_payloads.FINDING_CAVEATS)


class TestTheRunPairingIsTheBoundary:
    """A finding id alone is a key into every run; the pairing prevents that."""

    def test_a_finding_from_another_run_is_reported_as_absent(self, db):
        """The id is real, but not in this run, and the answer says neither."""
        other = db.ids_b[0]
        with pytest.raises(PayloadError, match=f"No such finding: {other}"):
            mcp_payloads.get_finding_payload(db.conn, db.run_a, other)

    def test_that_answer_is_identical_to_one_that_never_existed(self, db):
        """Otherwise the difference between the two maps the id space."""
        other = db.ids_b[0]
        try:
            mcp_payloads.get_finding_payload(db.conn, db.run_a, other)
        except PayloadError as e:
            real = str(e)
        try:
            mcp_payloads.get_finding_payload(db.conn, db.run_a, 999999)
        except PayloadError as e:
            absent = str(e)
        assert real.replace(str(other), "N") == absent.replace("999999", "N")

    def test_an_unknown_run_refuses_the_same_way(self, db):
        """A run outside the caller's reach answers like a missing finding."""
        with pytest.raises(PayloadError, match="No such finding"):
            mcp_payloads.get_finding_payload(db.conn, 4242, db.ids_a[0])

    def test_the_refusal_names_whichever_listing_tool_the_caller_exposes(self, db):
        """Two surfaces, two tool names."""
        with pytest.raises(PayloadError, match="Call list_findings"):
            mcp_payloads.get_finding_payload(db.conn, db.run_a, 999999)
        with pytest.raises(PayloadError, match="Call browse_findings"):
            mcp_payloads.get_finding_payload(
                db.conn, db.run_a, 999999, listing_tool="browse_findings"
            )

    @pytest.mark.parametrize("bad", ["1", 1.5, True])
    def test_a_non_integer_id_is_refused(self, db, bad):
        """Neither id may be anything but an integer."""
        with pytest.raises(PayloadError, match="must be an integer"):
            mcp_payloads.get_finding_payload(db.conn, db.run_a, bad)
        with pytest.raises(PayloadError, match="must be an integer"):
            mcp_payloads.get_finding_payload(db.conn, bad, db.ids_a[0])


class TestItChangesNothing:
    """Read-only by construction, asserted rather than assumed."""

    def test_reading_writes_nothing(self, db):
        """Neither payload changes a row."""
        def count(table: str) -> int:
            return db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        tables = ("runs", "findings", "images", "root_causes", "heatmaps")
        before = {t: count(t) for t in tables}
        mcp_payloads.list_findings_payload(db.conn, db.run_a)
        mcp_payloads.get_finding_payload(db.conn, db.run_a, db.ids_a[3])
        assert {t: count(t) for t in tables} == before

    def test_it_reads_through_a_connection_opened_read_only(self, db):
        """Ramanujan opens the database read-only; anything writing would raise."""
        path = db.conn.execute("PRAGMA database_list").fetchone()[2]
        ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        ro.row_factory = sqlite3.Row
        try:
            p = mcp_payloads.list_findings_payload(ro, db.run_a)
            assert p["total"] == 6
            one = mcp_payloads.get_finding_payload(ro, db.run_a, db.ids_a[3])
            assert one["finding"]["factors"] == ["small_object"]
        finally:
            ro.close()
