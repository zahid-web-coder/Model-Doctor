"""Relations as the read surfaces expose them: HTTP, MCP, and the pure summary.

Two things are tested. That the evidence reaches a consumer intact — counts
matching storage, partners resolvable, the provisional reading carried with the
bounds that produced it. And that exposing it changed nothing: the read API
declares no non-GET route, the existing `get_analysis` blocks keep their shape,
and no relation reaches the factor pipeline.

**`duplicate_prediction` must never read as a cause.** It is a provisional
interpretation of a continuous measurement, and the wording that says so is
asserted here rather than left to a reviewer's memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession

import config
from app import comparison, relations, storage
from app.storage import RunContext

SIZE = 100


def square(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@pytest.fixture()
def measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, int]:
    """A run carrying every relation this pass can produce.

    One image holds a merge candidate, a box/mask disagreement and two
    unmatched predictions — one that duplicates a found object and one that
    touches nothing.
    """
    from app.image_diagnosis import analyse_run as diagnose

    database = tmp_path / "surface.db"
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="s" * 64,
                dataset_yaml="/data/things/data.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?,'/i.jpg','i.jpg',?,?,3,2)",
            (run_id, SIZE, SIZE),
        )
        image_id = connection.execute(
            "SELECT id FROM images WHERE run_id=?", (run_id,)
        ).fetchone()[0]

        ids: list[int] = []
        for outcome, truth in (
            # Covers both objects: a box/mask disagreement and, for the second
            # object, the prediction that makes it a merge candidate.
            ("poor_localization", square(5, 5, 45, 45)),
            ("false_negative", square(50, 50, 90, 90)),
            ("false_positive", None),
            ("false_positive", None),
        ):
            ids.append(
                int(
                    connection.execute(
                        "INSERT INTO findings (run_id, image_id, outcome, "
                        "class_name, truth_polygon) VALUES (?,?,?,'thing',?)",
                        (
                            run_id,
                            image_id,
                            outcome,
                            json.dumps(truth) if truth else None,
                        ),
                    ).lastrowid
                )
            )
        storage.save_mask_findings(
            connection,
            run_id,
            [
                (ids[0], 0.2, "poor_localization", square(0, 0, 95, 95)),
                # Sits inside the matched object: duplicate-like.
                (ids[2], None, None, square(10, 10, 40, 40)),
                # Far from anything found: not duplicate-like.
                (ids[3], None, None, square(96, 0, 100, 4)),
            ],
        )
        diagnose(connection, run_id)
        relations.analyse_run(connection, run_id)

    monkeypatch.setattr(config, "DB_PATH", database)
    return database, run_id


def call(tool: str, arguments: dict[str, Any] | None = None) -> Any:
    """One MCP tool call over the in-memory transport."""
    from app.mcp_server import create_server

    async def go() -> Any:
        async with (
            InMemoryTransport(create_server()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await session.call_tool(tool, arguments or {})

    result = anyio.run(go)
    assert not result.is_error, result.content[0].text
    return result.structured_content


class TestTheHttpSurface:
    """The read API's two new routes."""

    def client(self) -> TestClient:
        """A client on the read API, pointed at the fixture database."""
        from app.api import create_app

        return TestClient(create_app())

    def test_every_stored_relation_is_returned(
        self, measured: tuple[Path, int]
    ) -> None:
        """What the endpoint serves must equal what storage holds."""
        database, run_id = measured
        rows = self.client().get(f"/runs/{run_id}/relations").json()
        with storage.connect(database) as connection:
            stored = storage.load_finding_relations(connection, run_id)
        assert len(rows) == len(stored)
        assert {r["finding_id"] for r in rows} == {s.finding_id for s in stored}

    def test_filtering_by_relation(self, measured: tuple[Path, int]) -> None:
        """A caller can ask for one relation without reading them all."""
        database, run_id = measured
        client = self.client()
        for name in comparison.RELATION_ORDER:
            rows = client.get(f"/runs/{run_id}/relations?relation={name}").json()
            assert all(r["relation"] == name for r in rows)
        merges = client.get(
            f"/runs/{run_id}/relations?relation=merge_candidate"
        ).json()
        assert merges, "the fixture holds a merge candidate"

    def test_filtering_by_finding(self, measured: tuple[Path, int]) -> None:
        """One finding's relations, without the rest of the run."""
        database, run_id = measured
        client = self.client()
        every = client.get(f"/runs/{run_id}/relations").json()
        wanted = every[0]["finding_id"]
        rows = client.get(f"/runs/{run_id}/relations?finding_id={wanted}").json()
        assert rows and all(r["finding_id"] == wanted for r in rows)

    def test_an_unknown_relation_returns_nothing_rather_than_failing(
        self, measured: tuple[Path, int]
    ) -> None:
        """A filter naming something that does not exist is empty, not an error."""
        database, run_id = measured
        response = self.client().get(
            f"/runs/{run_id}/relations?relation=invented"
        )
        assert response.status_code == 200
        assert response.json() == []

    def test_the_summary_matches_the_rows(self, measured: tuple[Path, int]) -> None:
        """The aggregate must be derivable from the rows themselves."""
        database, run_id = measured
        client = self.client()
        rows = client.get(f"/runs/{run_id}/relations").json()
        summary = client.get(f"/runs/{run_id}/relations/summary").json()
        assert summary["total"] == len(rows)
        assert sum(summary["counts"].values()) == len(rows)

    def test_the_summary_reports_the_provisional_reading(
        self, measured: tuple[Path, int]
    ) -> None:
        """The label, its status, and the bounds that produced it."""
        database, run_id = measured
        summary = (
            self.client().get(f"/runs/{run_id}/relations/summary").json()
        )
        duplicate = summary["duplicate_prediction"]
        assert duplicate["status"] == "provisional"
        assert duplicate["rule"]["prediction_on_matched_object_at_least"] == (
            relations.DUPLICATE_MIN_ONOBJECT
        )
        assert duplicate["rule"]["best_coverage_at_least"] == (
            relations.DUPLICATE_COVERAGE_FLOOR
        )
        assert "does_not_mean" in duplicate

    def test_an_unknown_run_is_refused(self, measured: tuple[Path, int]) -> None:
        """A run that does not exist is a 404, not an empty list."""
        database, _ = measured
        assert self.client().get("/runs/9999/relations").status_code == 404

    def test_the_reader_still_declares_no_non_get_route(self) -> None:
        """Adding a surface must not add a way to write through it (D-037)."""
        from app.api import create_app

        methods = {
            method
            for route in create_app().routes
            for method in getattr(route, "methods", set())
        }
        assert methods <= {"GET", "HEAD"}

    def test_reading_changes_nothing(self, measured: tuple[Path, int]) -> None:
        """Every route is safe to call repeatedly."""
        database, run_id = measured

        def snapshot() -> list[tuple]:
            with storage.connect(database) as connection:
                return [
                    tuple(r)
                    for r in connection.execute(
                        "SELECT * FROM finding_relations ORDER BY rowid"
                    )
                ]

        before = snapshot()
        client = self.client()
        for _ in range(3):
            client.get(f"/runs/{run_id}/relations")
            client.get(f"/runs/{run_id}/relations/summary")
        assert snapshot() == before


class TestPartnerIntegrityOverTheApi:
    """A relation must point at something a consumer can resolve."""

    def test_partners_are_valid_and_share_the_run_and_image(
        self, measured: tuple[Path, int]
    ) -> None:
        """Every partner id resolves to a finding beside it on the image."""
        database, run_id = measured
        from app.api import create_app

        client = TestClient(create_app())
        rows = client.get(f"/runs/{run_id}/relations").json()
        partnered = [r for r in rows if r["partner_finding_id"] is not None]
        assert partnered, "the fixture produces partnered relations"
        with storage.connect(database) as connection:
            for row in partnered:
                pair = connection.execute(
                    "SELECT a.run_id, a.image_id, b.run_id, b.image_id "
                    "FROM findings a JOIN findings b ON b.id = ? WHERE a.id = ?",
                    (row["partner_finding_id"], row["finding_id"]),
                ).fetchone()
                assert pair is not None, "the partner must exist"
                assert pair[0] == pair[2] == run_id
                assert pair[1] == pair[3]


class TestTheMcpSurface:
    """What a reasoning model receives."""

    def test_list_runs_reports_relation_availability(
        self, measured: tuple[Path, int]
    ) -> None:
        """Consistent with image_diagnoses: a boolean beside the others."""
        _, run_id = measured
        payload = call("list_runs")
        row = next(r for r in payload["runs"] if r["run_id"] == run_id)
        assert row["evidence"]["relations"] is True
        assert "image_diagnoses" in row["evidence"]

    def test_get_analysis_carries_a_relations_block(
        self, measured: tuple[Path, int]
    ) -> None:
        """A third lens, beside outcomes and images."""
        _, run_id = measured
        payload = call("get_analysis", {"run_ids": [run_id]})
        block = payload["per_run"][str(run_id)]["relations"]
        assert block is not None
        assert set(block["counts"]) == set(comparison.RELATION_ORDER)

    def test_the_block_names_the_provisional_reading_and_its_limits(
        self, measured: tuple[Path, int]
    ) -> None:
        """The wording that stops it reading as a cause is part of the payload."""
        _, run_id = measured
        block = call("get_analysis", {"run_ids": [run_id]})["per_run"][str(run_id)][
            "relations"
        ]
        duplicate = block["duplicate_prediction"]
        assert duplicate["status"] == "provisional"
        # The field is named for the negation it carries, and it names the
        # mechanisms a reader might otherwise assume.
        assert "does_not_mean" in duplicate
        assert "no mechanism" in duplicate["does_not_mean"]
        for mechanism in ("suppression", "decoding", "assignment"):
            assert mechanism in duplicate["does_not_mean"]

    def test_the_caveats_say_a_relation_is_not_a_cause(
        self, measured: tuple[Path, int]
    ) -> None:
        """Shipped with every response, so it cannot be read past."""
        _, run_id = measured
        caveats = " ".join(call("get_analysis", {"run_ids": [run_id]})["caveats"])
        assert "not a proven cause" in caveats
        assert "duplicate_prediction is a provisional reading" in caveats

    def test_counts_match_storage(self, measured: tuple[Path, int]) -> None:
        """The aggregate must be derivable from the rows a caller can fetch."""
        database, run_id = measured
        block = call("get_analysis", {"run_ids": [run_id]})["per_run"][str(run_id)][
            "relations"
        ]
        with storage.connect(database) as connection:
            rows = storage.load_finding_relations(connection, run_id)
        assert block["total"] == len(rows)
        for name in comparison.RELATION_ORDER:
            assert block["counts"][name] == sum(1 for r in rows if r.relation == name)
        # Scoped to the measurement it interprets: `qualifies` is also true on
        # merge_candidate and box_mask_disagreement rows, which exist only when
        # they meet their own rule.
        assert block["duplicate_prediction"]["count"] == sum(
            1
            for r in rows
            if r.relation == "prediction_on_matched_object" and r.qualifies
        )

    def test_a_run_without_relations_reports_none_not_zero(
        self, measured: tuple[Path, int]
    ) -> None:
        """Not measured and measured-and-empty must stay distinguishable."""
        database, _ = measured
        with storage.connect(database) as connection:
            other = storage.save_run(
                connection,
                RunContext(
                    model_path="m.pt",
                    model_sha256="z" * 64,
                    dataset_yaml="/data/things/data.yaml",
                    split="test",
                    confidence_threshold=0.25,
                    match_iou_threshold=0.5,
                    localization_iou_floor=0.1,
                    image_size=640,
                ),
            )
        payload = call("get_analysis", {"run_ids": [other]})
        assert payload["per_run"][str(other)]["relations"] is None

    def test_calling_twice_changes_nothing(self, measured: tuple[Path, int]) -> None:
        """The tools are annotated read-only; this checks they are."""
        database, run_id = measured

        def snapshot() -> list[tuple]:
            with storage.connect(database) as connection:
                return [
                    tuple(r)
                    for r in connection.execute(
                        "SELECT * FROM finding_relations ORDER BY rowid"
                    )
                ]

        before = snapshot()
        call("get_analysis", {"run_ids": [run_id]})
        call("list_runs")
        assert snapshot() == before


class TestBackwardCompatibility:
    """Existing consumers must not have to change."""

    def test_every_previous_per_run_block_is_still_present(
        self, measured: tuple[Path, int]
    ) -> None:
        """Existing consumers must not have to change."""
        _, run_id = measured
        block = call("get_analysis", {"run_ids": [run_id]})["per_run"][str(run_id)]
        for key in (
            "outcomes",
            "evaluation",
            "mask_summary",
            "factors",
            "groups",
            "recommendations",
            "benchmarks",
            "images",
        ):
            assert key in block, f"{key} disappeared from get_analysis"

    def test_the_top_level_shape_only_gained_nothing(
        self, measured: tuple[Path, int]
    ) -> None:
        """Relations are nested inside per_run, not a new top-level key."""
        _, run_id = measured
        payload = call("get_analysis", {"run_ids": [run_id]})
        assert set(payload) == {
            "schema_version",
            "runs",
            "comparability",
            "per_run",
            "cross_run",
            "evidence_gaps",
            "caveats",
        }

    def test_outcome_counts_are_untouched_by_the_new_block(
        self, measured: tuple[Path, int]
    ) -> None:
        """A relation is additive evidence and changes no count."""
        database, run_id = measured
        block = call("get_analysis", {"run_ids": [run_id]})["per_run"][str(run_id)]
        with storage.connect(database) as connection:
            counts = storage.outcome_counts(connection, run_id)
        for outcome, n in counts.items():
            assert block["outcomes"]["counts"][outcome] == n


class TestTheSummaryItself:
    """`comparison.relation_summary` is pure and is tested as such."""

    def test_unmeasured_is_none(self) -> None:
        """None, so a caller cannot read absence as a finding."""
        assert comparison.relation_summary([]) is None

    def test_directions_belong_to_their_relation(self) -> None:
        """A direction is a property of its relation, not a flat tally."""
        rows = [
            storage.FindingRelationRow(
                finding_id=1,
                relation="box_mask_disagreement",
                direction="mask_ok_box_fails",
                value=0.9,
                qualifies=True,
                cover_hit=0.5,
            )
        ]
        got = comparison.relation_summary(rows)
        assert got["directions"] == {
            "box_mask_disagreement": {"mask_ok_box_fails": 1}
        }

    def test_the_distribution_is_reported_beside_the_reading(self) -> None:
        """A consumer disagreeing with the bounds can re-read the numbers."""
        rows = [
            storage.FindingRelationRow(
                finding_id=i,
                relation="prediction_on_matched_object",
                value=v,
                qualifies=v >= 0.15,
                cover_hit=0.5,
                best_coverage=0.9,
                threshold=0.15,
                coverage_floor=0.05,
            )
            for i, v in enumerate([0.0, 0.005, 0.2, 0.9, 0.95])
        ]
        got = comparison.relation_summary(rows)
        assert got["prediction_on_matched_object"]["measured"] == 5
        assert got["prediction_on_matched_object"]["at_or_below_0.01"] == 2
        assert got["prediction_on_matched_object"]["at_or_above_0.50"] == 2
        assert got["duplicate_prediction"]["count"] == 3

    def test_no_relation_name_is_a_factor(self, measured: tuple[Path, int]) -> None:
        """Relations must never reach the factor pipeline."""
        database, run_id = measured
        with storage.connect(database) as connection:
            factors = {
                r.factor for r in storage.load_factor_rates(connection, run_id)
            } | {
                r["factor"]
                for r in connection.execute(
                    "SELECT factor FROM root_causes WHERE run_id=?", (run_id,)
                )
            }
        assert not factors & set(comparison.RELATION_ORDER)
        assert "duplicate_prediction" not in factors
