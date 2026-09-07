"""The MCP server is a read-only projection, and these tests hold it to that.

Three things matter. The two tools must be reachable through the real protocol,
not just as Python functions. Every failure a caller can cause — no such run,
too many runs, an absent database — must come back as a tool error carrying
the remedy, never as a crash. And nothing the server does may write: the
database is opened ``mode=ro`` and the process must not need the ML stack,
because a surface that reasons over evidence has no business loading a model.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession

import config
from app import storage
from app.clustering import DISCRIMINATING_METHOD
from app.mcp_server import create_server
from app.storage import RunContext

SAME_MODEL = "5" * 64
OTHER_MODEL = "9" * 64


def _seed_run(
    connection: sqlite3.Connection,
    *,
    sha: str,
    image_size: int,
    outcomes: dict[str, int],
    dataset: str = "/ws/stairs_1280_yolo/data.yaml",
) -> tuple[int, list[int]]:
    """One run with an image and one finding per requested outcome."""
    run_id = storage.save_run(
        connection,
        RunContext(
            model_path=f"/ws/model/best_{image_size}.pt",
            model_sha256=sha,
            dataset_yaml=dataset,
            split="test",
            confidence_threshold=0.25,
            match_iou_threshold=0.5,
            localization_iou_floor=0.1,
            image_size=image_size,
        ),
    )
    connection.execute(
        "INSERT INTO images (run_id, path, filename, width, height, "
        "prediction_count, truth_count) "
        "VALUES (?, '/ws/a.jpg', 'a.jpg', 100, 100, 1, 1)",
        (run_id,),
    )
    image_id = connection.execute(
        "SELECT id FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    finding_ids: list[int] = []
    for outcome, n in outcomes.items():
        for _ in range(n):
            cur = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, "
                "confidence) VALUES (?, ?, ?, 'staircase', 0.9)",
                (run_id, image_id, outcome),
            )
            finding_ids.append(cur.lastrowid)
    return run_id, finding_ids


def _seed_evidence(
    connection: sqlite3.Connection, run_id: int, failure_ids: list[int], map50: float
) -> None:
    """Evaluation, factor rates, one group and one recommendation for a run."""
    storage.save_evaluation(
        connection, run_id, task="bbox", evaluator="pycocotools COCOeval",
        evaluator_version="2.0.7", sweep_confidence=0.01, iou_thresholds="0.50:0.95",
        max_detections=100, ground_truth="yolo", gt_images=1, gt_annotations=1,
        prediction_count=3,
        metrics={"map50": map50, "map50_95": map50 - 0.2, "map_small": -1.0},
    )
    storage.save_factor_rates(
        connection, run_id,
        [("small_object", 4, 5, 1, 10, 8.0, 0.001),
         ("edge_truncation", 4, 5, 9, 10, 0.89, 0.6),
         ("blur", 0, 5, 0, 10, None, 1.0)],
    )
    storage.save_clusters(
        connection, run_id, DISCRIMINATING_METHOD,
        [("small_object", failure_ids)],
    )
    cluster_id = connection.execute(
        "SELECT id FROM clusters WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    storage.save_recommendations(
        connection, run_id,
        [(cluster_id, "recall_on_factor", "Improve recall for small_object.",
          "4 of 5 failures are misses.", "replicated", True, 5, 5.0)],
    )


@pytest.fixture()
def populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three runs: two of one model at different sizes, one of another model."""
    database = tmp_path / "mcp.db"
    with storage.connect(database) as connection:
        run4, ids4 = _seed_run(
            connection, sha=SAME_MODEL, image_size=640,
            outcomes={"correct": 10, "false_positive": 4, "false_negative": 1},
        )
        run5, ids5 = _seed_run(
            connection, sha=SAME_MODEL, image_size=448,
            outcomes={"correct": 12, "false_positive": 2, "false_negative": 1},
        )
        run7, _ = _seed_run(
            connection, sha=OTHER_MODEL, image_size=448,
            outcomes={"correct": 5, "false_negative": 5},
        )
        _seed_evidence(connection, run4, ids4[10:], map50=0.69)
        _seed_evidence(connection, run5, ids5[12:], map50=0.80)
        # Run 5 came from the browser, so it has a job naming its family.
        storage.create_job(
            connection, token="tok5", detector="yolo", split="test",
            model_name="best_448.pt", dataset_name="stairs_1280_yolo",
            workspace="/ws", image_size=448, confidence=0.25,
        )
        storage.update_job(connection, "tok5", run_id=run5)
    monkeypatch.setattr(config, "DB_PATH", database)
    return database


def _call(tool: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call one tool through the real protocol and return the raw result."""

    async def go() -> Any:
        async with (
            InMemoryTransport(create_server()) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await session.call_tool(tool, arguments or {})

    return anyio.run(go)


def _payload(result: Any) -> dict[str, Any]:
    """The structured content of a successful call."""
    assert not result.is_error, result.content[0].text
    assert result.structured_content is not None
    return result.structured_content


def _error_text(result: Any) -> str:
    assert result.is_error, "expected a tool error"
    return result.content[0].text


class TestProtocol:
    """The tools exist, describe themselves honestly, and answer over MCP."""

    @pytest.mark.usefixtures("populated")
    def test_exactly_three_read_only_tools_are_offered(self) -> None:
        """Exactly two read only tools are offered."""

        async def go() -> Any:
            async with (
                InMemoryTransport(create_server()) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                return await session.list_tools()

        listed = anyio.run(go)
        names = sorted(t.name for t in listed.tools)
        assert names == ["experiment_feasibility", "get_analysis", "list_runs"]
        for tool in listed.tools:
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False

    @pytest.mark.usefixtures("populated")
    def test_get_analysis_schema_requires_integer_run_ids(self) -> None:
        """Get analysis schema requires integer run ids."""

        async def go() -> Any:
            async with (
                InMemoryTransport(create_server()) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                return await session.list_tools()

        tool = next(t for t in anyio.run(go).tools if t.name == "get_analysis")
        schema = tool.input_schema
        assert "run_ids" in schema["required"]
        assert schema["properties"]["run_ids"]["type"] == "array"
        assert schema["properties"]["run_ids"]["items"]["type"] == "integer"
        assert schema["properties"]["run_ids"]["minItems"] == 1

    @pytest.mark.usefixtures("populated")
    def test_list_runs_reports_identity_evidence_and_replication_peers(
        self,
    ) -> None:
        """List runs reports identity evidence and replication peers."""
        body = _payload(_call("list_runs"))
        assert body["schema_version"] == storage.SCHEMA_VERSION
        runs = {r["run_id"]: r for r in body["runs"]}
        assert [r["run_id"] for r in body["runs"]] == [3, 2, 1], "newest first"
        five = runs[2]
        assert five["model"]["name"] == "best_448.pt"
        assert five["model"]["family"] == "yolo", "family comes from the job"
        assert runs[1]["model"]["family"] is None, "a CLI run has no job: unknown"
        assert five["dataset"]["classes"] == ["staircase"]
        assert five["counts"]["counts"]["false_positive"] == 2
        assert five["headline"]["map50_box"] == 0.80
        assert five["evidence"]["evaluation"] == ["bbox"]
        assert five["evidence"]["recommendations"] == 1
        assert five["same_model_runs"] == [1, 2]
        assert runs[3]["same_model_runs"] == [3]
        assert "path" not in five["model"]


class TestGetAnalysis:
    """The evidence, shaped for comparison."""

    @pytest.mark.usefixtures("populated")
    def test_compares_two_runs_of_one_model(self) -> None:
        """Compares two runs of one model."""
        body = _payload(_call("get_analysis", {"run_ids": [1, 2]}))
        assert body["comparability"]["same_model"] is True
        assert body["comparability"]["evaluation_settings_match"] is True
        diffs = {
            d["field"]: d["values"]
            for d in body["comparability"]["config_differences"]
        }
        assert diffs == {"image_size": {"1": 640, "2": 448}}
        cross = body["cross_run"]
        assert cross["baseline"] == 1
        assert cross["outcome_deltas"]["2"]["false_positive"] == -2
        assert cross["metric_deltas"]["2"]["map50_bbox"] == 0.11
        assert cross["factor_lift"]["small_object"] == {"1": 8.0, "2": 8.0}
        assert cross["group_replication"]["small_object"]["present_in"] == [1, 2]
        assert cross["shared_actionable_recommendations"] == [
            "recall_on_factor:small_object"
        ]

    @pytest.mark.usefixtures("populated")
    def test_per_run_blocks_carry_every_kind_of_evidence(self) -> None:
        """Per run blocks carry every kind of evidence."""
        body = _payload(_call("get_analysis", {"run_ids": [2]}))
        block = body["per_run"]["2"]
        assert block["outcomes"]["derived"]["rule"]
        metrics = block["evaluation"]["bbox"]["metrics"]
        assert metrics["map_small"] is None, "sentinel"
        assert block["evaluation"]["bbox"]["settings"]["max_detections"] == 100
        factors = {f["factor"]: f for f in block["factors"]}
        assert factors["small_object"]["qualifies"] is True
        assert factors["edge_truncation"]["qualifies"] is False
        assert factors["blur"]["lift"] is None
        assert block["groups"][0]["label"] == "small_object"
        assert block["recommendations"][0]["status"] == "replicated"
        assert body["caveats"]

    @pytest.mark.usefixtures("populated")
    def test_different_models_are_compared_with_a_warning(self) -> None:
        """Different models are compared with a warning."""
        body = _payload(_call("get_analysis", {"run_ids": [2, 3]}))
        assert body["comparability"]["same_model"] is False
        warnings = body["comparability"]["warnings"]
        assert any("different checkpoints" in w for w in warnings)

    @pytest.mark.usefixtures("populated")
    def test_missing_evidence_is_listed_with_its_remedy(self) -> None:
        """Run 3 was seeded with nothing but findings."""
        body = _payload(_call("get_analysis", {"run_ids": [3]}))
        gaps = {g["missing"]: g["how"] for g in body["evidence_gaps"]}
        assert "evaluation" in gaps and "--run-id 3" in gaps["evaluation"]
        assert "factor_rates" in gaps
        assert body["per_run"]["3"]["factors"] == []
        assert body["per_run"]["3"]["recommendations"] == []

    @pytest.mark.usefixtures("populated")
    def test_baseline_can_be_chosen(self) -> None:
        """Baseline can be chosen."""
        body = _payload(_call("get_analysis", {"run_ids": [1, 2], "baseline": 2}))
        assert body["cross_run"]["baseline"] == 2
        assert body["cross_run"]["outcome_deltas"]["1"]["false_positive"] == 2

    @pytest.mark.usefixtures("populated")
    def test_paths_only_on_request(self) -> None:
        """Paths only on request."""
        without = _payload(_call("get_analysis", {"run_ids": [2]}))
        assert "path" not in without["runs"][0]["model"]
        with_paths = _payload(
            _call("get_analysis", {"run_ids": [2], "include_paths": True})
        )
        assert with_paths["runs"][0]["model"]["path"].endswith("best_448.pt")

    @pytest.mark.usefixtures("populated")
    def test_descriptive_groups_only_on_request(self) -> None:
        """Descriptive groups only on request."""
        body = _payload(_call("get_analysis", {"run_ids": [2]}))
        assert "descriptive_groups" not in body["per_run"]["2"]
        body = _payload(
            _call("get_analysis", {"run_ids": [2], "include_descriptive_groups": True})
        )
        assert "descriptive_groups" in body["per_run"]["2"]

    @pytest.mark.usefixtures("populated")
    def test_never_returns_per_object_rows(self) -> None:
        """The response is aggregates; findings and images stay in the database."""
        body = _payload(_call("get_analysis", {"run_ids": [1, 2]}))
        text = json.dumps(body)
        assert "finding_id" not in text
        assert "image_id" not in text
        assert "a.jpg" not in text


class TestRefusals:
    """Every mistake a caller can make comes back as a tool error with a reason."""

    @pytest.mark.usefixtures("populated")
    def test_unknown_run_names_the_missing_id(self) -> None:
        """Unknown run names the missing id."""
        text = _error_text(_call("get_analysis", {"run_ids": [2, 999]}))
        assert "999" in text and "list_runs" in text

    @pytest.mark.usefixtures("populated")
    def test_empty_run_ids_are_refused(self) -> None:
        """Refused at the schema boundary, before any code of ours runs."""
        text = _error_text(_call("get_analysis", {"run_ids": []}))
        assert "at least" in text and "run_ids" in text

    @pytest.mark.usefixtures("populated")
    def test_too_many_runs_are_refused_with_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too many runs are refused with the limit."""
        monkeypatch.setattr(config, "MCP_MAX_RUNS", 2)
        text = _error_text(_call("get_analysis", {"run_ids": [1, 2, 3]}))
        assert "at most 2" in text

    @pytest.mark.usefixtures("populated")
    def test_duplicates_are_collapsed_not_refused(self) -> None:
        """Duplicates are collapsed not refused."""
        body = _payload(_call("get_analysis", {"run_ids": [2, 2, 1]}))
        assert [r["run_id"] for r in body["runs"]] == [2, 1]

    @pytest.mark.usefixtures("populated")
    def test_a_baseline_outside_the_set_is_refused(self) -> None:
        """A baseline outside the set is refused."""
        text = _error_text(_call("get_analysis", {"run_ids": [1], "baseline": 2}))
        assert "baseline" in text

    def test_a_missing_database_carries_the_command_that_makes_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing database carries the command that makes one."""
        monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.db")
        text = _error_text(_call("list_runs"))
        assert "app.diagnosis" in text


class TestReadOnly:
    """Nothing here can write, by construction rather than by discipline."""

    def test_calls_leave_the_database_byte_identical(self, populated: Path) -> None:
        """Calls leave the database byte identical."""
        before = populated.read_bytes()
        _payload(_call("list_runs"))
        _payload(_call("get_analysis", {"run_ids": [1, 2, 3]}))
        assert populated.read_bytes() == before

    def test_the_shared_opener_refuses_writes_at_the_engine(
        self, populated: Path
    ) -> None:
        """The shared opener refuses writes at the engine."""
        with (
            storage.connect_read_only(populated) as connection,
            pytest.raises(sqlite3.OperationalError, match="readonly"),
        ):
            connection.execute("DELETE FROM runs")

    def test_optional_tables_absent_degrade_to_gaps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A database from before the optional tables existed still answers."""
        database = tmp_path / "old.db"
        with storage.connect(database) as connection:
            _seed_run(
                connection, sha=SAME_MODEL, image_size=640, outcomes={"correct": 3}
            )
            for table in (
                "recommendations", "cluster_members", "clusters", "factor_rates",
                "run_evaluations", "run_benchmarks", "mask_findings", "heatmaps",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
        monkeypatch.setattr(config, "DB_PATH", database)
        body = _payload(_call("get_analysis", {"run_ids": [1]}))
        assert body["per_run"]["1"]["groups"] == []
        assert body["per_run"]["1"]["evaluation"] == {}
        missing = {g["missing"] for g in body["evidence_gaps"]}
        assert missing >= {"evaluation", "groups"}


def test_importing_the_server_loads_neither_torch_nor_fastapi() -> None:
    """The server reasons over evidence; it must not need a model or an HTTP stack.

    Run in a fresh interpreter, since this process has already imported both
    through other suites and ``sys.modules`` here would prove nothing.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.mcp_server; "
            "print(sorted(m for m in ('torch', 'ultralytics', 'fastapi') "
            "if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )
    assert result.stdout.strip() == "[]", (
        f"importing app.mcp_server loaded {result.stdout.strip()}"
    )
