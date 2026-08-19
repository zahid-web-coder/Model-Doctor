"""Contract tests for the read-only API.

Built against a **real** database — created through ``app.storage`` so it has
the actual schema, indexes and foreign keys — rather than against mocked
readers. A mocked storage layer would pass while the published contract was
broken, which is the failure these tests exist to prevent.

The file-serving cases are the security surface and are tested as three
distinct outcomes: served, absent, and refused. Collapsing "outside the allowed
roots" into 404 would hide a misconfiguration; collapsing it into 500 would
report a caller's ordinary situation as a server fault.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import config
from app import storage
from app.api import create_app
from app.clustering import DISCRIMINATING_METHOD
from app.storage import RunContext

RUN_ID = 1


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Confine the API to a temporary directory it is allowed to read."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(config, "API_FILE_ROOTS", (allowed.resolve(),))
    return allowed


@pytest.fixture
def populated(tmp_path: Path, roots: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a complete schema-v8 database with one run and every table filled."""
    database = tmp_path / "model_doctor.db"

    present = roots / "present.jpg"
    Image.new("RGB", (40, 40), (128, 128, 128)).save(present)
    heatmap = roots / "heat.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(heatmap)

    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="sha-api-tests",
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        for path in (present, roots / "vanished.jpg", Path("/etc/hosts")):
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?, ?, ?, 40, 40, 1, 1)",
                (run_id, str(path), path.name),
            )
        # The two cases a findings-derived count silently loses: an image that
        # processed cleanly and contained nothing, and one that never ran.
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) "
            "VALUES (?, '/x/empty.jpg', 'empty.jpg', 40, 40, 0, 0)",
            (run_id,),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count, error) "
            "VALUES (?, '/x/broken.jpg', 'broken.jpg', NULL, NULL, 0, 0, "
            "'Unreadable image')",
            (run_id,),
        )
        image_id = connection.execute(
            "SELECT id FROM images ORDER BY id LIMIT 1"
        ).fetchone()["id"]

        finding_ids = []
        for index in range(12):
            cursor = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, iou) "
                "VALUES (?, ?, ?, 'door_frame', 0.9)",
                (
                    run_id,
                    image_id,
                    "false_negative" if index < 10 else "correct",
                ),
            )
            finding_ids.append(int(cursor.lastrowid))

        storage.save_root_causes(
            connection,
            run_id,
            [(fid, "small_object", 0.8, "tiny") for fid in finding_ids],
        )
        storage.save_factor_rates(
            connection, run_id, [("small_object", 10, 12, 2, 40, 2.5, 0.001)]
        )
        storage.save_clusters(
            connection, run_id, DISCRIMINATING_METHOD, [("small_object", finding_ids)]
        )
        cluster_id = connection.execute("SELECT id FROM clusters").fetchone()["id"]
        storage.save_recommendations(
            connection,
            run_id,
            [
                (
                    cluster_id,
                    "recall_on_factor",
                    "Improve recall for small_object.",
                    "10 of 12 are misses.",
                    "provisional",
                    True,
                    12,
                    12.0,
                ),
                (
                    cluster_id,
                    "unexplained_backlog",
                    "Investigate.",
                    "Nothing accounts for these.",
                    "insufficient_evidence",
                    False,
                    12,
                    12.0,
                ),
            ],
        )
        storage.save_mask_findings(
            connection,
            run_id,
            [
                (finding_ids[0], 0.1, "poor_localization", [[0, 0], [5, 0], [5, 5]]),
                (finding_ids[1], None, None, None),
            ],
        )
        storage.save_embeddings(
            connection,
            run_id,
            "test-encoder",
            [
                (finding_ids[0], [1.0, 0.0, 0.0]),
                (finding_ids[1], [0.9, 0.1, 0.0]),
                (finding_ids[2], [0.0, 0.0, 1.0]),
            ],
        )
        storage.save_heatmaps(
            connection, run_id, "grad-cam", "layers", [(finding_ids[0], str(heatmap))]
        )

    monkeypatch.setattr(config, "DB_PATH", database)
    return database


@pytest.fixture
def client(populated: Path) -> TestClient:
    """A client bound to the fully populated database.

    Depends on `populated` for its side effect — it points `config.DB_PATH` at
    the database it built — and asserts that, so the dependency is visible
    rather than implicit.
    """
    assert populated.is_file()
    return TestClient(create_app())


@pytest.fixture
def minimal_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """A client bound to a database with only the three guaranteed tables."""
    database = tmp_path / "minimal.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_info (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, model_path TEXT NOT NULL,
            model_sha256 TEXT NOT NULL, dataset_yaml TEXT NOT NULL, split TEXT NOT NULL,
            confidence_threshold REAL NOT NULL,
            match_iou_threshold REAL NOT NULL,
            localization_iou_floor REAL NOT NULL, image_size INTEGER NOT NULL);
        CREATE TABLE images (
            id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, path TEXT NOT NULL,
            filename TEXT NOT NULL, width INTEGER, height INTEGER,
            prediction_count INTEGER NOT NULL, truth_count INTEGER NOT NULL,
            error TEXT);
        CREATE TABLE findings (
            id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, image_id INTEGER NOT NULL,
            outcome TEXT NOT NULL, class_id INTEGER, class_name TEXT NOT NULL,
            confidence REAL, iou REAL,
            pred_x1 REAL, pred_y1 REAL, pred_x2 REAL, pred_y2 REAL,
            truth_x1 REAL, truth_y1 REAL, truth_x2 REAL, truth_y2 REAL,
            truth_polygon TEXT);
        INSERT INTO schema_info VALUES (1, 't');
        INSERT INTO runs VALUES (1, 't', 'm.pt', 'sha', 'd.yaml', 'test',
                                 0.25, 0.5, 0.1, 640);
        INSERT INTO images VALUES (1, 1, '/x/a.jpg', 'a.jpg', 10, 10, 0, 0, NULL);
        INSERT INTO findings (id, run_id, image_id, outcome, class_name)
            VALUES (1, 1, 1, 'correct', 'door');
        """
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(config, "DB_PATH", database)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def test_runs_are_listed_with_their_provenance(client: TestClient) -> None:
    """A finding without provenance is an assertion with no context."""
    response = client.get("/runs")

    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["model_sha256"] == "sha-api-tests"
    assert runs[0]["split"] == "test"


def test_one_run_returns_its_thresholds(client: TestClient) -> None:
    """The numbers needed to reproduce or compare a run."""
    body = client.get(f"/runs/{RUN_ID}").json()

    assert body["confidence_threshold"] == 0.25
    assert body["match_iou_threshold"] == 0.5
    assert body["image_size"] == 640


def test_an_unknown_run_is_not_found(client: TestClient) -> None:
    """404 rather than an empty body, which would read as 'no data'."""
    assert client.get("/runs/999").status_code == 404


def test_outcome_counts_are_served(client: TestClient) -> None:
    """The aggregate a consumer opens with."""
    body = client.get(f"/runs/{RUN_ID}/outcomes").json()

    assert body["false_negative"] == 10
    assert body["correct"] == 2


# ---------------------------------------------------------------------------
# Findings and pagination
# ---------------------------------------------------------------------------
def test_findings_are_paginated_with_a_total(client: TestClient) -> None:
    """A page without a total cannot be paged through."""
    body = client.get(f"/runs/{RUN_ID}/findings?limit=5&offset=0").json()

    assert body["total"] == 12
    assert len(body["items"]) == 5
    assert body["limit"] == 5
    assert body["offset"] == 0


def test_pagination_offset_advances_the_window(client: TestClient) -> None:
    """Two pages must not return the same rows."""
    first = client.get(f"/runs/{RUN_ID}/findings?limit=5&offset=0").json()["items"]
    second = client.get(f"/runs/{RUN_ID}/findings?limit=5&offset=5").json()["items"]

    assert {row["id"] for row in first}.isdisjoint({row["id"] for row in second})


def test_an_oversized_page_is_rejected(client: TestClient) -> None:
    """The ceiling stops a request materialising an entire table."""
    assert client.get(f"/runs/{RUN_ID}/findings?limit=99999").status_code == 422


# ---------------------------------------------------------------------------
# Groups, causes, rates, recommendations, masks
# ---------------------------------------------------------------------------
def test_groups_default_to_the_discriminating_partition(client: TestClient) -> None:
    """SCHEMA.md instructs consumers to show this one (D-033)."""
    groups = client.get(f"/runs/{RUN_ID}/groups").json()

    assert len(groups) == 1
    assert groups[0]["label"] == "small_object"
    assert groups[0]["method"] == DISCRIMINATING_METHOD


def test_group_members_carry_their_image_filename(client: TestClient) -> None:
    """Enough to open the failure without a second request."""
    cluster_id = client.get(f"/runs/{RUN_ID}/groups").json()[0]["id"]

    members = client.get(f"/groups/{cluster_id}/members").json()

    assert len(members) == 12
    assert "filename" in members[0]


def test_root_causes_are_served(client: TestClient) -> None:
    """Attributed conditions, with the measurement behind each."""
    causes = client.get(f"/runs/{RUN_ID}/root-causes").json()

    assert causes
    assert causes[0]["factor"] == "small_object"
    assert causes[0]["evidence"]


def test_factor_rates_carry_lift_and_significance(client: TestClient) -> None:
    """A count without its base rate is not evidence (D-031)."""
    rates = client.get(f"/runs/{RUN_ID}/factor-rates").json()

    assert len(rates) == 1
    assert rates[0]["lift"] == pytest.approx(2.5)
    assert rates[0]["p_value"] == pytest.approx(0.001)
    assert rates[0]["correct_total"] == 40


def test_recommendations_keep_the_documented_order_and_the_refusals(
    client: TestClient,
) -> None:
    """Actionable first — and the refusal is returned, not filtered (D-035)."""
    rows = client.get(f"/runs/{RUN_ID}/recommendations").json()

    assert len(rows) == 2
    assert rows[0]["actionable"] is True
    assert rows[1]["status"] == "insufficient_evidence"
    assert rows[1]["actionable"] is False


def test_mask_findings_distinguish_unmeasured_from_zero(client: TestClient) -> None:
    """NULL means never measured. Rendering it as 0.0 would invent a failure."""
    rows = client.get(f"/runs/{RUN_ID}/mask-findings").json()

    scores = {row["finding_id"]: row["mask_iou"] for row in rows}
    assert None in scores.values()
    assert any(value == pytest.approx(0.1) for value in scores.values() if value)


def test_mask_disagreements_can_be_requested_alone(client: TestClient) -> None:
    """The set mask diagnosis exists to surface."""
    response = client.get(f"/runs/{RUN_ID}/mask-findings?disagreements=true")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# Neighbours — unreachable from a SQL-only consumer
# ---------------------------------------------------------------------------
def test_neighbours_rank_the_most_similar_failure_first(client: TestClient) -> None:
    """Finding 2 was placed near finding 1 and must outrank finding 3."""
    findings = client.get(f"/runs/{RUN_ID}/findings").json()["items"]
    query = findings[0]["id"]

    neighbours = client.get(
        f"/runs/{RUN_ID}/findings/{query}/neighbours?limit=2"
    ).json()

    assert len(neighbours) == 2
    assert neighbours[0]["similarity"] > neighbours[1]["similarity"]
    assert all(row["finding_id"] != query for row in neighbours)


def test_neighbours_for_a_finding_without_an_embedding_are_not_found(
    client: TestClient,
) -> None:
    """An actionable 404 beats an empty list that reads as 'nothing similar'."""
    response = client.get(f"/runs/{RUN_ID}/findings/9999/neighbours")

    assert response.status_code == 404
    assert "embedding" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Files — the security surface
# ---------------------------------------------------------------------------
def test_an_existing_image_is_served(client: TestClient) -> None:
    """The ordinary case."""
    response = client.get("/images/1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content


def test_a_stale_image_path_is_not_found(client: TestClient) -> None:
    """Images move; a run outlives the directory it was diagnosed from."""
    response = client.get("/images/2")

    assert response.status_code == 404
    assert "no longer present" in response.json()["detail"]


def test_a_path_outside_the_allowed_roots_is_refused(client: TestClient) -> None:
    """The stored path is input. /etc/hosts exists and must still be refused."""
    response = client.get("/images/3")

    assert response.status_code == 403
    assert "outside the directories" in response.json()["detail"]


def test_an_unknown_image_id_is_not_found(client: TestClient) -> None:
    """No row, as distinct from a row whose file is gone."""
    assert client.get("/images/9999").status_code == 404


def test_a_heatmap_is_served_by_finding(client: TestClient) -> None:
    """Explanations are files, recorded against the finding they explain."""
    finding_id = client.get(f"/runs/{RUN_ID}/findings").json()["items"][0]["id"]

    response = client.get(f"/findings/{finding_id}/heatmap")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_a_missing_heatmap_is_not_found(client: TestClient) -> None:
    """Most findings have none, and that is not an error."""
    assert client.get("/findings/9999/heatmap").status_code == 404


def test_no_endpoint_accepts_a_filesystem_path(client: TestClient) -> None:
    """Traversal has no entry point: files are addressed by id only."""
    for attempt in ("/images/../../etc/hosts", "/images/%2e%2e%2fetc%2fhosts"):
        assert client.get(attempt).status_code in (404, 422)


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def test_optional_tables_absent_returns_empty_not_an_error(
    minimal_client: TestClient,
) -> None:
    """A version-1 database still answers; the surfaces needing more are empty."""
    for path in (
        "/runs/1/groups",
        "/runs/1/root-causes",
        "/runs/1/factor-rates",
        "/runs/1/recommendations",
        "/runs/1/mask-findings",
        "/runs/1/findings/1/neighbours",
    ):
        response = minimal_client.get(path)
        assert response.status_code == 200, path
        assert response.json() == [], path


def test_the_guaranteed_tables_still_work_without_the_optional_ones(
    minimal_client: TestClient,
) -> None:
    """runs, images and findings are the contract that never changes (D-020)."""
    assert minimal_client.get("/runs").status_code == 200
    assert minimal_client.get("/runs/1/findings").json()["total"] == 1


def test_a_missing_database_is_service_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 with the command that produces one, not a stack trace."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.db")
    client = TestClient(create_app())

    response = client.get("/runs")

    assert response.status_code == 503
    assert "app.diagnosis" in response.json()["detail"]


def test_health_reports_a_missing_database_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A health check that raises when unhealthy is not a health check."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.db")

    body = TestClient(create_app()).get("/health").json()

    assert body["status"] == "no-database"


def test_health_reports_the_schema_version(client: TestClient) -> None:
    """A consumer detects a mismatch from this."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["schema_version"] == storage.SCHEMA_VERSION
    assert body["expected_schema_version"] == storage.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The API adds nothing
# ---------------------------------------------------------------------------
def test_the_api_never_writes(populated: Path) -> None:
    """Read-only by construction: a write must fail at the connection."""
    from app.api import read_only

    with read_only(populated) as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM findings")


def test_cors_is_restricted_rather_than_open() -> None:
    """The API serves local file contents; '*' would expose them to any page."""
    assert "*" not in config.CORS_ORIGINS
    assert any("localhost:3000" in origin for origin in config.CORS_ORIGINS)



# ---------------------------------------------------------------------------
# Images — the rows a findings-derived count would lose
# ---------------------------------------------------------------------------
def test_every_attempted_image_is_returned(client: TestClient) -> None:
    """Including the ones that produced no findings and the one that failed."""
    images = client.get(f"/runs/{RUN_ID}/images").json()

    names = [row["filename"] for row in images]
    assert len(images) == 5
    assert "empty.jpg" in names, "an image with no findings must still appear"
    assert "broken.jpg" in names, "an image that never ran must still appear"


def test_an_errored_image_keeps_its_error_text(client: TestClient) -> None:
    """`error` is what separates "found nothing" from "never ran"."""
    images = client.get(f"/runs/{RUN_ID}/images").json()

    broken = next(row for row in images if row["filename"] == "broken.jpg")
    empty = next(row for row in images if row["filename"] == "empty.jpg")

    assert broken["error"] == "Unreadable image"
    assert empty["error"] is None
    # Both have no findings. Only the error column tells them apart.
    assert broken["prediction_count"] == empty["prediction_count"] == 0


def test_image_rows_carry_what_the_design_needs(client: TestClient) -> None:
    """Id for the byte endpoint, filename for display, path for reference."""
    row = client.get(f"/runs/{RUN_ID}/images").json()[0]

    for field in (
        "id",
        "run_id",
        "path",
        "filename",
        "width",
        "height",
        "prediction_count",
        "truth_count",
        "error",
    ):
        assert field in row, field


def test_image_counts_cannot_be_derived_from_findings(client: TestClient) -> None:
    """The reason this endpoint exists, asserted rather than argued.

    Four of the five images produce no findings, so a count taken from the
    findings list sees only one of them — and cannot report the errored one in
    any case.
    """
    images = client.get(f"/runs/{RUN_ID}/images").json()
    findings = client.get(f"/runs/{RUN_ID}/findings?limit=1000").json()["items"]

    derived = {row["image_id"] for row in findings}

    assert len(images) == 5
    assert len(derived) == 1
    assert len(images) - len(derived) == 4


def test_images_for_an_unknown_run_are_not_found(client: TestClient) -> None:
    """404 rather than an empty list, which would read as "no images"."""
    assert client.get("/runs/999/images").status_code == 404


def test_images_work_without_any_optional_table(minimal_client: TestClient) -> None:
    """Images is one of the three guaranteed tables (D-020)."""
    response = minimal_client.get("/runs/1/images")

    assert response.status_code == 200
    assert len(response.json()) == 1


# ---------------------------------------------------------------------------
# Deployability — the API must not drag in the ML stack
# ---------------------------------------------------------------------------
def test_importing_the_api_does_not_load_torch() -> None:
    """The API never touches a model, and must not pay for one to be deployed.

    `config.DEVICE` was resolved at import, and `resolve_device()` imports
    torch. Every module imports `config`, so `import app.api` pulled in roughly
    500 MB of ML stack before serving a request — the difference between a
    100 MB container and a 2.5 GB one (D-038).

    Run in a fresh interpreter: this test process has already imported torch
    through the other suites, so checking `sys.modules` here would prove
    nothing.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.api; "
            "print('torch' in sys.modules or 'ultralytics' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )

    assert result.stdout.strip() == "False", (
        "importing app.api loaded the ML stack; a deployed API would need it"
    )


def test_device_is_still_readable_and_cached() -> None:
    """Deferring the computation must not change what callers see."""
    import config as config_module

    first = config_module.DEVICE
    second = config_module.DEVICE

    assert isinstance(first, str)
    assert first
    assert first is second, "resolved once, then cached"


def test_an_unknown_config_attribute_still_raises() -> None:
    """The lazy hook must not swallow genuine typos."""
    import config as config_module

    with pytest.raises(AttributeError, match="no attribute"):
        _ = config_module.NOT_A_REAL_SETTING
