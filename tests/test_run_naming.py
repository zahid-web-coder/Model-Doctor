"""Naming a run: the column migration, the rule, and the route.

A run's name is the only writable field on a run. Everything else records what
an analysis pass did, and rewriting any of it would leave the stored parameters
disagreeing with the findings derived from them — but what a run is *called*
belongs to the reader. Ultralytics writes every checkpoint to the same
filename, so several runs are otherwise distinguishable only by id.

**The migration is the part with teeth.** Every schema change before version 12
added a whole table, which ``CREATE TABLE IF NOT EXISTS`` handles by itself. A
column on an existing table does not work that way: the create statement is
skipped for a database that already has the table, so without an explicit
``ALTER TABLE`` the column would exist only in freshly created files while
every database already on disk kept working without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import storage
from app.storage import RunContext

CONTEXT = RunContext(
    model_path="/weights/best.pt",
    model_sha256="a" * 64,
    dataset_yaml="/data/staircase/data.yaml",
    split="test",
    confidence_threshold=0.25,
    match_iou_threshold=0.5,
    localization_iou_floor=0.1,
    image_size=640,
)


@pytest.fixture()
def database(tmp_path: Path) -> Path:
    path = tmp_path / "named.db"
    with storage.connect(path) as connection:
        storage.save_run(connection, CONTEXT)
    return path


class TestTheColumnMigration:
    """A database written before version 12 has to gain the column."""

    def test_a_database_without_the_column_gains_it(self, tmp_path: Path) -> None:
        """The case `CREATE TABLE IF NOT EXISTS` cannot reach."""
        path = tmp_path / "old.db"
        with storage.connect(path) as connection:
            storage.save_run(connection, CONTEXT)
            # Reproduce a pre-12 database by removing the column again. SQLite
            # supports dropping a column from 3.35, and this is exactly the
            # shape every database on disk had before this change.
            connection.execute("ALTER TABLE runs DROP COLUMN name")
            columns = {
                r["name"] for r in connection.execute("PRAGMA table_info(runs)")
            }
            assert "name" not in columns, "the fixture did not reproduce a pre-12 file"

        # Opening it again must migrate it.
        with storage.connect(path) as connection:
            columns = {
                r["name"] for r in connection.execute("PRAGMA table_info(runs)")
            }
            assert "name" in columns
            run = storage.load_run(connection, 1)
            assert run is not None
            assert run.name is None, "an unnamed run must not acquire a name"

    def test_migrating_does_not_disturb_the_run_it_widens(
        self, tmp_path: Path
    ) -> None:
        """Adding the column must leave every recorded parameter alone."""
        path = tmp_path / "intact.db"
        with storage.connect(path) as connection:
            run_id = storage.save_run(connection, CONTEXT)
            before = storage.load_run(connection, run_id)
            connection.execute("ALTER TABLE runs DROP COLUMN name")
        with storage.connect(path) as connection:
            after = storage.load_run(connection, run_id)
        assert before is not None and after is not None
        assert after == before

    def test_running_it_twice_changes_nothing(self, database: Path) -> None:
        """Opening a database is not a migration event, so it must be safe."""
        with storage.connect(database) as connection:
            storage.rename_run(connection, 1, "first look")
        for _ in range(3):
            with storage.connect(database) as connection:
                run = storage.load_run(connection, 1)
                assert run is not None
                assert run.name == "first look", "a reopen overwrote the name"

    def test_the_recorded_version_moves(self, database: Path) -> None:
        """A migrated database reports the version it was migrated to."""
        with storage.connect(database) as connection:
            version = connection.execute("SELECT version FROM schema_info").fetchone()
        assert version["version"] == storage.SCHEMA_VERSION
        assert storage.SCHEMA_VERSION >= 12


class TestRenaming:
    """The rule, which lives with the column rather than in a route."""

    def test_a_name_survives_a_round_trip(self, database: Path) -> None:
        """Written, closed, reopened, and still there."""
        with storage.connect(database) as connection:
            updated = storage.rename_run(connection, 1, "448px baseline")
            assert updated is not None and updated.name == "448px baseline"
        with storage.connect(database) as connection:
            assert storage.load_run(connection, 1).name == "448px baseline"

    def test_a_run_starts_unnamed_rather_than_blank(self, database: Path) -> None:
        """A new run has no name at all, not an empty one."""
        # None and "" behave differently everywhere; only one of them can mean
        # "never named", and it is the one that renders as the id alone.
        with storage.connect(database) as connection:
            assert storage.load_run(connection, 1).name is None

    def test_clearing_stores_null_not_an_empty_string(self, database: Path) -> None:
        """A name that renders as nothing is not the same as no name."""
        with storage.connect(database) as connection:
            storage.rename_run(connection, 1, "temporary")
            cleared = storage.rename_run(connection, 1, "")
            assert cleared is not None
            assert cleared.name is None
            stored = connection.execute("SELECT name FROM runs WHERE id=1").fetchone()
            assert stored["name"] is None

    def test_whitespace_is_not_a_name(self, database: Path) -> None:
        """Spaces alone leave the run unnamed rather than named invisibly."""
        with storage.connect(database) as connection:
            assert storage.rename_run(connection, 1, "   ").name is None

    def test_surrounding_whitespace_is_trimmed(self, database: Path) -> None:
        """A stray leading space must not make two names look different."""
        with storage.connect(database) as connection:
            assert storage.rename_run(connection, 1, "  baseline  ").name == "baseline"

    def test_an_over_long_name_is_refused_rather_than_truncated(
        self, database: Path
    ) -> None:
        """Silently shortening a label makes the reader's text disappear."""
        with storage.connect(database) as connection:
            with pytest.raises(ValueError, match="at most"):
                storage.rename_run(connection, 1, "x" * (storage.MAX_RUN_NAME + 1))
            assert storage.load_run(connection, 1).name is None

    def test_a_name_at_the_limit_is_accepted(self, database: Path) -> None:
        """The bound is inclusive, so the limit itself is usable."""
        with storage.connect(database) as connection:
            name = "x" * storage.MAX_RUN_NAME
            assert storage.rename_run(connection, 1, name).name == name

    def test_renaming_a_run_that_does_not_exist_reports_it(
        self, database: Path
    ) -> None:
        """A rename that matched nothing is reported, not silently accepted."""
        with storage.connect(database) as connection:
            assert storage.rename_run(connection, 999, "ghost") is None

    def test_two_runs_may_share_a_name(self, database: Path) -> None:
        """Names are labels, not keys — the id remains the identity.

        Refusing a duplicate would make naming a second run of the same
        experiment an error, which is the case a reader most wants to label.
        """
        with storage.connect(database) as connection:
            second = storage.save_run(connection, CONTEXT)
            storage.rename_run(connection, 1, "448px")
            storage.rename_run(connection, second, "448px")
            names = [r.name for r in storage.list_runs(connection)]
        assert names.count("448px") == 2

    def test_renaming_touches_nothing_else_about_the_run(
        self, database: Path
    ) -> None:
        """The other columns are evidence, and a rename is not a re-analysis."""
        with storage.connect(database) as connection:
            before = storage.load_run(connection, 1)
            storage.rename_run(connection, 1, "renamed")
            after = storage.load_run(connection, 1)
        assert before is not None and after is not None
        for field in (
            "created_at", "model_path", "model_sha256", "dataset_yaml", "split",
            "confidence_threshold", "match_iou_threshold",
            "localization_iou_floor", "image_size",
        ):
            assert getattr(after, field) == getattr(before, field), field

    def test_renaming_one_run_leaves_its_neighbours_alone(
        self, database: Path
    ) -> None:
        """Naming one run must not name any other."""
        with storage.connect(database) as connection:
            second = storage.save_run(connection, CONTEXT)
            storage.rename_run(connection, second, "only me")
            assert storage.load_run(connection, 1).name is None


class TestTheReaderStaysReadOnly:
    """Renaming must not become a reason to write from the read API."""

    def test_the_read_api_still_declares_no_non_get_route(self) -> None:
        """The rename lives on the control API, and must stay there."""
        from app.api import create_app

        methods = {
            method
            for route in create_app().routes
            for method in getattr(route, "methods", set())
        }
        assert methods <= {"GET", "HEAD"}, (
            f"the reader gained {methods - {'GET', 'HEAD'}}"
        )

    def test_the_read_api_serves_the_name_it_cannot_write(
        self, database: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name is only useful if the screens reading runs can see it."""
        from fastapi.testclient import TestClient

        from app.api import create_app

        with storage.connect(database) as connection:
            storage.rename_run(connection, 1, "448px baseline")
        monkeypatch.setattr("config.DB_PATH", database)
        client = TestClient(create_app())
        rows = client.get("/runs").json()
        assert rows[0]["name"] == "448px baseline"


class TestTheRoute:
    """The control API's rename, including what it refuses."""

    @pytest.fixture()
    def client(self, database: Path, monkeypatch: pytest.MonkeyPatch):
        """A control app pointed at a database holding one unnamed run."""
        from fastapi.testclient import TestClient

        from app.control import create_app

        monkeypatch.setattr("config.DB_PATH", database)
        return TestClient(create_app())

    def test_a_rename_is_applied_and_echoed(self, client, database: Path) -> None:
        """The route stores the name and returns what it stored."""
        response = client.patch("/runs/1", json={"name": "448px baseline"})
        assert response.status_code == 200
        assert response.json() == {"run_id": 1, "name": "448px baseline"}
        with storage.connect(database) as connection:
            assert storage.load_run(connection, 1).name == "448px baseline"

    def test_clearing_is_expressible_without_a_second_route(self, client) -> None:
        """Both null and blank undo a name through the same endpoint."""
        client.patch("/runs/1", json={"name": "temporary"})
        assert client.patch("/runs/1", json={"name": None}).json()["name"] is None
        client.patch("/runs/1", json={"name": "temporary"})
        assert client.patch("/runs/1", json={"name": ""}).json()["name"] is None

    def test_an_unknown_run_is_a_404_not_a_silent_success(self, client) -> None:
        """Naming a run that is not there must not look like it worked."""
        assert client.patch("/runs/999", json={"name": "ghost"}).status_code == 404

    def test_an_over_long_name_is_refused(self, client) -> None:
        """The length rule holds at the route, not only in storage."""
        response = client.patch(
            "/runs/1", json={"name": "x" * (storage.MAX_RUN_NAME + 1)}
        )
        assert response.status_code == 422

    def test_the_route_cannot_reach_any_other_column(
        self, client, database: Path
    ) -> None:
        """Extra fields must not become a way to rewrite a run's parameters."""
        client.patch("/runs/1", json={"name": "ok", "split": "train", "image_size": 1})
        with storage.connect(database) as connection:
            run = storage.load_run(connection, 1)
        assert run.split == "test"
        assert run.image_size == 640
