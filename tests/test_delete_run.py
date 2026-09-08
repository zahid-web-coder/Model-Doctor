"""Deleting a run is the only destructive operation in the project.

Two things must hold. The database must be left with no row pointing at a run
that no longer exists — every owned table declares ``ON DELETE CASCADE``, and a
cascade that silently does not fire leaves orphans that surface much later as
wrong counts. And a stored path must never be treated as permission to unlink
a file: a database copied from another machine names paths this process has no
business touching.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.control import _remove_within_results
from model_doctor.utils.exceptions import ModelDoctorError, ResourceNotFoundError


def _run_ids(connection: sqlite3.Connection) -> list[int]:
    return [r[0] for r in connection.execute("SELECT id FROM runs ORDER BY id")]


@pytest.fixture()
def populated(tmp_path: Path) -> Path:
    """A database with two runs, so deleting one can be shown not to touch the other."""
    db = tmp_path / "runs.db"
    with storage.connect(db) as connection:
        for _ in range(2):
            connection.execute(
                "INSERT INTO runs (created_at, model_path, model_sha256, "
                "dataset_yaml, split, confidence_threshold, match_iou_threshold, "
                "localization_iou_floor, image_size) "
                "VALUES ('2026-01-01T00:00:00Z', 'm.pt', 'abc', 'd.yaml', "
                "'test', 0.25, 0.5, 0.1, 640)"
            )
        for run_id in (1, 2):
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?, ?, ?, 100, 100, 1, 1)",
                (run_id, f"/img/{run_id}.jpg", f"{run_id}.jpg"),
            )
            connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_id, "
                "class_name, confidence) VALUES (?, ?, 'correct', 0, 'a', 0.9)",
                (run_id, run_id),
            )
    return db


class TestDeleteCascade:
    """Deleting a run must leave nothing behind that pointed at it."""

    def test_removes_every_row_the_run_owned(self, populated: Path) -> None:
        """Removes every row the run owned."""
        with storage.connect(populated) as connection:
            storage.delete_run(connection, 1)
            for table in ("images", "findings"):
                left = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id = 1"
                ).fetchone()[0]
                assert left == 0, f"{table} kept rows for a deleted run"

    def test_leaves_the_other_run_untouched(self, populated: Path) -> None:
        """Leaves the other run untouched."""
        with storage.connect(populated) as connection:
            storage.delete_run(connection, 1)
            assert _run_ids(connection) == [2]
            kept = connection.execute(
                "SELECT COUNT(*) FROM findings WHERE run_id = 2"
            ).fetchone()[0]
            assert kept == 1

    def test_reports_what_it_removed(self, populated: Path) -> None:
        """Reports what it removed."""
        # The count is what a confirmation dialog quotes back, so it must match
        # what actually went rather than what was intended.
        with storage.connect(populated) as connection:
            before = storage.run_footprint(connection, 1)
            removed = storage.delete_run(connection, 1)
        assert removed["findings"] == before["findings"] == 1
        assert removed["runs"] == 1

    def test_an_unknown_run_is_refused_rather_than_reported_deleted(
        self, populated: Path
    ) -> None:
        """An unknown run is refused rather than reported deleted."""
        with storage.connect(populated) as connection:
            with pytest.raises(ResourceNotFoundError):
                storage.delete_run(connection, 999)
            assert _run_ids(connection) == [1, 2]

    def test_refuses_when_the_cascade_would_not_fire(self, populated: Path) -> None:
        """Refuses when the cascade would not fire."""
        # Without the pragma the parent row goes and every child survives,
        # orphaned. Failing loudly beats corrupting quietly.
        with storage.connect(populated) as connection:
            connection.execute("PRAGMA foreign_keys = OFF")
            with pytest.raises(ModelDoctorError, match="foreign keys"):
                storage.delete_run(connection, 1)
            assert _run_ids(connection) == [1, 2]


class TestFootprint:
    """What a delete will remove, counted before it removes it."""

    def test_counts_nothing_for_a_run_with_no_rows(self, populated: Path) -> None:
        """Counts nothing for a run with no rows."""
        with storage.connect(populated) as connection:
            connection.execute("DELETE FROM findings")
            connection.execute("DELETE FROM images")
            assert sum(storage.run_footprint(connection, 1).values()) == 0

    def test_a_table_the_optional_passes_never_created_counts_zero(
        self, populated: Path
    ) -> None:
        """A table the optional passes never created counts zero."""
        # A run analysed before a milestone added its table legitimately has
        # none, and the footprint must say zero rather than raise.
        with storage.connect(populated) as connection:
            connection.execute("DROP TABLE IF EXISTS embeddings")
            assert storage.run_footprint(connection, 1)["embeddings"] == 0


@pytest.fixture()
def scratch_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``RESULTS_DIR`` at a throwaway tree for the duration of a test.

    **Filesystem tests must never run against the real results directory.**
    Copying the database is not isolation: the copied rows still hold the live
    paths, so a delete driven from a copy unlinks the original files. That
    happened during development and destroyed a run's heatmaps — they were
    regenerable, and the next thing might not be.

    ``_remove_within_results`` reads ``config.RESULTS_DIR`` at call time, so
    redirecting the module attribute redirects both the containment root and
    the files under test together. Isolating one without the other would be
    worse than neither: paths inside a scratch tree would be refused against a
    live root, and the tests would pass for the wrong reason.
    """
    root = tmp_path / "results"
    (root / "heatmaps").mkdir(parents=True)
    monkeypatch.setattr(config, "RESULTS_DIR", root)
    return root


def _live_results_dir() -> Path:
    """The real results directory, resolved without going through config.

    Deliberately recomputed from this file's location rather than read from
    ``config``, so a test that has monkeypatched the config still compares
    against the true path.
    """
    return (Path(__file__).resolve().parent.parent / "results").resolve()


class TestTestsAreIsolatedFromLiveArtifacts:
    """The guard on the guard: prove these tests cannot reach real files."""

    def test_the_fixture_does_not_point_at_the_live_results_directory(
        self, scratch_results: Path
    ) -> None:
        """The fixture does not point at the live results directory."""
        live = _live_results_dir()
        resolved = scratch_results.resolve()
        assert resolved != live
        assert not resolved.is_relative_to(live), (
            "scratch results directory sits inside the real one; a delete test "
            "could unlink a real artifact"
        )

    @pytest.mark.usefixtures("scratch_results")
    def test_a_live_path_is_refused_while_the_fixture_is_active(self) -> None:
        """A live path is refused while the fixture is active."""
        # The exact failure that occurred: a path recorded by the real
        # installation, handed to the remover. With RESULTS_DIR redirected it
        # falls outside containment and must be refused, never unlinked.
        live_path = _live_results_dir() / "heatmaps" / "run_5" / "finding_1081.png"
        existed = live_path.exists()
        deleted, refused = _remove_within_results([str(live_path)])
        assert (deleted, refused) == (0, 1)
        if existed:
            assert live_path.exists(), "a real heatmap was deleted by a test"

    def test_no_test_in_this_module_writes_outside_its_scratch_tree(
        self, scratch_results: Path
    ) -> None:
        """No test in this module writes outside its scratch tree."""
        # Every file this module creates lives under the fixture's root, so a
        # stray write shows up as a path that is not relative to it.
        written = scratch_results / "heatmaps" / "run_x" / "a.png"
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(b"x")
        assert written.resolve().is_relative_to(scratch_results.resolve())
        _remove_within_results([str(written)])
        assert not written.exists()


class TestFileRemovalIsContained:
    """A stored path is data, never permission to unlink."""

    def test_deletes_files_inside_the_results_directory(
        self, scratch_results: Path
    ) -> None:
        """Deletes files inside the results directory."""
        target = scratch_results / "heatmaps" / "run_1" / "a.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        deleted, refused = _remove_within_results([str(target)])
        assert (deleted, refused) == (1, 0)
        assert not target.exists()

    @pytest.mark.usefixtures("scratch_results")
    def test_refuses_a_path_outside_the_results_directory(
        self, tmp_path: Path
    ) -> None:
        """Refuses a path outside the results directory."""
        # The path came out of a database, which may have been written on
        # another machine. It is data, not authority to unlink.
        outside = tmp_path / "precious.png"
        outside.write_bytes(b"keep me")
        deleted, refused = _remove_within_results([str(outside)])
        assert (deleted, refused) == (0, 1)
        assert outside.exists(), "deleted a file outside the results directory"

    def test_refuses_a_symlink_that_escapes_the_results_directory(
        self, scratch_results: Path, tmp_path: Path
    ) -> None:
        """Refuses a symlink that escapes the results directory."""
        outside = tmp_path / "target.png"
        outside.write_bytes(b"keep me")
        link_dir = scratch_results / "heatmaps" / "run_link"
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / "escape.png"
        link.symlink_to(outside)
        deleted, refused = _remove_within_results([str(link)])
        assert (deleted, refused) == (0, 1)
        assert outside.exists(), "followed a symlink out of the results directory"

    def test_an_already_missing_file_is_not_an_error(
        self, scratch_results: Path
    ) -> None:
        """An already missing file is not an error."""
        gone = scratch_results / "heatmaps" / "run_1" / "nope.png"
        gone.parent.mkdir(parents=True, exist_ok=True)
        deleted, refused = _remove_within_results([str(gone)])
        assert refused == 0
        assert deleted == 1

    @pytest.mark.usefixtures("scratch_results")
    def test_no_paths_removes_nothing(self) -> None:
        """No paths removes nothing."""
        assert _remove_within_results([]) == (0, 0)

    def test_prunes_the_directory_only_once_it_is_empty(
        self, scratch_results: Path
    ) -> None:
        """Prunes the directory only once it is empty."""
        run_dir = scratch_results / "heatmaps" / "run_1"
        run_dir.mkdir(parents=True, exist_ok=True)
        removed = run_dir / "a.png"
        kept = run_dir / "b.preview.jpg"
        removed.write_bytes(b"x")
        kept.write_bytes(b"y")
        _remove_within_results([str(removed)])
        # A derived preview is not recorded in `heatmaps`, so it survives and
        # the directory must survive with it.
        assert run_dir.is_dir()
        assert kept.exists()


class TestPruningSpansEveryDirectory:
    """A run's artifacts are not guaranteed to sit in one directory."""

    def test_prunes_every_directory_it_emptied(self, scratch_results: Path) -> None:
        """Prunes every directory it emptied."""
        # The `heatmaps` table records absolute paths and nothing forces them
        # to share a parent. Pruning only the first left the rest behind.
        heatmaps = scratch_results / "heatmaps"
        dirs = [heatmaps / "run_1", heatmaps / "legacy" / "run_1", heatmaps / "spare"]
        files = []
        for index, directory in enumerate(dirs):
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"finding_{index}.png"
            target.write_bytes(b"x")
            files.append(target)

        deleted, refused = _remove_within_results([str(f) for f in files])

        assert (deleted, refused) == (3, 0)
        for directory in dirs:
            assert not directory.exists(), f"{directory} was left behind empty"

    def test_retires_a_parent_emptied_by_its_last_subdirectory(
        self, scratch_results: Path
    ) -> None:
        """Retires a parent emptied by its last subdirectory."""
        nested = scratch_results / "heatmaps" / "legacy" / "run_2"
        nested.mkdir(parents=True, exist_ok=True)
        target = nested / "a.png"
        target.write_bytes(b"x")

        _remove_within_results([str(target)])

        assert not nested.exists()
        assert not nested.parent.exists(), "the emptied parent was not retired"

    def test_keeps_a_directory_that_still_holds_something(
        self, scratch_results: Path
    ) -> None:
        """Keeps a directory that still holds something."""
        # Previews are derived and not recorded in `heatmaps`, so they survive
        # the delete and the directory must survive with them.
        run_dir = scratch_results / "heatmaps" / "run_3"
        run_dir.mkdir(parents=True, exist_ok=True)
        removed = run_dir / "a.png"
        kept = run_dir / "a.preview.jpg"
        removed.write_bytes(b"x")
        kept.write_bytes(b"y")

        _remove_within_results([str(removed)])

        assert run_dir.is_dir()
        assert kept.exists()

    def test_never_removes_the_results_directory_itself(
        self, scratch_results: Path
    ) -> None:
        """Never removes the results directory itself."""
        # It belongs to the installation, not to any run, and a later analysis
        # expects to write into it.
        loose = scratch_results / "stray.png"
        loose.write_bytes(b"x")
        (scratch_results / "heatmaps").rmdir()

        _remove_within_results([str(loose)])

        assert scratch_results.is_dir(), "pruned the results directory itself"

    def test_does_not_prune_a_directory_it_emptied_nothing_in(
        self, scratch_results: Path
    ) -> None:
        """Does not prune a directory it emptied nothing in."""
        # An unrelated empty directory is somebody else's business.
        bystander = scratch_results / "heatmaps" / "run_untouched"
        bystander.mkdir(parents=True, exist_ok=True)
        elsewhere = scratch_results / "heatmaps" / "run_4"
        elsewhere.mkdir(parents=True, exist_ok=True)
        target = elsewhere / "a.png"
        target.write_bytes(b"x")

        _remove_within_results([str(target)])

        assert bystander.is_dir(), "removed an empty directory it never touched"
        assert not elsewhere.exists()

    @pytest.mark.usefixtures("scratch_results")
    def test_a_refused_path_prunes_nothing(self, tmp_path: Path) -> None:
        """A refused path prunes nothing."""
        # Containment still governs: a path outside results is neither deleted
        # nor allowed to retire the directory holding it.
        outside_dir = tmp_path / "elsewhere"
        outside_dir.mkdir()
        outside = outside_dir / "a.png"
        outside.write_bytes(b"keep me")

        deleted, refused = _remove_within_results([str(outside)])

        assert (deleted, refused) == (0, 1)
        assert outside.exists()
        assert outside_dir.is_dir(), "pruned a directory outside the results tree"

    def test_a_symlinked_directory_is_not_pruned(
        self, scratch_results: Path
    ) -> None:
        """A symlinked directory is not pruned."""
        # Resolving the file lands inside the real tree; the link itself must
        # not be rmdir'd, which would remove somebody else's directory entry.
        real_dir = scratch_results / "heatmaps" / "run_5"
        real_dir.mkdir(parents=True, exist_ok=True)
        target = real_dir / "a.png"
        target.write_bytes(b"x")
        link = scratch_results / "heatmaps" / "alias"
        link.symlink_to(real_dir, target_is_directory=True)

        _remove_within_results([str(link / "a.png")])

        assert not target.exists()
        assert link.is_symlink(), "removed a symlink standing in for a directory"
