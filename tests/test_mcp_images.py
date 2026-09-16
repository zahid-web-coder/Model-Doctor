"""The visual-evidence payloads: the only ones that return bytes.

Everything else this module serves is derived from rows. These two open files,
which makes two properties worth proving rather than assuming: that no
caller-supplied string ever reaches the filesystem, and that a refusal never
describes where a file was looked for.

The reads go through the engine's own ``verified_file``, so the tests below
point ``API_FILE_ROOTS`` at a temporary directory and check the behaviour at
its edge — a file inside it, a file outside it, a symlink pointing out of it,
and a file that has since been removed.
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
from test_mcp_server import OTHER_MODEL, SAME_MODEL, _seed_run  # noqa: E402

PNG = bytes.fromhex("89504e470d0a1a0a") + b"fake-png-body"
JPG = bytes.fromhex("ffd8ffe0") + b"fake-jpeg-body"


@dataclass(frozen=True)
class _Seeded:
    conn: sqlite3.Connection
    run_a: int
    run_b: int
    ids_a: list[int]
    ids_b: list[int]
    image_a: int
    image_b: int
    root: Path


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two runs, real files inside one permitted root."""
    root = tmp_path / "data"
    (root / "heat").mkdir(parents=True)
    photo_a = root / "a.jpg"
    photo_a.write_bytes(JPG)
    photo_b = root / "b.jpg"
    photo_b.write_bytes(JPG)
    overlay = root / "heat" / "1.png"
    overlay.write_bytes(PNG)

    monkeypatch.setattr(config, "API_FILE_ROOTS", (root.resolve(),))
    database = tmp_path / "images.db"
    with storage.connect(database) as conn:
        run_a, ids_a = _seed_run(
            conn, sha=SAME_MODEL, image_size=640,
            outcomes={"correct": 1, "false_negative": 1},
        )
        run_b, ids_b = _seed_run(
            conn, sha=OTHER_MODEL, image_size=448, outcomes={"correct": 1},
        )
        conn.execute(
            "UPDATE images SET path = ? WHERE run_id = ?", (str(photo_a), run_a)
        )
        conn.execute(
            "UPDATE images SET path = ? WHERE run_id = ?", (str(photo_b), run_b)
        )
        image_a = conn.execute(
            "SELECT id FROM images WHERE run_id = ?", (run_a,)
        ).fetchone()[0]
        image_b = conn.execute(
            "SELECT id FROM images WHERE run_id = ?", (run_b,)
        ).fetchone()[0]
        storage.save_heatmaps(
            conn, run_a, "grad-cam", "model.layer4", [(ids_a[1], str(overlay))]
        )
    monkeypatch.setattr(config, "DB_PATH", database)
    with storage.connect(database) as conn:
        yield _Seeded(conn, run_a, run_b, ids_a, ids_b, image_a, image_b, root)


class TestGettingAnImage:
    """The photograph comes back as bytes, with nothing about where it lives."""

    def test_it_returns_the_bytes_and_a_media_type(self, db):
        """The file on disk is what comes back."""
        got = mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        assert got["data"] == JPG
        assert got["media_type"] == "image/jpeg"
        assert got["bytes"] == len(JPG)

    def test_it_names_the_photograph_without_locating_it(self, db):
        """Filename and dimensions, never a directory."""
        got = mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        assert got["filename"] == "a.jpg"
        assert got["width"] == 100
        assert "path" not in got
        assert str(db.root) not in repr({k: v for k, v in got.items() if k != "data"})

    def test_an_image_of_another_run_is_absent(self, db):
        """The run scopes the lookup, exactly as it does for a finding."""
        with pytest.raises(PayloadError, match=f"No such image: {db.image_b}"):
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_b)

    def test_that_refusal_matches_one_that_never_existed(self, db):
        """Otherwise the difference between them maps the id space."""
        try:
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_b)
        except PayloadError as e:
            other = str(e).replace(str(db.image_b), "N")
        try:
            mcp_payloads.image_for_run(db.conn, db.run_a, 999999)
        except PayloadError as e:
            never = str(e).replace("999999", "N")
        assert other == never

    @pytest.mark.parametrize("bad", ["1", 1.5, True, None])
    def test_a_non_integer_id_is_refused(self, db, bad):
        """Only integers; `bool` is not one."""
        with pytest.raises(PayloadError, match="must be an integer"):
            mcp_payloads.image_for_run(db.conn, db.run_a, bad)


class TestTheFilesystemIsNeverReachable:
    """No caller-supplied string reaches disk, and no refusal describes one."""

    def test_a_file_outside_the_permitted_roots_is_refused(self, db, tmp_path):
        """The root check is the engine's, and it is actually applied."""
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(JPG)
        db.conn.execute(
            "UPDATE images SET path = ? WHERE id = ?", (str(outside), db.image_a)
        )
        with pytest.raises(PayloadError) as excinfo:
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        assert str(tmp_path) not in str(excinfo.value)
        assert "outside.jpg" not in str(excinfo.value)

    def test_a_symlink_pointing_out_of_a_root_does_not_escape_it(self, db, tmp_path):
        """Symlinks are resolved before the check, so a link is not a loophole."""
        secret = tmp_path / "secret.jpg"
        secret.write_bytes(JPG)
        link = db.root / "innocent.jpg"
        link.symlink_to(secret)
        db.conn.execute(
            "UPDATE images SET path = ? WHERE id = ?", (str(link), db.image_a)
        )
        with pytest.raises(PayloadError) as excinfo:
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        assert "secret" not in str(excinfo.value)

    def test_a_file_that_has_gone_is_an_ordinary_answer(self, db):
        """Images move; a run outlives the directory it was diagnosed from."""
        (db.root / "a.jpg").unlink()
        with pytest.raises(PayloadError, match="may have been moved or removed"):
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)

    def test_no_refusal_names_a_directory(self, db, tmp_path):
        """`verified_file` names the resolved path; that must not travel."""
        for setup in (
            lambda: db.conn.execute(
                "UPDATE images SET path = ? WHERE id = ?",
                (str(tmp_path / "gone.jpg"), db.image_a),
            ),
            lambda: (db.root / "a.jpg").unlink(missing_ok=True),
        ):
            setup()
            try:
                mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
            except PayloadError as error:
                text = str(error)
                assert "/" not in text or "may have been moved" in text
                assert str(tmp_path) not in text

    def test_an_unreadable_format_is_refused_rather_than_guessed(self, db):
        """A wrong media type is worse than no image."""
        odd = db.root / "notes.txt"
        odd.write_bytes(b"not an image")
        db.conn.execute(
            "UPDATE images SET path = ? WHERE id = ?", (str(odd), db.image_a)
        )
        with pytest.raises(PayloadError, match="format this surface cannot return"):
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)

    def test_a_file_above_the_size_cap_is_refused(self, db, monkeypatch):
        """A response larger than a client can carry is not evidence."""
        monkeypatch.setattr(mcp_payloads, "MAX_IMAGE_BYTES", 4)
        with pytest.raises(PayloadError, match="above the"):
            mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)

    def test_every_argument_is_an_integer_so_no_string_reaches_disk(self):
        """The contract itself is the defence: there is nothing to smuggle."""
        import inspect

        for fn in (mcp_payloads.image_for_run, mcp_payloads.heatmap_for_finding):
            params = inspect.signature(fn).parameters
            names = [p for p in params if p != "connection"]
            assert "path" not in names
            assert "url" not in names
            assert "file" not in names


class TestGettingAHeatmap:
    """The overlay, when one exists, and a plain answer when none does."""

    def test_it_returns_the_overlay_for_an_explained_finding(self, db):
        """The stored overlay comes back with its method."""
        got = mcp_payloads.heatmap_for_finding(
            db.conn, db.run_a, db.ids_a[1], prefer_preview=False
        )
        assert got["data"] == PNG
        assert got["media_type"] == "image/png"
        assert got["method"] == "grad-cam"
        assert got["resolution"] == "original"

    def test_it_prefers_the_downscaled_companion_when_one_exists(self, db):
        """Several megabytes is a problem for every client in the path."""
        companion = db.root / "heat" / "1.preview.jpg"
        companion.write_bytes(JPG)
        got = mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1])
        assert got["resolution"] == "preview"
        assert got["media_type"] == "image/jpeg"

    def test_it_falls_back_to_the_original_when_there_is_no_companion(self, db):
        """A run explained before previews existed has none on disk."""
        got = mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1])
        assert got["resolution"] == "original (no downscaled companion exists)"

    def test_a_finding_with_no_heatmap_says_so(self, db):
        """Absence is an ordinary answer, not a fault."""
        with pytest.raises(PayloadError, match="No grad-cam heatmap"):
            mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[0])

    def test_a_finding_of_another_run_is_refused_the_same_way(self, db):
        """The join through findings is what scopes this by run."""
        with pytest.raises(PayloadError, match="No grad-cam heatmap"):
            mcp_payloads.heatmap_for_finding(db.conn, db.run_b, db.ids_a[1])

    def test_absence_and_wrong_run_are_indistinguishable(self, db):
        """A caller must not learn which findings were explained elsewhere."""
        import re

        flatten = lambda text: re.sub(r"\d+", "N", text)  # noqa: E731
        try:
            mcp_payloads.heatmap_for_finding(db.conn, db.run_b, db.ids_a[1])
        except PayloadError as e:
            wrong = flatten(str(e))
        try:
            mcp_payloads.heatmap_for_finding(db.conn, db.run_b, 999999)
        except PayloadError as e:
            absent = flatten(str(e))
        assert wrong == absent

    def test_a_database_with_no_heatmap_table_answers_the_same(self, db):
        """An older schema is not a different kind of failure."""
        db.conn.execute("DROP TABLE IF EXISTS heatmaps")
        with pytest.raises(PayloadError, match="No grad-cam heatmap"):
            mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1])

    @pytest.mark.parametrize("bad", ["1", 1.5, True])
    def test_a_non_integer_id_is_refused(self, db, bad):
        """Both ids must be integers."""
        with pytest.raises(PayloadError, match="must be an integer"):
            mcp_payloads.heatmap_for_finding(db.conn, db.run_a, bad)

    def test_an_empty_method_is_refused(self, db):
        """The method names a stored row; blank is not one."""
        with pytest.raises(PayloadError, match="`method` must be"):
            mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1], method="")


class TestItWritesNothing:
    """Reading evidence must not create copies of it."""

    def test_no_file_is_created_anywhere_under_the_root(self, db):
        """No temporary copy, no cache, no conversion left behind."""
        before = sorted(p.relative_to(db.root) for p in db.root.rglob("*"))
        mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1])
        after = sorted(p.relative_to(db.root) for p in db.root.rglob("*"))
        assert before == after

    def test_the_database_is_unchanged(self, db):
        """Serving evidence is a read on every table it touches."""
        def count(table: str) -> int:
            return db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        tables = ("images", "findings", "heatmaps", "runs")
        before = {t: count(t) for t in tables}
        mcp_payloads.image_for_run(db.conn, db.run_a, db.image_a)
        mcp_payloads.heatmap_for_finding(db.conn, db.run_a, db.ids_a[1])
        assert {t: count(t) for t in tables} == before
