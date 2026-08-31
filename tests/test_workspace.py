"""Uploads are untrusted input, and these are the tests that say so.

Most of this file is adversarial. A dataset and an attack differ by a
filename, so the cases that matter are the malformed ones: the archive that
tries to write outside its directory, the name that is really a path, the file
that claims to be a checkpoint. Each is asserted to be *refused*, not
sanitised-and-continued, because an archive containing a traversal entry is not
a dataset with a typo in it.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

import config
from app import workspace


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the workspace root at a temporary directory."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", root)
    return root


class TestSafeName:
    """A client-supplied filename never determines a directory."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "passwd"),
            ("a/b/../c.pt", "c.pt"),
            ("/absolute/path.pt", "path.pt"),
            ("..", "upload"),
            ("....//....//x.pt", "x.pt"),
            ("model .pt", "model_.pt"),
            ("", "upload"),
        ],
    )
    def test_reduces_hostile_names(self, raw: str, expected: str) -> None:
        """Traversal and separators cannot survive."""
        assert workspace.safe_name(raw) == expected

    def test_never_returns_a_path_separator(self) -> None:
        """Whatever the input, the result addresses one file in one directory."""
        for raw in ("a/b/c", "..\\..\\win.pt", "x\x00y.pt", "\n/etc/hosts"):
            assert "/" not in workspace.safe_name(raw)
            assert "\\" not in workspace.safe_name(raw)


@pytest.mark.usefixtures("isolated")
class TestWorkspaceLifecycle:
    """Workspaces are private, and live outside the source tree."""

    def test_created_under_the_configured_root(self, isolated: Path) -> None:
        """Created under the configured root."""
        space = workspace.create()
        assert isolated.resolve() in space.root.parents
        assert space.models.is_dir() and space.dataset.is_dir()

    def test_is_not_inside_the_repository_by_default(self) -> None:
        """The default must never place untrusted files beside the source."""
        assert config.PROJECT_ROOT not in config.WORKSPACE_DIR.parents
        assert config.WORKSPACE_DIR != config.PROJECT_ROOT

    def test_two_jobs_cannot_see_each_other(self) -> None:
        """Two jobs cannot see each other."""
        first, second = workspace.create(), workspace.create()
        assert first.root != second.root

    def test_destroy_refuses_a_path_outside_the_root(
        self, tmp_path: Path
    ) -> None:
        """The delete guard is what stops a bad token removing something else."""
        outsider = workspace.Workspace(token="x", root=tmp_path / "not-ours")
        with pytest.raises(workspace.WorkspaceError):
            workspace.destroy(outsider)


@pytest.mark.usefixtures("isolated")
class TestStoreUpload:
    """Type and size are checked before anything is written."""

    def test_rejects_an_unlisted_suffix(self) -> None:
        """Rejects an unlisted suffix."""
        space = workspace.create()
        with pytest.raises(workspace.WorkspaceError, match="not a supported file type"):
            workspace.store_upload(
                space.models, "payload.sh", b"#!/bin/sh\n",
                allowed_suffixes=workspace.MODEL_SUFFIXES, max_bytes=1000,
            )

    def test_rejects_a_file_over_the_limit(self) -> None:
        """Rejects a file over the limit."""
        space = workspace.create()
        with pytest.raises(workspace.WorkspaceError, match="above the"):
            workspace.store_upload(
                space.models, "big.pt", b"x" * 50,
                allowed_suffixes=workspace.MODEL_SUFFIXES, max_bytes=10,
            )

    def test_rejects_an_empty_file(self) -> None:
        """Rejects an empty file."""
        space = workspace.create()
        with pytest.raises(workspace.WorkspaceError, match="empty"):
            workspace.store_upload(
                space.models, "e.pt", b"",
                allowed_suffixes=workspace.MODEL_SUFFIXES, max_bytes=10,
            )

    def test_a_traversal_filename_lands_inside_the_workspace(
        self, isolated: Path
    ) -> None:
        """The written file is in the workspace regardless of what was asked for."""
        space = workspace.create()
        written = workspace.store_upload(
            space.models, "../../../../evil.pt", b"weights",
            allowed_suffixes=workspace.MODEL_SUFFIXES, max_bytes=1000,
        )
        assert space.models.resolve() == written.parent.resolve()
        assert not (isolated.parent / "evil.pt").exists()


def _zip_with(entries: dict[str, bytes]) -> bytes:
    """Build a zip archive in memory from name -> content."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.usefixtures("isolated")
class TestExtractArchive:
    """The archive tests are the ones that matter most in this file."""

    def test_extracts_an_ordinary_archive(self, tmp_path: Path) -> None:
        """Extracts an ordinary archive."""
        archive = tmp_path / "ds.zip"
        archive.write_bytes(_zip_with({"data.yaml": b"names: [a]\n"}))
        root = workspace.extract_archive(
            archive, tmp_path / "out", max_bytes=10_000
        )
        assert (root / "data.yaml").is_file()

    def test_descends_into_a_single_top_level_directory(
        self, tmp_path: Path
    ) -> None:
        """A dataset zipped with its own folder is the common case."""
        archive = tmp_path / "ds.zip"
        archive.write_bytes(_zip_with({"mydata/data.yaml": b"names: [a]\n"}))
        root = workspace.extract_archive(archive, tmp_path / "out", max_bytes=10_000)
        assert root.name == "mydata"
        assert (root / "data.yaml").is_file()

    def test_refuses_a_traversal_member(self, tmp_path: Path) -> None:
        """The whole archive is refused, not the one entry."""
        archive = tmp_path / "evil.zip"
        archive.write_bytes(
            _zip_with({"ok.txt": b"fine", "../../escaped.txt": b"pwned"})
        )
        destination = tmp_path / "out"
        with pytest.raises(workspace.WorkspaceError, match="outside the workspace"):
            workspace.extract_archive(archive, destination, max_bytes=10_000)
        assert not (tmp_path.parent / "escaped.txt").exists()
        # Nothing was written, including the member that was individually fine.
        assert not (destination / "ok.txt").exists()

    def test_refuses_an_absolute_member(self, tmp_path: Path) -> None:
        """Refuses an absolute member."""
        archive = tmp_path / "abs.zip"
        archive.write_bytes(_zip_with({"/tmp/owned.txt": b"x"}))
        with pytest.raises(workspace.WorkspaceError):
            workspace.extract_archive(archive, tmp_path / "out", max_bytes=10_000)

    def test_refuses_an_archive_that_expands_past_the_limit(
        self, tmp_path: Path
    ) -> None:
        """A zip bomb is refused on declared size, before it is written."""
        archive = tmp_path / "bomb.zip"
        archive.write_bytes(_zip_with({"big.bin": b"0" * 100_000}))
        with pytest.raises(workspace.WorkspaceError, match="expands to more than"):
            workspace.extract_archive(archive, tmp_path / "out", max_bytes=1_000)

    def test_refuses_a_tar_containing_a_symlink(
        self, tmp_path: Path
    ) -> None:
        """A link in an upload can point anywhere on the host."""
        archive = tmp_path / "link.tar"
        with tarfile.open(archive, "w") as bundle:
            link = tarfile.TarInfo("shortcut")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            bundle.addfile(link)
        with pytest.raises(workspace.WorkspaceError, match="is a link"):
            workspace.extract_archive(archive, tmp_path / "out", max_bytes=10_000)

    def test_refuses_something_that_is_not_an_archive(
        self, tmp_path: Path
    ) -> None:
        """Refuses something that is not an archive."""
        plain = tmp_path / "notes.zip"
        plain.write_bytes(b"this is not a zip")
        with pytest.raises(workspace.WorkspaceError, match="not a readable"):
            workspace.extract_archive(plain, tmp_path / "out", max_bytes=10_000)
