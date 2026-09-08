"""Isolated storage for files a browser uploaded.

**Everything here treats its input as hostile.** These files arrive over HTTP
from outside the process, and the difference between a dataset and an attack is
a filename. The rules this module enforces:

* Uploads live in a workspace directory **outside the source tree**, so a
  malicious archive cannot reach `app/`, `models/`, `datasets/` or the
  database. The default is under the user's home, not the repository.
* Each job gets its own directory. Two jobs cannot see or overwrite each
  other's inputs, and deleting one deletes everything it brought.
* Archive members are checked **before** extraction, not after. A member whose
  resolved destination escapes the target directory aborts the whole archive
  rather than being skipped, because an archive containing one such entry is
  not a dataset with a mistake in it.
* Nothing extracted is ever executed, and nothing is added to `sys.path`.

Size limits exist to stop a request exhausting the disk, and are configurable
because a legitimate detection dataset is genuinely large.
"""

from __future__ import annotations

import re
import secrets
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from model_doctor import config
from model_doctor.utils.exceptions import ModelDoctorError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Extensions accepted for each kind of upload. Deliberately a allowlist: the
# set of things that are definitely fine is knowable, the set of things that
# are definitely dangerous is not.
MODEL_SUFFIXES: frozenset[str] = frozenset({".pt", ".pth"})
ARCHIVE_SUFFIXES: frozenset[str] = frozenset({".zip", ".tar", ".tar.gz", ".tgz"})

# A filename is reduced to this before it touches a filesystem. Anything else
# — separators, traversal, control characters, leading dots — is replaced.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class WorkspaceError(ModelDoctorError):
    """An upload cannot be stored or unpacked safely, with the reason why."""


@dataclass(frozen=True)
class Workspace:
    """One job's private directory tree."""

    token: str
    root: Path

    @property
    def models(self) -> Path:
        """Where uploaded weights are written."""
        return self.root / "model"

    @property
    def dataset(self) -> Path:
        """Where an uploaded dataset is unpacked."""
        return self.root / "dataset"

    @property
    def logs(self) -> Path:
        """Where each stage's captured output is written."""
        return self.root / "logs"


def new_token() -> str:
    """Return an unguessable job token.

    A job token addresses a workspace and is handed to a browser, so it is
    generated the way a session identifier would be rather than from a counter.
    """
    return secrets.token_urlsafe(16)


def safe_name(raw: str, *, fallback: str = "upload") -> str:
    """Reduce a client-supplied filename to something safe to write.

    The name is taken for its *suffix and readability only*; it never
    determines a directory. Traversal sequences, separators and control
    characters cannot survive this, so a caller cannot be tricked by a
    filename alone.
    """
    stem = Path(raw).name  # discards any directory component the client sent
    cleaned = _SAFE_NAME.sub("_", stem).lstrip(".")
    return cleaned[:120] or fallback


def create(token: str | None = None) -> Workspace:
    """Create an empty workspace and return it."""
    resolved = token or new_token()
    root = (config.WORKSPACE_DIR / resolved).resolve()
    if config.WORKSPACE_DIR.resolve() not in root.parents:
        # Only reachable if the token contained separators, which new_token
        # never produces — but this is the check that makes that guaranteed
        # rather than assumed.
        raise WorkspaceError("Refusing to create a workspace outside the root.")
    for directory in (root / "model", root / "dataset", root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Created workspace %s", root)
    return Workspace(token=resolved, root=root)


def destroy(workspace: Workspace) -> None:
    """Remove a workspace and everything in it, if it is one of ours."""
    root = workspace.root.resolve()
    if config.WORKSPACE_DIR.resolve() not in root.parents:
        raise WorkspaceError("Refusing to delete a path outside the workspace root.")
    shutil.rmtree(root, ignore_errors=True)


def prune(keep: int) -> int:
    """Delete all but the newest ``keep`` workspaces, and say how many went.

    **Uploads are copies, and copies accumulate.** Each analysis keeps its own
    dataset — the demo dataset alone is hundreds of megabytes — so without a
    bound the workspace root grows by the size of a dataset per run until the
    disk fills. Nothing downstream needs the upload once the run is saved: the
    findings, heatmaps and metrics are in the database and the results
    directory, and the images are read during the run, not after it.

    Newest-first rather than oldest-first, and by modification time, so a
    workspace still being uploaded into is never the one chosen for deletion.

    Args:
        keep: How many workspaces to retain. Zero removes all of them.

    Returns:
        The number of workspaces removed.
    """
    root = config.WORKSPACE_DIR
    if not root.is_dir():
        return 0
    spaces = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for stale in spaces[max(0, keep) :]:
        shutil.rmtree(stale, ignore_errors=True)
        removed += 1
    if removed:
        logger.info("Pruned %d old workspace(s) from %s", removed, root)
    return removed


def _within(base: Path, candidate: Path) -> bool:
    """Report whether ``candidate`` resolves inside ``base``."""
    try:
        candidate.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True


def store_upload(
    destination: Path,
    filename: str,
    data: bytes,
    *,
    allowed_suffixes: frozenset[str],
    max_bytes: int,
) -> Path:
    """Write one uploaded file into ``destination`` and return its path.

    Args:
        destination: Directory to write into. Must already exist.
        filename: The client's name for the file. Used for its suffix only.
        data: The bytes received.
        allowed_suffixes: Suffixes this upload kind accepts.
        max_bytes: Reject anything larger.

    Raises:
        WorkspaceError: The file is empty, too large, or has a suffix this
            kind of upload does not accept.
    """
    if not data:
        raise WorkspaceError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise WorkspaceError(
            f"That file is {len(data) / 1e6:.0f} MB, above the "
            f"{max_bytes / 1e6:.0f} MB limit for this upload."
        )

    name = safe_name(filename)
    suffix = "".join(Path(name).suffixes[-2:]).lower()
    if suffix not in allowed_suffixes:
        suffix = Path(name).suffix.lower()
    if suffix not in allowed_suffixes:
        raise WorkspaceError(
            f"{name!r} is not a supported file type here. "
            f"Accepted: {', '.join(sorted(allowed_suffixes))}."
        )

    target = destination / name
    if not _within(destination, target):
        raise WorkspaceError("Refusing to write outside the workspace.")
    target.write_bytes(data)
    return target


def extract_archive(archive: Path, destination: Path, *, max_bytes: int) -> Path:
    """Unpack an archive into ``destination``, refusing anything unsafe.

    Every member is inspected before a single byte is written. An archive
    containing a traversal path, an absolute path, a link, or more data than
    the limit allows is rejected **whole** — a partially extracted hostile
    archive is worse than none, and an archive with one such member is not
    something to salvage.

    Returns:
        The directory containing the extracted tree. When the archive holds a
        single top-level directory, that directory is returned, because a
        dataset zipped with its own folder is the common case and descending
        into it is what the caller means.

    Raises:
        WorkspaceError: The archive is unreadable or contains an unsafe member.
    """
    destination.mkdir(parents=True, exist_ok=True)
    total = 0

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                total += info.file_size
                _check_member(info.filename, destination, total, max_bytes)
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if member.issym() or member.islnk():
                    raise WorkspaceError(
                        f"{member.name!r} is a link. Links in an uploaded "
                        "archive can point anywhere on this machine, so the "
                        "archive is refused."
                    )
                total += member.size
                _check_member(member.name, destination, total, max_bytes)
            # `data` filter (Python 3.12+) additionally strips ownership and
            # permission bits; harmless and ignored on older runtimes.
            try:
                bundle.extractall(destination, filter="data")
            except TypeError:  # pragma: no cover - older Python
                bundle.extractall(destination)
    else:
        raise WorkspaceError(
            "That file is not a readable .zip or .tar archive."
        )

    entries = [p for p in destination.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


def _check_member(
    name: str, destination: Path, running_total: int, max_bytes: int
) -> None:
    """Raise unless one archive member is safe to extract."""
    if running_total > max_bytes:
        raise WorkspaceError(
            f"The archive expands to more than {max_bytes / 1e6:.0f} MB."
        )
    if name.startswith("/") or Path(name).is_absolute():
        raise WorkspaceError(f"{name!r} is an absolute path. Archive refused.")
    target = destination / name
    if not _within(destination, target):
        raise WorkspaceError(
            f"{name!r} would extract outside the workspace. Archive refused."
        )
