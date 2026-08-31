"""The write API, including the guarantee that the read API stays read-only.

The first class here is the important one. This feature's whole risk is that
adding a write path quietly dissolves the property D-037 relies on, so that is
asserted directly rather than left to code review.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
from app import jobs, storage
from app.control import create_app


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point workspace and database at temporary locations."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", root)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "control.db")
    return root


@pytest.fixture
def client(isolated: Path) -> TestClient:  # noqa: ARG001 - fixture dependency
    """A control API bound to the isolated locations."""
    return TestClient(create_app())


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[jobs.JobRequest]:
    """Capture submissions instead of running them."""
    captured: list[jobs.JobRequest] = []
    monkeypatch.setattr(jobs.RUNNER, "submit", captured.append)
    return captured


def _dataset_zip() -> bytes:
    """A minimal valid YOLO dataset, zipped."""
    import PIL.Image

    image = io.BytesIO()
    PIL.Image.new("RGB", (32, 32)).save(image, format="JPEG")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("ds/data.yaml", "test: test/images\nnames:\n  0: column\n")
        bundle.writestr("ds/test/images/a.jpg", image.getvalue())
        bundle.writestr("ds/test/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    return buffer.getvalue()


def _model_bytes(marker: bytes = b"ultralytics") -> bytes:
    """A file that identifies as a checkpoint of a given family."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("m/data.pkl", marker)
    return buffer.getvalue()


class TestTheReadApiStaysReadOnly:
    """The guarantee this feature was most likely to break."""

    def test_the_read_api_still_declares_no_write_route(self) -> None:
        """app.api must remain GET-only after the control API exists."""
        from app.api import create_app as create_reader

        methods: set[str] = set()
        for route in create_reader().routes:
            methods |= set(getattr(route, "methods", set()) or set())
        assert methods <= {"GET", "HEAD"}, f"a write route appeared: {methods}"

    def test_the_reader_still_opens_the_database_read_only(self) -> None:
        """The structural half of the guarantee, not just the route list."""
        import inspect

        from app import api

        assert "mode=ro" in inspect.getsource(api.read_only)


class TestCapabilities:
    """The UI is told what this build supports rather than guessing."""

    def test_lists_only_families_with_an_adapter(self, client: TestClient) -> None:
        """Lists only families with an adapter."""
        from app.detectors import SUPPORTED_FAMILIES

        body = client.get("/capabilities").json()
        assert [d["family"] for d in body["detectors"]] == list(SUPPORTED_FAMILIES)

    def test_states_explainability_honestly_per_family(
        self, client: TestClient
    ) -> None:
        """RF-DETR must never be advertised as having Grad-CAM."""
        detectors = client.get("/capabilities").json()["detectors"]
        by_family = {d["family"]: d for d in detectors}
        assert by_family["yolo"]["explainability"] is True
        assert by_family["rfdetr"]["explainability"] is False
        assert "faithfulness" in by_family["rfdetr"]["explainability_note"]


class TestUploads:
    """Uploads are validated at the boundary, and answer with the verdict."""

    def test_unknown_session_is_not_found(self, client: TestClient) -> None:
        """Unknown session is not found."""
        response = client.post(
            "/uploads/made-up/model",
            files={"file": ("m.pt", b"x", "application/octet-stream")},
        )
        assert response.status_code == 404

    def test_a_traversing_token_cannot_escape_the_workspace_root(
        self, client: TestClient
    ) -> None:
        """A token is a directory name, so it is the obvious thing to attack."""
        response = client.get("/analyses/..%2F..%2Fetc")
        assert response.status_code == 404

    def test_model_upload_reports_the_detected_family(self, client: TestClient) -> None:
        """Model upload reports the detected family."""
        token = client.post("/uploads").json()["token"]
        response = client.post(
            f"/uploads/{token}/model",
            files={"file": ("best.pt", _model_bytes(), "application/octet-stream")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["facts"]["detected_family"] == "yolo"

    def test_model_upload_refuses_an_unlisted_type(self, client: TestClient) -> None:
        """Model upload refuses an unlisted type."""
        token = client.post("/uploads").json()["token"]
        response = client.post(
            f"/uploads/{token}/model",
            files={"file": ("run.sh", b"#!/bin/sh", "text/plain")},
        )
        assert response.status_code == 400
        assert "not a supported file type" in response.json()["detail"]

    def test_dataset_upload_validates_the_unpacked_tree(
        self, client: TestClient
    ) -> None:
        """Dataset upload validates the unpacked tree."""
        token = client.post("/uploads").json()["token"]
        response = client.post(
            f"/uploads/{token}/dataset",
            files={"file": ("ds.zip", _dataset_zip(), "application/zip")},
            data={"split": "test"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True
        assert body["facts"]["class_names"] == {"0": "column"}

    def test_dataset_upload_refuses_a_hostile_archive(self, client: TestClient) -> None:
        """Dataset upload refuses a hostile archive."""
        token = client.post("/uploads").json()["token"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("../escape.txt", b"pwned")
        response = client.post(
            f"/uploads/{token}/dataset",
            files={"file": ("evil.zip", buffer.getvalue(), "application/zip")},
        )
        assert response.status_code == 400
        assert "outside the workspace" in response.json()["detail"]


class TestStartingAnalyses:
    """Starting a run validates again, records a job, and returns immediately."""

    def _prepare(self, client: TestClient) -> tuple[str, str, str]:
        token = client.post("/uploads").json()["token"]
        model = client.post(
            f"/uploads/{token}/model",
            files={"file": ("best.pt", _model_bytes(), "application/octet-stream")},
        ).json()
        dataset = client.post(
            f"/uploads/{token}/dataset",
            files={"file": ("ds.zip", _dataset_zip(), "application/zip")},
        ).json()
        return token, model["path"], dataset["facts"]["data_yaml"]

    def test_queues_a_job_without_running_it(
        self, client: TestClient, queued: list[jobs.JobRequest]
    ) -> None:
        """Queues a job without running it."""
        token, model_path, data_yaml = self._prepare(client)
        response = client.post("/analyses", json={
            "token": token, "detector": "yolo", "model_path": model_path,
            "data_yaml": data_yaml, "split": "test",
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "queued"
        assert body["run_id"] is None
        assert len(queued) == 1

    def test_refuses_an_unsupported_detector(self, client: TestClient) -> None:
        """Refuses an unsupported detector."""
        token, model_path, data_yaml = self._prepare(client)
        response = client.post("/analyses", json={
            "token": token, "detector": "detectron", "model_path": model_path,
            "data_yaml": data_yaml,
        })
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "not a detector family this build implements" in detail

    def test_refuses_a_path_outside_the_upload_session(
        self, client: TestClient
    ) -> None:
        """A caller must not be able to point a run at an arbitrary file."""
        token, _, data_yaml = self._prepare(client)
        response = client.post("/analyses", json={
            "token": token, "detector": "yolo",
            "model_path": "/etc/passwd", "data_yaml": data_yaml,
        })
        assert response.status_code == 400
        assert "not part of this upload session" in response.json()["detail"]

    @pytest.mark.usefixtures("queued")
    def test_rfdetr_is_not_offered_an_explainability_stage(
        self, client: TestClient
    ) -> None:
        """The progress bar must not promise a step that will not run."""
        token = client.post("/uploads").json()["token"]
        model = client.post(
            f"/uploads/{token}/model",
            files={
                "file": ("r.pt", _model_bytes(b"rfdetr"), "application/octet-stream")
            },
        ).json()
        dataset = client.post(
            f"/uploads/{token}/dataset",
            files={"file": ("ds.zip", _dataset_zip(), "application/zip")},
        ).json()
        body = client.post("/analyses", json={
            "token": token, "detector": "rfdetr", "model_path": model["path"],
            "data_yaml": dataset["facts"]["data_yaml"], "split": "test",
        }).json()
        assert "explainability" not in body["stages"]
        assert body["explainability_supported"] is False

    @pytest.mark.usefixtures("queued")
    def test_status_is_pollable_and_404s_for_an_unknown_token(
        self, client: TestClient
    ) -> None:
        """Status is pollable and 404s for an unknown token."""
        token, model_path, data_yaml = self._prepare(client)
        client.post("/analyses", json={
            "token": token, "detector": "yolo", "model_path": model_path,
            "data_yaml": data_yaml,
        })
        assert client.get(f"/analyses/{token}").json()["token"] == token
        assert client.get("/analyses/nope").status_code == 404


class TestStageCommands:
    """The worker runs the CLI rather than a second copy of the pipeline."""

    def _request(self, detector: str, tmp_path: Path) -> jobs.JobRequest:
        return jobs.JobRequest(
            token="t", detector=detector, model_path=tmp_path / "m.pt",
            data_yaml=tmp_path / "data.yaml", split="test", image_size=640,
            confidence=0.25, database=tmp_path / "d.db", workspace=tmp_path,
        )

    def test_first_stage_creates_the_run(self, tmp_path: Path) -> None:
        """First stage creates the run."""
        stages = jobs.stage_commands(self._request("yolo", tmp_path), None)
        assert [name for name, _ in stages] == ["inference+diagnosis"]
        assert "--save" in stages[0][1]

    def test_yolo_gets_explainability_and_rfdetr_does_not(self, tmp_path: Path) -> None:
        """Yolo gets explainability and rfdetr does not."""
        yolo = [n for n, _ in jobs.stage_commands(self._request("yolo", tmp_path), 1)]
        rfdetr = [
            n for n, _ in jobs.stage_commands(self._request("rfdetr", tmp_path), 1)
        ]
        assert "explainability" in yolo
        assert "explainability" not in rfdetr

    def test_every_command_invokes_an_existing_entry_point(
        self, tmp_path: Path
    ) -> None:
        """Guards against a stage naming a module that was renamed away."""
        for _, command in jobs.stage_commands(self._request("yolo", tmp_path), 1):
            if "-m" in command:
                module = command[command.index("-m") + 1]
                source = (
                    config.PROJECT_ROOT / Path(*module.split("."))
                ).with_suffix(".py")
                assert source.is_file(), module
            else:
                script = next(c for c in command if c.endswith(".py"))
                assert (config.PROJECT_ROOT / script).is_file(), script

    def test_the_dataset_reaches_a_stage_through_the_environment(
        self, tmp_path: Path
    ) -> None:
        """The dataset reaches a stage through the environment."""
        environment = jobs.stage_environment(self._request("yolo", tmp_path))
        assert environment["MD_DATA_YAML"] == str(tmp_path / "data.yaml")
        assert environment["MD_DB_PATH"] == str(tmp_path / "d.db")
        assert environment["MD_DETECTOR"] == "yolo"


class TestJobRecords:
    """A job is observable from the moment it is accepted."""

    def test_a_job_exists_before_any_work_starts(self, isolated: Path) -> None:
        """A job exists before any work starts."""
        with storage.connect() as connection:
            storage.create_job(
                connection, token="tok", detector="yolo", split="test",
                model_name="m.pt", dataset_name="ds", workspace=str(isolated),
                image_size=640, confidence=0.25,
            )
            record = storage.load_job(connection, "tok")
        assert record is not None
        assert record.status == "queued"
        assert record.run_id is None

    def test_immutable_columns_cannot_be_rewritten(self, isolated: Path) -> None:
        """What was asked for must not be editable by the thing running it."""
        with storage.connect() as connection:
            storage.create_job(
                connection, token="tok", detector="yolo", split="test",
                model_name="m.pt", dataset_name="ds", workspace=str(isolated),
                image_size=640, confidence=0.25,
            )
            with pytest.raises(ValueError, match="Not writable"):
                storage.update_job(connection, "tok", workspace="/elsewhere")
