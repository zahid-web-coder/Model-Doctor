"""The write API, including the guarantee that the read API stays read-only.

The first class here is the important one. This feature's whole risk is that
adding a write path quietly dissolves the property D-037 relies on, so that is
asserted directly rather than left to code review.
"""

from __future__ import annotations

import importlib
import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_doctor import config
from model_doctor.app import jobs, storage, workspace
from model_doctor.app.control import create_app


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
        """model_doctor.app.api must remain GET-only after the control API exists."""
        from model_doctor.app.api import create_app as create_reader

        methods: set[str] = set()
        for route in create_reader().routes:
            methods |= set(getattr(route, "methods", set()) or set())
        assert methods <= {"GET", "HEAD"}, f"a write route appeared: {methods}"

    def test_the_reader_still_opens_the_database_read_only(self) -> None:
        """The structural half of the guarantee, not just the route list.

        The open moved into ``service`` when the engine gained a non-HTTP
        caller, so the chain is checked rather than one function's text: the
        API's reader must go through ``service``, and ``service`` must reach
        the ``mode=ro`` opener in ``storage``.
        """
        import inspect

        from model_doctor.app import api, service, storage

        assert "service.read_only" in inspect.getsource(api.read_only)
        assert "connect_read_only" in inspect.getsource(service.read_only)
        assert "mode=ro" in inspect.getsource(storage.connect_read_only)


class TestCapabilities:
    """The UI is told what this build supports rather than guessing."""

    def test_lists_only_families_with_an_adapter(self, client: TestClient) -> None:
        """Lists only families with an adapter."""
        from model_doctor.app.detectors import SUPPORTED_FAMILIES

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

    def test_the_additive_passes_run_and_in_a_workable_order(
        self, tmp_path: Path
    ) -> None:
        """Image diagnosis and relations run, and in that order.

        The order is a requirement, not a preference: ``model_doctor.app.relations``
        takes
        its coverage thresholds from the stored image diagnosis and refuses to
        run without one. Both also come after mask-diagnosis, whose outlines
        relations measures on.
        """
        names = [n for n, _ in jobs.stage_commands(self._request("yolo", tmp_path), 1)]
        assert names.index("mask-diagnosis") < names.index("image-diagnosis")
        assert names.index("image-diagnosis") < names.index("relations")

    def test_both_families_get_the_additive_passes(self, tmp_path: Path) -> None:
        """Neither pass is family-specific, unlike explainability."""
        for detector in ("yolo", "rfdetr"):
            names = [
                n for n, _ in jobs.stage_commands(self._request(detector, tmp_path), 1)
            ]
            assert "image-diagnosis" in names, detector
            assert "relations" in names, detector

    def test_the_declared_stages_cover_what_the_worker_runs(
        self, tmp_path: Path
    ) -> None:
        """``JOB_STAGES`` is what the dashboard derives progress from.

        A stage the worker runs but the tuple omits would make a job report
        progress against a list it is not following. Explainability is the one
        conditional stage, so the tuple is a superset rather than an equality.
        """
        request = self._request("yolo", tmp_path)
        produced = [n for n, _ in jobs.stage_commands(request, None)]
        produced += [n for n, _ in jobs.stage_commands(request, 1)]
        assert set(produced) <= set(storage.JOB_STAGES)
        ordering = [s for s in storage.JOB_STAGES if s in set(produced)]
        assert ordering == produced, "JOB_STAGES disagrees with execution order"

    def test_every_stage_accepts_the_arguments_it_is_given(
        self, tmp_path: Path
    ) -> None:
        """Each stage's flags are parsed by that stage's own parser.

        The modules do not agree on a spelling — ``model_doctor.app.relations`` takes
        ``--database`` where the others take ``--db`` — so a plausible-looking
        flag is not evidence that the stage would start. Parsing with the real
        parser is.
        """
        for _, command in jobs.stage_commands(self._request("yolo", tmp_path), 1):
            if "-m" not in command:
                continue  # a script, covered by the entry-point test
            module = importlib.import_module(command[command.index("-m") + 1])
            parser = getattr(module, "build_parser", None)
            if parser is None:
                continue
            parser().parse_args(command[command.index("-m") + 2:])

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


class TestAuditRegressions:
    """Defects found by the production-readiness audit, each with its bug."""

    def test_a_run_where_every_image_failed_is_not_treated_as_success(
        self, isolated: Path
    ) -> None:
        """A checkpoint that will not load must fail the stage that used it.

        The diagnosis pass records a per-image failure and continues, which is
        right for one unreadable image and wrong when none of them worked. A
        checkpoint that will not load produces exactly that: every image errors,
        no findings are written, and the stage exits zero.

        Before this check the job carried on and the *next* stage failed, so the
        operator was told mask-diagnosis was broken when the real problem was
        the file they uploaded — and an empty run appeared in the dashboard
        looking like a genuine result.
        """
        database = isolated.parent / "empty.db"
        with storage.connect(database) as connection:
            run_id = connection.execute(
                """
                INSERT INTO runs (
                    model_path, model_sha256, dataset_yaml, split,
                    confidence_threshold, match_iou_threshold,
                    localization_iou_floor, image_size, created_at
                ) VALUES ('m.pt','sha','d.yaml','test',0.25,0.5,0.1,640,'now')
                """
            ).lastrowid
            for name in ("a.jpg", "b.jpg"):
                connection.execute(
                    """
                    INSERT INTO images (run_id, path, filename, prediction_count,
                                        truth_count, error)
                    VALUES (?, ?, ?, 0, 0, 'Could not load the checkpoint')
                    """,
                    (run_id, f"/x/{name}", name),
                )

        with pytest.raises(RuntimeError, match="Every one of the 2 image"):
            jobs.JobRunner._verify_usable(database, int(run_id or 0))

    def test_a_run_with_some_failures_is_still_usable(self, isolated: Path) -> None:
        """One unreadable image must not abandon an otherwise good run."""
        database = isolated.parent / "partial.db"
        with storage.connect(database) as connection:
            run_id = connection.execute(
                """
                INSERT INTO runs (
                    model_path, model_sha256, dataset_yaml, split,
                    confidence_threshold, match_iou_threshold,
                    localization_iou_floor, image_size, created_at
                ) VALUES ('m.pt','sha','d.yaml','test',0.25,0.5,0.1,640,'now')
                """
            ).lastrowid
            connection.execute(
                "INSERT INTO images (run_id, path, filename, prediction_count,"
                " truth_count, error) VALUES (?,'/x/a.jpg','a.jpg',2,2,NULL)",
                (run_id,),
            )
            connection.execute(
                "INSERT INTO images (run_id, path, filename, prediction_count,"
                " truth_count, error) VALUES (?,'/x/b.jpg','b.jpg',0,0,'unreadable')",
                (run_id,),
            )
        jobs.JobRunner._verify_usable(database, int(run_id or 0))  # must not raise

    def test_a_job_left_running_by_a_dead_process_is_closed_out(
        self, isolated: Path
    ) -> None:
        """A restart must not leave the UI polling a job that cannot finish.

        Job status lives in the database while the work lives in a process. When
        the process ends, nothing moves the row, so the browser polls `running`
        for as long as the tab is open.
        """
        with storage.connect() as connection:
            for token, status in (("alive", "running"), ("waiting", "queued")):
                storage.create_job(
                    connection, token=token, detector="yolo", split="test",
                    model_name="m.pt", dataset_name="ds", workspace=str(isolated),
                    image_size=640, confidence=0.25,
                )
                storage.update_job(connection, token, status=status)

            closed = storage.reconcile_stale_jobs(connection)
            assert closed == 2
            for token in ("alive", "waiting"):
                record = storage.load_job(connection, token)
                assert record is not None
                assert record.status == "failed"
                assert "stopped while this job was running" in (record.error or "")

    def test_reconciliation_leaves_finished_jobs_alone(self, isolated: Path) -> None:
        """Only in-flight jobs are dead; a finished one is a record."""
        with storage.connect() as connection:
            storage.create_job(
                connection, token="done", detector="yolo", split="test",
                model_name="m.pt", dataset_name="ds", workspace=str(isolated),
                image_size=640, confidence=0.25,
            )
            # No run_id: the foreign key correctly refuses one that does not
            # exist, which this test has no reason to create.
            storage.update_job(connection, "done", status="succeeded")
            assert storage.reconcile_stale_jobs(connection) == 0
            record = storage.load_job(connection, "done")
            assert record is not None and record.status == "succeeded"

    def test_old_workspaces_are_pruned_so_the_disk_does_not_fill(
        self, isolated: Path
    ) -> None:
        """Each analysis keeps a full dataset copy; without a bound they stack up."""
        import os
        import time

        spaces = []
        for _ in range(4):
            space = workspace.create()
            (space.dataset / "payload.bin").write_bytes(b"x" * 1000)
            spaces.append(space)
            time.sleep(0.01)
        # Make the ordering unambiguous regardless of filesystem timestamp
        # granularity, which is what this test would otherwise be at the mercy of.
        for index, space in enumerate(spaces):
            stamp = time.time() + index
            os.utime(space.root, (stamp, stamp))

        assert workspace.prune(keep=2) == 2
        surviving = {p.name for p in isolated.iterdir() if p.is_dir()}
        assert surviving == {spaces[-1].token, spaces[-2].token}

    def test_pruning_keeps_everything_when_asked_to(self, isolated: Path) -> None:
        """A generous retention must not delete anything."""
        for _ in range(3):
            workspace.create()
        assert workspace.prune(keep=10) == 0
        assert len(list(isolated.iterdir())) == 3

    def test_the_reader_may_serve_images_from_the_upload_workspace(self) -> None:
        """A self-service run keeps its images in the workspace, not datasets/.

        Found by driving the UI: the run rendered a complete diagnosis in which
        every single image was a broken box. `API_FILE_ROOTS` listed only
        `datasets/` and `results/`, so the reader answered **403** for every
        image belonging to a browser-created run — thumbnails, prediction
        overlays and heatmap tiles alike.

        The numbers were all correct, which is what made it easy to miss: the
        failure is entirely visual and invisible to an API-level test that
        checks findings and metrics.
        """
        assert config.WORKSPACE_DIR.resolve() in config.API_FILE_ROOTS, (
            "the reader cannot serve images from runs started in the browser"
        )

    def test_the_reader_may_serve_images_from_the_default_workspace_too(
        self,
    ) -> None:
        """One database accumulates runs from sessions configured differently.

        `MD_WORKSPACE_DIR` is set per session, and a run analysed under the
        default becomes unreadable the moment a later session overrides it —
        every image 403s while the diagnosis around it renders perfectly. That
        happened here across runs in one database: some images sat under
        `~/.model-doctor/workspace` and others under an override, and whichever
        did not match the reader's current setting showed as broken boxes.

        Both are this installation's own upload directory, so both are served.
        Anywhere else still has to be named in `MD_API_FILE_ROOTS`.
        """
        assert config.DEFAULT_WORKSPACE_DIR.resolve() in config.API_FILE_ROOTS, (
            "runs analysed under the default workspace cannot serve their images"
        )

    def test_permitted_roots_are_free_of_duplicates(self) -> None:
        """With no override the two workspace entries coincide.

        Harmless to serving, but the roots are quoted back in the 403 message,
        and a list that repeats itself reads as a bug in the diagnosis rather
        than as configuration.
        """
        roots = list(config.API_FILE_ROOTS)
        assert len(roots) == len(set(roots))

    def test_permitted_roots_do_not_include_the_whole_home_directory(self) -> None:
        """Widening the roots must stay narrow: the workspace, not its parent."""
        for root in config.API_FILE_ROOTS:
            assert root != Path.home(), "API_FILE_ROOTS must not contain $HOME"
            assert root != Path("/"), "API_FILE_ROOTS must not contain /"
            assert root != Path.home() / ".model-doctor", (
                "the workspace's parent is not a permitted root"
            )
