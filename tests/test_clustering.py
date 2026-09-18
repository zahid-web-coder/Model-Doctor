"""Tests for failure grouping and similarity retrieval.

The grouping rule is tested as a pure function wherever possible: it is the
part that decides what an engineer is shown, and it should not need a database
to pin down.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.clustering import (
    DISCRIMINATING_METHOD,
    FACTOR_SIGNATURE_METHOD,
    UNEXPLAINED_LABEL,
    FailureGroupingError,
    group_by_factor_signature,
    group_run,
    signature_for,
)
from model_doctor.app.similarity import (
    Neighbour,
    SimilarityError,
    cosine_similarity,
    nearest_neighbours,
)
from model_doctor.app.storage import RunContext


# ---------------------------------------------------------------------------
# The grouping rule, with no database involved
# ---------------------------------------------------------------------------
def test_signature_sorts_factors_so_order_cannot_change_membership() -> None:
    """The same factors in any order must produce the same group."""
    assert signature_for(["crowding", "blur"]) == signature_for(["blur", "crowding"])
    assert signature_for(["crowding", "blur"]) == "blur + crowding"


def test_signature_collapses_duplicates() -> None:
    """A repeated factor must not produce 'blur + blur'."""
    assert signature_for(["blur", "blur"]) == "blur"


def test_signature_of_nothing_is_the_unexplained_label() -> None:
    """A failure with no attributed factor is named, not blank."""
    assert signature_for([]) == UNEXPLAINED_LABEL


def test_every_failure_lands_in_exactly_one_group() -> None:
    """Group sizes must sum to the failure count, or the report lies."""
    failures = [1, 2, 3, 4, 5]
    factors = {1: ["blur"], 2: ["blur"], 3: ["blur", "crowding"], 4: ["crowding"]}

    groups = group_by_factor_signature(failures, factors)

    assert sum(len(members) for members in groups.values()) == len(failures)
    assigned = [fid for members in groups.values() for fid in members]
    assert sorted(assigned) == failures


def test_failures_without_factors_become_the_unexplained_group() -> None:
    """Finding 5 is absent from the factor mapping, not skipped."""
    groups = group_by_factor_signature([1, 5], {1: ["blur"]})

    assert groups[UNEXPLAINED_LABEL] == [5]
    assert groups["blur"] == [1]


def test_groups_are_ordered_largest_first() -> None:
    """A consumer that does not sort should still see the biggest problem first."""
    factors = {1: ["blur"], 2: ["crowding"], 3: ["crowding"], 4: ["crowding"]}

    labels = list(group_by_factor_signature([1, 2, 3, 4], factors))

    assert labels[0] == "crowding"


def test_members_are_sorted_within_a_group() -> None:
    """Membership order must be stable regardless of input order."""
    factors = {1: ["blur"], 2: ["blur"], 3: ["blur"]}

    groups = group_by_factor_signature([3, 1, 2], factors)

    assert groups["blur"] == [1, 2, 3]


def test_grouping_is_deterministic() -> None:
    """Running twice on the same input must give an identical partition."""
    failures = [4, 1, 3, 2]
    factors = {1: ["blur", "crowding"], 2: ["crowding", "blur"], 3: ["low_light"]}

    assert group_by_factor_signature(failures, factors) == group_by_factor_signature(
        failures, factors
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _saved_run(database: Path) -> int:
    """Create a run with four failures and attributed factors."""
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="model.pt",
                model_sha256="abc123",
                dataset_yaml="data.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?, 'a.jpg', 'a.jpg', 10, 10, 1, 1)",
            (run_id,),
        )
        image_id = connection.execute("SELECT id FROM images").fetchone()["id"]
        for outcome, class_name in (
            ("false_negative", "door"),
            ("false_negative", "door"),
            ("false_positive", "door_frame"),
            ("correct", "door"),
        ):
            connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name) "
                "VALUES (?, ?, ?, ?)",
                (run_id, image_id, outcome, class_name),
            )
        # Findings 1 and 2 share a signature; 3 has none and is unexplained.
        storage.save_root_causes(
            connection,
            run_id,
            [
                (1, "blur", 0.9, "Laplacian variance 4.5 < 100"),
                (1, "crowding", 0.5, "overlaps 2 annotations"),
                (2, "crowding", 0.5, "overlaps 2 annotations"),
                (2, "blur", 0.9, "Laplacian variance 5.1 < 100"),
            ],
        )
    return run_id


@pytest.fixture
def grouped_database(tmp_path: Path) -> Path:
    """Provide a database with one run whose failures have been grouped."""
    database = tmp_path / "grouping.db"
    run_id = _saved_run(database)
    with storage.connect(database) as connection:
        group_run(connection, run_id)
    return database


def test_grouping_covers_every_failure_and_excludes_correct_findings(
    tmp_path: Path,
) -> None:
    """Correct findings are not failures and must not be grouped."""
    database = tmp_path / "grouping.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        report = group_run(connection, run_id)

    assert report.failure_count == 3
    assert report.grouped_count == 3
    assert report.unexplained_count == 1
    assert report.largest_label == "blur + crowding"
    assert report.largest_size == 2


def test_stored_size_matches_actual_membership(grouped_database: Path) -> None:
    """A size column that disagrees with its members would silently mislead."""
    with storage.connect(grouped_database) as connection:
        for group in storage.load_clusters(connection, 1):
            members = storage.load_cluster_members(connection, group.id)
            assert group.size == len(members)


def test_regrouping_replaces_rather_than_accumulates(grouped_database: Path) -> None:
    """Grouping is derived data: recomputing corrects, it does not append."""
    with storage.connect(grouped_database) as connection:
        before = storage.load_clusters(connection, 1)
        group_run(connection, 1)
        after = storage.load_clusters(connection, 1)

        assert len(after) == len(before)
        member_rows = connection.execute(
            "SELECT COUNT(*) AS n FROM cluster_members"
        ).fetchone()["n"]
        assert member_rows == 3


def test_a_second_method_does_not_disturb_the_first(grouped_database: Path) -> None:
    """`method` is the seam that lets another approach be added later."""
    with storage.connect(grouped_database) as connection:
        storage.save_clusters(connection, 1, "kmeans", [("cluster-0", [1, 2, 3])])

        signature_groups = storage.load_clusters(
            connection, 1, FACTOR_SIGNATURE_METHOD
        )
        kmeans_groups = storage.load_clusters(connection, 1, "kmeans")

        assert len(signature_groups) == 2
        assert len(kmeans_groups) == 1


def test_empty_groups_are_not_stored(tmp_path: Path) -> None:
    """A group nothing belongs to would be a row that explains nothing."""
    database = tmp_path / "empty.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        written = storage.save_clusters(
            connection, run_id, "test-method", [("empty", []), ("real", [1])]
        )

        assert written == 1
        assert len(storage.load_clusters(connection, run_id, "test-method")) == 1


def test_deleting_a_run_removes_its_groups(grouped_database: Path) -> None:
    """Cascade must reach cluster_members, or orphan rows accumulate."""
    with storage.connect(grouped_database) as connection:
        connection.execute("DELETE FROM runs WHERE id = 1")

        assert connection.execute("SELECT COUNT(*) AS n FROM clusters").fetchone()[
            "n"
        ] == 0
        assert connection.execute(
            "SELECT COUNT(*) AS n FROM cluster_members"
        ).fetchone()["n"] == 0


def test_grouping_a_run_with_no_failures_is_an_actionable_error(
    tmp_path: Path,
) -> None:
    """An empty run explains itself rather than producing an empty report."""
    database = tmp_path / "empty_run.db"
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="a",
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        with pytest.raises(FailureGroupingError, match="no failures to group"):
            group_run(connection, run_id)


def test_run_without_root_causes_groups_everything_as_unexplained(
    tmp_path: Path,
) -> None:
    """Grouping reads attribution; it must not silently invent it."""
    database = tmp_path / "no_causes.db"
    run_id = _saved_run(database)
    with storage.connect(database) as connection:
        connection.execute("DELETE FROM root_causes")
        report = group_run(connection, run_id)

    assert report.group_count == 1
    assert report.largest_label == UNEXPLAINED_LABEL
    assert report.unexplained_count == 3


def test_schema_version_is_recorded(grouped_database: Path) -> None:
    """Consumers detect a mismatch from this number."""
    with storage.connect(grouped_database) as connection:
        version = connection.execute("SELECT version FROM schema_info").fetchone()[
            "version"
        ]
    assert version == storage.SCHEMA_VERSION


def test_existing_tables_are_unchanged_by_this_milestone(
    grouped_database: Path,
) -> None:
    """D-020: findings is a published contract and must keep its exact shape."""
    with storage.connect(grouped_database) as connection:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(findings)").fetchall()
        ]

    assert columns == [
        "id",
        "run_id",
        "image_id",
        "outcome",
        "class_id",
        "class_name",
        "confidence",
        "iou",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "truth_x1",
        "truth_y1",
        "truth_x2",
        "truth_y2",
        "truth_polygon",
    ]


# ---------------------------------------------------------------------------
# Similarity retrieval
# ---------------------------------------------------------------------------
def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    """The self-similarity case anchors the scale."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    """Perpendicular vectors share no direction."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_ignores_magnitude() -> None:
    """Direction is what similarity means; length must not affect it."""
    assert cosine_similarity([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_a_zero_vector_is_zero() -> None:
    """A zero vector points nowhere; dividing by its norm would raise."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_comparing_different_lengths_names_the_real_cause() -> None:
    """Mismatched dimensions mean two encoders, not a maths error."""
    with pytest.raises(SimilarityError, match="different encoders"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def _database_with_embeddings(tmp_path: Path) -> Path:
    """Provide a run whose failures carry deliberately positioned vectors."""
    database = tmp_path / "similarity.db"
    run_id = _saved_run(database)
    with storage.connect(database) as connection:
        storage.save_embeddings(
            connection,
            run_id,
            "test-encoder",
            [
                (1, [1.0, 0.0, 0.0]),
                (2, [0.9, 0.1, 0.0]),  # close to finding 1
                (3, [0.0, 0.0, 1.0]),  # far from finding 1
            ],
        )
    return database


def test_nearest_neighbours_ranks_the_closest_first(tmp_path: Path) -> None:
    """Finding 2 was placed near finding 1 and must outrank finding 3."""
    database = _database_with_embeddings(tmp_path)

    with storage.connect(database) as connection:
        neighbours = nearest_neighbours(connection, 1, finding_id=1)

    assert [n.finding_id for n in neighbours] == [2, 3]
    assert neighbours[0].similarity > neighbours[1].similarity


def test_nearest_neighbours_excludes_the_query_itself(tmp_path: Path) -> None:
    """A finding is trivially its own nearest neighbour, which helps nobody."""
    database = _database_with_embeddings(tmp_path)

    with storage.connect(database) as connection:
        neighbours = nearest_neighbours(connection, 1, finding_id=1)

    assert all(n.finding_id != 1 for n in neighbours)


def test_nearest_neighbours_respects_the_limit(tmp_path: Path) -> None:
    """The caller controls how many results come back."""
    database = _database_with_embeddings(tmp_path)

    with storage.connect(database) as connection:
        neighbours = nearest_neighbours(connection, 1, finding_id=1, limit=1)

    assert len(neighbours) == 1


def test_nearest_neighbours_carries_labels_for_display(tmp_path: Path) -> None:
    """A caller should not need a second query to say what a neighbour is."""
    database = _database_with_embeddings(tmp_path)

    with storage.connect(database) as connection:
        neighbours = nearest_neighbours(connection, 1, finding_id=1)

    assert isinstance(neighbours[0], Neighbour)
    assert neighbours[0].outcome
    assert neighbours[0].class_name


def test_querying_a_finding_without_an_embedding_explains_why(
    tmp_path: Path,
) -> None:
    """The likely cause — a region too small to extract — is named."""
    database = _database_with_embeddings(tmp_path)

    with (
        storage.connect(database) as connection,
        pytest.raises(SimilarityError, match="no embedding"),
    ):
        nearest_neighbours(connection, 1, finding_id=999)


def test_querying_a_run_without_embeddings_gives_the_command_to_fix_it(
    tmp_path: Path,
) -> None:
    """An actionable error beats an empty list that looks like 'no matches'."""
    database = tmp_path / "no_embeddings.db"
    run_id = _saved_run(database)

    with (
        storage.connect(database) as connection,
        pytest.raises(SimilarityError, match="model_doctor.app.features"),
    ):
        nearest_neighbours(connection, run_id, finding_id=1)


def test_embeddings_table_is_untouched_by_this_milestone(tmp_path: Path) -> None:
    """Retrieval reads embeddings; it must not reshape or repurpose them."""
    database = _database_with_embeddings(tmp_path)

    with storage.connect(database) as connection:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(embeddings)").fetchall()
        ]

    assert columns == [
        "id",
        "finding_id",
        "run_id",
        "model_name",
        "dimensions",
        "vector",
    ]


def test_foreign_keys_are_enforced_for_group_members(tmp_path: Path) -> None:
    """A member pointing at no finding would corrupt every group count."""
    database = tmp_path / "fk.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        cursor = connection.execute(
            "INSERT INTO clusters (run_id, method, label, size) VALUES (?, ?, ?, ?)",
            (run_id, "test", "label", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO cluster_members (cluster_id, finding_id) VALUES (?, ?)",
                (int(cursor.lastrowid), 9999),
            )


# ---------------------------------------------------------------------------
# Discriminating groups (D-033)
# ---------------------------------------------------------------------------
def test_allowed_factors_restricts_which_factors_split_failures(
    tmp_path: Path,
) -> None:
    """Findings 1 and 2 carry blur + crowding; allowing only blur merges them."""
    database = tmp_path / "restricted.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"blur"},
        )
        labels = {
            group.label: group.size
            for group in storage.load_clusters(
                connection, run_id, DISCRIMINATING_METHOD
            )
        }

    assert labels == {"blur": 2, UNEXPLAINED_LABEL: 1}


def test_a_disallowed_factor_does_not_drop_its_finding(tmp_path: Path) -> None:
    """A failure whose only factors are excluded is unexplained, not missing.

    Dropping it would make group sizes stop summing to the failure count, which
    is the arithmetic every report depends on.
    """
    database = tmp_path / "excluded.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        report = group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"nothing_matches_this"},
        )

    assert report.grouped_count == report.failure_count
    assert report.unexplained_count == report.failure_count


def test_both_partitions_coexist_over_the_same_findings(tmp_path: Path) -> None:
    """The full signature describes; the restricted one discriminates.

    Both are stored, and neither disturbs the other — that is what
    `clusters.method` is for.
    """
    database = tmp_path / "both.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        full = group_run(connection, run_id)
        restricted = group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"blur"},
        )

        assert storage.load_clusters(connection, run_id, FACTOR_SIGNATURE_METHOD)
        assert storage.load_clusters(connection, run_id, DISCRIMINATING_METHOD)
        # Every failure is accounted for under each method independently.
        assert full.grouped_count == full.failure_count
        assert restricted.grouped_count == restricted.failure_count


def test_restricting_factors_cannot_increase_the_group_count(
    tmp_path: Path,
) -> None:
    """Fewer factors can only merge groups, never split them further."""
    database = tmp_path / "monotonic.db"
    run_id = _saved_run(database)

    with storage.connect(database) as connection:
        full = group_run(connection, run_id)
        restricted = group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"blur"},
        )

    assert restricted.group_count <= full.group_count


def test_the_discriminating_set_is_configured_not_computed() -> None:
    """Recomputing membership per run is what makes runs incomparable (D-033)."""
    assert isinstance(config.DISCRIMINATING_FACTORS, frozenset)
    assert {"small_object", "thin_structure"} == config.DISCRIMINATING_FACTORS
