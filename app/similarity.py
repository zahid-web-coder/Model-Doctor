"""Find failures that look like a given failure, using stored CLIP embeddings.

Separate from :mod:`app.clustering` on purpose. Grouping answers "what kinds of
failure are there", and does it deterministically from attributed factors.
Retrieval answers a different question — "show me more that look like *this
one*" — and is the job the embeddings are actually suited to.

The distinction is statistical, not stylistic. Partitioning 126 vectors in 512
dimensions into clusters requires structure that measurement showed is not
there (D-030). Ranking those same vectors by similarity to one query point
requires no such structure: it is a sort, and it degrades gracefully — the
worst case is that the nearest neighbour is not very near, which the returned
score states plainly instead of hiding inside a cluster assignment.

Nothing here is persisted. A neighbour list is cheap to recompute and depends
on which finding is being asked about, so storing it would mean a table of
answers to questions nobody has asked yet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from utils.exceptions import ModelDoctorError


class SimilarityError(ModelDoctorError):
    """A similarity query cannot be answered, with the reason."""


@dataclass(frozen=True)
class Neighbour:
    """One retrieved failure and how similar it is to the query.

    Carries the outcome and class so a caller can label a result without a
    second query, matching how :class:`~app.storage.EmbeddingRow` behaves.
    """

    finding_id: int
    similarity: float
    outcome: str
    class_name: str


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors.

    The stored vectors are already L2-normalised, so a dot product would
    usually do. The norms are divided out anyway: it costs one pass and keeps
    the function correct for any caller, rather than correct only while an
    invariant maintained in another module happens to hold.

    Args:
        first: A vector.
        second: A vector of the same length.

    Returns:
        Similarity in ``[-1, 1]``. Zero when either vector has no magnitude,
        which is the only defensible answer — a zero vector points nowhere, so
        it is neither similar nor dissimilar to anything.

    Raises:
        SimilarityError: If the vectors have different lengths, which means
            embeddings from different encoders are being compared.
    """
    if len(first) != len(second):
        raise SimilarityError(
            f"Cannot compare vectors of length {len(first)} and {len(second)}. "
            "Embeddings from different encoders are not comparable."
        )

    dot = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(a * a for a in first))
    second_norm = math.sqrt(sum(b * b for b in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot / (first_norm * second_norm)


def nearest_neighbours(
    connection: Any,
    run_id: int,
    finding_id: int,
    limit: int = 5,
    model_name: str | None = None,
) -> list[Neighbour]:
    """Return the failures most visually similar to one failure.

    Args:
        connection: An open database connection.
        run_id: Run to search within. Neighbours are not sought across runs:
            two runs may use different models, and a similar-looking failure
            under different weights is a different observation.
        finding_id: The query finding. It must have a stored embedding.
        limit: Maximum neighbours to return.
        model_name: Restrict to one encoder. Needed once more than one has been
            used, since vectors from different encoders are not comparable.

    Returns:
        Neighbours ordered by descending similarity, excluding the query
        itself. Ties break on finding id so repeated calls agree.

    Raises:
        SimilarityError: If the run has no embeddings, or the query finding has
            none. Both are actionable: run feature extraction first.
    """
    from app import storage

    rows = storage.load_embeddings(connection, run_id, model_name)
    if not rows:
        raise SimilarityError(
            f"Run {run_id} has no embeddings. "
            f"Run: python -m app.features --run {run_id}"
        )

    query = next((row for row in rows if row.finding_id == finding_id), None)
    if query is None:
        raise SimilarityError(
            f"Finding {finding_id} has no embedding in run {run_id}. It may be a "
            "failure whose region was too small to extract."
        )

    scored = [
        Neighbour(
            finding_id=row.finding_id,
            similarity=cosine_similarity(query.vector, row.vector),
            outcome=row.outcome,
            class_name=row.class_name,
        )
        for row in rows
        if row.finding_id != finding_id
    ]
    scored.sort(key=lambda n: (-n.similarity, n.finding_id))
    return scored[: max(0, limit)]
