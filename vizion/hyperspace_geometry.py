"""Project a tensor's index domain into a two-dimensional view."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt, tau
from typing import Final

import numpy as np
from numpy.typing import NDArray

MIN_DIMENSIONS: Final = 4
MAX_DIMENSIONS: Final = 10


@dataclass(frozen=True, slots=True)
class TensorProjection:
    outline: NDArray[np.float64]
    index_lines: NDArray[np.float64]


def orthographic_basis(dimensions: int, phase: float) -> NDArray[np.float64]:
    """Return a projection plane that gives each axis equal foreshortening."""
    phase %= tau
    if dimensions == 1:
        return np.array([[np.cos(phase), np.sin(phase)]])

    order = np.arange(dimensions)
    # Give the first two axes the largest angle between them when using a
    # higher-rank camera for lower-rank outputs.
    order[1], order[dimensions // 2] = order[dimensions // 2], order[1]
    angles = pi * order / dimensions + phase
    return np.column_stack((np.cos(angles), np.sin(angles))) * sqrt(2 / dimensions)


def _outline(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    # The projected box is a sum of line segments. Walking their directions
    # in angular order traces its exact convex boundary.
    x, y = vectors.T
    reverse = (y < 0) | ((y == 0) & (x < 0))
    forward = np.where(reverse[:, None], -vectors, vectors)
    side = forward[np.argsort(np.arctan2(forward[:, 1], forward[:, 0]))]
    steps = np.concatenate((np.zeros((1, 2)), side, -side[:-1]))
    return np.cumsum(steps, axis=0) - side.sum(axis=0) / 2


def project_tensor(
    shape: tuple[int | None, ...], phase: float, dimensions: int | None = None,
) -> TensorProjection | None:
    """Project the known index spans and their central index lines."""
    if 0 in shape:
        return None

    spans = tuple(0 if size is None else size - 1 for size in shape)
    largest = max(spans, default=0)
    if largest == 0:
        if None in shape:
            return None
        return TensorProjection(np.zeros((1, 2)), np.empty((0, 2, 2)))

    basis = orthographic_basis(max(len(shape), dimensions or 0), phase)[:len(shape)]
    vectors = basis * np.array([span / largest for span in spans])[:, None]
    active = np.any(vectors, axis=1)
    # For even axis sizes, use the lower of the two middle indices.
    offsets = basis * np.array([-(span % 2) / (2 * largest) for span in spans])[:, None]
    midpoints = offsets.sum(axis=0) - offsets
    lines = np.stack((midpoints - vectors / 2, midpoints + vectors / 2), axis=1)
    return TensorProjection(_outline(vectors[active]), lines[active])
