from itertools import product

import numpy as np
import pytest

from vizion.hyperspace_geometry import orthographic_basis, project_tensor


@pytest.mark.parametrize("dimensions", [2, 3, 5, 10, 13])
def test_projection_plane_is_orthonormal_with_equal_axis_foreshortening(dimensions: int) -> None:
    basis = orthographic_basis(dimensions, phase=0.7)
    np.testing.assert_allclose(basis.T @ basis, np.identity(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(basis, axis=1), np.sqrt(2 / dimensions))


def test_matrix_outline_and_lines_match_centered_index_coordinates() -> None:
    projection = project_tensor((3, 4), phase=0)
    assert projection is not None

    np.testing.assert_allclose(
        projection.outline,
        [[-1 / 3, -1 / 2], [1 / 3, -1 / 2], [1 / 3, 1 / 2], [-1 / 3, 1 / 2]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        projection.index_lines,
        [[[-1 / 3, -1 / 6], [1 / 3, -1 / 6]], [[0, -1 / 2], [0, 1 / 2]]],
        atol=1e-12,
    )


def test_high_rank_outline_bounds_match_the_projected_corner_extrema() -> None:
    shape = (2, 4, 8, 3, 4, 8, 1, 5, 9, 3)
    phase = 0.31
    projection = project_tensor(shape, phase)
    assert projection is not None

    corners = np.array(list(product(*[(1 - size, size - 1) for size in shape]))) / 16
    projected = corners @ orthographic_basis(len(shape), phase)
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    directions = np.stack((np.cos(angles), np.sin(angles)))
    np.testing.assert_allclose(
        (projection.outline @ directions).max(axis=0),
        (projected @ directions).max(axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(projection.index_lines, axis=1)[:, 0], axis=1),
        np.array([1, 3, 7, 2, 3, 7, 4, 8, 2]) / 8 * np.sqrt(2 / 10),
    )


def test_unknown_and_singleton_axes_do_not_invent_visible_spans() -> None:
    projection = project_tensor((None, 1, 7), phase=0.4, dimensions=10)
    assert projection is not None
    assert projection.outline.shape == (2, 2)
    assert projection.index_lines.shape == (1, 2, 2)
    np.testing.assert_allclose(projection.outline, projection.index_lines[0])


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_scalar_and_singleton_outputs_project_to_one_tensor_index(shape: tuple[int, ...]) -> None:
    projection = project_tensor(shape, phase=0)
    assert projection is not None
    np.testing.assert_array_equal(projection.outline, [[0, 0]])
    assert len(projection.index_lines) == 0


@pytest.mark.parametrize("shape", [(0, 4), (4, 0, None), (None, 1)])
def test_empty_or_unmeasurable_output_has_no_index_outline(shape: tuple[int | None, ...]) -> None:
    assert project_tensor(shape, phase=0) is None
