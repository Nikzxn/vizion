"""Place network layers in columns or wrap a long chain into rows."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .layers import Connection

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Layout:
    centers: tuple[Point, ...]
    tile_size: int
    pitch: float


def _columns(count: int, connections: tuple[Connection, ...]) -> list[list[int]]:
    successors: list[list[int]] = [[] for _ in range(count)]
    for source, target in connections:
        successors[source].append(target)

    ranks = [0] * count
    for source, targets in enumerate(successors):
        for target in targets:
            ranks[target] = max(ranks[target], ranks[source] + 1)

    columns: list[list[int]] = [[] for _ in range(max(ranks) + 1)]
    for index, rank in enumerate(ranks):
        columns[rank].append(index)
    return columns


def arrange(count: int, connections: tuple[Connection, ...], size: tuple[int, int], labels: bool) -> Layout:
    """Lay out ordered chains in reading order and other networks by graph rank."""
    if count == 0:
        return Layout((), 1, 0)
    width, height = size
    margin = min(50, width * 0.07)
    top = min(106, height * 0.18) if labels else height * 0.08
    bottom = height - (min(76, height * 0.15) if labels else height * 0.08)
    usable_width = width - 2 * margin
    centers: list[Point] = [(0, 0)] * count

    if connections == tuple((index, index + 1) for index in range(count - 1)):
        columns = min(count, max(1, int(usable_width / 132)))
        rows = ceil(count / columns)
        pitch = usable_width / columns
        row_pitch = (bottom - top) / rows
        for index in range(count):
            row, position = divmod(index, columns)
            column = position if row % 2 == 0 else columns - 1 - position
            centers[index] = margin + (column + 0.5) * pitch, top + (row + 0.5) * row_pitch
    else:
        graph_columns = _columns(count, connections)
        pitch = usable_width / len(graph_columns)
        row_pitch = (bottom - top) / max(map(len, graph_columns))
        for column, members in enumerate(graph_columns):
            for row, index in enumerate(members):
                centers[index] = (
                    margin + (column + 0.5) * pitch,
                    (top + bottom) / 2 + (row + 0.5 - len(members) / 2) * row_pitch,
                )

    tile_size = max(1, int(min(pitch * 0.72, row_pitch * 0.66, 190)))
    return Layout(tuple(centers), tile_size, pitch)
