"""Route skip edges above the linear renderer's layer shapes."""

from __future__ import annotations


def skip_height(count: int) -> int:
    return 12 + count * 8 if count else 0


def draw_skip(draw, pen, source: tuple[float, float], target: tuple[float, float], lane: int) -> None:
    """Draw a directed edge between two top anchors."""
    x1, y1 = source
    x2, y2 = target
    top = 6 + lane * 8
    draw.line([x1, y1, x1, top, x2, top, x2, y2], pen)
    draw.line([x2 - 3, y2 - 5, x2, y2, x2 + 3, y2 - 5], pen)
