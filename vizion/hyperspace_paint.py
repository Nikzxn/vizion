"""Draw Hyperspace glyphs, directed edges, and layer annotations."""

from __future__ import annotations

from math import hypot
from typing import Final, TypeAlias

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hyperspace_geometry import project_tensor
from .hyperspace_layout import Layout, Point, arrange
from .layers import Connection, Layer
from .options import HyperspaceOptions, _format_shape
from .sequential import Sequential, _Group

_FONT: TypeAlias = ImageFont.ImageFont | ImageFont.FreeTypeFont
_INK: Final = 32
_MUTED: Final = 95


def _font(size: int) -> _FONT:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _trim_label(text: str, font: _FONT, width: int) -> str:
    try:
        font.getbbox(text)
    except UnicodeEncodeError:
        text = text.encode("ascii", errors="replace").decode()
    if font.getbbox(text)[2] <= width:
        return text
    if font.getbbox("...")[2] > width:
        return ""
    while text and font.getbbox(text + "...")[2] > width:
        text = text[:-1]
    return text.rstrip() + "..."


def _centered_text(
    draw: ImageDraw.ImageDraw, position: Point, text: str, font: _FONT, *, width: int, fill: int,
) -> None:
    text = _trim_label(text, font, width)
    x, y = position
    draw.text((x - font.getbbox(text)[2] / 2, y), text, font=font, fill=fill)


def _tensor_mask(layer: Layer, options: HyperspaceOptions, side: int) -> Image.Image | None:
    projection = project_tensor(layer.output_shape, options.phase, options.dimensions)
    if projection is None:
        return None
    samples = 4
    resolution = side * samples
    center = (resolution - 1) / 2
    mask = Image.new("L", (resolution, resolution))
    draw = ImageDraw.Draw(mask)
    if len(projection.outline) == 1:
        draw.ellipse((center - samples, center - samples, center + samples, center + samples), fill=255)
    else:
        scale = center * 0.86 / np.max(np.abs(projection.outline))
        for start, end in np.rint(projection.index_lines * scale + center).astype(np.intp):
            draw.line((*start, *end), fill=125, width=samples)
        outline = [tuple(point) for point in np.rint(projection.outline * scale + center).astype(np.intp)]
        draw.line([*outline, outline[0]], fill=255, width=samples)
    return mask.resize((side, side), Image.Resampling.LANCZOS)


def _shape_lines(shape: str, font: _FONT, width: int) -> list[str]:
    first, *rest = shape.split(" x ")
    lines = [first]
    for extent in rest:
        addition = f"x {extent}"
        combined = f"{lines[-1]} {addition}"
        if font.getbbox(combined)[2] <= width:
            lines[-1] = combined
        else:
            lines.append(addition)
    return lines


def _arrow(draw: ImageDraw.ImageDraw, points: tuple[Point, Point, Point, Point]) -> None:
    a, b, c, d = map(np.asarray, points)
    path: list[Point] = []
    for t in np.linspace(0, 1, 25):
        x, y = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * c + t**3 * d
        path.append((round(float(x)), round(float(y))))
    draw.line(path, fill=_MUTED, width=1)
    x, y = d
    delta_x, delta_y = d - c
    distance = hypot(delta_x, delta_y)
    if distance:
        dx, dy = delta_x / distance, delta_y / distance
        point = (x - 6 * dx, y - 6 * dy)
        draw.polygon(
            [(x, y), (point[0] - 3 * dy, point[1] + 3 * dx), (point[0] + 3 * dy, point[1] - 3 * dx)],
            fill=_MUTED,
        )


def _paint_connections(draw: ImageDraw.ImageDraw, connections: tuple[Connection, ...], layout: Layout) -> None:
    pad = layout.tile_size * 0.46
    for source, target in connections:
        x1, y1 = layout.centers[source]
        x2, y2 = layout.centers[target]
        if abs(x2 - x1) < 1:
            direction = 1 if y2 > y1 else -1
            bend = min(layout.pitch * 0.38, pad + abs(y2 - y1) * 0.22)
            _arrow(draw, (
                (x1 + pad, y1), (x1 + bend, y1 + direction * pad),
                (x2 + bend, y2 - direction * pad), (x2 + pad, y2),
            ))
        else:
            direction = 1 if x2 > x1 else -1
            mid_x = (x1 + x2) / 2
            blocked = any(
                min(x1, x2) < x < max(x1, x2)
                and abs(y - (y1 + (y2 - y1) * (x - x1) / (x2 - x1))) < pad
                for x, y in layout.centers
            )
            lift = min(layout.tile_size, abs(x2 - x1) / 4) if blocked else 0
            _arrow(draw, (
                (x1 + direction * pad, y1), (mid_x, y1 - lift),
                (mid_x, y2 - lift), (x2 - direction * pad, y2),
            ))


def _paint_groups(draw: ImageDraw.ImageDraw, layout: Layout, groups: tuple[_Group, ...], size: tuple[int, int]) -> None:
    width, height = size
    box_padding = layout.tile_size * 0.54
    font = _font(13)
    for group in groups:
        corners = [layout.centers[index] for index in group.nodes]
        left = max(2, min(x for x, _ in corners) - box_padding)
        right = min(width - 2, max(x for x, _ in corners) + box_padding)
        top = max(height * 0.16, min(y for _, y in corners) - box_padding - 24)
        bottom = min(height * 0.92, max(y for _, y in corners) + box_padding + 51)
        if right <= left or bottom <= top:
            continue
        draw.rectangle((left, top, right, bottom), outline=205)
        label = _trim_label(group.name, font, int(right - left - 16))
        draw.text((left + 8, top + 5), label, fill=_MUTED, font=font)


def paint_model(model: Sequential, options: HyperspaceOptions) -> Image.Image:
    """Project output shapes and route the connections created by layer calls."""
    layers = tuple(layer for layer in model if isinstance(layer, Layer))
    layout = arrange(len(layers), model.connections, options.size, options.labels)
    image = Image.new("L", options.size, 255)
    draw = ImageDraw.Draw(image)
    if options.labels:
        _paint_groups(draw, layout, model._diagram().groups, options.size)
    _paint_connections(draw, model.connections, layout)
    for layer, (x, y) in zip(layers, layout.centers):
        glyph = _tensor_mask(layer, options, layout.tile_size)
        if glyph is not None:
            image.paste(_INK, (round(x - glyph.width / 2), round(y - glyph.height / 2)), glyph)

    if options.labels:
        width, height = options.size
        inset = min(50, width * 0.07)
        note = _font(13)
        label_font = _font(14)
        draw.text((inset, height * 0.06), "Projected tensor shapes", font=note, fill=_MUTED)
        text_width = max(1, int(layout.pitch - 10))
        for layer, (x, y) in zip(layers, layout.centers):
            y += layout.tile_size * 0.49 + 5
            _centered_text(draw, (x, y), layer.kind, label_font, width=text_width, fill=_INK)
            captions = _shape_lines(_format_shape(layer), note, text_width)
            if layer.name:
                captions.append(layer.name)
            for row, caption in enumerate(captions):
                _centered_text(draw, (x, y + 19 + 15 * row), caption, note, width=text_width, fill=_MUTED)
    return image.convert("RGBA")
