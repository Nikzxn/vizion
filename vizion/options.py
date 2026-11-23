"""Typed configuration objects and reusable presets for vizion renderers.

These dataclasses mirror the keyword arguments of the layered and LeNet renderers.

The renderer functions continue to accept plain keyword arguments; options and
presets are an opt-in convenience layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from os import PathLike
from difflib import get_close_matches
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union


from .layers import Layer, LayerType


StyleMap = Mapping[Union[str, type], Mapping[str, Any]]
TextCallable = Callable[[int, Layer], Tuple[str, bool]]

_EXPLICIT_ARGUMENTS = ContextVar("vizion_explicit_arguments", default=frozenset())


def _track_explicit_arguments(renderer):
    renderer_signature = signature(renderer)

    @wraps(renderer)
    def wrapped(*args, **kwargs):
        arguments = renderer_signature.bind(*args, **kwargs).arguments
        token = _EXPLICIT_ARGUMENTS.set(frozenset(arguments))
        try:
            return renderer(*args, **kwargs)
        finally:
            _EXPLICIT_ARGUMENTS.reset(token)

    return wrapped


def _is_explicit(name):
    return name in _EXPLICIT_ARGUMENTS.get()


def _validate_option_keys(values, valid_keys, renderer_name):
    unknown = sorted(set(values) - set(valid_keys), key=str)
    if not unknown:
        return
    message = "Unknown {} option(s): {}.".format(
        renderer_name, ", ".join(repr(key) for key in unknown)
    )
    suggestions = []
    for key in unknown:
        if isinstance(key, str):
            matches = get_close_matches(key, valid_keys, n=1)
            if matches:
                suggestions.append("{!r} -> {!r}".format(key, matches[0]))
    if suggestions:
        message += " Did you mean {}?".format(", ".join(suggestions))
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Text callable templates
# ---------------------------------------------------------------------------

def _layer_name(index: int, layer: Layer) -> str:
    return layer.name or f"{layer.kind}_{index}"


def _format_shape(layer: Layer) -> str:
    return " x ".join("?" if dim is None else str(dim) for dim in layer.output_shape) or "scalar"


LAYERED_TEXT_CALLABLES: Dict[str, TextCallable] = {
    # (text, above)
    "name": lambda i, layer: (_layer_name(i, layer), False),
    "type": lambda i, layer: (layer.kind, False),
    "shape": lambda i, layer: (_format_shape(layer), False),
    "name_shape": lambda i, layer: (f"{_layer_name(i, layer)}\n{_format_shape(layer)}", False),
}
"""Built-in layer annotation callables for layered diagrams."""


# ---------------------------------------------------------------------------
# Options dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayeredOptions:
    """Typed configuration bundle for :func:`vizion.layered_view`.

    This dataclass mirrors the keyword arguments accepted by ``layered_view`` so
    a layered configuration can be defined once and reused across multiple
    renders.

    The fields correspond directly to the parameters documented on
    :func:`vizion.layered_view`. Use this object when you want to keep
    layered sizing, labeling, grouping, connector styling, and per-layer
    overrides together as one reusable configuration.

    ``LayeredOptions`` is most useful when the same visual style should be
    applied to several related models, notebooks, or documentation examples. It
    keeps a long list of renderer settings in one typed object rather than
    repeating them across many calls.

    The class is intentionally thin. It does not introduce a second styling
    system or any extra resolution rules beyond what ``layered_view`` already
    supports. Its main job is to make a layered configuration easier to read,
    store, and reuse.
    """
    # Mirrors layered_view kwargs (excluding `layers`)
    to_file: str | PathLike[str] | None = None
    min_z: int = 20
    min_xy: int = 20
    max_z: int = 400
    max_xy: int = 2000
    scale_z: float = 1.5
    scale_xy: float = 4.0
    type_ignore: Optional[Sequence[LayerType]] = None
    index_ignore: Optional[Sequence[int]] = None
    color_map: Optional[Mapping[LayerType, Mapping[str, Any]]] = None
    one_dim_orientation: str = "z"
    index_2D: Sequence[int] = field(default_factory=tuple)
    background_fill: Any = "white"
    draw_volume: bool = True
    draw_reversed: bool = False
    padding: int = 10
    text_callable: Optional[Union[str, TextCallable]] = None
    text_vspacing: int = 4
    spacing: int = 10
    draw_funnel: bool = True
    shade_step: int = 10
    legend: bool = False
    legend_text_spacing_offset: int = 15
    font: Any = None
    font_color: Any = "black"
    show_dimension: bool = False
    sizing_mode: str = "accurate"
    dimension_caps: Optional[Mapping[str, int]] = None
    relative_base_size: int = 20
    connector_fill: Any = "gray"
    connector_width: int = 1
    image_fit: str = "fill"
    image_axis: str = "z"
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None
    logo_groups: Optional[Sequence[Dict[str, Any]]] = None
    logos_legend: Union[bool, Dict[str, Any]] = False
    styles: Optional[StyleMap] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Return the options object as a plain keyword-argument mapping."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class LenetOptions:
    """Typed configuration bundle for :func:`vizion.lenet_view`.

    The fields correspond directly to the parameters documented on
    :func:`vizion.lenet_view`. Use this object when you want to preserve a
    consistent LeNet-style layout, connector behavior, patch styling, label
    spacing, and per-layer overrides across multiple renders.

    ``LenetOptions`` is most helpful when publication-oriented LeNet-style
    figures should share the same stack spacing, label treatment, patch
    appearance, and embedded-image behavior across several examples.

    As with the other options classes, this dataclass does not change renderer
    semantics. It provides a clearer, reusable way to package a LeNet-style
    configuration before passing it to ``lenet_view`` or ``show``.
    """
    to_file: str | PathLike[str] | None = None
    min_xy: int = 20
    max_xy: int = 220
    scale_xy: float = 4.0
    type_ignore: Optional[Sequence[LayerType]] = None
    index_ignore: Optional[Sequence[int]] = None
    color_map: Optional[Mapping[LayerType, Mapping[str, Any]]] = None
    background_fill: Any = "black"
    padding: int = 20
    layer_spacing: int = 40
    map_spacing: int = 4
    max_visual_channels: int = 12
    connector_fill: Any = "gray"
    connector_width: int = 1
    patch_fill: Any = "#7db7ff"
    patch_outline: Any = "black"
    patch_scale: float = 1.0
    patch_alpha_on_image: int = 140
    seed: Optional[int] = None
    draw_connections: bool = True
    draw_patches: bool = True
    font: Any = None
    font_color: Any = "white"
    top_label_padding: int = 6
    bottom_label_padding: int = 6
    top_label: bool = True
    bottom_label: bool = True
    styles: Optional[StyleMap] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Return the options object as a plain keyword-argument mapping."""
        return dict(self.__dict__)


@dataclass(frozen=True, slots=True)
class HyperspaceOptions:
    """Configure the canvas, minimum camera rank, angle in radians, and labels."""

    to_file: str | PathLike[str] | None = None
    dimensions: int | None = None
    phase: float = 0.0
    size: tuple[int, int] = (1200, 720)
    labels: bool = True


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

LAYERED_PRESETS: Dict[str, LayeredOptions] = {
    # Mirrors the function defaults used across the README's quick-start.
    "default": LayeredOptions(),
    # Tighter spacing for dense figures (added in 0.3.0).
    "compact": LayeredOptions(spacing=6, padding=6, connector_width=1),
    # Matches the "flat style" example showcased in the documentation.
    # Kept identical to v0.2.0 for backward compatibility.
    "flat": LayeredOptions(draw_volume=False, draw_funnel=False, shade_step=0),
    # Emphasizes clean spacing and 2D presentation for print-ready figures.
    # Kept identical to v0.2.0 for backward compatibility.
    "presentation": LayeredOptions(
        draw_volume=False,
        draw_funnel=True,
        spacing=40,
        padding=30,
        scale_xy=2.5,
        scale_z=1.0,
        legend=True,
        show_dimension=True,
        sizing_mode="balanced",
        text_callable=LAYERED_TEXT_CALLABLES["name_shape"],
    ),
}
"""Curated presets for layered renderings keyed by human-friendly names."""


LENET_PRESETS: Dict[str, LenetOptions] = {
    "default": LenetOptions(),
    "compact": LenetOptions(layer_spacing=28, map_spacing=3, max_xy=180, padding=15),
    "presentation": LenetOptions(layer_spacing=55, map_spacing=5, max_xy=260, connector_width=2),
}
"""Curated presets for lenet-style renderings keyed by human-friendly names."""


HYPERSPACE_PRESETS = {
    "default": HyperspaceOptions(),
    "compact": HyperspaceOptions(size=(720, 440)),
    "presentation": HyperspaceOptions(size=(1600, 960)),
}
