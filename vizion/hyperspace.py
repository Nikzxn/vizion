"""Hyperspace network viewer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from math import isfinite
from os import PathLike
from typing import Any

from PIL import Image

from .hyperspace_geometry import MAX_DIMENSIONS, MIN_DIMENSIONS
from .hyperspace_paint import paint_model
from .layered import layered_view
from .layers import LayerSequence
from .options import (
    HYPERSPACE_PRESETS,
    HyperspaceOptions,
    _is_explicit,
    _track_explicit_arguments,
    _validate_option_keys,
)
from .sequential import compose


@_track_explicit_arguments
def _tensor_view(
    layers: LayerSequence,
    to_file: str | PathLike[str] | None = None,
    *,
    dimensions: int | None = None,
    phase: float = 0.0,
    size: tuple[int, int] = (1200, 720),
    labels: bool = True,
    options: HyperspaceOptions | Mapping[str, Any] | None = None,
    preset: str | None = None,
) -> Image.Image:
    """Draw the orthographic projection of each layer's output index domain.

    Each output uses its own rank and axis extents. An explicit dimensions
    value sets a minimum camera rank while keeping every actual output axis.
    Connections follow the layer calls recorded by Sequential.
    """
    settings = HyperspaceOptions(to_file=to_file, dimensions=dimensions, phase=phase, size=size, labels=labels)
    if preset is not None or options is not None:
        explicit = {
            "to_file": to_file,
            "dimensions": dimensions,
            "phase": phase,
            "size": size,
            "labels": labels,
        }
        try:
            settings = HYPERSPACE_PRESETS["default" if preset is None else preset]
        except KeyError as exc:
            raise ValueError(f"Unknown hyperspace preset {preset!r}.") from exc
        if isinstance(options, HyperspaceOptions):
            settings = options
        elif options is not None:
            if not isinstance(options, Mapping):
                raise TypeError("options must be HyperspaceOptions or a mapping of keyword arguments.")
            _validate_option_keys(options, explicit, "hyperspace")
            settings = replace(settings, **options)
        settings = replace(settings, **{key: value for key, value in explicit.items() if _is_explicit(key)})

    if settings.dimensions is not None and (
        type(settings.dimensions) is not int or not MIN_DIMENSIONS <= settings.dimensions <= MAX_DIMENSIONS
    ):
        raise ValueError(f"Hyperspace dimensions must be an integer from {MIN_DIMENSIONS} to {MAX_DIMENSIONS}.")
    if not isfinite(settings.phase):
        raise ValueError("Hyperspace phase must be finite.")
    if len(settings.size) != 2 or any(type(value) is not int or value <= 0 for value in settings.size):
        raise ValueError("Hyperspace size needs two positive integers.")

    model = compose(layers)
    image = paint_model(model, settings)
    if settings.to_file is not None:
        image.save(settings.to_file)
    return image


def hyperspace_view(
    layers: LayerSequence,
    *to_file: str | PathLike[str] | None,
    d: int | None = None,
    **options: Any,
) -> Image.Image:
    """Render a model as layered boxes (d=3) or projected tensor shapes."""
    if d is not None:
        if type(d) is not int or not 3 <= d <= MAX_DIMENSIONS:
            raise ValueError(f"Hyperspace d must be an integer from 3 to {MAX_DIMENSIONS}.")
        if "dimensions" in options:
            raise TypeError("Pass either d or dimensions.")
        if d == 3:
            return layered_view(layers, *to_file, **options)
        options["dimensions"] = d
    return _tensor_view(layers, *to_file, **options)


class Hyperspace:
    """Render a model in 3D layered mode or inspect its projected tensor shapes."""

    render: Callable[..., Image.Image] = staticmethod(hyperspace_view)

    def __init__(self, *, d: int | None = None) -> None:
        self.render = partial(hyperspace_view, d=d)
