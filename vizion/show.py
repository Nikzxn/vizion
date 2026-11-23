"""Select a Vizion renderer by mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL.Image import Image

from .hyperspace import hyperspace_view
from .layers import LayerSequence
from .lenet import lenet_view
from .options import HyperspaceOptions, LayeredOptions, LenetOptions


def show(
    layers: LayerSequence,
    *,
    mode: str = "layered",
    preset: str | None = None,
    options: LayeredOptions | LenetOptions | HyperspaceOptions | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Image:
    """Return a Pillow image rendered from a sequence of Vizion layers.

    Use mode "layered", "lenet", or "hyperspace". Options apply after the selected preset,
    with explicit keyword arguments taking precedence.
    """
    mode_name = mode.strip().lower()
    if mode_name in ("layered", "layers"):
        if isinstance(options, (LenetOptions, HyperspaceOptions)):
            raise TypeError("Layered mode requires LayeredOptions or an options mapping.")
        return hyperspace_view(layers, d=3, preset=preset, options=options, **overrides)
    if mode_name in ("lenet", "lenet_style"):
        if isinstance(options, (LayeredOptions, HyperspaceOptions)):
            raise TypeError("LeNet mode requires LenetOptions or an options mapping.")
        return lenet_view(layers, preset=preset, options=options, **overrides)
    if mode_name == "hyperspace":
        return hyperspace_view(layers, preset=preset, options=options, **overrides)
    raise ValueError(f"Unknown mode {mode!r}. Expected 'layered', 'lenet', or 'hyperspace'.")
