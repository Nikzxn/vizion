"""Built-in convolution and pooling layers."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from numbers import Integral
from typing import Literal

from ._layer_shapes import compatible_dimension, positive_int, single_input
from .layers import Layer, LayerDefinitionError, _InferredLayer, Shape, Window

Padding = Literal["SAME", "VALID"] | int


@dataclass(frozen=True, slots=True)
class _Window2d:
    size: tuple[int, int]
    strides: tuple[int, int]
    padding: Padding
    dilation: tuple[int, int]


def _pair(value: Window, label: str) -> tuple[int, int]:
    if isinstance(value, Integral):
        item = positive_int(value, label)
        return item, item
    try:
        height, width = value
    except (TypeError, ValueError) as exc:
        raise LayerDefinitionError(f"{label} must be an integer or a pair of integers.") from exc
    return positive_int(height, label), positive_int(width, label)


def _window(size: Window, strides: Window, padding: Padding, dilation: Window) -> _Window2d:
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ("SAME", "VALID"):
            raise LayerDefinitionError("padding must be SAME, VALID, or a non-negative integer.")
    elif isinstance(padding, bool) or not isinstance(padding, Integral) or padding < 0:
        raise LayerDefinitionError("padding must be SAME, VALID, or a non-negative integer.")
    return _Window2d(
        _pair(size, "kernel_size"), _pair(strides, "strides"), padding, _pair(dilation, "dilation"),
    )


def _output_extent(
    length: int | None, kernel: int, stride: int, padding: Padding, dilation: int,
) -> int | None:
    if length is None:
        return None
    if padding == "SAME":
        return (length + stride - 1) // stride
    total_padding = 0 if padding == "VALID" else 2 * int(padding)
    effective_kernel = (kernel - 1) * dilation + 1
    return max(0, (length + total_padding - effective_kernel) // stride + 1)


def _output_shape(layer: Layer, window: _Window2d, channels: int | None) -> Shape:
    spatial = layer.output_shape[-2:] if layer.data_format == "channels_first" else layer.output_shape[-3:-1]
    height, width = (
        _output_extent(length, kernel, stride, window.padding, dilation)
        for length, kernel, stride, dilation in zip(spatial, window.size, window.strides, window.dilation)
    )
    if layer.data_format == "channels_first":
        return (channels, *layer.output_shape[1:-2], height, width)
    return (*layer.output_shape[:-3], height, width, channels)


def _channels(layer: Layer, kind: str) -> int | None:
    if len(layer.output_shape) < 3:
        raise LayerDefinitionError(f"{kind} needs height, width, and channel axes.")
    return layer.output_shape[0 if layer.data_format == "channels_first" else -1]


@dataclass(frozen=True, slots=True)
class Conv2d(_InferredLayer):
    """Slide a 2D kernel over the input's spatial axes."""

    out_channels: int
    kernel_size: Window
    _: KW_ONLY
    strides: Window = 1
    padding: Padding = "SAME"
    dilation: Window = 1
    in_channels: int | None = None
    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        layer = single_input(inputs, "Conv2D")
        channels = _channels(layer, "Conv2D")
        if self.in_channels is not None:
            expected = positive_int(self.in_channels, "in_channels")
            compatible_dimension(expected, channels, f"Conv2d expected {expected} input channels, got {channels}.")
        window = _window(self.kernel_size, self.strides, self.padding, self.dilation)
        output = _output_shape(layer, window, positive_int(self.out_channels, "out_channels"))
        return Layer("Conv2D", output, name=self.name, data_format=layer.data_format, kernel_size=window.size)


@dataclass(frozen=True, slots=True)
class _Pool2d(_InferredLayer):
    kind: str
    kernel_size: Window
    strides: Window | None
    padding: Padding
    name: str | None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        layer = single_input(inputs, self.kind)
        channels = _channels(layer, self.kind)
        strides = self.kernel_size if self.strides is None else self.strides
        window = _window(self.kernel_size, strides, self.padding, 1)
        output = _output_shape(layer, window, channels)
        return Layer(self.kind, output, name=self.name, data_format=layer.data_format, pool_size=window.size)


def MaxPool2d(
    kernel_size: Window, *, strides: Window | None = None, padding: Padding = "VALID", name: str | None = None,
) -> _Pool2d:
    """Pool by maximum value over a 2D window."""
    return _Pool2d("MaxPool2D", kernel_size, strides, padding, name)


def AvgPool2d(
    kernel_size: Window, *, strides: Window | None = None, padding: Padding = "VALID", name: str | None = None,
) -> _Pool2d:
    """Pool by average value over a 2D window."""
    return _Pool2d("AvgPool2D", kernel_size, strides, padding, name)
