"""Framework-independent descriptions of layers and their output shapes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .sequential import _Diagram

Shape = Sequence[int | None]
DataFormat = Literal["channels_last", "channels_first"]
Window = int | Sequence[int]


class LayerDefinitionError(ValueError):
    """A layer or spacer contains invalid diagram metadata."""


class Layer:
    """Describe one layer in a diagram.

    Use a batch-free output shape. Individual dimensions may be None when
    unknown, and an empty shape represents a scalar. Kind describes the
    operation and is also a styling key. Kernel and pool sizes are optional
    receptive-field hints for LeNet connections. Use spatial_1d to mark a
    two-axis output as a 1D feature map in LeNet.
    """

    __slots__ = ("kind", "output_shape", "name", "data_format", "spatial_1d", "kernel_size", "pool_size", "_source_type")

    kind: str
    output_shape: tuple[int | None, ...]
    name: str | None
    data_format: DataFormat
    spatial_1d: bool
    kernel_size: Window | None
    pool_size: Window | None

    def __init__(
        self,
        kind: str = "Layer",
        output_shape: Shape = (),
        *,
        name: str | None = None,
        data_format: DataFormat = "channels_last",
        spatial_1d: bool = False,
        kernel_size: Window | None = None,
        pool_size: Window | None = None,
    ) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise LayerDefinitionError("Layer kind must be a non-empty string.")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise LayerDefinitionError("Layer name must be a non-empty string when supplied.")
        if data_format not in ("channels_last", "channels_first"):
            raise LayerDefinitionError("Layer data_format must be channels_last or channels_first.")
        if isinstance(output_shape, (str, bytes)):
            raise LayerDefinitionError("Layer output_shape must be a sequence of dimensions.")
        try:
            shape = tuple(output_shape)
        except TypeError as exc:
            raise LayerDefinitionError("Layer output_shape must be a sequence of dimensions.") from exc
        if any(
            dim is not None and (isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 0)
            for dim in shape
        ):
            raise LayerDefinitionError("Layer dimensions must be non-negative integers or None.")
        if spatial_1d and len(shape) != 2:
            raise LayerDefinitionError("A 1D feature map needs exactly two output dimensions.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "output_shape", tuple(None if dim is None else int(dim) for dim in shape))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "data_format", data_format)
        object.__setattr__(self, "spatial_1d", spatial_1d)
        object.__setattr__(self, "kernel_size", _normalize_window(kernel_size))
        object.__setattr__(self, "pool_size", _normalize_window(pool_size))

    def __setattr__(self, name: str, value: object) -> None:
        if name in Layer.__slots__ and hasattr(self, name):
            raise AttributeError(f"Layer metadata {name!r} is read-only.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in Layer.__slots__:
            raise AttributeError(f"Layer metadata {name!r} is read-only.")
        object.__delattr__(self, name)

    def __call__(self, *inputs: Value) -> Value:
        """Record one use of this layer and the values wired into it."""
        return Value(self, inputs)


def _normalize_window(value: Window | None) -> int | tuple[int, ...] | None:
    """Freeze a receptive-field window and check its dimensions."""
    if value is None:
        return None
    scalar = isinstance(value, Integral)
    try:
        sizes = (value,) if scalar else tuple(value)
    except TypeError as exc:
        raise LayerDefinitionError("A kernel or pool size must be an integer or a sequence.") from exc
    if len(sizes) not in (1, 2) or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0 for size in sizes
    ):
        raise LayerDefinitionError("A kernel or pool size needs one or two positive integer dimensions.")
    normalized = tuple(int(size) for size in sizes)
    return normalized[0] if scalar else normalized


@dataclass(frozen=True, slots=True)
class Spacer:
    """Add horizontal space in layered diagrams, in pixels."""

    spacing: int

    def __post_init__(self) -> None:
        if isinstance(self.spacing, bool) or not isinstance(self.spacing, Integral) or self.spacing < 0:
            raise LayerDefinitionError("Spacer spacing must be a non-negative integer.")


@dataclass(frozen=True, slots=True, eq=False)
class Value:
    """A symbolic layer output and the values used to produce it."""

    layer: Layer
    inputs: tuple[Value, ...]
    _diagram: _Diagram | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layer, Layer) or not hasattr(self.layer, "output_shape"):
            raise LayerDefinitionError("A Value needs a layer with a resolved output shape.")
        inputs = tuple(self.inputs)
        if not all(isinstance(value, Value) for value in inputs):
            raise LayerDefinitionError("Connect layers by passing Value objects produced by layer calls.")
        object.__setattr__(self, "inputs", inputs)

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.layer.output_shape

    def __add__(self, other: Value) -> Value:
        from ._merge_layers import Add

        return Add()(self, other)


class _InferredLayer(Layer):
    __slots__ = ()

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        raise NotImplementedError

    def __call__(self, *inputs: Value) -> Value:
        if not all(isinstance(value, Value) for value in inputs):
            raise LayerDefinitionError("Connect layers by passing Value objects produced by layer calls.")
        layer = self.resolve(tuple(value.layer for value in inputs))
        object.__setattr__(layer, "_source_type", type(self))
        return Value(layer, inputs)


Connection = tuple[int, int]
LayerSequence = Iterable[Layer | Spacer] | Layer | Value
LayerSelector = str | type[Layer] | Layer
LayerType = str | type[Layer]


def layer_class(layer: Layer) -> type[Layer]:
    return getattr(layer, "_source_type", type(layer))


def matches_layer(layer: Layer, selector: LayerSelector) -> bool:
    """Match a layer by kind, name, class, or identity."""
    if isinstance(selector, str):
        return selector in (layer.kind, layer.name)
    if isinstance(selector, type):
        return issubclass(layer_class(layer), selector)
    return layer is selector
