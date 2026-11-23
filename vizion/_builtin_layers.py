"""Built-in declarative input, dense, attention, and shape layers."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from math import isfinite, prod

from ._layer_shapes import compatible_dimension, known_volume, positive_int, single_input
from .layers import DataFormat, Layer, LayerDefinitionError, _InferredLayer, Shape


def Input(shape: Shape, *, name: str | None = None, data_format: DataFormat = "channels_last") -> Layer:
    """Declare a batch-free input shape."""
    return Layer("Input", shape, name=name, data_format=data_format)


def Tokens(size: int | None, *, name: str | None = None) -> Layer:
    """Declare a token sequence with a fixed or unknown length."""
    return Layer("Tokens", (size,), name=name)


@dataclass(frozen=True, slots=True)
class Embedding(_InferredLayer):
    """Append an embedding feature axis to a token shape."""

    features: int
    _: KW_ONLY
    num_embeddings: int | None = None
    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        input_layer = single_input(inputs, "Embedding")
        if self.num_embeddings is not None:
            positive_int(self.num_embeddings, "num_embeddings")
        features = positive_int(self.features, "features")
        return Layer("Embedding", (*input_layer.output_shape, features), name=self.name)


@dataclass(frozen=True, slots=True)
class Linear(_InferredLayer):
    """Replace the final feature dimension of the input."""

    out_features: int
    _: KW_ONLY
    in_features: int | None = None
    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        layer = single_input(inputs, "Linear")
        shape = layer.output_shape
        if not shape:
            raise LayerDefinitionError("Linear needs an input with a feature axis.")
        if self.in_features is not None:
            expected = positive_int(self.in_features, "in_features")
            compatible_dimension(expected, shape[-1], f"Linear expected {expected} input features, got {shape[-1]}.")
        features = positive_int(self.out_features, "out_features")
        return Layer(
            "Linear", (*shape[:-1], features), name=self.name,
            data_format=layer.data_format, spatial_1d=layer.spatial_1d,
        )


@dataclass(frozen=True, slots=True)
class Attention(_InferredLayer):
    """Describe self-attention or cross-attention over sequences."""

    num_heads: int
    _: KW_ONLY
    qkv_features: int | None = None
    out_features: int | None = None
    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        if not 1 <= len(inputs) <= 3 or any(len(layer.output_shape) < 2 for layer in inputs):
            raise LayerDefinitionError("Attention needs one to three inputs with sequence and feature axes.")
        query = inputs[0].output_shape
        memory = inputs[1].output_shape if len(inputs) > 1 else query
        values = inputs[2].output_shape if len(inputs) == 3 else memory
        if len(query) != len(memory) or len(memory) != len(values):
            raise LayerDefinitionError("Attention inputs must have the same rank.")
        prefix = []
        for lead, key, value in zip(query[:-2], memory[:-2], values[:-2]):
            known = compatible_dimension(lead, key, "Attention input prefixes must match.")
            prefix.append(compatible_dimension(known, value, "Attention input prefixes must match."))
        compatible_dimension(memory[-2], values[-2], "Attention key and value sequence lengths must match.")
        heads = positive_int(self.num_heads, "num_heads")
        features = query[-1] if self.qkv_features is None else positive_int(self.qkv_features, "qkv_features")
        if features is not None and features % heads:
            raise LayerDefinitionError(f"Attention features ({features}) must be divisible by num_heads ({heads}).")
        output = query[-1] if self.out_features is None else positive_int(self.out_features, "out_features")
        return Layer("Attention", (*prefix, query[-2], output), name=self.name)


@dataclass(frozen=True, slots=True)
class Flatten(_InferredLayer):
    """Combine every axis into one feature axis."""

    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        input_layer = single_input(inputs, "Flatten")
        return Layer("Flatten", (known_volume(input_layer.output_shape),), name=self.name)


@dataclass(frozen=True, slots=True, init=False)
class Reshape(_InferredLayer):
    """Declare a new shape with the same output volume."""

    shape: tuple[int | None, ...]
    name: str | None

    def __init__(self, shape: Shape, *, name: str | None = None) -> None:
        layer = Layer("Reshape", shape, name=name)
        object.__setattr__(self, "shape", tuple(layer.output_shape))
        object.__setattr__(self, "name", name)

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        input_layer = single_input(inputs, "Reshape")
        volume = known_volume(input_layer.output_shape)
        target_volume = known_volume(self.shape)
        message = "Reshape must preserve the number of elements."
        compatible_dimension(volume, target_volume, message)
        shape = self.shape
        if volume is not None and target_volume is None:
            known = prod(dim for dim in shape if dim is not None)
            inferred, remainder = divmod(volume, known)
            if remainder:
                raise LayerDefinitionError(message)
            if shape.count(None) == 1:
                shape = tuple(inferred if dim is None else dim for dim in shape)
        return Layer("Reshape", shape, name=self.name)


@dataclass(frozen=True, slots=True)
class _ShapePreserving(_InferredLayer):
    kind: str
    name: str | None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        layer = single_input(inputs, self.kind)
        return Layer(
            self.kind,
            layer.output_shape,
            name=self.name,
            data_format=layer.data_format,
            spatial_1d=layer.spatial_1d,
        )


def ReLU(*, name: str | None = None) -> _ShapePreserving:
    """Add a ReLU activation."""
    return _ShapePreserving("ReLU", name)


def GELU(*, name: str | None = None) -> _ShapePreserving:
    """Add a GELU activation."""
    return _ShapePreserving("GELU", name)


def Softmax(*, name: str | None = None) -> _ShapePreserving:
    """Add a softmax activation."""
    return _ShapePreserving("Softmax", name)


def LayerNorm(*, name: str | None = None) -> _ShapePreserving:
    """Add layer normalization."""
    return _ShapePreserving("LayerNorm", name)


def BatchNorm(*, name: str | None = None) -> _ShapePreserving:
    """Add batch normalization."""
    return _ShapePreserving("BatchNorm", name)


def Dropout(rate: float = 0.5, *, name: str | None = None) -> _ShapePreserving:
    """Add dropout with a rate between zero and one."""
    if not isfinite(rate) or not 0 <= rate <= 1:
        raise LayerDefinitionError("Dropout rate must be between zero and one.")
    return _ShapePreserving("Dropout", name)


@dataclass(frozen=True, slots=True)
class _Recurrent(_InferredLayer):
    kind: str
    hidden_size: int
    return_sequences: bool
    name: str | None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        shape = single_input(inputs, self.kind).output_shape
        if len(shape) < 2:
            raise LayerDefinitionError(f"{self.kind} needs an input with sequence and feature axes.")
        hidden_size = positive_int(self.hidden_size, "hidden_size")
        prefix = shape[:-1] if self.return_sequences else shape[:-2]
        return Layer(self.kind, (*prefix, hidden_size), name=self.name)


def LSTM(hidden_size: int, *, return_sequences: bool = True, name: str | None = None) -> _Recurrent:
    """Describe an LSTM producing hidden features."""
    return _Recurrent("LSTM", hidden_size, return_sequences, name)


def GRU(hidden_size: int, *, return_sequences: bool = True, name: str | None = None) -> _Recurrent:
    """Describe a GRU producing hidden features."""
    return _Recurrent("GRU", hidden_size, return_sequences, name)
