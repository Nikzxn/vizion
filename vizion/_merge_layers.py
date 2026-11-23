"""Built-in layers that combine multiple directed inputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from numbers import Integral

from ._layer_shapes import compatible_dimension
from .layers import DataFormat, Layer, LayerDefinitionError, _InferredLayer


def _merged_layout(inputs: tuple[Layer, ...], rank: int) -> tuple[DataFormat, bool]:
    spatial_inputs = tuple(layer for layer in inputs if len(layer.output_shape) >= 3 or layer.spatial_1d)
    layouts = {layer.data_format for layer in spatial_inputs}
    if len(layouts) > 1:
        raise LayerDefinitionError("Merged inputs must use the same channel layout.")
    layout = next(iter(layouts), inputs[0].data_format)
    spatial_1d = rank == 2 and any(layer.spatial_1d for layer in spatial_inputs)
    return layout, spatial_1d


@dataclass(frozen=True, slots=True)
class Add(_InferredLayer):
    """Combine inputs with right-aligned broadcasting."""

    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        if len(inputs) < 2:
            raise LayerDefinitionError("Add needs at least two input layers.")
        dimensions: list[int | None] = []
        for axis in zip_longest(*(reversed(layer.output_shape) for layer in inputs), fillvalue=1):
            known = {dim for dim in axis if dim not in (1, None)}
            if len(known) > 1:
                raise LayerDefinitionError("Add needs broadcast-compatible input shapes.")
            dimensions.append(next(iter(known), None if None in axis else 1))
        layout, spatial_1d = _merged_layout(inputs, len(dimensions))
        return Layer("Add", tuple(reversed(dimensions)), name=self.name, data_format=layout, spatial_1d=spatial_1d)


@dataclass(frozen=True, slots=True)
class Concatenate(_InferredLayer):
    """Join inputs along one axis."""

    axis: int = -1
    name: str | None = None

    def resolve(self, inputs: tuple[Layer, ...]) -> Layer:
        if len(inputs) < 2:
            raise LayerDefinitionError("Concatenate needs at least two input layers.")
        shape = list(inputs[0].output_shape)
        if isinstance(self.axis, bool) or not isinstance(self.axis, Integral) or not -len(shape) <= self.axis < len(shape):
            raise LayerDefinitionError("Concatenate axis is out of range.")
        axis = self.axis % len(shape)
        for layer in inputs[1:]:
            if len(layer.output_shape) != len(shape):
                raise LayerDefinitionError("Concatenate needs inputs with the same rank.")
            for index, other in enumerate(layer.output_shape):
                dim = shape[index]
                if index == axis:
                    shape[index] = None if dim is None or other is None else dim + other
                else:
                    shape[index] = compatible_dimension(dim, other, "Concatenate needs matching non-join dimensions.")
        layout, spatial_1d = _merged_layout(inputs, len(shape))
        return Layer("Concatenate", tuple(shape), name=self.name, data_format=layout, spatial_1d=spatial_1d)
