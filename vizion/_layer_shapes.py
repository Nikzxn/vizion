"""Validation and shape arithmetic for built-in layer specs."""

from __future__ import annotations

from math import prod
from numbers import Integral

from .layers import Layer, LayerDefinitionError, Shape


def positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise LayerDefinitionError(f"{label} must be a positive integer.")
    return int(value)


def single_input(inputs: tuple[Layer, ...], kind: str) -> Layer:
    if len(inputs) != 1:
        raise LayerDefinitionError(f"{kind} needs one input layer, got {len(inputs)}.")
    return inputs[0]


def known_volume(shape: Shape) -> int | None:
    if 0 in shape:
        return 0
    if None in shape:
        return None
    return prod(dim for dim in shape if dim is not None)


def compatible_dimension(left: int | None, right: int | None, message: str) -> int | None:
    if left is not None and right is not None and left != right:
        raise LayerDefinitionError(message)
    return left if left is not None else right
