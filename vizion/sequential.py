"""Compose layers and collect the directed diagram from their values."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from .layers import Connection, Layer, LayerDefinitionError, LayerSequence, Spacer, Value, _InferredLayer


@dataclass(frozen=True, slots=True)
class _Group:
    name: str
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Diagram:
    layers: tuple[Layer | Spacer, ...]
    connections: tuple[Connection, ...]
    groups: tuple[_Group, ...]
    values: tuple[Value, ...]
    output_index: int | None


class Sequential(Layer):
    """Compose positional layers. Calling the sequence connects its steps in order."""

    __slots__ = ("_steps", "_compiled")

    def __init__(self, *layers: Layer | Spacer | Value) -> None:
        for index, layer in enumerate(layers):
            if not isinstance(layer, (Layer, Spacer, Value)):
                raise LayerDefinitionError(f"Expected a layer, spacer, or value at index {index}, got {type(layer).__name__}.")
        self._steps = tuple(
            step
            for layer in layers
            for step in (layer._steps if type(layer) is Sequential else (layer,))
        )
        self._compiled: tuple[_Diagram, Value | None] | None = None
        if len(layers) == 1 and isinstance(layers[0], Value) and layers[0]._diagram is not None:
            self._compiled = (layers[0]._diagram, layers[0])

    def _apply(self, current: Value | None) -> Iterator[tuple[Layer | Spacer | Value, Value | Spacer]]:
        for layer in self._steps:
            if isinstance(layer, Spacer):
                yield layer, layer
                continue
            if isinstance(layer, Value):
                current = layer
            else:
                current = layer() if current is None else layer(current)
            if not isinstance(current, Value):
                raise LayerDefinitionError(f"{type(layer).__name__} must return a Value from its layer calls.")
            yield layer, current

    def _trace(self) -> tuple[_Diagram, Value | None]:
        if self._compiled is not None:
            return self._compiled
        layers: list[Layer | Spacer] = []
        indices: dict[Value, int] = {}
        connections: list[Connection] = []
        output: Value | None = None
        groups: list[_Group] = []
        captures: list[Value] = []
        for entry, step in self._apply(None):
            if isinstance(step, Spacer):
                layers.append(step)
                continue
            output = step
            first = len(indices)
            pending = [(step, False)]
            while pending:
                node, expanded = pending.pop()
                if node in indices:
                    continue
                if not expanded:
                    pending.append((node, True))
                    pending.extend((parent, False) for parent in reversed(node.inputs))
                    continue
                index = len(indices)
                indices[node] = index
                if node._diagram is not None:
                    captures.append(node)
                layers.append(node.layer)
                connections.extend((indices[parent], index) for parent in dict.fromkeys(node.inputs))
            if (
                len(indices) > first
                and isinstance(entry, Layer)
                and not isinstance(entry, _InferredLayer)
                and type(entry).__call__ is not Layer.__call__
            ):
                name = getattr(entry, "name", None) or type(entry).__name__
                groups.append(_Group(name, tuple(range(first, len(indices)))))
        if captures:
            layers, inherited_groups = _inherit_layout(layers, indices, captures)
            groups = list(dict.fromkeys((*inherited_groups, *groups)))
        output_index = indices[output] if output is not None else None
        diagram = _Diagram(tuple(layers), tuple(connections), tuple(groups), tuple(indices), output_index)
        final = replace(output, _diagram=diagram) if output is not None else None
        self._compiled = (diagram, final)
        return self._compiled

    def _diagram(self) -> _Diagram:
        return self._trace()[0]

    @property
    def layers(self) -> tuple[Layer | Spacer, ...]:
        return self._diagram().layers

    @property
    def connections(self) -> tuple[Connection, ...]:
        """Return directed indices into the rendered layers, excluding spacers."""
        return self._diagram().connections

    @property
    def output(self) -> Value:
        _, output = self._trace()
        if output is None:
            raise LayerDefinitionError("This sequence has no output value.")
        return output

    def __call__(self, *inputs: Value) -> Value:
        if not inputs:
            return self.output
        if len(inputs) != 1 or not isinstance(inputs[0], Value):
            raise LayerDefinitionError("Sequential accepts one input value.")
        current = inputs[0]
        for _, step in self._apply(current):
            if isinstance(step, Value):
                current = step
        return current

    def __iter__(self) -> Iterator[Layer | Spacer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)


def _inherit_layout(
    layers: list[Layer | Spacer], indices: dict[Value, int], captures: list[Value],
) -> tuple[list[Layer | Spacer], list[_Group]]:
    """Retain layout details when a captured value is used as another layer's input."""
    groups: list[_Group] = []
    insertions: dict[int, dict[tuple[int, int], Spacer]] = {}
    for capture in captures:
        diagram = capture._diagram
        if diagram is None:
            continue
        actual = {
            index: indices[capture if index == diagram.output_index else value]
            for index, value in enumerate(diagram.values)
            if (capture if index == diagram.output_index else value) in indices
        }
        for group in diagram.groups:
            nodes = tuple(actual[index] for index in group.nodes if index in actual)
            if nodes:
                groups.append(_Group(group.name, nodes))

        source_index = 0
        pending: list[Spacer] = []
        for layer in diagram.layers:
            if isinstance(layer, Spacer):
                pending.append(layer)
                continue
            if pending and source_index in actual:
                entries = insertions.setdefault(actual[source_index], {})
                entries.update(((id(spacer), index), spacer) for index, spacer in enumerate(pending))
            pending.clear()
            source_index += 1
        if pending and diagram.output_index in actual:
            entries = insertions.setdefault(actual[diagram.output_index] + 1, {})
            entries.update(((id(spacer), index), spacer) for index, spacer in enumerate(pending))

    result: list[Layer | Spacer] = []
    target_index = 0
    for layer in layers:
        if isinstance(layer, Layer):
            result.extend(insertions.get(target_index, {}).values())
            target_index += 1
        result.append(layer)
    result.extend(insertions.get(target_index, {}).values())
    return result, groups


def compose(layers: LayerSequence) -> Sequential:
    """Normalize renderer input to one compiled sequence."""
    if isinstance(layers, Sequential):
        return layers
    if isinstance(layers, (Layer, Value)):
        return Sequential(layers)
    return Sequential(*layers)


def visible_connections(model: Sequential, positions: tuple[int, ...]) -> tuple[Connection, ...]:
    """Route through hidden layers until a visible destination is reached."""
    nodes = tuple(index for index, layer in enumerate(model.layers) if isinstance(layer, Layer))
    visible = {position: index for index, position in enumerate(positions)}
    successors: dict[int, list[int]] = {}
    for source, destination in model.connections:
        successors.setdefault(nodes[source], []).append(nodes[destination])

    edges: set[Connection] = set()
    for start, source in visible.items():
        pending = list(successors.get(start, ()))
        seen: set[int] = set()
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            if target in visible:
                edges.add((source, visible[target]))
            else:
                pending.extend(successors.get(target, ()))
    return tuple(sorted(edges))
