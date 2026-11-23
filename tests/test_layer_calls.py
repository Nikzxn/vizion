import pytest

import vizion as vz


class Residual(vz.Layer):
    def __init__(self) -> None:
        self.branch = vz.Sequential(vz.Linear(16), vz.GELU(), vz.Linear(8))

    def __call__(self, x: vz.Value) -> vz.Value:
        return self.branch(x) + x


def test_layer_call_tracks_input_value_and_output_shape() -> None:
    tokens = vz.Tokens(size=128)()
    output = vz.Embedding(features=8)(tokens)
    assert output.shape == (128, 8)
    assert output.inputs == (tokens,)
    assert tokens.inputs == ()


def test_custom_layer_builds_a_skip_inside_sequential() -> None:
    model = vz.Sequential(vz.Input((32, 8)), Residual(), vz.Linear(3))
    assert [layer.kind for layer in model] == ["Input", "Linear", "GELU", "Linear", "Add", "Linear"]
    assert model.output.shape == (32, 3)
    residual = model.output.inputs[0]
    assert residual.inputs[1].layer is model.layers[0]
    assert model.connections == ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (4, 5))
    assert vz.Hyperspace.render(model, size=(540, 320)).getcolors(maxcolors=1) is None


def test_shared_layer_calls_create_distinct_connected_values() -> None:
    shared = vz.Linear(8)
    source = vz.Input((32, 8))()
    left = shared(source)
    right = shared(source)
    output = vz.Add()(left, right)
    model = vz.Sequential(output)
    assert left is not right
    assert model.connections == ((0, 1), (0, 2), (1, 3), (2, 3))
    assert vz.Hyperspace.render(output, size=(440, 290)).tobytes() == vz.Hyperspace.render(model, size=(440, 290)).tobytes()



@pytest.mark.parametrize("renderer", [vz.Layered, vz.LeNet])
def test_linear_renderers_route_a_skip_from_custom_layer_calls(renderer) -> None:
    model = vz.Sequential(vz.Input((32, 8)), Residual())
    chained = vz.Sequential(*model.layers)
    traced = renderer.render(model)
    flattened = renderer.render(chained)
    assert traced.height > flattened.height


def test_rendering_a_model_output_keeps_block_boundaries_without_calling_it_again() -> None:
    class Counted(Residual):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def __call__(self, x: vz.Value) -> vz.Value:
            self.calls += 1
            return super().__call__(x)

    block = Counted()
    model = vz.Sequential(vz.Input((32, 8)), vz.Spacer(14), block)
    output = model()
    assert block.calls == 1
    settings = {"size": (680, 410)}
    assert vz.Hyperspace.render(output, **settings).tobytes() == vz.Hyperspace.render(model, **settings).tobytes()
    assert vz.Layered.render(output).tobytes() == vz.Layered.render(model).tobytes()
    assert block.calls == 1


def test_nested_sequences_keep_custom_block_labels_and_spacers() -> None:
    model = vz.Sequential(vz.Input((32, 8)), vz.Sequential(vz.Spacer(12), Residual()))
    direct = vz.Sequential(vz.Input((32, 8)), vz.Spacer(12), Residual())
    assert vz.Hyperspace.render(model).tobytes() == vz.Hyperspace.render(direct).tobytes()
    assert vz.Layered.render(model).tobytes() == vz.Layered.render(direct).tobytes()



def test_extending_a_captured_value_keeps_its_block_and_spacer() -> None:
    prefix = vz.Sequential(vz.Input((32, 8)), vz.Spacer(18), Residual())
    output = vz.Linear(4)(prefix())
    full = vz.Sequential(vz.Input((32, 8)), vz.Spacer(18), Residual(), vz.Linear(4))
    assert vz.Hyperspace.render(output).tobytes() == vz.Hyperspace.render(full).tobytes()
    assert vz.Layered.render(output).tobytes() == vz.Layered.render(full).tobytes()


def test_layer_class_filters_apply_to_inferred_built_ins() -> None:
    model = vz.Sequential(vz.Input((8,)), vz.Linear(4))
    with_classes = vz.Layered.render(model, type_ignore=[vz.Linear])
    with_kinds = vz.Layered.render(model, type_ignore=["Linear"])
    assert with_classes.tobytes() == with_kinds.tobytes()
    assert with_classes.width < vz.Layered.render(model).width
