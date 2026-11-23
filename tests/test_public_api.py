from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.parametrize("mode", ["layered", "lenet"])
def test_layers_render_to_file(mode: str, tmp_path: Path) -> None:
    from vizion import Layer, show

    layers = [
        Layer("Input", (28, 28, 1)),
        Layer("Conv2D", (26, 26, 8), name="conv", kernel_size=3),
        Layer("MaxPool2D", (13, 13, 8), pool_size=(2, 2)),
        Layer("Dense", (10,), name="classes"),
    ]
    target = tmp_path / f"{mode}.png"
    image = show(iter(layers), mode=mode, preset="compact", to_file=target)
    with Image.open(target) as saved:
        assert image.tobytes() == saved.tobytes()
        assert saved.getcolors(maxcolors=1) is None


@pytest.mark.parametrize("mode", ["layered", "lenet"])
def test_native_kind_and_name_styles(mode: str) -> None:
    from vizion import Layer, show

    layers = [Layer("Dense", (8,), name="hidden"), Layer("Dense", (4,), name="out")]
    image = show(
        layers,
        mode=mode,
        styles={"Dense": {"fill": "#ff00ff"}, "out": {"fill": "#00ffff"}},
    )
    colors = {color for _, color in image.getcolors(image.width * image.height)}
    assert {(255, 0, 255, 255), (0, 255, 255, 255)} <= colors


def test_batch_free_shapes_include_dynamic_and_scalar_dimensions() -> None:
    from vizion import LAYERED_TEXT_CALLABLES, Layer

    shape_label = LAYERED_TEXT_CALLABLES["shape"]
    assert shape_label(0, Layer("Sequence", (None, 8))) == ("? x 8", False)
    assert shape_label(0, Layer("Scalar", ())) == ("scalar", False)


def test_spacer_expands_layered_view() -> None:
    from vizion import Layer, Spacer, layered_view

    layers = [Layer("Input", (8,)), Layer("Dense", (4,))]
    normal = layered_view(layers)
    spaced = layered_view([layers[0], Spacer(40), layers[1]])
    assert spaced.width == normal.width + 40


@pytest.mark.parametrize("mode", ["layered", "lenet"])
def test_filtering_every_layer_produces_a_blank_image(mode: str, tmp_path: Path) -> None:
    from vizion import Layer, show

    target = tmp_path / "empty.png"
    image = show(
        [Layer("Dense", (4,))], mode=mode, type_ignore=["Dense"],
        background_fill="white", padding=5, to_file=target,
    )
    assert image.getcolors() == [(100, (255, 255, 255, 255))]
    with Image.open(target) as saved:
        assert saved.size == (10, 10)


@pytest.mark.parametrize("mode", ["layered", "lenet"])
def test_channel_order_preserves_spatial_geometry(mode: str) -> None:
    from vizion import Layer, show

    channels_last = [Layer("Conv2D", (11, 19, 7))]
    channels_first = [Layer("Conv2D", (7, 11, 19), data_format="channels_first")]
    image = show(channels_first, mode=mode)
    reference = show(channels_last, mode=mode)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_lenet_uses_pool_size_to_draw_receptive_field_patches() -> None:
    from vizion import Layer, lenet_view

    layers = [Layer("Input", (20, 20, 1)), Layer("Pool", (10, 10, 1), pool_size=2)]
    image = lenet_view(layers, patch_fill="magenta", seed=3)
    plain = lenet_view(layers, patch_fill="magenta", seed=3, draw_patches=False)
    assert any(color == (255, 0, 255, 255) for _, color in image.getcolors(image.width * image.height))
    assert all(color != (255, 0, 255, 255) for _, color in plain.getcolors(plain.width * plain.height))


def test_layered_group_accepts_overlapping_kind_name_and_identity_selectors() -> None:
    from vizion import Layer, layered_view

    layers = [Layer("Input", (12,)), Layer("Dense", (8,), name="first"), Layer("Dense", (4,))]
    group = {"name": "Dense block", "fill": "#bde5ff", "padding": 10}
    image = layered_view(layers, layered_groups=[{**group, "layers": ["Dense", "first", layers[1]]}])
    reference = layered_view(layers, layered_groups=[{**group, "layers": layers[1:]}])
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
    assert image.tobytes() != layered_view(layers).tobytes()


@pytest.mark.parametrize("kind", ["Conv1D", "MaxPooling1D"])
def test_lenet_1d_feature_maps_honor_channel_order(kind: str) -> None:
    from vizion import Layer, lenet_view

    geometry = []

    def capture_shape(layer, rendered_shape):
        geometry.append((rendered_shape.kind, rendered_shape.w, rendered_shape.c))
        return layer.kind

    layers = [Layer(kind, (12, 3)), Layer(kind, (3, 12), data_format="channels_first")]
    lenet_view(layers, top_label_callable=capture_shape)
    assert geometry
    assert set(geometry) == {("spatial", 12, 3)}


def test_lenet_draws_patches_from_1d_input_with_explicit_spatial_shape() -> None:
    from vizion import Layer, lenet_view

    layers = [
        Layer("Input", (20, 2), spatial_1d=True),
        Layer("Conv1D", (18, 8), kernel_size=(3,)),
    ]
    image = lenet_view(layers, patch_fill="magenta", seed=3)
    assert any(color == (255, 0, 255, 255) for _, color in image.getcolors(image.width * image.height))


@pytest.mark.parametrize("reversed_view", [False, True])
def test_layered_connector_style_changes_pixels(reversed_view: bool) -> None:
    from vizion import Layer, layered_view

    layers = [Layer("Input", (32, 32, 1)), Layer("Conv2D", (8, 8, 3))]
    options = {"draw_reversed": reversed_view, "spacing": 60, "connector_fill": "magenta"}
    narrow = layered_view(layers, connector_width=1, **options)
    wide = layered_view(layers, connector_width=4, **options)
    magenta = (255, 0, 255, 255)
    narrow_count = sum(count for count, color in narrow.getcolors(narrow.width * narrow.height) if color == magenta)
    wide_count = sum(count for count, color in wide.getcolors(wide.width * wide.height) if color == magenta)
    assert wide_count > narrow_count


def test_layered_caption_index_matches_the_input_sequence() -> None:
    from vizion import Layer, Spacer, layered_view

    layers = [Layer("Input", (8,)), Spacer(12), Layer("Dense", (4,))]
    indices = []

    def caption(index, layer):
        indices.append(index)
        return layer.kind, False

    layered_view(layers, index_ignore=[0], text_callable=caption)
    assert indices == [2]


def test_lenet_labels_unknown_dimensions_as_unknown() -> None:
    from vizion import Layer, lenet_view

    layers = [Layer("Input", (None, 28, 3))]
    image = lenet_view(layers, top_label=False)
    reference = lenet_view(
        layers, top_label=False, bottom_label_callable=lambda layer, shape: "? x 28 x 3",
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_hyperspace_exports_a_monochrome_wireframe(tmp_path: Path) -> None:
    from vizion import Layer, show

    layers = [Layer("Input", (16, 8, 6, 4)), Layer("Mix", (12, 8, 5, 4, 2))]
    output = tmp_path / "hyperspace.png"
    image = show(layers, mode="hyperspace", size=(480, 300), labels=False, to_file=output)
    with Image.open(output) as saved:
        assert saved.size == (480, 300)
        assert saved.tobytes() == image.tobytes()
    pixels = {color for _, color in image.getcolors(image.width * image.height)}
    assert len(pixels) > 1
    assert all(r == g == b and alpha == 255 for r, g, b, alpha in pixels)


def test_hyperspace_projection_tracks_high_rank_shape_without_labels() -> None:
    from vizion import Hyperspace, Input

    shape = (2, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    reshaped = (4, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    settings = {"size": (320, 260), "labels": False, "dimensions": 10}

    original = Hyperspace.render(Input(shape), **settings)
    assert original.tobytes() != Hyperspace.render(Input(reshaped), **settings).tobytes()
    assert original.tobytes() != Hyperspace.render(Input((20,)), **settings).tobytes()


def test_hyperspace_dimensions_and_phase_change_the_camera() -> None:
    from vizion import HyperspaceOptions, Layer, hyperspace_view

    layers = [Layer("Input", (16, 2, 4, 8, 3))]
    options = HyperspaceOptions(size=(420, 260), labels=False)
    automatic = hyperspace_view(layers, options=options)
    projected = hyperspace_view(layers, options=options, dimensions=5)
    smaller_camera = hyperspace_view(layers, options=options, dimensions=4)
    larger_camera = hyperspace_view(layers, options=options, dimensions=6)
    rotated = hyperspace_view(layers, options=options, dimensions=5, phase=1.0)
    assert automatic.tobytes() == projected.tobytes()
    assert smaller_camera.tobytes() == projected.tobytes()
    assert larger_camera.tobytes() != projected.tobytes()
    assert rotated.tobytes() != projected.tobytes()


@pytest.mark.parametrize("dimensions", [2, 11])
def test_hyperspace_rejects_unsupported_projection_dimensions(dimensions: int) -> None:
    from vizion import Layer, hyperspace_view

    with pytest.raises(ValueError, match="dimensions"):
        hyperspace_view([Layer("Input", (2,))], dimensions=dimensions)


def test_hyperspace_accepts_large_finite_orbit_angles() -> None:
    from vizion import Layer, hyperspace_view

    image = hyperspace_view([Layer("Input", (4,))], phase=1e308, size=(180, 120), labels=False)
    assert image.getcolors(maxcolors=1) is None


def test_hyperspace_three_dimensions_renders_the_layered_view(tmp_path: Path) -> None:
    from vizion import Conv2d, Hyperspace, Input, Layered, LayeredOptions, Sequential, show

    model = Sequential(Input((28, 28, 1)), Conv2d(8, 3))
    target = tmp_path / "layered.png"
    options = LayeredOptions(to_file=target, text_callable="name_shape", spacing=25)
    expected = Layered.render(model, text_callable="name_shape", spacing=25)

    configured = Hyperspace(d=3).render(model, options=options)
    with Image.open(target) as saved:
        assert saved.tobytes() == expected.tobytes()
    assert configured.tobytes() == expected.tobytes()
    assert Hyperspace.render(model, d=3, options=options).tobytes() == expected.tobytes()
    assert show(model, mode="hyperspace", d=3, options=options).tobytes() == expected.tobytes()


def test_hyperspace_rejects_empty_preset_name() -> None:
    from vizion import Layer, hyperspace_view

    with pytest.raises(ValueError, match="Unknown hyperspace preset"):
        hyperspace_view([Layer("Input", (4,))], preset="")


@pytest.mark.parametrize("mode", ["layered", "lenet", "hyperspace"])
def test_sequential_accepts_layers_as_positional_arguments(mode: str) -> None:
    from vizion import Layer, Sequential, show

    input_layer = Layer("Input", (32, 8), name="tokens")
    output_layer = Layer("Projection", (32, 16), name="features")
    model = Sequential(input_layer, output_layer)
    assert model.layers == (input_layer, output_layer)
    assert len(model) == 2
    assert show(model, mode=mode).tobytes() == show([input_layer, output_layer], mode=mode).tobytes()
