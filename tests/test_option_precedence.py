import pytest
import vizion
from PIL import Image

def _build_layers():
    return [
        vizion.Layer("Input", (8, 8, 3), name="option_input"),
        vizion.Layer("Conv2D", (6, 6, 4), name="option_conv", kernel_size=3),
        vizion.Layer("Flatten", (144,), name="option_flatten"),
        vizion.Layer("Dense", (3,), name="option_output"),
    ]


def _render(mode, layers, via_show, **kwargs):
    if via_show:
        return vizion.show(layers, mode=mode, **kwargs)
    return getattr(vizion, mode + "_view")(layers, **kwargs)


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_default_spacing_overrides_preset(via_show):
    layers = _build_layers()
    image = _render("layered", layers, via_show, preset="compact", spacing=10)
    reference = vizion.layered_view(layers, spacing=10, padding=6)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_none_resets_option_caption(via_show):
    layers = _build_layers()
    image = _render(
        "layered", layers, via_show,
        options=vizion.LayeredOptions(text_callable=vizion.LAYERED_TEXT_CALLABLES["name"]),
        text_callable=None,
    )
    reference = vizion.layered_view(layers)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_false_resets_mapping_legend(via_show):
    layers = _build_layers()
    image = _render("layered", layers, via_show, options={"legend": True}, legend=False)
    reference = vizion.layered_view(layers)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_layered_options_forward_logo_fields(tmp_path):
    layers = _build_layers()
    icon = tmp_path / "options-logo.png"
    Image.new("RGBA", (24, 16), "magenta").save(icon)
    logos = [{"name": "Option logo", "file": str(icon), "layers": ["option_conv"], "axis": "z", "size": 0.4}]
    image = vizion.layered_view(
        layers,
        options=vizion.LayeredOptions(logo_groups=logos, logos_legend=True),
    )
    reference = vizion.layered_view(layers, logo_groups=logos, logos_legend=True)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_layered_explicit_image_defaults_override_mapping(tmp_path):
    layers = _build_layers()
    icon = tmp_path / "options-face.png"
    Image.new("RGBA", (24, 16), "magenta").save(icon)
    styles = {"option_conv": {"image": str(icon)}}
    image = vizion.layered_view(
        layers,
        options={"image_fit": "cover", "image_axis": "x", "styles": styles},
        image_fit="fill", image_axis="z",
    )
    reference = vizion.layered_view(layers, styles=styles, image_fit="fill", image_axis="z")
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_lenet_explicit_default_spacing_overrides_preset(via_show):
    layers = _build_layers()
    image = _render("lenet", layers, via_show, preset="compact", layer_spacing=40)
    reference = vizion.lenet_view(layers, layer_spacing=40, map_spacing=3, max_xy=180, padding=15)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_lenet_explicit_none_resets_seed(via_show):
    layers = _build_layers()
    image = _render("lenet", layers, via_show, options=vizion.LenetOptions(seed=7), seed=None)
    reference = vizion.lenet_view(layers, seed=None)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_lenet_explicit_true_resets_mapping_labels(via_show):
    layers = _build_layers()
    image = _render("lenet", layers, via_show, options={"top_label": False}, top_label=True)
    reference = vizion.lenet_view(layers, top_label=True)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_lenet_mapping_forwards_label_callables():
    layers = _build_layers()
    image = vizion.lenet_view(layers, options={
        "top_label_callable": lambda layer, shape: layer.name.upper(),
        "bottom_label_callable": lambda layer, shape: "custom bottom",
    })
    reference = vizion.lenet_view(
        layers,
        top_label_callable=lambda layer, shape: layer.name.upper(),
        bottom_label_callable=lambda layer, shape: "custom bottom",
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
