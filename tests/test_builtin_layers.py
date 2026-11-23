import pytest
from PIL import Image

import vizion as vz
from vizion import LayerDefinitionError


def test_token_model_infers_shapes_and_renders_directly() -> None:
    model = vz.Sequential(
        vz.Tokens(size=128),
        vz.Embedding(num_embeddings=32000, features=256),
        vz.Attention(num_heads=8),
        vz.LayerNorm(),
        vz.Linear(out_features=32),
    )
    assert [layer.kind for layer in model] == ["Tokens", "Embedding", "Attention", "LayerNorm", "Linear"]
    assert [layer.output_shape for layer in model] == [(128,), (128, 256), (128, 256), (128, 256), (128, 32)]
    assert isinstance(vz.Hyperspace.render(model, size=(480, 300)), Image.Image)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_cnn_infers_shapes_and_preserves_receptive_field_metadata(data_format: str) -> None:
    shape = (3, 20, 23) if data_format == "channels_first" else (20, 23, 3)
    model = vz.Sequential(
        vz.Input(shape, data_format=data_format),
        vz.Conv2d(out_channels=12, kernel_size=(3, 5), strides=(2, 3), padding="VALID"),
        vz.ReLU(),
        vz.MaxPool2d(2),
        vz.Flatten(),
        vz.Linear(4),
    )
    expected_conv = (12, 9, 7) if data_format == "channels_first" else (9, 7, 12)
    expected_pool = (12, 4, 3) if data_format == "channels_first" else (4, 3, 12)
    assert [layer.output_shape for layer in model] == [shape, expected_conv, expected_conv, expected_pool, (144,), (4,)]
    assert model.layers[1].kernel_size == (3, 5)
    assert model.layers[3].pool_size == (2, 2)
    assert vz.LeNet.render(model).getcolors(maxcolors=1) is None


def test_spacer_keeps_the_current_shape_and_custom_layers_can_reset_it() -> None:
    model = vz.Sequential(vz.Layer("Custom", (5, 10)), vz.Spacer(12), vz.Linear(7))
    assert model.layers[-1].output_shape == (5, 7)


def test_attention_rejects_incompatible_head_count() -> None:
    with pytest.raises(LayerDefinitionError, match="divisible by num_heads"):
        vz.Sequential(vz.Input((8, 10)), vz.Attention(3))()


def test_attention_keeps_query_length_and_infers_prefix_from_memory() -> None:
    query = vz.Input((None, 5, 32))()
    memory = vz.Input((3, 96, 64))()
    assert vz.Attention(8, out_features=16)(query, memory).shape == (3, 5, 16)


def test_concatenate_infers_joined_axis_and_known_non_join_dimensions() -> None:
    assert vz.Concatenate()(vz.Input((None, 32))(), vz.Input((8, 16))()).shape == (8, 48)


def test_recurrent_layers_can_keep_or_reduce_the_sequence_axis() -> None:
    model = vz.Sequential(vz.Input((None, 32)), vz.GRU(16), vz.LSTM(8, return_sequences=False))
    assert [layer.output_shape for layer in model] == [(None, 32), (None, 16), (8,)]


def test_reshape_rejects_a_different_output_volume() -> None:
    with pytest.raises(LayerDefinitionError, match="preserve the number of elements"):
        vz.Sequential(vz.Input((6, 8)), vz.Reshape((49,)))()


@pytest.mark.parametrize("merge", [vz.Add(), vz.Concatenate(axis=-1)])
def test_merges_keep_1d_feature_map_metadata(merge: vz.Layer) -> None:
    left = vz.Layer("Signal", (3, 24), data_format="channels_first", spatial_1d=True)()
    right = vz.Layer("Branch", (3, 24), data_format="channels_first", spatial_1d=True)()
    assert merge(left, right).layer.spatial_1d
    assert merge(left, right).layer.data_format == "channels_first"


def test_merge_rejects_incompatible_channel_layouts() -> None:
    left = vz.Input((4, 4, 4))()
    right = vz.Input((4, 4, 4), data_format="channels_first")()
    with pytest.raises(LayerDefinitionError, match="channel layout"):
        vz.Add()(left, right)


def test_convolution_preserves_first_channel_axis_for_stacked_images() -> None:
    model = vz.Sequential(
        vz.Input((3, 8, 20, 23), data_format="channels_first"),
        vz.Conv2d(12, 3, in_channels=3),
        vz.AvgPool2d(2),
    )
    assert model.layers[1].output_shape == (12, 8, 20, 23)
    assert model.layers[2].output_shape == (12, 8, 10, 11)


def test_reshape_infers_one_unknown_axis() -> None:
    model = vz.Sequential(vz.Input((6,)), vz.Reshape((3, None)))
    assert model.layers[1].output_shape == (3, 2)
    with pytest.raises(LayerDefinitionError, match="preserve the number of elements"):
        vz.Sequential(vz.Input((6,)), vz.Reshape((4, None)))()
