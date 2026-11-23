# Vizion

Vizion draws neural network diagrams from layers you define in Python. It uses
Pillow, NumPy, and aggdraw. No machine learning framework is required.

## Install

From the repository root, run `python -m pip install .`. Vizion requires Python
3.10 or later.

## Draw a network

Pass ready-made layers to `Sequential` in the order you want them drawn.
Render the model directly. Vizion follows its layer calls to infer shapes and connections.

```python
from vizion import Attention, Embedding, Hyperspace, LayerNorm, Linear, Sequential, Tokens

model = Sequential(
    Tokens(size=128),
    Embedding(num_embeddings=32_000, features=256),
    Attention(num_heads=8),
    LayerNorm(),
    Linear(out_features=32),
)

Hyperspace.render(model).save("hyperspace.png")
```

Use `Input` to declare a batch-free shape for an image, vector, or other
tensor. Pooling and convolution preserve the input's channel layout.

```python
from vizion import Conv2d, Flatten, Hyperspace, Input, LeNet, Linear, MaxPool2d, Sequential

layers = Sequential(
    Input((32, 32, 1), name="image"),
    Conv2d(6, kernel_size=5, padding="VALID", name="conv1"),
    MaxPool2d(2, name="pool1"),
    Conv2d(16, kernel_size=5, padding="VALID", name="conv2"),
    MaxPool2d(2, name="pool2"),
    Flatten(),
    Linear(120, name="hidden"),
    Linear(10, name="classes"),
)

Hyperspace(d=3).render(layers, text_callable="name_shape", legend=True, spacing=24).save("layered.png")
LeNet.render(layers, seed=7).save("lenet.png")
```

Both renderers return a Pillow image. `Hyperspace(d=3)` uses the shaded boxes
and options of `Layered.render`. LeNet shows feature maps and receptive fields.

### Hyperspace view

Hyperspace projects each layer's actual output index box. The outline shows
the projected boundary, and the thin lines follow the tensor axes through
their middle indices. Each shape is fitted to its tile with a uniform scale
that preserves its internal proportions. Labels give the exact shape, and
arrows show the data flow between layers.

Singleton axes collapse to a point in their direction. Unknown extents are
marked `?` and omitted from the geometry. A tensor with a zero-length axis
has no index box.

### Custom layers

Subclass `Layer` and connect child layers in `__call__`. A call returns a
`Value` with an output shape and references to its input values. Renderers
read those connections from the result. Hyperspace labels the operations
inside a custom block with the block name.

```python
from vizion import Attention, Embedding, Hyperspace, Layer, Linear, Sequential, Tokens, Value


class Encoder(Layer):
    def __init__(self, width: int, heads: int):
        self.branch = Sequential(Attention(num_heads=heads), Linear(width))

    def __call__(self, x: Value) -> Value:
        return x + self.branch(x)


encoder = Sequential(
    Tokens(size=128),
    Embedding(features=256),
    Encoder(width=256, heads=8),
)

Hyperspace.render(encoder).save("encoder.png")
```

Each layer uses its actual output rank by default. Choose `Hyperspace(d=10)`
to use a camera of at least ten dimensions while retaining every actual
output axis. Change `phase` to rotate the projection.

### Layer API

Ready-made layers include `Input`, `Tokens`, `Embedding`, `Attention`,
`Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Reshape`,
`ReLU`, `GELU`, `Softmax`, `LayerNorm`, `BatchNorm`, `Dropout`, `LSTM`,
`GRU`, `Add`, and `Concatenate`.
For custom operations or explicit shapes, pass `Layer(kind, output_shape)`.

- `kind` is a string you choose, such as `"Conv2D"` or `"Dense"`.
- `output_shape` takes dimensions such as `(10,)` for a vector or
  `(28, 28, 6)` for height, width, and channels. Use `None` for an unknown
  dimension and `()` for a scalar.
- `name` gives an individual layer a display name and styling key.
- `data_format="channels_first"` places channels at the start of the shape
  instead of at the end. In LeNet, rank-two shapes with a kind containing
  `Conv1D`, `Pool1D`, or `Pooling1D` are drawn as 1D feature maps. Use
  `spatial_1d=True` on other layers in the same 1D pipeline, such as
  `Layer("Input", (128, 2), spatial_1d=True)`.
- `kernel_size` and `pool_size` let LeNet draw receptive-field patches. For
  example, use `3` for a square window or `(3, 5)` for height and width.
  For a 1D window, use `(3,)`.
- `Spacer(30)` adds 30 pixels between layers in layered view. LeNet skips it.

### Style a diagram

Use `styles` to override visual properties by kind or by individual layer
name. Name overrides take precedence. You can also filter layers by kind
with `type_ignore` or by position with `index_ignore`.

```python
from vizion import LayeredOptions, show

image = show(
    layers,
    mode="layered",
    preset="compact",
    options=LayeredOptions(legend=True, text_callable="name"),
    styles={
        "Conv2D": {"fill": "#68a7d6"},
        "classes": {"fill": "#ffa868"},
    },
)
image.save("styled.png")
```

For additional styling parameters, see the docstrings for
`layered_view` and `lenet_view`.

## Development

Install tests with `python -m pip install -e '.[test]'` and run
`python -m pytest`.

Vizion is based on the MIT-licensed renderers from Visualkeras by Paul
Gavrikov, Santosh Patapati, and contributors. [Visualkeras upstream](https://github.com/paulgavrikov/visualkeras)
