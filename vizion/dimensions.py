"""Translate batch-free output shapes into layered diagram dimensions."""

import math
import warnings

from .utils import self_multiply


def calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
                               one_dim_orientation='y', sizing_mode='accurate',
                               dimension_caps=None, relative_base_size=20):
    """Calculate rendered box dimensions for a layer shape.

    This helper converts a tensor shape into the pixel dimensions used by
    the layered renderer. The exact conversion depends on the
    selected sizing mode and on the scaling, minimum, and maximum bounds that
    accompany it.

    Parameters
    ----------
    shape : tuple
        Batch-free output shape to convert.
    scale_z : float
        Multiplier applied to depth-like dimensions before clamping.
    scale_xy : float
        Multiplier applied to width and height dimensions before clamping.
    max_z : int
        Upper bound for the rendered depth dimension.
    max_xy : int
        Upper bound for rendered width and height.
    min_z : int
        Lower bound for the rendered depth dimension.
    min_xy : int
        Lower bound for rendered width and height.
    one_dim_orientation : {'y', 'z'}, default='y'
        Axis used when rendering one-dimensional layers.
    sizing_mode : {'accurate', 'capped', 'balanced', 'logarithmic', 'relative'}, default='accurate'
        Strategy used to transform tensor dimensions into pixels.
    dimension_caps : mapping, optional
        Optional per-dimension caps used by sizing modes that support them.
    relative_base_size : int, default=20
        Base pixel size used by ``relative`` mode.

    Returns
    -------
    tuple of int
        Three-item tuple ``(x, y, z)`` describing the rendered dimensions.

    """
    dims = [1 if dim is None else dim for dim in shape]

    # Default fallback
    if not dims:
        return (min_xy, min_xy, min_z)

    # Setup caps
    channel_cap = dimension_caps.get('channels', max_z) if dimension_caps else max_z
    sequence_cap = dimension_caps.get('sequence', max_xy) if dimension_caps else max_xy

    def smart_scale(value, base_scale, min_val, cap_val):
        """
        Smart scaling that maintains relative proportions while preventing extremely large visualizations.
        This provides the relative scaling functionality that was promised by the relative_scaling parameter.
        """
        if value <= 64:
            # Small dimensions: use full scaling to make them visible
            return min(max(value * base_scale, min_val), cap_val)
        elif value <= 512:
            # Medium dimensions: reduce scaling to balance visibility and proportion
            return min(max(value * base_scale * 0.6, min_val), cap_val)
        elif value <= 2048:
            # Large dimensions: further reduce scaling but maintain relative differences
            return min(max(value * base_scale * 0.3, min_val), cap_val)
        else:
            # Very large dimensions: use logarithmic scaling but still maintain relativity
            log_scale = math.log10(value) * base_scale * 15
            return min(max(log_scale, min_val), cap_val)

    def log_scale(value, base_scale, min_val, cap_val):
        if value <= 1:
            return min_val
        log_val = math.log10(value) * base_scale * 20
        return min(max(log_val, min_val), cap_val)

    # Accurate mode: mirror original block
    if sizing_mode == 'accurate':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = min(max(dims[0] * scale_xy, min_xy), max_xy)
                return (min_xy, y, min_z)
            else:
                z = min(max(dims[0] * scale_z, min_z), max_z)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(dims[1] * scale_z, min_z), max_z)
            return (x, y, z)
        else:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(self_multiply(dims[2:]) * scale_z, min_z), max_z)
            return (x, y, z)

    # Capped mode
    elif sizing_mode == 'capped':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
                return (min_xy, y, min_z)
            else:
                z = max(min(dims[0] * scale_z, channel_cap), min_z)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
            y = max(min(dims[1] * scale_xy, sequence_cap), min_xy)
            z = max(min(dims[2] * scale_z if len(dims)>2 else min_z, channel_cap), min_z)
            return (x, y, z)
        else:
            x = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
            y = max(min(dims[1] * scale_xy, sequence_cap), min_xy)
            z = max(min(self_multiply(dims[2:]) * scale_z, channel_cap), min_z)
            return (x, y, z)

    # Balanced mode
    elif sizing_mode == 'balanced':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return (min_xy, int(y), min_z)
            else:
                z = smart_scale(dims[0], scale_z, min_z, channel_cap)
                return (min_xy, min_xy, int(z))
        else:
            x = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = smart_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = smart_scale(self_multiply(dims[2:]) if len(dims)>2 else 1, scale_z, min_z, channel_cap)
            return (int(x), int(y), int(z))

    # Logarithmic mode
    elif sizing_mode == 'logarithmic':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return (min_xy, int(y), min_z)
            else:
                z = log_scale(dims[0], scale_z, min_z, channel_cap)
                return (min_xy, min_xy, int(z))
        else:
            x = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = log_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = log_scale(self_multiply(dims[2:]) if len(dims)>2 else 1, scale_z, min_z, channel_cap)
            return (int(x), int(y), int(z))

    # Relative mode - True proportional scaling where each layer's size is directly proportional to its dimension
    elif sizing_mode == 'relative':
        def proportional_scale(dimension, relative_base_size, min_val, max_val):
            """
            Scale dimension proportionally where relative_base_size represents
            the visual size for dimension=1.

            Args:
                dimension (int): The dimension value to scale
                relative_base_size (int): Visual size (in pixels) for dimension=1
                min_val (int): Minimum allowed scaled value
                max_val (int): Maximum allowed scaled value

            Returns:
                int: Scaled dimension with true proportional relationships

            Formula: visual_size = dimension * relative_base_size
            """
            if dimension <= 0:
                return min_val

            # True proportional scaling: dimension * base_size
            scaled = dimension * relative_base_size

            # Apply min/max constraints while preserving proportionality as much as possible
            return max(min_val, min(scaled, max_val))

        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
                return (min_xy, y, min_z)
            else:
                z = proportional_scale(dims[0], relative_base_size, min_z, channel_cap)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(dims[1], relative_base_size, min_xy, sequence_cap)

            # For 2D layers, use the second dimension for z-scaling as well
            z = proportional_scale(dims[1], relative_base_size, min_z, channel_cap)
            return (x, y, z)
        else:
            # 3D+ layers: handle spatial dimensions and channels separately
            x = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(dims[1], relative_base_size, min_xy, sequence_cap)

            # For channels (typically dims[2:]), use product for z-dimension
            channel_product = self_multiply(dims[2:]) if len(dims) > 2 else 1
            z = proportional_scale(channel_product, relative_base_size, min_z, channel_cap)
            return (x, y, z)
    else:
        warnings.warn(
            f"Unknown sizing mode '{sizing_mode}'. Defaulting to accurate.",
            UserWarning,
            stacklevel=3
        )

        # Recursive call to handle accurate sizing
        return calculate_layer_dimensions(
            shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
            one_dim_orientation=one_dim_orientation, sizing_mode='accurate',
            dimension_caps=dimension_caps, relative_base_size=relative_base_size
        )

    # Fallback
    return (min_xy, min_xy, min_z)
