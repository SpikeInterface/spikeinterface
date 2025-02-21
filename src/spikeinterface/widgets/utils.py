from __future__ import annotations

from warnings import warn
import numpy as np


def get_some_colors(
    keys,
    color_engine="auto",
    map_name="gist_ncar",
    format="RGBA",
    shuffle=None,
    seed=None,
    margin=None,
    resample=True,
):
    """
    Return a dict of colors for given keys

    Parameters
    ----------
    color_engine : "auto" | "matplotlib" | "colorsys" | "distinctipy", default: "auto"
        The engine to generate colors
    map_name : str
        Used for matplotlib
    format: str, default: "RGBA"
        The output formats
    shuffle : bool or None, default: None
        Shuffle or not the colors.
        If None then:
        * set to True for matplotlib and colorsys
        * set to False for distinctipy
    seed: int or None, default: None
        Set the seed
    margin: None or int
        If None, put a margin to remove colors on borders of some colomap of matplotlib.
    resample : bool, dafult True
        For matplotlib, only resample the cmap to the number of keys + eventualy maring

    Returns
    -------
    dict_colors: dict
        A dict of colors for given keys.

    """
    try:
        import matplotlib.pyplot as plt

        HAVE_MPL = True
    except ImportError:
        HAVE_MPL = False

    try:
        import distinctipy

        HAVE_DISTINCTIPY = True
    except ImportError:
        HAVE_DISTINCTIPY = False

    assert color_engine in ("auto", "distinctipy", "matplotlib", "colorsys")

    possible_formats = ("RGBA",)
    assert format in possible_formats, f"format must be {possible_formats}"

    # select the colormap engine
    if color_engine == "auto":
        if HAVE_MPL:
            color_engine = "matplotlib"
        elif HAVE_DISTINCTIPY:
            # this is the third choice because this is very slow
            color_engine = "distinctipy"
        else:
            color_engine = "colorsys"

    if shuffle is None:
        # distinctipy then False
        shuffle = color_engine != "distinctipy"
        seed = 91

    N = len(keys)

    if color_engine == "distinctipy":
        colors = distinctipy.get_colors(N)
        # add the alpha
        colors = [color + (1.0,) for color in colors]

    elif color_engine == "matplotlib":
        # some map have black or white at border so +10

        cmap = plt.colormaps[map_name]
        if resample:
            if margin is None:
                margin = max(4, int(N * 0.08))
            cmap = cmap.resampled(N + 2 * margin)
        colors = [cmap(i + margin) for i, key in enumerate(keys)]

    elif color_engine == "colorsys":
        import colorsys

        colors = [colorsys.hsv_to_rgb(x * 1.0 / N, 0.5, 0.5) + (1.0,) for x in range(N)]

    if shuffle:
        rng = np.random.RandomState(seed=seed)
        inds = np.arange(N)
        rng.shuffle(inds)
        colors = [colors[i] for i in inds]

    dict_colors = dict(zip(keys, colors))

    return dict_colors


def get_unit_colors(
    sorting_or_analyzer_or_templates, color_engine="auto", map_name="gist_ncar", format="RGBA", shuffle=None, seed=None
):
    """
    Return a dict colors per units.
    """
    colors = get_some_colors(
        sorting_or_analyzer_or_templates.unit_ids,
        color_engine=color_engine,
        map_name=map_name,
        format=format,
        shuffle=shuffle,
        seed=seed,
    )
    return colors


def array_to_image(
    data,
    colormap="RdGy",
    clim=None,
    spatial_zoom=(0.75, 1.25),
    num_timepoints_per_row=30000,
    row_spacing=0.25,
    scalebar=False,
    sampling_frequency=None,
):
    """
    Converts a 2D numpy array (width x height) to a
    3D image array (width x height x RGB color).

    Useful for visualizing data before/after preprocessing

    Parameters
    ----------
    data : np.array
        2D numpy array
    colormap : str
        Identifier for a Matplotlib colormap
    clim : tuple or None
        The color limits. If None, the clim is the range of the traces
    spatial_zoom : tuple
        Tuple specifying width & height scaling
    num_timepoints_per_row : int
        Max number of samples before wrapping
    row_spacing : float
        Ratio of row spacing to overall channel height

    Returns
    -------
    output_image : 3D numpy array

    """
    import matplotlib.pyplot as plt

    from scipy.ndimage import zoom

    if clim is not None:
        assert len(clim) == 2, "'clim' should have a minimum and maximum value"
    else:
        clim = [np.min(data), np.max(data)]

    num_timepoints = data.shape[0]
    num_channels = data.shape[1]
    spacing = int(num_channels * spatial_zoom[1] * row_spacing)

    cmap = plt.colormaps[colormap]
    zoomed_data = zoom(data, spatial_zoom)
    num_timepoints_after_scaling, num_channels_after_scaling = zoomed_data.shape
    num_timepoints_per_row_after_scaling = int(np.min([num_timepoints_per_row, num_timepoints]) * spatial_zoom[0])

    scaled_data = zoomed_data
    scaled_data[scaled_data < clim[0]] = clim[0]
    scaled_data[scaled_data > clim[1]] = clim[1]
    scaled_data = (scaled_data - clim[0]) / np.ptp(clim)
    a = np.flip((cmap(scaled_data.T)[:, :, :3] * 255).astype(np.uint8), axis=0)  # colorize and convert to uint8

    num_rows = int(np.ceil(num_timepoints / num_timepoints_per_row))

    output_image = (
        np.ones(
            (num_rows * (num_channels_after_scaling + spacing), num_timepoints_per_row_after_scaling, 3), dtype=np.uint8
        )
        * 255
    )

    for ir in range(num_rows):
        i1 = ir * num_timepoints_per_row_after_scaling
        i2 = min(i1 + num_timepoints_per_row_after_scaling, num_timepoints_after_scaling)
        output_image[
            ir * (num_channels_after_scaling + spacing) : ir * (num_channels_after_scaling + spacing)
            + num_channels_after_scaling,
            : i2 - i1,
            :,
        ] = a[:, i1:i2, :]

    if scalebar:
        assert sampling_frequency is not None

        try:
            from PIL import Image, ImageFont, ImageDraw
        except ImportError:
            raise ImportError("To add a scalebar, you need pillow: >>> pip install pillow")
        import platform

        y_scalebar = output_image.shape[0] - 10
        fontsize = int(0.8 * spacing)
        num_time_points = np.min([num_timepoints_per_row, num_timepoints])
        row_ms = (num_time_points / sampling_frequency) * 1000

        try:
            if platform.system() == "Linux":
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize)
            else:
                font = ImageFont.truetype("arial.ttf", fontsize)
        except:
            print(f"Could not load font to use in scalebar. Scalebar will not be drawn.")
            font = None

        if font is not None:
            image = Image.fromarray(output_image)
            image_editable = ImageDraw.Draw(image)

            # bar should be around 1/5 of row and ultiple of 5ms
            if row_ms / 5 > 5:
                bar_ms = int(np.ceil((row_ms / 5) // 5 * 5))
                text_offset = 0.3
            else:
                bar_ms = int(np.ceil(row_ms / 5))
                text_offset = -0.1
            bar_px = bar_ms * num_time_points / row_ms

            x_offset = int(0.1 * num_time_points)

            image_editable.line((x_offset, y_scalebar, x_offset + bar_px, y_scalebar), fill=(0, 0, 0), width=10)
            image_editable.text(
                (x_offset + text_offset * (bar_px), y_scalebar - 1.1 * fontsize),
                text=f"{bar_ms}ms",
                font=font,
                fill=(0, 0, 0),
            )

            output_image = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(output_image.shape)

    return output_image


def make_units_table_from_sorting(sorting, units_table=None):
    """
    Make a DataFrame from sorting properties.
    Only for properties with ndim=1

    Parameters
    ----------
    sorting : Sorting
        The Sorting object
    units_table : None | pd.DataFrame
        Optionally a existing dataframe.

    Returns
    -------
    units_table : pd.DataFrame
        Table containing all columns.
    """

    if units_table is None:
        import pandas as pd

        units_table = pd.DataFrame(index=sorting.unit_ids)

    for col in sorting.get_property_keys():
        values = sorting.get_property(col)
        if values.dtype.kind in "iuUSfb" and values.ndim == 1:
            units_table.loc[:, col] = values

    return units_table


def make_units_table_from_analyzer(
    analyzer,
    extra_properties=None,
):
    """
    Make a DataFrame by aggregating :
      * quality metrics
      * template metrics
      * unit_position
      * sorting properties
      * extra columns

    This used in sortingview and spikeinterface-gui to display the units table in a flexible way.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    extra_properties : None | dict
        Extra columns given as dict.

    Returns
    -------
    units_table : pd.DataFrame
        Table containing all columns.
    """
    import pandas as pd

    all_df = []

    if analyzer.get_extension("unit_locations") is not None:
        locs = analyzer.get_extension("unit_locations").get_data()
        df = pd.DataFrame(locs[:, :2], columns=["x", "y"], index=analyzer.unit_ids)
        all_df.append(df)

    if analyzer.get_extension("quality_metrics") is not None:
        df = analyzer.get_extension("quality_metrics").get_data()
        all_df.append(df)

    if analyzer.get_extension("template_metrics") is not None:
        df = analyzer.get_extension("template_metrics").get_data()
        all_df.append(df)

    if len(all_df) > 0:
        units_table = pd.concat(all_df, axis=1)
    else:
        units_table = pd.DataFrame(index=analyzer.unit_ids)

    make_units_table_from_sorting(analyzer.sorting, units_table=units_table)

    if extra_properties is not None:
        for col, values in extra_properties.items():
            # the ndim = 1 is important because we need  column only for the display in gui.
            if values.dtype.kind in "iuUSfb" and values.ndim == 1:
                units_table.loc[:, col] = values
            else:
                warn(
                    f"Extra property {col} not added to the units table because it has ndim > 1 or dtype not supported",
                )

    return units_table
