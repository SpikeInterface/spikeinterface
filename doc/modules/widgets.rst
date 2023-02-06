.. _modulewidgets:


TODO restart this page from scratch

Widgets module
==============

The :py:mod:`spikeinterface.widgets` module includes plotting function to visualize recordings,
sortings, waveforms, and more.

Since version 0.95.0, the :py:mod:`spikeinterface.widgets` module supports multiple backends:

* "matplotlib": static rendering using the `matplotlib <https://matplotlib.org/>`_ package
* "ipywidgets": interactive rendering within a jupyter notebook using the `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_ package
* "sortingview": web-based and interactive rendering using the `sortingview <https://github.com/magland/sortingview>`_ and `FIGURL <https://github.com/flatironinstitute/figurl>`_ packages.


Installation
------------

The backends are loaded at run-time and can be installed separately.

matplotlib
~~~~~~~~~~

The "matplotlib" backend (default) uses :code:`matplotlib` to generate static figures. 

To install it, run:

.. code-block:: bash

   pip install matplotlib

ipywidgets
~~~~~~~~~~

The "ipywidgets" backend allows users to interact with the plot, for example, by selecting units or 
scrolling through a time series.

To install it, run:

.. code-block:: bash

    pip install matplotlib ipympl ipywidgets 

To enable interactive widgets in your notebook, add and run a cell with:

.. code-block:: python

    %matplotlib widget

sortingview
~~~~~~~~~~~

The "sortingview" backends generates web-based and shareable links that can be viewed in the browser.

To install it, run:

.. code-block:: bash

    pip install sortingview

Internally, the processed data to be rendered are uploaded to a public bucket in the cloud, so that they
can be visualized via the web. 
To set up the backend, you need to authenticate to `kachery-cloud` using your google account by running 
the following command (you will be prompted a link):

.. code-block:: bash

    kachery-cloud-init

Finally, if you wish to set up another cloud provider, follow the instruction from the 
`kachery-cloud <https://github.com/flatironinstitute/kachery-cloud>`_ package ("Using your own storage bucket").


Examples
--------

You can specify which backend to use with the :code:`backend` argument. In addition, each backend 
comes with specific arguments that can be set when calling the plotting function.

.. code-block:: python

    import spikeinterface.extractors as se
    import spikeinterface.widgets as sw

    # recording is a BaseRecording object
    recording = se.read_spikeglx("spikeglx-folder")

    # matplotlib backend
    sw.plot_timeseries(recording, backend="matplotlib")

    # ipywidgets backend
    sw.plot_timeseries(recording, backend="ipywidgets")

    # sortingview backend
    sw.plot_timeseries(recording, backend="sortingview")

To inspect which backends are available for each function and what are the additional backend-specific 
arguments you can use the following notation:

.. code-block:: python
    
    sw.plot_timeseries?

which prints:

.. code-block:: bash

    Signature:
    si.plot_timeseries(
        recording,
        segment_index=None,
        channel_ids=None,
        order_channel_by_depth=False,
        time_range=None,
        mode='auto',
        cmap='RdBu',
        show_channel_ids=False,
        color_groups=False,
        color=None,
        clim=None,
        tile_size=512,
        seconds_per_row=0.2,
        with_colorbar=True,
        backend=None,
        **backend_kwargs,
    )
    Docstring:     
    Plots recording timeseries.

    Parameters
    ----------
    recording: RecordingExtractor or dict or list
        The recording extractor object
        If dict (or list) then it is a multi layer display to compare some processing
        for instance
    segment_index: None or int
        The segment index.
    channel_ids: list
        The channel ids to display.
    order_channel_by_depth: boolean
        Reorder channel by depth.
    time_range: list
        List with start time and end time
    mode: 'line' or 'map' or 'auto'
        2 possible mode:
            * 'line' : classical for low channel count
            * 'map' : for high channel count use color heat map
            * 'auto' : auto switch depending the channel count <32ch
    cmap: str default 'RdBu'
        matplotlib colormap used in mode 'map'
    show_channel_ids: bool
        Set yticks with channel ids
    color_groups: bool
        If True groups are plotted with different colors
    color:   str default: None
        The color used to draw the traces.
    clim: None, tuple, or dict
        When mode='map' this control color lims. 
        If dict, keys should be the same as recording keys
    with_colorbar: bool default True
        When mode='map' add colorbar
    tile_size: int
        For sortingview backend, the size of each tile in the rendered image
    seconds_per_row: float
        For 'map' mode and sortingview backend, seconds to render in each row

    Returns
    -------
    W: TimeseriesWidget
        The output widget

    Backends
    --------

    backends: str
        ['matplotlib', 'sortingview', 'ipywidgets']
    backend_kwargs: kwargs

        matplotlib:
        - figure: Matplotlib figure. When None, it is created. Default None
        - ax: Single matplotlib axis. When None, it is created. Default None
        - axes: Multiple matplotlib axes. When None, they is created. Default None
        - ncols: Number of columns to create in subplots.  Default 5
        - figsize: Size of matplotlib figure. Default None
        - figtitle: The figure title. Default None
        sortingview:
        - generate_url: If True, the figurl URL is generated and printed. Default is True
        - figlabel: The figurl figure label. Default None
        ipywidgets:
        - width_cm: Width of the figure in cm (default 10)
        - height_cm: Height of the figure in cm (default 6)


Checkout the :ref:`_sphx_glr_modules_widgets` tutorials for an overview of available widgets!
