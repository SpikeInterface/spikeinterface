.. _modulewidgets:

Widgets module
==============

The :py:mod:`spikeinterface.widgets` module includes plotting function to visualize recordings,
sortings, waveforms, and more.

Since version 0.95.0, the :py:mod:`spikeinterface.widgets` module supports multiple backends:

* "matplotlib": static rendering using the `matplotlib <https://matplotlib.org/>`_ package
* "ipywidgets": interactive rendering within a jupyter notebook using the `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_ package
* "sortingview": web-based and interactive rendering using the `sortingview <https://github.com/magland/sortingview>`_ and `FIGURL <https://github.com/flatironinstitute/figurl>`_ packages.


Installaling backends
---------------------

The backends are loaded at run-time and can be installed separately. Alternatively, all dependencies from all 
backends can be installed with:

.. code-block:: bash

   pip install spikeinterface[widgets]


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

    pip install sortingview figurl-jupyter

Internally, the processed data to be rendered are uploaded to a public bucket in the cloud, so that they
can be visualized via the web. 
To set up the backend, you need to authenticate to `kachery-cloud` using your google account by running 
the following command (you will be prompted a link):

.. code-block:: bash

    kachery-cloud-init

Finally, if you wish to set up another cloud provider, follow the instruction from the 
`kachery-cloud <https://github.com/flatironinstitute/kachery-cloud>`_ package ("Using your own storage bucket").

***Jupyter backend***


Usage
-----

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

TODO: Static figure

GIF

Screenshot * link

Backend-specific 

Sortingview, combination

To inspect which backends are available for each function and what are the additional backend-specific 
arguments you can use the following notation:

.. code-block:: python
    
    sw.plot_timeseries?

# TODO list of all plots
