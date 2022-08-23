Widgets module
==============

The :py:mod:`spikeinterface.widgets` module includes plotting function to visualize recordings,
sortings, waveforms, and more.

Since version 0.95.0, the :py:mod:`spikeinterface.widgets` module supports multiple backends:

* "matplotlib": static rendering based on the matplotlib package
* "ipywidgets": interactive rendering within a jupyter notebook using the ipywidgets package
* "sortingview": web-based and interactive rendering using the sortingview package.


Installation
------------

The backends are loaded at runtime and can be installed separately.

matplotlib
~~~~~~~~~~

.. code-block:: bash
    pip install matplotlib

ipywidgets
~~~~~~~~~~

.. code-block:: bash
    pip install matplotlib ipympl ipywidgets 

To enable interactive widgets in your notebook, add and run a cell with:

.. code-block:: python
    %matplotlib widget

sortingview
~~~~~~~~~~~

...

Checkout the :ref:`_sphx_glr_modules_widgets` tutorials for an overview of available widgets!
