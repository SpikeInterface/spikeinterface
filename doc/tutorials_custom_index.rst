.. This page provides a custom index to the 'Tutorials' page, rather than the default sphinx-gallery
.. generated page. The benefits of this are flexibility in design and inclusion of non-sphinx files in the index.
..
.. To update this index with a new documentation page
.. 1) Copy the grid-item-card and associated ".. raw:: html" section.
.. 2) change :link: to a link to your page. If this is an `.rst` file, point to the rst file directly.
..    If it is a sphinx-gallery generated file, format the path as separated by underscore and prefix `sphx_glr`,
..    pointing to the .py file. e.g. `tutorials/my/page.py` -> `sphx_glr_tutorials_my_page.py
.. 3) Change :img-top: to point to the thumbnail image of your choosing. You can point to images generated
..    in the sphinx gallery page if you wish.
.. 4) In the `html` section, change the `default-title` to your pages title and `hover-content` to the subtitle.

:orphan:

Tutorials
============

Longer form tutorials about using SpikeInterface. Many of these are downloadable
as notebooks or Python scripts so that you can "code along" with the tutorials.

If you're new to SpikeInterface, we recommend trying out the
:ref:`get_started/quickstart:Quickstart tutorial` first.

Updating from legacy
--------------------

.. toctree::
   :maxdepth: 1

   tutorials/waveform_extractor_to_sorting_analyzer

Core tutorials
--------------

These tutorials focus on the :py:mod:`spikeinterface.core` module.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Recording objects
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_1_recording_extractor.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_1_recording_extractor_thumb.png
      :img-alt: Recording objects
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Sorting objects
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_2_sorting_extractor.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_2_sorting_extractor_thumb.png
      :img-alt: Sorting objects
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Handling probe information
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_3_handle_probe_info.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_3_handle_probe_info_thumb.png
      :img-alt: Handling probe information
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: SortingAnalyzer
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_4_sorting_analyzer.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_4_sorting_analyzer_thumb.png
      :img-alt: SortingAnalyzer
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Append and/or concatenate segments
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_5_append_concatenate_segments.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_5_append_concatenate_segments_thumb.png
      :img-alt: Append/Concatenate segments
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Handle time information
      :link-type: ref
      :link: sphx_glr_tutorials_core_plot_6_handle_times.py
      :img-top: /tutorials/core/images/thumb/sphx_glr_plot_6_handle_times_thumb.png
      :img-alt: Handle time information
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Build full pipeline with dicts
      :link: how_to/build_pipeline_with_dicts.html
      :img-top: /images/logo.png
      :img-alt: Build full pipeline with dicts
      :class-card: gallery-card
      :text-align: center

Extractors tutorials
--------------------

The :py:mod:`spikeinterface.extractors` module is designed to load and save recorded and sorted data, and to handle probe information.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Read various formats
      :link: how_to/read_various_formats.html
      :img-top: how_to/read_various_formats_files/read_various_formats_12_0.png
      :img-alt: Read various formats
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Working with unscaled traces
      :link-type: ref
      :link: sphx_glr_tutorials_extractors_plot_2_working_with_unscaled_traces.py
      :img-top: /tutorials/extractors/images/thumb/sphx_glr_plot_2_working_with_unscaled_traces_thumb.png
      :img-alt: Unscaled traces
      :class-card: gallery-card
      :text-align: center

Quality metrics tutorial
------------------------

The :code:`spikeinterface.metrics.quality` module allows users to compute various quality metrics to assess the goodness of a spike sorting output.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Quality Metrics
      :link-type: ref
      :link: sphx_glr_tutorials_metrics_plot_3_quality_metrics.py
      :img-top: /tutorials/metrics/images/thumb/sphx_glr_plot_3_quality_metrics_thumb.png
      :img-alt: Quality Metrics
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Curation Tutorial
      :link-type: ref
      :link: sphx_glr_tutorials_metrics_plot_4_curation.py
      :img-top: /tutorials/metrics/images/thumb/sphx_glr_plot_4_curation_thumb.png
      :img-alt: Curation Tutorial
      :class-card: gallery-card
      :text-align: center

Automated curation tutorials
----------------------------

Learn how to curate your units using a trained machine learning model. Or how to create
and share your own model.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Model-based curation
      :link-type: ref
      :link: sphx_glr_tutorials_curation_plot_1_automated_curation.py
      :img-top: /tutorials/curation/images/sphx_glr_plot_1_automated_curation_002.png
      :img-alt: Model-based curation
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Train your own model
      :link-type: ref
      :link: sphx_glr_tutorials_curation_plot_2_train_a_model.py
      :img-top: /tutorials/curation/images/thumb/sphx_glr_plot_2_train_a_model_thumb.png
      :img-alt: Train your own model
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Upload your model to HuggingFaceHub
      :link-type: ref
      :link: sphx_glr_tutorials_curation_plot_3_upload_a_model.py
      :img-top: /images/hf-logo.svg
      :img-alt: Upload your model
      :class-card: gallery-card
      :text-align: center

Comparison tutorial
-------------------

The :code:`spikeinterface.comparison` module allows you to compare sorter outputs or benchmark against ground truth.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Sorter Comparison
      :link-type: ref
      :link: sphx_glr_tutorials_comparison_plot_5_comparison_sorter_weaknesses.py
      :img-top: /tutorials/comparison/images/thumb/sphx_glr_plot_5_comparison_sorter_weaknesses_thumb.png
      :img-alt: Sorter Comparison
      :class-card: gallery-card
      :text-align: center

Widgets tutorials
-----------------

The :code:`widgets` module contains several plotting routines (widgets) for visualizing recordings, sorting data, probe layout, and more.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: RecordingExtractor Widgets
      :link-type: ref
      :link: sphx_glr_tutorials_widgets_plot_1_rec_gallery.py
      :img-top: /tutorials/widgets/images/thumb/sphx_glr_plot_1_rec_gallery_thumb.png
      :img-alt: Recording Widgets
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: SortingExtractor Widgets
      :link-type: ref
      :link: sphx_glr_tutorials_widgets_plot_2_sort_gallery.py
      :img-top: /tutorials/widgets/images/thumb/sphx_glr_plot_2_sort_gallery_thumb.png
      :img-alt: Sorting Widgets
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Waveforms Widgets
      :link-type: ref
      :link: sphx_glr_tutorials_widgets_plot_3_waveforms_gallery.py
      :img-top: /tutorials/widgets/images/thumb/sphx_glr_plot_3_waveforms_gallery_thumb.png
      :img-alt: Waveforms Widgets
      :class-card: gallery-card
      :text-align: center

   .. grid-item-card:: Peaks Widgets
      :link-type: ref
      :link: sphx_glr_tutorials_widgets_plot_4_peaks_gallery.py
      :img-top: /tutorials/widgets/images/thumb/sphx_glr_plot_4_peaks_gallery_thumb.png
      :img-alt: Peaks Widgets
      :class-card: gallery-card
      :text-align: center

Download All Examples
---------------------

- :download:`Download all examples in Python source code </tutorials/tutorials_python.zip>`
- :download:`Download all examples in Jupyter notebooks </tutorials/tutorials_jupyter.zip>`
