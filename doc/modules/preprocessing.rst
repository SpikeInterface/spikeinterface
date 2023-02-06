Preprocessing module
====================

Overview : chain concept
------------------------

The :py:mod:`~spikeinterface.preprocessing` module includes preprocessing steps to apply before run a sorter.
The main idea is to to filter, remove noise, remove bad channels.
Preprocessors are *lazy*, meaning that no computation is performed until it is required (usually at the
spike sorting step). This enables one to build preprocessing chains to be applied in sequence to a
:code:`RecordingExtractor` object.
This is possible because each preprocessing step returns a new :code:`RecordingExtractor` that can be input to the next
step in the chain.

In this code example, we build a preprocessing chain with 2 steps:

1) bandpass filter
2) common median reference (CMR)

.. code-block:: python

    import spikeinterface.preprocessing import bandpass_filter, common_reference

    # recording is a RecordingEctractor object
    recording_f = bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_cmr = common_reference(recording_f, operator="median")

This 2 preprocessor haven't been computed yet.
The computation will be "on-demand" aka "on-the-fly" when getting traces.

.. code-block:: python

    traces = recording_cmr.get_traces(start_frame=100_000, end_frame=200_000)

Some internal sorter can work directly in this preprocessed objects so there is no need to save then.

.. code-block:: python

    # here the circus2 sorter engine use directly the lazy "recording_cmr" object
    sorting = run_sorter(recording_cmr, 'spykingcircus2')

But many external sorter will need a binary file as input, so we can optionally save the processed recording
with the efficient SI :code:`save()` function:

.. code-block:: python

    recording_saved = recording_cmr.save(folder="/path/to/reprocessed", n_jobs=8, chunk_duration='1s')

In this case, the :code:`save()` function will process in parallel our original recording with the bandpass filter and
CMR and save it to a binary file in the "preprocessed" folder. The :code:`recording_saved` is yet another
:code:`RecordignExtractor` which maps directly to the newly created binary file, for very quick access.


impact on dtype
---------------

By default the dtype of a preprocessing do not change the dtype of a recording.
Even if the internally the computation is done using float
For instance if we have a 'int16' recording a applying filtering will conserve the 'int16' dtype.


.. code-block:: python

    # spikeglx is int16
    rec_int16 = read_spikeglx("my_folder")
    # by default the int16 is kept
    rec_f = bandpass_filter(rec_int16, freq_min=300, freq_max=6000)
    # but we can force to float32
    rec_f2 = bandpass_filter(rec_int16, freq_min=300, freq_max=6000, dtype='float32')

Some scaling pre processors force the output to float32. For instance `whiten()` or `zscore()`


Available preprocessing
-----------------------

We have many preprocessing class they are used through a function.

They can of course be combined/chain.

The full list of preprocessing function is here :ref:`api_preprocessing`

Here a full list of possible preprocessings:


filter() / bandpass_filter() / notch_filter() / highpass_filter()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have some variant of filtering : band pass, high pass, ....
Filtering is implemented using `scipy.signal`
Importants:
  * they use a margin internally for chunk_duration
  * they are doing forward-backward filtering (filtfilt)
  * they can use 'ba' or 'sos' mode


.. code-block:: python

    rec_f = bandpass_filter(rec, freq_min=300, freq_max=6000)


:py:func: `spikeinterface.preprocessing.filter()`
:py:func: `spikeinterface.preprocessing.bandpass_filter()`
:py:func: `spikeinterface.preprocessing.notch_filter()`
:py:func: `spikeinterface.preprocessing.highpass_filter()`


common_reference()
^^^^^^^^^^^^^^^^^^

A very common operation to remove the noise is to re-reference tarces.
This is implemented with the `common_reference()` function.

There are some various usages when combining `operator` and `reference`:
  * using "median" or "average" : this have a big impact in the speed
  * using "global" / "local" /"single"

.. code-block:: python

    rec_cmr = common_reference(rec, operator="median", reference="global")

:py:func: `spikeinterface.preprocessing.common_reference()`

phase_shift()
^^^^^^^^^^^^^^

Recording system do not sample all channels simultaneously.
Infact, there is a small delay (less that a sampling period) in between channels.
For instance this is the case for neuropixel devices.

Applying `common_reference()` on this data do not remove correctly artifacts we need to compensate first the
small delays! This is exactly what `phase_shift()` compensate the small delays.

This rely on an internal property of the recording : "inter_sample_shift".

Calling `phase_shift()` alone have almost no effect but bombined with `common_reference()` make a real differences
on artifact removal.


.. code-block:: python

    rec_shift = phase_shift(rec)
    rec_cmr = common_reference(rec_shift, operator="median", reference="global")



CatGT and IBL destripe are based on this idea of fft data shifting see  :ref:`ibl_destripe`.


:py:func: `spikeinterface.preprocessing.phase_shift()`


normalize_by_quantile() /scale() / center() / zscore()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have several "scaler" to apply some gains and offsets on traces.

`scale()` is the neutral function to apply gain and offset to every channels.

`zscore()` estimate median/mad (or mean/std) and then apply the scale to get cenetred and variance of 1 on 
every channels.


.. code-block:: python

    rec_normed = zscore(rec)

:py:func: `spikeinterface.preprocessing.normalize_by_quantile()`
:py:func: `spikeinterface.preprocessing.scale()`
:py:func: `spikeinterface.preprocessing.center()`
:py:func: `spikeinterface.preprocessing.zscore()`

whiten()
^^^^^^^^

Many sorter use this pre processing step internally but you want to combine this to others preprocessing steps, 
you can compute the whitening with spikeinterface.
The whitenning matrix W is constructed by estimating the covariance across channels and then inverse it.

the whiten traces are then the dot product of traces by this W matrix.

.. code-block:: python

    rec_w = whiten(rec)


:py:func: `spikeinterface.preprocessing.whiten()`

clip() / blank_staturation()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can limit traces between min and max using `clip()`.
`blank_staturation()` do the same but auto estimate the limts by using quantile.

.. code-block:: python

    rec_w = clip(rec, a_min=-250., a_max=260)

:py:func: `spikeinterface.preprocessing.clip()`
:py:func: `spikeinterface.preprocessing.blank_staturation()`

highpass_spatial_filter()
^^^^^^^^^^^^^^^^^^^^^^^^^

`highpass_spatial_filter()` is a preprocessing step introduced by Olivier Winter.
It apply a filter on the spatial axis of the tarces after ordering then.
It is some kind of spatial detending. This can be usefull for big probe like neuropixel.

This is part of the "destripe" from IBL see :ref:`ibl_destripe`..

:py:func: `spikeinterface.preprocessing.highpass_spatial_filter()`

detect_bad_channels() / interpolate_bad_channels()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have function to detect bad channels with several methods.

Then it can be use either to remove channels from the recording using `recording.remove_channels(bad_channel_ids)`
or to interpolate then.

.. code-block:: python

    # detect
    bad_channel_ids = detect_bad_channels(rec)
    # Case 1 : remove then
    rec_clean = recording.remove_channels(bad_channel_ids)
    # Case 2 : interpolate then
    rec_clean = interpolate_bad_channels(rec, bad_channel_ids)


:py:func: `spikeinterface.preprocessing.detect_bad_channels()`
:py:func: `spikeinterface.preprocessing.interpolate_bad_channels()`

rectify()
^^^^^^^^^

To make traces absolute we can use the `rectify()` function.

:py:func: `spikeinterface.preprocessing.rectify()`

remove_artifacts()
^^^^^^^^^^^^^^

Given an external list of trigger time `remove_artifacts()` can remove or at least blank artifacts with several strategies:
put zeros, remove median or average, make linear or cubic interpolation.


.. code-block:: python

    rec_clean = remove_artifacts(rec, list_triggers)


:py:func: `spikeinterface.preprocessing.remove_artifacts()`


zero_channel_pad()
^^^^^^^^^^^^^^^^^^

Pads a recording with channels that contain only zero.

.. code-block:: python

    rec_with_more_channels = zero_channel_pad(rec, 128)

:py:func: `spikeinterface.preprocessing.zero_channel_pad()`

deepinterpolation() (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Very experimental port of the deep interpolation denoiser publish by Jerome Lecoq in 2021

:py:func: `spikeinterface.preprocessing.deepinterpolation()`



.. _ibl_destripe:

Howto implement "IBL destripe" or "spikeglx CatGT" in spiekinterface
--------------------------------------------------------------------



SpikeGLX have a built-in function called **CatGT** to apply some preprocessing on traces to remove noise and artifacts.

IBL also have a standardized pipeline to preprocessed traces a bit similar to CatGT which is called **"destripe"**.

In theses 2 cases the traces are entiely read, processed and written back to a file.

spikeinterface can build similar results without the need to write back to a file by building a preprocessing chain.
Optionaly, the result can still be writen to binary (or zarr) file.


Here a recipe to mimic **ibl destriping**:

.. code-block:: python

    rec = read_spikeglx('my_spikeglx_folder')
    rec = highpass_filter(rec)
    rec = phase_shift(rec)
    bad_channel_ids = detect_bad_channels(rec)
    rec = interpolate_bad_channels(rec, bad_channel_ids)
    rec = highpass_spatial_filter(rec)
    # optional
    rec.save(folder='clean_traces', n_jobs=10, chunk_duration='1s', progres_bar=True)



Here a recipe to mimic **spikeglx CatGt**:

.. code-block:: python

    rec = read_spikeglx('my_spikeglx_folder')
    rec = phase_shift(rec)
    rec = common_reference(rec, operator="median", reference="global")
    # optional
    rec.save(folder='clean_traces', n_jobs=10, chunk_duration='1s', progres_bar=True)


Or course anyone can build its own custum preprocessing mixing all possible function availables in `spikeinterface.preprocessing`



Preprocessing on snippets
-------------------------

Some preprocessing work on top of snippet object.


align_snippets()
^^^^^^^^^^^^^^

:py:func: `spikeinterface.preprocessing.align_snippets()`
