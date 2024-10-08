Preprocessing module
====================

Overview: the "chain" concept
-----------------------------

The :py:mod:`~spikeinterface.preprocessing` module includes preprocessing steps to apply before running a sorter.
Example processing steps include filtering, removing noise, removing bad channels, etc.
Preprocessors are *lazy*, meaning that no computation is performed until it is required (usually at the
spike sorting step). This enables one to build preprocessing chains to be applied in sequence to a
:code:`~spikeinterface.core.BaseRecording` object.
This is possible because each preprocessing step returns a new :code:`~spikeinterface.core.BaseRecording` that can be input to the next
step in the chain.

In this code example, we build a preprocessing chain with two steps:

1) bandpass filter
2) common median reference (CMR)

.. code-block:: python

    import spikeinterface.preprocessing import bandpass_filter, common_reference

    # recording is a RecordingExtractor object
    recording_f = bandpass_filter(recording=recording, freq_min=300, freq_max=6000)
    recording_cmr = common_reference(recording=recording_f, operator="median")

These two preprocessors will not compute anything at instantiation, but the computation will be "on-demand"
("on-the-fly") when getting traces.

.. code-block:: python

    traces = recording_cmr.get_traces(start_frame=100_000, end_frame=200_000)

Some internal sorters (see :ref:`modules/sorters:Internal Sorters`) can work directly on these preprocessed objects so there is no need to
save the object:

.. code-block:: python

    # here the spykingcircus2 sorter engine directly uses the lazy "recording_cmr" object
    sorting = run_sorter(sorter='spykingcircus2', recording=recording_cmr, sorter_name='spykingcircus2')

Most of the external sorters, however, will need a binary file as input, so we can optionally save the processed
recording with the efficient SpikeInterface :code:`save()` function:

.. code-block:: python

    recording_saved = recording_cmr.save(folder="/path/to/preprocessed", n_jobs=8, chunk_duration='1s')

In this case, the :code:`save()` function will process in parallel our original recording with the bandpass filter and
CMR, and save it to a binary file in the "/path/to/preprocessed" folder. The :code:`recording_saved` is yet another
:code:`~spikeinterface.core.BaseRecording` which maps directly to the newly created binary file, for very quick access.

**NOTE:** all sorters will automatically perform the saving operation internally.

Impact on recording dtype
-------------------------

By default the dtype of a preprocessed recording does not change the recording's dtype, even if, internally, the
computation is performed using a different dtype.
For instance if we have a :code:`int16`` recording, the application of a bandpass filter will preserve the original
dtype (unless specified otherwise):


.. code-block:: python

    import spikeinterface.extractors as se
    # spikeGLX is int16
    rec_int16 = se.read_spikeglx(folder_path"my_folder")
    # by default the int16 is kept
    rec_f = bandpass_filter(recording=rec_int16, freq_min=300, freq_max=6000)
    # we can force a float32 casting
    rec_f2 = bandpass_filter(recording=rec_int16, freq_min=300, freq_max=6000, dtype='float32')

Some scaling pre-processors, such as :code:`whiten()` or :code:`zscore()`, will force the output to :code:`float32`.

When converting from a :code:`float` to an :code:`int`, the value will first be rounded to the nearest integer.


Available preprocessing
-----------------------

We have many preprocessing functions that can be flexibly added to a pipeline.

The full list of preprocessing functions can be found here: :ref:`api_preprocessing`

Here is a full list of possible preprocessing steps, grouped by type of processing:

For all examples :code:`rec` is a :code:`RecordingExtractor`.


filter() / bandpass_filter() / notch_filter() / highpass_filter()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several variants of filtering (e.g., bandpass, highpass, notch).

Filtering steps are implemented using :code:`scipy.signal`.

Important aspects of filtering functions:
  * they use a margin internally to deal with border effects
  * they perform forward-backward filtering (:code:`filtfilt`)
  * they can use 'ba' or 'sos' mode

.. code-block:: python

    rec_f = bandpass_filter(recording=rec, freq_min=300, freq_max=6000)


* :py:func:`~spikeinterface.preprocessing.filter()`
* :py:func:`~spikeinterface.preprocessing.bandpass_filter()`
* :py:func:`~spikeinterface.preprocessing.notch_filter()`
* :py:func:`~spikeinterface.preprocessing.highpass_filter()`


common_reference()
^^^^^^^^^^^^^^^^^^

A very common operation to remove the noise is to re-reference traces.
This is implemented with the :code:`common_reference()` function.

There are various options when combining :code:`operator` and :code:`reference` arguments:
  * using "median" or "average" (average is faster, but median is less sensitive to outliers)
  * using "global" / "local" / "single" references

.. code-block:: python

    rec_cmr = common_reference(recording=rec, operator="median", reference="global")

* :py:func:`~spikeinterface.preprocessing.common_reference()`

phase_shift()
^^^^^^^^^^^^^^

Recording system often do not sample all channels simultaneously.
In fact, there is a small delay (less that a sampling period) in between channels.
For instance this is the case for Neuropixels devices.

Applying :code:`common_reference()` on this data does not correctly remove artifacts, since we first need to compensate
for these small delays! This is exactly what :code:`phase_shift()` does.

This function relies on an internal property of the recording called :code:`inter_sample_shift`.
For Neuropixels recordings (read with the :py:func:`~spikeinterface.extractors.read_spikeglx` or the
:py:func:`~spikeinterface.extractors.read_openephys` functions), the :code:`inter_sample_shift` is automatically loaded
from the metadata and set.

Calling :code:`phase_shift()` alone has almost no effect, but combined with :code:`common_reference()` it makes a real
difference on artifact removal.


.. code-block:: python

    rec_shift = phase_shift(recording=rec)
    rec_cmr = common_reference(recording=rec_shift, operator="median", reference="global")



CatGT and IBL destriping are both based on this idea (see :ref:`ibl_destripe`).


* :py:func:`~spikeinterface.preprocessing.phase_shift()`


normalize_by_quantile() /scale() / center() / zscore()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have several "scalers" to apply some gains and offsets on traces.

:code:`scale()` is the base function to apply user-defined gains and offsets to every channels.

:code:`zscore()` estimates median/mad (or mean/std) of each channel and then applies the scale function to obtain
centered with unitary variance on each channel.


.. code-block:: python

    rec_normed = zscore(recording=rec)

* :py:func:`~spikeinterface.preprocessing.normalize_by_quantile()`
* :py:func:`~spikeinterface.preprocessing.scale()`
* :py:func:`~spikeinterface.preprocessing.center()`
* :py:func:`~spikeinterface.preprocessing.zscore()`

whiten()
^^^^^^^^

Many sorters use this pre-processing step internally, but if you want to combine this operation with other preprocessing
steps, you can use the :code:`whiten()` implemented in SpikeInterface.
The whitenning matrix :code:`W` is constructed by estimating the covariance across channels and then inverting it.

The whitened traces are then the dot product between the traces and the :code:`W` matrix.

.. code-block:: python

    rec_w = whiten(recording=rec)


* :py:func:`~spikeinterface.preprocessing.whiten()`

clip() / blank_staturation()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can limit traces between a user-defined minimum and maximum using :code:`clip()` function.
The :code:`blank_staturation()` function is similar, but it automatically estimates the limits by using quantiles.

.. code-block:: python

    rec_w = clip(recording=rec, a_min=-250., a_max=260)

* :py:func:`~spikeinterface.preprocessing.clip()`
* :py:func:`~spikeinterface.preprocessing.blank_staturation()`


highpass_spatial_filter()
^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`highpass_spatial_filter()` is a preprocessing step introduced by the International Brain Laboratory [IBL_spikesorting]_.
It applies a filter in the spatial axis of the traces after ordering the channels by depth.
It is similar to common reference, but it can deal with "stripes" that are uneven across depth.
This preprocessing step can be super useful for long probes like Neuropixels.

This is part of the "destriping" from IBL (see :ref:`ibl_destripe`).

* :py:func:`~spikeinterface.preprocessing.highpass_spatial_filter()`


detect_bad_channels() / interpolate_bad_channels()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`detect_bad_channels()` can be used to detect bad channels with several methods, including an :code:`std`- or :code:`mad`-based
approach to detect bad channels with abnormally high power and the :code:`coherence+psd` method (introduced by [IBL_spikesorting]_),
which detects bad channels looking at both coherence with other channels and PSD power in the high-frequency range.

Note: The :code:`coherence+psd` method must be run on individual probes/shanks separately since it uses the coherence of the signal across the depth of the probe. See `Processing a Recording by Channel Group <https://spikeinterface.readthedocs.io/en/latest/how_to/process_by_channel_group.html?highlight=split_by>`_ for more information.

The function returns both the :code:`bad_channel_ids` and :code:`channel_labels`, which can be :code:`good`, :code:`noise`, :code:`dead`,
or :code:`out` (outside of the brain). Note that the :code:`dead` and :code:`out` are only available with the :code:`coherence+psd` method.

Bad channels can then either be removed from the recording using :code:`recording.remove_channels(bad_channel_ids)` or be
interpolated with the :code:`interpolate_bad_channels()` function (channels labeled as :code:`out` should always be removed):

.. code-block:: python

    # detect
    bad_channel_ids, channel_labels = detect_bad_channels(recording=rec)
    # Case 1 : remove then
    rec_clean = recording.remove_channels(remove_channel_ids=bad_channel_ids)
    # Case 2 : interpolate then
    rec_clean = interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids)


* :py:func:`~spikeinterface.preprocessing.detect_bad_channels()`
* :py:func:`~spikeinterface.preprocessing.interpolate_bad_channels()`

rectify()
^^^^^^^^^

This step returns traces in absolute values. It could be used to compute a proxy signal of multi-unit activity (MUA).

* :py:func:`~spikeinterface.preprocessing.rectify()`

remove_artifacts()
^^^^^^^^^^^^^^^^^^

Given an external list of trigger times,  :code:`remove_artifacts()` function can remove artifacts with several
strategies:

* replace with zeros (blank) :code:`'zeros'`
* make a linear (:code:`'linear'`) or cubic (:code:`'cubic'`) interpolation
* remove the median (:code:`'median'`) or average (:code:`'avereage'`) template (with optional time jitter and amplitude scaling correction)

.. code-block:: python

    rec_clean = remove_artifacts(recording=rec, list_triggers=[100, 200, 300], mode='zeros')


* :py:func:`~spikeinterface.preprocessing.remove_artifacts()`


astype() / unsigned_to_signed()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly to :code:`numpy.astype()`, the :code:`astype()` casts the traces to the desired :code:`dtype`:

.. code-block:: python

    rec_int16 = astype(recording=rec_float, dtype="int16")


For recordings whose traces are unsigned (e.g. Maxwell Biosystems), the :code:`unsigned_to_signed()` function makes them
signed by removing the unsigned "offset". For example, :code:`uint16` traces will be first upcast to :code:`uint32`, 2**15
is subtracted, and the traces are finally cast to :code:`int16`:


.. code-block:: python

    rec_int16 = unsigned_to_signed(recording=rec_uint16)

* :py:func:`~spikeinterface.preprocessing.astype()`
* :py:func:`~spikeinterface.preprocessing.unsigned_to_signed()`


zero_channel_pad()
^^^^^^^^^^^^^^^^^^

Pads a recording with extra channels that containing only zeros. This step can be useful when a certain shape is
required.

.. code-block:: python

    rec_with_more_channels = zero_channel_pad(parent_recording=rec, num_channels=128)

* :py:func:`~spikeinterface.preprocessing.zero_channel_pad()`


gaussian_filter()
^^^^^^^^^^^^^^^^^

Implementation of a gaussian filter for high/low/bandpass filters. Note that the the gaussian filter
response is not very steep.

.. code-block:: python

    # highpass
    rec_hp = gaussian_filter(recording=rec, freq_min=300, freq_max=None)
    # lowpass
    rec_lp = gaussian_filter(recording=rec, freq_min=None, freq_max=500)
    # bandpass
    rec_bp = gaussian_filter(recording=rec, freq_min=300, freq_max=2000)

* :py:func:`~spikeinterface.preprocessing.gaussian_filter()`


Motion/drift correction
^^^^^^^^^^^^^^^^^^^^^^^

Motion/drift correction is one of the most sophisticated preprocessing. See the :ref:`motion_correction` page for a full
explanation.



deepinterpolation() (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The step (experimental) applies the inference step of a DeepInterpolation denoiser model [DeepInterpolation]_.

* :py:func:`~spikeinterface.preprocessing.deepinterpolation()`


.. _ibl_destripe:

How to implement "IBL destriping" or "SpikeGLX CatGT" in SpikeInterface
-----------------------------------------------------------------------


SpikeGLX has a built-in function called `CatGT <https://billkarsh.github.io/SpikeGLX/help/dmx_vs_gbl/dmx_vs_gbl/>`_
to apply some preprocessing on the traces to remove noise and artifacts.
IBL also has a standardized pipeline for preprocessed traces a bit similar to CatGT which is called "destriping" [IBL_spikesorting]_.
In both these cases, the traces are entirely read, processed and written back to a file.

SpikeInterface can reproduce similar results without the need to write back to a file by building a *lazy*
preprocessing chain. Optionally, the result can still be written to a binary (or a zarr) file.


Here is a recipe to mimic the **IBL destriping**:

.. code-block:: python

    rec = read_spikeglx(folder_path='my_spikeglx_folder')
    rec = highpass_filter(recording=rec, n_channel_pad=60)
    rec = phase_shift(recording=rec)
    bad_channel_ids = detect_bad_channels(recording=rec)
    rec = interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids)
    rec = highpass_spatial_filter(recording=rec)
    # optional
    rec.save(folder='clean_traces', n_jobs=10, chunk_duration='1s', progres_bar=True)



Here is a recipe to mimic the **SpikeGLX CatGT**:

.. code-block:: python

    rec = read_spikeglx(folder_path='my_spikeglx_folder')
    rec = phase_shift(recording=rec)
    rec = common_reference(recording=rec, operator="median", reference="global")
    # optional
    rec.save(folder='clean_traces', n_jobs=10, chunk_duration='1s', progres_bar=True)


Of course, these pipelines can be enhanced and customized using other available steps in the
:py:mod:`spikeinterface.preprocessing` module!




Preprocessing on Snippets
-------------------------


Some preprocessing steps are available also for :py:class:`~spikeinterface.core.BaseSnippets` objects:

align_snippets()
^^^^^^^^^^^^^^^^

This function aligns waveform snippets.

* :py:func:`~spikeinterface.preprocessing.align_snippets()`



References
----------

.. [IBL_spikesorting] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022.

.. [DeepInterpolation] Lecoq, Jérôme, et al. "Removing independent noise in systems neuroscience data using DeepInterpolation." Nature methods 18.11 (2021): 1401-1408.
