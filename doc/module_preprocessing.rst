Preprocessing module
====================

The :py:mod:`~spikeinterface.toolkit.preprocessing` module includes preprocessing steps to apply before spike
sorting. Preprocessors are *lazy*, meaning that no computation is performed until it is required (usually at the
spike sorting step). This enables one to build preprocessing chains to be applied in sequence to a
:code:`RecordingExtractor` object.
This is possible because each preprocessing step returns a new :code:`RecordingExtractor` that can be input to the next
step in the chain.

In this code example, we build a preprocessing chain with 2 steps:

1) bandpass filter
2) common median reference (CMR)

.. code-block:: python

    import spikeinterface.toolkit as st

    # recording is a RecordingEctractor object
    recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_cmr = st.common_reference(recording_f, operator="median")

After preprocessing, we can optionally save the processed recording with the efficient SI :code:`save()` function:

.. code-block:: python

    recording_saved = recording_cmr.save(folder="preprocessed", n_jobs=8, total_memory="2G")

In this case, the :code:`save()` function will process in parallel our original recording with the bandpass filter and
CMR and save it to a binary file in the "preprocessed" folder. The :code:`recording_saved` is yet another
:code:`RecordignExtractor` which maps directly to the newly created binary file, for very quick access.






