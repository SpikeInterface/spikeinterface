Toolkit module
==============

The :code:`toolkit` module includes tools to process SI objects throughout your analysis.


Preprocessing
-------------

The :code:`toolkit.preprocessing` sub-module includes preprocessing steps to apply before spike sorting.
Preprocessors are *lazy*, meaning that no computation is performed until it is required (usually at the spike sorting
step). This enables one to build preprocessing chains to be applied in sequence to a :code:`RecordingExtractor` object.
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


Postprocessing
--------------

After spike sorting, we can use the :code:`toolkit.postprocessing` sub-module to further post-process the spike sorting
output. Most of the post-processing functions require a :code:`WaveformExtractor` as input. Available postprocessing
tools are:

* compute principal component scores
* compute template similarity
* compute template waveform metrics
* get amplitudes for each spikes
* compute auto- and cross-correlogram


Quality Metrics
---------------

Quality metrics allows to quantitatively assess to *goodness* of a spike sorting output. 
The :code:`toolkit.qualitymetrics` sub-module includes functions to compute a large variety of available metrics.
All of the metrics currently implemented in spikeInterface are *per unit* (pairwise metrics do appear in the literature).

Each metric aims to identify some quality of the unit.
Contamination metrics (also sometimes called 'false positive' or 'type I' metrics) aim to identify the amount of noise present in the unit.
Examples include: ISI violations, sliding refractory period violations, SNR, NN-hit rate.
Completeness metrics (or 'false negative'/'type II' metrics) aim to identify whether any of the spiking activity is missing from a unit.
Examples include: presence ratio, amplitude cutoff, NN-miss rate.
Drift metrics aim to identify changes in waveforms which occur when spike sorters fail to successfully track neurons in the case of electrode drift.

Some metrics make use of principal component analysis (PCA) to reduce the dimensionality of computations.
Various approaches to computing the principal components are possible, and choice should be carefully considered in relation to the recording equipment used.
The following metrics make use of PCA: isolation distance, L-ratio, D-prime, Silhouette score and NN-metrics.
By contrast, the following metrics are based on spike times only: firing rate, ISI violations, presence ratio.
And amplitude cutoff and SNR are based on spike times as well as waveforms.

For more details about each metric and it's availability and use within SpikeInterface, see the individual pages for each metrics.

.. toctree::
    :maxdepth: 1
    :glob:

    quality_metrics/*


For more information about quality metrics, check out this excellent
`documentation <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html>`_
from the Allen Institute.


