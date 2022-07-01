Postprocessing module
=====================

After spike sorting, we can use the :py:mod:`~spikeinterface.postprocessing` module to further post-process
the spike sorting output. Most of the post-processing functions require a
:py:class:`~spikeinterface.core.WaveformExtractor` as input. Available postprocessing tools are:

* compute principal component scores
* compute template similarity
* compute template waveform metrics
* get amplitudes for each spikes
* compute auto- and cross-correlogram 
