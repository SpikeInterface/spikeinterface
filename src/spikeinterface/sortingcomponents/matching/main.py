from threadpoolctl import threadpool_limits
import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
from spikeinterface.core import get_chunk_with_margin, compute_sparsity, WaveformExtractor

from threadpoolctl import threadpool_limits
import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
from spikeinterface.core import get_chunk_with_margin


class TemplatesDictionary(object):
    """
    Class to extract handle the templates in order to work with the matching
    engines

    Parameters
    ----------
    data: array
        The template matrix, as a numpy array of size (n_templates, n_samples, n_channels)
    unit_ids: list
        The list of unit_ids
    nbefore: int
        The number of samples before the peaks of the templates
    nafter: int
        The number of samples after the peaks of the templates
    sparsity_mask: None or array
        If not None, an array of size (n_templates, n_channels) to set some sparsity mask
    Returns
    -------
    templates: TemplatesDictionary
        The TemplatesDictionary object
    """

    def __init__(
        self, 
        data : np.array,
        unit_ids : list, 
        nbefore : int,
        nafter : int,
        sparsity_mask=None
    ) -> None:

        self.data = data.copy().astype(np.float32, casting="safe")
        self.unit_ids = unit_ids
        self.nbefore = nbefore
        self.nafter = nafter

        assert self.nbefore + self.nafter == data.shape[1]

        if sparsity_mask is None:
            self.sparsity_mask = np.sum(data, axis=(1)) == 0
        else:
            assert sparsity_mask.shape == (data.shape[0], data.shape[2]), "sparsity_mask has the wrong shape"
            self.sparsity_mask = sparsity_mask

        for i in range(len(self.data)):
            active_channels = self.sparsity_mask[i]
            self.data[i][:, ~active_channels] = 0

    def __getitem__(self, template_id):
        return self.data[template_id]

    def __getslice__(self, start, stop):
        return self.data[start:stop]

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_templates(self):
        return self.data.shape[0]

    @property
    def num_channels(self):
        return self.data.shape[2]

    @property
    def nsamples(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def get_amplitudes(
        self, 
        peak_sign: str = "neg",
        mode: str = "extremum"
    ):
        """
        Get amplitude per channel for each unit.

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
            The waveform extractor
        peak_sign: str
            Sign of the template to compute best channels ('neg', 'pos', 'both')
        mode: str
            'extremum':  max or min
            'at_index': take value at spike index

        Returns
        -------
        peak_values: dict
            Dictionary with unit ids as keys and template amplitudes as values
        """
        assert peak_sign in ("both", "neg", "pos")
        assert mode in ("extremum", "at_index")
        peak_values = {}
        for unit_ind, unit_id in enumerate(self.unit_ids):
            template = self.data[unit_ind]

            if mode == "extremum":
                if peak_sign == "both":
                    values = np.max(np.abs(template), axis=0)
                elif peak_sign == "neg":
                    values = -np.min(template, axis=0)
                elif peak_sign == "pos":
                    values = np.max(template, axis=0)
            elif mode == "at_index":
                if peak_sign == "both":
                    values = np.abs(template[self.nbefore, :])
                elif peak_sign == "neg":
                    values = -template[self.before, :]
                elif peak_sign == "pos":
                    values = template[self.before, :]

            peak_values[unit_id] = values

        return peak_values

    def get_extremum_channel(
        self,
        peak_sign: str = "neg",
        mode: str = "extremum",
    ):
        """
        Compute the channel with the extremum peak for each unit.

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
            The waveform extractor
        peak_sign: str
            Sign of the template to compute best channels ('neg', 'pos', 'both')
        mode: str
            'extremum':  max or min
            'at_index': take value at spike index

        Returns
        -------
        extremum_channels: dict
            Dictionary with unit ids as keys and extremum channels (id or index based on 'outputs')
            as values
        """

        assert peak_sign in ("both", "neg", "pos")
        assert mode in ("extremum", "at_index")

        peak_values = self.get_amplitudes(peak_sign=peak_sign, mode=mode)
        extremum_channels_index = {}
        for unit_id in self.unit_ids:
            max_ind = np.argmax(peak_values[unit_id])
            extremum_channels_index[unit_id] = max_ind

        return extremum_channels_index


def create_templates_from_waveform_extractor(waveform_extractor, mode="median", sparsity=None):
    if sparsity is not None and not waveform_extractor.is_sparse():
        sparsity_mask = compute_sparsity(waveform_extractor, **sparsity)
    else:
        sparsity_mask = None

    data = waveform_extractor.get_all_templates(mode=mode)
    unit_ids = waveform_extractor.unit_ids
    nbefore = waveform_extractor.nbefore
    nafter = waveform_extractor.nafter
    return TemplatesDictionary(data, waveform_extractor.unit_ids, nbefore, nafter, sparsity_mask)


def find_spikes_from_templates(
    recording,
    waveform_extractor,
    sparsity={"method": "ptp", "threshold": 1},
    templates_dictionary=None,
    method="naive",
    method_kwargs={},
    extra_outputs=False,
    **job_kwargs,
):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    waveform_extractor: WaveformExtractor
        The waveform extractor to get the templates (if templates are not provided manually)
    sparsity: dict or None
        Parameters that should be given to sparsify the templates, if waveform_extractor
        is not already sparse
    templates_dictionary: TemplatesDictionnary
        If provided, then these templates are used instead of the ones from the waveform_extractor
    method: str
        Which method to use ('naive' | 'tridesclous' | 'circus' | 'circus-omp' | 'wobble')
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_outputs: bool
        If True then method_kwargs is also return
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor

    Returns
    -------
    spikes: ndarray
        Spikes found from templates.
    method_kwargs:
        Optionaly returns for debug purpose.

    Notes
    -----
    Templates are represented as WaveformExtractor so statistics can be extracted.
    """
    from .method_list import matching_methods

    assert method in matching_methods, "The method %s is not a valid one" % method

    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = matching_methods[method]

    # initialize the templates
    method_kwargs = method_class.initialize_and_sparsify_templates(
        method_kwargs, waveform_extractor, sparsity, templates_dictionary
    )

    # initialize
    method_kwargs = method_class.initialize_and_check_kwargs(recording, method_kwargs)

    # add
    method_kwargs["margin"] = method_class.get_margin(recording, method_kwargs)

    # serialiaze for worker
    method_kwargs_seralized = method_class.serialize_method_kwargs(method_kwargs)

    # and run
    func = _find_spikes_chunk
    init_func = _init_worker_find_spikes
    init_args = (recording, method, method_kwargs_seralized)
    processor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, handle_returns=True, job_name=f"find spikes ({method})", **job_kwargs
    )
    spikes = processor.run()

    spikes = np.concatenate(spikes)

    if extra_outputs:
        return spikes, method_kwargs
    else:
        return spikes


def _init_worker_find_spikes(recording, method, method_kwargs):
    """Initialize worker for finding spikes."""

    from .method_list import matching_methods

    method_class = matching_methods[method]
    method_kwargs = method_class.unserialize_in_worker(method_kwargs)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["method"] = method
    worker_ctx["method_kwargs"] = method_kwargs
    worker_ctx["function"] = method_class.main_function

    return worker_ctx


def _find_spikes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Find spikes from a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx["recording"]
    method = worker_ctx["method"]
    method_kwargs = worker_ctx["method_kwargs"]
    margin = method_kwargs["margin"]

    # load trace in memory given some margin
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(
        recording_segment, start_frame, end_frame, None, margin, add_zeros=True
    )

    function = worker_ctx["function"]

    with threadpool_limits(limits=1):
        spikes = function(traces, method_kwargs)

    # remove spikes in margin
    if margin > 0:
        keep = (spikes["sample_index"] >= margin) & (spikes["sample_index"] < (traces.shape[0] - margin))
        spikes = spikes[keep]

    spikes["sample_index"] += start_frame - margin
    spikes["segment_index"] = segment_index
    return spikes


# generic class for template engine
class BaseTemplateMatchingEngine:
    @classmethod
    def initialize_and_sparsify_templates(cls, kwargs, waveform_extractor, templates_dictionary, sparsity):
        assert isinstance(waveform_extractor, WaveformExtractor)

        if templates_dictionary is not None:
            templates_dictionary = create_templates_from_waveform_extractor(waveform_extractor, sparsity=sparsity)

        assert isinstance(templates_dictionary, TemplatesDictionary)

        kwargs["templates"] = templates_dictionary

        return kwargs

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        """This function serializes kwargs to distribute them to workers"""
        kwargs = dict(kwargs)
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        """This function unserializes kwargs in workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def get_margin(cls, recording, kwargs):
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def main_function(cls, traces, method_kwargs):
        """This function returns the number of samples for the chunk margins"""
        # need to be implemented in subclass
        raise NotImplementedError
