class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""


def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


class RecordingExtractorOldAPI:
    """
    This class mimic the old API of spikeextractors with:
      * reversed shape (channels, samples):
      * unique segment
    This is internally used for:
      * Montainsort4
      * Herdinspikes
    Because theses sorters are based on the old API
    """

    def __init__(self, recording):
        assert recording.get_num_segments() == 1
        self._recording = recording

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame, end_frame=end_frame,
                                            segment_index=0)
        return traces.T

    def get_num_frames(self):
        return self._recording.get_num_frames(segment_index=0)

    def get_num_channels(self):
        return self._recording.get_num_channels()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_channel_property(self, channel_id, property):
        rec = self._recording
        values = rec.get_property(property)
        ind = rec.ids_to_indices([channel_id])
        v = values[ind[0]]
        return v
