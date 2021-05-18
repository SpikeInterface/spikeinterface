import neo
from .neobaseextractor import NeoBaseRecordingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'AxonaRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, **kargs):
        super().__init__(**kargs)

        # Read channel groups by tetrode IDs
        self.set_channel_groups(groups=[x - 1 for x in self.neo_reader.raw_annotations[
            'blocks'][0]['segments'][0]['signals'][0]['__array_annotations__']['tetrode_id']])

        header_channels = self.neo_reader.header['signal_channels'][slice(None)]

        names = header_channels['name']
        for i, ind in enumerate(self.get_channel_ids()):
            self.set_channel_property(channel_id=ind, property_name='name', value=names[i])
