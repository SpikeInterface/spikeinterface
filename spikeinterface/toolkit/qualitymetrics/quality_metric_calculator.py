

class QualityMetricCalculator:
    def __init__(waveform_extractor):
        self.waveform_extractor = waveform_extractor
        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting
    
    