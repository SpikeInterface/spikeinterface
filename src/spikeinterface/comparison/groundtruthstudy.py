_txt_error_message = """
GroundTruthStudy has been replaced by SorterStudy with similar API but not back compatible folder loading.
You can do:
from spikeinterface.benchmark import SorterStudy
study = SorterStudy.create(study_folder, datasets=..., cases=..., levels=...)
study.run() # this run sorters
study.compute_results() # this run the comparisons
# and then some ploting
study.plot_agreements()
study.plot_performances_vs_snr()
...
"""


class GroundTruthStudy:
    def __init__(self, study_folder):
        raise RuntimeError(_txt_error_message)

    @classmethod
    def create(cls, study_folder, datasets={}, cases={}, levels=None):
        raise RuntimeError(_txt_error_message)
