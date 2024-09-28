"""
This module will be deprecated and will be removed in 0.102.0

All ploting for the previous GTStudy is now centralized in spikeinterface.benchmark.benchmark_plot_tools
Please not that GTStudy is replaced by SorterStudy wich is based more generic BenchmarkStudy.
"""

from __future__ import annotations

from .base import BaseWidget

import warnings

class StudyRunTimesWidget(BaseWidget):
    """
    Plot sorter run times for a SorterStudy.

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.

    """

    def __init__(self, study, case_keys=None, backend=None, **backend_kwargs):
        warnings.warn("plot_study_run_times is to be deprecated. Use spikeinterface.benchmark.benchmark_plot_tools instead.")
        plot_data = dict(study=study, case_keys=case_keys)
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from spikeinterface.benchmark.benchmark_plot_tools import plot_run_times
        plot_run_times(data_plot["study"], case_keys=data_plot["case_keys"])


class StudyUnitCountsWidget(BaseWidget):
    """
    Plot unit counts for a study: "num_well_detected", "num_false_positive", "num_redundant", "num_overmerged"

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.

    """

    def __init__(self, study, case_keys=None, backend=None, **backend_kwargs):
        warnings.warn("plot_study_unit_counts is to be deprecated. Use spikeinterface.benchmark.benchmark_plot_tools instead.")
        plot_data = dict(study=study, case_keys=case_keys)
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from spikeinterface.benchmark.benchmark_plot_tools import plot_unit_counts
        plot_unit_counts(data_plot["study"], case_keys=data_plot["case_keys"])


class StudyPerformances(BaseWidget):
    """
    Plot performances over case for a study.


    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    mode : "ordered" | "snr" | "swarm", default: "ordered"
        Which plot mode to use:

        * "ordered": plot performance metrics vs unit indices ordered by decreasing accuracy
        * "snr": plot performance metrics vs snr
        * "swarm": plot performance metrics as a swarm plot (see seaborn.swarmplot for details)
    performance_names : list or tuple, default: ("accuracy", "precision", "recall")
        Which performances to plot ("accuracy", "precision", "recall")
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    """

    def __init__(
        self,
        study,
        mode="ordered",
        performance_names=("accuracy", "precision", "recall"),
        case_keys=None,
        backend=None,
        **backend_kwargs,
    ):
        warnings.warn("plot_study_performances is to be deprecated. Use spikeinterface.benchmark.benchmark_plot_tools instead.")
        plot_data = dict(
            study=study,
            mode=mode,
            performance_names=performance_names,
            case_keys=case_keys,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from spikeinterface.benchmark.benchmark_plot_tools import plot_performances
        plot_performances(
            data_plot["study"],
            mode=data_plot["mode"],
            performance_names=data_plot["performance_names"],
            case_keys=data_plot["case_keys"]
        )

class StudyAgreementMatrix(BaseWidget):
    """
    Plot agreement matrix.

    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    ordered : bool
        Order units with best agreement scores.
        This enable to see agreement on a diagonal.
    """

    def __init__(
        self,
        study,
        ordered=True,
        case_keys=None,
        backend=None,
        **backend_kwargs,
    ):
        warnings.warn("plot_study_agreement_matrix is to be deprecated. Use spikeinterface.benchmark.benchmark_plot_tools instead.")
        plot_data = dict(
            study=study,
            case_keys=case_keys,
            ordered=ordered,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from spikeinterface.benchmark.benchmark_plot_tools import plot_agreement_matrix
        plot_agreement_matrix(
            data_plot["study"],
            ordered=data_plot["ordered"],
            case_keys=data_plot["case_keys"]
        )


class StudySummary(BaseWidget):
    """
    Plot a summary of a ground truth study.
    Internally this plotting function runs:

      * plot_study_run_times
      * plot_study_unit_counts
      * plot_study_performances
      * plot_study_agreement_matrix

    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None, default: None
        A selection of cases to plot, if None, then all.
    """

    def __init__(
        self,
        study,
        case_keys=None,
        backend=None,
        **backend_kwargs,
    ):
        
        warnings.warn("plot_study_summary is to be deprecated. Use spikeinterface.benchmark.benchmark_plot_tools instead.")
        plot_data = dict(study=study, case_keys=case_keys)
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        study = data_plot["study"]
        case_keys = data_plot["case_keys"]

        from spikeinterface.benchmark.benchmark_plot_tools import plot_agreement_matrix, plot_performances, plot_unit_counts, plot_run_times

        plot_performances(study=study, case_keys=case_keys, mode="ordered")
        plot_performances(study=study, case_keys=case_keys, mode="snr")
        plot_agreement_matrix(study=study, case_keys=case_keys)
        plot_run_times(study=study, case_keys=case_keys)
        plot_unit_counts(study=study, case_keys=case_keys)
