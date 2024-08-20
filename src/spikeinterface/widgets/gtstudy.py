from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


def handle_levels(df, study, case_keys, levels):
    if case_keys is None:
        case_keys = list(study.cases.keys())
        labels = {key: study.cases[key]["label"] for key in case_keys}

    if levels is not None:
        drop_levels = [l for l in study.levels if l not in levels]
        df = df.droplevel(drop_levels).sort_index()
        case_keys = list(np.unique(df.index))
        if isinstance(df.index, df.MultiIndex):
            labels = {key: "-".join(key) for key in case_keys}
        else:
            labels = {key: key for key in case_keys}

    return df, case_keys, labels


class StudyRunTimesWidget(BaseWidget):
    """
    Plot sorter run times for a GroundTruthStudy


    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None, default: None
        A selection of cases to plot, if None, then all cases are plotted.
    levels : str or list-like or None, default: None
        A selection of levels to group cases by, if None, then all
        cases are treated as separate.
    """

    def __init__(
        self,
        study,
        case_keys=None,
        levels=None,
        backend=None,
        **backend_kwargs,
    ):
        if case_keys is None:
            case_keys = list(study.cases.keys())

        if levels is not None:
            assert all([l in study.levels for l in levels]), f"levels must be in {study.levels}"

        plot_data = dict(
            study=study,
            run_times=study.get_run_times(case_keys),
            case_keys=case_keys,
            levels=levels,
            colors=study.get_colors(),
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        run_times, case_keys, labels = handle_levels(dp.run_times, dp.study, dp.case_keys, dp.levels)

        for i, key in enumerate(case_keys):
            label = dp.study.cases[key]["label"]
            rt = run_times.loc[key]
            self.ax.bar(i, rt, width=0.8, label=label, facecolor=dp.colors[key])
        self.ax.set_ylabel("run time (s)")
        self.ax.legend()


class StudyUnitCountsWidget(BaseWidget):
    """
    Plot unit counts for a study: "num_well_detected", "num_false_positive", "num_redundant", "num_overmerged"


    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None, default: None
        A selection of cases to plot, if None, then all cases are plotted.
    levels : str or list-like or None, default: None
        A selection of levels to group cases by, if None, then all
        cases are treated as separate.
    """

    def __init__(
        self,
        study,
        case_keys=None,
        levels=None,
        backend=None,
        **backend_kwargs,
    ):
        if levels is not None:
            assert all([l in study.levels for l in levels]), f"levels must be in {study.levels}"
        plot_data = dict(
            study=study, count_units=study.get_count_units(case_keys=case_keys), case_keys=case_keys, levels=levels
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from .utils_matplotlib import make_mpl_figure
        from .utils import get_some_colors

        import pandas as pd

        dp = to_attr(data_plot)
        count_units = dp.count_units

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        count_units, case_keys, labels = handle_levels(dp.count_units, dp.study, dp.case_keys, dp.levels)

        columns = count_units.columns.tolist()
        columns.remove("num_gt")
        columns.remove("num_sorter")

        ncol = len(columns)

        colors = get_some_colors(columns, color_engine="auto", map_name="hot")
        colors["num_well_detected"] = "green"

        xticklabels = []
        for i, key in enumerate(case_keys):
            for c, col in enumerate(columns):
                x = i + 1 + c / (ncol + 1)
                y = count_units.loc[key, col]
                if not "well_detected" in col:
                    y = -y

                if i == 0:
                    label = col.replace("num_", "").replace("_", " ").title()
                else:
                    label = None

                self.ax.bar([x], [y], width=1 / (ncol + 2), label=label, color=colors[col])
            xticklabels.append(labels[key])

        self.ax.set_xticks(np.arange(len(dp.case_keys)) + 1)
        self.ax.set_xticklabels(xticklabels)
        self.ax.legend()


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
    case_keys : list or None, default: None
        A selection of cases to plot, if None, then all cases are plotted.
    levels : str or list-like or None, default: None
        A selection of levels to group cases by, if None, then all
        cases are treated as separate.
    """

    def __init__(
        self,
        study,
        mode="ordered",
        performance_names=("accuracy", "precision", "recall"),
        case_keys=None,
        levels=None,
        cmap="tab20",
        backend=None,
        **backend_kwargs,
    ):
        if levels is not None:
            assert all([l in study.levels for l in levels]), f"levels must be in {study.levels}"
        perfs_by_unit = study.get_performance_by_unit(case_keys=case_keys)
        if mode == "snr":
            metrics = study.get_metrics(case_keys=case_keys)
            perfs_by_unit = perfs_by_unit.merge(metrics, on=list(study.levels) + ["gt_unit_id"])

        plot_data = dict(
            study=study,
            perfs=perfs_by_unit,
            mode=mode,
            performance_names=performance_names,
            levels=levels,
            case_keys=case_keys,
            cmap=cmap,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from .utils import get_some_colors

        import pandas as pd
        import seaborn as sns

        dp = to_attr(data_plot)
        perfs, case_keys, labels = handle_levels(dp.perfs, dp.study, dp.case_keys, dp.levels)

        colors = get_some_colors(case_keys, map_name=dp.cmap, color_engine="matplotlib", shuffle=False, margin=0)

        if dp.mode in ("ordered", "snr"):
            backend_kwargs["num_axes"] = len(dp.performance_names)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        if dp.mode == "ordered":
            for count, performance_name in enumerate(dp.performance_names):
                ax = self.axes.flatten()[count]
                for key in case_keys:
                    label = labels[key]
                    val = perfs.xs(key).loc[:, performance_name].values
                    val = np.sort(val)[::-1]
                    ax.plot(val, label=label, c=colors[key])
                ax.set_title(performance_name)
                if count == len(dp.performance_names) - 1:
                    ax.legend(bbox_to_anchor=(0.05, 0.05), loc="lower left", framealpha=0.8)

        elif dp.mode == "snr":
            metric_name = dp.mode
            for count, performance_name in enumerate(dp.performance_names):
                ax = self.axes.flatten()[count]

                max_metric = 0
                for key in case_keys:
                    x = perfs.xs(key).loc[:, metric_name].values
                    y = perfs.xs(key).loc[:, performance_name].values
                    label = labels[key]
                    ax.scatter(x, y, s=10, label=label, color=colors[key], alpha=0.5)
                    max_metric = max(max_metric, np.max(x))
                ax.set_title(performance_name)
                ax.set_xlim(0, max_metric * 1.05)
                ax.set_ylim(0, 1.05)
                if count == 0:
                    ax.legend(loc="lower right")

        elif dp.mode == "swarm":
            levels = perfs.index.names if dp.levels is None else dp.levels
            df = pd.melt(
                perfs.reset_index(),
                id_vars=levels,
                var_name="Metric",
                value_name="Score",
                value_vars=dp.performance_names,
            )
            df["x"] = df.apply(lambda r: " ".join([r[col] for col in levels]), axis=1)
            sns.swarmplot(data=df, x="x", y="Score", hue="Metric", dodge=True)


class StudyAgreementMatrix(BaseWidget):
    """
    Plot agreement matrix.

    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None, default: None
        A selection of cases to plot, if None, then all cases are plotted.
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
        if case_keys is None:
            case_keys = list(study.cases.keys())

        plot_data = dict(
            study=study,
            case_keys=case_keys,
            ordered=ordered,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from .comparison import AgreementMatrixWidget

        dp = to_attr(data_plot)
        study = dp.study

        backend_kwargs["num_axes"] = len(dp.case_keys)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        for count, key in enumerate(dp.case_keys):
            ax = self.axes.flatten()[count]
            comp = study.comparisons[key]
            unit_ticks = len(comp.sorting1.unit_ids) <= 16
            count_text = len(comp.sorting1.unit_ids) <= 16

            AgreementMatrixWidget(
                comp, ordered=dp.ordered, count_text=count_text, unit_ticks=unit_ticks, backend="matplotlib", ax=ax
            )
            label = study.cases[key]["label"]
            ax.set_xlabel(label)

            if count > 0:
                ax.set_ylabel(None)
                ax.set_yticks([])
            ax.set_xticks([])

        # ax0 = self.axes.flatten()[0]
        # for ax in self.axes.flatten()[1:]:
        #     ax.sharey(ax0)


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
        A selection of cases to plot, if None, then all cases are plotted.
    """

    def __init__(
        self,
        study,
        case_keys=None,
        backend=None,
        **backend_kwargs,
    ):
        if case_keys is None:
            case_keys = list(study.cases.keys())

        plot_data = dict(
            study=study,
            case_keys=case_keys,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        study = data_plot["study"]
        case_keys = data_plot["case_keys"]

        StudyPerformances(study=study, case_keys=case_keys, mode="ordered", backend="matplotlib", **backend_kwargs)
        StudyPerformances(study=study, case_keys=case_keys, mode="snr", backend="matplotlib", **backend_kwargs)
        StudyAgreementMatrix(study=study, case_keys=case_keys, backend="matplotlib", **backend_kwargs)
        StudyRunTimesWidget(study=study, case_keys=case_keys, backend="matplotlib", **backend_kwargs)
        StudyUnitCountsWidget(study=study, case_keys=case_keys, backend="matplotlib", **backend_kwargs)
