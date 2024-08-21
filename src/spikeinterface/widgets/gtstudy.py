from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


def handle_levels(df, study, case_keys, levels):
    import pandas as pd

    if case_keys is None:
        case_keys = list(study.cases.keys())
    labels = {key: study.cases[key]["label"] for key in case_keys}

    if levels is not None:
        drop_levels = [l for l in study.levels if l not in levels]
        df = df.droplevel(drop_levels).sort_index()
        if len(levels) > 1:
            df = df.reorder_levels(levels)
        case_keys = list(np.unique(df.index))
        if isinstance(df.index, pd.MultiIndex):
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
        cases are treated as separate in a bar plot.
        When specified, if levels is a string or a 1-element tuple/list,
        then it will be treated as the "x" variable of a boxplot. In case it's a
        2-element object, the first element is "x", the second is "hue".
        More than 2 elements are not supported
    """

    def __init__(
        self,
        study,
        case_keys=None,
        levels=None,
        cmap="tab20",
        backend=None,
        **backend_kwargs,
    ):
        if case_keys is None:
            case_keys = list(study.cases.keys())

        if levels is not None:
            if isinstance(levels, str):
                levels = [levels]
            assert len(levels) < 3, "You can pass at most 2 levels to plot against!"
            assert all([l in study.levels for l in levels]), f"levels must be in {study.levels}"

        plot_data = dict(
            study=study,
            run_times=study.get_run_times(case_keys),
            case_keys=case_keys,
            levels=levels,
            colors=study.get_colors(),
            cmap=cmap,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        from .utils import get_some_colors
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        run_times, case_keys, labels = handle_levels(dp.run_times, dp.study, dp.case_keys, dp.levels)

        if dp.levels is None:
            x = None
            hue = case_keys
            colors = get_some_colors(case_keys, map_name=dp.cmap, color_engine="matplotlib", shuffle=False, margin=0)
            plt_fun = sns.barplot
        elif len(dp.levels) == 1:
            x = None
            colors = get_some_colors(case_keys, map_name=dp.cmap, color_engine="matplotlib", shuffle=False, margin=0)
            hue = dp.levels[0]
            plt_fun = sns.boxplot
        elif len(dp.levels) == 2:
            x, hue = dp.levels
            hues = np.unique([c[1] for c in case_keys])
            colors = get_some_colors(hues, map_name=dp.cmap, color_engine="matplotlib", shuffle=False, margin=0)
            plt_fun = sns.boxplot

        plt_fun(data=run_times, y="run_time", x=x, hue=hue, ax=self.ax, palette=colors)

        self.ax.set_ylabel("run time (s)")
        sns.despine(ax=self.ax)
        if dp.levels is None:
            h, l = self.ax.get_legend_handles_labels()
            self.ax.legend(h, list(labels.values()))


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
        When specified, if levels is a string or a 1-element tuple/list,
        then it will be treated as the "x" variable of a boxplot. In case it's a
        2-element object, the first element is "x", the second is "hue".
        More than 2 elements are not supported.
        If the number of counts to plot is more than one (e.g., in case of exhaustive
        ground truth), then only one level at a time is supported.
    labels : dict or None, default: None
        The labels to use for each case key in case levels is None.
    rotation : int or None, default: 45
        The rotation for the x tick labels

    """

    def __init__(
        self,
        study,
        case_keys=None,
        levels=None,
        labels=None,
        rotation=45,
        backend=None,
        cmap="tab20",
        **backend_kwargs,
    ):
        if levels is not None:
            if isinstance(levels, str):
                levels = [levels]
            assert all([l in study.levels for l in levels]), f"levels must be in {study.levels}"
        plot_data = dict(
            study=study,
            count_units=study.get_count_units(case_keys=case_keys),
            case_keys=case_keys,
            levels=levels,
            labels=labels,
            cmap=cmap,
            rotation=rotation
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import seaborn as sns
        import pandas as pd

        from .utils_matplotlib import make_mpl_figure
        from .utils import get_some_colors

        dp = to_attr(data_plot)
        study = dp.study
        count_units = dp.count_units
        levels = dp.levels

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        count_units, case_keys, labels = handle_levels(dp.count_units, dp.study, dp.case_keys, levels)
        count_units = count_units.drop(columns=["num_gt", "num_sorter"])

        if dp.labels is not None:
            labels = dp.labels

        for col in count_units.columns:
            vals = count_units[col].values
            if not "well_detected" in col:
                vals = -vals
            col_name = col.replace("num_", "").replace("_", " ").title()
            count_units.loc[:, col_name] = vals
            del count_units[col]

        columns = count_units.columns.tolist()
        ncol = len(columns)

        count_units = count_units.reset_index()
        if levels is not None:
            if len(levels) == 1:
                var_name = "Metric"
                x = levels[0]
                y = "Num Units"
                hue = "Metric"
                color_list = columns
            else:
                assert len(columns) == 1, (
                    f"Multi-levels is not supported when multiple metrics counts are available ({columns})"
                )
                var_name = None
                x, hue = levels
                y = columns[0]
                color_list = list(np.unique(count_units[hue]))
        else:
            count_units.loc[:, "Label"] = labels.values()
            levels = study.levels + ["Label"]
            var_name = "Metric"
            x = "Label"
            y = "Num Units"
            hue = "Metric"
            color_list = columns

        colors = get_some_colors(color_list, color_engine="auto", map_name=dp.cmap)
        # Well Detected is always present
        colors["Well Detected"] = "green"
        if var_name is not None:
            df = count_units.melt(id_vars=levels, var_name=var_name, value_name="Num Units")
        else:
            df = count_units

        sns.barplot(df, x=x, y=y, hue=hue, ax=self.ax, palette=colors,)
        _ = self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=dp.rotation)
        sns.despine(ax=self.ax)


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
        import pandas as pd
        import seaborn as sns

        from .utils_matplotlib import make_mpl_figure
        from .utils import get_some_colors

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
                sns.despine(ax=ax)

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
                sns.despine(ax=ax)

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
            sns.swarmplot(data=df, x="x", y="Score", hue="Metric", dodge=True, ax=self.ax)
            sns.despine(ax=self.ax)


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
