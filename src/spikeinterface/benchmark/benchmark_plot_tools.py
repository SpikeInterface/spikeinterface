import warnings

import numpy as np
from spikeinterface.widgets import get_some_colors
from .benchmark_tools import sigmoid, fit_sigmoid


def despine(ax_or_axes):
    import seaborn as sns

    if not isinstance(ax_or_axes, (list, tuple, np.ndarray)):
        ax_or_axes = [ax_or_axes]
    for ax in np.array(ax_or_axes).flatten():
        sns.despine(ax=ax)


def clean_axis(ax):
    for loc in ("top", "right", "left", "bottom"):
        ax.spines[loc].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_study_legend(study, case_keys=None, ax=None):
    """
    Make a ax with only legend
    """
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = study.get_colors()

    for k in case_keys:
        ax.plot([], color=colors[k], label=study.cases[k]["label"])
    ax.legend()
    clean_axis(ax)
    return fig


def aggregate_levels(df, study, case_keys=None, levels_to_keep=None):
    """
    Aggregate a DataFrame by dropping levels not to keep.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a MultiIndex.
    study : BenchmarkStudy
        A study object.
    case_keys : list | None, default: None
        A list of case keys to use. If None, then all cases are used.
    levels_to_keep : list | None, default: None
        A list of levels to keep. If None, the original dataframe, keys, labels and colros are returned.
    map_name : str | None, default: None
        The name of the map to use for colors.

    Returns
    -------
    df : pd.DataFrame
        The aggregated DataFrame.
    new_case_keys : list
        The aggregated case keys.
    labels : dict
        A dictionary of labels.
    colors : dict
        A dictionary of colors for each new case key.
    """
    import pandas as pd

    if case_keys is None:
        case_keys = list(study.cases.keys())

    if levels_to_keep is not None:
        if not isinstance(levels_to_keep, list):
            levels_to_keep = [levels_to_keep]
        drop_levels = [l for l in study.levels if l not in levels_to_keep]
        df = df.droplevel(drop_levels).sort_index()
        if len(levels_to_keep) > 1:
            df = df.reorder_levels(levels_to_keep)
        new_case_keys = list(np.unique(df.index))
        if isinstance(df.index, pd.MultiIndex):
            labels = {key: "-".join(key) for key in new_case_keys}
        else:
            labels = {key: key for key in new_case_keys}
        # get colors
        colors = study.get_colors(levels_to_group_by=levels_to_keep)
    else:
        new_case_keys = case_keys
        labels = {key: study.cases[key]["label"] for key in case_keys}
        colors = study.get_colors()

    return df, new_case_keys, labels, colors


def plot_run_times(study, case_keys=None, levels_to_keep=None, figsize=None, ax=None):
    """
    Plot run times for a BenchmarkStudy.

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list | None, default: None
        A selection of cases to plot, if None, then all.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Run times are aggregated by these levels.
    show_legend : bool, default True
        Show legend or not
    ax : matplotlib.axes.Axes | None, default: None
        The axes to use for plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots
    """
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    run_times = study.get_run_times(case_keys=case_keys)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if levels_to_keep is None:
        colors = study.get_colors()
        labels = []
        for i, key in enumerate(case_keys):
            labels.append(study.cases[key]["label"])
            rt = run_times.at[key, "run_times"]
            ax.bar(i, rt, width=0.8, color=colors[key])
        ax.set_xticks(np.arange(len(case_keys)))
        ax.set_xticklabels(labels, rotation=45.0)
        ax.set_ylabel("Run times (s)")

    else:
        import seaborn as sns

        run_times, case_keys, labels, colors = aggregate_levels(run_times, study, case_keys, levels_to_keep)
        palette_keys = case_keys

        if levels_to_keep is None:
            x = None
            hue = case_keys
            plt_fun = sns.barplot
        elif len(levels_to_keep) == 1:
            x = None
            hue = levels_to_keep[0]
            plt_fun = sns.boxplot
        elif len(levels_to_keep) == 2:
            # here we need to override the colors, since we are using x and hue
            # to displaye the 2 levels. We need to set the colors for the hue level alone
            x, hue = levels_to_keep
            hues = np.unique([c[1] for c in case_keys])
            colors = study.get_colors(levels_to_group_by=[hue])
            plt_fun = sns.boxplot
            palette_keys = hues
        else:
            # we aggregate levels into the same column and use the last level as hue
            levels_to_aggregate = levels_to_keep[:-1]
            hue = levels_to_keep[-1]
            x = " / ".join(levels_to_aggregate)
            run_times.loc[:, x] = run_times.index.map(lambda x: " / ".join(map(str, x[:-1])))
            hues = np.unique([c[-1] for c in case_keys])
            colors = study.get_colors(levels_to_group_by=[hue])
            plt_fun = sns.barplot
            palette_keys = hues

        assert all(
            [key in colors for key in palette_keys]
        ), f"colors must have a color for each palette key: {palette_keys}"

        plt_fun(data=run_times, y="run_times", x=x, hue=hue, ax=ax, palette=colors)

        despine(ax)
        if levels_to_keep is None:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, list(labels.values()))
        ax.set_ylabel("run time (s)")
    return fig


def plot_unit_counts(study, case_keys=None, levels_to_keep=None, colors=None, figsize=None, ax=None):
    """
    Plot unit counts for a study: "num_well_detected", "num_false_positive", "num_redundant", "num_overmerged"

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Unit counts are aggregated by these levels.
    colors : dict | None, default: None
        A dictionary of colors to use for each class ("Well Detected", "False Positive", "Redundant", "Overmerged").
    figsize : tuple | None, default: None
        The size of the figure.
    ax : matplotlib.axes.Axes | None, default: None
        The axes to use for plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from spikeinterface.widgets.utils import get_some_colors

    if case_keys is None:
        case_keys = list(study.cases.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    count_units = study.get_count_units(case_keys=case_keys)

    if levels_to_keep is None:
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
                ax.bar([x], [y], width=1 / (ncol + 2), label=label, color=colors[col])
            xticklabels.append(study.cases[key]["label"])
        ax.set_xticks(np.arange(len(case_keys)) + 1)
        ax.set_xticklabels(xticklabels, rotation=45.0)
        ax.legend()

    else:

        count_units, case_keys, _, _ = aggregate_levels(count_units, study, case_keys, levels_to_keep)
        count_units = count_units.drop(columns=["num_gt", "num_sorter"])
        if "num_bad" in count_units.columns:
            count_units = count_units.drop(columns=["num_bad"])

        # set hue based on exhaustive GT
        if "num_overmerged" in count_units.columns:
            hue_order = ["Well Detected", "False Positive", "Redundant", "Overmerged"]
        else:
            hue_order = ["Well Detected"]

        for col in count_units.columns:
            vals = count_units[col].values
            if not "well_detected" in col:
                vals = -vals
            col_name = col.replace("num_", "").replace("_", " ").title()
            count_units.loc[:, col_name] = vals
            del count_units[col]

        columns = count_units.columns.tolist()
        if colors is None:
            colors = get_some_colors(columns, color_engine="auto", map_name="hot")
            colors["Well Detected"] = "green"
        else:
            assert all([col in colors for col in columns]), f"colors must have a color for each column: {columns}"

        df = pd.melt(
            count_units.reset_index(),
            id_vars=levels_to_keep,
            value_vars=columns,
            var_name="Unit class",
            value_name="Count",
        )

        if len(levels_to_keep) > 1:
            x = " / ".join(levels_to_keep)
            df.loc[:, x] = df.apply(lambda r: " / ".join([str(r[col]) for col in levels_to_keep]), axis=1)
            df = df.drop(columns=levels_to_keep)
        else:
            x = levels_to_keep[0]

        sns.barplot(
            data=df,
            x=x,
            y="Count",
            hue="Unit class",
            ax=ax,
            hue_order=hue_order,
            palette=colors,
        )

        despine(ax)

    return fig


def plot_agreement_matrix(study, ordered=True, case_keys=None, axs=None):
    """
    Plot agreement matri ces for cases in a study.

    Parameters
    ----------
    study : GroundTruthStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    ordered : bool
        Order units with best agreement scores.
        This enable to see agreement on a diagonal.
    axs : matplotlib.axes.Axes | None, default: None
        The axs to use for plotting. Should be the same size as len(case_keys).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots
    """

    import matplotlib.pyplot as plt
    from spikeinterface.widgets import AgreementMatrixWidget

    if case_keys is None:
        case_keys = list(study.cases.keys())

    num_axes = len(case_keys)
    if axs is None:
        fig, axs = plt.subplots(ncols=num_axes, squeeze=True)
    else:
        assert len(axs) == num_axes, "axs should have the same number of axes as case_keys"
        fig = axs[0].get_figure()

    for count, key in enumerate(case_keys):
        ax = axs[count]
        comp = study.get_result(key)["gt_comparison"]

        unit_ticks = len(comp.sorting1.unit_ids) <= 16
        count_text = len(comp.sorting1.unit_ids) <= 16

        AgreementMatrixWidget(
            comp, ordered=ordered, count_text=count_text, unit_ticks=unit_ticks, backend="matplotlib", ax=ax
        )
        label = study.cases[key]["label"]
        ax.set_xlabel(label)

        if count > 0:
            ax.set_ylabel(None)
            ax.set_yticks([])
        ax.set_xticks([])

    return fig


def plot_performances(study, mode="ordered", performance_names=("accuracy", "precision", "recall"), case_keys=None):
    """
    Plot performances over case for a study.

    Parameters
    ----------
    study : BenchmarkStudy
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots
    """
    if mode == "snr":
        warnings.warn(
            "Use study.plot_performances_vs_snr() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_performances_vs_snr(study, case_keys=case_keys, performance_names=performance_names)
    elif mode == "ordered":
        warnings.warn(
            "Use study.plot_performances_ordered() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_performances_ordered(study, case_keys=case_keys, performance_names=performance_names)
    elif mode == "swarm":
        warnings.warn(
            "Use study.plot_performances_swarm() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_performances_swarm(study, case_keys=case_keys, performance_names=performance_names)
    else:
        raise ValueError("plot_performances() : wrong mode ")


def plot_performances_vs_snr(
    study,
    case_keys=None,
    figsize=None,
    performance_names=("accuracy", "recall", "precision"),
    snr_dataset_reference=None,
    levels_to_keep=None,
    orientation="vertical",
    show_legend=True,
    axs=None,
):
    """
    Plots performance metrics against signal-to-noise ratio (SNR) for different cases in a study.

    Parameters
    ----------
    study : object
        The study object containing the cases and results.
    case_keys : list | None, default: None
        List of case keys to include in the plot. If None, all cases in the study are included.
    figsize : tuple | None, default: None
        Size of the figure.
    performance_names : tuple, default: ("accuracy", "recall", "precision")
        Names of the performance metrics to plot. Default is ("accuracy", "recall", "precision").
    snr_dataset_reference : str | None, default: None
        Reference dataset key to use for SNR. If None, the SNR of each dataset is used.
    levels_to_keep : list | None, default: None
        Levels to group by when mapping case keys.
    orientation : "vertical" | "horizontal", default: "vertical"
        The orientation of the plot.
    show_legend : bool, default True
        Show legend or not
    axs : matplotlib.axes.Axes | None, default: None
        The axs to use for plotting. Should be the same size as len(performance_names).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots.
    """
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    if orientation == "vertical":
        ncols = 1
        nrows = len(performance_names)
    elif orientation == "horizontal":
        ncols = len(performance_names)
        nrows = 1
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if axs is None:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, squeeze=True)
    else:
        assert len(axs) == len(performance_names), "axs should have the same number of axes as performance_names"
        fig = axs[0].get_figure()

    for count, performance_name in enumerate(performance_names):

        ax = axs[count]
        if levels_to_keep is not None:
            case_group_keys, labels = study.get_grouped_keys_mapping(levels_to_group_by=levels_to_keep)
        else:
            labels = {k: study.cases[k]["label"] for k in case_keys}
            case_group_keys = {k: [k] for k in case_keys}

        colors = study.get_colors(levels_to_group_by=levels_to_keep)

        assert all(
            [key in colors for key in case_group_keys]
        ), f"colors must have a color for each case key: {case_group_keys}"

        for key, key_list in case_group_keys.items():
            color = colors[key]
            label = labels[key]
            all_xs = []
            all_ys = []
            for sub_key in key_list:
                if snr_dataset_reference is None:
                    # use the SNR of each dataset
                    analyzer = study.get_sorting_analyzer(sub_key)
                else:
                    # use the same SNR from a reference dataset
                    analyzer = study.get_sorting_analyzer(dataset_key=snr_dataset_reference)

                quality_metrics = analyzer.get_extension("quality_metrics").get_data()
                x = quality_metrics["snr"].values
                y = study.get_result(sub_key)["gt_comparison"].get_performance()[performance_name].values
                all_xs.append(x)
                all_ys.append(y)

            # accumulate x and y and make one final plot
            all_xs = np.concatenate(all_xs)
            all_ys = np.concatenate(all_ys)

            ax.scatter(all_xs, all_ys, marker=".", label=label, color=color)
            ax.set_ylabel(performance_name)

            popt = fit_sigmoid(all_xs, all_ys, p0=None)
            xfit = np.linspace(0, max(x), 100)
            ax.plot(xfit, sigmoid(xfit, *popt), color=color)

        ax.set_ylim(-0.05, 1.05)

        if show_legend and (count == len(performance_names) - 1):
            ax.legend()

    despine(axs)

    return fig


def plot_performances_ordered(
    study,
    case_keys=None,
    performance_names=("accuracy", "recall", "precision"),
    levels_to_keep=None,
    orientation="vertical",
    show_legend=True,
    figsize=None,
    axs=None,
):
    """
    Plot performances ordered by decreasing performance.

    Parameters
    ----------
    study : BenchmarkStudy
        A study object.
    case_keys : list | None, default: None
        A selection of cases to plot, if None, then all.
    performance_names : list | tuple, default: ("accuracy", "recall", "precision")
        A list of performance names to plot.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Performances are aggregated by these levels.
    orientation : "vertical" | "horizontal", default: "vertical"
        The orientation of the plot.
    show_legend : bool, default True
        Show legend or not
    figsize : tuple | None, default: None
        The size of the figure.
    axs : matplotlib.axes.Axes | None, default: None
        The axs to use for plotting. Should be the same size as len(performance_names).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots.
    """
    import matplotlib.pyplot as plt

    num_axes = len(performance_names)

    if case_keys is None:
        case_keys = list(study.cases.keys())

    perfs = study.get_performance_by_unit(case_keys=case_keys)
    perfs, case_keys, labels, colors = aggregate_levels(perfs, study, case_keys, levels_to_keep)
    assert all([key in colors for key in case_keys]), f"colors must have a color for each case key: {case_keys}"

    if axs is None:
        if orientation == "vertical":
            fig, axs = plt.subplots(nrows=num_axes, figsize=figsize, squeeze=True)
        elif orientation == "horizontal":
            fig, axs = plt.subplots(ncols=num_axes, figsize=figsize, squeeze=True)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
    else:
        assert len(axs) == num_axes, "axs should have the same number of axes as performance_names"
        fig = axs[0].get_figure()

    for count, performance_name in enumerate(performance_names):
        ax = axs[count]

        for key in case_keys:
            color = colors[key]
            label = labels[key]

            val = perfs.xs(key).loc[:, performance_name].values
            val = np.sort(val)[::-1]
            ax.plot(val, label=label, c=color)

        ax.set_title(performance_name)
        if show_legend and (count == len(performance_names) - 1):
            ax.legend(bbox_to_anchor=(0.05, 0.05), loc="lower left", framealpha=0.8)

    despine(axs)

    return fig


def plot_performances_swarm(
    study,
    case_keys=None,
    performance_names=("accuracy", "recall", "precision"),
    figsize=None,
    levels_to_keep=None,
    performance_colors={"accuracy": "g", "recall": "b", "precision": "r"},
    ax=None,
):
    """
    Parameters
    ----------
    study : BenchmarkStudy
        A study object.
    case_keys : list | None, default: None
        A selection of cases to plot, if None, then all.
    performance_names : list | tuple, default: ("accuracy", "recall", "precision")
        A list of performance names to plot.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Performances are aggregated by these levels.
    performance_colors : dict, default: {"accuracy": "g", "recall": "b", "precision": "r"}
        A dictionary of colors to use for each performance name.
    figsize : tuple | None, default: None
        The size of the figure.
    ax : matplotlib.axes.Axes | None, default: None
        The ax to use for plotting


    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if case_keys is None:
        case_keys = list(study.cases.keys())

    perfs = study.get_performance_by_unit(case_keys=case_keys)
    perfs, case_keys, _, _ = aggregate_levels(perfs, study, case_keys, levels_to_keep)

    assert all(
        [key in performance_colors for key in performance_names]
    ), f"performance_colors must have a color for each performance name: {performance_names}"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    levels = perfs.index.names

    df = pd.melt(
        perfs.reset_index(),
        id_vars=levels,
        var_name="Metric",
        value_name="Score",
        value_vars=performance_names,
    )
    df["x"] = df.apply(lambda r: " ".join([str(r[col]) for col in levels]), axis=1)
    sns.swarmplot(data=df, x="x", y="Score", hue="Metric", dodge=True, ax=ax, palette=performance_colors)

    despine(ax)

    return fig


def plot_performances_comparison(
    study,
    case_keys=None,
    figsize=None,
    performance_names=("accuracy", "recall", "precision"),
    performance_colors={"accuracy": "g", "recall": "b", "precision": "r"},
    levels_to_keep=None,
    ylim=(-0.1, 1.1),
):
    """
    Plot performances comparison for a study.

    Parameters
    ----------
    study : BenchmarkStudy
        A study object.
    case_keys : list | None, default: None
        A selection of cases to plot, if None, then all.
    figsize : tuple | None, default: None
        The size of the figure.
    performance_names : list | tuple, default: ("accuracy", "recall", "precision")
        A list of performance names to plot.
    performance_colors : dict, default: {"accuracy": "g", "recall": "b", "precision": "r"}
        A dictionary of colors to use for each performance name.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Performances are aggregated by these levels.
    ylim : tuple, default: (-0.1, 1.1)
        The y-axis limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots.
    """
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())
    case_keys, labels = study.get_grouped_keys_mapping(levels_to_group_by=levels_to_keep)

    num_methods = len(case_keys)
    assert num_methods >= 2, "plot_performances_comparison need at least 2 cases!"

    assert all(
        [key in performance_colors for key in performance_names]
    ), f"performance_colors must have a color for each performance name: {performance_names}"

    fig, axs = plt.subplots(ncols=num_methods - 1, nrows=num_methods - 1, figsize=figsize, squeeze=False)
    for i, key1 in enumerate(case_keys):
        for j, key2 in enumerate(case_keys):
            if i < j:
                ax = axs[i, j - 1]
                label1 = labels[key1]
                label2 = labels[key2]
                if i == j - 1:
                    ax.set_xlabel(label2)
                    ax.set_ylabel(label1)

                for sub_key1 in case_keys[key1]:
                    for sub_key2 in case_keys[key2]:
                        comp1 = study.get_result(sub_key1)["gt_comparison"]
                        comp2 = study.get_result(sub_key2)["gt_comparison"]

                        for performance_name, color in performance_colors.items():
                            perf1 = comp1.get_performance()[performance_name]
                            perf2 = comp2.get_performance()[performance_name]
                            ax.scatter(perf2, perf1, marker=".", label=performance_name, color=color)

                ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                ax.set_ylim(ylim)
                ax.set_xlim(ylim)
                despine(ax)
                ax.set_aspect("equal")
                if i != j - 1:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
            else:
                if j >= 1 and i < num_methods - 1:
                    ax = axs[i, j - 1]
                    ax.axis("off")

    ax = axs[num_methods - 2, 0]
    patches = []
    from matplotlib.patches import Patch

    for name, color in performance_colors.items():
        patches.append(Patch(color=color, label=name))
    ax.legend(handles=patches)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig


def plot_performances_vs_depth_and_snr(
    study, performance_name="accuracy", case_keys=None, figsize=None, levels_to_keep=None, map_name="viridis", axs=None
):
    """
    Plot performances vs depth and snr for a study.
    Parameters
    ----------
    study : BenchmarkStudy
        A study object.
    performance_name : str, default: "accuracy"
        The performance metric to plot.
    case_keys : list | None, default: None
        A selection of cases to plot, if None, then all.
    levels_to_keep : list | None, default: None
        A list of levels to keep. Performances are aggregated by these levels.
    map_name : str | None, default: "viridis"
        The name of the map to use for colors.
    figsize : tuple | None, default: None
        The size of the figure.
    axs : matplotlib.axes.Axes | None, default: None
        The axs to use for plotting. Should be the same size as len(case_keys).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots.
    """
    import pylab as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    case_keys, labels = study.get_grouped_keys_mapping(levels_to_group_by=levels_to_keep)

    if axs is None:
        fig, axs = plt.subplots(ncols=len(case_keys), figsize=figsize, squeeze=True)
    else:
        assert len(axs) == len(case_keys), "axs should have the same number of axes as case_keys"
        fig = axs[0].get_figure()

    for count, (key, key_list) in enumerate(case_keys.items()):
        all_snrs = []
        all_perfs = []
        all_depths = []
        for sub_key in key_list:
            result = study.get_result(sub_key)
            analyzer = study.get_sorting_analyzer(sub_key)
            positions = study.get_gt_unit_locations(sub_key)
            depth = positions[:, 1]

            metrics = analyzer.get_extension("quality_metrics").get_data()
            snr = metrics["snr"]
            perfs = result["gt_comparison"].get_performance()[performance_name].values
            all_snrs.append(snr)
            all_perfs.append(perfs)
            all_depths.append(depth)

        snr = np.concatenate(all_snrs)
        perfs = np.concatenate(all_perfs)
        depth = np.concatenate(all_depths)

        ax = axs[count]
        points = ax.scatter(depth, snr, c=perfs, label="matched", cmap=map_name)
        points.set_clim(0, 1)
        ax.set_xlabel("depth")
        ax.set_ylabel("snr")
        label = labels[key]
        ax.set_title(label)
        if count > 0:
            ax.set_ylabel("")
            ax.set_yticks([], [])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
    fig.colorbar(points, cax=cbar_ax, label=performance_name)

    despine(axs)

    return fig


def plot_performance_losses(
    study, case0, case1, performance_names=["accuracy"], map_name="coolwarm", figsize=None, axs=None
):
    """
    Plot performance losses between two cases.

    Parameters
    ----------
    study : BenchmarkStudy
        A study object.
    case0 : str | tuple
        The first case key.
    case1 : str | tuple
        The second case key.
    performance_names : list | tuple, default: ["accuracy"]
        A list of performance names to plot.
    map_name : str, default: "coolwarm"
        The name of the map to use for colors.
    figsize : tuple | None, default: None
        The size of the figure.
    axs : matplotlib.axes.Axes | None, default: None
        The axs to use for plotting. Should be the same size as len(performance_names).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the plots.
    """
    import matplotlib.pyplot as plt

    if axs is None:
        fig, axs = plt.subplots(nrows=len(performance_names), figsize=figsize, squeeze=True)
    else:
        assert len(axs) == len(performance_names), "axs should have the same number of axes as performance_names"
        fig = axs[0].get_figure()

    for count, perf_name in enumerate(performance_names):

        ax = axs[count]

        positions = study.get_gt_unit_locations(case0)

        analyzer = study.get_sorting_analyzer(case0)
        metrics_case0 = analyzer.get_extension("quality_metrics").get_data()
        x = metrics_case0["snr"].values

        y_case0 = study.get_result(case0)["gt_comparison"].get_performance()[perf_name].values
        y_case1 = study.get_result(case1)["gt_comparison"].get_performance()[perf_name].values

        ax.set_xlabel("depth (um)")
        im = ax.scatter(positions[:, 1], x, c=(y_case1 - y_case0), cmap=map_name)
        fig.colorbar(im, ax=ax, label=perf_name)
        im.set_clim(-1, 1)

        label0 = study.cases[case0]["label"]
        label1 = study.cases[case1]["label"]
        ax.set_title(f"{label0}\n vs \n{label1}")
        ax.set_ylabel("snr")

    despine(axs)

    return fig
