import numpy as np



def _simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_run_times(study, case_keys=None):
    """
    Plot run times for a BenchmarkStudy.

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    """
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    run_times = study.get_run_times(case_keys=case_keys)

    colors = study.get_colors()
    

    fig, ax = plt.subplots()
    labels = []
    for i, key in enumerate(case_keys):
        labels.append(study.cases[key]["label"])
        rt = run_times.at[key, "run_times"]
        ax.bar(i, rt, width=0.8, color=colors[key])
    ax.set_xticks(np.arange(len(case_keys)))
    ax.set_xticklabels(labels, rotation=45.0)
    return fig


def plot_unit_counts(study, case_keys=None):
    """
    Plot unit counts for a study: "num_well_detected", "num_false_positive", "num_redundant", "num_overmerged"

    Parameters
    ----------
    study : SorterStudy
        A study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    """
    import matplotlib.pyplot as plt
    from spikeinterface.widgets.utils import get_some_colors

    if case_keys is None:
        case_keys = list(study.cases.keys())


    count_units = study.get_count_units(case_keys=case_keys)

    fig, ax = plt.subplots()

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
    ax.set_xticklabels(xticklabels)
    ax.legend()

    return fig

def plot_performances(study, mode="ordered", performance_names=("accuracy", "precision", "recall"), case_keys=None):
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
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if case_keys is None:
        case_keys = list(study.cases.keys())

    perfs=study.get_performance_by_unit(case_keys=case_keys)
    colors = study.get_colors()


    if mode in ("ordered", "snr"):
        num_axes = len(performance_names)
        fig, axs = plt.subplots(ncols=num_axes)
    else:
        fig, ax = plt.subplots()

    if mode == "ordered":
        for count, performance_name in enumerate(performance_names):
            ax = axs.flatten()[count]
            for key in case_keys:
                label = study.cases[key]["label"]
                val = perfs.xs(key).loc[:, performance_name].values
                val = np.sort(val)[::-1]
                ax.plot(val, label=label, c=colors[key])
            ax.set_title(performance_name)
            if count == len(performance_names) - 1:
                ax.legend(bbox_to_anchor=(0.05, 0.05), loc="lower left", framealpha=0.8)

    elif mode == "snr":
        metric_name = mode
        for count, performance_name in enumerate(performance_names):
            ax = axs.flatten()[count]

            max_metric = 0
            for key in case_keys:
                x = study.get_metrics(key).loc[:, metric_name].values
                y = perfs.xs(key).loc[:, performance_name].values
                label = study.cases[key]["label"]
                ax.scatter(x, y, s=10, label=label, color=colors[key])
                max_metric = max(max_metric, np.max(x))
            ax.set_title(performance_name)
            ax.set_xlim(0, max_metric * 1.05)
            ax.set_ylim(0, 1.05)
            if count == 0:
                ax.legend(loc="lower right")

    elif mode == "swarm":
        levels = perfs.index.names
        df = pd.melt(
            perfs.reset_index(),
            id_vars=levels,
            var_name="Metric",
            value_name="Score",
            value_vars=performance_names,
        )
        df["x"] = df.apply(lambda r: " ".join([r[col] for col in levels]), axis=1)
        sns.swarmplot(data=df, x="x", y="Score", hue="Metric", dodge=True, ax=ax)


def plot_agreement_matrix(study, ordered=True, case_keys=None):
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
    """

    import matplotlib.pyplot as plt
    from spikeinterface.widgets import AgreementMatrixWidget

    if case_keys is None:
        case_keys = list(study.cases.keys())


    num_axes = len(case_keys)
    fig, axs = plt.subplots(ncols=num_axes)

    for count, key in enumerate(case_keys):
        ax = axs.flatten()[count]
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


def plot_performances_vs_snr(study, case_keys=None, figsize=None, metrics=["accuracy", "recall", "precision"]):
    import matplotlib.pyplot as plt

    if case_keys is None:
        case_keys = list(study.cases.keys())

    fig, axs = plt.subplots(ncols=1, nrows=len(metrics), figsize=figsize, squeeze=False)

    for count, k in enumerate(metrics):

        ax = axs[count, 0]
        for key in case_keys:
            label = study.cases[key]["label"]

            analyzer = study.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()
            x = metrics["snr"].values
            y = study.get_result(key)["gt_comparison"].get_performance()[k].values
            ax.scatter(x, y, marker=".", label=label)
            ax.set_title(k)
        
        ax.set_ylim(0, 1.05)
        
        if count == 2:
            ax.legend()

    return fig
