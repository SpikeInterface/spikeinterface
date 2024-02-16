from __future__ import annotations

from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.postprocessing import compute_template_similarity
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
    plot_unit_waveforms,
)

import time
import os
from pathlib import Path
import string, random
import pylab as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shutil
import copy
from tqdm.auto import tqdm


def running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        notebook_shells = {"ZMQInteractiveShell", "TerminalInteractiveShell"}
        # if a shell is missing from this set just check get_ipython().__class__.__name__ and add it to the set
        return shell in notebook_shells
    except NameError:
        return False


class BenchmarkMatching:
    """Benchmark a set of template matching methods on a given recording and ground-truth sorting."""

    def __init__(
        self,
        recording,
        gt_sorting,
        waveform_extractor,
        methods,
        methods_kwargs=None,
        exhaustive_gt=True,
        tmp_folder=None,
        template_mode="median",
        **job_kwargs,
    ):
        self.methods = methods
        if methods_kwargs is None:
            methods_kwargs = {method: {} for method in methods}
        self.methods_kwargs = methods_kwargs
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.exhaustive_gt = exhaustive_gt
        self.sampling_rate = self.recording.get_sampling_frequency()

        if tmp_folder is None:
            tmp_folder = os.path.join(".", "".join(random.choices(string.ascii_uppercase + string.digits, k=8)))
        self.tmp_folder = Path(tmp_folder)
        self.sort_folders = []

        self.we = waveform_extractor
        for method in self.methods:
            self.methods_kwargs[method]["waveform_extractor"] = self.we
        self.templates = self.we.get_all_templates(mode=template_mode)
        self.metrics = compute_quality_metrics(self.we, metric_names=["snr"], load_if_exists=True)
        self.similarity = compute_template_similarity(self.we)
        self.parameter_name2matching_fn = dict(
            num_spikes=self.run_matching_num_spikes,
            fraction_misclassed=self.run_matching_misclassed,
            fraction_missing=self.run_matching_missing_units,
        )

    def __enter__(self):
        self.tmp_folder.mkdir(exist_ok=True)
        self.sort_folders = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tmp_folder.exists():
            shutil.rmtree(self.tmp_folder)
        for sort_folder in self.sort_folders:
            if sort_folder.exists():
                shutil.rmtree(sort_folder)

    def run_matching(self, methods_kwargs, unit_ids):
        """Run template matching on the recording with settings in methods_kwargs.

        Parameters
        ----------
        methods_kwargs: dict
            A dictionary of method_kwargs for each method.
        unit_ids: array-like
            The unit ids to use for the output sorting.

        Returns
        -------
        sortings: dict
            A dictionary that maps method --> NumpySorting.
        runtimes: dict
            A dictionary that maps method --> runtime.
        """
        sortings, runtimes = {}, {}
        for method in self.methods:
            t0 = time.time()
            spikes = find_spikes_from_templates(
                self.recording, method=method, method_kwargs=methods_kwargs[method], **self.job_kwargs
            )
            runtimes[method] = time.time() - t0
            sorting = NumpySorting.from_times_labels(
                spikes["sample_index"], unit_ids[spikes["cluster_index"]], self.sampling_rate
            )
            sortings[method] = sorting
        return sortings, runtimes

    def run_matching_num_spikes(self, spike_num, seed=0, we_kwargs=None, template_mode="median"):
        """Run template matching with a given number of spikes per unit.

        Parameters
        ----------
        spike_num: int
            The maximum number of spikes per unit
        seed: int, default: 0
            Random seed
        we_kwargs: dict
            A dictionary of keyword arguments for the WaveformExtractor
        template_mode: "mean" | "median" | "std", default: "median"
            The mode to use to extract templates from the WaveformExtractor

        Returns
        -------

        sortings: dict
            A dictionary that maps method --> NumpySorting.
        gt_sorting: NumpySorting
            The ground-truth sorting used for template matching (= self.gt_sorting).
        """
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(
            dict(max_spikes_per_unit=spike_num, seed=seed, overwrite=True, load_if_exists=False, **self.job_kwargs)
        )

        # Generate New Waveform Extractor with New Spike Numbers
        we = extract_waveforms(self.recording, self.gt_sorting, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        sortings, _ = self.run_matching(methods_kwargs, we.unit_ids)
        shutil.rmtree(self.tmp_folder)
        return sortings, self.gt_sorting

    def update_methods_kwargs(self, we, template_mode="median"):
        """Update the methods_kwargs dictionary with the new WaveformExtractor.

        Parameters
        ----------
        we: WaveformExtractor
            The new WaveformExtractor.
        template_mode: "mean" | "median" | "std", default: "median"
            The mode to use to extract templates from the WaveformExtractor

        Returns
        -------
        methods_kwargs: dict
            A dictionary of method_kwargs for each method.
        """
        templates = we.get_all_templates(we.unit_ids, mode=template_mode)
        methods_kwargs = copy.deepcopy(self.methods_kwargs)
        for method in self.methods:
            method_kwargs = methods_kwargs[method]
            if method == "wobble":
                method_kwargs.update(dict(templates=templates, nbefore=we.nbefore, nafter=we.nafter))
            else:
                method_kwargs["waveform_extractor"] = we
        return methods_kwargs

    def run_matching_misclassed(
        self, fraction_misclassed, min_similarity=-1, seed=0, we_kwargs=None, template_mode="median"
    ):
        """Run template matching with a given fraction of misclassified spikes.

        Parameters
        ----------
        fraction_misclassed: float
            The fraction of misclassified spikes.
        min_similarity: float, default: -1
            The minimum cosine similarity between templates to be considered similar
        seed: int, default: 0
            Random seed
        we_kwargs: dict
            A dictionary of keyword arguments for the WaveformExtractor
        template_mode: "mean" | "median" | "std", default: "median"
            The mode to use to extract templates from the WaveformExtractor

        Returns
        -------
        sortings: dict
            A dictionary that maps method --> NumpySorting.
        gt_sorting: NumpySorting
            The ground-truth sorting used for template matching (with misclassified spike trains).
        """
        try:
            assert 0 <= fraction_misclassed <= 1
        except AssertionError:
            raise ValueError("'fraction_misclassed' must be between 0 and 1.")
        try:
            assert -1 <= min_similarity <= 1
        except AssertionError:
            raise ValueError("'min_similarity' must be between -1 and 1.")
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(dict(seed=seed, overwrite=True, load_if_exists=False, **self.job_kwargs))
        rng = np.random.default_rng(seed)

        # Randomly misclass spike trains
        spike_time_indices, labels = [], []
        for unit_index, unit_id in enumerate(self.we.unit_ids):
            unit_spike_train = self.gt_sorting.get_unit_spike_train(unit_id=unit_id)
            unit_similarity = self.similarity[unit_index, :]
            unit_similarity[unit_index] = min_similarity - 1  # skip self
            similar_unit_ids = self.we.unit_ids[unit_similarity >= min_similarity]
            at_least_one_similar_unit = len(similar_unit_ids)
            num_spikes = int(len(unit_spike_train) * fraction_misclassed)
            unit_misclass_idx = rng.choice(np.arange(len(unit_spike_train)), size=num_spikes, replace=False)
            unit_labels = np.repeat(unit_id, len(unit_spike_train))
            if at_least_one_similar_unit:
                unit_labels[unit_misclass_idx] = rng.choice(similar_unit_ids, size=num_spikes)
            spike_time_indices.extend(list(unit_spike_train))
            labels.extend(list(unit_labels))
        spike_time_indices = np.array(spike_time_indices)
        labels = np.array(labels)
        sort_idx = np.argsort(spike_time_indices)
        spike_time_indices = spike_time_indices[sort_idx]
        labels = labels[sort_idx]
        gt_sorting = NumpySorting.from_times_labels(
            spike_time_indices, labels, self.sampling_rate, unit_ids=self.we.unit_ids
        )
        sort_folder = Path(self.tmp_folder.stem + f"_sorting{len(self.sort_folders)}")
        gt_sorting = gt_sorting.save(folder=sort_folder)
        self.sort_folders.append(sort_folder)

        # Generate New Waveform Extractor with Misclassed Spike Trains
        we = extract_waveforms(self.recording, gt_sorting, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        sortings, _ = self.run_matching(methods_kwargs, we.unit_ids)
        shutil.rmtree(self.tmp_folder)
        return sortings, gt_sorting

    def run_matching_missing_units(
        self, fraction_missing, snr_threshold=0, seed=0, we_kwargs=None, template_mode="median"
    ):
        """Run template matching with a given fraction of missing units.

        Parameters
        ----------
        fraction_missing: float
            The fraction of missing units.
        snr_threshold: float, default: 0
            The SNR threshold below which units are considered missing
        seed: int, default: 0
            Random seed
        we_kwargs: dict
            A dictionary of keyword arguments for the WaveformExtractor.
        template_mode: "mean" | "median" | "std", default: "median"
            The mode to use to extract templates from the WaveformExtractor

        Returns
        -------
        sortings: dict
            A dictionary that maps method --> NumpySorting.
        gt_sorting: NumpySorting
            The ground-truth sorting used for template matching (with missing units).
        """
        try:
            assert 0 <= fraction_missing <= 1
        except AssertionError:
            raise ValueError("'fraction_missing' must be between 0 and 1.")
        try:
            assert snr_threshold >= 0
        except AssertionError:
            raise ValueError("'snr_threshold' must be greater than or equal to 0.")
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(dict(seed=seed, overwrite=True, load_if_exists=False, **self.job_kwargs))
        rng = np.random.default_rng(seed)

        # Omit fraction_missing of units with lowest SNR
        metrics = self.metrics.sort_values("snr")
        missing_units = np.array(metrics.index[metrics.snr < snr_threshold])
        num_missing = int(len(missing_units) * fraction_missing)
        missing_units = rng.choice(missing_units, size=num_missing, replace=False)
        present_units = np.setdiff1d(self.we.unit_ids, missing_units)
        # spike_time_indices, spike_cluster_ids = [], []
        # for unit in present_units:
        #     spike_train = self.gt_sorting.get_unit_spike_train(unit)
        #     for time_index in spike_train:
        #         spike_time_indices.append(time_index)
        #         spike_cluster_ids.append(unit)
        # spike_time_indices = np.array(spike_time_indices)
        # spike_cluster_ids = np.array(spike_cluster_ids)
        # gt_sorting = NumpySorting.from_times_labels(spike_time_indices, spike_cluster_ids, self.sampling_rate,
        #                                             unit_ids=present_units)
        gt_sorting = self.gt_sorting.select_units(present_units)
        sort_folder = Path(self.tmp_folder.stem + f"_sorting{len(self.sort_folders)}")
        gt_sorting = gt_sorting.save(folder=sort_folder)
        self.sort_folders.append(sort_folder)

        # Generate New Waveform Extractor with Missing Units
        we = extract_waveforms(self.recording, gt_sorting, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        sortings, _ = self.run_matching(methods_kwargs, we.unit_ids)
        shutil.rmtree(self.tmp_folder)
        return sortings, gt_sorting

    def run_matching_vary_parameter(
        self,
        parameters,
        parameter_name,
        num_replicates=1,
        we_kwargs=None,
        template_mode="median",
        progress_bars=[],
        **kwargs,
    ):
        """Run template matching varying the values of a given parameter.

        Parameters
        ----------
        parameters: array-like
            The values of the parameter to vary.
        parameter_name: "num_spikes", "fraction_misclassed", "fraction_missing"
            The name of the parameter to vary.
        num_replicates: int, default: 1
            The number of replicates to run for each parameter value
        we_kwargs: dict
            A dictionary of keyword arguments for the WaveformExtractor
        template_mode: "mean" | "median" | "std", default: "median"
            The mode to use to extract templates from the WaveformExtractor
        **kwargs
            Keyword arguments for the run_matching method

        Returns
        -------
        matching_df : pandas.DataFrame
            A dataframe of NumpySortings for each method/parameter_value/iteration combination.
        """
        try:
            run_matching_fn = self.parameter_name2matching_fn[parameter_name]
        except KeyError:
            raise ValueError(f"Parameter name must be one of {list(self.parameter_name2matching_fn.keys())}")
        try:
            progress_bar = self.job_kwargs["progress_bar"]
        except KeyError:
            progress_bar = False
        try:
            assert isinstance(num_replicates, int)
            assert num_replicates > 0
        except AssertionError:
            raise ValueError("num_replicates must be a positive integer")

        sortings, gt_sortings, parameter_values, parameter_names, iter_nums, methods = [], [], [], [], [], []
        if progress_bar:
            parameters = tqdm(parameters, desc=f"Vary Parameter ({parameter_name})")
        for parameter in parameters:
            if progress_bar and num_replicates > 1:
                replicates = tqdm(range(1, num_replicates + 1), desc=f"Replicating for Variability")
            else:
                replicates = range(1, num_replicates + 1)
            for i in replicates:
                sorting_per_method, gt_sorting = run_matching_fn(
                    parameter, seed=i, we_kwargs=we_kwargs, template_mode=template_mode, **kwargs
                )
                for method in self.methods:
                    sortings.append(sorting_per_method[method])
                    gt_sortings.append(gt_sorting)
                    parameter_values.append(parameter)
                    parameter_names.append(parameter_name)
                    iter_nums.append(i)
                    methods.append(method)
                if running_in_notebook():
                    from IPython.display import clear_output

                    clear_output(wait=True)
                    for bar in progress_bars:
                        display(bar.container)
                    display(parameters.container)
                    if num_replicates > 1:
                        display(replicates.container)
        matching_df = pd.DataFrame(
            {
                "sorting": sortings,
                "gt_sorting": gt_sortings,
                "parameter_value": parameter_values,
                "parameter_name": parameter_names,
                "iter_num": iter_nums,
                "method": methods,
            }
        )
        return matching_df

    def compare_sortings(self, gt_sorting, sorting, collision=False, **kwargs):
        """Compare a sorting to a ground-truth sorting.

        Parameters
        ----------
        gt_sorting: SortingExtractor
            The ground-truth sorting extractor.
        sorting: SortingExtractor
            The sorting extractor to compare to the ground-truth.
        collision: bool
            If True, use the CollisionGTComparison class. If False, use the compare_sorter_to_ground_truth function.
        **kwargs
            Keyword arguments for the comparison function.

        Returns
        -------
        comparison: GroundTruthComparison
            The comparison object.
        """
        if collision:
            return CollisionGTComparison(gt_sorting, sorting, exhaustive_gt=self.exhaustive_gt, **kwargs)
        else:
            return compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=self.exhaustive_gt, **kwargs)

    def compare_all_sortings(self, matching_df, collision=False, ground_truth="from_self", **kwargs):
        """Compare all sortings in a matching dataframe to their ground-truth sortings.

        Parameters
        ----------
        matching_df: pandas.DataFrame
            A dataframe of NumpySortings for each method/parameter_value/iteration combination.
        collision: bool
            If True, use the CollisionGTComparison class. If False, use the compare_sorter_to_ground_truth function.
        ground_truth: "from_self" | "from_df", default: "from_self"
            If "from_self", use the ground-truth sorting stored in the BenchmarkMatching object. If "from_df", use the
            ground-truth sorting stored in the matching_df.
        **kwargs
            Keyword arguments for the comparison function.

        Notes
        -----
        This function adds a new column to the matching_df called "comparison" that contains the GroundTruthComparison
        object for each row.
        """
        if ground_truth == "from_self":
            comparison_fn = lambda row: self.compare_sortings(
                self.gt_sorting, row["sorting"], collision=collision, **kwargs
            )
        elif ground_truth == "from_df":
            comparison_fn = lambda row: self.compare_sortings(
                row["gt_sorting"], row["sorting"], collision=collision, **kwargs
            )
        else:
            raise ValueError("'ground_truth' must be either 'from_self' or 'from_df'")
        matching_df["comparison"] = matching_df.apply(comparison_fn, axis=1)

    def plot(self, comp, title=None):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        ax.set_title(title)
        plot_agreement_matrix(comp, ax=ax)
        ax.set_title(title)

        ax = axs[1, 0]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for k in ("accuracy", "recall", "precision"):
            x = comp.get_performance()[k]
            y = self.metrics["snr"]
            ax.scatter(x, y, markersize=10, marker=".", label=k)
        ax.legend()

        ax = axs[0, 1]
        if self.exhaustive_gt:
            plot_comparison_collision_by_similarity(
                comp, self.templates, ax=ax, show_legend=True, mode="lines", good_only=False
            )
        return fig, axs


def plot_errors_matching(benchmark, comp, unit_id, nb_spikes=200, metric="cosine"):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

    benchmark.we.sorting.get_unit_spike_train(unit_id)
    template = benchmark.we.get_template(unit_id)
    a = template.reshape(template.size, 1).T
    count = 0
    colors = ["r", "b"]
    for label in ["TP", "FN"]:
        seg_num = 0  # TODO: make compatible with multiple segments
        idx_1 = np.where(comp.get_labels1(unit_id)[seg_num] == label)
        idx_2 = benchmark.we.get_sampled_indices(unit_id)["spike_index"]
        intersection = np.where(np.isin(idx_2, idx_1))[0]
        intersection = np.random.permutation(intersection)[:nb_spikes]
        if len(intersection) == 0:
            print(f"No {label}s found for unit {unit_id}")
            continue
        ### Should be able to give a subset of waveforms only...
        ax = axs[count, 0]
        plot_unit_waveforms(
            benchmark.we,
            unit_ids=[unit_id],
            axes=[ax],
            unit_selected_waveforms={unit_id: intersection},
            unit_colors={unit_id: colors[count]},
        )
        ax.set_title(label)

        wfs = benchmark.we.get_waveforms(unit_id)
        wfs = wfs[intersection, :, :]

        import sklearn

        nb_spikes = len(wfs)
        b = wfs.reshape(nb_spikes, -1)
        distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
        ax = axs[count, 1]
        ax.set_title(label)
        ax.hist(distances, color=colors[count])
        ax.set_ylabel("# waveforms")
        ax.set_xlabel(metric)

        count += 1
    return fig, axs


def plot_errors_matching_all_neurons(benchmark, comp, nb_spikes=200, metric="cosine"):
    templates = benchmark.templates
    nb_units = len(benchmark.we.unit_ids)
    colors = ["r", "b"]

    results = {"TP": {"mean": [], "std": []}, "FN": {"mean": [], "std": []}}

    for i in range(nb_units):
        unit_id = benchmark.we.unit_ids[i]
        idx_2 = benchmark.we.get_sampled_indices(unit_id)["spike_index"]
        wfs = benchmark.we.get_waveforms(unit_id)
        template = benchmark.we.get_template(unit_id)
        a = template.reshape(template.size, 1).T

        for label in ["TP", "FN"]:
            idx_1 = np.where(comp.get_labels1(unit_id) == label)[0]
            intersection = np.where(np.isin(idx_2, idx_1))[0]
            intersection = np.random.permutation(intersection)[:nb_spikes]
            wfs_sliced = wfs[intersection, :, :]

            import sklearn

            all_spikes = len(wfs_sliced)
            if all_spikes > 0:
                b = wfs_sliced.reshape(all_spikes, -1)
                if metric == "cosine":
                    distances = sklearn.metrics.pairwise.cosine_similarity(a, b).flatten()
                else:
                    distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
                results[label]["mean"] += [np.nanmean(distances)]
                results[label]["std"] += [np.nanstd(distances)]
            else:
                results[label]["mean"] += [0]
                results[label]["std"] += [0]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    for count, label in enumerate(["TP", "FN"]):
        ax = axs[count]
        idx = np.argsort(benchmark.metrics.snr)
        means = np.array(results[label]["mean"])[idx]
        stds = np.array(results[label]["std"])[idx]
        ax.errorbar(benchmark.metrics.snr[idx], means, yerr=stds, c=colors[count])
        ax.set_title(label)
        ax.set_xlabel("snr")
        ax.set_ylabel(metric)
    return fig, axs


def plot_comparison_matching(
    benchmark,
    comp_per_method,
    performance_names=["accuracy", "recall", "precision"],
    colors=["g", "b", "r"],
    ylim=(-0.1, 1.1),
):
    num_methods = len(benchmark.methods)
    fig, axs = plt.subplots(ncols=num_methods, nrows=num_methods, figsize=(10, 10))
    for i, method1 in enumerate(benchmark.methods):
        for j, method2 in enumerate(benchmark.methods):
            if len(axs.shape) > 1:
                ax = axs[i, j]
            else:
                ax = axs[j]
            comp1, comp2 = comp_per_method[method1], comp_per_method[method2]
            if i <= j:
                for performance, color in zip(performance_names, colors):
                    perf1 = comp1.get_performance()[performance]
                    perf2 = comp2.get_performance()[performance]
                    ax.plot(perf2, perf1, ".", label=performance, color=color)

                ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                ax.set_ylim(ylim)
                ax.set_xlim(ylim)
                ax.spines[["right", "top"]].set_visible(False)
                ax.set_aspect("equal")

                if j == i:
                    ax.set_ylabel(f"{method1}")
                else:
                    ax.set_yticks([])
                if i == j:
                    ax.set_xlabel(f"{method2}")
                else:
                    ax.set_xticks([])
                if i == num_methods - 1 and j == num_methods - 1:
                    patches = []
                    for color, name in zip(colors, performance_names):
                        patches.append(mpatches.Patch(color=color, label=name))
                    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            else:
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout(h_pad=0, w_pad=0)
    return fig, axs


def compute_rejection_rate(comp, method="by_unit"):
    missing_unit_ids = set(comp.unit1_ids) - set(comp.unit2_ids)
    performance = comp.get_performance()
    rejection_rates = np.zeros(len(missing_unit_ids))
    for i, missing_unit_id in enumerate(missing_unit_ids):
        rejection_rates[i] = performance.miss_rate[performance.index == missing_unit_id]
    if method == "by_unit":
        return rejection_rates
    elif method == "pooled_with_average":
        return np.mean(rejection_rates)
    else:
        raise ValueError(f'method must be "by_unit" or "pooled_with_average" but got {method}')


def plot_vary_parameter(
    matching_df, performance_metric="accuracy", method_colors=None, parameter_transform=lambda x: x
):
    parameter_names = matching_df.parameter_name.unique()
    methods = matching_df.method.unique()
    if method_colors is None:
        method_colors = {method: f"C{i}" for i, method in enumerate(methods)}
    figs, axs = [], []
    for parameter_name in parameter_names:
        df_parameter = matching_df[matching_df.parameter_name == parameter_name]
        parameters = df_parameter.parameter_value.unique()
        method_means = {method: [] for method in methods}
        method_stds = {method: [] for method in methods}
        for parameter in parameters:
            for method in methods:
                method_param_mask = np.logical_and(
                    df_parameter.method == method, df_parameter.parameter_value == parameter
                )
                comps = df_parameter.comparison[method_param_mask]
                performance_metrics = []
                for comp in comps:
                    try:
                        perf_metric = comp.get_performance(method="pooled_with_average")[performance_metric]
                    except KeyError:  # benchmarking-specific metric
                        assert performance_metric == "rejection_rate", f"{performance_metric} is not a valid metric"
                        perf_metric = compute_rejection_rate(comp, method="pooled_with_average")
                    performance_metrics.append(perf_metric)
                # Average / STD over replicates
                method_means[method].append(np.mean(performance_metrics))
                method_stds[method].append(np.std(performance_metrics))

        parameters_transformed = parameter_transform(parameters)
        fig, ax = plt.subplots()
        for method in methods:
            mean, std = method_means[method], method_stds[method]
            ax.errorbar(
                parameters_transformed, mean, std, color=method_colors[method], marker="o", markersize=5, label=method
            )
        if parameter_name == "num_spikes":
            xlabel = "Number of Spikes"
        elif parameter_name == "fraction_misclassed":
            xlabel = "Fraction of Spikes Misclassified"
        elif parameter_name == "fraction_missing":
            xlabel = "Fraction of Low SNR Units Missing"
        ax.set_xticks(parameters_transformed, parameters)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Average Unit {performance_metric}")
        ax.legend()
        figs.append(fig)
        axs.append(ax)
    return figs, axs
