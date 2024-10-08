from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class ComparisonCollisionBySimilarityWidget(BaseWidget):
    """
    Plots CollisionGTComparison pair by pair orderer by cosine_similarity

    Parameters
    ----------
    comp : CollisionGTComparison
        The collision ground truth comparison object
    templates : array
        template of units
    mode : "heatmap" or "lines"
        to see collision curves for every pairs ("heatmap") or as lines averaged over pairs.
    similarity_bins : array
        if mode is "lines", the bins used to average the pairs
    cmap : string
        colormap used to show averages if mode is "lines"
    metric : "cosine_similarity"
        metric for ordering
    good_only : True
        keep only the pairs with a non zero accuracy (found templates)
    min_accuracy : float
        If good only, the minimum accuracy every cell should have, individually, to be
        considered in a putative pair
    unit_ids : list
        List of considered units
    """

    def __init__(
        self,
        comp,
        templates,
        unit_ids=None,
        metric="cosine_similarity",
        figure=None,
        ax=None,
        mode="heatmap",
        similarity_bins=np.linspace(-0.4, 1, 8),
        cmap="winter",
        good_only=False,
        min_accuracy=0.9,
        show_legend=False,
        ylim=(0, 1),
        backend=None,
        **backend_kwargs,
    ):
        if unit_ids is None:
            unit_ids = comp.sorting1.get_unit_ids()

        data_plot = dict(
            comp=comp,
            templates=templates,
            unit_ids=unit_ids,
            metric=metric,
            mode=mode,
            similarity_bins=similarity_bins,
            cmap=cmap,
            good_only=good_only,
            min_accuracy=min_accuracy,
            show_legend=show_legend,
            ylim=ylim,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import sklearn
        import matplotlib

        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        # self.make_mpl_figure(**backend_kwargs)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        comp = dp.comp

        # compute similarity
        # take index of template (respect unit_ids order)
        all_unit_ids = list(comp.sorting1.get_unit_ids())
        template_inds = [all_unit_ids.index(u) for u in dp.unit_ids]

        templates = dp.templates[template_inds, :, :].copy()
        flat_templates = templates.reshape(templates.shape[0], -1)
        if dp.metric == "cosine_similarity":
            similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
        else:
            raise NotImplementedError("metric=...")

        fs = comp.sorting1.get_sampling_frequency()
        lags = comp.bins / fs * 1000

        n = len(dp.unit_ids)

        similarities, recall_scores, pair_names = comp.compute_collision_by_similarity(
            similarity_matrix, unit_ids=dp.unit_ids, good_only=dp.good_only, min_accuracy=dp.min_accuracy
        )

        if dp.mode == "heatmap":
            fig = self.figure
            for ax in fig.axes:
                ax.remove()

            n_pair = len(similarities)

            ax0 = fig.add_axes([0.1, 0.1, 0.25, 0.8])
            ax1 = fig.add_axes([0.4, 0.1, 0.5, 0.8], sharey=ax0)

            plt.setp(ax1.get_yticklabels(), visible=False)

            im = ax1.imshow(
                recall_scores[::-1, :],
                cmap="viridis",
                aspect="auto",
                interpolation="none",
                extent=(lags[0], lags[-1], -0.5, n_pair - 0.5),
            )
            im.set_clim(0, 1)

            ax0.plot(similarities, np.arange(n_pair), color="k")

            ax0.set_yticks(np.arange(n_pair))
            ax0.set_yticklabels(pair_names)
            # ax0.set_xlim(0,1)

            ax0.set_xlabel(dp.metric)
            ax0.set_ylabel("pairs")

            ax1.set_xlabel("lag (ms)")
        elif dp.mode == "lines":
            my_cmap = plt.colormaps[dp.cmap]
            cNorm = matplotlib.colors.Normalize(vmin=dp.similarity_bins.min(), vmax=dp.similarity_bins.max())
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

            # plot by similarity bins
            if self.ax is None:
                fig, ax = plt.subplots()
            else:
                ax = self.ax
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            order = np.argsort(similarities)
            similarities = similarities[order]
            recall_scores = recall_scores[order, :]

            for i in range(dp.similarity_bins.size - 1):
                cmin, cmax = dp.similarity_bins[i], dp.similarity_bins[i + 1]

                amin, amax = np.searchsorted(similarities, [cmin, cmax])
                mean_recall_scores = np.nanmean(recall_scores[amin:amax], axis=0)

                colorVal = scalarMap.to_rgba((cmin + cmax) / 2)
                ax.plot(
                    lags[:-1] + (lags[1] - lags[0]) / 2,
                    mean_recall_scores,
                    label="CS in [%g,%g]" % (cmin, cmax),
                    c=colorVal,
                )

            if dp.show_legend:
                ax.legend()
            ax.set_ylim(dp.ylim)
            ax.set_xlabel("lags (ms)")
            ax.set_ylabel("collision recall")


class StudyComparisonCollisionBySimilarityWidget(BaseWidget):
    """
    Plots CollisionGTComparison pair by pair orderer by cosine_similarity for all
    cases in a study.

    Parameters
    ----------
    study : CollisionGTStudy
        The collision study object.
    case_keys : list or None
        A selection of cases to plot, if None, then all.
    metric : "cosine_similarity"
        metric for ordering
    similarity_bins : array
        if mode is "lines", the bins used to average the pairs
    cmap : string
        colormap used to show averages if mode is "lines"
    good_only : False
        keep only the pairs with a non zero accuracy (found templates)
    min_accuracy : float
        If good only, the minimum accuracy every cell should have, individually, to be
        considered in a putative pair
    """

    def __init__(
        self,
        study,
        case_keys=None,
        metric="cosine_similarity",
        similarity_bins=np.linspace(-0.4, 1, 8),
        show_legend=False,
        ylim=(0.5, 1),
        good_only=False,
        min_accuracy=0.9,
        cmap="winter",
        backend=None,
        **backend_kwargs,
    ):
        if case_keys is None:
            case_keys = list(study.cases.keys())

        data_plot = dict(
            study=study,
            case_keys=case_keys,
            metric=metric,
            similarity_bins=similarity_bins,
            show_legend=show_legend,
            ylim=ylim,
            good_only=good_only,
            min_accuracy=min_accuracy,
            cmap=cmap,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import sklearn
        import matplotlib

        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        num_axes = len(dp.case_keys)
        backend_kwargs["num_axes"] = num_axes

        # self.make_mpl_figure(**backend_kwargs)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        study = dp.study

        my_cmap = plt.colormaps[dp.cmap]
        cNorm = matplotlib.colors.Normalize(vmin=dp.similarity_bins.min(), vmax=dp.similarity_bins.max())
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)
        study.precompute_scores_by_similarities(
            case_keys=dp.case_keys,
            good_only=dp.good_only,
            min_accuracy=dp.min_accuracy,
        )

        for count, key in enumerate(dp.case_keys):
            lags = study.get_lags(key)

            curves = study.get_lag_profile_over_similarity_bins(dp.similarity_bins, key)

            # plot by similarity bins
            ax = self.axes.flatten()[count]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for i in range(dp.similarity_bins.size - 1):
                cmin, cmax = dp.similarity_bins[i], dp.similarity_bins[i + 1]
                colorVal = scalarMap.to_rgba((cmin + cmax) / 2)
                ax.plot(
                    lags[:-1] + (lags[1] - lags[0]) / 2,
                    curves[(cmin, cmax)],
                    label="CS in [%g,%g]" % (cmin, cmax),
                    c=colorVal,
                )

            if count % self.axes.shape[1] == 0:
                ax.set_ylabel("collision recall")

            if count > (len(dp.case_keys) // self.axes.shape[1]):
                ax.set_xlabel("lags (ms)")

            label = study.cases[key]["label"]
            ax.set_title(label)
            if dp.show_legend:
                ax.legend()

            if dp.ylim is not None:
                ax.set_ylim(dp.ylim)
