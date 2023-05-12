import numpy as np
from typing import Union

from .base import BaseWidget
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting


class TemplateSimilarityWidget(BaseWidget):
    """
    Plots unit template similarity.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The object to compute/get template similarity from
    unit_ids : list
        List of unit ids default None
    display_diagonal_values : bool
        If False, the diagonal is displayed as zeros.
        If True, the similarity values (all 1s) are displayed, default False
    cmap : str
        The matplotlib colormap. Default 'viridis'.
    show_unit_ticks : bool
        If True, ticks display unit ids, default False.
    show_colorbar : bool
        If True, color bar is displayed, default True.
    """

    possible_backends = {}

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        unit_ids=None,
        cmap="viridis",
        display_diagonal_values=False,
        show_unit_ticks=False,
        show_colorbar=True,
        backend=None,
        **backend_kwargs,
    ):
        self.check_extensions(waveform_extractor, "similarity")
        tsc = waveform_extractor.load_extension("similarity")
        similarity = tsc.get_data().copy()

        sorting = waveform_extractor.sorting
        if unit_ids is None:
            unit_ids = sorting.unit_ids
        else:
            unit_indices = sorting.ids_to_indices(unit_ids)
            similarity = similarity[unit_indices][:, unit_indices]

        if not display_diagonal_values:
            np.fill_diagonal(similarity, 0)

        plot_data = dict(
            similarity=similarity,
            unit_ids=unit_ids,
            cmap=cmap,
            show_unit_ticks=show_unit_ticks,
            show_colorbar=show_colorbar,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)
