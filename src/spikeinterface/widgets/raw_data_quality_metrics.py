import numpy as np

from spikeinterface.core import order_channels_by_depth

from .base import BaseWidget


class RawDataQualityMetricsWidget(BaseWidget):
    """
    Plot rolling RMS and PSD, with one column per recording type.

    When rolling RMS was not computed, show only its sampled median summary and PSD.

    Parameters
    ----------
    results : dict
        Output of ``spikeinterface.preprocessing.raw_data_quality_metrics``.
    recordings : dict
        Mapping from result names to recordings, used for channel IDs and locations.
    segment_index : int, default: 0
        Segment to display.
    order_by_depth : bool, default: True
        Order channels by depth when a recording with channel locations is supplied.
        Otherwise, retain recording order. The lowest channel is at the bottom.
    """

    def __init__(self, results, recordings, segment_index=0, order_by_depth=True, backend=None, **backend_kwargs):

        display_names = {
            "raw_ap_recording": "Raw",
            "preprocessed_ap_recording": "Preprocessed",
            "raw_lfp_recording": "LFP",
        }

        recording_data = {}
        for recording_name, summary in results["chunked_results"].items():

            recording = recordings[recording_name]
            channel_indices = np.arange(recording.get_num_channels())
            if order_by_depth and recording.has_channel_location():
                channel_indices, _ = order_channels_by_depth(recording)

            channel_ids = recording.get_channel_ids()[channel_indices]

            rolling = results["rolling_rms"].get(recording_name)

            title = display_names.get(recording_name, recording_name)
            if len(summary["rms_median"]) > 1:
                title += f" | segment {segment_index}"

            # The saved RMS medians are per channel; summarize them across channels for the title.
            recording_data[recording_name] = dict(
                title=title,
                rms=rolling["rms"][segment_index][:, channel_indices] if rolling is not None else None,
                times=rolling["times"][segment_index] if rolling is not None else None,
                psd=summary["psd"][segment_index][:, channel_indices],
                freqs=summary["freqs"][segment_index],
                median_channel_rms=np.median(summary["rms_median"][segment_index]),
                channel_ids=channel_ids,
            )

        data_plot = dict(recording_data=recording_data)
        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):

        from matplotlib.colors import LogNorm
        from .utils_matplotlib import make_mpl_figure

        num_recordings = len(data_plot["recording_data"])
        backend_kwargs["ncols"] = num_recordings
        backend_kwargs["num_axes"] = 2 * num_recordings
        if backend_kwargs["figsize"] is None:
            backend_kwargs["figsize"] = (5 * num_recordings, 7)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        self.axes = np.asarray(self.axes).reshape(2, num_recordings)

        for column_index, (recording_name, recording_data) in enumerate(data_plot["recording_data"].items()):

            rms_axis, psd_axis = self.axes[:, column_index]

            rms_axis.text(
                0.5,
                1.28,
                recording_data["title"],
                transform=rms_axis.transAxes,
                ha="center",
                fontsize=18,
                fontweight="bold",
            )

            if recording_data["rms"] is not None:

                rms_axis.set_title("RMS", fontsize=15, pad=26)
                rms_axis.text(
                    0.5,
                    1.03,
                    f"(median RMS: {recording_data['median_channel_rms']:.2f} µV)",
                    transform=rms_axis.transAxes,
                    ha="center",
                    fontsize=10,
                )

                rms_image = self._plot_map(rms_axis, recording_data["times"], recording_data["rms"])
                rms_axis.set_xlabel("Time (s)")
                self.figure.colorbar(rms_image, ax=rms_axis, label="RMS (µV)")

            else:
                rms_axis.set_axis_off()

            psd_axis.set_title("PSD", fontsize=15)

            # Zero power cannot be shown on a log scale; keep all-zero data linear.
            psd = recording_data["psd"]
            norm = None
            if np.any(psd > 0):
                psd = np.ma.masked_less_equal(psd, 0)
                norm = LogNorm()

            psd_image = self._plot_map(psd_axis, recording_data["freqs"], psd, norm=norm)
            psd_axis.set_xlabel("Frequency (Hz)")
            psd_axis.set_xlim(left=0)
            self.figure.colorbar(psd_image, ax=psd_axis, label="PSD (µV²/Hz)")

            num_channels = len(recording_data["channel_ids"])
            tick_indices = np.linspace(0, num_channels - 1, min(8, num_channels), dtype=int)

            for axis in (psd_axis, rms_axis):
                axis.set_yticks(tick_indices, labels=recording_data["channel_ids"][tick_indices])
                axis.set_ylim(-0.5, num_channels - 0.5)
                axis.set_ylabel("Channel ID")

        self.figure.tight_layout()

    def _plot_map(self, axis, coordinates, values, norm=None):
        """Plot a channel map."""
        channel_indices = np.arange(values.shape[1])
        return axis.pcolormesh(coordinates, channel_indices, values.T, shading="nearest", cmap="viridis", norm=norm)
