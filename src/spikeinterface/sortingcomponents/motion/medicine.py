import numpy as np

import tempfile
import shutil
from pathlib import Path

from spikeinterface.core.motion import Motion
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows


class MedicineRegistration:
    """ """

    name = "medicine"
    need_peak_location = True
    params_doc = """

    """

    @classmethod
    def run(
        cls,
        recording,
        peaks,
        peak_locations,
        direction,
        # unsed need to be adapted
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        verbose,
        progress_bar,
        extra,
        bin_s=1.0,
        ## medicine specific kwargs propagated to the lib
        motion_bound=800,
        time_kernel_width=30,
        activity_network_hidden_features=(256, 256),
        amplitude_threshold_quantile=0.0,
        batch_size=4096,
        training_steps=10_000,
        initial_motion_noise=0.1,
        motion_noise_steps=2000,
        optimizer=None,
        learning_rate=0.0005,
        epsilon=1e-3,
    ):

        from medicine import run_medicine

        # folder = Path(tempfile.gettempdir())

        if rigid:
            # force one bin
            num_depth_bins = 1
        else:

            # we use the spatial window mechanism only to estimate the number one spatial bins
            dim = ["x", "y", "z"].index(direction)
            contact_depths = recording.get_channel_locations()[:, dim]

            deph_range = max(contact_depths) - min(contact_depths)
            if win_margin_um is not None:
                deph_range = deph_range - 2 * win_margin_um
            num_depth_bins = max(int(np.round(deph_range / win_scale_um)), 1)
            print("num_depth_bins", num_depth_bins)

        if optimizer is None:
            import torch

            optimizer = torch.optim.Adam

        trainer, time_bins, depth_bins, pred_motion = run_medicine(
            peak_amplitudes=peaks["amplitude"],
            peak_depths=peak_locations[direction],
            peak_times=peaks["sample_index"] / recording.get_sampling_frequency(),
            time_bin_size=bin_s,
            num_depth_bins=num_depth_bins,
            output_dir=None,
            plot_figures=False,
            motion_bound=motion_bound,
            time_kernel_width=time_kernel_width,
            activity_network_hidden_features=activity_network_hidden_features,
            amplitude_threshold_quantile=amplitude_threshold_quantile,
            batch_size=batch_size,
            training_steps=training_steps,
            initial_motion_noise=initial_motion_noise,
            motion_noise_steps=motion_noise_steps,
            optimizer=optimizer,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )

        motion = Motion(
            displacement=[np.array(pred_motion)],
            temporal_bins_s=[np.array(time_bins)],
            spatial_bins_um=np.array(depth_bins),
        )

        return motion
