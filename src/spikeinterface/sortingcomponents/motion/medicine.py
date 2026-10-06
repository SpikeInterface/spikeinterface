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
    output_dir : str | Path | None, default: None
        Directory to save MEDiCINe outputs (parameters, figures, motion arrays). If None, nothing is saved to disk.
    plot_figures : bool, default: False
        Whether MEDiCINe should plot and save figures summarizing the model results.
    num_depth_bins : int | None, default: None
        Number of depth bins for motion estimation, passed to MEDiCINe. If None, this defaults to MEDiCINe's own
        recommended default: 1 if `rigid` else 2. Note this is intentionally independent of the generic
        `win_scale_um` / `win_step_um` / `win_margin_um` windowing parameters used by other motion estimation
        methods, since MEDiCINe is not tuned for the number of depth bins those would otherwise imply.
    motion_bound : float, default: 800
        Bound on the maximum absolute motion allowed, in the same units as the spike depths (typically microns).
    time_kernel_width : float, default: 30
        Width of the temporal smoothing kernel, in the same units as the spike times (typically seconds).
    activity_network_hidden_features : tuple, default: (256, 256)
        Hidden layer sizes for MEDiCINe's activity network.
    amplitude_threshold_quantile : float, default: 0.0
        Cutoff quantile in [-1, 1] for peak amplitudes. See MEDiCINe's documentation for details.
    batch_size : int, default: 4096
        Batch size used for training.
    training_steps : int, default: 10000
        Number of optimization steps to take.
    initial_motion_noise : float, default: 0.1
        Initial magnitude of noise added to the motion function output, annealed to 0 over `motion_noise_steps`.
    motion_noise_steps : int, default: 2000
        Number of training steps over which `initial_motion_noise` is annealed to 0.
    optimizer : torch.optim.Optimizer | None, default: None
        Optimizer class used for training. If None, `torch.optim.Adam` is used.
    learning_rate : float, default: 0.0005
        Learning rate used for training.
    epsilon : float, default: 1e-3
        Small value to prevent divide-by-zero instabilities.
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
        output_dir=None,
        plot_figures=False,
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
        num_depth_bins=None,
    ):

        from medicine import run_medicine

        if num_depth_bins is None:
            # MEDiCINe's own recommended default (see medicine.run.run_medicine) is 2 non-rigid depth bins.
            # We intentionally do not derive this from win_scale_um/win_margin_um (the generic windowing
            # parameters used by other motion estimation methods), since those are tuned for different
            # algorithms and would otherwise silently produce far more depth bins than MEDiCINe expects.
            num_depth_bins = 1 if rigid else 2

        if optimizer is None:
            import torch

            optimizer = torch.optim.Adam

        trainer, time_bins, depth_bins, pred_motion = run_medicine(
            peak_amplitudes=peaks["amplitude"],
            peak_depths=peak_locations[direction],
            peak_times=peaks["sample_index"] / recording.get_sampling_frequency(),
            time_bin_size=bin_s,
            num_depth_bins=num_depth_bins,
            output_dir=output_dir,
            plot_figures=plot_figures,
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
