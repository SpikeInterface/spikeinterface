import numpy as np

from .motion_utils import Motion
import tempfile
import shutil
from pathlib import Path

from .motion_utils import get_spatial_windows


class MedecineRegistration:
    """ """

    name = "medecine"
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
        # bin_um=5.0,
        # hist_margin_um=20.0,
        bin_s=1.0,
        time_kernel_width=30.0,
        amplitude_threshold_quantile=0.0,
        ####
        training_steps=10_000,
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

        trainer, motion = run_medicine(
            peak_amplitudes=peaks["amplitude"],
            peak_depths=peak_locations[direction],
            peak_times=peaks["sample_index"] / recording.get_sampling_frequency(),
            time_bin_size=bin_s,
            num_depth_bins=num_depth_bins,
            training_steps=training_steps,
            time_kernel_width=time_kernel_width,
            amplitude_threshold_quantile=amplitude_threshold_quantile,
            output_dir=None,
            plot_figures=False,
            return_motion=True,
        )

        # Load motion estimated by MEDiCINe
        # motion_array = np.load(folder / 'motion.npy')
        # time_bins = np.load(folder / 'time_bins.npy')
        # depth_bins = np.load(folder / 'depth_bins.npy')

        # # Use interpolation to correct for motion estimated by MEDiCINe
        # motion = Motion(
        #     displacement=[motion_array],
        #     temporal_bins_s=[time_bins],
        #     spatial_bins_um=depth_bins,
        # )

        # TODO check why not working
        # shutil.rmtree(folder)

        return motion
