import numpy as np

from .motion_utils import Motion
import tempfile
import shutil
from pathlib import Path


class MedecineRegistration:
    """
    """

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

        #unsed need to be adapted
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        verbose,
        progress_bar,
        extra,

        bin_um=5.0,
        hist_margin_um=20.0,
        bin_s=2.0,

    ):
        
        from medicine import run_medicine

        folder = Path(tempfile.gettempdir())

        run_medicine(
            peak_amplitudes=peaks['amplitude'],
            peak_depths=peak_locations[direction],
            peak_times=peaks['sample_index'] / recording.get_sampling_frequency(),
            output_dir=folder,
            plot_figures=False,
        )

        # Load motion estimated by MEDiCINe
        motion_array = np.load(folder / 'motion.npy')
        time_bins = np.load(folder / 'time_bins.npy')
        depth_bins = np.load(folder / 'depth_bins.npy')

        # Use interpolation to correct for motion estimated by MEDiCINe
        motion = Motion(
            displacement=[motion_array],
            temporal_bins_s=[time_bins],
            spatial_bins_um=depth_bins,
        )

        shutil.rmtree(folder)

        return motion
