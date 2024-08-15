"""
TODO: some notes on this debugging script.
"""
import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import \
    make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
import alignment_utils
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

if __name__ == "__main__":

    print("Number of cpu : ", multiprocessing.cpu_count())

    # SEED = 45
    # np.random.seed(SEED)

    from pathlib import Path
    output_path = Path("/Users/joeziminski/data/alignment_benchmarking")

    si.set_global_job_kwargs(n_jobs=1)

    shifts = np.arange(0, 210, 15)

    benchmark = "num_units"

    if benchmark == "num_units":
        unit_numbers = np.array([5, 10, 15, 20, 25, 40, 70, 100, 150, 200])

        from itertools import product

        method_output_path = output_path / "num_units"
        method_output_path.mkdir(exist_ok=True)

        from mp_funcs import num_units_run_func

        all_comb = list(product(shifts, unit_numbers))

        # for i in range(len(all_comb)):
          #   num_units_run_func(all_comb[i])

        with ProcessPoolExecutor() as executor:
            executor.map(num_units_run_func, all_comb)

    elif benchmark == "bin_size":
        pass

    elif benchmark == "firing_rates":
        pass


    if False:
        results_arrays = {d: [] for d in results[0]["motion_arrays"].keys()}
        legend = []
        for key in results[0]["motion_arrays"].keys():
            for i in range(len(results)):
                ses_info = results[i]
                results_arrays[key].append(ses_info["motion_arrays"][key][1, 0] - all_shifts[i])
            plt.plot(all_shifts, results_arrays[key])
            legend.append(key)

        plt.legend(legend)
        plt.show()
