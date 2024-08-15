"""
TODO: some notes on this debugging script.
"""
import sys

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
import mp_funcs
from itertools import product
import sys

if __name__ == "__main__":

    BASE_PATH = sys.argv[1]
    METHOD = sys.argv[2]

    print("Number of cpu : ", multiprocessing.cpu_count())

    # SEED = 45
    # np.random.seed(SEED)

    from pathlib import Path
    # output_path = Path("/ceph/neuroinformatics/neuroinformatics/scratch/jziminski")
    output_path = Path(BASE_PATH)  # "/Users/joeziminski/data/alignment_benchmarking"
    si.set_global_job_kwargs(n_jobs=multiprocessing.cpu_count() - 2)

    shifts = np.arange(0, 210, 15)

    benchmark = METHOD

    method_output_path = output_path / benchmark
    method_output_path.mkdir(exist_ok=True, parents=True)

    # TODO: a lot of DRY!
    if benchmark == "num_units":
        unit_numbers = np.array([5, 10, 15, 20, 25, 40, 70, 100, 150, 200])

        all_comb = list(product(shifts, unit_numbers))

        for args in all_comb:  #    num_units_run_func(method_output_path, all_comb[i])

            shift, num_units = args

            all_results = mp_funcs.run_benchmarking(
                shift=shift,
                recording_durations=(100, 100),
                num_units=num_units,
                bin_um=2.5,
                seed=None,
            )

            filename = f"nu-{num_units}_shift-{shift}.pickle"
            mp_funcs.dump_results(method_output_path, filename, all_results)

        if False:
            # This is being super-slow and weird,  ithink because even with
            # n_jobs = 1 spikeinterface is using MP machinery during chunking
            # Not that, it is slow on recording_tools.get_random_data_chunks
            # the get traces call. Not 100% sure why... something to do with
            # how the generator methods are used?
            import multiprocessing
            from functools import partial

            func = partial(num_units_run_func, method_output_path)
            pool = multiprocessing.Pool()
            results = pool.map(func, all_comb)

         #   with ProcessPoolExecutor() as executor:
          #      executor.map(num_units_run_func, all_comb)

    elif benchmark == "bin_size":

        bin_size = np.array([2.5, 5, 10, 20, 40, 60, 90, 120, 240])

        all_comb = list(product(shifts, bin_size))

        for args in all_comb:  #    num_units_run_func(method_output_path, all_comb[i])

            shift, bin_size = args

            all_results = mp_funcs.run_benchmarking(
                shift=shift,
                recording_durations=(100, 100),
                num_units=25,
                bin_um=bin_size,
                seed=None,
            )

            filename = f"bin-{bin_size}_shift-{shift}.pickle"
            mp_funcs.dump_results(method_output_path, filename, all_results)

    elif benchmark == "firing_rates":

        firing_rates = [(0.25, 2), (1, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 150), (150, 300)]

        all_comb = list(product(shifts, firing_rates))

        for args in all_comb:  #    num_units_run_func(method_output_path, all_comb[i])

            shift, bin_size = args

            all_results = mp_funcs.run_benchmarking(
                shift=25,
                recording_durations=(100, 100),
                num_units=25,
                bin_um=2.5,
                firing_rates=firing_rates,
                seed=None,
            )

            filename = f"fr-{firing_rates}_shift-{shift}.pickle"
            mp_funcs.dump_results(method_output_path, filename, all_results)


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
