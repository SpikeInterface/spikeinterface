from __future__ import annotations

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
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks
from spikeinterface.widgets.motion import DriftRasterMapWidget
from spikeinterface.widgets.base import BaseWidget
import plotting


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
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

# TODO: add different modes (to mean, to nth session...)
# TODO: document that the output is Hz

# What we really want to find is maximal subset of the data that matches

SAVE = False
PLOT = False
BIN_UM = 2

if SAVE:
    scalings = [np.ones(25), np.r_[np.zeros(10), np.ones(15)]]  # TODO: there is something wrong here, because why are the maximum histograms not removed?
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=0.1, # None,
        num_units=25,
        recording_durations=(100, 100),
        recording_shifts=(
            (0, 0), (0, 75),
        ),
        recording_amplitude_scalings=None, # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        seed=None,
    )

    peaks_list = []
    peak_locations_list = []

    for recording in recordings_list:
        peaks, peak_locations = alignment_utils.prep_recording(
            recording, plot=PLOT,
        )
        peaks_list.append(peaks)
        peak_locations_list.append(peak_locations)

    # something relatively easy, only 15 units
    with open('all_recordings.pickle', 'wb') as handle:
        pickle.dump((recordings_list, peaks_list, peak_locations_list),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('all_recordings.pickle', 'rb') as handle:
    recordings_list, peaks_list, peak_locations_list = pickle.load(handle)


corrected_recordings_list, motion_objects_list, extra_info = session_alignment.run_inter_session_displacement_correction(
    recordings_list, peaks_list, peak_locations_list, bin_um=BIN_UM, histogram_estimation_method="entire_session", alignment_method="mean_crosscorr", rigid=True
)

plotting.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["histogram_info"]["all_session_hists"],
    **extra_info["corrected"]
)
plt.show()
