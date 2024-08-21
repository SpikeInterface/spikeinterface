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

SAVE = False
PLOT = False
BIN_UM = 2

if SAVE:

    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None,
        num_units=15,
        recording_durations=(200, 200),
        recording_shifts=(
            (0, 0), (0, 25),
        ),
        recording_amplitude_scalings=None,
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
    recordings_list, peaks_list, peak_locations_list, bin_um=BIN_UM, histogram_estimation_method="chunked_mean", alignment_method="mean_crosscorr"
)

corrected_peak_locations_list = []

for i in range(len(corrected_recordings_list)):
    corrected_peak_locations_list.append(
        correct_motion_on_peaks(
            peaks_list[i],
            peak_locations_list[i],
            motion_objects_list[i],        # TODO: should probably return directly... or return motion objects not motion estimates
            corrected_recordings_list[i],  #  TODO: why is the recording needed
        )
    )
corrected_hists = []
for i in range(len(corrected_peak_locations_list)):
    corrected_hists.append(
        alignment_utils.get_entire_session_hist(
            recordings_list[1], peaks_list[1], corrected_peak_locations_list[1], bin_um=BIN_UM
        )[0]
    )
x = extra_info["all_spatial_bin_centers"][0]

plot = plotting.SessionAlignmentHistogramWidget(
    [hist_ for hist_ in extra_info["histogram_info"]["all_session_hists"]] + corrected_hists,
    extra_info["all_spatial_bin_centers"][0],
    legend=["session 1", "session 2", "session_1_corrected", "session 2 corrected"],
    linewidths=(1, 1, 2, 2)
)
plt.show()
