from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import numpy as np
import plotting
import alignment_utils
import matplotlib.pyplot as plt
import pickle
scalings = [np.ones(10), np.r_[np.zeros(3), np.ones(7)]]

SAVE = True

if SAVE:
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None,
        num_units=35,
        recording_durations=(100, 100),
        recording_shifts=(
            (0, 0), (0, 0),
        ),
        recording_amplitude_scalings=None, # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        seed=None,
    )

    peaks_list = []
    peak_locations_list = []

    for recording in recordings_list:
        peaks, peak_locations = alignment_utils.prep_recording(
            recording, plot=False,
        )
        peaks_list.append(peaks)
        peak_locations_list.append(peak_locations)

    # something relatively easy, only 15 units
    with open('all_recordings.pickle', 'wb') as handle:
        pickle.dump((recordings_list, peaks_list, peak_locations_list),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('all_recordings.pickle', 'rb') as handle:
    recordings_list, peaks_list, peak_locations_list = pickle.load(handle)

bin_um = 2

# TODO: own function
min_y = np.min([np.min(locs["y"]) for locs in peak_locations_list])
max_y = np.max([np.max(locs["y"]) for locs in peak_locations_list])

spatial_bin_edges = np.arange(min_y, max_y + bin_um, bin_um)  # TODO: expose a margin...
spatial_bin_centers = alignment_utils.get_bin_centers(spatial_bin_edges)  # TODO: own function

session_histogram_list = []
for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

    hist, temp, spat = alignment_utils.get_entire_session_hist(recording, peaks, peak_locations, spatial_bin_edges, log_scale=False)

    session_histogram_list.append(
        hist
    )
# TODO: need to get all outputs and check are same size
plotting.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    session_histogram_list,
    histogram_spatial_bin_centers=spatial_bin_centers,
)
plt.show()
