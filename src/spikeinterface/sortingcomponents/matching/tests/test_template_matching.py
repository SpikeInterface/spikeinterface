import pytest
import numpy as np

from spikeinterface import (
    NumpyRecording,
    Templates,
    create_sorting_analyzer,
    compute_sparsity,
    get_noise_levels,
)

from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.sortingcomponents.matching.method_list import matching_methods


from spikeinterface.sortingcomponents.tests.common import make_dataset

# job_kwargs = dict(n_jobs=-1, chunk_duration="500ms", progress_bar=True)
job_kwargs = dict(n_jobs=1, chunk_duration="500ms", progress_bar=True)


def get_sorting_analyzer():
    recording, sorting = make_dataset()
    sorting_analyzer = create_sorting_analyzer(sorting, recording, sparse=False)
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("templates", **job_kwargs)
    sorting_analyzer.compute("noise_levels")
    return sorting_analyzer


@pytest.fixture(name="sorting_analyzer", scope="module")
def sorting_analyzer_fixture():
    return get_sorting_analyzer()


@pytest.mark.parametrize("method", matching_methods.keys())
def test_find_spikes_from_templates(method, sorting_analyzer):
    recording = sorting_analyzer.recording
    # waveform = waveform_extractor.get_waveforms(waveform_extractor.unit_ids[0])
    # num_waveforms, _, _ = waveform.shape
    # assert num_waveforms != 0

    templates = sorting_analyzer.get_extension("templates").get_data(outputs="Templates")
    sparsity = compute_sparsity(sorting_analyzer, method="snr", threshold=0.5)
    templates = templates.to_sparse(sparsity)

    # sorting_analyzer
    method_kwargs = {}

    if method == "kilosort-matching":
        from spikeinterface.sortingcomponents.tools import get_prototype_and_waveforms

        prototype, wfs, _ = get_prototype_and_waveforms(recording, ms_before=1, ms_after=2)

        n_components = 5
        from sklearn.cluster import KMeans

        wfs /= np.linalg.norm(wfs, axis=1)[:, None]
        model = KMeans(n_clusters=n_components, n_init=10).fit(wfs)
        temporal_components = model.cluster_centers_
        temporal_components = temporal_components / np.linalg.norm(temporal_components, axis=1)[:, None]
        temporal_components = temporal_components.astype(np.float32)

        from sklearn.decomposition import TruncatedSVD

        model = TruncatedSVD(n_components=n_components).fit(wfs)
        spatial_components = model.components_.astype(np.float32)
        method_kwargs["spatial_components"] = spatial_components
        method_kwargs["temporal_components"] = temporal_components

    if matching_methods[method].need_noise_levels:
        method_kwargs["noise_levels"] = get_noise_levels(recording, return_in_uV=False)

    if method == "nearest-svd":
        from spikeinterface.sortingcomponents.tools import get_prototype_and_waveforms

        _, wfs, _ = get_prototype_and_waveforms(recording, ms_before=1, ms_after=2)
        n_components = 5
        from sklearn.decomposition import TruncatedSVD

        svd_model = TruncatedSVD(n_components=n_components)
        svd_model.fit(wfs)
        method_kwargs["svd_model"] = svd_model

    # method_kwargs["wobble"] = {
    #     "templates": waveform_extractor.get_all_templates(),
    #     "nbefore": waveform_extractor.nbefore,
    #     "nafter": waveform_extractor.nafter,
    # }

    spikes, info = find_spikes_from_templates(
        recording, templates, method=method, method_kwargs=method_kwargs, extra_outputs=True, job_kwargs=job_kwargs
    )

    # print(info)

    # DEBUG = True
    # DEBUG = False

    # if DEBUG:
    #     sorting_analyzer.compute("waveforms")
    #     sorting_analyzer.compute("templates")

    #     gt_sorting = sorting_analyzer.sorting

    #     sorting = NumpySorting.from_samples_and_labels(
    #         spikes["sample_index"], spikes["cluster_index"], recording.sampling_frequency
    #     )

    #     #metrics = si.compute_quality_metrics(sorting_analyzer, metric_names=["snr"])

    #     fig, ax = plt.subplots()
    #     comp = si.compare_sorter_to_ground_truth(gt_sorting, sorting)
    #     si.plot_agreement_matrix(comp, ax=ax)
    #     ax.set_title(method)
    #     plt.show()


def test_circus_omp_overlapping_spikes():
    # noiseless overlapping spikes, spaced so that re-fits reach spikes whose neighbours lie outside them,
    # where selecting a template twice at the same time loses or misfits a spike
    sampling_frequency, nbefore, num_samples, num_channels = 30000.0, 20, 60, 4
    t = np.arange(num_samples) - nbefore
    templates_array = np.zeros((3, num_samples, num_channels), dtype=np.float32)
    for unit in range(3):
        width = 3 + 1.5 * unit
        waveform = -np.exp(-0.5 * (t / width) ** 2) + 0.4 * np.exp(-0.5 * ((t - 2.5 * width) / (2 * width)) ** 2)
        profile = np.exp(-0.5 * ((np.arange(num_channels) - 1.5 * unit) / 1.5) ** 2)
        # rank 1, so the matcher's SVD keeps the templates exactly
        templates_array[unit] = 10 * np.outer(waveform, profile)

    units, times = [1, 0, 0, 1, 2], [120, 138, 178, 220, 240]
    traces = np.zeros((420, num_channels), dtype=np.float32)
    for unit, time in zip(units, times):
        traces[time - nbefore : time - nbefore + num_samples] += templates_array[unit]
    recording = NumpyRecording([traces], sampling_frequency)
    templates = Templates(
        templates_array, sampling_frequency, nbefore, channel_ids=recording.channel_ids, unit_ids=np.arange(3)
    )

    spikes = find_spikes_from_templates(recording, templates, method="circus-omp", job_kwargs=dict(n_jobs=1))
    assert spikes["cluster_index"].tolist() == units
    assert spikes["sample_index"].tolist() == times
    assert np.allclose(spikes["amplitude"], 1, atol=0.01)


if __name__ == "__main__":
    sorting_analyzer = get_sorting_analyzer()
    # method = "nearest"
    method = "nearest-svd"
    # method = "tdc-peeler"
    # method = "circus-omp-svd"
    # method = "wobble"
    # method = "kilosort-matching"

    test_find_spikes_from_templates(method, sorting_analyzer)
