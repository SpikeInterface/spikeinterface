from spikeinterface import WaveformExtractor
import datetime
import numpy as np
from spikeinterface.toolkit.postprocessing.template_tools import (
    get_template_best_channels,
)
from spikeinterface.core.job_tools import (
    ensure_n_jobs,
    ensure_chunk_size,
    devide_recording_into_chunks,
)
from scipy.sparse import csr_matrix
from pathlib2 import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from joblib import Parallel
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from umap.umap_ import nearest_neighbors, check_random_state
import matplotlib.pyplot as plt
import seaborn as sns
import numba
import tensorflow as tf



class WaveformParametricUMAP:
    def __init__(
        self,
        n_channels,
        model_save_path,
        n_neighbors=5,
        batch_size=1024,
        load_if_exists=True,
        nbefore=30,
        nafter=50,
        n_components=2,
        max_channels_per_template=8,
        encoder_network=None,
        decoder_network=None,
        parametric_umap_kwargs={"min_dist": 0.5},
        verbose=True,
        temporal_neighborhood=False,
        temporal_neighborhood_time_hours=5,
    ):
        self.n_components = n_components
        self.max_channels_per_template = np.min([n_channels, max_channels_per_template])
        self.nbefore = nbefore
        self.nafter = nafter
        self.model_save_path = Path(model_save_path)
        self.verbose = verbose
        self.n_channels = n_channels
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.dims = [self.nbefore + self.nafter, self.n_channels, 1]
        self.temporal_neighborhood = temporal_neighborhood
        self.temporal_neighborhood_time_hours = temporal_neighborhood_time_hours

        if encoder_network is None:
            n_samples = nbefore + nafter
            n_dense = max_channels_per_template
            encoder_network = make_encoder_network(
                n_components,
                n_dense=n_dense,
                n_samples=n_samples,
                n_channels=self.n_channels,
                batch_size=batch_size,
                verbose=verbose,
            )

        if self.model_save_path.exists() and load_if_exists:
            self.model = load_ParametricUMAP(self.model_save_path)
            # load encoder tflite
        else:
            # create the model
            self.model = ParametricUMAP(
                n_components=n_components,
                encoder=encoder_network,
                decoder=decoder_network,
                n_neighbors=self.n_neighbors,
                metric="precomputed",
                # dims=self.dims,
                verbose=self.verbose,
                batch_size=self.batch_size,
                **parametric_umap_kwargs
            )

    def train(
        self,
        waveform_extractor_paths,
        recording_good_units_list,
        best_channels_index_list,
        n_spikes_per_recording_list,
        epoch_size=100000,
        sample_chunk_size=10000,
        min_chunks_to_sample=10,
        n_training_epochs=1,
        template_best_channels_args={"peak_sign": "neg"},
        n_jobs=1,
    ):
        # set recording best channel list as None if it hasn't been computed yet
        if best_channels_index_list is None:
            best_channels_index_list = np.repeat(None, len(waveform_extractor_paths))

        # set recording best channel list as None
        if recording_good_units_list is None:
            recording_good_units_list = np.repeat(None, len(waveform_extractor_paths))

        # get the total number of spikes per recording
        if n_spikes_per_recording_list is None:
            n_spikes_per_recording_list = [
                len(
                    WaveformExtractor.load_from_folder(i)
                    .sorting._sorting_segments[0]
                    .spike_indexes
                )
                for i in tqdm(waveform_extractor_paths, leave=False)
            ]
        # number of spikes to sample per recording for the epoch
        sample_spikes_per_recording = (
            (
                np.array(n_spikes_per_recording_list)
                / np.sum(n_spikes_per_recording_list)
            )
            * epoch_size
        ).astype(int)

        for epoch in tqdm(range(n_training_epochs), desc="Training Epoch"):
            train_parametric_umap_single_epoch(
                model=self.model,
                waveform_extractor_paths=waveform_extractor_paths,
                best_channels_index_list=best_channels_index_list,
                max_channels_per_template=self.max_channels_per_template,
                sample_chunk_size=sample_chunk_size,
                sample_spikes_per_recording=sample_spikes_per_recording,
                recording_good_units_list=recording_good_units_list,
                min_chunks_to_sample=min_chunks_to_sample,
                template_best_channels_args=template_best_channels_args,
                n_channels=self.n_channels,
                nbefore=self.nbefore,
                nafter=self.nafter,
                n_jobs=n_jobs,
                verbose=self.verbose,
                n_neighbors=self.n_neighbors,
            )
            self.save()

    def load(self):
        """
        load parametric UMAP model
        """
        self.model = load_ParametricUMAP(self.model_save_path)
        return

    def save(self):
        # save full model
        self.model.save(self.model_save_path)
        # save encoder model
        self.model.encoder.save(self.model_save_path / "encoder")
        # save tflite encoder
        create_tflite_model_from_keras_model(
            keras_filepath=self.model_save_path / "encoder",
            tflite_filepath=self.model_save_path / "model.tflite",
        )

    def run_for_extractor(
        self,
        waveform_extractor_path,
        projection_filename="umap_projection",
        best_channels_index=None,
        projection_memmap_file_path=None,
        n_jobs=1,
        chunk_size=None,
        total_memory="500M",
        chunk_memory=None,
        save_extension="",
        template_best_channels_args={"peak_sign": "neg"},
    ):
        # load waveform extractor
        we = WaveformExtractor.load_from_folder(waveform_extractor_path)

        # get recording and sorter
        sorting = we.sorting
        recording = we.recording

        # TODO is this needed?
        assert recording.get_num_segments() == 1

        # determine number of jobs
        n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)

        # get chunk size for iterating over waveforms (or spikes)
        chunk_size = ensure_chunk_size(
            recording,
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            n_jobs=n_jobs,
        )

        # get all spikes
        spike_times = sorting._sorting_segments[0].spike_indexes
        spike_labels = sorting._sorting_segments[0].spike_labels

        # get best channel index if it does not exist
        if best_channels_index is None:
            best_channels_index = get_template_best_channels(
                we, self.max_channels_per_template, **template_best_channels_args
            )

        # prep projection memmap
        if projection_memmap_file_path is None:
            projection_memmap_file_path = Path(
                waveform_extractor_path
            ) / "{}{}.npy".format(projection_filename, save_extension)
        else:
            projection_memmap_file_path = Path(projection_memmap_file_path)

        # save spike times and labels
        np.save(
            projection_memmap_file_path.parent
            / "{}_label{}.npy".format(projection_filename, save_extension),
            spike_labels,
        )
        np.save(
            projection_memmap_file_path.parent
            / "{}_frame{}.npy".format(projection_filename, save_extension),
            spike_times,
        )

        # break recording into chunks
        recording_chunks = devide_recording_into_chunks(recording, chunk_size)
        spike_time_bins = [i[1] for i in recording_chunks]
        spike_bin_index = np.digitize(spike_times, spike_time_bins)

        # create a memmap object to write to
        projection_memmap_shape = (spike_times.size, self.n_components)
        all_umap = np.lib.format.open_memmap(
            projection_memmap_file_path,
            mode="w+",
            dtype="float32",
            shape=projection_memmap_shape,
        )

        with tf.device("/CPU:0"):
            # iterative over chunks
            ProgressParallel(
                n_jobs=n_jobs,
                desc="Project spikes",
                leave=False,
                total=len(recording_chunks),
                # prefer="processes"
                # prefer="threads"
            )(
                delayed(project_spike_chunk)(
                    waveform_extractor_path=waveform_extractor_path,
                    spike_times_chunk=spike_times[spike_bin_index == chunk],
                    spike_labels_chunk=spike_labels[spike_bin_index == chunk],
                    spike_indices_chunk=np.where(spike_bin_index == chunk)[0],
                    projection_memmap_file_path=projection_memmap_file_path,
                    projection_memmap_shape=projection_memmap_shape,
                    encoder_filepath_tflite=self.model_save_path / "model.tflite",
                    max_channels_per_template=self.max_channels_per_template,
                    best_channels_index=best_channels_index,
                    nafter=self.nafter,
                    nbefore=self.nbefore,
                )
                for chunk in range(1, len(recording_chunks) + 1)
            )


def sample_spikes(
    waveform_extractor_path,
    nbefore=None,
    nafter=None,
    best_channels_index=None,
    n_spikes_to_sample=5000,
    max_channels_per_template=8,
    n_jobs=1,
    min_chunks_to_sample=32,
    good_unit_labels=None,
    chunk_size=30000,
    total_memory=None,
    chunk_memory=None,
    template_best_channels_args={"peak_sign": "neg"},
    verbose=False,
):
    """
    [summary]

    TODO allow samples across all channels

    Parameters
    ----------
    waveform_extractor_path : (pathlib2.Path or str)
        Path to the waveform extractor
    best_channels_index : dict, optional
        dictionary of best channels, by default None
    n_spikes_to_sample : int, optional
        number of spikes to sample from recording, by default 5000
    max_channels_per_template : int, optional
        number of channels to sample per recording, by default 8
    n_jobs : int, optional
        [description], by default 1
    min_chunks_to_sample : int, optional
        minimum number of chunks to sample from for random sample of spikes
        so that we don't sample every spike from the same few time bins, by default 32
    good_unit_labels : [type], optional
        [description], by default None
    chunk_size : [type], optional
        [description], by default None
    total_memory : [type], optional
        [description], by default None
    chunk_memory : [type], optional
        [description], by default None
    template_best_channels_args : dict, optional
        [description], by default {"peak_sign": "neg"}
    verbose : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    if verbose:
        start_time = datetime.datetime.now()
        print("Sampling spikes for each waveform extractor")
        print("\tLoading waveform extractor: {}".format(start_time))

    # load waveform extractor
    we = WaveformExtractor.load_from_folder(waveform_extractor_path)

    # get recording and sorter
    sorting = we.sorting
    recording = we.recording

    assert recording.get_num_segments() == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)

    # get chunk size for iterating over waveforms (or spikes)
    chunk_size = ensure_chunk_size(
        recording,
        total_memory=total_memory,
        chunk_size=chunk_size,
        chunk_memory=chunk_memory,
        n_jobs=n_jobs,
    )

    # get num samples
    recording_samples = recording.get_num_samples(segment_index=0)

    # get all spikes
    spike_times = sorting._sorting_segments[0].spike_indexes
    spike_labels = sorting._sorting_segments[0].spike_labels

    if verbose:
        print("\tSampling spikes: {}".format(datetime.datetime.now() - start_time))

    spike_times_chunks, spike_labels_chunks = sample_spikes_in_bins(
        recording,
        spike_times,
        spike_labels,
        chunk_size,
        min_chunks_to_sample,
        n_spikes_to_sample,
    )

    if verbose:
        print("\tPrepping spikes: {}".format(datetime.datetime.now() - start_time))

    # subset only good units, if a set of good units is provided
    if good_unit_labels is not None:
        good_unit_masks = [
            np.array([i in good_unit_labels for i in spike_labels_chunk])
            for spike_labels_chunk in spike_labels_chunks
        ]
        spike_times_chunks = [
            spike_times_chunk[good_unit_mask]
            for spike_times_chunk, good_unit_mask in zip(
                spike_times_chunks, good_unit_masks
            )
        ]
        spike_labels_chunks = [
            spike_labels_chunk[good_unit_mask]
            for spike_labels_chunk, good_unit_mask in zip(
                spike_labels_chunks, good_unit_masks
            )
        ]

    # ensure that the full spike template can be grabbed
    if nafter is None:
        nafter = we.nafter
    if nbefore is None:
        nbefore = we.nbefore

    spike_masks = [
        np.array(
            (spike_times_chunk > nbefore)
            & (spike_times_chunk < (recording_samples - nafter))
        )
        for spike_times_chunk in spike_times_chunks
    ]
    spike_times_chunks = [
        spike_times_chunk[spike_mask]
        for spike_times_chunk, spike_mask in zip(spike_times_chunks, spike_masks)
    ]
    spike_labels_chunks = [
        spike_labels_chunk[spike_mask]
        for spike_labels_chunk, spike_mask in zip(spike_labels_chunks, spike_masks)
    ]

    # get best channel index if it does not exist
    # TODO this should be stored in the waveformextractor object so it
    #    doesn't have to be re-computed
    if best_channels_index is None:
        max_channels_per_template = min(
            max_channels_per_template, we.recording.get_num_channels()
        )
        best_channels_index = get_template_best_channels(
            we, max_channels_per_template, **template_best_channels_args
        )

    if verbose:
        print("\tGrabbing chunks: {}".format(datetime.datetime.now() - start_time))
    # grab samples

    spike_waveforms = ProgressParallel(
        n_jobs=n_jobs,
        desc="batch sample spikes",
        leave=False,
        total=len(spike_times_chunks),
    )(
        delayed(grab_spike_chunk)(
            spike_times_chunk,
            spike_labels_chunk,
            nbefore,
            nafter,
            max_channels_per_template,
            waveform_extractor_path,
            best_channels_index,
        )
        for spike_times_chunk, spike_labels_chunk in zip(
            spike_times_chunks, spike_labels_chunks
        )
    )

    # seperate labels from waveforms
    spike_waveforms = np.vstack(spike_waveforms)
    spike_labels = np.concatenate(spike_labels_chunks)
    spike_times = np.concatenate(spike_times_chunks)
    channel_index = [best_channels_index[i] for i in spike_labels]

    if verbose:
        print("\tCompleted: {}".format(datetime.datetime.now() - start_time))

    return spike_waveforms, spike_labels, spike_times, channel_index


def grab_spike_chunk(
    spike_times,
    spike_labels,
    nbefore,
    nafter,
    max_channels_per_template,
    waveform_extractor_path,
    best_channels_index,
):
    """
    Grabs spikes from a chunk of the recording

    Returns
    -------
    unit_spike_waveforms [type]
        Waveform for each unit
    spike_labels: [type]
        Label for each unit
    """
    # load the extractor
    we = WaveformExtractor.load_from_folder(waveform_extractor_path)
    recording = we.recording

    shape = (spike_times.size, nbefore + nafter, max_channels_per_template)
    unit_spike_waveforms = np.zeros(dtype="float32", shape=shape)

    t0 = spike_times[0]
    t1 = spike_times[-1]

    trace = recording.get_traces(start_frame=t0 - nbefore, end_frame=t1 + nafter)
    for spike_i, (spike_time, spike_label) in enumerate(
        zip(
            tqdm(spike_times, desc="extract batch spikes", leave=False),
            spike_labels,
        )
    ):
        unit_spike_waveforms[spike_i] = trace[
            spike_time - t0 : spike_time - t0 + nbefore + nafter,
            best_channels_index[spike_label][:max_channels_per_template],
        ]

    return unit_spike_waveforms


def sample_spikes_in_bins(
    recording,
    spike_times,
    spike_labels,
    chunk_size,
    min_chunks_to_sample,
    n_spikes_to_sample,
):
    """
    Split spikes into chunks
    """
    # break recording into chunks
    recording_chunks = devide_recording_into_chunks(recording, chunk_size)

    spike_time_bins = [i[1] for i in recording_chunks]
    spike_bin_index = np.digitize(spike_times, spike_time_bins)
    # number of spikes per bin
    spikes_in_bins = np.bincount(spike_bin_index)

    # randomize order of sampling for time bins
    randomized_bins = np.random.permutation(np.arange(len(recording_chunks) - 1))
    # select from the random chunks, making sure to select at least min_chunks_to_sample chunks
    n_chunks_to_sample = np.max(
        [
            np.where(
                np.cumsum(spikes_in_bins[1:][randomized_bins]) > n_spikes_to_sample
            )[0][0],
            min_chunks_to_sample,
        ]
    )
    bins_to_sample = randomized_bins[:n_chunks_to_sample]
    # grab a roughly equal number of spikes to grab per bin, to amount to
    spikes_to_sample_per_bin = (
        spikes_in_bins[bins_to_sample]
        / np.sum(spikes_in_bins[bins_to_sample])
        * n_spikes_to_sample
    ).astype(int)
    # ensure no bins are empty
    bins_to_sample = bins_to_sample[spikes_to_sample_per_bin > 0]
    spikes_to_sample_per_bin = spikes_to_sample_per_bin[spikes_to_sample_per_bin > 0]

    # randomly sample from each bin
    spike_index_to_sample = [
        np.sort(
            np.random.permutation(np.where(spike_bin_index == bin_to_sample)[0])[
                :n_spikes_to_sample
            ]
        )
        for bin_to_sample, n_spikes_to_sample in tqdm(
            (zip(bins_to_sample, spikes_to_sample_per_bin)),
            total=len(bins_to_sample),
            leave=False,
            desc="sampling spikes from each bin",
        )
    ]
    spike_times_chunks = [spike_times[i] for i in spike_index_to_sample]
    spike_labels_chunks = [spike_labels[i] for i in spike_index_to_sample]
    return spike_times_chunks, spike_labels_chunks


class ProgressParallel(Parallel):
    """
    Modified version of joblib.parallel that allows for tqdm visualization of progress
    From: https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    Parameters
    ----------
    Parallel : joblib.Parallel
    """

    def __init__(
        self, use_tqdm=True, desc=None, total=None, leave=True, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._leave = leave
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            desc=self._desc,
            total=self._total,
            leave=self._leave,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def create_tflite_model_from_keras_model(keras_filepath, tflite_filepath):
    converter = tf.lite.TFLiteConverter.from_saved_model(keras_filepath.as_posix())
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    with open(tflite_filepath, "wb") as f:
        f.write(tflite_model)


def load_tflite_model(tflite_filepath):
    interpreter = tf.lite.Interpreter(tflite_filepath.as_posix(), num_threads=1)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    return interpreter, input_details, output_details, input_shape


def inference_tflite(interpreter, input_details, output_details, input_data):
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def project_spike_chunk(
    waveform_extractor_path,
    spike_times_chunk,
    spike_labels_chunk,
    spike_indices_chunk,
    projection_memmap_file_path,
    projection_memmap_shape,
    encoder_filepath_tflite,
    max_channels_per_template,
    best_channels_index,
    nafter,
    nbefore,
):
    """Projects all spikes in a chunk of a recording"""
    # with tf.device('/CPU:0'):
    # load memmap
    all_umap = np.lib.format.open_memmap(
        projection_memmap_file_path,
        mode="r+",
        dtype="float32",
        shape=projection_memmap_shape,
    )

    # load encoder
    interpreter, input_details, output_details, input_shape = load_tflite_model(
        encoder_filepath_tflite
    )

    # load the extractor
    we = WaveformExtractor.load_from_folder(waveform_extractor_path)
    recording = we.recording

    # get times to grab from recording
    t0 = spike_times_chunk[0]
    t1 = spike_times_chunk[-1]
    start_frame = np.max([0, t0 - nbefore])
    end_frame = np.min([recording.get_num_frames(segment_index=0), t1 + nafter])

    # grab recording
    trace = recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # if a spike exists near the borders, pad with zeros so we can grab the whole spike
    if (t0 - nbefore) < 0:
        trace = np.vstack([np.zeros((np.abs(t0 - nbefore), trace.shape[1])), trace])
    if (t1 + nafter) > recording.get_num_frames(segment_index=0):
        trace = np.vstack(
            [
                trace,
                np.zeros(
                    (
                        np.abs(t1 + nafter - recording.get_num_frames(segment_index=0)),
                        trace.shape[1],
                    )
                ),
            ]
        )
    z = np.zeros((len(spike_times_chunk), all_umap.shape[1]))
    for si, (spike_time, spike_label, spike_index) in enumerate(
        zip(
            spike_times_chunk,
            spike_labels_chunk,
            tqdm(spike_indices_chunk, leave=False, total=len(spike_indices_chunk) - 1),
        )
    ):
        # array([  1, 105,  64,   1],
        spike = np.zeros((1, nafter + nbefore, trace.shape[1], 1), dtype="float32")
        spike_subset = trace[
            spike_time - t0 : spike_time - t0 + nbefore + nafter,
            best_channels_index[spike_label][:max_channels_per_template],
        ]
        spike[
            :, :, best_channels_index[spike_label][:max_channels_per_template], :
        ] = np.expand_dims(spike_subset, [0, -1])
        # z = loaded_encoder.predict(spike)
        z[si] = inference_tflite(interpreter, input_details, output_details, spike)

    all_umap[spike_indices_chunk] = z


def make_encoder_network(
    n_components, n_dense, n_samples, n_channels, batch_size, verbose=False
):
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=n_dense + n_dense * n_samples),
            SparseInputLayer(
                n_dense=n_dense,
                n_samples=n_samples,
                n_channels=n_channels,
                batch_size=batch_size,
                name="sparselayer",
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(1, n_samples),
                strides=(1, 1),
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation="relu"),
            tf.keras.layers.Dense(units=512, activation="relu"),
            tf.keras.layers.Dense(units=512, activation="relu"),
            tf.keras.layers.Dense(units=n_components),
        ]
    )
    if verbose:
        encoder.summary()
    return encoder


def quick_plot_projection(z, category):
    palette = sns.color_palette("tab20", len(np.unique(category)))
    palette_dict = {i: palette[ii] for ii, i in enumerate(np.unique(category))}
    color_list = [palette_dict[i] for i in category]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(z[:, 0], z[:, 1], s=1, alpha=0.1, c=color_list)
    plt.show()


def train_parametric_umap_single_epoch(
    model,
    waveform_extractor_paths,
    best_channels_index_list,
    max_channels_per_template,
    sample_spikes_per_recording,
    recording_good_units_list,
    sample_chunk_size,
    n_channels,
    min_chunks_to_sample,
    n_neighbors,
    nbefore=None,
    nafter=None,
    template_best_channels_args={"peak_sign": "neg"},
    verbose=True,
    n_jobs=1,
):
    all_spike_waveforms = []
    all_spike_labels = []
    all_channel_index = []
    all_spike_times = []
    with tqdm(
        total=np.sum(sample_spikes_per_recording),
        desc="sampling spikes from each recording",
        leave=False,
    ) as t:
        for (
            waveform_extractor_path,
            best_channels_index,
            n_spikes_to_sample,
            good_unit_labels,
        ) in zip(
            waveform_extractor_paths,
            best_channels_index_list,
            sample_spikes_per_recording,
            recording_good_units_list,
        ):
            spike_waveforms, spike_labels, spike_times, channel_index = sample_spikes(
                waveform_extractor_path,
                best_channels_index=best_channels_index,
                n_spikes_to_sample=n_spikes_to_sample,
                max_channels_per_template=max_channels_per_template,
                chunk_size=sample_chunk_size,
                good_unit_labels=good_unit_labels,
                n_jobs=n_jobs,
                min_chunks_to_sample=min_chunks_to_sample,
                nbefore=nbefore,
                nafter=nafter,
                template_best_channels_args=template_best_channels_args,
                verbose=verbose,
            )
            all_spike_waveforms.append(spike_waveforms)
            all_channel_index.append(channel_index)
            all_spike_labels.append(spike_labels)
            t.update(n_spikes_to_sample)
    # n_waveforms_per_recording = [len(i) for i in all_spike_waveforms]
    all_spike_waveforms = np.vstack(all_spike_waveforms)
    all_spike_labels = np.concatenate(all_spike_labels)
    all_spike_times = np.concatenate(all_spike_labels)

    # make sample waveforms flat
    all_spike_waveforms = all_spike_waveforms.reshape(
        (
            len(all_spike_waveforms),
            np.product(all_spike_waveforms.shape[1:]),
        )
    )
    all_channel_index = np.vstack(all_channel_index)
    all_spike_waveforms = np.hstack([all_channel_index, all_spike_waveforms])

    n_samples = nbefore + nafter
    n_dense = max_channels_per_template
    knn_indices, knn_dists, _ = nearest_neighbors(
        all_spike_waveforms,
        metric=sparse_euclidean,
        metric_kwds={"n_samples": n_samples, "n_dense": n_dense},
        random_state=check_random_state(None),
        angular=False,
        n_neighbors=n_neighbors + 1,
        verbose=verbose,
    )

    sparse_nn_graph = knn_to_sparse_distance(knn_indices, knn_dists, n_neighbors + 1)

    model.fit(all_spike_waveforms, precomputed_distances=sparse_nn_graph)
    z = model.transform(all_spike_waveforms)

    if verbose:
        # plot the output
        quick_plot_projection(z, all_spike_labels)


def make_waveform_array_dense(
    all_spike_waveforms,
    max_channels_per_template,
    n_waveforms_per_recording,
    all_spike_labels,
    best_channels_index_list,
    n_channels,
):
    """Convert sparse representation of waveforms from best channels only, to dense representation
    with non-best channels as zeros
    """
    all_spike_waveforms_dense = np.zeros(
        (all_spike_waveforms.shape[0], all_spike_waveforms.shape[1], n_channels),
        dtype="float32",
    )
    n = 0
    for n_waveforms, best_channel_index in tqdm(
        zip(n_waveforms_per_recording, best_channels_index_list),
        total=len(n_waveforms_per_recording),
        leave=False,
        desc="expanding sparse waveforms",
    ):
        for wi, (label, waveform) in enumerate(
            zip(
                all_spike_labels[n : n_waveforms + n],
                all_spike_waveforms[n : n_waveforms + n],
            )
        ):
            all_spike_waveforms_dense[
                wi + n, :, best_channel_index[label][:max_channels_per_template]
            ] = waveform.T
        n += n_waveforms
    return all_spike_waveforms_dense


def project_spikes_extractor(
    waveform_extractor_path,
    encoder_filepath_tflite,
    n_components,
    best_channels_index=None,
    projection_memmap_file_path=None,
    max_channels_per_template=8,
    n_jobs=1,
    chunk_size=30000,
    total_memory=None,
    chunk_memory=None,
    template_best_channels_args={"peak_sign": "neg"},
    verbose=False,
):

    with tf.device("/CPU:0"):
        # load waveform extractor
        we = WaveformExtractor.load_from_folder(waveform_extractor_path)

        # get recording and sorter
        sorting = we.sorting
        recording = we.recording

        # TODO is this needed?
        assert recording.get_num_segments() == 1

        # determine number of jobs
        n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)

        # get chunk size for iterating over waveforms (or spikes)
        chunk_size = ensure_chunk_size(
            recording,
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            n_jobs=n_jobs,
        )

        nafter = we.nafter
        nbefore = we.nbefore

        # get all spikes
        spike_times = sorting._sorting_segments[0].spike_indexes
        spike_labels = sorting._sorting_segments[0].spike_labels

        # get best channel index if it does not exist
        if best_channels_index is None:
            max_channels_per_template = min(
                max_channels_per_template, we.recording.get_num_channels()
            )
            best_channels_index = get_template_best_channels(
                we, max_channels_per_template, **template_best_channels_args
            )

        if projection_memmap_file_path is None:
            projection_memmap_file_path = (
                Path(waveform_extractor_path) / "umap_projection.npy"
            )
        else:
            projection_memmap_file_path = Path(projection_memmap_file_path)

        # save spike times and labels
        np.save(
            projection_memmap_file_path.parent / "label_projection.npy", spike_labels
        )
        np.save(
            projection_memmap_file_path.parent / "frame_projection.npy", spike_times
        )

        # create a memmap object to write to
        projection_memmap_shape = (spike_times.size, n_components)
        all_umap = np.lib.format.open_memmap(
            projection_memmap_file_path,
            mode="w+",
            dtype="float32",
            shape=projection_memmap_shape,
        )

        # break recording into chunks
        recording_chunks = devide_recording_into_chunks(recording, chunk_size)
        spike_time_bins = [i[1] for i in recording_chunks]
        spike_bin_index = np.digitize(spike_times, spike_time_bins)

        # iterative over chunks

        ProgressParallel(
            n_jobs=n_jobs,
            desc="Project spikes",
            leave=False,
            total=len(recording_chunks),
            # prefer="processes"
            # prefer="threads"
        )(
            delayed(project_spike_chunk)(
                waveform_extractor_path=waveform_extractor_path,
                spike_times_chunk=spike_times[spike_bin_index == chunk],
                spike_labels_chunk=spike_labels[spike_bin_index == chunk],
                spike_indices_chunk=np.where(spike_bin_index == chunk)[0],
                projection_memmap_file_path=projection_memmap_file_path,
                projection_memmap_shape=projection_memmap_shape,
                encoder_filepath_tflite=encoder_filepath_tflite,
                max_channels_per_template=max_channels_per_template,
                best_channels_index=best_channels_index,
                nafter=nafter,
                nbefore=nbefore,
            )
            for chunk in range(1, len(recording_chunks) + 1)
        )


class SparseInputLayer(tf.keras.layers.Layer):
    """Takes in an input data comprised of data & columns, and reshapes it into a dense vector"""

    def __init__(self, n_dense, n_samples, n_channels, batch_size, **kwargs):
        super(SparseInputLayer, self).__init__(**kwargs)
        self.n_dense = n_dense
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.batch_size = batch_size

    def call(self, inputs):
        batch_size = self.batch_size
        # batch_size = tf.shape(inputs)[0]
        verbose = False
        if verbose:
            print(batch_size)
        # split batch into index and batch
        idx, x_batch = tf.split(
            inputs, (self.n_dense, self.n_dense * self.n_samples), axis=1
        )
        # print(idx, x_batch, self.n_dense*self.n_samples, self.n_channels, self.batch_size)

        # unflatten batch
        x_batch = tf.reshape(x_batch, (batch_size, self.n_dense, self.n_samples))
        if verbose:
            print(x_batch)
        # cast index to integer for indexing
        idx = tf.cast(idx, tf.int32)
        if verbose:
            print("idx", idx)
        ### reshape in to 2D (batch_size*channels, frames) and index
        x_reshape = tf.reshape(x_batch, (batch_size * self.n_dense, self.n_samples))
        if verbose:
            print("x_reshape", x_reshape)
        # add to index
        idx_added = idx + tf.expand_dims(tf.range(batch_size) * self.n_channels, 1)
        if verbose:
            print("idx_added", idx_added)
        # reshape to a single list
        idx_reshape = tf.squeeze(tf.reshape(idx_added, (batch_size * self.n_dense, 1)))
        if verbose:
            print("idx_reshape", idx_reshape)
        # expand to dense
        dense_data = tf.IndexedSlices(
            x_reshape,
            idx_reshape,
            dense_shape=(batch_size * self.n_channels, self.n_samples),
        )
        if verbose:
            print("dense_data", dense_data)
        # reshape data to correct value
        dense_data = tf.reshape(
            dense_data, (batch_size, self.n_channels, self.n_samples)
        )
        dense_data = tf.expand_dims(dense_data, -1)
        if verbose:
            print(dense_data)
        return dense_data


@numba.jit(fastmath=True, cache=True)
def sparse_euclidean(x, y, n_samples, n_dense):
    """Euclidean distance metric over sparse vectors, where first n_dense
    elements are indices, and n_samples is the length of the second dimension
    """
    # break out sparse into columns and data
    x_best = x[:n_dense]
    x = x[n_dense:]
    y_best = y[:n_dense]
    y = y[n_dense:]
    result = 0.0

    xi = 0
    for xb in x_best:
        calc = False
        yi = 0
        for yb in y_best:
            if xb == yb:
                calc = True
                # calculate euclidean
                for i in range(n_samples):
                    result += (x[xi * n_samples + i] - y[yi * n_samples + i]) ** 2

            yi += 1
        if calc == False:
            # add x squared
            for i in range(n_samples):
                result += x[xi * n_samples + i] ** 2
        xi += 1
    yi = 0
    for yb in y_best:
        calc = False
        for xb in x_best:
            if xb == yb:
                calc = True
        if calc == False:
            # add y squared
            for i in range(n_samples):
                result += y[yi * n_samples + i] ** 2

        yi += 1
    return np.sqrt(result)


def knn_to_sparse_distance(knn_indices, knn_dists, n_neighbors):
    csr_rows1 = np.repeat(np.arange(len(knn_indices)), n_neighbors - 1)
    csr_cols1 = knn_indices[:, 1:].flatten()
    csr_rows = np.concatenate([csr_rows1, csr_cols1, np.arange(len(knn_indices))])
    csr_cols = np.concatenate([csr_cols1, csr_rows1, np.arange(len(knn_indices))])

    csr_vals = knn_dists[:, 1:].flatten()
    csr_vals = np.concatenate([csr_vals, csr_vals, np.zeros(len(knn_indices))])
    sparse_nn_graph = csr_matrix(
        (csr_vals, (csr_rows, csr_cols)), shape=(len(knn_indices), len(knn_indices))
    )
    return sparse_nn_graph