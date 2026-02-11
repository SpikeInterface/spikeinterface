from __future__ import annotations
from typing import Optional
import numpy as np

from spikeinterface.core import concatenate_recordings, BaseRecording, BaseRecordingSegment

from deepinterpolation.generator_collection import SequentialGenerator


class SpikeInterfaceRecordingGenerator(SequentialGenerator):
    """
    This generator is used when dealing with a SpikeInterface recording.
    The desired shape controls the reshaping of the input data before convolutions.
    """

    def __init__(
        self,
        recordings: BaseRecording | list[BaseRecording],
        pre_frame: int = 30,
        post_frame: int = 30,
        pre_post_omission: int = 1,
        desired_shape: tuple = (192, 2),
        batch_size: int = 100,
        steps_per_epoch: int = 10,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        total_samples: int = -1,
    ):
        if not isinstance(recordings, list):
            recordings = [recordings]
        self.recordings = recordings
        if len(recordings) > 1:
            assert (
                r.get_num_channels() == recordings[0].get_num_channels() for r in recordings[1:]
            ), "All recordings must have the same number of channels"

        total_num_samples = sum([r.get_total_samples() for r in recordings])
        # In case of multiple recordings and/or multiple segments, we calculate the frame periods to be excluded (borders)
        exclude_intervals = []
        pre_extended = pre_frame + pre_post_omission
        post_extended = post_frame + pre_post_omission
        for i, recording in enumerate(recordings):
            total_samples_pre = sum([r.get_total_samples() for r in recordings[:i]])
            for segment_index in range(recording.get_num_segments()):
                # exclude samples at the border of the recordings
                num_samples_segment_pre = sum([recording.get_num_samples(s) for s in np.arange(segment_index)])
                if num_samples_segment_pre > 0:
                    exclude_intervals.append(
                        (
                            total_samples_pre + num_samples_segment_pre - pre_extended - 1,
                            total_samples_pre + num_samples_segment_pre + post_extended,
                        )
                    )
            # exclude samples at the border of the recordings
            if total_samples_pre > 0:
                exclude_intervals.append((total_samples_pre - pre_extended - 1, total_samples_pre + post_extended))

        total_valid_samples = (
            total_num_samples - sum([end - start for start, end in exclude_intervals]) - pre_extended - post_extended
        )
        self.total_samples = int(total_valid_samples) if total_samples == -1 else total_samples
        assert len(desired_shape) == 2, "desired_shape should be 2D"
        assert (
            desired_shape[0] * desired_shape[1] == recording.get_num_channels()
        ), f"The product of desired_shape dimensions should be the number of channels: {recording.get_num_channels()}"
        self.desired_shape = desired_shape

        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else total_num_samples
        assert end_frame > start_frame, "end_frame must be greater than start_frame"

        sequential_generator_params = dict()
        sequential_generator_params["steps_per_epoch"] = steps_per_epoch
        sequential_generator_params["pre_frame"] = pre_frame
        sequential_generator_params["post_frame"] = post_frame
        sequential_generator_params["batch_size"] = batch_size
        sequential_generator_params["start_frame"] = start_frame
        sequential_generator_params["end_frame"] = end_frame
        sequential_generator_params["total_samples"] = self.total_samples
        sequential_generator_params["pre_post_omission"] = pre_post_omission

        super().__init__(sequential_generator_params)

        self._update_end_frame(total_num_samples)

        # self.list_samples will exclude the border intervals on the concat recording
        self.recording_concat = concatenate_recordings(recordings)
        self.exclude_intervals = exclude_intervals
        self._calculate_list_samples(total_num_samples, exclude_intervals=exclude_intervals)

        self._kwargs = dict(
            recordings=recordings,
            pre_frame=pre_frame,
            post_frame=post_frame,
            pre_post_omission=pre_post_omission,
            desired_shape=desired_shape,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    # this is overridden to exclude samples from borders
    def _calculate_list_samples(self, total_frame_per_movie, exclude_intervals=[]):
        # We first cut if start and end frames are too close to the edges.
        self.start_sample = np.max([self.pre_frame + self.pre_post_omission, self.start_frame])
        self.end_sample = np.min(
            [
                self.end_frame,
                total_frame_per_movie - 1 - self.post_frame - self.pre_post_omission,
            ]
        )

        if (self.end_sample - self.start_sample + 1) < self.batch_size:
            raise Exception(
                "Not enough frames to construct one "
                + str(self.batch_size)
                + " frame(s) batch between "
                + str(self.start_sample)
                + " and "
                + str(self.end_sample)
                + " frame number."
            )

        # +1 to make sure end_samples is included
        list_samples_all = np.arange(self.start_sample, self.end_sample + 1)

        if len(exclude_intervals) > 0:
            for start, end in exclude_intervals:
                list_samples_all = list_samples_all[(list_samples_all <= start) | (list_samples_all >= end)]
            self.list_samples = list_samples_all
        else:
            self.list_samples = list_samples_all

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if self.total_samples > 0 and self.total_samples < len(self.list_samples):
            self.list_samples = self.list_samples[0 : self.total_samples]

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        shuffle_indexes = self.generate_batch_indexes(index)  # this not used during inferenc

        input_full = np.zeros(
            [len(shuffle_indexes), self.desired_shape[0], self.desired_shape[1], self.pre_frame + self.post_frame],
            dtype="float32",
        )
        output_full = np.zeros([len(shuffle_indexes), self.desired_shape[0], self.desired_shape[1], 1], dtype="float32")

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # We reorganize to follow true geometry of probe for convolution
        input_full = np.zeros(
            [1, self.desired_shape[0], self.desired_shape[1], self.pre_frame + self.post_frame], dtype="float32"
        )
        output_full = np.zeros([1, self.desired_shape[0], self.desired_shape[1], 1], dtype="float32")

        start_frame = index_frame - self.pre_frame - self.pre_post_omission
        end_frame = index_frame + self.post_frame + self.pre_post_omission + 1
        full_traces = self.recording_concat.get_traces(start_frame=start_frame, end_frame=end_frame).astype("float32")

        if full_traces.shape[0] == 0:
            print(f"Error! {index_frame}-{start_frame}-{end_frame}", flush=True)
        output_frame_index = self.pre_frame + self.pre_post_omission
        mask = np.ones(len(full_traces), dtype=bool)
        mask = np.ones(len(full_traces), dtype=bool)
        mask[output_frame_index - 1 : output_frame_index + 2] = False

        data_img_input = full_traces[mask]
        data_img_output = full_traces[output_frame_index][np.newaxis, :]

        # make 3d based on desired shape
        data_input_3d = data_img_input.reshape((-1, self.desired_shape[0], self.desired_shape[1]))
        data_output_3d = data_img_output.reshape((-1, self.desired_shape[0], self.desired_shape[1]))

        input_full[0] = data_input_3d.swapaxes(0, 1).swapaxes(1, 2)
        output_full[0] = data_output_3d.swapaxes(0, 1).swapaxes(1, 2)

        return input_full, output_full

    def reshape_output(self, output):
        return output.squeeze().reshape(-1, self.recording.get_num_channels())


class SpikeInterfaceRecordingSegmentGenerator(SequentialGenerator):
    """This generator is used when dealing with a SpikeInterface recording.
    The desired shape controls the reshaping of the input data before convolutions."""

    def __init__(
        self,
        recording_segment: BaseRecordingSegment,
        start_frame: int,
        end_frame: int,
        pre_frame: int = 30,
        post_frame: int = 30,
        pre_post_omission: int = 1,
        desired_shape: tuple = (192, 2),
        batch_size: int = 100,
        steps_per_epoch: int = 10,
    ):
        self.recording_segment = recording_segment
        self.num_channels = int(desired_shape[0] * desired_shape[1])
        assert len(desired_shape) == 2, "desired_shape should be 2D"
        self.desired_shape = desired_shape

        num_segment_samples = recording_segment.get_num_samples()
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else num_segment_samples
        assert end_frame > start_frame, "end_frame must be greater than start_frame"
        self.total_samples = end_frame - start_frame

        sequential_generator_params = dict()
        sequential_generator_params["steps_per_epoch"] = steps_per_epoch
        sequential_generator_params["pre_frame"] = pre_frame
        sequential_generator_params["post_frame"] = post_frame
        sequential_generator_params["batch_size"] = batch_size
        sequential_generator_params["start_frame"] = start_frame
        sequential_generator_params["end_frame"] = end_frame
        sequential_generator_params["total_samples"] = self.total_samples
        sequential_generator_params["pre_post_omission"] = pre_post_omission

        super().__init__(sequential_generator_params)

        self._update_end_frame(num_segment_samples)
        # IMPORTANT: this is used for inference, so we don't want to shuffle
        self.randomize = False
        self._calculate_list_samples(num_segment_samples)

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        shuffle_indexes = self.generate_batch_indexes(index)  # this not used during inferenc

        input_full = np.zeros(
            [len(shuffle_indexes), self.desired_shape[0], self.desired_shape[1], self.pre_frame + self.post_frame],
            dtype="float32",
        )
        output_full = np.zeros([len(shuffle_indexes), self.desired_shape[0], self.desired_shape[1], 1], dtype="float32")

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # We reorganize to follow true geometry of probe for convolution
        input_full = np.zeros(
            [1, self.desired_shape[0], self.desired_shape[1], self.pre_frame + self.post_frame], dtype="float32"
        )
        output_full = np.zeros([1, self.desired_shape[0], self.desired_shape[1], 1], dtype="float32")

        start_frame = index_frame - self.pre_frame - self.pre_post_omission
        end_frame = index_frame + self.post_frame + self.pre_post_omission + 1
        channel_indices = slice(None)
        full_traces = self.recording_segment.get_traces(
            start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices
        ).astype("float32")

        if full_traces.shape[0] == 0:
            print(f"Error! {index_frame}-{start_frame}-{end_frame}", flush=True)
        output_frame_index = self.pre_frame + self.pre_post_omission
        mask = np.ones(len(full_traces), dtype=bool)
        mask = np.ones(len(full_traces), dtype=bool)
        mask[output_frame_index - 1 : output_frame_index + 2] = False

        data_img_input = full_traces[mask]
        data_img_output = full_traces[output_frame_index][np.newaxis, :]

        # make 3d based on desired shape
        data_input_3d = data_img_input.reshape((-1, self.desired_shape[0], self.desired_shape[1]))
        data_output_3d = data_img_output.reshape((-1, self.desired_shape[0], self.desired_shape[1]))

        input_full[0] = data_input_3d.swapaxes(0, 1).swapaxes(1, 2)
        output_full[0] = data_output_3d.swapaxes(0, 1).swapaxes(1, 2)

        return input_full, output_full

    def reshape_output(self, output):
        return output.squeeze().reshape(-1, self.num_channels)
