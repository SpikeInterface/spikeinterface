"""Sorting components: peak detection."""

from __future__ import annotations
from typing import Tuple, List, Optional

import numpy as np


from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.node_pipeline import (
    PeakDetector,
    WaveformsNode,
    ExtractSparseWaveforms,
    base_peak_dtype,
)

expanded_base_peak_dtype = np.dtype(base_peak_dtype + [("iteration", "int8")])


class IterativePeakDetector(PeakDetector):
    """
    A class that iteratively detects peaks in the recording by applying a peak detector, waveform extraction,
    and waveform denoising node. The algorithm runs for a specified number of iterations or until no peaks are found.
    """

    def __init__(
        self,
        recording: BaseRecording,
        peak_detector_node: PeakDetector,
        waveform_extraction_node: WaveformsNode,
        waveform_denoising_node,
        num_iterations: int = 2,
        return_output: bool = True,
        tresholds: Optional[List[float]] = None,
    ):
        """
        Initialize the iterative peak detector.

        Parameters
        ----------
        recording : BaseRecording
            The recording to process
        peak_detector_node : PeakDetector
            The peak detector node to use
        waveform_extraction_node : WaveformsNode
            The waveform extraction node to use
        waveform_denoising_node
            The waveform denoising node to use
        num_iterations : int, default: 2
            The number of iterations to run the algorithm
        return_output : bool, default: True
            Whether to return the output of the algorithm
        """
        PeakDetector.__init__(self, recording, return_output=return_output)
        self.peak_detector_node = peak_detector_node
        self.waveform_extraction_node = waveform_extraction_node
        self.waveform_denoising_node = waveform_denoising_node
        self.num_iterations = num_iterations
        self.tresholds = tresholds

    def get_trace_margin(self) -> int:
        """
        Calculate the maximum trace margin from the internal pipeline.
        Using the strategy as use by the Node pipeline


        Returns
        -------
        int
            The maximum trace margin.
        """
        internal_pipeline = (self.peak_detector_node, self.waveform_extraction_node, self.waveform_denoising_node)
        pipeline_margin = (node.get_trace_margin() for node in internal_pipeline if hasattr(node, "get_trace_margin"))
        return max(pipeline_margin)

    def compute(self, traces_chunk, start_frame, end_frame, segment_index, max_margin) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the iterative peak detection, waveform extraction, and denoising.

        Parameters
        ----------
        traces_chunk : array-like
            The chunk of traces to process.
        start_frame : int
            The starting frame for the chunk.
        end_frame : int
            The ending frame for the chunk.
        segment_index : int
            The segment index.
        max_margin : int
            The maximum margin for the traces.

        Returns
        -------
        tuple of ndarray
            A tuple containing a single ndarray with the detected peaks.
        """

        traces_chunk = np.array(traces_chunk, copy=True, dtype="float32")
        local_peaks_list = []
        all_waveforms = []

        for iteration in range(self.num_iterations):
            # Hack because of lack of either attribute or named references
            # I welcome suggestions on how to improve this but I think it is an architectural issue
            if self.tresholds is not None:
                old_detect_treshold = self.peak_detector_node.detect_threshold
                old_abs_thresholds = self.peak_detector_node.abs_thresholds
                self.peak_detector_node.detect_threshold = self.tresholds[iteration]
                self.peak_detector_node.abs_tresholds = (
                    old_abs_thresholds * self.tresholds[iteration] / old_detect_treshold
                )

            (local_peaks,) = self.peak_detector_node.compute(
                traces=traces_chunk,
                start_frame=start_frame,
                end_frame=end_frame,
                segment_index=segment_index,
                max_margin=max_margin,
            )

            local_peaks = self.add_iteration_to_peaks_dtype(local_peaks=local_peaks, iteration=iteration)
            local_peaks_list.append(local_peaks)

            # End algorith if no peak is found
            if local_peaks.size == 0:
                break

            waveforms = self.waveform_extraction_node.compute(traces=traces_chunk, peaks=local_peaks)
            denoised_waveforms = self.waveform_denoising_node.compute(
                traces=traces_chunk, peaks=local_peaks, waveforms=waveforms
            )

            self.substract_waveforms_from_traces(
                local_peaks=local_peaks,
                traces_chunk=traces_chunk,
                waveforms=denoised_waveforms,
            )

            all_waveforms.append(waveforms)
        all_local_peaks = np.concatenate(local_peaks_list, axis=0)
        all_waveforms = np.concatenate(all_waveforms, axis=0) if len(all_waveforms) != 0 else np.empty((0, 0, 0))

        # Sort as iterative method implies peaks might not be discovered ordered in time
        sorting_indices = np.argsort(all_local_peaks["sample_index"])
        all_local_peaks = all_local_peaks[sorting_indices]
        all_waveforms = all_waveforms[sorting_indices]

        return (all_local_peaks, all_waveforms)

    def substract_waveforms_from_traces(
        self,
        local_peaks: np.ndarray,
        traces_chunk: np.ndarray,
        waveforms: np.ndarray,
    ):
        """
        Substract inplace the cleaned waveforms from the traces_chunk.

        Parameters
        ----------
        sample_indices : ndarray
            The indices where the waveforms are maximum (peaks["sample_index"]).
        traces_chunk : ndarray
            A chunk of the traces.
        waveforms : ndarray
            The waveforms extracted from the traces.
        """

        nbefore = self.waveform_extraction_node.nbefore
        nafter = self.waveform_extraction_node.nafter
        if isinstance(self.waveform_extraction_node, ExtractSparseWaveforms):
            neighbours_mask = self.waveform_extraction_node.neighbours_mask
        else:
            neighbours_mask = None

        for peak_index, peak in enumerate(local_peaks):
            center_sample = peak["sample_index"]
            first_sample = center_sample - nbefore
            last_sample = center_sample + nafter
            if neighbours_mask is None:
                traces_chunk[first_sample:last_sample, :] -= waveforms[peak_index, :, :]
            else:
                (channels,) = np.nonzero(neighbours_mask[peak["channel_index"]])
                traces_chunk[first_sample:last_sample, channels] -= waveforms[peak_index, :, : len(channels)]

    def add_iteration_to_peaks_dtype(self, local_peaks, iteration) -> np.ndarray:
        """
        Add the iteration number to the peaks dtype.

        Parameters
        ----------
        local_peaks : ndarray
            The array of local peaks.
        iteration : int
            The iteration number.

        Returns
        -------
        ndarray
            An array of local peaks with the iteration number added.
        """
        # Expand dtype to also contain an iteration field
        local_peaks_expanded = np.zeros_like(local_peaks, dtype=expanded_base_peak_dtype)
        fields_in_base_type = np.dtype(base_peak_dtype).names
        for field in fields_in_base_type:
            local_peaks_expanded[field] = local_peaks[field]
        local_peaks_expanded["iteration"] = iteration

        return local_peaks_expanded
