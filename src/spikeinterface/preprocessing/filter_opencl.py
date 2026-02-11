from __future__ import annotations

import numpy as np
import scipy.signal

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from spikeinterface.core import get_chunk_with_margin

try:
    import pyopencl

    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False


class FilterOpenCLRecording(BasePreprocessor):
    """
    Simple implementation of FilterRecording in OpenCL.

    Only filter_mode="sos" is supported.

    Author : Samuel Garcia
    This kernel is ported from "tridesclous"

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced

    N: order
    filter_mode: "sos" only

    ftypestr: "butter" / "cheby1" / ... all possible of scipy.signal.iirdesign

    margin: margin in second on border to avoid border effect

    """

    def __init__(
        self,
        recording,
        band=[300.0, 6000.0],
        btype="bandpass",
        filter_order=5,
        ftype="butter",
        filter_mode="sos",
        margin_ms=5.0,
    ):
        assert HAVE_PYOPENCL, "You need to install pyopencl (and GPU driver!!)"
        btype_modes = ("bandpass", "lowpass", "highpass", "bandstop")
        assert btype in btype_modes, f"'btype' must be in {btype_modes}"
        assert filter_mode in ("sos",), "'filter_mode' must be 'sos'"

        # coefficient
        sf = recording.get_sampling_frequency()
        if btype in ("bandpass", "bandstop"):
            assert len(band) == 2
            Wn = [e / sf * 2 for e in band]
        else:
            Wn = float(band) / sf * 2
        N = filter_order

        coefficients = scipy.signal.iirfilter(N, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

        BasePreprocessor.__init__(self, recording)

        margin = int(margin_ms * sf / 1000.0)
        num_channels = recording.get_num_channels()
        dtype = recording.get_dtype()
        # DEBUG force float32 at the moement
        dtype = "float32"
        executor = OpenCLFilterExecutor(coefficients, num_channels, dtype, margin)

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterOpenCLRecordingSegment(parent_segment, executor, margin))

        self._kwargs = dict(
            recording=recording,
            band=band,
            btype=btype,
            filter_order=filter_order,
            ftype=ftype,
            filter_mode=filter_mode,
            margin_ms=margin_ms,
        )


class FilterOpenCLRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, executor, margin):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.executor = executor
        self.margin = margin

    def get_traces(self, start_frame, end_frame, channel_indices):
        assert start_frame is not None, "FilterOpenCLRecording only works with fixed chunk_size"
        assert end_frame is not None, "FilterOpenCLRecording only works with fixed chunk_size"

        chunk_size = end_frame - start_frame
        if chunk_size != self.executor.chunk_size:
            self.executor.create_buffers_and_compile(chunk_size)

        #  get with margin and force zeros!!
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment, start_frame, end_frame, channel_indices, self.margin, add_zeros=True
        )

        # DEBUG
        traces_chunk_float32 = traces_chunk.astype("float32")

        filtered_traces = self.executor.process(traces_chunk_float32)

        # DEBUG
        filtered_traces = filtered_traces.astype(traces_chunk.dtype)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]

        return filtered_traces


class OpenCLFilterExecutor:
    """
    Executor function shared across FilterOpenCLRecordingSegment.

    The input/ouput can be  float32 only (int16 will be implemented soon).

    Internally it is computed as float32 with coeff

    """

    def __init__(self, coefficients, num_channels, dtype, margin):
        self.coefficients = np.ascontiguousarray(coefficients, dtype="float32")
        self.num_channels = num_channels
        self.dtype = np.dtype(dtype)
        self.margin = margin

        #  plat = pyopencl.get_platforms()
        #  dev = plat[0].get_devices()
        #  ctx = pyopencl.Context(dev)

        self.ctx = pyopencl.create_some_context(interactive=False)
        # print(self.ctx)
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)

        self.chunk_size = None
        self.full_size = None

    def process(self, traces):
        assert traces.dtype == self.dtype

        if traces.shape[0] != self.full_size:
            if self.full_size is not None:
                print(f"Warning : chunk_size has changed {self.chunk_size} {traces.shape[0]}, need to recompile CL!!!")
            self.create_buffers_and_compile()

        event = pyopencl.enqueue_copy(self.queue, self.input_cl, traces)
        event = self.kern_sosfiltfilt(
            self.queue,
            (self.num_channels,),
            (self.num_channels,),
            self.input_cl,
            self.output_cl,
            self.coefficients_cl,
            self.zi1_cl,
            self.zi2_cl,
        )
        event.wait()
        event = pyopencl.enqueue_copy(self.queue, self.output, self.output_cl)

        return self.output

    def create_buffers_and_compile(self, chunk_size):
        self.chunk_size = chunk_size
        self.full_size = chunk_size + self.margin * 2
        n_section = self.coefficients.shape[0]

        buffer_nbytes = self.full_size * self.num_channels * self.dtype.itemsize

        # this is for stream processing
        self.zi1 = np.zeros((self.num_channels, n_section, 2), dtype="float32")
        self.zi2 = np.zeros((self.num_channels, n_section, 2), dtype="float32")
        self.output = np.zeros((self.full_size, self.num_channels), dtype=self.dtype)

        # GPU buffers
        self.coefficients_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.coefficients)
        self.zi1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.zi1)
        self.zi2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.zi2)
        self.input_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=buffer_nbytes)
        self.output_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=buffer_nbytes)

        variables = dict(
            full_size=self.full_size,
            n_section=n_section,
            num_channels=self.num_channels,
        )
        kernel_formated = processor_kernel % variables
        # print(kernel_formated)

        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build()  # options='-cl-mad-enable'

        self.kern_sosfiltfilt = getattr(self.opencl_prg, "sosfiltfilt")


processor_kernel = """
#define full_size %(full_size)d
#define n_section %(n_section)d
#define num_channels %(num_channels)d


__kernel void sos_filter(__global  float *input,
                                    __global  float *output,
                                    __constant  float *coefficients,
                                    __global float *zi,
                                    int local_chunksize,
                                    int direction) {

    int chan = get_global_id(0); //channel indice

    int offset_filt2;  //offset channel within section
    int offset_zi = chan*n_section*2;

    int idx;

    float w0, w1,w2;
    float res;

    for (int section=0; section<n_section; section++){

        //offset_filt2 = chan*n_section*6+section*6;
        offset_filt2 = section*6;

        w1 = zi[offset_zi+section*2+0];
        w2 = zi[offset_zi+section*2+1];

        for (int s=0; s<local_chunksize;s++){

            if (direction==1) {idx = s*num_channels+chan;}
            else if (direction==-1) {idx = (local_chunksize-s-1)*num_channels+chan;}

            if (section==0)  {w0 = input[idx];}
            else {w0 = output[idx];}

            w0 -= coefficients[offset_filt2+4] * w1;
            w0 -= coefficients[offset_filt2+5] * w2;
            res = coefficients[offset_filt2+0] * w0 + coefficients[offset_filt2+1] * w1 +  coefficients[offset_filt2+2] * w2;
            w2 = w1; w1 =w0;

            output[idx] = res;
        }

        zi[offset_zi+section*2+0] = w1;
        zi[offset_zi+section*2+1] = w2;

    }

}


__kernel void sosfiltfilt(__global  float *input,
                                        __global  float *output,
                                        __constant  float *coefficients,
                                        __global float * zi1,
                                        __global float * zi2
                                        ){

    int chan = get_global_id(0); //channel indice

    // filter forward
    sos_filter(input, input, coefficients, zi1, full_size, 1);

    // filter backward
    sos_filter(input, output, coefficients, zi2, full_size, -1);

}



"""
