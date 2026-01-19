import numpy as np
import importlib.util

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

from spikeinterface.core.node_pipeline import (
    PeakDetector,
)
from spikeinterface.core.recording_tools import get_channel_distances
from .by_channel import ByChannelTorchPeakDetector

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    torch_nn_functional_spec = importlib.util.find_spec("torch.nn")
    if torch_nn_functional_spec is not None:
        HAVE_TORCH = True
    else:
        HAVE_TORCH = False
else:
    HAVE_TORCH = False

opencl_spec = importlib.util.find_spec("pyopencl")
if opencl_spec is not None:
    HAVE_PYOPENCL = True
else:
    HAVE_PYOPENCL = False

from .by_channel import ByChannelPeakDetector


class LocallyExclusivePeakDetector(PeakDetector):
    """Detect peaks using the "locally exclusive" method."""

    name = "locally_exclusive"
    engine = "numba"
    need_noise_levels = True
    preferred_mp_context = None
    params_doc = (
        ByChannelPeakDetector.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        return_output=True,
    ):
        if not HAVE_NUMBA:
            raise ModuleNotFoundError('"locally_exclusive" needs numba which is not installed')

        PeakDetector.__init__(self, recording, return_output=return_output)

        assert peak_sign in ("both", "neg", "pos")
        assert noise_levels is not None
        self.noise_levels = noise_levels

        self.abs_thresholds = self.noise_levels * detect_threshold
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.radius_um = radius_um
        self.detect_threshold = detect_threshold
        self.peak_sign = peak_sign
        # if remove_median:

        #     chunks = get_random_data_chunks(recording, return_in_uV=False, concatenated=True, **random_chunk_kwargs)
        #     medians = np.median(chunks, axis=0)
        #     medians = medians[None, :]
        #     print('medians', medians, noise_levels)
        # else:
        #     medians = None

        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

    def get_trace_margin(self):
        return self.exclude_sweep_size

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        assert HAVE_NUMBA, "You need to install numba"

        peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
            traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, self.neighbours_mask
        )

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        return (local_peaks,)


if HAVE_NUMBA:
    import numba

    def detect_peaks_numba_locally_exclusive_on_chunk(
        traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask
    ):

        # if medians is not None:
        #     traces = traces - medians

        traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]

        if peak_sign in ("pos", "both"):
            peak_mask = traces_center > abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_pos(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

        if peak_sign in ("neg", "both"):
            if peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_neg(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

            if peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # Find peaks and correct for time shift
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        peak_sample_ind += exclude_sweep_size

        return peak_sample_ind, peak_chan_ind

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_pos(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_neg(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask


class LocallyExclusiveTorchPeakDetector(ByChannelTorchPeakDetector):
    """Detect peaks using the "locally exclusive" method with pytorch."""

    name = "locally_exclusive_torch"
    engine = "torch"
    need_noise_levels = True
    preferred_mp_context = "spawn"
    params_doc = (
        ByChannelPeakDetector.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        device=None,
        radius_um=50,
        return_tensor=False,
        return_output=True,
    ):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')

        ByChannelTorchPeakDetector.__init__(
            self,
            recording,
            peak_sign,
            detect_threshold,
            exclude_sweep_ms,
            noise_levels,
            device,
            return_tensor,
            return_output,
        )

        self.channel_distance = get_channel_distances(recording)
        self.neighbour_indices_by_chan = []
        self.radius_um = radius_um
        self.num_channels = recording.get_num_channels()
        for chan in range(self.num_channels):
            self.neighbour_indices_by_chan.append(np.nonzero(self.channel_distance[chan] <= self.radius_um)[0])
        self.max_neighbs = np.max([len(neigh) for neigh in self.neighbour_indices_by_chan])
        self.neighbours_idxs = self.num_channels * np.ones((self.num_channels, self.max_neighbs), dtype=int)
        for i, neigh in enumerate(self.neighbour_indices_by_chan):
            self.neighbours_idxs[i, : len(neigh)] = neigh

    def get_trace_margin(self):
        return self.exclude_sweep_size

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        from .by_channel import _torch_detect_peaks

        peak_sample_ind, peak_chan_ind, peak_amplitude = _torch_detect_peaks(
            traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, self.neighbours_idxs, self.device
        )

        if not self.return_tensor:
            peak_sample_ind = np.array(peak_sample_ind.cpu())
            peak_chan_ind = np.array(peak_chan_ind.cpu())
            peak_amplitude = np.array(peak_amplitude.cpu())
            local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
            local_peaks["sample_index"] = peak_sample_ind
            local_peaks["channel_index"] = peak_chan_ind
            local_peaks["amplitude"] = peak_amplitude
            local_peaks["segment_index"] = segment_index
        return (local_peaks,)


class LocallyExclusiveOpenCLPeakDetector(LocallyExclusivePeakDetector):
    name = "locally_exclusive_cl"
    engine = "opencl"
    preferred_mp_context = None
    need_noise_levels = True
    params_doc = (
        LocallyExclusiveTorchPeakDetector.params_doc
        + """
    opencl_context_kwargs: None or dict
       kwargs to create the opencl context
    """
    )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        opencl_context_kwargs={},
    ):
        if not HAVE_PYOPENCL:
            raise ModuleNotFoundError('"locally_exclusive_cl" needs pyopencl which is not installed')

        LocallyExclusivePeakDetector.__init__(
            self, recording, peak_sign, detect_threshold, exclude_sweep_ms, radius_um, noise_levels
        )

        self.executor = OpenCLDetectPeakExecutor(
            self.abs_thresholds, self.exclude_sweep_size, self.neighbours_mask, self.peak_sign, **opencl_context_kwargs
        )

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        peak_sample_ind, peak_chan_ind = self.executor.detect_peak(traces)
        peak_sample_ind += self.exclude_sweep_size
        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        return (local_peaks,)


class OpenCLDetectPeakExecutor:
    def __init__(self, abs_thresholds, exclude_sweep_size, neighbours_mask, peak_sign):

        self.chunk_size = None

        self.abs_thresholds = abs_thresholds.astype("float32")
        self.exclude_sweep_size = exclude_sweep_size
        self.neighbours_mask = neighbours_mask.astype("uint8")
        self.peak_sign = peak_sign
        self.ctx = None
        self.queue = None
        self.x = 0

    def create_buffers_and_compile(self, chunk_size):
        import pyopencl

        mf = pyopencl.mem_flags
        try:
            self.device = pyopencl.get_platforms()[0].get_devices()[0]
            self.ctx = pyopencl.Context(devices=[self.device])
        except Exception as e:
            print("error create context ", e)

        self.queue = pyopencl.CommandQueue(self.ctx)
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        self.chunk_size = chunk_size

        self.neighbours_mask_cl = pyopencl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.neighbours_mask
        )
        self.abs_thresholds_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.abs_thresholds)

        num_channels = self.neighbours_mask.shape[0]
        self.traces_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=int(chunk_size * num_channels * 4))

        # TODO estimate smaller
        self.num_peaks = np.zeros(1, dtype="int32")
        self.num_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.num_peaks)

        nb_max_spike_in_chunk = num_channels * chunk_size
        self.peaks = np.zeros(nb_max_spike_in_chunk, dtype=[("sample_index", "int32"), ("channel_index", "int32")])
        self.peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.peaks)

        variables = dict(
            chunk_size=int(self.chunk_size),
            exclude_sweep_size=int(self.exclude_sweep_size),
            peak_sign={"pos": 1, "neg": -1}[self.peak_sign],
            num_channels=num_channels,
        )

        kernel_formated = processor_kernel % variables
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build()  # options='-cl-mad-enable'
        self.kern_detect_peaks = getattr(self.opencl_prg, "detect_peaks")

        self.kern_detect_peaks.set_args(
            self.traces_cl, self.neighbours_mask_cl, self.abs_thresholds_cl, self.peaks_cl, self.num_peaks_cl
        )

        s = self.chunk_size - 2 * self.exclude_sweep_size
        self.global_size = (s,)
        self.local_size = None

    def detect_peak(self, traces):
        self.x += 1

        import pyopencl

        if self.chunk_size is None or self.chunk_size != traces.shape[0]:
            self.create_buffers_and_compile(traces.shape[0])
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))

        pyopencl.enqueue_nd_range_kernel(
            self.queue,
            self.kern_detect_peaks,
            self.global_size,
            self.local_size,
        )

        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.num_peaks, self.num_peaks_cl)
        event = pyopencl.enqueue_copy(self.queue, self.peaks, self.peaks_cl)
        event.wait()

        n = self.num_peaks[0]
        peaks = self.peaks[:n]
        peak_sample_ind = peaks["sample_index"].astype("int64")
        peak_chan_ind = peaks["channel_index"].astype("int64")

        return peak_sample_ind, peak_chan_ind


processor_kernel = """
#define chunk_size %(chunk_size)d
#define exclude_sweep_size %(exclude_sweep_size)d
#define peak_sign %(peak_sign)d
#define num_channels %(num_channels)d


typedef struct st_peak{
    int sample_index;
    int channel_index;
} st_peak;


__kernel void detect_peaks(
                        //in
                        __global  float *traces,
                        __global  uchar *neighbours_mask,
                        __global  float *abs_thresholds,
                        //out
                        __global  st_peak *peaks,
                        volatile __global int *num_peaks
                ){
    int pos = get_global_id(0);

    if (pos == 0){
        *num_peaks = 0;
    }
    // this barrier OK if the first group is run first
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (pos>=(chunk_size - (2 * exclude_sweep_size))){
        return;
    }


    float v;
    uchar peak;
    uchar is_neighbour;

    int index;

    int i_peak;


    for (int chan=0; chan<num_channels; chan++){

        //v = traces[(pos + exclude_sweep_size)*num_channels + chan];
        index = (pos + exclude_sweep_size) * num_channels + chan;
        v = traces[index];

        if(peak_sign==1){
            if (v>abs_thresholds[chan]){peak=1;}
            else {peak=0;}
        }
        else if(peak_sign==-1){
            if (v<-abs_thresholds[chan]){peak=1;}
            else {peak=0;}
        }

        if (peak == 1){
            for (int chan_neigh=0; chan_neigh<num_channels; chan_neigh++){

                is_neighbour = neighbours_mask[chan * num_channels + chan_neigh];
                if (is_neighbour == 0){continue;}
                //if (chan == chan_neigh){continue;}

                index = (pos + exclude_sweep_size) * num_channels + chan_neigh;
                if(peak_sign==1){
                    peak = peak && (v>=traces[index]);
                }
                else if(peak_sign==-1){
                    peak = peak && (v<=traces[index]);
                }

                if (peak==0){break;}

                if(peak_sign==1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v>traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v>=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }
                else if(peak_sign==-1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v<traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v<=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }

            }

        }

        if (peak==1){
            //append to
            i_peak = atomic_inc(num_peaks);
            // sample_index is LOCAL to fifo
            peaks[i_peak].sample_index = pos + exclude_sweep_size;
            peaks[i_peak].channel_index = chan;
        }
    }

}
"""
