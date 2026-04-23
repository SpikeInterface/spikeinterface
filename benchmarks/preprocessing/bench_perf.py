"""Benchmark script for the parallel preprocessing speedups.

Runs four head-to-head comparisons on synthetic NumpyRecording fixtures
so the numbers are reproducible without external ephys data:

1. BandpassFilter: stock (n_workers=1) vs n_workers=8
2. CommonReferenceRecording median: n_workers=1 vs n_workers=16
3. PhaseShiftRecording: method="fft" vs method="fir" (same parent dtype)
4. PhaseShiftRecording int16-native: method="fft" int16 vs
   method="fir" + output_dtype=float32

Measured on a 24-core x86_64 host with 1M x 384 chunks (SI 0.103 dev,
numpy 2.1, scipy 1.14, numba 0.60, full get_traces() path end-to-end):

    === Bandpass (5th-order Butterworth 300-6000 Hz, 1M x 384 float32) ===
      stock (n_workers=1):       8.67 s
      parallel (n_workers=8):    3.34 s   (2.60x)
      output matches stock within float32 tolerance

    === CMR median (global, 1M x 384 float32) ===
      stock (n_workers=1):       3.95 s
      parallel (n_workers=16):   0.83 s   (4.76x)
      output is bitwise-identical to stock

    === PhaseShift (1M x 384 float32) ===
      method="fft":             68.07 s
      method="fir":             0.695 s   (97.94x)
      spike-band RMS error / signal RMS: 0.198%

    === PhaseShift int16-native (1M x 384 int16) ===
      method="fft" (int16 out):    69.53 s
      method="fir" + f32 out:       0.446 s   (156.06x)

The FIR speedup is larger end-to-end than kernel-only because it also
bypasses the 40 ms margin and float64 round-trip required by the FFT
path.  See the phase_shift.py docstring for the correctness analysis.

Bandpass and CMR scale sub-linearly with thread count due to memory
bandwidth saturation; 2.6x / 4.76x on 8 / 16 threads respectively is
consistent with the DRAM ceiling at these chunk sizes, not a
parallelism bug.

Run with ``python -m benchmarks.preprocessing.bench_perf`` from repo root.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.signal

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    CommonReferenceRecording,
    HighpassFilterRecording,
    PhaseShiftRecording,
)


def _make_aind_pipeline(source_rec, method, inner=1, preserve_f32=False):
    """Build the AIND production preprocessing chain: PS → HP → CMR.

    Dtype handling:
      - int16 (AIND production, default): HP and CMR explicitly set dtype=int16.
        Each stage round-trips through float internally (scipy's f64, PS's f32)
        then casts back to int16 at its output.  Matches the saved provenance
        in AIND analyzer zarrs.
      - f32 propagation (preserve_f32=True): PS uses method=fir with
        output_dtype=float32 (when method allows), HP and CMR set dtype=float32.
        Avoids the per-stage round-back-to-int16.  Matches what a
        mipmap-zarr-style consumer could do if it rewrote the provenance.
    """
    if preserve_f32:
        ps_output_dtype = np.float32 if method == "fir" else None
        ps = PhaseShiftRecording(source_rec, method=method, output_dtype=ps_output_dtype)
        hp = HighpassFilterRecording(ps, freq_min=300.0, dtype=np.float32, n_workers=max(inner, 1))
        cmr = CommonReferenceRecording(hp, dtype=np.float32, n_workers=max(inner, 1))
    else:
        ps = PhaseShiftRecording(source_rec, method=method)
        hp = HighpassFilterRecording(ps, freq_min=300.0, dtype=np.int16, n_workers=max(inner, 1))
        cmr = CommonReferenceRecording(hp, dtype=np.int16, n_workers=max(inner, 1))
    return cmr


def _make_recording(T: int = 1_048_576, C: int = 384, fs: float = 30_000.0, dtype=np.float32):
    """Synthetic NumpyRecording matching typical Neuropixels shard shape."""
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        traces = rng.standard_normal((T, C)).astype(dtype) * 100.0
    else:
        traces = rng.integers(-1000, 1000, size=(T, C), dtype=dtype)
    rec = NumpyRecording([traces], sampling_frequency=fs)
    rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=C))
    return rec


def _time_get_traces(rec, *, n_reps=3, warmup=1):
    """Median-of-N timing of rec.get_traces() for the full single segment."""
    for _ in range(warmup):
        rec.get_traces()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        rec.get_traces()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_callable(fn, *, n_reps=3, warmup=1):
    """Best-of-N timing for a bare callable.  Used for component-level benches
    where we want to isolate the hot operation from surrounding glue."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def bench_sosfiltfilt_component():
    """Component-level bench: just scipy.signal.sosfiltfilt vs channel-parallel.

    Isolates the hot SOS operation from the full BandpassFilter.get_traces
    path so you can see the kernel-only speedup (no margin fetch, no dtype
    cast, no slice).
    """
    from concurrent.futures import ThreadPoolExecutor

    print("--- [component] sosfiltfilt (1M x 384 float32) ---")
    T, C = 1_048_576, 384
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, C)).astype(np.float32) * 100.0
    sos = scipy.signal.butter(5, [300.0, 6000.0], btype="bandpass", fs=30_000.0, output="sos")

    pool = ThreadPoolExecutor(max_workers=8)

    def parallel_call():
        block = (C + 8 - 1) // 8
        bounds = [(c0, min(c0 + block, C)) for c0 in range(0, C, block)]

        def _work(c0, c1):
            return c0, c1, scipy.signal.sosfiltfilt(sos, x[:, c0:c1], axis=0)

        results = [fut.result() for fut in [pool.submit(_work, c0, c1) for c0, c1 in bounds]]
        out = np.empty((T, C), dtype=results[0][2].dtype)
        for c0, c1, block_out in results:
            out[:, c0:c1] = block_out
        return out

    t_stock = _time_callable(lambda: scipy.signal.sosfiltfilt(sos, x, axis=0))
    t_par = _time_callable(parallel_call)
    pool.shutdown()
    print(f"  scipy.sosfiltfilt serial:      {t_stock:6.2f} s")
    print(f"  scipy.sosfiltfilt 8 threads:   {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
    print()


def bench_median_component():
    """Component-level bench: just np.median(axis=1) vs threaded across time blocks."""
    from concurrent.futures import ThreadPoolExecutor

    print("--- [component] np.median axis=1 (1M x 384 float32) ---")
    T, C = 1_048_576, 384
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, C)).astype(np.float32) * 100.0

    pool = ThreadPoolExecutor(max_workers=16)

    def parallel_call():
        block = (T + 16 - 1) // 16
        bounds = [(t0, min(t0 + block, T)) for t0 in range(0, T, block)]

        def _work(t0, t1):
            return t0, t1, np.median(x[t0:t1, :], axis=1)

        results = [fut.result() for fut in [pool.submit(_work, t0, t1) for t0, t1 in bounds]]
        out = np.empty(T, dtype=results[0][2].dtype)
        for t0, t1, block_out in results:
            out[t0:t1] = block_out
        return out

    t_stock = _time_callable(lambda: np.median(x, axis=1))
    t_par = _time_callable(parallel_call)
    pool.shutdown()
    print(f"  np.median serial:              {t_stock:6.2f} s")
    print(f"  np.median 16 threads:          {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
    print()


def bench_bandpass():
    print("=== Bandpass (5th-order Butterworth 300-6000 Hz, 1M x 384 float32) ===")
    rec = _make_recording(dtype=np.float32)
    stock = BandpassFilterRecording(rec, freq_min=300.0, freq_max=6000.0, margin_ms=40.0)
    fast = BandpassFilterRecording(rec, freq_min=300.0, freq_max=6000.0, margin_ms=40.0, n_workers=8)

    t_stock = _time_get_traces(stock)
    t_fast = _time_get_traces(fast)
    print(f"  stock (n_workers=1):     {t_stock:6.2f} s")
    print(f"  parallel (n_workers=8):  {t_fast:6.2f} s   ({t_stock / t_fast:4.2f}x)")
    # Equivalence check
    ref = stock.get_traces(start_frame=1000, end_frame=10_000)
    out = fast.get_traces(start_frame=1000, end_frame=10_000)
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-4), "parallel bandpass output mismatch"
    print("  output matches stock within float32 tolerance")
    print()


def bench_cmr():
    print("=== CMR median (global, 1M x 384 float32) ===")
    rec = _make_recording(dtype=np.float32)
    stock = CommonReferenceRecording(rec, operator="median", reference="global")
    fast = CommonReferenceRecording(rec, operator="median", reference="global", n_workers=16)

    t_stock = _time_get_traces(stock)
    t_fast = _time_get_traces(fast)
    print(f"  stock (n_workers=1):     {t_stock:6.2f} s")
    print(f"  parallel (n_workers=16): {t_fast:6.2f} s   ({t_stock / t_fast:4.2f}x)")
    ref = stock.get_traces(start_frame=1000, end_frame=10_000)
    out = fast.get_traces(start_frame=1000, end_frame=10_000)
    np.testing.assert_array_equal(out, ref)
    print("  output is bitwise-identical to stock")
    print()


def bench_phase_shift_float32():
    print("=== PhaseShift (1M x 384 float32) ===")
    rec = _make_recording(dtype=np.float32)
    fft_rec = PhaseShiftRecording(rec, method="fft")
    fir_rec = PhaseShiftRecording(rec, method="fir")
    t_fft = _time_get_traces(fft_rec)
    t_fir = _time_get_traces(fir_rec)
    print(f'  method="fft":            {t_fft:6.2f} s')
    print(f'  method="fir":            {t_fir:6.3f} s   ({t_fft / t_fir:4.2f}x)')
    # Spike-band RMS error (300-5000 Hz) as a correctness check.
    edge = 5000
    ref = fft_rec.get_traces(start_frame=edge, end_frame=rec.get_num_samples() - edge)
    out = fir_rec.get_traces(start_frame=edge, end_frame=rec.get_num_samples() - edge)
    sos = scipy.signal.butter(4, [300.0, 5000.0], btype="bandpass", fs=30_000.0, output="sos")
    ref_bp = scipy.signal.sosfiltfilt(sos, ref.astype(np.float64), axis=0)
    out_bp = scipy.signal.sosfiltfilt(sos, out.astype(np.float64), axis=0)
    sig_rms = float(np.sqrt(np.mean(ref_bp**2)))
    err_rms = float(np.sqrt(np.mean((out_bp - ref_bp) ** 2)))
    print(f"  spike-band RMS error / signal RMS: {100 * err_rms / sig_rms:.3f}%")
    print()


def bench_phase_shift_int16():
    print("=== PhaseShift int16-native (1M x 384 int16) ===")
    rec = _make_recording(dtype=np.int16)
    fft_rec = PhaseShiftRecording(rec, method="fft")  # stock: int16 in -> int16 out
    fir_rec = PhaseShiftRecording(rec, method="fir", output_dtype=np.float32)
    t_fft = _time_get_traces(fft_rec)
    t_fir = _time_get_traces(fir_rec)
    print(f'  method="fft" (int16 out):    {t_fft:6.2f} s')
    print(f'  method="fir" + f32 out:      {t_fir:6.3f} s   ({t_fft / t_fir:4.2f}x)')
    print()


def bench_pipeline_int16():
    """AIND production pipeline end-to-end (PS → HP → CMR, int16 throughout).

    Matches what the saved AIND sorting provenance actually does: PS first to
    correct ADC staggering, then 300 Hz highpass, then global CMR, all with
    explicit dtype=int16 on HP and CMR.  Output is int16.  The FIR
    algorithmic change at PS still helps (int16-native kernel reads int16
    directly, accumulates in f32), even though the downstream int16 cast
    defeats the f32 output-propagation optimization.
    """
    print("=== Pipeline AIND-style (PS → HP → CMR, int16 throughout, 1M x 384) ===")
    rec = _make_recording(dtype=np.int16)

    stock = _make_aind_pipeline(rec, method="fft")
    fast = _make_aind_pipeline(rec, method="fir", inner=8)

    t_stock = _time_get_traces(stock)
    t_par = _time_get_traces(fast)
    print(f"  stock (FFT, serial):        {t_stock:6.2f} s")
    print(f"  parallel+FIR (int16):       {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
    assert stock.get_dtype() == np.int16, f"stock output dtype {stock.get_dtype()} != int16"
    assert fast.get_dtype() == np.int16, f"fast output dtype {fast.get_dtype()} != int16"
    print(f"  output dtype: {fast.get_dtype()} (AIND production contract)")
    print()


def bench_pipeline_mipmap_f32():
    """Mipmap-style pipeline end-to-end (PS → HP → CMR, f32 propagated).

    A variant where the consumer rewrites the AIND provenance to set
    dtype=float32 on HP and CMR (or builds a fresh chain from scratch),
    and PS uses output_dtype=float32.  Each stage skips the round-back-to-int16
    step.  Output is float32 — different contract than AIND-preserving but
    what a viewer / mipmap builder that already consumes float32 downstream
    could use.
    """
    print("=== Pipeline mipmap-style (PS → HP → CMR, f32 propagated, 1M x 384) ===")
    rec = _make_recording(dtype=np.int16)

    stock = _make_aind_pipeline(rec, method="fft", preserve_f32=True)
    fast = _make_aind_pipeline(rec, method="fir", inner=8, preserve_f32=True)

    t_stock = _time_get_traces(stock)
    t_par = _time_get_traces(fast)
    print(f"  stock (FFT, serial) f32:    {t_stock:6.2f} s")
    print(f"  parallel+FIR f32 native:    {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
    print(f"  output dtype: {fast.get_dtype()} (f32 propagated end-to-end)")
    print()


def _time_cre(executor, *, n_reps=2, warmup=1):
    """Min-of-N timing for a TimeSeriesChunkExecutor invocation."""
    for _ in range(warmup):
        executor.run()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        executor.run()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _cre_init(recording):
    return {"recording": recording}


def _cre_func(segment_index, start_frame, end_frame, worker_dict):
    worker_dict["recording"].get_traces(
        start_frame=start_frame, end_frame=end_frame, segment_index=segment_index
    )


def bench_cre_outer_vs_intra():
    """SI's ChunkRecordingExecutor outer parallelism vs our intra-chunk
    parallelism, and their combinations.  Same 1M × 384 int16 pipeline, only
    the parallelization strategy varies.

    NumpyRecording source: this is a **CPU-only** measurement.  For
    file-backed sources (binary, zarr, wavpack, S3), CRE outer chunking
    additionally hides read latency; our intra-chunk parallelism only
    reduces per-chunk compute and composes with IO-oriented scheduling
    rather than replacing it.

    pool_engine="thread" throughout: pool_engine="process" on a NumpyRecording
    would pickle the ~768 MB buffer to each worker and dominate wall-clock.
    In production on file-backed recordings, pickling is cheap and the
    three-way thread-pool contention analysis here still applies.
    """
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== CRE outer × intra-chunk parallelism (1M × 384 int16, chunk=1s) ===")
    print("    (CPU-only — NumpyRecording source, no IO; see notes above)")
    print()

    rec = _make_recording(dtype=np.int16)

    # (label, n_jobs, inner, method, preserve_f32)
    configs = [
        ("CRE n=1, stock AIND",              1,  1, "fft", False),
        ("CRE n=1, fast AIND (int16)",       1,  8, "fir", False),
        ("CRE n=8 thread, stock AIND",       8,  1, "fft", False),
        ("CRE n=8 thread, fast AIND (int16)",8,  8, "fir", False),
        ("CRE n=24 thread, fast AIND (int16)",24, 1, "fir", False),
        ("CRE n=8 thread, fast f32 (mipmap)",8,  8, "fir", True),
        ("CRE n=24 thread, fast f32 (mipmap)",24, 1, "fir", True),
    ]

    results = []
    for label, n_jobs, inner, method, preserve_f32 in configs:
        pipeline = _make_aind_pipeline(rec, method=method, inner=inner, preserve_f32=preserve_f32)
        ex = TimeSeriesChunkExecutor(
            time_series=pipeline,
            func=_cre_func,
            init_func=_cre_init,
            init_args=(pipeline,),
            pool_engine="thread",
            n_jobs=n_jobs,
            chunk_duration="1s",
            progress_bar=False,
        )
        t = _time_cre(ex)
        results.append((label, t))

    baseline = results[0][1]
    print(f"  {'config':<30} {'time':>8}   {'speedup':>8}")
    for label, t in results:
        print(f"  {label:<30} {t:6.2f} s   {baseline / t:6.2f}×")
    print()


def bench_bandpass_cre_interaction():
    """Bandpass: outer (CRE) vs inner (n_workers) parallelism at matched chunk size.

    No algorithmic change here — BP is the same scipy sosfiltfilt either way.
    Question is whether outer-only parallelism already saturates (so the
    intra-chunk ``n_workers`` kwarg is redundant) or whether they compose.
    """
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== Bandpass: outer (CRE) × inner (n_workers) parallelism (1M × 384 float32, chunk=1s) ===")
    rec = _make_recording(dtype=np.float32)

    def make_cre(bp_rec, n_jobs):
        return TimeSeriesChunkExecutor(
            time_series=bp_rec, func=_cre_func, init_func=_cre_init, init_args=(bp_rec,),
            pool_engine="thread", n_jobs=n_jobs, chunk_duration="1s", progress_bar=False,
        )

    t_stock_n1 = _time_cre(make_cre(BandpassFilterRecording(rec), n_jobs=1))
    t_stock_n8 = _time_cre(make_cre(BandpassFilterRecording(rec), n_jobs=8))
    t_fast_n1 = _time_cre(make_cre(BandpassFilterRecording(rec, n_workers=8), n_jobs=1))
    t_fast_n8 = _time_cre(make_cre(BandpassFilterRecording(rec, n_workers=8), n_jobs=8))

    print(f"  {'config':<40} {'time':>8}   {'vs baseline':>12}")
    print(f"  {'stock, CRE n=1 (baseline)':<40} {t_stock_n1:6.2f} s   {'1.00×':>12}")
    print(f"  {'stock, CRE n=8 thread':<40} {t_stock_n8:6.2f} s   {t_stock_n1/t_stock_n8:5.2f}× (outer only)")
    print(f"  {'n_workers=8, CRE n=1':<40} {t_fast_n1:6.2f} s   {t_stock_n1/t_fast_n1:5.2f}× (inner only)")
    print(f"  {'n_workers=8, CRE n=8 thread':<40} {t_fast_n8:6.2f} s   {t_stock_n1/t_fast_n8:5.2f}× (both)")
    print()


def bench_cmr_cre_interaction():
    """CMR: outer (CRE) vs inner (n_workers) parallelism at matched chunk size."""
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== CMR: outer (CRE) × inner (n_workers) parallelism (1M × 384 float32, chunk=1s) ===")
    rec = _make_recording(dtype=np.float32)

    def make_cre(cmr_rec, n_jobs):
        return TimeSeriesChunkExecutor(
            time_series=cmr_rec, func=_cre_func, init_func=_cre_init, init_args=(cmr_rec,),
            pool_engine="thread", n_jobs=n_jobs, chunk_duration="1s", progress_bar=False,
        )

    t_stock_n1 = _time_cre(make_cre(CommonReferenceRecording(rec), n_jobs=1))
    t_stock_n8 = _time_cre(make_cre(CommonReferenceRecording(rec), n_jobs=8))
    t_fast_n1 = _time_cre(make_cre(CommonReferenceRecording(rec, n_workers=16), n_jobs=1))
    t_fast_n8 = _time_cre(make_cre(CommonReferenceRecording(rec, n_workers=16), n_jobs=8))

    print(f"  {'config':<40} {'time':>8}   {'vs baseline':>12}")
    print(f"  {'stock, CRE n=1 (baseline)':<40} {t_stock_n1:6.2f} s   {'1.00×':>12}")
    print(f"  {'stock, CRE n=8 thread':<40} {t_stock_n8:6.2f} s   {t_stock_n1/t_stock_n8:5.2f}× (outer only)")
    print(f"  {'n_workers=16, CRE n=1':<40} {t_fast_n1:6.2f} s   {t_stock_n1/t_fast_n1:5.2f}× (inner only)")
    print(f"  {'n_workers=16, CRE n=8 thread':<40} {t_fast_n8:6.2f} s   {t_stock_n1/t_fast_n8:5.2f}× (both)")
    print()


def bench_peak_memory():
    """Measure peak RSS for CRE configs at varying n_jobs and chunk_duration.

    Each config runs in a fresh subprocess so per-config peak RSS is clean
    (Python's allocator retains memory within a process, confounding same-process
    measurements).  pool_engine="thread" throughout; process engine would add a
    per-worker recording-footprint term on top.
    """
    import subprocess
    import sys
    import textwrap

    def measure(n_jobs, method, inner, chunk_duration, preserve_f32, T, C):
        code = textwrap.dedent(f"""
            import numpy as np, numba, resource, threading, time, psutil, os
            import sys
            sys.path.insert(0, {repr(str(__file__).rsplit('/', 3)[0])})
            from benchmarks.preprocessing.bench_perf import (
                _make_recording, _make_aind_pipeline, _cre_func, _cre_init,
            )
            from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

            proc = psutil.Process(os.getpid())
            rec = _make_recording(T={T}, C={C}, dtype=np.int16)
            baseline = proc.memory_info().rss

            numba.set_num_threads(max({inner}, 1))
            pipeline = _make_aind_pipeline(rec, method="{method}", inner={inner}, preserve_f32={preserve_f32})
            ex = TimeSeriesChunkExecutor(
                time_series=pipeline, func=_cre_func, init_func=_cre_init, init_args=(pipeline,),
                pool_engine="thread", n_jobs={n_jobs}, chunk_duration="{chunk_duration}", progress_bar=False,
            )
            # warmup + sampled run
            ex.run()
            peak = [proc.memory_info().rss]
            stop = threading.Event()
            def sampler():
                while not stop.wait(0.02):
                    peak[0] = max(peak[0], proc.memory_info().rss)
            thr = threading.Thread(target=sampler, daemon=True)
            thr.start()
            ex.run()
            stop.set()
            thr.join()
            print(f"BASELINE_B {{baseline}}")
            print(f"PEAK_B {{peak[0]}}")
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [measurement failed] {{result.stderr[-300:]}}")
            return None, None
        baseline_b = peak_b = None
        for line in result.stdout.splitlines():
            if line.startswith("BASELINE_B"):
                baseline_b = int(line.split()[1])
            elif line.startswith("PEAK_B"):
                peak_b = int(line.split()[1])
        return baseline_b, peak_b

    print("=== Peak RSS by n_jobs × chunk_duration (1M × 384 int16, thread engine) ===")
    print("    Each config runs in a fresh subprocess for clean peak RSS.")
    print()

    # (label, n_jobs, method, inner, chunk_duration, preserve_f32)
    configs = [
        ("CRE n=1, stock, chunk=1s",         1, "fft",  1,  "1s", False),
        ("CRE n=1, fast, chunk=1s",          1, "fir",  8,  "1s", False),
        ("CRE n=4, stock, chunk=1s",         4, "fft",  1,  "1s", False),
        ("CRE n=8, stock, chunk=1s",         8, "fft",  1,  "1s", False),
        ("CRE n=24, stock, chunk=1s",       24, "fft",  1,  "1s", False),
        ("CRE n=24, fast, chunk=1s",        24, "fir",  1,  "1s", False),
        # larger chunks
        ("CRE n=1, stock, chunk=10s",        1, "fft",  1, "10s", False),
        ("CRE n=4, stock, chunk=10s",        4, "fft",  1, "10s", False),
        ("CRE n=8, stock, chunk=10s",        8, "fft",  1, "10s", False),
        ("CRE n=24, stock, chunk=10s",      24, "fft",  1, "10s", False),
        ("CRE n=24, fast, chunk=10s",       24, "fir",  1, "10s", False),
    ]

    print(f"  {'config':<32} {'baseline':>10} {'peak':>10} {'Δ':>10}")
    for label, n_jobs, method, inner, chunk, preserve_f32 in configs:
        baseline_b, peak_b = measure(n_jobs, method, inner, chunk, preserve_f32, 1_048_576, 384)
        if baseline_b is None:
            continue
        delta_gb = (peak_b - baseline_b) / 2**30
        print(f"  {label:<32} {baseline_b/2**30:>8.2f}GB {peak_b/2**30:>8.2f}GB {delta_gb:>8.2f}GB")
    print()


def bench_thread_split_sweep():
    """Sweep (outer CRE n_jobs, inner n_workers) splits holding total thread
    budget ≈ core count, to find the empirical best combination for the full
    int16 pipeline.

    Same 1M × 384 int16 pipeline; only the outer/inner split varies.
    Chunk size = 1s (SI default).  Numba threads are pinned to ``inner`` per
    config so PS's numba pool matches the other stages' thread budget
    (otherwise numba defaults to all cores and oversubscribes on combined
    configs).
    """
    import numba
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== Thread-split sweep: outer × inner (1M × 384 int16 pipeline, chunk=1s) ===")
    rec = _make_recording(dtype=np.int16)

    def time_config(method, n_jobs, inner, preserve_f32=False):
        saved = numba.get_num_threads()
        try:
            numba.set_num_threads(max(inner, 1))
            pipeline = _make_aind_pipeline(rec, method=method, inner=inner, preserve_f32=preserve_f32)
            ex = TimeSeriesChunkExecutor(
                time_series=pipeline, func=_cre_func, init_func=_cre_init, init_args=(pipeline,),
                pool_engine="thread", n_jobs=n_jobs, chunk_duration="1s", progress_bar=False,
            )
            return _time_cre(ex)
        finally:
            numba.set_num_threads(saved)

    # (label, method, outer_n_jobs, inner_n_workers) — total threads ≈ outer × inner
    configs = [
        ("stock, outer=1 (baseline)",        "fft",  1,  1),
        ("stock, outer=24",                  "fft", 24,  1),
        ("fast, outer=1,  inner=24",         "fir",  1, 24),
        ("fast, outer=2,  inner=12",         "fir",  2, 12),
        ("fast, outer=3,  inner=8",          "fir",  3,  8),
        ("fast, outer=4,  inner=6",          "fir",  4,  6),
        ("fast, outer=6,  inner=4",          "fir",  6,  4),
        ("fast, outer=8,  inner=3",          "fir",  8,  3),
        ("fast, outer=12, inner=2",          "fir", 12,  2),
        ("fast, outer=24, inner=1",          "fir", 24,  1),
        # Inner oversubscription (total threads > core count)
        ("fast, outer=24, inner=2 (OS 2x)",  "fir", 24,  2),
        ("fast, outer=24, inner=4 (OS 4x)",  "fir", 24,  4),
        ("fast, outer=24, inner=8 (OS 8x)",  "fir", 24,  8),
        ("fast, outer=12, inner=8 (OS 4x)",  "fir", 12,  8),
        ("fast, outer=8,  inner=8 (OS ~3x)", "fir",  8,  8),
    ]

    results = []
    for label, method, n_jobs, inner in configs:
        t = time_config(method, n_jobs, inner)
        results.append((label, t))

    baseline = results[0][1]
    best = min(r[1] for r in results)
    print(f"  {'config':<40} {'time':>8}   {'vs baseline':>12}   {'vs best':>8}")
    for label, t in results:
        marker = "  ←" if t == best else ""
        print(f"  {label:<40} {t:6.2f} s   {baseline/t:8.2f}×      {best/t:5.2f}×{marker}")
    print()


def bench_phase_shift_algo_vs_parallelism():
    """Decompose the phase-shift speedup into algorithmic (FFT → FIR) and
    parallel components, at **matched chunk size**.

    All four configs go through CRE with ``chunk_duration="1s"`` so chunk
    size is constant; only n_jobs, method, and numba threads vary.  This
    isolates algorithm-change-alone vs parallelism-alone from the
    chunk-size effect (scipy FFT scales as O(N log N); smaller chunks run
    much faster per sample independently of parallelism).

    Answers: "Is the FIR speedup just what CRE n_jobs=N gives me on stock
    FFT?"  No — even at identical chunk size, the algorithmic change alone
    beats CRE's best parallelism on stock, and the two compose.
    """
    import numba
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== Phase-shift: algorithm vs parallelism (1M × 384 int16, chunk=1s) ===")
    rec = _make_recording(dtype=np.int16)
    fft_rec = PhaseShiftRecording(rec, method="fft")
    fir_rec = PhaseShiftRecording(rec, method="fir")

    def make_cre(rec, n_jobs):
        return TimeSeriesChunkExecutor(
            time_series=rec, func=_cre_func, init_func=_cre_init, init_args=(rec,),
            pool_engine="thread", n_jobs=n_jobs, chunk_duration="1s", progress_bar=False,
        )

    # 1. FFT, CRE n=1 — baseline at chunk=1s
    t_fft_n1 = _time_cre(make_cre(fft_rec, n_jobs=1))

    # 2. FFT, CRE n=8 thread — outer parallelism only on stock algorithm
    t_fft_n8 = _time_cre(make_cre(fft_rec, n_jobs=8))

    # 3. FIR, CRE n=1, numba 1-thread — algorithm only (no parallelism at all)
    saved = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        t_fir_serial = _time_cre(make_cre(fir_rec, n_jobs=1))
    finally:
        numba.set_num_threads(saved)

    # 4. FIR, CRE n=1, numba default — algorithm + inner parallelism only
    t_fir_inner = _time_cre(make_cre(fir_rec, n_jobs=1))

    # 5. FIR, CRE n=8 thread, numba default — algorithm + inner + outer
    t_fir_full = _time_cre(make_cre(fir_rec, n_jobs=8))

    print(f"  {'config':<40} {'time':>8}   {'vs baseline':>12}")
    print(f"  {'FFT, CRE n=1 (baseline)':<40} {t_fft_n1:6.2f} s   {'1.00×':>12}")
    print(f"  {'FFT, CRE n=8 thread':<40} {t_fft_n8:6.2f} s   {t_fft_n1/t_fft_n8:5.2f}× (outer only)")
    print(f"  {'FIR, CRE n=1, numba 1-thread':<40} {t_fir_serial:6.2f} s   {t_fft_n1/t_fir_serial:5.2f}× (algorithm only)")
    print(f"  {'FIR, CRE n=1, numba default':<40} {t_fir_inner:6.2f} s   {t_fft_n1/t_fir_inner:5.2f}× (algo + inner only)")
    print(f"  {'FIR, CRE n=8 thread, numba default':<40} {t_fir_full:6.2f} s   {t_fft_n1/t_fir_full:5.2f}× (algo + inner + outer)")
    print()


def main():
    # Component-level: isolated hot operation, fixed buffer.  Shows the raw
    # kernel speedup without the surrounding get_traces() plumbing.
    print("### COMPONENT-LEVEL (hot operation only) ###")
    print()
    bench_sosfiltfilt_component()
    bench_median_component()

    # End-to-end (per-stage): full rec.get_traces() through a single
    # preprocessing class.  Includes margin fetch, dtype cast, slicing,
    # subtract — the overhead users actually experience.  These ratios are
    # lower than the component ones because the non-parallelizable glue
    # dilutes the speedup.
    print("### END-TO-END per stage (rec.get_traces()) ###")
    print()
    bench_bandpass()
    bench_cmr()
    bench_phase_shift_float32()
    bench_phase_shift_int16()

    # End-to-end full pipeline: all three stages chained, int16 preserved.
    # This is the headline number — what a user running the full
    # preprocessing chain actually saves with every option enabled.
    print("### END-TO-END full pipeline (int16 preserved) ###")
    print()
    bench_pipeline_int16()


if __name__ == "__main__":
    main()
