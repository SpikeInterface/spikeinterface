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
    CommonReferenceRecording,
    HighpassFilterRecording,
    PhaseShiftRecording,
)


def _make_aind_pipeline(source_rec, method, preserve_f32=False):
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
        hp = HighpassFilterRecording(ps, freq_min=300.0, dtype=np.float32)
        cmr = CommonReferenceRecording(hp, dtype=np.float32)
    else:
        ps = PhaseShiftRecording(source_rec, method=method)
        hp = HighpassFilterRecording(ps, freq_min=300.0, dtype=np.int16)
        cmr = CommonReferenceRecording(hp, dtype=np.int16)
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
    fast = _make_aind_pipeline(rec, method="fir")

    t_stock = _time_get_traces(stock)
    t_par = _time_get_traces(fast)
    print(f"  stock (FFT, serial):        {t_stock:6.2f} s")
    print(f"  FIR (int16):                {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
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
    fast = _make_aind_pipeline(rec, method="fir", preserve_f32=True)

    t_stock = _time_get_traces(stock)
    t_par = _time_get_traces(fast)
    print(f"  stock (FFT, serial) f32:    {t_stock:6.2f} s")
    print(f"  FIR f32 native:             {t_par:6.2f} s   ({t_stock / t_par:4.2f}x)")
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


def bench_cre_ps_algorithm():
    """FIR algorithmic change (PS) composed with SI's TimeSeriesChunkExecutor
    outer parallelism, AIND pipeline (PS → HP → CMR) at chunk=1s.

    Shows that the FIR algorithmic change multiplies cleanly with the
    existing outer chunk parallelism SI already ships — no downside to
    enabling ``method="fir"`` at any n_jobs.

    NumpyRecording source: CPU-only measurement (no IO).  For file-backed
    recordings, outer chunking additionally hides read latency; FIR only
    reduces per-chunk compute and composes with IO-oriented scheduling.

    pool_engine="thread" throughout.
    """
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== FIR × CRE n_jobs on AIND pipeline (1M × 384 int16, chunk=1s) ===")
    print("    (CPU-only — NumpyRecording source, no IO)")
    print()

    rec = _make_recording(dtype=np.int16)

    # (label, n_jobs, method, preserve_f32)
    configs = [
        ("CRE n=1, stock AIND",                 1, "fft", False),
        ("CRE n=1, FIR AIND (int16)",           1, "fir", False),
        ("CRE n=8 thread, stock AIND",          8, "fft", False),
        ("CRE n=8 thread, FIR AIND (int16)",    8, "fir", False),
        ("CRE n=24 thread, stock AIND",        24, "fft", False),
        ("CRE n=24 thread, FIR AIND (int16)",  24, "fir", False),
        ("CRE n=8 thread, FIR f32 (mipmap)",    8, "fir", True),
        ("CRE n=24 thread, FIR f32 (mipmap)",  24, "fir", True),
    ]

    results = []
    for label, n_jobs, method, preserve_f32 in configs:
        pipeline = _make_aind_pipeline(rec, method=method, preserve_f32=preserve_f32)
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
    print(f"  {'config':<40} {'time':>8}   {'speedup':>8}")
    for label, t in results:
        print(f"  {label:<40} {t:6.2f} s   {baseline / t:6.2f}×")
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
    # PS isolated benchmarks — FFT vs FIR on a single-stage PhaseShift.
    print("### PER-STAGE PhaseShift (rec.get_traces()) ###")
    print()
    bench_phase_shift_float32()
    bench_phase_shift_int16()

    # Algorithm vs parallelism decomposition (matched chunk size via CRE).
    print("### Algorithm vs parallelism ###")
    print()
    bench_phase_shift_algo_vs_parallelism()

    # End-to-end full AIND pipeline (PS → HP → CMR) with and without FIR.
    print("### END-TO-END AIND pipeline ###")
    print()
    bench_pipeline_int16()
    bench_pipeline_mipmap_f32()

    # FIR + CRE outer parallelism on the AIND pipeline.
    print("### FIR × CRE outer parallelism ###")
    print()
    bench_cre_ps_algorithm()

    # Peak memory by n_jobs × chunk size, FFT vs FIR.
    print("### Peak memory scaling ###")
    print()
    bench_peak_memory()


if __name__ == "__main__":
    main()
