"""Benchmark script for the parallel bandpass + CMR speedups.

Runs head-to-head comparisons on synthetic NumpyRecording fixtures so the
numbers are reproducible without external ephys data:

1. Component-level (hot operation only, no SI plumbing):
    - scipy.signal.sosfiltfilt serial vs channel-parallel threads
    - np.median(axis=1) serial vs time-parallel threads
2. Per-stage end-to-end (``rec.get_traces()`` path):
    - BandpassFilterRecording stock vs n_workers=8
    - CommonReferenceRecording stock vs n_workers=16
3. CRE (``TimeSeriesChunkExecutor``) × inner (n_workers) interaction at
   matched chunk_duration="1s".

FilterRecordingSegment and CommonReferenceRecordingSegment use
**per-caller-thread inner pools** (WeakKeyDictionary keyed by the calling
Thread object).  Each outer thread that calls get_traces() gets its own
inner ThreadPoolExecutor, so n_workers composes cleanly with CRE's outer
parallelism — no shared-pool queueing pathology.  See
``tests/test_parallel_pool_semantics.py`` for the contract.

Measured on a 24-core x86_64 host with 1M x 384 float32 chunks (SI 0.103
dev, numpy 2.1, scipy 1.14, full get_traces() path end-to-end):

    === Component-level (hot kernel only, no SI plumbing) ===
    sosfiltfilt serial → 8 threads:   7.80 s →  2.67 s (2.92x)
    np.median serial   → 16 threads:  3.51 s →  0.33 s (10.58x)

    === Per-stage end-to-end (rec.get_traces) ===
    Bandpass (5th-order, 300-6k Hz):  8.59 s →  3.20 s (2.69x)
    CMR median (global):              4.01 s →  0.81 s (4.95x)

    === CRE outer × inner (chunk=1s, per-caller pools) ===
    Bandpass: stock n=1 → stock n=8 thread:       7.42 s → 1.40 s (5.3x outer)
                         n_workers=8 n=1:          3.18 s (2.3x inner)
                         n_workers=8 n=8 thread:   1.24 s (combined)
    CMR:      stock n=1 → stock n=8 thread:       3.98 s → 0.61 s (6.5x outer)
                         n_workers=16 n=1:         1.58 s (2.5x inner)
                         n_workers=16 n=8 thread:  0.36 s (11.0x combined)

Bandpass and CMR scale sub-linearly with thread count due to memory
bandwidth saturation; 2.7x / 5x per stage on 8 / 16 threads respectively
is consistent with the DRAM ceiling at these chunk sizes, not a
parallelism bug.  Under CRE, the outer-vs-inner combination depends on
whether the inner pool has headroom over n_jobs — per-caller pools make
this deterministic regardless.

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
)


def _make_recording(T: int = 1_048_576, C: int = 384, fs: float = 30_000.0, dtype=np.float32):
    """Synthetic NumpyRecording matching typical Neuropixels shard shape."""
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((T, C)).astype(dtype) * 100.0
    rec = NumpyRecording([traces], sampling_frequency=fs)
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
    """End-to-end bench: BandpassFilterRecording stock vs n_workers=8."""
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
    """End-to-end bench: CommonReferenceRecording stock vs n_workers=16."""
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


def bench_bandpass_cre_interaction():
    """Bandpass: outer (TimeSeriesChunkExecutor) × inner (n_workers) parallelism.

    At SI's default ``chunk_duration="1s"``, the intra-chunk ``n_workers``
    kwarg is only useful when outer CRE workers don't already saturate cores.
    When combined, the result depends on whether inner-pool ``max_workers``
    exceeds outer ``n_jobs``.
    """
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== Bandpass: outer (CRE) × inner (n_workers), 1M × 384 float32, chunk=1s ===")
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
    """CMR: outer (TimeSeriesChunkExecutor) × inner (n_workers) parallelism."""
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=== CMR: outer (CRE) × inner (n_workers), 1M × 384 float32, chunk=1s ===")
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


def main():
    print("### COMPONENT-LEVEL (hot operation only) ###")
    print()
    bench_sosfiltfilt_component()
    bench_median_component()

    print("### PER-STAGE END-TO-END (rec.get_traces()) ###")
    print()
    bench_bandpass()
    bench_cmr()

    print("### CRE OUTER × INNER (chunk=1s) ###")
    print()
    bench_bandpass_cre_interaction()
    bench_cmr_cre_interaction()


if __name__ == "__main__":
    main()
