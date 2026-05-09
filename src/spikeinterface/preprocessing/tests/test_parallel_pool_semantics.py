"""Tests for the per-caller-thread pool semantics used by
``BaseRecording.get_traces_multi_thread`` (FilterRecording, CommonReferenceRecording).

Contract: each outer thread that calls ``get_traces_multi_thread`` gets its own
inner ``ThreadPoolExecutor`` (keyed in a module-global registry by
``(Thread, max_threads)``).  Keying by Thread avoids the shared-pool queueing
pathology that arises when many outer workers submit concurrently into a
single inner pool with fewer max_workers than outer callers.

The pool registry lives in ``core/job_tools._inner_pools`` rather than on each
segment, so a chained pipeline reuses one pool per ``(Thread, max_threads)``
pair across segments.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading

import numpy as np
import pytest

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    CommonReferenceRecording,
)
from spikeinterface.core.job_tools import _inner_pools, get_inner_pool


def _make_recording(T: int = 50_000, C: int = 64, fs: float = 30_000.0):
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((T, C)).astype(np.float32) * 100.0
    return NumpyRecording([traces], sampling_frequency=fs)


@pytest.fixture
def filter_rec():
    return BandpassFilterRecording(_make_recording(), freq_min=300.0, freq_max=6000.0)


@pytest.fixture
def cmr_rec():
    return CommonReferenceRecording(_make_recording(), operator="median", reference="global")


def _pool_for_current_thread(max_threads: int):
    sized = _inner_pools.get(threading.current_thread())
    if sized is None:
        return None
    return sized.get(max_threads)


class TestPerCallerThreadPool:
    """Verify each calling thread gets its own inner pool, keyed by max_threads."""

    @pytest.mark.parametrize("rec_fixture", ["filter_rec", "cmr_rec"])
    def test_single_caller_reuses_pool(self, rec_fixture, request):
        """Repeated calls from the same thread reuse the same inner pool."""
        rec = request.getfixturevalue(rec_fixture)
        rec.get_traces_multi_thread(start_frame=0, end_frame=50_000, max_threads=4)
        pool_a = _pool_for_current_thread(4)
        rec.get_traces_multi_thread(start_frame=0, end_frame=50_000, max_threads=4)
        pool_b = _pool_for_current_thread(4)
        assert pool_a is not None
        assert pool_a is pool_b, "expected the same inner pool to be reused across calls from the same thread"

    @pytest.mark.parametrize("rec_fixture", ["filter_rec", "cmr_rec"])
    def test_concurrent_callers_get_distinct_pools(self, rec_fixture, request):
        """Two outer threads calling get_traces_multi_thread concurrently must
        receive different inner pools — not a shared one that would queue their
        tasks through a single bottleneck.
        """
        rec = request.getfixturevalue(rec_fixture)

        ready = threading.Barrier(2)
        captured = {}

        def worker(name):
            ready.wait()
            rec.get_traces_multi_thread(start_frame=0, end_frame=50_000, max_threads=4)
            captured[name] = _pool_for_current_thread(4)

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert captured["t1"] is not None
        assert captured["t2"] is not None
        assert captured["t1"] is not captured["t2"], (
            "expected distinct inner pools for concurrent callers; a shared "
            "single-pool design would cause queueing pathology"
        )

    def test_distinct_max_threads_get_distinct_pools(self):
        """Same caller, different max_threads => different pools.

        get_inner_pool is keyed by (Thread, max_threads) so a viewer that
        flips between budgets gets a fresh pool of the right size each time
        rather than sharing one undersized pool.
        """
        pool_a = get_inner_pool(2)
        pool_b = get_inner_pool(8)
        assert pool_a is not None
        assert pool_b is not None
        assert pool_a is not pool_b
        # repeated lookups of the same size return the same pool
        assert get_inner_pool(2) is pool_a
        assert get_inner_pool(8) is pool_b

    def test_single_thread_max_threads_is_passthrough(self):
        """max_threads <= 1 returns None — no pool is ever created."""
        assert get_inner_pool(1) is None
        assert get_inner_pool(0) is None


# --- Post-fork pid-guard regression test --------------------------------------
#
# The pid guard in get_inner_pool detects when the calling process has
# changed (i.e. after os.fork()) and rebuilds the registry so we don't
# inherit the parent's ThreadPoolExecutors — whose worker OS threads were not
# copied across fork() and would deadlock on the child's first submit().


def _child_uses_inherited_recording(rec, queue):
    """Child entry point: exercise the parent-inherited recording.

    Under fork, the parent's ``_inner_pools`` registry is copied via fork's
    COW.  Without the pid guard in ``get_inner_pool``, the child's first
    ``submit()`` blocks because the worker threads of the inherited
    ``ThreadPoolExecutor`` don't exist in this process.
    """
    try:
        rec.get_traces_multi_thread(start_frame=0, end_frame=50_000, max_threads=4)
        queue.put("ok")
    except Exception as e:  # pragma: no cover — failure path
        queue.put(f"error: {type(e).__name__}: {e}")


@pytest.mark.skipif(sys.platform == "win32", reason="fork is POSIX-only")
@pytest.mark.parametrize(
    "builder",
    [
        lambda: BandpassFilterRecording(_make_recording(), freq_min=300.0, freq_max=6000.0),
        lambda: CommonReferenceRecording(_make_recording(), operator="median", reference="global"),
    ],
    ids=["filter", "cmr"],
)
def test_pool_recovers_after_fork(builder):
    """After fork, the child must rebuild its inner pool rather than reuse the
    parent's stale one — so ``get_traces_multi_thread`` completes promptly.

    Trigger: the parent pre-warms the pool *before* fork.  Without the pid
    guard in ``get_inner_pool``, the child's first ``submit()`` deadlocks on
    the inherited pool's queue because the parent's worker OS threads were
    not copied across ``fork()``.
    """
    rec = builder()
    rec.get_traces_multi_thread(start_frame=0, end_frame=50_000, max_threads=4)
    parent_pid = os.getpid()
    parent_pool = _pool_for_current_thread(4)
    assert parent_pool is not None, "fixture failed to pre-warm the parent pool"

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child_uses_inherited_recording, args=(rec, queue))
    proc.start()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        pytest.fail(
            "child get_traces_multi_thread() deadlocked after fork: pid guard "
            "in get_inner_pool is missing or broken (parent pre-warmed the pool before fork)"
        )
    result = queue.get_nowait()
    assert result == "ok", f"child failed: {result}"
    assert proc.exitcode == 0, f"child exited non-zero: {proc.exitcode}"

    # Parent's pool is unchanged after the child runs.
    assert os.getpid() == parent_pid
    assert _pool_for_current_thread(4) is parent_pool
