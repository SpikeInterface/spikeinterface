"""Tests for the per-caller-thread pool semantics used by FilterRecording and
CommonReferenceRecording when ``n_workers > 1``.

Contract: each outer thread that calls ``get_traces()`` on a parallel-enabled
segment gets its own inner ThreadPoolExecutor.  Keying by thread avoids the
shared-pool queueing pathology that arises when many outer workers submit
concurrently into a single inner pool with fewer max_workers than outer
callers.  See the module-level comments in filter.py and common_reference.py
for the full rationale.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    CommonReferenceRecording,
)


def _make_recording(T: int = 50_000, C: int = 64, fs: float = 30_000.0):
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((T, C)).astype(np.float32) * 100.0
    return NumpyRecording([traces], sampling_frequency=fs)


@pytest.fixture
def filter_segment():
    rec = _make_recording()
    bp = BandpassFilterRecording(rec, freq_min=300.0, freq_max=6000.0, n_workers=4)
    return bp, bp._recording_segments[0]


@pytest.fixture
def cmr_segment():
    rec = _make_recording()
    cmr = CommonReferenceRecording(rec, operator="median", reference="global", n_workers=4)
    return cmr, cmr._recording_segments[0]


class TestPerCallerThreadPool:
    """Verify each calling thread gets its own inner pool."""

    @pytest.mark.parametrize(
        "segment_fixture,pools_attr",
        [
            ("filter_segment", "_filter_pools"),
            ("cmr_segment", "_cmr_pools"),
        ],
    )
    def test_single_caller_reuses_pool(self, segment_fixture, pools_attr, request):
        """Repeated calls from the same thread reuse the same inner pool."""
        rec, seg = request.getfixturevalue(segment_fixture)
        rec.get_traces(start_frame=0, end_frame=50_000)
        pool_a = getattr(seg, pools_attr).get(threading.current_thread())
        rec.get_traces()
        pool_b = getattr(seg, pools_attr).get(threading.current_thread())
        assert pool_a is not None
        assert pool_a is pool_b, "expected the same inner pool to be reused across calls from the same thread"

    @pytest.mark.parametrize(
        "segment_fixture,pools_attr",
        [
            ("filter_segment", "_filter_pools"),
            ("cmr_segment", "_cmr_pools"),
        ],
    )
    def test_concurrent_callers_get_distinct_pools(self, segment_fixture, pools_attr, request):
        """Two outer threads calling get_traces concurrently must receive
        different inner pools — not a shared one that would queue their
        tasks through a single bottleneck.
        """
        rec, seg = request.getfixturevalue(segment_fixture)

        ready = threading.Barrier(2)
        captured = {}

        def worker(name):
            # Align the two threads so they're definitely live concurrently
            # when they touch the pool-map, exercising the double-checked
            # locking path.
            ready.wait()
            rec.get_traces(start_frame=0, end_frame=50_000)
            captured[name] = getattr(seg, pools_attr).get(threading.current_thread())

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert captured["t1"] is not None
        assert captured["t2"] is not None
        assert captured["t1"] is not captured["t2"], (
            "expected distinct inner pools for concurrent callers; Model 1 "
            "shared-pool semantics would cause queueing pathology"
        )
