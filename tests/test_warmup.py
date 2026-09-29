"""
tests/test_warmup.py — Warm-up censoring is applied uniformly across collectors.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats.collector import QueueStats, LatencyHistogram, ReplicationStats


def test_queue_stats_censors_pre_warmup_events():
    """Arrivals and sojourns before warmup must not contribute to L, λ, W."""
    q = QueueStats("test", warmup_s=100.0)
    # Pre-warmup traffic: one long-lived item
    q.arrival(time=10.0)
    q.departure(time=20.0, arrival_time=10.0)
    # Post-warmup traffic
    q.arrival(time=150.0)
    q.departure(time=160.0, arrival_time=150.0)
    q.finalise(sim_end_time=200.0)

    # Only the post-warmup item should appear in counters.
    assert q._n_arrivals == 1
    assert len(q._sojourn_times) == 1
    assert q._sojourn_times[0] == pytest.approx(10.0)

    # L is averaged over [100, 200] = 100 s, with one unit present for 10 s.
    assert q.mean_L() == pytest.approx(10.0 / 100.0, rel=1e-9)

    # λ is 1 arrival in 100 s.
    assert q.lambda_hat() == pytest.approx(1.0 / 100.0, rel=1e-9)


def test_queue_stats_straddling_arrival_is_censored():
    """An arrival before warmup paired with a departure after must be dropped."""
    q = QueueStats("test", warmup_s=100.0)
    q.arrival(time=50.0)           # pre-warmup arrival, still tracked for length
    q.departure(time=150.0, arrival_time=50.0)  # straddles warmup
    q.finalise(sim_end_time=200.0)
    assert q._sojourn_times == []  # sojourn censored because arrival pre-warmup
    # Queue length starts at 1 at t=100 and drops to 0 at t=150 ⇒ area = 50.
    assert q.mean_L() == pytest.approx(50.0 / 100.0, rel=1e-9)


def test_latency_histogram_drops_pre_warmup_samples():
    h = LatencyHistogram("lat", warmup_s=50.0)
    h.record(1.0, time=10.0)   # dropped
    h.record(2.0, time=49.9)   # dropped
    h.record(3.0, time=50.0)   # kept (>= warmup)
    h.record(4.0, time=200.0)  # kept
    h.record(5.0)              # kept (no time ⇒ legacy callers preserved)
    assert h._samples == [3.0, 4.0, 5.0]


def test_replication_stats_threads_warmup():
    rep = ReplicationStats(replication_id=0, warmup_s=100.0)
    # Sanity: the nested collectors inherit warmup_s.
    for q in rep.queues.values():
        assert q._warmup_s == 100.0
    assert rep.recall_latency._warmup_s == 100.0
    assert rep.migrate_latency._warmup_s == 100.0
    assert rep.mount_latency._warmup_s == 100.0

    # record_recall_latency respects warmup when a time is passed.
    rep.record_recall_latency(1.0, time=50.0)   # dropped
    rep.record_recall_latency(2.0, time=150.0)  # kept
    assert rep.recall_latency._samples == [2.0]
    assert rep.n_recalls == 1


def test_warmup_zero_is_neutral():
    """warmup_s=0 must behave identically to the un-censored classic path."""
    q_old_like = QueueStats("q", warmup_s=0.0)
    q_old_like.arrival(time=0.0)
    q_old_like.arrival(time=1.0)
    q_old_like.departure(time=2.0, arrival_time=0.0)
    q_old_like.departure(time=3.0, arrival_time=1.0)
    q_old_like.finalise(sim_end_time=10.0)
    assert q_old_like._n_arrivals == 2
    assert len(q_old_like._sojourn_times) == 2
