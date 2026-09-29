"""
tests/test_fail_fast.py — Replication failure handling.

Phase 3.6 replaces the best-effort ``except Exception: continue`` in the
main CLI with fail-fast semantics: any replication error aborts the run
unless the caller explicitly opts in via ``--allow-partial``. When a
partial run is allowed, the report metadata is flagged so downstream
tooling can suppress or caveat confidence intervals.

These tests drive the behaviour through Click's ``CliRunner`` against a
minimal config, monkeypatching ``run_replication`` so we can inject a
deterministic crash at a chosen replication index.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from stats.collector import ReplicationStats  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default_config.yaml"


def _config_path() -> str:
    """Use the project's default config; Pydantic validation requires more
    sections than a minimal test fixture would hand-roll. The stub
    ``run_replication`` means the config is validated but never executed
    in the simulator — so the only thing that matters is that it loads."""
    assert DEFAULT_CONFIG.exists(), f"missing {DEFAULT_CONFIG}"
    return str(DEFAULT_CONFIG)


def _make_stub_run_replication(fail_at: int):
    """Return a stub ``run_replication`` that succeeds normally but raises
    RuntimeError on the replication index ``fail_at``."""
    def _stub(cfg, replication_id, progress_callback=None):
        if replication_id == fail_at:
            raise RuntimeError(f"injected crash on rep {replication_id}")
        # Minimal stats object — ReplicationStats is safe to build empty.
        return ReplicationStats(replication_id=replication_id, warmup_s=0.0)
    return _stub


def test_default_behaviour_is_fail_fast(tmp_path, monkeypatch):
    """With no --allow-partial, the first replication crash exits the CLI."""
    cfg_path = _config_path()
    out_dir = tmp_path / "results"
    monkeypatch.setattr(main, "run_replication",
                        _make_stub_run_replication(fail_at=1))
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        ["run",
         "--config", cfg_path,
         "--output-dir", str(out_dir),
         "--replications", "3",
         "--duration", "3600"],   # > warmup_s in default_config.yaml
    )
    assert result.exit_code == 1, (
        f"Expected non-zero exit; got {result.exit_code}\n{result.output}"
    )
    assert "Aborting" in result.output
    assert "--allow-partial" in result.output
    # A report must not be written when the run aborts.
    assert not (out_dir / "simulation_report.json").exists()


def test_allow_partial_continues_and_flags_report(tmp_path, monkeypatch):
    """With --allow-partial, a single crash is recorded in metadata and
    the run proceeds to emit a (flagged) report."""
    cfg_path = _config_path()
    out_dir = tmp_path / "results"
    monkeypatch.setattr(main, "run_replication",
                        _make_stub_run_replication(fail_at=1))
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        ["run",
         "--config", cfg_path,
         "--output-dir", str(out_dir),
         "--replications", "3",
         "--duration", "3600",
         "--allow-partial"],
    )
    assert result.exit_code == 0, (
        f"Expected success; got {result.exit_code}\n{result.output}"
    )
    report_path = out_dir / "simulation_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    meta = report["metadata"]
    assert meta["partial_run"] is True
    assert meta["n_successful"] == 2
    assert meta["n_replications"] == 3
    assert len(meta["failed_replications"]) == 1
    failed = meta["failed_replications"][0]
    assert failed["replication"] == 1
    assert failed["error_type"] == "RuntimeError"
    assert "ci_warning" in meta


def test_no_failures_means_no_partial_flag(tmp_path, monkeypatch):
    """When every replication succeeds the report must not be marked
    partial — even if --allow-partial was supplied."""
    cfg_path = _config_path()
    out_dir = tmp_path / "results"
    monkeypatch.setattr(main, "run_replication",
                        _make_stub_run_replication(fail_at=-1))  # never fires
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        ["run",
         "--config", cfg_path,
         "--output-dir", str(out_dir),
         "--replications", "2",
         "--duration", "3600",
         "--allow-partial"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out_dir / "simulation_report.json").read_text())
    meta = report["metadata"]
    assert meta["partial_run"] is False
    assert meta["failed_replications"] == []
    assert meta["n_successful"] == 2
    assert "ci_warning" not in meta
