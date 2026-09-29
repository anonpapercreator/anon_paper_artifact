"""
core/sweeper.py — Parameter Sweep Runner (Unified Path)
=========================================================
Executes parameter sweeps using the component-based DES via main.run_replication().

The legacy HSMSimulator path has been removed; all sweeps now use the full-fidelity
component architecture (Filesystem + TapeSubsystem + AdaptivePolicyAdvisor).
"""
from __future__ import annotations
import time
import os
from typing import List, Dict, Any, Optional, Callable
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

from main import run_replication, load_config
from stats.converter import replication_stats_to_simulation_result
from db.database import SweepDB, ConfigDB


def get_max_workers() -> int:
    """Calculate maximum workers based on available RAM (~2 GB per process)."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        max_workers = max(1, int(available_gb / 2))
        return min(max_workers, os.cpu_count() or 4)
    except ImportError:
        return min(4, os.cpu_count() or 4)


def _run_single_simulation(args: tuple) -> Dict[str, Any]:
    """
    Run a single simulation in a worker process.

    Args:
        args: Tuple of (config_dict, combo, rep, seed)

    Returns:
        Flat dict result (SimulationResult-compatible).
    """
    config_dict, combo, rep, seed = args

    # Apply combo parameters to config
    cfg = copy.deepcopy(config_dict)
    for param, value in combo.items():
        _apply_param(cfg, param, value)
    cfg["simulation"]["seed"] = seed

    # Run the component-based DES
    rep_stats = run_replication(cfg, replication_id=rep)

    # Convert to dict format expected by DB / web UI
    result = replication_stats_to_simulation_result(
        rep_stats, cfg, replication_id=rep, wall_time_s=0.0
    )
    result["replication"] = rep
    return result


def _apply_param(cfg: Dict[str, Any], param: str, value: Any) -> None:
    """Apply a single sweep parameter to the config dict."""
    if param == "age_threshold_minutes":
        cfg["policy_engine"]["candidate_age_threshold_s"] = value * 60
    elif param == "age_threshold_seconds":
        cfg["policy_engine"]["candidate_age_threshold_s"] = value
    elif param == "hwm_fraction":
        cfg["disk_cache"]["hwm_fraction"] = value
    elif param == "lwm_fraction":
        cfg["disk_cache"]["lwm_fraction"] = value
    elif param == "cache_priority":
        # Cache priority maps to adaptive policy baseline
        if "adaptive_policy" not in cfg:
            cfg["adaptive_policy"] = {}
        base_age = 30 * 24 * 3600  # 30 days default
        derived_age = int(base_age * (value**0.5))
        cfg["adaptive_policy"]["baseline_age_threshold_s"] = derived_age
    elif param == "archiver_interval_minutes":
        cfg["policy_engine"]["cycle_period_s"] = value * 60
    elif param == "archiver_interval_seconds":
        cfg["policy_engine"]["cycle_period_s"] = value
    else:
        # Generic: try to find in known sections
        for section in ["simulation", "disk_cache", "tape_drives", "policy_engine", "workload", "adaptive_policy"]:
            if param in cfg.get(section, {}):
                cfg[section][param] = value
                break


class SweepRunner:
    """Sequential parameter sweep runner using the unified component DES."""

    def __init__(self, base_config: Dict[str, Any], sweep_params: Dict[str, List[Any]]):
        self.base_config = base_config
        self.sweep_params = sweep_params
        self.sweep_db = SweepDB()
        self.config_db = ConfigDB()

    def run(
        self,
        replications: int = 30,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run parameter sweep sequentially.

        Args:
            replications: Number of replications per parameter combination.
            progress_callback: Called with (progress, current, total, combo, rep, result).

        Returns:
            List of result dicts, one per run.
        """
        from itertools import product

        keys = list(self.sweep_params.keys())
        values = list(self.sweep_params.values())
        combinations = [dict(zip(keys, combo)) for combo in product(*values)]

        total_runs = len(combinations) * replications

        sweep_id = self.sweep_db.create_sweep(
            name="parameter_sweep",
            description="Unified DES sweep",
            params_swept=self.sweep_params,
        )
        self.sweep_db.update_sweep_status(sweep_id, "running")

        all_results: List[Dict[str, Any]] = []
        current_run = 0

        try:
            for combo_idx, combo in enumerate(combinations):
                for rep in range(replications):
                    seed = self.base_config["simulation"]["seed"] + combo_idx * 1000 + rep
                    result = _run_single_simulation(
                        (self.base_config, combo, rep, seed)
                    )

                    self._save_result(sweep_id, result)
                    all_results.append(result)

                    current_run += 1
                    if progress_callback:
                        progress_callback(
                            progress=current_run / total_runs,
                            current=current_run,
                            total=total_runs,
                            combo=combo,
                            rep=rep,
                            result=result,
                        )

            self.sweep_db.update_sweep_status(sweep_id, "completed")
        except Exception:
            self.sweep_db.update_sweep_status(sweep_id, "failed")
            raise

        return all_results

    def _save_result(self, sweep_id: int, result: Dict[str, Any]) -> None:
        """Save result to database."""
        config_id = self.config_db.save_config(result)
        self.sweep_db.save_result(sweep_id, config_id, result)


class SweepRunnerParallel:
    """Parallel parameter sweep runner using ProcessPoolExecutor."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        sweep_params: Dict[str, List[Any]],
        n_workers: int = 4,
    ):
        self.base_config = base_config
        self.sweep_params = sweep_params
        self.n_workers = min(n_workers, get_max_workers())
        self.sweep_db = SweepDB()
        self.config_db = ConfigDB()

    def run(
        self,
        replications: int = 30,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Run parameter sweep in parallel."""
        from itertools import product

        keys = list(self.sweep_params.keys())
        values = list(self.sweep_params.values())
        combinations = [dict(zip(keys, combo)) for combo in product(*values)]

        total_runs = len(combinations) * replications

        sweep_id = self.sweep_db.create_sweep(
            name="parameter_sweep_parallel",
            description="Unified DES parallel sweep",
            params_swept=self.sweep_params,
        )
        self.sweep_db.update_sweep_status(sweep_id, "running")

        all_runs = []
        for combo_idx, combo in enumerate(combinations):
            for rep in range(replications):
                seed = (
                    self.base_config["simulation"]["seed"] + combo_idx * 1000 + rep
                )
                all_runs.append((self.base_config, combo, rep, seed))

        all_results: List[Dict[str, Any]] = []
        completed = 0

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_idx = {
                    executor.submit(_run_single_simulation, run_args): idx
                    for idx, run_args in enumerate(all_runs)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    combo_idx = idx // replications
                    rep = idx % replications

                    try:
                        result = future.result()
                        self._save_result(sweep_id, result)
                        all_results.append(result)
                        completed += 1

                        if progress_callback:
                            progress_callback(
                                progress=completed / total_runs,
                                current=completed,
                                total=total_runs,
                                worker_id=idx % self.n_workers,
                                combo=combinations[combo_idx],
                                rep=rep,
                                result=result,
                            )
                    except Exception as e:
                        print(f"Error in worker {idx}: {e}")

            self.sweep_db.update_sweep_status(sweep_id, "completed")
        except Exception:
            self.sweep_db.update_sweep_status(sweep_id, "failed")
            raise

        return all_results

    def _save_result(self, sweep_id: int, result: Dict[str, Any]) -> None:
        config_id = self.config_db.save_config(result)
        self.sweep_db.save_result(sweep_id, config_id, result)


def run_sweep(
    base_config: Dict[str, Any],
    sweep_params: Dict[str, List[Any]],
    replications: int = 30,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for sequential parameter sweep."""
    runner = SweepRunner(base_config, sweep_params)
    return runner.run(replications=replications, progress_callback=progress_callback)
