"""
Database Module for HSM Simulator
================================
SQLite database for storing configurations and sweep results.
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import threading


DB_PATH = Path.home() / ".hsm_sim" / "hsm_sim.db"

_connection_lock = threading.Lock()
_connections = {}


def get_db_path() -> Path:
    """Get or create database path."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get thread-safe database connection."""
    thread_id = threading.get_ident()
    
    if thread_id not in _connections:
        with _connection_lock:
            conn = sqlite3.connect(str(get_db_path()), timeout=30.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            _connections[thread_id] = conn
    
    return _connections[thread_id]


def close_connection():
    """Close connection for current thread."""
    thread_id = threading.get_ident()
    if thread_id in _connections:
        _connections[thread_id].close()
        del _connections[thread_id]


def init_db() -> sqlite3.Connection:
    """Initialize database with schema."""
    conn = sqlite3.connect(str(get_db_path()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    
    # Create tables
    conn.executescript("""
        -- Configuration profiles
        CREATE TABLE IF NOT EXISTS configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            
            -- ESS 3500
            ess_nodes INTEGER DEFAULT 4,
            ess_throughput_gib_s REAL DEFAULT 65.0,
            ess_capacity_tib REAL DEFAULT 809.0,
            network_type TEXT DEFAULT 'hdr_infiniband',
            
            -- TS1160
            ts1160_drives INTEGER DEFAULT 16,
            
            -- DMF Policy
            age_threshold_minutes INTEGER DEFAULT 30,
            hwm_fraction REAL DEFAULT 0.80,
            lwm_fraction REAL DEFAULT 0.70,
            large_file_threshold_gb INTEGER DEFAULT 100,
            
            -- Declarative Policy
            cache_priority REAL DEFAULT 0.5,
            policy_preset TEXT DEFAULT 'balanced',
            prefetch_enabled INTEGER DEFAULT 1,
            
            -- Cost
            power_cost_per_kw_month REAL DEFAULT 427.0,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Sweep metadata
        CREATE TABLE IF NOT EXISTS sweeps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            parameters_swept TEXT NOT NULL,  -- JSON
            total_runs INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
        
        -- Sweep results
        CREATE TABLE IF NOT EXISTS sweep_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sweep_id INTEGER NOT NULL,
            config_id INTEGER,
            replication INTEGER NOT NULL,
            
            -- Performance Metrics
            cache_hit_ratio REAL,
            latency_p50 REAL,
            latency_p95 REAL,
            latency_p99 REAL,
            latency_mean REAL,
            tape_mounts INTEGER,
            bytes_migrated INTEGER,
            total_cost REAL,
            cost_per_recall REAL,
            
            -- NEW: Media Consumption
            archived_data_tb REAL,
            tape_cartridges_used REAL,
            tape_capacity_utilization_pct REAL,
            
            -- NEW: Bandwidth
            migration_bandwidth_gbs REAL,
            scratch_effective_bandwidth_gbs REAL,
            scratch_bandwidth_degradation_pct REAL,
            
            -- NEW: Cache
            cache_resident_tb REAL,
            cache_churn_tb REAL,
            
            -- Parameters used
            age_threshold_minutes INTEGER,
            hwm_fraction REAL,
            lwm_fraction REAL,
            cache_priority REAL,
            archiver_interval_minutes INTEGER,
            
            -- Run info
            run_time_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- NEW: IOR Benchmark Metrics
            ior_config_name TEXT,
            ior_iteration INTEGER,
            total_bytes_written INTEGER,
            total_bytes_read INTEGER,
            write_operations INTEGER,
            read_operations INTEGER,
            write_duration_s REAL,
            read_duration_s REAL,
            
            FOREIGN KEY (sweep_id) REFERENCES sweeps(id),
            FOREIGN KEY (config_id) REFERENCES configs(id)
        );
        
        -- Create indices
        CREATE INDEX IF NOT EXISTS idx_results_sweep ON sweep_results(sweep_id);
        CREATE INDEX IF NOT EXISTS idx_results_config ON sweep_results(config_id);
        CREATE INDEX IF NOT EXISTS idx_sweeps_status ON sweeps(status);
    """)
    
    conn.commit()
    return conn


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    id: Optional[int] = None
    name: str = "default"
    description: str = ""
    
    # ESS 3500
    ess_nodes: int = 4
    ess_throughput_gib_s: float = 65.0
    ess_capacity_tib: float = 809.0
    network_type: str = "hdr_infiniband"
    
    # TS1160
    ts1160_drives: int = 16
    
    # DMF Policy
    age_threshold_minutes: int = 30
    hwm_fraction: float = 0.80
    lwm_fraction: float = 0.70
    large_file_threshold_gb: int = 100
    
    # Declarative Policy
    cache_priority: float = 0.5
    policy_preset: str = "balanced"
    prefetch_enabled: bool = True
    
    # Cost
    power_cost_per_kw_month: float = 427.0
    
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ess_nodes": self.ess_nodes,
            "ess_throughput_gib_s": self.ess_throughput_gib_s,
            "ess_capacity_tib": self.ess_capacity_tib,
            "network_type": self.network_type,
            "ts1160_drives": self.ts1160_drives,
            "age_threshold_minutes": self.age_threshold_minutes,
            "hwm_fraction": self.hwm_fraction,
            "lwm_fraction": self.lwm_fraction,
            "large_file_threshold_gb": self.large_file_threshold_gb,
            "cache_priority": self.cache_priority,
            "policy_preset": self.policy_preset,
            "prefetch_enabled": self.prefetch_enabled,
            "power_cost_per_kw_month": self.power_cost_per_kw_month,
        }
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SimulationConfig:
        """Create from database row."""
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            ess_nodes=row["ess_nodes"],
            ess_throughput_gib_s=row["ess_throughput_gib_s"],
            ess_capacity_tib=row["ess_capacity_tib"],
            network_type=row["network_type"],
            ts1160_drives=row["ts1160_drives"],
            age_threshold_minutes=row["age_threshold_minutes"],
            hwm_fraction=row["hwm_fraction"],
            lwm_fraction=row["lwm_fraction"],
            large_file_threshold_gb=row["large_file_threshold_gb"],
            cache_priority=row["cache_priority"],
            policy_preset=row["policy_preset"],
            prefetch_enabled=bool(row["prefetch_enabled"]),
            power_cost_per_kw_month=row["power_cost_per_kw_month"],
            created_at=row["created_at"],
        )


class ConfigDB:
    """Database operations for configurations."""
    
    def __init__(self):
        self.conn = get_connection()
    
    def save_config(self, config) -> int:
        """Save configuration, return ID. Accepts SimulationConfig object or dict."""
        # Handle dict input (from to_db_dict())
        if isinstance(config, dict):
            config_name = config.get('name', 'default')
            
            # Check if config with this name already exists
            existing = self.conn.execute(
                "SELECT id FROM configs WHERE name = ?", (config_name,)
            ).fetchone()
            
            if existing:
                return existing['id']
            
            # Insert new config
            cursor = self.conn.execute("""
                INSERT INTO configs (
                    name, description,
                    ess_nodes, ess_throughput_gib_s, ess_capacity_tib,
                    network_type, ts1160_drives,
                    age_threshold_minutes, hwm_fraction, lwm_fraction,
                    large_file_threshold_gb, cache_priority, policy_preset,
                    prefetch_enabled, power_cost_per_kw_month
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.get('name'), config.get('description'),
                config.get('ess_nodes'), config.get('ess_throughput_gib_s'), config.get('ess_capacity_tib'),
                config.get('network_type'), config.get('ts1160_drives'),
                config.get('age_threshold_minutes'), config.get('hwm_fraction'), config.get('lwm_fraction'),
                config.get('large_file_threshold_gb'), config.get('cache_priority'), config.get('policy_preset'),
                int(config.get('prefetch_enabled', 1)), config.get('power_cost_per_kw_month')
            ))
            self.conn.commit()
            return cursor.lastrowid if cursor.lastrowid else 0
        
        # Handle SimulationConfig object (original behavior)
        if config.id:
            # Update existing
            self.conn.execute("""
                UPDATE configs SET
                    name = ?, description = ?,
                    ess_nodes = ?, ess_throughput_gib_s = ?, ess_capacity_tib = ?,
                    network_type = ?, ts1160_drives = ?,
                    age_threshold_minutes = ?, hwm_fraction = ?, lwm_fraction = ?,
                    large_file_threshold_gb = ?, cache_priority = ?, policy_preset = ?,
                    prefetch_enabled = ?, power_cost_per_kw_month = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                config.name, config.description,
                config.ess_nodes, config.ess_throughput_gib_s, config.ess_capacity_tib,
                config.network_type, config.ts1160_drives,
                config.age_threshold_minutes, config.hwm_fraction, config.lwm_fraction,
                config.large_file_threshold_gb, config.cache_priority, config.policy_preset,
                int(config.prefetch_enabled), config.power_cost_per_kw_month,
                config.id
            ))
            self.conn.commit()
            return config.id
        else:
            # Check if exists by name
            existing = self.conn.execute(
                "SELECT id FROM configs WHERE name = ?", (config.name,)
            ).fetchone()
            if existing:
                return existing['id']
            
            # Insert new
            cursor = self.conn.execute("""
                INSERT INTO configs (
                    name, description,
                    ess_nodes, ess_throughput_gib_s, ess_capacity_tib,
                    network_type, ts1160_drives,
                    age_threshold_minutes, hwm_fraction, lwm_fraction,
                    large_file_threshold_gb, cache_priority, policy_preset,
                    prefetch_enabled, power_cost_per_kw_month
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.name, config.description,
                config.ess_nodes, config.ess_throughput_gib_s, config.ess_capacity_tib,
                config.network_type, config.ts1160_drives,
                config.age_threshold_minutes, config.hwm_fraction, config.lwm_fraction,
                config.large_file_threshold_gb, config.cache_priority, config.policy_preset,
                int(config.prefetch_enabled), config.power_cost_per_kw_month,
            ))
            self.conn.commit()
            return cursor.lastrowid if cursor.lastrowid else 0
    
    def load_config(self, name: str) -> Optional[SimulationConfig]:
        """Load configuration by name."""
        row = self.conn.execute(
            "SELECT * FROM configs WHERE name = ?", (name,)
        ).fetchone()
        
        if row:
            return SimulationConfig.from_row(row)
        return None
    
    def list_configs(self) -> List[SimulationConfig]:
        """List all configurations."""
        rows = self.conn.execute(
            "SELECT * FROM configs ORDER BY updated_at DESC"
        ).fetchall()
        return [SimulationConfig.from_row(row) for row in rows]
    
    def delete_config(self, name: str) -> bool:
        """Delete configuration by name."""
        cursor = self.conn.execute(
            "DELETE FROM configs WHERE name = ?", (name,)
        )
        self.conn.commit()
        return cursor.rowcount > 0


class SweepDB:
    """Database operations for sweeps."""
    
    def __init__(self):
        self.conn = get_connection()
    
    def create_sweep(self, name: str, description: str, 
                    params_swept: Dict[str, List[Any]]) -> int:
        """Create new sweep."""
        cursor = self.conn.execute("""
            INSERT INTO sweeps (name, description, parameters_swept, status)
            VALUES (?, ?, ?, 'pending')
        """, (name, description, json.dumps(params_swept)))
        self.conn.commit()
        return cursor.lastrowid or 0
    
    def update_sweep_status(self, sweep_id: int, status: str):
        """Update sweep status."""
        if status == "completed":
            self.conn.execute("""
                UPDATE sweeps SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, sweep_id))
        else:
            self.conn.execute("""
                UPDATE sweeps SET status = ? WHERE id = ?
            """, (status, sweep_id))
        self.conn.commit()
    
    def save_result(self, sweep_id: int, config_id: Optional[int],
                   result: Dict[str, Any]) -> int:
        """Save sweep result."""
        cursor = self.conn.execute("""
            INSERT INTO sweep_results (
                sweep_id, config_id, replication,
                cache_hit_ratio, latency_p50, latency_p95, latency_p99, latency_mean,
                tape_mounts, bytes_migrated, total_cost, cost_per_recall,
                archived_data_tb, tape_cartridges_used, tape_capacity_utilization_pct,
                migration_bandwidth_gbs, scratch_effective_bandwidth_gbs, scratch_bandwidth_degradation_pct,
                cache_resident_tb, cache_churn_tb,
                age_threshold_minutes, hwm_fraction, lwm_fraction, cache_priority,
                archiver_interval_minutes,
                run_time_seconds,
                ior_config_name, ior_iteration, total_bytes_written, total_bytes_read,
                write_operations, read_operations, write_duration_s, read_duration_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sweep_id, config_id, result.get("replication", 0),
            result.get("cache_hit_ratio"),
            result.get("latency_p50"), result.get("latency_p95"), 
            result.get("latency_p99"), result.get("latency_mean"),
            result.get("tape_mounts"), result.get("bytes_migrated"),
            result.get("total_cost"), result.get("cost_per_recall"),
            # NEW: Media Consumption
            result.get("archived_data_tb"),
            result.get("tape_cartridges_used"),
            result.get("tape_capacity_utilization_pct"),
            # NEW: Bandwidth
            result.get("migration_bandwidth_gbs"),
            result.get("scratch_effective_bandwidth_gbs"),
            result.get("scratch_bandwidth_degradation_pct"),
            # NEW: Cache
            result.get("cache_resident_tb"),
            result.get("cache_churn_tb"),
            # Parameters
            result.get("age_threshold_minutes"),
            result.get("hwm_fraction"), result.get("lwm_fraction"),
            result.get("cache_priority"),
            result.get("archiver_interval_minutes"),
            result.get("run_time_seconds"),
            # NEW: IOR Benchmark Metrics
            result.get("ior_config_name"),
            result.get("ior_iteration"),
            result.get("total_bytes_written"),
            result.get("total_bytes_read"),
            result.get("write_operations"),
            result.get("read_operations"),
            result.get("write_duration_s"),
            result.get("read_duration_s")
        ))
        self.conn.commit()
        
        # Update total runs
        self.conn.execute("""
            UPDATE sweeps SET total_runs = total_runs + 1 WHERE id = ?
        """, (sweep_id,))
        self.conn.commit()
        
        return cursor.lastrowid or 0
    
    def get_all_results(self) -> List[Dict]:
        """Get all results from all sweeps."""
        rows = self.conn.execute("""
            SELECT * FROM sweep_results ORDER BY sweep_id, replication
        """).fetchall()
        return [dict(row) for row in rows]
    
    def get_sweep_results(self, sweep_id: int) -> List[Dict]:
        """Get all results for a sweep."""
        rows = self.conn.execute("""
            SELECT * FROM sweep_results WHERE sweep_id = ? ORDER BY replication
        """, (sweep_id,)).fetchall()
        return [dict(row) for row in rows]
    
    def get_sweep_summary(self, sweep_id: int) -> Dict:
        """Get aggregated summary for a sweep."""
        row = self.conn.execute("""
            SELECT 
                COUNT(*) as n_runs,
                AVG(cache_hit_ratio) as avg_hit_ratio,
                AVG(latency_p50) as avg_latency_p50,
                AVG(latency_p95) as avg_latency_p95,
                AVG(cost_per_recall) as avg_cost_per_recall,
                AVG(tape_mounts) as avg_tape_mounts
            FROM sweep_results WHERE sweep_id = ?
        """, (sweep_id,)).fetchone()
        
        return dict(row) if row else {}
    
    def list_sweeps(self) -> List[Dict]:
        """List all sweeps."""
        rows = self.conn.execute("""
            SELECT * FROM sweeps ORDER BY created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


# Initialize on import
init_db()
