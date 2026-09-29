"""
Migration Script: Add archiver_interval_minutes to sweep_results
================================================================
Adds the archiver_interval_minutes column to support archiver interval parameter sweeps.

Usage:
    python migrations/add_archiver_interval.py
"""

import sqlite3
from pathlib import Path

def migrate(db_path: Path = Path.home() / ".hsm_sim" / "hsm_sim.db"):
    """Add archiver_interval_minutes column to sweep_results table."""
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if column already exists
    cursor.execute("PRAGMA table_info(sweep_results)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "archiver_interval_minutes" in columns:
        print("Column 'archiver_interval_minutes' already exists. No migration needed.")
        conn.close()
        return
    
    # Add column
    cursor.execute("ALTER TABLE sweep_results ADD COLUMN archiver_interval_minutes INTEGER")
    conn.commit()
    conn.close()
    
    print("Migration complete: Added 'archiver_interval_minutes' column to sweep_results")

if __name__ == "__main__":
    migrate()
