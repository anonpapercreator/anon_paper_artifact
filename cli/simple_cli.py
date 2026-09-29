#!/usr/bin/env python3
"""
HSM Simulator - Simple CLI
A fallback CLI using click for when Textual TUI doesn't work.
"""

import click
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workload.dmf_log_parser import DMFLogParser
from config.models import SimulationConfig
from db.database import ConfigDB, SweepDB, init_db


@click.group()
def cli():
    """HSM Simulator v2.0 - Hierarchical Storage Management"""
    init_db()


@cli.command()
@click.argument('log_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output trace file')
@click.option('--stats', '-s', is_flag=True, help='Show statistics')
def parse(log_file, output, stats):
    """Parse a DMF log file."""
    click.echo(f"Parsing: {log_file}")
    
    parser = DMFLogParser(log_file)
    events = parser.parse_log()
    
    if stats:
        stats_data = parser.get_statistics()
        click.echo(f"\n=== Statistics ===")
        click.echo(f"Total events: {stats_data['total_events']}")
        click.echo(f"Recalls: {stats_data['recall_count']}")
        click.echo(f"Migrations: {stats_data['migrate_count']}")
        click.echo(f"Completed requests: {stats_data['completed_requests']}")
        dur_s = stats_data['duration_seconds']
        click.echo(f"Trace span: {dur_s:.0f}s ({dur_s/3600:.2f}h / {dur_s/86400:.2f}d)")
        click.echo(f"Mean latency: {stats_data['latency_mean']:.2f}s")
        click.echo(f"p50 latency: {stats_data['latency_p50']:.2f}s")
        click.echo(f"p95 latency: {stats_data['latency_p95']:.2f}s")
        click.echo(f"p99 latency: {stats_data['latency_p99']:.2f}s")
        click.echo(f"max latency: {stats_data['latency_max']:.2f}s")
        click.echo("\nNOTE: for trace_replay, set simulation.sim_duration_s >= trace span "
                   "above,\n  or long recalls will be censored and the tail under-counted.")
    
    if output:
        parser.export_normalized(output)
        click.echo(f"\nExported to: {output}")
    
    n_events = events if isinstance(events, int) else len(events)
    click.echo(f"\nParsed {n_events} events ({len(parser.events)} retained)")


@cli.command()
def config():
    """Show current configuration."""
    from config.models import SimulationConfig, ESS3500Config, TS1160Config, DMFPolicyConfig
    
    cfg_db = ConfigDB()
    configs = cfg_db.list_configs()
    
    if not configs:
        click.echo("No saved configurations. Using defaults:")
        cfg = SimulationConfig()
    else:
        c = configs[0]
        cfg = SimulationConfig(
            name=c.name,
            ess3500=ESS3500Config(
                nsd_nodes=c.ess_nodes,
                aggregate_throughput_gib_s=c.ess_throughput_gib_s,
                usable_capacity_tib=c.ess_capacity_tib,
            ),
            ts1160=TS1160Config(
                n_drives=c.ts1160_drives,
            ),
            dmf_policy=DMFPolicyConfig(
                age_threshold_minutes=c.age_threshold_minutes,
                hwm_fraction=c.hwm_fraction,
                lwm_fraction=c.lwm_fraction,
            )
        )
    
    click.echo(f"\n=== Configuration: {cfg.name} ===")
    click.echo(f"ESS 3500:")
    click.echo(f"  NSD Nodes: {cfg.ess3500.nsd_nodes}")
    click.echo(f"  Throughput: {cfg.ess3500.aggregate_throughput_gib_s} GiB/s")
    click.echo(f"  Capacity: {cfg.ess3500.usable_capacity_tib} TiB")
    click.echo(f"\nTS1160:")
    click.echo(f"  Drives: {cfg.ts1160.n_drives}")
    click.echo(f"\nDMF Policy:")
    click.echo(f"  Age Threshold: {cfg.dmf_policy.age_threshold_minutes} minutes")
    click.echo(f"  HWM: {cfg.dmf_policy.hwm_fraction*100:.0f}%")
    click.echo(f"  LWM: {cfg.dmf_policy.lwm_fraction*100:.0f}%")


@cli.command()
def sweeps():
    """List previous sweeps."""
    sweep_db = SweepDB()
    all_sweeps = sweep_db.list_sweeps()
    
    if not all_sweeps:
        click.echo("No sweeps found.")
        return
    
    click.echo(f"\n=== Saved Sweeps ===")
    for s in all_sweeps:
        click.echo(f"ID: {s['id']} | {s['name']} | {s['status']} | Runs: {s['total_runs']}")


@cli.command()
def status():
    """Show simulator status."""
    cfg_db = ConfigDB()
    sweep_db = SweepDB()
    
    configs = cfg_db.list_configs()
    sweeps = sweep_db.list_sweeps()
    
    click.echo("=== HSM Simulator Status ===")
    click.echo(f"Configurations: {len(configs)}")
    click.echo(f"Sweeps: {len(sweeps)}")


if __name__ == '__main__':
    cli()
