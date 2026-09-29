"""
DMF 7 Log Parser
================
Parses raw DMF ls-agent logs into normalized trace format for simulation.

Log Format (from your actual DMF system):
- Header line with column names
- Events: "Request Started" (GetFile/PutFile) and "Request Completed"
- JSON embedded in message field

Output: Normalized CSV/JSONL with fields required by DMFTraceReader

Author: Simulation Team
"""

from __future__ import annotations
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys


class DMFLogParser:
    """
    Parses raw DMF 7 ls-agent logs into normalized trace format.
    
    Input: Raw log from <dmf-ls-log>
    Output: Normalized events for simulation replay
    """
    
    # Regex patterns for extracting events
    PATTERN_STARTED = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})\s+'
        r'[\d.]+\s+\S+\s+\S+\s+\d+\s+\d+\s+(\w+)\s+'
        r'---------- Request Started, JobId=([^,]+), request=(\{.+\})'
    )
    
    PATTERN_COMPLETED = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})\s+'
        r'[\d.]+\s+\S+\s+\S+\s+\d+\s+\d+\s+(\w+)\s+'
        r'---------- Request Completed, JobId=([^,]+), response=(\{.+\})'
    )
    
    # Map DMF operations to normalized event types
    OPERATION_MAP = {
        'GetFile': 'RECALL',    # Tape -> Disk
        'PutFile': 'MIGRATE',   # Disk -> Tape
    }
    
    def __init__(self, log_path: str = None):
        self.log_path = Path(log_path) if log_path else None
        if log_path and not Path(log_path).exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        self.events: List[Dict[str, Any]] = []
        self.latencies: Dict[str, float] = {}  # job_id -> latency_seconds
        self._started_requests: Dict[str, Dict] = {}  # job_id -> request data
    
    def scan_directory(self, directory: str) -> List[str]:
        """
        Scan directory for DMF log files (ls.*.log pattern).
        
        Args:
            directory: Path to directory containing log files
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = sorted(dir_path.glob("ls.*.log"))
        return [str(f) for f in files]
    
    def parse_directory(self, directory: str) -> int:
        """
        Parse all DMF log files in a directory.
        
        Args:
            directory: Path to directory containing log files
            
        Returns:
            Total number of events parsed
        """
        files = self.scan_directory(directory)
        
        total_events = 0
        for file_path in files:
            self.log_path = Path(file_path)
            count = self._parse_single_file()
            total_events += count
        
        return total_events
        
    def _parse_single_file(self) -> int:
        """Parse a single log file."""
        events_before = len(self.events)
        
        with open(self.log_path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                    
                match_started = self.PATTERN_STARTED.search(line)
                if match_started:
                    self._handle_started(match_started, line_no)
                    continue
                    
                match_completed = self.PATTERN_COMPLETED.search(line)
                if match_completed:
                    self._handle_completed(match_completed, line_no)
        
        return len(self.events) - events_before
    
    def parse_timestamp(self, ts_str: str) -> float:
        """Convert ISO timestamp to seconds since epoch."""
        dt = datetime.fromisoformat(ts_str.replace('+10:00', '+10:00'))
        return dt.timestamp()
    
    def parse_log(self, log_path: str = None) -> List[Dict[str, Any]]:
        """Parse log file or directory."""
        if log_path:
            self.log_path = Path(log_path)
        
        if not self.log_path:
            raise ValueError("No log path specified")
        
        if self.log_path.is_dir():
            return self.parse_directory(str(self.log_path))
        
        return self._parse_single_file()
    
    def _parse_log_legacy(self) -> List[Dict[str, Any]]:
        """Parse the entire log file (legacy method)."""
        print(f"Parsing DMF log: {self.log_path}")
        
        with open(self.log_path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                # Skip header lines
                if line.startswith('#'):
                    continue
                    
                # Try to match "Request Started"
                match_started = self.PATTERN_STARTED.search(line)
                if match_started:
                    self._handle_started(match_started, line_no)
                    continue
                    
                # Try to match "Request Completed"
                match_completed = self.PATTERN_COMPLETED.search(line)
                if match_completed:
                    self._handle_completed(match_completed, line_no)
                    
        print(f"Parsed {len(self.events)} events")
        print(f"Found {len(self.latencies)} completed requests with latency")
        return self.events
    
    def _handle_started(self, match: re.Match, line_no: int):
        """Handle Request Started event."""
        ts_str, level, job_id, request_json_str = match.groups()
        
        try:
            request_data = json.loads(request_json_str)
        except json.JSONDecodeError:
            return
            
        operation = request_data.get('operation', '')
        if operation not in self.OPERATION_MAP:
            return
            
        # Store for later latency calculation
        self._started_requests[job_id] = {
            'timestamp': self.parse_timestamp(ts_str),
            'operation': operation,
            'file_id': request_data.get('key', job_id),
            'file_size_bytes': request_data.get('length', 0),
            'vol_grp': request_data.get('volGrp', 'unknown'),
            'fs_type': request_data.get('fsType', 'unknown'),
            'mount_point': request_data.get('mountPoint', '/'),
        }
        
        # Create event record
        event = {
            'timestamp_s': self.parse_timestamp(ts_str),
            'event_type': self.OPERATION_MAP[operation],
            'file_id': request_data.get('key', job_id),
            'file_size_bytes': request_data.get('length', 0),
            'vsn': request_data.get('volGrp', ''),
            'job_id': job_id,
            'operation': operation,
            'file_state_before': None,
            'file_state_after': None,
        }
        self.events.append(event)
    
    def _handle_completed(self, match: re.Match, line_no: int):
        """Handle Request Completed event."""
        ts_str, level, job_id, response_json = match.groups()
        
        # Look up the start time
        if job_id not in self._started_requests:
            return
            
        start_data = self._started_requests[job_id]
        start_ts = start_data['timestamp']
        end_ts = self.parse_timestamp(ts_str)
        latency = end_ts - start_ts
        
        # Store latency for validation
        self.latencies[job_id] = latency
        
        # Find and update the matching started event
        for event in reversed(self.events):
            if event.get('job_id') == job_id:
                event['latency_s'] = latency
                event['file_state_after'] = 'COMPLETED'
                break
        
        # Create completion event
        try:
            response_data = json.loads(response_json)
            bytes_moved = response_data.get('bytesMoved', 0)
        except:
            bytes_moved = 0
            
        completion_event = {
            'timestamp_s': end_ts,
            'event_type': 'REQUEST_COMPLETED',
            'file_id': start_data['file_id'],
            'file_size_bytes': bytes_moved,
            'vsn': start_data['vol_grp'],
            'job_id': job_id,
            'operation': start_data['operation'],
            'latency_s': latency,
            'file_state_before': 'IN_PROGRESS',
            'file_state_after': 'COMPLETED',
        }
        self.events.append(completion_event)
    
    def get_latency_distribution(self) -> List[float]:
        """Get list of all latency values for validation."""
        return list(self.latencies.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from the log."""
        recall_events = [e for e in self.events if e.get('event_type') == 'RECALL']
        migrate_events = [e for e in self.events if e.get('event_type') == 'MIGRATE']
        
        latencies = list(self.latencies.values())
        
        return {
            'total_events': len(self.events),
            'recall_count': len(recall_events),
            'migrate_count': len(migrate_events),
            'completed_requests': len(latencies),
            'duration_seconds': max(e['timestamp_s'] for e in self.events) - min(e['timestamp_s'] for e in self.events),
            'latency_mean': sum(latencies) / len(latencies) if latencies else 0,
            'latency_p50': self._percentile(latencies, 0.5),
            'latency_p95': self._percentile(latencies, 0.95),
            'latency_p99': self._percentile(latencies, 0.99),
            'latency_min': min(latencies) if latencies else 0,
            'latency_max': max(latencies) if latencies else 0,
        }
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def export_normalized(self, output_path: str, format: str = 'jsonl') -> None:
        """Export normalized events to file."""
        output_path = Path(output_path)
        
        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for event in sorted(self.events, key=lambda x: x['timestamp_s']):
                    # Remove internal fields
                    export_event = {
                        'timestamp_s': event['timestamp_s'],
                        'event_type': event['event_type'],
                        'file_id': event['file_id'],
                        'file_size_bytes': event['file_size_bytes'],
                    }
                    if 'vsn' in event:
                        export_event['vsn'] = event['vsn']
                    if 'latency_s' in event:
                        export_event['latency_s'] = event['latency_s']
                    f.write(json.dumps(export_event) + '\n')
        elif format == 'csv':
            fieldnames = ['timestamp_s', 'event_type', 'file_id', 'file_size_bytes', 'vsn', 'latency_s']
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for event in sorted(self.events, key=lambda x: x['timestamp_s']):
                    writer.writerow(event)
                    
        print(f"Exported {len(self.events)} events to {output_path}")


def main():
    """Standalone execution for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse DMF 7 logs')
    parser.add_argument('log_file', help='Path to DMF log file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--stats', '-s', action='store_true', help='Print statistics')
    parser.add_argument('--format', '-f', choices=['jsonl', 'csv'], default='jsonl',
                       help='Output format')
    
    args = parser.parse_args()
    
    parser_obj = DMFLogParser(args.log_file)
    events = parser_obj.parse_log()
    
    if args.stats:
        stats = parser_obj.get_statistics()
        print("\n=== Log Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if args.output:
        parser_obj.export_normalized(args.output, args.format)


if __name__ == '__main__':
    main()
