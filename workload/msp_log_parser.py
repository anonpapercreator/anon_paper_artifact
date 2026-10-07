"""
MSP Log Parser
==============
Parses DMF Managed Storage Proxy (MSP) logs into normalized trace format.

Log Format (from ls1 system):
- msplog.YYYYMMDD files in /ls1 directory
- Events: Put_File (migration), Get_File (recall)
- Example: Req=7486238,18940a9734e9371e,Put_File,key=0abf..., good Put_File - size 380011541.

Output: Normalized CSV/JSONL with fields required for simulation replay

Author: Simulation Team
"""

from __future__ import annotations
import re
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import glob


class MSPLogParser:
    """
    Parses MSP (Managed Storage Proxy) logs into normalized trace format.
    
    Input: msplog.YYYYMMDD files from /ls1 directory
    Output: Normalized events for simulation replay
    """
    
    PUT_FILE_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)-[IO]\s+'
        r'(\S+)\s+\d+-\S+\s+'
        r'Req=(\d+),([^,]+),Put_File,key=([^,]+),\s*'
        r'(good|bad) Put_File - size (\d+)'
    )
    
    GET_FILE_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)-[IO]\s+'
        r'(\S+)\s+\d+-\S+\s+'
        r'Req=(\d+),([^,]+),Get_File,key=([^,]+),\s*'
        r'(good|bad) Get_File - size (\d+)'
    )
    
    GET_FILE_QUEUED_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)-[IO]\s+'
        r'(\S+)\s+\d+-\S+\s+'
        r'Req=(\d+),([^,]+),Get_File,key=([^,]+),\s*'
        r'volume group=(\S+) pri=(\d+) queued'
    )
    
    OPERATION_MAP = {
        'Put_File': 'MIGRATE',
        'Get_File': 'RECALL',
    }
    
    def __init__(self, log_path: str = None):
        self.log_path = Path(log_path) if log_path else None
        self.events: List[Dict[str, Any]] = []
        self._file_cache: Dict[str, List[Dict]] = {}
    
    def scan_directory(self, directory: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
        """
        Scan directory for msplog.YYYYMMDD files.
        
        Args:
            directory: Path to directory containing msplog files
            start_date: Optional start date (YYYYMMDD)
            end_date: Optional end date (YYYYMMDD)
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = str(dir_path / "msplog.*")
        files = glob.glob(pattern)
        files.sort()
        
        if start_date:
            files = [f for f in files if Path(f).name >= f"msplog.{start_date}"]
        if end_date:
            files = [f for f in files if Path(f).name <= f"msplog.{end_date}"]
        
        return files
    
    def parse_file(self, file_path: str) -> int:
        """
        Parse a single MSP log file.
        
        Args:
            file_path: Path to msplog.YYYYMMDD file
            
        Returns:
            Number of events parsed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        events_before = len(self.events)
        
        with open(file_path, 'r') as f:
            for line in f:
                self._parse_line(line)
        
        return len(self.events) - events_before
    
    def parse_directory(self, directory: str, start_date: str = None, 
                       end_date: str = None) -> int:
        """
        Parse all MSP logs in a directory within date range.
        
        Args:
            directory: Path to directory containing msplog files
            start_date: Optional start date (YYYYMMDD)
            end_date: Optional end date (YYYYMMDD)
            
        Returns:
            Total number of events parsed
        """
        files = self.scan_directory(directory, start_date, end_date)
        
        total_events = 0
        for file_path in files:
            count = self.parse_file(file_path)
            total_events += count
        
        return total_events
    
    def _parse_line(self, line: str):
        """Parse a single log line."""
        put_match = self.PUT_FILE_PATTERN.search(line)
        if put_match:
            self._handle_put_file(put_match)
            return
        
        get_match = self.GET_FILE_PATTERN.search(line)
        if get_match:
            self._handle_get_file(get_match)
            return
    
    def _handle_put_file(self, match: re.Match):
        """Handle Put_File (migration) event."""
        ts_str, hostname, req_id, uuid, key, status, size_str = match.groups()
        
        try:
            timestamp = self._parse_timestamp(ts_str)
            file_size = int(size_str)
        except (ValueError, TypeError):
            return
        
        event = {
            'timestamp_s': timestamp,
            'event_type': 'MIGRATE',
            'file_id': key[:32] if len(key) > 32 else key,
            'file_size_bytes': file_size,
            'status': status,
            'operation': 'Put_File',
            'req_id': req_id,
            'uuid': uuid,
        }
        
        self.events.append(event)
    
    def _handle_get_file(self, match: re.Match):
        """Handle Get_File (recall) event."""
        ts_str, hostname, req_id, uuid, key, status, size_str = match.groups()
        
        try:
            timestamp = self._parse_timestamp(ts_str)
            file_size = int(size_str)
        except (ValueError, TypeError):
            return
        
        event = {
            'timestamp_s': timestamp,
            'event_type': 'RECALL',
            'file_id': key[:32] if len(key) > 32 else key,
            'file_size_bytes': file_size,
            'status': status,
            'operation': 'Get_File',
            'req_id': req_id,
            'uuid': uuid,
        }
        
        self.events.append(event)
    
    def _parse_timestamp(self, ts_str: str) -> float:
        """Convert ISO timestamp to seconds since epoch."""
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.timestamp()
        except ValueError:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from parsed events."""
        migrate_events = [e for e in self.events if e.get('event_type') == 'MIGRATE']
        recall_events = [e for e in self.events if e.get('event_type') == 'RECALL']
        
        migrate_sizes = [e.get('file_size_bytes', 0) for e in migrate_events]
        recall_sizes = [e.get('file_size_bytes', 0) for e in recall_events]
        
        return {
            'total_events': len(self.events),
            'migrate_count': len(migrate_events),
            'recall_count': len(recall_events),
            'total_migrate_bytes': sum(migrate_sizes),
            'total_recall_bytes': sum(recall_sizes),
            'duration_seconds': (
                max(e['timestamp_s'] for e in self.events) - 
                min(e['timestamp_s'] for e in self.events)
            ) if self.events else 0,
            'avg_migrate_size_bytes': sum(migrate_sizes) / len(migrate_sizes) if migrate_sizes else 0,
            'avg_recall_size_bytes': sum(recall_sizes) / len(recall_sizes) if recall_sizes else 0,
        }
    
    def export_normalized(self, output_path: str, format: str = 'jsonl') -> None:
        """
        Export normalized events to file.
        
        Args:
            output_path: Output file path
            format: 'jsonl' or 'csv'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for event in sorted(self.events, key=lambda x: x.get('timestamp_s', 0)):
                    export_event = {
                        'timestamp_s': event['timestamp_s'],
                        'event_type': event['event_type'],
                        'file_id': event['file_id'],
                        'file_size_bytes': event['file_size_bytes'],
                    }
                    f.write(json.dumps(export_event) + '\n')
        elif format == 'csv':
            fieldnames = ['timestamp_s', 'event_type', 'file_id', 'file_size_bytes']
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for event in sorted(self.events, key=lambda x: x.get('timestamp_s', 0)):
                    writer.writerow({
                        'timestamp_s': event['timestamp_s'],
                        'event_type': event['event_type'],
                        'file_id': event['file_id'],
                        'file_size_bytes': event['file_size_bytes'],
                    })
        
        print(f"Exported {len(self.events)} events to {output_path}")


def main():
    """Standalone execution for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse MSP logs')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--stats', '-s', action='store_true', help='Print statistics')
    parser.add_argument('--format', '-f', choices=['jsonl', 'csv'], default='jsonl',
                       help='Output format')
    parser.add_argument('--start-date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', help='End date (YYYYMMDD)')
    
    args = parser.parse_args()
    
    msp = MSPLogParser()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        print(f"Scanning directory: {input_path}")
        files = msp.scan_directory(str(input_path), args.start_date, args.end_date)
        print(f"Found {len(files)} files")
        for f in files:
            print(f"  {f}")
        
        count = msp.parse_directory(str(input_path), args.start_date, args.end_date)
        print(f"Parsed {count} events")
    else:
        count = msp.parse_file(str(input_path))
        print(f"Parsed {count} events")
    
    if args.stats:
        stats = msp.get_statistics()
        print(f"\n=== Statistics ===")
        print(f"Total events: {stats['total_events']}")
        print(f"Migrations: {stats['migrate_count']}")
        print(f"Recalls: {stats['recall_count']}")
        print(f"Total migrated: {stats['total_migrate_bytes'] / 1024**4:.2f} TB")
        print(f"Total recalled: {stats['total_recall_bytes'] / 1024**4:.2f} TB")
        print(f"Duration: {stats['duration_seconds'] / 3600:.2f} hours")
    
    if args.output:
        msp.export_normalized(args.output, args.format)


if __name__ == '__main__':
    main()
