"""
IOR Benchmark Output Parser
==========================
Parses IOR benchmark output files and extracts test configurations,
iteration results, and summary statistics.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class IORConfig:
    """IOR test configuration parameters."""
    xfersize_bytes: int
    blocksize_bytes: int
    tasks: int
    nodes: int
    repetitions: int
    segments: int
    aggregate_filesize_bytes: int
    api: str = "POSIX"
    access: str = "file-per-process"


@dataclass
class IORIteration:
    """Single iteration result from IOR test."""
    iteration: int
    operation: str  # "write" or "read"
    bw_mib_s: float
    iops: float
    latency_s: float
    block_kib: float
    xfer_kib: float
    open_s: float
    wr_rd_s: float
    close_s: float
    total_s: float


@dataclass
class IORSummary:
    """Summary statistics across all iterations."""
    operation: str
    max_bw_mib_s: float
    min_bw_mib_s: float
    mean_bw_mib_s: float
    stddev_bw: float
    max_iops: float
    min_iops: float
    mean_iops: float
    stddev_iops: float
    mean_time_s: float


@dataclass
class IORTest:
    """Complete IOR test with config, iterations, and summary."""
    start_time: str
    finish_time: str
    config: IORConfig
    iterations: List[IORIteration] = field(default_factory=list)
    write_summary: Optional[IORSummary] = None
    read_summary: Optional[IORSummary] = None


class IORParser:
    """
    Parser for IOR benchmark output files.
    
    Supports parsing multiple test configurations from a single file.
    """
    
    def __init__(self):
        self.tests: List[IORTest] = []
    
    def parse_file(self, filepath: str) -> List[IORTest]:
        """Parse an IOR output file and return all tests."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into individual test runs
        test_blocks = self._split_test_blocks(content)
        
        for block in test_blocks:
            test = self._parse_test_block(block)
            if test:
                self.tests.append(test)
        
        return self.tests
    
    def _split_test_blocks(self, content: str) -> List[str]:
        """Split content into individual test blocks."""
        # Tests start with "IOR-..." or "Began"
        test_blocks = []
        current_block = []
        in_test = False
        
        for line in content.split('\n'):
            # Check for start of new test
            if 'IOR-' in line and 'MPI Coordinated Test' in line:
                if current_block:
                    test_blocks.append('\n'.join(current_block))
                current_block = [line]
                in_test = True
            elif 'Began' in line and in_test:
                # This is the start line, include it
                current_block.append(line)
            elif in_test:
                current_block.append(line)
                
                # Check for end of test
                if 'Finished' in line and 'IOR-' in content[content.find(line):content.find(line)+100]:
                    pass  # Continue until next test
        
        # Add last block
        if current_block:
            test_blocks.append('\n'.join(current_block))
        
        return test_blocks
    
    def _parse_test_block(self, block: str) -> Optional[IORTest]:
        """Parse a single test block into IORTest object."""
        lines = block.strip().split('\n')
        if not lines:
            return None
        
        # Extract start and finish times
        start_time = ""
        finish_time = ""
        for line in lines:
            if 'Began' in line:
                start_time = line.replace('Began', '').strip()
            if 'Finished' in line:
                finish_time = line.replace('Finished', '').strip()
        
        # Extract configuration
        config = self._parse_config(lines)
        if not config:
            return None
        
        # Extract iterations
        iterations = self._parse_iterations(lines, config.repetitions)
        
        # Extract summaries
        write_summary, read_summary = self._parse_summaries(lines)
        
        return IORTest(
            start_time=start_time,
            finish_time=finish_time,
            config=config,
            iterations=iterations,
            write_summary=write_summary,
            read_summary=read_summary
        )
    
    def _parse_config(self, lines: List[str]) -> Optional[IORConfig]:
        """Parse configuration parameters from test lines."""
        config = {}
        
        for line in lines:
            # Parse key: value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Map to known config fields
                if 'xfersize' in key:
                    config['xfersize'] = self._parse_size(value)
                elif 'blocksize' in key:
                    config['blocksize'] = self._parse_size(value)
                elif key == 'tasks':
                    config['tasks'] = int(value)
                elif key == 'nodes':
                    config['nodes'] = int(value)
                elif 'repetitions' in key:
                    config['repetitions'] = int(value)
                elif 'segments' in key:
                    config['segments'] = int(value)
                elif 'aggregate_filesize' in key:
                    config['aggregate_filesize'] = self._parse_size(value)
                elif key == 'api':
                    config['api'] = value
                elif 'access' in key:
                    config['access'] = value
        
        if 'xfersize' not in config or 'blocksize' not in config:
            return None
        
        return IORConfig(
            xfersize_bytes=config.get('xfersize', 0),
            blocksize_bytes=config.get('blocksize', 0),
            tasks=config.get('tasks', 0),
            nodes=config.get('nodes', 0),
            repetitions=config.get('repetitions', 0),
            segments=config.get('segments', 0),
            aggregate_filesize_bytes=config.get('aggregate_filesize', 0),
            api=config.get('api', 'POSIX'),
            access=config.get('access', 'file-per-process')
        )
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '64 MiB' or '4 GiB' or '4096 bytes' to bytes."""
        size_str = size_str.strip()
        
        # Match patterns like "64 MiB", "4 GiB", "128 KiB", "4096 bytes"
        match = re.match(r'([\d.]+)\s*([KMGT]i?B|bytes?|BYTES?)', size_str, re.IGNORECASE)
        if not match:
            # Try to parse plain number (bytes)
            try:
                return int(size_str)
            except ValueError:
                return 0
        
        value = float(match.group(1))
        unit = match.group(2).upper().rstrip('S')  # Remove trailing 's' for 'bytes'
        
        units = {
            'BYTE': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3,
            'TB': 1024**4,
            'KIB': 1024,
            'MIB': 1024**2,
            'GIB': 1024**3,
            'TIB': 1024**4,
        }
        
        return int(value * units.get(unit, 1))
    
    def _parse_iterations(self, lines: List[str], num_iterations: int) -> List[IORIteration]:
        """Parse iteration results from test lines."""
        iterations = []
        
        # Find Results section
        in_results = False
        for line in lines:
            if 'Results:' in line:
                in_results = True
                continue
            
            if in_results and '------' in line:
                continue
            
            if in_results and line.strip():
                # Parse iteration line
                parts = line.split()
                if len(parts) >= 11:
                    try:
                        # Format: write/write 1169.89 299504 0.000448 65536 4.00 0.645693 336.10 314.72 336.11 0
                        operation = parts[0]
                        iteration = int(parts[10])
                        
                        iter_result = IORIteration(
                            iteration=iteration,
                            operation=operation,
                            bw_mib_s=float(parts[1]),
                            iops=float(parts[2]),
                            latency_s=float(parts[3]),
                            block_kib=float(parts[4]),
                            xfer_kib=float(parts[5]),
                            open_s=float(parts[6]),
                            wr_rd_s=float(parts[7]),
                            close_s=float(parts[8]),
                            total_s=float(parts[9])
                        )
                        iterations.append(iter_result)
                    except (ValueError, IndexError):
                        continue
            
            if in_results and 'Summary' in line:
                break
        
        return iterations
    
    def _parse_summaries(self, lines: List[str]) -> tuple:
        """Parse summary statistics."""
        write_summary = None
        read_summary = None
        
        in_summary = False
        for i, line in enumerate(lines):
            if 'Summary of all tests:' in line:
                in_summary = True
                continue
            
            if in_summary and line.strip():
                parts = line.split()
                # Summary line format:
                # Operation Max(MiB) Min(MiB) Mean(MiB) StdDev Max(OPs) Min(OPs) Mean(OPs) StdDev Mean(s) ...
                # Example: write 1460.23 1169.89 1316.48 76.07 373818.85 299492.59 337019.02 19474.58 299.69923 ...
                if len(parts) >= 11:
                    try:
                        operation = parts[0].lower()
                        summary = IORSummary(
                            operation=operation,
                            max_bw_mib_s=float(parts[1]),
                            min_bw_mib_s=float(parts[2]),
                            mean_bw_mib_s=float(parts[3]),
                            stddev_bw=float(parts[4]),
                            max_iops=float(parts[5]),
                            min_iops=float(parts[6]),
                            mean_iops=float(parts[7]),
                            stddev_iops=float(parts[8]),
                            mean_time_s=float(parts[9])
                        )
                        
                        if operation == 'write':
                            write_summary = summary
                        elif operation == 'read':
                            read_summary = summary
                    except (ValueError, IndexError):
                        continue
        
        return write_summary, read_summary
    
    def get_tests_by_config(self) -> Dict[str, List[IORTest]]:
        """Group tests by configuration (transfer_size_block_size)."""
        grouped = {}
        for test in self.tests:
            config_name = f"ior_{test.config.xfersize_bytes}_{test.config.blocksize_bytes}"
            if config_name not in grouped:
                grouped[config_name] = []
            grouped[config_name].append(test)
        return grouped
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from parsed tests."""
        if not self.tests:
            return {}
        
        total_iterations = sum(len(t.iterations) for t in self.tests)
        
        return {
            'total_tests': len(self.tests),
            'total_iterations': total_iterations,
            'configurations': list(self.get_tests_by_config().keys())
        }


def parse_ior_file(filepath: str) -> List[IORTest]:
    """Convenience function to parse an IOR file."""
    parser = IORParser()
    return parser.parse_file(filepath)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ior_parser.py <ior_output_file>")
        sys.exit(1)
    
    parser = IORParser()
    tests = parser.parse_file(sys.argv[1])
    
    print(f"Parsed {len(tests)} IOR tests")
    
    for i, test in enumerate(tests):
        print(f"\nTest {i+1}:")
        print(f"  Config: {test.config.xfersize_bytes}B transfer, {test.config.blocksize_bytes}B block")
        print(f"  Tasks: {test.config.tasks}, Repetitions: {test.config.repetitions}")
        print(f"  Iterations: {len(test.iterations)}")
        
        if test.write_summary:
            print(f"  Write Summary: {test.write_summary.mean_bw_mib_s:.2f} MiB/s, {test.write_summary.mean_iops:.2f} IOPS")
        if test.read_summary:
            print(f"  Read Summary: {test.read_summary.mean_bw_mib_s:.2f} MiB/s, {test.read_summary.mean_iops:.2f} IOPS")
