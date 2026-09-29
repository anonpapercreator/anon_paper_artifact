"""
IOR Trace Generator
==================
Generates JSONL trace files from parsed IOR benchmark data.
These traces can be used to simulate IOR workloads in the HSM simulator.
"""

import json
from typing import List, Dict, Any
from pathlib import Path

from workload.ior_parser import IORParser, IORTest, IORIteration


class IORTraceGenerator:
    """
    Generates JSONL traces from IOR benchmark data.
    
    Creates per-operation event traces that can be replayed
    by the HSM simulator to analyze archive behavior.
    """
    
    def __init__(self, output_dir: str = "./traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_traces(self, ior_tests: List[IORTest]) -> List[str]:
        """
        Generate JSONL trace files for all IOR tests.
        
        Args:
            ior_tests: List of parsed IOR tests
            
        Returns:
            List of generated trace file paths
        """
        generated_files = []
        
        for test in ior_tests:
            # Generate traces for each iteration
            iteration_traces = self._generate_test_traces(test)
            generated_files.extend(iteration_traces)
        
        return generated_files
    
    def _generate_test_traces(self, test: IORTest) -> List[str]:
        """Generate traces for a single IOR test (all iterations)."""
        generated_files = []
        
        # Group iterations by write/read
        # Each iteration has 2 entries: write then read
        for i in range(0, len(test.iterations), 2):
            if i + 1 >= len(test.iterations):
                break
                
            write_iter = test.iterations[i]
            read_iter = test.iterations[i + 1] if i + 1 < len(test.iterations) else None
            
            # Generate trace for this iteration
            filename = self._generate_iteration_trace(test, write_iter, read_iter)
            if filename:
                generated_files.append(filename)
        
        return generated_files
    
    def _generate_iteration_trace(self, test: IORTest, 
                                  write_iter: IORIteration,
                                  read_iter: IORIteration) -> str:
        """Generate a single iteration trace file."""
        
        # Create config name for the trace file
        config_name = self._get_config_name(test)
        filename = f"trace_{config_name}_iter{write_iter.iteration}.jsonl"
        filepath = self.output_dir / filename
        
        events = []
        current_time = 0.0
        
        # Generate WRITE events
        write_events = self._generate_operation_events(
            operation='WRITE',
            iops=write_iter.iops,
            duration_s=write_iter.total_s,
            num_tasks=test.config.tasks,
            block_size_bytes=test.config.blocksize_bytes,
            start_time=current_time
        )
        events.extend(write_events)
        current_time += write_iter.total_s
        
        # Generate READ events (start after writes complete)
        if read_iter:
            read_events = self._generate_operation_events(
                operation='READ',
                iops=read_iter.iops,
                duration_s=read_iter.total_s,
                num_tasks=test.config.tasks,
                block_size_bytes=test.config.blocksize_bytes,
                start_time=current_time
            )
            events.extend(read_events)
        
        # Write events to file
        with open(filepath, 'w') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        
        return str(filepath)
    
    def _generate_operation_events(self, operation: str, iops: float, 
                                  duration_s: float, num_tasks: int,
                                  block_size_bytes: int,
                                  start_time: float) -> List[Dict[str, Any]]:
        """
        Generate events for a single operation (WRITE or READ).
        
        Generates file-level events (one per task), not I/O-level events.
        This is more suitable for HSM simulation.
        
        Args:
            operation: 'WRITE' or 'READ'
            iops: Operations per second (used for timing)
            duration_s: Duration in seconds
            num_tasks: Number of parallel tasks
            block_size_bytes: Size per file in bytes
            start_time: Starting timestamp
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        if duration_s <= 0:
            return events
        
        # Generate one event per task (file-per-process pattern)
        # This represents each MPI task writing/reading one file
        # Distribute evenly across the duration
        
        for task_id in range(num_tasks):
            # Spread tasks evenly across the duration
            event_time = start_time + (task_id / num_tasks) * duration_s
            
            # File size is the block size (each task writes one block)
            file_size = block_size_bytes
            
            event = {
                'timestamp_s': event_time,
                'event_type': operation,
                'file_id': f"task{task_id}",
                'file_size_bytes': file_size,
                'task_id': task_id,
            }
            
            events.append(event)
        
        return events
    
    def _get_config_name(self, test: IORTest) -> str:
        """Generate configuration name from test config."""
        xfer = test.config.xfersize_bytes
        block = test.config.blocksize_bytes
        
        # Convert to human-readable
        xfer_str = f"{xfer}" if xfer < 1024 else f"{xfer//1024}k"
        block_str = f"{block}" if block < 1024 else f"{block//1024}k"
        
        return f"ior_{xfer_str}_{block_str}"
    
    def generate_summary_csv(self, ior_tests: List[IORTest], output_file: str = None) -> str:
        """
        Generate a CSV summary of all IOR tests.
        
        Args:
            ior_tests: List of parsed IOR tests
            output_file: Optional output file path
            
        Returns:
            Path to generated CSV file
        """
        if output_file is None:
            output_file = self.output_dir / "ior_summary.csv"
        
        lines = []
        lines.append("config,iteration,operation,bw_mib_s,iops,latency_s,time_s,total_bytes")
        
        for test in ior_tests:
            config_name = self._get_config_name(test)
            
            for i in range(0, len(test.iterations), 2):
                write_iter = test.iterations[i]
                lines.append(f"{config_name},{write_iter.iteration},write,"
                           f"{write_iter.bw_mib_s},{write_iter.iops},"
                           f"{write_iter.latency_s},{write_iter.total_s},"
                           f"{int(write_iter.bw_mib_s * 1024**2 * write_iter.total_s)}")
                
                if i + 1 < len(test.iterations):
                    read_iter = test.iterations[i + 1]
                    lines.append(f"{config_name},{read_iter.iteration},read,"
                               f"{read_iter.bw_mib_s},{read_iter.iops},"
                               f"{read_iter.latency_s},{read_iter.total_s},"
                               f"{int(read_iter.bw_mib_s * 1024**2 * read_iter.total_s)}")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(output_file)


def generate_ior_traces(ior_file: str, output_dir: str = "./traces") -> List[str]:
    """
    Convenience function to generate traces from an IOR file.
    
    Args:
        ior_file: Path to IOR output file
        output_dir: Directory to write trace files
        
    Returns:
        List of generated trace file paths
    """
    # Parse IOR file
    parser = IORParser()
    tests = parser.parse_file(ior_file)
    
    # Generate traces
    generator = IORTraceGenerator(output_dir)
    trace_files = generator.generate_traces(tests)
    
    # Also generate summary
    generator.generate_summary_csv(tests)
    
    return trace_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ior_trace_generator.py <ior_output_file> [output_dir]")
        sys.exit(1)
    
    ior_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./traces"
    
    trace_files = generate_ior_traces(ior_file, output_dir)
    print(f"Generated {len(trace_files)} trace files in {output_dir}/")
