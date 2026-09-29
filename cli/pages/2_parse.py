"""
DESCASSI - Trace Generator Page
=============================
Unified interface for parsing log files into simulation traces.
Supports IOR Benchmark, DMF Logs, and P4/MSP Logs.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import json
import os

st.set_page_config(
    page_title="DESCASSI - Trace Generator",
    page_icon="📂",
    layout="wide")

st.title("📂 DESCASSI Trace Generator")
st.markdown("*Convert log files to simulation-ready traces*")

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def init_log_state():
    """Initialize session state for logging."""
    if 'parse_log' not in st.session_state:
        st.session_state.parse_log = []
    if 'last_parse_result' not in st.session_state:
        st.session_state.last_parse_result = None

def log_message(msg, level="info"):
    """Add a message to the session log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}.get(level, "•")
    st.session_state.parse_log.append(f"[{timestamp}] {prefix} {msg}")

def save_log():
    """Save log to file."""
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"parse_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write("\n".join(st.session_state.parse_log))
    return log_file

def clear_log():
    """Clear the session log."""
    st.session_state.parse_log = []
    st.session_state.last_parse_result = None

# Initialize logging state
init_log_state()

# ============================================================================
# FILE BROWSER UTILITIES
# ============================================================================

def validate_path(path_str, path_type="file"):
    """Validate that a path exists and is the correct type."""
    if not path_str:
        return False, "Path is empty"
    
    p = Path(path_str)
    if not p.exists():
        return False, "Path does not exist"
    if path_type == "file" and not p.is_file():
        return False, "Path is not a file"
    if path_type == "directory" and not p.is_dir():
        return False, "Path is not a directory"
    return True, "Valid"

def browse_file(label, key, help_text="", default_value=""):
    """File input with text field + validate button."""
    col1, col2 = st.columns([4, 1])
    current_value = st.session_state.get(key, default_value)
    with col1:
        path = st.text_input(
            label,
            value=current_value,
            help=help_text,
            key=f"{key}_text"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✓", key=f"{key}_validate"):
            if path:
                valid, msg = validate_path(path, "file")
                if valid:
                    st.success("Valid")
                    st.session_state[key] = path
                else:
                    st.error(msg)
    return path

def browse_directory(label, key, help_text="", default_value="./traces"):
    """Directory input with text field + validate button."""
    col1, col2 = st.columns([4, 1])
    current_value = st.session_state.get(key, default_value)
    with col1:
        path = st.text_input(
            label,
            value=current_value,
            help=help_text,
            key=f"{key}_text"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✓", key=f"{key}_validate"):
            if path:
                valid, msg = validate_path(path, "directory")
                if valid:
                    st.success("Valid")
                    st.session_state[key] = path
                else:
                    st.error(msg)
    return path

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Log type selector
log_type = st.selectbox(
    "Log Type",
    ["IOR Benchmark", "DMF Log", "P4/MSP Log"],
    index=0,
    help="Select the type of log file to parse"
)

st.markdown("---")

# ============================================================================
# IOR BENCHMARK PARSER
# ============================================================================

if log_type == "IOR Benchmark":
    st.markdown("### IOR Benchmark Settings")
    
    # File inputs
    ior_file = browse_file(
        "Input File",
        "ior_input_file",
        "Path to IOR benchmark output file (.txt)",
        default_value=""
    )
    
    output_dir = browse_directory(
        "Output Directory",
        "ior_output_dir",
        "Directory to save trace files",
        default_value="./traces"
    )
    
    st.markdown("#### Options")
    
    # Output format options
    col1, col2 = st.columns(2)
    with col1:
        include_summary = st.checkbox("Include summary CSV", value=True, key="ior_summary")
    with col2:
        show_details = st.checkbox("Show iteration details", value=True, key="ior_details")
    
    # Parse button
    st.markdown("")
    if st.button("🔍 Generate IOR Traces", type="primary", key="ior_parse_btn"):
        if not ior_file:
            st.error("Please select an IOR output file")
        else:
            valid, msg = validate_path(ior_file, "file")
            if not valid:
                st.error(f"Invalid input file: {msg}")
            else:
                log_message(f"Parsing IOR file: {ior_file}")
                
                try:
                    from workload.ior_parser import IORParser
                    from workload.ior_trace_generator import IORTraceGenerator
                    
                    with st.spinner("Parsing IOR file..."):
                        parser = IORParser()
                        tests = parser.parse_file(ior_file)
                    
                    log_message(f"Found {len(tests)} test configurations", "success")
                    
                    # Show test configurations
                    if show_details:
                        st.markdown("#### Test Configurations Found")
                        for i, test in enumerate(tests):
                            xfer = test.config.xfersize_bytes
                            block = test.config.blocksize_bytes
                            xfer_str = f'{xfer}B' if xfer < 1024 else f'{xfer//1024}KB'
                            block_str = f'{block}B' if block < 1024 else f'{block//1024}KB'
                            
                            write_bw = test.write_summary.mean_bw_mib_s if test.write_summary else 0
                            read_bw = test.read_summary.mean_bw_mib_s if test.read_summary else 0
                            
                            st.markdown(f"**{i+1}. {xfer_str} transfer, {block_str} block**")
                            st.caption(f"   Write: {write_bw:.1f} MiB/s | Read: {read_bw:.1f} MiB/s | {len(test.iterations)//2} iterations")
                            log_message(f"  {i+1}. {xfer_str}/{block_str} - {len(test.iterations)//2} iterations")
                    
                    # Generate traces
                    with st.spinner("Generating trace files..."):
                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        generator = IORTraceGenerator(str(out_dir))
                        trace_files = generator.generate_traces(tests)
                        
                        if include_summary:
                            generator.generate_summary_csv(tests)
                    
                    log_message(f"Generated {len(trace_files)} trace files", "success")
                    st.session_state.last_parse_result = {
                        "type": "ior",
                        "files": trace_files,
                        "tests": len(tests)
                    }
                    
                    # Show generated files
                    if show_details:
                        st.markdown("#### Generated Trace Files")
                        for f in trace_files[:15]:
                            st.text(f"  • {Path(f).name}")
                        if len(trace_files) > 15:
                            st.caption(f"  ... and {len(trace_files) - 15} more")
                    
                    summary_file = out_dir / "ior_summary.csv"
                    if summary_file.exists() and include_summary:
                        st.success(f"Summary saved to: {summary_file}")
                    st.success(f"Generated {len(trace_files)} trace files")
                        
                except Exception as e:
                    log_message(f"Error: {str(e)}", "error")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# DMF LOG PARSER
# ============================================================================

elif log_type == "DMF Log":
    st.markdown("### DMF Log Settings")
    
    log_dir = browse_directory(
        "DMF Log Directory",
        "dmf_input_dir",
        "Directory containing DMF ls.*.log files",
        default_value=""
    )
    
    output_dir = browse_directory(
        "Output Directory",
        "dmf_output_dir",
        "Directory to save the trace file",
        default_value="./traces"
    )
    
    st.markdown("#### Options")
    
    output_filename = st.text_input(
        "Output Filename",
        value="trace_dmf.jsonl",
        key="dmf_output_filename"
    )
    
    st.markdown("""
    **Expected Log Format:**
    - Files matching `ls.*.log` pattern
    - With `Request Started` and `Request Completed` events
    """)
    
    if st.button("📁 Scan Directory", key="dmf_scan_btn"):
        if log_dir:
            valid, msg = validate_path(log_dir, "directory")
            if valid:
                try:
                    from workload.dmf_log_parser import DMFLogParser
                    parser = DMFLogParser()
                    files = parser.scan_directory(log_dir)
                    log_message(f"Found {len(files)} log files", "success")
                    
                    st.markdown(f"**Found {len(files)} files:**")
                    for f in files[:20]:
                        st.text(f"  • {Path(f).name}")
                    if len(files) > 20:
                        st.caption(f"  ... and {len(files) - 20} more")
                except Exception as e:
                    log_message(f"Error scanning: {str(e)}", "error")
            else:
                st.error(f"Invalid directory: {msg}")
    
    if st.button("🔍 Parse DMF Logs", type="primary", key="dmf_parse_btn"):
        if not log_dir:
            st.error("Please select a DMF log directory")
        else:
            valid, msg = validate_path(log_dir, "directory")
            if not valid:
                st.error(f"Invalid directory: {msg}")
            else:
                log_message(f"Parsing DMF logs from: {log_dir}")
                
                try:
                    from workload.dmf_log_parser import DMFLogParser
                    
                    with st.spinner("Parsing log files..."):
                        parser = DMFLogParser()
                        count = parser.parse_directory(log_dir)
                        stats = parser.get_statistics()
                    
                    # Export
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_path = out_dir / output_filename
                    parser.export_normalized(str(output_path))
                    
                    log_message(f"Parsed {count} files", "success")
                    log_message(f"Exported to: {output_path}", "success")
                    
                    st.session_state.last_parse_result = {
                        "type": "dmf",
                        "file": str(output_path),
                        "stats": stats
                    }
                    
                    # Show results
                    st.markdown("#### Results")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Events", stats['total_events'])
                    c2.metric("Recalls", stats['recall_count'])
                    c3.metric("Migrations", stats['migrate_count'])
                    
                    st.markdown("#### Latency Statistics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean", f"{stats['latency_mean']:.1f}s")
                    c2.metric("p50", f"{stats['latency_p50']:.1f}s")
                    c3.metric("p95", f"{stats['latency_p95']:.1f}s")
                    c4.metric("p99", f"{stats['latency_p99']:.1f}s")
                    
                except Exception as e:
                    log_message(f"Error: {str(e)}", "error")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# P4/MSP LOG PARSER
# ============================================================================

elif log_type == "P4/MSP Log":
    st.markdown("### P4/MSP Log Settings")
    
    log_dir = browse_directory(
        "P4 Log Directory",
        "p4_input_dir",
        "Directory containing msplog.YYYYMMDD files",
        default_value=""
    )
    
    output_dir = browse_directory(
        "Output Directory",
        "p4_output_dir",
        "Directory to save the trace file",
        default_value="./traces"
    )
    
    st.markdown("#### Options")
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input(
            "Start Date (YYYYMMDD)",
            value="",
            help="Leave empty for earliest date",
            key="p4_start_date"
        )
    with col2:
        end_date = st.text_input(
            "End Date (YYYYMMDD)",
            value="",
            help="Leave empty for latest date",
            key="p4_end_date"
        )
    
    output_filename = st.text_input(
        "Output Filename",
        value="trace_p4.jsonl",
        key="p4_output_filename"
    )
    
    st.markdown("""
    **Expected Log Format:**
    - `msplog.YYYYMMDD` files
    - `Put_File` = migration to tape
    - `Get_File` = recall from tape
    """)
    
    if st.button("📁 Scan Directory", key="p4_scan_btn"):
        if log_dir:
            valid, msg = validate_path(log_dir, "directory")
            if valid:
                try:
                    from workload.msp_log_parser import MSPLogParser
                    parser = MSPLogParser()
                    files = parser.scan_directory(log_dir, start_date or None, end_date or None)
                    log_message(f"Found {len(files)} log files", "success")
                    
                    st.markdown(f"**Found {len(files)} files:**")
                    for f in files[:20]:
                        st.text(f"  • {Path(f).name}")
                    if len(files) > 20:
                        st.caption(f"  ... and {len(files) - 20} more")
                except Exception as e:
                    log_message(f"Error scanning: {str(e)}", "error")
            else:
                st.error(f"Invalid directory: {msg}")
    
    if st.button("🔍 Parse P4 Logs", type="primary", key="p4_parse_btn"):
        if not log_dir:
            st.error("Please select a P4 log directory")
        else:
            valid, msg = validate_path(log_dir, "directory")
            if not valid:
                st.error(f"Invalid directory: {msg}")
            else:
                log_message(f"Parsing P4 logs from: {log_dir}")
                if start_date:
                    log_message(f"  Date range: {start_date} to {end_date or 'latest'}")
                
                try:
                    from workload.msp_log_parser import MSPLogParser
                    
                    with st.spinner("Parsing log files..."):
                        parser = MSPLogParser()
                        count = parser.parse_directory(
                            log_dir,
                            start_date if start_date else None,
                            end_date if end_date else None
                        )
                        stats = parser.get_statistics()
                    
                    # Export
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_path = out_dir / output_filename
                    parser.export_normalized(str(output_path))
                    
                    log_message(f"Parsed {count} files", "success")
                    log_message(f"Exported to: {output_path}", "success")
                    
                    st.session_state.last_parse_result = {
                        "type": "p4",
                        "file": str(output_path),
                        "stats": stats
                    }
                    
                    # Show results
                    st.markdown("#### Results")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Events", stats['total_events'])
                    c2.metric("Migrations", stats['migrate_count'])
                    c3.metric("Recalls", stats['recall_count'])
                    
                    c4, c5 = st.columns(2)
                    c4.metric("Total Migrated", f"{stats['total_migrate_bytes'] / 1024**4:.2f} TB")
                    c5.metric("Total Recalled", f"{stats['total_recall_bytes'] / 1024**4:.2f} TB")
                    
                    st.metric("Duration", f"{stats['duration_seconds'] / 3600:.2f} hours")
                    
                except Exception as e:
                    log_message(f"Error: {str(e)}", "error")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# OUTPUT LOG SECTION
# ============================================================================

st.markdown("---")
st.markdown("### Output Log")

# Log display
if st.session_state.parse_log:
    log_container = st.container()
    with log_container:
        for msg in st.session_state.parse_log[-50:]:
            if "❌" in msg:
                st.error(msg)
            elif "✅" in msg:
                st.success(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            else:
                st.text(msg)
else:
    st.info("No log messages yet. Parse a file to see output here.")

# Log controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Save Log"):
        if st.session_state.parse_log:
            log_file = save_log()
            st.success(f"Log saved to: {log_file}")
        else:
            st.warning("No log to save")

with col2:
    if st.button("🗑️ Clear Log"):
        clear_log()
        st.rerun()

with col3:
    if st.button("📥 Download Traces") and st.session_state.last_parse_result:
        result = st.session_state.last_parse_result
        if result["type"] == "ior" and result.get("files"):
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in result["files"]:
                    zf.write(f, Path(f).name)
            
            st.download_button(
                "Download ZIP",
                data=zip_buffer.getvalue(),
                file_name="ior_traces.zip",
                mime="application/zip"
            )
        elif result.get("file"):
            filepath = result["file"]
            with open(filepath, 'r') as f:
                content = f.read()
            st.download_button(
                "Download Trace",
                data=content,
                file_name=Path(filepath).name,
                mime="application/json"
            )
