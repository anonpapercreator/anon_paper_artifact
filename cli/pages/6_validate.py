"""
DESCASSI - Validate Page
=======================
Compare simulation results against real-world trace data and validate parameter relationships.
"""

import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="DESCASSI - Validate",
    page_icon="✅",
    layout="wide")

st.title("✅ Validate Simulation")
st.markdown("*Compare simulation results against real-world data and validate model relationships*")

from validation.data_service import ValidationDataService, get_available_traces

data_service = ValidationDataService()
sweeps = data_service.get_sweeps()
all_traces = get_available_traces()

# Separate traces by type
dmf_traces = [t for t in all_traces if 'dmf' in t.lower()]
p4_traces = [t for t in all_traces if 'p4' in t.lower()]
ior_traces = [t for t in all_traces if 'ior' in t.lower()]

tab_compare, tab_sweep, tab_ior, tab_summary = st.tabs([
    "📊 Compare Simulation vs Trace", 
    "📈 Parameter Sweep Analysis",
    "🔬 IOR Benchmark Validation",
    "📋 Summary"
])

# ============================================================================
# TAB 1: COMPARE SIMULATION VS TRACE
# ============================================================================

with tab_compare:
    st.markdown("### Compare Simulation vs Real-World Trace Data")
    
    st.info("""
    **What this does:**
    
    This tab compares the output of a simulation run against real-world trace data that was previously parsed and saved as `.jsonl` files.
    
    **Workflow:**
    1. Select the trace type (DMF, P4, or IOR) from the dropdown
    2. Choose the specific trace file to compare against (DMF/P4) or use the IOR summary
    3. Select a simulation sweep to compare
    4. Click "Run Validation" to see the comparison
    """)
    
    st.markdown("---")
    
    # Trace type selector
    trace_type = st.selectbox(
        "Trace Type",
        ["DMF Trace", "P4 Trace", "IOR Trace"],
        index=0,
        help="Select the type of trace to compare against"
    )
    
    # Filter traces by type
    if trace_type == "DMF Trace":
        available_traces = dmf_traces
        trace_prefix = "DMF"
        is_ior = False
    elif trace_type == "P4 Trace":
        available_traces = p4_traces
        trace_prefix = "P4"
        is_ior = False
    else:
        available_traces = ior_traces
        trace_prefix = "IOR"
        is_ior = True
    
    if is_ior:
        # For IOR, we use the summary CSV instead of individual trace files
        ior_summary_path = "./traces/ior_summary.csv"
        summary_path = Path(ior_summary_path)
        
        if summary_path.exists():
            st.success(f"Using IOR summary: {ior_summary_path}")
            real_metrics = data_service.get_metrics_from_ior_summary(ior_summary_path)
            
            if real_metrics:
                st.markdown("#### Real IOR Benchmark Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Events", f"{real_metrics.get('total_events', 0):,}")
                c2.metric("Write Operations", f"{real_metrics.get('write_count', 0):,}")
                c3.metric("Read Operations", f"{real_metrics.get('read_count', 0):,}")
                c4.metric("Total Data", f"{real_metrics.get('total_bytes', 0) / 1024**4:.2f} TiB")
                
                # Show IOR-specific metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Write Throughput", f"{real_metrics.get('write_bw_mean_mibs', 0):.2f} MiB/s")
                c2.metric("Read Throughput", f"{real_metrics.get('read_bw_mean_mibs', 0):.2f} MiB/s")
                c3.metric("Write IOPS", f"{real_metrics.get('write_iops_mean', 0):,.0f}")
                c4.metric("Read IOPS", f"{real_metrics.get('read_iops_mean', 0):,.0f}")
                
                c1, c2 = st.columns(2)
                c1.metric("Write Latency", f"{real_metrics.get('write_latency_mean_s', 0):.6f} s")
                c2.metric("Read Latency", f"{real_metrics.get('read_latency_mean_s', 0):.6f} s")
        else:
            st.warning(f"IOR summary not found. Parse an IOR file first.")
            real_metrics = {}
    elif available_traces:
        selected_trace = st.selectbox(
            f"Select {trace_prefix} Trace",
            available_traces,
            help=f"Choose a {trace_prefix} trace file to compare against"
        )
        
        trace_path = Path(selected_trace)
        st.caption(f"File: {trace_path.name}")
        
        if trace_type == "DMF Trace":
            real_metrics = data_service.get_metrics_from_dmf_trace(selected_trace)
        else:
            real_metrics = data_service.get_metrics_from_p4_trace(selected_trace)
        
        if real_metrics:
            st.markdown("#### Real Trace Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Events", f"{real_metrics.get('total_events', 0):,}")
            c2.metric("Recalls", f"{real_metrics.get('recall_count', 0):,}")
            c3.metric("Migrations", f"{real_metrics.get('migrate_count', 0):,}")
            c4.metric("Duration", f"{real_metrics.get('duration_hours', 0):.1f} hrs")
    else:
        st.warning(f"No {trace_prefix} traces found. Parse some {trace_prefix} logs first.")
        selected_trace = None
        real_metrics = {}
    
    st.markdown("---")
    st.markdown("#### Select Simulation to Compare")
    
    if sweeps:
        sweep_options = {s['id']: f"{s['name']} (ID: {s['id']})" for s in sweeps}
        selected_sweep = st.selectbox(
            "Simulation Sweep",
            options=list(sweep_options.keys()),
            format_func=lambda x: sweep_options[x],
            help="Select the simulation sweep to compare against the trace"
        )
        
        if st.button("🔬 Run Validation", type="primary"):
            if not is_ior and not selected_trace:
                st.error("Please select a trace file first.")
            else:
                with st.spinner("Running validation..."):
                    sim_results = data_service.get_simulation_results(selected_sweep)
                    
                    if sim_results.empty:
                        st.error("No simulation results found for this sweep.")
                    else:
                        st.markdown("---")
                        st.markdown("#### Validation Results")
                        
                        # Check if this is IOR comparison
                        if is_ior and real_metrics:
                            # IOR-specific comparison
                            st.markdown("##### Metric Comparison (IOR)")
                            
                            comparison_data = []
                            
                            # Write Throughput
                            real_write_bw = real_metrics.get('write_bw_mean_mibs', 0)
                            sim_write_bw = sim_results['migration_bandwidth_gbs'].mean() * 1024  # Convert GB/s to MiB/s
                            write_bw_diff = ((sim_write_bw - real_write_bw) / real_write_bw * 100) if real_write_bw > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Write Throughput (MiB/s)",
                                "Simulated": f"{sim_write_bw:,.2f}",
                                "Real (IOR)": f"{real_write_bw:,.2f}",
                                "Difference": f"{write_bw_diff:+.1f}%",
                                "Status": "🟢" if abs(write_bw_diff) < 20 else "🔴"
                            })
                            
                            # Read Throughput
                            real_read_bw = real_metrics.get('read_bw_mean_mibs', 0)
                            # Simulation doesn't have read throughput, show N/A
                            comparison_data.append({
                                "Metric": "Read Throughput (MiB/s)",
                                "Simulated": "N/A",
                                "Real (IOR)": f"{real_read_bw:,.2f}",
                                "Difference": "N/A",
                                "Status": "⚪"
                            })
                            
                            # Write IOPS
                            real_write_iops = real_metrics.get('write_iops_mean', 0)
                            sim_write_iops = sim_results.get('write_iops', pd.Series([0])).mean() if 'write_iops' in sim_results.columns else 0
                            write_iops_diff = ((sim_write_iops - real_write_iops) / real_write_iops * 100) if real_write_iops > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Write IOPS",
                                "Simulated": f"{sim_write_iops:,.0f}" if sim_write_iops > 0 else "N/A",
                                "Real (IOR)": f"{real_write_iops:,.0f}",
                                "Difference": f"{write_iops_diff:+.1f}%" if sim_write_iops > 0 else "N/A",
                                "Status": "🟢" if 0 < abs(write_iops_diff) < 20 else "⚪" if sim_write_iops == 0 else "🔴"
                            })
                            
                            # Read IOPS
                            real_read_iops = real_metrics.get('read_iops_mean', 0)
                            comparison_data.append({
                                "Metric": "Read IOPS",
                                "Simulated": "N/A",
                                "Real (IOR)": f"{real_read_iops:,.0f}",
                                "Difference": "N/A",
                                "Status": "⚪"
                            })
                            
                            # Write Latency
                            real_write_lat = real_metrics.get('write_latency_mean_s', 0)
                            sim_write_lat = sim_results['latency_mean'].mean()
                            write_lat_diff = ((sim_write_lat - real_write_lat) / real_write_lat * 100) if real_write_lat > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Write Latency (s)",
                                "Simulated": f"{sim_write_lat:.6f}",
                                "Real (IOR)": f"{real_write_lat:.6f}",
                                "Difference": f"{write_lat_diff:+.1f}%",
                                "Status": "🟢" if abs(write_lat_diff) < 20 else "🔴"
                            })
                            
                            # Read Latency
                            real_read_lat = real_metrics.get('read_latency_mean_s', 0)
                            comparison_data.append({
                                "Metric": "Read Latency (s)",
                                "Simulated": "N/A",
                                "Real (IOR)": f"{real_read_lat:.6f}",
                                "Difference": "N/A",
                                "Status": "⚪"
                            })
                            
                            # Cache Hit Ratio (N/A for IOR)
                            sim_ch = sim_results['cache_hit_ratio'].mean()
                            comparison_data.append({
                                "Metric": "Cache Hit Ratio",
                                "Simulated": f"{sim_ch:.1%}",
                                "Real (IOR)": "N/A",
                                "Difference": "N/A",
                                "Status": "⚪"
                            })
                            
                            st.dataframe(comparison_data, width='stretch', hide_index=True)
                            
                            # Metric cards
                            st.markdown("##### Detailed Metrics")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Simulated**")
                                st.metric(
                                    "Write Throughput (MiB/s)",
                                    f"{sim_write_bw:,.2f}",
                                    delta=f"{write_bw_diff:+.1f}% vs IOR"
                                )
                                st.metric("Read Throughput (MiB/s)", "N/A")
                                st.metric(
                                    "Write Latency (s)",
                                    f"{sim_write_lat:.6f}",
                                    delta=f"{write_lat_diff:+.1f}% vs IOR"
                                )
                                st.metric("Read Latency (s)", "N/A")
                                st.metric(
                                    "Cache Hit Ratio",
                                    f"{sim_ch:.1%}"
                                )
                            
                            with col2:
                                st.markdown("**Real (IOR Benchmark)**")
                                st.metric("Write Throughput (MiB/s)", f"{real_write_bw:,.2f}")
                                st.metric("Read Throughput (MiB/s)", f"{real_read_bw:,.2f}")
                                st.metric("Write Latency (s)", f"{real_write_lat:.6f}")
                                st.metric("Read Latency (s)", f"{real_read_lat:.6f}")
                                st.metric("Cache Hit Ratio", "N/A")
                            
                            # Summary
                            st.markdown("##### Validation Summary")
                            passed = sum(1 for d in comparison_data if "🟢" in d["Status"])
                            comparable = sum(1 for d in comparison_data if "⚪" not in d["Status"])
                            failed = comparable - passed
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Comparable Metrics", comparable)
                            c2.metric("Passed", passed, delta_color="normal")
                            c3.metric("Failed", failed, delta_color="inverse")
                            
                            if failed == 0 and comparable > 0:
                                st.success("All comparable metrics within acceptable range (±20%)")
                            elif comparable == 0:
                                st.info("No directly comparable metrics available for IOR traces.")
                            else:
                                st.warning(f"{failed} metric(s) outside acceptable range")
                        
                        else:
                            # DMF/P4 comparison (original logic)
                            sim_metrics = {
                                'latency_mean': sim_results['latency_mean'].mean(),
                                'latency_p95': sim_results['latency_p95'].mean(),
                                'throughput_gbs': sim_results['migration_bandwidth_gbs'].mean(),
                                'cache_hit_ratio': sim_results['cache_hit_ratio'].mean(),
                            }
                            
                            comparison_data = []
                            
                            # Latency
                            real_lat = real_metrics.get('latency_mean', 0)
                            sim_lat = sim_metrics['latency_mean']
                            lat_diff = ((sim_lat - real_lat) / real_lat * 100) if real_lat > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Mean Latency (s)",
                                "Simulated": f"{sim_lat:.3f}",
                                "Real": f"{real_lat:.3f}" if real_lat > 0 else "N/A",
                                "Difference": f"{lat_diff:+.1f}%",
                                "Status": "🟢" if abs(lat_diff) < 20 else "🔴"
                            })
                            
                            # Throughput
                            real_tput = real_metrics.get('throughput_gbs', 0)
                            sim_tput = sim_metrics['throughput_gbs']
                            tput_diff = ((sim_tput - real_tput) / real_tput * 100) if real_tput > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Throughput (GB/s)",
                                "Simulated": f"{sim_tput:.4f}",
                                "Real": f"{real_tput:.4f}" if real_tput > 0 else "N/A",
                                "Difference": f"{tput_diff:+.1f}%",
                                "Status": "🟢" if abs(tput_diff) < 20 else "🔴"
                            })
                            
                            # Cache Hit
                            real_ch = real_metrics.get('cache_hit_ratio', 0)
                            sim_ch = sim_metrics['cache_hit_ratio']
                            ch_diff = ((sim_ch - real_ch) / real_ch * 100) if real_ch > 0 else 0
                            
                            comparison_data.append({
                                "Metric": "Cache Hit Ratio",
                                "Simulated": f"{sim_ch:.1%}",
                                "Real": f"{real_ch:.1%}" if real_ch > 0 else "N/A",
                                "Difference": f"{ch_diff:+.1f}%" if real_ch > 0 else "N/A",
                                "Status": "🟢" if abs(ch_diff) < 20 else "🔴"
                            })
                            
                            st.dataframe(comparison_data, width='stretch', hide_index=True)
                            
                            # Metric cards
                            st.markdown("##### Detailed Metrics")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Simulated**")
                                st.metric(
                                    "Mean Latency (s)",
                                    f"{sim_lat:.3f}",
                                    delta=f"{lat_diff:+.1f}% vs Real"
                                )
                                st.metric(
                                    "Throughput (GB/s)",
                                    f"{sim_tput:.4f}",
                                    delta=f"{tput_diff:+.1f}% vs Real"
                                )
                                st.metric(
                                    "Cache Hit Ratio",
                                    f"{sim_ch:.1%}",
                                    delta=f"{ch_diff:+.1f}%" if real_ch > 0 else None
                                )
                            
                            with col2:
                                st.markdown("**Real (from trace)**")
                                st.metric("Mean Latency (s)", f"{real_lat:.3f}" if real_lat > 0 else "N/A")
                                st.metric("Throughput (GB/s)", f"{real_tput:.4f}" if real_tput > 0 else "N/A")
                                st.metric("Cache Hit Ratio", f"{real_ch:.1%}" if real_ch > 0 else "N/A")
                            
                            # Summary
                            st.markdown("##### Validation Summary")
                            
                            passed = sum(1 for d in comparison_data if "🟢" in d["Status"])
                            failed = len(comparison_data) - passed
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Metrics Compared", len(comparison_data))
                            c2.metric("Passed", passed, delta_color="normal")
                            c3.metric("Failed", failed, delta_color="inverse")
                            
                            if failed == 0:
                                st.success("All metrics within acceptable range (±20%)")
                            else:
                                st.warning(f"{failed} metric(s) outside acceptable range")
                        
    else:
        st.warning("No simulation sweeps found. Run a parameter sweep first.")

# ============================================================================
# TAB 2: PARAMETER SWEEP ANALYSIS
# ============================================================================

with tab_sweep:
    st.markdown("### Parameter Sweep Analysis")
    
    st.info("""
    **What this does:**
    
    This tab analyzes the results of a parameter sweep to verify that the simulator produces expected relationships between parameters.
    
    **What to look for:**
    - Higher age threshold → LESS data archived (files stay in cache longer)
    - More frequent archiver → MORE bandwidth used, MORE media usage
    - Monotonic relationships indicate the model is behaving correctly
    """)
    
    st.markdown("---")
    
    if sweeps:
        sweep_options = {s['id']: f"{s['name']} (ID: {s['id']})" for s in sweeps}
        selected_sweep = st.selectbox(
            "Select Sweep to Analyze",
            options=list(sweep_options.keys()),
            format_func=lambda x: sweep_options[x],
            help="Choose a sweep to analyze"
        )
        
        results_df = data_service.get_simulation_results(selected_sweep)
        
        if not results_df.empty:
            analysis = data_service.compare_parameter_sweep(results_df)
            
            # Age Threshold Analysis
            if 'age_thresholds_tested' in analysis:
                st.markdown("#### Age Threshold Relationships")
                
                ages = analysis['age_thresholds_tested']
                st.write(f"**Age thresholds tested:** {ages}")
                st.write("**Expected:** Higher age thresholds → LESS archived data, LESS bandwidth")
                
                age_groups = results_df.groupby('age_threshold_minutes').agg({
                    'archived_data_tb': 'mean',
                    'migration_bandwidth_gbs': 'mean',
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Archived Data vs Age Threshold")
                    archived_trend = analysis.get('archived_data_trend', 'unknown')
                    if archived_trend == "decreasing":
                        st.success("✓ Model correctly archives LESS data with higher age threshold")
                    else:
                        st.warning(f"⚠ Unexpected trend: {archived_trend}")
                    st.line_chart(age_groups.set_index('age_threshold_minutes')['archived_data_tb'])
                
                with col2:
                    st.markdown("##### Bandwidth (GB/s) vs Age Threshold")
                    bandwidth_trend = analysis.get('bandwidth_trend', 'unknown')
                    if bandwidth_trend == "decreasing":
                        st.success("✓ Model correctly uses LESS bandwidth with higher age threshold")
                    else:
                        st.warning(f"⚠ Unexpected trend: {bandwidth_trend}")
                    st.line_chart(age_groups.set_index('age_threshold_minutes')['migration_bandwidth_gbs'])
                
                st.markdown("##### Summary Table")
                st.dataframe(age_groups, width='stretch', hide_index=True)
            
            # Archiver Interval Analysis
            if 'archiver_intervals_tested' in analysis:
                st.markdown("---")
                st.markdown("#### Archiver Interval Relationships")
                
                intervals = analysis['archiver_intervals_tested']
                st.write(f"**Archiver intervals tested:** {intervals}")
                st.write("**Expected:** Lower interval (more frequent archiver) → MORE bandwidth, MORE media usage")
                
                arch_groups = results_df.groupby('archiver_interval_minutes').agg({
                    'archived_data_tb': 'mean',
                    'migration_bandwidth_gbs': 'mean',
                    'tape_cartridges_used': 'mean',
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Archived Data (TB) vs Interval")
                    archived_trend = analysis.get('archiver_archived_trend', 'unknown')
                    if archived_trend == "decreasing":
                        st.success("✓ Model correctly archives MORE data with more frequent archiver")
                    else:
                        st.warning(f"⚠ Unexpected trend: {archived_trend}")
                    st.line_chart(arch_groups.set_index('archiver_interval_minutes')['archived_data_tb'])
                
                with col2:
                    st.markdown("##### Bandwidth (GB/s) vs Interval")
                    bandwidth_trend = analysis.get('archiver_bandwidth_trend', 'unknown')
                    if bandwidth_trend == "decreasing":
                        st.success("✓ Model correctly uses MORE bandwidth with more frequent archiver")
                    else:
                        st.warning(f"⚠ Unexpected trend: {bandwidth_trend}")
                    st.line_chart(arch_groups.set_index('archiver_interval_minutes')['migration_bandwidth_gbs'])
                
                st.markdown("##### Tape Cartridges Used vs Interval")
                cartridges_trend = analysis.get('archiver_cartridges_trend', 'unknown')
                if cartridges_trend == "decreasing":
                    st.success("✓ Model correctly uses MORE media with more frequent archiver")
                else:
                    st.warning(f"⚠ Unexpected trend: {cartridges_trend}")
                st.line_chart(arch_groups.set_index('archiver_interval_minutes')['tape_cartridges_used'])
                
                st.markdown("##### Summary Table")
                st.dataframe(arch_groups, width='stretch', hide_index=True)
            
            if 'age_thresholds_tested' not in analysis and 'archiver_intervals_tested' not in analysis:
                st.warning("This sweep doesn't vary age thresholds or archiver intervals.")
        else:
            st.warning("No results for this sweep.")
    else:
        st.warning("No sweeps found.")

# ============================================================================
# TAB 3: IOR BENCHMARK VALIDATION
# ============================================================================

with tab_ior:
    st.markdown("### IOR Benchmark Validation")
    
    st.info("""
    **What this does:**
    
    This tab validates simulation results against IOR benchmark data. IOR (Interleaved Or Random) is a standard benchmark for parallel file systems.
    
    **Workflow:**
    1. Parse an IOR benchmark output file (Parse page)
    2. Run simulations using IOR trace files
    3. Select the simulation sweep here to compare
    """)
    
    st.markdown("---")
    
    # Check for IOR summary
    traces_dir = Path("./traces")
    ior_summary_file = traces_dir / "ior_summary.csv"
    ior_trace_files = list(traces_dir.glob("trace_ior_*.jsonl"))
    
    if not ior_summary_file.exists():
        st.warning("No IOR data found. Parse an IOR file first in the Parse page.")
    else:
        # Load IOR summary
        ior_df = pd.read_csv(ior_summary_file)
        
        st.markdown("#### IOR Benchmark Summary")
        st.dataframe(ior_df.head(20), width='stretch')
        
        # Performance by configuration
        if 'config' in ior_df.columns:
            st.markdown("#### Performance by Configuration")
            
            config_stats = ior_df.groupby('config').agg({
                'bw_mib_s': ['mean', 'std'],
                'iops': ['mean', 'std'],
            }).round(2)
            
            st.dataframe(config_stats, width='stretch')
            
            # Charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("##### Bandwidth by Config (MiB/s)")
                bw_by_config = ior_df.groupby('config')['bw_mib_s'].mean()
                st.bar_chart(bw_by_config)
            
            with c2:
                st.markdown("##### IOPS by Config")
                iops_by_config = ior_df.groupby('config')['iops'].mean()
                st.bar_chart(iops_by_config)
        
        # Compare with simulation
        st.markdown("---")
        st.markdown("#### Compare with Simulation")
        
        if sweeps:
            sweep_options = {s['id']: f"{s['name']} (ID: {s['id']})" for s in sweeps}
            selected_sweep = st.selectbox(
                "Select Simulation to Compare",
                options=list(sweep_options.keys()),
                format_func=lambda x: sweep_options[x]
            )
            
            alpha = st.slider(
                "Significance Level (alpha)",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="p-value threshold for statistical significance"
            )
            
            if st.button("🔬 Run IOR Validation"):
                try:
                    from validation.ior_validation import IORValidationService
                    from db.database import SweepDB
                    
                    with st.spinner("Validating..."):
                        sweep_db = SweepDB()
                        sim_results = sweep_db.get_sweep_results(selected_sweep)
                        
                        if not sim_results:
                            st.warning("No simulation results found.")
                        else:
                            sim_df = pd.DataFrame(sim_results)
                            
                            validator = IORValidationService()
                            comparison = validator.compare_results(ior_df, sim_df, alpha)
                            
                            if comparison.empty:
                                st.warning("No matching configurations found. Make sure simulations used IOR traces.")
                            else:
                                # Summary
                                summary = validator.generate_summary(comparison)
                                
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Comparisons", summary.get('total_comparisons', 0))
                                c2.metric("Passed", summary.get('passed', 0))
                                c3.metric("Failed", summary.get('failed', 0))
                                c4.metric("Pass Rate", f"{summary.get('pass_rate', 0):.1f}%")
                                
                                st.markdown("##### Comparison Results")
                                st.dataframe(comparison, width='stretch', hide_index=True)
                                
                                # Charts
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("##### Difference by Config (%)")
                                    diff_by_config = comparison.groupby('config')['difference_pct'].mean()
                                    st.bar_chart(diff_by_config)
                                with c2:
                                    st.markdown("##### Pass Rate by Config (%)")
                                    pass_by_config = comparison.groupby('config')['passed'].mean() * 100
                                    st.bar_chart(pass_by_config)
                                
                                # Export
                                csv = comparison.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Comparison CSV",
                                    csv,
                                    "ior_comparison_results.csv",
                                    "text/csv"
                                )
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
        else:
            st.info("Run a simulation first to compare with IOR benchmarks.")

# ============================================================================
# TAB 4: SUMMARY
# ============================================================================

with tab_summary:
    st.markdown("### Validation Summary")
    
    st.markdown("""
    #### How Validation Works
    
    **1. Simulation vs Trace Comparison (DMF/P4)**
    
    Compares simulation output metrics against real-world trace data:
    - **Latency**: Mean response time
    - **Throughput**: Data migration bandwidth (GB/s)
    - **Cache Hit Ratio**: Cache effectiveness (%)
    
    **Pass criteria**: Within ±20% of real-world values
    
    **2. Simulation vs Trace Comparison (IOR)**
    
    IOR benchmarks provide direct filesystem performance metrics:
    - **Write Throughput (MiB/s)**: Average write bandwidth
    - **Read Throughput (MiB/s)**: Average read bandwidth
    - **Write IOPS**: Input/output operations per second (writes)
    - **Read IOPS**: Input/output operations per second (reads)
    - **Write Latency (s)**: Average time per write operation
    - **Read Latency (s)**: Average time per read operation
    - **Cache Hit Ratio**: N/A for IOR (benchmark traffic, not HSM workload)
    
    **3. Parameter Sweep Analysis**
    
    Validates expected monotonic relationships:
    - **Age Threshold vs Archived**: Higher threshold → LESS data archived
    - **Age Threshold vs Bandwidth**: Higher threshold → LESS bandwidth used
    - **Archiver Interval vs Bandwidth**: More frequent → MORE bandwidth used
    - **Archiver Interval vs Media**: More frequent → MORE media usage
    
    **4. IOR Benchmark Validation**
    
    Statistical comparison of simulation vs benchmark results:
    - Uses t-test to compare means
    - Pass if p-value ≥ alpha OR |difference| < 20%
    
    ---
    
    **Significance Level:** p < 0.05 = statistically significant difference
    
    **Status Indicators:**
    - 🟢 = Within acceptable range (±20%)
    - 🔴 = Outside acceptable range
    - ⚪ = Not applicable / No comparison available
    
    A passing validation means the simulator accurately represents real-world behavior.
    """)
