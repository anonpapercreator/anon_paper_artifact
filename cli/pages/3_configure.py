"""
DESCASSI - Configure Page
=========================
Configure simulation parameters including hardware specs, policy, cost, and workload.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DESCASSI - Configure",
    page_icon="⚙️",
    layout="wide")

st.title("⚙️ Configure Simulation")

cfg = st.session_state.config

# Run derive_parameters to get current values
cfg.declarative_policy.derive_parameters()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["ESS 3500", "TS1160", "DMF Policy", "Cost", "Workload"])

with tab1:
    st.markdown("### IBM ESS 3500 Configuration")
    
    nsd_nodes = st.slider("NSD Nodes", 1, 32, cfg.ess3500.nsd_nodes, help="Number of NSD servers")
    throughput = st.slider("Aggregate Throughput (GiB/s)", 1.0, 200.0, cfg.ess3500.aggregate_throughput_gib_s, 0.5, help="Total throughput across all nodes")
    capacity = st.slider("Usable Capacity (TiB)", 1.0, 5000.0, cfg.ess3500.usable_capacity_tib, 1.0, help="Usable capacity")
    
    network = st.selectbox(
        "Network Type",
        ["100Gb Ethernet", "100Gb HDR InfiniBand", "200Gb HDR InfiniBand"],
        index=2
    )
    
    if st.button("Apply ESS 3500 Settings"):
        cfg.ess3500.nsd_nodes = nsd_nodes
        cfg.ess3500.aggregate_throughput_gib_s = throughput
        cfg.ess3500.usable_capacity_tib = capacity
        cfg.ess3500.network_type = network
        st.success("ESS 3500 settings updated!")

with tab2:
    st.markdown("### IBM TS1160 Configuration")
    
    drives = st.slider("Number of Drives", 1, 64, cfg.ts1160.n_drives, help="Concurrent tape drives")
    rate = st.slider("Native Rate (MB/s)", 100, 800, int(cfg.ts1160.native_rate_bytes_per_s / 1024 / 1024), help="Drive native rate")
    
    if st.button("Apply TS1160 Settings"):
        cfg.ts1160.n_drives = drives
        cfg.ts1160.native_rate_bytes_per_s = rate * 1024 * 1024
        st.success("TS1160 settings updated!")

with tab3:
    st.markdown("### DMF Policy Configuration")
    
    age_threshold = st.slider("Age Threshold (minutes)", 1, 1440, cfg.dmf_policy.age_threshold_minutes, help="Files older than this are migration candidates")
    archiver_interval = st.slider("Archiver Interval (minutes)", 1, 60, cfg.dmf_policy.archiver_interval_minutes, help="How often the archiver scans for files to migrate")
    hwm = st.slider("High Water Mark (%)", 50, 99, int(cfg.dmf_policy.hwm_fraction * 100), help="Trigger migration when cache usage exceeds this")
    lwm = st.slider("Low Water Mark (%)", 30, 95, int(cfg.dmf_policy.lwm_fraction * 100), help="Stop migration when cache usage reaches this")
    large_file = st.slider("Large File Threshold (GB)", 1, 1000, cfg.dmf_policy.large_file_threshold_gb, help="Files larger than this are 'large files'")
    
    if st.button("Apply Policy Settings"):
        cfg.dmf_policy.age_threshold_minutes = age_threshold
        cfg.dmf_policy.archiver_interval_minutes = archiver_interval
        cfg.dmf_policy.hwm_fraction = hwm / 100
        cfg.dmf_policy.lwm_fraction = lwm / 100
        cfg.dmf_policy.large_file_threshold_gb = large_file
        st.success("Policy settings updated!")

with tab4:
    st.markdown("### Cost Configuration")
    
    # Info box explaining presets
    st.info("""
    **ℹ️ What does this preset do?**
    
    This preset controls **COST MODELING** only. It does **NOT** affect simulation timing or migration decisions.
    
    The simulation behavior (when files migrate, recall latency, etc.) is controlled by the **DMF Policy** tab settings (HWM, LWM, Age Threshold, Archiver Interval).
    
    The preset only affects how costs are calculated for reporting purposes.
    """)
    
    preset = st.selectbox(
        "Policy Preset",
        ["maximum_performance", "balanced", "archive_optimized", "user_defined"],
        index=1
    )
    
    # Preset descriptions
    preset_descriptions = {
        "maximum_performance": "Minimize recall latency, accept higher storage cost",
        "balanced": "Balance between cost and performance (default)",
        "archive_optimized": "Minimize tape/media costs, accept slower recalls",
        "user_defined": "Customize all parameters manually"
    }
    st.markdown(f"**{preset_descriptions.get(preset, '')}**")
    
    st.markdown("---")
    st.markdown("#### Preset Comparison")
    
    # Show preset comparison table
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**maximum_performance**")
        st.caption("Cache Priority: 100%")
        st.caption("Prefetch: Full")
        st.caption("Cost Impact: Highest")
    with col2:
        st.markdown("**balanced**")
        st.caption("Cache Priority: 70%")
        st.caption("Prefetch: Moderate")
        st.caption("Cost Impact: Medium")
    with col3:
        st.markdown("**archive_optimized**")
        st.caption("Cache Priority: 30%")
        st.caption("Prefetch: Disabled")
        st.caption("Cost Impact: Lowest")
    with col4:
        st.markdown("**user_defined**")
        st.caption("Cache Priority: Custom")
        st.caption("Prefetch: Custom")
        st.caption("Cost Impact: Variable")
    
    st.markdown("---")
    
    # Show derived parameters (read-only)
    st.markdown("#### Derived Parameters (calculated from preset)")
    
    derived_age_days = cfg.declarative_policy.age_threshold_seconds / 86400 if cfg.declarative_policy.age_threshold_seconds else 0
    derived_batch_mb = cfg.declarative_policy.migration_batch_mb or 0
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Cache Priority", f"{cfg.declarative_policy.cache_priority:.2f}")
        st.caption("Used in cost calculations")
    with c2:
        st.metric("Derived Age Threshold", f"{derived_age_days:.1f} days")
        st.caption("Reference only (not used in simulation)")
    
    st.markdown("---")
    
    # Override option (for age threshold override)
    override_age = st.checkbox("Override age threshold", value=False, key="override_age")
    if override_age:
        override_value = st.number_input(
            "Override age threshold (minutes)",
            min_value=1,
            max_value=1440,
            value=cfg.dmf_policy.age_threshold_minutes,
            help="This overrides the age threshold in the DMF Policy tab"
        )
        if st.button("Apply Override"):
            cfg.dmf_policy.age_threshold_minutes = override_value
            st.success("Age threshold overridden!")
    
    st.markdown("---")
    st.markdown("#### Cost Settings")
    
    power_cost = st.number_input(
        "Power Cost ($/kW/month)",
        min_value=0.0,
        max_value=1000.0,
        value=cfg.cost.power_cost_per_kw_month,
        step=1.0,
        help="Your actual power cost"
    )
    
    if st.button("Apply All Settings"):
        cfg.declarative_policy.preset = preset
        cfg.declarative_policy.cache_priority = {
            "maximum_performance": 1.0,
            "balanced": 0.7,
            "archive_optimized": 0.3,
            "user_defined": cfg.declarative_policy.cache_priority
        }.get(preset, 0.5)
        cfg.declarative_policy.prefetch_aggression = {
            "maximum_performance": 1.0,
            "balanced": 0.5,
            "archive_optimized": 0.0,
            "user_defined": cfg.declarative_policy.prefetch_aggression
        }.get(preset, 0.5)
        cfg.declarative_policy.derive_parameters()
        cfg.cost.power_cost_per_kw_month = power_cost
        st.success("All settings updated!")

with tab5:
    st.markdown("### Workload Configuration")
    
    # Get available traces
    traces_dir = Path("./traces")
    available_traces = []
    if traces_dir.exists():
        available_traces = [str(p) for p in traces_dir.glob("*.jsonl")]
    
    # Workload mode selection
    workload_mode = st.radio(
        "Workload Mode",
        ["synthetic", "trace_replay"],
        index=0 if cfg.workload_mode == "synthetic" else 1,
        help="synthetic: Generate synthetic workload | trace_replay: Replay from trace file"
    )

    st.info(
        "**Multi-tenant noisy-neighbour mode** is configured via YAML, not "
        "this UI. Add `workload.multi_tenant.enabled: true` to your config "
        "(see `config/default_config.yaml` for the full schema and "
        "`workload/multi_tenant.py` for the model). The probabilistic-model "
        "page has a tab that shows the per-tenant spec distribution at any "
        "noise level β so you can preview the heterogeneity before running "
        "a full DES."
    )
    
    # Trace file selection (only if trace_replay selected)
    trace_file = None
    if workload_mode == "trace_replay":
        if available_traces:
            trace_idx = 0
            if cfg.trace_file:
                for i, t in enumerate(available_traces):
                    if str(cfg.trace_file) in t:
                        trace_idx = i
                        break
            
            selected_trace = st.selectbox(
                "Trace File",
                available_traces,
                index=trace_idx,
                help="Select trace file to replay"
            )
            trace_file = Path(selected_trace)
            
            # Show trace info
            from validation.data_service import ValidationDataService
            ds = ValidationDataService()
            if "dmf" in selected_trace.lower():
                metrics = ds.get_metrics_from_dmf_trace(selected_trace)
            elif "p4" in selected_trace.lower():
                metrics = ds.get_metrics_from_p4_trace(selected_trace)
            else:
                metrics = None
            
            if metrics:
                st.caption(f"Trace: {metrics.get('total_events', 0)} events, {metrics.get('duration_hours', 0):.1f} hours")
        else:
            st.warning("No trace files found in ./traces/")
    
    # Simulation duration
    duration = st.number_input(
        "Simulation Duration (hours)",
        min_value=0.1,
        max_value=168.0,
        value=min(cfg.sim_duration_hours, 24.0),
        step=1.0,
        help="How long to run the simulation"
    )
    
    if st.button("Apply Workload Settings"):
        cfg.workload_mode = workload_mode
        cfg.trace_file = trace_file
        cfg.sim_duration_hours = duration
        st.success("Workload settings updated!")

st.markdown("---")
st.markdown("### Current Configuration Summary")

st.code(f"""
ESS 3500:
  Nodes: {cfg.ess3500.nsd_nodes}
  Throughput: {cfg.ess3500.aggregate_throughput_gib_s} GiB/s
  Capacity: {cfg.ess3500.usable_capacity_tib} TiB
  
TS1160:
  Drives: {cfg.ts1160.n_drives}
  
DMF Policy:
  Age Threshold: {cfg.dmf_policy.age_threshold_minutes} min
  Archiver Interval: {cfg.dmf_policy.archiver_interval_minutes} min
  HWM: {cfg.dmf_policy.hwm_fraction*100:.0f}%
  LWM: {cfg.dmf_policy.lwm_fraction*100:.0f}%
  
Workload:
  Mode: {cfg.workload_mode}
  Trace: {cfg.trace_file}
  Duration: {cfg.sim_duration_hours} hours

Cost Settings:
  Policy Preset: {cfg.declarative_policy.preset}
  Cache Priority: {cfg.declarative_policy.cache_priority:.2f}
  Power Cost: ${cfg.cost.power_cost_per_kw_month:.2f}/kW/month
""")
