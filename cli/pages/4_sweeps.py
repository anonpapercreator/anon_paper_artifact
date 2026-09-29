"""
DESCASSI - Parameter Sweeps Page
"""

import streamlit as st
from config.models import SimulationConfig, SweepConfig
from core.sweeper import SweepRunnerParallel, get_max_workers

st.set_page_config(
    page_title="DESCASSI - Parameter Sweeps",
    page_icon="🔬",
    layout="wide")

st.title("🔬 Parameter Sweeps")

st.markdown("""
Run parameter sweeps to understand the impact of different policy configurations.
This will show how media consumption and bandwidth usage change with different archival intervals.
""")

# Sweep configuration
st.markdown("### Configure Sweep")

col1, col2 = st.columns(2)

with col1:
    sweep_name = st.text_input("Sweep Name", value="archiver_interval_sweep")
    sweep_desc = st.text_area("Description", value="Testing different archiver intervals to verify bandwidth and media utilization relationship")

with col2:
    replications = st.number_input("Replications per Config", 1, 100, 20, help="Number of simulation runs per configuration")

st.markdown("#### Parameters to Sweep")

sweep_age = st.checkbox("Age Threshold (minutes)", value=False)
sweep_hwm = st.checkbox("HWM (%)", value=False)
sweep_priority = st.checkbox("Cache Priority", value=False)
sweep_archiver = st.checkbox("Archiver Interval (minutes)", value=True)

age_values = None
hwm_values = None
priority_values = None
archiver_values = None

if sweep_age:
    age_values = st.multiselect(
        "Age Threshold Values",
        [5, 10, 20, 30, 40, 50, 60, 120, 240, 480, 720, 1440],
        default=[5, 10, 20, 30, 60, 120, 240, 1440]
    )

if sweep_hwm:
    hwm_values = st.multiselect(
        "HWM Values",
        [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        default=[0.70, 0.80, 0.90]
    )

if sweep_priority:
    priority_values = st.multiselect(
        "Cache Priority Values",
        [0.1, 0.3, 0.5, 0.7, 1.0],
        default=[0.3, 0.5, 0.7, 1.0]
    )

if sweep_archiver:
    archiver_values = st.multiselect(
        "Archiver Interval Values (minutes)",
        [1, 5, 15, 30, 60, 120],
        default=[1, 5, 15, 30, 60, 120],
        help="How often the archiver scans for files to migrate. More frequent = higher bandwidth, more media usage."
    )
    archiver_manual = st.text_input(
        "Or enter custom values (comma-separated)",
        value="",
        placeholder="e.g., 1, 5, 30, 60"
    )
    if archiver_manual.strip():
        try:
            archiver_values = [int(x.strip()) for x in archiver_manual.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid input. Please enter comma-separated integers.")

# Preview
st.markdown("### Preview")
combinations = []
if sweep_age and age_values:
    for v in age_values:
        combinations.append({"age_threshold_minutes": v})
if sweep_hwm and hwm_values:
    for v in hwm_values:
        combinations.append({"hwm_fraction": v})
if sweep_priority and priority_values:
    for v in priority_values:
        combinations.append({"cache_priority": v})
if sweep_archiver and archiver_values:
    for v in archiver_values:
        combinations.append({"archiver_interval_minutes": v})

total_runs = len(combinations) * replications
st.info(f"Total configurations to run: {len(combinations)} × {replications} = {total_runs} runs")

# Worker configuration
st.markdown("#### ⚡ Parallel Execution")
col_workers, col_est = st.columns(2)

with col_workers:
    max_workers = get_max_workers()
    n_workers = st.number_input(
        "Parallel Workers",
        min_value=1,
        max_value=max_workers,
        value=min(4, max_workers),
        help=f"Run simulations in parallel. Max available: {max_workers}"
    )

with col_est:
    st.metric("Estimated Speedup", f"~{n_workers}x faster")
    est_time = (total_runs * 30) // (60 * n_workers) if total_runs > 0 else 0
    st.caption(f"Est. time: ~{est_time} minutes (sequential: ~{total_runs * 30 // 60} min)")

# Run button
st.markdown("---")

if st.button("🚀 Start Parameter Sweep", type="primary", disabled=len(combinations) == 0):
    if len(combinations) == 0:
        st.warning("Please select at least one parameter to sweep")
    else:
        sweep_config = SweepConfig(
            name=sweep_name,
            description=sweep_desc,
            sweep_age_threshold=sweep_age,
            age_threshold_values=age_values or [],
            sweep_hwm=sweep_hwm,
            hwm_values=hwm_values or [],
            sweep_cache_priority=sweep_priority,
            cache_priority_values=priority_values or [],
            replications=replications,
            n_workers=n_workers
        )
        
        with st.spinner("Running parameter sweep... This may take a few minutes."):
            overall_progress = st.progress(0)
            overall_status = st.empty()
            worker_progress = st.container()
            
            try:
                def progress_callback(progress, current, total, worker_id, combo, rep, result):
                    overall_progress.progress(progress)
                    combo_str = ", ".join(f"{k}={v}" for k, v in combo.items()) if combo else "N/A"
                    overall_status.text(f"Overall: {current}/{total} ({progress*100:.1f}%) - Worker {worker_id}: {combo_str} (Rep {rep+1}/{replications})")
                    
                    with worker_progress:
                        st.text(f"Worker {worker_id}: Config {combo_str}, Rep {rep+1}")
                
                base_config = st.session_state.config
                runner = SweepRunnerParallel(base_config, sweep_config, n_workers=n_workers)
                results = runner.run(progress_callback=progress_callback)
                
                overall_progress.empty()
                overall_status.empty()
                worker_progress.empty()
                
                st.session_state.sweep_results = results
                
                st.success(f"✅ Sweep completed! {len(results)} results saved.")
                
            except Exception as e:
                overall_progress.empty()
                overall_status.empty()
                worker_progress.empty()
                st.error(f"Error running sweep: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
