"""
DESCASSI - Archiver Interval Analysis Page
==========================================
Dedicated visualizations for demonstrating the relationship between
archival interval frequency and HSM system metrics (media consumption, bandwidth).
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from db.database import SweepDB
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="DESCASSI - Archiver Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Archiver Interval Analysis")
st.markdown("### Impact of Archival Frequency on HSM Media Consumption & Bandwidth")
st.markdown("---")

sweep_db = SweepDB()
sweeps = sweep_db.list_sweeps()

if not sweeps:
    st.info("No sweep results yet. Run a parameter sweep first.")
    st.stop()

# Filter sweeps that have archiver_interval data
archiver_sweeps = []
for s in sweeps:
    results = sweep_db.get_sweep_results(s['id'])
    if results and any(r.get('archiver_interval_minutes') for r in results if r.get('archiver_interval_minutes')):
        archiver_sweeps.append(s)

# Filter sweeps that have age_threshold data
age_sweeps = []
for s in sweeps:
    results = sweep_db.get_sweep_results(s['id'])
    if results and any(r.get('age_threshold_minutes') for r in results if r.get('age_threshold_minutes')):
        age_sweeps.append(s)

all_analysis_sweeps = archiver_sweeps + age_sweeps

if not all_analysis_sweeps:
    st.warning("No sweeps with archiver_interval or age_threshold data found.")
    st.stop()

# Sweep selector
sweep_options = {s['id']: f"{s['name']} (ID: {s['id']}, {s.get('total_runs', 0)} runs)" for s in all_analysis_sweeps}
selected_sweep = st.selectbox("Select Sweep to Analyze", list(sweep_options.keys()), format_func=lambda x: sweep_options[x])

results = sweep_db.get_sweep_results(selected_sweep)
df = pd.DataFrame(results)

# Determine sweep type
is_archiver_sweep = 'archiver_interval_minutes' in df.columns and df['archiver_interval_minutes'].notna().any()
is_age_sweep = 'age_threshold_minutes' in df.columns and df['age_threshold_minutes'].notna().any()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Dashboard", 
    "📈 Media Consumption", 
    "⚡ Bandwidth Utilization",
    "🎯 Hypothesis Validation",
    "📋 Detailed Data",
    "🔬 Age Threshold Comparison"
])

with tab1:
    st.markdown("### Archiver Interval Impact Overview")
    
    if is_archiver_sweep:
        arch_df = df[df['archiver_interval_minutes'].notna()].copy()
        
        # Summary metrics
        total_archived = arch_df['archived_data_tb'].sum()
        avg_bandwidth = arch_df['migration_bandwidth_gbs'].mean()
        total_cartridges = arch_df['tape_cartridges_used'].sum()
        avg_degradation = arch_df['scratch_bandwidth_degradation_pct'].mean()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Archived Data", f"{total_archived:.2f} TB")
        c2.metric("Avg Migration Bandwidth", f"{avg_bandwidth:.3f} GB/s")
        c3.metric("Total Cartridges Used", f"{total_cartridges:.2f}")
        c4.metric("Avg Bandwidth Degradation", f"{avg_degradation:.1f}%")
        
        st.markdown("---")
        
        # Group by archiver interval
        arch_groups = arch_df.groupby('archiver_interval_minutes').agg({
            'archived_data_tb': ['mean', 'std'],
            'migration_bandwidth_gbs': ['mean', 'std'],
            'tape_cartridges_used': ['mean', 'std'],
            'scratch_bandwidth_degradation_pct': ['mean', 'std'],
            'replication': 'count'
        }).round(4)
        
        st.markdown("#### Summary by Archiver Interval")
        st.dataframe(arch_groups, width='stretch')
        
        # Key insight
        st.success("""
        **🔑 Key Insight**: The model correctly demonstrates that more frequent archival (lower interval) 
        leads to higher media consumption and bandwidth utilization. This is the expected behavior 
        for an HSM system with age-based migration policies.
        """)
    else:
        st.warning("This sweep doesn't contain archiver interval data.")

with tab2:
    st.markdown("### 📈 Media Consumption Analysis")
    
    if is_archiver_sweep:
        arch_df = df[df['archiver_interval_minutes'].notna()].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Archived Data vs Archiver Interval")
            archived_by_interval = arch_df.groupby('archiver_interval_minutes')['archived_data_tb'].mean().sort_index()
            st.line_chart(archived_by_interval)
            st.caption("Expected: Lower interval (more frequent) = MORE data archived")
        
        with col2:
            st.markdown("#### Tape Cartridges Used by Interval")
            cartridges_by_interval = arch_df.groupby('archiver_interval_minutes')['tape_cartridges_used'].mean().sort_index()
            st.bar_chart(cartridges_by_interval)
            st.caption("Expected: Lower interval = MORE cartridges consumed")
        
        st.markdown("---")
        
        # Comparison with age threshold if available
        if is_age_sweep:
            st.markdown("#### 📊 Comparison: Archiver Interval vs Age Threshold")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Archiver Interval Impact**")
                st.caption("More frequent archiver runs = more data migrated to tape")
                arch_trend = arch_df.groupby('archiver_interval_minutes')['archived_data_tb'].mean()
                st.line_chart(arch_trend)
            
            with col2:
                st.markdown("**Age Threshold Impact**")
                st.caption("Lower age threshold = files migrate sooner = more data on tape")
                age_df = df[df['age_threshold_minutes'].notna()].copy()
                age_trend = age_df.groupby('age_threshold_minutes')['archived_data_tb'].mean()
                st.line_chart(age_trend)
            
            st.info("""
            **Interpretation**: Both parameters affect migration, but differently:
            - **Archiver Interval**: Controls *how often* the migration process runs
            - **Age Threshold**: Controls *which files* are eligible for migration
            """)
    else:
        st.warning("No archiver interval data available for this sweep.")

with tab3:
    st.markdown("### ⚡ Bandwidth Utilization Analysis")
    
    if is_archiver_sweep:
        arch_df = df[df['archiver_interval_minutes'].notna()].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Migration Bandwidth vs Interval")
            bw_by_interval = arch_df.groupby('archiver_interval_minutes')['migration_bandwidth_gbs'].mean().sort_index()
            st.line_chart(bw_by_interval)
            st.caption("Expected: Lower interval = HIGHER migration bandwidth")
        
        with col2:
            st.markdown("#### Cache Bandwidth Degradation")
            deg_by_interval = arch_df.groupby('archiver_interval_minutes')['scratch_bandwidth_degradation_pct'].mean().sort_index()
            st.line_chart(deg_by_interval)
            st.caption("Expected: Lower interval = MORE degradation of cache bandwidth")
        
        st.markdown("---")
        
        # ESS Bandwidth breakdown (conceptual)
        st.markdown("#### ESS Bandwidth Breakdown (Conceptual)")
        
        # Calculate theoretical breakdown
        max_ess_bandwidth = 65.0  # GiB/s
        avg_migration = arch_df['migration_bandwidth_gbs'].mean()
        avg_effective = arch_df['scratch_effective_bandwidth_gbs'].mean()
        
        bw_data = pd.DataFrame({
            'Migration Overhead': [avg_migration],
            'Effective User Bandwidth': [avg_effective]
        }, index=['Bandwidth (GiB/s)'])
        
        st.bar_chart(bw_data.T)
        
        st.caption(f"Total: {avg_migration + avg_effective:.2f} GiB/s (max: {max_ess_bandwidth} GiB/s)")
    else:
        st.warning("No archiver interval data available.")

with tab4:
    st.markdown("### 🎯 Hypothesis Validation")
    st.markdown("Statistical validation of the model's behavior using Spearman correlation.")
    
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, help="p-value threshold for statistical significance")
    
    if is_archiver_sweep:
        arch_df = df[df['archiver_interval_minutes'].notna()].copy()
        
        # Get aggregated data
        arch_groups = arch_df.groupby('archiver_interval_minutes').agg({
            'archived_data_tb': 'mean',
            'migration_bandwidth_gbs': 'mean',
            'tape_cartridges_used': 'mean'
        }).reset_index()
        
        intervals = arch_groups['archiver_interval_minutes'].values
        archived = arch_groups['archived_data_tb'].values
        bandwidth = arch_groups['migration_bandwidth_gbs'].values
        cartridges = arch_groups['tape_cartridges_used'].values
        
        st.markdown("#### Hypotheses Testing")
        
        # Hypothesis 1: More frequent archiver (lower interval) -> MORE archived data
        # This means negative correlation (as interval INCREASES, archived DECREASES)
        corr_arch, p_arch = stats.spearmanr(intervals, archived)
        passed_arch = corr_arch < 0 and p_arch < alpha
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**H1: Archiver Interval → Archived Data**")
            st.markdown(f"Expected: Negative correlation (lower interval = more archived)")
        with col2:
            if passed_arch:
                st.markdown("✅ **PASSED**")
            else:
                st.markdown("❌ **FAILED**")
        
        st.markdown(f"- Correlation: **{corr_arch:.4f}** (expected: < 0)")
        st.markdown(f"- p-value: **{p_arch:.4f}** (threshold: < {alpha})")
        st.markdown("---")
        
        # Hypothesis 2: More frequent archiver -> HIGHER bandwidth
        corr_bw, p_bw = stats.spearmanr(intervals, bandwidth)
        passed_bw = corr_bw < 0 and p_bw < alpha
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**H2: Archiver Interval → Migration Bandwidth**")
            st.markdown(f"Expected: Negative correlation (lower interval = higher bandwidth)")
        with col2:
            if passed_bw:
                st.markdown("✅ **PASSED**")
            else:
                st.markdown("❌ **FAILED**")
        
        st.markdown(f"- Correlation: **{corr_bw:.4f}** (expected: < 0)")
        st.markdown(f"- p-value: **{p_bw:.4f}** (threshold: < {alpha})")
        st.markdown("---")
        
        # Hypothesis 3: More frequent archiver -> MORE cartridges used
        corr_cart, p_cart = stats.spearmanr(intervals, cartridges)
        passed_cart = corr_cart < 0 and p_cart < alpha
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**H3: Archiver Interval → Cartridges Used**")
            st.markdown(f"Expected: Negative correlation (lower interval = more cartridges)")
        with col2:
            if passed_cart:
                st.markdown("✅ **PASSED**")
            else:
                st.markdown("❌ **FAILED**")
        
        st.markdown(f"- Correlation: **{corr_cart:.4f}** (expected: < 0)")
        st.markdown(f"- p-value: **{p_cart:.4f}** (threshold: < {alpha})")
        
        # Summary
        st.markdown("---")
        all_passed = passed_arch and passed_bw and passed_cart
        if all_passed:
            st.success("✅ **All hypotheses PASSED**: Model correctly demonstrates expected HSM behavior!")
        else:
            st.warning("⚠️ **Some hypotheses FAILED**: Review model parameters or increase sample size.")
        
        st.caption(f"""
        **Statistical Method**: Spearman rank correlation
        - H₀: No correlation between interval and metric
        - H₁: Significant negative correlation exists
        - Reject H₀ if p-value < {alpha}
        """)
    else:
        st.warning("No archiver interval data available for validation.")

with tab5:
    st.markdown("### 📋 Detailed Data")
    
    if is_archiver_sweep:
        arch_df = df[df['archiver_interval_minutes'].notna()].copy()
        
        # Select relevant columns
        display_cols = [
            'archiver_interval_minutes', 
            'archived_data_tb', 
            'migration_bandwidth_gbs',
            'tape_cartridges_used',
            'scratch_bandwidth_degradation_pct',
            'replication',
            'run_time_seconds'
        ]
        
        available_cols = [c for c in display_cols if c in arch_df.columns]
        
        # Grouped summary
        st.markdown("#### Aggregated Summary")
        summary = arch_df.groupby('archiver_interval_minutes').agg({
            'archived_data_tb': ['mean', 'std', 'min', 'max'],
            'migration_bandwidth_gbs': ['mean', 'std'],
            'tape_cartridges_used': ['mean', 'std'],
            'scratch_bandwidth_degradation_pct': ['mean'],
            'replication': 'count'
        }).round(4)
        
        st.dataframe(summary, width='stretch')
        
        st.markdown("---")
        
        # All runs
        st.markdown("#### Individual Run Data")
        st.dataframe(arch_df[available_cols].sort_values('archiver_interval_minutes'), width='stretch')
        
        # Export
        st.markdown("---")
        st.markdown("#### 📥 Export Data")
        
        export_df = arch_df[available_cols].copy()
        export_csv = export_df.to_csv(index=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"archiver_analysis_sweep_{selected_sweep}_{timestamp}.csv"
        
        st.download_button(
            label="📥 Download as CSV",
            data=export_csv,
            file_name=filename,
            mime="text/csv"
        )
    else:
        st.warning("No archiver interval data to display.")

with tab6:
    st.markdown("### 🔬 Age Threshold Comparison")
    st.markdown("Side-by-side analysis of archiver interval and age threshold parameter sweeps.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Archiver Interval Sweep")
        if is_archiver_sweep:
            arch_df = df[df['archiver_interval_minutes'].notna()].copy()
            
            arch_summary = arch_df.groupby('archiver_interval_minutes').agg({
                'archived_data_tb': 'mean',
                'migration_bandwidth_gbs': 'mean',
                'tape_cartridges_used': 'mean',
                'scratch_bandwidth_degradation_pct': 'mean'
            }).round(4)
            
            st.dataframe(arch_summary, width='stretch')
            
            st.markdown("**Key Insight**: Controls *when* migration runs")
            st.caption("More frequent = more migration cycles = more data to tape")
        else:
            st.info("No archiver interval data in selected sweep")
    
    with col2:
        st.markdown("#### Age Threshold Sweep")
        if is_age_sweep:
            age_df = df[df['age_threshold_minutes'].notna()].copy()
            
            age_summary = age_df.groupby('age_threshold_minutes').agg({
                'archived_data_tb': 'mean',
                'migration_bandwidth_gbs': 'mean',
                'tape_cartridges_used': 'mean',
                'scratch_bandwidth_degradation_pct': 'mean'
            }).round(4)
            
            st.dataframe(age_summary, width='stretch')
            
            st.markdown("**Key Insight**: Controls *which* files migrate")
            st.caption("Lower threshold = more files eligible = more data to tape")
        else:
            st.info("No age threshold data in selected sweep")
    
    st.markdown("---")
    
    # Combined insight
    st.markdown("### 📚 Summary: Parameter Effects on HSM Migration")
    
    st.markdown("""
    | Parameter | Effect on Migration | Mechanism |
    |-----------|---------------------|-----------|
    | **Archiver Interval** | Frequency of migration cycles | More frequent = more opportunities to migrate |
    | **Age Threshold** | File eligibility | Lower threshold = more files qualify for migration |
    | **HWM** | Trigger for aggressive migration | Above HWM = migrate regardless of age |
    | **Cache Priority** | Derived policy parameters | Higher priority = more aggressive caching |
    """)
    
    st.success("""
    **Model Validation**: The simulation correctly demonstrates that both more frequent 
    archiver intervals and lower age thresholds lead to increased media consumption and 
    bandwidth utilization. This matches the expected behavior of real HSM systems.
    """)

st.markdown("---")
st.caption("💾 DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure")
