"""
DESCASSI - Reports Page
"""

import streamlit as st

st.set_page_config(
    page_title="DESCASSI - Reports",
    page_icon="📊",
    layout="wide")

st.title("📊 Reports")

from db.database import SweepDB

sweep_db = SweepDB()
sweeps = sweep_db.list_sweeps()

if not sweeps:
    st.info("No sweep results yet. Run a parameter sweep first.")
    st.stop()

sweep_options = {s['id']: f"{s['name']} ({s['total_runs']} runs)" for s in sweeps}
selected_sweep = st.selectbox("Select Sweep", list(sweep_options.keys()), format_func=lambda x: sweep_options[x])

if selected_sweep:
    results = sweep_db.get_sweep_results(selected_sweep)
    summary = sweep_db.get_sweep_summary(selected_sweep)
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    st.markdown("### Summary Metrics")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", summary.get('n_runs', 0))
    c2.metric("Avg Cache Hit", f"{summary.get('avg_hit_ratio', 0)*100:.1f}%")
    c3.metric("Avg p95 Latency", f"{summary.get('avg_latency_p95', 0):.1f}s")
    c4.metric("Avg Cost/Recall", f"${summary.get('avg_cost_per_recall', 0):.4f}")
    
    st.markdown("---")
    st.markdown("### 📀 Media Consumption by Policy Interval")
    
    if 'age_threshold_minutes' in df.columns and 'archived_data_tb' in df.columns:
        media_by_age = df.groupby('age_threshold_minutes').agg({
            'archived_data_tb': 'mean',
            'tape_cartridges_used': 'mean',
            'cache_resident_tb': 'mean'
        })
        
        st.dataframe(media_by_age.reset_index(), width='stretch')
        st.markdown("#### Media Consumption Chart")
        st.line_chart(media_by_age)
    
    st.markdown("---")
    st.markdown("### 📶 Bandwidth Impact by Policy Interval")
    
    if 'age_threshold_minutes' in df.columns and 'migration_bandwidth_gbs' in df.columns:
        bw_by_age = df.groupby('age_threshold_minutes').agg({
            'migration_bandwidth_gbs': 'mean',
            'scratch_effective_bandwidth_gbs': 'mean',
            'scratch_bandwidth_degradation_pct': 'mean'
        })
        
        st.dataframe(bw_by_age.reset_index(), width='stretch')
        st.markdown("#### Bandwidth Impact Chart")
        st.line_chart(bw_by_age)
    
    st.markdown("---")
    st.markdown("### Results Table")
    
    if not df.empty:
        display_cols = ['replication', 'age_threshold_minutes', 'cache_hit_ratio', 
                      'latency_p95', 'archived_data_tb', 'migration_bandwidth_gbs', 
                      'scratch_bandwidth_degradation_pct', 'cost_per_recall']
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(df[available_cols], width='stretch')
        
        st.markdown("### Charts")
        
        c1, c2 = st.columns(2)
        
        with c1:
            if 'cache_hit_ratio' in df.columns and 'age_threshold_minutes' in df.columns:
                st.markdown("#### Cache Hit Ratio")
                st.bar_chart(df.groupby('age_threshold_minutes')['cache_hit_ratio'].mean())
            elif 'cache_hit_ratio' in df.columns:
                st.bar_chart(df['cache_hit_ratio'])
        
        with c2:
            if 'latency_p95' in df.columns and 'age_threshold_minutes' in df.columns:
                st.markdown("#### p95 Latency")
                st.bar_chart(df.groupby('age_threshold_minutes')['latency_p95'].mean())
            elif 'latency_p95' in df.columns:
                st.bar_chart(df['latency_p95'])
        
        st.markdown("### Export")
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "sweep_results.csv",
            "text/csv"
        )
    else:
        st.warning("No results for this sweep yet.")
