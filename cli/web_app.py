"""
DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure
===============================================================================================
Streamlit-based web interface entry point.
Pages are auto-discovered from the pages/ directory.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import SimulationConfig
from db.database import init_db

st.set_page_config(
    page_title="DESCASSI - Discrete Event Simulator for Combined Archival and Scratch Storage Infrastructure",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = SimulationConfig()
    if 'sweep_results' not in st.session_state:
        st.session_state.sweep_results = None
    if 'sweep_running' not in st.session_state:
        st.session_state.sweep_running = False
    if 'sweep_progress' not in st.session_state:
        st.session_state.sweep_progress = 0.0
    if 'sweep_current' not in st.session_state:
        st.session_state.sweep_current = ""
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0

def main():
    """Main application entry point."""
    init_session_state()
    init_db()
    
    st.sidebar.title("💾 DESCASSI")
    st.sidebar.markdown("---")
    st.sidebar.info("Use the pages in the main area to navigate.")

if __name__ == "__main__":
    main()
