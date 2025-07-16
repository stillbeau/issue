# -*- coding: utf-8 -*-
"""
FloForm Unified Dashboard - Main Application
Modular, clean, and performant business intelligence platform
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import warnings
from pricing_analysis_ui import render_pricing_analysis_tab
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from data_processing import load_and_process_data
    from business_logic import calculate_business_health_score
    from ui_components import (
        render_login_screen, 
        render_sidebar_config,
        render_overall_health_tab,
        render_operational_dashboard,
        render_profitability_dashboard
    )
    from visualization import setup_plotly_theme
    import pricing_config as pc
except ImportError as e:
    st.error(f"❌ **Module Import Error**: {e}")
    st.error("Please ensure all required files are in the same directory as app.py")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="FloForm Unified Dashboard",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- Initialize Plotly Theme ---
setup_plotly_theme()

# --- Constants ---
APP_VERSION = "2.0.0"
LAST_UPDATED = "December 2024"

def render_app_header():
    """Render the main application header with styling."""
    st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0;">🚀 FloForm Unified Dashboard</h1>
            <p style="color: white; opacity: 0.9; margin: 0;">Operations & Profitability Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'data_last_loaded' not in st.session_state:
        st.session_state.data_last_loaded = None
    
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'salesperson': 'All',
            'division': 'All',
            'status_filter': ['Active']
        }

def render_footer():
    """Render application footer with version info."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
            <div style="text-align: center; color: #888; font-size: 0.8rem;">
                FloForm Dashboard v{APP_VERSION} | Last Updated: {LAST_UPDATED}
            </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Authentication check
    if not render_login_screen():
        return
    
    # Render header
    render_app_header()
    
    # Sidebar configuration
    config = render_sidebar_config()
    
    # Data loading with error handling
    try:
        with st.spinner("🔄 Loading and processing job data..."):
            df_stone, df_laminate, df_full = load_and_process_data(
                config['today_dt'], 
                config['install_cost']
            )
    except Exception as e:
        st.error("❌ **Data Loading Error**")
        st.error(f"An unexpected error occurred: {e}")
        
        with st.expander("🔍 Debug Information"):
            st.exception(e)
        
        st.info("💡 **Troubleshooting Tips:**")
        st.markdown("""
        - Check your Google Sheets connection in Streamlit secrets
        - Verify the worksheet name is correct  
        - Ensure the pricing_config.py file is up to date
        - Try refreshing the page or clearing cache
        """)
        st.stop()
    
    # Data validation
    if df_full.empty:
        st.error("❌ **No Data Available**")
        st.error("No data was loaded from Google Sheets. Please check your connection and data source.")
        st.stop()
    
    # Update sidebar with data summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Data Summary**")
    st.sidebar.info(f"✅ {len(df_full)} total jobs loaded")
    
    if 'Division_Type' in df_full.columns:
        division_counts = df_full['Division_Type'].value_counts()
        for division, count in division_counts.items():
            st.sidebar.caption(f"• {division}: {count} jobs")
    
    # Pricing validation summary
    if 'Pricing_Issues_Count' in df_full.columns:
        critical_issues = df_full['Pricing_Issues_Count'].sum()
        warnings_count = df_full['Pricing_Warnings_Count'].sum()
        st.sidebar.markdown("🔍 **Pricing Status:**")
        st.sidebar.caption(f"• 🔴 {critical_issues} critical issues")
        st.sidebar.caption(f"• 🟡 {warnings_count} warnings")
    
    # Last refresh timestamp
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕒 Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
    
# Main dashboard tabs
main_tabs = st.tabs([
    "📈 Business Health",
    "⚙️ Operations", 
    "💰 Profitability",
    "🔍 Pricing Analysis"
])

# Business Health Tab
with main_tabs[0]:
    render_overall_health_tab(df_full, config['today_dt'])

# Operations Tab  
with main_tabs[1]:
    render_operational_dashboard(df_full, config['today_dt'])

# Profitability Tab
with main_tabs[2]:
    render_profitability_dashboard(df_stone, df_laminate, config['today_dt'])

# Pricing Analysis Tab
with main_tabs[3]:
    render_pricing_analysis_tab(df_full)  # Make sure this is indented!
    
    # Render footer
    render_footer()

# --- Application Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ **Application Error**")
        st.error(f"An unexpected error occurred: {e}")
        
        with st.expander("🔍 Debug Information"):
            st.exception(e)
        
        st.info("💡 Please refresh the page or contact support if the issue persists.")
