# -*- coding: utf-8 -*-
"""
Unified Operations and Profitability Dashboard

This script provides a holistic view of the business, integrating operational performance
and financial analysis. It uses a dedicated pricing module for all financial calculations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- NEW: Import the dedicated pricing module ---
import pricing_config as pc

# --- Page & App Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Unified Business Dashboard", page_icon="🚀")

# --- Constants & Global Configuration ---
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Timeline & Risk Thresholds (Operational constants remain here) ---
TIMELINE_THRESHOLDS = {
    'template_to_rtf': 3,
    'rtf_to_product_rcvd': 7,
    'product_rcvd_to_install': 5,
    'template_to_install': 15,
    'template_to_ship': 10,
    'ship_to_install': 5,
    'days_in_stage_warning': 5,
    'stale_job_threshold': 7,
    'avg_stage_durations': {
        'Post-Template': 3,
        'In Fabrication': 7,
        'Product Received': 5,
        'Shipped': 5,
        'default': 5
    }
}

# --- Division-Specific Processing Configuration ---
STONE_CONFIG = {
    "name": "Stone/Quartz",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue', 'Phase_Dollars_Plant_Invoice_': 'Cost_From_Plant',
        'Total_Job_SqFT': 'Total_Job_SqFt', 'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price', 'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor', 'Job_Throughput_Total_COGS': 'Total_COGS'
    },
    "cost_components": ['Cost_From_Plant', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price', 'Rework_COGS', 'Rework_Labor'],
    "has_shop_profit": True
}

LAMINATE_CONFIG = {
    "name": "Laminate",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue', 'Branch_INV_': 'Shop_Cost', 'Plant_INV_': 'Material_Cost',
        'Total_Job_SqFT': 'Total_Job_SqFt', 'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
    },
    "cost_components": ['Shop_Cost', 'Material_Cost', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price'],
    "has_shop_profit": False
}


# --- Authentication Function ---
def render_login_screen():
    """ Displays a PIN-based login screen. Returns True if authenticated, False otherwise. """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS hidden for brevity

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Secure Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Enter your PIN to access the dashboard</div>', unsafe_allow_html=True)

    with st.form("pin_auth_form"):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed", placeholder="Enter PIN")
        submitted = st.form_submit_button("🔓 Unlock", use_container_width=True)
        if submitted:
            correct_pin = st.secrets.get("APP_PIN", "1234")
            if pin == correct_pin:
                st.session_state.authenticated = True
                st.success("Authentication successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid PIN. Please try again.")
    st.markdown('</div>', unsafe_allow_html=True)
    return False


# --- NEW: Data Cleaning and Pricing Logic (Refactored) ---

def extract_material_from_description(material_description):
    """
    Extracts a plausible material name from a raw job material description string using regex.
    This is a data-cleaning step before looking up the material in the pricing config.
    """
    if pd.isna(material_description):
        return None, "no_description"

    material_desc = str(material_description).lower()
    
    # Skip laminate materials
    if any(indicator in material_desc for indicator in ['wilsonart pl', 'formica', 'corian']):
        return None, "laminate_skipped"

    # Define regex patterns to find material names from various suppliers
    patterns = [
        r'hanstone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)',
        r'rona\s+quartz[^-]*-\s*([^(/]+)',
        r'vicostone\s*\([^)]*\)\s*([^b]+?)\s*bq\d+',
        r'wilsonart\s+quartz\s*\([^)]*\)\s*([^mq]+?)(?:\s*matte)?\s*q\d+',
        r'silestone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)',
        r'cambria\s*\([^)]*\)\s*([^2-3]+?)\s*[23]cm',
        r'caesarstone\s*\([^)]+\)\s*([^#]+?)\s*#\d+',
        r'dekton\s*\([^)]+\)\s*([^m]+?)\s*matte',
        r'natural\s+stone\s*\([^)]+\)\s*([^2-3]+?)\s*[23]cm'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, material_desc, re.IGNORECASE)
        if matches:
            # Clean up and return the first valid match
            material = matches[0].strip()
            cleaned = re.sub(r'\s*(ex|ss|eternal|leathered|polished|matte)\s*', ' ', material, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if len(cleaned) > 2:
                return cleaned, "extracted"
    
    return None, "unrecognized_pattern"


def detect_and_validate_pricing(row):
    """
    Orchestrates the full pricing validation for a single job row.
    1. Extracts material name from description.
    2. Uses pricing_config to get group and validate pricing.
    """
    material_desc = row.get('Job_Material', '')
    
    # 1. Extract a clean material name
    extracted_material, status = extract_material_from_description(material_desc)

    # Handle cases where extraction fails
    if status == "laminate_skipped":
        return {'status': 'laminate_skipped', 'message': 'Laminate material'}
    if status == "unrecognized_pattern":
        return {'status': 'unrecognized_material', 'message': f'Could not identify material from: {material_desc[:50]}...'}
    if not extracted_material:
        return {'status': 'error', 'message': 'Material name could not be extracted.'}
        
    # 2. Get material group from the pricing module
    material_group = pc.get_material_group(extracted_material)

    if material_group is None:
        # Check if it's a known but unassigned material
        if extracted_material.lower() in pc.UNASSIGNED_MATERIALS:
            return {'status': 'unassigned_material', 'message': f'Material "{extracted_material}" needs group assignment.'}
        else:
            return {'status': 'unknown_material', 'message': f'Extracted material "{extracted_material}" not in any group.'}

    # 3. Get all necessary values for validation
    try:
        sqft = float(str(row.get('Total_Job_SqFT', 0)).replace(',', ''))
        revenue = float(str(row.get('Total_Job_Price_', 0)).replace('$', '').replace(',', ''))
        plant_cost = float(str(row.get('Phase_Dollars_Plant_Invoice_', 0)).replace('$', '').replace(',', ''))
        customer_type = row.get('Job_Type', 'Retail') # Default to Retail if missing
    except (ValueError, TypeError):
        return {'status': 'error', 'message': 'Invalid numeric data for SqFt, Revenue, or Cost.'}

    if sqft <= 0:
        return {'status': 'error', 'message': 'Job has zero or invalid SqFt.'}

    # 4. Perform validation using the pricing module
    validation_results = pc.validate_job_pricing(
        material_group=material_group,
        sqft=sqft,
        customer_type=customer_type,
        actual_revenue=revenue,
        actual_plant_cost=plant_cost
    )
    # Add extracted material name for reference
    validation_results['extracted_material'] = extracted_material
    return validation_results


# --- Helper & Calculation Functions (Operational) ---

def parse_material_brand(s: str) -> tuple[str, str]:
    # ... (function remains the same)
    pass

def get_current_stage(row):
    # ... (function remains the same)
    pass

def calculate_days_in_stage(row, today):
    # ... (function remains the same)
    pass

def calculate_risk_score(row):
    # ... (function remains the same)
    pass

def calculate_delay_probability(row):
    # ... (function remains the same, but uses TIMELINE_THRESHOLDS constant)
    risk_score = 0
    factors = []
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40
        factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_durations = TIMELINE_THRESHOLDS['avg_stage_durations']
        expected_duration = avg_durations.get(row.get('Current_Stage', ''), avg_durations['default'])
        if row['Days_In_Current_Stage'] > expected_duration:
            risk_score += 20
            factors.append(f"Stuck in {row['Current_Stage']} for {row['Days_In_Current_Stage']:.0f} days")
    if row.get('Has_Rework', False):
        risk_score += 15
        factors.append("Has rework")
    if pd.isna(row.get('Next_Sched_Activity')):
        risk_score += 15
        factors.append("No next activity scheduled")
    return min(risk_score, 100), ", ".join(factors)


# --- Data Loading and Processing ---

def _process_financial_data(df: pd.DataFrame, config: dict, install_cost_per_sqft: float) -> pd.DataFrame:
    # ... (function remains the same)
    pass

@st.cache_data(ttl=300)
def load_and_process_data(today: pd.Timestamp, install_cost: float):
    """
    Loads data from Google Sheets and performs all processing.
    """
    # ... (GSheets connection and initial column cleaning remains the same)
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=WORKSHEET_NAME, ttl=300)
        df = pd.DataFrame(df)
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # Standardize columns
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    # ... (ensuring all expected columns exist remains the same)
    
    # ... (date parsing and operational calculations remain the same)
    # e.g., Last_Activity_Date, Days_Behind, Current_Stage, Risk_Score, etc.
    
    # --- REFACTORED PRICING VALIDATION ---
    # Apply the new, streamlined validation function
    pricing_analysis_series = df.apply(detect_and_validate_pricing, axis=1)
    df['Pricing_Analysis'] = pricing_analysis_series
    
    # Extract key metrics for easier filtering, using keys from the new config module
    def get_metric(analysis, key, default=None):
        return analysis.get(key, default) if isinstance(analysis, dict) else default

    df['Material_Group'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'material_group'))
    df['Pricing_Issues_Count'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'critical_issues', 0))
    df['Pricing_Warnings_Count'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'warnings', 0))
    df['Revenue_Variance'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'revenue_variance', 0))
    df['Cost_Variance'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'plant_cost_variance', 0)) # Note key change
    
    # --- Financial Processing (remains the same) ---
    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    
    df_stone_processed = _process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)

    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)

    return df_stone_processed, df_laminate_processed, df_combined


# --- UI Rendering Functions ---
# All UI functions (render_daily_priorities, render_workload_calendar, etc.)
# remain the same as they operate on the processed DataFrame.
# The `render_pricing_validation_tab` will now use the updated column names
# (`Cost_Variance` for plant cost) and dictionary keys (`critical_issues`).

def render_pricing_validation_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔍 {division_name} Pricing Validation")
    # ... (This function's logic remains largely the same, but you might need to
    #      update dictionary keys if you display details from the 'Pricing_Analysis' column.
    #      For example, `analysis.get('critical_issues')` instead of `analysis.get('total_issues')`.)


# --- Main Application ---
def main():
    if not render_login_screen():
        return

    st.title("🚀 Unified Operations & Profitability Dashboard")
    
    st.sidebar.header("⚙️ Configuration")
    today_dt = pd.to_datetime(st.sidebar.date_input("Select 'Today's' Date", value=datetime.now().date()))
    install_cost_sqft = st.sidebar.number_input("Install Cost per SqFt ($)", min_value=0.0, value=15.0, step=0.50)

    try:
        with st.spinner("Loading and processing job data..."):
            df_stone, df_laminate, df_full = load_and_process_data(today_dt, install_cost_sqft)
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.exception(e)
        st.stop()

    if df_full.empty:
        st.error("No data loaded. Please check your Google Sheets connection and data.")
        st.stop()

    # ... (Rest of the main function, including sidebar info and tab rendering, remains the same)

if __name__ == "__main__":
    main()
