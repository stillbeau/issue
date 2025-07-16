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

# --- NEW: Robustly import the dedicated pricing module ---
try:
    import pricing_config as pc
    # Check for essential functions to provide a helpful error message
    required_items = ['get_material_group', 'validate_job_pricing', 'UNASSIGNED_MATERIALS']
    if not all(hasattr(pc, item) for item in required_items):
        st.error(
            "**Error: Your `pricing_config.py` file is incomplete or outdated.**\n\n"
            "It is missing required functions or variables. Please ensure you have the correct version of `pricing_config.py` saved in the same directory as this app. "
            "I am providing the correct version in a new 'pricing_config.py' document."
        )
        st.stop()
except ImportError:
    st.error(
        "**Error: `pricing_config.py` not found.**\n\n"
        "Please make sure the `pricing_config.py` file is saved in the same directory as this app. "
        "I am providing the correct version in a new 'pricing_config.py' document for you to save."
    )
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while importing `pricing_config.py`: {e}")
    st.stop()


# --- Page & App Configuration ---
st.set_page_config(layout="wide", page_title="Unified Business Dashboard", page_icon="🚀")

# --- Constants & Global Configuration ---
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Timeline & Risk Thresholds ---
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

    st.markdown("""
        <style>
            .login-container {
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                height: 70vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px; padding: 2rem; margin: auto; color: white;
                text-align: center; max-width: 500px;
            }
            .login-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
            .login-subtitle { font-size: 1.2rem; opacity: 0.9; margin-bottom: 2rem; }
        </style>
    """, unsafe_allow_html=True)

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


# --- Data Cleaning and Pricing Logic (Refactored) ---
def extract_material_from_description(material_description: str):
    """
    Extracts a plausible material name from a raw job material description string using regex.
    This is a data-cleaning step before looking up the material in the pricing config.
    """
    if pd.isna(material_description):
        return None, "no_description"

    material_desc = str(material_description).lower()
    
    laminate_indicators = ['wilsonart pl', 'formica', 'corian']
    if any(indicator in material_desc for indicator in laminate_indicators):
        return None, "laminate_skipped"

    patterns = [
        r'hanstone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)', r'rona\s+quartz[^-]*-\s*([^(/]+)',
        r'vicostone\s*\([^)]*\)\s*([^b]+?)\s*bq\d+', r'wilsonart\s+quartz\s*\([^)]*\)\s*([^mq]+?)(?:\s*matte)?\s*q\d+',
        r'silestone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)', r'cambria\s*\([^)]*\)\s*([^2-3]+?)\s*[23]cm',
        r'caesarstone\s*\([^)]+\)\s*([^#]+?)\s*#\d+', r'dekton\s*\([^)]+\)\s*([^m]+?)\s*matte',
        r'natural\s+stone\s*\([^)]+\)\s*([^2-3]+?)\s*[23]cm'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, material_desc, re.IGNORECASE)
        if matches:
            material = matches[0].strip()
            cleaned = re.sub(r'\s*(ex|ss|eternal|leathered|polished|matte)\s*', ' ', material, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if len(cleaned) > 2:
                return cleaned, "extracted"
    
    return None, "unrecognized_pattern"


def analyze_job_pricing(row):
    """
    Orchestrates the full pricing validation for a single job row.
    """
    material_desc = row.get('Job_Material', '')
    extracted_material, status = extract_material_from_description(material_desc)

    if status == "laminate_skipped":
        return {'status': 'laminate_skipped', 'message': 'Laminate material'}
    if status == "unrecognized_pattern":
        return {'status': 'unrecognized_pattern', 'message': f'Could not identify material from: "{material_desc[:50]}..."'}
    if not extracted_material:
        return {'status': 'error', 'message': 'Material name could not be extracted.'}

    material_group = pc.get_material_group(extracted_material)

    if material_group is None:
        if extracted_material.lower() in pc.UNASSIGNED_MATERIALS:
            return {'status': 'unassigned_material', 'message': f'Material "{extracted_material}" needs group assignment.'}
        else:
            return {'status': 'unknown_material', 'message': f'Extracted material "{extracted_material}" not in any group.'}

    try:
        sqft = float(str(row.get('Total_Job_SqFT', 0)).replace(',', ''))
        revenue = float(str(row.get('Total_Job_Price_', 0)).replace('$', '').replace(',', ''))
        plant_cost = float(str(row.get('Phase_Dollars_Plant_Invoice_', 0)).replace('$', '').replace(',', ''))
        customer_type = row.get('Job_Type', 'Retail')
    except (ValueError, TypeError):
        return {'status': 'error', 'message': 'Invalid numeric data for SqFt, Revenue, or Cost.'}

    if sqft <= 0:
        return {'status': 'error', 'message': 'Job has zero or invalid SqFt.'}

    validation_results = pc.validate_job_pricing(
        material_group=material_group,
        sqft=sqft,
        customer_type=customer_type,
        actual_revenue=revenue,
        actual_plant_cost=plant_cost
    )
    validation_results['extracted_material'] = extracted_material
    return validation_results


# --- Helper & Calculation Functions (Operational) ---
def parse_material(s: str) -> tuple[str, str]:
    """Parses material description to extract brand and color."""
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

def get_current_stage(row):
    """Determine the current operational stage of a job based on dates."""
    if pd.notna(row.get('Install_Date')) or pd.notna(row.get('Pick_Up_Date')): return 'Completed'
    elif pd.notna(row.get('Ship_Date')): return 'Shipped'
    elif pd.notna(row.get('Product_Rcvd_Date')): return 'Product Received'
    elif pd.notna(row.get('Ready_to_Fab_Date')): return 'In Fabrication'
    elif pd.notna(row.get('Template_Date')): return 'Post-Template'
    else: return 'Pre-Template'

def calculate_days_in_stage(row, today):
    """Calculate how many days a job has been in its current stage."""
    stage = row['Current_Stage']
    date_map = {'Shipped': 'Ship_Date', 'Product Received': 'Product_Rcvd_Date', 'In Fabrication': 'Ready_to_Fab_Date', 'Post-Template': 'Template_Date'}
    if stage in date_map and pd.notna(row.get(date_map[stage])):
        return (today - row[date_map[stage]]).days
    return 0 if stage == 'Completed' else np.nan

def calculate_risk_score(row):
    """Calculate an operational risk score for each job."""
    score = 0
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0: score += min(row['Days_Behind'] * 2, 20)
    if pd.notna(row.get('Days_In_Current_Stage')) and row['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']: score += 10
    if pd.isna(row.get('Ready_to_Fab_Date')) and pd.notna(row.get('Template_Date')):
        if (pd.Timestamp.now() - row['Template_Date']).days > TIMELINE_THRESHOLDS['template_to_rtf']: score += 15
    if row.get('Has_Rework', False): score += 10
    if pd.isna(row.get('Next_Sched_Activity')): score += 5
    return score

def calculate_delay_probability(row):
    """Calculate a simple probability of delay based on risk factors."""
    risk_score, factors = 0, []
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40; factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_durations = TIMELINE_THRESHOLDS['avg_stage_durations']
        expected_duration = avg_durations.get(row.get('Current_Stage', ''), avg_durations['default'])
        if row['Days_In_Current_Stage'] > expected_duration:
            risk_score += 20; factors.append(f"Stuck in {row['Current_Stage']} for {row['Days_In_Current_Stage']:.0f} days")
    if row.get('Has_Rework', False): risk_score += 15; factors.append("Has rework")
    if pd.isna(row.get('Next_Sched_Activity')): risk_score += 15; factors.append("No next activity scheduled")
    return min(risk_score, 100), ", ".join(factors)


# --- Data Loading and Processing ---
def _process_financial_data(df: pd.DataFrame, config: dict, install_cost_per_sqft: float) -> pd.DataFrame:
    """Processes financial data for a specific division."""
    df_processed = df.copy()
    for original, new in config["numeric_map"].items():
        if original in df_processed.columns:
            df_processed[new] = pd.to_numeric(df_processed[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else: df_processed[new] = 0.0
    df_processed['Install_Cost'] = df_processed.get('Total_Job_SqFt', 0) * install_cost_per_sqft
    df_processed['Total_Rework_Cost'] = sum([df_processed.get(c, 0) for c in config["rework_components"]])
    df_processed['Total_Branch_Cost'] = sum([df_processed.get(c, 0) for c in config["cost_components"]])
    revenue = df_processed.get('Revenue', 0)
    df_processed['Branch_Profit'] = revenue - df_processed['Total_Branch_Cost']
    df_processed['Branch_Profit_Margin_%'] = np.where(revenue != 0, (df_processed['Branch_Profit'] / revenue * 100), 0)
    df_processed['Profit_Variance'] = df_processed['Branch_Profit'] - df_processed.get('Original_GM', 0)
    if config["has_shop_profit"]:
        cost_from_plant = df_processed.get('Cost_From_Plant', 0)
        total_cogs = df_processed.get('Total_COGS', 0)
        df_processed['Shop_Profit'] = cost_from_plant - total_cogs
        df_processed['Shop_Profit_Margin_%'] = np.where(cost_from_plant != 0, (df_processed['Shop_Profit'] / cost_from_plant * 100), 0)
    return df_processed

@st.cache_data(ttl=300)
def load_and_process_data(today: pd.Timestamp, install_cost: float):
    """Loads data from Google Sheets and performs all processing."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=WORKSHEET_NAME, ttl=300)
        df = pd.DataFrame(df)
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Streamlit secrets configuration for gsheets.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    all_expected_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date', 'Delivery_Date',
        'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date', 'Job_Material',
        'Rework_Stone_Shop_Rework_Price', 'Production_', 'Total_Job_Price_', 'Total_Job_SqFT',
        'Job_Throughput_Job_GM_original', 'Salesperson', 'Division', 'Next_Sched_Activity',
        'Install_Assigned_To', 'Template_Assigned_To', 'Job_Name', 'Rework_Stone_Shop_Reason',
        'Ready_to_Fab_Status', 'Job_Type', 'Order_Type', 'Lead_Source', 'Phase_Dollars_Plant_Invoice_',
        'Job_Throughput_Rework_COGS', 'Job_Throughput_Rework_Job_Labor', 'Job_Throughput_Total_COGS',
        'Branch_INV_', 'Plant_INV_', 'Job_Status', 'Invoice_Status', 'Install_Status', 'Pick_Up_Status', 'Delivery_Status'
    ]
    for col in all_expected_cols:
        if col not in df.columns: df[col] = None
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols: df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Last_Activity_Date'] = df[date_cols].max(axis=1)
    df['Days_Since_Last_Activity'] = (today - df['Last_Activity_Date']).dt.days
    df['Days_Behind'] = np.where(df['Next_Sched_Date'].notna(), (today - df['Next_Sched_Date']).dt.days, np.nan)
    df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    df['Link'] = df['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
    df['Division_Type'] = df['Division'].apply(lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz')
    df['Current_Stage'] = df.apply(get_current_stage, axis=1)
    df['Days_In_Current_Stage'] = df.apply(lambda row: calculate_days_in_stage(row, today), axis=1)
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
    df['Days_RTF_to_Product_Rcvd'] = (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days
    df['Days_Product_Rcvd_to_Install'] = (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days
    df['Days_Template_to_Ship'] = (df['Ship_Date'] - df['Template_Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
    df['Has_Rework'] = df['Rework_Stone_Shop_Rework_Price'].notna() & (df['Rework_Stone_Shop_Rework_Price'] != '')
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    df[['Delay_Probability', 'Risk_Factors']] = df.apply(lambda row: pd.Series(calculate_delay_probability(row)), axis=1)
    
    pricing_analysis_series = df.apply(analyze_job_pricing, axis=1)
    def get_metric(analysis, key, default=None): return analysis.get(key, default) if isinstance(analysis, dict) else default
    df['Pricing_Analysis'] = pricing_analysis_series
    df['Material_Group'] = pricing_analysis_series.apply(lambda x: get_metric(x.get('expected_retail', {}), 'material_group'))
    df['Pricing_Issues_Count'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'critical_issues', 0))
    df['Pricing_Warnings_Count'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'warnings', 0))
    df['Revenue_Variance'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'revenue_variance', 0))
    df['Cost_Variance'] = pricing_analysis_series.apply(lambda x: get_metric(x, 'plant_cost_variance', 0))

    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    df_stone_processed = _process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)
    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)
    return df_stone_processed, df_laminate_processed, df_combined


# --- UI Rendering Functions (Placeholders) ---
# NOTE: The full implementation of these functions is required for the app to run.
def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp): pass
def render_workload_calendar(df: pd.DataFrame, today: pd.Timestamp): pass
def render_timeline_analytics(df: pd.DataFrame): pass
def render_predictive_analytics(df: pd.DataFrame): pass
def render_performance_scorecards(df: pd.DataFrame): pass
def render_historical_trends(df: pd.DataFrame): pass
def render_profitability_tabs(df_stone, df_laminate, today_dt): pass
def render_pricing_validation_tab(df: pd.DataFrame, division_name: str): pass
def render_overview_tab(df: pd.DataFrame, division_name: str): pass
def render_detailed_data_tab(df: pd.DataFrame, division_name: str): pass
def render_profit_drivers_tab(df: pd.DataFrame, division_name: str): pass
def render_rework_tab(df: pd.DataFrame, division_name: str): pass
def render_pipeline_issues_tab(df: pd.DataFrame, division_name: str, today: pd.Timestamp): pass
def render_field_workload_tab(df: pd.DataFrame, division_name: str): pass
def render_forecasting_tab(df: pd.DataFrame, division_name: str): pass
def render_overall_health_tab(df: pd.DataFrame, today: pd.Timestamp): pass


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

    st.sidebar.info(f"Data loaded for {len(df_full)} jobs.")
    
    if 'Pricing_Issues_Count' in df_full.columns:
        critical_issues = df_full['Pricing_Issues_Count'].sum()
        warnings = df_full['Pricing_Warnings_Count'].sum()
        st.sidebar.markdown("**🔍 Pricing Validation:**")
        st.sidebar.markdown(f"- 🔴 {critical_issues} critical issues")
        st.sidebar.markdown(f"- 🟡 {warnings} warnings")
    
    st.sidebar.info(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    main_tabs = st.tabs(["📈 Overall Business Health", "⚙️ Operational Performance", "💰 Profitability Analysis"])

    with main_tabs[0]:
        render_overall_health_tab(df_full, today_dt)

    with main_tabs[1]:
        st.header("Operational Performance Dashboard")
        st.markdown("Analyze real-time operational efficiency, risks, and workload.")
        
        op_cols = st.columns(3)
        with op_cols[0]:
            status_options = ["Active", "Complete", "30+ Days Old", "Unscheduled"]
            status_filter = st.multiselect("Filter by Job Status", status_options, default=["Active"], key="op_status_multi")
        with op_cols[1]:
            salesperson_list = ['All'] + sorted(df_full['Salesperson'].dropna().unique().tolist())
            salesperson_filter = st.selectbox("Filter by Salesperson", salesperson_list, key="op_sales")
        with op_cols[2]:
            division_list = ['All'] + sorted(df_full['Division_Type'].dropna().unique().tolist())
            division_filter = st.selectbox("Filter by Division", division_list, key="op_div")

        df_op_filtered = df_full.copy()

        if status_filter:
            final_mask = pd.Series([False] * len(df_full), index=df_full.index)
            if "Active" in status_filter: final_mask |= (df_full['Job_Status'] != 'Complete')
            if "Complete" in status_filter: final_mask |= (df_full['Job_Status'] == 'Complete')
            if "30+ Days Old" in status_filter:
                thirty_days_ago = today_dt - timedelta(days=30)
                final_mask |= ((df_full['Job_Creation'] < thirty_days_ago) & (df_full['Job_Status'] != 'Complete'))
            if "Unscheduled" in status_filter: final_mask |= (df_full['Next_Sched_Date'].isna() & (df_full['Job_Status'] != 'Complete'))
            df_op_filtered = df_full[final_mask]
        else:
            df_op_filtered = pd.DataFrame(columns=df_full.columns)

        if salesperson_filter != 'All': df_op_filtered = df_op_filtered[df_op_filtered['Salesperson'] == salesperson_filter]
        if division_filter != 'All': df_op_filtered = df_op_filtered[df_op_filtered['Division_Type'] == division_filter]
        
        st.info(f"Displaying {len(df_op_filtered)} jobs based on filters.")

        op_sub_tabs = st.tabs(["🚨 Daily Priorities", "📅 Workload Calendar", "📊 Timeline Analytics", "🔮 Predictive Analytics", "🎯 Performance Scorecards", "📈 Historical Trends"])
        with op_sub_tabs[0]: render_daily_priorities(df_op_filtered, today_dt)
        with op_sub_tabs[1]: render_workload_calendar(df_op_filtered, today_dt)
        with op_sub_tabs[2]: render_timeline_analytics(df_op_filtered)
        with op_sub_tabs[3]: render_predictive_analytics(df_op_filtered)
        with op_sub_tabs[4]: render_performance_scorecards(df_op_filtered)
        with op_sub_tabs[5]: render_historical_trends(df_full)

    with main_tabs[2]:
        render_profitability_tabs(df_stone, df_laminate, today_dt)

if __name__ == "__main__":
    main()
