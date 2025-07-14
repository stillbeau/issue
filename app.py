# -*- coding: utf-8 -*-
"""
Unified Operations and Profitability Dashboard

This script combines two separate dashboards into a single, comprehensive Streamlit application.
It provides a holistic view of the business, integrating:
1.  Operational Performance: Timeline management, risk assessment, daily warnings, and workload planning.
2.  Financial Analysis: Job profitability, cost analysis, and revenue tracking by division.
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
import streamlit.components.v1 as components

# --- Page & App Configuration ---
st.set_page_config(layout="wide", page_title="Unified Business Dashboard", page_icon="🚀")

# --- Constants & Global Configuration ---
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Timeline & Risk Thresholds (from Operations Dashboard) ---
TIMELINE_THRESHOLDS = {
    'template_to_rtf': 3,          # days
    'rtf_to_product_rcvd': 7,      # days
    'product_rcvd_to_install': 5,  # days
    'template_to_install': 15,     # days
    'template_to_ship': 10,        # days
    'ship_to_install': 5,          # days
    'days_in_stage_warning': 5,    # days stuck in any stage
    'stale_job_threshold': 7       # days since last activity
}

# --- Division-Specific Processing Configuration (from Profitability Dashboard) ---
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


# --- AUTHENTICATION SYSTEM ---

def render_faceid_style_auth():
    """Face ID-style authentication with modern UI"""
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.auth_attempts = 0
    
    if st.session_state.authenticated:
        return True
    
    # Face ID-style CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .faceid-container {
            font-family: 'Inter', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
            background: linear-gradient(145deg, #000000, #1a1a1a);
            border-radius: 30px;
            padding: 3rem;
            color: white;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .faceid-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0,122,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        
        .faceid-icon {
            font-size: 5rem;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 2;
        }
        
        .faceid-title {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 2;
        }
        
        .faceid-subtitle {
            font-size: 1.1rem;
            opacity: 0.7;
            margin-bottom: 3rem;
            position: relative;
            z-index: 2;
        }
        
        .auth-button {
            background: #007AFF;
            color: white;
            border: none;
            border-radius: 15px;
            padding: 15px 30px;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0.5rem;
        }
        
        .auth-button:hover {
            background: #0056CC;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main Face ID interface
    st.markdown("""
        <div class="faceid-container">
            <div class="faceid-icon">🔒</div>
            <div class="faceid-title">Business Dashboard</div>
            <div class="faceid-subtitle">Secure authentication required</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Authentication methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔐 Face ID Authentication")
        if st.button("🔐 Authenticate with Face ID", key="faceid_main", use_container_width=True):
            with st.spinner("Authenticating..."):
                time.sleep(2)  # Simulate processing time
                st.session_state.authenticated = True
                st.success("✅ Face ID authentication successful!")
                time.sleep(1)
                st.rerun()
    
    with col2:
        st.markdown("### 🔢 PIN Authentication")
        with st.form("pin_form", clear_on_submit=True):
            pin = st.text_input("Enter 4-digit PIN", type="password", max_chars=4, key="pin_input")
            submitted = st.form_submit_button("Verify PIN", use_container_width=True)
            
            if submitted:
                # Get PIN from secrets
                correct_pin = st.secrets.get("APP_PIN", "1332")
                
                if pin == correct_pin:
                    st.session_state.authenticated = True
                    st.success("✅ PIN authentication successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.auth_attempts += 1
                    if st.session_state.auth_attempts >= 3:
                        st.error("❌ Too many failed attempts. Please wait...")
                        time.sleep(3)
                    else:
                        st.error(f"❌ Invalid PIN. {3 - st.session_state.auth_attempts} attempts remaining.")
    
    return False

# --- DATA LOADING AND PROCESSING ---

# --- Helper & Calculation Functions (Consolidated) ---

def parse_material(s: str) -> tuple[str, str]:
    """Parses a material description string to extract brand and color."""
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

def get_current_stage(row):
    """Determine the current operational stage of a job based on dates."""
    if pd.notna(row.get('Install_Date')) or pd.notna(row.get('Pick_Up_Date')):
        return 'Completed'
    elif pd.notna(row.get('Ship_Date')):
        return 'Shipped'
    elif pd.notna(row.get('Product_Rcvd_Date')):
        return 'Product Received'
    elif pd.notna(row.get('Ready_to_Fab_Date')):
        return 'In Fabrication'
    elif pd.notna(row.get('Template_Date')):
        return 'Post-Template'
    else:
        return 'Pre-Template'

def calculate_days_in_stage(row, today):
    """Calculate how many days a job has been in its current stage."""
    stage = row['Current_Stage']
    date_map = {
        'Shipped': 'Ship_Date',
        'Product Received': 'Product_Rcvd_Date',
        'In Fabrication': 'Ready_to_Fab_Date',
        'Post-Template': 'Template_Date'
    }
    if stage in date_map and pd.notna(row.get(date_map[stage])):
        return (today - row[date_map[stage]]).days
    return 0 if stage == 'Completed' else np.nan

def calculate_risk_score(row):
    """Calculate an operational risk score for each job."""
    score = 0
    if pd.notna(row.get('Days_Behind', np.nan)) and row['Days_Behind'] > 0:
        score += min(row['Days_Behind'] * 2, 20)
    if pd.notna(row.get('Days_In_Current_Stage', np.nan)) and row['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']:
        score += 10
    if pd.isna(row.get('Ready_to_Fab_Date')) and pd.notna(row.get('Template_Date')):
        days_since_template = (pd.Timestamp.now() - row['Template_Date']).days
        if days_since_template > TIMELINE_THRESHOLDS['template_to_rtf']:
            score += 15
    if row.get('Has_Rework', False):
        score += 10
    if pd.isna(row.get('Next_Sched_Activity')):
        score += 5
    return score

def calculate_delay_probability(row):
    """Calculate a simple probability of delay based on risk factors."""
    risk_score = 0
    factors = []
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40
        factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_stage_duration = {'Post-Template': 3, 'In Fabrication': 7, 'Product Received': 5, 'Shipped': 5}
        expected_duration = avg_stage_duration.get(row.get('Current_Stage', ''), 5)
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
    """Processes financial data for a specific division using a configuration dictionary."""
    df_processed = df.copy()
    for original, new in config["numeric_map"].items():
        if original in df_processed.columns:
            df_processed[new] = pd.to_numeric(df_processed[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else:
            df_processed[new] = 0.0
            
    df_processed['Install_Cost'] = df_processed.get('Total_Job_SqFt', 0) * install_cost_per_sqft
    
    rework_costs = [df_processed.get(c, 0) for c in config["rework_components"]]
    df_processed['Total_Rework_Cost'] = sum(rework_costs)
    
    branch_costs = [df_processed.get(c, 0) for c in config["cost_components"]]
    df_processed['Total_Branch_Cost'] = sum(branch_costs)

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
    """
    Loads data from Google Sheets using Streamlit secrets and performs all processing for both
    operational and profitability analysis.
    """
    try:
        # 1. Load data using Streamlit connection
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=WORKSHEET_NAME, ttl=300)
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Streamlit secrets configuration.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)

    # 2. Ensure Core Columns Exist
    all_expected_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date', 'Delivery_Date',
        'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date', 'Job_Material',
        'Rework_Stone_Shop_Rework_Price', 'Production_', 'Total_Job_Price_', 'Total_Job_SqFT',
        'Job_Throughput_Job_GM_original', 'Salesperson', 'Division', 'Next_Sched_Activity',
        'Install_Assigned_To', 'Template_Assigned_To', 'Job_Name', 'Rework_Stone_Shop_Reason',
        'Ready_to_Fab_Status', 'Job_Type', 'Order_Type', 'Lead_Source', 'Phase_Dollars_Plant_Invoice_',
        'Job_Throughput_Rework_COGS', 'Job_Throughput_Rework_Job_Labor', 'Job_Throughput_Total_COGS',
        'Branch_INV_', 'Plant_INV_', 'Job_Status', 'Customer_Category', 'City', 'Lead_Type',
        'Supplied_By', 'Job_Throughput_Job_Rev', 'Job_Throughput_Total_Job_Cost'
    ]
    for col in all_expected_cols:
        if col not in df.columns:
            df[col] = None

    # 3. Parse Dates
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date',
                 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 4. Basic Calculations & Enrichments
    df['Last_Activity_Date'] = df[date_cols].max(axis=1)
    df['Days_Since_Last_Activity'] = (today - df['Last_Activity_Date']).dt.days
    df['Days_Behind'] = np.where(df['Next_Sched_Date'].notna(), (today - df['Next_Sched_Date']).dt.days, np.nan)
    df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    df['Link'] = df['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
    df['Division_Type'] = df['Division'].apply(lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz')

    # 5. Operational Metrics
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
    
    # Calculate Total_Rework_Cost for quality analysis
    df['Total_Rework_Cost'] = pd.to_numeric(df['Rework_Stone_Shop_Rework_Price'].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)

    # 6. Financial Metrics Processing & Splitting by Division
    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    
    df_stone_processed = _process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)

    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)

    # Return all three dataframes for efficiency
    return df_stone_processed, df_laminate_processed, df_combined

# --- UI Rendering Functions for OPERATIONAL PERFORMANCE ---

def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚨 Daily Priorities & Warnings")
    
    # Metrics at the top respect the main filters applied to the incoming dataframe `df`
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 High Risk Jobs", len(df[df['Risk_Score'] >= 30]))
    with col2:
        st.metric("⏰ Behind Schedule", len(df[df['Days_Behind'] > 0]))
    with col3:
        st.metric("🚧 Stuck Jobs", len(df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]))
    with col4:
        # For the metric, also only count active stale jobs from the filtered df
        stale_jobs_metric = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
        st.metric("💨 Stale Jobs", len(stale_jobs_metric))

    st.markdown("---")
    st.subheader("⚡ Critical Issues Requiring Immediate Attention")
    st.caption("This section only shows jobs where 'Job Status' is not 'Complete'.")

    # Define dataframes for each critical issue, ensuring they are active jobs
    # by filtering for Job_Status != 'Complete' from the incoming dataframe `df`.
    
    missing_activity = df[(df['Next_Sched_Activity'].isna()) & (df['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received'])) & (df['Job_Status'] != 'Complete')]
    
    stale_jobs = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
    
    template_to_rtf_stuck = df[(df['Template_Date'].notna()) & (df['Ready_to_Fab_Date'].isna()) & ((today - df['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf']) & (df['Job_Status'] != 'Complete')]
    
    upcoming_installs = df[(df['Install_Date'].notna()) & (df['Install_Date'] <= today + timedelta(days=7)) & (df['Product_Rcvd_Date'].isna()) & (df['Job_Status'] != 'Complete')]

    # Render expanders using the filtered dataframes
    if not missing_activity.empty:
        with st.expander(f"🚨 Jobs Missing Next Activity ({len(missing_activity)} jobs)", expanded=True):
            display_cols = ['Link', 'Job_Name', 'Current_Stage', 'Days_In_Current_Stage', 'Salesperson']
            st.dataframe(missing_activity[display_cols].sort_values('Days_In_Current_Stage', ascending=False),
                         column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}, use_container_width=True)

    if not stale_jobs.empty:
        with st.expander(f"💨 Stale Jobs (No activity for >{TIMELINE_THRESHOLDS['stale_job_threshold']} days)", expanded=False):
            stale_jobs_display = stale_jobs.copy()
            stale_jobs_display['Last_Activity_Date'] = stale_jobs_display['Last_Activity_Date'].dt.strftime('%Y-%m-%d')
            display_cols = ['Link', 'Job_Name', 'Current_Stage', 'Last_Activity_Date', 'Days_Since_Last_Activity']
            st.dataframe(stale_jobs_display[display_cols].sort_values('Days_Since_Last_Activity', ascending=False),
                         column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}, use_container_width=True)

    if not template_to_rtf_stuck.empty:
        with st.expander(f"📋 Stuck: Template → Ready to Fab ({len(template_to_rtf_stuck)} jobs)"):
            template_to_rtf_stuck_display = template_to_rtf_stuck.copy()
            template_to_rtf_stuck_display['Days_Since_Template'] = (today - template_to_rtf_stuck_display['Template_Date']).dt.days
            display_cols = ['Link', 'Job_Name', 'Template_Date', 'Days_Since_Template', 'Salesperson']
            st.dataframe(template_to_rtf_stuck_display[display_cols].sort_values('Days_Since_Template', ascending=False),
                         column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}, use_container_width=True)

    if not upcoming_installs.empty:
        with st.expander(f"⚠️ Upcoming Installs Missing Product ({len(upcoming_installs)} jobs)", expanded=True):
            upcoming_installs_display = upcoming_installs.copy()
            upcoming_installs_display['Days_Until_Install'] = (upcoming_installs_display['Install_Date'] - today).dt.days
            display_cols = ['Link', 'Job_Name', 'Install_Date', 'Days_Until_Install', 'Install_Assigned_To']
            st.dataframe(upcoming_installs_display[display_cols].sort_values('Days_Until_Install'),
                         column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}, use_container_width=True)

def render_workload_calendar(df: pd.DataFrame, today: pd.Timestamp):
    st.header("📅 Workload Calendar")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=today.date(), key="cal_start")
    with col2:
        end_date = st.date_input("End Date", value=(today + timedelta(days=14)).date(), key="cal_end")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    activity_type = st.selectbox("Select Activity Type", ["Templates", "Installs", "All Activities"], key="cal_activity")
    
    activity_df = pd.DataFrame()
    if activity_type == "Templates":
        if 'Template_Date' in df.columns:
            activity_df = df[df['Template_Date'].notna()].copy()
            date_col, assignee_col = 'Template_Date', 'Template_Assigned_To'
    elif activity_type == "Installs":
        if 'Install_Date' in df.columns:
            activity_df = df[df['Install_Date'].notna()].copy()
            date_col, assignee_col = 'Install_Date', 'Install_Assigned_To'
    else:
        activities = []
        if 'Template_Date' in df.columns:
            temp_df = df[df['Template_Date'].notna()].copy()
            temp_df.rename(columns={'Template_Date': 'Activity_Date', 'Template_Assigned_To': 'Assignee'}, inplace=True)
            temp_df['Activity_Type'] = 'Template'
            activities.append(temp_df)
        if 'Install_Date' in df.columns:
            inst_df = df[df['Install_Date'].notna()].copy()
            inst_df.rename(columns={'Install_Date': 'Activity_Date', 'Install_Assigned_To': 'Assignee'}, inplace=True)
            inst_df['Activity_Type'] = 'Install'
            activities.append(inst_df)
        if activities:
            activity_df = pd.concat(activities, ignore_index=True)
        date_col, assignee_col = 'Activity_Date', 'Assignee'

    if activity_df.empty or date_col not in activity_df.columns:
        st.warning("No activities found for the selected type and filters.")
        return
        
    activity_df = activity_df[(activity_df[date_col] >= pd.Timestamp(start_date)) & (activity_df[date_col] <= pd.Timestamp(end_date))]
    
    daily_summary = []
    if assignee_col in activity_df.columns:
        for date in date_range:
            day_activities = activity_df[activity_df[date_col].dt.date == date.date()]
            if not day_activities.empty:
                assignee_counts = day_activities[assignee_col].value_counts()
                for assignee, count in assignee_counts.items():
                    if assignee and str(assignee).strip():
                        assignee_jobs = day_activities[day_activities[assignee_col] == assignee]
                        total_sqft = assignee_jobs['Total_Job_SqFt'].sum() if 'Total_Job_SqFt' in assignee_jobs else 0
                        daily_summary.append({'Date': date, 'Assignee': str(assignee), 'Job_Count': int(count), 'Total_SqFt': float(total_sqft)})

    if daily_summary:
        summary_df = pd.DataFrame(daily_summary)
        try:
            pivot_df = summary_df.pivot_table(index='Assignee', columns=summary_df['Date'].dt.strftime('%m/%d'), values='Job_Count', fill_value=0, aggfunc='sum').astype(int)
            if not pivot_df.empty:
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_df) * 0.5)))
                sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Jobs'})
                ax.set_title('Jobs by Assignee and Date'); ax.set_xlabel('Date'); ax.set_ylabel('Assignee')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception as e:
            st.warning(f"Unable to create heatmap visualization. Error: {e}")
            st.dataframe(summary_df)
            
        st.subheader("💡 Days with Light Workload")
        threshold = st.slider("Jobs threshold for 'light' day", 1, 10, 3, key="cal_slider")
        daily_totals = summary_df.groupby('Date')['Job_Count'].sum()
        light_days_data = [{'Date': date.strftime('%A, %m/%d'), 'Total_Jobs': int(daily_totals.get(date, 0)), 'Available_Capacity': int(threshold - daily_totals.get(date, 0))} for date in date_range if daily_totals.get(date, 0) < threshold]
        if light_days_data:
            st.dataframe(pd.DataFrame(light_days_data), use_container_width=True)
        else:
            st.success(f"No days found with fewer than {threshold} jobs.")

def render_timeline_analytics(df: pd.DataFrame):
    st.header("📊 Timeline Analytics & Bottlenecks")
    timeline_metrics = {
        "Template to Install": "Days_Template_to_Install", "Ready to Fab to Product Received": "Days_RTF_to_Product_Rcvd",
        "Template to Ready to Fab": "Days_Template_to_RTF", "Product Received to Install": "Days_Product_Rcvd_to_Install",
        "Template to Ship": "Days_Template_to_Ship", "Ship to Install": "Days_Ship_to_Install",
    }
    
    st.subheader("⏱️ Average Timeline by Division")
    divisions = df['Division_Type'].unique()
    cols = st.columns(len(divisions))
    
    for idx, division in enumerate(divisions):
        with cols[idx]:
            st.markdown(f"**{division}**")
            division_df = df[df['Division_Type'] == division]
            for metric_name, col_name in timeline_metrics.items():
                if col_name in division_df.columns:
                    avg_days = division_df[col_name].mean()
                    if pd.notna(avg_days):
                        st.metric(metric_name, f"{avg_days:.1f} days")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Bottlenecks")
        stage_counts = df['Current_Stage'].value_counts()
        if not stage_counts.empty:
            st.bar_chart(stage_counts)
            bottleneck_stage = stage_counts.idxmax()
            st.info(f"📍 Potential bottleneck: **{bottleneck_stage}** ({stage_counts[bottleneck_stage]} jobs)")
    with col2:
        st.subheader("Stage Duration Analysis")
        stuck_threshold = st.number_input("Days threshold for 'stuck' jobs", min_value=3, max_value=30, value=7, key="stuck_threshold")
        stuck_jobs = df[df['Days_In_Current_Stage'] > stuck_threshold]
        if not stuck_jobs.empty:
            stuck_by_stage = stuck_jobs['Current_Stage'].value_counts()
            st.bar_chart(stuck_by_stage)
            st.warning(f"⚠️ {len(stuck_jobs)} jobs stuck > {stuck_threshold} days")
        else:
            st.success(f"✅ No jobs stuck > {stuck_threshold} days")

def render_predictive_analytics(df: pd.DataFrame):
    st.header("🔮 Predictive Analytics")
    active_jobs = df[~df['Current_Stage'].isin(['Completed'])].copy()
    if active_jobs.empty:
        st.warning("No active jobs to analyze.")
        return
        
    high_risk_threshold = st.slider("High risk threshold (%)", min_value=50, max_value=90, value=70, key="risk_threshold")
    high_risk_jobs = active_jobs[active_jobs['Delay_Probability'] >= high_risk_threshold].sort_values('Delay_Probability', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"🚨 Jobs with >{high_risk_threshold}% Delay Risk ({len(high_risk_jobs)} jobs)")
        if not high_risk_jobs.empty:
            for _, row in high_risk_jobs.head(10).iterrows():
                with st.container(border=True):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{row.get('Job_Name', 'Unknown')}** - {row.get('Current_Stage', 'Unknown')}")
                        st.caption(f"Salesperson: {row.get('Salesperson', 'N/A')} | Next: {row.get('Next_Sched_Activity', 'None scheduled')}")
                        if row['Risk_Factors']:
                            st.warning(f"Risk factors: {row['Risk_Factors']}")
                    with col_b:
                        color = "🔴" if row['Delay_Probability'] >= 80 else "🟡"
                        st.metric("Delay Risk", f"{color} {row['Delay_Probability']:.0f}%")
        else:
            st.success(f"No jobs with delay risk above {high_risk_threshold}%")
    with col2:
        st.subheader("📊 Risk Distribution")
        risk_bins = [0, 30, 50, 70, 101]; risk_labels = ['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (70-100%)']
        active_jobs['Risk_Category'] = pd.cut(active_jobs['Delay_Probability'], bins=risk_bins, labels=risk_labels, right=False)
        risk_dist = active_jobs['Risk_Category'].value_counts()
        for category in risk_labels:
            st.metric(f"{'🔴' if 'Critical' in category else '🟡' if 'High' in category else '🟠' if 'Medium' in category else '🟢'} {category}", risk_dist.get(category, 0))

def render_performance_scorecards(df: pd.DataFrame):
    st.header("🎯 Performance Scorecards")
    role_type = st.selectbox("Select Role to Analyze", ["Salesperson", "Template Assigned To", "Install Assigned To"], key="role_select")
    role_col = {"Salesperson": "Salesperson", "Template Assigned To": "Template_Assigned_To", "Install Assigned To": "Install_Assigned_To"}[role_type]
    
    if role_col not in df.columns:
        st.warning(f"Column '{role_col}' not found in data.")
        return
        
    employees = [e for e in df[role_col].dropna().unique() if str(e).strip()]
    if not employees:
        st.warning(f"No {role_type} data available.")
        return
        
    scorecards = []
    now = pd.Timestamp.now()
    week_ago = now - pd.Timedelta(days=7)

    for employee in employees:
        emp_jobs = df[df[role_col] == employee]
        if role_type == "Salesperson":
            metrics = {
                'Employee': employee, 'Total Jobs': len(emp_jobs), 'Active Jobs': len(emp_jobs[~emp_jobs['Current_Stage'].isin(['Completed'])]),
                'Avg Days Behind': emp_jobs['Days_Behind'].mean(), 'Jobs w/ Rework': len(emp_jobs[emp_jobs.get('Has_Rework', False) == True]),
                'Avg Timeline': emp_jobs['Days_Template_to_Install'].mean(), 'High Risk Jobs': len(emp_jobs[emp_jobs.get('Risk_Score', 0) >= 30])
            }
        elif role_type == "Template Assigned To":
            template_jobs = emp_jobs[emp_jobs['Template_Date'].notna()]
            metrics = {
                'Employee': employee, 'Total Templates': len(template_jobs), 'Avg Template to RTF': template_jobs['Days_Template_to_RTF'].mean(),
                'Templates This Week': len(template_jobs[(template_jobs['Template_Date'] >= week_ago) & (template_jobs['Template_Date'] <= now)]),
                'Upcoming Templates': len(template_jobs[template_jobs['Template_Date'] > now]),
                'Overdue RTF': len(template_jobs[(template_jobs.get('Ready_to_Fab_Date', pd.NaT) < template_jobs['Template_Date'])])
            }
        else: # Install Assigned To
            install_jobs = emp_jobs[emp_jobs['Install_Date'].notna()]
            metrics = {
                'Employee': employee, 'Total Installs': len(install_jobs),
                'Installs This Week': len(install_jobs[(install_jobs['Install_Date'] >= week_ago) & (install_jobs['Install_Date'] <= now)]),
                'Upcoming Installs': len(install_jobs[install_jobs['Install_Date'] > now]),
                'Avg Ship to Install': install_jobs['Days_Ship_to_Install'].mean(), 'Total SqFt': install_jobs['Total_Job_SqFt'].sum()
            }
        scorecards.append(metrics)
        
    if not scorecards:
        st.warning(f"No performance data available for {role_type}.")
        return
        
    scorecards_df = pd.DataFrame(scorecards).sort_values("Active Jobs" if role_type == "Salesperson" else "Total Templates" if role_type == "Template Assigned To" else "Total Installs", ascending=False)
    st.subheader(f"🏆 Top Performers: {role_type}")
    cols = st.columns(min(3, len(scorecards_df)))
    for idx, (_, row) in enumerate(scorecards_df.head(3).iterrows()):
        if idx < len(cols):
            with cols[idx], st.container(border=True):
                st.markdown(f"### {row['Employee']}")
                if role_type == "Salesperson":
                    st.metric("Active Jobs", f"{row['Active Jobs']:.0f}")
                    st.metric("Avg Days Behind", f"{row['Avg Days Behind']:.1f}" if pd.notna(row['Avg Days Behind']) else "N/A")
                    st.metric("High Risk Jobs", f"{row['High Risk Jobs']:.0f}", delta_color="inverse")
                elif role_type == "Template Assigned To":
                    st.metric("Templates This Week", f"{row['Templates This Week']:.0f}")
                    st.metric("Avg Template→RTF", f"{row['Avg Template to RTF']:.1f} days" if pd.notna(row['Avg Template to RTF']) else "N/A")
                    st.metric("Overdue RTF", f"{row['Overdue RTF']:.0f}", delta_color="inverse")
                else:
                    st.metric("Installs This Week", f"{row['Installs This Week']:.0f}")
                    st.metric("Upcoming Installs", f"{row['Upcoming Installs']:.0f}")
                    st.metric("Total SqFt", f"{row['Total SqFt']:,.0f}")
    
    with st.expander("View All Employees"):
        st.dataframe(scorecards_df.style.format({col: '{:.1f}' for col in scorecards_df.columns if 'Avg' in col or 'Days' in col} | {col: '{:,.0f}' for col in scorecards_df.columns if 'SqFt' in col}, na_rep='N/A'), use_container_width=True)

def render_historical_trends(df: pd.DataFrame):
    st.header("📈 Historical Trends")
    st.markdown("Analyze performance and quality trends over time. This view uses all data, ignoring local filters.")

    df['Job_Creation'] = pd.to_datetime(df['Job_Creation'], errors='coerce')
    df['Install_Date'] = pd.to_datetime(df['Install_Date'], errors='coerce')
    df.dropna(subset=['Job_Creation', 'Install_Date'], how='all', inplace=True)

    if df.empty:
        st.warning("Not enough data to build historical trends.")
        return

    st.subheader("Job Throughput (Monthly)")
    created = df.set_index('Job_Creation').resample('M').size().rename('Jobs Created')
    completed = df[df['Current_Stage'] == 'Completed'].set_index('Install_Date').resample('M').size().rename('Jobs Completed')
    throughput_df = pd.concat([created, completed], axis=1).fillna(0).astype(int)
    throughput_df.index = throughput_df.index.strftime('%Y-%m')
    st.line_chart(throughput_df)
    st.caption("Compares new jobs created vs. jobs completed each month.")

    st.markdown("---")
    st.subheader("Average Job Cycle Time Trend (Template to Install)")
    completed_jobs = df[df['Days_Template_to_Install'].notna()].copy()
    if not completed_jobs.empty:
        cycle_time_trend = completed_jobs.set_index('Install_Date')['Days_Template_to_Install'].resample('M').mean().fillna(0)
        cycle_time_trend.index = cycle_time_trend.index.strftime('%Y-%m')
        st.line_chart(cycle_time_trend)
        st.caption("Tracks the average number of days from template to installation.")
    else:
        st.info("No completed jobs with both Template and Install dates to analyze cycle time.")

    st.markdown("---")
    st.subheader("Rework Rate Trend (%)")
    rework_jobs = df[df['Install_Date'].notna()].copy()
    if not rework_jobs.empty:
        rework_jobs['Month'] = rework_jobs['Install_Date'].dt.to_period('M')
        monthly_rework = rework_jobs.groupby('Month').agg(
            Total_Jobs=('Job_Name', 'count'),
            Rework_Jobs=('Has_Rework', 'sum')
        )
        monthly_rework['Rework_Rate'] = (monthly_rework['Rework_Jobs'] / monthly_rework['Total_Jobs']) * 100
        rework_rate_trend = monthly_rework['Rework_Rate'].fillna(0)
        rework_rate_trend.index = rework_rate_trend.index.strftime('%Y-%m')
        st.line_chart(rework_rate_trend)
        st.caption("Monitors the percentage of completed jobs that required rework.")
    else:
        st.info("No completed jobs to analyze rework trends.")

# --- NEW ADVANCED ANALYTICS FUNCTIONS ---

def render_cash_flow_forecasting(df: pd.DataFrame, today: pd.Timestamp):
    st.header("💰 Predictive Cash Flow Forecasting")
    
    # Calculate average cycle times for prediction
    completed_jobs = df[df['Current_Stage'] == 'Completed'].copy()
    if completed_jobs.empty:
        st.warning("No completed jobs available for cash flow forecasting.")
        return
    
    avg_cycle_time = completed_jobs['Days_Template_to_Install'].mean()
    if pd.isna(avg_cycle_time):
        avg_cycle_time = 30  # Default fallback
    
    col1, col2, col3 = st.columns(3)
    
    # Current pipeline value
    pipeline_jobs = df[df['Current_Stage'] != 'Completed'].copy()
    total_pipeline_value = pipeline_jobs['Revenue'].sum()
    
    with col1:
        st.metric("Total Pipeline Value", f"${total_pipeline_value:,.0f}")
    
    # Predicted monthly revenue
    monthly_avg = completed_jobs.groupby(completed_jobs['Install_Date'].dt.to_period('M'))['Revenue'].sum().mean()
    with col2:
        st.metric("Avg Monthly Revenue", f"${monthly_avg:,.0f}" if pd.notna(monthly_avg) else "N/A")
    
    # Jobs at risk of delay
    at_risk_value = pipeline_jobs[pipeline_jobs['Risk_Score'] >= 30]['Revenue'].sum()
    with col3:
        st.metric("At-Risk Revenue", f"${at_risk_value:,.0f}", delta_color="inverse")
    
    st.markdown("---")
    
    # Cash flow forecast
    st.subheader("📊 90-Day Cash Flow Forecast")
    
    # Create forecast periods
    periods = []
    for i in range(13):  # 13 weeks = ~90 days
        week_start = today + timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        periods.append({'week': i+1, 'start': week_start, 'end': week_end})
    
    forecast_data = []
    for period in periods:
        # Estimate jobs likely to complete in this period
        period_revenue = 0
        
        # Jobs with scheduled install dates in this period
        scheduled_installs = pipeline_jobs[
            (pipeline_jobs['Install_Date'] >= period['start']) & 
            (pipeline_jobs['Install_Date'] <= period['end'])
        ]
        period_revenue += scheduled_installs['Revenue'].sum()
        
        # Jobs without install dates but likely to complete based on current stage
        stage_multipliers = {
            'Shipped': 0.8,  # 80% likely to install this week
            'Product Received': 0.6,  # 60% likely
            'In Fabrication': 0.3,   # 30% likely
            'Post-Template': 0.1     # 10% likely
        }
        
        for stage, multiplier in stage_multipliers.items():
            stage_jobs = pipeline_jobs[
                (pipeline_jobs['Current_Stage'] == stage) & 
                (pipeline_jobs['Install_Date'].isna())
            ]
            period_revenue += (stage_jobs['Revenue'].sum() * multiplier) / 13  # Spread over 13 weeks
        
        forecast_data.append({
            'Week': f"Week {period['week']}",
            'Forecasted_Revenue': period_revenue,
            'Period': period['start'].strftime('%m/%d')
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Display forecast chart
    st.bar_chart(forecast_df.set_index('Week')['Forecasted_Revenue'])
    
    # Cash flow alerts
    st.markdown("---")
    st.subheader("🚨 Cash Flow Alerts")
    
    low_weeks = forecast_df[forecast_df['Forecasted_Revenue'] < monthly_avg/4]  # Less than 25% of monthly average
    if not low_weeks.empty:
        with st.expander("⚠️ Low Cash Flow Periods", expanded=True):
            st.warning(f"Found {len(low_weeks)} weeks with potentially low cash flow:")
            for _, week in low_weeks.iterrows():
                st.write(f"• {week['Week']} ({week['Period']}): ${week['Forecasted_Revenue']:,.0f}")
    else:
        st.success("✅ No significant cash flow gaps detected in the next 90 days")

def render_material_cost_intelligence(df: pd.DataFrame):
    st.header("🔬 Material Cost Intelligence")
    
    if 'Job_Material' not in df.columns or df['Job_Material'].isna().all():
        st.warning("Material data not available for analysis.")
        return
    
    # Parse material data that was already processed
    material_df = df[df['Material_Brand'].notna() & df['Material_Color'].notna()].copy()
    
    if material_df.empty:
        st.warning("No parsed material data available.")
        return
    
    st.subheader("💎 Material Profitability Analysis")
    
    # Material profitability by brand
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Profitable Material Brands**")
        brand_profit = material_df.groupby('Material_Brand').agg({
            'Branch_Profit': 'sum',
            'Branch_Profit_Margin_%': 'mean',
            'Job_Name': 'count'
        }).round(2)
        brand_profit.columns = ['Total_Profit', 'Avg_Margin_%', 'Job_Count']
        brand_profit = brand_profit.sort_values('Total_Profit', ascending=False).head(10)
        
        st.dataframe(brand_profit.style.format({
            'Total_Profit': '${:,.0f}',
            'Avg_Margin_%': '{:.1f}%'
        }), use_container_width=True)
    
    with col2:
        st.markdown("**Material Cost Trends**")
        if 'Cost_From_Plant' in material_df.columns and 'Total_Job_SqFt' in material_df.columns:
            material_df['Cost_Per_SqFt'] = material_df['Cost_From_Plant'] / material_df['Total_Job_SqFt'].replace(0, 1)
            
            # Monthly cost per sqft trend
            material_df['Month'] = pd.to_datetime(material_df['Job_Creation']).dt.to_period('M')
            monthly_costs = material_df.groupby('Month')['Cost_Per_SqFt'].mean()
            monthly_costs.index = monthly_costs.index.strftime('%Y-%m')
            st.line_chart(monthly_costs)
        else:
            st.info("Cost per square foot data not available")
    
    st.markdown("---")
    st.subheader("📊 Material Waste & Efficiency Analysis")
    
    # Material efficiency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Total_Job_SqFt' in material_df.columns:
            avg_sqft = material_df['Total_Job_SqFt'].mean()
            st.metric("Avg Job Size (SqFt)", f"{avg_sqft:.1f}")
    
    with col2:
        rework_rate = (material_df['Has_Rework'].sum() / len(material_df)) * 100
        st.metric("Material Rework Rate", f"{rework_rate:.1f}%", delta_color="inverse")
    
    with col3:
        if 'Revenue' in material_df.columns and 'Total_Job_SqFt' in material_df.columns:
            revenue_per_sqft = (material_df['Revenue'] / material_df['Total_Job_SqFt'].replace(0, 1)).mean()
            st.metric("Avg Revenue/SqFt", f"${revenue_per_sqft:.2f}")
    
    # High-waste materials identification
    st.subheader("⚠️ High-Risk Materials (Quality Issues)")
    rework_by_material = material_df.groupby('Material_Brand').agg({
        'Has_Rework': 'sum',
        'Job_Name': 'count'
    })
    rework_by_material['Rework_Rate_%'] = (rework_by_material['Has_Rework'] / rework_by_material['Job_Name']) * 100
    rework_by_material = rework_by_material[rework_by_material['Job_Name'] >= 5]  # At least 5 jobs
    high_rework = rework_by_material[rework_by_material['Rework_Rate_%'] > 10].sort_values('Rework_Rate_%', ascending=False)
    
    if not high_rework.empty:
        st.dataframe(high_rework.style.format({'Rework_Rate_%': '{:.1f}%'}), use_container_width=True)
    else:
        st.success("✅ No materials with consistently high rework rates identified")

def render_resource_optimization(df: pd.DataFrame):
    st.header("👥 Resource Optimization Dashboard")
    
    # Template crew analysis
    st.subheader("📋 Template Crew Efficiency")
    
    if 'Template_Assigned_To' in df.columns:
        template_crews = df[df['Template_Assigned_To'].notna()].copy()
        
        if not template_crews.empty:
            crew_metrics = template_crews.groupby('Template_Assigned_To').agg({
                'Job_Name': 'count',
                'Days_Template_to_RTF': 'mean',
                'Total_Job_SqFt': 'sum',
                'Has_Rework': 'sum'
            }).round(2)
            
            crew_metrics.columns = ['Total_Jobs', 'Avg_Template_to_RTF_Days', 'Total_SqFt', 'Rework_Count']
            crew_metrics['Rework_Rate_%'] = (crew_metrics['Rework_Count'] / crew_metrics['Total_Jobs']) * 100
            crew_metrics['SqFt_Per_Job'] = crew_metrics['Total_SqFt'] / crew_metrics['Total_Jobs']
            
            # Sort by efficiency (lower days to RTF is better)
            crew_metrics = crew_metrics.sort_values('Avg_Template_to_RTF_Days')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Template Crew Performance**")
                display_cols = ['Total_Jobs', 'Avg_Template_to_RTF_Days', 'Rework_Rate_%']
                st.dataframe(crew_metrics[display_cols].style.format({
                    'Avg_Template_to_RTF_Days': '{:.1f}',
                    'Rework_Rate_%': '{:.1f}%'
                }), use_container_width=True)
            
            with col2:
                st.markdown("**Workload Distribution**")
                st.bar_chart(crew_metrics['Total_Jobs'])
        else:
            st.info("No template crew data available")
    
    st.markdown("---")
    
    # Install crew analysis
    st.subheader("🔧 Install Crew Performance")
    
    if 'Install_Assigned_To' in df.columns:
        install_crews = df[df['Install_Assigned_To'].notna()].copy()
        
        if not install_crews.empty:
            install_metrics = install_crews.groupby('Install_Assigned_To').agg({
                'Job_Name': 'count',
                'Days_Ship_to_Install': 'mean',
                'Total_Job_SqFt': 'sum',
                'Revenue': 'sum'
            }).round(2)
            
            install_metrics.columns = ['Total_Installs', 'Avg_Ship_to_Install_Days', 'Total_SqFt', 'Total_Revenue']
            install_metrics['Revenue_Per_Install'] = install_metrics['Total_Revenue'] / install_metrics['Total_Installs']
            install_metrics['SqFt_Per_Install'] = install_metrics['Total_SqFt'] / install_metrics['Total_Installs']
            
            # Sort by total revenue
            install_metrics = install_metrics.sort_values('Total_Revenue', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Install Crew Performance**")
                display_cols = ['Total_Installs', 'Avg_Ship_to_Install_Days', 'Revenue_Per_Install']
                st.dataframe(install_metrics[display_cols].style.format({
                    'Avg_Ship_to_Install_Days': '{:.1f}',
                    'Revenue_Per_Install': '${:,.0f}'
                }), use_container_width=True)
            
            with col2:
                st.markdown("**Revenue Distribution**")
                st.bar_chart(install_metrics['Total_Revenue'])
        else:
            st.info("No install crew data available")
    
    st.markdown("---")
    
    # Capacity planning
    st.subheader("📈 Capacity Planning Analysis")
    
    completed_jobs = df[df['Current_Stage'] == 'Completed'].copy()
    if not completed_jobs.empty and 'Job_Creation' in completed_jobs.columns:
        completed_jobs['Month'] = pd.to_datetime(completed_jobs['Job_Creation']).dt.to_period('M')
        monthly_capacity = completed_jobs.groupby('Month').agg({
            'Job_Name': 'count',
            'Total_Job_SqFt': 'sum',
            'Revenue': 'sum'
        })
        monthly_capacity.columns = ['Jobs_Completed', 'Total_SqFt', 'Total_Revenue']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Monthly Jobs", f"{monthly_capacity['Jobs_Completed'].mean():.0f}")
        
        with col2:
            st.metric("Avg Monthly SqFt", f"{monthly_capacity['Total_SqFt'].mean():,.0f}")
        
        with col3:
            st.metric("Avg Monthly Revenue", f"${monthly_capacity['Total_Revenue'].mean():,.0f}")
        
        # Capacity trend
        monthly_capacity.index = monthly_capacity.index.strftime('%Y-%m')
        st.line_chart(monthly_capacity['Jobs_Completed'])
    else:
        st.info("Insufficient data for capacity planning analysis")

def render_quality_control_center(df: pd.DataFrame):
    st.header("🎯 Quality Control Center")
    
    # Overall quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_jobs = len(df)
    rework_jobs = df['Has_Rework'].sum()
    rework_rate = (rework_jobs / total_jobs) * 100 if total_jobs > 0 else 0
    
    with col1:
        st.metric("Total Jobs", total_jobs)
    
    with col2:
        st.metric("Jobs with Rework", rework_jobs, delta_color="inverse")
    
    with col3:
        st.metric("Overall Rework Rate", f"{rework_rate:.1f}%", delta_color="inverse")
    
    with col4:
        if 'Total_Rework_Cost' in df.columns:
            total_rework_cost = df['Total_Rework_Cost'].sum()
            st.metric("Total Rework Cost", f"${total_rework_cost:,.0f}", delta_color="inverse")
    
    st.markdown("---")
    
    # Rework analysis by different dimensions
    st.subheader("🔍 Rework Analysis by Category")
    
    analysis_tabs = st.tabs(["By Material", "By Crew", "By Customer Type", "By Time Period"])
    
    with analysis_tabs[0]:
        # Rework by material
        if 'Material_Brand' in df.columns:
            material_rework = df.groupby('Material_Brand').agg({
                'Has_Rework': 'sum',
                'Job_Name': 'count',
                'Total_Rework_Cost': 'sum'
            }).reset_index()
            material_rework['Rework_Rate_%'] = (material_rework['Has_Rework'] / material_rework['Job_Name']) * 100
            material_rework = material_rework[material_rework['Job_Name'] >= 3]  # At least 3 jobs
            material_rework = material_rework.sort_values('Rework_Rate_%', ascending=False)
            
            if not material_rework.empty:
                st.dataframe(material_rework.style.format({
                    'Rework_Rate_%': '{:.1f}%',
                    'Total_Rework_Cost': '${:,.0f}'
                }), use_container_width=True)
            else:
                st.info("No material rework data available")
        else:
            st.info("Material data not available for rework analysis")
    
    with analysis_tabs[1]:
        # Rework by crew
        crew_columns = ['Template_Assigned_To', 'Install_Assigned_To']
        for crew_type in crew_columns:
            if crew_type in df.columns:
                st.markdown(f"**{crew_type.replace('_', ' ')} Rework Analysis**")
                crew_rework = df[df[crew_type].notna()].groupby(crew_type).agg({
                    'Has_Rework': 'sum',
                    'Job_Name': 'count',
                    'Total_Rework_Cost': 'sum'
                }).reset_index()
                crew_rework['Rework_Rate_%'] = (crew_rework['Has_Rework'] / crew_rework['Job_Name']) * 100
                crew_rework = crew_rework[crew_rework['Job_Name'] >= 5]  # At least 5 jobs
                crew_rework = crew_rework.sort_values('Rework_Rate_%', ascending=False)
                
                if not crew_rework.empty:
                    st.dataframe(crew_rework.style.format({
                        'Rework_Rate_%': '{:.1f}%',
                        'Total_Rework_Cost': '${:,.0f}'
                    }), use_container_width=True)
                else:
                    st.info(f"No {crew_type} rework data available")
                st.markdown("---")
    
    with analysis_tabs[2]:
        # Rework by customer category
        if 'Customer_Category' in df.columns:
            customer_rework = df.groupby('Customer_Category').agg({
                'Has_Rework': 'sum',
                'Job_Name': 'count',
                'Total_Rework_Cost': 'sum'
            }).reset_index()
            customer_rework['Rework_Rate_%'] = (customer_rework['Has_Rework'] / customer_rework['Job_Name']) * 100
            customer_rework = customer_rework.sort_values('Rework_Rate_%', ascending=False)
            
            st.dataframe(customer_rework.style.format({
                'Rework_Rate_%': '{:.1f}%',
                'Total_Rework_Cost': '${:,.0f}'
            }), use_container_width=True)
        else:
            st.info("Customer category data not available")
    
    with analysis_tabs[3]:
        # Rework trends over time
        if 'Job_Creation' in df.columns:
            df_time = df.copy()
            df_time['Month'] = pd.to_datetime(df_time['Job_Creation']).dt.to_period('M')
            monthly_rework = df_time.groupby('Month').agg({
                'Has_Rework': 'sum',
                'Job_Name': 'count',
                'Total_Rework_Cost': 'sum'
            })
            monthly_rework['Rework_Rate_%'] = (monthly_rework['Has_Rework'] / monthly_rework['Job_Name']) * 100
            
            monthly_rework.index = monthly_rework.index.strftime('%Y-%m')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Monthly Rework Rate Trend**")
                st.line_chart(monthly_rework['Rework_Rate_%'])
            
            with col2:
                st.markdown("**Monthly Rework Cost Trend**")
                st.line_chart(monthly_rework['Total_Rework_Cost'])
        else:
            st.info("Time-based rework analysis not available")
    
    st.markdown("---")
    
    # Cost of quality analysis
    st.subheader("💰 Cost of Quality Analysis")
    
    if 'Total_Rework_Cost' in df.columns and 'Revenue' in df.columns:
        total_revenue = df['Revenue'].sum()
        total_rework_cost = df['Total_Rework_Cost'].sum()
        cost_of_quality_pct = (total_rework_cost / total_revenue) * 100 if total_revenue > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cost of Quality %", f"{cost_of_quality_pct:.2f}%")
            st.caption("Rework cost as % of total revenue")
        
        with col2:
            # Calculate potential savings
            if rework_rate > 0:
                potential_savings = total_rework_cost * 0.5  # Assume 50% reduction possible
                st.metric("Potential Annual Savings", f"${potential_savings:,.0f}")
                st.caption("50% rework reduction target")
    else:
        st.info("Cost of quality data not available")

# --- UI Rendering Functions for PROFITABILITY ANALYSIS ---

def render_overview_tab(df: pd.DataFrame, division_name: str):
    st.header(f"📈 {division_name} Overview")
    if df.empty:
        st.warning(f"No {division_name} data available for the selected period.")
        return

    total_revenue = df['Revenue'].sum()
    total_profit = df['Branch_Profit'].sum()
    avg_margin = (total_profit / total_revenue * 100) if total_revenue != 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Branch Profit", f"${total_profit:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

    st.markdown("---")
    st.subheader("Profit by Salesperson")
    if 'Salesperson' in df.columns and not df.empty:
        sales_profit = df.groupby('Salesperson')['Branch_Profit'].sum().sort_values(ascending=False)
        st.bar_chart(sales_profit)

def render_detailed_data_tab(df: pd.DataFrame, division_name: str):
    st.header(f"📋 {division_name} Detailed Data")
    df_display = df.copy()

    col1, col2 = st.columns(2)
    with col1:
        job_name_filter = st.text_input("Filter by Job Name", key=f"job_name_{division_name}")
    with col2:
        prod_num_filter = st.text_input("Filter by Production #", key=f"prod_num_{division_name}")

    if job_name_filter and 'Job_Name' in df_display.columns:
        df_display = df_display[df_display['Job_Name'].str.contains(job_name_filter, case=False, na=False)]
    if prod_num_filter and 'Production_' in df_display.columns:
        df_display = df_display[df_display['Production_'].str.contains(prod_num_filter, case=False, na=False)]

    if df_display.empty:
        st.warning("No data matches the current filters.")
        return

    base_cols = ['Link', 'Job_Name', 'Next_Sched_Activity', 'Days_Behind', 'Revenue', 'Total_Job_SqFt']
    profit_cols = ['Total_Branch_Cost', 'Branch_Profit', 'Branch_Profit_Margin_%']
    
    if division_name == 'Laminate':
        middle_cols = ['Material_Cost', 'Shop_Cost']
    else:
        middle_cols = ['Cost_From_Plant']
        profit_cols.append('Shop_Profit_Margin_%')

    column_order = base_cols + middle_cols + profit_cols
    final_column_order = [c for c in column_order if c in df_display.columns]
    
    st.dataframe(df_display[final_column_order], use_container_width=True,
        column_config={
            "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
            "Days_Behind": st.column_config.NumberColumn("Days Behind/Ahead", help="Positive: Behind. Negative: Ahead."),
            "Revenue": st.column_config.NumberColumn(format='$%.2f'),
            "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
            "Cost_From_Plant": st.column_config.NumberColumn("Production Cost", format='$%.2f'),
            "Material_Cost": st.column_config.NumberColumn("Material Cost", format='$%.2f'),
            "Shop_Cost": st.column_config.NumberColumn("Shop Cost", format='$%.2f'),
            "Total_Branch_Cost": st.column_config.NumberColumn(format='$%.2f'),
            "Branch_Profit": st.column_config.NumberColumn(format='$%.2f'),
            "Branch_Profit_Margin_%": st.column_config.ProgressColumn("Branch Profit %", format='%.2f%%', min_value=-50, max_value=100),
            "Shop_Profit_Margin_%": st.column_config.ProgressColumn("Shop Profit %", format='%.2f%%', min_value=-50, max_value=100),
        }
    )

def render_profit_drivers_tab(df: pd.DataFrame, division_name: str):
    st.header(f"💸 {division_name} Profitability Drivers")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return

    driver_options = ['Job_Type', 'Order_Type', 'Lead_Source', 'Salesperson', 'Material_Brand']
    valid_drivers = [d for d in driver_options if d in df.columns and df[d].notna().any()]

    if not valid_drivers:
        st.warning("No valid columns found for this analysis (e.g., 'Job_Type', 'Salesperson').")
        return

    selected_driver = st.selectbox("Analyze Profitability by:", valid_drivers, key=f"driver_{division_name}")
    if selected_driver:
        agg_dict = {
            'Avg_Branch_Profit_Margin': ('Branch_Profit_Margin_%', 'mean'),
            'Total_Profit': ('Branch_Profit', 'sum'),
            'Job_Count': ('Job_Name', 'count')
        }
        driver_analysis = df.groupby(selected_driver).agg(**agg_dict).sort_values('Total_Profit', ascending=False)
        st.dataframe(driver_analysis.style.format({
            'Avg_Branch_Profit_Margin': '{:.2f}%',
            'Total_Profit': '${:,.2f}'
        }), use_container_width=True)

def render_rework_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔬 {division_name} Rework & Variance")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rework Analysis")
        if 'Total_Rework_Cost' in df and 'Rework_Stone_Shop_Reason' in df.columns:
            rework_jobs = df[df['Total_Rework_Cost'] > 0].copy()
            if not rework_jobs.empty:
                st.metric("Total Rework Cost", f"${rework_jobs['Total_Rework_Cost'].sum():,.2f}", f"{len(rework_jobs)} jobs affected")
                agg_rework = rework_jobs.groupby('Rework_Stone_Shop_Reason')['Total_Rework_Cost'].agg(['sum', 'count'])
                agg_rework.columns = ['Total Rework Cost', 'Number of Jobs']
                st.dataframe(agg_rework.sort_values('Total Rework Cost', ascending=False).style.format({'Total Rework Cost': '${:,.2f}'}))
            else:
                st.info("No rework costs recorded.")
        else:
            st.info("Rework data not available.")
    with c2:
        st.subheader("Profit Variance Analysis")
        if 'Profit_Variance' in df.columns and 'Original_GM' in df.columns:
            variance_jobs = df[df['Profit_Variance'].abs() > 0.01].copy()
            if not variance_jobs.empty:
                st.metric("Jobs with Profit Variance", f"{len(variance_jobs)}")
                display_cols = ['Job_Name', 'Original_GM', 'Branch_Profit', 'Profit_Variance']
                st.dataframe(
                    variance_jobs[display_cols].sort_values(by='Profit_Variance', key=abs, ascending=False).head(20),
                    column_config={
                        "Original_GM": st.column_config.NumberColumn("Est. Profit", format='$%.2f'),
                        "Branch_Profit": st.column_config.NumberColumn("Actual Profit", format='$%.2f'),
                        "Profit_Variance": st.column_config.NumberColumn("Variance", format='$%.2f')
                    }
                )
            else:
                st.info("No significant profit variance found.")
        else:
            st.info("Profit variance data not available.")

def render_pipeline_issues_tab(df: pd.DataFrame, division_name: str, today: pd.Timestamp):
    st.header(f"🚧 {division_name} Pipeline & Issues")
    
    st.subheader("Jobs Awaiting Ready-to-Fab")
    required_cols_rtf = ['Ready_to_Fab_Status', 'Template_Date']
    if all(col in df.columns for col in required_cols_rtf):
        conditions = (df['Template_Date'].notna() & (df['Template_Date'] <= today) & (df['Ready_to_Fab_Status'].fillna('').str.lower() != 'complete'))
        stuck_jobs = df[conditions].copy()
        if not stuck_jobs.empty:
            stuck_jobs['Days_Since_Template'] = (today - stuck_jobs['Template_Date']).dt.days
            display_cols = ['Link', 'Job_Name', 'Salesperson', 'Template_Date', 'Days_Since_Template']
            st.dataframe(stuck_jobs[display_cols].sort_values(by='Days_Since_Template', ascending=False),
                         use_container_width=True, column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")})
        else:
            st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    else:
        st.warning("Could not check for jobs awaiting RTF. Required columns missing.")

    st.markdown("---")
    st.subheader("Jobs in Fabrication (Not Shipped)")
    required_cols_fab = ['Ready_to_Fab_Date', 'Ship_Date']
    if all(col in df.columns for col in required_cols_fab):
        conditions = (df['Ready_to_Fab_Date'].notna() & df['Ship_Date'].isna())
        fab_jobs = df[conditions].copy()
        if not fab_jobs.empty:
            fab_jobs['Days_Since_RTF'] = (today - fab_jobs['Ready_to_Fab_Date']).dt.days
            display_cols = ['Link', 'Job_Name', 'Salesperson', 'Ready_to_Fab_Date', 'Days_Since_RTF']
            st.dataframe(fab_jobs[display_cols].sort_values(by='Days_Since_RTF', ascending=False),
                         use_container_width=True, column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")})
        else:
            st.success("✅ No jobs are currently in fabrication without a ship date.")
    else:
        st.warning("Could not check for jobs in fabrication. Required columns missing.")

def render_workload_card(df_filtered: pd.DataFrame, activity_name: str, date_col: str, assignee_col: str):
    st.subheader(activity_name)
    if date_col not in df_filtered.columns or assignee_col not in df_filtered.columns:
        st.warning(f"Required columns for {activity_name} analysis not found: {date_col}, {assignee_col}")
        return
        
    activity_df = df_filtered.dropna(subset=[date_col, assignee_col]).copy()
    if activity_df.empty:
        st.info(f"No {activity_name.lower()} data available.")
        return

    assignees = sorted([name for name in activity_df[assignee_col].unique() if name and str(name).strip()])
    
    for assignee in assignees:
        with st.container(border=True):
            assignee_df = activity_df[activity_df[assignee_col] == assignee]
            total_jobs = len(assignee_df)
            total_sqft = assignee_df['Total_Job_SqFt'].sum() if 'Total_Job_SqFt' in assignee_df.columns else 0

            col1, col2 = st.columns(2)
            col1.metric(f"{assignee} - Total Jobs", f"{total_jobs}")
            col2.metric(f"{assignee} - Total SqFt", f"{total_sqft:,.2f}")

            with st.expander("View Weekly Breakdown"):
                agg_cols = {'Jobs': ('Production_', 'count')}
                if 'Total_Job_SqFt' in assignee_df.columns:
                    agg_cols['Total_SqFt'] = ('Total_Job_SqFt', 'sum')
                
                weekly_summary = assignee_df.set_index(date_col).resample('W-Mon', label='left', closed='left').agg(**agg_cols).reset_index()
                weekly_summary = weekly_summary[weekly_summary['Jobs'] > 0]
                
                if not weekly_summary.empty:
                    st.dataframe(weekly_summary.rename(columns={date_col: 'Week_Start_Date'}), use_container_width=True)
                else:
                    st.write("No scheduled work for this person in the selected period.")

def render_field_workload_tab(df: pd.DataFrame, division_name: str):
    st.header(f"👷 {division_name} Field Workload Planner")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        render_workload_card(df, "Templates", "Template_Date", "Template_Assigned_To")
    with col2:
        render_workload_card(df, "Installs", "Install_Date", "Install_Assigned_To")

def render_forecasting_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔮 {division_name} Forecasting & Trends")
    if 'Job_Creation' not in df.columns or df['Job_Creation'].isnull().all():
        st.warning("Job Creation date column is required for trend analysis.")
        return
    if df.empty or len(df) < 2:
        st.warning(f"Not enough {division_name} data to create a forecast.")
        return

    df_trends = df.copy().set_index('Job_Creation').sort_index()
    st.subheader("Monthly Performance Trends")
    monthly_summary = df_trends.resample('M').agg({'Revenue': 'sum', 'Branch_Profit': 'sum', 'Job_Name': 'count'}).rename(columns={'Job_Name': 'Job_Count'})
    if monthly_summary.empty:
        st.info("No data in the selected range to display monthly trends.")
        return
    st.line_chart(monthly_summary[['Revenue', 'Branch_Profit']])
    st.bar_chart(monthly_summary['Job_Count'])

# --- UI Rendering for NEW "Overall Business Health" Tab ---
def render_overall_health_tab(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚀 Overall Business Health at a Glance")
    
    df_active = df[df['Current_Stage'] != 'Completed']
    df_completed_last_30 = df[(df['Current_Stage'] == 'Completed') & (df['Install_Date'].notna()) & (df['Install_Date'] >= today - timedelta(days=30))]

    st.markdown("### Key Performance Indicators (Last 30 Days)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_profit = df_completed_last_30['Branch_Profit'].sum()
        st.metric("Total Profit", f"${total_profit:,.0f}")
    with col2:
        avg_margin = df_completed_last_30['Branch_Profit_Margin_%'].mean()
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%" if pd.notna(avg_margin) else "N/A")
    with col3:
        avg_timeline = df_completed_last_30['Days_Template_to_Install'].mean()
        st.metric("Avg Cycle Time", f"{avg_timeline:.1f} days" if pd.notna(avg_timeline) else "N/A")
    with col4:
        rework_rate = df_completed_last_30['Has_Rework'].mean() * 100
        st.metric("Rework Rate", f"{rework_rate:.1f}%" if pd.notna(rework_rate) else "N/A")

    st.markdown("---")
    st.markdown("### Current Operational Status (Active Jobs)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Jobs", len(df_active))
    with col2:
        st.metric("🔴 High Risk Jobs", len(df_active[df_active['Risk_Score'] >= 30]))
    with col3:
        st.metric("🚧 Stuck Jobs", len(df_active[df_active['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]))
    with col4:
        st.metric("⏰ Behind Schedule", len(df_active[df_active['Days_Behind'] > 0]))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current Bottlenecks (Active Jobs)")
        stage_counts = df_active['Current_Stage'].value_counts()
        if not stage_counts.empty:
            st.bar_chart(stage_counts)
    with c2:
        st.subheader("Profitability by Division (Last 30 Days)")
        profit_by_div = df_completed_last_30.groupby('Division_Type')['Branch_Profit'].sum()
        if not profit_by_div.empty:
            st.bar_chart(profit_by_div)

# --- Main Application ---
def main():
    # 🔐 AUTHENTICATION - Add this at the very beginning
    if not render_faceid_style_auth():  # Choose your preferred method
        return
    
    st.title("🚀 Unified Operations & Profitability Dashboard")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Configuration")
    
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Sign Out"):
            st.session_state.authenticated = False
            st.success("👋 Signed out successfully!")
            time.sleep(1)
            st.rerun()
    
    today_dt = pd.to_datetime(st.sidebar.date_input("Select 'Today's' Date", value=datetime.now().date()))
    
    install_cost_sqft = st.sidebar.number_input("Install Cost per SqFt ($)", min_value=0.0, value=15.0, step=0.50)

    # --- Data Loading ---
    try:
        with st.spinner("Loading and processing all job data..."):
            # The data loading function now uses Streamlit secrets
            df_stone, df_laminate, df_full = load_and_process_data(today_dt, install_cost_sqft)
    except Exception as e:
        st.error(f"Failed to load or process data. Error: {e}")
        st.error("Please check your Streamlit secrets configuration in the app settings.")
        st.info("Make sure you have set up the 'gsheets' connection in your secrets.toml file.")
        st.stop()

    if df_full.empty:
        st.error("No data loaded. Please check your Google Sheets connection and data.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.info(f"Data loaded for {len(df_full)} jobs.")
    st.sidebar.info(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display connection status
    with st.sidebar:
        if len(df_full) > 0:
            st.success("✅ Connected to Google Sheets")
        else:
            st.error("❌ Connection failed")
    
    # Display connection status
    with st.sidebar:
        if len(df_full) > 0:
            st.success("✅ Connected to Google Sheets")
        else:
            st.error("❌ Connection failed")

    # --- Main Application Tabs ---
    main_tabs = st.tabs(["📈 Overall Business Health", "⚙️ Operational Performance", "💰 Profitability Analysis"])

    with main_tabs[0]:
        render_overall_health_tab(df_full, today_dt)

    with main_tabs[1]:
        st.header("Operational Performance Dashboard")
        st.markdown("Analyze real-time operational efficiency, risks, and workload.")
        
        op_cols = st.columns(2)
        with op_cols[0]:
            salesperson_list = ['All'] + sorted(df_full['Salesperson'].dropna().unique().tolist())
            salesperson_filter = st.selectbox("Filter by Salesperson", salesperson_list, key="op_sales")
        with op_cols[1]:
            division_list = ['All'] + sorted(df_full['Division_Type'].dropna().unique().tolist())
            division_filter = st.selectbox("Filter by Division", division_list, key="op_div")

        # --- Filtering Logic (Simplified - No Status Filter) ---
        df_op_filtered = df_full.copy()

        # Apply filters
        if salesperson_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Salesperson'] == salesperson_filter]
        
        if division_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Division_Type'] == division_filter]
        
        st.info(f"Displaying {len(df_op_filtered)} jobs based on filters.")

        op_sub_tabs = st.tabs(["🚨 Daily Priorities", "📅 Workload Calendar", "📊 Timeline Analytics", "🔮 Predictive Analytics", "🎯 Performance Scorecards", "📈 Historical Trends", "💰 Cash Flow Forecast", "🔬 Material Intelligence", "👥 Resource Optimization", "🎯 Quality Control"])
        with op_sub_tabs[0]: render_daily_priorities(df_op_filtered, today_dt)
        with op_sub_tabs[1]: render_workload_calendar(df_op_filtered, today_dt)
        with op_sub_tabs[2]: render_timeline_analytics(df_op_filtered)
        with op_sub_tabs[3]: render_predictive_analytics(df_op_filtered)
        with op_sub_tabs[4]: render_performance_scorecards(df_op_filtered)
        with op_sub_tabs[5]: render_historical_trends(df_full)
        with op_sub_tabs[6]: render_cash_flow_forecasting(df_op_filtered, today_dt)
        with op_sub_tabs[7]: render_material_cost_intelligence(df_op_filtered)
        with op_sub_tabs[8]: render_resource_optimization(df_op_filtered)
        with op_sub_tabs[9]: render_quality_control_center(df_op_filtered)

    with main_tabs[2]:
        st.header("Profitability Analysis Dashboard")
        st.markdown("Analyze financial performance, costs, and profit drivers by division.")

        # Use the pre-processed dataframes directly
        profit_sub_tabs = st.tabs(["💎 Stone/Quartz", "🪵 Laminate"])
        
        with profit_sub_tabs[0]:
            stone_tabs = st.tabs(["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"])
            with stone_tabs[0]: render_overview_tab(df_stone, "Stone/Quartz")
            with stone_tabs[1]: render_detailed_data_tab(df_stone, "Stone/Quartz")
            with stone_tabs[2]: render_profit_drivers_tab(df_stone, "Stone/Quartz")
            with stone_tabs[3]: render_rework_tab(df_stone, "Stone/Quartz")
            with stone_tabs[4]: render_pipeline_issues_tab(df_stone, "Stone/Quartz", today_dt)
            with stone_tabs[5]: render_field_workload_tab(df_stone, "Stone/Quartz")
            with stone_tabs[6]: render_forecasting_tab(df_stone, "Stone/Quartz")

        with profit_sub_tabs[1]:
            laminate_tabs = st.tabs(["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"])
            with laminate_tabs[0]: render_overview_tab(df_laminate, "Laminate")
            with laminate_tabs[1]: render_detailed_data_tab(df_laminate, "Laminate")
            with laminate_tabs[2]: render_profit_drivers_tab(df_laminate, "Laminate")
            with laminate_tabs[3]: render_rework_tab(df_laminate, "Laminate")
            with laminate_tabs[4]: render_pipeline_issues_tab(df_laminate, "Laminate", today_dt)
            with laminate_tabs[5]: render_field_workload_tab(df_laminate, "Laminate")
            with laminate_tabs[6]: render_forecasting_tab(df_laminate, "Laminate")


if __name__ == "__main__":
    main()
