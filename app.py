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
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-
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

# --- Page & App Configuration (Must be the first Streamlit command) ---
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

# --- Authentication Function ---
def render_login_screen():
    """
    Displays a PIN-based login screen. Returns True if authenticated, False otherwise.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Custom CSS for the login screen
    st.markdown("""
        <style>
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 70vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: auto;
                color: white;
                text-align: center;
                max-width: 500px;
            }
            .login-title {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            .login-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Secure Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Enter your PIN to access the dashboard</div>', unsafe_allow_html=True)

    # Use a form for PIN input
    with st.form("pin_auth_form"):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed", placeholder="Enter PIN")
        submitted = st.form_submit_button("🔓 Unlock", use_container_width=True)

        if submitted:
            # Check the PIN against a secret
            # IMPORTANT: Set this PIN in your Streamlit secrets!
            correct_pin = st.secrets.get("APP_PIN", "1234") # Fallback to "1234" if not in secrets
            if pin == correct_pin:
                st.session_state.authenticated = True
                st.success("Authentication successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid PIN. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)
    return False

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
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=WORKSHEET_NAME, ttl=300)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Streamlit secrets configuration.")
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
        if col not in df.columns:
            df[col] = None

    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date',
                 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

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

    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    
    df_stone_processed = _process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)

    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)

    return df_stone_processed, df_laminate_processed, df_combined

# --- UI Rendering Functions ---

def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚨 Daily Priorities & Warnings")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 High Risk Jobs", len(df[df['Risk_Score'] >= 30]))
    with col2:
        st.metric("⏰ Behind Schedule", len(df[df['Days_Behind'] > 0]))
    with col3:
        st.metric("🚧 Stuck Jobs", len(df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]))
    with col4:
        stale_jobs_metric = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
        st.metric("💨 Stale Jobs", len(stale_jobs_metric))

    st.markdown("---")
    st.subheader("⚡ Critical Issues Requiring Immediate Attention")
    st.caption("This section only shows jobs where 'Job Status' is not 'Complete'.")
    
    missing_activity = df[(df['Next_Sched_Activity'].isna()) & (df['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received'])) & (df['Job_Status'] != 'Complete')]
    stale_jobs = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
    template_to_rtf_stuck = df[(df['Template_Date'].notna()) & (df['Ready_to_Fab_Date'].isna()) & ((today - df['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf']) & (df['Job_Status'] != 'Complete')]
    upcoming_installs = df[(df['Install_Date'].notna()) & (df['Install_Date'] <= today + timedelta(days=7)) & (df['Product_Rcvd_Date'].isna()) & (df['Job_Status'] != 'Complete')]

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

def render_profitability_tabs(df_stone, df_laminate, today_dt):
    st.header("Profitability Analysis Dashboard")
    st.markdown("Analyze financial performance, costs, and profit drivers by division.")

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

    st.markdown("---")
    st.subheader("Jobs Ready for Invoicing")
    
    # Check for jobs that should be invoiced
    # Logic: If Install, Pick Up, or Delivery is complete but Invoice is not complete
    required_cols_invoice = ['Invoice_Status', 'Install_Status', 'Pick_Up_Status', 'Delivery_Status']
    
    if all(col in df.columns for col in required_cols_invoice):
        # Helper function to safely check status
        def safe_status_check(status_value):
            if pd.isna(status_value):
                return False
            return str(status_value).lower() == 'complete'
        
        # Jobs where at least one completion activity is done but invoice is not complete
        completion_conditions = (
            df['Install_Status'].apply(safe_status_check) |
            df['Pick_Up_Status'].apply(safe_status_check) |
            df['Delivery_Status'].apply(safe_status_check)
        )
        
        invoice_not_complete = ~df['Invoice_Status'].apply(safe_status_check)
        
        jobs_to_invoice = df[completion_conditions & invoice_not_complete].copy()
        
        if not jobs_to_invoice.empty:
            # Add completion status column for display
            def get_completion_status(row):
                completed_activities = []
                if safe_status_check(row.get('Install_Status')):
                    completed_activities.append('Install')
                if safe_status_check(row.get('Pick_Up_Status')):
                    completed_activities.append('Pick Up')
                if safe_status_check(row.get('Delivery_Status')):
                    completed_activities.append('Delivery')
                return ', '.join(completed_activities) if completed_activities else 'None'
            
            jobs_to_invoice['Completion_Status'] = jobs_to_invoice.apply(get_completion_status, axis=1)
            
            display_cols = ['Link', 'Job_Name', 'Salesperson', 'Completion_Status', 'Invoice_Status']
            st.dataframe(
                jobs_to_invoice[display_cols].sort_values(by='Job_Name'),
                use_container_width=True, 
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
                    "Completion_Status": st.column_config.TextColumn("Completed Activities"),
                    "Invoice_Status": st.column_config.TextColumn("Invoice Status")
                }
            )
            st.info(f"📋 {len(jobs_to_invoice)} jobs are ready for invoicing")
        else:
            st.success("✅ No jobs are currently awaiting invoicing.")
    else:
        missing_cols = [col for col in required_cols_invoice if col not in df.columns]
        st.warning(f"Could not check for jobs ready for invoicing. Missing columns: {', '.join(missing_cols)}")

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

def render_overall_health_tab(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚀 Overall Business Health at a Glance")
    
    df_active = df[df['Job_Status'] != 'Complete']
    df_completed_last_30 = df[(df['Job_Status'] == 'Complete') & (df['Install_Date'].notna()) & (df['Install_Date'] >= today - timedelta(days=30))]

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
    if not render_login_screen():
        return

    st.title("🚀 Unified Operations & Profitability Dashboard")
    
    st.sidebar.header("⚙️ Configuration")
    today_dt = pd.to_datetime(st.sidebar.date_input("Select 'Today's' Date", value=datetime.now().date()))
    install_cost_sqft = st.sidebar.number_input("Install Cost per SqFt ($)", min_value=0.0, value=15.0, step=0.50)

    try:
        with st.spinner("Loading and processing all job data..."):
            df_stone, df_laminate, df_full = load_and_process_data(today_dt, install_cost_sqft)
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.exception(e)
        st.stop()

    if df_full.empty:
        st.error("No data loaded. Please check your Google Sheets connection and data.")
        st.stop()

    st.sidebar.info(f"Data loaded for {len(df_full)} jobs.")
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
            if "Active" in status_filter:
                final_mask |= (df_full['Job_Status'] != 'Complete')
            if "Complete" in status_filter:
                final_mask |= (df_full['Job_Status'] == 'Complete')
            if "30+ Days Old" in status_filter:
                thirty_days_ago = today_dt - timedelta(days=30)
                final_mask |= ((df_full['Job_Creation'] < thirty_days_ago) & (df_full['Job_Status'] != 'Complete'))
            if "Unscheduled" in status_filter:
                final_mask |= (df_full['Next_Sched_Date'].isna() & (df_full['Job_Status'] != 'Complete'))
            df_op_filtered = df_full[final_mask]
        else:
            df_op_filtered = pd.DataFrame(columns=df_full.columns)

        if salesperson_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Salesperson'] == salesperson_filter]
        if division_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Division_Type'] == division_filter]
        
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
import seaborn as sns
import time

# --- Page & App Configuration (Must be the first Streamlit command) ---
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

# --- Authentication Function ---
def render_login_screen():
    """
    Displays a PIN-based login screen. Returns True if authenticated, False otherwise.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Custom CSS for the login screen
    st.markdown("""
        <style>
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 70vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: auto;
                color: white;
                text-align: center;
                max-width: 500px;
            }
            .login-title {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            .login-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Secure Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Enter your PIN to access the dashboard</div>', unsafe_allow_html=True)

    # Use a form for PIN input
    with st.form("pin_auth_form"):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed", placeholder="Enter PIN")
        submitted = st.form_submit_button("🔓 Unlock", use_container_width=True)

        if submitted:
            # Check the PIN against a secret
            # IMPORTANT: Set this PIN in your Streamlit secrets!
            correct_pin = st.secrets.get("APP_PIN", "1234") # Fallback to "1234" if not in secrets
            if pin == correct_pin:
                st.session_state.authenticated = True
                st.success("Authentication successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid PIN. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)
    return False

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
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=WORKSHEET_NAME, ttl=300)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Streamlit secrets configuration.")
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
        if col not in df.columns:
            df[col] = None

    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date',
                 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

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

    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    
    df_stone_processed = _process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)

    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)

    return df_stone_processed, df_laminate_processed, df_combined

# --- UI Rendering Functions ---

def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚨 Daily Priorities & Warnings")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 High Risk Jobs", len(df[df['Risk_Score'] >= 30]))
    with col2:
        st.metric("⏰ Behind Schedule", len(df[df['Days_Behind'] > 0]))
    with col3:
        st.metric("🚧 Stuck Jobs", len(df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]))
    with col4:
        stale_jobs_metric = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
        st.metric("💨 Stale Jobs", len(stale_jobs_metric))

    st.markdown("---")
    st.subheader("⚡ Critical Issues Requiring Immediate Attention")
    st.caption("This section only shows jobs where 'Job Status' is not 'Complete'.")
    
    missing_activity = df[(df['Next_Sched_Activity'].isna()) & (df['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received'])) & (df['Job_Status'] != 'Complete')]
    stale_jobs = df[(df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & (df['Job_Status'] != 'Complete')]
    template_to_rtf_stuck = df[(df['Template_Date'].notna()) & (df['Ready_to_Fab_Date'].isna()) & ((today - df['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf']) & (df['Job_Status'] != 'Complete')]
    upcoming_installs = df[(df['Install_Date'].notna()) & (df['Install_Date'] <= today + timedelta(days=7)) & (df['Product_Rcvd_Date'].isna()) & (df['Job_Status'] != 'Complete')]

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

def render_profitability_tabs(df_stone, df_laminate, today_dt):
    st.header("Profitability Analysis Dashboard")
    st.markdown("Analyze financial performance, costs, and profit drivers by division.")

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

    st.markdown("---")
    st.subheader("Jobs Ready for Invoicing")
    
    # Check for jobs that should be invoiced
    # Logic: If Install, Pick Up, or Delivery is complete but Invoice is not complete
    required_cols_invoice = ['Invoice_Status', 'Install_Status', 'Pick_Up_Status', 'Delivery_Status']
    
    if all(col in df.columns for col in required_cols_invoice):
        # Jobs where at least one completion activity is done but invoice is not complete
        completion_conditions = (
            (df['Install_Status'].fillna('').str.lower() == 'complete') |
            (df['Pick_Up_Status'].fillna('').str.lower() == 'complete') |
            (df['Delivery_Status'].fillna('').str.lower() == 'complete')
        )
        
        invoice_not_complete = df['Invoice_Status'].fillna('').str.lower() != 'complete'
        
        jobs_to_invoice = df[completion_conditions & invoice_not_complete].copy()
        
        if not jobs_to_invoice.empty:
            # Add completion status column for display
            jobs_to_invoice['Completion_Status'] = jobs_to_invoice.apply(
                lambda row: ', '.join([
                    status for status in [
                        'Install' if row.get('Install_Status', '').lower() == 'complete' else None,
                        'Pick Up' if row.get('Pick_Up_Status', '').lower() == 'complete' else None,
                        'Delivery' if row.get('Delivery_Status', '').lower() == 'complete' else None
                    ] if status is not None
                ]), axis=1
            )
            
            display_cols = ['Link', 'Job_Name', 'Salesperson', 'Completion_Status', 'Invoice_Status']
            st.dataframe(
                jobs_to_invoice[display_cols].sort_values(by='Job_Name'),
                use_container_width=True, 
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
                    "Completion_Status": st.column_config.TextColumn("Completed Activities"),
                    "Invoice_Status": st.column_config.TextColumn("Invoice Status")
                }
            )
            st.info(f"📋 {len(jobs_to_invoice)} jobs are ready for invoicing")
        else:
            st.success("✅ No jobs are currently awaiting invoicing.")
    else:
        missing_cols = [col for col in required_cols_invoice if col not in df.columns]
        st.warning(f"Could not check for jobs ready for invoicing. Missing columns: {', '.join(missing_cols)}")

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

def render_overall_health_tab(df: pd.DataFrame, today: pd.Timestamp):
    st.header("🚀 Overall Business Health at a Glance")
    
    df_active = df[df['Job_Status'] != 'Complete']
    df_completed_last_30 = df[(df['Job_Status'] == 'Complete') & (df['Install_Date'].notna()) & (df['Install_Date'] >= today - timedelta(days=30))]

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
    if not render_login_screen():
        return

    st.title("🚀 Unified Operations & Profitability Dashboard")
    
    st.sidebar.header("⚙️ Configuration")
    today_dt = pd.to_datetime(st.sidebar.date_input("Select 'Today's' Date", value=datetime.now().date()))
    install_cost_sqft = st.sidebar.number_input("Install Cost per SqFt ($)", min_value=0.0, value=15.0, step=0.50)

    try:
        with st.spinner("Loading and processing all job data..."):
            df_stone, df_laminate, df_full = load_and_process_data(today_dt, install_cost_sqft)
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.exception(e)
        st.stop()

    if df_full.empty:
        st.error("No data loaded. Please check your Google Sheets connection and data.")
        st.stop()

    st.sidebar.info(f"Data loaded for {len(df_full)} jobs.")
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
            if "Active" in status_filter:
                final_mask |= (df_full['Job_Status'] != 'Complete')
            if "Complete" in status_filter:
                final_mask |= (df_full['Job_Status'] == 'Complete')
            if "30+ Days Old" in status_filter:
                thirty_days_ago = today_dt - timedelta(days=30)
                final_mask |= ((df_full['Job_Creation'] < thirty_days_ago) & (df_full['Job_Status'] != 'Complete'))
            if "Unscheduled" in status_filter:
                final_mask |= (df_full['Next_Sched_Date'].isna() & (df_full['Job_Status'] != 'Complete'))
            df_op_filtered = df_full[final_mask]
        else:
            df_op_filtered = pd.DataFrame(columns=df_full.columns)

        if salesperson_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Salesperson'] == salesperson_filter]
        if division_filter != 'All':
            df_op_filtered = df_op_filtered[df_op_filtered['Division_Type'] == division_filter]
        
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
