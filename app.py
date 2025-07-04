# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This version creates two separate, dedicated dashboards for Stone/Quartz and
Laminate divisions, each with its own tailored calculations and views. All
previous analytical tabs have been restored.
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime, timedelta

# --- Attempt to import optional libraries for advanced features ---
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- Page & App Configuration ---
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="💰")

# --- Constants & Global Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="
INSTALL_COST_PER_SQFT = 15.0


# --- Helper Functions ---

def parse_material(s: str) -> tuple[str, str]:
    """Parses a material description string to extract brand and color."""
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

# --- Data Loading and Processing Sub-Functions ---

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes all column names for easier access."""
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    return df

def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses all known date columns to datetime objects."""
    date_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date',
        'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date',
        'Product_Rcvd_Date', 'Pick_Up_Date'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def _calculate_days_behind(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Calculates if a job is ahead or behind its next scheduled date."""
    if 'Next_Sched_Date' in df.columns:
        df['Days_Behind'] = (today - df['Next_Sched_Date']).dt.days
    else:
        df['Days_Behind'] = np.nan
    return df

def _calculate_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates durations between key process stages."""
    if 'Ready_to_Fab_Date' in df.columns and 'Template_Date' in df.columns:
        df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
        df.loc[df['Days_Template_to_RTF'] < 0, 'Days_Template_to_RTF'] = np.nan
    if 'Ship_Date' in df.columns and 'Ready_to_Fab_Date' in df.columns:
        df['Days_RTF_to_Ship'] = (df['Ship_Date'] - df['Ready_to_Fab_Date']).dt.days
        df.loc[df['Days_RTF_to_Ship'] < 0, 'Days_RTF_to_Ship'] = np.nan
    if 'Install_Date' in df.columns and 'Ship_Date' in df.columns:
        df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
        df.loc[df['Days_Ship_to_Install'] < 0, 'Days_Ship_to_Install'] = np.nan
    return df

def _enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds supplementary data like material parsing."""
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material_Brand'] = "N/A"
        df['Material_Color'] = "N/A"
    return df

# --- Division-Specific Processing Pipelines ---

def process_stone_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the data specifically for Stone/Quartz jobs."""
    df_stone = df.copy()
    
    numeric_map = {
        'Total_Job_Price_': 'Revenue',
        'Phase_Dollars_Plant_Invoice_': 'Cost_From_Plant',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
        'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor',
        'Job_Throughput_Total_COGS': 'Total_COGS'
    }
    for original, new in numeric_map.items():
        if original in df_stone.columns:
            df_stone[new] = pd.to_numeric(df_stone[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else:
            df_stone[new] = 0.0
            
    df_stone['Install_Cost'] = df_stone.get('Total_Job_SqFt', 0) * INSTALL_COST_PER_SQFT
    df_stone['Total_Rework_Cost'] = df_stone.get('Rework_Price', 0) + df_stone.get('Rework_COGS', 0) + df_stone.get('Rework_Labor', 0)
    df_stone['Total_Branch_Cost'] = df_stone.get('Cost_From_Plant', 0) + df_stone['Install_Cost'] + df_stone['Total_Rework_Cost']
    df_stone['Branch_Profit'] = df_stone.get('Revenue', 0) - df_stone['Total_Branch_Cost']
    df_stone['Branch_Profit_Margin_%'] = df_stone.apply(
        lambda row: (row['Branch_Profit'] / row['Revenue'] * 100) if row.get('Revenue') != 0 else 0, axis=1
    )
    df_stone['Profit_Variance'] = df_stone['Branch_Profit'] - df_stone.get('Original_GM', 0)
    
    # Shop Profitability
    df_stone['Shop_Profit'] = df_stone.get('Cost_From_Plant', 0) - df_stone.get('Total_COGS', 0)
    df_stone['Shop_Profit_Margin_%'] = df_stone.apply(
        lambda row: (row['Shop_Profit'] / row['Cost_From_Plant'] * 100) if row.get('Cost_From_Plant') != 0 else 0, axis=1
    )
    
    return df_stone

def process_laminate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the data specifically for Laminate jobs."""
    df_lam = df.copy()

    numeric_map = {
        'Total_Job_Price_': 'Revenue',
        'Branch_INV_': 'Shop_Cost',
        'Plant_INV_': 'Material_Cost',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
    }
    for original, new in numeric_map.items():
        if original in df_lam.columns:
            df_lam[new] = pd.to_numeric(df_lam[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else:
            df_lam[new] = 0.0

    df_lam['Install_Cost'] = df_lam.get('Total_Job_SqFt', 0) * INSTALL_COST_PER_SQFT
    df_lam['Total_Rework_Cost'] = df_lam.get('Rework_Price', 0)
    df_lam['Total_Branch_Cost'] = df_lam.get('Shop_Cost', 0) + df_lam.get('Material_Cost', 0) + df_lam['Install_Cost'] + df_lam['Total_Rework_Cost']
    df_lam['Branch_Profit'] = df_lam.get('Revenue', 0) - df_lam['Total_Branch_Cost']
    df_lam['Branch_Profit_Margin_%'] = df_lam.apply(
        lambda row: (row['Branch_Profit'] / row['Revenue'] * 100) if row.get('Revenue') != 0 else 0, axis=1
    )
    df_lam['Profit_Variance'] = df_lam['Branch_Profit'] - df_lam.get('Original_GM', 0)
    return df_lam

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data and splits it into two processed dataframes for each division."""
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(worksheet.get_all_records())

    df = _clean_column_names(df)
    df = _parse_date_columns(df)
    df = _calculate_days_behind(df, today)
    df = _calculate_durations(df)
    df = _enrich_data(df)

    if 'Division' not in df.columns:
        st.error("Your Google Sheet must have a 'Division' column to separate Stone and Laminate jobs.")
        st.stop()

    df_stone = df[df['Division'].str.contains("Stone/Quartz", na=False)].copy()
    df_laminate = df[df['Division'].str.contains("Laminate", na=False)].copy()

    df_stone_processed = process_stone_data(df_stone)
    df_laminate_processed = process_laminate_data(df_laminate)

    return df_stone_processed, df_laminate_processed


# --- UI Rendering Functions ---

def render_overview_tab(df: pd.DataFrame, division_name: str):
    st.header(f"📈 {division_name} Overview")
    if df.empty:
        st.warning(f"No {division_name} data available.")
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

    if job_name_filter:
        df_display = df_display[df_display['Job_Name'].str.contains(job_name_filter, case=False, na=False)]
    if prod_num_filter:
        df_display = df_display[df_display['Production_'].str.contains(prod_num_filter, case=False, na=False)]

    if df_display.empty:
        st.warning("No data matches the current filters.")
        return

    if 'Production_' in df_display.columns:
        df_display['Link'] = df_display['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)

    column_order = [
        'Link', 'Job_Name', 'Next_Sched_Activity', 'Days_Behind', 'Revenue', 
        'Total_Job_SqFt', 'Total_Branch_Cost', 'Branch_Profit', 'Branch_Profit_Margin_%'
    ]
    
    if division_name == 'Laminate':
        column_order.insert(6, 'Shop_Cost')
        column_order.insert(6, 'Material_Cost')
    else: # Stone
        column_order.insert(6, 'Cost_From_Plant')
        column_order.append('Shop_Profit_Margin_%')

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
    valid_drivers = [d for d in driver_options if d in df.columns]

    if not valid_drivers:
        st.warning("No valid columns found for this analysis.")
        return

    selected_driver = st.selectbox("Analyze Profitability by:", valid_drivers, key=f"driver_{division_name}")
    if selected_driver:
        agg_dict = {'Avg_Branch_Profit_Margin': ('Branch_Profit_Margin_%', 'mean'), 'Total_Profit': ('Branch_Profit', 'sum'), 'Job_Count': ('Job_Name', 'count')}
        driver_analysis = df.groupby(selected_driver).agg(**agg_dict).sort_values('Total_Profit', ascending=False)
        st.dataframe(driver_analysis.style.format({'Avg_Branch_Profit_Margin': '{:.2f}%', 'Total_Profit': '${:,.2f}'}), use_container_width=True)

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
                st.dataframe(variance_jobs[display_cols].sort_values(by='Profit_Variance', key=abs, ascending=False).head(20),
                    column_config={"Original_GM": st.column_config.NumberColumn("Est. Profit", format='$%.2f'), "Branch_Profit": st.column_config.NumberColumn("Actual Profit", format='$%.2f'), "Profit_Variance": st.column_config.NumberColumn("Variance", format='$%.2f')})
            else:
                st.info("No significant profit variance found.")
        else:
            st.info("Profit variance data not available.")

def render_pipeline_issues_tab(df: pd.DataFrame, division_name: str, today: pd.Timestamp):
    st.header(f"🚧 {division_name} Pipeline & Issues")
    
    st.subheader("Jobs Awaiting Ready-to-Fab")
    if 'Ready_to_Fab_Status' in df.columns and 'Template_Date' in df.columns:
        conditions = (df['Template_Date'].notna() & (df['Template_Date'] <= today) & (df['Ready_to_Fab_Status'].fillna('').str.lower() != 'complete'))
        stuck_jobs = df[conditions].copy()
        if not stuck_jobs.empty:
            stuck_jobs['Days_Since_Template'] = (today - stuck_jobs['Template_Date']).dt.days
            display_cols = ['Job_Name', 'Salesperson', 'Template_Date', 'Days_Since_Template']
            st.dataframe(stuck_jobs[display_cols].sort_values(by='Days_Since_Template', ascending=False), use_container_width=True)
        else:
            st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    else:
        st.warning("Could not check for jobs awaiting RTF. Required columns missing.")

    st.markdown("---")
    st.subheader("Jobs with Scheduling Conflicts")
    activity_cols = ['Install_Date', 'Service_Date', 'Pick_Up_Date', 'Delivery_Date']
    product_received_col = 'Product_Rcvd_Date'
    if all(col in df.columns for col in activity_cols + [product_received_col]):
        conflict_conditions = ((df['Install_Date'].notna() & (df['Install_Date'] < df[product_received_col])) | (df['Service_Date'].notna() & (df['Service_Date'] < df[product_received_col])) | (df['Pick_Up_Date'].notna() & (df['Pick_Up_Date'] < df[product_received_col])) | (df['Delivery_Date'].notna() & (df['Delivery_Date'] < df[product_received_col])))
        conflict_jobs = df[conflict_conditions].copy()
        if not conflict_jobs.empty:
            display_cols = ['Job_Name', 'Product_Rcvd_Date', 'Install_Date', 'Service_Date']
            st.dataframe(conflict_jobs[[c for c in display_cols if c in conflict_jobs.columns]], use_container_width=True)
        else:
            st.success("✅ No scheduling conflicts found.")
    else:
        st.warning("Could not perform scheduling conflict analysis. Required columns are missing.")

def render_field_workload_tab(df: pd.DataFrame, division_name: str):
    st.header(f"👷 {division_name} Field Workload Planner")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return

    def render_workload_analysis(df_filtered: pd.DataFrame, activity_name: str, date_col: str, assignee_col: str):
        st.subheader(activity_name)
        if date_col not in df_filtered.columns or assignee_col not in df_filtered.columns:
            st.warning(f"Required columns for {activity_name} analysis not found.")
            return
        activity_df = df_filtered.dropna(subset=[date_col, assignee_col]).copy()
        if activity_df.empty:
            st.info(f"No {activity_name.lower()} data available.")
            return
        assignees = sorted([name for name in activity_df[assignee_col].unique() if name and str(name).strip()])
        for assignee in assignees:
            with st.expander(f"**{assignee}**"):
                assignee_df = activity_df[activity_df[assignee_col] == assignee].copy()
                weekly_summary = assignee_df.set_index(date_col).resample('W-Mon', label='left', closed='left').agg(Jobs=('Production_', 'count'), Total_SqFt=('Total_Job_SqFt', 'sum')).reset_index()
                weekly_summary = weekly_summary[weekly_summary['Jobs'] > 0]
                if not weekly_summary.empty:
                    st.dataframe(weekly_summary.rename(columns={date_col: 'Week_Start_Date'}), use_container_width=True)
                else:
                    st.write("No scheduled work for this person.")

    render_workload_analysis(df, "Templates", "Template_Date", "Template_Assigned_To")
    render_workload_analysis(df, "Installs", "Install_Date", "Install_Assigned_To")

def render_forecasting_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔮 {division_name} Forecasting & Trends")
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.error("Forecasting features require scikit-learn and matplotlib.")
        return
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

# --- Main Application Logic ---

def main():
    st.title("💰 Enhanced Job Profitability Dashboard")
    st.markdown("An interactive dashboard to analyze job data and drive profitability, separated by division.")
    st.sidebar.header("⚙️ Configuration")
    creds = None
    if "google_creds_json" in st.secrets:
        creds = json.loads(st.secrets["google_creds_json"])
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Google Service Account JSON", type="json")
        if uploaded_file:
            creds = json.load(uploaded_file)
    if not creds:
        st.sidebar.error("Please provide Google credentials to load data.")
        st.stop()

    today_date = st.sidebar.date_input("Select 'Today's' Date for Calculations", value=datetime.now().date(), help="This date is used for 'Days Behind' calculations.")
    today_dt = pd.to_datetime(today_date)

    try:
        df_stone, df_laminate = load_and_process_data(creds, today_dt)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.exception(e)
        st.stop()

    division_tabs = st.tabs(["Stone/Quartz Dashboard", "Laminate Dashboard"])
    with division_tabs[0]:
        stone_sub_tabs = st.tabs(["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"])
        with stone_sub_tabs[0]: render_overview_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[1]: render_detailed_data_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[2]: render_profit_drivers_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[3]: render_rework_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[4]: render_pipeline_issues_tab(df_stone, "Stone/Quartz", today_dt)
        with stone_sub_tabs[5]: render_field_workload_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[6]: render_forecasting_tab(df_stone, "Stone/Quartz")
    with division_tabs[1]:
        laminate_sub_tabs = st.tabs(["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"])
        with laminate_sub_tabs[0]: render_overview_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[1]: render_detailed_data_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[2]: render_profit_drivers_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[3]: render_rework_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[4]: render_pipeline_issues_tab(df_laminate, "Laminate", today_dt)
        with laminate_sub_tabs[5]: render_field_workload_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[6]: render_forecasting_tab(df_laminate, "Laminate")

if __name__ == "__main__":
    main()
