# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This refactored version incorporates best practices such as centralized constants,
DRY (Don't Repeat Yourself) principles for UI generation, and more granular
error handling to create a more robust and maintainable application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime

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

# --- REFACTOR: Centralized Column Name Constants ---
# Using a class to hold column names makes the code more robust.
# If a column name changes in the source data, you only need to update it here.
class COLS:
    # Input Columns (from Google Sheet, after cleaning)
    DIVISION = 'Division'
    JOB_NAME = 'Job_Name'
    PRODUCTION_NUM = 'Production_'
    SALESPERSON = 'Salesperson'
    JOB_TYPE = 'Job_Type'
    ORDER_TYPE = 'Order_Type'
    LEAD_SOURCE = 'Lead_Source'
    JOB_MATERIAL = 'Job_Material'
    REWORK_REASON = 'Rework_Stone_Shop_Reason'
    RTF_STATUS = 'Ready_to_Fab_Status'
    NEXT_SCHED_ACTIVITY = 'Next_Sched_Activity'

    # Date Columns
    TEMPLATE_DATE = 'Template_Date'
    RTF_DATE = 'Ready_to_Fab_Date'
    SHIP_DATE = 'Ship_Date'
    INSTALL_DATE = 'Install_Date'
    SERVICE_DATE = 'Service_Date'
    DELIVERY_DATE = 'Delivery_Date'
    JOB_CREATION_DATE = 'Job_Creation'
    NEXT_SCHED_DATE = 'Next_Sched_Date'
    PRODUCT_RCVD_DATE = 'Product_Rcvd_Date'
    PICK_UP_DATE = 'Pick_Up_Date'
    
    # Assignee Columns
    TEMPLATE_ASSIGNEE = 'Template_Assigned_To'
    INSTALL_ASSIGNEE = 'Install_Assigned_To'

    # Raw Numeric/Financial Columns (pre-calculation)
    RAW_REVENUE = 'Total_Job_Price_'
    RAW_COST_PLANT = 'Phase_Dollars_Plant_Invoice_' # Stone
    RAW_SHOP_COST = 'Branch_INV_' # Laminate
    RAW_MATERIAL_COST = 'Plant_INV_' # Laminate
    RAW_SQFT = 'Total_Job_SqFT'
    RAW_ORIGINAL_GM = 'Job_Throughput_Job_GM_original'
    RAW_REWORK_PRICE = 'Rework_Stone_Shop_Rework_Price'
    RAW_REWORK_COGS = 'Job_Throughput_Rework_COGS'
    RAW_REWORK_LABOR = 'Job_Throughput_Rework_Job_Labor'
    RAW_TOTAL_COGS = 'Job_Throughput_Total_COGS'

    # Calculated Columns (created during processing)
    PRODUCT_TYPE = 'Product_Type'
    DAYS_BEHIND = 'Days_Behind'
    MATERIAL_BRAND = 'Material_Brand'
    MATERIAL_COLOR = 'Material_Color'
    REVENUE = 'Revenue'
    COST_FROM_PLANT = 'Cost_From_Plant'
    SHOP_COST = 'Shop_Cost'
    MATERIAL_COST = 'Material_Cost'
    TOTAL_SQFT = 'Total_Job_SqFt'
    ORIGINAL_GM = 'Original_GM'
    REWORK_PRICE = 'Rework_Price'
    REWORK_COGS = 'Rework_COGS'
    REWORK_LABOR = 'Rework_Labor'
    TOTAL_COGS = 'Total_COGS'
    INSTALL_COST = 'Install_Cost'
    TOTAL_REWORK_COST = 'Total_Rework_Cost'
    TOTAL_BRANCH_COST = 'Total_Branch_Cost'
    BRANCH_PROFIT = 'Branch_Profit'
    BRANCH_PROFIT_MARGIN = 'Branch_Profit_Margin_%'
    PROFIT_VARIANCE = 'Profit_Variance'
    SHOP_PROFIT = 'Shop_Profit'
    SHOP_PROFIT_MARGIN = 'Shop_Profit_Margin_%'
    LINK = 'Link'


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
        COLS.TEMPLATE_DATE, COLS.RTF_DATE, COLS.SHIP_DATE, COLS.INSTALL_DATE,
        COLS.SERVICE_DATE, COLS.DELIVERY_DATE, COLS.JOB_CREATION_DATE, COLS.NEXT_SCHED_DATE,
        COLS.PRODUCT_RCVD_DATE, COLS.PICK_UP_DATE
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def _calculate_days_behind(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Calculates if a job is ahead or behind its next scheduled date."""
    if COLS.NEXT_SCHED_DATE in df.columns:
        df[COLS.DAYS_BEHIND] = (today - df[COLS.NEXT_SCHED_DATE]).dt.days
    else:
        df[COLS.DAYS_BEHIND] = np.nan
    return df

def _calculate_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates durations between key process stages."""
    if COLS.RTF_DATE in df.columns and COLS.TEMPLATE_DATE in df.columns:
        df['Days_Template_to_RTF'] = (df[COLS.RTF_DATE] - df[COLS.TEMPLATE_DATE]).dt.days
        df.loc[df['Days_Template_to_RTF'] < 0, 'Days_Template_to_RTF'] = np.nan
    if COLS.SHIP_DATE in df.columns and COLS.RTF_DATE in df.columns:
        df['Days_RTF_to_Ship'] = (df[COLS.SHIP_DATE] - df[COLS.RTF_DATE]).dt.days
        df.loc[df['Days_RTF_to_Ship'] < 0, 'Days_RTF_to_Ship'] = np.nan
    if COLS.INSTALL_DATE in df.columns and COLS.SHIP_DATE in df.columns:
        df['Days_Ship_to_Install'] = (df[COLS.INSTALL_DATE] - df[COLS.SHIP_DATE]).dt.days
        df.loc[df['Days_Ship_to_Install'] < 0, 'Days_Ship_to_Install'] = np.nan
    return df

def _enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds supplementary data like material parsing."""
    if COLS.JOB_MATERIAL in df.columns:
        df[[COLS.MATERIAL_BRAND, COLS.MATERIAL_COLOR]] = df[COLS.JOB_MATERIAL].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df[COLS.MATERIAL_BRAND] = "N/A"
        df[COLS.MATERIAL_COLOR] = "N/A"
    return df

# --- Division-Specific Processing Pipelines ---

def process_stone_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the data specifically for Stone/Quartz jobs."""
    df_stone = df.copy()
    
    numeric_map = {
        COLS.RAW_REVENUE: COLS.REVENUE,
        COLS.RAW_COST_PLANT: COLS.COST_FROM_PLANT,
        COLS.RAW_SQFT: COLS.TOTAL_SQFT,
        COLS.RAW_ORIGINAL_GM: COLS.ORIGINAL_GM,
        COLS.RAW_REWORK_PRICE: COLS.REWORK_PRICE,
        COLS.RAW_REWORK_COGS: COLS.REWORK_COGS,
        COLS.RAW_REWORK_LABOR: COLS.REWORK_LABOR,
        COLS.RAW_TOTAL_COGS: COLS.TOTAL_COGS
    }
    
    # --- REFACTOR: Granular Error Handling ---
    for original, new in numeric_map.items():
        if original in df_stone.columns:
            df_stone[new] = pd.to_numeric(df_stone[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else:
            # If a critical column is missing, create it with zeros and warn the user.
            st.warning(f"Missing critical data column '{original}' for Stone calculations. Results may be inaccurate.")
            df_stone[new] = 0.0
            
    df_stone[COLS.INSTALL_COST] = df_stone.get(COLS.TOTAL_SQFT, 0) * INSTALL_COST_PER_SQFT
    df_stone[COLS.TOTAL_REWORK_COST] = df_stone.get(COLS.REWORK_PRICE, 0) + df_stone.get(COLS.REWORK_COGS, 0) + df_stone.get(COLS.REWORK_LABOR, 0)
    df_stone[COLS.TOTAL_BRANCH_COST] = df_stone.get(COLS.COST_FROM_PLANT, 0) + df_stone[COLS.INSTALL_COST] + df_stone[COLS.TOTAL_REWORK_COST]
    df_stone[COLS.BRANCH_PROFIT] = df_stone.get(COLS.REVENUE, 0) - df_stone[COLS.TOTAL_BRANCH_COST]
    df_stone[COLS.BRANCH_PROFIT_MARGIN] = df_stone.apply(
        lambda row: (row[COLS.BRANCH_PROFIT] / row[COLS.REVENUE] * 100) if row.get(COLS.REVENUE) != 0 else 0, axis=1
    )
    df_stone[COLS.PROFIT_VARIANCE] = df_stone[COLS.BRANCH_PROFIT] - df_stone.get(COLS.ORIGINAL_GM, 0)
    
    # Shop Profitability
    df_stone[COLS.SHOP_PROFIT] = df_stone.get(COLS.COST_FROM_PLANT, 0) - df_stone.get(COLS.TOTAL_COGS, 0)
    df_stone[COLS.SHOP_PROFIT_MARGIN] = df_stone.apply(
        lambda row: (row[COLS.SHOP_PROFIT] / row[COLS.COST_FROM_PLANT] * 100) if row.get(COLS.COST_FROM_PLANT) != 0 else 0, axis=1
    )
    
    return df_stone

def process_laminate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the data specifically for Laminate jobs."""
    df_lam = df.copy()

    numeric_map = {
        COLS.RAW_REVENUE: COLS.REVENUE,
        COLS.RAW_SHOP_COST: COLS.SHOP_COST,
        COLS.RAW_MATERIAL_COST: COLS.MATERIAL_COST,
        COLS.RAW_SQFT: COLS.TOTAL_SQFT,
        COLS.RAW_ORIGINAL_GM: COLS.ORIGINAL_GM,
        COLS.RAW_REWORK_PRICE: COLS.REWORK_PRICE,
    }

    # --- REFACTOR: Granular Error Handling ---
    for original, new in numeric_map.items():
        if original in df_lam.columns:
            df_lam[new] = pd.to_numeric(df_lam[original].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)
        else:
            st.warning(f"Missing critical data column '{original}' for Laminate calculations. Results may be inaccurate.")
            df_lam[new] = 0.0

    df_lam[COLS.INSTALL_COST] = df_lam.get(COLS.TOTAL_SQFT, 0) * INSTALL_COST_PER_SQFT
    df_lam[COLS.TOTAL_REWORK_COST] = df_lam.get(COLS.REWORK_PRICE, 0)
    df_lam[COLS.TOTAL_BRANCH_COST] = df_lam.get(COLS.SHOP_COST, 0) + df_lam.get(COLS.MATERIAL_COST, 0) + df_lam[COLS.INSTALL_COST] + df_lam[COLS.TOTAL_REWORK_COST]
    df_lam[COLS.BRANCH_PROFIT] = df_lam.get(COLS.REVENUE, 0) - df_lam[COLS.TOTAL_BRANCH_COST]
    df_lam[COLS.BRANCH_PROFIT_MARGIN] = df_lam.apply(
        lambda row: (row[COLS.BRANCH_PROFIT] / row[COLS.REVENUE] * 100) if row.get(COLS.REVENUE) != 0 else 0, axis=1
    )
    df_lam[COLS.PROFIT_VARIANCE] = df_lam[COLS.BRANCH_PROFIT] - df_lam.get(COLS.ORIGINAL_GM, 0)
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

    if COLS.DIVISION not in df.columns:
        st.error(f"Your Google Sheet must have a '{COLS.DIVISION}' column to separate Stone and Laminate jobs.")
        st.stop()
    
    df[COLS.PRODUCT_TYPE] = df[COLS.DIVISION].apply(lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz')

    df_stone = df[df[COLS.PRODUCT_TYPE] == 'Stone/Quartz'].copy()
    df_laminate = df[df[COLS.PRODUCT_TYPE] == 'Laminate'].copy()

    df_stone_processed = process_stone_data(df_stone)
    df_laminate_processed = process_laminate_data(df_laminate)

    return df_stone_processed, df_laminate_processed


# --- UI Rendering Functions ---

def render_overview_tab(df: pd.DataFrame, division_name: str):
    st.header(f"📈 {division_name} Overview")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return

    total_revenue = df[COLS.REVENUE].sum()
    total_profit = df[COLS.BRANCH_PROFIT].sum()
    avg_margin = (total_profit / total_revenue * 100) if total_revenue != 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Branch Profit", f"${total_profit:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

    st.markdown("---")
    st.subheader("Profit by Salesperson")
    if COLS.SALESPERSON in df.columns and not df.empty:
        sales_profit = df.groupby(COLS.SALESPERSON)[COLS.BRANCH_PROFIT].sum().sort_values(ascending=False)
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
        df_display = df_display[df_display[COLS.JOB_NAME].str.contains(job_name_filter, case=False, na=False)]
    if prod_num_filter:
        df_display = df_display[df_display[COLS.PRODUCTION_NUM].str.contains(prod_num_filter, case=False, na=False)]

    if df_display.empty:
        st.warning("No data matches the current filters.")
        return

    if COLS.PRODUCTION_NUM in df_display.columns:
        df_display[COLS.LINK] = df_display[COLS.PRODUCTION_NUM].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)

    column_order = [
        COLS.LINK, COLS.JOB_NAME, COLS.NEXT_SCHED_ACTIVITY, COLS.DAYS_BEHIND, COLS.REVENUE, 
        COLS.TOTAL_SQFT, COLS.TOTAL_BRANCH_COST, COLS.BRANCH_PROFIT, COLS.BRANCH_PROFIT_MARGIN
    ]
    
    if division_name == 'Laminate':
        column_order.insert(6, COLS.SHOP_COST)
        column_order.insert(6, COLS.MATERIAL_COST)
    else: # Stone
        column_order.insert(6, COLS.COST_FROM_PLANT)
        column_order.append(COLS.SHOP_PROFIT_MARGIN)

    final_column_order = [c for c in column_order if c in df_display.columns]
    st.dataframe(df_display[final_column_order], use_container_width=True,
        column_config={
            COLS.LINK: st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
            COLS.DAYS_BEHIND: st.column_config.NumberColumn("Days Behind/Ahead", help="Positive: Behind. Negative: Ahead."),
            COLS.REVENUE: st.column_config.NumberColumn(format='$%.2f'),
            COLS.TOTAL_SQFT: st.column_config.NumberColumn("SqFt", format='%.2f'),
            COLS.COST_FROM_PLANT: st.column_config.NumberColumn("Production Cost", format='$%.2f'),
            COLS.MATERIAL_COST: st.column_config.NumberColumn("Material Cost", format='$%.2f'),
            COLS.SHOP_COST: st.column_config.NumberColumn("Shop Cost", format='$%.2f'),
            COLS.TOTAL_BRANCH_COST: st.column_config.NumberColumn(format='$%.2f'),
            COLS.BRANCH_PROFIT: st.column_config.NumberColumn(format='$%.2f'),
            COLS.BRANCH_PROFIT_MARGIN: st.column_config.ProgressColumn("Branch Profit %", format='%.2f%%', min_value=-50, max_value=100),
            COLS.SHOP_PROFIT_MARGIN: st.column_config.ProgressColumn("Shop Profit %", format='%.2f%%', min_value=-50, max_value=100),
        }
    )

def render_profit_drivers_tab(df: pd.DataFrame, division_name: str):
    st.header(f"💸 {division_name} Profitability Drivers")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return

    driver_options = [COLS.JOB_TYPE, COLS.ORDER_TYPE, COLS.LEAD_SOURCE, COLS.SALESPERSON, COLS.MATERIAL_BRAND]
    valid_drivers = [d for d in driver_options if d in df.columns]

    if not valid_drivers:
        st.warning("No valid columns found for this analysis.")
        return

    selected_driver = st.selectbox("Analyze Profitability by:", valid_drivers, key=f"driver_{division_name}")
    if selected_driver:
        agg_dict = {'Avg_Branch_Profit_Margin': (COLS.BRANCH_PROFIT_MARGIN, 'mean'), 'Total_Profit': (COLS.BRANCH_PROFIT, 'sum'), 'Job_Count': (COLS.JOB_NAME, 'count')}
        driver_analysis = df.groupby(selected_driver).agg(**agg_dict).sort_values('Total_Profit', ascending=False)
        st.dataframe(driver_analysis.style.format({'Avg_Branch_Profit_Margin': '{:.2f}%', 'Total_Profit': '${:,.2f}'}), use_container_width=True)

def render_rework_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔬 {division_name} Rework & Variance")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rework Analysis")
        if COLS.TOTAL_REWORK_COST in df and COLS.REWORK_REASON in df.columns:
            rework_jobs = df[df[COLS.TOTAL_REWORK_COST] > 0].copy()
            if not rework_jobs.empty:
                st.metric("Total Rework Cost", f"${rework_jobs[COLS.TOTAL_REWORK_COST].sum():,.2f}", f"{len(rework_jobs)} jobs affected")
                agg_rework = rework_jobs.groupby(COLS.REWORK_REASON)[COLS.TOTAL_REWORK_COST].agg(['sum', 'count'])
                agg_rework.columns = ['Total Rework Cost', 'Number of Jobs']
                st.dataframe(agg_rework.sort_values('Total Rework Cost', ascending=False).style.format({'Total Rework Cost': '${:,.2f}'}))
            else:
                st.info("No rework costs recorded.")
        else:
            st.info("Rework data not available (missing Rework Cost or Rework Reason columns).")
    with c2:
        st.subheader("Profit Variance Analysis")
        if COLS.PROFIT_VARIANCE in df.columns and COLS.ORIGINAL_GM in df.columns:
            variance_jobs = df[df[COLS.PROFIT_VARIANCE].abs() > 0.01].copy()
            if not variance_jobs.empty:
                st.metric("Jobs with Profit Variance", f"{len(variance_jobs)}")
                display_cols = [COLS.JOB_NAME, COLS.ORIGINAL_GM, COLS.BRANCH_PROFIT, COLS.PROFIT_VARIANCE]
                st.dataframe(variance_jobs[display_cols].sort_values(by=COLS.PROFIT_VARIANCE, key=abs, ascending=False).head(20),
                    column_config={COLS.ORIGINAL_GM: st.column_config.NumberColumn("Est. Profit", format='$%.2f'), COLS.BRANCH_PROFIT: st.column_config.NumberColumn("Actual Profit", format='$%.2f'), COLS.PROFIT_VARIANCE: st.column_config.NumberColumn("Variance", format='$%.2f')})
            else:
                st.info("No significant profit variance found.")
        else:
            st.info("Profit variance data not available.")

def render_pipeline_issues_tab(df: pd.DataFrame, division_name: str, today: pd.Timestamp):
    st.header(f"🚧 {division_name} Pipeline & Issues")
    
    st.subheader("Jobs Awaiting Ready-to-Fab")
    if COLS.RTF_STATUS in df.columns and COLS.TEMPLATE_DATE in df.columns:
        conditions = (df[COLS.TEMPLATE_DATE].notna() & (df[COLS.TEMPLATE_DATE] <= today) & (df[COLS.RTF_STATUS].fillna('').str.lower() != 'complete'))
        stuck_jobs = df[conditions].copy()
        if not stuck_jobs.empty:
            stuck_jobs['Days_Since_Template'] = (today - stuck_jobs[COLS.TEMPLATE_DATE]).dt.days
            display_cols = [COLS.JOB_NAME, COLS.SALESPERSON, COLS.TEMPLATE_DATE, 'Days_Since_Template']
            st.dataframe(stuck_jobs[display_cols].sort_values(by='Days_Since_Template', ascending=False), use_container_width=True)
        else:
            st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    else:
        st.warning("Could not check for jobs awaiting RTF. Required columns missing.")

    st.markdown("---")
    st.subheader("Jobs with Scheduling Conflicts")
    activity_cols = [COLS.INSTALL_DATE, COLS.SERVICE_DATE, COLS.PICK_UP_DATE, COLS.DELIVERY_DATE]
    product_received_col = COLS.PRODUCT_RCVD_DATE
    if all(col in df.columns for col in activity_cols + [product_received_col]):
        conflict_conditions = ((df[COLS.INSTALL_DATE].notna() & (df[COLS.INSTALL_DATE] < df[product_received_col])) | (df[COLS.SERVICE_DATE].notna() & (df[COLS.SERVICE_DATE] < df[product_received_col])) | (df[COLS.PICK_UP_DATE].notna() & (df[COLS.PICK_UP_DATE] < df[product_received_col])) | (df[COLS.DELIVERY_DATE].notna() & (df[COLS.DELIVERY_DATE] < df[product_received_col])))
        conflict_jobs = df[conflict_conditions].copy()
        if not conflict_jobs.empty:
            display_cols = [COLS.JOB_NAME, COLS.PRODUCT_RCVD_DATE, COLS.INSTALL_DATE, COLS.SERVICE_DATE]
            st.dataframe(conflict_jobs[[c for c in display_cols if c in conflict_jobs.columns]], use_container_width=True)
        else:
            st.success("✅ No scheduling conflicts found.")
    else:
        st.warning("Could not perform scheduling conflict analysis. Required columns are missing.")

def render_workload_card(df_filtered: pd.DataFrame, activity_name: str, date_col: str, assignee_col: str):
    """Helper function to display a workload card for a given activity."""
    st.subheader(activity_name)
    
    if date_col not in df_filtered.columns or assignee_col not in df_filtered.columns:
        st.warning(f"Required columns ('{date_col}', '{assignee_col}') for {activity_name} analysis not found.")
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
            total_sqft = assignee_df[COLS.TOTAL_SQFT].sum()

            col1, col2 = st.columns(2)
            col1.metric(f"{assignee} - Total Jobs", f"{total_jobs}")
            col2.metric(f"{assignee} - Total SqFt", f"{total_sqft:,.2f}")

            with st.expander("View Weekly Breakdown"):
                weekly_summary = assignee_df.set_index(date_col).resample('W-Mon', label='left', closed='left').agg(
                    Jobs=(COLS.PRODUCTION_NUM, 'count'),
                    Total_SqFt=(COLS.TOTAL_SQFT, 'sum')
                ).reset_index()
                weekly_summary = weekly_summary[weekly_summary['Jobs'] > 0]
                
                if not weekly_summary.empty:
                    st.dataframe(weekly_summary.rename(columns={date_col: 'Week_Start_Date'}), use_container_width=True)
                else:
                    st.write("No scheduled work for this person in the selected period.")

def render_field_workload_tab(df: pd.DataFrame, division_name: str):
    """Renders the enhanced tab for Template, Install, and Service workloads."""
    st.header(f"� {division_name} Field Workload Planner")
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        render_workload_card(df, "Templates", COLS.TEMPLATE_DATE, COLS.TEMPLATE_ASSIGNEE)
    with col2:
        render_workload_card(df, "Installs", COLS.INSTALL_DATE, COLS.INSTALL_ASSIGNEE)

def render_forecasting_tab(df: pd.DataFrame, division_name: str):
    st.header(f"🔮 {division_name} Forecasting & Trends")
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.error("Forecasting features require scikit-learn and matplotlib.")
        return
    if COLS.JOB_CREATION_DATE not in df.columns or df[COLS.JOB_CREATION_DATE].isnull().all():
        st.warning(f"'{COLS.JOB_CREATION_DATE}' column is required for trend analysis.")
        return
    if df.empty or len(df) < 2:
        st.warning(f"Not enough {division_name} data to create a forecast.")
        return

    df_trends = df.copy().set_index(COLS.JOB_CREATION_DATE).sort_index()
    st.subheader("Monthly Performance Trends")
    monthly_summary = df_trends.resample('M').agg({COLS.REVENUE: 'sum', COLS.BRANCH_PROFIT: 'sum', COLS.JOB_NAME: 'count'}).rename(columns={COLS.JOB_NAME: 'Job_Count'})
    if monthly_summary.empty:
        st.info("No data in the selected range to display monthly trends.")
        return
    st.line_chart(monthly_summary[[COLS.REVENUE, COLS.BRANCH_PROFIT]])
    st.bar_chart(monthly_summary['Job_Count'])

# --- NEW: Company-Wide Rendering Functions ---

def render_company_workload_tab(df_combined: pd.DataFrame):
    st.header("👷 Company-Wide Field Workload")
    if df_combined.empty:
        st.warning("No data available to display workload.")
        return

    divisions = sorted(df_combined[COLS.PRODUCT_TYPE].unique())
    selected_divisions = st.multiselect("Filter by Division:", options=divisions, default=list(divisions))
    
    if not selected_divisions:
        st.warning("Please select at least one division.")
        return
        
    df_filtered = df_combined[df_combined[COLS.PRODUCT_TYPE].isin(selected_divisions)]

    col1, col2 = st.columns(2)
    with col1:
        render_workload_card(df_filtered, "Templates", COLS.TEMPLATE_DATE, COLS.TEMPLATE_ASSIGNEE)
    with col2:
        render_workload_card(df_filtered, "Installs", COLS.INSTALL_DATE, COLS.INSTALL_ASSIGNEE)

def render_company_forecasting_tab(df_combined: pd.DataFrame):
    st.header("🔮 Company-Wide Forecasting & Trends")
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.error("Forecasting features require scikit-learn and matplotlib.")
        return
    if COLS.JOB_CREATION_DATE not in df_combined.columns or df_combined[COLS.JOB_CREATION_DATE].isnull().all():
        st.warning(f"'{COLS.JOB_CREATION_DATE}' column is required for trend analysis.")
        return
    if df_combined.empty or len(df_combined) < 2:
        st.warning("Not enough data to create a forecast.")
        return

    df_trends = df_combined.copy().set_index(COLS.JOB_CREATION_DATE)
    
    monthly_combined = df_trends.resample('M').agg({COLS.REVENUE: 'sum'}).rename(columns={COLS.REVENUE: 'Total Revenue'})
    monthly_by_division = df_trends.groupby(COLS.PRODUCT_TYPE).resample('M').agg({COLS.REVENUE: 'sum'}).unstack(level=0)
    monthly_by_division.columns = monthly_by_division.columns.droplevel(0)
    
    plot_df = pd.concat([monthly_combined, monthly_by_division], axis=1).fillna(0)

    st.subheader("Monthly Revenue Trends by Division")
    st.line_chart(plot_df)

    st.subheader("Total Company Revenue Forecast")
    forecast_df = monthly_combined.reset_index()
    forecast_df['Time'] = np.arange(len(forecast_df.index))
    
    if len(forecast_df) < 2:
        st.info("Not enough historical data to generate a forecast.")
        return

    model = LinearRegression()
    X = forecast_df[['Time']]
    y = forecast_df['Total Revenue']
    model.fit(X, y)

    future_periods = st.slider("Months to Forecast:", 1, 12, 3, key="company_forecast_slider")
    last_time = forecast_df['Time'].max()
    last_date = forecast_df[COLS.JOB_CREATION_DATE].max()

    future_dates = pd.to_datetime([last_date + pd.DateOffset(months=i) for i in range(1, future_periods + 1)])
    future_df = pd.DataFrame({COLS.JOB_CREATION_DATE: future_dates, 'Time': np.arange(last_time + 1, last_time + 1 + future_periods)})
    future_df['Forecast'] = model.predict(future_df[['Time']])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df[COLS.JOB_CREATION_DATE], forecast_df['Total Revenue'], label='Actual Revenue', marker='o')
    ax.plot(future_df[COLS.JOB_CREATION_DATE], future_df['Forecast'], label='Forecasted Revenue', linestyle='--', marker='o')
    ax.set_title('Total Company Monthly Revenue and Forecast')
    ax.set_ylabel('Revenue ($)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)

# --- REFACTOR: UI logic moved to a dedicated function to avoid repetition ---
def build_division_dashboard(df: pd.DataFrame, division_name: str, today_dt: pd.Timestamp):
    """Creates the set of tabs for a specific division dashboard."""
    tabs = ["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"]
    overview_tab, data_tab, drivers_tab, rework_tab, pipeline_tab, workload_tab, forecast_tab = st.tabs(tabs)
    
    with overview_tab:
        render_overview_tab(df, division_name)
    with data_tab:
        render_detailed_data_tab(df, division_name)
    with drivers_tab:
        render_profit_drivers_tab(df, division_name)
    with rework_tab:
        render_rework_tab(df, division_name)
    with pipeline_tab:
        render_pipeline_issues_tab(df, division_name, today_dt)
    with workload_tab:
        render_field_workload_tab(df, division_name)
    with forecast_tab:
        render_forecasting_tab(df, division_name)

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
        df_combined = pd.concat([df_stone, df_laminate], ignore_index=True)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.exception(e)
        st.stop()

    # --- Main Dashboard Tabs ---
    company_tab, stone_tab, laminate_tab = st.tabs(["🏢 Company-Wide", "💎 Stone/Quartz Dashboard", "🪵 Laminate Dashboard"])

    with company_tab:
        workload_sub_tab, forecast_sub_tab = st.tabs(["👷 Field Workload", "🔮 Forecasting"])
        with workload_sub_tab:
            render_company_workload_tab(df_combined)
        with forecast_sub_tab:
            render_company_forecasting_tab(df_combined)

    with stone_tab:
        build_division_dashboard(df_stone, "Stone/Quartz", today_dt)

    with laminate_tab:
        build_division_dashboard(df_laminate, "Laminate", today_dt)

if __name__ == "__main__":
    main()
