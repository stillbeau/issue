# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This refactored version consolidates data processing logic, centralizes
configuration, and improves performance and maintainability. It retains the
three-dashboard structure: Stone/Quartz, Laminate, and a Company-Wide view.
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

# --- Division-Specific Processing Configuration ---
# Centralizing configuration makes the code cleaner and easier to update.
# The keys here are the names AFTER cleaning by _clean_column_names.
STONE_CONFIG = {
    "name": "Stone/Quartz",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue',
        'Phase_Dollars_Plant_Invoice_': 'Cost_From_Plant',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
        'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor',
        'Job_Throughput_Total_COGS': 'Total_COGS'
    },
    "cost_components": ['Cost_From_Plant', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price', 'Rework_COGS', 'Rework_Labor'],
    "has_shop_profit": True
}

LAMINATE_CONFIG = {
    "name": "Laminate",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue',
        'Branch_INV_': 'Shop_Cost',
        'Plant_INV_': 'Material_Cost',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
    },
    "cost_components": ['Shop_Cost', 'Material_Cost', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price'],
    "has_shop_profit": False
}


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
    """Calculates durations between key process stages, handling invalid dates."""
    duration_pairs = {
        'Days_Template_to_RTF': ('Ready_to_Fab_Date', 'Template_Date'),
        'Days_RTF_to_Ship': ('Ship_Date', 'Ready_to_Fab_Date'),
        'Days_Ship_to_Install': ('Install_Date', 'Ship_Date'),
        'Days_Template_to_Install': ('Install_Date', 'Template_Date'),
        'Days_RTF_to_Product_Rcvd': ('Product_Rcvd_Date', 'Ready_to_Fab_Date'),
        'Days_Product_Rcvd_to_Install': ('Install_Date', 'Product_Rcvd_Date'),
        'Days_Template_to_Ship': ('Ship_Date', 'Template_Date')
    }
    for new_col, (end_col, start_col) in duration_pairs.items():
        if start_col in df.columns and end_col in df.columns:
            duration = (df[end_col] - df[start_col]).dt.days
            df[new_col] = np.where(duration >= 0, duration, np.nan)
    return df

def _enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds supplementary data like material parsing."""
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material_Brand'] = "N/A"
        df['Material_Color'] = "N/A"
    return df

# --- Unified Division-Specific Processing Pipeline ---

def _process_division_data(df: pd.DataFrame, config: dict, install_cost_per_sqft: float) -> pd.DataFrame:
    """Processes data for a specific division using a configuration dictionary."""
    df_processed = df.copy()
    
    for original, new in config["numeric_map"].items():
        if original in df_processed.columns:
            df_processed[new] = pd.to_numeric(
                df_processed[original].astype(str).str.replace(r'[$,%]', '', regex=True),
                errors='coerce'
            ).fillna(0)
        else:
            df_processed[new] = 0.0
            
    df_processed['Install_Cost'] = df_processed.get('Total_Job_SqFt', 0) * install_cost_per_sqft
    
    rework_costs = [df_processed.get(c, 0) for c in config["rework_components"]]
    df_processed['Total_Rework_Cost'] = sum(rework_costs)
    
    branch_costs = [df_processed.get(c, 0) for c in config["cost_components"]]
    df_processed['Total_Branch_Cost'] = sum(branch_costs)

    revenue = df_processed.get('Revenue', 0)
    df_processed['Branch_Profit'] = revenue - df_processed['Total_Branch_Cost']
    
    df_processed['Branch_Profit_Margin_%'] = np.where(
        revenue != 0, (df_processed['Branch_Profit'] / revenue * 100), 0
    )
    
    df_processed['Profit_Variance'] = df_processed['Branch_Profit'] - df_processed.get('Original_GM', 0)

    if config["has_shop_profit"]:
        cost_from_plant = df_processed.get('Cost_From_Plant', 0)
        total_cogs = df_processed.get('Total_COGS', 0)
        df_processed['Shop_Profit'] = cost_from_plant - total_cogs
        df_processed['Shop_Profit_Margin_%'] = np.where(
            cost_from_plant != 0, (df_processed['Shop_Profit'] / cost_from_plant * 100), 0
        )
        
    return df_processed

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict, today: pd.Timestamp, install_cost: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from Google Sheets, performs universal processing, and then
    delegates to the division-specific processor.
    """
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df_raw = pd.DataFrame(worksheet.get_all_records())
    df = df_raw.copy()

    # --- Universal Processing Steps ---
    df = _clean_column_names(df)
    df = _parse_date_columns(df)
    df = _calculate_days_behind(df, today)
    df = _calculate_durations(df)
    df = _enrich_data(df)

    if 'Division' not in df.columns:
        st.error("Your Google Sheet must have a 'Division' column to separate Stone and Laminate jobs.")
        st.stop()
    
    df['Product_Type'] = df['Division'].apply(lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz')

    # --- Split and Process by Division ---
    df_stone = df[df['Product_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Product_Type'] == 'Laminate'].copy()

    df_stone_processed = _process_division_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = _process_division_data(df_laminate, LAMINATE_CONFIG, install_cost)

    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)

    return df_stone_processed, df_laminate_processed, df_combined


# --- UI Rendering Functions ---

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

    if 'Production_' in df_display.columns:
        df_display['Link'] = df_display['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)

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
    valid_drivers = [d for d in driver_options if d in df.columns]

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
        conditions = (
            df['Template_Date'].notna() &
            (df['Template_Date'] <= today) &
            (df['Ready_to_Fab_Status'].fillna('').str.lower() != 'complete')
        )
        stuck_jobs = df[conditions].copy()
        if not stuck_jobs.empty:
            stuck_jobs['Days_Since_Template'] = (today - stuck_jobs['Template_Date']).dt.days
            
            if 'Production_' in stuck_jobs.columns:
                stuck_jobs['Link'] = stuck_jobs['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
            
            display_cols = ['Link', 'Job_Name', 'Salesperson', 'Template_Date', 'Days_Since_Template']
            final_display_cols = [c for c in display_cols if c in stuck_jobs.columns]

            st.dataframe(
                stuck_jobs[final_display_cols].sort_values(by='Days_Since_Template', ascending=False),
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}
            )
        else:
            st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    else:
        st.warning("Could not check for jobs awaiting RTF. Required columns missing: " + ", ".join([c for c in required_cols_rtf if c not in df.columns]))

    st.markdown("---")
    st.subheader("Jobs with Scheduling Conflicts")
    required_cols_conflict = ['Install_Date', 'Service_Date', 'Pick_Up_Date', 'Delivery_Date', 'Product_Rcvd_Date']
    if all(col in df.columns for col in required_cols_conflict):
        conflict_conditions = (
            (df['Install_Date'].notna() & (df['Install_Date'] < df['Product_Rcvd_Date'])) |
            (df['Service_Date'].notna() & (df['Service_Date'] < df['Product_Rcvd_Date'])) |
            (df['Pick_Up_Date'].notna() & (df['Pick_Up_Date'] < df['Product_Rcvd_Date'])) |
            (df['Delivery_Date'].notna() & (df['Delivery_Date'] < df['Product_Rcvd_Date']))
        )
        conflict_jobs = df[conflict_conditions].copy()
        if not conflict_jobs.empty:
            if 'Production_' in conflict_jobs.columns:
                conflict_jobs['Link'] = conflict_jobs['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)

            display_cols = ['Link', 'Job_Name', 'Product_Rcvd_Date', 'Install_Date', 'Service_Date']
            final_display_cols = [c for c in display_cols if c in conflict_jobs.columns]
            
            st.dataframe(
                conflict_jobs[final_display_cols],
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")}
            )
        else:
            st.success("✅ No scheduling conflicts found.")
    else:
        st.warning("Could not perform scheduling conflict analysis. Required columns are missing: " + ", ".join([c for c in required_cols_conflict if c not in df.columns]))


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
    st.header(f"� {division_name} Field Workload Planner")
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
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.error("Forecasting features require scikit-learn and matplotlib. Please install them.")
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

# --- Company-Wide Performance Dashboard ---
def render_company_performance_tab(df_combined: pd.DataFrame, today: pd.Timestamp):
    """Displays key timeline, efficiency, and workload metrics for the whole company."""
    st.header("🏢 Company-Wide Performance")
    if df_combined.empty:
        st.warning("No data available to display company-wide metrics.")
        return

    df_stone = df_combined[df_combined['Product_Type'] == 'Stone/Quartz']
    df_laminate = df_combined[df_combined['Product_Type'] == 'Laminate']

    # --- 1. Timeline Metrics ---
    st.subheader("⏱️ Timeline Metrics")
    st.markdown("Average number of days between key process stages.")
    
    timeline_metrics = {
        "Template to Install": "Days_Template_to_Install",
        "Ready to Fab to Product Received": "Days_RTF_to_Product_Rcvd",
        "Template to Ready to Fab": "Days_Template_to_RTF",
        "Product Received to Install": "Days_Product_Rcvd_to_Install",
        "Template to Ship": "Days_Template_to_Ship",
        "Ship to Install": "Days_Ship_to_Install",
    }

    for title, col_name in timeline_metrics.items():
        if col_name in df_combined.columns:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{title}**")
            avg_stone = df_stone[col_name].mean()
            c2.metric("💎 Stone/Quartz", f"{avg_stone:.1f} days" if pd.notna(avg_stone) else "N/A")
            avg_laminate = df_laminate[col_name].mean()
            c3.metric("🪵 Laminate", f"{avg_laminate:.1f} days" if pd.notna(avg_laminate) else "N/A")

    # --- 2. Efficiency Metrics ---
    st.markdown("---")
    st.subheader("⚡ Efficiency Metrics")
    
    if 'Revenue' in df_combined.columns and 'Days_Template_to_Install' in df_combined.columns:
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Revenue per Day (Template to Install)**")
        
        df_stone_eff = df_stone.dropna(subset=['Revenue', 'Days_Template_to_Install'])
        if not df_stone_eff.empty and df_stone_eff['Days_Template_to_Install'].sum() > 0:
            rev_per_day_stone = df_stone_eff['Revenue'].sum() / df_stone_eff['Days_Template_to_Install'].sum()
            c2.metric("💎 Stone/Quartz", f"${rev_per_day_stone:,.2f}")
        else:
            c2.metric("💎 Stone/Quartz", "N/A")

        df_laminate_eff = df_laminate.dropna(subset=['Revenue', 'Days_Template_to_Install'])
        if not df_laminate_eff.empty and df_laminate_eff['Days_Template_to_Install'].sum() > 0:
            rev_per_day_laminate = df_laminate_eff['Revenue'].sum() / df_laminate_eff['Days_Template_to_Install'].sum()
            c3.metric("🪵 Laminate", f"${rev_per_day_laminate:,.2f}")
        else:
            c3.metric("🪵 Laminate", "N/A")
    else:
        st.warning("Revenue or timeline data missing for Efficiency Metrics.")


    # --- 3. Work-in-Progress (WIP) & Throughput ---
    st.markdown("---")
    st.subheader("🏭 Work-in-Progress (WIP) & Throughput")

    c1, c2 = st.columns([2,2,3]) # Adjust column widths
    
    with c1:
        st.markdown("**Jobs Currently in Fabrication**")
        if 'Ready_to_Fab_Date' in df_stone.columns and 'Ship_Date' in df_stone.columns:
            fab_cond_stone = (df_stone['Ready_to_Fab_Date'].notna()) & (df_stone['Ship_Date'].isna())
            st.metric("💎 Stone/Quartz", f"{fab_cond_stone.sum()} Jobs")
        else:
            st.metric("💎 Stone/Quartz", "N/A")
            
        if 'Ready_to_Fab_Date' in df_laminate.columns and 'Ship_Date' in df_laminate.columns:
            fab_cond_laminate = (df_laminate['Ready_to_Fab_Date'].notna()) & (df_laminate['Ship_Date'].isna())
            st.metric("🪵 Laminate", f"{fab_cond_laminate.sum()} Jobs")
        else:
            st.metric("🪵 Laminate", "N/A")

    with c2:
        st.markdown("**Weekly Install Throughput**")
        if 'Install_Date' in df_stone.columns:
            installs_stone = df_stone.dropna(subset=['Install_Date']).set_index('Install_Date')
            if not installs_stone.empty:
                weekly_installs_stone = installs_stone.resample('W').size().mean()
                st.metric("💎 Stone/Quartz", f"{weekly_installs_stone:.1f} Jobs/Week")
            else:
                st.metric("💎 Stone/Quartz", "N/A")
        else:
            st.metric("💎 Stone/Quartz", "N/A")

        if 'Install_Date' in df_laminate.columns:
            installs_laminate = df_laminate.dropna(subset=['Install_Date']).set_index('Install_Date')
            if not installs_laminate.empty:
                weekly_installs_laminate = installs_laminate.resample('W').size().mean()
                st.metric("🪵 Laminate", f"{weekly_installs_laminate:.1f} Jobs/Week")
            else:
                st.metric("🪵 Laminate", "N/A")
        else:
            st.metric("🪵 Laminate", "N/A")


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

    today_date = st.sidebar.date_input(
        "Select 'Today's' Date for Calculations",
        value=datetime.now().date(),
        help="This date is used for 'Days Behind' calculations."
    )
    today_dt = pd.to_datetime(today_date)
    
    install_cost_sqft = st.sidebar.number_input(
        "Install Cost per SqFt ($)",
        min_value=0.0,
        value=15.0,
        step=0.50,
        help="Adjust the assumed cost of installation per square foot."
    )

    try:
        df_stone, df_laminate, df_combined = load_and_process_data(creds, today_dt, install_cost_sqft)
    except Exception as e:
        st.error(f"Failed to load or process data. Please ensure your Google Sheet format matches the script's expectations. Error: {e}")
        st.exception(e)
        st.stop()

    main_tabs = st.tabs(["🏢 Company-Wide Performance", "💎 Stone/Quartz Dashboard", "🪵 Laminate Dashboard"])

    with main_tabs[0]:
        render_company_performance_tab(df_combined, today_dt)

    with main_tabs[1]:
        stone_tabs = ["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"]
        stone_sub_tabs = st.tabs(stone_tabs)
        with stone_sub_tabs[0]: render_overview_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[1]: render_detailed_data_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[2]: render_profit_drivers_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[3]: render_rework_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[4]: render_pipeline_issues_tab(df_stone, "Stone/Quartz", today_dt)
        with stone_sub_tabs[5]: render_field_workload_tab(df_stone, "Stone/Quartz")
        with stone_sub_tabs[6]: render_forecasting_tab(df_stone, "Stone/Quartz")

    with main_tabs[2]:
        laminate_tabs = ["📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance", "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting"]
        laminate_sub_tabs = st.tabs(laminate_tabs)
        with laminate_sub_tabs[0]: render_overview_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[1]: render_detailed_data_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[2]: render_profit_drivers_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[3]: render_rework_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[4]: render_pipeline_issues_tab(df_laminate, "Laminate", today_dt)
        with laminate_sub_tabs[5]: render_field_workload_tab(df_laminate, "Laminate")
        with laminate_sub_tabs[6]: render_forecasting_tab(df_laminate, "Laminate")

if __name__ == "__main__":
    main()
