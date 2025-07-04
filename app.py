# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This refactored version includes interactive sidebar filters, improved code
modularity, and enhanced UI/UX for a more powerful and user-friendly analysis
experience.
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

def get_lat_lon(city_name: str) -> tuple[float, float]:
    """Placeholder function to geocode city names."""
    city_coords = {
        "vernon": (50.267, -119.272), "kelowna": (49.888, -119.496),
        "penticton": (49.492, -119.593), "kamloops": (50.676, -120.341),
        "calgary": (51.044, -114.071), "edmonton": (53.546, -113.493),
        "red deer": (52.269, -113.811)
    }
    return city_coords.get(str(city_name).lower(), (0, 0))

def calculate_production_cost(value: str) -> float:
    """
    Calculates the total production cost from a string that may contain
    multiple numeric values separated by newlines.
    """
    total_cost = 0.0
    if isinstance(value, str):
        numbers = re.findall(r'-?\$?[\d,]*\.?\d+', value)
        for num_str in numbers:
            try:
                cleaned_num = num_str.replace('$', '').replace(',', '')
                total_cost += float(cleaned_num)
            except (ValueError, TypeError):
                continue
    return total_cost


# --- Data Loading and Processing Sub-Functions ---

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes all column names for easier access."""
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    return df

def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses all known date columns to datetime objects."""
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date', 'Delivery_Date', 'Job_Creation']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames and cleans key numeric columns."""
    numeric_column_map = {
        'Total_Job_Price_': 'Revenue',
        'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_T': 'Throughput_T',
        'Job_Throughput_Job_T_': 'Throughput_T_Percent',
        'Job_Throughput_Total_COGS': 'Total_COGS'
    }
    for original_name, new_name in numeric_column_map.items():
        if original_name in df.columns:
            cleaned_series = df[original_name].astype(str).str.replace(r'[$,%]', '', regex=True)
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            df[new_name] = numeric_series.fillna(0)
        else:
            df[new_name] = 0.0
    return df

def _calculate_profitability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all core profitability and cost metrics."""
    # Branch Profitability
    df['Install_Cost'] = df.get('Total_Job_SqFt', 0) * INSTALL_COST_PER_SQFT
    df['Total_Rework_Cost'] = df.get('Rework_COGS', 0) + df.get('Rework_Labor', 0) + df.get('Rework_Price', 0)
    df['Total_Branch_Cost'] = df['Cost_From_Plant'] + df['Install_Cost'] + df['Total_Rework_Cost']
    df['Branch_Profit'] = df.get('Revenue', 0) - df['Total_Branch_Cost']
    df['Branch_Profit_Margin_%'] = df.apply(
        lambda row: (row['Branch_Profit'] / row['Revenue'] * 100) if row.get('Revenue') and row['Revenue'] != 0 else 0,
        axis=1
    )
    df['Profit_Variance'] = df['Branch_Profit'] - df.get('Original_GM', 0)

    # Shop Profitability
    if 'Cost_From_Plant' in df.columns and 'Total_COGS' in df.columns:
        df['Shop_Profit'] = df['Cost_From_Plant'] - df['Total_COGS']
        df['Shop_Profit_Margin_%'] = df.apply(
            lambda row: (row['Shop_Profit'] / row['Cost_From_Plant'] * 100) if row.get('Cost_From_Plant') and row['Cost_From_Plant'] != 0 else 0,
            axis=1
        )
    else:
        df['Shop_Profit'] = 0.0
        df['Shop_Profit_Margin_%'] = 0.0
    return df

def _calculate_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates durations between key process stages and filters out negative values."""
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
    """Adds supplementary data like material parsing and geocoding."""
    # Parse material information
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material_Brand'] = "N/A"
        df['Material_Color'] = "N/A"

    # Geocode city data
    if 'City' in df.columns:
        df[['lat', 'lon']] = df['City'].apply(lambda x: pd.Series(get_lat_lon(str(x))))

    return df


@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict) -> pd.DataFrame:
    """Main data pipeline: Loads, cleans, and processes data from Google Sheets."""
    # 1. Load data from Google Sheets
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(worksheet.get_all_records())

    # 2. Initial Cost Calculation (before renaming columns)
    prod_cost_col = 'Phase Dollars - Plant Invoice $'
    fallback_prod_cost_col = 'Job Throughput - Job Plant Invoice'
    if prod_cost_col in df.columns:
        df['Cost_From_Plant'] = df[prod_cost_col].apply(calculate_production_cost)
    elif fallback_prod_cost_col in df.columns:
        df['Cost_From_Plant'] = df[fallback_prod_cost_col].apply(calculate_production_cost)
    else:
        df['Cost_From_Plant'] = 0.0

    # 3. Processing Pipeline
    df = _clean_column_names(df)
    df = _parse_date_columns(df)
    df = _clean_numeric_columns(df)
    df = _calculate_profitability_metrics(df)
    df = _calculate_durations(df)
    df = _enrich_data(df)

    # 4. Final cleaning
    if 'Production_' in df.columns:
        df['Production_'] = df['Production_'].fillna('').astype(str).str.strip()

    return df.copy()


# --- UI Rendering Functions ---

def render_overview_tab(df: pd.DataFrame):
    """Renders the 'Overall' performance tab."""
    st.header("📈 Overall Performance")
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    total_revenue = df['Revenue'].sum()
    total_profit = df['Branch_Profit'].sum()
    avg_margin = (total_profit / total_revenue * 100) if total_revenue else 0
    avg_throughput = df['Throughput_T'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Branch Profit", f"${total_profit:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
    c4.metric("Avg Throughput (T$)", f"${avg_throughput:,.0f}")

    st.markdown("---")
    st.subheader("Profit by Salesperson")
    if 'Salesperson' in df.columns and not df.empty:
        sales_profit = df.groupby('Salesperson')['Branch_Profit'].sum().sort_values(ascending=False)
        st.bar_chart(sales_profit)

def render_detailed_data_tab(df: pd.DataFrame):
    """Renders the 'Detailed Data' tab with filtering options."""
    st.header("📋 Detailed Data View")

    df_display = df.copy()

    # --- Add filter widgets ---
    st.markdown("#### Filters")
    col1, col2 = st.columns(2)
    with col1:
        job_name_filter = st.text_input("Filter by Job Name (contains)", key="job_name_filter")
    with col2:
        prod_num_filter = st.text_input("Filter by Production # (contains)", key="prod_num_filter")

    # Apply filters if they are not empty
    if job_name_filter and 'Job_Name' in df_display.columns:
        df_display = df_display[df_display['Job_Name'].str.contains(job_name_filter, case=False, na=False)]

    if prod_num_filter and 'Production_' in df_display.columns:
        df_display = df_display[df_display['Production_'].str.contains(prod_num_filter, case=False, na=False)]

    st.markdown("---")

    if df_display.empty:
        st.warning("No data available for the selected filters.")
        return

    # Create a new column containing the full URL for the link.
    if 'Production_' in df.columns:
        df_display['Link'] = df_display['Production_'].apply(
            lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None
        )

    column_config = {
        "Link": st.column_config.LinkColumn("Prod #", help="Click to search in Moraware", display_text=r".*search=(.*)"),
        "Production_": None, # Hide original column
        "Next_Sched_Activity": "Next Activity",
        "Revenue": st.column_config.NumberColumn(format='$%.2f'),
        "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
        "Cost_From_Plant": st.column_config.NumberColumn("Production Cost", format='$%.2f'),
        "Install_Cost": st.column_config.NumberColumn(format='$%.2f'),
        "Total_Branch_Cost": st.column_config.NumberColumn(format='$%.2f'),
        "Branch_Profit": st.column_config.NumberColumn(format='$%.2f'),
        "Profit_Variance": st.column_config.NumberColumn(format='$%.2f'),
        "Branch_Profit_Margin_%": st.column_config.ProgressColumn("Branch Profit Margin", format='%.2f%%', min_value=-100, max_value=100),
        "Shop_Profit_Margin_%": st.column_config.ProgressColumn("Shop Profit Margin", format='%.2f%%', min_value=-100, max_value=100),
    }

    # Column order now includes 'Next_Sched_Activity'
    column_order = [
        'Link', 'Job_Name', 'Next_Sched_Activity', 'Revenue', 'Total_Job_SqFt',
        'Cost_From_Plant', 'Install_Cost', 'Total_Branch_Cost', 'Branch_Profit',
        'Branch_Profit_Margin_%', 'Shop_Profit_Margin_%', 'Profit_Variance'
    ]
    # Ensure we only try to display columns that actually exist in the dataframe
    final_column_order = [c for c in column_order if c in df_display.columns]

    st.dataframe(
        df_display,
        use_container_width=True,
        column_config=column_config,
        column_order=final_column_order
    )

def render_profit_drivers_tab(df: pd.DataFrame):
    """Renders the 'Profit Drivers' tab."""
    st.header("💸 Profitability Drivers")
    st.markdown("Analyze which business segments are most profitable.")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    driver_options = ['Job_Type', 'Order_Type', 'Lead_Source', 'Material_Brand']
    valid_drivers = [d for d in driver_options if d in df.columns]

    if not valid_drivers:
        st.warning("No valid columns (Job_Type, Order_Type, etc.) found for this analysis.")
        return

    selected_driver = st.selectbox("Analyze Profitability by:", valid_drivers)

    if selected_driver:
        agg_dict = {
            'Avg_Branch_Profit_Margin': ('Branch_Profit_Margin_%', 'mean'),
            'Total_Profit': ('Branch_Profit', 'sum'),
            'Job_Count': ('Production_', 'count')
        }
        if 'Shop_Profit_Margin_%' in df.columns:
            agg_dict['Avg_Shop_Profit_Margin'] = ('Shop_Profit_Margin_%', 'mean')

        driver_analysis = df.groupby(selected_driver).agg(**agg_dict).sort_values('Avg_Branch_Profit_Margin', ascending=False)

        st.subheader(f"Profitability by {selected_driver.replace('_', ' ')}")
        format_dict = {
            'Avg_Branch_Profit_Margin': '{:.2f}%',
            'Total_Profit': '${:,.2f}'
        }
        if 'Avg_Shop_Profit_Margin' in driver_analysis.columns:
            format_dict['Avg_Shop_Profit_Margin'] = '{:.2f}%'

        st.dataframe(driver_analysis.style.format(format_dict), use_container_width=True)

def render_rework_tab(df: pd.DataFrame):
    """Renders the 'Rework & Variance' tab."""
    st.header("🔬 Rework Insights & Profit Variance")
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rework Analysis")
        if 'Total_Rework_Cost' in df and 'Rework_Stone_Shop_Reason' in df.columns:
            rework_jobs = df[df['Total_Rework_Cost'] > 0].copy()
            if not rework_jobs.empty:
                st.metric("Total Rework Cost", f"${rework_jobs['Total_Rework_Cost'].sum():,.2f}", f"{len(rework_jobs)} jobs affected")
                st.write("**Rework Costs by Reason**")
                agg_rework = rework_jobs.groupby('Rework_Stone_Shop_Reason')['Total_Rework_Cost'].agg(['sum', 'count'])
                agg_rework.columns = ['Total Rework Cost', 'Number of Jobs']
                st.dataframe(agg_rework.sort_values('Total Rework Cost', ascending=False).style.format({'Total Rework Cost': '${:,.2f}'}))
            else:
                st.info("No rework costs recorded for the current selection.")
        else:
            st.info("Rework data not available in the source sheet.")

    with c2:
        st.subheader("Profit Variance Analysis")
        if 'Profit_Variance' in df.columns and 'Original_GM' in df.columns:
            variance_jobs = df[df['Profit_Variance'].abs() > 0.01].copy()
            if not variance_jobs.empty:
                st.metric("Jobs with Profit Variance", f"{len(variance_jobs)}")
                st.write("**Jobs with Largest Profit Variance**")
                if 'Production_' in variance_jobs.columns:
                    variance_jobs['Link'] = variance_jobs['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
                
                variance_display_cols = ['Link', 'Job_Name', 'Next_Sched_Activity', 'Original_GM', 'Branch_Profit', 'Profit_Variance']
                
                st.dataframe(
                    variance_jobs[[c for c in variance_display_cols if c in variance_jobs.columns]].sort_values(by='Profit_Variance', key=abs, ascending=False).head(20),
                    use_container_width=True,
                    column_config={
                        "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
                        "Next_Sched_Activity": "Next Activity",
                        "Original_GM": st.column_config.NumberColumn("Est. Profit", format='$%.2f'),
                        "Branch_Profit": st.column_config.NumberColumn("Actual Profit", format='$%.2f'),
                        "Profit_Variance": st.column_config.NumberColumn("Variance", format='$%.2f'),
                    }
                )
            else:
                st.info("No significant profit variance found.")
        else:
            st.info("Profit variance data not available.")

def render_pipeline_issues_tab(df: pd.DataFrame, today: pd.Timestamp):
    """Renders the 'Pipeline & Issues' tab, using a specific date as 'today'."""
    st.header("🚧 Job Pipeline & Issues")
    # This tab receives the full, unfiltered dataframe.
    # The main sidebar filters do NOT apply here.

    # --- Section for Jobs Awaiting RTF ---
    st.subheader("Jobs Awaiting Ready-to-Fab")
    st.markdown("Jobs that have been templated but are not yet marked as 'Ready to Fab' as of the selected date. This view ignores the sidebar filters.")
    
    if 'Ready_to_Fab_Status' in df.columns:
        conditions = (
            (df['Template_Date'].notna()) &
            (df['Template_Date'] <= today) &
            (df['Ready_to_Fab_Status'].fillna('').str.lower() != 'complete')
        )
        stuck_jobs = df[conditions].copy()

        if not stuck_jobs.empty:
            stuck_jobs['Days_Since_Template'] = (today - stuck_jobs['Template_Date']).dt.days
            if 'Production_' in stuck_jobs.columns:
                stuck_jobs['Link'] = stuck_jobs['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
            
            display_cols = ['Link', 'Job_Name', 'Next_Sched_Activity', 'Salesperson', 'Template_Date', 'Days_Since_Template']
            
            st.dataframe(
                stuck_jobs[[c for c in display_cols if c in stuck_jobs.columns]].sort_values(by='Days_Since_Template', ascending=False),
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
                    "Next_Sched_Activity": "Next Activity",
                    "Template_Date": st.column_config.DateColumn("Template Date", format="YYYY-MM-DD")
                }
            )
        else:
            st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    else:
        st.warning("Could not find a 'Ready_to_Fab_Status' column in the data. This section requires that column to function.")


    st.markdown("---")
    
    # --- Section for Missing Plant Invoice ---
    st.subheader("Jobs with Missing Plant Invoice $")
    st.markdown("Jobs that are missing the 'Phase Dollars - Plant Invoice $' amount. This view ignores the sidebar filters.")
    if 'Cost_From_Plant' in df.columns and 'Job_Creation' in df.columns:
        missing_invoice_jobs = df[df['Cost_From_Plant'] == 0].copy()
        if not missing_invoice_jobs.empty:
            if 'Production_' in missing_invoice_jobs.columns:
                missing_invoice_jobs['Link'] = missing_invoice_jobs['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
            
            display_cols = ['Link', 'Job_Name', 'Next_Sched_Activity', 'Salesperson', 'Job_Creation']
            
            st.dataframe(
                missing_invoice_jobs[[c for c in display_cols if c in missing_invoice_jobs.columns]].sort_values(by='Job_Creation', ascending=False),
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
                    "Next_Sched_Activity": "Next Activity",
                    "Job_Creation": st.column_config.DateColumn("Job Creation Date", format="YYYY-MM-DD")
                }
            )
        else:
            st.success("✅ No jobs with missing plant invoices in the current selection.")
    else:
        st.warning("Could not check for missing invoices. Required columns not found.")

def render_field_workload_tab(df: pd.DataFrame):
    """Renders the enhanced tab for Template, Install, and Service workloads."""
    st.header("👷 Field Workload Planner")
    st.markdown("Weekly breakdown of jobs and square footage for each team member, based on the selected filters.")
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    def render_workload_analysis(df_filtered: pd.DataFrame, activity_name: str, date_col: str, assignee_col: str):
        """A reusable function to display weekly workload for a given activity."""
        st.subheader(activity_name)
        if date_col not in df_filtered.columns or assignee_col not in df_filtered.columns:
            st.warning(f"Required columns not found for {activity_name} analysis.")
            return

        activity_df = df_filtered.dropna(subset=[date_col, assignee_col]).copy()
        if activity_df.empty:
            st.info(f"No {activity_name.lower()} data available for the current selection.")
            return

        assignees = sorted([name for name in activity_df[assignee_col].unique() if name and str(name).strip()])
        for assignee in assignees:
            with st.expander(f"**{assignee}**"):
                assignee_df = activity_df[activity_df[assignee_col] == assignee].copy()
                weekly_summary = assignee_df.set_index(date_col).resample('W-Mon', label='left', closed='left').agg(
                    Jobs=('Production_', 'count'),
                    Total_SqFt=('Total_Job_SqFt', 'sum')
                ).reset_index()
                weekly_summary = weekly_summary[weekly_summary['Jobs'] > 0]
                if not weekly_summary.empty:
                    st.dataframe(weekly_summary.rename(columns={date_col: 'Week_Start_Date'}), use_container_width=True)
                else:
                    st.write("No scheduled work for this person in the selected period.")

    render_workload_analysis(df, "Templates", "Template_Date", "Template_Assigned_To")
    st.markdown("---")
    render_workload_analysis(df, "Installs", "Install_Date", "Install_Assigned_To")
    st.markdown("---")
    render_workload_analysis(df, "Service", "Service_Date", "Service_Assigned_To")
    st.markdown("---")
    render_workload_analysis(df, "Delivery", "Delivery_Date", "Delivery_Assigned_To")

def render_forecasting_tab(df: pd.DataFrame):
    """Renders the 'Forecasting & Trends' tab."""
    st.header("🔮 Forecasting & Trends")

    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.error("Forecasting features require scikit-learn and matplotlib. Please install them.")
        return

    if 'Job_Creation' not in df.columns or df['Job_Creation'].isnull().all():
        st.warning("Job Creation date column is required for trend analysis and is missing or empty.")
        return
        
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    df_trends = df.copy()
    df_trends = df_trends.set_index('Job_Creation').sort_index()

    # --- Monthly Performance Trends ---
    st.subheader("Monthly Performance Trends")
    st.caption("Trends are based on the data within the selected date range and filters.")
    monthly_summary = df_trends.resample('M').agg({
        'Revenue': 'sum',
        'Branch_Profit': 'sum',
        'Production_': 'count'
    }).rename(columns={'Production_': 'Job_Count'})
    monthly_summary['Branch_Profit_Margin_%'] = (monthly_summary['Branch_Profit'] / monthly_summary['Revenue'] * 100).fillna(0)

    if monthly_summary.empty or len(monthly_summary) < 2:
        st.info("Not enough historical data in the selected range to display monthly trends or forecasts.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(monthly_summary[['Revenue', 'Branch_Profit']])
    with c2:
        st.line_chart(monthly_summary[['Branch_Profit_Margin_%']])
    st.bar_chart(monthly_summary['Job_Count'])

    st.markdown("---")

    # --- Revenue Forecast ---
    st.subheader("Simple Revenue Forecast")
    st.caption("Note: This is a simple linear forecast based on the filtered data. For better results, select a longer date range (12+ months).")

    forecast_df = monthly_summary[['Revenue']].reset_index()
    forecast_df['Time'] = np.arange(len(forecast_df.index))

    model = LinearRegression()
    X = forecast_df[['Time']]
    y = forecast_df['Revenue']
    model.fit(X, y)

    future_periods = st.slider("Months to Forecast:", 1, 12, 3)
    last_time = forecast_df['Time'].max()
    last_date = forecast_df['Job_Creation'].max()

    future_dates = pd.to_datetime([last_date + pd.DateOffset(months=i) for i in range(1, future_periods + 1)])
    future_df = pd.DataFrame({
        'Job_Creation': future_dates,
        'Time': np.arange(last_time + 1, last_time + 1 + future_periods)
    })
    future_df['Forecast'] = model.predict(future_df[['Time']])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df['Job_Creation'], forecast_df['Revenue'], label='Actual Revenue', marker='o')
    ax.plot(future_df['Job_Creation'], future_df['Forecast'], label='Forecasted Revenue', linestyle='--', marker='o')
    ax.set_title('Monthly Revenue and Forecast')
    ax.set_ylabel('Revenue ($)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.title("💰 Enhanced Job Profitability Dashboard")
    st.markdown("An interactive dashboard to analyze job data and drive profitability.")

    # --- Sidebar: Credentials ---
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

    # --- Data Loading ---
    try:
        df_full = load_and_process_data(creds)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.exception(e)
        st.stop()

    if df_full.empty:
        st.warning("No data was loaded from the Google Sheet.")
        st.stop()

    # --- Sidebar: Date Selection for Pipeline ---
    st.sidebar.header("🗓️ Date Selection")
    today_date = st.sidebar.date_input(
        "Select 'Today's' Date for Pipeline Calculations",
        value=datetime.now().date(),
        help="This date is used to calculate metrics like 'Days Since Template'."
    )
    today_dt = pd.to_datetime(today_date)

    # --- Main Content: Tabs ---
    tab_names = [
        "📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "� Rework & Variance",
        "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting & Trends"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]: render_overview_tab(df_full)
    with tabs[1]: render_detailed_data_tab(df_full)
    with tabs[2]: render_profit_drivers_tab(df_full)
    with tabs[3]: render_rework_tab(df_full)
    with tabs[4]: render_pipeline_issues_tab(df_full, today_dt)
    with tabs[5]: render_field_workload_tab(df_full)
    with tabs[6]: render_forecasting_tab(df_full)

if __name__ == "__main__":
    main()
