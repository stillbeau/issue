# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This application connects to a specified Google Sheet, processes job data,
calculates various financial and operational metrics, and presents them in an
interactive, multi-tabbed dashboard with forecasting capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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


# --- Data Loading and Processing ---

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict) -> pd.DataFrame:
    """Loads, cleans, and processes data from Google Sheets for analysis."""
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(worksheet.get_all_records())

    # --- Data Cleaning and Transformation ---
    
    # 1. Calculate Production Cost FIRST, using original column names for reliability
    prod_cost_col = 'Phase Dollars - Plant Invoice $'
    fallback_prod_cost_col = 'Job Throughput - Job Plant Invoice'
    
    if prod_cost_col in df.columns:
        df['Cost_From_Plant'] = df[prod_cost_col].apply(calculate_production_cost)
    elif fallback_prod_cost_col in df.columns:
        df['Cost_From_Plant'] = df[fallback_prod_cost_col].apply(calculate_production_cost)
    else:
        df['Cost_From_Plant'] = 0.0

    # 2. Standardize all column names for easier access later
    original_cols = df.columns.tolist()
    new_cols = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    df.columns = new_cols
    # Create a mapping from new to original names for later use if needed
    col_map = dict(zip(new_cols, original_cols))

    # 3. Clean the Production_ column to ensure it's a string for link generation
    if 'Production_' in df.columns:
        # This sequence is important: fill NA, then convert to string, then strip.
        # This avoids converting None to the string 'None' or NaN to 'nan'.
        df['Production_'] = df['Production_'].fillna('').astype(str).str.strip()

    # 4. Rename and clean remaining numeric columns
    numeric_column_map = {
        'Total_Job_Price_': 'Revenue',
        'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Total_Job_SqFT': 'Total_Job_SqFt',
        'Job_Throughput_Job_T': 'Throughput_T',
        'Job_Throughput_Job_T_': 'Throughput_T_Percent'
    }
    for original_name, new_name in numeric_column_map.items():
        if original_name in df.columns:
            cleaned_series = df[original_name].astype(str).str.replace(r'[$,%]', '', regex=True)
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            df[new_name] = numeric_series.fillna(0)
        else:
            df[new_name] = 0.0

    # 5. Parse date columns using the new standardized names
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Service_Date', 'Delivery_Date', 'Job_Creation']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 6. Calculate profitability metrics
    df['Install_Cost'] = df.get('Total_Job_SqFt', 0) * INSTALL_COST_PER_SQFT
    df['Total_Rework_Cost'] = df.get('Rework_COGS', 0) + df.get('Rework_Labor', 0) + df.get('Rework_Price', 0)
    df['Total_Branch_Cost'] = df['Cost_From_Plant'] + df['Install_Cost'] + df['Total_Rework_Cost']
    df['Branch_Profit'] = df.get('Revenue', 0) - df['Total_Branch_Cost']
    df['Branch_Profit_Margin_%'] = df.apply(
        lambda row: (row['Branch_Profit'] / row['Revenue'] * 100) if row.get('Revenue') and row['Revenue'] != 0 else 0,
        axis=1
    )
    df['Profit_Variance'] = df['Branch_Profit'] - df.get('Original_GM', 0)

    # 7. Parse material information
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material_Brand'] = "N/A"

    # 8. Calculate stage durations and filter out negative values
    if 'Ready_to_Fab_Date' in df.columns and 'Template_Date' in df.columns:
        df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
        df = df[df['Days_Template_to_RTF'] >= 0]
    if 'Ship_Date' in df.columns and 'Ready_to_Fab_Date' in df.columns:
        df['Days_RTF_to_Ship'] = (df['Ship_Date'] - df['Ready_to_Fab_Date']).dt.days
        df = df[df['Days_RTF_to_Ship'] >= 0]
    if 'Install_Date' in df.columns and 'Ship_Date' in df.columns:
        df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
        df = df[df['Days_Ship_to_Install'] >= 0]

    # 9. Geocode city data
    if 'City' in df.columns:
        df[['lat', 'lon']] = df['City'].apply(lambda x: pd.Series(get_lat_lon(str(x))))

    return df.copy()


# --- UI Rendering Functions ---

def render_overview_tab(df: pd.DataFrame):
    """Renders the 'Overall' performance tab."""
    st.header("📈 Overall Performance")
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
    if 'Salesperson' in df.columns:
        st.bar_chart(df.groupby('Salesperson')['Branch_Profit'].sum())

def render_detailed_data_tab(df: pd.DataFrame):
    """Renders the 'Detailed Data' tab."""
    st.header("📋 Detailed Data View")
    display_cols = [
        'Production_', 'Job_Name', 'Revenue', 'Total_Job_SqFt', 
        'Cost_From_Plant', 'Install_Cost', 'Total_Branch_Cost', 'Branch_Profit', 
        'Branch_Profit_Margin_%', 'Profit_Variance'
    ]
    df_display = df[[c for c in display_cols if c in df.columns]].copy()
    
    column_config = {
        "Revenue": st.column_config.NumberColumn(format='$%.2f'),
        "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
        "Cost_From_Plant": st.column_config.NumberColumn("Production Cost", format='$%.2f'),
        "Install_Cost": st.column_config.NumberColumn(format='$%.2f'),
        "Total_Branch_Cost": st.column_config.NumberColumn(format='$%.2f'),
        "Branch_Profit": st.column_config.NumberColumn(format='$%.2f'),
        "Profit_Variance": st.column_config.NumberColumn(format='$%.2f'),
        "Branch_Profit_Margin_%": st.column_config.ProgressColumn(
            "Profit Margin", format='%.2f%%', min_value=-100, max_value=100
        ),
    }
    
    if 'Production_' in df_display.columns:
        column_config["Production_"] = st.column_config.LinkColumn(
            "Prod #", 
            href=f"{MORAWARE_SEARCH_URL}%s"
        )

    st.dataframe(
        df_display, use_container_width=True,
        column_config=column_config
    )

def render_profit_drivers_tab(df: pd.DataFrame):
    """Renders the 'Profit Drivers' tab."""
    st.header("💸 Profitability Drivers")
    st.markdown("Analyze which business segments are most profitable.")
    
    driver_options = ['Job_Type', 'Order_Type', 'Lead_Source', 'Material_Brand']
    valid_drivers = [d for d in driver_options if d in df.columns]

    if not valid_drivers:
        st.warning("No valid columns (Job_Type, Order_Type, etc.) found for this analysis.")
        return

    selected_driver = st.selectbox("Analyze Profitability by:", valid_drivers)
    
    if selected_driver:
        driver_analysis = df.groupby(selected_driver).agg(
            Avg_Profit_Margin=('Branch_Profit_Margin_%', 'mean'),
            Total_Profit=('Branch_Profit', 'sum'),
            Job_Count=('Production_', 'count')
        ).sort_values('Avg_Profit_Margin', ascending=False)
        
        st.subheader(f"Profitability by {selected_driver.replace('_', ' ')}")
        st.dataframe(driver_analysis.style.format({
            'Avg_Profit_Margin': '{:.2f}%',
            'Total_Profit': '${:,.2f}'
        }))

def render_rework_tab(df: pd.DataFrame):
    """Renders the 'Rework & Variance' tab with detailed job links."""
    st.header("🔬 Rework Insights & Profit Variance")
    
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

                with st.expander("View Rework Job Details"):
                    rework_display_cols = ['Production_', 'Job_Name', 'Total_Rework_Cost', 'Rework_Stone_Shop_Reason']
                    st.dataframe(
                        rework_jobs[[c for c in rework_display_cols if c in rework_jobs.columns]],
                        use_container_width=True,
                        column_config={
                            "Production_": st.column_config.LinkColumn("Prod #", href=f"{MORAWARE_SEARCH_URL}%s"),
                            "Total_Rework_Cost": st.column_config.NumberColumn("Rework Cost", format='$%.2f'),
                        }
                    )
            else:
                st.info("No rework costs recorded for the current selection.")
        else:
            st.info("Rework data not available.")

    with c2:
        st.subheader("Profit Variance Analysis")
        if 'Profit_Variance' in df.columns and 'Original_GM' in df.columns:
            variance_jobs = df[df['Profit_Variance'].abs() > 0.01].copy()
            if not variance_jobs.empty:
                st.metric("Jobs with Profit Variance", f"{len(variance_jobs)}")

                st.write("**Jobs with Largest Profit Variance**")
                variance_display_cols = ['Production_', 'Job_Name', 'Original_GM', 'Branch_Profit', 'Profit_Variance']
                st.dataframe(
                    variance_jobs[[c for c in variance_display_cols if c in variance_jobs.columns]].sort_values(by='Profit_Variance', key=abs, ascending=False).head(20),
                    use_container_width=True,
                    column_config={
                        "Production_": st.column_config.LinkColumn("Prod #", href=f"{MORAWARE_SEARCH_URL}%s"),
                        "Original_GM": st.column_config.NumberColumn("Est. Profit", format='$%.2f'),
                        "Branch_Profit": st.column_config.NumberColumn("Actual Profit", format='$%.2f'),
                        "Profit_Variance": st.column_config.NumberColumn("Variance", format='$%.2f'),
                    }
                )
            else:
                st.info("No significant profit variance found.")
        else:
            st.info("Profit variance data not available.")

def render_pipeline_issues_tab(df: pd.DataFrame):
    """Renders the 'Pipeline & Issues' tab."""
    st.header("🚧 Job Pipeline & Issues")
    
    st.subheader("Jobs Awaiting Ready-to-Fab")
    st.markdown("Jobs that have been templated but are not yet marked as 'Ready to Fab'.")

    today = pd.to_datetime('today').normalize()
    stuck_jobs = df[
        (df['Template_Date'].notna()) & (df['Template_Date'] <= today) & (df['Ready_to_Fab_Date'].isna())
    ].copy()

    if not stuck_jobs.empty:
        stuck_jobs['Days_Since_Template'] = (today - stuck_jobs['Template_Date']).dt.days
        display_cols = ['Production_', 'Job_Name', 'Salesperson', 'Template_Date', 'Days_Since_Template']
        st.dataframe(
            stuck_jobs[[c for c in display_cols if c in stuck_jobs.columns]].sort_values(by='Days_Since_Template', ascending=False),
            use_container_width=True,
            column_config={
                "Production_": st.column_config.LinkColumn("Prod #", href=f"{MORAWARE_SEARCH_URL}%s"),
                "Template_Date": st.column_config.DateColumn("Template Date", format="YYYY-MM-DD")
            }
        )
    else:
        st.success("✅ No jobs are currently stuck between Template and Ready to Fab.")
    
    st.markdown("---")
    
    issue_cols = ['Job_Issues', 'Account_Issues']
    valid_issue_cols = [i for i in issue_cols if i in df.columns]
    if valid_issue_cols:
        st.subheader("Jobs with Reported Issues")
        jobs_with_issues = df[df[valid_issue_cols].notna().any(axis=1) & (df[valid_issue_cols] != '').any(axis=1)].copy()
        if not jobs_with_issues.empty:
            display_cols = ['Production_', 'Job_Name', 'Branch_Profit_Margin_%'] + valid_issue_cols
            st.dataframe(
                jobs_with_issues[[c for c in display_cols if c in jobs_with_issues.columns]],
                column_config={
                    "Production_": st.column_config.LinkColumn("Prod #", href=f"{MORAWARE_SEARCH_URL}%s"),
                }
            )
        else:
            st.info("No jobs with issues in the current selection.")

def render_workload_analysis(df: pd.DataFrame, activity_name: str, date_col: str, assignee_col: str):
    """A reusable function to display weekly workload for a given activity."""
    st.subheader(activity_name)
    
    if date_col not in df.columns or assignee_col not in df.columns:
        st.warning(f"Required columns not found for {activity_name} analysis.")
        return

    activity_df = df.dropna(subset=[date_col, assignee_col]).copy()
    
    if activity_df.empty:
        st.info(f"No {activity_name.lower()} data available for the current selection.")
        return

    assignees = sorted([name for name in activity_df[assignee_col].unique() if name and str(name).strip()])
    
    for assignee in assignees:
        with st.expander(f"**{assignee}**"):
            assignee_df = activity_df[activity_df[assignee_col] == assignee]
            
            weekly_summary = assignee_df.set_index(date_col).resample('W-Mon', label='left', closed='left').agg(
                Jobs=('Production_', 'count'),
                Total_SqFt=('Total_Job_SqFt', 'sum')
            ).reset_index()
            weekly_summary = weekly_summary[weekly_summary['Jobs'] > 0]
            
            if not weekly_summary.empty:
                st.write("**Weekly Summary**")
                st.dataframe(weekly_summary.rename(columns={date_col: 'Week_Start_Date'}), use_container_width=True)

                with st.expander("Show Job Details"):
                    job_detail_cols = ['Production_', 'Job_Name', 'Total_Job_SqFt', date_col]
                    st.dataframe(
                        assignee_df[[c for c in job_detail_cols if c in assignee_df.columns]].sort_values(by=date_col),
                        use_container_width=True,
                        column_config={
                            "Production_": st.column_config.LinkColumn("Prod #", href=f"{MORAWARE_SEARCH_URL}%s"),
                            date_col: st.column_config.DateColumn("Scheduled Date", format="YYYY-MM-DD")
                        }
                    )
            else:
                st.write("No scheduled work for this person in the selected period.")

def render_field_workload_tab(df: pd.DataFrame):
    """Renders the enhanced tab for Template, Install, and Service workloads."""
    st.header("👷 Field Workload Planner")
    st.markdown("Weekly breakdown of jobs and square footage for each team member.")

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

    df_trends = df.copy()
    df_trends = df_trends.set_index('Job_Creation').sort_index()
    
    # --- Monthly Performance Trends ---
    st.subheader("Monthly Performance Trends")
    monthly_summary = df_trends.resample('M').agg({
        'Revenue': 'sum',
        'Branch_Profit': 'sum',
        'Production_': 'count'
    }).rename(columns={'Production_': 'Job_Count'})
    monthly_summary['Branch_Profit_Margin_%'] = (monthly_summary['Branch_Profit'] / monthly_summary['Revenue'] * 100).fillna(0)
    
    # Remove the last month if it's incomplete
    if not monthly_summary.empty:
        last_month = monthly_summary.index[-1]
        if last_month.month == datetime.now().month and last_month.year == datetime.now().year:
            monthly_summary = monthly_summary[:-1]
    
    if monthly_summary.empty:
        st.info("Not enough historical data to display monthly trends.")
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
    
    forecast_df = monthly_summary[['Revenue']].reset_index()
    forecast_df['Time'] = np.arange(len(forecast_df.index))
    
    # Train model
    model = LinearRegression()
    X = forecast_df[['Time']]
    y = forecast_df['Revenue']
    model.fit(X, y)
    
    # Create future dataframe
    future_periods = st.slider("Months to Forecast:", 1, 12, 3)
    last_time = forecast_df['Time'].max()
    last_date = forecast_df['Job_Creation'].max()
    
    future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq='M')[1:]
    future_df = pd.DataFrame({
        'Job_Creation': future_dates,
        'Time': np.arange(last_time + 1, last_time + 1 + future_periods)
    })
    
    # Predict future revenue
    future_df['Forecast'] = model.predict(future_df[['Time']])
    
    # Combine and plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df['Job_Creation'], forecast_df['Revenue'], label='Actual Revenue', marker='o')
    ax.plot(future_df['Job_Creation'], future_df['Forecast'], label='Forecasted Revenue', linestyle='--', marker='o')
    
    ax.set_title('Monthly Revenue and Forecast')
    ax.set_ylabel('Revenue ($)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    
    st.caption("Note: This is a simple linear regression forecast and should be used for directional guidance only.")


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.title("💰 Enhanced Job Profitability Dashboard")
    st.markdown("An efficient, multi-faceted dashboard to analyze job data and drive profitability.")

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

    try:
        df_full = load_and_process_data(creds)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.exception(e) # Provides a full traceback for debugging
        st.stop()

    st.sidebar.header("📊 Filters")
    if 'Job_Creation' in df_full.columns and not df_full['Job_Creation'].dropna().empty:
        min_date = df_full['Job_Creation'].min().date()
        max_date = df_full['Job_Creation'].max().date()
        
        # Default to the last 12 months if the date range is very large
        default_start = max_date - relativedelta(months=12)
        if default_start < min_date:
            default_start = min_date

        start_date, end_date = st.sidebar.date_input(
            "Filter by Job Creation Date", value=[default_start, max_date],
            min_value=min_date, max_value=max_date
        )
        df_filtered = df_full[
            (df_full['Job_Creation'].dt.date >= start_date) &
            (df_full['Job_Creation'].dt.date <= end_date)
        ]
    else:
        df_filtered = df_full

    def get_unique_options(df, col_name):
        return sorted(df[col_name].dropna().unique()) if col_name in df else []

    filter_cols = {'Salesperson': 'Salesperson', 'Customer_Category': 'Customer Category', 
                   'Material_Brand': 'Material Brand', 'City': 'City'}
    for col, label in filter_cols.items():
        if col in df_filtered:
            options = get_unique_options(df_filtered, col)
            # Use a key for multiselect to prevent state issues on rerun
            selected = st.sidebar.multiselect(label, options, default=options, key=f"select_{col}")
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    if df_filtered.empty:
        st.warning("No data matches the current filter selection.")
        st.stop()

    tab_names = [
        "📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance",
        "🚧 Pipeline & Issues", "👷 Field Workload", "🔮 Forecasting & Trends"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]: render_overview_tab(df_filtered)
    with tabs[1]: render_detailed_data_tab(df_filtered)
    with tabs[2]: render_profit_drivers_tab(df_filtered)
    with tabs[3]: render_rework_tab(df_filtered)
    with tabs[4]: render_pipeline_issues_tab(df_filtered)
    with tabs[5]: render_field_workload_tab(df_filtered)
    with tabs[6]: render_forecasting_tab(df_filtered)

if __name__ == "__main__":
    main()
