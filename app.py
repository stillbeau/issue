# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This application connects to a specified Google Sheet, processes job data,
calculates various financial and operational metrics, and presents them in an
interactive, multi-tabbed dashboard.
"""

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime
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
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="�")

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
        # Find all occurrences of numbers (including decimals)
        # This regex handles currency symbols, commas, and negative values
        numbers = re.findall(r'-?\$?[\d,]*\.?\d+', value)
        for num_str in numbers:
            try:
                # Clean the string and convert to float
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
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)

    # 3. Rename and clean remaining numeric columns
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

    # 4. Parse date columns
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 'Job_Creation']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 5. Calculate profitability metrics using the now-correct 'Cost_From_Plant'
    df['Install_Cost'] = df.get('Total_Job_SqFt', 0) * INSTALL_COST_PER_SQFT
    df['Total_Rework_Cost'] = df.get('Rework_COGS', 0) + df.get('Rework_Labor', 0) + df.get('Rework_Price', 0)
    df['Total_Branch_Cost'] = df['Cost_From_Plant'] + df['Install_Cost'] + df['Total_Rework_Cost']
    df['Branch_Profit'] = df.get('Revenue', 0) - df['Total_Branch_Cost']
    df['Branch_Profit_Margin_%'] = df.apply(
        lambda row: (row['Branch_Profit'] / row['Revenue'] * 100) if row.get('Revenue') and row['Revenue'] != 0 else 0,
        axis=1
    )
    df['Profit_Variance'] = df['Branch_Profit'] - df.get('Original_GM', 0)

    # 6. Parse material information
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material_Brand'] = "N/A"

    # 7. Calculate stage durations and filter out negative (irregular) values
    if 'Ready_to_Fab_Date' in df.columns and 'Template_Date' in df.columns:
        df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
        df = df[df['Days_Template_to_RTF'] >= 0]
    if 'Ship_Date' in df.columns and 'Ready_to_Fab_Date' in df.columns:
        df['Days_RTF_to_Ship'] = (df['Ship_Date'] - df['Ready_to_Fab_Date']).dt.days
        df = df[df['Days_RTF_to_Ship'] >= 0]
    if 'Install_Date' in df.columns and 'Ship_Date' in df.columns:
        df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
        df = df[df['Days_Ship_to_Install'] >= 0]

    # 8. Create dynamic job link
    if 'Production_' in df.columns:
        df['Job_Link'] = MORAWARE_SEARCH_URL + df['Production_'].astype(str)

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
        'Production_', 'Job_Link', 'Job_Name', 'Revenue', 'Total_Job_SqFt', 
        'Cost_From_Plant', 'Install_Cost', 'Total_Branch_Cost', 'Branch_Profit', 
        'Branch_Profit_Margin_%', 'Profit_Variance'
    ]
    df_display = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(
        df_display, use_container_width=True,
        column_config={
            "Production_": "Prod #",
            "Job_Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗"),
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
    """Renders the 'Rework & Variance' tab."""
    st.header("🔬 Rework Insights & Profit Variance")
    if 'Total_Rework_Cost' in df and 'Rework_Stone_Shop_Reason' in df.columns:
        rework_jobs = df[df['Total_Rework_Cost'] > 0]
        if not rework_jobs.empty:
            st.subheader("Rework Costs by Reason")
            agg_rework = rework_jobs.groupby('Rework_Stone_Shop_Reason')['Total_Rework_Cost'].agg(['sum', 'count'])
            agg_rework.columns = ['Total Rework Cost', 'Number of Jobs']
            st.dataframe(agg_rework.sort_values('Total Rework Cost', ascending=False).style.format({'Total Rework Cost': '${:,.2f}'}))
        else:
            st.info("No rework costs recorded for the current selection.")

def render_pipeline_issues_tab(df: pd.DataFrame):
    """Renders the 'Pipeline & Issues' tab."""
    st.header("🚧 Job Pipeline & Issues")
    
    status_cols = ['Template_Status', 'Ready_to_Fab_Status', 'Ship_Status', 'Install_Status']
    valid_status_cols = [s for s in status_cols if s in df.columns]

    if valid_status_cols:
        selected_status = st.selectbox("View Pipeline Stage:", valid_status_cols)
        st.subheader(f"Current Status for {selected_status.replace('_', ' ')}")
        status_counts = df[selected_status].value_counts()
        st.bar_chart(status_counts)
    
    issue_cols = ['Job_Issues', 'Account_Issues']
    valid_issue_cols = [i for i in issue_cols if i in df.columns]
    if valid_issue_cols:
        st.markdown("---")
        st.subheader("Jobs with Reported Issues")
        jobs_with_issues = df[df[valid_issue_cols].notna().any(axis=1) & (df[valid_issue_cols] != '').any(axis=1)]
        if not jobs_with_issues.empty:
            st.dataframe(jobs_with_issues[['Production_', 'Job_Name', 'Branch_Profit_Margin_%'] + valid_issue_cols])
        else:
            st.info("No jobs with issues in the current selection.")

def render_template_install_tab(df: pd.DataFrame):
    """Renders the weekly forecast for Template and Install activities."""
    st.header("👷 Weekly Template & Install Forecast")

    # --- Template Section ---
    st.subheader("Templates")
    if 'Template_Date' in df.columns and 'Template_Assigned_To' in df.columns:
        template_df = df.dropna(subset=['Template_Date', 'Template_Assigned_To']).copy()
        if not template_df.empty:
            template_weekly = template_df.set_index('Template_Date').resample('W-Mon', label='left', closed='left').agg(
                Jobs=('Production_', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).reset_index()
            st.write("**Weekly Template Volume**")
            st.dataframe(template_weekly)

    # --- Install Section ---
    st.subheader("Installs")
    if 'Install_Date' in df.columns and 'Install_Assigned_To' in df.columns:
        install_df = df.dropna(subset=['Install_Date', 'Install_Assigned_To']).copy()
        if not install_df.empty:
            install_weekly = install_df.set_index('Install_Date').resample('W-Mon', label='left', closed='left').agg(
                Jobs=('Production_', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).reset_index()
            st.write("**Weekly Install Volume**")
            st.dataframe(install_weekly)

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
        st.stop()

    st.sidebar.header("📊 Filters")
    # Date-range filter
    if 'Job_Creation' in df_full.columns and not df_full['Job_Creation'].dropna().empty:
        min_date = df_full['Job_Creation'].min().date()
        max_date = df_full['Job_Creation'].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Filter by Job Creation Date", value=[min_date, max_date],
            min_value=min_date, max_value=max_date
        )
        df_filtered = df_full[
            (df_full['Job_Creation'].dt.date >= start_date) &
            (df_full['Job_Creation'].dt.date <= end_date)
        ]
    else:
        df_filtered = df_full

    # Multiselect filters
    def get_unique_options(df, col_name):
        return sorted(df[col_name].dropna().unique()) if col_name in df else []

    filter_cols = {'Salesperson': 'Salesperson', 'Customer_Category': 'Customer Category', 
                   'Material_Brand': 'Material Brand', 'City': 'City'}
    for col, label in filter_cols.items():
        if col in df_filtered:
            options = get_unique_options(df_filtered, col)
            selected = st.sidebar.multiselect(label, options, default=options)
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    if df_filtered.empty:
        st.warning("No data matches the current filter selection.")
        st.stop()

    tab_names = [
        "📈 Overview", "📋 Detailed Data", "💸 Profit Drivers", "🔬 Rework & Variance",
        "🚧 Pipeline & Issues", "👷 Template & Install"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]: render_overview_tab(df_filtered)
    with tabs[1]: render_detailed_data_tab(df_filtered)
    with tabs[2]: render_profit_drivers_tab(df_filtered)
    with tabs[3]: render_rework_tab(df_filtered)
    with tabs[4]: render_pipeline_issues_tab(df_filtered)
    with tabs[5]: render_template_install_tab(df_filtered)

if __name__ == "__main__":
    main()
�
