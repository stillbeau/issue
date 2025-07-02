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
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="💰")

# --- Constants & Global Configuration ---
# It's good practice to group constants for easy modification.
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="
INSTALL_COST_PER_SQFT = 15.0


# --- Helper Functions ---

def parse_material(s: str) -> tuple[str, str]:
    """
    Parses a material description string to extract brand and color.

    Args:
        s: The material string, e.g., "- ,123 - BrandName (ColorName)".

    Returns:
        A tuple containing the extracted brand and color.
    """
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

def get_lat_lon(city_name: str) -> tuple[float, float]:
    """
    Geocodes a city name to latitude and longitude.
    
    Note: This is a placeholder. For a real application, you would use a
    geocoding service like Geopy with Nominatim or the Google Maps API.
    """
    # Placeholder coordinates for cities in British Columbia and Alberta
    city_coords = {
        "vernon": (50.267, -119.272),
        "kelowna": (49.888, -119.496),
        "penticton": (49.492, -119.593),
        "kamloops": (50.676, -120.341),
        "calgary": (51.044, -114.071),
        "edmonton": (53.546, -113.493),
        "red deer": (52.269, -113.811)
    }
    return city_coords.get(str(city_name).lower(), (0, 0))


# --- Data Loading and Processing ---

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict) -> pd.DataFrame:
    """
    Loads data from Google Sheets, cleans, and processes it for analysis.

    This function is cached to avoid reloading data on every interaction.

    Args:
        creds_dict: A dictionary containing Google service account credentials.

    Returns:
        A processed pandas DataFrame with calculated metrics.
    """
    # --- Authenticate and Fetch Data ---
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(worksheet.get_all_records())

    # --- Data Cleaning and Transformation ---

    # 1. Rename and clean numeric columns
    numeric_column_map = {
        'Total Job Price $': 'Revenue',
        'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
        'Job Throughput - Rework COGS': 'Rework_COGS',
        'Job Throughput - Rework Job Labor': 'Rework_Labor',
        'Rework - Stone Shop - Rework Price': 'Rework_Price',
        'Job Throughput - Job GM (original)': 'Original_GM',
        'Total Job SqFT': 'Total_Job_SqFt'
    }
    for original_name, new_name in numeric_column_map.items():
        if original_name in df.columns:
            # Ensure column exists before processing
            cleaned_series = df[original_name].astype(str).str.replace(r'[$,]', '', regex=True)
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            df[new_name] = numeric_series.fillna(0)
        else:
            # If the source column doesn't exist, create a zero-filled column
            df[new_name] = 0.0

    # 2. Parse date columns
    date_cols = ['Template - Date', 'Ready to Fab - Date', 'Ship - Date', 'Install - Date', 'Job Creation']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Calculate profitability metrics
    df['Install Cost'] = df['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT
    df['Total Rework Cost'] = df.get('Rework_COGS', 0) + df.get('Rework_Labor', 0) + df.get('Rework_Price', 0)
    df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost'] + df['Total Rework Cost']
    df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
    df['Branch Profit Margin %'] = df.apply(
        lambda row: (row['Branch Profit'] / row['Revenue'] * 100) if row.get('Revenue') and row['Revenue'] != 0 else 0,
        axis=1
    )
    df['Profit Variance'] = df['Branch Profit'] - df.get('Original_GM', 0)

    # 4. Parse material information
    if 'Job Material' in df.columns:
        df[['Material Brand', 'Material Color']] = df['Job Material'].apply(
            lambda x: pd.Series(parse_material(str(x)))
        )
    else:
        df['Material Brand'] = "N/A"
        df['Material Color'] = "N/A"

    # 5. Calculate stage durations
    if 'Ready to Fab - Date' in df.columns and 'Template - Date' in df.columns:
        df['Days_Template_to_RTF'] = (df['Ready to Fab - Date'] - df['Template - Date']).dt.days
    if 'Ship - Date' in df.columns and 'Ready to Fab - Date' in df.columns:
        df['Days_RTF_to_Ship'] = (df['Ship - Date'] - df['Ready to Fab - Date']).dt.days
    if 'Install - Date' in df.columns and 'Ship - Date' in df.columns:
        df['Days_Ship_to_Install'] = (df['Install - Date'] - df['Ship - Date']).dt.days

    # 6. Create dynamic job link
    if 'Production #' in df.columns:
        df['Job Link'] = MORAWARE_SEARCH_URL + df['Production #'].astype(str)

    # 7. Geocode city data
    if 'City' in df.columns:
        df[['lat', 'lon']] = df['City'].apply(lambda x: pd.Series(get_lat_lon(str(x))))

    return df


# --- UI Rendering Functions ---

def render_overview_tab(df: pd.DataFrame):
    """Renders the 'Overall' performance tab."""
    st.header("📈 Overall Performance")
    total_revenue = df['Revenue'].sum()
    total_profit = df['Branch Profit'].sum()
    avg_margin = (total_profit / total_revenue * 100) if total_revenue else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Profit", f"${total_profit:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

    st.markdown("---")
    st.subheader("Profit by Salesperson")
    if 'Salesperson' in df.columns:
        st.bar_chart(df.groupby('Salesperson')['Branch Profit'].sum())

    st.subheader("Material Brand Leaderboard")
    if 'Material Brand' in df.columns:
        mat_leaderboard = df.groupby('Material Brand').agg(
            Total_Profit=('Branch Profit', 'sum'),
            Avg_Margin=('Branch Profit Margin %', 'mean')
        ).sort_values('Total_Profit', ascending=False)
        st.dataframe(mat_leaderboard.style.format({'Total_Profit': '${:,.2f}', 'Avg_Margin': '{:.2f}%'}))

    st.subheader("Low Profit Alerts")
    threshold = st.number_input("Show jobs with margin below (%)", -100.0, 100.0, 10.0, 1.0)
    low_profit_jobs = df[df['Branch Profit Margin %'] < threshold]
    if not low_profit_jobs.empty:
        st.dataframe(low_profit_jobs[['Production #', 'Job Name', 'Branch Profit Margin %', 'Branch Profit']])
    else:
        st.success("No low-profit jobs found for the current selection.")


def render_detailed_data_tab(df: pd.DataFrame):
    """Renders the 'Detailed Data' tab with a searchable table."""
    st.header("📋 Detailed Data View")
    display_cols = [
        'Production #', 'Job Link', 'Job Name', 'Revenue', 
        'Total_Job_SqFt', 'Cost_From_Plant', 'Install Cost', 'Total Branch Cost',
        'Branch Profit', 'Branch Profit Margin %', 'Profit Variance'
    ]
    # Ensure we only try to display columns that actually exist
    df_display = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(
        df_display,
        use_container_width=True,
        column_config={
            "Job Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗"),
            "Revenue": st.column_config.NumberColumn(format='$%.2f'),
            "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
            "Cost_From_Plant": st.column_config.NumberColumn("Production Cost", format='$%.2f'),
            "Install Cost": st.column_config.NumberColumn(format='$%.2f'),
            "Total Branch Cost": st.column_config.NumberColumn(format='$%.2f'),
            "Branch Profit": st.column_config.NumberColumn(format='$%.2f'),
            "Profit Variance": st.column_config.NumberColumn(format='$%.2f'),
            "Branch Profit Margin %": st.column_config.ProgressColumn(
                "Profit Margin", format='%.2f%%', min_value=-100, max_value=100
            ),
        }
    )


def render_rework_tab(df: pd.DataFrame):
    """Renders the 'Rework & Variance' tab."""
    st.header("🔬 Rework Insights & Profit Variance")
    if 'Total Rework Cost' in df and 'Rework - Stone Shop - Reason' in df.columns:
        rework_jobs = df[df['Total Rework Cost'] > 0]
        if not rework_jobs.empty:
            st.subheader("Rework Costs by Reason")
            agg_rework = rework_jobs.groupby('Rework - Stone Shop - Reason')['Total Rework Cost'].agg(['sum', 'count'])
            agg_rework.columns = ['Total Rework Cost', 'Number of Jobs']
            st.dataframe(agg_rework.sort_values('Total Rework Cost', ascending=False).style.format({'Total Rework Cost': '${:,.2f}'}))
        else:
            st.info("No rework costs were recorded for the current selection.")
    else:
        st.warning("Required rework columns ('Total Rework Cost', 'Rework - Stone Shop - Reason') not found in data.")

    st.subheader("Jobs with Negative Profit (Losses)")
    loss_jobs = df[df['Branch Profit'] < 0]
    if not loss_jobs.empty:
        st.dataframe(loss_jobs[['Production #', 'Job Name', 'Branch Profit', 'Total Rework Cost']])
    else:
        st.success("No jobs resulted in a loss for the current selection.")


def render_stage_durations_tab(df: pd.DataFrame):
    """Renders the 'Stage Durations' tab."""
    st.header("⏱️ Stage Durations Analysis")
    duration_cols = {
        'Template → Ready-to-Fab': 'Days_Template_to_RTF',
        'Ready-to-Fab → Ship': 'Days_RTF_to_Ship',
        'Ship → Install': 'Days_Ship_to_Install'
    }
    
    avg_durations = {}
    for friendly_name, actual_col in duration_cols.items():
        if actual_col in df.columns:
            avg_durations[friendly_name] = df[actual_col].mean()

    st.write("**Average Days in Each Stage**")
    st.json({k: f"{v:.1f}" if pd.notna(v) else "N/A" for k, v in avg_durations.items()})

    st.subheader("Duration Distributions")
    if MATPLOTLIB_AVAILABLE:
        for friendly_name, actual_col in duration_cols.items():
            if actual_col in df.columns and pd.notna(df[actual_col].mean()):
                fig, ax = plt.subplots()
                df[actual_col].dropna().hist(bins=20, ax=ax)
                ax.set_title(f"Distribution of: {friendly_name}")
                ax.set_xlabel("Days")
                ax.set_ylabel("Number of Jobs")
                st.pyplot(fig)
    else:
        st.warning("Duration charts require 'matplotlib'. Please install it.")


def render_forecast_tab(df: pd.DataFrame):
    """Reworks the 'Forecasts & Trends' tab."""
    st.header("📅 Forecasts & Trends")
    if 'Template - Date' in df.columns and not df['Template - Date'].dropna().empty:
        df_trends = df.copy()
        df_trends['Template - Date'] = pd.to_datetime(df_trends['Template - Date'], errors='coerce')
        df_trends = df_trends.dropna(subset=['Template - Date'])

        if not df_trends.empty:
            ts_data = df_trends.set_index('Template - Date').resample('M').agg(
                Revenue=('Revenue', 'sum'),
                Jobs=('Production #', 'count')
            )
            st.subheader("Monthly Trend: Revenue & Job Count")
            st.line_chart(ts_data)

            st.subheader("Simple Linear Revenue Forecast (Next 3 Months)")
            if SKLEARN_AVAILABLE:
                # Use last 6 months for trend projection
                recent_revenue = ts_data.tail(6)['Revenue'].reset_index(drop=True)
                if len(recent_revenue) > 1:
                    model = LinearRegression()
                    X = recent_revenue.index.values.reshape(-1, 1)
                    y = recent_revenue.values
                    model.fit(X, y)

                    future_indices = pd.DataFrame({'x': range(len(recent_revenue), len(recent_revenue) + 3)})
                    predictions = model.predict(future_indices[['x']])
                    
                    forecast_dates = pd.date_range(start=ts_data.index[-1] + relativedelta(months=1), periods=3, freq='M')
                    forecast_series = pd.Series(predictions, index=forecast_dates, name="Forecasted Revenue")
                    st.line_chart(forecast_series, height=200)
                else:
                    st.info("Not enough monthly data points to generate a forecast.")
            else:
                st.warning("Forecasting requires 'scikit-learn'. Please install it.")
        else:
            st.info("No valid date data available for trend analysis in the current selection.")
    else:
        st.warning("'Template - Date' column is required for trend analysis.")

def render_geo_analysis_tab(df: pd.DataFrame):
    """Renders the 'Geospatial Analysis' tab."""
    st.header("🗺️ Geospatial Analysis")
    if 'City' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
        st.subheader("Profit by City")
        
        city_profit = df.groupby('City')['Branch Profit'].sum().sort_values(ascending=False)
        st.bar_chart(city_profit)

        st.subheader("Geographic Profit Distribution")
        # Filter out cities with no coordinates
        map_data = df[df['lat'] != 0].copy()
        if not map_data.empty:
            st.map(map_data[['lat', 'lon']])
        else:
            st.warning("No valid coordinates found for the selected cities.")
    else:
        st.warning("City and coordinate data are required for this tab.")

def render_template_install_tab(df: pd.DataFrame):
    """Renders the weekly forecast for Template and Install activities."""
    st.header("📅 Weekly Template & Install Forecast")

    # --- Template Section ---
    st.subheader("Templates")
    if 'Template - Date' in df.columns and 'Template - Assigned To' in df.columns:
        template_df = df.dropna(subset=['Template - Date', 'Template - Assigned To']).copy()
        if not template_df.empty:
            # Weekly Forecast
            template_weekly = template_df.set_index('Template - Date').resample('W-Mon', label='left', closed='left').agg(
                Jobs=('Production #', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).reset_index()
            st.write("**Weekly Template Volume**")
            st.dataframe(template_weekly)

            # Assigned To Breakdown
            st.write("**Template Workload by Person (for selected period)**")
            template_assigned = template_df.groupby('Template - Assigned To').agg(
                Jobs=('Production #', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).sort_values(by='Jobs', ascending=False)
            st.dataframe(template_assigned)
        else:
            st.info("No template data available for the current selection.")
    else:
        st.warning("Template data columns ('Template - Date', 'Template - Assigned To') not found.")

    st.markdown("---")

    # --- Install Section ---
    st.subheader("Installs")
    if 'Install - Date' in df.columns and 'Install - Assigned To' in df.columns:
        install_df = df.dropna(subset=['Install - Date', 'Install - Assigned To']).copy()
        if not install_df.empty:
            # Weekly Forecast
            install_weekly = install_df.set_index('Install - Date').resample('W-Mon', label='left', closed='left').agg(
                Jobs=('Production #', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).reset_index()
            st.write("**Weekly Install Volume**")
            st.dataframe(install_weekly)

            # Assigned To Breakdown
            st.write("**Install Workload by Person (for selected period)**")
            install_assigned = install_df.groupby('Install - Assigned To').agg(
                Jobs=('Production #', 'count'),
                SqFt=('Total_Job_SqFt', 'sum')
            ).sort_values(by='Jobs', ascending=False)
            st.dataframe(install_assigned)
        else:
            st.info("No install data available for the current selection.")
    else:
        st.warning("Install data columns ('Install - Date', 'Install - Assigned To') not found.")


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.title("💰 Job Profitability Dashboard")
    st.markdown("Analyzes job data from Google Sheets to calculate profitability metrics.")

    # --- Credentials & Initial Load ---
    st.sidebar.header("⚙️ Configuration")
    creds = None
    # Try loading credentials from Streamlit secrets first
    if "google_creds_json" in st.secrets:
        creds = json.loads(st.secrets["google_creds_json"])
    else:
        # Fallback to file uploader if secrets are not available
        uploaded_file = st.sidebar.file_uploader("Upload Google Service Account JSON", type="json")
        if uploaded_file:
            creds = json.load(uploaded_file)

    if not creds:
        st.sidebar.error("Please provide Google credentials to load data.")
        st.info("Please upload your service account credentials in the sidebar to begin.")
        st.stop()

    try:
        df_full = load_and_process_data(creds)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.stop()


    # --- Sidebar Filters ---
    st.sidebar.header("📊 Filters")

    # 1. Date-range filter
    if 'Job Creation' in df_full.columns and not df_full['Job Creation'].dropna().empty:
        min_date = df_full['Job Creation'].min().date()
        max_date = df_full['Job Creation'].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Filter by Job Creation Date",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        # Filter the DataFrame based on the selected date range
        df_filtered = df_full[
            (df_full['Job Creation'].dt.date >= start_date) &
            (df_full['Job Creation'].dt.date <= end_date)
        ].copy()
    else:
        st.sidebar.warning("'Job Creation' column not found or is empty. Cannot apply date filter.")
        df_filtered = df_full.copy()

    # 2. Multiselect filters
    def get_unique_options(col_name):
        return sorted(df_filtered[col_name].dropna().unique()) if col_name in df_filtered else []

    salesperson_options = get_unique_options('Salesperson')
    category_options = get_unique_options('Customer Category')
    material_options = get_unique_options('Material Brand')
    city_options = get_unique_options('City')

    selected_salespeople = st.sidebar.multiselect("Salesperson", salesperson_options, default=salesperson_options)
    selected_categories = st.sidebar.multiselect("Customer Category", category_options, default=category_options)
    selected_materials = st.sidebar.multiselect("Material Brand", material_options, default=material_options)
    selected_cities = st.sidebar.multiselect("City", city_options, default=city_options)

    # Apply multiselect filters sequentially
    if selected_salespeople: df_filtered = df_filtered[df_filtered['Salesperson'].isin(selected_salespeople)]
    if selected_categories: df_filtered = df_filtered[df_filtered['Customer Category'].isin(selected_categories)]
    if selected_materials: df_filtered = df_filtered[df_filtered['Material Brand'].isin(selected_materials)]
    if selected_cities: df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]


    # --- Main Panel with Tabs ---
    if df_filtered.empty:
        st.warning("No data matches the current filter selection.")
        st.stop()

    tab_names = [
        "📈 Overview",
        "📋 Detailed Data",
        "🔬 Rework & Variance",
        "⏱️ Stage Durations",
        "📅 Forecasts & Trends",
        "🗺️ Geospatial Analysis",
        "👷 Template & Install"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview_tab(df_filtered)

    with tabs[1]:
        render_detailed_data_tab(df_filtered)

    with tabs[2]:
        render_rework_tab(df_filtered)

    with tabs[3]:
        render_stage_durations_tab(df_filtered)
        
    with tabs[4]:
        render_forecast_tab(df_filtered)

    with tabs[5]:
        render_geo_analysis_tab(df_filtered)
        
    with tabs[6]:
        render_template_install_tab(df_filtered)


if __name__ == "__main__":
    main()
