import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import json
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="💰")

# --- App Title ---
st.title("💰 Job Profitability Dashboard")
st.markdown("Analyzes job data from Google Sheets to provide profitability insights.")

# --- Constants & Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
DATA_WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="
INSTALL_COST_PER_SQFT = 15.0
LBS_PER_SQFT = 20.0

# --- Helper Functions ---

def get_google_creds():
    """Loads Google credentials from Streamlit secrets or file uploader."""
    if "google_creds_json" in st.secrets:
        try:
            return json.loads(st.secrets["google_creds_json"])
        except json.JSONDecodeError:
            st.sidebar.error("Error parsing Google credentials from Streamlit Secrets.")
            return None
    return None

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict, spreadsheet_id, worksheet_name):
    """Loads, preprocesses, and calculates profitability for the job data."""
    if creds_dict is None:
        st.error("Google credentials not provided or invalid.")
        return None

    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        if not data:
            st.info(f"Worksheet '{worksheet_name}' is empty.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)

        # --- Preprocessing Step ---
        # Define columns to convert to numeric
        numeric_cols = {
            'Total Job Price $': 'Revenue',
            'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
            'Total Job SqFT': 'Total_Job_SqFt',
            'Rework - Stone Shop - Rework Price': 'Rework_Cost' # New Rework Cost column
        }
        
        for col_original, col_new in numeric_cols.items():
            if col_original in df.columns:
                df[col_new] = df[col_original].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col_new] = pd.to_numeric(df[col_new], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col_original}' not found. Calculations may be inaccurate.")
                df[col_new] = 0.0

        # Ensure other critical columns exist
        critical_cols = ['Order Type', 'Production #', 'Job Name', 'Invoice - Status', 
                         'Salesperson', 'Customer Category', 'Rework - Stone Shop - Reason']
        for col in critical_cols:
            if col not in df.columns:
                df[col] = ''
        
        # Convert date columns
        for col in ['Orders - Sale Date', 'Template - Date', 'Ship - Date', 'Invoice - Date']:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

        # --- Filter for Completed Jobs ---
        df_completed = df[df['Invoice - Status'].astype(str).str.lower().str.strip() == 'complete'].copy()

        if df_completed.empty:
            st.info("No jobs with 'Invoice - Status' as 'Complete' were found. Profitability is based on completed jobs only.")
            return pd.DataFrame()

        # --- Profitability Calculation Step ---
        df_completed['Install Cost'] = df_completed.apply(
            lambda row: row['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT 
            if 'pickup' not in str(row.get('Order Type', '')).lower().replace('-', '').replace(' ', '') else 0,
            axis=1
        )
        
        df_completed['Total Branch Cost'] = df_completed['Cost_From_Plant'] + df_completed['Install Cost'] + df_completed['Rework_Cost']
        df_completed['Branch Profit'] = df_completed['Revenue'] - df_completed['Total Branch Cost']
        
        df_completed['Branch Profit Margin %'] = df_completed.apply(
            lambda row: (row['Branch Profit'] / row['Revenue']) * 100 if row['Revenue'] != 0 else 0,
            axis=1
        )

        df_completed['Job Link'] = MORAWARE_SEARCH_URL + df_completed['Production #'].astype(str)
        
        return df_completed

    except gspread.exceptions.GSpreadException as e:
        if "duplicate" in str(e).lower():
            st.error(f"Error: The header row in '{worksheet_name}' contains duplicate column names. Please ensure all headers are unique.")
        else:
            st.error(f"Error loading Google Sheet: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Main App UI ---

st.sidebar.header("⚙️ Configuration")
final_creds = get_google_creds()

if not final_creds:
    st.sidebar.subheader("Google Sheets Credentials")
    st.sidebar.markdown("Upload your Google Cloud Service Account JSON key file.")
    uploaded_file = st.sidebar.file_uploader("Upload Service Account JSON", type="json")
    if uploaded_file:
        try:
            final_creds = json.load(StringIO(uploaded_file.getvalue().decode("utf-8")))
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")

if 'df_profit' not in st.session_state:
    st.session_state.df_profit = None

if final_creds:
    if st.sidebar.button("🔄 Load and Calculate Profitability"):
        with st.spinner("Loading and analyzing job data..."):
            st.session_state.df_profit = load_and_process_data(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
        if st.session_state.df_profit is not None:
            st.success(f"Successfully processed profitability for {len(st.session_state.df_profit)} completed jobs.")
        else:
            st.error("Failed to load or process data.")
else:
    st.info("Please configure your Google credentials in Streamlit Secrets or upload your JSON key file to begin.")

# --- Main Display Area ---
if st.session_state.df_profit is not None and not st.session_state.df_profit.empty:
    df_full = st.session_state.df_profit

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Salesperson Filter
    salesperson_options = sorted(df_full['Salesperson'].dropna().unique())
    selected_salespersons = st.sidebar.multiselect("Filter by Salesperson:", salesperson_options, default=salesperson_options)

    # Customer Category Filter
    category_options = sorted(df_full['Customer Category'].dropna().unique())
    selected_categories = st.sidebar.multiselect("Filter by Customer Category:", category_options, default=category_options)

    # Apply filters
    df_filtered = df_full[
        df_full['Salesperson'].isin(selected_salespersons) &
        df_full['Customer Category'].isin(selected_categories)
    ]

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Overall Dashboard", "📋 Detailed Profitability Data", "🛠️ Forecasts & Tools"])

    with tab1:
        st.header("📈 Overall Performance Dashboard")
        if not df_filtered.empty:
            # Summary Metrics
            total_revenue = df_filtered['Revenue'].sum()
            total_profit = df_filtered['Branch Profit'].sum()
            avg_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
            
            summary_cols = st.columns(3)
            summary_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
            summary_cols[1].metric("Total Branch Profit", f"${total_profit:,.2f}")
            summary_cols[2].metric("Average Profit Margin", f"{avg_margin:.2f}%")

            st.markdown("---")

            # Charts
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                st.subheader("Profit by Salesperson")
                profit_by_salesperson = df_filtered.groupby('Salesperson')['Branch Profit'].sum().sort_values(ascending=False)
                st.bar_chart(profit_by_salesperson)

            with chart_cols[1]:
                st.subheader("Revenue by Customer Category")
                revenue_by_category = df_filtered.groupby('Customer Category')['Revenue'].sum().sort_values(ascending=False)
                st.bar_chart(revenue_by_category)
            
            st.markdown("---")
            
            st.subheader("Rework Cost Analysis")
            rework_df = df_filtered[df_filtered['Rework_Cost'] > 0]
            if not rework_df.empty:
                rework_summary = rework_df.groupby('Rework - Stone Shop - Reason')['Rework_Cost'].agg(['sum', 'count']).reset_index()
                rework_summary = rework_summary.rename(columns={'sum': 'Total Rework Cost', 'count': 'Number of Jobs'})
                st.dataframe(rework_summary.style.format({'Total Rework Cost': '${:,.2f}'}), use_container_width=True)
            else:
                st.info("No rework costs recorded for the selected jobs.")

        else:
            st.warning("No data matches the current filter selection.")

    with tab2:
        st.header("📋 Detailed Job Profitability")
        display_cols = [
            'Production #', 'Job Link', 'Job Name', 'Revenue', 'Total Branch Cost', 'Branch Profit', 'Branch Profit Margin %',
            'Cost_From_Plant', 'Install Cost', 'Rework_Cost', 'Total_Job_SqFt', 'Order Type', 'Salesperson', 'Customer Category'
        ]
        display_cols_exist = [col for col in display_cols if col in df_filtered.columns]
        display_df = df_filtered[display_cols_exist].rename(columns={
            'Cost_From_Plant': 'Cost from Plant', 'Total_Job_SqFt': 'Total Job SqFt', 'Rework_Cost': 'Rework Cost'
        })
        st.dataframe(display_df, column_config={"Job Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗")}, use_container_width=True)


    with tab3:
        st.header("🛠️ Forecasts & Tools")
        
        st.subheader("🗓️ Upcoming Template Forecast")
        if 'Template - Date' in df_full.columns:
            future_templates_df = df_full[df_full['Template - Date'] > datetime.now()].copy()
            if not future_templates_df.empty:
                st.write("**Weekly Forecast**")
                future_templates_df['Week Start'] = future_templates_df['Template - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                weekly_summary = future_templates_df.groupby('Week Start').agg(Jobs=('Job Name', 'count'), SqFt=('Total_Job_SqFt', 'sum'), Value=('Revenue', 'sum')).reset_index()
                st.dataframe(weekly_summary.style.format({'SqFt': '{:,.2f}', 'Value': '${:,.2f}'}), use_container_width=True)
                
                st.write("**Monthly Forecast**")
                future_templates_df['Month'] = future_templates_df['Template - Date'].dt.to_period('M')
                monthly_summary = future_templates_df.groupby('Month').agg(Jobs=('Job Name', 'count'), SqFt=('Total_Job_SqFt', 'sum'), Value=('Revenue', 'sum')).reset_index()
                monthly_summary['Month'] = monthly_summary['Month'].astype(str)
                st.dataframe(monthly_summary.style.format({'SqFt': '{:,.2f}', 'Value': '${:,.2f}'}), use_container_width=True)
            else:
                st.info("No upcoming templates found in the data.")
        else:
            st.warning("'Template - Date' column not found, cannot generate forecast.")

        st.markdown("---")

        st.subheader("🚚 Truck Weight Calculator")
        st.info("The Truck Weight Calculator can be re-enabled and updated here if needed based on the new data structure.")
