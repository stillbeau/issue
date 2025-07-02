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
st.markdown("Analyzes job data from Google Sheets to calculate profitability metrics.")

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
    """Loads, preprocesses, and calculates profitability for all job data."""
    if creds_dict is None:
        st.error("Google credentials not provided or invalid.")
        return None, None

    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        if not data:
            st.info(f"Worksheet '{worksheet_name}' is empty.")
            return pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(data)

        # --- Preprocessing Step ---
        numeric_cols = {
            'Total Job Price $': 'Revenue',
            'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
            'Total Job SqFT': 'Total_Job_SqFt',
            'Rework - Stone Shop - Rework Price': 'Rework_Price' # Using a temporary name
        }
        
        for col_original, col_new in numeric_cols.items():
            if col_original in df.columns:
                df[col_new] = df[col_original].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col_new] = pd.to_numeric(df[col_new], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col_original}' not found. Calculations may be inaccurate.")
                df[col_new] = 0.0

        critical_cols = ['Order Type', 'Production #', 'Job Name', 'Invoice - Status', 
                         'Salesperson', 'Customer Category', 'Rework - Stone Shop - Reason', 'Rework - Stone Shop - Bill To']
        for col in critical_cols:
            if col not in df.columns:
                df[col] = ''
        
        date_cols = ['Orders - Sale Date', 'Template - Date', 'Ship - Date', 'Invoice - Date', 'Job Creation', 'Ready to Fab - Date']
        for col in date_cols:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

        df['Job Link'] = MORAWARE_SEARCH_URL + df['Production #'].astype(str)
        
        # --- Profitability Calculation on ALL jobs ---
        df['Install Cost'] = df.apply(
            lambda row: row['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT 
            if 'pickup' not in str(row.get('Order Type', '')).lower().replace('-', '').replace(' ', '') else 0,
            axis=1
        )

        # Calculate Rework Cost conditionally
        df['Rework Cost'] = df.apply(
            lambda row: row['Rework_Price'] if str(row.get('Rework - Stone Shop - Bill To', '')).strip() == 'VER Branch - 14' else 0,
            axis=1
        )
        
        df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost'] + df['Rework Cost']
        df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
        
        df['Branch Profit Margin %'] = df.apply(
            lambda row: (row['Branch Profit'] / row['Revenue']) * 100 if row['Revenue'] != 0 else 0,
            axis=1
        )
        
        # --- Create a separate DataFrame for completed jobs for the main dashboard ---
        df_completed = df[df['Invoice - Status'].astype(str).str.lower().str.strip() == 'complete'].copy()

        if df_completed.empty:
            st.info("No jobs with 'Invoice - Status' as 'Complete' were found. Profitability dashboard will be based on 0 completed jobs.")
        
        return df, df_completed # Return BOTH full df and completed df

    except gspread.exceptions.GSpreadException as e:
        if "duplicate" in str(e).lower():
            st.error(f"Error: The header row in '{worksheet_name}' contains duplicate column names. Please ensure all headers are unique.")
        else:
            st.error(f"Error loading Google Sheet: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

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

if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'df_profit' not in st.session_state:
    st.session_state.df_profit = None

if final_creds:
    if st.sidebar.button("🔄 Load and Calculate Profitability"):
        with st.spinner("Loading and analyzing job data..."):
            st.session_state.df_full, st.session_state.df_profit = load_and_process_data(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
        
        if st.session_state.df_profit is not None:
            st.success(f"Successfully processed profitability for {len(st.session_state.df_profit)} completed jobs.")
        else:
            st.error("Failed to load or process data.")
else:
    st.info("Please configure your Google credentials in Streamlit Secrets or upload your JSON key file to begin.")

# --- Main Display Area ---
if st.session_state.df_profit is not None and st.session_state.df_full is not None:
    df_profit_display = st.session_state.df_profit
    df_full_display = st.session_state.df_full

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Use the completed jobs dataframe for filtering options
    df_for_filters = df_profit_display
    
    selected_salespersons = []
    if 'Salesperson' in df_for_filters.columns:
        salesperson_options = sorted(df_for_filters['Salesperson'].dropna().unique())
        selected_salespersons = st.sidebar.multiselect("Filter by Salesperson:", salesperson_options, default=salesperson_options)
    
    selected_categories = []
    if 'Customer Category' in df_for_filters.columns:
        category_options = sorted(df_for_filters['Customer Category'].dropna().unique())
        selected_categories = st.sidebar.multiselect("Filter by Customer Category:", category_options, default=category_options)

    # Apply filters
    df_filtered = df_profit_display.copy()
    if selected_salespersons:
        df_filtered = df_filtered[df_filtered['Salesperson'].isin(selected_salespersons)]
    if selected_categories:
        df_filtered = df_filtered[df_filtered['Customer Category'].isin(selected_categories)]

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Dashboard", "📋 Detailed Profitability Data", "� Rework Analysis", "🛠️ Forecasts & Tools"])

    with tab1:
        st.header("📈 Overall Performance Dashboard")
        if not df_filtered.empty:
            total_revenue = df_filtered['Revenue'].sum()
            total_profit = df_filtered['Branch Profit'].sum()
            avg_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
            
            summary_cols = st.columns(3)
            summary_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
            summary_cols[1].metric("Total Branch Profit", f"${total_profit:,.2f}")
            summary_cols[2].metric("Average Profit Margin", f"{avg_margin:.2f}%")
            st.markdown("---")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Profit by Salesperson")
                if 'Salesperson' in df_filtered.columns:
                    st.bar_chart(df_filtered.groupby('Salesperson')['Branch Profit'].sum().sort_values(ascending=False))
            with chart_cols[1]:
                st.subheader("Revenue by Customer Category")
                if 'Customer Category' in df_filtered.columns:
                    st.bar_chart(df_filtered.groupby('Customer Category')['Revenue'].sum().sort_values(ascending=False))
            
        else:
            st.warning("No data matches the current filter selection.")

    with tab2:
        st.header("📋 Detailed Job Profitability")
        display_cols = [
            'Production #', 'Job Link', 'Job Name', 'Revenue', 'Total Branch Cost', 'Branch Profit', 'Branch Profit Margin %',
            'Cost_From_Plant', 'Install Cost', 'Rework Cost', 'Total_Job_SqFt', 'Order Type', 'Salesperson', 'Customer Category'
        ]
        display_cols_exist = [col for col in display_cols if col in df_filtered.columns]
        display_df = df_filtered[display_cols_exist].rename(columns={
            'Cost_From_Plant': 'Cost from Plant', 'Total_Job_SqFt': 'Total Job SqFt'
        })
        st.dataframe(display_df, column_config={"Job Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗")}, use_container_width=True)

    with tab3:
        st.header("🔬 Rework Analysis")
        rework_cols_exist = 'Rework Cost' in df_full_display.columns and \
                            'Rework - Stone Shop - Bill To' in df_full_display.columns and \
                            'Job Creation' in df_full_display.columns and \
                            pd.api.types.is_datetime64_any_dtype(df_full_display['Job Creation'])

        if rework_cols_exist:
            rework_df = df_full_display[df_full_display['Rework Cost'] > 0].copy()
            if not rework_df.empty:
                rework_df['Month'] = rework_df['Job Creation'].dt.to_period('M')
                rework_summary = rework_df.groupby(['Month', 'Rework - Stone Shop - Bill To'])['Rework Cost'].agg(['sum', 'count']).reset_index()
                rework_summary = rework_summary.rename(columns={'sum': 'Total Rework Cost', 'count': 'Number of Jobs'})
                rework_summary['Month'] = rework_summary['Month'].astype(str)
                st.dataframe(rework_summary.sort_values(by=['Month', 'Total Rework Cost'], ascending=[False, False]).style.format({'Total Rework Cost': '${:,.2f}'}), use_container_width=True)
            else:
                st.info("No jobs with rework costs were found in the dataset.")
        else:
            st.warning("Could not generate rework analysis. Required columns are missing or have incorrect data types (Rework Cost, Rework - Stone Shop - Bill To, Job Creation).")


    with tab4:
        st.header("🛠️ Forecasts & Tools")
        forecast_tab1, forecast_tab2 = st.tabs(["🗓️ Upcoming Template Forecast", "🏭 Production Forecast"])

        with forecast_tab1:
            if 'Template - Date' in df_full_display.columns:
                future_templates_df = df_full_display[df_full_display['Template - Date'] > datetime.now()].copy()
                if not future_templates_df.empty:
                    st.write("**Weekly Template Forecast**")
                    future_templates_df['Week Start'] = future_templates_df['Template - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                    weekly_summary = future_templates_df.groupby('Week Start').agg(Jobs=('Job Name', 'count'), SqFt=('Total_Job_SqFt', 'sum'), Value=('Revenue', 'sum'), Profit=('Branch Profit', 'sum')).reset_index()
                    weekly_summary['Margin %'] = weekly_summary.apply(lambda row: (row['Profit'] / row['Value']) * 100 if row['Value'] != 0 else 0, axis=1)
                    st.dataframe(weekly_summary.style.format({'SqFt': '{:,.2f}', 'Value': '${:,.2f}', 'Profit': '${:,.2f}', 'Margin %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("No upcoming templates found in the data.")
            else:
                st.warning("'Template - Date' column not found.")

        with forecast_tab2:
            if 'Ready to Fab - Date' in df_full_display.columns:
                st.subheader("Recent Jobs Sent to Production")
                recent_rtf_df = df_full_display[df_full_display['Ready to Fab - Date'].notna()].sort_values(by='Ready to Fab - Date', ascending=False)
                st.dataframe(recent_rtf_df[['Job Name', 'Production #', 'Ready to Fab - Date', 'Total_Job_SqFt', 'Revenue']].head(15).style.format({'Total_Job_SqFt': '{:,.2f}', 'Revenue': '${:,.2f}'}), use_container_width=True)

                st.subheader("Weekly Production Forecast (by RTF Date)")
                rtf_df = df_full_display[df_full_display['Ready to Fab - Date'].notna()].copy()
                rtf_df['Week Start'] = rtf_df['Ready to Fab - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                weekly_rtf_summary = rtf_df.groupby('Week Start').agg(Jobs=('Job Name', 'count'), SqFt=('Total_Job_SqFt', 'sum'), Value=('Revenue', 'sum'), Profit=('Branch Profit', 'sum')).reset_index().sort_values(by='Week Start', ascending=False)
                weekly_rtf_summary['Margin %'] = weekly_rtf_summary.apply(lambda row: (row['Profit'] / row['Value']) * 100 if row['Value'] != 0 else 0, axis=1)
                st.dataframe(weekly_rtf_summary.style.format({'SqFt': '{:,.2f}', 'Value': '${:,.2f}', 'Profit': '${:,.2f}', 'Margin %': '{:.2f}%'}), use_container_width=True)
            else:
                st.warning("'Ready to Fab - Date' column not found.")
�
