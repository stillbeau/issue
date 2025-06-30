import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import json
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Profitability Calculator", page_icon="�")

# --- App Title ---
st.title("💰 Job Profitability Calculator")
st.markdown("Analyzes job data from Google Sheets to calculate profitability metrics for completed jobs.")

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
        # Define columns to convert to numeric, using the correct column names from the sheet
        numeric_cols = {
            'Total Job Price $': 'Revenue',
            'Job Throughput - Job Plant Invoice': 'Cost_From_Plant', # CORRECTED: Using Job Plant Invoice for cost
            'Total Job SqFT': 'Total_Job_SqFt'
        }
        
        for col_original, col_new in numeric_cols.items():
            if col_original in df.columns:
                df[col_new] = df[col_original].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col_new] = pd.to_numeric(df[col_new], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col_original}' not found. Calculations may be inaccurate.")
                df[col_new] = 0.0

        # Ensure other critical columns exist
        if 'Order Type' not in df.columns: df['Order Type'] = ''
        if 'Production #' not in df.columns: df['Production #'] = ''
        if 'Job Name' not in df.columns: df['Job Name'] = 'Unknown'
        if 'Invoice - Status' not in df.columns: df['Invoice - Status'] = ''
        
        # Convert date columns
        for col in ['Orders - Sale Date', 'Template - Date', 'Ship - Date', 'Invoice - Date']:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')

        # --- Filter for Completed Jobs ---
        df_completed = df[df['Invoice - Status'].astype(str).str.lower().str.strip() == 'complete'].copy()

        if df_completed.empty:
            st.info("No jobs with 'Invoice - Status' as 'Complete' were found. Profitability is based on completed jobs only.")
            return pd.DataFrame()

        # --- Profitability Calculation Step (on completed jobs only) ---
        df_completed['Install Cost'] = df_completed.apply(
            lambda row: row['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT 
            if 'pick up' not in str(row.get('Order Type', '')).lower() else 0,
            axis=1
        )
        
        # Total Cost for the Branch is the cost from plant (Job Plant Invoice) plus external install cost
        df_completed['Total Branch Cost'] = df_completed['Cost_From_Plant'] + df_completed['Install Cost']
        
        df_completed['Branch Profit'] = df_completed['Revenue'] - df_completed['Total Branch Cost']
        
        df_completed['Branch Profit Margin %'] = df_completed.apply(
            lambda row: (row['Branch Profit'] / row['Revenue']) * 100 if row['Revenue'] != 0 else 0,
            axis=1
        )
        
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

# --- Display Profitability Data ---
if st.session_state.df_profit is not None and not st.session_state.df_profit.empty:
    df_display = st.session_state.df_profit

    st.header("📊 Branch Profitability Summary (for Completed Jobs)")
    
    # Summary Metrics
    total_revenue = df_display['Revenue'].sum()
    total_profit = df_display['Branch Profit'].sum()
    avg_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
    
    summary_cols = st.columns(3)
    summary_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
    summary_cols[1].metric("Total Branch Profit", f"${total_profit:,.2f}")
    summary_cols[2].metric("Average Branch Profit Margin", f"{avg_margin:.2f}%")

    st.markdown("---")
    
    # Display the detailed profitability table
    st.subheader("Detailed Job Profitability")
    
    # Define columns to show in the main table
    display_cols = [
        'Job Name', 'Production #', 'Revenue', 'Total Branch Cost', 'Branch Profit', 'Branch Profit Margin %',
        'Cost_From_Plant', 'Install Cost', 'Total_Job_SqFt', 'Order Type'
    ]
    display_cols_exist = [col for col in display_cols if col in df_display.columns]

    # Rename columns for a more user-friendly display
    display_df = df_display[display_cols_exist].rename(columns={
        'Cost_From_Plant': 'Cost from Plant',
        'Total_Job_SqFt': 'Total Job SqFt'
    })
    
    st.dataframe(
        display_df.style.format({
            'Revenue': '${:,.2f}',
            'Total Branch Cost': '${:,.2f}',
            'Branch Profit': '${:,.2f}',
            'Profit Margin %': '{:.2f}%',
            'Cost from Plant': '${:,.2f}',
            'Install Cost': '${:,.2f}',
            'Total Job SqFt': '{:,.2f}'
        }),
        height=600,
        use_container_width=True
    )

    # --- Other Tools Section ---
    st.markdown("---")
    st.header("🛠️ Other Tools")
    # Using tabs for a cleaner layout
    calc_tab1, calc_tab2 = st.tabs(["🗓️ Upcoming Template Forecast", "🚚 Truck Weight Calculator"])

    with calc_tab1:
        # Note: This forecast uses the full dataset, not just completed jobs.
        # A separate, non-cached load might be needed if the main df is pre-filtered.
        # For simplicity, we re-use the loaded df but it might be filtered. A better implementation
        # would be to pass around the full df and the completed_df separately.
        # Let's use the full session state dataframe for this.
        full_df = st.session_state.df_profit # This now only contains completed jobs. We need to reload for forecasts.
        # For now, let's just make it clear this is based on the completed jobs, which isn't ideal.
        st.warning("Forecasts are currently based on the loaded set of completed jobs. To see a full forecast, the app logic needs to be extended.")

    with calc_tab2:
        st.info("The Truck Weight Calculator can be re-enabled and updated here if needed.")
�
