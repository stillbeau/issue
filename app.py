import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import json
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Profitability Calculator", page_icon="💰")

# --- App Title ---
st.title("💰 Job Profitability Calculator")
st.markdown("Analyzes job data from Google Sheets to calculate profitability metrics for the branch.")

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
        # Define columns to convert to numeric, using the new "Job Throughput" and "Orders" names
        numeric_cols = {
            'Orders - Total Price': 'Revenue',
            'Job Throughput - Total COGS': 'Cost_From_Plant',
            'Job Throughput - Job SqFt': 'Total_Job_SqFt'
        }
        
        for col_original, col_new in numeric_cols.items():
            if col_original in df.columns:
                # Convert column to string to handle various formats before cleaning
                df[col_new] = df[col_original].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col_new] = pd.to_numeric(df[col_new], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col_original}' not found. Calculations may be inaccurate.")
                df[col_new] = 0.0

        # Ensure other critical columns exist
        if 'Order Type' not in df.columns: df['Order Type'] = ''
        if 'Production #' not in df.columns: df['Production #'] = ''
        if 'Job Name' not in df.columns: df['Job Name'] = 'Unknown'
        
        # Convert date columns
        for col in ['Orders - Sale Date', 'Template - Date', 'Ship - Date']:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')


        # --- Profitability Calculation Step ---
        # Calculate Install Cost based on Order Type
        df['Install Cost'] = df.apply(
            lambda row: row['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT 
            if 'pick up' not in str(row.get('Order Type', '')).lower() else 0,
            axis=1
        )
        
        # Calculate Total Cost for the Branch
        df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost']
        
        # Calculate Branch Profit
        df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
        
        # Calculate Branch Profit Margin %
        df['Branch Profit Margin %'] = df.apply(
            lambda row: (row['Branch Profit'] / row['Revenue']) * 100 if row['Revenue'] != 0 else 0,
            axis=1
        )
        
        return df

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
            st.success(f"Successfully loaded and processed {len(st.session_state.df_profit)} jobs.")
        else:
            st.error("Failed to load or process data.")
else:
    st.info("Please configure your Google credentials in Streamlit Secrets or upload your JSON key file to begin.")

# --- Display Profitability Data ---
if st.session_state.df_profit is not None and not st.session_state.df_profit.empty:
    df_display = st.session_state.df_profit

    st.header("📊 Branch Profitability Summary")
    
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
        'Cost_From_Plant': 'Cost from Plant (COGS)',
        'Total_Job_SqFt': 'Total Job SqFt'
    })
    
    st.dataframe(
        display_df.style.format({
            'Revenue': '${:,.2f}',
            'Total Branch Cost': '${:,.2f}',
            'Branch Profit': '${:,.2f}',
            'Branch Profit Margin %': '{:.2f}%',
            'Cost from Plant (COGS)': '${:,.2f}',
            'Install Cost': '${:,.2f}',
            'Total Job SqFt': '{:,.2f}'
        }),
        height=600,
        use_container_width=True
    )

    # --- Calculators Section ---
    st.markdown("---")
    st.header("🛠️ Forecasts & Calculators")
    calc_tab1, calc_tab2 = st.tabs(["🗓️ Upcoming Template Forecast", "🚚 Truck Weight Calculator"])

    with calc_tab1:
        if 'Template - Date' in df_display.columns:
            future_templates_df = df_display[df_display['Template - Date'] > datetime.now()].copy()
            
            if not future_templates_df.empty:
                st.subheader("Weekly Template Forecast")
                future_templates_df['Week Start'] = future_templates_df['Template - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                weekly_summary = future_templates_df.groupby('Week Start').agg(
                    Total_Jobs=('Job Name', 'count'),
                    Total_SqFt=('Total_Job_SqFt', 'sum'),
                    Total_Value=('Revenue', 'sum')
                ).reset_index()
                st.dataframe(weekly_summary.style.format({'Total_SqFt': '{:,.2f}', 'Total_Value': '${:,.2f}'}))
                
                st.subheader("Monthly Template Forecast")
                future_templates_df['Month'] = future_templates_df['Template - Date'].dt.to_period('M')
                monthly_summary = future_templates_df.groupby('Month').agg(
                    Total_Jobs=('Job Name', 'count'),
                    Total_SqFt=('Total_Job_SqFt', 'sum'),
                    Total_Value=('Revenue', 'sum')
                ).reset_index()
                monthly_summary['Month'] = monthly_summary['Month'].astype(str)
                st.dataframe(monthly_summary.style.format({'Total_SqFt': '{:,.2f}', 'Total_Value': '${:,.2f}'}))
            else:
                st.info("No upcoming templates found in the data.")
        else:
            st.warning("'Template - Date' column not found, cannot generate forecast.")

    with calc_tab2:
        st.info("The Truck Weight Calculator can be re-enabled and updated here if needed.")
