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
        # Define columns to convert to numeric, handling currency symbols and commas
        numeric_cols = ['Total Price', 'Total COGS', 'Total Job Labor', 'Total Job SqFT']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col}' not found. Calculations may be inaccurate.")
                df[col] = 0.0 # Add column with default value to prevent errors

        # Ensure other critical columns exist
        if 'Order Type' not in df.columns: df['Order Type'] = ''
        if 'Production #' not in df.columns: df['Production #'] = ''
        if 'Job Name' not in df.columns: df['Job Name'] = 'Unknown'
        if 'Ship - Date' not in df.columns: df['Ship - Date'] = None
        if 'Template - Date' not in df.columns: df['Template - Date'] = None
        
        # Convert date columns
        for col in ['Ship - Date', 'Template - Date']:
             if col in df.columns:
                 df[col] = pd.to_datetime(df[col], errors='coerce')


        # --- Profitability Calculation Step ---
        # Calculate Install Cost based on Order Type
        df['Install Cost'] = df.apply(
            lambda row: row['Total Job SqFT'] * INSTALL_COST_PER_SQFT 
            if row['Order Type'].lower().strip() != 'pick up' else 0,
            axis=1
        )
        
        # Calculate Total Cost
        df['Total Cost'] = df['Total COGS'] + df['Total Job Labor'] + df['Install Cost']
        
        # Calculate Profit
        df['Profit'] = df['Total Price'] - df['Total Cost']
        
        # Calculate Profit Margin %
        # Avoid division by zero
        df['Profit Margin %'] = df.apply(
            lambda row: (row['Profit'] / row['Total Price']) * 100 if row['Total Price'] != 0 else 0,
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

    st.header("📊 Profitability Summary")
    
    # Summary Metrics
    total_revenue = df_display['Total Price'].sum()
    total_profit = df_display['Profit'].sum()
    avg_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
    
    summary_cols = st.columns(3)
    summary_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
    summary_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
    summary_cols[2].metric("Average Profit Margin", f"{avg_margin:.2f}%")

    st.markdown("---")
    
    # Display the detailed profitability table
    st.subheader("Detailed Job Profitability")
    
    # Define columns to show in the main table
    display_cols = [
        'Job Name', 'Production #', 'Total Price', 'Total Cost', 'Profit', 'Profit Margin %',
        'Total COGS', 'Total Job Labor', 'Install Cost', 'Total Job SqFT', 'Order Type'
    ]
    # Filter to only include columns that actually exist in the dataframe
    display_cols = [col for col in display_cols if col in df_display.columns]
    
    st.dataframe(
        df_display[display_cols].style.format({
            'Total Price': '${:,.2f}',
            'Total Cost': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Profit Margin %': '{:.2f}%',
            'Total COGS': '${:,.2f}',
            'Total Job Labor': '${:,.2f}',
            'Install Cost': '${:,.2f}',
            'Total Job SqFT': '{:,.2f}'
        }),
        height=600,
        use_container_width=True
    )

    # --- Calculators Section ---
    st.markdown("---")
    st.header("🛠️ Calculators & Forecasts")
    calc_tab1, calc_tab2 = st.tabs(["🚚 Truck Weight Calculator", "🗓️ Upcoming Template Forecast"])

    with calc_tab1:
        selected_date_for_week = st.date_input("Select any date within the desired ship week:")
        if selected_date_for_week:
            start_of_week = selected_date_for_week - timedelta(days=selected_date_for_week.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            st.info(f"Calculating for week: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}")

            if 'Ship - Date' in df_display.columns:
                jobs_for_week = df_display[
                    (df_display['Ship - Date'].dt.date >= start_of_week) &
                    (df_display['Ship - Date'].dt.date <= end_of_week)
                ]
                
                if not jobs_for_week.empty:
                    stone_jobs = jobs_for_week[jobs_for_week['Production #'].str.startswith('09')]
                    laminate_jobs = jobs_for_week[jobs_for_week['Production #'].str.startswith('04')]
                    
                    st.subheader("Stone Jobs on Truck")
                    stone_sqft = stone_jobs['Total Job SqFT'].sum()
                    stone_weight = stone_sqft * LBS_PER_SQFT
                    stone_install_cost = stone_jobs[stone_jobs['Order Type'].str.lower().strip() != 'pick up']['Total Job SqFT'].sum() * INSTALL_COST_PER_SQFT
                    
                    calc_cols_stone = st.columns(4)
                    calc_cols_stone[0].metric("Stone Jobs", len(stone_jobs))
                    calc_cols_stone[1].metric("Stone SqFt", f"{stone_sqft:,.2f} sqft")
                    calc_cols_stone[2].metric("Estimated Stone Weight", f"{stone_weight:,.2f} lbs")
                    calc_cols_stone[3].metric("Estimated Install Cost", f"${stone_install_cost:,.2f}")
                    
                    st.subheader("Laminate Jobs on Truck")
                    laminate_sqft = laminate_jobs['Total Job SqFT'].sum()
                    laminate_install_cost = laminate_jobs[laminate_jobs['Order Type'].str.lower().strip() != 'pick up']['Total Job SqFT'].sum() * INSTALL_COST_PER_SQFT
                    
                    calc_cols_lam = st.columns(3)
                    calc_cols_lam[0].metric("Laminate Jobs", len(laminate_jobs))
                    calc_cols_lam[1].metric("Laminate SqFt", f"{laminate_sqft:,.2f} sqft")
                    calc_cols_lam[2].metric("Estimated Install Cost", f"${laminate_install_cost:,.2f}")

                else:
                    st.info(f"No jobs found for the selected week.")
            else:
                st.warning("'Ship - Date' column not found in the data.")

    with calc_tab2:
        if 'Template - Date' in df_display.columns:
            future_templates_df = df_display[df_display['Template - Date'] > datetime.now()].copy()
            
            if not future_templates_df.empty:
                st.subheader("Weekly Template Forecast")
                future_templates_df['Week Start'] = future_templates_df['Template - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                weekly_summary = future_templates_df.groupby('Week Start').agg(
                    Total_Jobs=('Job Name', 'count'),
                    Total_SqFt=('Total Job SqFT', 'sum'),
                    Total_Value=('Total Price', 'sum')
                ).reset_index()
                st.dataframe(weekly_summary.style.format({'Total_SqFt': '{:,.2f}', 'Total_Value': '${:,.2f}'}))
                
                st.subheader("Monthly Template Forecast")
                future_templates_df['Month'] = future_templates_df['Template - Date'].dt.to_period('M')
                monthly_summary = future_templates_df.groupby('Month').agg(
                    Total_Jobs=('Job Name', 'count'),
                    Total_SqFt=('Total Job SqFT', 'sum'),
                    Total_Value=('Total Price', 'sum')
                ).reset_index()
                monthly_summary['Month'] = monthly_summary['Month'].astype(str)
                st.dataframe(monthly_summary.style.format({'Total_SqFt': '{:,.2f}', 'Total_Value': '${:,.2f}'}))
            else:
                st.info("No upcoming templates found in the data.")
        else:
            st.warning("'Template - Date' column not found, cannot generate forecast.")
