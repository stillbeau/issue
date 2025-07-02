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
        numeric_cols = {
            'Total Job Price $': 'Revenue',
            'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
            'Total Job SqFT': 'Total_Job_SqFt',
            'Job Throughput - Rework COGS': 'Rework_COGS',
            'Job Throughput - Rework Job Labor': 'Rework_Labor',
            'Job Throughput - Job GM (original)': 'Original_GM'
        }
        
        for col_original, col_new in numeric_cols.items():
            if col_original in df.columns:
                df[col_new] = df[col_original].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col_new] = pd.to_numeric(df[col_new], errors='coerce').fillna(0)
            else:
                st.warning(f"Required column '{col_original}' not found. Calculations may be inaccurate.")
                df[col_new] = 0.0

        critical_cols = ['Order Type', 'Production #', 'Job Name', 'Invoice - Status', 
                         'Salesperson', 'Customer Category', 'Rework - Stone Shop - Reason', 
                         'Job Material', 'City']
        for col in critical_cols:
            if col not in df.columns:
                df[col] = ''
        
        date_cols = ['Orders - Sale Date', 'Template - Date', 'Ship - Date', 'Invoice - Date', 'Ready to Fab - Date']
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
        
        df['Total Rework Cost'] = df['Rework_COGS'] + df['Rework_Labor']
        df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost'] + df['Total Rework Cost']
        df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
        
        df['Branch Profit Margin %'] = df.apply(
            lambda row: (row['Branch Profit'] / row['Revenue']) * 100 if row['Revenue'] != 0 else 0,
            axis=1
        )
        
        df['Profit Variance'] = df['Branch Profit'] - df['Original_GM']
        
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

if 'df_full' not in st.session_state:
    st.session_state.df_full = None

if final_creds:
    if st.sidebar.button("🔄 Load and Calculate Profitability"):
        with st.spinner("Loading and analyzing job data..."):
            st.session_state.df_full = load_and_process_data(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
        
        if st.session_state.df_full is not None:
            st.success(f"Successfully processed profitability for {len(st.session_state.df_full)} jobs.")
        else:
            st.error("Failed to load or process data.")
else:
    st.info("Please configure your Google credentials in Streamlit Secrets or upload your JSON key file to begin.")

# --- Main Display Area ---
if st.session_state.df_full is not None and not st.session_state.df_full.empty:
    df_full = st.session_state.df_full

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Dashboard", "📋 Detailed Profitability Data", "� Rework & Variance Analysis", "🛠️ Forecasts"])

    with tab1:
        st.header("📈 Overall Performance Dashboard")
        
        # Filter for completed jobs for summary metrics
        df_completed = df_full[df_full['Invoice - Status'].astype(str).str.lower().str.strip() == 'complete'].copy()

        if not df_completed.empty:
            total_revenue = df_completed['Revenue'].sum()
            total_profit = df_completed['Branch Profit'].sum()
            avg_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
            
            st.subheader("Summary for Completed Jobs")
            summary_cols = st.columns(3)
            summary_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
            summary_cols[1].metric("Total Branch Profit", f"${total_profit:,.2f}")
            summary_cols[2].metric("Average Profit Margin", f"{avg_margin:.2f}%")
            st.markdown("---")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Profit by Salesperson")
                if 'Salesperson' in df_completed.columns:
                    st.bar_chart(df_completed.groupby('Salesperson')['Branch Profit'].sum().sort_values(ascending=False))
            with chart_cols[1]:
                st.subheader("Profit by Job Material")
                if 'Job Material' in df_completed.columns:
                    st.bar_chart(df_completed.groupby('Job Material')['Branch Profit'].sum().sort_values(ascending=False))
        else:
            st.warning("No completed jobs found to display summary statistics.")

    with tab2:
        st.header("📋 Detailed Job Profitability")
        display_cols = [
            'Production #', 'Job Link', 'Job Name', 'Revenue', 'Total Branch Cost', 'Branch Profit', 'Branch Profit Margin %', 'Profit Variance',
            'Cost_From_Plant', 'Install Cost', 'Total Rework Cost', 'Total_Job_SqFt', 'Order Type', 'Salesperson', 'Customer Category'
        ]
        display_cols_exist = [col for col in display_cols if col in df_full.columns]
        display_df = df_full[display_cols_exist].rename(columns={
            'Cost_From_Plant': 'Cost from Plant', 'Total_Job_SqFt': 'Total Job SqFt'
        })
        st.dataframe(display_df, column_config={"Job Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗")}, use_container_width=True)

    with tab3:
        st.header("🔬 Rework & Variance Analysis")
        
        st.subheader("Rework Cost Breakdown by Reason")
        if 'Rework_Cost' in df_full.columns and 'Rework - Stone Shop - Reason' in df_full.columns:
            rework_df = df_full[df_full['Rework_Cost'] > 0]
            if not rework_df.empty:
                rework_summary = rework_df.groupby('Rework - Stone Shop - Reason')['Rework_Cost'].agg(['sum', 'count']).reset_index().rename(columns={'sum': 'Total Rework Cost', 'count': 'Number of Jobs'})
                st.dataframe(rework_summary.style.format({'Total Rework Cost': '${:,.2f}'}), use_container_width=True)
            else:
                st.info("No rework costs recorded for the selected jobs.")
        else:
            st.info("Rework data not available for analysis.")

        st.markdown("---")
        st.subheader("Profit Variance Analysis")
        variance_df = df_full.copy()
        if 'Profit Variance' in variance_df.columns:
            variance_df['Abs_Variance'] = variance_df['Profit Variance'].abs()
            
            st.write("**Top 10 Jobs with Negative Variance (Underperformed Estimate)**")
            st.dataframe(
                variance_df.sort_values(by='Profit Variance', ascending=True).head(10)[['Job Name', 'Production #', 'Original_GM', 'Branch Profit', 'Profit Variance']],
                column_config={'Original_GM': st.column_config.NumberColumn(format='$%.2f'), 'Branch Profit': st.column_config.NumberColumn(format='$%.2f'), 'Profit Variance': st.column_config.NumberColumn(format='$%.2f')}
            )

            st.write("**Top 10 Jobs with Positive Variance (Overperformed Estimate)**")
            st.dataframe(
                variance_df.sort_values(by='Profit Variance', ascending=False).head(10)[['Job Name', 'Production #', 'Original_GM', 'Branch Profit', 'Profit Variance']],
                column_config={'Original_GM': st.column_config.NumberColumn(format='$%.2f'), 'Branch Profit': st.column_config.NumberColumn(format='$%.2f'), 'Profit Variance': st.column_config.NumberColumn(format='$%.2f')}
            )
        else:
            st.info("Profit Variance data not available for analysis.")

    with tab4:
        st.header("🛠️ Forecasts & Tools")
        forecast_tab1, forecast_tab2 = st.tabs(["🗓️ Upcoming Template Forecast", "🏭 Production Forecast"])

        with forecast_tab1:
            if 'Template - Date' in df_full.columns:
                future_templates_df = df_full[df_full['Template - Date'] > datetime.now()].copy()
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
            if 'Ready to Fab - Date' in df_full.columns:
                st.subheader("Recent Jobs Sent to Production")
                recent_rtf_df = df_full[df_full['Ready to Fab - Date'].notna()].sort_values(by='Ready to Fab - Date', ascending=False)
                st.dataframe(recent_rtf_df[['Job Name', 'Production #', 'Ready to Fab - Date', 'Total_Job_SqFt', 'Revenue']].head(15).style.format({'Total_Job_SqFt': '{:,.2f}', 'Revenue': '${:,.2f}'}), use_container_width=True)

                st.subheader("Weekly Production Forecast (by RTF Date)")
                rtf_df = df_full[df_full['Ready to Fab - Date'].notna()].copy()
                rtf_df['Week Start'] = rtf_df['Ready to Fab - Date'].dt.to_period('W').apply(lambda p: p.start_time).dt.date
                weekly_rtf_summary = rtf_df.groupby('Week Start').agg(Jobs=('Job Name', 'count'), SqFt=('Total_Job_SqFt', 'sum'), Value=('Revenue', 'sum'), Profit=('Branch Profit', 'sum')).reset_index().sort_values(by='Week Start', ascending=False)
                weekly_rtf_summary['Margin %'] = weekly_rtf_summary.apply(lambda row: (row['Profit'] / row['Value']) * 100 if row['Value'] != 0 else 0, axis=1)
                st.dataframe(weekly_rtf_summary.style.format({'SqFt': '{:,.2f}', 'Value': '${:,.2f}', 'Profit': '${:,.2f}', 'Margin %': '{:.2f}%'}), use_container_width=True)
            else:
                st.warning("'Ready to Fab - Date' column not found.")
