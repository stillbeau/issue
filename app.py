import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from io import StringIO
import json # For parsing JSON from secrets or uploaded file
import math # For pagination
from datetime import datetime # For timestamping notes

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Issue Detector", page_icon="⚙️")

# --- App Title ---
st.title("⚙️ Job Issue Detector Dashboard")
st.markdown("Analyzes job data from Google Sheets to identify issues and create a prioritized 'todo' list.")

# --- Constants & Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38" # YOUR GOOGLE SHEET ID
DATA_WORKSHEET_NAME = "jobs" # Your main data sheet
TODO_WORKSHEET_NAME = "todo" # Sheet for the priority list
ACTION_LOG_WORKSHEET_NAME = "notes" # New sheet for logging actions
ITEMS_PER_PAGE = 10 # For pagination

# --- Helper Functions (Authentication, Data Loading, Processing) ---

def get_google_creds():
    if "google_creds_json" in st.secrets:
        try:
            creds_str = st.secrets["google_creds_json"]
            creds_dict = json.loads(creds_str)
            return creds_dict
        except json.JSONDecodeError:
            st.sidebar.error("Error parsing Google credentials from Streamlit Secrets. Ensure it's valid JSON.")
            return None
        except Exception as e:
            st.sidebar.error(f"Error with Streamlit Secrets: {e}")
            return None
    return None 

@st.cache_data(ttl=300) # Cache data for 5 minutes
def load_google_sheet(creds_dict, spreadsheet_id, worksheet_name):
    if creds_dict is None:
        st.error("Google credentials not provided or invalid.")
        return None, None, None 
    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records() 
        if not data:
            st.info(f"Worksheet '{worksheet_name}' appears to be empty or has no data rows after headers.")
            return pd.DataFrame(), spreadsheet, gc 
        df = pd.DataFrame(data)
        critical_cols = ['Next Sched. - Activity', 'Next Sched. - Date', 'Next Sched. - Status', 
                         'Install - Date', 'Supplied By', 'Production #', 'Salesperson'] 
        for col in critical_cols:
            if col not in df.columns:
                df[col] = "" 
        return df, spreadsheet, gc 
    except gspread.exceptions.GSpreadException as e: 
        if "duplicate" in str(e).lower():
             st.error(f"Error loading Google Sheet '{worksheet_name}': The header row in the worksheet contains duplicate column names. Please ensure all column headers in the first row of your sheet are unique. Details: {e}")
        else:
             st.error(f"Error loading Google Sheet '{worksheet_name}': {e}")
        return None, None, None
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Error: Worksheet '{worksheet_name}' not found in the Google Sheet.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading Google Sheet '{worksheet_name}': {e}")
        return None, None, None

def preprocess_data(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df_processed = df.copy()
    date_columns_to_convert = [
        'Next Sched. - Date', 'Job Creation', 'Template - Date', 'Photo Layout - Date',
        'Ready to Fab - Date', 'Rework - Date', 'Cutlist - Date', 'Program - Date',
        'Material Pull - Date', 'Saw - Date', 'CNC - Date', 
        'Polish/Fab Completion - Date', 'Hone Splash - Date', 'QC - Date',
        'Ship - Date', 'Product Rcvd - Date', 'Repair - Date', 
        'Delivery - Date', 'Install - Date', 'Pick Up - Date', 
        'Service - Date', 'Callback - Date', 'Invoice - Date', 
        'Build Up - Date', 'Tearout - Date', 'Lift Help - Date', 'Courier - Date',
        'Tile Order - Date', 'Tile Install - Date',
        'Collect Final - Date', 'Follow Up Call - Date'
    ]
    unique_date_columns = []
    for col in date_columns_to_convert:
        if col not in unique_date_columns:
            unique_date_columns.append(col)
    date_columns_to_convert = unique_date_columns

    common_non_date_placeholders = ['', 'None', 'none', 'NAN', 'NaN', 'nan', 'NA', 'NaT', 'nat', 'Pending', 'TBD', 'No Date', '#N/A', 'NULL', 'null']

    for col_name in date_columns_to_convert:
        if col_name in df_processed.columns:
            df_processed[col_name] = df_processed[col_name].astype(str).str.strip()
            for placeholder in common_non_date_placeholders:
                df_processed[col_name] = df_processed[col_name].replace(placeholder, None, regex=False) 
            df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce')
    return df_processed

def flag_threshold_delays(df, current_date_str):
    if df is None or df.empty: return pd.DataFrame()
    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)

    df_flagged['Flag_Awaiting_RTF'] = False
    if 'Template - Date' in df_flagged.columns and 'Ready to Fab - Date' in df_flagged.columns:
        condition_rtf_pending = df_flagged['Template - Date'].notna() & \
                                df_flagged['Ready to Fab - Date'].isna() & \
                                pd.api.types.is_datetime64_any_dtype(df_flagged['Template - Date'])
        if condition_rtf_pending.any(): 
            df_flagged.loc[condition_rtf_pending, 'Flag_Awaiting_RTF'] = \
                (current_date - df_flagged.loc[condition_rtf_pending, 'Template - Date']).dt.days > 2

    df_flagged['Flag_Awaiting_Template'] = False
    if 'Job Creation' in df_flagged.columns and 'Template - Date' in df_flagged.columns:
        condition_template_pending = df_flagged['Job Creation'].notna() & \
                                     df_flagged['Template - Date'].isna() & \
                                     pd.api.types.is_datetime64_any_dtype(df_flagged['Job Creation'])
        if condition_template_pending.any():
            df_flagged.loc[condition_template_pending, 'Flag_Awaiting_Template'] = \
                (current_date - df_flagged.loc[condition_template_pending, 'Job Creation']).dt.days > 2
    
    df_flagged['Flag_Awaiting_Cutlist'] = False
    if 'Ready to Fab - Date' in df_flagged.columns and \
       'Cutlist - Date' in df_flagged.columns and \
       'Supplied By' in df_flagged.columns: 
        is_not_laminate = ~df_flagged['Supplied By'].astype(str).str.strip().eq('ABB PF - 12')
        rtf_date_exists = df_flagged['Ready to Fab - Date'].notna()
        cutlist_date_missing = df_flagged['Cutlist - Date'].isna()
        is_rtf_date_datetime = pd.api.types.is_datetime64_any_dtype(df_flagged['Ready to Fab - Date'])
        valid_rows_for_cutlist_flag = rtf_date_exists & cutlist_date_missing & is_rtf_date_datetime & is_not_laminate
        if valid_rows_for_cutlist_flag.any():
            df_flagged.loc[valid_rows_for_cutlist_flag, 'Flag_Awaiting_Cutlist'] = \
                (current_date - df_flagged.loc[valid_rows_for_cutlist_flag, 'Ready to Fab - Date']).dt.days > 3

    threshold_flag_cols = ['Flag_Awaiting_RTF', 'Flag_Awaiting_Template', 'Flag_Awaiting_Cutlist']
    threshold_flag_cols = [col for col in threshold_flag_cols if col in df_flagged.columns]
    df_flagged['Flag_Any_Threshold_Delay'] = df_flagged[threshold_flag_cols].any(axis=1) if threshold_flag_cols else False
    return df_flagged

def flag_keyword_issues(df):
    if df is None or df.empty: return pd.DataFrame()
    df_flagged = df.copy()
    keywords_to_flag = [
        'must be', 'urgent', 'rush', 'asap', 'delay', 'delayed', 'leaving town',
        'only time', 'call when', 'call on way', 'not home until', 'confirm time', 'schedule',
        'broke', 'broken', 'missing', 'not ready', 'issue', 'problem', 'concern',
        'damage', 'damaged', 'rework', 'remake', 'incorrect', 'error', 'hold',
        'leak', 'leaking', 'scratch', 'chipped',
        'crane', 'heavy', '3rd man', 'third man',
        'change', 'changed', 'revision', 'adjust', 'adjustment', 're-template', 'retemplate',
        'add', 'added', 'remove', 'removed', 'new drawing', 'cad adjustment',
        'confirm edge', 'confirm sink', 'confirm color', 'choose', 'pick',
        'stock sink', 'no sink onsite', 'sink not onsite'
    ]
    escaped_keywords = [re.escape(kw) for kw in keywords_to_flag]
    keyword_pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'

    def find_keywords_in_note(note_text, pattern):
        if pd.isna(note_text) or not isinstance(note_text, str): return False
        return bool(re.search(pattern, note_text, re.IGNORECASE))

    notes_columns_to_scan = [ 
        'Next Sched. - Notes', 'Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes', 
        'Ready to Fab - Notes', 'Rework - Notes', 'Cutlist - Notes', 'Program - Notes', 'Material Pull - Notes', 
        'CNC - Notes', 'Polish/Fab Completion - Notes', 'Hone Splash - Notes', 'Ship - Notes', 'Product Rcvd - Notes',
        'Repair - Notes', 'Delivery - Notes', 'Pick Up - Notes', 'Service - Notes', 'Callback - Notes', 'Invoice - Notes', 
        'Build Up - Notes', 'Tearout - Notes', 'Lift Help - Notes', 'Courier - Notes', 'Tile Order - Notes', 
        'Tile Install - Notes', 'Collect Final - Notes', 'Follow Up Call - Notes', 'Address Notes'
    ]
    actual_notes_cols_to_scan = [col for col in notes_columns_to_scan if col in df_flagged.columns]

    keyword_flag_columns_generated = []
    for col_name in actual_notes_cols_to_scan:
        clean_col_name_for_flag = re.sub(r'[^A-Za-z0-9_]+', '_', col_name) 
        new_flag_col_name = f'Flag_Keyword_In_{clean_col_name_for_flag}'
        df_flagged[new_flag_col_name] = df_flagged[col_name].apply(lambda note: find_keywords_in_note(note, keyword_pattern))
        keyword_flag_columns_generated.append(new_flag_col_name)
            
    df_flagged['Flag_Any_Keyword_Issue'] = df_flagged[keyword_flag_columns_generated].any(axis=1) if keyword_flag_columns_generated else False
    return df_flagged

def flag_past_due_activities(df, current_date_str):
    if df is None or df.empty: return pd.DataFrame()
    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)
    completion_terms = ['complete', 'completed', 'done', 'installed', 'invoiced', 'paid', 'sent', 'received', 'closed', 'fabricated']
    cancellation_terms = ['cancelled', 'canceled', 'void', 'voided']
    
    activities_to_check_past_due = [
        ('Next_Sched_Activity', 'Next Sched. - Date', 'Next Sched. - Status'),
        ('Template', 'Template - Date', 'Template - Status'),
        ('RTF', 'Ready_to_Fab', 'Ready to Fab - Date', 'Ready to Fab - Status'), 
        ('Install', 'Install - Date', 'Install - Status'),
        ('Invoice', 'Invoice - Date', 'Invoice - Status'),
        ('Collect_Final', 'Collect Final - Date', 'Collect Final - Status'),
        ('Saw', 'Saw - Date', 'Saw - Status'),
        ('Polish_Fab_Completion', 'Polish/Fab Completion - Date', 'Polish/Fab Completion - Status'),
        ('Cutlist', 'Cutlist - Date', 'Cutlist - Status'),
        ('Program', 'Program - Date', 'Program - Status'),
        ('QC', 'QC - Date', 'QC - Status'),
        ('Delivery', 'Delivery - Date', 'Delivery - Status'),
        ('Service', 'Service - Date', 'Service - Status'),
    ]
    past_due_flag_columns_generated = []

    for activity_tuple in activities_to_check_past_due:
        activity_name_for_flag, date_col, status_col = activity_tuple[0], activity_tuple[1], activity_tuple[2]
        if activity_tuple[0] == 'RTF': 
            activity_name_for_flag = activity_tuple[1] 

        new_flag_col = f'Flag_PastDue_{activity_name_for_flag}'
        df_flagged[new_flag_col] = False 

        if not (date_col in df_flagged.columns and \
                status_col in df_flagged.columns and \
                pd.api.types.is_datetime64_any_dtype(df_flagged[date_col])):
            continue 
        
        valid_date_rows = df_flagged[date_col].notna()
        if not valid_date_rows.any(): 
            continue

        condition_date_past = pd.Series(False, index=df_flagged.index)
        if valid_date_rows.sum() > 0: 
            condition_date_past.loc[valid_date_rows] = (df_flagged.loc[valid_date_rows, date_col] < current_date)
        
        status_cleaned = df_flagged[status_col].fillna('').astype(str).str.lower().str.strip()
        condition_not_complete = ~status_cleaned.isin(completion_terms)
        condition_not_cancelled = ~status_cleaned.isin(cancellation_terms)
        final_condition = condition_date_past & condition_not_complete & condition_not_cancelled
        
        df_flagged.loc[final_condition, new_flag_col] = True
        past_due_flag_columns_generated.append(new_flag_col)

    df_flagged['Flag_Any_PastDue_Activity'] = df_flagged[past_due_flag_columns_generated].any(axis=1) if past_due_flag_columns_generated else False
    return df_flagged

def determine_primary_issue_and_days(row, current_calc_date_ts):
    if not isinstance(current_calc_date_ts, pd.Timestamp):
        current_calc_date_ts = pd.Timestamp(current_calc_date_ts)

    date_cols_map = {
        'Flag_PastDue_Next_Sched_Activity': 'Next Sched. - Date',
        'Flag_PastDue_Install': 'Install - Date',
        'Flag_PastDue_Polish_Fab_Completion': 'Polish/Fab Completion - Date',
        'Flag_PastDue_Saw': 'Saw - Date',
        'Flag_PastDue_Ready_to_Fab': 'Ready to Fab - Date',
        'Flag_PastDue_Invoice': 'Invoice - Date',
        'Flag_PastDue_Collect_Final': 'Collect Final - Date',
        'Flag_PastDue_Template': 'Template - Date',
        'Flag_PastDue_Cutlist': 'Cutlist - Date',
        'Flag_PastDue_Program': 'Program - Date',
        'Flag_PastDue_QC': 'QC - Date',
        'Flag_PastDue_Delivery': 'Delivery - Date',
        'Flag_PastDue_Service': 'Service - Date',
        'Flag_Awaiting_Cutlist': 'Ready to Fab - Date', 
        'Flag_Awaiting_RTF': 'Template - Date',       
        'Flag_Awaiting_Template': 'Job Creation'      
    }
    issue_desc_map = {
        'Flag_PastDue_Next_Sched_Activity': "Past Due: Next Sched. Activity", 
        'Flag_PastDue_Install': "Past Due: Install",
        'Flag_PastDue_Polish_Fab_Completion': "Past Due: Polish/Fab",
        'Flag_PastDue_Saw': "Past Due: Saw",
        'Flag_PastDue_Ready_to_Fab': "Past Due: RTF", 
        'Flag_PastDue_Invoice': "Past Due: Invoice",
        'Flag_PastDue_Collect_Final': "Past Due: Collect Final",
        'Flag_PastDue_Template': "Past Due: Template",
        'Flag_PastDue_Cutlist': "Past Due: Cutlist",
        'Flag_PastDue_Program': "Past Due: Program",
        'Flag_PastDue_QC': "Past Due: QC",
        'Flag_PastDue_Delivery': "Past Due: Delivery",
        'Flag_PastDue_Service': "Past Due: Service",
        'Flag_Awaiting_Cutlist': "Delay: Awaiting Cutlist",
        'Flag_Awaiting_RTF': "Delay: Awaiting RTF",
        'Flag_Awaiting_Template': "Delay: Awaiting Template",
        'Flag_Keyword_In_Next_Sched_Notes': "Keyword: Next Sched. Notes", 
        'Flag_Keyword_In_Install_Notes': "Keyword: Install Notes",
        'Flag_Keyword_In_Template_Notes': "Keyword: Template Notes",
        'Flag_Keyword_In_Job_Issues': "Keyword: Job Issues", 
        'Flag_Keyword_In_QC_Notes': "Keyword: QC Notes",
        'Flag_Keyword_In_Saw_Notes': "Keyword: Saw Notes"
    }
    flag_check_order = [ 
        'Flag_PastDue_Install', 'Flag_PastDue_Next_Sched_Activity', 
        'Flag_PastDue_Polish_Fab_Completion', 'Flag_PastDue_Saw', 
        'Flag_PastDue_Ready_to_Fab', 'Flag_PastDue_Invoice', 'Flag_PastDue_Collect_Final', 
        'Flag_PastDue_Template', 'Flag_PastDue_Cutlist', 'Flag_PastDue_Program', 
        'Flag_PastDue_QC', 'Flag_PastDue_Delivery', 'Flag_PastDue_Service',
        'Flag_Awaiting_Cutlist', 'Flag_Awaiting_RTF', 'Flag_Awaiting_Template', 
        'Flag_Keyword_In_Install_Notes', 
        'Flag_Keyword_In_Template_Notes',
        'Flag_Keyword_In_Next_Sched_Notes', 
        'Flag_Keyword_In_Job_Issues', 
        'Flag_Keyword_In_QC_Notes', 
        'Flag_Keyword_In_Saw_Notes'
    ]

    for flag_col in flag_check_order:
        if row.get(flag_col, False): 
            issue_description = ""
            if flag_col == 'Flag_PastDue_Next_Sched_Activity':
                next_activity_text = str(row.get('Next Sched. - Activity', "")).strip()
                issue_description = f"Past Due: {next_activity_text}" if next_activity_text else "Past Due: Next Sched. Activity"
            else:
                issue_description = issue_desc_map.get(flag_col, f"Unknown Issue ({flag_col})")
            
            days_behind = "N/A"
            date_col_for_days = date_cols_map.get(flag_col)
            
            if date_col_for_days and \
               date_col_for_days in row and \
               pd.notna(row[date_col_for_days]) and \
               isinstance(row[date_col_for_days], pd.Timestamp):
                try:
                    days_behind = (current_calc_date_ts - row[date_col_for_days]).days
                except TypeError: 
                    days_behind = "Error Calc Days" 
            return issue_description, days_behind
            
    return "Other Issue", "N/A" 

def append_action_log(spreadsheet_obj, worksheet_name, log_entry_dict):
    """Appends a log entry to the specified worksheet."""
    if spreadsheet_obj is None:
        st.warning("Spreadsheet object not available. Cannot write action log.")
        return False
    try:
        try:
            log_sheet = spreadsheet_obj.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            st.info(f"Action log worksheet '{worksheet_name}' not found. Creating it...")
            # Define headers for the new action log sheet
            action_log_headers = ["Timestamp", "Job Name", "Production #", "Action", "Details", "Assigned To"]
            log_sheet = spreadsheet_obj.add_worksheet(title=worksheet_name, rows=100, cols=len(action_log_headers) + 5)
            log_sheet.append_row(action_log_headers) # Add header row
            st.info(f"Worksheet '{worksheet_name}' created with headers.")

        # Prepare row data, ensuring all header columns are present even if value is empty
        action_log_headers = log_sheet.row_values(1) # Get headers to ensure order
        if not action_log_headers : # If sheet was just created and append_row hasn't flushed
             action_log_headers = ["Timestamp", "Job Name", "Production #", "Action", "Details", "Assigned To"]


        row_to_append = [log_entry_dict.get(header, "") for header in action_log_headers]
        log_sheet.append_row(row_to_append, value_input_option='USER_ENTERED')
        # st.toast(f"Action logged to '{worksheet_name}'.") # Use toast for less intrusive message
        return True
    except Exception as e:
        st.error(f"Error writing action log to '{worksheet_name}': {e}")
        return False


def write_to_google_sheet(spreadsheet_obj, worksheet_name, df_to_write):
    if spreadsheet_obj is None or df_to_write is None:
        st.warning("Spreadsheet object or DataFrame to write is missing. Cannot write to Google Sheet.")
        return False
    try:
        worksheet = spreadsheet_obj.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        st.info(f"Worksheet '{worksheet_name}' not found. Creating it...")
        num_cols = len(df_to_write.columns) if not df_to_write.empty else 20 
        worksheet = spreadsheet_obj.add_worksheet(title=worksheet_name, rows=max(100, len(df_to_write) + 5), cols=num_cols + 5)
    
    worksheet.clear()
    if df_to_write.empty:
        st.info(f"No data to write to '{worksheet_name}'. Sheet cleared.")
        return True

    df_export = df_to_write.fillna('').astype(str) 
    export_values = [df_export.columns.values.tolist()] + df_export.values.tolist()
    worksheet.update(export_values, value_input_option='USER_ENTERED')
    st.success(f"Successfully wrote {len(df_to_write)} rows to worksheet: '{worksheet_name}'")
    return True
    
# --- Main App Logic ---
st.sidebar.header("⚙️ Configuration")
creds_from_secrets = get_google_creds()
uploaded_creds_dict = None

if not creds_from_secrets:
    st.sidebar.subheader("Google Sheets Credentials")
    st.sidebar.markdown("Upload your Google Cloud Service Account JSON key file. This is not stored after your session ends.")
    uploaded_file = st.sidebar.file_uploader("Upload Service Account JSON", type="json")
    if uploaded_file:
        try:
            creds_json_str = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            uploaded_creds_dict = json.loads(creds_json_str)
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file. Please upload a valid Google Service Account key file.")
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")

final_creds = creds_from_secrets if creds_from_secrets else uploaded_creds_dict

default_calc_date = pd.Timestamp('2025-06-04').date() 
current_calc_date_input = st.sidebar.date_input("Date for Calculations", value=default_calc_date)
current_calc_date_str = current_calc_date_input.strftime('%Y-%m-%d')
current_calc_date_ts = pd.Timestamp(current_calc_date_input)

# Initialize session state
if 'df_analyzed' not in st.session_state: st.session_state.df_analyzed = None
if 'spreadsheet_obj' not in st.session_state: st.session_state.spreadsheet_obj = None
if 'resolved_job_indices' not in st.session_state: st.session_state.resolved_job_indices = set()
if 'snoozed_job_indices' not in st.session_state: st.session_state.snoozed_job_indices = set()
if 'assignments' not in st.session_state: st.session_state.assignments = {} 
if 'current_page' not in st.session_state: st.session_state.current_page = 0
if 'selected_next_activities' not in st.session_state: st.session_state.selected_next_activities = []
if 'selected_salespersons' not in st.session_state: st.session_state.selected_salespersons = []
if 'selected_supplied_by' not in st.session_state: st.session_state.selected_supplied_by = []
if 'sort_by_column' not in st.session_state: st.session_state.sort_by_column = "Install - Date"


if final_creds:
    if st.sidebar.button("🔄 Load and Analyze Job Data", key="load_analyze_button"):
        st.session_state.df_analyzed = None 
        st.session_state.resolved_job_indices = set() 
        st.session_state.snoozed_job_indices = set()
        st.session_state.assignments = {}
        st.session_state.current_page = 0 
        # Don't reset filters on load, allow users to keep them
        # st.session_state.selected_next_activities = [] 
        # st.session_state.selected_salespersons = []
        # st.session_state.selected_supplied_by = []


        with st.spinner("Loading data from Google Sheets..."):
            raw_df, spreadsheet, gc_instance = load_google_sheet(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
            st.session_state.spreadsheet_obj = spreadsheet
        
        if raw_df is not None and not raw_df.empty:
            st.success(f"Successfully loaded {len(raw_df)} jobs from '{DATA_WORKSHEET_NAME}'.")
            with st.spinner("Processing and flagging data..."):
                df_processed = preprocess_data(raw_df) 
                if df_processed.empty and not raw_df.empty: 
                    st.warning("Preprocessing returned an empty DataFrame. Date parsing might have issues.")
                df_threshold_flagged = flag_threshold_delays(df_processed, current_calc_date_str)
                df_keyword_flagged = flag_keyword_issues(df_threshold_flagged)
                df_past_due_flagged = flag_past_due_activities(df_keyword_flagged, current_calc_date_str)
            st.session_state.df_analyzed = df_past_due_flagged

            if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty:
                all_flag_summary_cols = [col for col in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 'Flag_Any_PastDue_Activity'] if col in st.session_state.df_analyzed.columns]
                st.session_state.df_analyzed['Flag_Overall_Needs_Attention'] = st.session_state.df_analyzed[all_flag_summary_cols].any(axis=1) if all_flag_summary_cols else False
                st.success("Data analysis complete!")
            elif st.session_state.df_analyzed is not None and st.session_state.df_analyzed.empty:
                 st.info("Data analysis complete, but no jobs matched criteria or the source data led to an empty result after processing.")
            else: 
                st.error("Data analysis failed after loading. df_analyzed is None.")
        elif raw_df is not None and raw_df.empty:
             st.info(f"Source data sheet '{DATA_WORKSHEET_NAME}' is empty. No analysis performed.")
        else: 
            st.error("Failed to load data from Google Sheets. Check credentials and sheet sharing.")
elif not creds_from_secrets :
     st.sidebar.info("Please upload your Google Service Account JSON key to begin.")

# --- Filters and Sort Options ---
if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty:
    df_for_filters = st.session_state.df_analyzed[st.session_state.df_analyzed.get('Flag_Overall_Needs_Attention', False)].copy()
    
    filter_cols = st.columns(3)
    with filter_cols[0]:
        if 'Next Sched. - Activity' in df_for_filters.columns:
            unique_next_activities = sorted(df_for_filters['Next Sched. - Activity'].dropna().astype(str).unique())
            if unique_next_activities:
                st.session_state.selected_next_activities = st.multiselect(
                    "Filter by Next Sched. Activity:", options=unique_next_activities,
                    default=st.session_state.selected_next_activities, key="next_activity_filter"
                )
    with filter_cols[1]:
        if 'Salesperson' in df_for_filters.columns:
            unique_salespersons = sorted(df_for_filters['Salesperson'].dropna().astype(str).unique())
            if unique_salespersons:
                st.session_state.selected_salespersons = st.multiselect(
                    "Filter by Salesperson:", options=unique_salespersons,
                    default=st.session_state.selected_salespersons, key="salesperson_filter"
                )
    with filter_cols[2]:
        if 'Supplied By' in df_for_filters.columns:
            unique_supplied_by = sorted(df_for_filters['Supplied By'].dropna().astype(str).unique())
            if unique_supplied_by:
                st.session_state.selected_supplied_by = st.multiselect(
                    "Filter by Supplied By:", options=unique_supplied_by,
                    default=st.session_state.selected_supplied_by, key="supplied_by_filter"
                )
    
    st.session_state.sort_by_column = st.selectbox(
        "Sort jobs by:",
        options=["Install - Date", "Days Behind"],
        index=0 if st.session_state.sort_by_column == "Install - Date" else 1,
        key="sort_by_select"
    )


# --- Display Interactive "todo" List ---
if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty and 'Flag_Overall_Needs_Attention' in st.session_state.df_analyzed.columns:
    df_display_full = st.session_state.df_analyzed.copy()
    if not df_display_full.index.is_unique: 
        df_display_full = df_display_full.reset_index(drop=True)

    priority_jobs_df_all_original = df_display_full[df_display_full['Flag_Overall_Needs_Attention'] == True].copy()

    if not priority_jobs_df_all_original.empty:
        issues_and_days_series = priority_jobs_df_all_original.apply(lambda row: determine_primary_issue_and_days(row, current_calc_date_ts), axis=1)
        priority_jobs_df_all_original.loc[:, 'Primary Issue'] = [item[0] for item in issues_and_days_series]
        priority_jobs_df_all_original.loc[:, 'Days Behind'] = [item[1] for item in issues_and_days_series]
        
        # Apply Filters
        priority_jobs_df_filtered = priority_jobs_df_all_original.copy()
        if st.session_state.selected_next_activities and 'Next Sched. - Activity' in priority_jobs_df_filtered.columns:
            priority_jobs_df_filtered = priority_jobs_df_filtered[
                priority_jobs_df_filtered['Next Sched. - Activity'].isin(st.session_state.selected_next_activities)
            ]
        if st.session_state.selected_salespersons and 'Salesperson' in priority_jobs_df_filtered.columns:
            priority_jobs_df_filtered = priority_jobs_df_filtered[
                priority_jobs_df_filtered['Salesperson'].isin(st.session_state.selected_salespersons)
            ]
        if st.session_state.selected_supplied_by and 'Supplied By' in priority_jobs_df_filtered.columns:
            priority_jobs_df_filtered = priority_jobs_df_filtered[
                priority_jobs_df_filtered['Supplied By'].isin(st.session_state.selected_supplied_by)
            ]

        # Apply Sorting
        sort_ascending = True
        if st.session_state.sort_by_column == "Install - Date":
            if 'Install - Date' in priority_jobs_df_filtered.columns and pd.api.types.is_datetime64_any_dtype(priority_jobs_df_filtered['Install - Date']):
                priority_jobs_df_filtered.sort_values(by='Install - Date', ascending=True, na_position='last', inplace=True)
        elif st.session_state.sort_by_column == "Days Behind":
            # Convert 'Days Behind' to numeric for sorting, coercing errors for 'N/A'
            priority_jobs_df_filtered['Days Behind_numeric'] = pd.to_numeric(priority_jobs_df_filtered['Days Behind'], errors='coerce')
            priority_jobs_df_filtered.sort_values(by='Days Behind_numeric', ascending=False, na_position='last', inplace=True) # Show most days behind first
            priority_jobs_df_filtered.drop(columns=['Days Behind_numeric'], inplace=True) # Drop temporary sort column
        

        visible_priority_jobs_df = priority_jobs_df_filtered[
            ~priority_jobs_df_filtered.index.isin(list(st.session_state.resolved_job_indices)) &
            ~priority_jobs_df_filtered.index.isin(list(st.session_state.snoozed_job_indices))
        ].copy() 

        st.header(f"🚩 Jobs Requiring Attention ({len(visible_priority_jobs_df)} currently shown)")

        if not visible_priority_jobs_df.empty:
            total_jobs_to_display = len(visible_priority_jobs_df)
            total_pages = math.ceil(total_jobs_to_display / ITEMS_PER_PAGE) if ITEMS_PER_PAGE > 0 else 1
            
            # Reset current_page if it's out of bounds due to filtering
            if st.session_state.current_page >= total_pages and total_pages > 0:
                st.session_state.current_page = total_pages - 1
            if st.session_state.current_page < 0: 
                 st.session_state.current_page = 0


            start_idx = st.session_state.current_page * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            jobs_to_display_on_page = visible_priority_jobs_df.iloc[start_idx:end_idx]

            for job_index, row_data in jobs_to_display_on_page.iterrows(): 
                job_name_display = row_data.get('Job Name', f"Job Index {job_index}")
                prod_number_display = row_data.get('Production #', '') 
                subheader_text = f"{job_name_display}"
                if prod_number_display and str(prod_number_display).strip() != "":
                    subheader_text += f" (PO: {prod_number_display})"
                
                st.subheader(subheader_text)

                info_cols = st.columns([3,1,2,2])
                primary_issue_display = row_data.get('Primary Issue', "N/A")
                days_behind_display = row_data.get('Days Behind', "N/A")
                install_date_display = row_data.get('Install - Date') 
                
                install_date_str = "N/A"
                if pd.notna(install_date_display) and isinstance(install_date_display, pd.Timestamp):
                    install_date_str = install_date_display.strftime('%Y-%m-%d')
                elif pd.notna(install_date_display): 
                    install_date_str = str(install_date_display)

                info_cols[0].markdown(f"**Issue:** {primary_issue_display}")
                info_cols[1].markdown(f"**Days:** {days_behind_display}")
                info_cols[2].markdown(f"**Install:** {install_date_str}")
                info_cols[3].markdown(f"**Next Sched:** {row_data.get('Next Sched. - Activity', 'N/A')}")
                
                action_button_cols = st.columns([1,1,5]) 
                with action_button_cols[0]:
                    if st.button("Resolve", key=f"resolve_{job_index}", help="Mark as resolved for this session & log action"):
                        st.session_state.resolved_job_indices.add(job_index)
                        log_entry = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Job Name": job_name_display,
                            "Production #": prod_number_display,
                            "Action": "Resolved",
                            "Details": primary_issue_display,
                            "Assigned To": st.session_state.assignments.get(job_index, "")
                        }
                        append_action_log(st.session_state.spreadsheet_obj, ACTION_LOG_WORKSHEET_NAME, log_entry)
                        st.rerun() 
                with action_button_cols[1]:
                    if st.button("Snooze", key=f"snooze_{job_index}", help="Hide from view for this session & log action"):
                        st.session_state.snoozed_job_indices.add(job_index)
                        log_entry = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Job Name": job_name_display,
                            "Production #": prod_number_display,
                            "Action": "Snoozed",
                            "Details": primary_issue_display,
                            "Assigned To": st.session_state.assignments.get(job_index, "")
                        }
                        append_action_log(st.session_state.spreadsheet_obj, ACTION_LOG_WORKSHEET_NAME, log_entry)
                        st.rerun() 

                with st.expander("Show Full Details & Notes", expanded=False):
                    st.write("--- Job Details ---")
                    st.write("**Key Dates:**")
                    date_cols_to_show_detail = [ 
                        'Next Sched. - Date', 'Job Creation', 'Template - Date', 'Ready to Fab - Date', 
                        'Cutlist - Date', 'Saw - Date', 'Polish/Fab Completion - Date', 
                        'Install - Date', 'Invoice - Date', 'Collect Final - Date', 'Service - Date'
                        ]
                    for col in date_cols_to_show_detail:
                        display_col_name = col 
                        if col in row_data and pd.notna(row_data[col]) and isinstance(row_data[col], pd.Timestamp):
                            st.markdown(f"  *{display_col_name}:* {row_data[col].strftime('%Y-%m-%d')}")
                        elif col in row_data and pd.notna(row_data[col]): 
                             st.markdown(f"  *{display_col_name}:* {row_data[col]}")

                    st.write("**Notes:**")
                    notes_cols_to_display = [ 
                        'Next Sched. - Notes', 'Template - Notes', 'Install - Notes', 
                        'Saw - Notes', 'Job Issues', 'QC - Notes', 'Address Notes'
                        ] 
                    for note_col in notes_cols_to_display:
                        if note_col in row_data and pd.notna(row_data[note_col]) and str(row_data[note_col]).strip() != '':
                            display_note_col_name = note_col
                            st.markdown(f"**{display_note_col_name}:**")
                            st.text_area(f"", value=str(row_data[note_col]), height=100, key=f"note_{note_col}_{job_index}", disabled=True)
                    
                    current_assignment = st.session_state.assignments.get(job_index, "")
                    new_assignment = st.text_input("Assign/Notify To:", value=current_assignment, key=f"assign_{job_index}")
                    if new_assignment != current_assignment:
                        st.session_state.assignments[job_index] = new_assignment
                        log_entry = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Job Name": job_name_display,
                            "Production #": prod_number_display,
                            "Action": "Assigned",
                            "Details": f"Assigned to: {new_assignment}",
                            "Assigned To": new_assignment
                        }
                        append_action_log(st.session_state.spreadsheet_obj, ACTION_LOG_WORKSHEET_NAME, log_entry)
                        st.info(f"Assignment for {job_name_display} updated to: {new_assignment} and logged.")
                st.markdown("---") 
            
            if total_pages > 1:
                page_cols = st.columns(5) 
                if page_cols[0].button("⬅️ Previous", disabled=(st.session_state.current_page == 0), key="prev_page_button"):
                    st.session_state.current_page -= 1
                    st.rerun()
                with page_cols[1]: 
                     st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")

                if page_cols[2].button("Next ➡️", disabled=(st.session_state.current_page >= total_pages - 1), key="next_page_button"):
                    st.session_state.current_page += 1
                    st.rerun()

            if st.button("✍️ Update 'todo' Sheet (with current filtered & paged view)", key="update_todo_sheet_button_interactive"):
                if st.session_state.spreadsheet_obj:
                    with st.spinner(f"Writing to '{TODO_WORKSHEET_NAME}' sheet..."):
                        # Export the currently VISIBLE and FILTERED jobs
                        export_cols = ['Primary Issue', 'Days Behind', 'Job Name', 'Production #'] 
                        if 'Salesperson' in visible_priority_jobs_df.columns: export_cols.append('Salesperson')
                        for key_col in ['Job Creation', 'Template - Date', 'Install - Date', 'Next Sched. - Activity', 'Next Sched. - Date']: 
                             if key_col in visible_priority_jobs_df.columns:
                                export_cols.append(key_col)
                        
                        if 'Next Sched. - Activity' in visible_priority_jobs_df.columns:
                            if 'Next Sched. - Activity' not in export_cols: 
                                export_cols.append('Next Sched. - Activity')
                        
                        true_flag_cols_in_visible = [col for col in visible_priority_jobs_df.columns if col.startswith('Flag_') and \
                                          col not in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 
                                                      'Flag_Any_PastDue_Activity', 'Flag_Overall_Needs_Attention', 
                                                      'Primary Issue', 'Days Behind'] and \
                                          visible_priority_jobs_df[col].any()]
                        export_cols.extend(sorted(true_flag_cols_in_visible))
                        export_cols = [col for col in export_cols if col in visible_priority_jobs_df.columns] 
                        
                        export_df_final = visible_priority_jobs_df[export_cols].copy()
                        write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, export_df_final)
                else:
                    st.error("Spreadsheet object not available. Cannot write to Google Sheet. Try reloading data.")
        else: 
            if not priority_jobs_df_all_original.empty and visible_priority_jobs_df.empty :
                 st.info("All priority jobs matching current filters have been locally resolved or snoozed for this session.")
            else: 
                st.info("No jobs currently require attention based on the defined criteria (or active filters).")
            
            if st.session_state.spreadsheet_obj and st.button("Clear 'todo' Sheet (as no jobs to show/list)", key="clear_todo_button_interactive"):
                 with st.spinner(f"Clearing '{TODO_WORKSHEET_NAME}' sheet..."):
                    write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, pd.DataFrame())
    else: 
        st.warning("Overall attention flag not found. Analysis might be incomplete or no jobs require attention.")

    with st.expander("Show Detailed Flag Counts (from full analyzed data before filtering/local actions)"):
        df_full_for_counts = st.session_state.df_analyzed 
        st.subheader("Threshold-Based Delays")
        if 'Flag_Awaiting_RTF' in df_full_for_counts.columns: st.metric("Awaiting RTF (>2 days)", df_full_for_counts['Flag_Awaiting_RTF'].sum())
        if 'Flag_Awaiting_Template' in df_full_for_counts.columns: st.metric("Awaiting Template (>2 days)", df_full_for_counts['Flag_Awaiting_Template'].sum())
        if 'Flag_Awaiting_Cutlist' in df_full_for_counts.columns: st.metric("Awaiting Cutlist (>3 days, non-laminate)", df_full_for_counts['Flag_Awaiting_Cutlist'].sum())
        
        st.subheader("Keyword Issues in Notes")
        notes_cols_for_keyword_summary = [ 
            'Next Sched. - Notes', 'Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes', 
            'Ready to Fab - Notes','Rework - Notes', 'Cutlist - Notes', 'Program - Notes', 'Material Pull - Notes', 
            'CNC - Notes', 'Polish/Fab Completion - Notes', 'Hone Splash - Notes', 'Ship - Notes', 'Product Rcvd - Notes',
            'Repair - Notes', 'Delivery - Notes', 'Pick Up - Notes', 'Service - Notes', 'Callback - Notes', 'Invoice - Notes', 
            'Build Up - Notes', 'Tearout - Notes', 'Lift Help - Notes', 'Courier - Notes', 'Tile Order - Notes', 
            'Tile Install - Notes', 'Collect Final - Notes', 'Follow Up Call - Notes', 'Address Notes'
            ]
        actual_notes_cols_for_summary = [col for col in notes_cols_for_keyword_summary if col in df_full_for_counts.columns]

        for note_col_original_name in actual_notes_cols_for_summary:
            clean_col_name_for_flag = re.sub(r'[^A-Za-z0-9_]+', '_', note_col_original_name)
            flag_col_name = f'Flag_Keyword_In_{clean_col_name_for_flag}'
            if flag_col_name in df_full_for_counts.columns:
                display_note_name = note_col_original_name 
                st.write(f"*Keyword in '{display_note_name}'*: {df_full_for_counts[flag_col_name].sum()} jobs")
        
        st.subheader("Past Due Activities")
        past_due_activities_display = [
            ('Next Sched. Activity', 'Flag_PastDue_Next_Sched_Activity'),
            ('Template', 'Flag_PastDue_Template'), ('RTF', 'Flag_PastDue_Ready_to_Fab'), ('Install', 'Flag_PastDue_Install'),
            ('Invoice', 'Flag_PastDue_Invoice'), ('Collect Final', 'Flag_PastDue_Collect_Final'),
            ('Saw', 'Flag_PastDue_Saw'), ('Polish/Fab Completion', 'Flag_PastDue_Polish_Fab_Completion'),
            ('Cutlist', 'Flag_PastDue_Cutlist'), ('Program', 'Flag_PastDue_Program'),
            ('QC', 'Flag_PastDue_QC'), ('Delivery', 'Flag_PastDue_Delivery'), ('Service', 'Flag_PastDue_Service')
        ]
        for friendly_name, flag_col in past_due_activities_display:
            if flag_col in df_full_for_counts.columns: 
                st.write(f"*Past Due '{friendly_name}'*: {df_full_for_counts[flag_col].sum()} jobs")

    if st.checkbox("Show Full Analyzed Data Table (with all flags)", key="show_full_data_interactive"):
        st.subheader("Full Analyzed Data (before local resolve/snooze)")
        st.dataframe(df_display_full, use_container_width=True)

elif final_creds and st.session_state.df_analyzed is not None and st.session_state.df_analyzed.empty:
    st.info("Analysis complete. The source data might be empty or no jobs matched the flagging criteria after processing.")
elif not final_creds and not creds_from_secrets :
     st.info("Please upload your Google Service Account JSON key in the sidebar and click 'Load and Analyze Job Data'.")

st.sidebar.markdown("---")
st.sidebar.info("Remember to configure `GOOGLE_CREDS_JSON` in Streamlit Secrets for deployed apps.")
