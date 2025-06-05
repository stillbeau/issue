import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from io import StringIO
import json # For parsing JSON from secrets or uploaded file
import math # For pagination

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Issue Detector", page_icon="⚙️")

# --- App Title ---
st.title("⚙️ Job Issue Detector Dashboard")
st.markdown("Analyzes job data from Google Sheets to identify issues and create a prioritized 'todo' list.")

# --- Constants & Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38" # YOUR GOOGLE SHEET ID
DATA_WORKSHEET_NAME = "jobs" # Your main data sheet
TODO_WORKSHEET_NAME = "todo" # Sheet for the priority list
ITEMS_PER_PAGE = 10 # For pagination

# --- Helper Functions (Authentication, Data Loading, Processing) ---

def get_google_creds():
    """
    Tries to load Google credentials from Streamlit secrets.
    If not found (e.g., local development), it will expect a file upload.
    Returns the credentials dictionary or None.
    """
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
        return df, spreadsheet, gc 
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Error: Worksheet '{worksheet_name}' not found in the Google Sheet.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading Google Sheet '{worksheet_name}': {e}")
        return None, None, None

def preprocess_data(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df_processed = df.copy()
    date_columns_to_convert = [
        'Job Creation', 'Template - Date', 'Send Quote - Date', 'Photo Layout - Date',
        'Ready to Fab - Date', 'Rework - Date', 'Cutlist - Date', 'Program - Date',
        'Material Pull - Date', 'Saw - Date', 'CNC - Date', 'Glue-Up Fabrication - Date',
        'Polish/Fab Completion - Date', 'Hone Splash - Date', 'QC - Date',
        'Packaging - Date', 'Ship - Date', 'Transfer - Date', 'Product Rcvd - Date',
        'Repair - Date', 'Build Up - Date', 'Tearout - Date', 'Delivery - Date',
        'Install - Date', 'Lift Help - Date', 'Pick Up - Date', 'Courier - Date',
        'Tile Order - Date', 'Tile Install - Date', 'Service - Date', 'Callback - Date',
        'Invoice - Date', 'Collect Final - Date', 'Follow Up Call - Date'
    ]
    common_non_date_placeholders = ['', 'None', 'none', 'NAN', 'NaN', 'nan', 'NA', 'NaT', 'nat', 'Pending', 'TBD', 'No Date', '#N/A', 'NULL', 'null']

    # Remove existing debug prints from here, or make them optional
    # st.write("--- Debugging: Raw Date Values Before Conversion ---") 
    debug_cols_sample = ['Job Creation', 'Template - Date', 'Install - Date', 'Invoice - Date']


    for col_name in date_columns_to_convert:
        if col_name in df_processed.columns:
            df_processed[col_name] = df_processed[col_name].astype(str).str.strip()
            for placeholder in common_non_date_placeholders:
                df_processed[col_name] = df_processed[col_name].replace(placeholder, None, regex=False) 
            
            # # Debug: Print a sample of values for key date columns AFTER cleaning, BEFORE to_datetime
            # if col_name in debug_cols_sample and st.session_state.get("debug_mode", False): # Optional debug mode
            #     st.write(f"Sample values for '{col_name}' (after cleaning, before pd.to_datetime):")
            #     unique_vals = df_processed[col_name].dropna().unique()
            #     st.write(unique_vals[:15]) 

            df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce')
            
            # # Debug: Check for NaT proportions after conversion
            # if col_name in debug_cols_sample and df_processed[col_name].isnull().any() and st.session_state.get("debug_mode", False):
            #      nat_percentage = (df_processed[col_name].isnull().sum() / len(df_processed[col_name])) * 100
            #      if nat_percentage > 0:
            #         st.write(f"**Warning for '{col_name}'**: {nat_percentage:.2f}% values became NaT (Not a Time) after conversion.")
    
    # st.write("--- End Debugging: Raw Date Values ---")
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

    notes_columns_to_scan = ['Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes']
    keyword_flag_columns_generated = []
    for col_name in notes_columns_to_scan:
        if col_name in df_flagged.columns:
            new_flag_col_name = f'Flag_Keyword_In_{col_name.replace(" - ", "_").replace(" ", "_")}'
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
        ('Template', 'Template - Date', 'Template - Status'),
        ('RTF', 'Ready_to_Fab', 'Ready to Fab - Date', 'Ready to Fab - Status'),
        ('Install', 'Install - Date', 'Install - Status'),
        ('Invoice', 'Invoice - Date', 'Invoice - Status'),
        ('Collect_Final', 'Collect Final - Date', 'Collect Final - Status'),
        ('Saw', 'Saw - Date', 'Saw - Status'),
        ('Polish_Fab_Completion', 'Polish/Fab Completion - Date', 'Polish/Fab Completion - Status')
    ]
    past_due_flag_columns_generated = []

    for activity_tuple in activities_to_check_past_due:
        if len(activity_tuple) == 4 and activity_tuple[0] == 'RTF':
             activity_name_for_flag, date_col, status_col = activity_tuple[1], activity_tuple[2], activity_tuple[3]
        elif len(activity_tuple) == 3:
             activity_name_for_flag, date_col, status_col = activity_tuple[0], activity_tuple[1], activity_tuple[2]
        else: continue

        new_flag_col = f'Flag_PastDue_{activity_name_for_flag}'
        df_flagged[new_flag_col] = False
        if not (date_col in df_flagged.columns and status_col in df_flagged.columns and pd.api.types.is_datetime64_any_dtype(df_flagged[date_col])):
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
        'Flag_PastDue_Install': 'Install - Date',
        'Flag_PastDue_Polish_Fab_Completion': 'Polish/Fab Completion - Date',
        'Flag_PastDue_Saw': 'Saw - Date',
        'Flag_PastDue_Ready_to_Fab': 'Ready to Fab - Date',
        'Flag_PastDue_Invoice': 'Invoice - Date',
        'Flag_PastDue_Collect_Final': 'Collect Final - Date',
        'Flag_PastDue_Template': 'Template - Date',
        'Flag_Awaiting_Cutlist': 'Ready to Fab - Date', 
        'Flag_Awaiting_RTF': 'Template - Date',       
        'Flag_Awaiting_Template': 'Job Creation'      
    }
    issue_desc_map = {
        'Flag_PastDue_Install': "Past Due: Install",
        'Flag_PastDue_Polish_Fab_Completion': "Past Due: Polish/Fab",
        'Flag_PastDue_Saw': "Past Due: Saw",
        'Flag_PastDue_Ready_to_Fab': "Past Due: RTF",
        'Flag_PastDue_Invoice': "Past Due: Invoice",
        'Flag_PastDue_Collect_Final': "Past Due: Collect Final",
        'Flag_PastDue_Template': "Past Due: Template",
        'Flag_Awaiting_Cutlist': "Delay: Awaiting Cutlist",
        'Flag_Awaiting_RTF': "Delay: Awaiting RTF",
        'Flag_Awaiting_Template': "Delay: Awaiting Template",
        'Flag_Keyword_In_Install_Notes': "Keyword: Install Notes",
        'Flag_Keyword_In_Template_Notes': "Keyword: Template Notes",
        'Flag_Keyword_In_Job_Issues': "Keyword: Job Issues",
        'Flag_Keyword_In_QC_Notes': "Keyword: QC Notes",
        'Flag_Keyword_In_Saw_Notes': "Keyword: Saw Notes"
    }
    flag_check_order = [
        'Flag_PastDue_Install', 'Flag_PastDue_Polish_Fab_Completion', 'Flag_PastDue_Saw', 
        'Flag_PastDue_Ready_to_Fab', 'Flag_PastDue_Invoice', 'Flag_PastDue_Collect_Final', 
        'Flag_PastDue_Template', 'Flag_Awaiting_Cutlist', 'Flag_Awaiting_RTF', 
        'Flag_Awaiting_Template', 'Flag_Keyword_In_Install_Notes', 'Flag_Keyword_In_Template_Notes',
        'Flag_Keyword_In_Job_Issues', 'Flag_Keyword_In_QC_Notes', 'Flag_Keyword_In_Saw_Notes'
    ]

    for flag_col in flag_check_order:
        if row.get(flag_col, False):
            issue_description = issue_desc_map.get(flag_col, "Unknown Issue")
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


if final_creds:
    if st.sidebar.button("🔄 Load and Analyze Job Data", key="load_analyze_button"):
        st.session_state.df_analyzed = None 
        st.session_state.resolved_job_indices = set() 
        st.session_state.snoozed_job_indices = set()
        st.session_state.assignments = {}
        st.session_state.current_page = 0 # Reset page on new load

        with st.spinner("Loading data from Google Sheets..."):
            raw_df, spreadsheet, gc_instance = load_google_sheet(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
            st.session_state.spreadsheet_obj = spreadsheet
        
        if raw_df is not None and not raw_df.empty:
            st.success(f"Successfully loaded {len(raw_df)} jobs from '{DATA_WORKSHEET_NAME}'.")
            with st.spinner("Processing and flagging data..."):
                df_processed = preprocess_data(raw_df) 
                if df_processed.empty and not raw_df.empty: 
                    st.warning("Preprocessing returned an empty DataFrame. Check debug output for date parsing issues.")
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


# --- Display Interactive "todo" List ---
if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty and 'Flag_Overall_Needs_Attention' in st.session_state.df_analyzed.columns:
    df_display_full = st.session_state.df_analyzed.copy()
    # Ensure DataFrame has a unique index if it doesn't already
    if not df_display_full.index.is_unique:
        df_display_full = df_display_full.reset_index(drop=True)

    priority_jobs_df_all = df_display_full[df_display_full['Flag_Overall_Needs_Attention'] == True].copy()

    if not priority_jobs_df_all.empty:
        # Apply determine_primary_issue_and_days only if priority_jobs_df_all is not empty
        issues_and_days_series = priority_jobs_df_all.apply(lambda row: determine_primary_issue_and_days(row, current_calc_date_ts), axis=1)
        priority_jobs_df_all.loc[:, 'Primary Issue'] = [item[0] for item in issues_and_days_series]
        priority_jobs_df_all.loc[:, 'Days Behind'] = [item[1] for item in issues_and_days_series]
        
        if 'Install - Date' in priority_jobs_df_all.columns and pd.api.types.is_datetime64_any_dtype(priority_jobs_df_all['Install - Date']):
            priority_jobs_df_all.sort_values(by='Install - Date', ascending=True, na_position='last', inplace=True)
        
        visible_priority_jobs_df = priority_jobs_df_all[
            ~priority_jobs_df_all.index.isin(list(st.session_state.resolved_job_indices)) &
            ~priority_jobs_df_all.index.isin(list(st.session_state.snoozed_job_indices))
        ].copy() 

        st.header(f"🚩 Jobs Requiring Attention ({len(visible_priority_jobs_df)} currently shown)")

        if not visible_priority_jobs_df.empty:
            # --- Pagination Logic ---
            total_jobs = len(visible_priority_jobs_df)
            total_pages = math.ceil(total_jobs / ITEMS_PER_PAGE)
            
            # Ensure current_page is within valid bounds
            if st.session_state.current_page >= total_pages and total_pages > 0:
                st.session_state.current_page = total_pages - 1
            if st.session_state.current_page < 0:
                 st.session_state.current_page = 0


            start_idx = st.session_state.current_page * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            jobs_to_display_on_page = visible_priority_jobs_df.iloc[start_idx:end_idx]

            for job_index, row_data in jobs_to_display_on_page.iterrows(): 
                job_name_display = row_data.get('Job Name', f"Job Index {job_index}")
                primary_issue_display = row_data.get('Primary Issue', "N/A")
                days_behind_display = row_data.get('Days Behind', "N/A")
                install_date_display = row_data.get('Install - Date')
                if pd.notna(install_date_display) and isinstance(install_date_display, pd.Timestamp):
                    install_date_str = install_date_display.strftime('%Y-%m-%d')
                else:
                    install_date_str = "N/A"

                st.subheader(f"{job_name_display}")
                item_cols = st.columns([3, 2, 3])
                item_cols[0].markdown(f"**Issue:** {primary_issue_display}")
                item_cols[1].markdown(f"**Days:** {days_behind_display}")
                item_cols[2].markdown(f"**Install Date:** {install_date_str}")

                with st.expander("Show Details & Actions", expanded=False):
                    st.write("--- Job Details ---")
                    st.write("**Key Dates:**")
                    date_cols_to_show_detail = ['Job Creation', 'Template - Date', 'Ready to Fab - Date', 
                                               'Cutlist - Date', 'Saw - Date', 'Polish/Fab Completion - Date', 
                                               'Install - Date', 'Invoice - Date', 'Collect Final - Date']
                    for col in date_cols_to_show_detail:
                        if col in row_data and pd.notna(row_data[col]) and isinstance(row_data[col], pd.Timestamp):
                            st.markdown(f"  *{col.replace(' - Date', '')}:* {row_data[col].strftime('%Y-%m-%d')}")
                        elif col in row_data and pd.notna(row_data[col]):
                             st.markdown(f"  *{col.replace(' - Date', '')}:* {row_data[col]}")

                    st.write("**Notes:**")
                    notes_cols_to_display = ['Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes']
                    for note_col in notes_cols_to_display:
                        if note_col in row_data and pd.notna(row_data[note_col]) and str(row_data[note_col]).strip() != '':
                            st.markdown(f"**{note_col}:**")
                            st.text_area(f"", value=str(row_data[note_col]), height=100, key=f"note_{note_col}_{job_index}", disabled=True)
                    
                    st.write("**Actions:**")
                    action_cols = st.columns(3)
                    if action_cols[0].button("Mark Resolved (Session)", key=f"resolve_{job_index}"):
                        st.session_state.resolved_job_indices.add(job_index)
                        st.rerun() 
                    if action_cols[1].button("Snooze (Session)", key=f"snooze_{job_index}"):
                        st.session_state.snoozed_job_indices.add(job_index)
                        st.rerun() 
                    
                    current_assignment = st.session_state.assignments.get(job_index, "")
                    new_assignment = st.text_input("Assign/Notify To:", value=current_assignment, key=f"assign_{job_index}")
                    if new_assignment != current_assignment:
                        st.session_state.assignments[job_index] = new_assignment
                        st.info(f"Assignment for {job_name_display} updated in session to: {new_assignment}")
                st.markdown("---") 
            
            # --- Pagination Controls ---
            if total_pages > 1:
                page_cols = st.columns(3)
                if page_cols[0].button("⬅️ Previous Page", disabled=(st.session_state.current_page == 0)):
                    st.session_state.current_page -= 1
                    st.rerun()
                page_cols[1].write(f"Page {st.session_state.current_page + 1} of {total_pages}")
                if page_cols[2].button("Next Page ➡️", disabled=(st.session_state.current_page >= total_pages - 1)):
                    st.session_state.current_page += 1
                    st.rerun()

            if st.button("✍️ Update 'todo' Sheet (with current view)", key="update_todo_sheet_button_interactive"):
                if st.session_state.spreadsheet_obj:
                    with st.spinner(f"Writing to '{TODO_WORKSHEET_NAME}' sheet..."):
                        export_cols = ['Primary Issue', 'Days Behind', 'Job Name', 'Salesperson', 
                                       'Job Creation', 'Template - Date', 'Install - Date', 'Job Status']
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
        else: # If visible_priority_jobs_df is empty
            if not priority_jobs_df_all.empty and visible_priority_jobs_df.empty :
                 st.info("All priority jobs have been locally resolved or snoozed for this session.")
            else: # If priority_jobs_df_all was also empty
                st.info("No jobs currently require attention based on the defined criteria.")
            
            if st.session_state.spreadsheet_obj and st.button("Clear 'todo' Sheet (as no jobs to show/list)", key="clear_todo_button_interactive"):
                 with st.spinner(f"Clearing '{TODO_WORKSHEET_NAME}' sheet..."):
                    write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, pd.DataFrame())
    else: 
        st.warning("Overall attention flag not found. Analysis might be incomplete.")

    with st.expander("Show Detailed Flag Counts (from full analyzed data)"):
        df_full_for_counts = st.session_state.df_analyzed 
        st.subheader("Threshold-Based Delays")
        if 'Flag_Awaiting_RTF' in df_full_for_counts.columns: st.metric("Awaiting RTF (>2 days)", df_full_for_counts['Flag_Awaiting_RTF'].sum())
        if 'Flag_Awaiting_Template' in df_full_for_counts.columns: st.metric("Awaiting Template (>2 days)", df_full_for_counts['Flag_Awaiting_Template'].sum())
        if 'Flag_Awaiting_Cutlist' in df_full_for_counts.columns: st.metric("Awaiting Cutlist (>3 days, non-laminate)", df_full_for_counts['Flag_Awaiting_Cutlist'].sum())
        
        st.subheader("Keyword Issues in Notes")
        notes_cols_for_keyword_summary = ['Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes']
        for note_col_original_name in notes_cols_for_keyword_summary:
            flag_col_name = f'Flag_Keyword_In_{note_col_original_name.replace(" - ", "_").replace(" ", "_")}'
            if flag_col_name in df_full_for_counts.columns:
                st.write(f"*Keyword in '{note_col_original_name}'*: {df_full_for_counts[flag_col_name].sum()} jobs")
        
        st.subheader("Past Due Activities")
        past_due_activities_display = [
            ('Template', 'Flag_PastDue_Template'), ('RTF', 'Flag_PastDue_Ready_to_Fab'), ('Install', 'Flag_PastDue_Install'),
            ('Invoice', 'Flag_PastDue_Invoice'), ('Collect Final', 'Flag_PastDue_Collect_Final'),
            ('Saw', 'Flag_PastDue_Saw'), ('Polish/Fab Completion', 'Flag_PastDue_Polish_Fab_Completion')
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
