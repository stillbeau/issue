import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from io import StringIO
import json # For parsing JSON from secrets or uploaded file

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Issue Detector", page_icon="⚙️")

# --- App Title ---
st.title("⚙️ Job Issue Detector Dashboard")
st.markdown("Analyzes job data from Google Sheets to identify issues and create a prioritized 'todo' list.")

# --- Constants & Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38" # YOUR GOOGLE SHEET ID
DATA_WORKSHEET_NAME = "jobs" # Your main data sheet
TODO_WORKSHEET_NAME = "todo" # Sheet for the priority list

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
            # st.sidebar.success("Credentials loaded from Streamlit Secrets.")
            return creds_dict
        except json.JSONDecodeError:
            st.sidebar.error("Error parsing Google credentials from Streamlit Secrets. Ensure it's valid JSON.")
            return None
        except Exception as e:
            st.sidebar.error(f"Error with Streamlit Secrets: {e}")
            return None
    return None # Indicates credentials should be uploaded

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_google_sheet(creds_dict, spreadsheet_id, worksheet_name):
    """Loads data from a Google Sheet into a pandas DataFrame."""
    if creds_dict is None:
        st.error("Google credentials not provided or invalid.")
        return None, None, None # Added None for gc
    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records() # More robust for varying empty cells
        if not data:
            st.info(f"Worksheet '{worksheet_name}' appears to be empty or has no data rows after headers.")
            return pd.DataFrame(), spreadsheet, gc # Return empty DataFrame but valid spreadsheet and gc
        df = pd.DataFrame(data)
        return df, spreadsheet, gc # Return spreadsheet and gc for writing back
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
    common_non_date_placeholders = ['', 'None', 'none', 'NAN', 'NaN', 'nan', 'NA', 'NaT', 'nat', 'Pending', 'TBD', 'No Date', '#N/A']

    for col_name in date_columns_to_convert:
        if col_name in df_processed.columns:
            # Convert to string, strip whitespace, and replace common placeholders with None
            df_processed[col_name] = df_processed[col_name].astype(str).str.strip()
            for placeholder in common_non_date_placeholders:
                df_processed[col_name] = df_processed[col_name].replace(placeholder, None, regex=False)
            
            # Attempt to convert to datetime
            df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce')
            
            # Optional: If you know the exact format and the above still fails, you can try:
            # df_processed[col_name] = pd.to_datetime(df_processed[col_name], format='%m/%d/%Y', errors='coerce')
            # or other formats like '%d/%m/%Y', '%Y-%m-%d', etc.
    return df_processed

def flag_threshold_delays(df, current_date_str):
    if df is None or df.empty: return pd.DataFrame()
    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)

    # Rule 1: Template Done, Awaiting RTF (Threshold: 2 days)
    df_flagged['Flag_Awaiting_RTF'] = False
    if 'Template - Date' in df_flagged.columns and 'Ready to Fab - Date' in df_flagged.columns:
        condition_rtf_pending = df_flagged['Template - Date'].notna() & \
                                df_flagged['Ready to Fab - Date'].isna() & \
                                pd.api.types.is_datetime64_any_dtype(df_flagged['Template - Date'])
        if condition_rtf_pending.any(): 
            df_flagged.loc[condition_rtf_pending, 'Flag_Awaiting_RTF'] = \
                (current_date - df_flagged.loc[condition_rtf_pending, 'Template - Date']).dt.days > 2

    # Rule 2: Job Created, Awaiting Template (Threshold: 2 days)
    df_flagged['Flag_Awaiting_Template'] = False
    if 'Job Creation' in df_flagged.columns and 'Template - Date' in df_flagged.columns:
        condition_template_pending = df_flagged['Job Creation'].notna() & \
                                     df_flagged['Template - Date'].isna() & \
                                     pd.api.types.is_datetime64_any_dtype(df_flagged['Job Creation'])
        if condition_template_pending.any():
            df_flagged.loc[condition_template_pending, 'Flag_Awaiting_Template'] = \
                (current_date - df_flagged.loc[condition_template_pending, 'Job Creation']).dt.days > 2
    
    # Rule 3: Ready to Fab, Awaiting Cutlist (Threshold: 3 days) - MODIFIED
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
    if threshold_flag_cols:
        df_flagged['Flag_Any_Threshold_Delay'] = df_flagged[threshold_flag_cols].any(axis=1)
    else:
        df_flagged['Flag_Any_Threshold_Delay'] = False
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
            
    if keyword_flag_columns_generated:
        df_flagged['Flag_Any_Keyword_Issue'] = df_flagged[keyword_flag_columns_generated].any(axis=1)
    else:
        df_flagged['Flag_Any_Keyword_Issue'] = False
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
        condition_date_past.loc[valid_date_rows] = (df_flagged.loc[valid_date_rows, date_col] < current_date)
        
        status_cleaned = df_flagged[status_col].fillna('').astype(str).str.lower().str.strip()
        condition_not_complete = ~status_cleaned.isin(completion_terms)
        condition_not_cancelled = ~status_cleaned.isin(cancellation_terms)
        
        final_condition = condition_date_past & condition_not_complete & condition_not_cancelled
        
        df_flagged.loc[final_condition, new_flag_col] = True
        past_due_flag_columns_generated.append(new_flag_col)

    if past_due_flag_columns_generated:
        df_flagged['Flag_Any_PastDue_Activity'] = df_flagged[past_due_flag_columns_generated].any(axis=1)
    else:
        df_flagged['Flag_Any_PastDue_Activity'] = False
    return df_flagged

def determine_primary_issue_and_days(row, current_calc_date_ts):
    """Determines a primary issue string and days behind, based on a hierarchy of flags."""
    if not isinstance(current_calc_date_ts, pd.Timestamp):
        current_calc_date_ts = pd.Timestamp(current_calc_date_ts)

    # Past Due Flags (Higher Priority)
    if row.get('Flag_PastDue_Install', False) and pd.notna(row.get('Install - Date')) and isinstance(row.get('Install - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Install - Date']).days
        return "Past Due: Install", days
    if row.get('Flag_PastDue_Polish_Fab_Completion', False) and pd.notna(row.get('Polish/Fab Completion - Date')) and isinstance(row.get('Polish/Fab Completion - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Polish/Fab Completion - Date']).days
        return "Past Due: Polish/Fab", days
    if row.get('Flag_PastDue_Saw', False) and pd.notna(row.get('Saw - Date')) and isinstance(row.get('Saw - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Saw - Date']).days
        return "Past Due: Saw", days
    if row.get('Flag_PastDue_Ready_to_Fab', False) and pd.notna(row.get('Ready to Fab - Date')) and isinstance(row.get('Ready to Fab - Date'), pd.Timestamp): 
        days = (current_calc_date_ts - row['Ready to Fab - Date']).days
        return "Past Due: RTF", days
    if row.get('Flag_PastDue_Invoice', False) and pd.notna(row.get('Invoice - Date')) and isinstance(row.get('Invoice - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Invoice - Date']).days
        return "Past Due: Invoice", days
    if row.get('Flag_PastDue_Collect_Final', False) and pd.notna(row.get('Collect Final - Date')) and isinstance(row.get('Collect Final - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Collect Final - Date']).days
        return "Past Due: Collect Final", days
    if row.get('Flag_PastDue_Template', False) and pd.notna(row.get('Template - Date')) and isinstance(row.get('Template - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Template - Date']).days
        return "Past Due: Template", days
    
    # Threshold Delay Flags
    if row.get('Flag_Awaiting_Cutlist', False) and pd.notna(row.get('Ready to Fab - Date')) and isinstance(row.get('Ready to Fab - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Ready to Fab - Date']).days 
        return "Delay: Awaiting Cutlist", days 
    if row.get('Flag_Awaiting_RTF', False) and pd.notna(row.get('Template - Date')) and isinstance(row.get('Template - Date'), pd.Timestamp):
        days = (current_calc_date_ts - row['Template - Date']).days
        return "Delay: Awaiting RTF", days 
    if row.get('Flag_Awaiting_Template', False) and pd.notna(row.get('Job Creation')) and isinstance(row.get('Job Creation'), pd.Timestamp):
        days = (current_calc_date_ts - row['Job Creation']).days
        return "Delay: Awaiting Template", days 
    
    # Keyword Flags (Days Behind = "N/A")
    if row.get('Flag_Keyword_In_Install_Notes', False): return "Keyword: Install Notes", "N/A"
    if row.get('Flag_Keyword_In_Template_Notes', False): return "Keyword: Template Notes", "N/A"
    if row.get('Flag_Keyword_In_Job_Issues', False): return "Keyword: Job Issues", "N/A"
    if row.get('Flag_Keyword_In_QC_Notes', False): return "Keyword: QC Notes", "N/A"
    if row.get('Flag_Keyword_In_Saw_Notes', False): return "Keyword: Saw Notes", "N/A"
    
    return "Other Issue", "N/A" # Fallback


def write_to_google_sheet(spreadsheet_obj, worksheet_name, df_to_write):
    """Writes a DataFrame to the specified Google Sheet worksheet."""
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

    df_export = df_to_write.fillna('').astype(str) # Ensure all are strings for gspread
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
current_calc_date_ts = pd.Timestamp(current_calc_date_input) # For passing to determine_primary_issue_and_days

# Initialize session state
if 'df_analyzed' not in st.session_state: st.session_state.df_analyzed = None
if 'spreadsheet_obj' not in st.session_state: st.session_state.spreadsheet_obj = None
if 'gc_obj' not in st.session_state: st.session_state.gc_obj = None


if final_creds:
    if st.sidebar.button("🔄 Load and Analyze Job Data", key="load_analyze_button"):
        st.session_state.df_analyzed = None 
        with st.spinner("Loading data from Google Sheets..."):
            raw_df, spreadsheet, gc_instance = load_google_sheet(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
            st.session_state.spreadsheet_obj = spreadsheet
            st.session_state.gc_obj = gc_instance 
        
        if raw_df is not None and not raw_df.empty:
            st.success(f"Successfully loaded {len(raw_df)} jobs from '{DATA_WORKSHEET_NAME}'.")
            
            with st.spinner("Processing and flagging data... This may take a moment."):
                df_processed = preprocess_data(raw_df)
                df_threshold_flagged = flag_threshold_delays(df_processed, current_calc_date_str)
                df_keyword_flagged = flag_keyword_issues(df_threshold_flagged)
                df_past_due_flagged = flag_past_due_activities(df_keyword_flagged, current_calc_date_str)
            
            st.session_state.df_analyzed = df_past_due_flagged

            if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty:
                all_flag_summary_cols = []
                for flag_type_col in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 'Flag_Any_PastDue_Activity']:
                    if flag_type_col in st.session_state.df_analyzed.columns:
                        all_flag_summary_cols.append(flag_type_col)
                
                if all_flag_summary_cols: 
                    st.session_state.df_analyzed['Flag_Overall_Needs_Attention'] = st.session_state.df_analyzed[all_flag_summary_cols].any(axis=1)
                else: 
                    st.session_state.df_analyzed['Flag_Overall_Needs_Attention'] = False
                st.success("Data analysis complete!")
            elif st.session_state.df_analyzed is not None and st.session_state.df_analyzed.empty:
                 st.info("Data analysis complete, but no jobs matched criteria or the source data led to an empty result after processing.")
            else:
                st.error("Data analysis failed after loading.")
        elif raw_df is not None and raw_df.empty:
             st.info(f"Source data sheet '{DATA_WORKSHEET_NAME}' is empty. No analysis performed.")
        else:
            st.error("Failed to load data from Google Sheets. Check credentials and sheet sharing.")
elif not creds_from_secrets:
    st.sidebar.info("Please upload your Google Service Account JSON key to begin.")


# --- Display Results and Write to 'todo' Sheet ---
if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty:
    df_display = st.session_state.df_analyzed.copy()
    st.header("🚩 Jobs Requiring Attention ('todo' List Preview)")
    
    if 'Flag_Overall_Needs_Attention' in df_display.columns:
        priority_jobs_df = df_display[df_display['Flag_Overall_Needs_Attention'] == True].copy()

        if not priority_jobs_df.empty:
            # Apply the new function to get two columns
            issues_and_days = priority_jobs_df.apply(
                lambda row: determine_primary_issue_and_days(row, current_calc_date_ts), axis=1
            )
            priority_jobs_df.loc[:, 'Primary Issue'] = [item[0] for item in issues_and_days]
            priority_jobs_df.loc[:, 'Days Behind'] = [item[1] for item in issues_and_days]

            if 'Install - Date' in priority_jobs_df.columns and pd.api.types.is_datetime64_any_dtype(priority_jobs_df['Install - Date']):
                priority_jobs_df.sort_values(by='Install - Date', ascending=True, na_position='last', inplace=True)
            
            display_cols_priority = ['Primary Issue', 'Days Behind', 'Job Name', 'Salesperson', 'Job Creation', 
                                     'Template - Date', 'Install - Date', 'Job Status']
            true_flag_cols_in_priority = [col for col in priority_jobs_df.columns if col.startswith('Flag_') and \
                                          col not in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 
                                                      'Flag_Any_PastDue_Activity', 'Flag_Overall_Needs_Attention', 
                                                      'Primary Issue', 'Days Behind'] and \
                                          priority_jobs_df[col].any()] # Check if any True in this column
            display_cols_priority.extend(sorted(true_flag_cols_in_priority))
            display_cols_priority = [col for col in display_cols_priority if col in priority_jobs_df.columns]


            st.dataframe(priority_jobs_df[display_cols_priority], height=600, use_container_width=True)
            st.markdown(f"**Total Jobs for 'todo' List: {len(priority_jobs_df)}**")

            if st.button("✍️ Update 'todo' Sheet in Google Sheets", key="update_todo_sheet_button"):
                if st.session_state.spreadsheet_obj:
                    with st.spinner(f"Writing to '{TODO_WORKSHEET_NAME}' sheet..."):
                        export_df = priority_jobs_df[display_cols_priority].copy() 
                        write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, export_df)
                else:
                    st.error("Spreadsheet object not available. Cannot write to Google Sheet. Try reloading data.")
        else:
            st.info("No jobs currently require attention based on the defined criteria.")
            if st.session_state.spreadsheet_obj and st.button("Clear 'todo' Sheet (as no jobs to list)", key="clear_todo_button"):
                 with st.spinner(f"Clearing '{TODO_WORKSHEET_NAME}' sheet..."):
                    write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, pd.DataFrame())


    # --- Detailed Breakdown Expander ---
    with st.expander("Show Detailed Flag Counts"):
        st.subheader("Threshold-Based Delays")
        if 'Flag_Awaiting_RTF' in df_display.columns: st.metric("Awaiting RTF (>2 days)", df_display['Flag_Awaiting_RTF'].sum())
        if 'Flag_Awaiting_Template' in df_display.columns: st.metric("Awaiting Template (>2 days)", df_display['Flag_Awaiting_Template'].sum())
        if 'Flag_Awaiting_Cutlist' in df_display.columns: st.metric("Awaiting Cutlist (>3 days, non-laminate)", df_display['Flag_Awaiting_Cutlist'].sum())
        
        st.subheader("Keyword Issues in Notes")
        notes_cols_for_keyword_summary = ['Template - Notes', 'Install - Notes', 'Saw - Notes', 'Job Issues', 'QC - Notes']
        for note_col_original_name in notes_cols_for_keyword_summary:
            flag_col_name = f'Flag_Keyword_In_{note_col_original_name.replace(" - ", "_").replace(" ", "_")}'
            if flag_col_name in df_display.columns:
                st.write(f"*Keyword in '{note_col_original_name}'*: {df_display[flag_col_name].sum()} jobs")
        
        st.subheader("Past Due Activities")
        past_due_activities_display = [
            ('Template', 'Flag_PastDue_Template'), ('RTF', 'Flag_PastDue_Ready_to_Fab'), ('Install', 'Flag_PastDue_Install'),
            ('Invoice', 'Flag_PastDue_Invoice'), ('Collect Final', 'Flag_PastDue_Collect_Final'),
            ('Saw', 'Flag_PastDue_Saw'), ('Polish/Fab Completion', 'Flag_PastDue_Polish_Fab_Completion')
        ]
        for friendly_name, flag_col in past_due_activities_display:
            if flag_col in past_due_activities_display: # This condition was incorrect, should check df_display.columns
                 if flag_col in df_display.columns:
                    st.write(f"*Past Due '{friendly_name}'*: {df_display[flag_col].sum()} jobs")


    if st.checkbox("Show Full Analyzed Data Table (with all flags)"):
        st.subheader("Full Analyzed Data")
        st.dataframe(df_display, use_container_width=True)

elif final_creds and st.session_state.df_analyzed is not None and st.session_state.df_analyzed.empty:
    st.info("Analysis complete. The source data might be empty or no jobs matched the flagging criteria after processing.")
elif not final_creds and not creds_from_secrets :
     st.info("Please upload your Google Service Account JSON key in the sidebar and click 'Load and Analyze Job Data'.")


st.sidebar.markdown("---")
st.sidebar.info("Remember to configure `GOOGLE_CREDS_JSON` in Streamlit Secrets for deployed apps.")
