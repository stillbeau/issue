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
        return None
    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records() # More robust for varying empty cells
        if not data:
            st.info(f"Worksheet '{worksheet_name}' appears to be empty or has no data rows after headers.")
            return pd.DataFrame() # Return empty DataFrame
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
    for col_name in date_columns_to_convert:
        if col_name in df_processed.columns:
            df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce', dayfirst=False, yearfirst=False) # Common date formats
    return df_processed

def flag_threshold_delays(df, current_date_str):
    if df is None or df.empty: return pd.DataFrame()
    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)

    df_flagged['Flag_Awaiting_RTF'] = False
    if 'Template - Date' in df_flagged.columns and 'Ready to Fab - Date' in df_flagged.columns:
        valid_rows = df_flagged['Template - Date'].notna() & df_flagged['Ready to Fab - Date'].isna() & pd.api.types.is_datetime64_any_dtype(df_flagged['Template - Date'])
        df_flagged.loc[valid_rows, 'Flag_Awaiting_RTF'] = \
            (current_date - df_flagged.loc[valid_rows, 'Template - Date']).dt.days > 2

    df_flagged['Flag_Awaiting_Template'] = False
    if 'Job Creation' in df_flagged.columns and 'Template - Date' in df_flagged.columns:
        valid_rows = df_flagged['Job Creation'].notna() & df_flagged['Template - Date'].isna() & pd.api.types.is_datetime64_any_dtype(df_flagged['Job Creation'])
        df_flagged.loc[valid_rows, 'Flag_Awaiting_Template'] = \
            (current_date - df_flagged.loc[valid_rows, 'Job Creation']).dt.days > 2
    
    df_flagged['Flag_Awaiting_Cutlist'] = False
    if 'Ready to Fab - Date' in df_flagged.columns and 'Cutlist - Date' in df_flagged.columns:
        valid_rows = df_flagged['Ready to Fab - Date'].notna() & df_flagged['Cutlist - Date'].isna() & pd.api.types.is_datetime64_any_dtype(df_flagged['Ready to Fab - Date'])
        df_flagged.loc[valid_rows, 'Flag_Awaiting_Cutlist'] = \
            (current_date - df_flagged.loc[valid_rows, 'Ready to Fab - Date']).dt.days > 3

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

        condition_date_past = df_flagged[date_col].notna() & (df_flagged[date_col] < current_date)
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

def determine_primary_issue(row):
    if row.get('Flag_PastDue_Install', False): return "Past Due: Install"
    if row.get('Flag_PastDue_Polish_Fab_Completion', False): return "Past Due: Polish/Fab"
    # ... (add other primary issue checks in order of priority) ...
    if row.get('Flag_Awaiting_Template', False): return "Delay: Awaiting Template"
    if row.get('Flag_Keyword_In_Install_Notes', False): return "Keyword: Install Notes"
    # Add all other primary issue checks from your Colab script here
    if row.get('Flag_PastDue_Saw', False): return "Past Due: Saw"
    if row.get('Flag_PastDue_RTF', False): return "Past Due: RTF"
    if row.get('Flag_PastDue_Invoice', False): return "Past Due: Invoice"
    if row.get('Flag_PastDue_Collect_Final', False): return "Past Due: Collect Final"
    if row.get('Flag_PastDue_Template', False): return "Past Due: Template"
    if row.get('Flag_Awaiting_Cutlist', False): return "Delay: Awaiting Cutlist"
    if row.get('Flag_Awaiting_RTF', False): return "Delay: Awaiting RTF"
    if row.get('Flag_Keyword_In_Template_Notes', False): return "Keyword: Template Notes"
    if row.get('Flag_Keyword_In_Job_Issues', False): return "Keyword: Job Issues"
    if row.get('Flag_Keyword_In_QC_Notes', False): return "Keyword: QC Notes"
    if row.get('Flag_Keyword_In_Saw_Notes', False): return "Keyword: Saw Notes"
    return "Other Issue"

def write_to_google_sheet(spreadsheet_obj, worksheet_name, df_to_write):
    """Writes a DataFrame to the specified Google Sheet worksheet."""
    if spreadsheet_obj is None or df_to_write is None:
        st.warning("Spreadsheet object or DataFrame to write is missing. Cannot write to Google Sheet.")
        return False
    try:
        worksheet = spreadsheet_obj.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        st.info(f"Worksheet '{worksheet_name}' not found. Creating it...")
        worksheet = spreadsheet_obj.add_worksheet(title=worksheet_name, rows=100, cols=len(df_to_write.columns) + 5) # Add some buffer cols
    
    worksheet.clear()
    # Convert NaT, NaN, None to empty strings for cleaner sheet output and ensure all data is string
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

# Initialize session state
if 'df_analyzed' not in st.session_state: st.session_state.df_analyzed = None
if 'spreadsheet_obj' not in st.session_state: st.session_state.spreadsheet_obj = None
if 'gc_obj' not in st.session_state: st.session_state.gc_obj = None


if final_creds:
    if st.sidebar.button("🔄 Load and Analyze Job Data", key="load_analyze_button"):
        with st.spinner("Loading data from Google Sheets..."):
            raw_df, spreadsheet, gc_instance = load_google_sheet(final_creds, SPREADSHEET_ID, DATA_WORKSHEET_NAME)
            st.session_state.spreadsheet_obj = spreadsheet
            st.session_state.gc_obj = gc_instance # Store gc for potential future use
        
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
            else:
                st.error("Data analysis resulted in an empty DataFrame or failed.")
        elif raw_df is not None and raw_df.empty:
             st.info("Source data sheet is empty. No analysis performed.")
        else:
            st.error("Failed to load data from Google Sheets.")
elif not creds_from_secrets:
    st.sidebar.info("Please upload your Google Service Account JSON key to begin.")


# --- Display Results and Write to 'todo' Sheet ---
if st.session_state.df_analyzed is not None and not st.session_state.df_analyzed.empty:
    df_display = st.session_state.df_analyzed.copy()
    st.header("🚩 Jobs Requiring Attention ('todo' List Preview)")
    
    if 'Flag_Overall_Needs_Attention' in df_display.columns:
        priority_jobs_df = df_display[df_display['Flag_Overall_Needs_Attention'] == True].copy()

        if not priority_jobs_df.empty:
            priority_jobs_df.loc[:, 'Primary Issue'] = priority_jobs_df.apply(determine_primary_issue, axis=1)
            if 'Install - Date' in priority_jobs_df.columns:
                priority_jobs_df.sort_values(by='Install - Date', ascending=True, na_position='last', inplace=True)
            
            display_cols_priority = ['Primary Issue', 'Job Name', 'Salesperson', 'Job Creation', 
                                     'Template - Date', 'Install - Date', 'Job Status']
            true_flag_cols_in_priority = [col for col in priority_jobs_df.columns if col.startswith('Flag_') and \
                                          col not in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 
                                                      'Flag_Any_PastDue_Activity', 'Flag_Overall_Needs_Attention', 'Primary Issue'] and \
                                          priority_jobs_df[col].any()]
            display_cols_priority.extend(sorted(true_flag_cols_in_priority))
            display_cols_priority = [col for col in display_cols_priority if col in priority_jobs_df.columns]

            st.dataframe(priority_jobs_df[display_cols_priority], height=600)
            st.markdown(f"**Total Jobs for 'todo' List: {len(priority_jobs_df)}**")

            if st.button("✍️ Update 'todo' Sheet in Google Sheets", key="update_todo_sheet_button"):
                if st.session_state.spreadsheet_obj:
                    with st.spinner(f"Writing to '{TODO_WORKSHEET_NAME}' sheet..."):
                        # Prepare only the necessary columns for export
                        export_df = priority_jobs_df[display_cols_priority].copy()
                        write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, export_df)
                else:
                    st.error("Spreadsheet object not available. Cannot write to Google Sheet. Try reloading data.")
        else:
            st.info("No jobs currently require attention based on the defined criteria.")
            # Optionally clear the 'todo' sheet if it's empty
            if st.session_state.spreadsheet_obj and st.button("Clear 'todo' Sheet (as no jobs to list)", key="clear_todo_button"):
                 with st.spinner(f"Clearing '{TODO_WORKSHEET_NAME}' sheet..."):
                    write_to_google_sheet(st.session_state.spreadsheet_obj, TODO_WORKSHEET_NAME, pd.DataFrame())


    # --- Detailed Breakdown Expander ---
    with st.expander("Show Detailed Flag Counts"):
        # ... (Detailed flag counts as in previous version) ...
        st.subheader("Threshold-Based Delays")
        if 'Flag_Awaiting_RTF' in df_display.columns: st.metric("Awaiting RTF (>2 days)", df_display['Flag_Awaiting_RTF'].sum())
        if 'Flag_Awaiting_Template' in df_display.columns: st.metric("Awaiting Template (>2 days)", df_display['Flag_Awaiting_Template'].sum())
        if 'Flag_Awaiting_Cutlist' in df_display.columns: st.metric("Awaiting Cutlist (>3 days)", df_display['Flag_Awaiting_Cutlist'].sum())
        # ... (add other flag counts here) ...

    if st.checkbox("Show Full Analyzed Data Table"):
        st.subheader("Full Analyzed Data")
        st.dataframe(df_display)

elif final_creds and st.session_state.df_analyzed is not None and st.session_state.df_analyzed.empty:
    st.info("Analysis complete, but the resulting dataset is empty (e.g. source sheet was empty or no jobs matched criteria).")

st.sidebar.markdown("---")
st.sidebar.info("Remember to configure `GOOGLE_CREDS_JSON` in Streamlit Secrets for deployed apps.")
