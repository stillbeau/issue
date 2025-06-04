import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from io import StringIO

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Issue Detector", page_icon="⚙️")

# --- App Title ---
st.title("⚙️ Job Issue Detector Dashboard")
st.markdown("This app analyzes job data from a Google Sheet to identify potential delays, keyword issues in notes, and past due activities.")

# --- Constants & Configuration ---
# These would ideally be managed via secrets in a deployed app
# For local development, you can set them here or use Streamlit's secrets management
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38" # YOUR GOOGLE SHEET ID
WORKSHEET_NAME = "jobs" # YOUR WORKSHEET/TAB NAME (original data)
# TODO_WORKSHEET_NAME = "todo" # For future implementation if writing back

# --- Helper Functions from Colab (Refactored) ---

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_google_sheet(creds_json_str, spreadsheet_id, worksheet_name):
    """Loads data from a Google Sheet into a pandas DataFrame."""
    try:
        creds_dict = creds_json_str # Already a dict if parsed from st.file_uploader
        creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_values()
        if not data:
            st.error("Error: The main data worksheet appears to be empty.")
            return None
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return None

def preprocess_data(df):
    """Converts date columns and performs other necessary preprocessing."""
    if df is None:
        return None
    
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
            df_processed[col_name] = pd.to_datetime(df_processed[col_name], errors='coerce')
    return df_processed

def flag_threshold_delays(df, current_date_str):
    """Flags jobs based on threshold-based workflow delays."""
    if df is None:
        return df
    
    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)

    # Rule 1: Template Done, Awaiting RTF (Threshold: 2 days)
    df_flagged['Flag_Awaiting_RTF'] = False
    if 'Template - Date' in df_flagged.columns and 'Ready to Fab - Date' in df_flagged.columns:
        temp_dates_for_rtf_check = df_flagged.loc[df_flagged['Template - Date'].notna() & df_flagged['Ready to Fab - Date'].isna(), 'Template - Date']
        if pd.api.types.is_datetime64_any_dtype(temp_dates_for_rtf_check) and not temp_dates_for_rtf_check.empty:
            condition_rtf_pending = df_flagged['Template - Date'].notna() & df_flagged['Ready to Fab - Date'].isna()
            df_flagged.loc[condition_rtf_pending, 'Flag_Awaiting_RTF'] = \
                (current_date - df_flagged.loc[condition_rtf_pending, 'Template - Date']).dt.days > 2

    # Rule 2: Job Created, Awaiting Template (Threshold: 2 days)
    df_flagged['Flag_Awaiting_Template'] = False
    if 'Job Creation' in df_flagged.columns and 'Template - Date' in df_flagged.columns:
        job_creation_dates_for_template_check = df_flagged.loc[df_flagged['Job Creation'].notna() & df_flagged['Template - Date'].isna(), 'Job Creation']
        if pd.api.types.is_datetime64_any_dtype(job_creation_dates_for_template_check) and not job_creation_dates_for_template_check.empty:
            condition_template_pending = df_flagged['Job Creation'].notna() & df_flagged['Template - Date'].isna()
            df_flagged.loc[condition_template_pending, 'Flag_Awaiting_Template'] = \
                (current_date - df_flagged.loc[condition_template_pending, 'Job Creation']).dt.days > 2
    
    # Rule 3: Ready to Fab, Awaiting Cutlist (Threshold: 3 days)
    df_flagged['Flag_Awaiting_Cutlist'] = False
    if 'Ready to Fab - Date' in df_flagged.columns and 'Cutlist - Date' in df_flagged.columns:
        rtf_dates_for_cutlist_check = df_flagged.loc[df_flagged['Ready to Fab - Date'].notna() & df_flagged['Cutlist - Date'].isna(), 'Ready to Fab - Date']
        if pd.api.types.is_datetime64_any_dtype(rtf_dates_for_cutlist_check) and not rtf_dates_for_cutlist_check.empty:
            condition_cutlist_pending = df_flagged['Ready to Fab - Date'].notna() & df_flagged['Cutlist - Date'].isna()
            df_flagged.loc[condition_cutlist_pending, 'Flag_Awaiting_Cutlist'] = \
                (current_date - df_flagged.loc[condition_cutlist_pending, 'Ready to Fab - Date']).dt.days > 3

    threshold_flag_cols = ['Flag_Awaiting_RTF', 'Flag_Awaiting_Template', 'Flag_Awaiting_Cutlist']
    threshold_flag_cols = [col for col in threshold_flag_cols if col in df_flagged.columns]
    if threshold_flag_cols:
        df_flagged['Flag_Any_Threshold_Delay'] = df_flagged[threshold_flag_cols].any(axis=1)
    return df_flagged

def flag_keyword_issues(df):
    """Analyzes notes columns for keywords."""
    if df is None:
        return df
        
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
        if pd.isna(note_text) or not isinstance(note_text, str):
            return False
        return bool(re.search(pattern, note_text, re.IGNORECASE))

    notes_columns_to_scan = [
        'Template - Notes', 'Install - Notes', 'Saw - Notes',
        'Job Issues', 'QC - Notes'
    ]
    keyword_flag_columns_generated = []
    for col_name in notes_columns_to_scan:
        if col_name in df_flagged.columns:
            new_flag_col_name = f'Flag_Keyword_In_{col_name.replace(" - ", "_").replace(" ", "_")}'
            df_flagged[new_flag_col_name] = df_flagged[col_name].apply(lambda note: find_keywords_in_note(note, keyword_pattern))
            keyword_flag_columns_generated.append(new_flag_col_name)
            
    if keyword_flag_columns_generated:
        df_flagged['Flag_Any_Keyword_Issue'] = df_flagged[keyword_flag_columns_generated].any(axis=1)
    return df_flagged

def flag_past_due_activities(df, current_date_str):
    """Flags activities that are past their scheduled date and not complete/cancelled."""
    if df is None:
        return df

    df_flagged = df.copy()
    current_date = pd.Timestamp(current_date_str)
    completion_terms = [
        'complete', 'completed', 'done', 'installed', 'invoiced',
        'paid', 'sent', 'received', 'closed', 'fabricated'
    ]
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
        else:
            continue

        new_flag_col = f'Flag_PastDue_{activity_name_for_flag}'
        df_flagged[new_flag_col] = False

        if date_col not in df_flagged.columns or status_col not in df_flagged.columns or not pd.api.types.is_datetime64_any_dtype(df_flagged[date_col]):
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
    return df_flagged

def determine_primary_issue(row):
    """Determines a primary issue string based on a hierarchy of flags."""
    if row.get('Flag_PastDue_Install', False): return "Past Due: Install"
    if row.get('Flag_PastDue_Polish_Fab_Completion', False): return "Past Due: Polish/Fab"
    if row.get('Flag_PastDue_Saw', False): return "Past Due: Saw"
    if row.get('Flag_PastDue_RTF', False): return "Past Due: RTF"
    if row.get('Flag_PastDue_Invoice', False): return "Past Due: Invoice"
    if row.get('Flag_PastDue_Collect_Final', False): return "Past Due: Collect Final"
    if row.get('Flag_PastDue_Template', False): return "Past Due: Template"
    if row.get('Flag_Awaiting_Cutlist', False): return "Delay: Awaiting Cutlist"
    if row.get('Flag_Awaiting_RTF', False): return "Delay: Awaiting RTF"
    if row.get('Flag_Awaiting_Template', False): return "Delay: Awaiting Template"
    if row.get('Flag_Keyword_In_Install_Notes', False): return "Keyword: Install Notes"
    if row.get('Flag_Keyword_In_Template_Notes', False): return "Keyword: Template Notes"
    if row.get('Flag_Keyword_In_Job_Issues', False): return "Keyword: Job Issues"
    if row.get('Flag_Keyword_In_QC_Notes', False): return "Keyword: QC Notes"
    if row.get('Flag_Keyword_In_Saw_Notes', False): return "Keyword: Saw Notes"
    return "Other Issue"

# --- Main App Logic ---

# Sidebar for Inputs
st.sidebar.header("⚙️ Configuration")

# Google Service Account Credentials
st.sidebar.subheader("Google Sheets Credentials")
st.sidebar.markdown("Upload your Google Cloud Service Account JSON key file. This file is not stored after your session ends.")
uploaded_file = st.sidebar.file_uploader("Upload Service Account JSON", type="json")

# Current date for calculations
# Using Streamlit's date_input, defaulting to June 4, 2025, as per Colab
default_calc_date = pd.Timestamp('2025-06-04').date() # Convert to datetime.date object
current_calc_date_input = st.sidebar.date_input("Date for Calculations (Past Due/Delays)", value=default_calc_date)
current_calc_date_str = current_calc_date_input.strftime('%Y-%m-%d')


# Initialize session state for DataFrame
if 'df_analyzed' not in st.session_state:
    st.session_state.df_analyzed = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

if uploaded_file is not None:
    # To read file as string:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    creds_json_str = stringio.read() # This is a string
    
    # Convert string to dictionary for gspread
    import json
    try:
        creds_dict = json.loads(creds_json_str)
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid Google Service Account key file.")
        creds_dict = None

    if creds_dict:
        if st.button("Load and Analyze Job Data"):
            with st.spinner("Loading data from Google Sheets..."):
                st.session_state.raw_df = load_google_sheet(creds_dict, SPREADSHEET_ID, WORKSHEET_NAME)
            
            if st.session_state.raw_df is not None:
                st.success(f"Successfully loaded {len(st.session_state.raw_df)} jobs from '{WORKSHEET_NAME}'.")
                
                with st.spinner("Preprocessing data..."):
                    df_processed = preprocess_data(st.session_state.raw_df)
                
                with st.spinner("Flagging threshold-based delays..."):
                    df_threshold_flagged = flag_threshold_delays(df_processed, current_calc_date_str)
                
                with st.spinner("Analyzing notes for keywords..."):
                    df_keyword_flagged = flag_keyword_issues(df_threshold_flagged)
                
                with st.spinner("Flagging past due activities..."):
                    df_past_due_flagged = flag_past_due_activities(df_keyword_flagged, current_calc_date_str)
                
                st.session_state.df_analyzed = df_past_due_flagged

                if st.session_state.df_analyzed is not None:
                    # Create Overall Needs Attention Flag
                    all_flag_summary_cols = []
                    if 'Flag_Any_Threshold_Delay' in st.session_state.df_analyzed.columns: all_flag_summary_cols.append('Flag_Any_Threshold_Delay')
                    if 'Flag_Any_Keyword_Issue' in st.session_state.df_analyzed.columns: all_flag_summary_cols.append('Flag_Any_Keyword_Issue')
                    if 'Flag_Any_PastDue_Activity' in st.session_state.df_analyzed.columns: all_flag_summary_cols.append('Flag_Any_PastDue_Activity')

                    if all_flag_summary_cols:
                        st.session_state.df_analyzed['Flag_Overall_Needs_Attention'] = st.session_state.df_analyzed[all_flag_summary_cols].any(axis=1)
                    else:
                        st.session_state.df_analyzed['Flag_Overall_Needs_Attention'] = False
                    
                    st.success("Data analysis complete!")
                else:
                    st.error("Data analysis failed after loading.")
            else:
                st.error("Failed to load data from Google Sheets.")
else:
    st.sidebar.info("Please upload your Google Service Account JSON key to begin.")


# --- Display Results ---
if st.session_state.df_analyzed is not None:
    df_display = st.session_state.df_analyzed.copy()

    st.header("🚩 Jobs Requiring Attention")
    
    priority_jobs_df = df_display[df_display['Flag_Overall_Needs_Attention'] == True].copy()

    if not priority_jobs_df.empty:
        priority_jobs_df['Primary Issue'] = priority_jobs_df.apply(determine_primary_issue, axis=1)
        
        # Sort by 'Install - Date'
        if 'Install - Date' in priority_jobs_df.columns:
            priority_jobs_df.sort_values(by='Install - Date', ascending=True, na_position='last', inplace=True)
        
        # Columns to display in the Streamlit table
        display_cols_priority = ['Primary Issue', 'Job Name', 'Salesperson', 'Job Creation', 
                                 'Template - Date', 'Install - Date', 'Job Status']
        
        # Add individual flag columns that are True for any job in the priority list
        true_flag_cols_in_priority = []
        for col in priority_jobs_df.columns:
            if col.startswith('Flag_') and col not in ['Flag_Any_Threshold_Delay', 'Flag_Any_Keyword_Issue', 
                                                      'Flag_Any_PastDue_Activity', 'Flag_Overall_Needs_Attention', 'Primary Issue']:
                if priority_jobs_df[col].any(): # Only include if at least one job has this flag True
                    true_flag_cols_in_priority.append(col)
        
        display_cols_priority.extend(sorted(true_flag_cols_in_priority))
        display_cols_priority = [col for col in display_cols_priority if col in priority_jobs_df.columns] # Ensure existence

        st.dataframe(priority_jobs_df[display_cols_priority], height=600)
        
        st.markdown(f"**Total Jobs Requiring Attention: {len(priority_jobs_df)}**")

        # --- Detailed Breakdown of Flags (Optional Expander) ---
        with st.expander("Show Detailed Flag Counts & Samples"):
            st.subheader("Threshold-Based Delays")
            if 'Flag_Awaiting_RTF' in df_display.columns:
                st.metric("Awaiting RTF (>2 days)", df_display['Flag_Awaiting_RTF'].sum())
            if 'Flag_Awaiting_Template' in df_display.columns:
                st.metric("Awaiting Template (>2 days)", df_display['Flag_Awaiting_Template'].sum())
            if 'Flag_Awaiting_Cutlist' in df_display.columns:
                st.metric("Awaiting Cutlist (>3 days)", df_display['Flag_Awaiting_Cutlist'].sum())

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
                if flag_col in df_display.columns:
                    st.write(f"*Past Due '{friendly_name}'*: {df_display[flag_col].sum()} jobs")
    else:
        st.info("No jobs currently require attention based on the defined criteria.")

    # --- Option to view Raw Data (for debugging or inspection) ---
    if st.checkbox("Show Full Analyzed Data Table (with all flags)"):
        st.subheader("Full Analyzed Data")
        st.dataframe(df_display)

else:
    if uploaded_file is not None and st.session_state.raw_df is None and st.session_state.df_analyzed is None:
        # This case might happen if loading failed after button press but before df_analyzed is set
        st.warning("Data loading or analysis may have failed. Please check for error messages above.")
    elif uploaded_file is None:
        st.info("Upload your Google Service Account JSON key in the sidebar and click 'Load and Analyze Job Data'.")


st.sidebar.markdown("---")
st.sidebar.markdown("Developed with the help of an AI assistant.")
