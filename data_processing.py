"""
Data Processing Module for FloForm Dashboard - CLEAN VERSION
Handles all data loading, cleaning, and preprocessing operations
Updated based on actual Excel column structure - NO SYNTAX ERRORS
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from business_logic import (
    parse_material,
    calculate_risk_score,
    calculate_delay_probability,
    analyze_job_pricing,
)

# --- Constants ---
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- CORRECTED Division Configuration Based on Actual Column Names ---
STONE_CONFIG = {
    "name": "Stone/Quartz",
    "numeric_map": {
        'Total Job Price $': 'Revenue',
        'Phase Throughput - Phase Plant Invoice': 'Cost_From_Plant',
        'Total Job SqFT': 'Total_Job_SqFt',
        'Job Throughput - Job GM (original)': 'Original_GM',
        'Rework - Stone Shop - Rework Price': 'Rework_Price',
        'Job Throughput - Rework COGS': 'Rework_COGS',
        'Job Throughput - Rework Job Labor': 'Rework_Labor',
        'Job Throughput - Total COGS': 'Total_COGS'
    },
    "cost_components": ['Cost_From_Plant', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price', 'Rework_COGS', 'Rework_Labor'],
    "has_shop_profit": True
}

LAMINATE_CONFIG = {
    "name": "Laminate",
    "numeric_map": {
        'Total Job Price $': 'Revenue',
        'Branch INV $': 'Shop_Cost',
        'Plant INV $': 'Material_Cost',
        'Total Job SqFT': 'Total_Job_SqFt',
        'Job Throughput - Job GM (original)': 'Original_GM',
        'Rework - Stone Shop - Rework Price': 'Rework_Price'
    },
    "cost_components": ['Shop_Cost', 'Material_Cost', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price'],
    "has_shop_profit": False
}

def clean_column_names(df):
    """Clean and standardize column names"""
    df.columns = df.columns.str.strip()
    return df

def add_missing_columns(df):
    """Add any missing expected columns with default values"""
    expected_cols = [
        'Template - Date', 'Ready to Fab - Date', 'Ship - Date', 'Install - Date',
        'Service - Date', 'Delivery - Date', 'Job Creation', 'Next Sched. - Date',
        'Product Rcvd - Date', 'Pick Up - Date', 'Invoice - Date', 'Plant INV - Date',
        'Job Material', 'Rework - Stone Shop - Rework Price', 'Production #',
        'Total Job Price $', 'Total Job SqFT', 'Job Throughput - Job GM (original)',
        'Salesperson', 'Division', 'Next Sched. - Activity', 'Install - Assigned To',
        'Template - Assigned To', 'Job Name', 'Rework - Stone Shop - Reason',
        'Ready to Fab - Status', 'Template - Status', 'Plant INV - Status',
        'Invoice - Status', 'Install - Status', 'Pick Up - Status', 'Delivery - Status',
        'Job Type', 'Order Type', 'Lead Source', 'Phase Throughput - Phase Plant Invoice',
        'Job Throughput - Job Plant Invoice',
        'Job Throughput - Rework COGS', 'Job Throughput - Rework Job Labor',
        'Job Throughput - Total COGS', 'Branch INV $', 'Plant INV $', 'Job Status'
    ]
    
    for col in expected_cols:
        if col not in df.columns: 
            df[col] = None
    
    return df

def process_date_columns(df):
    """Process and standardize all date columns"""
    date_cols = [
        'Template - Date', 'Ready to Fab - Date', 'Ship - Date', 'Install - Date',
        'Service - Date', 'Delivery - Date', 'Job Creation', 'Next Sched. - Date',
        'Product Rcvd - Date', 'Pick Up - Date', 'Invoice - Date', 'Plant INV - Date'
    ]
    
    for col in date_cols: 
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def create_standardized_column_names(df):
    """Create standardized column names that business_logic.py functions expect"""
    
    # Date columns
    df['Template_Date'] = df['Template - Date']
    df['Ready_to_Fab_Date'] = df['Ready to Fab - Date']
    df['Ship_Date'] = df['Ship - Date']
    df['Install_Date'] = df['Install - Date']
    df['Product_Rcvd_Date'] = df['Product Rcvd - Date']
    df['Pick_Up_Date'] = df['Pick Up - Date']
    df['Service_Date'] = df['Service - Date']
    df['Delivery_Date'] = df['Delivery - Date']
    df['Next_Sched_Date'] = df['Next Sched. - Date']
    df['Invoice_Date'] = df['Invoice - Date']
    df['Plant_INV_Date'] = df['Plant INV - Date']
    df['Job_Creation'] = df['Job Creation']
    
    # Other important columns for business logic
    df['Job_Material'] = df['Job Material']
    df['Total_Job_SqFT'] = df['Total Job SqFT']
    df['Total_Job_Price_'] = df['Total Job Price $']
    df['Phase_Dollars_Plant_Invoice_'] = df['Phase Throughput - Phase Plant Invoice']
    if 'Job Throughput - Job Plant Invoice' in df.columns:
        df['Job_Plant_Invoice'] = df['Job Throughput - Job Plant Invoice']
    else:
        df['Job_Plant_Invoice'] = df['Plant INV $']
    df['Job_Type'] = df['Job Type']
    df['Production_'] = df['Production #']
    df['Job_Name'] = df['Job Name']
    df['Next_Sched_Activity'] = df['Next Sched. - Activity']
    df['Install_Assigned_To'] = df['Install - Assigned To']
    df['Template_Assigned_To'] = df['Template - Assigned To']
    df['Rework_Stone_Shop_Rework_Price'] = df['Rework - Stone Shop - Rework Price']
    df['Rework_Stone_Shop_Reason'] = df['Rework - Stone Shop - Reason']
    df['Template_Status'] = df['Template - Status']
    df['Ready_to_Fab_Status'] = df['Ready to Fab - Status']
    df['Plant_INV_Status'] = df['Plant INV - Status']
    df['Invoice_Status'] = df['Invoice - Status']
    df['Install_Status'] = df['Install - Status']
    df['Job_Status'] = df['Job Status']
    df['Pick_Up_Status'] = df['Pick Up - Status']
    df['Delivery_Status'] = df['Delivery - Status']
    
    return df

def calculate_operational_metrics(df, today):
    """Calculate operational metrics for timeline and stage analysis"""
    
    # Create standardized column names first
    df = create_standardized_column_names(df)
    
    # Last activity tracking - Using standardized column names
    date_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 
        'Service_Date', 'Delivery_Date', 'Product_Rcvd_Date'
    ]
    df['Last_Activity_Date'] = df[date_cols].max(axis=1)
    df['Days_Since_Last_Activity'] = (today - df['Last_Activity_Date']).dt.days
    
    # Schedule tracking - Using standardized column name
    df['Days_Behind'] = np.where(
        df['Next_Sched_Date'].notna(), 
        (today - df['Next_Sched_Date']).dt.days, 
        np.nan
    )
    
    # Material parsing - Using standardized column name
    df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(
        lambda x: pd.Series(parse_material(str(x)))
    )
    
    # Moraware links - Using standardized column name
    df['Link'] = df['Production_'].apply(
        lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None
    )
    
    # Division classification
    df['Division_Type'] = df['Division'].apply(
        lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz'
    )
    
    return df

def calculate_stage_metrics(df, today):
    """Calculate stage and timeline metrics"""
    
    # Create standardized column names for the stage calculation function
    df = create_standardized_column_names(df)
    
    # Current stage determination using vectorized operations
    stage_conditions = [
        df['Install_Date'].notna() | df['Pick_Up_Date'].notna(),
        df['Ship_Date'].notna(),
        df['Product_Rcvd_Date'].notna(),
        df['Ready_to_Fab_Date'].notna(),
        df['Template_Date'].notna(),
    ]
    stage_choices = [
        'Completed',
        'Shipped',
        'Product Received',
        'In Fabrication',
        'Post-Template',
    ]
    df['Current_Stage'] = np.select(stage_conditions, stage_choices, default='Pre-Template')

    # Days spent in the current stage
    df['Days_In_Current_Stage'] = np.nan
    stage_to_date = {
        'Shipped': 'Ship_Date',
        'Product Received': 'Product_Rcvd_Date',
        'In Fabrication': 'Ready_to_Fab_Date',
        'Post-Template': 'Template_Date',
    }
    for stage, col in stage_to_date.items():
        mask = df['Current_Stage'] == stage
        df.loc[mask, 'Days_In_Current_Stage'] = (today - df.loc[mask, col]).dt.days
    df.loc[df['Current_Stage'] == 'Completed', 'Days_In_Current_Stage'] = 0
    
    # Timeline calculations - Using standardized column names
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
    df['Days_RTF_to_Product_Rcvd'] = (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days
    df['Days_Product_Rcvd_to_Install'] = (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days
    df['Days_Template_to_Ship'] = (df['Ship_Date'] - df['Template_Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
    
    return df

def calculate_quality_metrics(df):
    """Calculate quality and rework metrics"""
    
    # Rework identification - Using standardized column name
    df['Has_Rework'] = (
        df['Rework_Stone_Shop_Rework_Price'].notna() & 
        (df['Rework_Stone_Shop_Rework_Price'] != '')
    )
    
    return df

def calculate_risk_metrics(df):
    """Calculate risk scores and delay probabilities"""
    
    # Risk score calculation
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    
    # Delay probability calculation
    df[['Delay_Probability', 'Risk_Factors']] = df.apply(
        lambda row: pd.Series(calculate_delay_probability(row)), axis=1
    )
    
    return df

def calculate_close_out_metrics(df, today):
    """Compute metrics to identify jobs ready for billing and overdue items."""

    df = create_standardized_column_names(df)

    # Days since key milestones
    df['Days_Since_Install'] = (today - df['Install_Date']).dt.days
    df['Days_Since_Job_Creation'] = (today - df['Job_Creation']).dt.days

    # Overdue flag based on job age
    df['Is_Overdue'] = df['Days_Since_Job_Creation'] > 30

    incomplete_status = ['estimate', 'auto-schedule', 'no date', '']

    # Ready for billing when install complete but invoice incomplete
    install_complete = df['Install_Status'].str.lower().isin(['complete', 'confirmed'])
    invoice_incomplete = df['Invoice_Status'].fillna('').str.lower().isin(incomplete_status)
    df['Ready_For_Billing'] = install_complete & invoice_incomplete

    # Phase escalation when downstream is complete but upstream or invoice incomplete
    plant_incomplete = df['Plant_INV_Status'].fillna('').str.lower().isin(incomplete_status)
    df['Needs_Escalation'] = install_complete & (plant_incomplete | invoice_incomplete)

    # Missing required dates
    phase_date_pairs = [
        ('Template_Status', 'Template_Date', 'Template'),
        ('Ready_to_Fab_Status', 'Ready_to_Fab_Date', 'Ready to Fab'),
        ('Plant_INV_Status', 'Plant_INV_Date', 'Plant INV'),
        ('Install_Status', 'Install_Date', 'Install'),
        ('Invoice_Status', 'Invoice_Date', 'Invoice')
    ]

    def get_missing_phases(row):
        missing = [name for status_col, date_col, name in phase_date_pairs
                  if str(row.get(status_col, '')).lower() in ['complete', 'confirmed']
                  and pd.isna(row.get(date_col))]
        return ', '.join(missing)

    df['Missing_Dates'] = df.apply(get_missing_phases, axis=1)
    df['Has_Missing_Dates'] = df['Missing_Dates'] != ''

    # Phase summary for quick review
    summary_fields = [
        ('Template', 'Template_Status'),
        ('Ready to Fab', 'Ready_to_Fab_Status'),
        ('Plant INV', 'Plant_INV_Status'),
        ('Install', 'Install_Status'),
        ('Invoice', 'Invoice_Status')
    ]

    df['Phase_Summary'] = df.apply(
        lambda row: ' | '.join(
            f"{name}:{row.get(col, '')}" for name, col in summary_fields if row.get(col)
        ),
        axis=1
    )

    return df

def perform_pricing_analysis(df):
    """Perform comprehensive pricing analysis on all jobs"""
    
    with st.spinner("Analyzing job pricing and material recognition..."):
        pricing_analysis_series = df.apply(analyze_job_pricing, axis=1)
        
        def get_metric(analysis, key, default=None): 
            return analysis.get(key, default) if isinstance(analysis, dict) else default
        
        # Extract pricing analysis results
        df['Pricing_Analysis'] = pricing_analysis_series
        
        df['Material_Group'] = pricing_analysis_series.apply(
            lambda x: get_metric(x.get('expected_retail', {}), 'material_group') 
            if isinstance(x, dict) else None
        )
        
        df['Material_Type'] = pricing_analysis_series.apply(
            lambda x: get_metric(x, 'material_type')
        )
        
        df['Pricing_Issues_Count'] = pricing_analysis_series.apply(
            lambda x: get_metric(x, 'critical_issues', 0)
        )
        
        df['Pricing_Warnings_Count'] = pricing_analysis_series.apply(
            lambda x: get_metric(x, 'warnings', 0)
        )
        
        df['Revenue_Variance'] = pricing_analysis_series.apply(
            lambda x: get_metric(x, 'revenue_variance', 0)
        )
        
        df['Cost_Variance'] = pricing_analysis_series.apply(
            lambda x: get_metric(x, 'plant_cost_variance', 0)
        )
    
    return df

def process_financial_data(df, config, install_cost_per_sqft):
    """Process financial data for a specific division"""
    
    df_processed = df.copy()
    
    # Process numeric columns with error handling
    for original, new in config["numeric_map"].items():
        if original in df_processed.columns:
            df_processed[new] = pd.to_numeric(
                df_processed[original].astype(str).str.replace(r'[$,%]', '', regex=True), 
                errors='coerce'
            ).fillna(0)
        else: 
            df_processed[new] = 0.0
    
    # Calculate costs and profits
    df_processed['Install_Cost'] = df_processed.get('Total_Job_SqFt', 0) * install_cost_per_sqft
    df_processed['Total_Rework_Cost'] = sum([
        df_processed.get(c, 0) for c in config["rework_components"]
    ])
    df_processed['Total_Branch_Cost'] = sum([
        df_processed.get(c, 0) for c in config["cost_components"]
    ])
    
    # Branch profit calculations
    revenue = df_processed.get('Revenue', 0)
    df_processed['Branch_Profit'] = revenue - df_processed['Total_Branch_Cost']
    df_processed['Branch_Profit_Margin_%'] = np.where(
        revenue != 0, 
        (df_processed['Branch_Profit'] / revenue * 100), 
        0
    )
    df_processed['Profit_Variance'] = (
        df_processed['Branch_Profit'] - df_processed.get('Original_GM', 0)
    )
    
    # Shop profit calculations (for stone/quartz only)
    if config["has_shop_profit"]:
        cost_from_plant = df_processed.get('Cost_From_Plant', 0)
        total_cogs = df_processed.get('Total_COGS', 0)
        df_processed['Shop_Profit'] = cost_from_plant - total_cogs
        df_processed['Shop_Profit_Margin_%'] = np.where(
            cost_from_plant != 0, 
            (df_processed['Shop_Profit'] / cost_from_plant * 100), 
            0
        )
    
    return df_processed

@st.cache_data(ttl=300, show_spinner="Loading job data from Google Sheets...")
def load_raw_data():
    """Load raw data from Google Sheets using gspread"""
    
    try:
        # Setup Google Sheets connection using gspread
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Ensure credentials are available in Streamlit secrets
        if "gsheets" not in st.secrets:
            st.error("Google Sheets credentials not configured.")
            st.info(
                "Add a [gsheets] section to your Streamlit secrets. "
                "See https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management "
                "for more information."
            )
            return pd.DataFrame()

        # Get credentials from Streamlit secrets
        credentials_dict = dict(st.secrets["gsheets"])
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        
        # Authorize and create client
        client = gspread.authorize(credentials)
        
        # Open spreadsheet by URL
        spreadsheet = client.open_by_url(st.secrets["gsheets"]["spreadsheet"])
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        
        # Get all records as list of dictionaries
        data = worksheet.get_all_records()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            st.warning("No data received from Google Sheets.")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        st.info("Please check your Google Sheets connection and credentials.")
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.write("Error details:", str(e))
            st.write("Secrets available:", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets found")
        
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Processing and analyzing job data...")
def process_loaded_data(df, today, install_cost):
    """Process the loaded data through all transformation steps"""
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Step 1: Clean and standardize
    df = clean_column_names(df)
    df = add_missing_columns(df)
    df = process_date_columns(df)
    
    # Step 2: Calculate operational metrics
    df = calculate_operational_metrics(df, today)
    df = calculate_stage_metrics(df, today)
    df = calculate_quality_metrics(df)
    df = calculate_risk_metrics(df)
    df = calculate_close_out_metrics(df, today)
    
    # Step 3: Perform pricing analysis
    df = perform_pricing_analysis(df)
    
    # Step 4: Process division-specific financial data
    df_stone = df[df['Division_Type'] == 'Stone/Quartz'].copy()
    df_laminate = df[df['Division_Type'] == 'Laminate'].copy()
    
    df_stone_processed = process_financial_data(df_stone, STONE_CONFIG, install_cost)
    df_laminate_processed = process_financial_data(df_laminate, LAMINATE_CONFIG, install_cost)
    df_combined = pd.concat([df_stone_processed, df_laminate_processed], ignore_index=True)
    
    return df_stone_processed, df_laminate_processed, df_combined

def load_and_process_data(today, install_cost):
    """Main function to load and process all data with comprehensive error handling"""
    
    # Load raw data
    raw_df = load_raw_data()
    
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Process the data
    try:
        df_stone, df_laminate, df_combined = process_loaded_data(raw_df, today, install_cost)
        
        # Data quality checks
        total_jobs = len(df_combined)
        if total_jobs == 0:
            st.warning("No jobs found after processing.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        return df_stone, df_laminate, df_combined
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        with st.expander("🔍 Processing Error Details"):
            st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def get_data_summary(df):
    """Generate a comprehensive data summary for display"""
    
    if df.empty:
        return "No data available"
    
    summary = {
        'total_jobs': len(df),
        'divisions': df.get('Division_Type', pd.Series()).value_counts().to_dict(),
        'active_jobs': len(df[df.get('Job_Status', '') != 'Complete']),
        'completed_jobs': len(df[df.get('Job_Status', '') == 'Complete']),
        'total_revenue': df.get('Revenue', pd.Series([0])).sum(),
        'avg_job_value': df.get('Revenue', pd.Series([0])).mean(),
        'high_risk_jobs': len(df[df.get('Risk_Score', 0) >= 30]),
        'jobs_with_rework': len(df[df.get('Has_Rework', False) == True])
    }
    
    return summary

def filter_data(df, filters):
    """Apply filters to the dataset efficiently"""
    
    if df.empty:
        return df
    
    mask = pd.Series(True, index=df.index)

    # Status filters - Using standardized column names
    if filters.get('status_filter'):
        status_mask = pd.Series(False, index=df.index)

        if "Active" in filters['status_filter']:
            status_mask |= (df.get('Job_Status', '') != 'Complete')
        if "Complete" in filters['status_filter']:
            status_mask |= (df.get('Job_Status', '') == 'Complete')
        if "30+ Days Old" in filters['status_filter']:
            thirty_days_ago = pd.Timestamp.now() - timedelta(days=30)
            status_mask |= (
                (df.get('Job_Creation', pd.NaT) < thirty_days_ago) &
                (df.get('Job_Status', '') != 'Complete')
            )
        if "Unscheduled" in filters['status_filter']:
            status_mask |= (
                df.get('Next_Sched_Date', pd.NaT).isna() &
                (df.get('Job_Status', '') != 'Complete')
            )

        mask &= status_mask

    # Salesperson filter
    if filters.get('salesperson') and filters['salesperson'] != 'All':
        mask &= df.get('Salesperson', '') == filters['salesperson']

    # Division filter
    if filters.get('division') and filters['division'] != 'All':
        mask &= df.get('Division_Type', '') == filters['division']

    # Date range filter - Using standardized column name
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        mask &= (
            (df.get('Job_Creation', pd.NaT) >= start_date) &
            (df.get('Job_Creation', pd.NaT) <= end_date)
        )

    # Text search filter
    if filters.get('search_query'):
        query = str(filters['search_query']).strip().lower()
        if query:
            name_match = df.get('Job_Name', '').str.lower().str.contains(query, na=False)
            po_match = df.get('Production_', '').astype(str).str.contains(query, na=False)
            mask &= name_match | po_match

    return df[mask]

def export_data_summary(df, filename_prefix="floform_data"):
    """Generate data export summary for download"""
    
    if df.empty:
        return None
    
    # Create export timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Select key columns for export - Using standardized column names
    export_columns = [
        'Job_Name', 'Production_', 'Salesperson', 'Division_Type', 'Current_Stage',
        'Revenue', 'Total_Job_SqFt', 'Branch_Profit_Margin_%', 'Risk_Score',
        'Template_Date', 'Install_Date', 'Material_Type', 'Material_Group'
    ]
    
    # Filter to available columns
    available_columns = [col for col in export_columns if col in df.columns]
    export_df = df[available_columns].copy()
    
    # Format dates for export
    date_columns = ['Template_Date', 'Install_Date']
    for col in date_columns:
        if col in export_df.columns:
            export_df[col] = export_df[col].dt.strftime('%Y-%m-%d')
    
    return export_df.to_csv(index=False), filename
