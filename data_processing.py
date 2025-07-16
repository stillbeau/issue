"""
Data Processing Module for FloForm Dashboard - Fixed for Python 3.13
Handles all data loading, cleaning, and preprocessing operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from business_logic import (
    parse_material, get_current_stage, calculate_days_in_stage,
    calculate_risk_score, calculate_delay_probability, analyze_job_pricing
)

# --- Constants ---
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Division Configuration ---
STONE_CONFIG = {
    "name": "Stone/Quartz",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue', 
        'Phase_Dollars_Plant_Invoice_': 'Cost_From_Plant',
        'Total_Job_SqFT': 'Total_Job_SqFt', 
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price', 
        'Job_Throughput_Rework_COGS': 'Rework_COGS',
        'Job_Throughput_Rework_Job_Labor': 'Rework_Labor', 
        'Job_Throughput_Total_COGS': 'Total_COGS'
    },
    "cost_components": ['Cost_From_Plant', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price', 'Rework_COGS', 'Rework_Labor'],
    "has_shop_profit": True
}

LAMINATE_CONFIG = {
    "name": "Laminate",
    "numeric_map": {
        'Total_Job_Price_': 'Revenue', 
        'Branch_INV_': 'Shop_Cost', 
        'Plant_INV_': 'Material_Cost',
        'Total_Job_SqFT': 'Total_Job_SqFt', 
        'Job_Throughput_Job_GM_original': 'Original_GM',
        'Rework_Stone_Shop_Rework_Price': 'Rework_Price',
    },
    "cost_components": ['Shop_Cost', 'Material_Cost', 'Install_Cost', 'Total_Rework_Cost'],
    "rework_components": ['Rework_Price'],
    "has_shop_profit": False
}

def clean_column_names(df):
    """Clean and standardize column names."""
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r'[\s-]+', '_', regex=True)
                  .str.replace(r'[^\w]', '', regex=True))
    return df

def add_missing_columns(df):
    """Add any missing expected columns with default values."""
    expected_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 
        'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 
        'Product_Rcvd_Date', 'Pick_Up_Date', 'Job_Material', 'Rework_Stone_Shop_Rework_Price', 
        'Production_', 'Total_Job_Price_', 'Total_Job_SqFT', 'Job_Throughput_Job_GM_original', 
        'Salesperson', 'Division', 'Next_Sched_Activity', 'Install_Assigned_To', 
        'Template_Assigned_To', 'Job_Name', 'Rework_Stone_Shop_Reason', 'Ready_to_Fab_Status', 
        'Job_Type', 'Order_Type', 'Lead_Source', 'Phase_Dollars_Plant_Invoice_',
        'Job_Throughput_Rework_COGS', 'Job_Throughput_Rework_Job_Labor', 
        'Job_Throughput_Total_COGS', 'Branch_INV_', 'Plant_INV_', 'Job_Status', 
        'Invoice_Status', 'Install_Status', 'Pick_Up_Status', 'Delivery_Status',
        
        # Stone Details columns (for enhanced pricing analysis)
        'Stone_Details_Name', 'Stone_Details_Phase', 'Stone_Details_Product', 
        'Stone_Details_Colour', 'Stone_Details_Edge_Detail', 'Stone_Details_Thickness',
        'Stone_Details_Finish', 'Stone_Details_Special_Color', 'Stone_Details_Priced',
        'Stone_Details_Sq_Ft', 'Stone_Details_Edge_LF',
        
        # Phase Dollars columns (for plant invoice tracking)
        'Phase_Dollars_Name', 'Phase_Dollars_Phase', 'Phase_Dollars_Plant_Invoice_'
    ]
    
    for col in expected_cols:
        if col not in df.columns: 
            df[col] = None
    
    return df

def process_date_columns(df):
    """Process and standardize all date columns."""
    date_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 
        'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date', 
        'Product_Rcvd_Date', 'Pick_Up_Date'
    ]
    
    for col in date_cols: 
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def calculate_operational_metrics(df, today):
    """Calculate operational metrics for timeline and stage analysis."""
    
    # Last activity tracking
    date_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date', 
        'Service_Date', 'Delivery_Date', 'Product_Rcvd_Date'
    ]
    df['Last_Activity_Date'] = df[date_cols].max(axis=1)
    df['Days_Since_Last_Activity'] = (today - df['Last_Activity_Date']).dt.days
    
    # Schedule tracking
    df['Days_Behind'] = np.where(
        df['Next_Sched_Date'].notna(), 
        (today - df['Next_Sched_Date']).dt.days, 
        np.nan
    )
    
    # Material parsing
    df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(
        lambda x: pd.Series(parse_material(str(x)))
    )
    
    # Moraware links
    df['Link'] = df['Production_'].apply(
        lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None
    )
    
    # Division classification
    df['Division_Type'] = df['Division'].apply(
        lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz'
    )
    
    return df

def calculate_stage_metrics(df, today):
    """Calculate stage and timeline metrics."""
    
    # Current stage determination
    df['Current_Stage'] = df.apply(get_current_stage, axis=1)
    df['Days_In_Current_Stage'] = df.apply(
        lambda row: calculate_days_in_stage(row, today), axis=1
    )
    
    # Timeline calculations
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
    df['Days_RTF_to_Product_Rcvd'] = (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days
    df['Days_Product_Rcvd_to_Install'] = (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days
    df['Days_Template_to_Ship'] = (df['Ship_Date'] - df['Template_Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
    
    return df

def calculate_quality_metrics(df):
    """Calculate quality and rework metrics."""
    
    # Rework identification
    df['Has_Rework'] = (
        df['Rework_Stone_Shop_Rework_Price'].notna() & 
        (df['Rework_Stone_Shop_Rework_Price'] != '')
    )
    
    return df

def calculate_risk_metrics(df):
    """Calculate risk scores and delay probabilities."""
    
    # Risk score calculation
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    
    # Delay probability calculation
    df[['Delay_Probability', 'Risk_Factors']] = df.apply(
        lambda row: pd.Series(calculate_delay_probability(row)), axis=1
    )
    
    return df

def perform_pricing_analysis(df):
    """Perform comprehensive pricing analysis on all jobs."""
    
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
    """Process financial data for a specific division."""
    
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
    """Load raw data from Google Sheets using gspread (compatible with Python 3.13)."""
    
    try:
        # Setup Google Sheets connection using gspread
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
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
        
        st.success(f"✅ Successfully loaded {len(df)} rows from Google Sheets")
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
    """Process the loaded data through all transformation steps."""
    
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
    """Main function to load and process all data with comprehensive error handling."""
    
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
        
        # Log processing summary
        st.success(f"✅ Successfully processed {total_jobs} jobs")
        
        if 'Pricing_Analysis' in df_combined.columns:
            pricing_issues = df_combined['Pricing_Issues_Count'].sum()
            pricing_warnings = df_combined['Pricing_Warnings_Count'].sum()
            
            if pricing_issues > 0 or pricing_warnings > 0:
                st.info(f"🔍 Pricing Analysis: {pricing_issues} critical issues, {pricing_warnings} warnings")
        
        return df_stone, df_laminate, df_combined
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        with st.expander("🔍 Processing Error Details"):
            st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def get_data_summary(df):
    """Generate a comprehensive data summary for display."""
    
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
    """Apply filters to the dataset efficiently."""
    
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Status filters
    if filters.get('status_filter'):
        status_mask = pd.Series([False] * len(df), index=df.index)
        
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
        
        filtered_df = df[status_mask]
    
    # Salesperson filter
    if filters.get('salesperson') and filters['salesperson'] != 'All':
        filtered_df = filtered_df[
            filtered_df.get('Salesperson', '') == filters['salesperson']
        ]
    
    # Division filter
    if filters.get('division') and filters['division'] != 'All':
        filtered_df = filtered_df[
            filtered_df.get('Division_Type', '') == filters['division']
        ]
    
    # Date range filter
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df.get('Job_Creation', pd.NaT) >= start_date) & 
            (filtered_df.get('Job_Creation', pd.NaT) <= end_date)
        ]
    
    return filtered_df

def export_data_summary(df, filename_prefix="floform_data"):
    """Generate data export summary for download."""
    
    if df.empty:
        return None
    
    # Create export timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Select key columns for export
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
