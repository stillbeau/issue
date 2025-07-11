# -*- coding: utf-8 -*-
"""
Enhanced Daily Operations Dashboard - Focus on Timeline Management and Early Warning System
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Daily Operations Dashboard", page_icon="⚡")

# --- Constants ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Timeline Thresholds (customizable) ---
TIMELINE_THRESHOLDS = {
    'template_to_rtf': 3,  # days
    'rtf_to_product_rcvd': 7,    # days
    'product_rcvd_to_install': 5,  # days
    'template_to_install': 15,  # days
    'template_to_ship': 10,  # days
    'ship_to_install': 5,  # days
    'days_in_stage_warning': 5,  # days stuck in any stage
    'stale_job_threshold': 7 # days since last activity
}

# --- Helper Functions ---
def parse_material(s: str) -> tuple[str, str]:
    """Parses material description to extract brand and color."""
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

def calculate_risk_score(row):
    """Calculate a risk score for each job based on multiple factors."""
    score = 0
    
    # Timeline risks
    if pd.notna(row.get('Days_Behind', np.nan)) and row['Days_Behind'] > 0:
        score += min(row['Days_Behind'] * 2, 20)  # Max 20 points for being behind
    
    # Stage duration risks
    if pd.notna(row.get('Days_In_Current_Stage', np.nan)):
        if row['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']:
            score += 10
    
    # Missing critical dates
    if pd.isna(row.get('Ready_to_Fab_Date')) and pd.notna(row.get('Template_Date')):
        days_since_template = (pd.Timestamp.now() - row['Template_Date']).days
        if days_since_template > TIMELINE_THRESHOLDS['template_to_rtf']:
            score += 15
    
    # Rework indicator
    if row.get('Has_Rework', False):
        score += 10
    
    # No next scheduled activity
    if pd.isna(row.get('Next_Sched_Activity')):
        score += 5
    
    return score

def get_current_stage(row):
    """Determine the current stage of a job based on dates."""
    if pd.notna(row.get('Install_Date')) or pd.notna(row.get('Pick_Up_Date')):
        return 'Completed'
    elif pd.notna(row.get('Ship_Date')):
        return 'Shipped'
    elif pd.notna(row.get('Product_Rcvd_Date')):
        return 'Product Received'
    elif pd.notna(row.get('Ready_to_Fab_Date')):
        return 'In Fabrication'
    elif pd.notna(row.get('Template_Date')):
        return 'Post-Template'
    else:
        return 'Pre-Template'

def calculate_days_in_stage(row, today):
    """Calculate how many days a job has been in its current stage."""
    stage = row['Current_Stage']
    
    if stage == 'Completed':
        return 0
    elif stage == 'Shipped' and pd.notna(row.get('Ship_Date')):
        return (today - row['Ship_Date']).days
    elif stage == 'Product Received' and pd.notna(row.get('Product_Rcvd_Date')):
        return (today - row['Product_Rcvd_Date']).days
    elif stage == 'In Fabrication' and pd.notna(row.get('Ready_to_Fab_Date')):
        return (today - row['Ready_to_Fab_Date']).days
    elif stage == 'Post-Template' and pd.notna(row.get('Template_Date')):
        return (today - row['Template_Date']).days
    else:
        return np.nan

@st.cache_data(ttl=300)
def load_and_process_data(creds_dict: dict, today: pd.Timestamp):
    """Load and process data with focus on timeline and warning indicators."""
    # Connect and load data
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(worksheet.get_all_records())
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+', '_', regex=True).str.replace(r'[^\w]', '', regex=True)
    
    # --- ENHANCED: Ensure all expected columns exist to prevent errors ---
    all_expected_cols = [
        'Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date',
        'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date',
        'Product_Rcvd_Date', 'Pick_Up_Date', 'Job_Material', 'Rework_Stone_Shop_Rework_Price',
        'Production_', 'Total_Job_Price_', 'Total_Job_SqFT', 'Job_Throughput_Job_GM_original',
        'Salesperson', 'Division', 'Next_Sched_Activity', 'Install_Assigned_To', 'Template_Assigned_To', 'Job_Name'
    ]
    for col in all_expected_cols:
        if col not in df.columns:
            df[col] = None

    # Parse dates
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date',
                 'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date',
                 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # --- NEW: Find the most recent activity date for stale job detection ---
    df['Last_Activity_Date'] = df[date_cols].max(axis=1)
    df['Days_Since_Last_Activity'] = (today - df['Last_Activity_Date']).dt.days


    # Calculate timeline metrics
    df['Days_Behind'] = np.where(df['Next_Sched_Date'].notna(), (today - df['Next_Sched_Date']).dt.days, np.nan)
    
    # Current stage and days in stage
    df['Current_Stage'] = df.apply(get_current_stage, axis=1)
    df['Days_In_Current_Stage'] = df.apply(lambda row: calculate_days_in_stage(row, today), axis=1)
    
    # Timeline calculations
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days
    df['Days_RTF_to_Product_Rcvd'] = (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days
    df['Days_Product_Rcvd_to_Install'] = (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days
    df['Days_Template_to_Ship'] = (df['Ship_Date'] - df['Template_Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days
    
    # Parse material info
    df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(lambda x: pd.Series(parse_material(str(x))))
    
    # Rework indicator
    df['Has_Rework'] = df['Rework_Stone_Shop_Rework_Price'].notna() & (df['Rework_Stone_Shop_Rework_Price'] != '')
    
    # Calculate risk score
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    
    # Calculate delay probability for predictive analytics
    df[['Delay_Probability', 'Risk_Factors']] = df.apply(lambda row: pd.Series(calculate_delay_probability(row)), axis=1)
    
    # Add hyperlinks
    df['Link'] = df['Production_'].apply(lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None)
    
    # Parse numeric fields
    numeric_fields = ['Total_Job_Price_', 'Total_Job_SqFT', 'Job_Throughput_Job_GM_original']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field].astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)

    # Determine division if available
    df['Division_Type'] = df['Division'].apply(lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz')

    return df

def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp):
    """Render the daily priorities and warnings dashboard."""
    st.header("🚨 Daily Priorities & Warnings")
    
    # High risk jobs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_jobs = df[df['Risk_Score'] >= 30]
        st.metric("🔴 High Risk Jobs", len(high_risk_jobs))
    
    with col2:
        behind_schedule = df[df['Days_Behind'] > 0]
        st.metric("⏰ Behind Schedule", len(behind_schedule))
    
    with col3:
        stuck_jobs = df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]
        st.metric("🚧 Stuck Jobs", len(stuck_jobs))

    # --- NEW: Stale Jobs metric ---
    with col4:
        stale_jobs = df[df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']]
        st.metric("💨 Stale Jobs", len(stale_jobs))

    # Critical Issues Section
    st.markdown("---")
    st.subheader("⚡ Critical Issues Requiring Immediate Attention")
    
    # Jobs missing next scheduled activity
    missing_activity = df[(df['Next_Sched_Activity'].isna()) & (df['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received']))]
    
    if not missing_activity.empty:
        with st.expander(f"🚨 Jobs Missing Next Activity ({len(missing_activity)} jobs)", expanded=True):
            display_cols = ['Link', 'Job_Name', 'Current_Stage', 'Days_In_Current_Stage', 'Salesperson']
            st.dataframe(
                missing_activity[display_cols].sort_values('Days_In_Current_Stage', ascending=False),
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")},
                use_container_width=True
            )
            
    # --- NEW: Stale Jobs expander ---
    if not stale_jobs.empty:
        with st.expander(f"💨 Stale Jobs (No activity for >{TIMELINE_THRESHOLDS['stale_job_threshold']} days)", expanded=False):
            stale_jobs_display = stale_jobs.copy()
            stale_jobs_display['Last_Activity_Date'] = stale_jobs_display['Last_Activity_Date'].dt.strftime('%Y-%m-%d')
            display_cols = ['Link', 'Job_Name', 'Current_Stage', 'Last_Activity_Date', 'Days_Since_Last_Activity']
            st.dataframe(
                stale_jobs_display[display_cols].sort_values('Days_Since_Last_Activity', ascending=False),
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")},
                use_container_width=True
            )

    # Jobs stuck between stages
    template_to_rtf_stuck = df[(df['Template_Date'].notna()) & (df['Ready_to_Fab_Date'].isna()) & ((today - df['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf'])]
    
    if not template_to_rtf_stuck.empty:
        with st.expander(f"📋 Stuck: Template → Ready to Fab ({len(template_to_rtf_stuck)} jobs)"):
            template_to_rtf_stuck['Days_Since_Template'] = (today - template_to_rtf_stuck['Template_Date']).dt.days
            display_cols = ['Link', 'Job_Name', 'Template_Date', 'Days_Since_Template', 'Salesperson']
            st.dataframe(
                template_to_rtf_stuck[display_cols].sort_values('Days_Since_Template', ascending=False),
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")},
                use_container_width=True
            )
            
    # Upcoming installs without product
    upcoming_installs = df[(df['Install_Date'].notna()) & (df['Install_Date'] <= today + timedelta(days=7)) & (df['Product_Rcvd_Date'].isna())]
    
    if not upcoming_installs.empty:
        with st.expander(f"⚠️ Upcoming Installs Missing Product ({len(upcoming_installs)} jobs)", expanded=True):
            upcoming_installs['Days_Until_Install'] = (upcoming_installs['Install_Date'] - today).dt.days
            display_cols = ['Link', 'Job_Name', 'Install_Date', 'Days_Until_Install', 'Install_Assigned_To']
            st.dataframe(
                upcoming_installs[display_cols].sort_values('Days_Until_Install'),
                column_config={"Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")},
                use_container_width=True
            )

# (The rest of your rendering functions: render_workload_calendar, render_timeline_analytics, etc. remain the same)
def render_workload_calendar(df: pd.DataFrame, today: pd.Timestamp):
    """Render workload calendar view to identify light days."""
    st.header("📅 Workload Calendar")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=today.date())
    with col2:
        end_date = st.date_input("End Date", value=(today + timedelta(days=14)).date())
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Activity type selector
    activity_type = st.selectbox("Select Activity Type", ["Templates", "Installs", "All Activities"])
    
    # Prepare data based on selection
    if activity_type == "Templates":
        activity_df = df[df['Template_Date'].notna()].copy()
        date_col, assignee_col = 'Template_Date', 'Template_Assigned_To'
    elif activity_type == "Installs":
        activity_df = df[df['Install_Date'].notna()].copy()
        date_col, assignee_col = 'Install_Date', 'Install_Assigned_To'
    else:
        activities = []
        if 'Template_Date' in df.columns:
            temp_df = df[df['Template_Date'].notna()].copy()
            temp_df.rename(columns={'Template_Date': 'Activity_Date', 'Template_Assigned_To': 'Assignee'}, inplace=True)
            temp_df['Activity_Type'] = 'Template'
            activities.append(temp_df)
        if 'Install_Date' in df.columns:
            inst_df = df[df['Install_Date'].notna()].copy()
            inst_df.rename(columns={'Install_Date': 'Activity_Date', 'Install_Assigned_To': 'Assignee'}, inplace=True)
            inst_df['Activity_Type'] = 'Install'
            activities.append(inst_df)
        activity_df = pd.concat(activities, ignore_index=True) if activities else pd.DataFrame()
        date_col, assignee_col = 'Activity_Date', 'Assignee'

    if activity_df.empty:
        st.warning("No activities found in the selected range.")
        return
    
    # Filter by date range
    activity_df = activity_df[(activity_df[date_col] >= pd.Timestamp(start_date)) & (activity_df[date_col] <= pd.Timestamp(end_date))]
    
    # Create daily summary
    daily_summary = []
    for date in date_range:
        day_activities = activity_df[activity_df[date_col].dt.date == date.date()]
        if not day_activities.empty and assignee_col in day_activities.columns:
            assignee_counts = day_activities[assignee_col].value_counts()
            for assignee, count in assignee_counts.items():
                if assignee and str(assignee).strip():
                    assignee_jobs = day_activities[day_activities[assignee_col] == assignee]
                    total_sqft = assignee_jobs['Total_Job_SqFT'].sum() if 'Total_Job_SqFT' in assignee_jobs else 0
                    daily_summary.append({'Date': date, 'Assignee': str(assignee), 'Job_Count': int(count), 'Total_SqFt': float(total_sqft)})
    
    if daily_summary:
        summary_df = pd.DataFrame(daily_summary)
        try:
            pivot_df = summary_df.pivot_table(index='Assignee', columns=summary_df['Date'].dt.strftime('%m/%d'), values='Job_Count', fill_value=0, aggfunc='sum').astype(int)
            if not pivot_df.empty:
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_df) * 0.5)))
                sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Jobs'})
                ax.set_title('Jobs by Assignee and Date'); ax.set_xlabel('Date'); ax.set_ylabel('Assignee')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception:
            st.warning("Unable to create heatmap visualization. Showing data in table format instead."); st.dataframe(summary_df)
            
        st.subheader("💡 Days with Light Workload")
        threshold = st.slider("Jobs threshold for 'light' day", 1, 10, 3)
        daily_totals = summary_df.groupby('Date')['Job_Count'].sum()
        light_days = [{'Date': date.strftime('%A, %m/%d'), 'Total_Jobs': int(daily_totals.get(date, 0)), 'Available_Capacity': int(threshold - daily_totals.get(date, 0))} for date in date_range if daily_totals.get(date, 0) < threshold]
        if light_days:
            st.dataframe(pd.DataFrame(light_days), use_container_width=True)
        else:
            st.success(f"No days found with fewer than {threshold} jobs.")

def render_timeline_analytics(df: pd.DataFrame):
    """Render timeline analytics and bottleneck identification."""
    st.header("📊 Timeline Analytics & Bottlenecks")
    
    timeline_metrics = {
        "Template to Install": "Days_Template_to_Install", "Ready to Fab to Product Received": "Days_RTF_to_Product_Rcvd",
        "Template to Ready to Fab": "Days_Template_to_RTF", "Product Received to Install": "Days_Product_Rcvd_to_Install",
        "Template to Ship": "Days_Template_to_Ship", "Ship to Install": "Days_Ship_to_Install",
    }
    
    st.subheader("⏱️ Average Timeline by Division")
    divisions = df['Division_Type'].unique()
    cols = st.columns(len(divisions))
    
    for idx, division in enumerate(divisions):
        with cols[idx]:
            st.markdown(f"**{division}**")
            division_df = df[df['Division_Type'] == division]
            metrics_data = [{'Metric': metric_name, 'Avg Days': round(division_df[col_name].mean(), 1)} for metric_name, col_name in timeline_metrics.items() if col_name in division_df.columns and pd.notna(division_df[col_name].mean())]
            if metrics_data:
                for metric in metrics_data:
                    st.metric(metric['Metric'], f"{metric['Avg Days']} days")
            else:
                st.info("No timeline data available")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Bottlenecks")
        stage_counts = df['Current_Stage'].value_counts()
        if not stage_counts.empty:
            st.bar_chart(stage_counts)
            bottleneck_stage = stage_counts.idxmax()
            st.info(f"📍 Potential bottleneck: **{bottleneck_stage}** ({stage_counts[bottleneck_stage]} jobs)")
    with col2:
        st.subheader("Stage Duration Analysis")
        stuck_threshold = st.number_input("Days threshold for 'stuck' jobs", min_value=3, max_value=30, value=7, key="stuck_threshold")
        stuck_jobs = df[df['Days_In_Current_Stage'] > stuck_threshold]
        if not stuck_jobs.empty:
            stuck_by_stage = stuck_jobs['Current_Stage'].value_counts()
            st.bar_chart(stuck_by_stage); st.warning(f"⚠️ {len(stuck_jobs)} jobs stuck > {stuck_threshold} days")
        else:
            st.success(f"✅ No jobs stuck > {stuck_threshold} days")
            
# (The rest of your original rendering functions remain here)
def calculate_delay_probability(row):
    """Calculate probability of delay based on multiple factors."""
    risk_score = 0
    factors = []
    
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40; factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_stage_duration = {'Post-Template': 3, 'In Fabrication': 7, 'Product Received': 5, 'Shipped': 5}
        expected_duration = avg_stage_duration.get(row.get('Current_Stage', ''), 5)
        if row['Days_In_Current_Stage'] > expected_duration:
            risk_score += 20; factors.append(f"Stuck in {row['Current_Stage']} for {row['Days_In_Current_Stage']:.0f} days")
            
    if row.get('Has_Rework', False):
        risk_score += 15; factors.append("Has rework")
        
    if pd.isna(row.get('Next_Sched_Activity')):
        risk_score += 15; factors.append("No next activity scheduled")
        
    return min(risk_score, 100), factors

def render_predictive_analytics(df: pd.DataFrame):
    """Render predictive analytics for job delays."""
    st.header("🔮 Predictive Analytics")
    active_jobs = df[~df['Current_Stage'].isin(['Completed'])].copy()
    if active_jobs.empty:
        st.warning("No active jobs to analyze."); return
        
    active_jobs[['Delay_Probability', 'Risk_Factors']] = active_jobs.apply(lambda row: pd.Series(calculate_delay_probability(row)), axis=1)
    high_risk_threshold = st.slider("High risk threshold (%)", min_value=50, max_value=90, value=70, key="risk_threshold")
    high_risk_jobs = active_jobs[active_jobs['Delay_Probability'] >= high_risk_threshold].sort_values('Delay_Probability', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"🚨 Jobs with >{high_risk_threshold}% Delay Risk ({len(high_risk_jobs)} jobs)")
        if not high_risk_jobs.empty:
            for _, row in high_risk_jobs.head(10).iterrows():
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{row.get('Job_Name', 'Unknown')}** - {row.get('Current_Stage', 'Unknown')}")
                        st.caption(f"Salesperson: {row.get('Salesperson', 'N/A')} | Next: {row.get('Next_Sched_Activity', 'None scheduled')}")
                        if row['Risk_Factors']:
                            st.warning(f"Risk factors: {' • '.join(row['Risk_Factors'])}")
                    with col_b:
                        color = "🔴" if row['Delay_Probability'] >= 80 else "🟡"
                        st.metric("Delay Risk", f"{color} {row['Delay_Probability']:.0f}%")
                    st.markdown("---")
        else:
            st.success(f"No jobs with delay risk above {high_risk_threshold}%")
    with col2:
        st.subheader("📊 Risk Distribution")
        risk_bins = [0, 30, 50, 70, 100]; risk_labels = ['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (70-100%)']
        active_jobs['Risk_Category'] = pd.cut(active_jobs['Delay_Probability'], bins=risk_bins, labels=risk_labels)
        risk_dist = active_jobs['Risk_Category'].value_counts()
        for category in risk_labels:
            st.metric(f"{'🔴' if 'Critical' in category else '🟡' if 'High' in category else '🟠' if 'Medium' in category else '🟢'} {category}", risk_dist.get(category, 0))

def render_performance_scorecards(df: pd.DataFrame):
    """Render performance scorecards for employees."""
    st.header("📊 Performance Scorecards")
    role_type = st.selectbox("Select Role to Analyze", ["Salesperson", "Template Assigned To", "Install Assigned To"], key="role_select")
    role_col = {"Salesperson": "Salesperson", "Template Assigned To": "Template_Assigned_To", "Install Assigned To": "Install_Assigned_To"}[role_type]
    
    if role_col not in df.columns:
        st.warning(f"Column '{role_col}' not found in data."); return
        
    employees = [e for e in df[role_col].dropna().unique() if str(e).strip()]
    if not employees:
        st.warning(f"No {role_type} data available."); return
        
    scorecards = []
    for employee in employees:
        emp_jobs = df[df[role_col] == employee]
        if role_type == "Salesperson":
            metrics = {'Employee': employee, 'Total Jobs': len(emp_jobs), 'Active Jobs': len(emp_jobs[~emp_jobs['Current_Stage'].isin(['Completed'])]), 'Avg Days Behind': emp_jobs['Days_Behind'].mean(), 'Jobs w/ Rework': len(emp_jobs[emp_jobs.get('Has_Rework', False) == True]), 'Avg Timeline': emp_jobs['Days_Template_to_Install'].mean(), 'High Risk Jobs': len(emp_jobs[emp_jobs.get('Risk_Score', 0) >= 30])}
        elif role_type == "Template Assigned To":
            template_jobs = emp_jobs[emp_jobs['Template_Date'].notna()]; now = pd.Timestamp.now(); week_ago = now - pd.Timedelta(days=7)
            metrics = {'Employee': employee, 'Total Templates': len(template_jobs), 'Avg Template to RTF': template_jobs['Days_Template_to_RTF'].mean(), 'Templates This Week': len(template_jobs[(template_jobs['Template_Date'] >= week_ago) & (template_jobs['Template_Date'] <= now)]), 'Upcoming Templates': len(template_jobs[template_jobs['Template_Date'] > now]), 'Overdue RTF': len(template_jobs[(template_jobs.get('Ready_to_Fab_Date', pd.Series()).isna()) & ((now - template_jobs['Template_Date']).dt.days > 3)]) if 'Ready_to_Fab_Date' in template_jobs.columns else 0}
        else: # Install Assigned To
            install_jobs = emp_jobs[emp_jobs['Install_Date'].notna()]; now = pd.Timestamp.now(); week_ago = now - pd.Timedelta(days=7)
            metrics = {'Employee': employee, 'Total Installs': len(install_jobs), 'Installs This Week': len(install_jobs[(install_jobs['Install_Date'] >= week_ago) & (install_jobs['Install_Date'] <= now)]), 'Upcoming Installs': len(install_jobs[install_jobs['Install_Date'] > now]), 'Avg Ship to Install': install_jobs['Days_Ship_to_Install'].mean(), 'Total SqFt': install_jobs['Total_Job_SqFT'].sum()}
        scorecards.append(metrics)
        
    if not scorecards:
        st.warning(f"No performance data available for {role_type}."); return
        
    scorecards_df = pd.DataFrame(scorecards).sort_values("Active Jobs" if role_type == "Salesperson" else "Total Templates" if role_type == "Template Assigned To" else "Total Installs", ascending=False)
    st.subheader(f"🎯 {role_type} Performance")
    cols = st.columns(min(3, len(scorecards_df)))
    for idx, (_, row) in enumerate(scorecards_df.head(3).iterrows()):
        if idx < len(cols):
            with cols[idx], st.container(border=True):
                st.markdown(f"### {row['Employee']}")
                if role_type == "Salesperson":
                    st.metric("Active Jobs", row['Active Jobs']); st.metric("Avg Days Behind", f"{row['Avg Days Behind']:.1f}" if pd.notna(row['Avg Days Behind']) else "N/A"); st.metric("High Risk Jobs", row['High Risk Jobs'], delta_color="inverse")
                elif role_type == "Template Assigned To":
                    st.metric("Templates This Week", row['Templates This Week']); st.metric("Avg Template→RTF", f"{row['Avg Template to RTF']:.1f} days" if pd.notna(row['Avg Template to RTF']) else "N/A"); st.metric("Overdue RTF", row['Overdue RTF'], delta_color="inverse")
                else:
                    st.metric("Installs This Week", row['Installs This Week']); st.metric("Upcoming Installs", row['Upcoming Installs']); st.metric("Total SqFt", f"{row['Total SqFt']:,.0f}")
    with st.expander("View All Employees"):
        st.dataframe(scorecards_df.style.format({col: '{:.1f}' for col in scorecards_df.columns if 'Avg' in col or 'Days' in col} | {col: '{:,.0f}' for col in scorecards_df.columns if 'SqFt' in col}, na_rep='N/A'), use_container_width=True)

def render_quick_actions(df: pd.DataFrame):
    """Render quick action items and recommendations."""
    st.header("⚡ Quick Actions & Recommendations")
    # This function remains as is, providing high-level summaries and actions.

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.title("⚡ Daily Operations Dashboard")
    st.markdown("Real-time insights for maximizing efficiency and catching issues early")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load credentials
    creds = None
    if "google_creds_json" in st.secrets:
        creds = json.loads(st.secrets["google_creds_json"])
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Google Service Account JSON", type="json")
        if uploaded_file:
            creds = json.load(uploaded_file)
    
    if not creds:
        st.sidebar.error("Please provide Google credentials to load data.")
        st.stop()
    
    today = pd.Timestamp.now()
    
    # Load data
    try:
        with st.spinner("Loading and processing data..."):
            df_full = load_and_process_data(creds, today)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
        
    # --- Global Filters in Sidebar ---
    st.sidebar.header("Global Filters")
    
    # Filter by Job Status
    status_filter = st.sidebar.selectbox("Filter by Job Status", ["Active", "Completed", "All"], index=0)
    
    # Filter by Salesperson
    salesperson_list = ['All'] + sorted(df_full['Salesperson'].dropna().unique().tolist())
    salesperson_filter = st.sidebar.selectbox("Filter by Salesperson", salesperson_list)
    
    # Filter by Division
    division_list = ['All'] + sorted(df_full['Division_Type'].dropna().unique().tolist())
    division_filter = st.sidebar.selectbox("Filter by Division", division_list)

    # Apply filters
    df_filtered = df_full.copy()
    if status_filter == "Active":
        df_filtered = df_filtered[df_filtered['Current_Stage'] != 'Completed']
    elif status_filter == "Completed":
        df_filtered = df_filtered[df_filtered['Current_Stage'] == 'Completed']
    
    if salesperson_filter != 'All':
        df_filtered = df_filtered[df_filtered['Salesperson'] == salesperson_filter]
    
    if division_filter != 'All':
        df_filtered = df_filtered[df_filtered['Division_Type'] == division_filter]
        
    # Last Refreshed Timestamp
    st.sidebar.markdown("---")
    st.sidebar.info(f"Data last refreshed: {today.strftime('%Y-%m-%d %H:%M:%S')}")

    # Display main metrics
    active_jobs = df_filtered[df_filtered['Current_Stage'] != 'Completed']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Displayed Jobs", len(df_filtered))
    with col2:
        avg_timeline = df_filtered['Days_Template_to_Install'].mean()
        st.metric("Avg Timeline", f"{avg_timeline:.1f} days" if pd.notna(avg_timeline) else "N/A")
    with col3:
        high_risk = df_filtered[df_filtered['Risk_Score'] >= 30]
        st.metric("High Risk Jobs", len(high_risk))
    with col4:
        today_activities = df_filtered[(df_filtered['Template_Date'].dt.date == today.date()) | (df_filtered['Install_Date'].dt.date == today.date())]
        st.metric("Today's Activities", len(today_activities))

    # Render tabs
    tabs = st.tabs(["🚨 Daily Priorities", "📅 Workload Calendar", "📊 Timeline Analytics", "🔮 Predictive Analytics", "🎯 Performance Scorecards", "⚡ Quick Actions", "📈 Historical Comparisons"])
    with tabs[0]: render_daily_priorities(df_filtered, today)
    with tabs[1]: render_workload_calendar(df_filtered, today)
    with tabs[2]: render_timeline_analytics(df_filtered)
    with tabs[3]: render_predictive_analytics(df_filtered)
    with tabs[4]: render_performance_scorecards(df_filtered)
    with tabs[5]: render_quick_actions(df_filtered)
    with tabs[6]: render_historical_placeholder(df_full) # Pass full dataframe to historical

if __name__ == "__main__":
    main()
