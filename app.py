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
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Daily Operations Dashboard", page_icon="⚡")

# --- Constants ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="

# --- Timeline Thresholds (customizable) ---
TIMELINE_THRESHOLDS = {
    'template_to_rtf': 3,  # days
    'rtf_to_product': 7,   # days
    'product_to_install': 5,  # days
    'template_to_install': 15,  # days
    'days_in_stage_warning': 5,  # days stuck in any stage
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
    
    # Parse dates
    date_cols = ['Template_Date', 'Ready_to_Fab_Date', 'Ship_Date', 'Install_Date',
                 'Service_Date', 'Delivery_Date', 'Job_Creation', 'Next_Sched_Date',
                 'Product_Rcvd_Date', 'Pick_Up_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate timeline metrics
    df['Days_Behind'] = np.where(
        df['Next_Sched_Date'].notna(),
        (today - df['Next_Sched_Date']).dt.days,
        np.nan
    )
    
    # Current stage and days in stage
    df['Current_Stage'] = df.apply(get_current_stage, axis=1)
    df['Days_In_Current_Stage'] = df.apply(lambda row: calculate_days_in_stage(row, today), axis=1)
    
    # Timeline calculations
    df['Days_Template_to_RTF'] = np.where(
        (df['Template_Date'].notna() & df['Ready_to_Fab_Date'].notna()),
        (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days,
        np.nan
    )
    
    df['Days_RTF_to_Product'] = np.where(
        (df['Ready_to_Fab_Date'].notna() & df['Product_Rcvd_Date'].notna()),
        (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days,
        np.nan
    )
    
    df['Days_Product_to_Install'] = np.where(
        (df['Product_Rcvd_Date'].notna() & df['Install_Date'].notna()),
        (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days,
        np.nan
    )
    
    # Parse material info
    if 'Job_Material' in df.columns:
        df[['Material_Brand', 'Material_Color']] = df['Job_Material'].apply(
            lambda x: pd.Series(parse_material(str(x)))
        )
    
    # Rework indicator
    df['Has_Rework'] = df['Rework_Stone_Shop_Rework_Price'].notna() & (df['Rework_Stone_Shop_Rework_Price'] != '')
    
    # Calculate risk score
    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    
    # Add hyperlinks
    if 'Production_' in df.columns:
        df['Link'] = df['Production_'].apply(
            lambda po: f"{MORAWARE_SEARCH_URL}{po}" if po else None
        )
    
    # Parse numeric fields
    numeric_fields = ['Total_Job_Price_', 'Total_Job_SqFT', 'Job_Throughput_Job_GM_original']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(
                df[field].astype(str).str.replace(r'[$,%]', '', regex=True),
                errors='coerce'
            ).fillna(0)
    
    return df

def render_daily_priorities(df: pd.DataFrame, today: pd.Timestamp):
    """Render the daily priorities and warnings dashboard."""
    st.header("🚨 Daily Priorities & Warnings")
    
    # High risk jobs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_jobs = df[df['Risk_Score'] >= 30]
        st.metric("🔴 High Risk Jobs", len(high_risk_jobs))
    
    with col2:
        behind_schedule = df[df['Days_Behind'] > 0]
        st.metric("⏰ Behind Schedule", len(behind_schedule))
    
    with col3:
        stuck_jobs = df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']]
        st.metric("🚧 Stuck Jobs", len(stuck_jobs))
    
    # Critical Issues Section
    st.markdown("---")
    st.subheader("⚡ Critical Issues Requiring Immediate Attention")
    
    # Jobs missing next scheduled activity
    missing_activity = df[
        (df['Next_Sched_Activity'].isna()) & 
        (df['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received']))
    ]
    
    if not missing_activity.empty:
        with st.expander(f"🚨 Jobs Missing Next Activity ({len(missing_activity)} jobs)", expanded=True):
            display_cols = ['Link', 'Job_Name', 'Current_Stage', 'Days_In_Current_Stage', 'Salesperson']
            st.dataframe(
                missing_activity[display_cols].sort_values('Days_In_Current_Stage', ascending=False),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                }
            )
    
    # Jobs stuck between stages
    template_to_rtf_stuck = df[
        (df['Template_Date'].notna()) & 
        (df['Ready_to_Fab_Date'].isna()) & 
        ((today - df['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf'])
    ]
    
    if not template_to_rtf_stuck.empty:
        with st.expander(f"📋 Stuck: Template → Ready to Fab ({len(template_to_rtf_stuck)} jobs)"):
            template_to_rtf_stuck['Days_Since_Template'] = (today - template_to_rtf_stuck['Template_Date']).dt.days
            display_cols = ['Link', 'Job_Name', 'Template_Date', 'Days_Since_Template', 'Salesperson']
            st.dataframe(
                template_to_rtf_stuck[display_cols].sort_values('Days_Since_Template', ascending=False),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                }
            )
    
    # Upcoming installs without product
    upcoming_installs = df[
        (df['Install_Date'].notna()) & 
        (df['Install_Date'] <= today + timedelta(days=7)) &
        (df['Product_Rcvd_Date'].isna())
    ]
    
    if not upcoming_installs.empty:
        with st.expander(f"⚠️ Upcoming Installs Missing Product ({len(upcoming_installs)} jobs)", expanded=True):
            upcoming_installs['Days_Until_Install'] = (upcoming_installs['Install_Date'] - today).dt.days
            display_cols = ['Link', 'Job_Name', 'Install_Date', 'Days_Until_Install', 'Install_Assigned_To']
            st.dataframe(
                upcoming_installs[display_cols].sort_values('Days_Until_Install'),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                }
            )

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
    activity_type = st.selectbox(
        "Select Activity Type",
        ["Templates", "Installs", "All Activities"]
    )
    
    # Prepare data based on selection
    if activity_type == "Templates":
        activity_df = df[df['Template_Date'].notna()].copy()
        date_col = 'Template_Date'
        assignee_col = 'Template_Assigned_To'
    elif activity_type == "Installs":
        activity_df = df[df['Install_Date'].notna()].copy()
        date_col = 'Install_Date'
        assignee_col = 'Install_Assigned_To'
    else:
        # Combine all activities
        activities = []
        if 'Template_Date' in df.columns:
            temp_df = df[df['Template_Date'].notna()].copy()
            temp_df['Activity_Type'] = 'Template'
            temp_df['Activity_Date'] = temp_df['Template_Date']
            temp_df['Assignee'] = temp_df.get('Template_Assigned_To', 'Unassigned')
            activities.append(temp_df)
        
        if 'Install_Date' in df.columns:
            inst_df = df[df['Install_Date'].notna()].copy()
            inst_df['Activity_Type'] = 'Install'
            inst_df['Activity_Date'] = inst_df['Install_Date']
            inst_df['Assignee'] = inst_df.get('Install_Assigned_To', 'Unassigned')
            activities.append(inst_df)
        
        activity_df = pd.concat(activities, ignore_index=True) if activities else pd.DataFrame()
        date_col = 'Activity_Date'
        assignee_col = 'Assignee'
    
    if activity_df.empty:
        st.warning("No activities found in the selected range.")
        return
    
    # Filter by date range
    activity_df = activity_df[
        (activity_df[date_col] >= pd.Timestamp(start_date)) & 
        (activity_df[date_col] <= pd.Timestamp(end_date))
    ]
    
    # Create daily summary
    daily_summary = []
    for date in date_range:
        day_activities = activity_df[activity_df[date_col].dt.date == date.date()]
        
        if assignee_col in day_activities.columns:
            assignee_counts = day_activities[assignee_col].value_counts()
            for assignee, count in assignee_counts.items():
                if assignee and str(assignee).strip():
                    daily_summary.append({
                        'Date': date,
                        'Assignee': assignee,
                        'Job_Count': count,
                        'Total_SqFt': day_activities[day_activities[assignee_col] == assignee]['Total_Job_SqFT'].sum()
                    })
    
    if daily_summary:
        summary_df = pd.DataFrame(daily_summary)
        
        # Pivot for heatmap
        pivot_df = summary_df.pivot_table(
            index='Assignee',
            columns=summary_df['Date'].dt.strftime('%m/%d'),
            values='Job_Count',
            fill_value=0
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Date", y="Assignee", color="Jobs"),
            color_continuous_scale="RdYlGn_r",
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show days with light workload
        st.subheader("💡 Days with Light Workload")
        threshold = st.slider("Jobs threshold for 'light' day", 1, 10, 3)
        
        light_days = []
        for date in date_range:
            day_total = summary_df[summary_df['Date'] == date]['Job_Count'].sum()
            if day_total < threshold:
                light_days.append({
                    'Date': date.strftime('%A, %m/%d'),
                    'Total_Jobs': day_total,
                    'Available_Capacity': threshold - day_total
                })
        
        if light_days:
            light_df = pd.DataFrame(light_days)
            st.dataframe(light_df, use_container_width=True)
        else:
            st.success(f"No days found with fewer than {threshold} jobs.")

def render_timeline_analytics(df: pd.DataFrame):
    """Render timeline analytics and bottleneck identification."""
    st.header("📊 Timeline Analytics & Bottlenecks")
    
    # Average timelines by stage
    timeline_metrics = {
        'Template → RTF': 'Days_Template_to_RTF',
        'RTF → Product Received': 'Days_RTF_to_Product',
        'Product → Install': 'Days_Product_to_Install'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Timeline by Stage")
        avg_times = []
        for stage, col in timeline_metrics.items():
            if col in df.columns:
                avg_time = df[col].mean()
                if pd.notna(avg_time):
                    avg_times.append({
                        'Stage': stage,
                        'Avg_Days': round(avg_time, 1),
                        'Target_Days': TIMELINE_THRESHOLDS.get(col.lower(), 'N/A')
                    })
        
        if avg_times:
            avg_df = pd.DataFrame(avg_times)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=avg_df['Stage'],
                y=avg_df['Avg_Days'],
                name='Actual',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Average Days by Stage",
                yaxis_title="Days",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Current Bottlenecks")
        
        # Count jobs in each stage
        stage_counts = df['Current_Stage'].value_counts()
        
        # Identify bottlenecks (stages with unusually high counts)
        if not stage_counts.empty:
            fig = px.pie(
                values=stage_counts.values,
                names=stage_counts.index,
                title="Jobs by Current Stage"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Timeline trend analysis
    st.subheader("📈 Timeline Trends (Last 30 Days)")
    recent_jobs = df[df['Install_Date'] >= today - timedelta(days=30)]
    
    if not recent_jobs.empty and 'Days_Template_to_Install' in recent_jobs.columns:
        recent_jobs = recent_jobs.dropna(subset=['Days_Template_to_Install'])
        if not recent_jobs.empty:
            fig = px.scatter(
                recent_jobs,
                x='Install_Date',
                y='Days_Template_to_Install',
                color='Salesperson',
                size='Total_Job_SqFT',
                hover_data=['Job_Name', 'Material_Brand'],
                title="Template to Install Timeline Trend"
            )
            fig.add_hline(
                y=TIMELINE_THRESHOLDS['template_to_install'],
                line_dash="dash",
                line_color="red",
                annotation_text="Target"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_quick_actions(df: pd.DataFrame):
    """Render quick action items and recommendations."""
    st.header("⚡ Quick Actions & Recommendations")
    
    # Generate actionable insights
    actions = []
    
    # Check for jobs needing immediate attention
    urgent_jobs = df[
        (df['Days_Behind'] > 3) | 
        (df['Risk_Score'] > 40)
    ]
    
    if not urgent_jobs.empty:
        actions.append({
            'Priority': '🔴 HIGH',
            'Action': f"Contact customers for {len(urgent_jobs)} high-risk jobs",
            'Details': f"Jobs more than 3 days behind or with risk score > 40"
        })
    
    # Check for resource imbalances
    if 'Install_Assigned_To' in df.columns:
        upcoming_installs = df[
            (df['Install_Date'] >= today) & 
            (df['Install_Date'] <= today + timedelta(days=7))
        ]
        if not upcoming_installs.empty:
            installer_loads = upcoming_installs['Install_Assigned_To'].value_counts()
            if len(installer_loads) > 0:
                max_load = installer_loads.max()
                min_load = installer_loads.min()
                if max_load > min_load * 2:
                    actions.append({
                        'Priority': '🟡 MEDIUM',
                        'Action': "Rebalance installer workload",
                        'Details': f"Workload varies from {min_load} to {max_load} jobs per installer"
                    })
    
    # Check for scheduling opportunities
    light_days = []
    for i in range(1, 8):
        future_date = today + timedelta(days=i)
        day_installs = df[df['Install_Date'].dt.date == future_date.date()]
        if len(day_installs) < 3:
            light_days.append(future_date.strftime('%A %m/%d'))
    
    if light_days:
        actions.append({
            'Priority': '🟢 LOW',
            'Action': "Schedule additional jobs on light days",
            'Details': f"Light days: {', '.join(light_days[:3])}"
        })
    
    # Display actions
    if actions:
        actions_df = pd.DataFrame(actions)
        st.dataframe(
            actions_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Action": st.column_config.TextColumn("Action", width="medium"),
                "Details": st.column_config.TextColumn("Details", width="large")
            }
        )
    else:
        st.success("✅ No urgent actions required at this time!")

# --- Main Application ---
def main():
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
    
    # Timeline threshold customization
    with st.sidebar.expander("⏱️ Timeline Thresholds"):
        for key, value in TIMELINE_THRESHOLDS.items():
            TIMELINE_THRESHOLDS[key] = st.number_input(
                f"{key.replace('_', ' ').title()} (days)",
                value=value,
                min_value=1,
                max_value=30
            )
    
    today = pd.Timestamp.now()
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            df = load_and_process_data(creds, today)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
    
    # Quick metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        active_jobs = df[~df['Current_Stage'].isin(['Completed'])]
        st.metric("Active Jobs", len(active_jobs))
    with col2:
        avg_timeline = df['Days_Template_to_Install'].mean()
        st.metric("Avg Timeline", f"{avg_timeline:.1f} days" if pd.notna(avg_timeline) else "N/A")
    with col3:
        high_risk = df[df['Risk_Score'] >= 30]
        st.metric("High Risk Jobs", len(high_risk), delta=f"{len(high_risk)/len(active_jobs)*100:.0f}%")
    with col4:
        today_activities = df[
            (df['Template_Date'].dt.date == today.date()) | 
            (df['Install_Date'].dt.date == today.date())
        ]
        st.metric("Today's Activities", len(today_activities))
    
    # Main content tabs
    tabs = st.tabs([
        "🚨 Daily Priorities",
        "📅 Workload Calendar", 
        "📊 Timeline Analytics",
        "⚡ Quick Actions"
    ])
    
    with tabs[0]:
        render_daily_priorities(df, today)
    
    with tabs[1]:
        render_workload_calendar(df, today)
    
    with tabs[2]:
        render_timeline_analytics(df)
    
    with tabs[3]:
        render_quick_actions(df)
    
    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (5 min)", value=False):
        st.rerun()

if __name__ == "__main__":
    main()
