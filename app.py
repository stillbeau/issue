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
    'rtf_to_product_rcvd': 7,   # days
    'product_rcvd_to_install': 5,  # days
    'template_to_install': 15,  # days
    'template_to_ship': 10,  # days
    'ship_to_install': 5,  # days
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
    
    df['Days_RTF_to_Product_Rcvd'] = np.where(
        (df['Ready_to_Fab_Date'].notna() & df['Product_Rcvd_Date'].notna()),
        (df['Product_Rcvd_Date'] - df['Ready_to_Fab_Date']).dt.days,
        np.nan
    )
    
    df['Days_Product_Rcvd_to_Install'] = np.where(
        (df['Product_Rcvd_Date'].notna() & df['Install_Date'].notna()),
        (df['Install_Date'] - df['Product_Rcvd_Date']).dt.days,
        np.nan
    )
    
    df['Days_Template_to_Install'] = np.where(
        (df['Template_Date'].notna() & df['Install_Date'].notna()),
        (df['Install_Date'] - df['Template_Date']).dt.days,
        np.nan
    )
    
    df['Days_Template_to_Ship'] = np.where(
        (df['Template_Date'].notna() & df['Ship_Date'].notna()),
        (df['Ship_Date'] - df['Template_Date']).dt.days,
        np.nan
    )
    
    df['Days_Ship_to_Install'] = np.where(
        (df['Ship_Date'].notna() & df['Install_Date'].notna()),
        (df['Install_Date'] - df['Ship_Date']).dt.days,
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
    
    # Calculate delay probability for predictive analytics
    df[['Delay_Probability', 'Risk_Factors']] = df.apply(
        lambda row: pd.Series(calculate_delay_probability(row)), axis=1
    )
    
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
            display_cols = [col for col in display_cols if col in missing_activity.columns]
            st.dataframe(
                missing_activity[display_cols].sort_values('Days_In_Current_Stage', ascending=False),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                },
                use_container_width=True
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
            display_cols = [col for col in display_cols if col in template_to_rtf_stuck.columns]
            st.dataframe(
                template_to_rtf_stuck[display_cols].sort_values('Days_Since_Template', ascending=False),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                },
                use_container_width=True
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
            display_cols = [col for col in display_cols if col in upcoming_installs.columns]
            st.dataframe(
                upcoming_installs[display_cols].sort_values('Days_Until_Install'),
                column_config={
                    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)")
                },
                use_container_width=True
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
        
        if not day_activities.empty and assignee_col in day_activities.columns:
            assignee_counts = day_activities[assignee_col].value_counts()
            for assignee, count in assignee_counts.items():
                if assignee and str(assignee).strip():
                    sqft_col = 'Total_Job_SqFT' if 'Total_Job_SqFT' in day_activities.columns else None
                    total_sqft = 0
                    if sqft_col:
                        assignee_jobs = day_activities[day_activities[assignee_col] == assignee]
                        total_sqft = assignee_jobs[sqft_col].sum() if not assignee_jobs.empty else 0
                    
                    daily_summary.append({
                        'Date': date,
                        'Assignee': str(assignee),
                        'Job_Count': int(count),
                        'Total_SqFt': float(total_sqft)
                    })
        elif not day_activities.empty:
            # If no assignee column, just count total jobs for the day
            sqft_col = 'Total_Job_SqFT' if 'Total_Job_SqFT' in day_activities.columns else None
            total_sqft = day_activities[sqft_col].sum() if sqft_col else 0
            
            daily_summary.append({
                'Date': date,
                'Assignee': 'Unassigned',
                'Job_Count': len(day_activities),
                'Total_SqFt': float(total_sqft)
            })
    
    if daily_summary:
        summary_df = pd.DataFrame(daily_summary)
        
        # Create a simple heatmap using matplotlib
        try:
            pivot_df = summary_df.pivot_table(
                index='Assignee',
                columns=summary_df['Date'].dt.strftime('%m/%d'),
                values='Job_Count',
                fill_value=0,
                aggfunc='sum'  # Explicitly set aggregation function
            )
            
            # Convert to integers to avoid formatting issues
            pivot_df = pivot_df.fillna(0).astype(int)
            
            # Only create heatmap if we have data
            if not pivot_df.empty:
                # Display heatmap
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Jobs'})
                ax.set_title('Jobs by Assignee and Date')
                ax.set_xlabel('Date')
                ax.set_ylabel('Assignee')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No data available for heatmap visualization.")
        except Exception as e:
            st.warning(f"Unable to create heatmap visualization. Showing data in table format instead.")
            st.dataframe(summary_df)
        
        # Show days with light workload
        st.subheader("💡 Days with Light Workload")
        threshold = st.slider("Jobs threshold for 'light' day", 1, 10, 3)
        
        light_days = []
        for date in date_range:
            day_df = summary_df[summary_df['Date'] == date] if 'summary_df' in locals() else pd.DataFrame()
            day_total = day_df['Job_Count'].sum() if not day_df.empty else 0
            if day_total < threshold:
                light_days.append({
                    'Date': date.strftime('%A, %m/%d'),
                    'Total_Jobs': int(day_total),
                    'Available_Capacity': int(threshold - day_total)
                })
        
        if light_days:
            light_df = pd.DataFrame(light_days)
            st.dataframe(light_df, use_container_width=True)
        else:
            st.success(f"No days found with fewer than {threshold} jobs.")

def render_timeline_analytics(df: pd.DataFrame):
    """Render timeline analytics and bottleneck identification."""
    st.header("📊 Timeline Analytics & Bottlenecks")
    
    today = pd.Timestamp.now()
    
    # Determine division if available
    df['Division_Type'] = 'Unknown'
    if 'Division' in df.columns:
        df['Division_Type'] = df['Division'].apply(
            lambda x: 'Laminate' if 'laminate' in str(x).lower() else 'Stone/Quartz'
        )
    
    # Timeline metrics configuration
    timeline_metrics = {
        "Template to Install": "Days_Template_to_Install",
        "Ready to Fab to Product Received": "Days_RTF_to_Product_Rcvd",
        "Template to Ready to Fab": "Days_Template_to_RTF",
        "Product Received to Install": "Days_Product_Rcvd_to_Install",
        "Template to Ship": "Days_Template_to_Ship",
        "Ship to Install": "Days_Ship_to_Install",
    }
    
    # Display timeline metrics by division
    st.subheader("⏱️ Average Timeline by Division")
    
    divisions = df['Division_Type'].unique()
    
    # Create columns for each division
    cols = st.columns(len(divisions))
    
    for idx, division in enumerate(divisions):
        with cols[idx]:
            st.markdown(f"**{division}**")
            division_df = df[df['Division_Type'] == division]
            
            metrics_data = []
            for metric_name, col_name in timeline_metrics.items():
                if col_name in division_df.columns:
                    avg_days = division_df[col_name].mean()
                    if pd.notna(avg_days):
                        metrics_data.append({
                            'Metric': metric_name,
                            'Avg Days': round(avg_days, 1)
                        })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # Display as metric cards
                for _, row in metrics_df.iterrows():
                    st.metric(row['Metric'], f"{row['Avg Days']} days")
            else:
                st.info("No timeline data available")
    
    st.markdown("---")
    
    # Bottleneck analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Bottlenecks")
        
        # Count jobs in each stage
        stage_counts = df['Current_Stage'].value_counts()
        
        # Display as a bar chart
        if not stage_counts.empty:
            st.bar_chart(stage_counts)
            
            # Show which stage has the most jobs
            bottleneck_stage = stage_counts.idxmax()
            st.info(f"📍 Potential bottleneck: **{bottleneck_stage}** ({stage_counts[bottleneck_stage]} jobs)")
    
    with col2:
        st.subheader("Stage Duration Analysis")
        
        # Show jobs stuck too long in each stage
        stuck_threshold = st.number_input(
            "Days threshold for 'stuck' jobs", 
            min_value=3, 
            max_value=30, 
            value=7,
            key="stuck_threshold"
        )
        
        stuck_jobs = df[df['Days_In_Current_Stage'] > stuck_threshold]
        if not stuck_jobs.empty:
            stuck_by_stage = stuck_jobs['Current_Stage'].value_counts()
            st.bar_chart(stuck_by_stage)
            st.warning(f"⚠️ {len(stuck_jobs)} jobs stuck > {stuck_threshold} days")
        else:
            st.success(f"✅ No jobs stuck > {stuck_threshold} days")
    
    # Timeline trend analysis
    st.markdown("---")
    st.subheader("📈 Timeline Trends (Last 30 Days)")
    
    # Select metric to analyze
    selected_metric = st.selectbox(
        "Select timeline metric to analyze",
        list(timeline_metrics.keys()),
        key="timeline_metric_select"
    )
    
    metric_col = timeline_metrics[selected_metric]
    recent_jobs = df[df['Install_Date'] >= today - timedelta(days=30)]
    
    if not recent_jobs.empty and metric_col in recent_jobs.columns:
        recent_jobs = recent_jobs.dropna(subset=[metric_col])
        if not recent_jobs.empty:
            # Show trend by division
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for division in divisions:
                division_jobs = recent_jobs[recent_jobs['Division_Type'] == division]
                if not division_jobs.empty:
                    # Group by week and calculate average
                    division_jobs['Week'] = division_jobs['Install_Date'].dt.to_period('W')
                    weekly_avg = division_jobs.groupby('Week')[metric_col].mean()
                    
                    if not weekly_avg.empty:
                        weekly_avg.plot(kind='line', marker='o', ax=ax, label=division)
            
            ax.set_xlabel('Week')
            ax.set_ylabel('Average Days')
            ax.set_title(f'{selected_metric} Timeline Trend')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info(f"No completed jobs with {selected_metric} data in the last 30 days")

def calculate_delay_probability(row):
    """Calculate probability of delay based on multiple factors."""
    risk_score = 0
    factors = []
    
    # Factor 1: Already behind schedule
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40
        factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    
    # Factor 2: Stuck in current stage
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_stage_duration = {
            'Post-Template': 3,
            'In Fabrication': 7,
            'Product Received': 5,
            'Shipped': 5
        }
        expected_duration = avg_stage_duration.get(row.get('Current_Stage', ''), 5)
        if row['Days_In_Current_Stage'] > expected_duration:
            risk_score += 20
            factors.append(f"Stuck in {row['Current_Stage']} for {row['Days_In_Current_Stage']:.0f} days")
    
    # Factor 3: Has rework
    if row.get('Has_Rework', False):
        risk_score += 15
        factors.append("Has rework")
    
    # Factor 4: Missing next activity
    if pd.isna(row.get('Next_Sched_Activity')):
        risk_score += 15
        factors.append("No next activity scheduled")
    
    # Factor 5: Historical performance (if salesperson has pattern of delays)
    # This would be more accurate with historical data
    if pd.notna(row.get('Salesperson')):
        # Placeholder - in real implementation, calculate based on historical performance
        risk_score += 0
    
    # Cap at 100%
    risk_score = min(risk_score, 100)
    
    return risk_score, factors

def render_predictive_analytics(df: pd.DataFrame):
    """Render predictive analytics for job delays."""
    st.header("🔮 Predictive Analytics")
    
    # Calculate delay probability for active jobs
    active_jobs = df[~df['Current_Stage'].isin(['Completed'])].copy()
    
    if active_jobs.empty:
        st.warning("No active jobs to analyze.")
        return
    
    # Calculate delay probability
    active_jobs[['Delay_Probability', 'Risk_Factors']] = active_jobs.apply(
        lambda row: pd.Series(calculate_delay_probability(row)), axis=1
    )
    
    # High risk jobs
    high_risk_threshold = st.slider(
        "High risk threshold (%)", 
        min_value=50, 
        max_value=90, 
        value=70,
        key="risk_threshold"
    )
    
    high_risk_jobs = active_jobs[active_jobs['Delay_Probability'] >= high_risk_threshold].sort_values(
        'Delay_Probability', ascending=False
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🚨 Jobs with >{high_risk_threshold}% Delay Risk ({len(high_risk_jobs)} jobs)")
        
        if not high_risk_jobs.empty:
            # Display high risk jobs
            for idx, row in high_risk_jobs.head(10).iterrows():
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{row.get('Job_Name', 'Unknown')}** - {row.get('Current_Stage', 'Unknown')}")
                        salesperson = row.get('Salesperson', 'N/A')
                        next_activity = row.get('Next_Sched_Activity', 'None scheduled')
                        st.caption(f"Salesperson: {salesperson} | Next: {next_activity}")
                        
                        # Show risk factors
                        if row['Risk_Factors']:
                            factors_text = " • ".join(row['Risk_Factors'])
                            st.warning(f"Risk factors: {factors_text}")
                    
                    with col_b:
                        # Color code based on risk level
                        color = "🔴" if row['Delay_Probability'] >= 80 else "🟡"
                        st.metric(
                            "Delay Risk",
                            f"{color} {row['Delay_Probability']:.0f}%"
                        )
                    st.markdown("---")
        else:
            st.success(f"No jobs with delay risk above {high_risk_threshold}%")
    
    with col2:
        st.subheader("📊 Risk Distribution")
        
        # Create risk distribution
        risk_bins = [0, 30, 50, 70, 100]
        risk_labels = ['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (70-100%)']
        active_jobs['Risk_Category'] = pd.cut(
            active_jobs['Delay_Probability'], 
            bins=risk_bins, 
            labels=risk_labels
        )
        
        risk_dist = active_jobs['Risk_Category'].value_counts()
        
        # Display as metrics
        for category in risk_labels:
            count = risk_dist.get(category, 0)
            if 'Critical' in category:
                st.metric(f"🔴 {category}", count)
            elif 'High' in category:
                st.metric(f"🟡 {category}", count)
            elif 'Medium' in category:
                st.metric(f"🟠 {category}", count)
            else:
                st.metric(f"🟢 {category}", count)
    
    # Predictive insights
    st.subheader("💡 Predictive Insights")
    
    insights = []
    
    # Insight 1: Stage-specific delays
    stage_delays = active_jobs.groupby('Current_Stage')['Delay_Probability'].agg(['mean', 'count'])
    worst_stage = stage_delays['mean'].idxmax() if not stage_delays.empty else None
    if worst_stage:
        insights.append({
            'Insight': f"Jobs in '{worst_stage}' stage have highest average delay risk",
            'Action': f"Focus on clearing {stage_delays.loc[worst_stage, 'count']} jobs in {worst_stage}",
            'Impact': 'High'
        })
    
    # Insight 2: Installer capacity
    if 'Install_Date' in active_jobs.columns and 'Install_Assigned_To' in active_jobs.columns:
        upcoming_installs = active_jobs[
            active_jobs['Install_Date'].notna() & 
            (active_jobs['Install_Date'] <= pd.Timestamp.now() + pd.Timedelta(days=7))
        ]
        if not upcoming_installs.empty:
            installer_risk = upcoming_installs.groupby('Install_Assigned_To')['Delay_Probability'].mean()
            if not installer_risk.empty:
                high_risk_installer = installer_risk[installer_risk > 60]
                if not high_risk_installer.empty:
                    insights.append({
                        'Insight': f"{len(high_risk_installer)} installers have high-risk jobs scheduled",
                        'Action': "Consider redistributing workload or adding resources",
                        'Impact': 'Medium'
                    })
    
    if insights:
        insights_df = pd.DataFrame(insights)
        st.dataframe(insights_df, use_container_width=True, hide_index=True)

def render_performance_scorecards(df: pd.DataFrame):
    """Render performance scorecards for employees."""
    st.header("📊 Performance Scorecards")
    
    # Select role to analyze
    role_type = st.selectbox(
        "Select Role to Analyze",
        ["Salesperson", "Template Assigned To", "Install Assigned To"],
        key="role_select"
    )
    
    # Map role to column
    role_column_map = {
        "Salesperson": "Salesperson",
        "Template Assigned To": "Template_Assigned_To",
        "Install Assigned To": "Install_Assigned_To"
    }
    
    role_col = role_column_map[role_type]
    
    if role_col not in df.columns:
        st.warning(f"Column '{role_col}' not found in data.")
        return
    
    # Get unique employees
    employees = df[role_col].dropna().unique()
    employees = [e for e in employees if str(e).strip()]
    
    if not employees:
        st.warning(f"No {role_type} data available.")
        return
    
    # Calculate metrics for each employee
    scorecards = []
    
    for employee in employees:
        emp_jobs = df[df[role_col] == employee]
        
        if role_type == "Salesperson":
            metrics = {
                'Employee': employee,
                'Total Jobs': len(emp_jobs),
                'Active Jobs': len(emp_jobs[~emp_jobs['Current_Stage'].isin(['Completed'])]),
                'Avg Days Behind': emp_jobs['Days_Behind'].mean() if 'Days_Behind' in emp_jobs.columns else 0,
                'Jobs w/ Rework': len(emp_jobs[emp_jobs.get('Has_Rework', False) == True]),
                'Avg Timeline': emp_jobs['Days_Template_to_Install'].mean() if 'Days_Template_to_Install' in emp_jobs.columns else None,
                'High Risk Jobs': len(emp_jobs[emp_jobs.get('Risk_Score', 0) >= 30])
            }
        
        elif role_type == "Template Assigned To":
            # Calculate template-specific metrics
            template_jobs = emp_jobs[emp_jobs['Template_Date'].notna()]
            now = pd.Timestamp.now()
            week_ago = now - pd.Timedelta(days=7)
            metrics = {
                'Employee': employee,
                'Total Templates': len(template_jobs),
                'Avg Template to RTF': template_jobs['Days_Template_to_RTF'].mean() if 'Days_Template_to_RTF' in template_jobs.columns else None,
                'Templates This Week': len(template_jobs[
                    (template_jobs['Template_Date'] >= week_ago) &
                    (template_jobs['Template_Date'] <= now)
                ]),
                'Upcoming Templates': len(template_jobs[
                    template_jobs['Template_Date'] > now
                ]),
                'Overdue RTF': len(template_jobs[
                    (template_jobs.get('Ready_to_Fab_Date', pd.Series()).isna()) & 
                    ((now - template_jobs['Template_Date']).dt.days > 3)
                ]) if 'Ready_to_Fab_Date' in template_jobs.columns else 0
            }
        
        else:  # Install Assigned To
            install_jobs = emp_jobs[emp_jobs['Install_Date'].notna()]
            now = pd.Timestamp.now()
            week_ago = now - pd.Timedelta(days=7)
            metrics = {
                'Employee': employee,
                'Total Installs': len(install_jobs),
                'Installs This Week': len(install_jobs[
                    (install_jobs['Install_Date'] >= week_ago) &
                    (install_jobs['Install_Date'] <= now)
                ]),
                'Upcoming Installs': len(install_jobs[
                    install_jobs['Install_Date'] > now
                ]),
                'Avg Ship to Install': install_jobs['Days_Ship_to_Install'].mean() if 'Days_Ship_to_Install' in install_jobs.columns else None,
                'Total SqFt': install_jobs['Total_Job_SqFT'].sum() if 'Total_Job_SqFT' in install_jobs.columns else 0
            }
        
        scorecards.append(metrics)
    
    # Convert to DataFrame for display
    if not scorecards:
        st.warning(f"No performance data available for {role_type}.")
        return
        
    scorecards_df = pd.DataFrame(scorecards)
    
    # Sort by relevant metric
    if role_type == "Salesperson":
        sort_col = 'Active Jobs'
    elif role_type == "Template Assigned To":
        sort_col = 'Total Templates'
    else:
        sort_col = 'Total Installs'
    
    scorecards_df = scorecards_df.sort_values(sort_col, ascending=False)
    
    # Display as cards
    st.subheader(f"🎯 {role_type} Performance")
    
    # Top performers
    cols = st.columns(min(3, len(scorecards_df)))
    for idx, (i, row) in enumerate(scorecards_df.head(3).iterrows()):
        if idx < len(cols):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"### {row['Employee']}")
                    
                    if role_type == "Salesperson":
                        st.metric("Active Jobs", row['Active Jobs'])
                        st.metric("Avg Days Behind", f"{row['Avg Days Behind']:.1f}" if pd.notna(row['Avg Days Behind']) else "N/A")
                        st.metric("High Risk Jobs", row['High Risk Jobs'], delta_color="inverse")
                    
                    elif role_type == "Template Assigned To":
                        st.metric("Templates This Week", row['Templates This Week'])
                        st.metric("Avg Template→RTF", f"{row['Avg Template to RTF']:.1f} days" if pd.notna(row['Avg Template to RTF']) else "N/A")
                        st.metric("Overdue RTF", row['Overdue RTF'], delta_color="inverse")
                    
                    else:  # Install
                        st.metric("Installs This Week", row['Installs This Week'])
                        st.metric("Upcoming Installs", row['Upcoming Installs'])
                        st.metric("Total SqFt", f"{row['Total SqFt']:,.0f}")
    
    # Full table view
    with st.expander("View All Employees"):
        # Format numeric columns
        format_dict = {}
        for col in scorecards_df.columns:
            if 'Avg' in col or 'Days' in col:
                format_dict[col] = '{:.1f}'
            elif 'SqFt' in col:
                format_dict[col] = '{:,.0f}'
        
        st.dataframe(
            scorecards_df.style.format(format_dict, na_rep='N/A'),
            use_container_width=True
        )
    
    # Performance insights
    st.subheader("🎯 Performance Insights")
    
    if role_type == "Salesperson" and 'Avg Days Behind' in scorecards_df.columns:
        valid_performers = scorecards_df.dropna(subset=['Avg Days Behind'])
        if not valid_performers.empty:
            best_performer = valid_performers.loc[valid_performers['Avg Days Behind'].idxmin()]
            worst_performer = valid_performers.loc[valid_performers['Avg Days Behind'].idxmax()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best On-Time Performance:** {best_performer['Employee']} (Avg {best_performer['Avg Days Behind']:.1f} days)")
            with col2:
                st.warning(f"**Needs Improvement:** {worst_performer['Employee']} (Avg {worst_performer['Avg Days Behind']:.1f} days behind)")
        else:
            st.info("Not enough data to determine performance leaders.")

def render_quick_actions(df: pd.DataFrame):
    """Render quick action items and recommendations."""
    st.header("⚡ Quick Actions & Recommendations")
    
    today = pd.Timestamp.now()
    
    # Generate actionable insights based on all analytics
    actions = []
    
    # Check for jobs with high delay probability (from predictive analytics)
    if 'Delay_Probability' in df.columns:
        critical_delay_jobs = df[
            (df['Delay_Probability'] >= 80) & 
            (~df['Current_Stage'].isin(['Completed']))
        ]
        if not critical_delay_jobs.empty:
            top_jobs = critical_delay_jobs.nlargest(3, 'Delay_Probability')['Job_Name'].tolist()
            actions.append({
                'Priority': '🔴 CRITICAL',
                'Action': f"Immediate intervention needed for {len(critical_delay_jobs)} jobs",
                'Details': f"Top priority: {', '.join(top_jobs[:3])}"
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
    
    # Check for bottlenecks
    stage_counts = df[~df['Current_Stage'].isin(['Completed'])]['Current_Stage'].value_counts()
    if not stage_counts.empty:
        bottleneck_stage = stage_counts.idxmax()
        if stage_counts[bottleneck_stage] > len(df) * 0.3:  # More than 30% of jobs in one stage
            actions.append({
                'Priority': '🟡 MEDIUM',
                'Action': f"Clear bottleneck in {bottleneck_stage} stage",
                'Details': f"{stage_counts[bottleneck_stage]} jobs stuck in this stage"
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
    
    # Additional insights with new predictive data
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Days_Template_to_Install' in df.columns:
            avg_cycle_time = df['Days_Template_to_Install'].mean()
            if pd.notna(avg_cycle_time):
                st.metric("Avg Cycle Time", f"{avg_cycle_time:.1f} days")
            else:
                st.metric("Avg Cycle Time", "N/A")
        else:
            st.metric("Avg Cycle Time", "N/A")
    
    with col2:
        if 'Days_Behind' in df.columns:
            on_time_jobs = df[df['Days_Behind'] <= 0]
            on_time_pct = len(on_time_jobs) / len(df) * 100 if len(df) > 0 else 0
            st.metric("On-Time Performance", f"{on_time_pct:.1f}%")
        else:
            st.metric("On-Time Performance", "N/A")
    
    with col3:
        if 'Delay_Probability' in df.columns:
            high_risk_pct = len(df[df['Delay_Probability'] >= 70]) / len(df) * 100 if len(df) > 0 else 0
            st.metric("High Risk Jobs", f"{high_risk_pct:.1f}%", delta_color="inverse")
        elif 'Has_Rework' in df.columns:
            jobs_with_rework = df[df['Has_Rework'] == True]
            rework_pct = len(jobs_with_rework) / len(df) * 100 if len(df) > 0 else 0
            st.metric("Rework Rate", f"{rework_pct:.1f}%", delta_color="inverse")
        else:
            st.metric("Active Jobs", len(df[~df['Current_Stage'].isin(['Completed'])]))
    
    with col4:
        if 'Risk_Score' in df.columns:
            avg_risk = df['Risk_Score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1f}", delta_color="inverse")
        else:
            st.metric("Active Jobs", len(df[~df['Current_Stage'].isin(['Completed'])]))

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
    
    # Usage tips
    with st.sidebar.expander("💡 Dashboard Tips"):
        st.markdown("""
        **🔮 Predictive Analytics:**
        - Jobs with >70% delay risk need immediate attention
        - Risk factors explain WHY a job might be delayed
        - Use insights to prevent delays before they happen
        
        **🎯 Performance Scorecards:**
        - Compare employee performance objectively
        - Identify training needs early
        - Reward top performers with data
        
        **📅 Workload Calendar:**
        - Find light days to schedule more work
        - Prevent employee burnout
        - Balance resources effectively
        """)
    
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
        st.metric("High Risk Jobs", len(high_risk), delta=f"{len(high_risk)/len(active_jobs)*100:.0f}%" if len(active_jobs) > 0 else None)
    with col4:
        today_activities = df[
            (df['Template_Date'].dt.date == today.date()) | 
            (df['Install_Date'].dt.date == today.date())
        ]
        st.metric("Today's Activities", len(today_activities))
    
def render_historical_placeholder(df: pd.DataFrame):
    """Placeholder for historical comparisons - shows what will be available."""
    st.header("📈 Historical Comparisons (Coming Soon)")
    
    st.info(
        "⚠️ Historical comparisons require importing completed jobs. "
        "Currently showing only active jobs. Once you import historical data, you'll see:"
    )
    
    # Mock visualization of what historical data would show
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 What You'll Be Able to Track")
        
        metrics_list = [
            "**Timeline Trends**: See if jobs are getting faster or slower over time",
            "**Seasonal Patterns**: Identify busy/slow periods throughout the year",
            "**Employee Performance**: Track improvement or decline in performance",
            "**Rework Trends**: Monitor if quality issues are increasing or decreasing",
            "**Customer Satisfaction**: Link completion times to customer feedback",
            "**Profitability Trends**: See how efficiency impacts your bottom line"
        ]
        
        for metric in metrics_list:
            st.markdown(f"• {metric}")
    
    with col2:
        st.subheader("🎯 Benefits of Historical Tracking")
        
        benefits = [
            "**Predict Seasonal Demand**: Staff appropriately for busy periods",
            "**Identify Training Needs**: See which employees need support",
            "**Optimize Processes**: Find which stages consistently cause delays",
            "**Set Realistic Goals**: Base targets on actual historical performance",
            "**Reward Top Performers**: Data-driven employee recognition"
        ]
        
        for benefit in benefits:
            st.markdown(f"• {benefit}")
    
    # Show current period summary as a preview
    st.markdown("---")
    st.subheader("📅 Current Period Summary (Active Jobs Only)")
    
    # Calculate some basic metrics from current data
    current_metrics = {
        "Active Jobs": len(df[~df['Current_Stage'].isin(['Completed'])]),
        "Avg Timeline (Template→Install)": f"{df['Days_Template_to_Install'].mean():.1f} days" if 'Days_Template_to_Install' in df.columns else "N/A",
        "Jobs with Rework": len(df[df['Has_Rework'] == True]) if 'Has_Rework' in df.columns else 0,
        "High Risk Jobs": len(df[df['Risk_Score'] >= 30]) if 'Risk_Score' in df.columns else 0
    }
    
    cols = st.columns(4)
    for idx, (metric, value) in enumerate(current_metrics.items()):
        with cols[idx]:
            st.metric(metric, value)
    
    st.markdown("---")
    st.markdown(
        "💡 **To enable historical comparisons**: Include completed jobs in your Google Sheets export "
        "or set up automated data archiving for completed jobs."
    )

    # Main content tabs
    tabs = st.tabs([
        "🚨 Daily Priorities",
        "📅 Workload Calendar", 
        "📊 Timeline Analytics",
        "🔮 Predictive Analytics",
        "🎯 Performance Scorecards",
        "⚡ Quick Actions",
        "📈 Historical Comparisons"
    ])
    
    with tabs[0]:
        render_daily_priorities(df, today)
    
    with tabs[1]:
        render_workload_calendar(df, today)
    
    with tabs[2]:
        render_timeline_analytics(df)
    
    with tabs[3]:
        render_predictive_analytics(df)
    
    with tabs[4]:
        render_performance_scorecards(df)
    
    with tabs[5]:
        render_quick_actions(df)
    
    with tabs[6]:
        render_historical_placeholder(df)
    
    # Auto-refresh note
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Refresh the page to get latest data")

if __name__ == "__main__":
    main()
