"""
UI Components Module for FloForm Dashboard
Contains all user interface rendering functions and components
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from business_logic import (
    calculate_business_health_score, get_critical_issues, 
    calculate_performance_metrics, generate_business_insights,
    calculate_timeline_metrics, TIMELINE_THRESHOLDS
)
from data_processing import filter_data, get_data_summary
from visualization import (
    create_timeline_chart, create_risk_distribution_chart,
    create_performance_metrics_chart, create_health_score_gauge
)

def render_login_screen():
    """Enhanced login screen with professional styling."""
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("""
        <style>
            .login-container {
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                height: 70vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px; padding: 2rem; margin: auto; color: white;
                text-align: center; max-width: 500px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .login-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
            .login-subtitle { font-size: 1.2rem; opacity: 0.9; margin-bottom: 2rem; }
            .login-features {
                text-align: left; margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🚀 FloForm Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Unified Operations & Profitability Intelligence</div>', unsafe_allow_html=True)

    with st.form("pin_auth_form"):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed", placeholder="Enter PIN")
        submitted = st.form_submit_button("🔓 Access Dashboard", use_container_width=True)
        
        if submitted:
            correct_pin = st.secrets.get("APP_PIN", "1234")
            if pin == correct_pin:
                st.session_state.authenticated = True
                st.success("✅ Authentication successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid PIN. Please try again.")
    
    st.markdown("""
        <div class="login-features">
            <strong>Dashboard Features:</strong><br>
            • Real-time operational analytics<br>
            • Material pricing validation<br>
            • Risk assessment & predictions<br>
            • Performance scorecards<br>
            • Business health monitoring
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return False

def render_sidebar_config():
    """Enhanced sidebar configuration with data refresh controls."""
    
    st.sidebar.header("⚙️ Dashboard Configuration")
    
    # Date and cost configuration
    today_dt = pd.to_datetime(st.sidebar.date_input(
        "📅 Reference Date", 
        value=datetime.now().date(),
        help="Set the reference date for all calculations"
    ))
    
    install_cost_sqft = st.sidebar.number_input(
        "💰 Install Cost per SqFt ($)", 
        min_value=0.0, 
        value=15.0, 
        step=0.50,
        help="Standard installation cost per square foot"
    )
    
    # Data refresh controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("🔄 **Data Controls**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Reload data from Google Sheets"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", help="Clear all cached data"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox(
        "🔄 Auto-refresh (5 min)", 
        help="Automatically refresh data every 5 minutes"
    )
    
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Performance monitoring
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Performance**")
    
    # Memory usage (if psutil is available)
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        st.sidebar.caption(f"💾 Memory: {memory_mb:.1f} MB")
        
        if memory_mb > 500:
            st.sidebar.warning("⚠️ High memory usage")
    except ImportError:
        pass
    
    return {
        'today_dt': today_dt,
        'install_cost': install_cost_sqft,
        'auto_refresh': auto_refresh
    }

def render_overall_health_tab(df, today):
    """Enhanced business health overview with comprehensive metrics."""
    
    st.header("📈 Overall Business Health Dashboard")
    st.markdown("Comprehensive view of business performance across all divisions and metrics.")
    
    if df.empty:
        st.warning("No data available for analysis.")
        return
    
    # Calculate business health score
    overall_health, health_components = calculate_business_health_score(df)
    
    # Executive summary metrics
    st.subheader("🎯 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = len(df)
        active_jobs = len(df[df['Job_Status'] != 'Complete'])
        st.metric("Total Jobs", total_jobs, delta=f"{active_jobs} active")
    
    with col2:
        total_revenue = df.get('Revenue', pd.Series([0])).sum()
        avg_job_value = total_revenue / total_jobs if total_jobs > 0 else 0
        st.metric("Total Revenue", f"${total_revenue:,.0f}", delta=f"${avg_job_value:,.0f} avg/job")
    
    with col3:
        avg_profit_margin = df.get('Branch_Profit_Margin_%', pd.Series([0])).mean()
        profitable_jobs = len(df[df.get('Branch_Profit_Margin_%', 0) > 0])
        st.metric("Avg Profit Margin", f"{avg_profit_margin:.1f}%", delta=f"{profitable_jobs} profitable")
    
    with col4:
        high_risk_jobs = len(df[df.get('Risk_Score', 0) >= 30])
        risk_rate = (high_risk_jobs / active_jobs * 100) if active_jobs > 0 else 0
        st.metric("High Risk Jobs", high_risk_jobs, delta=f"{risk_rate:.1f}% of active")
    
    with col5:
        total_sqft = df.get('Total_Job_SqFt', pd.Series([0])).sum()
        avg_sqft = total_sqft / total_jobs if total_jobs > 0 else 0
        st.metric("Total SqFt", f"{total_sqft:,.0f}", delta=f"{avg_sqft:.0f} avg/job")
    
    st.markdown("---")
    
    # Business Health Score Display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Health score gauge
        fig = create_health_score_gauge(overall_health)
        st.plotly_chart(fig, use_container_width=True)
        
        # Health status
        if overall_health >= 80:
            st.success(f"🟢 **Excellent Performance** ({overall_health:.0f}/100)")
        elif overall_health >= 60:
            st.info(f"🟡 **Good Performance** ({overall_health:.0f}/100)")
        else:
            st.warning(f"🔴 **Needs Attention** ({overall_health:.0f}/100)")
    
    with col2:
        # Health components breakdown
        if health_components:
            st.subheader("📋 Health Score Components")
            
            components_data = []
            for component, (score, weight) in health_components.items():
                status = '🟢' if score >= 80 else '🟡' if score >= 60 else '🔴'
                components_data.append({
                    'Component': component,
                    'Score': f"{score:.0f}/100",
                    'Weight': f"{weight}%",
                    'Status': status
                })
            
            components_df = pd.DataFrame(components_data)
            st.dataframe(components_df, use_container_width=True, hide_index=True)
    
    # Division performance comparison
    if 'Division_Type' in df.columns:
        st.subheader("🏢 Division Performance Comparison")
        
        division_metrics = df.groupby('Division_Type').agg({
            'Job_Name': 'count',
            'Revenue': ['sum', 'mean'],
            'Branch_Profit_Margin_%': 'mean',
            'Total_Job_SqFt': 'sum',
            'Risk_Score': 'mean'
        }).round(2)
        
        # Flatten column names
        division_metrics.columns = ['_'.join(col).strip() for col in division_metrics.columns.values]
        division_metrics = division_metrics.rename(columns={
            'Job_Name_count': 'Total Jobs',
            'Revenue_sum': 'Total Revenue ($)',
            'Revenue_mean': 'Avg Revenue/Job ($)',
            'Branch_Profit_Margin_%_mean': 'Avg Profit Margin (%)',
            'Total_Job_SqFt_sum': 'Total SqFt',
            'Risk_Score_mean': 'Avg Risk Score'
        })
        
        st.dataframe(division_metrics, use_container_width=True)
    
    # Key insights and recommendations
    st.subheader("💡 Business Insights & Recommendations")
    insights = generate_business_insights(df, today)
    
    for insight in insights:
        st.info(insight)
    
    # Critical issues summary
    critical_issues = get_critical_issues(df, today)
    
    if critical_issues:
        st.subheader("⚠️ Critical Issues Summary")
        
        for issue_type, issue_data in critical_issues.items():
            severity_emoji = "🔴" if issue_data['severity'] == 'critical' else "🟡"
            st.warning(f"{severity_emoji} **{issue_data['description']}**: {issue_data['count']} jobs")
    else:
        st.success("✅ No critical issues identified across all operations.")

def render_operational_dashboard(df, today):
    """Enhanced operational performance dashboard."""
    
    st.header("⚙️ Operational Performance Dashboard")
    st.markdown("Real-time operational efficiency, risk assessment, and workload management.")
    
    # Operational filters
    with st.expander("🔧 Operational Filters", expanded=False):
        op_cols = st.columns(3)
        
        with op_cols[0]:
            status_options = ["Active", "Complete", "30+ Days Old", "Unscheduled"]
            status_filter = st.multiselect(
                "Job Status Filter", 
                status_options, 
                default=["Active"], 
                key="op_status_multi"
            )
        
        with op_cols[1]:
            salesperson_list = ['All'] + sorted(df['Salesperson'].dropna().unique().tolist())
            salesperson_filter = st.selectbox("Salesperson Filter", salesperson_list, key="op_sales")
        
        with op_cols[2]:
            division_list = ['All'] + sorted(df['Division_Type'].dropna().unique().tolist())
            division_filter = st.selectbox("Division Filter", division_list, key="op_div")

    # Apply filters
    filters = {
        'status_filter': status_filter,
        'salesperson': salesperson_filter,
        'division': division_filter
    }
    
    df_filtered = filter_data(df, filters)
    st.info(f"📊 Displaying **{len(df_filtered):,}** jobs based on current filters")

    # Operational sub-tabs
    op_tabs = st.tabs([
        "🚨 Daily Priorities", 
        "📅 Workload Calendar", 
        "📊 Timeline Analytics", 
        "🔮 Predictive Analytics", 
        "🎯 Performance Scorecards"
    ])
    
    with op_tabs[0]:
        render_daily_priorities(df_filtered, today)
    
    with op_tabs[1]:
        render_workload_calendar(df_filtered, today)
    
    with op_tabs[2]:
        render_timeline_analytics(df_filtered)
    
    with op_tabs[3]:
        render_predictive_analytics(df_filtered)
    
    with op_tabs[4]:
        render_performance_scorecards(df_filtered)

def render_daily_priorities(df, today):
    """Enhanced daily priorities with actionable insights."""
    
    st.subheader("🚨 Daily Priorities & Critical Alerts")
    
    # Key priority metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = len(df[df['Risk_Score'] >= 30])
        st.metric("🔴 High Risk Jobs", high_risk_count)
    
    with col2:
        behind_schedule = len(df[df['Days_Behind'] > 0])
        st.metric("⏰ Behind Schedule", behind_schedule)
    
    with col3:
        stuck_jobs = len(df[df['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']])
        st.metric("🚧 Stuck Jobs", stuck_jobs)
    
    with col4:
        stale_jobs = df[
            (df['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']) & 
            (df['Job_Status'] != 'Complete')
        ]
        st.metric("💨 Stale Jobs", len(stale_jobs))

    # Get critical issues
    critical_issues = get_critical_issues(df, today)
    
    if not critical_issues:
        st.success("✅ No critical issues found! All active jobs are on track.")
        return
    
    # Display critical issues
    for issue_type, issue_data in critical_issues.items():
        severity = issue_data['severity']
        expanded = severity == 'critical'
        
        severity_emoji = "🔴" if severity == 'critical' else "🟡"
        
        with st.expander(
            f"{severity_emoji} {issue_data['description']} ({issue_data['count']} jobs)", 
            expanded=expanded
        ):
            issue_df = issue_data['data']
            
            if not issue_df.empty:
                display_cols = ['Job_Name', 'Current_Stage', 'Salesperson', 'Days_In_Current_Stage']
                available_cols = [col for col in display_cols if col in issue_df.columns]
                
                if 'Link' in issue_df.columns:
                    available_cols.insert(0, 'Link')
                
                st.dataframe(
                    issue_df[available_cols],
                    column_config={
                        "Link": st.column_config.LinkColumn(
                            "Prod #", 
                            display_text=r".*search=(.*)"
                        )
                    },
                    use_container_width=True
                )

def render_workload_calendar(df, today):
    """Enhanced workload calendar with capacity planning."""
    
    st.subheader("📅 Workload Calendar & Resource Planning")
    
    # Date range and activity selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=today.date())
    with col2:
        end_date = st.date_input("End Date", value=(today + timedelta(days=14)).date())
    
    activity_type = st.selectbox("Activity Type", ["Templates", "Installs", "All Activities"])
    
    # Create workload visualization
    fig = create_timeline_chart(df, start_date, end_date, activity_type)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activities scheduled in the selected date range.")

def render_timeline_analytics(df):
    """Enhanced timeline analytics with process insights."""
    
    st.subheader("📊 Timeline Analytics & Process Bottlenecks")
    
    # Calculate timeline metrics
    timeline_metrics = calculate_timeline_metrics(df)
    
    if not timeline_metrics:
        st.warning("No timeline data available for analysis.")
        return
    
    # Display timeline performance
    st.subheader("⏱️ Process Performance Metrics")
    
    cols = st.columns(3)
    col_idx = 0
    
    for process, metrics in timeline_metrics.items():
        with cols[col_idx % 3]:
            st.metric(
                process,
                f"{metrics['mean']:.1f} days avg",
                delta=f"{metrics['median']:.1f} median"
            )
        col_idx += 1
    
    # Bottleneck analysis
    active_jobs = df[df['Job_Status'] != 'Complete']
    
    if not active_jobs.empty:
        st.subheader("🚧 Current Process Bottlenecks")
        
        stage_counts = active_jobs['Current_Stage'].value_counts()
        fig = px.bar(
            x=stage_counts.index,
            y=stage_counts.values,
            title="Active Jobs by Current Stage",
            labels={'x': 'Stage', 'y': 'Number of Jobs'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_predictive_analytics(df):
    """Enhanced predictive analytics with risk modeling."""
    
    st.subheader("🔮 Predictive Analytics & Risk Assessment")
    
    active_jobs = df[~df['Current_Stage'].isin(['Completed'])]
    
    if active_jobs.empty:
        st.warning("No active jobs to analyze.")
        return
    
    # Risk distribution
    fig = create_risk_distribution_chart(active_jobs)
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk jobs detail
    high_risk_threshold = st.slider("Risk Threshold (%)", 50, 90, 70)
    high_risk_jobs = active_jobs[active_jobs['Delay_Probability'] >= high_risk_threshold]
    
    if not high_risk_jobs.empty:
        st.subheader(f"⚠️ Jobs Above {high_risk_threshold}% Risk Threshold")
        
        risk_display_cols = ['Job_Name', 'Current_Stage', 'Delay_Probability', 'Risk_Factors', 'Salesperson']
        available_risk_cols = [col for col in risk_display_cols if col in high_risk_jobs.columns]
        
        st.dataframe(
            high_risk_jobs[available_risk_cols].sort_values('Delay_Probability', ascending=False),
            use_container_width=True
        )
    else:
        st.success(f"✅ No jobs exceed {high_risk_threshold}% risk threshold.")

def render_performance_scorecards(df):
    """Enhanced performance scorecards with detailed analytics."""
    
    st.subheader("🎯 Performance Scorecards & Team Analytics")
    
    # Role selection
    role_type = st.selectbox("Select Role", ["Salesperson", "Template Assigned To", "Install Assigned To"])
    
    # Calculate performance metrics
    today = pd.Timestamp.now()
    scorecards = calculate_performance_metrics(df, role_type, today)
    
    if not scorecards:
        st.warning(f"No {role_type} data available.")
        return
    
    scorecards_df = pd.DataFrame(scorecards)
    
    # Display top performers
    st.subheader(f"🏆 Top Performers: {role_type}")
    
    top_performers = min(3, len(scorecards_df))
    cols = st.columns(top_performers)
    
    for idx, (_, row) in enumerate(scorecards_df.head(top_performers).iterrows()):
        if idx < len(cols):
            with cols[idx]:
                with st.container(border=True):
                    rank_emoji = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "🏅"
                    st.markdown(f"### {rank_emoji} {row['Employee']}")
                    
                    # Display relevant metrics based on role
                    if role_type == "Salesperson":
                        st.metric("Active Jobs", f"{row['Active Jobs']:.0f}")
                        st.metric("Total Revenue", f"${row['Total Revenue']:,.0f}")
                        if pd.notna(row['Avg Profit Margin']):
                            st.metric("Avg Margin", f"{row['Avg Profit Margin']:.1f}%")
                    
                    elif role_type == "Template Assigned To":
                        st.metric("Templates This Week", f"{row['Templates This Week']:.0f}")
                        st.metric("Total SqFt", f"{row['Total SqFt']:,.0f}")
                        if pd.notna(row['Avg Template to RTF']):
                            st.metric("Avg Template→RTF", f"{row['Avg Template to RTF']:.1f} days")
                    
                    else:  # Install Assigned To
                        st.metric("Installs This Week", f"{row['Installs This Week']:.0f}")
                        st.metric("Total SqFt", f"{row['Total SqFt']:,.0f}")
                        st.metric("Jobs w/ Rework", f"{row['Jobs w/ Rework']:.0f}")
    
    # Detailed performance table
    with st.expander("📊 Detailed Performance Data"):
        st.dataframe(scorecards_df, use_container_width=True, hide_index=True)

def render_profitability_dashboard(df_stone, df_laminate, today):
    """Enhanced profitability dashboard with comprehensive financial analysis."""
    
    st.header("💰 Profitability Analysis Dashboard")
    st.markdown("Comprehensive financial performance analysis by division.")

    # Division tabs
    profit_tabs = st.tabs(["💎 Stone/Quartz", "🪵 Laminate", "📊 Combined Analysis"])
    
    with profit_tabs[0]:
        render_division_profitability(df_stone, "Stone/Quartz")
    
    with profit_tabs[1]:
        render_division_profitability(df_laminate, "Laminate")
    
    with profit_tabs[2]:
        render_combined_profitability_analysis(df_stone, df_laminate)

def render_division_profitability(df, division_name):
    """Render profitability analysis for a specific division."""
    
    st.subheader(f"💼 {division_name} Financial Performance")
    
    if df.empty:
        st.warning(f"No {division_name} data available.")
        return
    
    # Key financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df.get('Revenue', pd.Series([0])).sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_profit = df.get('Branch_Profit', pd.Series([0])).sum()
        st.metric("Total Profit", f"${total_profit:,.0f}")
    
    with col3:
        avg_margin = df.get('Branch_Profit_Margin_%', pd.Series([0])).mean()
        st.metric("Avg Margin", f"{avg_margin:.1f}%")
    
    with col4:
        job_count = len(df)
        avg_job_value = total_revenue / job_count if job_count > 0 else 0
        st.metric("Avg Job Value", f"${avg_job_value:,.0f}")
    
    # Profitability visualizations
    if 'Branch_Profit_Margin_%' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Profit Margin Distribution")
            fig = px.histogram(
                df, 
                x='Branch_Profit_Margin_%',
                nbins=20,
                title="Distribution of Profit Margins"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Material_Group' in df.columns:
                st.subheader("💎 Profitability by Material Group")
                material_profit = df.groupby('Material_Group')['Branch_Profit_Margin_%'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=material_profit.index,
                    y=material_profit.values,
                    title="Average Margin by Material Group"
                )
                st.plotly_chart(fig, use_container_width=True)

def render_combined_profitability_analysis(df_stone, df_laminate):
    """Render combined profitability analysis across divisions."""
    
    st.subheader("📊 Combined Division Analysis")
    
    # Combine data for comparison
    df_stone_summary = df_stone.copy() if not df_stone.empty else pd.DataFrame()
    df_laminate_summary = df_laminate.copy() if not df_laminate.empty else pd.DataFrame()
    
    if df_stone_summary.empty and df_laminate_summary.empty:
        st.warning("No data available for combined analysis.")
        return
    
    # Division comparison metrics
    comparison_data = []
    
    for division_name, df_div in [("Stone/Quartz", df_stone_summary), ("Laminate", df_laminate_summary)]:
        if not df_div.empty:
            comparison_data.append({
                'Division': division_name,
                'Total Jobs': len(df_div),
                'Total Revenue': df_div.get('Revenue', pd.Series([0])).sum(),
                'Total Profit': df_div.get('Branch_Profit', pd.Series([0])).sum(),
                'Avg Margin %': df_div.get('Branch_Profit_Margin_%', pd.Series([0])).mean(),
                'Avg Job Value': df_div.get('Revenue', pd.Series([0])).mean()
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format for display
        display_df = comparison_df.copy()
        display_df['Total Revenue'] = display_df['Total Revenue'].apply(lambda x: f"${x:,.0f}")
        display_df['Total Profit'] = display_df['Total Profit'].apply(lambda x: f"${x:,.0f}")
        display_df['Avg Margin %'] = display_df['Avg Margin %'].apply(lambda x: f"{x:.1f}%")
        display_df['Avg Job Value'] = display_df['Avg Job Value'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Division performance visualization
        if len(comparison_data) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    comparison_df,
                    values='Total Revenue',
                    names='Division',
                    title="Revenue Distribution by Division"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    comparison_df,
                    x='Division',
                    y='Avg Margin %',
                    title="Average Profit Margin by Division"
                )
                st.plotly_chart(fig, use_container_width=True)
