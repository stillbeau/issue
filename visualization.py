"""
Visualization Module for FloForm Dashboard
Contains all chart creation and plotting functions using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define consistent color schemes
FLOFORM_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

RISK_COLORS = {
    'low': '#28a745',      # Green
    'medium': '#ffc107',   # Yellow  
    'high': '#fd7e14',     # Orange
    'critical': '#dc3545'  # Red
}

DIVISION_COLORS = {
    'Stone/Quartz': '#667eea',
    'Laminate': '#764ba2'
}

def setup_plotly_theme():
    """Set up consistent Plotly theme for all charts."""
    
    # Configure default template
    plotly_template = {
        'layout': {
            'colorway': list(FLOFORM_COLORS.values()),
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'title': {'font': {'size': 16, 'color': FLOFORM_COLORS['dark']}},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'grid': {'color': 'rgba(0,0,0,0.1)'},
            'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
        }
    }
    
    # Apply theme to all new figures
    px.defaults.template = "plotly_white"

def create_health_score_gauge(score):
    """Create a gauge chart for business health score."""
    
    # Determine color based on score
    if score >= 80:
        color = RISK_COLORS['low']
        status = "Excellent"
    elif score >= 60:
        color = FLOFORM_COLORS['warning']
        status = "Good"
    else:
        color = RISK_COLORS['critical']
        status = "Needs Attention"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Business Health Score<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
        delta = {'reference': 75, 'position': "bottom"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, 60], 'color': "rgba(220,53,69,0.2)"},
                {'range': [60, 80], 'color': "rgba(255,193,7,0.2)"},
                {'range': [80, 100], 'color': "rgba(40,167,69,0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 14}
    )
    
    return fig

def create_timeline_chart(df, start_date, end_date, activity_type):
    """Create timeline chart for workload calendar."""
    
    if df.empty:
        return None
    
    # Filter data by date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Process activity data based on type
    if activity_type == "Templates" and 'Template_Date' in df.columns:
        activity_df = df[df['Template_Date'].notna()].copy()
        date_col, assignee_col = 'Template_Date', 'Template_Assigned_To'
        
    elif activity_type == "Installs" and 'Install_Date' in df.columns:
        activity_df = df[df['Install_Date'].notna()].copy()
        date_col, assignee_col = 'Install_Date', 'Install_Assigned_To'
        
    elif activity_type == "All Activities":
        activities = []
        
        if 'Template_Date' in df.columns:
            temp_df = df[df['Template_Date'].notna()].copy()
            temp_df = temp_df.rename(columns={
                'Template_Date': 'Activity_Date', 
                'Template_Assigned_To': 'Assignee'
            })
            temp_df['Activity_Type'] = 'Template'
            activities.append(temp_df)
            
        if 'Install_Date' in df.columns:
            inst_df = df[df['Install_Date'].notna()].copy()
            inst_df = inst_df.rename(columns={
                'Install_Date': 'Activity_Date', 
                'Install_Assigned_To': 'Assignee'
            })
            inst_df['Activity_Type'] = 'Install'
            activities.append(inst_df)
            
        if activities:
            activity_df = pd.concat(activities, ignore_index=True)
            date_col, assignee_col = 'Activity_Date', 'Assignee'
        else:
            return None
    else:
        return None
    
    # Filter by date range
    activity_df = activity_df[
        (activity_df[date_col] >= pd.Timestamp(start_date)) & 
        (activity_df[date_col] <= pd.Timestamp(end_date))
    ]
    
    if activity_df.empty:
        return None
    
    # Create daily summary
    daily_summary = []
    for date in date_range:
        day_activities = activity_df[activity_df[date_col].dt.date == date.date()]
        
        if not day_activities.empty and assignee_col in day_activities.columns:
            assignee_counts = day_activities[assignee_col].value_counts()
            
            for assignee, count in assignee_counts.items():
                if assignee and str(assignee).strip():
                    daily_summary.append({
                        'Date': date,
                        'Assignee': str(assignee),
                        'Job_Count': int(count),
                        'Activity_Type': activity_type
                    })
    
    if not daily_summary:
        return None
    
    # Create heatmap
    summary_df = pd.DataFrame(daily_summary)
    
    try:
        pivot_df = summary_df.pivot_table(
            index='Assignee',
            columns=summary_df['Date'].dt.strftime('%m/%d'),
            values='Job_Count',
            fill_value=0,
            aggfunc='sum'
        )
        
        fig = px.imshow(
            pivot_df,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title=f"{activity_type} Workload Heatmap",
            labels={'color': 'Job Count'}
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Assignee"
        )
        
        return fig
        
    except Exception:
        # Fallback to bar chart if heatmap fails
        daily_totals = summary_df.groupby('Date')['Job_Count'].sum().reset_index()
        daily_totals['Date_Str'] = daily_totals['Date'].dt.strftime('%m/%d')
        
        fig = px.bar(
            daily_totals,
            x='Date_Str',
            y='Job_Count',
            title=f"Daily {activity_type} Count",
            labels={'Job_Count': 'Number of Jobs', 'Date_Str': 'Date'}
        )
        
        return fig

def create_risk_distribution_chart(df):
    """Create risk distribution pie chart."""
    
    if df.empty or 'Delay_Probability' not in df.columns:
        return None
    
    # Categorize risk levels
    risk_bins = [0, 30, 50, 70, 101]
    risk_labels = ['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (70-100%)']
    
    df['Risk_Category'] = pd.cut(
        df['Delay_Probability'],
        bins=risk_bins,
        labels=risk_labels,
        right=False
    )
    
    risk_dist = df['Risk_Category'].value_counts()
    
    if risk_dist.empty:
        return None
    
    # Create pie chart with custom colors
    colors = [RISK_COLORS['low'], RISK_COLORS['medium'], RISK_COLORS['high'], RISK_COLORS['critical']]
    
    fig = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title="Risk Distribution of Active Jobs",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_performance_metrics_chart(scorecards_df, role_type):
    """Create performance metrics chart for team scorecards."""
    
    if scorecards_df.empty:
        return None
    
    # Determine key metric based on role
    if role_type == "Salesperson":
        y_col = 'Active Jobs'
        title = "Active Jobs by Salesperson"
    elif role_type == "Template Assigned To":
        y_col = 'Total Templates'
        title = "Templates by Team Member"
    else:  # Install Assigned To
        y_col = 'Total Installs'
        title = "Installs by Team Member"
    
    if y_col not in scorecards_df.columns:
        return None
    
    # Create horizontal bar chart
    sorted_df = scorecards_df.sort_values(y_col, ascending=True)
    
    fig = px.bar(
        sorted_df,
        x=y_col,
        y='Employee',
        orientation='h',
        title=title,
        color=y_col,
        color_continuous_scale='Blues',
        labels={y_col: y_col.replace('_', ' '), 'Employee': 'Team Member'}
    )
    
    fig.update_layout(
        height=max(400, len(scorecards_df) * 40),
        yaxis_title="Team Member",
        showlegend=False
    )
    
    return fig

def create_timeline_metrics_chart(timeline_data):
    """Create timeline metrics comparison chart."""
    
    if not timeline_data:
        return None
    
    processes = list(timeline_data.keys())
    means = [timeline_data[process]['mean'] for process in processes]
    medians = [timeline_data[process]['median'] for process in processes]
    
    # Create DataFrame for Plotly
    chart_data = pd.DataFrame({
        'Process': processes + processes,
        'Days': means + medians,
        'Metric': ['Average'] * len(processes) + ['Median'] * len(processes)
    })
    
    fig = px.bar(
        chart_data,
        x='Process',
        y='Days',
        color='Metric',
        title='Process Timeline Comparison',
        labels={'Days': 'Days', 'Process': 'Process'},
        barmode='group'
    )
    
    fig.update_layout(height=400)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_revenue_trend_chart(df):
    """Create revenue trend chart over time."""
    
    if df.empty or 'Job_Creation' not in df.columns or 'Revenue' not in df.columns:
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=['Job_Creation', 'Revenue'])
    
    if df_clean.empty:
        return None
    
    # Group by month
    df_clean['Month'] = df_clean['Job_Creation'].dt.to_period('M')
    monthly_revenue = df_clean.groupby('Month')['Revenue'].sum().reset_index()
    monthly_revenue['Month_Str'] = monthly_revenue['Month'].astype(str)
    
    # Create line chart
    fig = px.line(
        monthly_revenue,
        x='Month_Str',
        y='Revenue',
        title='Monthly Revenue Trend',
        markers=True,
        labels={'Revenue': 'Revenue ($)', 'Month_Str': 'Month'}
    )
    
    fig.update_traces(
        line_color=FLOFORM_COLORS['primary'],
        line_width=3,
        marker_size=8
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode='x unified'
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')

    return fig


def create_monthly_installs_trend(df, today):
    """Create line chart showing installs for the current month."""

    if df.empty or 'Install_Date' not in df.columns:
        return None

    start_of_month = pd.Timestamp(today.year, today.month, 1)
    month_df = df[(df['Install_Date'].notna()) &
                  (df['Install_Date'] >= start_of_month) &
                  (df['Install_Date'] <= today)]

    if month_df.empty:
        return None

    daily_installs = (month_df
                      .groupby(month_df['Install_Date'].dt.date)
                      .size()
                      .reset_index(name='Installs'))

    fig = px.line(
        daily_installs,
        x='Install_Date',
        y='Installs',
        title='Daily Installs This Month',
        markers=True,
        labels={'Install_Date': 'Date', 'Installs': 'Number of Installs'}
    )

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

    return fig

def create_profitability_scatter(df):
    """Create profitability scatter plot (Revenue vs Profit)."""
    
    if df.empty or 'Revenue' not in df.columns or 'Branch_Profit' not in df.columns:
        return None
    
    # Clean data
    plot_df = df.dropna(subset=['Revenue', 'Branch_Profit'])
    
    if plot_df.empty:
        return None
    
    # Add color coding by division if available
    color_col = 'Division_Type' if 'Division_Type' in plot_df.columns else None
    
    fig = px.scatter(
        plot_df,
        x='Revenue',
        y='Branch_Profit',
        color=color_col,
        title='Revenue vs Branch Profit',
        hover_data=['Job_Name'] if 'Job_Name' in plot_df.columns else None,
        color_discrete_map=DIVISION_COLORS if color_col else None
    )
    
    # Add diagonal line for break-even
    max_val = max(plot_df['Revenue'].max(), plot_df['Branch_Profit'].max())
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="gray", width=2, dash="dash"),
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Revenue ($)",
        yaxis_title="Branch Profit ($)"
    )
    
    # Format axes as currency
    fig.update_xaxes(tickformat='$,.0f')
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

def create_stage_distribution_chart(df):
    """Create current stage distribution chart."""
    
    if df.empty or 'Current_Stage' not in df.columns:
        return None
    
    # Filter to active jobs only
    active_jobs = df[df.get('Job_Status', '') != 'Complete']
    
    if active_jobs.empty:
        return None
    
    stage_counts = active_jobs['Current_Stage'].value_counts()
    
    # Create DataFrame for Plotly
    stage_data = pd.DataFrame({
        'Stage': stage_counts.index,
        'Job_Count': stage_counts.values
    })
    
    # Create pie chart
    fig = px.pie(
        stage_data,
        values='Job_Count',
        names='Stage',
        title="Active Jobs by Current Stage"
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_material_group_chart(df):
    """Create material group distribution chart."""
    
    if df.empty or 'Material_Group' not in df.columns:
        return None
    
    # Clean data and count material groups
    material_counts = df['Material_Group'].dropna().value_counts()
    
    if material_counts.empty:
        return None
    
    # Create bar chart
    chart_data = pd.DataFrame({
        'Material_Group': material_counts.index.astype(str),
        'Job_Count': material_counts.values
    })
    
    fig = px.bar(
        chart_data,
        x='Material_Group',
        y='Job_Count',
        title="Jobs by Material Group",
        labels={'Job_Count': 'Number of Jobs', 'Material_Group': 'Material Group'}
    )
    
    fig.update_traces(marker_color=FLOFORM_COLORS['primary'])
    
    fig.update_layout(height=400)
    
    return fig

def create_rework_analysis_chart(df):
    """Create rework analysis chart."""
    
    if df.empty or 'Has_Rework' not in df.columns:
        return None
    
    # Calculate rework rates
    total_jobs = len(df)
    rework_jobs = df['Has_Rework'].sum()
    no_rework_jobs = total_jobs - rework_jobs
    
    if total_jobs == 0:
        return None
    
    # Create pie chart
    fig = px.pie(
        values=[no_rework_jobs, rework_jobs],
        names=['No Rework', 'Has Rework'],
        title=f"Rework Rate: {(rework_jobs/total_jobs)*100:.1f}%",
        color_discrete_sequence=[RISK_COLORS['low'], RISK_COLORS['critical']]
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_combined_timeline_chart(df):
    """Create combined timeline chart showing multiple process stages."""
    
    if df.empty:
        return None
    
    # Define timeline columns
    timeline_cols = {
        'Days_Template_to_RTF': 'Template → RTF',
        'Days_RTF_to_Product_Rcvd': 'RTF → Product Received',
        'Days_Product_Rcvd_to_Install': 'Product → Install'
    }
    
    # Check which columns are available
    available_cols = {col: name for col, name in timeline_cols.items() if col in df.columns}
    
    if not available_cols:
        return None
    
    # Calculate averages for each stage
    averages = []
    for col, name in available_cols.items():
        avg_days = df[col].mean()
        if pd.notna(avg_days):
            averages.append({'Stage': name, 'Average_Days': avg_days})
    
    if not averages:
        return None
    
    averages_df = pd.DataFrame(averages)
    
    # Create horizontal bar chart
    chart_data = pd.DataFrame({
        'Stage': averages_df['Stage'],
        'Average_Days': averages_df['Average_Days']
    })
    
    fig = px.bar(
        chart_data,
        x='Average_Days',
        y='Stage',
        orientation='h',
        title='Average Process Stage Durations',
        color='Average_Days',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=300,
        xaxis_title="Average Days",
        yaxis_title="Process Stage",
        showlegend=False
    )
    
    return fig

def create_division_comparison_chart(df_stone, df_laminate):
    """Create division comparison chart."""
    
    comparison_data = []
    
    for division_name, df_div in [("Stone/Quartz", df_stone), ("Laminate", df_laminate)]:
        if not df_div.empty:
            comparison_data.append({
                'Division': division_name,
                'Jobs': len(df_div),
                'Revenue': df_div.get('Revenue', pd.Series([0])).sum(),
                'Avg_Margin': df_div.get('Branch_Profit_Margin_%', pd.Series([0])).mean()
            })
    
    if len(comparison_data) < 2:
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Job Count', 'Total Revenue', 'Average Margin %'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add job count
    fig.add_trace(
        go.Bar(x=comparison_df['Division'], y=comparison_df['Jobs'], name='Jobs'),
        row=1, col=1
    )
    
    # Add revenue
    fig.add_trace(
        go.Bar(x=comparison_df['Division'], y=comparison_df['Revenue'], name='Revenue'),
        row=1, col=2
    )
    
    # Add margin
    fig.add_trace(
        go.Bar(x=comparison_df['Division'], y=comparison_df['Avg_Margin'], name='Margin %'),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Division Performance Comparison',
        height=400,
        showlegend=False
    )
    
    return fig

def format_currency(value):
    """Format value as currency string."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"

def format_percentage(value):
    """Format value as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"

def get_color_scale(values, color_scheme='Blues'):
    """Get appropriate color scale for values."""
    color_scales = {
        'Blues': px.colors.sequential.Blues,
        'Reds': px.colors.sequential.Reds,
        'Greens': px.colors.sequential.Greens,
        'Oranges': px.colors.sequential.Oranges
    }
    
    return color_scales.get(color_scheme, px.colors.sequential.Blues)
