# pricing_analysis_ui.py
"""
UI Component for Pricing Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from enhanced_pricing_analysis import (
    generate_pricing_report, 
    get_pricing_summary_stats,
    analyze_job_interbranch_pricing
)

def render_pricing_analysis_tab(df):
    """
    Render the comprehensive pricing analysis tab.
    """
    st.header("🔍 Interbranch Pricing Analysis")
    st.markdown("Detailed analysis of plant invoice costs vs. expected interbranch pricing using Stone Details data.")
    
    if df.empty:
        st.warning("No data available for pricing analysis.")
        return
    
    # Generate pricing analysis
    with st.spinner("Analyzing interbranch pricing for all jobs..."):
        pricing_results = generate_pricing_report(df)
        summary_stats = get_pricing_summary_stats(pricing_results)
    
    # Summary metrics
    render_pricing_summary_metrics(summary_stats)
    
    # Analysis tabs
    analysis_tabs = st.tabs([
        "📊 Overview Dashboard",
        "🚨 Critical Variances", 
        "⚠️ Warnings",
        "📋 Detailed Analysis",
        "📈 Variance Trends"
    ])
    
    with analysis_tabs[0]:
        render_pricing_overview(pricing_results, summary_stats)
    
    with analysis_tabs[1]:
        render_critical_variances(pricing_results)
    
    with analysis_tabs[2]:
        render_variance_warnings(pricing_results)
    
    with analysis_tabs[3]:
        render_detailed_analysis(pricing_results)
    
    with analysis_tabs[4]:
        render_variance_trends(pricing_results)

def render_pricing_summary_metrics(summary_stats):
    """
    Render high-level summary metrics.
    """
    st.subheader("📈 Pricing Analysis Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Jobs Analyzed", 
            summary_stats['analyzed_jobs'],
            delta=f"{summary_stats['total_jobs']} total"
        )
    
    with col2:
        st.metric(
            "Jobs with Variances", 
            summary_stats['jobs_with_variance']
        )
    
    with col3:
        critical_count = summary_stats['critical_variances']
        st.metric(
            "Critical Issues", 
            critical_count,
            delta="🔴" if critical_count > 0 else "✅"
        )
    
    with col4:
        warning_count = summary_stats['warning_variances']
        st.metric(
            "Warnings", 
            warning_count,
            delta="🟡" if warning_count > 0 else "✅"
        )
    
    with col5:
        overall_variance = summary_stats['overall_variance_percent']
        st.metric(
            "Overall Variance", 
            f"{overall_variance:+.1f}%",
            delta=f"${summary_stats['overall_variance']:+,.0f}"
        )
    
    # Health indicator
    if summary_stats['critical_variances'] == 0 and summary_stats['warning_variances'] == 0:
        st.success("✅ All interbranch pricing appears to be within normal ranges.")
    elif summary_stats['critical_variances'] == 0:
        st.info(f"🟡 {summary_stats['warning_variances']} jobs have pricing variances that may need review.")
    else:
        st.error(f"🔴 {summary_stats['critical_variances']} jobs have critical pricing variances requiring immediate attention.")

def render_pricing_overview(pricing_results, summary_stats):
    """
    Render overview dashboard with charts and key insights.
    """
    st.subheader("📊 Pricing Analysis Overview")
    
    # Create variance distribution chart
    variance_jobs = [r for r in pricing_results 
                    if r['Analysis'].get('variance_analysis') is not None]
    
    if variance_jobs:
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance distribution pie chart
            severity_counts = {
                'Normal': len([r for r in variance_jobs if r['Analysis']['variance_analysis']['severity'] == 'normal']),
                'Warning': len([r for r in variance_jobs if r['Analysis']['variance_analysis']['severity'] == 'warning']),
                'Critical': len([r for r in variance_jobs if r['Analysis']['variance_analysis']['severity'] == 'critical'])
            }
            
            fig_pie = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Pricing Variance Severity Distribution",
                color_discrete_map={
                    'Normal': '#28a745',
                    'Warning': '#ffc107', 
                    'Critical': '#dc3545'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Variance amount histogram
            variance_amounts = [r['Analysis']['variance_analysis']['variance_percent'] 
                              for r in variance_jobs]
            
            fig_hist = px.histogram(
                x=variance_amounts,
                nbins=20,
                title="Distribution of Variance Percentages",
                labels={'x': 'Variance %', 'y': 'Number of Jobs'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Expected")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top variance jobs
        st.subheader("🔍 Largest Pricing Variances")
        
        variance_df = pd.DataFrame([
            {
                'Job Name': r['Job_Name'],
                'Production #': r['Production_'],
                'Expected Cost': f"${r['Analysis']['total_expected_cost']:,.2f}",
                'Actual Cost': f"${r['Analysis']['actual_plant_cost']:,.2f}",
                'Variance': f"${r['Analysis']['variance_analysis']['variance_amount']:+,.2f}",
                'Variance %': f"{r['Analysis']['variance_analysis']['variance_percent']:+.1f}%",
                'Severity': r['Analysis']['variance_analysis']['severity'].title()
            }
            for r in sorted(variance_jobs, 
                          key=lambda x: abs(x['Analysis']['variance_analysis']['variance_amount']), 
                          reverse=True)[:10]
        ])
        
        st.dataframe(
            variance_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Severity": st.column_config.TextColumn(
                    "Severity",
                    help="Variance severity level"
                )
            }
        )

def render_critical_variances(pricing_results):
    """
    Render critical variance issues requiring immediate attention.
    """
    st.subheader("🚨 Critical Pricing Variances")
    
    critical_jobs = [r for r in pricing_results 
                    if r['Analysis'].get('variance_analysis') and 
                    r['Analysis']['variance_analysis']['severity'] == 'critical']
    
    if not critical_jobs:
        st.success("✅ No critical pricing variances found!")
        return
    
    st.warning(f"Found {len(critical_jobs)} jobs with critical pricing variances (>20% difference)")
    
    for job in critical_jobs:
        analysis = job['Analysis']
        variance = analysis['variance_analysis']
        
        with st.expander(
            f"🔴 {job['Job_Name']} - Variance: {variance['variance_percent']:+.1f}%",
            expanded=True
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expected Cost", f"${analysis['total_expected_cost']:,.2f}")
            with col2:
                st.metric("Actual Cost", f"${analysis['actual_plant_cost']:,.2f}")
            with col3:
                st.metric("Variance", f"${variance['variance_amount']:+,.2f}")
            
            # Material breakdown
            st.subheader("Material Breakdown:")
            for material in analysis['material_details']:
                if material.get('recognized', False):
                    expected = material.get('expected_cost')
                    if expected:
                        st.write(f"• **{material['material']}** ({material['sqft']} sq ft)")
                        st.write(f"  - Group {material['material_group']} - Expected: ${expected['total_cost_avg']:.2f}")
                    else:
                        st.write(f"• **{material['material']}** ({material['sqft']} sq ft) - {material.get('note', 'No cost data')}")
                else:
                    st.write(f"• **{material['material']}** - ❌ {material.get('note', 'Not recognized')}")

def render_variance_warnings(pricing_results):
    """
    Render warning-level variances.
    """
    st.subheader("⚠️ Pricing Variance Warnings")
    
    warning_jobs = [r for r in pricing_results 
                   if r['Analysis'].get('variance_analysis') and 
                   r['Analysis']['variance_analysis']['severity'] == 'warning']
    
    if not warning_jobs:
        st.success("✅ No pricing variance warnings found!")
        return
    
    st.info(f"Found {len(warning_jobs)} jobs with moderate pricing variances (10-20% difference)")
    
    warning_df = pd.DataFrame([
        {
            'Job Name': job['Job_Name'],
            'Production #': job['Production_'],
            'Total SqFt': job['Total_SqFt'],
            'Expected Cost': job['Analysis']['total_expected_cost'],
            'Actual Cost': job['Analysis']['actual_plant_cost'],
            'Variance Amount': job['Analysis']['variance_analysis']['variance_amount'],
            'Variance %': job['Analysis']['variance_analysis']['variance_percent']
        }
        for job in warning_jobs
    ])
    
    st.dataframe(
        warning_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Expected Cost": st.column_config.NumberColumn("Expected Cost", format="$%.2f"),
            "Actual Cost": st.column_config.NumberColumn("Actual Cost", format="$%.2f"),
            "Variance Amount": st.column_config.NumberColumn("Variance Amount", format="$%.2f"),
            "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%")
        }
    )

def render_detailed_analysis(pricing_results):
    """
    Render detailed material-by-material analysis.
    """
    st.subheader("📋 Detailed Material Analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        show_recognized_only = st.checkbox("Show only recognized materials", value=True)
    
    with col2:
        show_variances_only = st.checkbox("Show only jobs with variances", value=False)
    
    # Process data for detailed view
    detailed_data = []
    
    for job in pricing_results:
        if job['Analysis']['status'] != 'analyzed':
            continue
            
        if show_variances_only and not job['Analysis'].get('variance_analysis'):
            continue
        
        for material in job['Analysis']['material_details']:
            if show_recognized_only and not material.get('recognized', False):
                continue
            
            detailed_data.append({
                'Job Name': job['Job_Name'],
                'Production #': job['Production_'],
                'Material': material['material'],
                'SqFt': material['sqft'],
                'Material Type': material.get('material_type', 'Unknown'),
                'Material Group': material.get('material_group', 'Unknown'),
                'Recognized': '✅' if material.get('recognized') else '❌',
                'Expected Cost/SqFt': material.get('expected_cost', {}).get('cost_per_sqft_avg', 0) if material.get('expected_cost') else 0,
                'Expected Total': material.get('expected_cost', {}).get('total_cost_avg', 0) if material.get('expected_cost') else 0,
                'Notes': material.get('note', '')
            })
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        
        st.dataframe(
            detailed_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Expected Cost/SqFt": st.column_config.NumberColumn("Expected Cost/SqFt", format="$%.2f"),
                "Expected Total": st.column_config.NumberColumn("Expected Total", format="$%.2f")
            }
        )
        
        # Download option
        csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Analysis",
            data=csv,
            file_name="pricing_analysis_detailed.csv",
            mime="text/csv"
        )
    else:
        st.info("No data matches the current filters.")

def render_variance_trends(pricing_results):
    """
    Render variance trend analysis.
    """
    st.subheader("📈 Variance Trends & Insights")
    
    variance_jobs = [r for r in pricing_results 
                    if r['Analysis'].get('variance_analysis') is not None]
    
    if not variance_jobs:
        st.info("No variance data available for trend analysis.")
        return
    
    # Material group variance analysis
    group_variances = {}
    
    for job in variance_jobs:
        for material in job['Analysis']['material_details']:
            if material.get('material_group') is not None:
                group = f"Group {material['material_group']}"
                if group not in group_variances:
                    group_variances[group] = []
                
                if material.get('expected_cost'):
                    # Calculate this material's contribution to variance
                    expected = material['expected_cost']['total_cost_avg']
                    # Approximate actual cost proportionally
                    job_variance_ratio = job['Analysis']['variance_analysis']['variance_amount'] / job['Analysis']['total_expected_cost']
                    actual = expected * (1 + job_variance_ratio)
                    material_variance = actual - expected
                    
                    group_variances[group].append(material_variance)
    
    # Create group variance chart
    if group_variances:
        group_avg_variances = {group: sum(variances)/len(variances) 
                              for group, variances in group_variances.items()}
        
        fig_group = px.bar(
            x=list(group_avg_variances.keys()),
            y=list(group_avg_variances.values()),
            title="Average Variance by Material Group",
            labels={'x': 'Material Group', 'y': 'Average Variance ($)'}
        )
        st.plotly_chart(fig_group, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    over_budget = len([j for j in variance_jobs if j['Analysis']['variance_analysis']['variance_amount'] > 0])
    under_budget = len([j for j in variance_jobs if j['Analysis']['variance_analysis']['variance_amount'] < 0])
    
    insights = [
        f"📊 **Budget Performance**: {over_budget} jobs over budget, {under_budget} jobs under budget",
        f"💰 **Average Variance**: ${sum(j['Analysis']['variance_analysis']['variance_amount'] for j in variance_jobs) / len(variance_jobs):,.2f}",
        f"📈 **Largest Overage**: ${max(j['Analysis']['variance_analysis']['variance_amount'] for j in variance_jobs):,.2f}",
        f"📉 **Largest Savings**: ${min(j['Analysis']['variance_analysis']['variance_amount'] for j in variance_jobs):,.2f}"
    ]
    
    for insight in insights:
        st.info(insight)
