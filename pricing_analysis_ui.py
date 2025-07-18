# pricing_analysis_ui.py
"""
UI Component for Pricing Analysis Dashboard - FIXED for Your Data Structure
Works with your actual plant invoice data instead of Stone Details
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from business_logic import analyze_job_pricing

def analyze_interbranch_pricing_direct(df):
    """
    Analyze interbranch pricing using your actual data structure
    (Phase Throughput - Phase Plant Invoice)
    """
    results = []
    
    # Filter for jobs that should have interbranch costs (Stone/Quartz with plant costs)
    candidates = df[
        (df.get('Division_Type', '') == 'Stone/Quartz') &
        (df['Job_Material'].notna()) &
        (df['Total_Job_SqFT'].notna()) &
        (df['Total_Job_SqFT'] > 0) &
        (df['Total_Job_Price_'].notna()) &
        (df['Total_Job_Price_'] > 0) &
        (df['Phase_Dollars_Plant_Invoice_'].notna()) &
        (df['Phase_Dollars_Plant_Invoice_'] > 0)
    ]
    
    for idx, row in candidates.iterrows():
        # Use the existing pricing analysis from business_logic
        analysis = analyze_job_pricing(row)
        
        if isinstance(analysis, dict):
            # Extract key information
            job_info = {
                'Job_Name': row.get('Job_Name', ''),
                'Production_': row.get('Production_', ''),
                'Division': row.get('Division', ''),
                'Total_SqFt': row.get('Total_Job_SqFT', 0),
                'Actual_Plant_Cost': row.get('Phase_Dollars_Plant_Invoice_', 0),
                'Analysis': analysis
            }
            
            # Check if this job has interbranch cost analysis
            if analysis.get('status') == 'analyzed' and 'expected_plant' in analysis:
                expected_plant = analysis['expected_plant']
                job_info['Expected_Plant_Cost'] = expected_plant.get('total_cost_avg', 0)
                job_info['Variance_Amount'] = job_info['Actual_Plant_Cost'] - job_info['Expected_Plant_Cost']
                job_info['Variance_Percent'] = (job_info['Variance_Amount'] / job_info['Expected_Plant_Cost'] * 100) if job_info['Expected_Plant_Cost'] > 0 else 0
                
                # Determine severity
                if abs(job_info['Variance_Percent']) > 20:
                    job_info['Severity'] = 'critical'
                elif abs(job_info['Variance_Percent']) > 10:
                    job_info['Severity'] = 'warning'
                else:
                    job_info['Severity'] = 'normal'
                
                job_info['Has_Variance'] = True
            else:
                job_info['Has_Variance'] = False
                job_info['Expected_Plant_Cost'] = 0
                job_info['Variance_Amount'] = 0
                job_info['Variance_Percent'] = 0
                job_info['Severity'] = 'no_analysis'
            
            results.append(job_info)
    
    return results

def get_pricing_summary_stats_direct(pricing_results):
    """
    Generate summary statistics for the direct pricing analysis
    """
    total_jobs = len(pricing_results)
    
    # Jobs with variance analysis
    variance_jobs = [r for r in pricing_results if r.get('Has_Variance', False)]
    
    # Count by severity
    critical_variances = len([r for r in variance_jobs if r.get('Severity') == 'critical'])
    warning_variances = len([r for r in variance_jobs if r.get('Severity') == 'warning'])
    
    # Calculate totals
    total_expected = sum([r.get('Expected_Plant_Cost', 0) for r in variance_jobs])
    total_actual = sum([r.get('Actual_Plant_Cost', 0) for r in variance_jobs])
    
    return {
        'total_jobs': total_jobs,
        'analyzed_jobs': total_jobs,
        'jobs_with_variance': len(variance_jobs),
        'critical_variances': critical_variances,
        'warning_variances': warning_variances,
        'total_expected_cost': total_expected,
        'total_actual_cost': total_actual,
        'overall_variance': total_actual - total_expected if total_expected > 0 else 0,
        'overall_variance_percent': ((total_actual - total_expected) / total_expected * 100) if total_expected > 0 else 0
    }

def render_pricing_analysis_tab(df):
    """
    Render the comprehensive pricing analysis tab - FIXED for your data structure
    """
    st.header("🔍 Interbranch Pricing Analysis")
    st.markdown("Detailed analysis of plant invoice costs vs. expected interbranch pricing using your actual data structure.")
    
    if df.empty:
        st.warning("No data available for pricing analysis.")
        return
    
    # Generate pricing analysis using your actual data structure
    with st.spinner("Analyzing interbranch pricing for all jobs..."):
        pricing_results = analyze_interbranch_pricing_direct(df)
        summary_stats = get_pricing_summary_stats_direct(pricing_results)
    
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
        render_pricing_overview_direct(pricing_results, summary_stats)
    
    with analysis_tabs[1]:
        render_critical_variances_direct(pricing_results)
    
    with analysis_tabs[2]:
        render_variance_warnings_direct(pricing_results)
    
    with analysis_tabs[3]:
        render_detailed_analysis_direct(pricing_results)
    
    with analysis_tabs[4]:
        render_variance_trends_direct(pricing_results)

def render_pricing_summary_metrics(summary_stats):
    """
    Render high-level summary metrics
    """
    st.subheader("📈 Pricing Analysis Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Jobs Analyzed", 
            summary_stats['analyzed_jobs'],
            delta=f"{summary_stats['total_jobs']} candidates"
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

def render_pricing_overview_direct(pricing_results, summary_stats):
    """
    Render overview dashboard with charts and key insights
    """
    st.subheader("📊 Pricing Analysis Overview")
    
    if not pricing_results:
        st.info("No jobs found with interbranch pricing data.")
        return
    
    # Create variance distribution chart
    variance_jobs = [r for r in pricing_results if r.get('Has_Variance', False)]
    
    if variance_jobs:
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance distribution pie chart
            severity_counts = {
                'Normal': len([r for r in variance_jobs if r.get('Severity') == 'normal']),
                'Warning': len([r for r in variance_jobs if r.get('Severity') == 'warning']),
                'Critical': len([r for r in variance_jobs if r.get('Severity') == 'critical'])
            }
            
            # Only show chart if we have data
            if sum(severity_counts.values()) > 0:
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
            variance_amounts = [r.get('Variance_Percent', 0) for r in variance_jobs if r.get('Has_Variance', False)]
            
            if variance_amounts:
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
        
        # Sort by absolute variance amount
        sorted_variance_jobs = sorted(
            variance_jobs, 
            key=lambda x: abs(x.get('Variance_Amount', 0)), 
            reverse=True
        )
        
        if sorted_variance_jobs:
            variance_df = pd.DataFrame([
                {
                    'Job Name': r['Job_Name'],
                    'Production #': r['Production_'],
                    'Moraware Link': f"https://floformcountertops.moraware.net/sys/search?&search={r['Production_']}" if r['Production_'] else None,
                    'Expected Cost': f"${r.get('Expected_Plant_Cost', 0):,.2f}",
                    'Actual Cost': f"${r.get('Actual_Plant_Cost', 0):,.2f}",
                    'Variance': f"${r.get('Variance_Amount', 0):+,.2f}",
                    'Variance %': f"{r.get('Variance_Percent', 0):+.1f}%",
                    'Severity': r.get('Severity', 'unknown').title()
                }
                for r in sorted_variance_jobs[:10]
            ])
            
            st.dataframe(
                variance_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Moraware Link": st.column_config.LinkColumn(
                        "🔗 Moraware",
                        display_text="Open Job"
                    ),
                    "Severity": st.column_config.TextColumn(
                        "Severity",
                        help="Variance severity level"
                    )
                }
            )
    
    else:
        st.info("No jobs with variance analysis found. This could mean:")
        st.markdown("""
        - No quartz jobs with plant invoice costs found
        - Material recognition issues preventing analysis
        - All jobs are within normal variance ranges
        """)
        
        # Show what jobs we do have
        st.subheader("📋 Available Jobs for Analysis")
        
        jobs_summary = pd.DataFrame([
            {
                'Job Name': r['Job_Name'],
                'Production #': r['Production_'],
                'Division': r['Division'],
                'SqFt': r['Total_SqFt'],
                'Plant Cost': f"${r['Actual_Plant_Cost']:,.2f}",
                'Analysis Status': r['Analysis'].get('status', 'unknown') if isinstance(r['Analysis'], dict) else 'error'
            }
            for r in pricing_results[:10]  # Show first 10
        ])
        
        st.dataframe(jobs_summary, use_container_width=True, hide_index=True)

def render_critical_variances_direct(pricing_results):
    """
    Render critical variance issues requiring immediate attention
    """
    st.subheader("🚨 Critical Pricing Variances")
    
    critical_jobs = [r for r in pricing_results 
                    if r.get('Severity') == 'critical']
    
    if not critical_jobs:
        st.success("✅ No critical pricing variances found!")
        return
    
    st.warning(f"Found {len(critical_jobs)} jobs with critical pricing variances (>20% difference)")
    
    for job in critical_jobs:
        variance_pct = job.get('Variance_Percent', 0)
        
        with st.expander(
            f"🔴 {job['Job_Name']} - Variance: {variance_pct:+.1f}%",
            expanded=True
        ):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Cost", f"${job.get('Expected_Plant_Cost', 0):,.2f}")
            with col2:
                st.metric("Actual Cost", f"${job.get('Actual_Plant_Cost', 0):,.2f}")
            with col3:
                st.metric("Variance", f"${job.get('Variance_Amount', 0):+,.2f}")
            with col4:
                # Add Moraware link button
                if job.get('Production_'):
                    moraware_url = f"https://floformcountertops.moraware.net/sys/search?&search={job['Production_']}"
                    st.link_button("🔗 Open in Moraware", moraware_url)
            
            # Show analysis details if available
            analysis = job.get('Analysis', {})
            if isinstance(analysis, dict) and analysis.get('status') == 'analyzed':
                st.subheader("Analysis Details:")
                
                if 'expected_retail' in analysis:
                    retail_info = analysis['expected_retail']
                    st.write(f"• **Material Group**: {retail_info.get('material_group', 'Unknown')}")
                    st.write(f"• **Material Type**: {retail_info.get('material_type', 'Unknown')}")
                
                if 'expected_plant' in analysis:
                    plant_info = analysis['expected_plant']
                    st.write(f"• **Expected Cost Range**: ${plant_info.get('total_cost_min', 0):.2f} - ${plant_info.get('total_cost_max', 0):.2f}")

def render_variance_warnings_direct(pricing_results):
    """
    Render warning-level variances
    """
    st.subheader("⚠️ Pricing Variance Warnings")
    
    warning_jobs = [r for r in pricing_results 
                   if r.get('Severity') == 'warning']
    
    if not warning_jobs:
        st.success("✅ No pricing variance warnings found!")
        return
    
    st.info(f"Found {len(warning_jobs)} jobs with moderate pricing variances (10-20% difference)")
    
    warning_df = pd.DataFrame([
        {
            'Job Name': job['Job_Name'],
            'Production #': job['Production_'],
            'Moraware Link': f"https://floformcountertops.moraware.net/sys/search?&search={job['Production_']}" if job['Production_'] else None,
            'Total SqFt': job['Total_SqFt'],
            'Expected Cost': job.get('Expected_Plant_Cost', 0),
            'Actual Cost': job.get('Actual_Plant_Cost', 0),
            'Variance Amount': job.get('Variance_Amount', 0),
            'Variance %': job.get('Variance_Percent', 0)
        }
        for job in warning_jobs
    ])
    
    st.dataframe(
        warning_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Moraware Link": st.column_config.LinkColumn(
                "🔗 Moraware",
                display_text="Open Job"
            ),
            "Expected Cost": st.column_config.NumberColumn("Expected Cost", format="$%.2f"),
            "Actual Cost": st.column_config.NumberColumn("Actual Cost", format="$%.2f"),
            "Variance Amount": st.column_config.NumberColumn("Variance Amount", format="$%.2f"),
            "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%")
        }
    )

def render_detailed_analysis_direct(pricing_results):
    """
    Render detailed analysis for all jobs
    """
    st.subheader("📋 Detailed Analysis")
    
    if not pricing_results:
        st.info("No pricing analysis results available.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        show_variances_only = st.checkbox("Show only jobs with variances", value=False)
    
    with col2:
        show_critical_only = st.checkbox("Show only critical/warning issues", value=False)
    
    # Apply filters
    filtered_results = pricing_results
    
    if show_variances_only:
        filtered_results = [r for r in filtered_results if r.get('Has_Variance', False)]
    
    if show_critical_only:
        filtered_results = [r for r in filtered_results if r.get('Severity') in ['critical', 'warning']]
    
    if filtered_results:
        detailed_df = pd.DataFrame([
            {
                'Job Name': r['Job_Name'],
                'Production #': r['Production_'],
                'Moraware Link': f"https://floformcountertops.moraware.net/sys/search?&search={r['Production_']}" if r['Production_'] else None,
                'SqFt': r['Total_SqFt'],
                'Expected Cost': r.get('Expected_Plant_Cost', 0),
                'Actual Cost': r.get('Actual_Plant_Cost', 0),
                'Variance': r.get('Variance_Amount', 0),
                'Variance %': r.get('Variance_Percent', 0),
                'Severity': r.get('Severity', 'unknown').title(),
                'Analysis Status': r['Analysis'].get('status', 'unknown') if isinstance(r['Analysis'], dict) else 'error'
            }
            for r in filtered_results
        ])
        
        st.dataframe(
            detailed_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Moraware Link": st.column_config.LinkColumn(
                    "🔗 Moraware",
                    display_text="Open Job"
                ),
                "Expected Cost": st.column_config.NumberColumn("Expected Cost", format="$%.2f"),
                "Actual Cost": st.column_config.NumberColumn("Actual Cost", format="$%.2f"),
                "Variance": st.column_config.NumberColumn("Variance", format="$%.2f"),
                "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%")
            }
        )
        
        # Download option
        csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Analysis",
            data=csv,
            file_name="interbranch_pricing_analysis.csv",
            mime="text/csv"
        )
    else:
        st.info("No data matches the current filters.")

def render_variance_trends_direct(pricing_results):
    """
    Render variance trend analysis
    """
    st.subheader("📈 Variance Trends & Insights")
    
    if not pricing_results:
        st.info("No variance data available for trend analysis.")
        return
    
    variance_jobs = [r for r in pricing_results if r.get('Has_Variance', False)]
    
    if not variance_jobs:
        st.info("No jobs with variance data found.")
        return
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    over_budget = len([j for j in variance_jobs if j.get('Variance_Amount', 0) > 0])
    under_budget = len([j for j in variance_jobs if j.get('Variance_Amount', 0) < 0])
    
    if variance_jobs:
        avg_variance = sum(j.get('Variance_Amount', 0) for j in variance_jobs) / len(variance_jobs)
        max_overage = max((j.get('Variance_Amount', 0) for j in variance_jobs), default=0)
        min_variance = min((j.get('Variance_Amount', 0) for j in variance_jobs), default=0)
        
        insights = [
            f"📊 **Budget Performance**: {over_budget} jobs over budget, {under_budget} jobs under budget",
            f"💰 **Average Variance**: ${avg_variance:,.2f}",
            f"📈 **Largest Overage**: ${max_overage:,.2f}",
            f"📉 **Largest Savings**: ${min_variance:,.2f}"
        ]
        
        for insight in insights:
            st.info(insight)
    else:
        st.info("No variance data available for insights.")
