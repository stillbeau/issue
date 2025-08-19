diff --git a/pricing_analysis_ui.py b/pricing_analysis_ui.py
index 36c9ac7b30dca9345abd88c3ca87f5ebcbcecad2..1d0fd0d79575a3fa9c413a986e194c6c76f30543 100644
--- a/pricing_analysis_ui.py
+++ b/pricing_analysis_ui.py
@@ -1,194 +1,284 @@
 # pricing_analysis_ui.py
 """
 UI Component for Pricing Analysis Dashboard - FIXED Data Type Issues
 Works with your actual plant invoice data and handles string/numeric conversion
 """
 
 import streamlit as st
 import pandas as pd
 import plotly.express as px
 import plotly.graph_objects as go
 from business_logic import analyze_job_pricing
 
+# Jobs where the plant invoice exceeds expected cost by more than this
+# percentage will be flagged as potentially overpriced.
+OVERPRICE_THRESHOLD = 15
+
 def safe_numeric_conversion(value):
     """Safely convert value to numeric, handling strings with $ and commas"""
     if pd.isna(value):
         return 0.0
     
     # Convert to string and clean
     str_value = str(value).replace('$', '').replace(',', '').strip()
     
     try:
         return float(str_value)
     except (ValueError, TypeError):
         return 0.0
 
 def analyze_interbranch_pricing_direct(df):
     """
     Analyze interbranch pricing using your actual data structure
     (Phase Throughput - Phase Plant Invoice) - FIXED for data type issues
     """
     results = []
     
     # Create a copy and ensure numeric columns are properly converted
     df_work = df.copy()
     
     # Convert key columns to numeric with safe conversion
-    numeric_columns = ['Total_Job_SqFT', 'Total_Job_Price_', 'Phase_Dollars_Plant_Invoice_']
+    numeric_columns = ['Total_Job_SqFT', 'Total_Job_Price_', 'Phase_Dollars_Plant_Invoice_', 'Job_Plant_Invoice']
     
     for col in numeric_columns:
         if col in df_work.columns:
             df_work[col] = df_work[col].apply(safe_numeric_conversion)
     
     # Filter for jobs that should have interbranch costs (Stone/Quartz with plant costs)
     try:
+        division_col = (
+            'Division_Type' if 'Division_Type' in df_work.columns
+            else 'Division' if 'Division' in df_work.columns
+            else None
+        )
+
+        division_mask = (
+            df_work[division_col] == 'Stone/Quartz'
+            if division_col else pd.Series([True] * len(df_work))
+        )
+
         candidates = df_work[
-            (df_work.get('Division_Type', '') == 'Stone/Quartz') &
+            division_mask &
             (df_work['Job_Material'].notna()) &
             (df_work['Total_Job_SqFT'].notna()) &
             (df_work['Total_Job_SqFT'] > 0) &
             (df_work['Total_Job_Price_'].notna()) &
             (df_work['Total_Job_Price_'] > 0) &
             (df_work['Phase_Dollars_Plant_Invoice_'].notna()) &
-            (df_work['Phase_Dollars_Plant_Invoice_'] > 0)
+            (df_work['Phase_Dollars_Plant_Invoice_'] >= 0)
         ]
     except Exception as e:
         st.error(f"Error filtering data: {e}")
         st.info("Attempting alternative filtering method...")
-        
+
         # Alternative filtering with more robust checks
         mask = pd.Series([True] * len(df_work))
-        
+
         # Division check
         if 'Division_Type' in df_work.columns:
             mask &= (df_work['Division_Type'] == 'Stone/Quartz')
-        
+        elif 'Division' in df_work.columns:
+            mask &= (df_work['Division'] == 'Stone/Quartz')
+
         # Material check
         if 'Job_Material' in df_work.columns:
             mask &= df_work['Job_Material'].notna()
-        
+
         # Numeric checks with safe conversion
         for col in ['Total_Job_SqFT', 'Total_Job_Price_', 'Phase_Dollars_Plant_Invoice_']:
             if col in df_work.columns:
-                mask &= (df_work[col] > 0)
-        
+                if col == 'Phase_Dollars_Plant_Invoice_':
+                    mask &= (df_work[col] >= 0)
+                else:
+                    mask &= (df_work[col] > 0)
+
         candidates = df_work[mask]
     
     st.info(f"Found {len(candidates)} candidate jobs for interbranch pricing analysis")
     
     for idx, row in candidates.iterrows():
         # Use the existing pricing analysis from business_logic
         try:
             analysis = analyze_job_pricing(row)
         except Exception as e:
             st.warning(f"Error analyzing job {row.get('Job_Name', 'Unknown')}: {e}")
             continue
         
         if isinstance(analysis, dict):
             # Extract key information with safe conversions
+            plant_cost = safe_numeric_conversion(row.get('Phase_Dollars_Plant_Invoice_', 0))
+            plant_invoice = safe_numeric_conversion(row.get('Job_Plant_Invoice', 0))
             job_info = {
                 'Job_Name': row.get('Job_Name', ''),
                 'Production_': row.get('Production_', ''),
                 'Division': row.get('Division', ''),
                 'Total_SqFt': safe_numeric_conversion(row.get('Total_Job_SqFT', 0)),
-                'Actual_Plant_Cost': safe_numeric_conversion(row.get('Phase_Dollars_Plant_Invoice_', 0)),
+                'Job_Revenue': safe_numeric_conversion(row.get('Total_Job_Price_', 0)),
+                'Actual_Plant_Cost': plant_cost,
+                'Actual_Plant_Invoice': plant_invoice,
                 'Analysis': analysis
             }
             
             # Check if this job has interbranch cost analysis
             if analysis.get('status') == 'analyzed' and 'expected_plant' in analysis:
                 expected_plant = analysis['expected_plant']
                 job_info['Expected_Plant_Cost'] = expected_plant.get('total_cost_avg', 0)
-                job_info['Variance_Amount'] = job_info['Actual_Plant_Cost'] - job_info['Expected_Plant_Cost']
-                job_info['Variance_Percent'] = (job_info['Variance_Amount'] / job_info['Expected_Plant_Cost'] * 100) if job_info['Expected_Plant_Cost'] > 0 else 0
+
+                # If no actual plant cost, fall back to estimated values
+                if plant_cost <= 0:
+                    job_info['Plant_Invoice_Used'] = job_info['Expected_Plant_Cost']
+                    job_info['Invoice_Type'] = 'Estimated'
+                    job_info['Plant_Profit'] = 0
+                    job_info['Plant_Profit_Margin_%'] = 0
+                else:
+                    # Determine which invoice value to use
+                    if plant_invoice > 0:
+                        job_info['Plant_Invoice_Used'] = plant_invoice
+                        job_info['Invoice_Type'] = 'Actual'
+                    else:
+                        job_info['Plant_Invoice_Used'] = job_info['Expected_Plant_Cost']
+                        job_info['Invoice_Type'] = 'Estimated'
+
+                    job_info['Plant_Profit'] = job_info['Plant_Invoice_Used'] - plant_cost
+                    job_info['Plant_Profit_Margin_%'] = (
+                        job_info['Plant_Profit'] / job_info['Plant_Invoice_Used'] * 100
+                    ) if job_info['Plant_Invoice_Used'] > 0 else 0
+                job_info['Expected_Cost_per_SqFt'] = (
+                    job_info['Expected_Plant_Cost'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Invoice_per_SqFt'] = (
+                    job_info['Plant_Invoice_Used'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Plant_Cost_per_SqFt'] = (
+                    plant_cost / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Plant_Profit_per_SqFt'] = (
+                    job_info['Plant_Profit'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+
+                # Compare actual plant cost to expected cost
+                job_info['Variance_Amount'] = plant_cost - job_info['Expected_Plant_Cost']
+                job_info['Variance_Percent'] = (
+                    job_info['Variance_Amount'] / job_info['Expected_Plant_Cost'] * 100
+                ) if job_info['Expected_Plant_Cost'] > 0 else 0
+                job_info['Overpriced'] = job_info['Variance_Percent'] > OVERPRICE_THRESHOLD
                 
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
+
+                if job_info['Actual_Plant_Cost'] > 0:
+                    job_info['Plant_Invoice_Used'] = job_info['Actual_Plant_Invoice']
+                    job_info['Invoice_Type'] = 'Actual' if job_info['Actual_Plant_Invoice'] > 0 else 'Estimated'
+                    job_info['Plant_Profit'] = job_info['Plant_Invoice_Used'] - job_info['Actual_Plant_Cost']
+                    job_info['Plant_Profit_Margin_%'] = (
+                        job_info['Plant_Profit'] / job_info['Plant_Invoice_Used'] * 100
+                    ) if job_info['Plant_Invoice_Used'] > 0 else 0
+                else:
+                    job_info['Plant_Invoice_Used'] = job_info['Actual_Plant_Invoice'] if job_info['Actual_Plant_Invoice'] > 0 else 0
+                    job_info['Invoice_Type'] = 'Estimated'
+                    job_info['Plant_Profit'] = 0
+                    job_info['Plant_Profit_Margin_%'] = 0
+
+                job_info['Expected_Cost_per_SqFt'] = 0
+                job_info['Invoice_per_SqFt'] = (
+                    job_info['Plant_Invoice_Used'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Plant_Cost_per_SqFt'] = (
+                    job_info['Actual_Plant_Cost'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Plant_Profit_per_SqFt'] = (
+                    job_info['Plant_Profit'] / job_info['Total_SqFt']
+                ) if job_info['Total_SqFt'] > 0 else 0
+                job_info['Overpriced'] = False
             
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
-    
+    total_actual_profit = sum([r.get('Plant_Profit', 0) for r in variance_jobs if r.get('Invoice_Type') == 'Actual'])
+    total_estimated_profit = sum([r.get('Plant_Profit', 0) for r in variance_jobs if r.get('Invoice_Type') == 'Estimated'])
+
     return {
         'total_jobs': total_jobs,
         'analyzed_jobs': total_jobs,
         'jobs_with_variance': len(variance_jobs),
         'critical_variances': critical_variances,
         'warning_variances': warning_variances,
         'total_expected_cost': total_expected,
         'total_actual_cost': total_actual,
         'overall_variance': total_actual - total_expected if total_expected > 0 else 0,
-        'overall_variance_percent': ((total_actual - total_expected) / total_expected * 100) if total_expected > 0 else 0
+        'overall_variance_percent': ((total_actual - total_expected) / total_expected * 100) if total_expected > 0 else 0,
+        'total_actual_plant_profit': total_actual_profit,
+        'total_estimated_plant_profit': total_estimated_profit
     }
 
 def render_pricing_analysis_tab(df):
     """
     Render the comprehensive pricing analysis tab - FIXED for data type issues
     """
     st.header("🔍 Interbranch Pricing Analysis")
     st.markdown("Detailed analysis of plant invoice costs vs. expected interbranch pricing using your actual data structure.")
     
     if df.empty:
         st.warning("No data available for pricing analysis.")
         return
     
     # Show data quality information
     with st.expander("📊 Data Quality Check"):
         st.subheader("Column Data Types")
         
-        key_columns = ['Division_Type', 'Job_Material', 'Total_Job_SqFT', 'Total_Job_Price_', 'Phase_Dollars_Plant_Invoice_']
+        key_columns = ['Division_Type', 'Job_Material', 'Total_Job_SqFT', 'Total_Job_Price_', 'Phase_Dollars_Plant_Invoice_', 'Job_Plant_Invoice']
         
         for col in key_columns:
             if col in df.columns:
                 sample_values = df[col].dropna().head(3).tolist()
                 st.write(f"**{col}**: {sample_values}")
             else:
                 st.write(f"**{col}**: ❌ Missing")
     
     # Generate pricing analysis using your actual data structure
     with st.spinner("Analyzing interbranch pricing for all jobs..."):
         try:
             pricing_results = analyze_interbranch_pricing_direct(df)
             summary_stats = get_pricing_summary_stats_direct(pricing_results)
         except Exception as e:
             st.error(f"Error during analysis: {e}")
             st.info("Please check the data quality above for potential issues.")
             return
     
     # Summary metrics
     render_pricing_summary_metrics(summary_stats)
     
     # Analysis tabs
     analysis_tabs = st.tabs([
         "📊 Overview Dashboard",
         "🚨 Critical Variances", 
diff --git a/pricing_analysis_ui.py b/pricing_analysis_ui.py
index 36c9ac7b30dca9345abd88c3ca87f5ebcbcecad2..1d0fd0d79575a3fa9c413a986e194c6c76f30543 100644
--- a/pricing_analysis_ui.py
+++ b/pricing_analysis_ui.py
@@ -196,88 +286,100 @@ def render_pricing_analysis_tab(df):
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
     
-    col1, col2, col3, col4, col5 = st.columns(5)
+    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
     
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
-            "Overall Variance", 
+            "Overall Variance",
             f"{overall_variance:+.1f}%",
             delta=f"${summary_stats['overall_variance']:+,.0f}"
         )
+
+    with col6:
+        st.metric(
+            "Actual Plant Profit",
+            f"${summary_stats['total_actual_plant_profit']:,.0f}"
+        )
+
+    with col7:
+        st.metric(
+            "Estimated Plant Profit",
+            f"${summary_stats['total_estimated_plant_profit']:,.0f}"
+        )
     
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
         
diff --git a/pricing_analysis_ui.py b/pricing_analysis_ui.py
index 36c9ac7b30dca9345abd88c3ca87f5ebcbcecad2..1d0fd0d79575a3fa9c413a986e194c6c76f30543 100644
--- a/pricing_analysis_ui.py
+++ b/pricing_analysis_ui.py
@@ -304,261 +406,326 @@ def render_pricing_overview_direct(pricing_results, summary_stats):
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
-        
         if sorted_variance_jobs:
             variance_df = pd.DataFrame([
                 {
                     'Job Name': r['Job_Name'],
                     'Production #': r['Production_'],
-                    'Moraware Link': f"https://floformcountertops.moraware.net/sys/search?&search={r['Production_']}" if r['Production_'] else None,
-                    'Expected Cost': f"${r.get('Expected_Plant_Cost', 0):,.2f}",
-                    'Actual Cost': f"${r.get('Actual_Plant_Cost', 0):,.2f}",
-                    'Variance': f"${r.get('Variance_Amount', 0):+,.2f}",
-                    'Variance %': f"{r.get('Variance_Percent', 0):+.1f}%",
-                    'Severity': r.get('Severity', 'unknown').title()
+                    'Moraware Link': (
+                        f"https://floformcountertops.moraware.net/sys/search?&search={r['Production_']}"
+                        if r['Production_'] else None
+                    ),
+                    'Job Revenue': r.get('Job_Revenue', 0),
+                    'Expected Cost': r.get('Expected_Plant_Cost', 0),
+                    'Plant Invoice': r.get('Plant_Invoice_Used', 0),
+                    'Actual Invoice': r.get('Actual_Plant_Invoice', 0),
+                    'Plant Cost': r.get('Actual_Plant_Cost', 0),
+                    'Invoice Type': r.get('Invoice_Type', ''),
+                    'Plant Profit': r.get('Plant_Profit', 0),
+                    'Plant Profit %': r.get('Plant_Profit_Margin_%', 0),
+                    'Expected $/SqFt': r.get('Expected_Cost_per_SqFt', 0),
+                    'Invoice $/SqFt': r.get('Invoice_per_SqFt', 0),
+                    'Cost $/SqFt': r.get('Plant_Cost_per_SqFt', 0),
+                    'Profit $/SqFt': r.get('Plant_Profit_per_SqFt', 0),
+                    'Variance Amount': r.get('Variance_Amount', 0),
+                    'Variance %': r.get('Variance_Percent', 0),
+                    'Overpriced': '🚩' if r.get('Overpriced') else '',
+                    'Severity': r.get('Severity', 'unknown').title(),
                 }
                 for r in sorted_variance_jobs[:10]
             ])
-            
+
             st.dataframe(
                 variance_df,
                 use_container_width=True,
                 hide_index=True,
                 column_config={
                     "Moraware Link": st.column_config.LinkColumn(
                         "🔗 Moraware",
-                        display_text="Open Job"
+                        display_text="Open Job",
                     ),
+                    "Job Revenue": st.column_config.NumberColumn("Job Revenue", format="$%.2f"),
+                    "Expected Cost": st.column_config.NumberColumn("Expected Cost", format="$%.2f"),
+                    "Plant Invoice": st.column_config.NumberColumn("Plant Invoice", format="$%.2f"),
+                    "Actual Invoice": st.column_config.NumberColumn("Actual Invoice", format="$%.2f"),
+                    "Plant Cost": st.column_config.NumberColumn("Plant Cost", format="$%.2f"),
+                    "Plant Profit": st.column_config.NumberColumn("Plant Profit", format="$%.2f"),
+                    "Plant Profit %": st.column_config.NumberColumn("Plant Profit %", format="%.1f%%"),
+                    "Expected $/SqFt": st.column_config.NumberColumn("Expected $/SqFt", format="$%.2f"),
+                    "Invoice $/SqFt": st.column_config.NumberColumn("Invoice $/SqFt", format="$%.2f"),
+                    "Cost $/SqFt": st.column_config.NumberColumn("Cost $/SqFt", format="$%.2f"),
+                    "Profit $/SqFt": st.column_config.NumberColumn("Profit $/SqFt", format="$%.2f"),
+                    "Variance Amount": st.column_config.NumberColumn("Variance Amount", format="$%.2f"),
+                    "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%"),
                     "Severity": st.column_config.TextColumn(
                         "Severity",
-                        help="Variance severity level"
-                    )
-                }
+                        help="Variance severity level",
+                    ),
+                },
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
         
         if pricing_results:
             jobs_summary = pd.DataFrame([
                 {
                     'Job Name': r['Job_Name'],
                     'Production #': r['Production_'],
                     'Division': r['Division'],
                     'SqFt': r['Total_SqFt'],
-                    'Plant Cost': f"${r['Actual_Plant_Cost']:,.2f}",
+                    'Plant Invoice': f"${r['Plant_Invoice_Used']:,.2f}",
+                    'Invoice Type': r['Invoice_Type'],
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
-    
+
     for job in critical_jobs:
         variance_pct = job.get('Variance_Percent', 0)
-        
+
+        icon = "🚩 " if job.get('Overpriced') else ""
         with st.expander(
-            f"🔴 {job['Job_Name']} - Variance: {variance_pct:+.1f}%",
+            f"🔴 {icon}{job['Job_Name']} - Variance: {variance_pct:+.1f}%",
             expanded=True
         ):
-            col1, col2, col3, col4 = st.columns(4)
-            
+            col1, col2, col3, col4, col5 = st.columns(5)
+
             with col1:
                 st.metric("Expected Cost", f"${job.get('Expected_Plant_Cost', 0):,.2f}")
             with col2:
-                st.metric("Actual Cost", f"${job.get('Actual_Plant_Cost', 0):,.2f}")
+                cost_label = "Plant Invoice" if job.get('Invoice_Type') == 'Actual' else "Plant Invoice (Estimated)"
+                st.metric(cost_label, f"${job.get('Plant_Invoice_Used', 0):,.2f}")
             with col3:
-                st.metric("Variance", f"${job.get('Variance_Amount', 0):+,.2f}")
+                st.metric("Plant Profit", f"${job.get('Plant_Profit', 0):+,.2f}")
             with col4:
+                st.metric("Variance", f"${job.get('Variance_Amount', 0):+,.2f}")
+            with col5:
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
+            'Job Revenue': job.get('Job_Revenue', 0),
             'Expected Cost': job.get('Expected_Plant_Cost', 0),
-            'Actual Cost': job.get('Actual_Plant_Cost', 0),
+            'Plant Invoice': job.get('Plant_Invoice_Used', 0),
+            'Invoice Type': job.get('Invoice_Type'),
+            'Plant Profit': job.get('Plant_Profit', 0),
+            'Plant Profit %': job.get('Plant_Profit_Margin_%', 0),
+            'Expected $/SqFt': job.get('Expected_Cost_per_SqFt', 0),
+            'Invoice $/SqFt': job.get('Invoice_per_SqFt', 0),
+            'Profit $/SqFt': job.get('Plant_Profit_per_SqFt', 0),
             'Variance Amount': job.get('Variance_Amount', 0),
-            'Variance %': job.get('Variance_Percent', 0)
+            'Variance %': job.get('Variance_Percent', 0),
+            'Overpriced': '🚩' if job.get('Overpriced') else ''
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
-            "Actual Cost": st.column_config.NumberColumn("Actual Cost", format="$%.2f"),
+            "Plant Invoice": st.column_config.NumberColumn("Plant Invoice", format="$%.2f"),
+            "Plant Profit": st.column_config.NumberColumn("Plant Profit", format="$%.2f"),
+            "Plant Profit %": st.column_config.NumberColumn("Plant Profit %", format="%.1f%%"),
+            "Expected $/SqFt": st.column_config.NumberColumn("Expected $/SqFt", format="$%.2f"),
+            "Invoice $/SqFt": st.column_config.NumberColumn("Invoice $/SqFt", format="$%.2f"),
+            "Profit $/SqFt": st.column_config.NumberColumn("Profit $/SqFt", format="$%.2f"),
             "Variance Amount": st.column_config.NumberColumn("Variance Amount", format="$%.2f"),
-            "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%")
+            "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%"),
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
+                'Job Revenue': r.get('Job_Revenue', 0),
                 'Expected Cost': r.get('Expected_Plant_Cost', 0),
-                'Actual Cost': r.get('Actual_Plant_Cost', 0),
+                'Plant Invoice': r.get('Plant_Invoice_Used', 0),
+                'Actual Invoice': r.get('Actual_Plant_Invoice', 0),
+                'Plant Cost': r.get('Actual_Plant_Cost', 0),
+                'Invoice Type': r.get('Invoice_Type'),
+                'Plant Profit': r.get('Plant_Profit', 0),
+                'Plant Profit %': r.get('Plant_Profit_Margin_%', 0),
+                'Expected $/SqFt': r.get('Expected_Cost_per_SqFt', 0),
+                'Invoice $/SqFt': r.get('Invoice_per_SqFt', 0),
+                'Cost $/SqFt': r.get('Plant_Cost_per_SqFt', 0),
+                'Profit $/SqFt': r.get('Plant_Profit_per_SqFt', 0),
                 'Variance': r.get('Variance_Amount', 0),
                 'Variance %': r.get('Variance_Percent', 0),
+                'Overpriced': '🚩' if r.get('Overpriced') else '',
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
+                "SqFt": st.column_config.NumberColumn("SqFt", format="%.1f"),
+                "Job Revenue": st.column_config.NumberColumn("Job Revenue", format="$%.2f"),
                 "Expected Cost": st.column_config.NumberColumn("Expected Cost", format="$%.2f"),
-                "Actual Cost": st.column_config.NumberColumn("Actual Cost", format="$%.2f"),
+                "Plant Invoice": st.column_config.NumberColumn("Plant Invoice", format="$%.2f"),
+                "Actual Invoice": st.column_config.NumberColumn("Actual Invoice", format="$%.2f"),
+                "Plant Cost": st.column_config.NumberColumn("Plant Cost", format="$%.2f"),
+                "Plant Profit": st.column_config.NumberColumn("Plant Profit", format="$%.2f"),
+                "Plant Profit %": st.column_config.NumberColumn("Plant Profit %", format="%.1f%%"),
+                "Expected $/SqFt": st.column_config.NumberColumn("Expected $/SqFt", format="$%.2f"),
+                "Invoice $/SqFt": st.column_config.NumberColumn("Invoice $/SqFt", format="$%.2f"),
+                "Cost $/SqFt": st.column_config.NumberColumn("Cost $/SqFt", format="$%.2f"),
+                "Profit $/SqFt": st.column_config.NumberColumn("Profit $/SqFt", format="$%.2f"),
                 "Variance": st.column_config.NumberColumn("Variance", format="$%.2f"),
-                "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%")
+                "Variance %": st.column_config.NumberColumn("Variance %", format="%.1f%%"),
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
