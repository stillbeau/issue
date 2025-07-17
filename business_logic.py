"""
Business Logic Module for FloForm Dashboard
Contains all business intelligence calculations, risk assessments, and analytics
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import pricing_config as pc

# --- Timeline & Risk Thresholds ---
TIMELINE_THRESHOLDS = {
    'template_to_rtf': 3,
    'rtf_to_product_rcvd': 7,
    'product_rcvd_to_install': 5,
    'template_to_install': 15,
    'template_to_ship': 10,
    'ship_to_install': 5,
    'days_in_stage_warning': 5,
    'stale_job_threshold': 7,
    'avg_stage_durations': {
        'Post-Template': 3,
        'In Fabrication': 7,
        'Product Received': 5,
        'Shipped': 5,
        'default': 5
    }
}

def parse_material(s: str) -> tuple[str, str]:
    """Enhanced material parsing with better brand/color extraction."""
    if pd.isna(s) or not s:
        return "N/A", "N/A"
    
    s = str(s)
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else "N/A"
    color = color_match.group(1).strip() if color_match else "N/A"
    return brand, color

def get_current_stage(row):
    """Determine the current operational stage of a job based on dates."""
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
    date_map = {
        'Shipped': 'Ship_Date', 
        'Product Received': 'Product_Rcvd_Date', 
        'In Fabrication': 'Ready_to_Fab_Date', 
        'Post-Template': 'Template_Date'
    }
    
    if stage in date_map and pd.notna(row.get(date_map[stage])):
        return (today - row[date_map[stage]]).days
    return 0 if stage == 'Completed' else np.nan

def calculate_risk_score(row):
    """Enhanced risk score calculation with multiple factors."""
    score = 0
    
    # Days behind schedule (0-20 points)
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0: 
        score += min(row['Days_Behind'] * 2, 20)
    
    # Time stuck in current stage (0-10 points)
    if pd.notna(row.get('Days_In_Current_Stage')) and row['Days_In_Current_Stage'] > TIMELINE_THRESHOLDS['days_in_stage_warning']: 
        score += 10
    
    # Template to RTF bottleneck (0-15 points)
    if pd.isna(row.get('Ready_to_Fab_Date')) and pd.notna(row.get('Template_Date')):
        days_since_template = (pd.Timestamp.now() - row['Template_Date']).days
        if days_since_template > TIMELINE_THRESHOLDS['template_to_rtf']: 
            score += 15
    
    # Rework flag (10 points)
    if row.get('Has_Rework', False): 
        score += 10
    
    # Missing next activity (5 points)
    if pd.isna(row.get('Next_Sched_Activity')): 
        score += 5
    
    # High-value job risk multiplier
    if pd.notna(row.get('Revenue')) and row['Revenue'] > 10000:
        score *= 1.2
    
    # Material recognition issues (5 points)
    if pd.isna(row.get('Material_Group')):
        score += 5
    
    return min(score, 100)  # Cap at 100

def calculate_delay_probability(row):
    """Enhanced delay probability calculation with detailed factors."""
    risk_score, factors = 0, []
    
    # Already behind schedule (40 points)
    if pd.notna(row.get('Days_Behind')) and row['Days_Behind'] > 0:
        risk_score += 40
        factors.append(f"Already {row['Days_Behind']:.0f} days behind")
    
    # Stuck in current stage (20 points)
    if pd.notna(row.get('Days_In_Current_Stage')):
        avg_durations = TIMELINE_THRESHOLDS['avg_stage_durations']
        expected_duration = avg_durations.get(row.get('Current_Stage', ''), avg_durations['default'])
        if row['Days_In_Current_Stage'] > expected_duration:
            risk_score += 20
            factors.append(f"Stuck in {row['Current_Stage']} for {row['Days_In_Current_Stage']:.0f} days")
    
    # Has rework (15 points)
    if row.get('Has_Rework', False): 
        risk_score += 15
        factors.append("Has rework")
    
    # No next activity scheduled (15 points)
    if pd.isna(row.get('Next_Sched_Activity')): 
        risk_score += 15
        factors.append("No next activity scheduled")
    
    # Material availability issues (10 points)
    if pd.isna(row.get('Material_Group')):
        risk_score += 10
        factors.append("Material pricing issues")
    
    # High-value job additional risk (5 points)
    if pd.notna(row.get('Revenue')) and row['Revenue'] > 15000:
        risk_score += 5
        factors.append("High-value job")
    
    return min(risk_score, 100), "; ".join(factors)

def extract_material_from_description(material_description: str):
    """Enhanced material extraction with comprehensive pattern recognition."""
    if pd.isna(material_description):
        return None, "no_description"

    material_desc = str(material_description).lower()
    
    # Skip laminate materials early
    laminate_indicators = ['wilsonart', 'formica', 'arborite']
    if any(indicator in material_desc for indicator in laminate_indicators):
        return None, "laminate_skipped"

    # Enhanced patterns for different material types
    patterns = [
        # Brand-specific patterns
        r'hanstone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)',
        r'rona\s+quartz[^-]*-\s*([^(/]+)',
        r'vicostone\s*\([^)]*\)\s*([^b]+?)\s*bq\d+',
        r'wilsonart\s+quartz\s*\([^)]*\)\s*([^mq]+?)(?:\s*matte)?\s*q\d+',
        r'silestone\s*\([^)]*\)\s*([^(]+?)\s*\([^)]*\)',
        r'cambria\s*\([^)]*\)\s*([^2-3]+?)\s*[23]cm',
        r'caesarstone\s*\([^)]+\)\s*([^#]+?)\s*#\d+',
        r'dekton\s*\([^)]+\)\s*([^m]+?)\s*matte',
        r'natural\s+stone\s*\([^)]+\)\s*([^2-3]+?)\s*[23]cm',
        r'granite\s*\([^)]*\)\s*([^2-3]+?)\s*[23]cm',
        r'corian\s*\([^)]*\)\s*([^(]+?)\s*\(',
        r'himacs\s*\([^)]*\)\s*([^(]+?)\s*\(',
        # Generic patterns
        r'\([^)]*\)\s*([^(]{3,}?)\s*\(',
        r'-\s*([^-]{3,}?)\s*-',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, material_desc, re.IGNORECASE)
        if matches:
            material = matches[0].strip()
            # Clean up the extracted material name
            cleaned = re.sub(r'\s*(ex|ss|eternal|leathered|polished|matte|honed|suede)\s*', ' ', material, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if len(cleaned) > 2:
                return cleaned, "extracted"
    
    return None, "unrecognized_pattern"

def analyze_job_pricing(row):
    """Enhanced pricing validation using the comprehensive pricing structure."""
    material_desc = row.get('Job_Material', '')
    extracted_material, status = extract_material_from_description(material_desc)

    if status == "laminate_skipped":
        return {'status': 'laminate_skipped', 'message': 'Laminate material'}
    if status == "unrecognized_pattern":
        return {'status': 'unrecognized_pattern', 'message': f'Could not identify material from: "{material_desc[:50]}..."'}
    if not extracted_material:
        return {'status': 'error', 'message': 'Material name could not be extracted.'}

    # Identify material type from description
    material_type = pc.identify_material_type(material_desc)
    
    # Get material group
    material_group, detected_type = pc.get_material_group(extracted_material, material_type)
    
    if material_group is None:
        if extracted_material.lower() in pc.UNASSIGNED_MATERIALS:
            return {'status': 'unassigned_material', 'message': f'Material "{extracted_material}" needs group assignment.'}
        else:
            return {'status': 'unknown_material', 'message': f'Extracted material "{extracted_material}" not in any group.'}

    try:
        sqft = float(str(row.get('Total_Job_SqFT', 0)).replace(',', ''))
        revenue = float(str(row.get('Total_Job_Price_', 0)).replace('$', '').replace(',', ''))
        plant_cost = float(str(row.get('Phase_Dollars_Plant_Invoice_', 0)).replace('$', '').replace(',', '')) if detected_type == 'quartz' else None
        customer_type = row.get('Job_Type', 'Retail')
    except (ValueError, TypeError):
        return {'status': 'error', 'message': 'Invalid numeric data for SqFt, Revenue, or Cost.'}

    if sqft <= 0:
        return {'status': 'error', 'message': 'Job has zero or invalid SqFt.'}

    validation_results = pc.validate_job_pricing(
        material_group=material_group,
        material_type=detected_type,
        sqft=sqft,
        customer_type=customer_type,
        actual_revenue=revenue,
        actual_plant_cost=plant_cost
    )
    
    validation_results['extracted_material'] = extracted_material
    validation_results['material_type'] = detected_type
    return validation_results

def calculate_business_health_score(df):
    """Calculate comprehensive business health score based on multiple factors."""
    
    if df.empty:
        return 50, {}  # Default neutral score
    
    health_components = {}
    
    # Revenue Health (30% weight)
    if 'Revenue' in df.columns and 'Job_Creation' in df.columns:
        df_with_dates = df.dropna(subset=['Job_Creation'])
        if len(df_with_dates) >= 2:
            monthly_revenue = df_with_dates.set_index(
                pd.to_datetime(df_with_dates['Job_Creation'], errors='coerce')
            ).resample('M')['Revenue'].sum()
            
            if len(monthly_revenue) >= 2:
                revenue_trend = (monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100
                revenue_health = min(100, max(0, 50 + revenue_trend * 2))
            else:
                revenue_health = 75
        else:
            revenue_health = 75
        health_components['Revenue Growth'] = (revenue_health, 30)
    
    # Operational Health (25% weight)
    if 'Risk_Score' in df.columns:
        avg_risk = df.get('Risk_Score', pd.Series([0])).mean()
        operational_health = max(0, 100 - avg_risk)
        health_components['Operational Efficiency'] = (operational_health, 25)
    
    # Quality Health (25% weight)
    if 'Has_Rework' in df.columns:
        rework_rate = df.get('Has_Rework', pd.Series([False])).sum() / len(df) * 100
        quality_health = max(0, 100 - rework_rate * 5)
        health_components['Quality'] = (quality_health, 25)
    
    # Profitability Health (20% weight)
    if 'Branch_Profit_Margin_%' in df.columns:
        avg_margin = df.get('Branch_Profit_Margin_%', pd.Series([0])).mean()
        profitability_health = min(100, max(0, avg_margin * 2))
        health_components['Profitability'] = (profitability_health, 20)
    
    # Calculate overall health score
    if health_components:
        total_weight = sum(weight for _, weight in health_components.values())
        overall_health = sum(score * weight for score, weight in health_components.values()) / total_weight
    else:
        overall_health = 50  # Default if no components available
    
    return overall_health, health_components

def get_critical_issues(df, today):
    """Identify and categorize critical issues requiring immediate attention."""
    
    active_jobs = df[df['Job_Status'] != 'Complete']
    
    if active_jobs.empty:
        return {}
    
    issues = {}
    
    # **NEW: Completed work not invoiced/closed**
    completed_not_closed = active_jobs[
        ((active_jobs['Install_Date'].notna()) & (active_jobs['Install_Date'] < today)) |
        ((active_jobs['Pick_Up_Date'].notna()) & (active_jobs['Pick_Up_Date'] < today)) |
        ((active_jobs['Delivery_Date'].notna()) & (active_jobs['Delivery_Date'] < today))
    ]
    
    if not completed_not_closed.empty:
        issues['completed_not_invoiced'] = {
            'count': len(completed_not_closed),
            'severity': 'critical',
            'description': 'Work completed but job not closed (likely invoicing issue)',
            'data': completed_not_closed,
            'priority': 1  # Highest priority for cash flow
        }
    
    # Missing next activity
    missing_activity = active_jobs[
        (active_jobs['Next_Sched_Activity'].isna()) & 
        (active_jobs['Current_Stage'].isin(['Post-Template', 'In Fabrication', 'Product Received']))
    ]
    if not missing_activity.empty:
        issues['missing_activity'] = {
            'count': len(missing_activity),
            'severity': 'critical',
            'description': 'Jobs missing next scheduled activity',
            'data': missing_activity,
            'priority': 2
        }
    
    # Stale jobs
    stale_jobs = active_jobs[active_jobs['Days_Since_Last_Activity'] > TIMELINE_THRESHOLDS['stale_job_threshold']]
    if not stale_jobs.empty:
        issues['stale_jobs'] = {
            'count': len(stale_jobs),
            'severity': 'warning',
            'description': f'Jobs with no activity for >{TIMELINE_THRESHOLDS["stale_job_threshold"]} days',
            'data': stale_jobs,
            'priority': 4
        }
    
    # Template to RTF bottleneck
    template_stuck = active_jobs[
        (active_jobs['Template_Date'].notna()) & 
        (active_jobs['Ready_to_Fab_Date'].isna()) & 
        ((today - active_jobs['Template_Date']).dt.days > TIMELINE_THRESHOLDS['template_to_rtf'])
    ]
    if not template_stuck.empty:
        issues['template_bottleneck'] = {
            'count': len(template_stuck),
            'severity': 'warning',
            'description': 'Template to Ready-to-Fab bottleneck',
            'data': template_stuck,
            'priority': 5
        }
    
    # Upcoming installs missing product
    upcoming_installs = active_jobs[
        (active_jobs['Install_Date'].notna()) & 
        (active_jobs['Install_Date'] <= today + timedelta(days=7)) & 
        (active_jobs['Product_Rcvd_Date'].isna())
    ]
    if not upcoming_installs.empty:
        issues['missing_product'] = {
            'count': len(upcoming_installs),
            'severity': 'critical',
            'description': 'Upcoming installs missing product',
            'data': upcoming_installs,
            'priority': 3
        }
    
    # High-risk jobs
    high_risk = active_jobs[active_jobs.get('Risk_Score', 0) >= 30]
    if not high_risk.empty:
        issues['high_risk'] = {
            'count': len(high_risk),
            'severity': 'warning',
            'description': 'High-risk jobs (Risk Score ≥ 30)',
            'data': high_risk,
            'priority': 6
        }
    
    # Sort issues by priority (lower number = higher priority)
    return dict(sorted(issues.items(), key=lambda x: x[1].get('priority', 999)))


def calculate_revenue_at_risk(df, today):
    """Calculate revenue at risk from completed but not invoiced jobs."""
    
    if df.empty:
        return {'total_revenue_at_risk': 0, 'jobs_at_risk': 0, 'avg_days_overdue': 0}
    
    # Identify completed but not closed jobs
    completed_not_closed = df[
        (df['Job_Status'] != 'Complete') &
        (
            ((df['Install_Date'].notna()) & (df['Install_Date'] < today)) |
            ((df['Pick_Up_Date'].notna()) & (df['Pick_Up_Date'] < today)) |
            ((df['Delivery_Date'].notna()) & (df['Delivery_Date'] < today))
        )
    ]
    
    if completed_not_closed.empty:
        return {'total_revenue_at_risk': 0, 'jobs_at_risk': 0, 'avg_days_overdue': 0}
    
    # Calculate metrics
    total_revenue = completed_not_closed.get('Revenue', pd.Series([0])).sum()
    job_count = len(completed_not_closed)
    
    # Calculate average days overdue
    completed_not_closed = completed_not_closed.copy()
    
    # Find the latest completion date for each job
    completion_dates = []
    for _, row in completed_not_closed.iterrows():
        dates = [
            row.get('Install_Date'),
            row.get('Pick_Up_Date'), 
            row.get('Delivery_Date')
        ]
        valid_dates = [d for d in dates if pd.notna(d) and d < today]
        if valid_dates:
            completion_dates.append(max(valid_dates))
        else:
            completion_dates.append(today)  # Fallback
    
    if completion_dates:
        days_overdue = [(today - date).days for date in completion_dates]
        avg_days_overdue = sum(days_overdue) / len(days_overdue)
    else:
        avg_days_overdue = 0
    
    return {
        'total_revenue_at_risk': total_revenue,
        'jobs_at_risk': job_count,
        'avg_days_overdue': avg_days_overdue,
        'jobs_data': completed_not_closed
    }

def calculate_performance_metrics(df, role_type, today):
    """Calculate performance metrics for different roles (Salesperson, Template, Install)."""
    
    role_col_map = {
        "Salesperson": "Salesperson", 
        "Template Assigned To": "Template_Assigned_To", 
        "Install Assigned To": "Install_Assigned_To"
    }
    
    role_col = role_col_map.get(role_type)
    
    if not role_col or role_col not in df.columns:
        return []
    
    employees = [e for e in df[role_col].dropna().unique() if str(e).strip()]
    scorecards = []
    
    week_ago = today - pd.Timedelta(days=7)
    month_ago = today - pd.Timedelta(days=30)

    for employee in employees:
        emp_jobs = df[df[role_col] == employee]
        
        if role_type == "Salesperson":
            active_jobs = emp_jobs[~emp_jobs['Current_Stage'].isin(['Completed'])]
            
            metrics = {
                'Employee': employee,
                'Total Jobs': len(emp_jobs),
                'Active Jobs': len(active_jobs),
                'Total Revenue': emp_jobs.get('Revenue', pd.Series([0])).sum(),
                'Avg Revenue/Job': emp_jobs.get('Revenue', pd.Series([0])).mean(),
                'Avg Days Behind': emp_jobs['Days_Behind'].mean(),
                'Jobs w/ Rework': len(emp_jobs[emp_jobs.get('Has_Rework', False) == True]),
                'High Risk Jobs': len(emp_jobs[emp_jobs.get('Risk_Score', 0) >= 30]),
                'Avg Profit Margin': emp_jobs.get('Branch_Profit_Margin_%', pd.Series([0])).mean(),
                'Jobs This Month': len(emp_jobs[emp_jobs['Job_Creation'] >= month_ago])
            }
            
        elif role_type == "Template Assigned To":
            template_jobs = emp_jobs[emp_jobs['Template_Date'].notna()]
            
            metrics = {
                'Employee': employee,
                'Total Templates': len(template_jobs),
                'Templates This Week': len(template_jobs[
                    (template_jobs['Template_Date'] >= week_ago) & 
                    (template_jobs['Template_Date'] <= today)
                ]),
                'Upcoming Templates': len(template_jobs[template_jobs['Template_Date'] > today]),
                'Avg Template to RTF': template_jobs['Days_Template_to_RTF'].mean(),
                'Overdue RTF': len(template_jobs[
                    (template_jobs['Template_Date'] < today) & 
                    (template_jobs['Ready_to_Fab_Date'].isna())
                ]),
                'Total SqFt': template_jobs.get('Total_Job_SqFt', pd.Series([0])).sum()
            }
            
        else:  # Install Assigned To
            install_jobs = emp_jobs[emp_jobs['Install_Date'].notna()]
            
            metrics = {
                'Employee': employee,
                'Total Installs': len(install_jobs),
                'Installs This Week': len(install_jobs[
                    (install_jobs['Install_Date'] >= week_ago) & 
                    (install_jobs['Install_Date'] <= today)
                ]),
                'Upcoming Installs': len(install_jobs[install_jobs['Install_Date'] > today]),
                'Avg Ship to Install': install_jobs['Days_Ship_to_Install'].mean(),
                'Total SqFt': install_jobs.get('Total_Job_SqFt', pd.Series([0])).sum(),
                'Jobs w/ Rework': len(install_jobs[install_jobs.get('Has_Rework', False) == True])
            }
        
        scorecards.append(metrics)
    
    return scorecards

def generate_business_insights(df, today):
    """Generate actionable business insights based on data analysis."""
    
    insights = []
    
    if df.empty:
        return ["No data available for analysis"]
    
    # Revenue insights
    if 'Revenue' in df.columns:
        high_value_jobs = df[df['Revenue'] > 15000]
        if len(high_value_jobs) > 0:
            high_value_at_risk = len(high_value_jobs[high_value_jobs.get('Risk_Score', 0) > 50])
            if high_value_at_risk > 0:
                insights.append(f"💰 **High-Value Alert:** {high_value_at_risk} high-value jobs (>$15K) are at risk")
    
    # Operational insights
    if 'Current_Stage' in df.columns:
        active_jobs = df[df['Job_Status'] != 'Complete']
        if not active_jobs.empty:
            bottleneck_stage = active_jobs['Current_Stage'].value_counts().idxmax()
            bottleneck_count = active_jobs['Current_Stage'].value_counts().iloc[0]
            if bottleneck_count > 10:
                insights.append(f"🚧 **Bottleneck Alert:** {bottleneck_count} jobs stuck in {bottleneck_stage}")
    
    # Quality insights
    if 'Has_Rework' in df.columns:
        rework_rate = df['Has_Rework'].sum() / len(df) * 100
        if rework_rate > 10:
            insights.append(f"🔧 **Quality Concern:** {rework_rate:.1f}% rework rate exceeds 10% threshold")
        elif rework_rate < 5:
            insights.append(f"✅ **Quality Excellence:** {rework_rate:.1f}% rework rate - excellent quality control")
    
    # Capacity insights
    if 'Install_Date' in df.columns:
        upcoming_week = df[
            (df['Install_Date'] >= today) &
            (df['Install_Date'] <= today + timedelta(days=7))
        ]
        if len(upcoming_week) > 20:
            insights.append(f"📅 **Capacity Warning:** {len(upcoming_week)} installs scheduled next week")
    
    # Material insights
    if 'Material_Group' in df.columns:
        unrecognized = df[df['Material_Group'].isna()]
        if len(unrecognized) > 0:
            insights.append(f"🔍 **Pricing Alert:** {len(unrecognized)} jobs have unrecognized materials")
    
    if not insights:
        insights.append("✅ **Operations Normal:** No critical issues detected in current analysis")
    
    return insights

def calculate_timeline_metrics(df):
    """Calculate comprehensive timeline performance metrics."""
    
    timeline_metrics = {}
    
    # Define timeline columns and their descriptions
    timeline_cols = {
        'Days_Template_to_Install': 'Template to Install',
        'Days_Template_to_RTF': 'Template to Ready-to-Fab', 
        'Days_RTF_to_Product_Rcvd': 'Ready-to-Fab to Product Received',
        'Days_Product_Rcvd_to_Install': 'Product Received to Install',
        'Days_Template_to_Ship': 'Template to Ship',
        'Days_Ship_to_Install': 'Ship to Install'
    }
    
    for col, description in timeline_cols.items():
        if col in df.columns:
            data = df[col].dropna()
            if not data.empty:
                timeline_metrics[description] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'count': len(data),
                    'percentile_90': data.quantile(0.9)
                }
    
    return timeline_metrics
