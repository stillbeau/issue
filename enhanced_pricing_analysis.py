# enhanced_pricing_analysis.py
"""
Enhanced Pricing Analysis Module for FloForm Dashboard
Uses Stone Details sections for more accurate material identification and pricing validation
"""

import pandas as pd
import numpy as np
import re
import pricing_config as pc

def parse_stone_details(row):
    """
    Parse structured stone details from the row data.
    Returns list of materials with their details.
    """
    materials = []
    
    # Check if we have stone details columns
    product_col = row.get('Stone_Details_Product', '')
    colour_col = row.get('Stone_Details_Colour', '')
    sqft_col = row.get('Stone_Details_Sq_Ft', '')
    
    if not product_col or not colour_col:
        return materials
    
    # Handle multiple materials (split by newlines or commas)
    products = str(product_col).split('\n') if '\n' in str(product_col) else [str(product_col)]
    colours = str(colour_col).split('\n') if '\n' in str(colour_col) else [str(colour_col)]
    sqfts = str(sqft_col).split('\n') if '\n' in str(sqft_col) else [str(sqft_col)]
    
    # Ensure all lists have the same length
    max_len = max(len(products), len(colours), len(sqfts))
    
    for i in range(max_len):
        product = products[i] if i < len(products) else products[-1] if products else ''
        colour = colours[i] if i < len(colours) else colours[-1] if colours else ''
        sqft = sqfts[i] if i < len(sqfts) else sqfts[-1] if sqfts else '0'
        
        # Clean and parse square footage
        try:
            sqft_value = float(re.sub(r'[^\d.]', '', str(sqft))) if sqft else 0
        except (ValueError, TypeError):
            sqft_value = 0
        
        if product.strip() and colour.strip() and sqft_value > 0:
            materials.append({
                'product': product.strip(),
                'colour': colour.strip(),
                'sqft': sqft_value,
                'material_line': f"{product.strip()} - {colour.strip()}"
            })
    
    return materials

def identify_material_from_stone_details(material_info):
    """
    Identify material type and group from structured stone details.
    """
    product = material_info['product'].lower()
    colour = material_info['colour'].lower()
    
    # Determine material type based on product
    if any(brand in product for brand in ['hanstone', 'silestone', 'caesarstone', 'cambria', 'vicostone', 'quartz']):
        material_type = 'quartz'
    elif 'cosentino granite' in product or 'granite' in product:
        material_type = 'granite'
    elif any(brand in product for brand in ['dekton', 'infinity', 'porcelain']):
        material_type = 'porcelain'
    elif any(brand in product for brand in ['corian', 'himacs']):
        material_type = 'solid_surface'
    elif 'signature' in product.lower():
        material_type = 'signature'
    else:
        material_type = 'quartz'  # Default assumption
    
    # Extract clean material name from colour
    # Remove brand suffixes like "- HanStone", "- Silestone", etc.
    clean_colour = re.sub(r'\s*-\s*(hanstone|silestone|caesarstone|cambria|vicostone|corian|himacs|dekton).*$', '', colour, flags=re.IGNORECASE)
    clean_colour = clean_colour.strip()
    
    # Find material group
    material_group, detected_type = pc.get_material_group(clean_colour, material_type)
    
    return {
        'material_type': detected_type or material_type,
        'material_group': material_group,
        'clean_material_name': clean_colour,
        'original_product': material_info['product'],
        'original_colour': material_info['colour']
    }

def calculate_expected_interbranch_cost(material_group, material_type, sqft):
    """
    Calculate expected interbranch cost based on material group and square footage.
    """
    if material_type != 'quartz' or material_group not in pc.QUARTZ_GROUPS:
        return None
    
    group_data = pc.QUARTZ_GROUPS[material_group]
    
    cost_min = group_data['ib_cost_min'] * sqft
    cost_max = group_data['ib_cost_max'] * sqft
    cost_avg = (cost_min + cost_max) / 2
    
    return {
        'cost_per_sqft_min': group_data['ib_cost_min'],
        'cost_per_sqft_max': group_data['ib_cost_max'],
        'cost_per_sqft_avg': (group_data['ib_cost_min'] + group_data['ib_cost_max']) / 2,
        'total_cost_min': cost_min,
        'total_cost_max': cost_max,
        'total_cost_avg': cost_avg,
        'group_name': group_data['name']
    }

def analyze_job_interbranch_pricing(row):
    """
    Comprehensive pricing analysis using stone details.
    """
    # Parse stone details
    materials = parse_stone_details(row)
    
    if not materials:
        return {
            'status': 'no_stone_details',
            'message': 'No structured stone details found',
            'materials_analyzed': 0
        }
    
    # Get actual plant invoice amount
    try:
        plant_invoice_str = str(row.get('Phase_Dollars_Plant_Invoice_', '0'))
        actual_plant_cost = float(re.sub(r'[^\d.]', '', plant_invoice_str.replace(',', '')))
    except (ValueError, TypeError):
        actual_plant_cost = 0
    
    # Analyze each material
    material_analyses = []
    total_expected_cost = 0
    recognized_materials = 0
    
    for material in materials:
        material_id = identify_material_from_stone_details(material)
        
        if material_id['material_group'] is not None:
            expected_cost = calculate_expected_interbranch_cost(
                material_id['material_group'],
                material_id['material_type'],
                material['sqft']
            )
            
            if expected_cost:
                total_expected_cost += expected_cost['total_cost_avg']
                recognized_materials += 1
                
                material_analyses.append({
                    'material': material['material_line'],
                    'sqft': material['sqft'],
                    'material_type': material_id['material_type'],
                    'material_group': material_id['material_group'],
                    'clean_name': material_id['clean_material_name'],
                    'expected_cost': expected_cost,
                    'recognized': True
                })
            else:
                material_analyses.append({
                    'material': material['material_line'],
                    'sqft': material['sqft'],
                    'material_type': material_id['material_type'],
                    'material_group': material_id['material_group'],
                    'clean_name': material_id['clean_material_name'],
                    'expected_cost': None,
                    'recognized': True,
                    'note': 'No interbranch cost data available'
                })
        else:
            material_analyses.append({
                'material': material['material_line'],
                'sqft': material['sqft'],
                'recognized': False,
                'note': f'Material "{material_id["clean_material_name"]}" not found in pricing groups'
            })
    
    # Calculate variance if we have expected costs
    variance_analysis = None
    if total_expected_cost > 0 and actual_plant_cost > 0:
        variance_amount = actual_plant_cost - total_expected_cost
        variance_percent = (variance_amount / total_expected_cost) * 100
        
        # Determine severity
        if abs(variance_percent) > 20:
            severity = 'critical'
        elif abs(variance_percent) > 10:
            severity = 'warning'
        else:
            severity = 'normal'
        
        variance_analysis = {
            'actual_cost': actual_plant_cost,
            'expected_cost': total_expected_cost,
            'variance_amount': variance_amount,
            'variance_percent': variance_percent,
            'severity': severity,
            'status': 'over_budget' if variance_amount > 0 else 'under_budget'
        }
    
    return {
        'status': 'analyzed',
        'materials_analyzed': len(materials),
        'recognized_materials': recognized_materials,
        'material_details': material_analyses,
        'variance_analysis': variance_analysis,
        'total_expected_cost': total_expected_cost,
        'actual_plant_cost': actual_plant_cost
    }

def generate_pricing_report(df):
    """
    Generate comprehensive pricing report for all jobs.
    """
    pricing_results = []
    
    for idx, row in df.iterrows():
        analysis = analyze_job_interbranch_pricing(row)
        
        pricing_results.append({
            'Job_Name': row.get('Job_Name', ''),
            'Production_': row.get('Production_', ''),
            'Division': row.get('Division', ''),
            'Total_SqFt': row.get('Total_Job_SqFT', 0),
            'Analysis': analysis
        })
    
    return pricing_results

def get_pricing_summary_stats(pricing_results):
    """
    Generate summary statistics for pricing analysis.
    """
    total_jobs = len(pricing_results)
    analyzed_jobs = len([r for r in pricing_results if r['Analysis']['status'] == 'analyzed'])
    
    variance_jobs = [r for r in pricing_results 
                    if r['Analysis'].get('variance_analysis') is not None]
    
    critical_variances = len([r for r in variance_jobs 
                             if r['Analysis']['variance_analysis']['severity'] == 'critical'])
    
    warning_variances = len([r for r in variance_jobs 
                            if r['Analysis']['variance_analysis']['severity'] == 'warning'])
    
    total_expected = sum([r['Analysis'].get('total_expected_cost', 0) for r in variance_jobs])
    total_actual = sum([r['Analysis'].get('actual_plant_cost', 0) for r in variance_jobs])
    
    return {
        'total_jobs': total_jobs,
        'analyzed_jobs': analyzed_jobs,
        'jobs_with_variance': len(variance_jobs),
        'critical_variances': critical_variances,
        'warning_variances': warning_variances,
        'total_expected_cost': total_expected,
        'total_actual_cost': total_actual,
        'overall_variance': total_actual - total_expected if total_expected > 0 else 0,
        'overall_variance_percent': ((total_actual - total_expected) / total_expected * 100) if total_expected > 0 else 0
    }
