"""
FloForm Express Pricing Configuration
June 2025 Pricing Structure

This module contains all pricing data and logic for the FloForm Express quartz program.
Includes retail pricing, interbranch billing costs, and customer discount structures.
"""

import pandas as pd
import numpy as np

# --- MATERIAL GROUPS WITH JUNE 2025 PRICING ---
MATERIAL_GROUPS = {
    0: {
        'name': 'Group 0',
        'retail_price': 92.00,  # Includes installation
        'ib_cost_min': 34.80,   # Interbranch billing - what shop charges branch (min)
        'ib_cost_max': 34.99,   # Interbranch billing - what shop charges branch (max)
        'materials': ['black coral', 'rocky shores', 'tofino grey', 'tofino']
    },
    1: {
        'name': 'Group 1', 
        'retail_price': 102.00,
        'ib_cost_min': 39.01,
        'ib_cost_max': 39.01,
        'materials': ['aspen', 'blackburn', 'leaden', 'uptown grey']
    },
    2: {
        'name': 'Group 2',
        'retail_price': 122.00,
        'ib_cost_min': 48.77,
        'ib_cost_max': 48.77,
        'materials': ['miami vena', 'whistler', 'whistler gold', 'miami white', 'silhouette', 
                      'artisan grey', 'drift', 'specchio white', 'lazio', 'urban cloud']
    },
    3: {
        'name': 'Group 3',
        'retail_price': 132.00,
        'ib_cost_min': 54.17,
        'ib_cost_max': 54.17,
        'materials': ['carrara codena', 'desert silver', 'calacatta west', 'organic white', 'aterra blanca']
    },
    4: {
        'name': 'Group 4',
        'retail_price': 144.00,
        'ib_cost_min': 58.69,
        'ib_cost_max': 60.47,
        'materials': ['aterra verity', 'brava marfil', 'charcoal soapstone', 'antello', 'celestial sky',
                      'embrace', 'empress', 'fresh concrete', 'frosty carrina', 'oceana', 'raw concrete',
                      'stellar snow', 'tranquility', 'bianco drift']
    },
    5: {
        'name': 'Group 5',
        'retail_price': 152.00,
        'ib_cost_min': 61.67,
        'ib_cost_max': 61.67,
        'materials': ['clouds rest', 'desert wind', 'haida', 'glencoe', 'marathi marble', 'moorland fog',
                      'nova serrana', 'river glen', 'rugged concrete', 'santiago', 'serene', 'verde peak',
                      'vicentia', 'eden', 'aurelia', 'montauk', 'chantilly']
    },
    6: {
        'name': 'Group 6',
        'retail_price': 162.00,
        'ib_cost_min': 70.68,
        'ib_cost_max': 70.68,
        'materials': ['calacatta olympos', 'fossa falls', 'calacatta volegno', 'calacatta pastino',
                      'coastal', 'enchanted rock', 'north cascades', 'raw a', 'raw g', 'et statuario',
                      'calacatta extra', 'calacatta mont']
    },
    7: {
        'name': 'Group 7',
        'retail_price': 178.00,
        'ib_cost_min': 73.38,
        'ib_cost_max': 83.29,
        'materials': ['tyrol', 'verdelia', 'et calacatta gold', 'ethereal glow', 'ethereal dusk',
                      'ethereal noctis', 'elba white', 'le blanc', 'matterhorn', 'eternal calacatta gold']
    },
    8: {
        'name': 'Group 8',
        'retail_price': 202.00,
        'ib_cost_min': 98.75,
        'ib_cost_max': 98.75,
        'materials': ['amarcord', 'berwyn', 'colton', 'calacatta nuvo', 'solenna', 'versailles ivory',
                      'romantic ash', 'riviere rose']
    },
    9: {
        'name': 'Group 9',
        'retail_price': 232.00,
        'ib_cost_min': 115.46,
        'ib_cost_max': 115.46,
        'materials': ['brittanicca', 'brittanicca gold warm', 'skara brae', 'inverness frost',
                      'everleigh', 'portrush', 'ironsbridge']
    }
}

# --- CUSTOMER DISCOUNT STRUCTURE ---
CUSTOMER_DISCOUNTS = {
    'Retail': 0.00,         # Full retail price
    'Dealer': 0.15,         # 15% discount
    'Contractor': 0.25,     # 25% discount
    'Home Builder': 0.25,   # 25% discount
    'Commercial': 0.25,     # 25% discount
    'LIA': 0.30,            # 30% discount
    'Home Depot': 'special',# Requires manual review
    'Costco': 'special'     # Requires manual review
}

# --- VALIDATION THRESHOLDS ---
VALIDATION_THRESHOLDS = {
    'revenue_warning': 50.00,
    'revenue_critical': 500.00,
    'cost_critical': 500.00
}

# --- MATERIALS PENDING ASSIGNMENT ---
UNASSIGNED_MATERIALS = {
    'royal blanc', 'mount royal', 'bianco delicato', 'upper canada', 'bianco modesto', 
    'noir terrain', 'calacatta nero', 'lithic luxe', 'calacatta marmo', 'markina leathered',
    'eternal bella', 'weybourne', 'crystal ice', 'alpine mist', 'super white', 'domoos',
    'tetons oro', 'brezza oro'
}

# --- PRICING FUNCTIONS ---

def get_material_group(material_name):
    """
    Find the material group for a given material name.
    """
    if not material_name:
        return None
    
    material_lower = material_name.lower().strip()
    
    for group_num, group_data in MATERIAL_GROUPS.items():
        for known_material in group_data['materials']:
            if known_material.lower() in material_lower or material_lower in known_material.lower():
                return group_num
    
    return None

def get_retail_price(material_group, sqft, customer_type):
    """
    Calculate retail price for a material group, square footage, and customer type.
    """
    if material_group not in MATERIAL_GROUPS or customer_type not in CUSTOMER_DISCOUNTS:
        return None
    
    group_data = MATERIAL_GROUPS[material_group]
    base_price = group_data['retail_price']
    discount_rate = CUSTOMER_DISCOUNTS[customer_type]
    
    if discount_rate == 'special':
        return {
            'status': 'special_pricing',
            'base_price_per_sqft': base_price,
            'customer_type': customer_type,
            'message': f'{customer_type} requires manual pricing review'
        }
    
    total_base_price = base_price * sqft
    discount_amount = total_base_price * discount_rate
    final_price = total_base_price - discount_amount
    
    return {
        'status': 'calculated', 'material_group': material_group, 'group_name': group_data['name'],
        'customer_type': customer_type, 'sqft': sqft, 'base_price_per_sqft': base_price,
        'discount_rate': discount_rate, 'discount_percent': discount_rate * 100,
        'total_base_price': total_base_price, 'discount_amount': discount_amount,
        'final_price': final_price, 'price_per_sqft': final_price / sqft if sqft > 0 else 0
    }

def get_expected_plant_cost(material_group, sqft):
    """
    Calculate expected interbranch billing cost (what the shop charges the branch).
    """
    if material_group not in MATERIAL_GROUPS:
        return None
    
    group_data = MATERIAL_GROUPS[material_group]
    
    cost_min = group_data['ib_cost_min'] * sqft
    cost_max = group_data['ib_cost_max'] * sqft
    cost_avg = (cost_min + cost_max) / 2
    
    return {
        'material_group': material_group, 'group_name': group_data['name'], 'sqft': sqft,
        'cost_per_sqft_min': group_data['ib_cost_min'], 'cost_per_sqft_max': group_data['ib_cost_max'],
        'cost_per_sqft_avg': (group_data['ib_cost_min'] + group_data['ib_cost_max']) / 2,
        'total_cost_min': cost_min, 'total_cost_max': cost_max, 'total_cost_avg': cost_avg
    }

def validate_job_pricing(material_group, sqft, customer_type, actual_revenue, actual_plant_cost):
    """
    Validate job pricing against expected pricing structure.
    """
    expected_retail = get_retail_price(material_group, sqft, customer_type)
    expected_plant = get_expected_plant_cost(material_group, sqft)
    
    if not expected_retail or not expected_plant:
        return {'status': 'error', 'message': 'Could not calculate expected pricing'}
    
    if expected_retail['status'] == 'special_pricing':
        return expected_retail
    
    revenue_variance = actual_revenue - expected_retail['final_price']
    revenue_variance_pct = (revenue_variance / expected_retail['final_price'] * 100) if expected_retail['final_price'] > 0 else 0
    
    plant_cost_variance = actual_plant_cost - expected_plant['total_cost_avg']
    plant_cost_variance_pct = (plant_cost_variance / expected_plant['total_cost_avg'] * 100) if expected_plant['total_cost_avg'] > 0 else 0
    
    issues = []
    
    if abs(revenue_variance) > VALIDATION_THRESHOLDS['revenue_warning']:
        severity = 'critical' if abs(revenue_variance) > VALIDATION_THRESHOLDS['revenue_critical'] else 'warning'
        issues.append({
            'type': 'revenue_variance', 'severity': severity,
            'message': f'Revenue ${actual_revenue:,.2f} vs expected ${expected_retail["final_price"]:,.2f} ({revenue_variance_pct:+.1f}%)',
            'variance_amount': revenue_variance
        })
    
    if actual_plant_cost < expected_plant['total_cost_min'] or actual_plant_cost > expected_plant['total_cost_max']:
        severity = 'critical' if abs(plant_cost_variance) > VALIDATION_THRESHOLDS['cost_critical'] else 'warning'
        issues.append({
            'type': 'plant_cost_variance', 'severity': severity,
            'message': f'Plant cost ${actual_plant_cost:,.2f} outside expected range ${expected_plant["total_cost_min"]:,.2f}-${expected_plant["total_cost_max"]:,.2f}',
            'variance_amount': plant_cost_variance
        })
    
    return {
        'status': 'analyzed', 'expected_retail': expected_retail, 'expected_plant': expected_plant,
        'actual_revenue': actual_revenue, 'actual_plant_cost': actual_plant_cost,
        'revenue_variance': revenue_variance, 'revenue_variance_pct': revenue_variance_pct,
        'plant_cost_variance': plant_cost_variance, 'plant_cost_variance_pct': plant_cost_variance_pct,
        'issues': issues, 'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
        'warnings': len([i for i in issues if i['severity'] == 'warning'])
    }
