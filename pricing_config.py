"""
FloForm Express Pricing Configuration
Updated July 2025 - Complete Material Pricing Structure

This module contains all pricing data and logic for all FloForm Express programs.
Includes retail pricing for quartz, granite, porcelain, laminate, and solid surface.
"""

import pandas as pd
import numpy as np

# --- QUARTZ PRICING (Updated June 2025) ---
QUARTZ_GROUPS = {
    0: {
        'name': 'Group 0',
        'retail_price': 92.00,
        'ib_cost_min': 34.80,
        'ib_cost_max': 34.99,
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

# --- GRANITE PRICING (Updated Dec 2024) ---
GRANITE_GROUPS = {
    1: {
        'name': 'Group 1',
        'retail_price': 99.00,
        'materials': ['majestic white', 'steel grey', 'via lattea']
    },
    2: {
        'name': 'Group 2',
        'retail_price': 119.00,
        'materials': ['star white', 'white lava', 'black pearl']
    },
    3: {
        'name': 'Group 3',
        'retail_price': 129.00,
        'materials': ['absolute black', 'white alamo', 'delicatus ice', 'naori']
    },
    4: {
        'name': 'Group 4',
        'retail_price': 169.00,
        'materials': ['dakar', 'white macaubas', 'nilo quartzite']
    }
}

# --- SIGNATURE SERIES PRICING (Updated June 2025) ---
SIGNATURE_GROUPS = {
    1: {
        'name': 'Level 1',
        'retail_price': 79.00,
        'materials': ['upper canada', 'tofino', 'tofino grey']
    },
    2: {
        'name': 'Level 2',
        'retail_price': 99.00,
        'materials': ['whistler', 'whistler gold', 'linen cream']
    },
    3: {
        'name': 'Level 3',
        'retail_price': 109.00,
        'materials': ['tempest gris', 'atacama oro', 'nimbus blanc', 'tetons oro', 'lithic lux',
                      'noir terrain', 'ashen marble', 'siberian frost']
    }
}

# --- PORCELAIN PRICING (Updated Jan 2025) ---
PORCELAIN_GROUPS = {
    0: {
        'name': 'Group 0',
        'retail_price': 120.00,
        'materials': ['keon']
    },
    1: {
        'name': 'Group 1',
        'retail_price': 130.00,
        'materials': ['uyuni', 'sirius', 'malibu', 'keyla', 'fossil']
    },
    2: {
        'name': 'Group 2',
        'retail_price': 140.00,
        'materials': ['absolute white', 'calacatta oro', 'classic statuario', 'metal dark', 
                      'milan stone', 'pierre blue', 'ceppo', 'grigio', 'laos', 'marina', 
                      'soke', 'trilium', 'trance']
    },
    3: {
        'name': 'Group 3',
        'retail_price': 155.00,
        'materials': ['calacatta hermitage', 'calacatta lincoln', 'calacatta magnifico', 
                      'panda white', 'entzo', 'morpheus', 'neural', 'rem', 'salina', 'somnia']
    },
    4: {
        'name': 'Group 4',
        'retail_price': 175.00,
        'materials': ['arga', 'awake', 'bergen', 'helena', 'laurent', 'lucid', 'trance']
    }
}

# --- SOLID SURFACE PRICING (Updated May 2025) ---
SOLID_SURFACE_GROUPS = {
    1: {
        'name': 'Group 1',
        'retail_price': 79.00,
        'materials': ['glacier white', 'linen', 'pearl grey', 'designer white', 'frosty white',
                      'pearl mirage', 'powder white', 'mirage']
    },
    2: {
        'name': 'Group 2',
        'retail_price': 89.00,
        'materials': ['antarctica', 'artista canvas', 'artista gray', 'terrazzo moderna', 'everest',
                      'dusk ice', 'luminous white', 'chilled earth', 'milk glass spectra', 
                      'morning ice', 'peace grey', 'night stars', 'soothing grey']
    },
    3: {
        'name': 'Group 3',
        'retail_price': 109.00,
        'materials': ['sandalwood', 'sparkling granita', 'silver birch', 'yukon riverstone',
                      'cloud mist', 'grey beola', 'gulf coast', 'calacatta perlato', 'europa',
                      'calacatta emporio', 'masoned concrete']
    },
    4: {
        'name': 'Group 4',
        'retail_price': 120.00,
        'materials': ['ash aggregate', 'carrara lino', 'clam shell', 'limestone prima', 'rain cloud',
                      'dune prima', 'angel falls', 'arctic dune', 'aspen quartzite', 'arctic drift',
                      'beige travertine', 'carrara emporio', 'flint rock', 'hidden space', 
                      'ice statuario', 'monte paradiso', 'cashmere mirage', 'carbone marmo',
                      'brooklyn concrete', 'carrara royale', 'monte amiata', 'tumbled stone',
                      'whisper white']
    }
}

# --- MATERIAL TYPE MAPPING ---
MATERIAL_TYPE_GROUPS = {
    'quartz': QUARTZ_GROUPS,
    'granite': GRANITE_GROUPS,
    'signature': SIGNATURE_GROUPS,
    'porcelain': PORCELAIN_GROUPS,
    'solid_surface': SOLID_SURFACE_GROUPS
}

# --- CUSTOMER DISCOUNT STRUCTURE ---
CUSTOMER_DISCOUNTS = {
    'Retail': 0.00,
    'Dealer': 0.15,
    'Contractor': 0.25,
    'Home Builder': 0.25,
    'Commercial': 0.25,
    'LIA': 0.30,
    'Home Depot': 'special',
    'Costco': 'special'
}

# --- VALIDATION THRESHOLDS ---
VALIDATION_THRESHOLDS = {
    'revenue_warning': 50.00,
    'revenue_critical': 500.00,
    'cost_critical': 500.00
}

# --- MATERIALS PENDING ASSIGNMENT ---
UNASSIGNED_MATERIALS = {
    'royal blanc', 'mount royal', 'bianco delicato', 'bianco modesto', 'calacatta nero',
    'calacatta marmo', 'markina leathered', 'eternal bella', 'weybourne', 'crystal ice',
    'alpine mist', 'super white', 'domoos', 'brezza oro'
}

# --- PRICING FUNCTIONS ---

def identify_material_type(material_description):
    """
    Identify the material type (quartz, granite, etc.) from description.
    """
    if not material_description:
        return None
    
    desc_lower = str(material_description).lower()
    
    # Check for laminate indicators
    laminate_indicators = ['wilsonart', 'formica', 'arborite', 'laminate']
    if any(indicator in desc_lower for indicator in laminate_indicators):
        return 'laminate'
    
    # Check for solid surface indicators
    solid_surface_indicators = ['corian', 'himacs', 'solid surface']
    if any(indicator in desc_lower for indicator in solid_surface_indicators):
        return 'solid_surface'
    
    # Check for granite indicators
    granite_indicators = ['granite', 'cosentino granite']
    if any(indicator in desc_lower for indicator in granite_indicators):
        return 'granite'
    
    # Check for porcelain indicators
    porcelain_indicators = ['dekton', 'infinity', 'porcelain']
    if any(indicator in desc_lower for indicator in porcelain_indicators):
        return 'porcelain'
    
    # Check for signature series indicators
    signature_indicators = ['signature series']
    if any(indicator in desc_lower for indicator in signature_indicators):
        return 'signature'
    
    # Default to quartz for most stone materials
    quartz_indicators = ['silestone', 'caesarstone', 'hanstone', 'cambria', 'vicostone', 'quartz']
    if any(indicator in desc_lower for indicator in quartz_indicators):
        return 'quartz'
    
    return 'quartz'  # Default assumption

def get_material_group(material_name, material_type=None):
    """
    Find the material group for a given material name and type.
    """
    if not material_name:
        return None, None
    
    material_lower = material_name.lower().strip()
    
    # If material type is specified, search only in that category
    if material_type and material_type in MATERIAL_TYPE_GROUPS:
        groups = MATERIAL_TYPE_GROUPS[material_type]
        for group_num, group_data in groups.items():
            for known_material in group_data['materials']:
                if known_material.lower() in material_lower or material_lower in known_material.lower():
                    return group_num, material_type
        return None, material_type
    
    # Search all material types
    for mat_type, groups in MATERIAL_TYPE_GROUPS.items():
        for group_num, group_data in groups.items():
            for known_material in group_data['materials']:
                if known_material.lower() in material_lower or material_lower in known_material.lower():
                    return group_num, mat_type
    
    return None, None

def get_retail_price(material_group, material_type, sqft, customer_type):
    """
    Calculate retail price for a material group, type, square footage, and customer type.
    """
    if material_type not in MATERIAL_TYPE_GROUPS or customer_type not in CUSTOMER_DISCOUNTS:
        return None
    
    groups = MATERIAL_TYPE_GROUPS[material_type]
    if material_group not in groups:
        return None
    
    group_data = groups[material_group]
    base_price = group_data['retail_price']
    discount_rate = CUSTOMER_DISCOUNTS[customer_type]
    
    if discount_rate == 'special':
        return {
            'status': 'special_pricing',
            'base_price_per_sqft': base_price,
            'customer_type': customer_type,
            'material_type': material_type,
            'message': f'{customer_type} requires manual pricing review'
        }
    
    total_base_price = base_price * sqft
    discount_amount = total_base_price * discount_rate
    final_price = total_base_price - discount_amount
    
    return {
        'status': 'calculated',
        'material_group': material_group,
        'material_type': material_type,
        'group_name': group_data['name'],
        'customer_type': customer_type,
        'sqft': sqft,
        'base_price_per_sqft': base_price,
        'discount_rate': discount_rate,
        'discount_percent': discount_rate * 100,
        'total_base_price': total_base_price,
        'discount_amount': discount_amount,
        'final_price': final_price,
        'price_per_sqft': final_price / sqft if sqft > 0 else 0
    }

def get_expected_plant_cost(material_group, material_type, sqft):
    """
    Calculate expected interbranch billing cost (only available for quartz currently).
    """
    if material_type != 'quartz' or material_group not in QUARTZ_GROUPS:
        return None
    
    group_data = QUARTZ_GROUPS[material_group]
    
    cost_min = group_data['ib_cost_min'] * sqft
    cost_max = group_data['ib_cost_max'] * sqft
    cost_avg = (cost_min + cost_max) / 2
    
    return {
        'material_group': material_group,
        'material_type': material_type,
        'group_name': group_data['name'],
        'sqft': sqft,
        'cost_per_sqft_min': group_data['ib_cost_min'],
        'cost_per_sqft_max': group_data['ib_cost_max'],
        'cost_per_sqft_avg': (group_data['ib_cost_min'] + group_data['ib_cost_max']) / 2,
        'total_cost_min': cost_min,
        'total_cost_max': cost_max,
        'total_cost_avg': cost_avg
    }

def validate_job_pricing(material_group, material_type, sqft, customer_type, actual_revenue, actual_plant_cost=None):
    """
    Validate job pricing against expected pricing structure.
    """
    expected_retail = get_retail_price(material_group, material_type, sqft, customer_type)
    
    if not expected_retail:
        return {'status': 'error', 'message': 'Could not calculate expected pricing'}
    
    if expected_retail['status'] == 'special_pricing':
        return expected_retail
    
    revenue_variance = actual_revenue - expected_retail['final_price']
    revenue_variance_pct = (revenue_variance / expected_retail['final_price'] * 100) if expected_retail['final_price'] > 0 else 0
    
    issues = []
    
    # Revenue validation
    if abs(revenue_variance) > VALIDATION_THRESHOLDS['revenue_warning']:
        severity = 'critical' if abs(revenue_variance) > VALIDATION_THRESHOLDS['revenue_critical'] else 'warning'
        issues.append({
            'type': 'revenue_variance',
            'severity': severity,
            'message': f'Revenue ${actual_revenue:,.2f} vs expected ${expected_retail["final_price"]:,.2f} ({revenue_variance_pct:+.1f}%)',
            'variance_amount': revenue_variance
        })
    
    result = {
        'status': 'analyzed',
        'expected_retail': expected_retail,
        'actual_revenue': actual_revenue,
        'revenue_variance': revenue_variance,
        'revenue_variance_pct': revenue_variance_pct,
        'issues': issues,
        'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
        'warnings': len([i for i in issues if i['severity'] == 'warning'])
    }
    
    # Plant cost validation (only for quartz with interbranch costs)
    if actual_plant_cost is not None and material_type == 'quartz':
        expected_plant = get_expected_plant_cost(material_group, material_type, sqft)
        if expected_plant:
            plant_cost_variance = actual_plant_cost - expected_plant['total_cost_avg']
            plant_cost_variance_pct = (plant_cost_variance / expected_plant['total_cost_avg'] * 100) if expected_plant['total_cost_avg'] > 0 else 0
            
            if actual_plant_cost < expected_plant['total_cost_min'] or actual_plant_cost > expected_plant['total_cost_max']:
                severity = 'critical' if abs(plant_cost_variance) > VALIDATION_THRESHOLDS['cost_critical'] else 'warning'
                issues.append({
                    'type': 'plant_cost_variance',
                    'severity': severity,
                    'message': f'Plant cost ${actual_plant_cost:,.2f} outside expected range ${expected_plant["total_cost_min"]:,.2f}-${expected_plant["total_cost_max"]:,.2f}',
                    'variance_amount': plant_cost_variance
                })
            
            result.update({
                'expected_plant': expected_plant,
                'actual_plant_cost': actual_plant_cost,
                'plant_cost_variance': plant_cost_variance,
                'plant_cost_variance_pct': plant_cost_variance_pct,
                'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
                'warnings': len([i for i in issues if i['severity'] == 'warning'])
            })
    
    return result
