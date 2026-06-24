"""
Company intelligence feature extraction: startup, product, and enterprise ratios.
"""
from src.jd_parser import IT_SERVICES_COMPANIES, PRODUCT_COMPANIES
from datetime import datetime

CURRENT_DATE = datetime(2026, 6, 20)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def compute_company_intelligence(candidate):
    """
    Computes career product ratios and current company type as secondary features.
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    
    current_company = p.get("current_company", "").strip().lower()
    
    total_months = 0
    product_months = 0
    recent_total_months = 0
    recent_product_months = 0
    
    # 1. Evaluate current company type
    current_type = 0.5  # Default neutral
    if current_company in PRODUCT_COMPANIES:
        current_type = 1.0
    elif current_company in IT_SERVICES_COMPANIES:
        current_type = 0.1
        
    for role in career:
        comp = role.get("company", "").strip().lower()
        desc = role.get("description", "").strip().lower()
        ind = role.get("industry", "").strip().lower()
        duration = role.get("duration_months", 0)
        
        is_services = comp in IT_SERVICES_COMPANIES or "it services" in ind or "consulting" in ind
        is_product = comp in PRODUCT_COMPANIES or "product company" in desc or "saas" in desc
        is_startup = "startup" in desc or "seed stage" in desc
        
        # Classify product or startup
        is_prod_or_startup = is_product or is_startup
        
        total_months += duration
        if is_prod_or_startup:
            product_months += duration
            
        # Recent history (within last 36 months)
        start_dt = parse_date(role.get("start_date"))
        if start_dt:
            months_ago = (CURRENT_DATE - start_dt).days / 30.5
            if months_ago <= 36.0:
                recent_total_months += duration
                if is_prod_or_startup:
                    recent_product_months += duration
                    
    career_product_ratio = 0.0
    if total_months > 0:
        career_product_ratio = product_months / total_months
        
    recent_product_ratio = 0.0
    if recent_total_months > 0:
        recent_product_ratio = recent_product_months / recent_total_months
        
    return {
        "career_product_ratio": float(career_product_ratio),
        "recent_product_ratio": float(recent_product_ratio),
        "current_company_type": float(current_type)
    }
