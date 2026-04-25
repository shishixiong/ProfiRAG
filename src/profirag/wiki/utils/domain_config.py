"""域名配置工具

提供域名配置的简单查询功能。
"""

from typing import Dict

DOMAIN_CONFIGS = {
    "wiki": "wiki",
    "edevops": "8e1fa06c-d38e-4522-b537-ca02a3b3290b",
    "codehub": "56f69231-0ee9-11ed-8d72-fa163ecf9d11",
    "gdemate": "5998a38d-a111-11ed-9853-fa163e389e12",
    "wisedevops": "5100bb54-16e5-11ed-8d72-fa163ecf9d11"
}

def get_domain_config(domain_key: str) -> str:
    """获取域名对应的app_key
    
    Args:
        domain_key: 域名关键字
        
    Returns:
        app_key字符串，如果域名不支持则返回None
    """
    return DOMAIN_CONFIGS.get(domain_key.lower())

def is_supported_domain(domain_key: str) -> bool:
    """检查域名是否支持
    
    Args:
        domain_key: 域名关键字
        
    Returns:
        如果域名支持返回True，否则返回False
    """
    return domain_key.lower() in DOMAIN_CONFIGS

def get_all_domains() -> list[str]:
    """获取所有支持的域名列表
    
    Returns:
        域名关键字列表
    """
    return list(DOMAIN_CONFIGS.keys())