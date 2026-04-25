
class WikiAuthenticationError(Exception):
    """认证失败（账号密码/Token 错误或缺失）"""
    def __init__(self, message="Wiki认证失败，请检查账号密码！"):
        self.message = message
        super().__init__(self.message)


class WikiBusinessError(Exception):
    """wiki业务提示"""
    def __init__(self, message="业务提示信息"):
        self.message = message
        super().__init__(self.message)


class DomainNotSupportedError(WikiBusinessError):
    """域名不支持错误"""
    def __init__(self, domain_key: str):
        self.domain_key = domain_key
        super().__init__(f"域名 '{domain_key}' 不被支持，支持域名: wiki, edevops, codehub, gdemate, wisedevops")


class WikiIdentifierError(WikiBusinessError):
    """WIKI标识符格式错误"""
    def __init__(self, wiki_sn: str, reason: str):
        self.wiki_sn = wiki_sn
        super().__init__(f"WIKI标识符 '{wiki_sn}' 格式错误: {reason}")


class InvalidURLError(WikiBusinessError):
    """URL格式错误"""
    def __init__(self, url: str, reason: str):
        self.url = url
        super().__init__(f"URL格式错误 '{url}': {reason}")


class ConfigurationError(WikiBusinessError):
    """配置错误"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"配置错误: {message}")


# 错误消息常量
ERROR_MESSAGES = {
    'domain_not_supported': "'{domain_key}' 不被支持，支持域名: wiki, edevops, codehub, gdemate, wisedevops",
    'invalid_wiki_sn': "WIKI标识符格式无效，应为 WIKI + 10~16位数字",
    'invalid_url': "URL格式无效",
    'config_not_found': "域名配置未找到",
    'unsupported_parameter_format': "不支持的参数格式，请检查域名支持的URL格式"
}


def format_domain_error(domain_key: str, error_type: str) -> str:
    """格式化域名错误消息
    
    Args:
        domain_key: 域名关键字
        error_type: 错误类型
        
    Returns:
        格式化的错误消息
    """
    template = ERROR_MESSAGES.get(error_type, "未知域名错误")
    return template.format(domain_key=domain_key)


def validate_supported_domains(domains: list[str]) -> bool:
    """验证域名列表是否都支持
    
    Args:
        domains: 域名列表
        
    Returns:
        所有域名都返回True，否则返回False
    """
    supported_domains = {'wiki', 'edevops', 'codehub', 'gdemate', 'wisedevops'}
    return all(domain in supported_domains for domain in domains)


def log_error(error: Exception, error_type: str = "general"):
    """记录错误日志（可选实现）

    Args:
        error: 异常对象
        error_type: 错误类型
    """
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"[{error_type}] {str(error)}")
    except Exception:
        # 如果日志记录失败，不影响主流程
        pass


def localize_error_message(error_type: str, params: dict[str, str]) -> str:
    """本地化错误消息（可选实现）

    Args:
        error_type: 错误类型
        params: 参数字典

    Returns:
        本地化的错误消息
    """
    # 简单实现，实际项目中可以结合多语言库
    message = ERROR_MESSAGES.get(error_type, str(params.get('default', '未知错误')))
    return message.format(**params)