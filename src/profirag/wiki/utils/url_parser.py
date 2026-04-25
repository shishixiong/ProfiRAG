"""URL解析模块

提供URL中域名关键字的提取和验证功能。
"""

import re
from urllib.parse import urlparse
from typing import Optional
from .exceptions import InvalidURLError, DomainNotSupportedError
from .domain_config import is_supported_domain, get_all_domains


class URLParser:
    """URL解析器

    负责从URL中提取域名关键字并验证URL格式。
    """

    def __init__(self):
        """初始化URL解析器"""
        self._supported_domains = get_all_domains()

    def parse_domain(self, url: str) -> str:
        """从URL中解析域名关键字

        Args:
            url: 待解析的URL字符串

        Returns:
            域名关键字

        Raises:
            InvalidURLError: URL格式无效或不包含支持的域名
            DomainNotSupportedError: 域名不被支持
        """
        if not self.validate_url_format(url):
            raise InvalidURLError(url, "URL格式无效")

        domain_key = self._extract_domain_key(url)

        if not domain_key:
            raise InvalidURLError(
                url, 
                f"URL中不包含支持的域名。支持的域名: {', '.join(self._supported_domains)}"
            )

        if not is_supported_domain(domain_key):
            raise DomainNotSupportedError(domain_key)

        return domain_key

    def validate_url_format(self, url: str) -> bool:
        """验证URL基本格式

        Args:
            url: 待验证的URL字符串

        Returns:
            URL格式有效返回True，否则返回False
        """
        if not url or not isinstance(url, str):
            return False

        try:
            parsed = urlparse(url)

            # 必须有scheme (http/https) 和 netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # scheme必须是http或https
            if parsed.scheme not in ('http', 'https'):
                return False

            return True

        except Exception:
            return False

    def _extract_domain_key(self, url: str) -> Optional[str]:
        """从URL中提取域名关键字

        Args:
            url: URL字符串

        Returns:
            域名关键字，如果未找到则返回None
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # 按优先级顺序匹配域名（避免部分匹配问题）
            # 例如：避免wisedevops匹配到edevops
            domain_patterns = sorted(
                self._supported_domains,
                key=lambda x: len(x),
                reverse=True
            )

            for supported_domain in domain_patterns:
                if supported_domain in domain:
                    return supported_domain

        except Exception:
            pass

        return None