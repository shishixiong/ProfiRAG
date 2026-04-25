"""WIKI标识符提取模块

提供WIKI标识符的提取和格式验证功能。
"""

import re
from typing import Tuple
from .exceptions import WikiIdentifierError


class WikiIdentifierExtractor:
    """WIKI标识符提取器

    负责从URL中提取WIKI标识符并验证其格式。
    """

    # WIKI标识符正则表达式：WIKI + 10~16位数字
    WIKI_PATTERN = r"WIKI(\d{10,16})"

    def extract_wiki_sn(self, url: str) -> str:
        """从URL中提取WIKI标识符

        Args:
            url: URL字符串

        Returns:
            WIKI标识符 (WIKI + 数字部分)

        Raises:
            ValueError: URL中不包含有效的WIKI标识符
        """
        match = re.search(self.WIKI_PATTERN, url, re.IGNORECASE)

        if not match:
            raise WikiIdentifierError(
                url,
                f"URL中不包含有效的WIKI标识符。WIKI标识符格式应为：WIKI + 10~16位数字 (例如: WIKI1234567890)"
            )

        wiki_sn = match.group(0)

        # 验证提取的标识符格式
        is_valid, error_msg = self.validate_wiki_sn(wiki_sn)
        if not is_valid:
            raise ValueError(error_msg)

        return wiki_sn

    def validate_wiki_sn(self, wiki_sn: str) -> Tuple[bool, str]:
        """验证WIKI标识符格式

        Args:
            wiki_sn: WIKI标识符字符串

        Returns:
            (验证结果, 错误信息) 元组
        """
        if not wiki_sn:
            return (False, "WIKI标识符不能为空")

        # 检查是否匹配WIKI+数字格式
        match = re.fullmatch(r"WIKI(\d{10,16})", wiki_sn, re.IGNORECASE)
        if not match:
            return (False, f"WIKI标识符格式无效: {wiki_sn}。应为WIKI + 10~16位数字")

        # 验证数字范围
        num_part = match.group(1)
        if len(num_part) < 10:
            return (False, f"WIKI标识符数字部分不足10位: {len(num_part)}位")

        if len(num_part) > 16:
            return (False, f"WIKI标识符数字部分超过16位: {len(num_part)}位")

        return (True, "")

    def _normalize_wiki_sn(self, wiki_sn: str) -> str:
        """标准化WIKI标识符（转换为大写格式）

        Args:
            wiki_sn: WIKI标识符字符串

        Returns:
            标准化后的WIKI标识符（WIKI + 数字）
        """
        if not wiki_sn:
            return wiki_sn

        match = re.search(r"WIKI(\d{10,16})", wiki_sn, re.IGNORECASE)
        if match:
            return f"WIKI{match.group(1)}"

        return wiki_sn