"""Rule-based extractor for deterministic information extraction."""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

from .cleaner_config import RuleResult, CleanerConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Regex Pattern Libraries (Extensible)
# ============================================================================

# Error code patterns for various systems
ERROR_CODE_PATTERNS = [
    # Database systems
    r'GAUSS-\d{5}',           # GaussDB error codes
    r'ORA-\d{5}',             # Oracle error codes
    r'PG-\d{4,5}',            # PostgreSQL error codes
    r'MYSQL-\d{4}',           # MySQL error codes
    r'SQL-\d{4,5}',           # Generic SQL errors

    # General error patterns
    r'ERROR\s+\d{4,5}',       # Generic ERROR format
    r'ERR-\w+-\d+',           # Custom error codes
    r'ERRCODE\s+\w+',         # Error code references
    r'ErrorCode:\s*\d+',      # ErrorCode prefix

    # Application specific
    r'E\d{3,5}',              # Short error codes
    r'\b\d{5}\b',             # 5-digit codes (context-aware)
]

# Log patterns / error messages
LOG_PATTERNS = [
    # Connection issues
    r'connection\s+refused',
    r'connection\s+failed',
    r'connection\s+timeout',
    r'connection\s+reset',
    r'could\s+not\s+connect',
    r'failed\s+to\s+connect',
    r'unable\s+to\s+connect',

    # Timeout issues
    r'timeout|timed\s+out',
    r'TIMEOUT',
    r'request\s+timeout',

    # Permission/Auth issues
    r'permission\s+denied',
    r'access\s+denied',
    r'authentication\s+failed',
    r'auth\s+error',
    r'login\s+failed',
    r'invalid\s+credentials',
    r'password\s+incorrect',

    # Resource issues
    r'out\s+of\s+memory',
    r'memory\s+allocation\s+failed',
    r'disk\s+full',
    r'no\s+space\s+left',
    r'storage\s+full',
    r'resource\s+unavailable',
    r'insufficient\s+resources',

    # Network issues
    r'network\s+error',
    r'network\s+unreachable',
    r'host\s+unreachable',
    r'no\s+route\s+to\s+host',
    r'socket\s+error',
    r'port\s+already\s+in\s+use',

    # Data issues
    r'data\s+corruption',
    r'invalid\s+data',
    r'data\s+format\s+error',
    r'parse\s+error',
    r'syntax\s+error',
    r'encoding\s+error',

    # Process issues
    r'process\s+terminated',
    r'process\s+killed',
    r'process\s+died',
    r'child\s+process\s+failed',
    r'fork\s+failed',

    # Configuration issues
    r'config\s+error',
    r'configuration\s+invalid',
    r'invalid\s+parameter',
    r'missing\s+config',
    r'config\s+file\s+not\s+found',

    # General failure patterns
    r'failed\s+to\s+\w+',
    r'error\s+in\s+\w+',
    r'exception\s+\w+',
    r'fatal\s+error',
    r'critical\s+error',
]

# Environment patterns
ENVIRONMENT_PATTERNS = {
    # Database versions
    'database': [
        r'(PostgreSQL|postgres)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(Oracle)\s+(\d+[cRg]?(?:\.\d+)?)',
        r'(MySQL)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(GaussDB|gaussdb)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(SQLServer|sqlserver)\s+(\d+)',
        r'(MongoDB|mongodb)\s+(\d+\.\d+(?:\.\d+)?)',
    ],
    # Operating systems
    'os': [
        r'(Linux)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(CentOS|centos)\s+(\d+)',
        r'(Ubuntu|ubuntu)\s+(\d+\.\d+)',
        r'(RedHat|RHEL|rhel)\s+(\d+)',
        r'(Windows)\s+(\d+)',
        r'(SUSE|suse)\s+(\d+)',
    ],
    # Middleware
    'middleware': [
        r'(Tomcat|tomcat)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(Nginx|nginx)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(Apache|apache)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(Redis|redis)\s+(\d+\.\d+(?:\.\d+)?)',
        r'(Kafka|kafka)\s+(\d+\.\d+(?:\.\d+)?)',
    ],
}

# Service component patterns
SERVICE_COMPONENT_PATTERNS = [
    # Database services
    r'gaussdb',
    r'gausskernel',
    r'gsql',
    r'gs_ctl',
    r'gs_dump',
    r'gs_restore',
    r'postgres',
    r'postgresql',
    r'mysql',
    r'oracle',
    r'sqlserver',

    # Cluster management
    r'om',
    r'gs_om',
    r'etcd',
    r'cms',  # Cluster Management Service
    r'gtm',  # Global Transaction Manager
    r'cm_agent',
    r'cm_server',

    # Middleware
    r'tomcat',
    r'nginx',
    r'apache',
    r'redis',
    r'kafka',
    r'zookeeper',
    r'memcached',

    # System services
    r'systemd',
    r'crond',
    r'sshd',
    r'networkd',
    r'logrotate',
    r'supervisord',
]

# Problem classification keywords
PROBLEM_KEYWORDS = {
    '性能问题': [
        '慢查询', '性能下降', '响应慢', '高延迟', '吞吐量低',
        '卡顿', 'hang', 'slow', 'performance', '瓶颈',
    ],
    '连接问题': [
        '连接失败', '无法连接', '连接超时', '断连', '掉线',
        'connection', '网络', '端口', 'socket',
    ],
    '数据问题': [
        '数据丢失', '数据不一致', '数据损坏', '数据错误',
        'corruption', 'inconsistency', 'data loss',
    ],
    '权限问题': [
        '权限不足', '拒绝访问', '认证失败', '登录失败',
        'permission', 'access denied', 'authentication',
    ],
    '配置问题': [
        '配置错误', '参数错误', '设置不当', '配置冲突',
        'config', 'parameter', 'setting',
    ],
    '资源问题': [
        '内存不足', '磁盘满', '空间不足', '资源耗尽',
        'memory', 'disk', 'space', 'resource',
    ],
    '集群问题': [
        '集群故障', '节点宕机', '主备切换', '脑裂',
        'cluster', 'node', 'failover', 'split-brain',
    ],
    '备份恢复': [
        '备份失败', '恢复失败', '数据导出', '数据导入',
        'backup', 'restore', 'dump', 'import',
    ],
}


class RuleExtractor:
    """规则提取器 - 提取确定性信息"""

    def __init__(self, config: Optional[CleanerConfig] = None):
        self.config = config or CleanerConfig()
        self._compiled_error_patterns = self._compile_patterns(ERROR_CODE_PATTERNS)
        self._compiled_log_patterns = self._compile_patterns(LOG_PATTERNS)
        self._compiled_env_patterns = self._compile_env_patterns()
        self._compiled_service_patterns = self._compile_patterns(
            SERVICE_COMPONENT_PATTERNS, case_insensitive=True
        )

    def _compile_patterns(
        self,
        patterns: List[str],
        case_insensitive: bool = True
    ) -> List[re.Pattern]:
        """编译正则表达式"""
        flags = re.IGNORECASE if case_insensitive else 0
        return [re.compile(p, flags) for p in patterns]

    def _compile_env_patterns(self) -> Dict[str, List[Tuple[re.Pattern, str]]]:
        """编译环境模式，返回(category, pattern, prefix)"""
        result = {}
        for category, patterns in ENVIRONMENT_PATTERNS.items():
            compiled = []
            for p in patterns:
                # Pattern captures both name and version
                compiled.append((re.compile(p, re.IGNORECASE), category))
            result[category] = compiled
        return result

    def extract(self, text: str) -> RuleResult:
        """从文本中提取所有规则信息"""
        if len(text) < self.config.min_document_length:
            logger.warning(f"Document too short: {len(text)} chars")
            return RuleResult()

        # 截取最大长度
        if len(text) > self.config.max_document_length:
            text = text[:self.config.max_document_length]

        return RuleResult(
            error_codes=self._extract_error_codes(text),
            log_patterns=self._extract_log_patterns(text),
            environment=self._extract_environment(text),
            service_components=self._extract_service_components(text),
            keywords=self._extract_keywords(text),
        )

    def _extract_error_codes(self, text: str) -> List[str]:
        """提取错误码"""
        codes: Set[str] = set()
        for pattern in self._compiled_error_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # 处理元组匹配（多个捕获组）
                if isinstance(match, tuple):
                    match = ''.join(match)
                codes.add(match.upper() if match.isalpha() or '-' in match else match)
        return sorted(codes)

    def _extract_log_patterns(self, text: str) -> List[str]:
        """提取日志特征"""
        patterns: Set[str] = set()
        for pattern in self._compiled_log_patterns:
            matches = pattern.findall(text)
            for match in matches:
                patterns.add(match.lower())
        return sorted(patterns)

    def _extract_environment(self, text: str) -> Dict[str, str]:
        """提取环境信息"""
        env: Dict[str, str] = {}
        for category, pattern_list in self._compiled_env_patterns.items():
            for pattern, _ in pattern_list:
                matches = pattern.findall(text)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        name = match[0]
                        version = match[1]
                        # 合并同类型信息
                        key = f"{category}_{name.lower()}"
                        if key not in env:
                            env[key] = f"{name} {version}"
                        # 也记录纯类别信息
                        if category not in env:
                            env[category] = f"{name} {version}"
        return env

    def _extract_service_components(self, text: str) -> List[str]:
        """提取相关服务组件"""
        components: Set[str] = set()
        for pattern in self._compiled_service_patterns:
            matches = pattern.findall(text)
            for match in matches:
                components.add(match.lower())
        return sorted(components)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取问题分类关键词"""
        keywords: Set[str] = set()
        text_lower = text.lower()

        for category, kw_list in PROBLEM_KEYWORDS.items():
            for kw in kw_list:
                if kw.lower() in text_lower:
                    keywords.add(category)
                    keywords.add(kw)

        return sorted(keywords)

    def get_hints_for_llm(self, result: RuleResult) -> str:
        """生成LLM提示信息"""
        hints = []

        if result.error_codes:
            hints.append(f"错误码: {', '.join(result.error_codes)}")

        if result.log_patterns:
            hints.append(f"日志特征: {', '.join(result.log_patterns)}")

        if result.environment:
            env_str = ", ".join(f"{k}: {v}" for k, v in result.environment.items())
            hints.append(f"环境信息: {env_str}")

        if result.service_components:
            hints.append(f"相关服务组件: {', '.join(result.service_components)}")

        if result.keywords:
            hints.append(f"问题关键词: {', '.join(result.keywords)}")

        return "\n".join(hints) if hints else "未识别到特定信息"