from logging import exception

import httpx
import os
import threading
from ..utils.exceptions import WikiAuthenticationError, WikiBusinessError
from ..utils.domain_config import get_domain_config


class RemoteAuthenticationError(Exception):
    """远程模式下X-Auth-Token缺失异常"""
    pass


def _fetch_new_cookie() -> str:
    """从环境变量读取配置并执行登录，获取拼接后的 Cookie"""
    login_url = "https://login.huawei.com/login1/rest/hwidcenter/login"
    headers = {"Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"}
    payload = {
        "lang": "zh_CN",
        "loginAccount": os.getenv("w3Account"),
        "password": os.getenv("password"),
        "uid": os.getenv("w3Account")
    }
    try:
        with httpx.Client(timeout=60.0, verify=False, trust_env=True) as client:
            resp = client.post(login_url, json=payload, headers=headers)
            resp.raise_for_status()

            # 核心：获取 Header 中所有的 Set-Cookie 并拼接
            cookie_items = []
            for key, value in resp.cookies.items():
                cookie_items.append(f"{key}={value}")
            if not cookie_items:
                raise WikiAuthenticationError()
            return "; ".join(cookie_items)
    except Exception as e:
        raise WikiAuthenticationError(e)


class HttpClientUtil:
    """HTTP客户端工具类，提供统一的GET/POST/PUT请求方法"""

    _cached_cookie = None
    _cookie_lock = threading.Lock()

    def __init__(self):
        """初始化HTTP客户端

        Args:
            is_remote: 是否为远程模式，默认从环境变量读取
        """
        self.base_url = "https://wiki.huawei.com"

    def get_url_prefix(self, domain_key: str) -> str:
        return self.base_url

    def get_common_headers(self, domain_key: str) -> dict:
        """构建通用的请求头

        Args:
            domain_key: 域名key
            ctx: MCP上下文

        Returns:
            包含认证信息的请求头字典
        """
        # 构建API端点URL和x-app-key
        if domain_key:
            app_key = get_domain_config(domain_key)
            if not app_key:
                raise ValueError(f"未找到域名配置: {domain_key}")
        else:
            app_key = "wiki"

        headers = {
            "Content-Type": "application/json",
            "x-app-key": app_key,
        }

            # Local模式：使用Cookie认证
        headers["Cookie"] = self.get_cookie()
        if domain_key == 'edevops':
            headers["x-requested-with"] = "XMLHttpRequest"
            headers["cftk"] = "wiki-mcp"
            headers["Cookie"] = headers.get("Cookie") + ";prod_cftk=wiki-mcp"

        return headers

    def get_cookie(self) -> str:
        if self._cached_cookie is None:
            self._cached_cookie = _fetch_new_cookie()
        return self._cached_cookie

    def _refresh_cookie(self) -> None:
        self._cached_cookie = _fetch_new_cookie()

    def _check_response(self, resp: httpx.Response) -> dict:
        # 登录校验 - 分别处理local和remote模式
        if resp.status_code == 401:
            # Local模式：标记cookie失效，由调用方处理重试
            self._cached_cookie = None
            raise WikiAuthenticationError()

        # 接口请求异常
        if resp.status_code != 200:
            raise WikiBusinessError(f"Wiki API error: {resp.status_code}")

        data = resp.json()

        # 无权限校验
        if data.get("code") in (-18014, -18031):
            raise WikiBusinessError(f"您暂无权限执行此操作")

        # 文档已删除校验
        if data.get("code") == -18002:
            raise WikiBusinessError("当前文档已被删除。")

        # 其余业务场景
        if data.get("code") != 200:
            raise WikiBusinessError(f"操作失败，原因：{data.get('message', '未知错误')}")

        return data

    async def get(self, url: str, params: dict, domain_key: str) -> dict | None:
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=60.0, verify=False, trust_env=True) as client:
                    resp = await client.get(self.get_url_prefix(domain_key) + url,
                                            headers=self.get_common_headers(domain_key), params=params)
                    return self._check_response(resp)
            except WikiAuthenticationError:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    self._refresh_cookie()
                    continue
                raise
            except httpx.TimeoutException:
                raise exception("连接超时，请稍后重试。")
            except Exception as e:
                raise e
        return None

    async def post(self, url: str, json_data, params: dict, domain_key: str) -> dict | None:
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=60.0, verify=False, trust_env=True) as client:
                    resp = await client.post(self.get_url_prefix(domain_key) + url,
                                             headers=self.get_common_headers(domain_key), json=json_data,
                                             params=params)
                    return self._check_response(resp)
            except WikiAuthenticationError:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    self._refresh_cookie()
                    continue
                raise
            except httpx.TimeoutException:
                raise exception("连接超时，请稍后重试。")
            except Exception as e:
                raise e
        return None

    async def put(self, url: str, json_data, params: dict, domain_key: str) -> dict | None:
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient(timeout=60.0, verify=False, trust_env=True) as client:
                    resp = await client.put(self.get_url_prefix(domain_key) + url,
                                            headers=self.get_common_headers(domain_key), json=json_data,
                                            params=params)
                    return self._check_response(resp)
            except WikiAuthenticationError:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    self._refresh_cookie()
                    continue
                raise
            except httpx.TimeoutException:
                raise exception("连接超时，请稍后重试。")
            except Exception as e:
                raise e
        return None
