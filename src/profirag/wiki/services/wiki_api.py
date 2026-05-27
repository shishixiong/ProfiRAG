import time
from typing import Optional

from ..utils.decoder import html_to_markdown, decode_html_entities
from ..utils.exceptions import WikiBusinessError
from ..utils.http_client_util import HttpClientUtil


def process_paragraphs(paragraphs: list) -> str:
    """根据 ui_source 转换数据"""
    parts = []
    for p in paragraphs:
        content = p.get("content", "")
        ui_source = p.get("ui_source")

        if ui_source == 1:
            parts.append(html_to_markdown(content))
        elif ui_source == 2:
            parts.append(decode_html_entities(content))
        else:
            parts.append(content)
    return "\n".join(parts)


class WikiService:
    def __init__(self, cookie: Optional[str] = None):
        # 基础配置 - 基于环境变量和常量
        self.http_client = HttpClientUtil(cookie=cookie)

    async def get_wiki_content(self, domain_id: int, kanban_id: int, wiki_sn: str, domain_key: str):
        payload = {
            "wiki_sn": wiki_sn,
            "type": "UI",
            "request_tag": str(int(time.time())),
            "domain_id": domain_id,
            "kanban_id": kanban_id
        }
        data = await self.http_client.post('/devops-knowledge-management/api/getWiki', json_data=payload, params={},
                                           domain_key=domain_key)
        if data.get("data").get("status") != "published":
            raise WikiBusinessError("该文档已删除")
        return data

    async def create_wiki(self, domain_id, kanban_id, parent_id, title, content, domain_key, ctx):
        # 获取文档sn
        data = await self.http_client.get('/devops-knowledge-management/api/wiki/number',
                                          params={"domainId": domain_id, "kanbanId": kanban_id},
                                          domain_key=domain_key, ctx=ctx)
        sn = data.get("data")

        # 创建文档
        payload = [{
            "title": title,
            "description": content,
            "category": "document",
            "parent_id": parent_id,
            "kanban_id": kanban_id,
            "ui_source": 2,
            "assigned_domain_id": domain_id,
            "sn": sn,
            "sub_classify": "completed",
            "article_config": "{\"isAutoNum\":false,\"isAutoLineBreak\":false}"
        }]
        data = await self.http_client.post('/devops-knowledge-management/api/wiki', json_data=payload, params={},
                                           domain_key=domain_key, ctx=ctx)

        failed_list = data.get("data", {}).get("failed", [])
        if failed_list:
            raise WikiBusinessError(
                f"新建文档失败，code：{failed_list[0].get('code')}, message:{failed_list[0].get('message')}")
        return sn

    async def update_wiki(self, wiki_id, article_config, title, content, domain_key, ctx):
        payload = [{
            "article_config": article_config,
            "id": wiki_id,
            "sub_classify": "completed",
            "ui_source": 2
        }]
        if title:
            payload[0]["title"] = title
        if content:
            payload[0]["description"] = content
        data = await self.http_client.put('/devops-knowledge-management/api/wiki', json_data=payload, params={},
                                          domain_key=domain_key, ctx=ctx)
        failed_list = data.get("data", {}).get("failed", [])
        if failed_list:
            raise WikiBusinessError(
                f"更新文档失败，code：{failed_list[0].get('code')}, message:{failed_list[0].get('message')}")
