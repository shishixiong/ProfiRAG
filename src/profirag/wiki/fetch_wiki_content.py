from dotenv import load_dotenv
from pathlib import Path

import asyncio
import json
from ..wiki.services.wiki_api import WikiService, process_paragraphs
from ..wiki.utils.url_parser import URLParser
from ..wiki.utils.wiki_identifier import WikiIdentifierExtractor

# Load .env from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env", override=True)


def fetch_wiki_content(wiki_url, cookie=None):
    wiki_service = WikiService(cookie=cookie)

    try:
        # 步骤1: 解析URL获取域名关键字
        domain_key = URLParser().parse_domain(wiki_url)

        # 步骤2: 提取WIKI标识符
        wiki_sn = WikiIdentifierExtractor().extract_wiki_sn(wiki_url)

        # 步骤3: 调用Wiki API获取内容
        res = asyncio.run(wiki_service.get_wiki_content(0, 0, wiki_sn, domain_key))

        # 步骤4: 数据转换
        data = res["data"]
        content = process_paragraphs(data.get("paragraphs", []))
        document_type = "Markdown" if data.get("paragraphs")[0].get("ui_source") == 2 else "富文本"
        print(data)
        # 步骤5: 格式化输出
        return {"document_type": document_type, "title": data.get('title'),
                  "viewCount": data.get('wikiStatistic').get('viewCount'),
                  "author": data.get('owner', {}).get('name_full'),
                  "lastUpdateTime": data.get('last_update_time'), "content": content}
        # print(result)
    except Exception as e:
        print(e)
        return None