"""Multi-turn conversation support for RAG Agents."""

import re
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field

# Explicit reference patterns for Chinese
EXPLICIT_PATTERNS = [
    r"基于上(面|述|文)",
    r"根据(刚才|之前|上文)",
    r"继续(讨论|说明|解释)",
    r"那个(问题|文档|概念)",
    r"它(是指|是什么|怎么样)",
    r"关于(这|那)(个|些)",
    r"进一步",
    r"还有(什么|哪些)",
    r"(更多|更详细)(的|地)",
]

# Prompts
CONTEXT_DECISION_PROMPT = """判断以下新问题是否需要参考之前的对话历史。

对话摘要: {summary}
最近问答: {last_turn}

新问题: {query}

判断标准:
- 问题中提到之前讨论的概念/术语 → 需要
- 问题是对之前回答的追问 → 需要
- 问题完全独立、新话题 → 不需要

输出JSON: {{"needs_context": true/false, "reason": "简短说明"}}"""

SUMMARIZATION_PROMPT = """请将以下对话历史压缩为简洁的摘要，保留关键信息。

要求:
1. 保留用户询问的主要问题（列举）
2. 保留讨论的关键概念/术语
3. 不包含具体回答细节（只需提及"已讨论X、Y、Z等概念"）
4. 控制在150字以内

对话历史:
{turns_text}

摘要:"""


class ConversationTurn(BaseModel):
    """Single conversation exchange."""
    query: str
    response: str
    timestamp: datetime
    tool_calls: List[Dict] = Field(default_factory=list)
    mode: str  # "react" or "plan"


class ConversationState(BaseModel):
    """Session conversation state."""
    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime
    last_activity: datetime

    def total_turns(self) -> int:
        """Return total number of turns."""
        return len(self.turns)

    def needs_summarization(self, threshold: int) -> bool:
        """Check if summarization is needed."""
        return len(self.turns) > threshold


class QueryEnrichmentResult(BaseModel):
    """Result of query processing."""
    original_query: str
    enriched_query: str
    injected_context: bool = False
    reference_detected: bool = False
    context_source: str = "none"  # "summary" | "recent_turns" | "none"