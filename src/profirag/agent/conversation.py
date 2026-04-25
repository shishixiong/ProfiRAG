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
    r"(更多|更详细)(的|地)?",
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


class ConversationManager:
    """Stateful wrapper for multi-turn conversations with any Agent."""

    def __init__(
        self,
        agent: Any,
        llm: Any,
        max_history_turns: int = 6,
        keep_recent_turns: int = 2,
        enable_auto_context: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            agent: RAGReActAgent or RAGPlanAgent instance
            llm: LLM instance for summarization and context decision
            max_history_turns: Maximum turns before summarization triggers
            keep_recent_turns: Number of recent turns kept verbatim
            enable_auto_context: Enable LLM-based context decision
            verbose: Print enrichment details
        """
        self.agent = agent
        self.llm = llm
        self.max_history_turns = max_history_turns
        self.keep_recent_turns = keep_recent_turns
        self.enable_auto_context = enable_auto_context
        self.verbose = verbose

        self.state = ConversationState(
            session_id=str(uuid.uuid4())[:8],
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )

    def reset(self) -> None:
        """Clear conversation state for new session."""
        self.state = ConversationState(
            session_id=str(uuid.uuid4())[:8],
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )

    def get_history(self) -> List[ConversationTurn]:
        """Return full conversation history."""
        return self.state.turns.copy()

    def get_summary(self) -> str:
        """Return current conversation summary."""
        return self.state.summary

    def export_state(self) -> Dict:
        """Export state for debugging/testing."""
        return {
            "session_id": self.state.session_id,
            "turns": [t.model_dump() for t in self.state.turns],
            "summary": self.state.summary,
            "created_at": self.state.created_at.isoformat(),
            "last_activity": self.state.last_activity.isoformat(),
        }

    def import_state(self, state_dict: Dict) -> None:
        """Import previous state for testing/debugging."""
        self.state = ConversationState(
            session_id=state_dict.get("session_id", str(uuid.uuid4())[:8]),
            turns=[ConversationTurn(**t) for t in state_dict.get("turns", [])],
            summary=state_dict.get("summary", ""),
            created_at=datetime.fromisoformat(state_dict["created_at"]) if "created_at" in state_dict else datetime.now(),
            last_activity=datetime.fromisoformat(state_dict["last_activity"]) if "last_activity" in state_dict else datetime.now(),
        )

    def _detect_explicit_reference(self, query: str) -> bool:
        """Detect explicit reference patterns in query.

        Args:
            query: User query string

        Returns:
            True if explicit reference pattern found
        """
        for pattern in EXPLICIT_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _enrich_query(self, query: str, use_recent_turns: bool = False) -> str:
        """Enrich query with conversation context.

        Args:
            query: Original user query
            use_recent_turns: Include recent turns in context

        Returns:
            Enriched query string
        """
        # Build context string
        context_parts = []

        if self.state.summary:
            context_parts.append(f"摘要: {self.state.summary}")

        if use_recent_turns and self.state.turns:
            recent = self.state.turns[-self.keep_recent_turns:]
            for turn in recent:
                context_parts.append(f"问: {turn.query}")
                context_parts.append(f"答: {turn.response[:200]}")

        if not context_parts:
            return query  # No enrichment

        context_str = "\n".join(context_parts)

        # Choose enrichment format
        if use_recent_turns:
            return f"【上下文】\n{context_str}\n\n用户问题：{query}"
        else:
            return f"【相关背景】\n{context_str}\n\n用户问题：{query}"

    def _should_inject_context_llm(self, query: str) -> bool:
        """Use LLM to decide if context should be injected.

        Args:
            query: User query

        Returns:
            True if context should be injected
        """
        if not self.enable_auto_context:
            return False

        if not self.state.summary and not self.state.turns:
            return False

        # Get last turn for context
        last_turn_str = ""
        if self.state.turns:
            last = self.state.turns[-1]
            last_turn_str = f"问: {last.query}\n答: {last.response[:100]}"

        prompt = CONTEXT_DECISION_PROMPT.format(
            summary=self.state.summary or "无",
            last_turn=last_turn_str or "无",
            query=query,
        )

        try:
            response = self.llm.complete(prompt)
            # Parse JSON response
            text = response.text.strip()
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("needs_context", False)
        except Exception:
            pass  # Fallback to False on any error

        return False

    def _summarize_history(self, turns: List[ConversationTurn]) -> str:
        """Generate summary from conversation turns.

        Args:
            turns: List of turns to summarize

        Returns:
            Summary string
        """
        if not turns:
            return ""

        # Build turns text
        turns_text = []
        for turn in turns:
            turns_text.append(f"问: {turn.query}")
            turns_text.append(f"答: {turn.response[:100]}")

        prompt = SUMMARIZATION_PROMPT.format(turns_text="\n".join(turns_text))

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception:
            # Fallback: simple concatenation of queries
            queries = [t.query for t in turns]
            return f"用户询问了: {', '.join(queries[:5])}"

    def _maybe_summarize(self) -> None:
        """Trigger summarization if threshold exceeded."""
        if not self.state.needs_summarization(self.max_history_turns):
            return

        # Turns to summarize (all except recent)
        turns_to_summarize = self.state.turns[:-self.keep_recent_turns]
        recent_turns = self.state.turns[-self.keep_recent_turns:]

        # Generate new summary (combine with existing)
        new_summary = self._summarize_history(turns_to_summarize)
        if self.state.summary:
            # Combine summaries
            self.state.summary = f"{self.state.summary}\n{new_summary}"
        else:
            self.state.summary = new_summary

        # Keep only recent turns
        self.state.turns = recent_turns

        if self.verbose:
            print(f"📋 Summarized {len(turns_to_summarize)} turns into summary")