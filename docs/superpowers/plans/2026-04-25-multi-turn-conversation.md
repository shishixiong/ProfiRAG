# Multi-Turn Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-based conversation memory to Agent system via ConversationManager wrapper.

**Architecture:** ConversationManager wraps any Agent (ReAct or Plan), maintains conversation state with auto-summarization, enriches queries with context when needed. No changes to existing Agent classes.

**Tech Stack:** Python, Pydantic, LlamaIndex, FastAPI

---

## File Structure

```
src/profirag/agent/
├── conversation.py          # NEW: ConversationTurn, ConversationState, ConversationManager
├── __init__.py              # MODIFY: Add exports

src/profirag/config/
├── settings.py              # MODIFY: Add ConversationConfig

src/profirag/pipeline/
├── rag_pipeline.py          # MODIFY: Add create_conversation_manager method

web/api/
├── schemas.py               # MODIFY: Add conversation fields to ChatRequest/Response
├── services.py              # MODIFY: Add session management in ChatService
├── routes/chat.py           # MODIFY: Add session endpoints

tests/agent/
├── test_conversation.py     # NEW: Unit tests for ConversationManager
```

---

### Task 1: Data Models and Constants

**Files:**
- Create: `src/profirag/agent/conversation.py`
- Create: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for ConversationTurn**

```python
# tests/agent/test_conversation.py
"""Tests for ConversationManager and related models."""

from datetime import datetime
from profirag.agent.conversation import ConversationTurn, ConversationState

def test_conversation_turn_creation():
    """Test ConversationTurn model creation."""
    turn = ConversationTurn(
        query="What is Qdrant?",
        response="Qdrant is a vector database...",
        timestamp=datetime.now(),
        mode="react",
    )
    assert turn.query == "What is Qdrant?"
    assert turn.response == "Qdrant is a vector database..."
    assert turn.mode == "react"
    assert turn.tool_calls == []

def test_conversation_turn_with_tool_calls():
    """Test ConversationTurn with tool calls."""
    turn = ConversationTurn(
        query="test",
        response="result",
        timestamp=datetime.now(),
        mode="plan",
        tool_calls=[{"tool": "vector_search", "input": {"query": "test"}}],
    )
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["tool"] == "vector_search"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'profirag.agent.conversation'"

- [ ] **Step 3: Create conversation.py with data models and constants**

```python
# src/profirag/agent/conversation.py
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

输出JSON: {{\"needs_context\": true/false, \"reason\": \"简短说明\"}}"""

SUMMARIZATION_PROMPT = """请将以下对话历史压缩为简洁的摘要，保留关键信息。

要求:
1. 保留用户询问的主要问题（列举）
2. 保留讨论的关键概念/术语
3. 不包含具体回答细节（只需提及"已讨论X、Y、Z等概念"）
4. 控制在150字以内

对话历史:
{turns_text}

摘要:""""


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add ConversationTurn and ConversationState data models"
```

---

### Task 2: ConversationState Methods

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for ConversationState methods**

```python
# Add to tests/agent/test_conversation.py

def test_conversation_state_creation():
    """Test ConversationState model creation."""
    state = ConversationState(
        session_id="test123",
        created_at=datetime.now(),
        last_activity=datetime.now(),
    )
    assert state.session_id == "test123"
    assert state.turns == []
    assert state.summary == ""
    assert state.total_turns() == 0

def test_conversation_state_needs_summarization():
    """Test needs_summarization method."""
    state = ConversationState(
        session_id="test",
        created_at=datetime.now(),
        last_activity=datetime.now(),
    )
    # Add 7 turns
    for i in range(7):
        state.turns.append(ConversationTurn(
            query=f"query{i}",
            response=f"response{i}",
            timestamp=datetime.now(),
            mode="react",
        ))
    assert state.needs_summarization(6) is True
    assert state.needs_summarization(10) is False
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py -v`
Expected: PASS (methods already defined in previous task)

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_conversation.py
git commit -m "test: add tests for ConversationState methods"
```

---

### Task 3: ConversationManager Core Methods

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for ConversationManager initialization and basic methods**

```python
# Add to tests/agent/test_conversation.py

from unittest.mock import MagicMock
from profirag.agent.conversation import ConversationManager

class MockAgent:
    """Mock agent for testing."""
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            "response": f"Mock response for: {question}",
            "question": question,
            "mode": "react",
        }

def test_conversation_manager_init():
    """Test ConversationManager initialization."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(
        agent=mock_agent,
        llm=mock_llm,
        max_history_turns=6,
    )
    assert manager.max_history_turns == 6
    assert manager.state.turns == []
    assert len(manager.state.session_id) == 8

def test_conversation_manager_reset():
    """Test reset clears state."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.turns.append(ConversationTurn(
        query="test",
        response="result",
        timestamp=datetime.now(),
        mode="react",
    ))
    manager.reset()
    assert manager.state.turns == []
    assert manager.state.summary == ""

def test_conversation_manager_get_history():
    """Test get_history returns turns."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.turns.append(ConversationTurn(
        query="q1",
        response="r1",
        timestamp=datetime.now(),
        mode="react",
    ))
    history = manager.get_history()
    assert len(history) == 1
    assert history[0].query == "q1"

def test_conversation_manager_get_summary():
    """Test get_summary returns current summary."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.summary = "User asked about X and Y."
    assert manager.get_summary() == "User asked about X and Y."

def test_conversation_manager_export_import_state():
    """Test export and import state."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.summary = "test summary"
    exported = manager.export_state()
    assert exported["summary"] == "test summary"

    manager2 = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager2.import_state(exported)
    assert manager2.state.summary == "test summary"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py -v`
Expected: FAIL with "ImportError: cannot import name 'ConversationManager'"

- [ ] **Step 3: Add ConversationManager class with core methods**

```python
# Add to src/profirag/agent/conversation.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add ConversationManager with core methods"
```

---

### Task 4: Explicit Reference Detection

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for explicit reference detection**

```python
# Add to tests/agent/test_conversation.py

def test_detect_explicit_reference_based_on_above():
    """Test detection of '基于上面/上述/上文' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("基于上面的回答，继续解释") is True
    assert manager._detect_explicit_reference("基于上述内容") is True
    assert manager._detect_explicit_reference("基于上文提到的概念") is True

def test_detect_explicit_reference_according_to():
    """Test detection of '根据刚才/之前/上文' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("根据刚才的查询结果") is True
    assert manager._detect_explicit_reference("根据之前的讨论") is True
    assert manager._detect_explicit_reference("根据上文") is True

def test_detect_explicit_reference_continue():
    """Test detection of '继续讨论/说明/解释' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("继续讨论这个问题") is True
    assert manager._detect_explicit_reference("继续说明") is True
    assert manager._detect_explicit_reference("继续解释") is True

def test_detect_explicit_reference_that():
    """Test detection of '那个问题/文档/概念' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("那个问题怎么解决") is True
    assert manager._detect_explicit_reference("那个文档在哪里") is True
    assert manager._detect_explicit_reference("那个概念是什么") is True

def test_detect_explicit_reference_it():
    """Test detection of '它是指/是什么' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("它是指什么") is True
    assert manager._detect_explicit_reference("它是什么意思") is True
    assert manager._detect_explicit_reference("它怎么样") is True

def test_detect_explicit_reference_about_this():
    """Test detection of '关于这/那个' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("关于这个问题") is True
    assert manager._detect_explicit_reference("关于那个文档") is True
    assert manager._detect_explicit_reference("关于这些参数") is True

def test_detect_explicit_reference_further():
    """Test detection of '进一步' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("进一步说明") is True
    assert manager._detect_explicit_reference("请进一步解释") is True

def test_detect_explicit_reference_more():
    """Test detection of '更多/更详细' pattern."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("更多详细信息") is True
    assert manager._detect_explicit_reference("更详细的说明") is True
    assert manager._detect_explicit_reference("还有哪些功能") is True

def test_detect_no_explicit_reference():
    """Test queries without explicit reference."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    assert manager._detect_explicit_reference("什么是Qdrant？") is False
    assert manager._detect_explicit_reference("如何配置向量数据库") is False
    assert manager._detect_explicit_reference("介绍一下RAG系统") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py::test_detect -v`
Expected: FAIL with "AttributeError: ... has no attribute '_detect_explicit_reference'"

- [ ] **Step 3: Add _detect_explicit_reference method**

```python
# Add to ConversationManager class in src/profirag/agent/conversation.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py::test_detect -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add explicit reference detection with Chinese patterns"
```

---

### Task 5: Query Enrichment

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for query enrichment**

```python
# Add to tests/agent/test_conversation.py

def test_enrich_query_with_recent_turns():
    """Test enrichment with recent turns and summary."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.summary = "讨论了向量数据库配置"
    manager.state.turns.append(ConversationTurn(
        query="什么是Qdrant?",
        response="Qdrant是一个向量数据库",
        timestamp=datetime.now(),
        mode="react",
    ))

    result = manager._enrich_query(
        query="基于上面的回答，如何配置?",
        use_recent_turns=True,
    )
    assert "【上下文】" in result
    assert "用户问题：基于上面的回答，如何配置?" in result

def test_enrich_query_with_summary_only():
    """Test enrichment with summary only."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    manager.state.summary = "讨论了向量数据库配置"

    result = manager._enrich_query(
        query="继续说明",
        use_recent_turns=False,
    )
    assert "【相关背景】" in result
    assert "讨论了向量数据库配置" in result
    assert "用户问题：继续说明" in result

def test_enrich_query_no_context():
    """Test no enrichment when context empty."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)

    result = manager._enrich_query(
        query="什么是向量数据库?",
        use_recent_turns=False,
    )
    assert result == "什么是向量数据库?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py::test_enrich -v`
Expected: FAIL with "AttributeError: ... has no attribute '_enrich_query'"

- [ ] **Step 3: Add _enrich_query method**

```python
# Add to ConversationManager class in src/profirag/agent/conversation.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py::test_enrich -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add query enrichment with context formatting"
```

---

### Task 6: LLM Context Decision

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for LLM context decision**

```python
# Add to tests/agent/test_conversation.py

def test_should_inject_context_with_llm_needs():
    """Test LLM decides context needed."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = '{"needs_context": true, "reason": "问题涉及之前讨论的概念"}'

    manager = ConversationManager(agent=mock_agent, llm=mock_llm, enable_auto_context=True)
    manager.state.summary = "讨论了Qdrant配置"
    manager.state.turns.append(ConversationTurn(
        query="Qdrant如何配置?",
        response="Qdrant配置需要...",
        timestamp=datetime.now(),
        mode="react",
    ))

    needs = manager._should_inject_context_llm("它的主要参数是什么?")
    assert needs is True

def test_should_inject_context_with_llm_not_needed():
    """Test LLM decides context not needed."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = '{"needs_context": false, "reason": "新问题独立"}'

    manager = ConversationManager(agent=mock_agent, llm=mock_llm, enable_auto_context=True)
    manager.state.summary = "讨论了Qdrant配置"

    needs = manager._should_inject_context_llm("什么是PostgreSQL?")
    assert needs is False

def test_should_inject_context_llm_fallback():
    """Test fallback when LLM fails."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("LLM error")

    manager = ConversationManager(agent=mock_agent, llm=mock_llm, enable_auto_context=True)
    manager.state.summary = "讨论了Qdrant"

    # Should fallback to False on error
    needs = manager._should_inject_context_llm("新问题")
    assert needs is False

def test_should_inject_context_disabled():
    """Test auto context disabled."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()

    manager = ConversationManager(agent=mock_agent, llm=mock_llm, enable_auto_context=False)

    # Should return False when disabled
    needs = manager._should_inject_context_llm("任何问题")
    assert needs is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py::test_should_inject_context -v`
Expected: FAIL with "AttributeError: ... has no attribute '_should_inject_context_llm'"

- [ ] **Step 3: Add _should_inject_context_llm method**

```python
# Add to ConversationManager class in src/profirag/agent/conversation.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py::test_should_inject_context -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add LLM-based context decision with fallback"
```

---

### Task 7: History Summarization

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for summarization**

```python
# Add to tests/agent/test_conversation.py

def test_summarize_history():
    """Test history summarization with LLM."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = "用户询问了Qdrant配置、混合检索原理。讨论了向量数据库、BM25等概念。"

    manager = ConversationManager(agent=mock_agent, llm=mock_llm)

    turns = [
        ConversationTurn(query="Qdrant如何配置?", response="...", timestamp=datetime.now(), mode="react"),
        ConversationTurn(query="混合检索是什么?", response="...", timestamp=datetime.now(), mode="react"),
        ConversationTurn(query="BM25怎么用?", response="...", timestamp=datetime.now(), mode="react"),
    ]

    summary = manager._summarize_history(turns)
    assert "Qdrant" in summary or "配置" in summary

def test_summarize_history_empty():
    """Test summarization with empty turns."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()

    manager = ConversationManager(agent=mock_agent, llm=mock_llm)
    summary = manager._summarize_history([])
    assert summary == ""

def test_trigger_summarization():
    """Test triggering summarization after threshold."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = "用户询问了多个问题。"

    manager = ConversationManager(agent=mock_agent, llm=mock_llm, max_history_turns=3, keep_recent_turns=1)

    # Add turns exceeding threshold
    for i in range(5):
        manager.state.turns.append(ConversationTurn(
            query=f"query{i}",
            response=f"response{i}",
            timestamp=datetime.now(),
            mode="react",
        ))

    manager._maybe_summarize()

    # Should have summary
    assert manager.state.summary != ""
    # Should keep recent turns
    assert len(manager.state.turns) == manager.keep_recent_turns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py::test_summarize -v`
Expected: FAIL with "AttributeError: ... has no attribute '_summarize_history'"

- [ ] **Step 3: Add summarization methods**

```python
# Add to ConversationManager class in src/profirag/agent/conversation.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py::test_summarize -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: add history summarization with LLM"
```

---

### Task 8: Main Query Method

**Files:**
- Modify: `src/profirag/agent/conversation.py`
- Modify: `tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing tests for query method**

```python
# Add to tests/agent/test_conversation.py

def test_query_first_turn_no_enrichment():
    """Test first query has no enrichment."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)

    result = manager.query("什么是Qdrant?")
    assert result["response"] == "Mock response for: 什么是Qdrant?"
    assert result["enriched_query"] == "什么是Qdrant?"
    assert result["injected_context"] is False
    assert result["conversation_turns"] == 1

def test_query_with_explicit_reference():
    """Test query with explicit reference gets enriched."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)

    # Add first turn
    manager.state.turns.append(ConversationTurn(
        query="什么是Qdrant?",
        response="Qdrant是向量数据库",
        timestamp=datetime.now(),
        mode="react",
    ))
    manager.state.summary = "讨论了向量数据库"

    # Query with explicit reference
    result = manager.query("基于上面的回答，如何配置?")
    assert "【上下文】" in result["enriched_query"]
    assert result["injected_context"] is True
    assert result["reference_detected"] is True
    assert result["conversation_turns"] == 2

def test_query_without_reference_no_enrichment():
    """Test query without reference stays original when LLM decides not needed."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = '{"needs_context": false, "reason": "独立问题"}'
    manager = ConversationManager(agent=mock_agent, llm=mock_llm)

    # Add first turn
    manager.state.turns.append(ConversationTurn(
        query="什么是Qdrant?",
        response="Qdrant是向量数据库",
        timestamp=datetime.now(),
        mode="react",
    ))

    # Independent query
    result = manager.query("什么是PostgreSQL?")
    assert result["enriched_query"] == "什么是PostgreSQL?"
    assert result["injected_context"] is False

def test_query_triggers_summarization():
    """Test query triggers summarization at threshold."""
    mock_agent = MockAgent()
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = '{"needs_context": false}'  # For context decision
    manager = ConversationManager(agent=mock_agent, llm=mock_llm, max_history_turns=3)

    # Add turns up to threshold
    for i in range(3):
        manager.state.turns.append(ConversationTurn(
            query=f"q{i}",
            response=f"r{i}",
            timestamp=datetime.now(),
            mode="react",
        ))

    # This query should trigger summarization
    manager.query("新问题")

    # Check summarization happened
    assert manager.state.summary != ""
    assert len(manager.state.turns) == manager.keep_recent_turns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_conversation.py::test_query -v`
Expected: FAIL with various assertion errors

- [ ] **Step 3: Implement full query method**

```python
# Replace/add to ConversationManager class in src/profirag/agent/conversation.py

    def query(self, question: str, **agent_kwargs) -> Dict[str, Any]:
        """
        Process query with conversation context.

        Args:
            question: User query
            **agent_kwargs: Additional args for wrapped agent

        Returns:
            Result dict with response, enriched_query, conversation metadata
        """
        # Step 1: Check if first query (no history)
        is_first_query = len(self.state.turns) == 0

        # Step 2: Determine enrichment
        enrichment_result = QueryEnrichmentResult(
            original_query=question,
            enriched_query=question,
            injected_context=False,
            reference_detected=False,
            context_source="none",
        )

        if not is_first_query:
            # Detect explicit reference
            explicit_ref = self._detect_explicit_reference(question)

            if explicit_ref:
                # Enrich with recent turns + summary
                enrichment_result.enriched_query = self._enrich_query(question, use_recent_turns=True)
                enrichment_result.injected_context = True
                enrichment_result.reference_detected = True
                enrichment_result.context_source = "recent_turns"
            elif self.enable_auto_context:
                # Use LLM to decide
                needs_context = self._should_inject_context_llm(question)
                if needs_context:
                    enrichment_result.enriched_query = self._enrich_query(question, use_recent_turns=False)
                    enrichment_result.injected_context = True
                    enrichment_result.context_source = "summary"

        if self.verbose and enrichment_result.injected_context:
            print(f"🔍 Enriched query: {enrichment_result.context_source}")

        # Step 3: Pass to wrapped agent
        agent_result = self.agent.query(enrichment_result.enriched_query, **agent_kwargs)

        # Step 4: Create ConversationTurn
        mode = agent_result.get("mode", "unknown")
        tool_calls = agent_result.get("tool_calls", [])
        turn = ConversationTurn(
            query=question,  # Original query, not enriched
            response=agent_result.get("response", ""),
            timestamp=datetime.now(),
            tool_calls=tool_calls,
            mode=mode,
        )
        self.state.turns.append(turn)
        self.state.last_activity = datetime.now()

        # Step 5: Trigger summarization if needed
        self._maybe_summarize()

        # Step 6: Build result
        result = {
            "response": agent_result.get("response", ""),
            "enriched_query": enrichment_result.enriched_query,
            "original_query": enrichment_result.original_query,
            "injected_context": enrichment_result.injected_context,
            "reference_detected": enrichment_result.reference_detected,
            "context_source": enrichment_result.context_source,
            "conversation_turns": self.state.total_turns(),
            "session_id": self.state.session_id,
            "mode": mode,
        }

        # Include other agent result fields
        for key in ["sources", "source_nodes", "plan", "execution_result", "metadata"]:
            if key in agent_result:
                result[key] = agent_result[key]

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation.py::test_query -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/agent/conversation.py tests/agent/test_conversation.py
git commit -m "feat: implement ConversationManager.query with full flow"
```

---

### Task 9: Update Agent Module Exports

**Files:**
- Modify: `src/profirag/agent/__init__.py`

- [ ] **Step 1: Update __init__.py exports**

```python
# src/profirag/agent/__init__.py
"""Agent module for RAG system

Provides ReAct Agent, Plan Agent, and ConversationManager for intelligent question answering.
"""

from .tools import RAGTools, ToolResultFormatter
from .react_agent import RAGReActAgent, AgentFactory
from .plan_agent import (
    RAGPlanAgent,
    ExecutionPlan,
    PlanStep,
    PlanGenerator,
    PlanExecutor,
    PlanExecutionResult,
    PlanApproval,
    PlanComplexity,
    StepStatus,
)
from .conversation import (
    ConversationManager,
    ConversationTurn,
    ConversationState,
    QueryEnrichmentResult,
)

__all__ = [
    "RAGTools",
    "ToolResultFormatter",
    "RAGReActAgent",
    "RAGPlanAgent",
    "AgentFactory",
    "ExecutionPlan",
    "PlanStep",
    "PlanGenerator",
    "PlanExecutor",
    "PlanExecutionResult",
    "PlanApproval",
    "PlanComplexity",
    "StepStatus",
    "ConversationManager",
    "ConversationTurn",
    "ConversationState",
    "QueryEnrichmentResult",
]
```

- [ ] **Step 2: Run import test**

Run: `python -c "from profirag.agent import ConversationManager; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/profirag/agent/__init__.py
git commit -m "feat: export ConversationManager from agent module"
```

---

### Task 10: Add ConversationConfig to Settings

**Files:**
- Modify: `src/profirag/config/settings.py`
- Create: `tests/config/test_conversation_config.py`

- [ ] **Step 1: Write failing tests for ConversationConfig**

```python
# tests/config/test_conversation_config.py
"""Tests for conversation configuration."""

from profirag.config.settings import ConversationConfig, EnvSettings

def test_conversation_config_defaults():
    """Test default values."""
    config = ConversationConfig()
    assert config.max_history_turns == 6
    assert config.keep_recent_turns == 2
    assert config.auto_context is True

def test_conversation_config_custom():
    """Test custom values."""
    config = ConversationConfig(max_history_turns=10, keep_recent_turns=3)
    assert config.max_history_turns == 10
    assert config.keep_recent_turns == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/config/test_conversation_config.py -v`
Expected: FAIL with "ImportError: cannot import name 'ConversationConfig'"

- [ ] **Step 3: Add ConversationConfig to settings.py**

```python
# Add to src/profirag/config/settings.py after PlanAgentConfig

class ConversationConfig(BaseModel):
    """ConversationManager configuration."""
    max_history_turns: int = 6      # Turns before summarization
    keep_recent_turns: int = 2      # Turns kept verbatim after summarization
    auto_context: bool = True       # Enable LLM-based context decision
```

- [ ] **Step 4: Add environment variable support**

```python
# Add to EnvSettings class in src/profirag/config/settings.py

    # Conversation Configuration
    profirag_conversation_max_history: int = 6
    profirag_conversation_keep_recent: int = 2
    profirag_conversation_auto_context: bool = True
```

- [ ] **Step 5: Add to AgentConfig**

```python
# Add to AgentConfig class in src/profirag/config/settings.py

class AgentConfig(BaseModel):
    """Agent configuration for ReAct-based question answering."""
    enabled: bool = False
    mode: str = "react"
    max_iterations: int = 10
    verbose: bool = True
    markdown_base_path: Optional[str] = None
    plan_config: PlanAgentConfig = PlanAgentConfig()
    conversation_config: ConversationConfig = ConversationConfig()  # NEW
    tools: List[str] = [...]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/config/test_conversation_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/profirag/config/settings.py tests/config/test_conversation_config.py
git commit -m "feat: add ConversationConfig to settings"
```

---

### Task 11: Add AgentFactory Conversation Method

**Files:**
- Modify: `src/profirag/agent/react_agent.py`

- [ ] **Step 1: Add create_conversation_agent to AgentFactory**

```python
# Add to AgentFactory class in src/profirag/agent/react_agent.py

    @staticmethod
    def create_conversation_agent(
        agent_type: str,  # "react" or "plan"
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        max_history_turns: int = 6,
        keep_recent_turns: int = 2,
        enable_auto_context: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        """Create ConversationManager wrapping specified agent type.

        Args:
            agent_type: "react" or "plan"
            retriever: Retriever instance
            synthesizer: Synthesizer instance
            llm: LLM instance
            max_history_turns: Max turns before summarization
            keep_recent_turns: Turns kept verbatim
            enable_auto_context: Enable LLM context decision
            verbose: Print enrichment details
            **kwargs: Additional args for underlying agent

        Returns:
            ConversationManager instance
        """
        from .conversation import ConversationManager

        if agent_type == "plan":
            agent = AgentFactory.create_plan_agent(
                retriever=retriever,
                synthesizer=synthesizer,
                llm=llm,
                **kwargs
            )
        else:
            agent = AgentFactory.create_react_agent(
                retriever=retriever,
                synthesizer=synthesizer,
                llm=llm,
                **kwargs
            )

        return ConversationManager(
            agent=agent,
            llm=llm,
            max_history_turns=max_history_turns,
            keep_recent_turns=keep_recent_turns,
            enable_auto_context=enable_auto_context,
            verbose=verbose,
        )
```

- [ ] **Step 2: Run import test**

Run: `python -c "from profirag.agent import AgentFactory; cm = AgentFactory.create_conversation_agent; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/profirag/agent/react_agent.py
git commit -m "feat: add create_conversation_agent to AgentFactory"
```

---

### Task 12: Integration Tests

**Files:**
- Create: `tests/agent/test_conversation_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/agent/test_conversation_integration.py
"""Integration tests for ConversationManager with mock agents."""

from datetime import datetime
from unittest.mock import MagicMock
from profirag.agent import ConversationManager, AgentFactory
from profirag.agent.conversation import ConversationTurn


class MockReActAgent:
    """Mock ReActAgent for integration tests."""
    def __init__(self):
        self.call_count = 0

    def query(self, question: str, **kwargs) -> dict:
        self.call_count += 1
        return {
            "response": f"Answer for: {question[:50]}",
            "question": question,
            "mode": "react",
            "tool_calls": [{"tool": "vector_search", "input": {"query": question}}],
        }


def test_multi_turn_conversation_flow():
    """Test complete multi-turn conversation."""
    agent = MockReActAgent()
    llm = MagicMock()
    llm.complete.return_value.text = '{"needs_context": false}'

    manager = ConversationManager(agent=agent, llm=llm)

    # First query - no context
    r1 = manager.query("什么是向量数据库?")
    assert r1["injected_context"] is False
    assert r1["conversation_turns"] == 1

    # Second query - explicit reference
    r2 = manager.query("基于上面的回答，继续解释")
    assert r2["injected_context"] is True
    assert r2["reference_detected"] is True
    assert r2["conversation_turns"] == 2

    # Third query - independent (LLM decides no context)
    r3 = manager.query("什么是Python?")
    assert r3["injected_context"] is False
    assert r3["conversation_turns"] == 3

    # Check history
    history = manager.get_history()
    assert len(history) == 3


def test_conversation_reset():
    """Test conversation reset."""
    agent = MockReActAgent()
    llm = MagicMock()

    manager = ConversationManager(agent=agent, llm=llm)
    manager.query("query1")
    manager.query("query2")

    assert len(manager.get_history()) == 2

    manager.reset()
    assert len(manager.get_history()) == 0


def test_summarization_flow():
    """Test summarization at threshold."""
    agent = MockReActAgent()
    llm = MagicMock()
    llm.complete.return_value.text = "讨论了向量数据库、检索配置等概念。"  # Summary

    manager = ConversationManager(
        agent=agent,
        llm=llm,
        max_history_turns=4,
        keep_recent_turns=2,
    )

    # Add 5 queries to trigger summarization
    for i in range(5):
        manager.query(f"query{i}")

    # Check summarization happened
    assert manager.state.summary != ""
    assert len(manager.state.turns) == 2  # Only recent kept


def test_export_import_flow():
    """Test export and import preserves state."""
    agent = MockReActAgent()
    llm = MagicMock()

    manager = ConversationManager(agent=agent, llm=llm)
    manager.query("query1")
    manager.query("query2")
    manager.state.summary = "Test summary"

    exported = manager.export_state()

    # Create new manager and import
    manager2 = ConversationManager(agent=agent, llm=llm)
    manager2.import_state(exported)

    assert len(manager2.get_history()) == 2
    assert manager2.get_summary() == "Test summary"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/agent/test_conversation_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_conversation_integration.py
git commit -m "test: add integration tests for ConversationManager"
```

---

### Task 13: Web API Schemas

**Files:**
- Modify: `web/api/schemas.py`

- [ ] **Step 1: Add conversation schemas**

```python
# Add to web/api/schemas.py after ChatMode enum

class ConversationRequest(BaseModel):
    """Conversation context in request."""
    session_id: Optional[str] = Field(None, description="Session ID to continue")
    continue_session: bool = Field(False, description="Continue existing conversation")

class ConversationInfo(BaseModel):
    """Conversation info in response."""
    session_id: str
    turn_count: int
    injected_context: bool = False
    reference_detected: bool = False

# Modify ChatRequest to include conversation
class ChatRequest(BaseModel):
    """Request for RAG chat query."""
    query: str = Field(..., description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    mode: ChatMode = Field(ChatMode.PIPELINE, description="Query mode")
    env_file: str = Field(".env", description="Path to .env config file")
    conversation: Optional[ConversationRequest] = Field(None, description="Conversation context")

# Modify ChatResponse to include conversation info
class ChatResponse(BaseModel):
    """Response for RAG chat query."""
    query: str
    response: str
    source_nodes: List[SourceNode] = Field(default_factory=list)
    images: List[ImageInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation: Optional[ConversationInfo] = Field(None, description="Conversation info")
```

- [ ] **Step 2: Run schema test**

Run: `python -c "from schemas import ChatRequest, ConversationRequest; print('OK')"`
Expected: OK (from web/api directory)

- [ ] **Step 3: Commit**

```bash
git add web/api/schemas.py
git commit -m "feat: add conversation fields to ChatRequest/Response schemas"
```

---

### Task 14: Web API Services

**Files:**
- Modify: `web/api/services.py`

- [ ] **Step 1: Add session management to ChatService**

```python
# Add to ChatService class in web/api/services.py

    # Session storage (in-memory for now)
    active_sessions: Dict[str, Any] = {}

    @staticmethod
    def query_with_conversation(
        query_str: str,
        session_id: Optional[str],
        mode: str,
        top_k: int,
        env_file: str,
    ) -> Dict[str, Any]:
        """Execute query with conversation context.

        Args:
            query_str: User query
            session_id: Existing session ID (None for new session)
            mode: Agent mode (react/plan)
            top_k: Retrieval count
            env_file: Config file path

        Returns:
            Response with conversation info
        """
        config_path = PROJECT_ROOT / env_file
        config = load_config(str(config_path))
        pipeline = RAGPipeline(config)

        # Get or create conversation manager
        if session_id and session_id in ChatService.active_sessions:
            conv_manager = ChatService.active_sessions[session_id]
        else:
            conv_manager = AgentFactory.create_conversation_agent(
                agent_type=mode,
                retriever=pipeline._hybrid_retriever,
                synthesizer=pipeline._synthesizer,
                llm=pipeline._llm,
                max_history_turns=config.agent.conversation_config.max_history_turns,
                keep_recent_turns=config.agent.conversation_config.keep_recent_turns,
                enable_auto_context=config.agent.conversation_config.auto_context,
                verbose=config.agent.verbose,
                markdown_base_path=config.agent.markdown_base_path,
                pre_retrieval=pipeline._pre_retrieval,
                reranker=pipeline._reranker,
            )
            ChatService.active_sessions[conv_manager.state.session_id] = conv_manager

        # Execute query
        result = conv_manager.query(query_str)

        # Format response
        response = {
            "query": result.get("original_query", query_str),
            "response": result.get("response", ""),
            "source_nodes": ChatService._extract_sources(result),
            "images": [],
            "metadata": {"mode": mode},
            "conversation": {
                "session_id": conv_manager.state.session_id,
                "turn_count": result.get("conversation_turns", 1),
                "injected_context": result.get("injected_context", False),
                "reference_detected": result.get("reference_detected", False),
            },
        }
        return response

    @staticmethod
    def _extract_sources(result: Dict) -> List[Dict]:
        """Extract sources from result."""
        sources = []
        for src in result.get("sources", result.get("source_nodes", [])):
            if hasattr(src, "node"):
                sources.append({
                    "node_id": src.node.node_id,
                    "text": src.node.text[:300],
                    "score": src.score,
                    "source_file": src.node.metadata.get("source_file"),
                })
            else:
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300],
                    "score": src.get("score", 0.0),
                    "source_file": src.get("source_file"),
                })
        return sources

    @staticmethod
    def clear_session(session_id: str) -> bool:
        """Clear conversation session."""
        if session_id in ChatService.active_sessions:
            del ChatService.active_sessions[session_id]
            return True
        return False

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """Get session info."""
        if session_id in ChatService.active_sessions:
            conv_manager = ChatService.active_sessions[session_id]
            return {
                "session_id": session_id,
                "turn_count": conv_manager.state.total_turns(),
                "summary": conv_manager.state.summary,
                "created_at": conv_manager.state.created_at.isoformat(),
            }
        return None
```

- [ ] **Step 2: Modify existing query method to support conversation**

```python
# Replace ChatService.query method in web/api/services.py

    @staticmethod
    def query(
        query_str: str,
        top_k: int = 10,
        mode: str = "pipeline",
        env_file: str = ".env",
        conversation: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute RAG query and return response with images.

        Args:
            query_str: User query string
            top_k: Number of results to retrieve
            mode: Query mode (pipeline, agent, plan)
            env_file: Path to environment config file
            conversation: Optional conversation context dict

        Returns:
            Dictionary with response, sources, and images
        """
        # Handle conversation mode
        if conversation and mode in ("agent", "plan", "react"):
            return ChatService.query_with_conversation(
                query_str=query_str,
                session_id=conversation.get("session_id"),
                mode=mode,
                top_k=top_k,
                env_file=env_file,
            )

        # Load configuration
        config_path = PROJECT_ROOT / env_file
        config = load_config(str(config_path))

        # Initialize pipeline
        pipeline = RAGPipeline(config)

        # Route based on mode (existing behavior)
        if mode == "plan":
            result = pipeline.query_with_agent(query_str, mode="plan", auto_approve=True)
            return ChatService._format_agent_response(result)
        elif mode == "agent":
            result = pipeline.query_with_agent(query_str, mode="agent")
            return ChatService._format_agent_response(result)
        else:
            # Pipeline mode - default behavior
            result = pipeline.query_with_images(query_str, top_k=top_k, include_images=True)
            result["query"] = query_str
            return result
```

- [ ] **Step 3: Commit**

```bash
git add web/api/services.py
git commit -m "feat: add conversation support to ChatService"
```

---

### Task 15: Web API Routes

**Files:**
- Modify: `web/api/routes/chat.py`

- [ ] **Step 1: Add session endpoints**

```python
# web/api/routes/chat.py
"""Chat/Q&A endpoints for RAG queries."""

from fastapi import APIRouter, HTTPException

import schemas
import services

router = APIRouter(prefix="/chat", tags=["Chat/Q&A"])


@router.post("/query", response_model=schemas.ChatResponse, summary="Execute RAG query")
async def query(request: schemas.ChatRequest):
    """Execute RAG query and return response with source references."""
    try:
        conversation_dict = None
        if request.conversation:
            conversation_dict = {
                "session_id": request.conversation.session_id,
                "continue": request.conversation.continue_session,
            }

        result = services.ChatService.query(
            query_str=request.query,
            top_k=request.top_k,
            mode=request.mode.value,
            env_file=request.env_file,
            conversation=conversation_dict,
        )
        return schemas.ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", summary="Create new conversation session")
async def create_session():
    """Create a new conversation session."""
    session_id = services.ChatService._create_session_id()
    return {"session_id": session_id}


@router.get("/session/{session_id}", summary="Get session info")
async def get_session(session_id: str):
    """Get conversation session info."""
    session = services.ChatService.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}", summary="Clear session")
async def clear_session(session_id: str):
    """Clear conversation session."""
    success = services.ChatService.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session_id": session_id}
```

- [ ] **Step 2: Add _create_session_id helper to ChatService**

```python
# Add to ChatService in web/api/services.py

    @staticmethod
    def _create_session_id() -> str:
        """Generate new session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
```

- [ ] **Step 3: Commit**

```bash
git add web/api/routes/chat.py web/api/services.py
git commit -m "feat: add session management endpoints to chat API"
```

---

### Task 16: Update RAGPipeline

**Files:**
- Modify: `src/profirag/pipeline/rag_pipeline.py`

- [ ] **Step 1: Add conversation methods to RAGPipeline**

```python
# Add to RAGPipeline class in src/profirag/pipeline/rag_pipeline.py

from ..agent import ConversationManager

    def create_conversation_manager(
        self,
        mode: str = "react",
        max_history_turns: int = 6,
    ) -> ConversationManager:
        """Create ConversationManager with pipeline's components.

        Args:
            mode: Agent mode ("react" or "plan")
            max_history_turns: Max turns before summarization

        Returns:
            ConversationManager instance
        """
        return AgentFactory.create_conversation_agent(
            agent_type=mode,
            retriever=self._hybrid_retriever,
            synthesizer=self._synthesizer,
            llm=self._llm,
            max_history_turns=max_history_turns,
            keep_recent_turns=self._agent_config.conversation_config.keep_recent_turns,
            enable_auto_context=self._agent_config.conversation_config.auto_context,
            verbose=self._agent_config.verbose,
            markdown_base_path=self._agent_config.markdown_base_path,
            pre_retrieval=self._pre_retrieval,
            reranker=self._reranker,
        )

    def query_with_conversation(
        self,
        question: str,
        conversation_manager: ConversationManager,
        **kwargs
    ) -> Dict[str, Any]:
        """Query using conversation manager.

        Args:
            question: User question
            conversation_manager: ConversationManager instance
            **kwargs: Additional arguments

        Returns:
            Query result with conversation metadata
        """
        return conversation_manager.query(question, **kwargs)
```

- [ ] **Step 2: Run import test**

Run: `python -c "from profirag.pipeline import RAGPipeline; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/profirag/pipeline/rag_pipeline.py
git commit -m "feat: add create_conversation_manager to RAGPipeline"
```

---

### Task 17: Run All Tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/agent/ tests/config/test_conversation_config.py -v`
Expected: All PASS

- [ ] **Step 2: Verify imports**

Run: `python -c "from profirag.agent import ConversationManager, ConversationTurn, ConversationState; print('All imports OK')"`
Expected: All imports OK

- [ ] **Step 3: Final commit with all changes**

```bash
git status
git add -A
git commit -m "feat: complete multi-turn conversation support

- ConversationManager wraps any Agent for session-based memory
- Hybrid context linking (explicit patterns + LLM decision)
- History summarization with configurable threshold
- Web API session management endpoints
- Full backward compatibility

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Spec Coverage Checklist

| Spec Section | Task |
|--------------|------|
| ConversationTurn model | Task 1 |
| ConversationState model | Task 1, 2 |
| QueryEnrichmentResult model | Task 1 |
| Explicit reference patterns | Task 4 |
| LLM context decision | Task 6 |
| Query enrichment format | Task 5 |
| History summarization | Task 7 |
| ConversationManager.query flow | Task 8 |
| ConversationManager methods (reset, get_history, etc.) | Task 3 |
| AgentFactory.create_conversation_agent | Task 11 |
| ConversationConfig | Task 10 |
| RAGPipeline integration | Task 16 |
| Web API schemas | Task 13 |
| Web API services | Task 14 |
| Web API routes/session endpoints | Task 15 |
| Unit tests | Tasks 1-8 |
| Integration tests | Task 12 |