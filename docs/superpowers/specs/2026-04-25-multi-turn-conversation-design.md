# Multi-Turn Conversation Support Design

**Feature:** Add session-based conversation memory to Agent system
**Date:** 2026-04-25
**Status:** Approved

## Overview

Add multi-turn conversation support for RAG Agents, enabling users to ask follow-up questions within a session. The system maintains conversation history with auto-summarization, detects when queries need previous context, and enriches queries appropriately before passing to the wrapped agent.

## Requirements

### User Requirements
- Support follow-up questions within a session
- Maintain full conversation history with summarization for efficiency
- Hybrid context linking: LLM auto-detects + explicit reference patterns
- Fresh retrieval for each query (no document reuse)
- Work with both ReActAgent and PlanAgent

### Technical Requirements
- Backward compatible - no changes to existing Agent classes
- Wrapper pattern: ConversationManager wraps any Agent
- History summarization triggered at configurable threshold
- Lightweight LLM prompt for context decision
- Export/import state for debugging and potential persistence

## Architecture

The `ConversationManager` is a stateful wrapper between user and Agent. It maintains conversation state and enriches queries with relevant context.

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationManager                       │
│                                                              │
│  State:                                                      │
│  ├── state: ConversationState                                │
│  │   ├── turns: List[ConversationTurn]                       │
│  │   ├── summary: str (LLM-generated compressed history)     │
│  │   └── session_id: str                                     │
│  └── agent: RAGReActAgent | RAGPlanAgent                     │
│                                                              │
│  Methods (internal helpers):                                 │
│  ├── _detect_reference(query) → bool                         │
│  ├── _enrich_query(query, context) → enriched_query          │
│  ├── _should_inject_context(query) → bool                    │
│  ├── _summarize(turns) → str                                 │
│  └── query(question, **kwargs) → Dict                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Wrapped Agent (RAGReActAgent | RAGPlanAgent)       │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Retrieval    │ Optimization │ Generation               │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** The wrapped agent remains unchanged. ConversationManager handles all state management externally.

**Note:** QueryProcessor and HistorySummarizer are internal helper methods, not separate classes. This keeps the implementation simple and avoids over-engineering.

## Data Models

### ConversationTurn

```python
class ConversationTurn(BaseModel):
    """Single conversation exchange"""
    query: str
    response: str
    timestamp: datetime
    tool_calls: List[Dict] = []  # For debugging/analysis
    mode: str  # "react" or "plan"
```

### ConversationState

```python
class ConversationState(BaseModel):
    """Session conversation state"""
    session_id: str
    turns: List[ConversationTurn] = []
    summary: str = ""
    created_at: datetime
    last_activity: datetime

    def total_turns(self) -> int:
        return len(self.turns)

    def needs_summarization(self, threshold: int) -> bool:
        return len(self.turns) > threshold
```

### QueryEnrichmentResult

```python
class QueryEnrichmentResult(BaseModel):
    """Result of query processing"""
    original_query: str
    enriched_query: str
    injected_context: bool
    reference_detected: bool
    context_source: str  # "summary" | "recent_turns" | "none"
```

## Query Processing (Hybrid Context Linking)

Two mechanisms to decide if context should be injected:

### 1. Explicit Reference Detection (Rule-based)

Pattern matching for common Chinese reference phrases:

```python
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
```

### 2. LLM-based Context Decision (Semantic)

For queries without explicit patterns, use a lightweight LLM prompt:

```python
CONTEXT_DECISION_PROMPT = """判断以下新问题是否需要参考之前的对话历史。

对话摘要: {summary}
最近问答: {last_turn}

新问题: {query}

判断标准:
- 问题中提到之前讨论的概念/术语 → 需要
- 问题是对之前回答的追问 → 需要
- 问题完全独立、新话题 → 不需要

输出JSON: {{\"needs_context\": true/false, \"reason\": \"简短说明\"}}"""
```

### 3. Enrichment Strategy

| Scenario | Context Source | Enrichment Format |
|----------|---------------|-------------------|
| Explicit reference detected | Recent 2 turns + summary | `【上下文】{context}\n\n用户问题：{query}` |
| LLM decides needs context | Summary only | `【相关背景】{summary}\n\n用户问题：{query}` |
| No context needed | None | Original query unchanged |

## History Summarization

### Trigger Condition

Summarization triggered when `turns.length > max_history_turns` (default 6).

### Summarization Strategy

- Keep last `keep_recent_turns` (default 2) verbatim
- Compress older turns into single summary string
- Uses LLM for summarization

### Summarization Prompt

```python
SUMMARIZATION_PROMPT = """请将以下对话历史压缩为简洁的摘要，保留关键信息。

要求:
1. 保留用户询问的主要问题（列举）
2. 保留讨论的关键概念/术语
3. 不包含具体回答细节（只需提及"已讨论X、Y、Z等概念"）
4. 控制在150字以内

对话历史:
{turns_text}

摘要:"""
```

### Summary Format Example

```
用户询问了如何配置Qdrant存储、HybridRetriever的工作原理、以及重排序器的选择。
讨论了混合检索、BM25、向量相似度、Cohere/DashScope重排序等概念。
```

## ConversationManager Class

```python
class ConversationManager:
    """Stateful wrapper for multi-turn conversations with any Agent"""

    def __init__(
        self,
        agent: Union[RAGReActAgent, RAGPlanAgent],
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
        )

    def query(self, question: str, **agent_kwargs) -> Dict[str, Any]:
        """
        Process query with conversation context.

        Args:
            question: User query
            **agent_kwargs: Additional args for wrapped agent (e.g., auto_approve for PlanAgent)

        Returns:
            Result dict with:
            - response: Agent response
            - enriched_query: Query after enrichment (or original if none)
            - injected_context: Whether context was injected
            - conversation_turns: Current turn count
            - mode: Agent mode ("react" or "plan")
            - ... (all other agent result fields)
        """

    def reset(self) -> None:
        """Clear conversation state for new session"""

    def get_history(self) -> List[ConversationTurn]:
        """Return full conversation history"""

    def get_summary(self) -> str:
        """Return current conversation summary"""

    def export_state(self) -> Dict:
        """Export state for debugging/testing"""

    def import_state(self, state_dict: Dict) -> None:
        """Import previous state for testing/debugging (not for cross-session persistence)"""
```

## Implementation Flow

### query() Method Flow

```
query(question)
    │
    ├─► 1. Detect explicit reference (regex patterns)
    │       │
    │       ├─► Match found? → enrich with recent turns + summary
    │       │
    │       └─► No match? ─► 2. LLM context decision (if enable_auto_context)
    │                           │
    │                           ├─► needs_context=true → enrich with summary
    │                           │
    │                           └─► needs_context=false → original query
    │
    ├─► 3. Pass enriched query to wrapped agent
    │       │
    │       └─► agent.query(enriched_query, **kwargs)
    │
    ├─► 4. Create ConversationTurn from result
    │       │
    │       └─► Append to state.turns
    │
    ├─► 5. Check summarization threshold
    │       │
    │       ├─► Exceeded? → summarize older turns, update state.summary
    │       │
    │       └─► Not exceeded? → skip
    │
    └─► 6. Return result + conversation metadata
```

## API Integration

### AgentFactory Extension

Add conversation-aware factory methods:

```python
class AgentFactory:
    # ... existing methods ...

    @staticmethod
    def create_conversation_agent(
        agent_type: str,  # "react" or "plan"
        retriever: Any,
        synthesizer: Any,
        llm: Any,
        max_history_turns: int = 6,
        **kwargs
    ) -> ConversationManager:
        """
        Create ConversationManager wrapping specified agent type.

        Args:
            agent_type: "react" or "plan"
            ... other args passed to underlying agent factory
        """
```

### RAGPipeline Integration

Add method for conversation mode:

```python
class RAGPipeline:
    # ... existing methods ...

    def create_conversation_manager(
        self,
        mode: str = "react",
        max_history_turns: int = 6,
    ) -> ConversationManager:
        """Create ConversationManager with pipeline's configured components"""

    def query_with_conversation(
        self,
        question: str,
        conversation_manager: ConversationManager,
        **kwargs
    ) -> Dict[str, Any]:
        """Query using conversation manager"""
```

## Web API Changes

### Chat Endpoint Enhancement

Request:
```json
{
  "query": "用户问题",
  "mode": "agent",
  "conversation": {
    "session_id": "abc123",
    "continue": true
  }
}
```

Response:
```json
{
  "query": "原始问题",
  "enriched_query": "【上下文】...用户问题：...",
  "response": "回答内容",
  "conversation": {
    "session_id": "abc123",
    "turn_count": 3,
    "injected_context": true
  },
  "source_nodes": [...],
  "metadata": {...}
}
```

### Session Management Endpoints

```
POST /api/chat/session        → Create new session, returns session_id
GET  /api/chat/session/{id}   → Get session state/history
DELETE /api/chat/session/{id} → Clear session
```

## File Structure

```
src/profirag/agent/
├── __init__.py              # Add ConversationManager exports
├── conversation.py          # NEW: ConversationManager, ConversationTurn, ConversationState
├── react_agent.py           # No changes
├── plan_agent.py            # No changes
├── tools.py                 # No changes
```

## Testing Strategy

### Unit Tests

1. `ConversationTurn` model validation
2. `ConversationState` summarization threshold logic
3. Explicit pattern detection (regex tests)
4. Query enrichment formatting
5. Summary generation (mock LLM)

### Integration Tests

1. Full query flow with mock agent
2. Multiple turns leading to summarization
3. Context injection behavior
4. Reset and state management
5. Both ReActAgent and PlanAgent wrapping

### Edge Cases

1. Empty history → no enrichment
2. First query in session → no enrichment
3. Explicit reference with empty summary
4. LLM context decision timeout → fallback to original query
5. Very long queries → truncation before enrichment

## Configuration

New environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFIRAG_CONVERSATION_MAX_HISTORY` | 6 | Max turns before summarization |
| `PROFIRAG_CONVERSATION_KEEP_RECENT` | 2 | Turns kept verbatim |
| `PROFIRAG_CONVERSATION_AUTO_CONTEXT` | true | Enable LLM context decision |

## Backward Compatibility

- No changes to existing Agent classes
- ConversationManager is opt-in
- Existing `query()` methods unchanged
- Web API backward compatible (conversation field optional)