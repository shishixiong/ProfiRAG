"""Tests for ConversationManager and related models."""

from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock

from profirag.agent.conversation import (
    ConversationTurn,
    ConversationState,
    ConversationManager,
)


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