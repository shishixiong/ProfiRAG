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