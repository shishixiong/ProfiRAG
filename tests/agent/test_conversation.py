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