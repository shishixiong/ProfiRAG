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