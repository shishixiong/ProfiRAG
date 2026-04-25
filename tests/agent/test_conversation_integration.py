"""Integration tests for ConversationManager with mock agents."""

from unittest.mock import MagicMock

from profirag.agent import ConversationManager


class MockReActAgent:
    """Mock ReActAgent for integration tests."""

    def query(self, question: str, **kwargs) -> dict:
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

    # Return JSON for context decision calls, plain text for summarization
    def mock_complete(prompt):
        response = MagicMock()
        if "needs_context" in str(prompt):
            response.text = '{"needs_context": false}'
        else:
            response.text = "讨论了向量数据库、检索配置等概念。"
        return response

    llm.complete.side_effect = mock_complete

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
