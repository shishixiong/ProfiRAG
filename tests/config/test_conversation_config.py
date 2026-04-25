"""Tests for conversation configuration."""

from profirag.config.settings import ConversationConfig

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