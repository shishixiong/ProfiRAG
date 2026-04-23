"""Tests for reranker implementations"""

import pytest
from unittest.mock import patch, MagicMock
from profirag.retrieval.reranker import BaseReranker
from llama_index.core.schema import NodeWithScore, TextNode


def test_base_reranker_is_abstract():
    """Test BaseReranker cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseReranker(top_n=5)


def test_base_reranker_has_rerank_method():
    """Test BaseReranker defines rerank abstract method."""
    from abc import ABC
    assert hasattr(BaseReranker, 'rerank')
    # Check it's abstract
    assert BaseReranker.__bases__[0] == ABC


def test_cohere_reranker_init():
    """Test CohereReranker initialization."""
    from profirag.retrieval.reranker import CohereReranker
    reranker = CohereReranker(
        api_key="test-key",
        base_url="https://api.cohere.ai",
        model="rerank-v1",
        top_n=5
    )
    assert reranker.api_key == "test-key"
    assert reranker.base_url == "https://api.cohere.ai"
    assert reranker.model == "rerank-v1"
    assert reranker.top_n == 5


def test_cohere_reranker_requires_api_key():
    """Test CohereReranker raises error without api_key."""
    from profirag.retrieval.reranker import CohereReranker
    with pytest.raises(ValueError, match="api_key is required"):
        CohereReranker(api_key=None, base_url="https://api.cohere.ai")


def test_cohere_reranker_requires_base_url():
    """Test CohereReranker raises error without base_url."""
    from profirag.retrieval.reranker import CohereReranker
    with pytest.raises(ValueError, match="base_url is required"):
        CohereReranker(api_key="test-key", base_url=None)


@patch("httpx.post")
def test_cohere_reranker_rerank(mock_post):
    """Test CohereReranker rerank method."""
    from profirag.retrieval.reranker import CohereReranker
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.75}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = CohereReranker(
        api_key="test-key",
        base_url="https://api.cohere.ai",
        model="rerank-v1",
        top_n=2
    )

    nodes = [
        NodeWithScore(node=TextNode(text="Document 0"), score=0.5),
        NodeWithScore(node=TextNode(text="Document 1"), score=0.6)
    ]

    result = reranker.rerank("test query", nodes)

    assert len(result) == 2
    assert result[0].node.text == "Document 1"
    assert result[0].score == 0.95
    assert result[1].node.text == "Document 0"
    assert result[1].score == 0.75