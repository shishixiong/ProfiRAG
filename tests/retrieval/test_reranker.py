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


# DashScopeReranker tests

from profirag.retrieval.reranker import DashScopeReranker


def test_dashscope_reranker_init():
    """Test DashScopeReranker initialization."""
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        model="rerank-v1",
        top_n=5
    )
    assert reranker.api_key == "test-key"
    assert reranker.base_url == "https://dashscope.aliyuncs.com"
    assert reranker.model == "rerank-v1"
    assert reranker.top_n == 5


def test_dashscope_reranker_requires_api_key():
    """Test DashScopeReranker raises error without api_key."""
    with pytest.raises(ValueError, match="api_key is required"):
        DashScopeReranker(api_key=None, base_url="https://dashscope.aliyuncs.com")


def test_dashscope_reranker_requires_base_url():
    """Test DashScopeReranker raises error without base_url."""
    with pytest.raises(ValueError, match="base_url is required"):
        DashScopeReranker(api_key="test-key", base_url=None)


@patch("httpx.post")
def test_dashscope_reranker_rerank(mock_post):
    """Test DashScopeReranker rerank method with wrapped response."""
    # Mock DashScope response format
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.68}
            ]
        },
        "request_id": "test-req-id"
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
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
    assert result[0].score == 0.92


@patch("httpx.post")
def test_dashscope_reranker_request_format(mock_post):
    """Test DashScopeReranker sends correct request format."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {"results": []},
        "request_id": "test"
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        model="rerank-v1",
        top_n=5
    )

    nodes = [NodeWithScore(node=TextNode(text="doc"), score=0.5)]
    reranker.rerank("query", nodes)

    # Verify request format
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    # DashScope uses nested input structure
    assert "input" in payload
    assert payload["input"]["query"] == "query"
    assert payload["input"]["documents"] == ["doc"]
    assert payload["input"]["top_n"] == 5


# CrossEncoderReranker tests

from profirag.retrieval.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_extends_base():
    """Test CrossEncoderReranker extends BaseReranker."""
    assert issubclass(CrossEncoderReranker, BaseReranker)


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_rerank_method(mock_ce):
    """Test CrossEncoderReranker has rerank method from BaseReranker."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.5]
    mock_ce.return_value = mock_model

    reranker = CrossEncoderReranker(model="test-model", top_n=2)
    nodes = [
        NodeWithScore(node=TextNode(text="doc1"), score=0.5),
        NodeWithScore(node=TextNode(text="doc2"), score=0.6)
    ]

    result = reranker.rerank("query", nodes)

    assert hasattr(reranker, 'rerank')
    assert len(result) == 2


# Reranker factory tests

from profirag.config.settings import RerankingConfig
from profirag.retrieval.reranker import Reranker, CohereReranker, DashScopeReranker, CrossEncoderReranker


def test_reranker_factory_local():
    """Test Reranker factory creates local reranker."""
    config = RerankingConfig(provider="local", model="test-model", top_n=5)
    reranker = Reranker(config)
    assert isinstance(reranker._impl, CrossEncoderReranker)


def test_reranker_factory_cohere():
    """Test Reranker factory creates Cohere reranker."""
    config = RerankingConfig(
        provider="cohere",
        api_key="test-key",
        base_url="https://api.cohere.ai",
        top_n=5
    )
    reranker = Reranker(config)
    assert isinstance(reranker._impl, CohereReranker)


def test_reranker_factory_dashscope():
    """Test Reranker factory creates DashScope reranker."""
    config = RerankingConfig(
        provider="dashscope",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        top_n=5
    )
    reranker = Reranker(config)
    assert isinstance(reranker._impl, DashScopeReranker)


def test_reranker_factory_disabled():
    """Test Reranker with enabled=False."""
    config = RerankingConfig(enabled=False, provider="local")
    reranker = Reranker(config)
    assert reranker.enabled == False


def test_reranker_factory_cohere_missing_key():
    """Test Reranker raises error for Cohere without api_key."""
    config = RerankingConfig(provider="cohere", base_url="https://api.cohere.ai")
    with pytest.raises(ValueError):
        Reranker(config)