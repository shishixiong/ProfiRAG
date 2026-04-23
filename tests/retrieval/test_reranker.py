"""Tests for reranker implementations"""

from profirag.retrieval.reranker import BaseReranker


def test_base_reranker_is_abstract():
    """Test BaseReranker cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        BaseReranker(top_n=5)


def test_base_reranker_has_rerank_method():
    """Test BaseReranker defines rerank abstract method."""
    from abc import ABC
    assert hasattr(BaseReranker, 'rerank')
    # Check it's abstract
    assert BaseReranker.__bases__[0] == ABC