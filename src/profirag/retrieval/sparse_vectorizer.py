"""Sparse vector calculator using jieba tokenization and TF-IDF weighting.

Used for Qdrant native BM25 hybrid retrieval.
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
import jieba


class SparseVectorizer:
    """jieba + TF-IDF based sparse vector calculator.

    Tokenizes text with jieba, computes TF-IDF weights, and produces
    sparse vectors compatible with Qdrant's sparse_vectors storage.
    """

    # Special payload key for storing IDF values
    IDF_PAYLOAD_KEY = "_sparse_idf"

    def __init__(
        self,
        tokenizer: str = "jieba",
        language: str = "zh",
        min_doc_freq: int = 1,
        max_vocab_size: int = 50000,
    ):
        """Initialize SparseVectorizer.

        Args:
            tokenizer: Tokenizer type ("jieba" for Chinese, "space" for simple split)
            language: Language code
            min_doc_freq: Minimum document frequency for IDF (filter rare words)
            max_vocab_size: Maximum vocabulary size (for hash-based mapping)
        """
        self.tokenizer = tokenizer
        self.language = language
        self.min_doc_freq = min_doc_freq
        self.max_vocab_size = max_vocab_size

        # Global IDF values (computed from corpus)
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

        # Cumulative document frequency for each token (for incremental updates)
        self._doc_freq: Counter = Counter()

        # Track seen document IDs to avoid double-counting on re-ingestion
        self._seen_doc_ids: set = set()

        # Token to vocabulary index mapping
        self._token_to_idx: Dict[str, int] = {}
        self._idx_to_token: Dict[int, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.tokenizer == "jieba":
            return list(jieba.cut(text))
        else:
            return text.split()

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalize tokens (lowercase, filter short)."""
        return [t.lower() for t in tokens if len(t) > 1]

    def fit(self, texts: List[str]) -> "SparseVectorizer":
        """Build vocabulary and IDF from corpus (incremental).

        Multiple calls accumulate document frequencies.

        Args:
            texts: List of text documents

        Returns:
            self
        """
        batch_doc_count = len(texts)
        if batch_doc_count == 0:
            return self

        # Accumulate document frequencies
        for text in texts:
            tokens = self._normalize_tokens(self._tokenize(text))
            unique_tokens = set(tokens)
            self._doc_freq.update(unique_tokens)

        self._doc_count += batch_doc_count

        # Rebuild vocabulary and recompute IDF
        # IDF = log((N + 1) / (df + 1)) + 1  (smoothed)
        vocab_idx = len(self._token_to_idx)
        for token, df in self._doc_freq.items():
            if df >= self.min_doc_freq and token not in self._token_to_idx:
                if vocab_idx >= self.max_vocab_size:
                    break
                self._token_to_idx[token] = vocab_idx
                self._idx_to_token[vocab_idx] = token
                vocab_idx += 1

        # Recompute IDF for all tokens
        for token in self._token_to_idx:
            df = self._doc_freq.get(token, 0)
            # Smoothed IDF
            self._idf[token] = math.log((self._doc_count + 1) / (df + 1)) + 1

        return self

    def fit_nodes(self, nodes: List) -> "SparseVectorizer":
        """Build vocabulary and IDF from LlamaIndex nodes.

        Args:
            nodes: List of TextNode objects

        Returns:
            self
        """
        texts = []
        new_ids = []
        for node in nodes:
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            if node_id and node_id in self._seen_doc_ids:
                continue  # Skip already-seen documents
            if node_id:
                self._seen_doc_ids.add(node_id)

            text = getattr(node, "text", "") or ""
            metadata = getattr(node, "metadata", {}) or {}
            # Also include combined text from metadata if available
            combined = text
            if isinstance(metadata, dict):
                combined = metadata.get("combined_text", text)
            texts.append(combined)
            new_ids.append(node_id)

        if not texts:
            return self  # No new documents

        return self.fit(texts)

    def compute_sparse_vector(
        self,
        text: str,
        with_idf: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """Compute sparse vector for a text.

        Args:
            text: Input text
            with_idf: If True, weight by IDF (use for queries).
                      If False, use raw TF (use for stored documents).

        Returns:
            Tuple of (indices, values) for SparseVector
        """
        tokens = self._normalize_tokens(self._tokenize(text))
        token_counts = Counter(tokens)

        indices: List[int] = []
        values: List[float] = []

        for token, count in token_counts.items():
            if token not in self._token_to_idx:
                continue

            idx = self._token_to_idx[token]
            tf = count  # term frequency

            if with_idf:
                # TF-IDF weight
                idf = self._idf.get(token, 1.0)
                weight = tf * idf
            else:
                # Just TF (for document storage)
                weight = tf

            indices.append(idx)
            values.append(float(weight))

        return indices, values

    def compute_query_sparse_vector(self, query: str) -> Tuple[List[int], List[float]]:
        """Compute sparse vector for a query (with IDF weighting).

        Args:
            query: Query text

        Returns:
            Tuple of (indices, values) for SparseVector
        """
        return self.compute_sparse_vector(query, with_idf=True)

    def get_idf_payload(self) -> Dict[str, float]:
        """Get IDF values for storage in Qdrant payload.

        Returns:
            Dict mapping tokens to IDF values
        """
        return {
            "idf": self._idf,
            "token_to_idx": self._token_to_idx,
            "doc_count": self._doc_count,
            "doc_freq": dict(self._doc_freq),
            "seen_doc_ids": list(self._seen_doc_ids),
        }

    def load_idf_from_payload(self, payload: Dict) -> None:
        """Load IDF values from stored payload.

        Args:
            payload: Payload dict with IDF data
        """
        self._idf = payload.get("idf", {})
        self._token_to_idx = payload.get("token_to_idx", {})
        self._idx_to_token = {v: k for k, v in self._token_to_idx.items()}
        self._doc_count = payload.get("doc_count", 0)
        self._doc_freq = Counter(payload.get("doc_freq", {}))
        self._seen_doc_ids = set(payload.get("seen_doc_ids", []))

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._token_to_idx)

    @property
    def has_idf(self) -> bool:
        """Check if IDF has been computed."""
        return self._doc_count > 0 and len(self._idf) > 0
