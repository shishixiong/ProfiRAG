"""Evaluation dataset definitions"""

import random
import re
from pathlib import Path
from typing import List, Optional, Any
import json

from pydantic import BaseModel


class EvalItem(BaseModel):
    """Single evaluation item.

    Contains a query, expected relevant document IDs, and optionally
    expected texts and reference answer.

    Attributes:
        query: Query string to evaluate
        expected_ids: List of relevant document/node IDs that should be retrieved
        expected_texts: Optional list of expected text content
        reference_answer: Optional reference answer for correctness evaluation
    """

    query: str
    expected_ids: List[str]
    expected_texts: Optional[List[str]] = None
    reference_answer: Optional[str] = None


class EvalDataset(BaseModel):
    """Evaluation dataset containing multiple evaluation items.

    Attributes:
        items: List of EvalItem objects
    """

    items: List[EvalItem]

    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        """Load dataset from JSON file.

        JSON format:
        {
            "items": [
                {
                    "query": "What is RAG?",
                    "expected_ids": ["doc_1", "doc_5"],
                    "expected_texts": ["...", "..."],  // optional
                    "reference_answer": "RAG is..."     // optional
                },
                ...
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            EvalDataset instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = [EvalItem(**item) for item in data.get("items", data)]
        return cls(items=items)

    @classmethod
    def from_csv(cls, path: str) -> "EvalDataset":
        """Load dataset from CSV file.

        CSV format:
        query,expected_ids,reference_answer
        "What is RAG?","doc_1,doc_5","RAG is..."

        Args:
            path: Path to CSV file

        Returns:
            EvalDataset instance
        """
        import csv

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected_ids = row.get("expected_ids", "").split(",")
                expected_ids = [id.strip() for id in expected_ids if id.strip()]

                item = EvalItem(
                    query=row.get("query", ""),
                    expected_ids=expected_ids,
                    reference_answer=row.get("reference_answer") or None,
                )
                items.append(item)

        return cls(items=items)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalDataset":
        """Create dataset from dictionary.

        Args:
            data: Dictionary with items list

        Returns:
            EvalDataset instance
        """
        items = [EvalItem(**item) for item in data.get("items", [])]
        return cls(items=items)

    def save(self, path: str) -> None:
        """Save dataset to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> EvalItem:
        return self.items[index]

    def get_queries(self) -> List[str]:
        """Get all queries."""
        return [item.query for item in self.items]

    def get_expected_ids(self) -> List[List[str]]:
        """Get all expected IDs."""
        return [item.expected_ids for item in self.items]

    def get_reference_answers(self) -> List[Optional[str]]:
        """Get all reference answers."""
        return [item.reference_answer for item in self.items]


def create_sample_dataset() -> EvalDataset:
    """Create a hardcoded sample evaluation dataset for testing.

    Returns:
        Sample EvalDataset with example queries
    """
    items = [
        EvalItem(
            query="What is RAG?",
            expected_ids=["doc_1", "doc_2"],
            reference_answer="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation.",
        ),
        EvalItem(
            query="How does vector search work?",
            expected_ids=["doc_3", "doc_5"],
            reference_answer="Vector search works by embedding documents and queries into vectors, then finding similar vectors.",
        ),
        EvalItem(
            query="What are the benefits of hybrid retrieval?",
            expected_ids=["doc_4"],
            reference_answer="Hybrid retrieval combines keyword and semantic search for better results.",
        ),
    ]
    return EvalDataset(items=items)


def extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text for generating queries.

    Args:
        text: Text content
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of keywords
    """
    # Simple keyword extraction using common patterns
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and", "but",
        "if", "or", "because", "until", "while", "although", "though",
        "这", "那", "是", "有", "和", "的", "了", "在", "不", "也", "就", "都",
        "可以", "会", "要", "能", "一个", "这个", "那个", "什么", "怎么", "如何",
    }

    # Extract words (English and Chinese)
    words = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", text.lower())

    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:max_keywords]]


def generate_query_from_text(text: str, style: str = "question") -> str:
    """Generate a query string from text content.

    Args:
        text: Source text content
        style: Query style - "question", "keyword", or "summary"

    Returns:
        Generated query string
    """
    keywords = extract_keywords_from_text(text, max_keywords=5)

    if not keywords:
        # Fallback: use first sentence
        first_sentence = text.split(". ")[0] if ". " in text else text[:100]
        return f"What is discussed in: {first_sentence[:50]}?"

    if style == "keyword":
        return " ".join(keywords[:3])

    if style == "summary":
        return f"Tell me about {keywords[0]}"

    # Default: question style
    if len(keywords) >= 2:
        return f"What is the relationship between {keywords[0]} and {keywords[1]}?"
    else:
        return f"What is {keywords[0]}?"


def create_dataset_from_nodes(
    nodes: List[Any],
    num_samples: int = 10,
    query_style: str = "question",
    include_texts: bool = True,
    llm: Optional[Any] = None,
    generate_answers: bool = False,
) -> EvalDataset:
    """Create evaluation dataset from a list of nodes.

    Args:
        nodes: List of TextNode objects from vector store
        num_samples: Number of samples to generate
        query_style: Query generation style ("question", "keyword", "summary")
        include_texts: Include expected_texts in items
        llm: Optional LLM for generating questions/answers
        generate_answers: Generate reference answers using LLM

    Returns:
        EvalDataset with generated evaluation items
    """
    if not nodes:
        raise ValueError("No nodes provided")

    # Sample nodes
    sample_size = min(num_samples, len(nodes))
    sampled_nodes = random.sample(nodes, sample_size)

    items = []
    for node in sampled_nodes:
        text = node.text
        node_id = node.node_id

        # Generate query
        if llm and query_style == "llm":
            query = generate_llm_query(llm, text)
        else:
            query = generate_query_from_text(text, style=query_style)

        # Generate reference answer if LLM provided
        reference_answer = None
        if generate_answers and llm:
            reference_answer = generate_llm_answer(llm, query, text)

        # Build expected_texts
        expected_texts = [text[:500]] if include_texts else None

        item = EvalItem(
            query=query,
            expected_ids=[node_id],
            expected_texts=expected_texts,
            reference_answer=reference_answer,
        )
        items.append(item)

    return EvalDataset(items=items)


def generate_llm_query(llm: Any, text: str) -> str:
    """Generate a question from text using LLM.

    Args:
        llm: LLM instance
        text: Source text

    Returns:
        Generated question
    """
    prompt = f"""Based on the following text, generate a specific question that can be answered using this content.
The question should be clear and directly related to the key information in the text.

Text:
{text[:500]}

Question:"""

    response = llm.complete(prompt)
    return response.text.strip()


def generate_llm_answer(llm: Any, query: str, context: str) -> str:
    """Generate a reference answer using LLM.

    Args:
        llm: LLM instance
        query: Question
        context: Source context

    Returns:
        Generated answer
    """
    prompt = f"""Answer the following question based on the provided context.
Provide a concise and accurate answer.

Context:
{context[:500]}

Question: {query}

Answer:"""

    response = llm.complete(prompt)
    return response.text.strip()


def create_dataset_from_documents(
    documents_dir: str,
    output_path: str,
    num_samples: int = 10,
    query_style: str = "question",
    chunk_size: int = 512,
    recursive: bool = True,
) -> EvalDataset:
    """Create evaluation dataset from documents directory.

    This loads documents, chunks them, and generates evaluation items
    from the chunks. Does NOT require vector store connection.

    Args:
        documents_dir: Path to documents directory
        output_path: Path to save the generated dataset
        num_samples: Number of samples to generate
        query_style: Query generation style
        chunk_size: Chunk size for splitting documents
        recursive: Search subdirectories

    Returns:
        EvalDataset with generated items
    """
    from ..ingestion.loaders import DocumentLoader
    from ..ingestion.splitters import TextSplitter

    # Load documents
    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_directory(documents_dir, recursive=recursive)

    if not documents:
        raise ValueError(f"No documents found in {documents_dir}")

    # Split documents
    splitter = TextSplitter(
        splitter_type="sentence",
        chunk_size=chunk_size,
        chunk_overlap=50,
    )
    nodes = splitter.split_documents(documents)

    # Create dataset
    dataset = create_dataset_from_nodes(
        nodes=nodes,
        num_samples=num_samples,
        query_style=query_style,
        include_texts=True,
        llm=None,
        generate_answers=False,
    )

    # Save to output
    dataset.save(output_path)

    return dataset


def create_dataset_from_pipeline(
    pipeline: Any,
    output_path: str,
    num_samples: int = 10,
    query_style: str = "question",
    generate_answers: bool = False,
) -> EvalDataset:
    """Create evaluation dataset from existing RAG pipeline.

    This extracts nodes from the pipeline's vector store and generates
    evaluation items. Useful for creating test data from already ingested content.

    Args:
        pipeline: RAGPipeline instance
        output_path: Path to save the generated dataset
        num_samples: Number of samples to generate
        query_style: Query generation style
        generate_answers: Generate reference answers using pipeline's LLM

    Returns:
        EvalDataset with generated items
    """
    # Get nodes from vector store
    vector_store = pipeline._vector_store
    count = vector_store.count()

    if count == 0:
        raise ValueError("Vector store is empty. Please ingest documents first.")

    # Sample nodes
    sample_size = min(num_samples, count)

    # Try to get nodes from vector store
    nodes = []
    try:
        # Get ref_doc_info to find node IDs
        if hasattr(vector_store, 'get_ref_doc_info'):
            # Sample some document IDs
            # This is a simplified approach - may need adjustment based on store type
            pass

        # Alternative: use random queries to retrieve nodes
        # Generate some generic queries to get sample nodes
        sample_queries = [
            "information",
            "data",
            "content",
            "document",
            "text",
        ]

        for query in sample_queries:
            result = pipeline.query(query, top_k=10)
            for node_with_score in result.get("source_nodes", []):
                nodes.append(node_with_score.node)

        # Deduplicate by node_id
        seen_ids = set()
        unique_nodes = []
        for node in nodes:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_nodes.append(node)

        nodes = unique_nodes[:sample_size]

    except Exception as e:
        raise ValueError(f"Failed to retrieve nodes from vector store: {e}")

    if not nodes:
        raise ValueError("Could not extract nodes from pipeline")

    # Create dataset
    llm = pipeline._llm if generate_answers else None
    dataset = create_dataset_from_nodes(
        nodes=nodes,
        num_samples=len(nodes),  # Use all extracted nodes
        query_style=query_style,
        include_texts=True,
        llm=llm,
        generate_answers=generate_answers,
    )

    # Save to output
    dataset.save(output_path)

    return dataset