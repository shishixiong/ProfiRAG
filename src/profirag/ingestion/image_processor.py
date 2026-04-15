"""Image processing module for PDF image handling with MiniMax VLM."""

import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.schema import TextNode


# Default prompt for image description
DEFAULT_IMAGE_DESCRIPTION_PROMPT = "描述这张图片的内容，包括图片中的文字、图形、图表等关键信息"


def understand_image_minimax(
    image_path: str,
    prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT,
    api_key: Optional[str] = None,
    api_host: str = "https://api.minimax.chat",
    timeout: int = 60,
) -> str:
    """Understand image using MiniMax VLM API.

    Args:
        image_path: Path to image file.
        prompt: Prompt for image understanding.
        api_key: MiniMax API key (uses environment variable if not provided).
        api_host: MiniMax API host URL.
        timeout: Request timeout in seconds.

    Returns:
        Image description text.

    Raises:
        EnvironmentError: If API key is not configured.
        FileNotFoundError: If image file does not exist.
    """
    # Get API key from environment if not provided
    key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not key:
        raise EnvironmentError("MINIMAX_API_KEY environment variable not set")

    # Check image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read and encode image
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    # Determine image type from file extension
    ext = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    # Build request
    url = f"{api_host}/v1/coding_plan/vlm"
    data = json.dumps({
        "prompt": prompt,
        "image_url": f"data:{mime_type};base64,{img_data}"
    }).encode()

    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    })

    # Send request and parse response
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
        return result.get("content", "")


class ImageProcessor:
    """Process extracted images and generate descriptions using MiniMax VLM.

    This class handles:
    - Scanning image directories from PDF extraction
    - Generating descriptions using multimodal LLM
    - Creating searchable ImageNodes for vector storage
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_host: str = "https://api.minimax.chat",
        description_prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT,
        storage_path: str = "./images",
        generate_descriptions: bool = True,
        timeout: int = 60,
    ):
        """Initialize image processor.

        Args:
            api_key: MiniMax API key
            api_host: MiniMax API host URL
            description_prompt: Prompt template for image description
            storage_path: Base path for image storage
            generate_descriptions: Whether to generate descriptions (can be disabled for testing)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_host = api_host
        self.description_prompt = description_prompt
        self.storage_path = storage_path
        self.generate_descriptions = generate_descriptions
        self.timeout = timeout

    def process_images_from_directory(
        self,
        image_directory: str,
        source_document: str,
        image_map: Optional[Dict[str, Any]] = None,
    ) -> List[TextNode]:
        """Process all images from a directory and create ImageNodes.

        Args:
            image_directory: Directory containing extracted images
            source_document: Source document name for metadata
            image_map: Optional mapping of image IDs to metadata (position, context)

        Returns:
            List of TextNode objects with image descriptions
        """
        if not os.path.exists(image_directory):
            return []

        image_nodes = []
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        # Scan for image files
        for image_file in Path(image_directory).iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue

            # Get image info from image_map if available
            image_id = image_file.stem
            img_info = image_map.get(image_id, {}) if image_map else {}

            # Generate description
            description = ""
            if self.generate_descriptions:
                try:
                    description = understand_image_minimax(
                        str(image_file),
                        prompt=self.description_prompt,
                        api_key=self.api_key,
                        api_host=self.api_host,
                        timeout=self.timeout,
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate description for {image_file}: {e}")
                    description = f"Image from {source_document}"

            # Create ImageNode
            node = self._create_image_node(
                image_path=str(image_file),
                description=description,
                source_document=source_document,
                page_number=img_info.get("page_number"),
                original_context=img_info.get("surrounding_text"),
            )
            image_nodes.append(node)

        return image_nodes

    def _create_image_node(
        self,
        image_path: str,
        description: str,
        source_document: str,
        page_number: Optional[int] = None,
        original_context: Optional[str] = None,
    ) -> TextNode:
        """Create a searchable TextNode from image description.

        Args:
            image_path: Path to the image file
            description: Generated description text
            source_document: Source document name
            page_number: Page number in source document
            original_context: Surrounding text from original document

        Returns:
            TextNode with image metadata
        """
        # Calculate image hash for deduplication
        image_hash = self._calculate_hash(image_path)

        # Build metadata
        metadata = {
            "node_type": "image",
            "image_path": image_path,
            "image_type": Path(image_path).suffix.lower(),
            "source_file": source_document,
            "image_hash": image_hash,
            "description": description,
        }

        if page_number:
            metadata["page_number"] = page_number
        if original_context:
            metadata["original_context"] = original_context

        # Node text is the description (searchable)
        return TextNode(
            text=description,
            metadata=metadata,
        )

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for deduplication.

        Args:
            file_path: Path to file

        Returns:
            SHA-256 hash string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_image_by_path(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Get image info by path for retrieval.

        Args:
            image_path: Path to the image file

        Returns:
            Image info dictionary if exists, None otherwise
        """
        if not os.path.exists(image_path):
            return None

        return {
            "path": image_path,
            "exists": True,
            "size": os.path.getsize(image_path),
            "type": Path(image_path).suffix.lower(),
        }


class ImageResult:
    """Result object for image retrieval."""

    def __init__(
        self,
        image_path: str,
        description: str,
        score: float = 0.0,
        source_chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize image result.

        Args:
            image_path: Path to the image file
            description: Image description text
            score: Relevance score
            source_chunk_id: ID of chunk that references this image
            metadata: Additional metadata
        """
        self.image_path = image_path
        self.description = description
        self.score = score
        self.source_chunk_id = source_chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "path": self.image_path,
            "description": self.description,
            "score": self.score,
            "source_chunk": self.source_chunk_id,
            "metadata": self.metadata,
        }


class RetrievalResult:
    """Result containing text nodes and associated images."""

    def __init__(
        self,
        text_nodes: List[Any],
        images: List[ImageResult],
    ):
        """Initialize retrieval result.

        Args:
            text_nodes: List of NodeWithScore objects
            images: List of ImageResult objects
        """
        self.text_nodes = text_nodes
        self.images = images

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "text_nodes": [
                {
                    "node_id": n.node.node_id if hasattr(n, 'node') else n.get('node_id'),
                    "text": n.node.text[:200] if hasattr(n, 'node') else n.get('text', '')[:200],
                    "score": n.score if hasattr(n, 'score') else n.get('score', 0),
                }
                for n in self.text_nodes
            ],
            "images": [img.to_dict() for img in self.images],
        }