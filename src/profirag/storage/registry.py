"""Storage backend registry for managing multiple vector store implementations"""

from typing import Dict, Type, Any, List
from .base import BaseVectorStore


class StorageRegistry:
    """Registry for vector store backends.

    Allows registration and retrieval of different vector store implementations.
    Supports dynamic registration of new backends.
    """

    _stores: Dict[str, Type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a vector store backend.

        Args:
            name: Unique name for the backend (e.g., "qdrant", "local", "postgres")

        Returns:
            Decorator function

        Example:
            @StorageRegistry.register("qdrant")
            class QdrantStore(BaseVectorStore):
                ...
        """
        def decorator(store_class: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
            if name in cls._stores:
                raise ValueError(f"Store '{name}' is already registered")
            cls._stores[name] = store_class
            return store_class
        return decorator

    @classmethod
    def get_store_class(cls, name: str) -> Type[BaseVectorStore]:
        """Get the registered store class by name.

        Args:
            name: Name of the registered backend

        Returns:
            The registered store class

        Raises:
            ValueError: If the store name is not registered
        """
        if name not in cls._stores:
            available = list(cls._stores.keys())
            raise ValueError(f"Unknown store: '{name}'. Available stores: {available}")
        return cls._stores[name]

    @classmethod
    def get_store(cls, name: str, config: Dict[str, Any]) -> BaseVectorStore:
        """Create and return a store instance by name and config.

        Args:
            name: Name of the registered backend
            config: Configuration dictionary for the store

        Returns:
            Instance of the vector store

        Raises:
            ValueError: If the store name is not registered
        """
        store_class = cls.get_store_class(name)
        return store_class.from_config(config)

    @classmethod
    def list_stores(cls) -> List[str]:
        """List all registered store names.

        Returns:
            List of registered store names
        """
        return list(cls._stores.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a store is registered.

        Args:
            name: Name of the store to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._stores

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a store by name.

        Args:
            name: Name of the store to unregister

        Returns:
            True if successfully unregistered, False if not found
        """
        if name in cls._stores:
            del cls._stores[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered stores."""
        cls._stores.clear()