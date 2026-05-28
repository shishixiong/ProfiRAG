"""Secure credential storage using OS keyring (Windows Credential Manager / Linux Secret Service)."""

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "profirag"


class KeyringCredentialStore:
    """Store and retrieve credentials securely using the OS keyring.

    - Windows: Windows Credential Manager
    - macOS: Keychain
    - Linux: Secret Service (GNOME Keyring / KDE Wallet)

    Usage:
        # Save credentials
        KeyringCredentialStore.save("auth_password", "my-secret-password")

        # Load credentials
        password = KeyringCredentialStore.load("auth_password")

        # Delete credentials
        KeyringCredentialStore.delete("auth_password")
    """

    @staticmethod
    def _get_keyring():
        try:
            import keyring
            return keyring
        except ImportError:
            logger.warning("keyring package not installed. Falling back to .env for credentials.")
            return None

    @staticmethod
    def save(key: str, value: str) -> bool:
        """Save a credential to the OS keyring.

        Args:
            key: Credential key name (e.g. "auth_password")
            value: Secret value to store

        Returns:
            True if saved successfully, False otherwise
        """
        kr = KeyringCredentialStore._get_keyring()
        if kr is None:
            return False
        try:
            kr.set_password(KEYRING_SERVICE, key, value)
            logger.info(f"Credential '{key}' saved to keyring")
            return True
        except Exception as e:
            logger.error(f"Failed to save credential '{key}' to keyring: {e}")
            return False

    @staticmethod
    def load(key: str) -> Optional[str]:
        """Load a credential from the OS keyring.

        Args:
            key: Credential key name

        Returns:
            The credential value, or None if not found
        """
        kr = KeyringCredentialStore._get_keyring()
        if kr is None:
            return None
        try:
            value = kr.get_password(KEYRING_SERVICE, key)
            if value is None:
                logger.debug(f"Credential '{key}' not found in keyring")
            return value
        except Exception as e:
            logger.error(f"Failed to load credential '{key}' from keyring: {e}")
            return None

    @staticmethod
    def delete(key: str) -> bool:
        """Delete a credential from the OS keyring.

        Args:
            key: Credential key name

        Returns:
            True if deleted successfully, False otherwise
        """
        kr = KeyringCredentialStore._get_keyring()
        if kr is None:
            return False
        try:
            kr.delete_password(KEYRING_SERVICE, key)
            logger.info(f"Credential '{key}' deleted from keyring")
            return True
        except Exception as e:
            logger.error(f"Failed to delete credential '{key}' from keyring: {e}")
            return False

    @staticmethod
    def is_available() -> bool:
        """Check if keyring backend is available and functional.

        Returns:
            True if keyring can be used
        """
        kr = KeyringCredentialStore._get_keyring()
        if kr is None:
            return False
        try:
            kr.get_password(KEYRING_SERVICE, "__test__")
            return True
        except Exception:
            return False


def main():
    """CLI for managing keyring credentials.

    Usage:
        python -m profirag.config.credentials set <key>       # Set a credential (prompts for value)
        python -m profirag.config.credentials get <key>       # Get a credential value
        python -m profirag.config.credentials delete <key>    # Delete a credential
        python -m profirag.config.credentials list            # List stored credential keys
        python -m profirag.config.credentials check           # Check if keyring is available
    """
    import getpass

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        available = KeyringCredentialStore.is_available()
        print(f"Keyring available: {available}")
        if available:
            import keyring
            print(f"Backend: {keyring.get_keyring().__class__.__name__}")

    elif command == "set":
        if len(sys.argv) < 3:
            print("Usage: python -m profirag.config.credentials set <key>")
            sys.exit(1)
        key = sys.argv[2]
        value = getpass.getpass(f"Enter value for '{key}': ")
        if KeyringCredentialStore.save(key, value):
            print(f"Credential '{key}' saved successfully")
        else:
            print(f"Failed to save credential '{key}'")
            sys.exit(1)

    elif command == "get":
        if len(sys.argv) < 3:
            print("Usage: python -m profirag.config.credentials get <key>")
            sys.exit(1)
        key = sys.argv[2]
        value = KeyringCredentialStore.load(key)
        if value is not None:
            print(f"{key}={value}")
        else:
            print(f"Credential '{key}' not found")
            sys.exit(1)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python -m profirag.config.credentials delete <key>")
            sys.exit(1)
        key = sys.argv[2]
        if KeyringCredentialStore.delete(key):
            print(f"Credential '{key}' deleted successfully")
        else:
            print(f"Failed to delete credential '{key}'")
            sys.exit(1)

    elif command == "list":
        import keyring
        kr = keyring.get_keyring()
        print(f"Keyring backend: {kr.__class__.__name__}")
        common_keys = ["auth_password", "auth_token", "api_key"]
        found = False
        for key in common_keys:
            value = KeyringCredentialStore.load(key)
            if value is not None:
                print(f"  {key}: *** (exists)")
                found = True
        if not found:
            print("  No profirag credentials found in keyring")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
