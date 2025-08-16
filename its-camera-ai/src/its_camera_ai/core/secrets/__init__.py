"""
Secrets management module for ITS Camera AI.

Provides secure access to secrets stored in HashiCorp Vault with
automatic authentication, token renewal, and secret caching.
"""

from .vault_client import (
    VaultAPIError,
    VaultAuthError,
    VaultAuthToken,
    VaultClient,
    VaultConfig,
    VaultConnectionError,
    VaultError,
    VaultPermissionError,
    VaultSecret,
    get_vault_client,
    load_secrets_from_vault,
)

__all__ = [
    "VaultClient",
    "VaultConfig",
    "VaultSecret",
    "VaultAuthToken",
    "VaultError",
    "VaultConnectionError",
    "VaultAuthError",
    "VaultPermissionError",
    "VaultAPIError",
    "get_vault_client",
    "load_secrets_from_vault",
]
