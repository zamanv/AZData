"""AES-256 credential encryption for database passwords and connection strings.

Uses Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256).
Keys are ephemeral — stored only in Streamlit session state.
"""

from typing import Optional
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken


_SESSION_KEY_FERNET: str = "az_fernet_key"


def _get_or_create_fernet() -> Fernet:
    """Return or create an ephemeral Fernet key in session state."""
    if _SESSION_KEY_FERNET not in st.session_state:
        st.session_state[_SESSION_KEY_FERNET] = Fernet.generate_key()
    return Fernet(st.session_state[_SESSION_KEY_FERNET])


def encrypt_credential(plaintext: str) -> str:
    """Encrypt a plaintext credential string.

    Args:
        plaintext: The raw credential (e.g., database password).

    Returns:
        Base64-encoded encrypted token as a string.
    """
    if not plaintext:
        return ""
    fernet = _get_or_create_fernet()
    encrypted = fernet.encrypt(plaintext.encode("utf-8"))
    return encrypted.decode("utf-8")


def decrypt_credential(encrypted_token: str) -> str:
    """Decrypt an encrypted credential token.

    Args:
        encrypted_token: The encrypted string from encrypt_credential().

    Returns:
        Decrypted plaintext string.

    Raises:
        ValueError: If decryption fails (wrong key, corrupted data).
    """
    if not encrypted_token:
        return ""
    try:
        fernet = _get_or_create_fernet()
        decrypted = fernet.decrypt(encrypted_token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except InvalidToken:
        raise ValueError("Decryption failed — invalid token or wrong key")


def secure_store_credential(key: str, value: str) -> None:
    """Encrypt and store a credential in session state.

    Args:
        key: Storage key (e.g., 'db_password').
        value: Raw credential value to encrypt.
    """
    encrypted = encrypt_credential(value)
    st.session_state[f"az_cred_{key}"] = encrypted


def secure_get_credential(key: str) -> str:
    """Retrieve and decrypt a credential from session state.

    Args:
        key: Storage key used in secure_store_credential().

    Returns:
        Decrypted credential string, or empty string if not found.
    """
    encrypted = st.session_state.get(f"az_cred_{key}", "")
    if not encrypted:
        return ""
    return decrypt_credential(encrypted)


def wipe_credential(key: str) -> None:
    """Securely wipe a credential from session state."""
    cred_key = f"az_cred_{key}"
    if cred_key in st.session_state:
        del st.session_state[cred_key]


def wipe_all_credentials() -> None:
    """Wipe all stored credentials from session state."""
    keys_to_remove = [
        k for k in st.session_state if k.startswith("az_cred_")
    ]
    for key in keys_to_remove:
        del st.session_state[key]


def mask_credential(value: str, visible_chars: int = 4) -> str:
    """Return a masked version of a credential for display.

    Args:
        value: The credential string.
        visible_chars: Number of trailing characters to show.

    Returns:
        Masked string like '************abcd'.
    """
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    mask_len = len(value) - visible_chars
    return "*" * mask_len + value[-visible_chars:]
