"""Tests for credential encryption and masking (security/crypto.py).

Streamlit's ``session_state`` is not available outside a Streamlit runtime, so
these tests substitute a plain dict-backed stub for the module's ``st``
reference. ``mask_credential`` is pure and is tested without the stub.
"""

import types

import pytest
from cryptography.fernet import Fernet

from security import crypto
from security.crypto import (
    encrypt_credential,
    mask_credential,
    secure_get_credential,
    secure_store_credential,
    wipe_all_credentials,
    wipe_credential,
)


@pytest.fixture
def fake_st(monkeypatch):
    """Replace streamlit with a dict-backed session_state stub."""
    stub = types.SimpleNamespace(session_state={})
    monkeypatch.setattr(crypto, "st", stub)
    return stub


# ── masking (pure) ──
def test_mask_credential_hides_all_but_last_characters():
    secret = "supersecretpassword"  # 19 chars
    assert len(secret) == 19
    assert mask_credential(secret) == "*" * 15 + "word"


def test_mask_credential_masks_short_values_entirely():
    assert mask_credential("abcd") == "****"
    assert mask_credential("abc", visible_chars=4) == "***"


def test_mask_credential_respects_visible_chars():
    assert mask_credential("abcdefgh", visible_chars=2) == "******gh"


def test_mask_credential_handles_empty_input():
    assert mask_credential("") == ""
    assert mask_credential(None) == ""


def test_mask_credential_never_leaks_the_bulk_of_the_secret():
    masked = mask_credential("hunter2hunter2hunter2")
    assert "hunter2hunter2" not in masked
    assert masked.endswith("ter2")


# ── encryption round trip ──
def test_encrypt_then_decrypt_round_trips(fake_st):
    secret = "p@ssw0rd-123"
    token = encrypt_credential(secret)
    assert token != secret
    assert crypto.decrypt_credential(token) == secret


def test_ciphertext_differs_between_calls(fake_st):
    """Fernet uses a random IV, so identical plaintext must not collide."""
    assert encrypt_credential("same") != encrypt_credential("same")


def test_ciphertext_does_not_contain_plaintext(fake_st):
    token = encrypt_credential("correcthorsebatterystaple")
    assert "correcthorse" not in token
    assert "battery" not in token


def test_empty_input_round_trips_to_empty_string(fake_st):
    assert encrypt_credential("") == ""
    assert crypto.decrypt_credential("") == ""


def test_decrypt_fails_with_a_foreign_key(fake_st, monkeypatch):
    token = encrypt_credential("secret")
    monkeypatch.setitem(fake_st.session_state, crypto._SESSION_KEY_FERNET,
                        Fernet.generate_key())
    with pytest.raises(ValueError, match="Decryption failed"):
        crypto.decrypt_credential(token)


def test_key_is_generated_once_per_session(fake_st):
    encrypt_credential("a")
    key = fake_st.session_state[crypto._SESSION_KEY_FERNET]
    encrypt_credential("b")
    assert fake_st.session_state[crypto._SESSION_KEY_FERNET] == key


# ── session-state store ──
def test_secure_store_and_get_credential(fake_st):
    secure_store_credential("db_password", "hunter2")
    stored = fake_st.session_state["az_cred_db_password"]
    assert stored != "hunter2"
    assert secure_get_credential("db_password") == "hunter2"


def test_secure_get_returns_empty_for_missing_key(fake_st):
    assert secure_get_credential("never_set") == ""


def test_wipe_credential_removes_single_entry(fake_st):
    secure_store_credential("a", "1")
    secure_store_credential("b", "2")
    wipe_credential("a")
    assert "az_cred_a" not in fake_st.session_state
    assert secure_get_credential("b") == "2"


def test_wipe_credential_is_safe_when_absent(fake_st):
    wipe_credential("does_not_exist")


def test_wipe_all_credentials_clears_every_credential(fake_st):
    secure_store_credential("a", "1")
    secure_store_credential("b", "2")
    fake_st.session_state["unrelated"] = "keep"
    wipe_all_credentials()
    assert secure_get_credential("a") == ""
    assert secure_get_credential("b") == ""
    assert fake_st.session_state["unrelated"] == "keep"
