"""Pure JWT/password tests — no DB, no network."""
import datetime as dt

import jwt as pyjwt
import pytest

from app.auth.jwt import (
    AuthError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth.models import User
from app.config import settings

_ALICE = User(
    username="alice",
    hashed_password=hash_password("password123"),
    role="sales_us",
    region="us",
)


def test_hash_password_then_verify_roundtrip():
    hashed = hash_password("password123")
    assert hashed != "password123"
    assert verify_password("password123", hashed)


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("password123")
    assert not verify_password("wrong-password", hashed)


def test_verify_password_rejects_malformed_hash():
    assert not verify_password("password123", "not-a-real-bcrypt-hash")


def test_access_token_roundtrip_carries_expected_claims():
    token = create_access_token(_ALICE)
    claims = decode_token(token)
    assert claims["sub"] == "alice"
    assert claims["role"] == "sales_us"
    assert claims["region"] == "us"
    assert claims["permissions"] == ["query:execute"]
    assert claims["type"] == "access"


def test_refresh_token_has_refresh_type_and_longer_expiry():
    access = decode_token(create_access_token(_ALICE))
    refresh = decode_token(create_refresh_token(_ALICE))
    assert refresh["type"] == "refresh"
    assert refresh["exp"] > access["exp"]


def test_expired_token_is_rejected():
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": "alice", "role": "sales_us", "region": "us", "permissions": ["query:execute"],
        "type": "access",
        "iat": now - dt.timedelta(hours=1),
        "exp": now - dt.timedelta(minutes=1),
    }
    expired = pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    with pytest.raises(AuthError):
        decode_token(expired)


def test_tampered_token_is_rejected():
    token = create_access_token(_ALICE)
    # Flip a character in the signature segment to invalidate it.
    header, payload, signature = token.split(".")
    tampered_sig = ("a" if signature[0] != "a" else "b") + signature[1:]
    tampered = f"{header}.{payload}.{tampered_sig}"
    with pytest.raises(AuthError):
        decode_token(tampered)


def test_token_signed_with_wrong_secret_is_rejected():
    bad_token = pyjwt.encode(
        {"sub": "alice", "role": "sales_us", "region": "us", "permissions": [], "type": "access"},
        "some-other-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(AuthError):
        decode_token(bad_token)
