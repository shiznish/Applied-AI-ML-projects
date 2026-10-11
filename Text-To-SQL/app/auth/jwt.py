"""Password hashing + access/refresh JWT issuance and decoding.

Password hashing uses the `bcrypt` library directly rather than through
passlib's `CryptContext`: as installed in this sandbox (passlib 1.7.4 +
bcrypt 5.0.0), passlib's bcrypt backend-detection code reads
`bcrypt.__about__.__version__`, which the bcrypt 4.x/5.x package line removed
(a well-known upstream incompatibility) — every passlib bcrypt call raises
before hashing anything. bcrypt itself works fine standalone, so we call it
directly; `passlib[bcrypt]` remains a declared dependency for compatibility
with older bcrypt pins in other environments, but this module doesn't require
passlib to import cleanly.
"""
import datetime as dt

import bcrypt
import jwt as pyjwt

from ..config import settings
from .models import User

# bcrypt hashes only the first 72 bytes of input; longer secrets are silently
# truncated by the C library, so we truncate explicitly (matching passlib's
# documented behavior) to avoid surprising truncation differences later.
_BCRYPT_MAX_BYTES = 72


class AuthError(Exception):
    """Raised for any invalid/expired/tampered token or bad credentials."""


def hash_password(plain: str) -> str:
    raw = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    raw = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(raw, hashed.encode("ascii"))
    except ValueError:
        # Malformed hash (e.g. empty string) — never a valid match.
        return False


def _claims_for(user: User) -> dict:
    """The claim shape embedded in both access and refresh tokens.

    This is the hand-off contract that RLS wiring reads via
    `app.auth.rbac.security_context_from_claims` — keep `sub`/`role`/`region`/
    `permissions` stable if this shape ever changes.
    """
    return {
        "sub": user.username,
        "role": user.role,
        "region": user.region,
        "permissions": sorted(user.permissions),
    }


def _encode(user: User, expiry_minutes: int, token_type: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        **_claims_for(user),
        "type": token_type,
        "iat": now,
        "exp": now + dt.timedelta(minutes=expiry_minutes),
    }
    return pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user: User) -> str:
    return _encode(user, settings.jwt_access_expiry_minutes, "access")


def create_refresh_token(user: User) -> str:
    return _encode(user, settings.jwt_refresh_expiry_minutes, "refresh")


def decode_token(token: str) -> dict:
    """Decode + validate a JWT, returning its claims dict.

    Raises AuthError with a clear, non-sensitive message on any failure:
    expired, tampered/invalid signature, or malformed.
    """
    try:
        return pyjwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except pyjwt.ExpiredSignatureError as e:
        raise AuthError("Token has expired") from e
    except pyjwt.InvalidTokenError as e:
        raise AuthError("Token is invalid") from e
