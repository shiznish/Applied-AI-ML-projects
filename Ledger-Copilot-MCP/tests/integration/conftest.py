from __future__ import annotations

import base64
from collections.abc import Callable

import httpx2
import pytest
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.utilities.asgi_transport import StreamingASGITransport
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from ledger_copilot.server import create_app

TEST_ISSUER = "https://issuer.test"
TEST_AUDIENCE = "ledger-copilot"
TEST_KID = "test-key"


def _b64url_uint(value: int) -> str:
    length = (value.bit_length() + 7) // 8 or 1
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode("ascii")


def _jwks_for(keypair: RSAKeyPair) -> dict[str, object]:
    public_key = load_pem_public_key(keypair.public_key.encode("utf-8"))
    assert isinstance(public_key, RSAPublicKey)
    numbers = public_key.public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": TEST_KID,
                "use": "sig",
                "alg": "RS256",
                "n": _b64url_uint(numbers.n),
                "e": _b64url_uint(numbers.e),
            }
        ]
    }


@pytest.fixture
def rsa_keypair() -> RSAKeyPair:
    return RSAKeyPair.generate()


@pytest.fixture
def make_token(rsa_keypair: RSAKeyPair) -> Callable[..., str]:
    def _make_token(
        scopes: list[str] | None = None,
        *,
        subject: str = "test-user",
        audience: str | list[str] | None = TEST_AUDIENCE,
        issuer: str = TEST_ISSUER,
        expires_in_seconds: int = 3600,
        kid: str | None = TEST_KID,
    ) -> str:
        return rsa_keypair.create_token(
            subject=subject,
            issuer=issuer,
            audience=audience,
            scopes=scopes or [],
            expires_in_seconds=expires_in_seconds,
            kid=kid,
        )

    return _make_token


@pytest.fixture
def http_app(rsa_keypair: RSAKeyPair) -> Starlette:
    """A real Starlette ASGI app (not the in-memory FastMCPTransport used
    elsewhere) fronted by a JWTVerifier that fetches its JWKS from an
    in-process mock endpoint signing with `rsa_keypair` — a local JWT
    signer stand-in, never a live IdP."""
    jwks = _jwks_for(rsa_keypair)

    def _handle_jwks(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=jwks)

    jwks_http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(_handle_jwks))
    verifier = JWTVerifier(
        jwks_uri=f"{TEST_ISSUER}/.well-known/jwks.json",
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
        http_client=jwks_http_client,
    )
    auth = RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[AnyHttpUrl(TEST_ISSUER)],
        base_url="http://testserver",
    )
    app = create_app(auth=auth)
    return app.http_app()


@pytest.fixture
def authorized_client_factory(
    http_app: Starlette,
) -> Callable[..., Callable[..., httpx2.AsyncClient]]:
    """Factory-of-factories handed to `StreamableHttpTransport(httpx_client_factory=...)`
    so a `fastmcp.Client` talks to `http_app` over the same in-process ASGI bridge."""

    def _factory() -> Callable[..., httpx2.AsyncClient]:
        def _make_client(**kwargs: object) -> httpx2.AsyncClient:
            kwargs["transport"] = StreamingASGITransport(http_app)
            kwargs["base_url"] = "http://testserver"
            return httpx2.AsyncClient(**kwargs)

        return _make_client

    return _factory
