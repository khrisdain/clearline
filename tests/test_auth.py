import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from clearline.auth import Authenticator
from clearline.config import Settings
from clearline.models import Classification

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
ISSUER, AUDIENCE = "https://sso.example.test/realms/clearline", "clearline-api"


def settings() -> Settings:
    return Settings(env="test", auth_mode="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE)


def token(**overrides) -> str:
    now = int(time.time())
    claims = {"sub": "u-1", "preferred_username": "dana", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300,
              "clearance": ["protected-b"], "groups": ["/staff", "/departments/finance"],
              "realm_access": {"roles": ["clearline-auditor"]}}
    claims.update(overrides)
    return jwt.encode(claims, KEY, algorithm="RS256")


def request(header: str | None) -> Request:
    headers = [(b"authorization", header.encode())] if header else []
    return Request({"type": "http", "headers": headers})


def authenticate(header: str | None):
    return Authenticator(settings(), key_for_token=lambda _: KEY.public_key())(request(header))


def test_a_valid_token_becomes_a_principal():
    user = authenticate(f"Bearer {token()}")
    assert user.subject == "dana"
    assert user.clearance is Classification.PROTECTED_B
    assert user.groups == frozenset({"staff", "finance"})
    assert user.is_auditor and not user.is_admin


@pytest.mark.parametrize("header", [
    None,
    "Basic abc",
    f"Bearer {token(exp=int(time.time()) - 10)}",
    f"Bearer {token(aud='another-app')}",
    f"Bearer {token(iss='https://evil.example/realms/clearline')}",
    "Bearer not-a-jwt",
])
def test_bad_or_missing_tokens_are_rejected(header):
    with pytest.raises(HTTPException) as error:
        authenticate(header)
    assert error.value.status_code == 401


def test_a_token_signed_by_another_key_is_rejected():
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(jwt.decode(token(), options={"verify_signature": False}), other, algorithm="RS256")
    with pytest.raises(HTTPException):
        authenticate(f"Bearer {forged}")


def test_an_unknown_clearance_falls_back_to_unclassified():
    assert authenticate(f"Bearer {token(clearance='top-secret-ish')}").clearance is Classification.UNCLASSIFIED


def test_dev_login_is_refused_in_production():
    with pytest.raises(ValidationError):
        Settings(env="production", auth_mode="dev")
