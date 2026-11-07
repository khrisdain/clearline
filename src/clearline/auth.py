"""Who is asking. In production: an OIDC bearer token (Keycloak), verified against the realm's
JWKS. Locally: a few fixed dev users picked with the X-Dev-User header (refused in production).

Clearance comes from a token claim (a Keycloak user attribute mapped into the token), groups
from the group-membership claim, roles from realm roles.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jwt
from fastapi import HTTPException, Request

from .config import Settings
from .models import Classification, Principal

DEV_USERS: dict[str, Principal] = {
    "ana": Principal("ana", Classification.UNCLASSIFIED, frozenset({"staff"})),
    "ben": Principal("ben", Classification.PROTECTED_B, frozenset({"staff", "finance"})),
    "chloe": Principal("chloe", Classification.PROTECTED_B, frozenset({"staff", "security", "it"}),
                       frozenset({"clearline-auditor"})),
    "admin": Principal("admin", Classification.PROTECTED_B, frozenset({"staff", "it"}), frozenset({"clearline-admin"})),
}


def principal_from_claims(claims: dict[str, Any], settings: Settings) -> Principal:
    raw = claims.get(settings.clearance_claim, "unclassified")
    if isinstance(raw, list):  # Keycloak maps multivalued attributes to lists
        raw = raw[0] if raw else "unclassified"
    try:
        clearance = Classification.parse(raw)
    except (KeyError, ValueError):
        clearance = Classification.UNCLASSIFIED
    groups = frozenset(str(g).strip("/").split("/")[-1].lower() for g in claims.get(settings.groups_claim, []))
    roles = frozenset(claims.get("realm_access", {}).get("roles", []))
    return Principal(subject=str(claims.get("preferred_username") or claims["sub"]), clearance=clearance, groups=groups,
                     roles=roles)


class Authenticator:
    def __init__(self, settings: Settings, key_for_token: Callable[[str], Any] | None = None) -> None:
        self.settings = settings
        if key_for_token is None and settings.auth_mode == "oidc":
            jwks = jwt.PyJWKClient(settings.jwks_url, cache_keys=True, lifespan=300)
            key_for_token = lambda token: jwks.get_signing_key_from_jwt(token).key  # noqa: E731
        self.key_for_token = key_for_token

    def __call__(self, request: Request) -> Principal:
        if self.settings.auth_mode == "dev":
            user = DEV_USERS.get(request.headers.get("x-dev-user", "ana"))
            if not user:
                raise HTTPException(401, "Unknown dev user")
            return user
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        token = header[7:]
        try:
            assert self.key_for_token is not None
            claims = jwt.decode(token, self.key_for_token(token), algorithms=["RS256"], audience=self.settings.oidc_audience,
                                issuer=self.settings.oidc_issuer, options={"require": ["exp", "iat", "sub"]})
        except jwt.PyJWTError as error:
            raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"}) from error
        return principal_from_claims(claims, self.settings)
