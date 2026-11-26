"""API-key authentication.   >>> YOUR TURN: implement ``require_api_key``. <<<

Tests that define "done": tests/api/test_auth.py
"""

import secrets
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from docaudit.api.deps import ServicesDep

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: Annotated[str | None, Security(api_key_header)],
    services: ServicesDep,
) -> None:
    """Let the request through only with the right key in ``X-API-Key``.

    Requirements:
      * ``settings.api_key`` is ``None``: auth is off, let everything through.
      * Missing or wrong key: ``401`` with a ``WWW-Authenticate: ApiKey`` header. Don't say which
        of the two it was.
      * Compare with ``secrets.compare_digest``, not ``==``: ``==`` stops at the first different
        character, and how long it takes tells an attacker how many characters were right.
    """
    expected_key = services.settings.api_key
    if expected_key is None:
        return
    if key is None or not secrets.compare_digest(key, expected_key.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
