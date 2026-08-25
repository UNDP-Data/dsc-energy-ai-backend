"""
Security routines for authenticating requests.
"""

import hmac
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-Api-Key",
    description="The API key for authentication.",
    auto_error=True,
)


async def authenticate(api_key: str = Security(api_key_header)) -> bool:
    """
    Authenticate a request using an API Key header.

    Parameters
    ----------
    api_key : str
        API key as supplied by the client.

    Returns
    -------
    bool
        True if the authentication has been successful.

    Raises
    ------
    HTTPException
        If no key is provided or the key is invalid.
    """
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server.",
        )
    if hmac.compare_digest(api_key, expected_key):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
    )


async def authenticate_sgp_project_access(
    api_key: str = Security(api_key_header),
) -> bool:
    """Authenticate the separate server-side credential for non-public SGP corpora."""
    expected_key = os.getenv("SGP_PROJECT_RAG_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SGP project-corpus access is not configured on the server.",
        )
    if hmac.compare_digest(api_key, expected_key):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid SGP project-corpus API key.",
    )
