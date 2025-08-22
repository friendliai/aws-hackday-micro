# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""CLI application settings."""

from __future__ import annotations

import os


class PatAuthBackend:
    """Authentication backend."""

    def __init__(self) -> None:
        """Initialize."""
        pass

    def store_credential(self, user_id: str, token: str) -> None:
        """This method currently does not store the token persistently."""
        pass

    def fetch_credential(self, user_id: str) -> str | None:
        """Fetch personal access token from the FRIENDLI_TOKEN environment variable."""
        return os.getenv("FRIENDLI_TOKEN")

    def clear_credential(self, user_id: str) -> None:
        """Clear personal access token from keychain."""
        pass
