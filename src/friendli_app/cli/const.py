# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Constants."""

from __future__ import annotations

from enum import Enum

APP_NAME = "friendli-suite"
SUITE_PAT_URL = "https://friendli.ai/suite/setting/tokens"
API_BASE_URL = "https://api.friendli.ai/"


# Command group names
class Panel(str, Enum):
    """Panel names."""

    COMMON = "Common Commands"
    APP = "App Commands"
