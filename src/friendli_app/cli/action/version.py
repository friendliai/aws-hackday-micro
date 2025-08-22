# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Show client and server version."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext) -> None:
    """Show cli version."""
    msg = "Friendli Agent CLI: [cyan bold]0.0.2 [/]"
    ctx.console.print(msg)
