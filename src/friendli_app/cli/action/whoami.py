# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli whoami action."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import typer

from friendli_app.cli.schema.user import WhoamiResponse

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext) -> None:
    """Show information about the logged in user."""
    http_client = ctx.http_client
    if not http_client.headers.get("Authorization"):
        ctx.console.print("FRIENDLI_TOKEN is not set. Please set [bold]'FRIENDLI_TOKEN'[/] first.")
        raise typer.Exit(1) from None

    try:
        response = http_client.get("/api/auth/whoami")
        response.raise_for_status()
        info = WhoamiResponse.model_validate_json(response.text)

        user_name = info.user_name or info.user_email
        ctx.console.print(f"Hi, [bold]{user_name}[/].")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            ctx.console.print(
                "Unauthorized. Please make sure your [bold]'FRIENDLI_TOKEN'[/] is set."
            )
        else:
            ctx.console.print(f"[error]❌ API request failed (Status: {e.response.status_code})[/]")
        raise typer.Exit(1) from None
    except httpx.RequestError as e:
        ctx.console.print(f"[error]❌ Connection failed: {e}[/]")
        raise typer.Exit(1) from None
