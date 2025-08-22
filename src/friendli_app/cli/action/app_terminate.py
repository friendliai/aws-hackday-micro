# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli App terminate action."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import typer

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext, app_id: str) -> None:
    """Terminate an app."""
    ctx.console.print(f"🔌 Terminating app with ID: {app_id}")

    http_client = ctx.http_client
    terminate_app_url = f"/beta/agent/agent-beta/{app_id}/terminate"

    try:
        response = http_client.post(terminate_app_url)
        response.raise_for_status()
        ctx.console.print(f"[success]✅ App '{app_id}' terminated successfully.[/]")
    except httpx.HTTPStatusError as e:
        ctx.console.print(
            f"[error]❌ Failed to terminate app. Server responded with {e.response.status_code}[/]"
        )
        try:
            error_detail = e.response.json()
            ctx.console.print(f"   Detail: {error_detail}")
        except json.JSONDecodeError:
            ctx.console.print(f"   Response: {e.response.text}")
        raise typer.Exit(1) from e
    except httpx.RequestError as e:
        ctx.console.print(
            "[error]❌ Failed to terminate app. An error occurred while making the request.[/]"
        )
        ctx.console.print(f"   Error: {e}")
        raise typer.Exit(1) from e
