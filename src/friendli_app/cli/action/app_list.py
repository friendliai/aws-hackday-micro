# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli Agent list action."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import typer
from rich.table import Table

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext, project_id: str | None) -> None:
    """List apps."""
    ctx.console.print(f"🔍 Listing apps in project: {project_id}")

    http_client = ctx.http_client
    list_apps_url = "/beta/agent/agent-beta/list"

    try:
        # Always include project_id in params, even when None as requested
        response = http_client.get(list_apps_url, params={"project_id": project_id})
        response.raise_for_status()
        response_json = response.json()
        agents = response_json.get("agents", [])

        if not agents:
            ctx.console.print("No apps found in this project.")
            return

        table = Table(
            "ID", "Name", "Status", "Creator ID", "Created At", "Updated At", title="Agents"
        )
        for agent in agents:
            table.add_row(
                agent["id"],
                agent["name"],
                agent["status"],
                agent["creator_id"],
                agent["created_at"],
                agent["updated_at"],
            )
        ctx.console.print(table)

    except httpx.HTTPStatusError as e:
        ctx.console.print(
            f"[error]❌ Failed to list apps. Server responded with {e.response.status_code}[/]"
        )
        try:
            error_detail = e.response.json()
            ctx.console.print(f"   Detail: {error_detail}")
        except json.JSONDecodeError:
            ctx.console.print(f"   Response: {e.response.text}")
        raise typer.Exit(1) from e
    except httpx.RequestError as e:
        ctx.console.print(
            "[error]❌ Failed to list agents. An error occurred while making the request.[/]"
        )
        ctx.console.print(f"   Error: {e}")
        raise typer.Exit(1) from e
