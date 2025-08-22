# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Application context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from typer import get_app_dir

from friendli_app.cli.const import API_BASE_URL, APP_NAME
from friendli_app.cli.typer_util import TyperContext

if TYPE_CHECKING:
    from types import EllipsisType, TracebackType
    from typing import Self


@dataclass
class RootContextObj:
    """Root context for the application."""

    base_url: str | None = None
    token: str | None = None


class TyperAppContext(TyperContext[RootContextObj]):
    """Typer's application context."""


class AppContext:
    """Application context."""

    def __init__(self, root_ctx: RootContextObj) -> None:
        """Initialize application context."""
        self.root = root_ctx

        from rich.console import Console
        from rich.theme import Theme

        from .backend.auth import PatAuthBackend
        from .backend.settings import SettingsBackend

        app_dir = get_app_dir(APP_NAME)
        self.settings_backend = SettingsBackend(Path(app_dir))
        self.auth_backend = PatAuthBackend()

        # get auth - prioritize --token option over environment variable
        auth = root_ctx.token
        if not auth and (user_info := self.settings_backend.settings.user_info):
            auth = self.auth_backend.fetch_credential(user_info.user_id)
        self.root.token = auth

        headers = {
            "Accept": "application/json",
        }
        if auth:
            headers["Authorization"] = f"Bearer {auth}"

        base_url = root_ctx.base_url or API_BASE_URL

        # Configure timeouts (tunable via env vars)
        timeout = httpx.Timeout(
            connect=float(os.getenv("FRIENDLI_HTTP_CONNECT_TIMEOUT", "10")),
            read=float(os.getenv("FRIENDLI_HTTP_READ_TIMEOUT", "60")),
            write=float(os.getenv("FRIENDLI_HTTP_WRITE_TIMEOUT", "60")),
            pool=float(os.getenv("FRIENDLI_HTTP_POOL_TIMEOUT", "60")),
        )

        self.http_client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

        theme = Theme(
            {
                "info": "dim cyan",
                "warn": "magenta",
                "danger": "bold red",
                "success": "bold green",
                "headline": "bold bright_blue",
                "subheadline": "dim bright_blue",
                "content": "bright_white",
                "highlight": "bold bright_yellow",
            }
        )
        self.console = Console(theme=theme)

    def refresh_client(self, auth: str | EllipsisType | None = ...) -> None:
        """Refresh http client."""
        if auth is not ...:
            self.root.token = auth

        if self.root.token:
            self.http_client.headers["Authorization"] = f"Bearer {self.root.token}"
        if self.root.base_url:
            self.http_client.base_url = self.root.base_url

    def __enter__(self) -> Self:
        """Context manager for application context."""
        self.http_client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.http_client.__exit__(exc_type, exc_val, exc_tb)
