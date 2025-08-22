# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli Suite CLI commands."""

from __future__ import annotations

from typer import Typer

from friendli_app.cli.command.agent import app as agent_app
from friendli_app.cli.command.auth import group as auth_group
from friendli_app.cli.command.version import group as version_group
from friendli_app.cli.typer_util import merge_typer

app = Typer()

merge_typer(app, auth_group)
merge_typer(app, version_group)
merge_typer(app, agent_app)
