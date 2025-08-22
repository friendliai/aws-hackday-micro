# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typer import Typer

from friendli_app.cli.const import Panel
from friendli_app.cli.context import AppContext, TyperAppContext

group = Typer()


@group.command(
    "version",
    help="Show Friendli Suite version information.",
    rich_help_panel=Panel.COMMON,
)
def version(ctx: TyperAppContext) -> None:
    from ..action.version import run

    with AppContext(ctx.obj) as app_ctx:
        run(app_ctx)
