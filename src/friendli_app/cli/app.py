# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI application definition."""

from __future__ import annotations

import warnings

from rich import print
from typer import Option, Typer

from friendli_app.cli.command import app as command_app
from friendli_app.cli.command.agent import app as agent_command_app
from friendli_app.cli.context import RootContextObj, TyperAppContext
from friendli_app.cli.typer_util import OrderedCommands, merge_typer

warnings.filterwarnings("ignore")


def _app_callback(
    ctx: TyperAppContext,
    token: str | None = Option(None, "--token", help="Login token", envvar="FRIENDLI_TOKEN"),
    base_url: str | None = Option(
        None,
        "--base-url",
        help="API URL",
        envvar="FRIENDLI_BASE_URL",
        hidden=True,
    ),
) -> None:
    if ctx.resilient_parsing:
        # Called when autocomplete is enabled
        return

    if base_url is not None:
        msg = f"[magenta]🔔 Heads up! You're using a custom URL:[/] {base_url}\n"
        print(msg)

    obj = RootContextObj(base_url=base_url, token=token)
    ctx.obj = obj


app = Typer(
    cls=OrderedCommands,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    callback=_app_callback,
)

# Merge commands
merge_typer(app, command_app)
merge_typer(app, agent_command_app)


if __name__ == "__main__":
    app()
