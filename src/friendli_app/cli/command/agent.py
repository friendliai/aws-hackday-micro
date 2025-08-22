# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Friendli Agent command group."""

from __future__ import annotations

from typer import Argument, Option, Typer

from friendli_app.cli.const import Panel
from friendli_app.cli.context import AppContext, TyperAppContext
from friendli_app.cli.typer_util import CommandUsageExample, format_examples

app = Typer(
    name="Friendli Apps",
    help="Manage apps.",
    rich_help_panel=Panel.APP,
)


@app.command(
    "deploy",
    help="Deploy an app.",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis="Deploy an app from a directory with a name and environment variables.",
                args="fa deploy ./my-app --name my-app-name -e KEY1=VALUE1 -e KEY2=VALUE2",
            ),
        ]
    ),
)
def deploy(
    ctx: TyperAppContext,
    app_dir_path: str = Argument(
        ...,
        help="The path to the app directory to deploy.",
    ),
    env: list[str] = Option(
        [],
        "--env",
        "-e",
        help="Environment variables to set for the app. (e.g., -e KEY=VALUE)",
    ),
    name: str | None = Option(
        None,
        "--name",
        "-n",
        help="The name of the app.",
    ),
    project_id: str | None = Option(
        None,
        "--project-id",
        "-p",
        help="The ID of the project to deploy the app to.",
    ),
) -> None:
    """Deploy an app."""
    from ..action.app_deploy import run as app_deploy

    with AppContext(ctx.obj) as app_ctx:
        app_deploy(
            app_ctx,
            app_dir_path=app_dir_path,
            env=env,
            name=name,
            project_id=project_id,
        )


@app.command(
    "update",
    help="Update an app's source archive.",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis="Update an app's source archive from a directory.",
                args="fa update <APP_ID> ./my-app",
            ),
        ]
    ),
)
def update(
    ctx: TyperAppContext,
    app_id: str = Argument(
        ...,
        help="The ID of the app to update.",
    ),
    app_dir_path: str = Argument(
        ...,
        help="The path to the app directory to update.",
    ),
) -> None:
    """Update an app."""
    from ..action.app_update import run as app_update

    with AppContext(ctx.obj) as app_ctx:
        app_update(app_ctx, app_id=app_id, app_dir_path=app_dir_path)


@app.command(
    "list",
    help="List apps.",
)
def list_agents(
    ctx: TyperAppContext,
    project_id: str | None = Option(
        None,
        "--project-id",
        "-p",
        help="The ID of the project to list apps from.",
    ),
) -> None:
    """List apps."""
    from ..action.app_list import run as app_list

    with AppContext(ctx.obj) as app_ctx:
        app_list(app_ctx, project_id=project_id)


@app.command(
    "terminate",
    help="Terminate an app.",
)
def terminate_app(
    ctx: TyperAppContext,
    app_id: str = Argument(
        ...,
        help="The ID of the app to terminate.",
    ),
) -> None:
    """Terminate an app."""
    from ..action.app_terminate import run as app_terminate

    with AppContext(ctx.obj) as app_ctx:
        app_terminate(app_ctx, app_id=app_id)


@app.command(
    "restart",
    help="Restart an app.",
)
def restart_app(
    ctx: TyperAppContext,
    app_id: str = Argument(
        ...,
        help="The ID of the app to restart.",
    ),
) -> None:
    """Restart an app."""
    from ..action.app_restart import run as app_restart

    with AppContext(ctx.obj) as app_ctx:
        app_restart(app_ctx, app_id=app_id)
