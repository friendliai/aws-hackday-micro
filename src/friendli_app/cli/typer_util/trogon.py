# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Typer utilities."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from typer.core import TyperGroup

if TYPE_CHECKING:
    from typing import LiteralString

    from typer import Context, Typer


def run_trogon(app: Typer, ctx: Context, *, ignore_names: Collection[LiteralString]) -> None:
    """Run trogon interactive command viewer."""
    from trogon import Trogon  # type: ignore[import-untyped]
    from typer.main import get_group

    group = get_group(app)
    ignore_names = frozenset(ignore_names)
    commands = [c for c in group.commands.values() if not c.hidden and c.name not in ignore_names]

    group = TyperGroup(
        name=group.name,
        commands=commands,
        rich_help_panel=group.rich_help_panel,
        rich_markup_mode=group.rich_markup_mode,
    )
    Trogon(group, click_context=ctx).run()
