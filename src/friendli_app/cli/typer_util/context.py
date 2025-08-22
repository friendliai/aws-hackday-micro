# Copyright (c) 2025-present, FriendliAI Inc. All rights reserved.

"""Click context."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from click.core import BaseCommand, Command, Context


class ExtendedContext(Context):
    """Extended context for Click context."""

    def __init__(
        self,
        command: Command,
        parent: Context | None = None,
        info_name: str | None = None,
        obj: Any | None = None,
        auto_envvar_prefix: str | None = None,
        default_map: MutableMapping[str, Any] | None = None,
        terminal_width: int | None = None,
        max_content_width: int | None = None,
        resilient_parsing: bool = False,
        allow_extra_args: bool | None = None,
        allow_interspersed_args: bool | None = None,
        ignore_unknown_options: bool | None = None,
        help_option_names: list[str] | None = None,
        token_normalize_func: Callable[[str], str] | None = None,
        color: bool | None = None,
        show_default: bool | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(
            command,
            parent,
            info_name,
            obj,
            auto_envvar_prefix,
            default_map,
            terminal_width,
            max_content_width,
            resilient_parsing,
            allow_extra_args,
            allow_interspersed_args,
            ignore_unknown_options,
            help_option_names,
            token_normalize_func,
            color,
            show_default,
        )


BaseCommand.context_class = ExtendedContext  # type: ignore[attr-defined]
