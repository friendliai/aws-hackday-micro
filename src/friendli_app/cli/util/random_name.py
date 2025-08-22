"""Utilities for generating human-friendly random names.

Generates a lowercase kebab-case string in the form '{adjective}-{noun}'.
"""

from __future__ import annotations

import random

_ADJECTIVES = (
    "brave",
    "calm",
    "clever",
    "daring",
    "eager",
    "fancy",
    "gentle",
    "happy",
    "jolly",
    "kind",
    "lively",
    "mighty",
    "noble",
    "quick",
    "rapid",
    "silent",
    "smart",
    "swift",
    "tiny",
    "vivid",
    "witty",
    "zesty",
)

_NOUNS = (
    "acorn",
    "beacon",
    "breeze",
    "canyon",
    "comet",
    "dawn",
    "eagle",
    "ember",
    "forest",
    "galaxy",
    "harbor",
    "horizon",
    "island",
    "lake",
    "meadow",
    "meteor",
    "mountain",
    "nebula",
    "ocean",
    "prairie",
    "quartz",
    "rainbow",
    "river",
    "savanna",
    "sky",
    "spark",
    "spring",
    "storm",
    "summit",
    "thunder",
    "tide",
    "valley",
    "willow",
    "zephyr",
)


def generate_random_name(rng: random.Random | None = None) -> str:
    """Return a random '{adjective}-{noun}' name.

    Args:
        rng: Optional random generator for testability.

    Returns:
        A lowercase kebab-case name like 'brave-ember'.
    """
    r = rng or random
    adjective = r.choice(_ADJECTIVES)
    noun = r.choice(_NOUNS)
    return f"{adjective}-{noun}"
