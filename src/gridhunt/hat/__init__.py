"""Human-agent teaming toolkit: run GridHunt with people in or on the loop.

- :class:`HumanSession`: the study window, keyboard control of agents, AI teammates, buttons
  that switch teams of policies, pausing, reaction times.
- :class:`SessionLog`: one JSON Lines file per participant session plus a metadata file.
- :class:`Questionnaire` with :class:`Likert`, :class:`Choice`, :class:`Text` and
  :class:`Message` items, shown between episodes.

Needs the rendering dependencies: ``pip install "gridhunt[hat]"``.
"""

try:
    import pygame  # noqa: F401
    import pygame_gui  # noqa: F401
except ImportError as error:  # pragma: no cover
    raise ImportError("gridhunt.hat needs the rendering dependencies: pip install 'gridhunt[hat]'") from error

from .log import SessionLog, read_log
from .questionnaire import Choice, Likert, Message, Questionnaire, Text
from .session import ARROW_KEYS, WASD_KEYS, Button, EpisodeResult, HumanSession

__all__ = [
    "ARROW_KEYS",
    "WASD_KEYS",
    "Button",
    "Choice",
    "EpisodeResult",
    "HumanSession",
    "Likert",
    "Message",
    "Questionnaire",
    "SessionLog",
    "Text",
    "read_log",
]
