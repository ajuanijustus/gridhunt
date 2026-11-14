"""Questionnaires and instruction screens shown in the study window between episodes.

Build a ``Questionnaire`` from items and pass it to ``HumanSession.ask``::

    trust = Questionnaire([
        Message("Please answer a few questions about the last round."),
        Likert("trust", "I trusted the AI teammates.", points=7,
               anchors=("Strongly disagree", "Strongly agree")),
        Choice("strategy", "Which team did you rely on most?", ["Stag team", "Hare team"]),
        Text("comments", "Any comments?", required=False),
    ])
    answers = session.ask(trust)

Items are shown one at a time. Every answer and its response time (milliseconds from the item
appearing to the answer) goes into the session log.
"""

import time
from dataclasses import dataclass, field

import pygame
import pygame_gui


@dataclass
class Message:
    """A screen of text with a single button, for instructions and breaks."""

    text: str
    button: str = "Continue"
    id: str | None = None


@dataclass
class Likert:
    """A rating on a scale of ``points`` (answer: 1..points)."""

    id: str
    prompt: str
    points: int = 7
    anchors: tuple = ("Strongly disagree", "Strongly agree")


@dataclass
class Choice:
    """One option out of several (answer: the option text)."""

    id: str
    prompt: str
    options: list = field(default_factory=list)


@dataclass
class Text:
    """Free text (answer: the text entered)."""

    id: str
    prompt: str
    required: bool = False
    button: str = "Next"


class QuitRequested(Exception):
    """The participant closed the window or pressed the quit key."""


class Questionnaire:
    """An ordered list of items: ``Message``, ``Likert``, ``Choice``, ``Text``.

    Args:
        items: the items, shown in order.
        title: shown above every item with the item number, and stored as the log's id.
        id: id stored in the log (default: the title).
        font_size: text size in pixels.
        theme: a pygame_gui theme (dict or path) to restyle the screens completely.
    """

    BACKGROUND = (20, 20, 20)
    MARGIN = 40
    BUTTON_HEIGHT = 44

    def __init__(self, items, title=None, id=None, font_size=20, theme=None):
        self.font_size = font_size
        self.theme = theme or {
            "defaults": {"font": {"name": "noto_sans", "size": str(font_size)}},
            "text_box": {"font": {"name": "noto_sans", "size": str(font_size)}},
            "label": {"font": {"name": "noto_sans", "size": str(max(12, font_size - 4))}},
            "button": {"font": {"name": "noto_sans", "size": str(max(12, font_size - 2))}},
        }
        self.items = list(items)
        self.title = title
        self.id = id or title
        ids = [item.id for item in self.items if not isinstance(item, Message)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"item ids must be unique, got {ids}")
        for item in self.items:
            if isinstance(item, Likert) and item.points < 2:
                raise ValueError(f"{item.id}: a Likert item needs at least 2 points")
            if isinstance(item, Choice) and not item.options:
                raise ValueError(f"{item.id}: a Choice item needs options")
        # Widgets of the item on screen ({"buttons": {label: element}, "entry": element}),
        # exposed so scripted tests can press them
        self.current_widgets = {}

    def run(self, screen, event_source=pygame.event.get, responder=None, quit_key=pygame.K_ESCAPE, fps=30,
            show=True):
        """Show every item and return ``(answers, response_ms)`` keyed by item id.

        responder: optional ``callable(item) -> answer`` that answers instead of the participant
            (for tests and dry runs); the items are still drawn.
        """
        answers, response_ms = {}, {}
        clock = pygame.time.Clock()
        for index, item in enumerate(self.items):
            manager = pygame_gui.UIManager(screen.get_size(), self.theme)
            self._build(manager, screen.get_size(), item, index)
            shown = time.perf_counter()
            answer = self._wait(manager, screen, item, event_source, responder, quit_key, clock, fps, show)
            if not isinstance(item, Message):
                answers[item.id] = answer
                response_ms[item.id] = round((time.perf_counter() - shown) * 1000, 1)
            elif item.id is not None:
                response_ms[item.id] = round((time.perf_counter() - shown) * 1000, 1)
        self.current_widgets = {}
        return answers, response_ms

    def _build(self, manager, size, item, index):
        width, height = size
        inner = width - 2 * self.MARGIN
        y = self.MARGIN
        buttons = {}
        if self.title:
            pygame_gui.elements.UILabel(
                pygame.Rect(self.MARGIN, y, inner, 30), f"{self.title} ({index + 1}/{len(self.items)})", manager
            )
            y += 40
        text = item.text if isinstance(item, Message) else item.prompt
        box_height = max(100, int(height * 0.25))
        pygame_gui.elements.UITextBox(text.replace("\n", "<br>"), pygame.Rect(self.MARGIN, y, inner, box_height),
                                      manager)
        y += box_height + 20

        if isinstance(item, Likert):
            gap = 8
            size_ = min(60, (inner - gap * (item.points - 1)) // item.points)
            total = item.points * size_ + gap * (item.points - 1)
            x = self.MARGIN + (inner - total) // 2
            for point in range(1, item.points + 1):
                rect = pygame.Rect(x + (point - 1) * (size_ + gap), y, size_, self.BUTTON_HEIGHT)
                buttons[str(point)] = pygame_gui.elements.UIButton(rect, str(point), manager)
            y += self.BUTTON_HEIGHT + 6
            if item.anchors:
                # Each anchor label is centred under the end point of the scale
                low, high = item.anchors
                label_width = min(inner // 2, 240)
                for text, centre in ((low, x + size_ // 2), (high, x + total - size_ // 2)):
                    left = min(max(self.MARGIN, centre - label_width // 2), self.MARGIN + inner - label_width)
                    pygame_gui.elements.UILabel(pygame.Rect(left, y, label_width, 28), text, manager)
        elif isinstance(item, Choice):
            for option in item.options:
                buttons[option] = pygame_gui.elements.UIButton(
                    pygame.Rect(self.MARGIN, y, inner, self.BUTTON_HEIGHT), option, manager
                )
                y += self.BUTTON_HEIGHT + 8
        elif isinstance(item, Text):
            entry = pygame_gui.elements.UITextEntryLine(pygame.Rect(self.MARGIN, y, inner, self.BUTTON_HEIGHT), manager)
            entry.focus()
            self.current_widgets["entry"] = entry
            y += self.BUTTON_HEIGHT + 12
            buttons[item.button] = pygame_gui.elements.UIButton(
                pygame.Rect(self.MARGIN + inner - 140, y, 140, self.BUTTON_HEIGHT), item.button, manager
            )
        else:  # Message
            buttons[item.button] = pygame_gui.elements.UIButton(
                pygame.Rect(self.MARGIN + inner - 160, y, 160, self.BUTTON_HEIGHT), item.button, manager
            )
        self.current_widgets["buttons"] = buttons

    def _answer_for(self, item, label):
        if isinstance(item, Likert):
            return int(label)
        if isinstance(item, Choice):
            return label
        if isinstance(item, Text):
            text = self.current_widgets["entry"].get_text().strip()
            if item.required and not text:
                return None
            return text
        return True

    def _wait(self, manager, screen, item, event_source, responder, quit_key, clock, fps, show):
        if responder is not None:
            self._draw(manager, screen, 0.0, show)
            return responder(item)
        while True:
            delta = clock.tick(fps) / 1000.0
            events = list(event_source())
            # The UI manager turns clicks into UI_BUTTON_PRESSED events on pygame's queue
            events += pygame.event.get(pygame_gui.UI_BUTTON_PRESSED)
            if event_source is not pygame.event.get:
                pygame.event.clear()
            for event in events:
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == quit_key):
                    raise QuitRequested
                if event.type != pygame_gui.UI_BUTTON_PRESSED:
                    manager.process_events(event)
                    continue
                label = getattr(event.ui_element, "text", None)
                if label in self.current_widgets.get("buttons", {}):
                    answer = self._answer_for(item, label)
                    if answer is not None:
                        return answer
            self._draw(manager, screen, delta, show)

    def _draw(self, manager, screen, delta, show):
        manager.update(delta)
        screen.fill(self.BACKGROUND)
        manager.draw_ui(screen)
        if show:
            pygame.display.flip()
