# Questionnaires

Questionnaires are shown in the study window between episodes, one item at a time. Every answer
and its response time go into the session log.

```python
from gridhunt.hat import Choice, Likert, Message, Questionnaire, Text

after_round = Questionnaire(
    [
        Message("Please answer a few questions about the last round."),
        Likert("trust", "I trusted the AI teammates.", points=7,
               anchors=("Strongly disagree", "Strongly agree")),
        Choice("strategy", "Which team did you rely on most?", ["Stag team", "Hare team"]),
        Text("comments", "Any comments?", required=False),
    ],
    title="About this round",
)
answers = session.ask(after_round)   # {"trust": 5, "strategy": "Hare team", "comments": "..."}
```

| Item | Shows | Answer |
|---|---|---|
| `Message(text, button="Continue")` | text and one button, for instructions and breaks | none |
| `Likert(id, prompt, points=7, anchors=(low, high))` | a row of numbered buttons with labels under the ends | an integer from 1 to `points` |
| `Scale(id, prompt, low=0, high=100, step=5, anchors=(low, high))` | a slider with labels under the ends and a Next button that works once the handle has moved | the value, rounded to a multiple of `step` |
| `Choice(id, prompt, options)` | one button per option | the option text |
| `Text(id, prompt, required=False)` | a text field and a Next button | the text entered |

A `Scale` suits visual analogue ratings such as the NASA-TLX scales (0 to 100 in steps of 5). Its
handle starts in the middle, and because Next only works after it has moved, no answer is ever
recorded by default. Pass `show_value=True` to show the number next to the slider.

Item ids must be unique within a questionnaire. The title, if given, is shown above every item with
its number and is stored as the questionnaire's id in the log.

Change the text size with `Questionnaire(..., font_size=24)`, or restyle the screens completely
with a [pygame_gui theme](https://pygame-gui.readthedocs.io/en/latest/theme_guide.html) (`theme=`).
Pressing Escape or closing the window ends the session.

The log records each answer and `response_ms`, the time from the item appearing to the answer. A
`Message` with an `id` also records how long the participant spent on it.
