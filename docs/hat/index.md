# Human-agent teaming

`gridhunt.hat` runs GridHunt with people taking part, for human factors and human-AI teaming
research. Install it with `pip install "gridhunt[hat]"`.

- **Human in the loop:** a participant controls one or more agents from the keyboard while AI
  teammates control the rest. Every move is logged with its reaction time.
- **Human on the loop:** AI agents play on their own and the participant supervises, for example
  by choosing which team of policies is in control, pausing, or anything a custom button does.
- **AI teammates** are any function `policy(observation, agent_id, env) -> action`: the scripted
  policies in `gridhunt.policies`, the trained benchmark teams in [`gridhunt.zoo`](../benchmark/zoo.md)
  (numpy only, no PyTorch), or your own.
- **Cues only the participant sees:** entities are drawn on the board but never observed by the
  agents and never change the game ([custom scenarios](../guides/custom-scenarios.md#human-only-entities)).
- **Questionnaires** between rounds: Likert scales, multiple choice, free text and instruction
  screens.
- **Logs:** one JSON Lines file per participant session, written as it happens, plus a metadata
  file with versions, condition and seeds.

```python
import gridhunt
from gridhunt.hat import ARROW_KEYS, HumanSession, Likert, Questionnaire, SessionLog
from gridhunt.policies import GreedyPolicy

env = gridhunt.make("GridHunt-ClassicStag-v0")
session = HumanSession(
    env,
    humans={"predator_0": ARROW_KEYS},           # the participant plays this hunter
    teammates={"predator_1": GreedyPolicy()},    # a scripted AI plays the other
    log=SessionLog("hat_logs", participant="P01", condition="scripted_partner"),
)
with session:
    for round_ in range(3):
        session.run_episode(seed=round_, max_steps=60)
        session.ask(Questionnaire([Likert("trust", "I trusted my AI teammate.")]))
```

Two complete studies are in the repository's `examples/` folder:

- `hitl_play_with_ai.py`: a stag hunt with an AI partner, two questions after each round;
- `hotl_policy_delegation.py`: the participant watches three AI hunters in Toxic stag and chooses
  which team plays (one that hunts stags, one that hunts only hares, one that avoids toxic stags),
  guided by berries on the board that only they can see.

Both run headless with `--dry-run`, which is how the test suite checks them.

Next: [sessions](sessions.md), [logs](logs.md), [questionnaires](questionnaires.md) and a checklist
for [running a study](running-a-study.md).
