"""Human on the loop: choose which AI team plays, from cues only you can see.

Three AI predators hunt in ToxicStag. In each round the stag is toxic (worth -5) with some
probability, and toxic stags look like healthy ones. Berries on the board hint at that
probability: the more yellow berries, the likelier a toxic stag. The berries are drawn for the
participant only; the agents never observe them and they do not change the game.

The participant supervises by choosing which team is in control:

- Stag team: hunts the stag whether or not it is toxic (best when stags are safe);
- Hare team: hunts only hares (safe but slow);
- Adaptive team: hunts the stag only when it is healthy (it can tell, the participant cannot).

Every switch, the team in control at each step and the round scores are logged, followed by
a short questionnaire after each round.

    python examples/hotl_policy_delegation.py --participant P01
    python examples/hotl_policy_delegation.py --dry-run      # headless, automatic choices

Logs go to hat_logs/ (one .jsonl file and one _meta.json file per session).
"""

import argparse
import random

import gridhunt
from gridhunt.agents import Entity, Hare, ToxicStag
from gridhunt.assets import BLUE_BERRY_SPRITE, RED_BERRY_SPRITE, STAG_SPRITE, YELLOW_BERRY_SPRITE
from gridhunt.hat import Button, Choice, HumanSession, Likert, Message, Questionnaire, SessionLog
from gridhunt.policies import GreedyPolicy

# toxic probability -> how many of the 7 berries are yellow (drawn uniformly from the range)
BERRY_CUES = {0.0: (0, 2), 0.5: (1, 4), 0.9: (3, 6)}
TOTAL_BERRIES = 7
ROUNDS = [0.0, 0.5, 0.9, 0.0, 0.5, 0.9, 0.0, 0.5, 0.9]
PREDATORS = ["predator_0", "predator_1", "predator_2"]


def berries(toxic_probability, rng):
    low, high = BERRY_CUES[toxic_probability]
    yellow = rng.randint(low, high)
    sprites = [YELLOW_BERRY_SPRITE] * yellow
    sprites += [rng.choice([RED_BERRY_SPRITE, BLUE_BERRY_SPRITE]) for _ in range(TOTAL_BERRIES - yellow)]
    rng.shuffle(sprites)
    return [Entity(f"berry_{i}", render_sprite=sprite) for i, sprite in enumerate(sprites)]


def prey(toxic_probability):
    # Toxic stags look like healthy ones, so only the berries hint at the risk
    return [
        ToxicStag("stag_0", toxic_probability=toxic_probability, toxic_render_sprite=STAG_SPRITE),
        Hare("hare_0"),
        Hare("hare_1"),
    ]


def teams():
    return {
        "Stag team": {a: GreedyPolicy(prey_types=("stag",), avoid_harmful=False) for a in PREDATORS},
        "Hare team": {a: GreedyPolicy(prey_types=("hare",)) for a in PREDATORS},
        "Adaptive team": {a: GreedyPolicy() for a in PREDATORS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant", default="anonymous")
    parser.add_argument("--condition", default="berry_cues")
    parser.add_argument("--steps", type=int, default=300, help="steps per round")
    parser.add_argument("--step-interval", type=float, default=0.15, help="seconds per step")
    parser.add_argument("--seed", type=int, default=0, help="seeds the rounds and the berry cues")
    parser.add_argument("--log-dir", default="hat_logs")
    parser.add_argument("--dry-run", action="store_true", help="headless, short rounds, automatic answers")
    args = parser.parse_args()
    rounds = ROUNDS[:2] if args.dry_run else ROUNDS
    steps = 20 if args.dry_run else args.steps
    cue_rng = random.Random(args.seed)

    env = gridhunt.make("GridHunt-ToxicStag-v0", prey=prey(rounds[0]))
    log = SessionLog(args.log_dir, participant=args.participant, condition=args.condition,
                     metadata={"rounds": rounds, "berry_cues": {str(k): v for k, v in BERRY_CUES.items()}})
    round_state = {"index": 0}
    session = HumanSession(
        env,
        teams=teams(),
        buttons=[Button("Skip round", lambda s: s.end_episode("skipped"))],
        info=lambda s: (f"Round {round_state['index'] + 1}/{len(rounds)}   Step {s.step}/{steps}   "
                        f"Score {s.team_return:g}\nIn control: {s.active_team}" + ("   (paused)" if s.paused else "")),
        log=log,
        step_interval=0.0 if args.dry_run else args.step_interval,
        headless=args.dry_run,
        event_source=(lambda: []) if args.dry_run else None,
    )
    auto = None
    if args.dry_run:
        def auto(item):
            return 4 if isinstance(item, Likert) else item.options[0] if isinstance(item, Choice) else True

    intro = Questionnaire([Message(
        "Three AI hunters are playing. Some stags are toxic and cost points, but toxic stags look healthy.\n\n"
        "The berries are a hint that only you can see: more yellow berries mean a toxic stag is more likely.\n\n"
        "Use the buttons to choose which team plays. You can switch at any time."
    )])
    after_round = Questionnaire(
        [
            Choice("risk", "How likely was a toxic stag in this round?", ["Unlikely", "Possible", "Very likely"]),
            Likert("confidence", "I am confident I chose the right team.", points=7),
        ],
        title="About this round",
    )

    with session:
        session.ask(intro, responder=auto)
        for index, toxic_probability in enumerate(rounds):
            round_state["index"] = index
            env.set_prey(prey(toxic_probability))
            env.set_entities(berries(toxic_probability, cue_rng))
            result = session.run_episode(seed=args.seed * 1000 + index, max_steps=steps,
                                         round=index + 1, toxic_probability=toxic_probability)
            if result is None or session.quit_requested:
                break
            print(f"Round {index + 1} (P(toxic)={toxic_probability}): score {result.team_return:g}")
            session.ask(after_round, responder=auto)
    print(f"Log: {log.path}")


if __name__ == "__main__":
    main()
