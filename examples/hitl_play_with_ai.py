"""Human in the loop: play a stag hunt with an AI teammate.

The participant controls the blue hunter with the arrow keys (Return stays put); a scripted
AI teammate controls the other. The stag is worth more but needs both hunters at once; hares
can be caught alone. Each round waits for the participant's move, logs every move with its
reaction time, and ends with two questions about the teammate.

    python examples/hitl_play_with_ai.py --participant P01
    python examples/hitl_play_with_ai.py --dry-run      # headless, automatic answers

Logs go to hat_logs/ (one .jsonl file and one _meta.json file per session).
"""

import argparse

import gridhunt
from gridhunt.hat import ARROW_KEYS, HumanSession, Likert, Message, Questionnaire, SessionLog
from gridhunt.policies import GreedyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant", default="anonymous")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60, help="steps per round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="hat_logs")
    parser.add_argument("--dry-run", action="store_true", help="headless, with automatic moves and answers")
    args = parser.parse_args()

    env = gridhunt.make("GridHunt-ClassicStag-v0")
    log = SessionLog(args.log_dir, participant=args.participant, condition="hitl_stag_hunt",
                     metadata={"rounds": args.rounds, "steps_per_round": args.steps})
    session = HumanSession(
        env,
        humans={"predator_0": ARROW_KEYS},
        teammates={"predator_1": GreedyPolicy()},
        log=log,
        turn_timeout=0.0 if args.dry_run else None,
        headless=args.dry_run,
        event_source=(lambda: []) if args.dry_run else None,
    )
    auto = (lambda item: 4 if isinstance(item, Likert) else True) if args.dry_run else None

    intro = Questionnaire([Message(
        "You are the blue hunter. Move with the arrow keys; press Return to stay put.\n\n"
        "The stag is worth 10 but needs both hunters at once. A hare is worth 1 and can be caught alone.\n\n"
        "Each round lasts a fixed number of moves. Try to score as much as you can as a team."
    )])
    after_round = Questionnaire(
        [
            Likert("trust", "I trusted my AI teammate.", points=7),
            Likert("predictable", "I could predict what my AI teammate would do.", points=7),
        ],
        title="About this round",
    )

    with session:
        session.ask(intro, responder=auto)
        for round_index in range(args.rounds):
            result = session.run_episode(seed=args.seed + round_index, max_steps=args.steps, round=round_index + 1)
            if result is None or session.quit_requested:
                break
            print(f"Round {round_index + 1}: team score {result.team_return:g} in {result.steps} moves")
            session.ask(after_round, responder=auto)
    print(f"Log: {log.path}")


if __name__ == "__main__":
    main()
