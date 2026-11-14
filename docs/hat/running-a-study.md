# Running a study

A checklist for studies built with `gridhunt.hat`.

## Before data collection

- **Ethics.** Get approval from your ethics board before recruiting. The logs contain no names,
  only the participant id you choose; keep the key linking ids to people separately.
- **Fix the software.** Record the GridHunt version (`gridhunt.__version__`; it is also written to
  every metadata file) and pin it, with every other package, for the whole study.
- **Fix the randomness.** Give every episode an explicit seed, and seed any cue generator of your
  own (`random.Random(seed)`), so each participant's sequence of boards and cues can be reproduced
  exactly. GridHunt's own randomness depends only on the seed you pass to `run_episode`.
- **Choose the teammates.** Scripted teammates (`gridhunt.policies`) behave the same for everyone.
  Trained teams (`gridhunt.zoo`) sample their actions; pass `rng=seed` to `zoo.load` to make that
  reproducible too. Report which model you used (scenario, algorithm, seed).
- **Decide the pacing.** Turn-based steps give clean reaction times; timed steps suit supervision.
  Note that the window's frame rate (`fps`, 30 by default) limits how finely key presses are timed.
- **Counterbalance.** Order the conditions and seeds per participant in your own script, and store
  the order with `run_episode(..., condition=..., order=...)` or `SessionLog(metadata=...)`.

## Piloting

- Run the script with `headless=True`, an automatic `event_source` and a `responder` (as the
  examples' `--dry-run` does) to check that it runs to the end and writes the log you expect.
- Run a few pilot sessions with people, then read their logs with `read_log` and check every field
  you plan to analyse before collecting real data.

## During a session

- Use the same machine, screen and keyboard for every participant, in a quiet room.
- Show instructions with `Message` items, so every participant reads the same text.
- Keep the experimenter's interventions out of the game; use `log_event` for anything notable.

## After a session

- The log is flushed after every record; the metadata file is completed when the session closes.
  Use the session as a context manager (`with session:`) so it closes even after an error.
- Back up `hat_logs/` after every session.
