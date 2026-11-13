"""Human-agent teaming toolkit: sessions, logs, questionnaires and human-only entities.

Sessions run headless (a hidden window) with scripted keyboard and button events.
"""

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")
pygame_gui = pytest.importorskip("pygame_gui")

import gridhunt  # noqa: E402
from gridhunt.hat import (  # noqa: E402
    ARROW_KEYS,
    Button,
    Choice,
    HumanSession,
    Likert,
    Message,
    Questionnaire,
    SessionLog,
    Text,
    read_log,
)
from gridhunt.policies import RIGHT, STILL, GreedyPolicy, IdlePolicy  # noqa: E402


class Script:
    """Event source returning one scripted frame per call (a list of events, or a callable)."""

    def __init__(self, frames, then=()):
        self.frames = list(frames)
        self.then = list(then)

    def __call__(self):
        frame = self.frames.pop(0) if self.frames else self.then
        return frame() if callable(frame) else list(frame)


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def button(label):
    return pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=SimpleNamespace(text=label))


def records(log, record_type):
    return [r for r in read_log(log.path) if r["type"] == record_type]


def test_human_in_the_loop_moves_follow_keys_and_reaction_times_are_logged(tmp_path):
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    log = SessionLog(tmp_path, participant="P01", condition="hitl")
    session = HumanSession(
        env, humans={"predator_0": ARROW_KEYS}, teammates={"predator_1": GreedyPolicy()},
        log=log, headless=True, event_source=Script([], then=[key(pygame.K_RIGHT)]),
    )
    result = session.run_episode(seed=3, max_steps=6)
    x_positions = [r["positions"]["predator_0"][0] for r in records(log, "step")]
    session.close()

    assert result.steps == 6 and result.end_reason == "max_steps"
    steps = records(log, "step")
    assert len(steps) == 6
    assert all(s["actions"]["predator_0"] == RIGHT and s["sources"]["predator_0"] == "human" for s in steps)
    assert all(s["sources"]["predator_1"] == "GreedyPolicy" for s in steps)
    assert all(s["reaction_ms"]["predator_0"] >= 0 for s in steps)
    assert all(len(s["state"]) == env.state_space.shape[0] for s in steps)
    assert x_positions == sorted(x_positions) and x_positions[-1] == env.grid_size[0] - 1
    keys = [e for e in records(log, "event") if e["name"] == "key"]
    assert len(keys) == 6 and keys[0]["key"] == "right"

    meta = json.loads(log.meta_path.read_text())
    assert meta["participant"] == "P01" and meta["condition"] == "hitl"
    assert meta["env"]["env_id"] == "GridHunt-ClassicStag-v0"
    assert meta["versions"]["gridhunt"] == gridhunt.__version__
    assert meta["episodes"] == [{"episode": 0, "seed": 3}]
    assert "ended" in meta


def test_turn_timeout_lets_a_silent_human_stay_still(tmp_path):
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    log = SessionLog(tmp_path)
    with HumanSession(env, humans={"predator_0": ARROW_KEYS}, teammates={"predator_1": IdlePolicy()},
                      turn_timeout=0.0, log=log, headless=True, event_source=Script([])) as session:
        session.run_episode(seed=0, max_steps=3)
    steps = records(log, "step")
    assert [s["sources"]["predator_0"] for s in steps] == ["timeout"] * 3
    assert all(s["actions"]["predator_0"] == STILL and s["reaction_ms"]["predator_0"] is None for s in steps)


def _toxic_teams():
    agents = ["predator_0", "predator_1", "predator_2"]
    return {
        "StagTeam": {a: GreedyPolicy(prey_types=("stag",), avoid_harmful=False) for a in agents},
        "HareTeam": {a: GreedyPolicy(prey_types=("hare",)) for a in agents},
    }


def test_human_on_the_loop_switches_teams_with_buttons(tmp_path):
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    log = SessionLog(tmp_path)
    session = HumanSession(env, teams=_toxic_teams(), step_interval=0.0, log=log, headless=True,
                           event_source=Script([[], [], [], [button("HareTeam")]]))
    session.run_episode(seed=1, max_steps=8)
    session.close()

    teams = [s["team"] for s in records(log, "step")]
    assert teams == ["StagTeam"] * 3 + ["HareTeam"] * 5
    assert records(log, "step")[-1]["sources"]["predator_0"] == "HareTeam"
    switches = [e for e in records(log, "event") if e["name"] == "team_switch"]
    assert switches == [dict(switches[0], team="HareTeam", previous="StagTeam", step=3)]
    assert [e["label"] for e in records(log, "event") if e["name"] == "button"] == ["HareTeam"]


def test_episode_start_records_the_cues_shown(tmp_path):
    env = gridhunt.make("GridHunt-ToxicStag-v0", entities=_berries(n=3, fixed=(0, 0)))
    log = SessionLog(tmp_path)
    with HumanSession(env, teams=_toxic_teams(), step_interval=0.0, log=log, headless=True,
                      event_source=Script([])) as session:
        session.run_episode(seed=2, max_steps=1)
    cues = records(log, "episode_start")[0]["entities"]
    assert set(cues) == {"berry_0", "berry_1", "berry_2", "marker"}
    assert cues["marker"] == {"position": [0, 0], "sprite": "plant_no_fruit.png"}


def test_a_real_mouse_click_on_a_team_button_switches_teams():
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    frames = {"count": 0}

    def click_hare_team_on_third_frame():
        frames["count"] += 1
        events = pygame.event.get()
        if frames["count"] == 3:
            sprites = session.renderer._ui_manager.get_sprite_group().sprites()
            target = next(e for e in sprites if isinstance(e, pygame_gui.elements.UIButton) and e.text == "HareTeam")
            pos = target.rect.center
            events += [
                pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)),
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1),
                pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1),
            ]
        return events

    session = HumanSession(env, teams=_toxic_teams(), step_interval=0.0, headless=True,
                           event_source=click_hare_team_on_third_frame)
    session.run_episode(seed=0, max_steps=5)
    session.close()
    assert session.active_team == "HareTeam"


def test_pause_holds_the_game_and_custom_buttons_run_their_callback(tmp_path):
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    log = SessionLog(tmp_path)
    pressed = []
    frames = [[key(pygame.K_p)], [], [], [button("Mark")], [key(pygame.K_p)]]
    session = HumanSession(env, teams=_toxic_teams(), step_interval=0.0, log=log, headless=True,
                           buttons=[Button("Mark", lambda s: pressed.append(s.step))], event_source=Script(frames))
    result = session.run_episode(seed=0, max_steps=2)
    session.close()

    names = [e["name"] for e in records(log, "event")]
    assert names == ["pause", "button", "resume"]
    assert pressed == [0]  # pressed while paused, before the first step
    assert result.steps == 2


def test_stop_when_and_skip_end_episodes(tmp_path):
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    teammates = {a: GreedyPolicy() for a in env.possible_agents}
    session = HumanSession(env, teammates=teammates, step_interval=0.0, headless=True, event_source=Script([]),
                           buttons=[Button("Skip", lambda s: s.end_episode())])
    result = session.run_episode(seed=0, stop_when=lambda s: s.captures >= 3)
    assert result.end_reason == "stop_when" and result.captures >= 3

    session.event_source = Script([[], [button("Skip")]])
    result = session.run_episode(seed=1)
    assert result.end_reason == "skipped" and result.steps == 1
    session.close()


def test_quit_key_ends_the_session(tmp_path):
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    log = SessionLog(tmp_path)
    session = HumanSession(env, humans={"predator_0": ARROW_KEYS}, teammates={"predator_1": IdlePolicy()},
                           log=log, headless=True, event_source=Script([[key(pygame.K_ESCAPE)]]))
    assert session.run_episode(seed=0).end_reason == "quit"
    assert session.quit_requested and session.run_episode(seed=1) is None
    session.close()
    assert records(log, "session_end")[0]["quit"] is True


def test_session_rejects_agents_controlled_twice_or_not_at_all():
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    with pytest.raises(ValueError, match="nobody controls"):
        HumanSession(env, humans={"predator_0": ARROW_KEYS}, headless=True)
    with pytest.raises(ValueError, match="both humans and teammates"):
        HumanSession(env, humans={"predator_0": ARROW_KEYS},
                     teammates={"predator_0": IdlePolicy(), "predator_1": IdlePolicy()}, headless=True)
    with pytest.raises(ValueError, match="unknown agents"):
        HumanSession(env, teammates={"predator_0": IdlePolicy(), "predator_1": IdlePolicy(), "ghost": IdlePolicy()},
                     headless=True)


def test_questionnaire_answers_and_response_times_are_logged(tmp_path):
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    log = SessionLog(tmp_path)
    questionnaire = Questionnaire(
        [
            Message("Thanks for playing.", id="intro"),
            Likert("trust", "I trusted my teammate.", points=7),
            Choice("plan", "What did you aim for?", ["Stag", "Hares"]),
            Text("comments", "Any comments?"),
        ],
        title="After the round",
    )

    def type_and_submit():
        questionnaire.current_widgets["entry"].set_text("  fine  ")
        return [button("Next")]

    frames = [[button("Continue")], [button("9"), button("5")], [], [button("Hares")], type_and_submit]
    session = HumanSession(env, teammates={a: IdlePolicy() for a in env.possible_agents}, log=log, headless=True,
                           event_source=Script(frames))
    answers = session.ask(questionnaire)
    session.close()

    assert answers == {"trust": 5, "plan": "Hares", "comments": "fine"}
    logged = records(log, "questionnaire")[0]
    assert logged["answers"] == answers and logged["id"] == "After the round"
    assert set(logged["response_ms"]) == {"intro", "trust", "plan", "comments"}


def test_questionnaire_responder_and_validation():
    with pytest.raises(ValueError, match="unique"):
        Questionnaire([Likert("a", "x"), Choice("a", "y", ["1"])])
    with pytest.raises(ValueError, match="options"):
        Questionnaire([Choice("a", "y", [])])
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    session = HumanSession(env, teammates={a: IdlePolicy() for a in env.possible_agents}, headless=True)
    answers = session.ask(Questionnaire([Likert("trust", "?"), Text("t", "?")]),
                          responder=lambda item: 4 if isinstance(item, Likert) else "ok")
    session.close()
    assert answers == {"trust": 4, "t": "ok"}


def test_log_forwards_records_and_never_overwrites(tmp_path):
    seen = []
    first = SessionLog(tmp_path, participant="P/01", on_record=seen.append)
    second = SessionLog(tmp_path, participant="P/01")
    first.write("event", name="marker", value=np.int64(3), where=(1, 2))
    first.close()
    second.close()
    assert first.path != second.path and "/" not in first.path.name
    assert seen[0]["value"] == 3 and seen[0]["where"] == [1, 2]
    assert read_log(first.path)[0]["name"] == "marker"


# ----------------------------------------------------------------- entities


def _berries(n=4, fixed=None):
    berries = [gridhunt.Entity(f"berry_{i}") for i in range(n)]
    if fixed is not None:
        berries.append(gridhunt.Entity("marker", position=fixed))
    return berries


def _trajectory(env, seed, steps=60):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = [copy.deepcopy(obs), env.state().copy()]
    for _ in range(steps):
        if not env.agents:
            obs, _ = env.reset()
        actions = {a: int(rng.integers(5)) for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        out += [copy.deepcopy(obs), copy.deepcopy(rewards), env.state().copy()]
    return out


def test_human_only_entities_never_change_what_the_agents_experience():
    for env_id in ("GridHunt-ToxicStag-v0", "GridHunt-Specialists-v0", "GridHunt-Military-v0"):
        plain = _trajectory(gridhunt.make(env_id, max_steps=25), seed=5)
        with_cues = gridhunt.make(env_id, max_steps=25, entities=_berries(fixed=(0, 0)))
        cued = _trajectory(with_cues, seed=5)
        assert len(plain) == len(cued)
        for a, b in zip(plain, cued, strict=True):
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a:
                    np.testing.assert_array_equal(a[k], b[k])
            else:
                np.testing.assert_array_equal(a, b)


def test_entities_are_placed_reproducibly_and_keep_fixed_positions():
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    env.set_entities(_berries(fixed=(2, 3)))
    env.reset(seed=7)
    first = {k: e.position for k, e in env.entities.items()}
    env.reset(seed=7)
    assert first == {k: e.position for k, e in env.entities.items()}
    assert first["marker"] == (2, 3)
    assert len(set(first.values())) == len(first)
    env.reset(seed=8)
    assert first != {k: e.position for k, e in env.entities.items()}


def test_entities_are_drawn_for_viewers():
    plain = gridhunt.make("GridHunt-ToxicStag-v0", render_mode="rgb_array", gui_config={"vision_mask": None})
    cued = gridhunt.make("GridHunt-ToxicStag-v0", render_mode="rgb_array", gui_config={"vision_mask": None},
                         entities=_berries())
    plain.reset(seed=0)
    cued.reset(seed=0)
    assert not np.array_equal(plain.render(), cued.render())
    plain.close()
    cued.close()


# ------------------------------------------------------------------ policies


def test_greedy_teams_follow_their_prey_preferences():
    # Predators passing over a stag together capture it, so incidental captures still happen
    def captures(policy, toxic_probability):
        env = gridhunt.make("GridHunt-ToxicStag-v0", max_steps=150)
        env.set_prey([gridhunt.ToxicStag("stag_0", toxic_probability=toxic_probability),
                      gridhunt.Hare("hare_0"), gridhunt.Hare("hare_1")])
        env.reset(seed=0)
        kinds, team_return = [], 0.0
        while env.agents:
            _, _, _, _, infos = env.step({a: policy(None, a, env) for a in env.agents})
            kinds += [c["type"] for c in infos["__common__"]["captures"]]
            team_return += infos["__common__"]["team_reward"]
        return kinds.count("hare"), kinds.count("stag"), team_return

    hares, stags, _ = captures(GreedyPolicy(prey_types=("hare",)), 0.0)
    assert hares >= 10 * stags
    hares, stags, reckless = captures(GreedyPolicy(prey_types=("stag",), avoid_harmful=False), 1.0)
    assert stags > hares and reckless < 0
    hares, stags, informed = captures(GreedyPolicy(), 1.0)  # skips stags it knows are toxic
    assert hares >= 10 * stags and informed > 0


def test_set_prey_applies_from_the_next_reset_and_keeps_ids():
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    env.set_prey([gridhunt.ToxicStag("stag_0", toxic_probability=1.0), gridhunt.Hare("hare_0"),
                  gridhunt.Hare("hare_1")])
    for seed in range(5):
        env.reset(seed=seed)
        assert env.prey["stag_0"].status.get("toxic")
    with pytest.raises(ValueError, match="same prey ids"):
        env.set_prey([gridhunt.Hare("hare_0")])


@pytest.mark.parametrize("example", ["hitl_play_with_ai.py", "hotl_policy_delegation.py"])
def test_examples_run_end_to_end(example, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parent.parent / "examples" / example
    subprocess.run([sys.executable, str(script), "--dry-run", "--log-dir", str(tmp_path)], check=True,
                   capture_output=True, env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
    logs = list(tmp_path.glob("*.jsonl"))
    assert len(logs) == 1
    types = [r["type"] for r in read_log(logs[0])]
    assert types[0] == "session_start" and types[-1] == "session_end"
    assert types.count("episode_end") >= 2 and "questionnaire" in types and "step" in types
