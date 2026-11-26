"""HumanSession: run GridHunt with people in the loop or on the loop.

Human in the loop: a participant controls one or more agents from the keyboard while AI
teammates control the rest. By default every step waits until each human has chosen a move,
and the time from the frame appearing to the key press is logged as the reaction time.

Human on the loop: AI agents play on their own (``step_interval`` seconds per step) and the
participant supervises with buttons, for example switching which team of policies is in
control (``teams``), pausing, or anything a ``Button`` callback does.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pygame
import pygame_gui

from ..policies import DOWN, LEFT, RIGHT, STILL, UP
from .questionnaire import Questionnaire, QuitRequested

ARROW_KEYS = {pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT, pygame.K_RETURN: STILL}
"""Arrow keys move, Return stays."""

WASD_KEYS = {pygame.K_w: UP, pygame.K_s: DOWN, pygame.K_a: LEFT, pygame.K_d: RIGHT, pygame.K_SPACE: STILL}
"""W/A/S/D move, Space stays."""

_RESET = object()  # returned by _collect_actions when reset_board() was requested


@dataclass
class Button:
    """A button in the information bar. ``on_press(session)`` runs when it is clicked."""

    label: str
    on_press: object


@dataclass
class EpisodeResult:
    episode: int
    seed: object
    steps: int
    team_return: float
    captures: int
    end_reason: str
    duration_s: float


class HumanSession:
    """Run episodes of a GridHunt environment with human participants.

    Args:
        env: a GridHunt environment, created without a render mode; the session opens the window.
        humans: ``{agent_id: key_map}`` for agents controlled from the keyboard, where a key map
            is ``{pygame_key: action}`` such as ``ARROW_KEYS`` or ``WASD_KEYS``.
        teammates: ``{agent_id: policy}`` for AI-controlled agents. A policy is called as
            ``policy(observation, agent_id, env)`` and returns an action; if it has a ``reset()``
            method it is called at the start of every episode. The scripted policies in
            ``gridhunt.policies`` and the pretrained ones in ``gridhunt.zoo`` work as is.
        teams: ``{name: {agent_id: policy}}``: alternative sets of teammates. Each team gets a
            button that puts it in control; ``initial_team`` (default: the first) starts.
        buttons: extra ``Button`` objects.
        pause_button: add a Pause button (the pause key works either way).
        info: text for the information bar, or ``callable(session) -> str``; default: episode,
            step, score and the team in control.
        log: a ``gridhunt.hat.SessionLog``, or None for no logging.
        step_interval: seconds per step. None (default when there are humans) waits for every
            human to choose; a number advances on a timer, and humans who have not pressed a
            key by then stay still.
        turn_timeout: when waiting for humans, give up after this many seconds and let them
            stay still (logged with the source ``"timeout"``). Default: wait indefinitely.
        gui_config: extra renderer settings. ``"vision_mask"`` defaults to the human's own view
            when one participant plays an agent with limited vision, and to the whole board
            otherwise; set it to None or an agent id to choose.
        info_lines: lines reserved for the information text.
        fps: frames drawn per second while waiting.
        headless: draw to a hidden window instead of showing one (tests, dry runs).
        event_source: callable returning the pending pygame events (default
            ``pygame.event.get``); scripted tests replace it.
        pause_key: key that pauses and resumes (default P).
        quit_key: key that ends the session (default Escape).
        keys: ``{pygame_key: callback}`` extra key bindings; ``callback(session)`` runs when the
            key is pressed (also while paused), like a button. For example
            ``{pygame.K_r: lambda s: s.reset_board()}``.
    """

    def __init__(self, env, humans=None, teammates=None, teams=None, initial_team=None, buttons=(),
                 pause_button=True, info=None, log=None, step_interval=None, turn_timeout=None, gui_config=None,
                 info_lines=2, fps=30, headless=False, event_source=None, pause_key=pygame.K_p,
                 quit_key=pygame.K_ESCAPE, keys=None):
        self.env = env
        self.base = env.unwrapped
        self.humans = dict(humans or {})
        self.teammates = dict(teammates or {})
        self.teams = {name: dict(policies) for name, policies in (teams or {}).items()}
        self.log = log
        self.info = info
        self.turn_timeout = turn_timeout
        self.fps = fps
        self.headless = headless
        self.event_source = event_source or pygame.event.get
        self.pause_key = pause_key
        self.quit_key = quit_key
        self.keys = dict(keys or {})
        reserved = {quit_key, pause_key}.union(*(set(key_map) for key_map in self.humans.values()))
        clashes = sorted(pygame.key.name(k) for k in set(self.keys) & reserved if k is not None)
        if clashes:
            raise ValueError(f"keys {clashes} are already used for quitting, pausing or moving")
        if step_interval is None and not self.humans:
            step_interval = 0.2
        self.step_interval = step_interval

        self._check_agents()
        self.active_team = None
        if self.teams:
            self.active_team = initial_team if initial_team is not None else next(iter(self.teams))
            if self.active_team not in self.teams:
                raise ValueError(f"initial_team {initial_team!r} is not one of {list(self.teams)}")

        self.buttons = [Button(name, lambda session, name=name: session.set_team(name)) for name in self.teams]
        if pause_button:
            self.buttons.append(Button("Pause", lambda session: session.toggle_pause()))
        self.buttons.extend(buttons)
        labels = [button.label for button in self.buttons]
        if len(labels) != len(set(labels)):
            raise ValueError(f"button labels must be unique, got {labels}")
        self._buttons_by_label = {button.label: button for button in self.buttons}

        self._attach_renderer(gui_config or {}, info_lines)
        self.episode = -1
        self.step = 0
        self.team_return = 0.0
        self.captures = 0
        self.paused = False
        self.quit_requested = False
        self.last_infos = {}
        self.last_rewards = {}
        self._end_reason = None
        self._episode_started = None
        self._reset_requested = False
        self._disabled_buttons = set()
        self._t0 = time.perf_counter()
        self._clock = pygame.time.Clock()
        self._pending_events = []
        if self.log is not None:
            self.log.update_metadata(**self._describe())
            self.log.write("session_start", **self._describe())

    # ------------------------------------------------------------------ setup

    def _check_agents(self):
        agents = set(self.base.possible_agents)
        team_agents = set().union(*self.teams.values()) if self.teams else set()
        for name, policies in self.teams.items():
            if set(policies) != team_agents:
                raise ValueError(f"every team must control the same agents; {name!r} controls {sorted(policies)}")
        groups = {"humans": set(self.humans), "teammates": set(self.teammates), "teams": team_agents}
        for first, second in (("humans", "teammates"), ("humans", "teams"), ("teammates", "teams")):
            overlap = groups[first] & groups[second]
            if overlap:
                raise ValueError(f"agents {sorted(overlap)} are in both {first} and {second}")
        controlled = groups["humans"] | groups["teammates"] | groups["teams"]
        if controlled - agents:
            raise ValueError(f"unknown agents {sorted(controlled - agents)}; the environment has {sorted(agents)}")
        if agents - controlled:
            raise ValueError(f"nobody controls {sorted(agents - controlled)}; add them to humans, teammates or teams")

    def _attach_renderer(self, gui_config, info_lines):
        from ..rendering import Renderer

        config = {**(self.base.gui_config or {}), **gui_config}
        if "vision_mask" not in gui_config:
            # Supervisors see the whole board; a participant playing one agent with limited
            # vision sees only what that agent sees
            limited = [a for a in self.humans if self.base.predators.get(a) and
                       self.base.predators[a].status.get("limited_vision")]
            config["vision_mask"] = limited[0] if len(self.humans) == 1 and limited else None
        config["info_bar"] = True
        config["buttons"] = [{"text": button.label} for button in self.buttons]
        config.setdefault("buttons_per_row", 4)
        config["info_text_lines"] = max(int(config.get("info_text_lines", 1)), info_lines)
        if self.base.renderer is not None:
            self.base.renderer.close()
        self.base.render_mode = "rgb_array" if self.headless else "gui"
        self.base.gui_config = config
        self.base.renderer = Renderer(env=self.base, gui_config=config)
        self.renderer = self.base.renderer

    def _describe(self):
        base = self.base
        return {
            "env": {
                "env_id": getattr(base, "env_id", None),
                "class": type(base).__name__,
                "grid_size": list(base.grid_size),
                "agents": list(base.possible_agents),
                "observation_mode": base.observation_mode,
                "reward_mode": base.reward_mode,
                "max_steps": base.max_steps,
            },
            "humans": sorted(self.humans),
            "teammates": {agent: _policy_name(policy) for agent, policy in self.teammates.items()},
            "teams": {name: {a: _policy_name(p) for a, p in team.items()} for name, team in self.teams.items()},
            "buttons": [button.label for button in self.buttons],
            "step_interval": self.step_interval,
            "turn_timeout": self.turn_timeout,
        }

    # -------------------------------------------------------------- controls

    def now(self):
        """Seconds since the session started (the log's clock when logging)."""
        return self.log.now() if self.log is not None else time.perf_counter() - self._t0

    def set_team(self, name):
        """Put a team of policies in control from the next step on."""
        if name not in self.teams:
            raise ValueError(f"unknown team {name!r}")
        previous, self.active_team = self.active_team, name
        for policy in self.teams[name].values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.log_event("team_switch", team=name, previous=previous)

    def toggle_pause(self):
        self.paused = not self.paused
        self.log_event("pause" if self.paused else "resume")

    def pause(self):
        """Hold the game: frames keep drawing and buttons keep working, but nothing steps."""
        if not self.paused:
            self.toggle_pause()

    def resume(self):
        """Continue after ``pause()`` or ``run_episode(start_paused=True)``."""
        if self.paused:
            self.toggle_pause()

    def set_button_enabled(self, label, enabled=True):
        """Grey out (or restore) a button; presses of a disabled button are ignored."""
        if label not in self._buttons_by_label:
            raise ValueError(f"unknown button {label!r}")
        if enabled:
            self._disabled_buttons.discard(label)
        else:
            self._disabled_buttons.add(label)
        widget = getattr(self.renderer, "buttons", {}).get(label)
        if widget is not None:
            widget.enable() if enabled else widget.disable()

    def reset_board(self):
        """Reset the environment without starting a new episode.

        Agents and prey (and entities without a fixed position) are placed again. The
        episode, step count, score and captures carry on. The environment's own reset
        rules apply, so a stock environment also re-draws state such as toxicity. Takes
        effect before the next step; logged as a ``reset_board`` event.
        """
        self._reset_requested = True

    def end_episode(self, reason="skipped"):
        """End the current episode after this frame (for example from a Skip button)."""
        self._end_reason = reason

    def quit(self):
        """End the episode and mark the session as finished."""
        self.quit_requested = True
        self._end_reason = "quit"
        self.log_event("quit")

    def log_event(self, name, **data):
        """Write an event record (if logging) tagged with the current episode and step.

        ``episode_t`` is the time in seconds since the episode started (None before the first
        episode), so the time from an episode appearing to a button press can be read directly.
        """
        if self.log is not None:
            episode_t = None if self._episode_started is None else round(self.now() - self._episode_started, 3)
            self.log.write("event", name=name, episode=self.episode, step=self.step, episode_t=episode_t, **data)

    # ------------------------------------------------------------- episodes

    def run_episode(self, seed=None, max_steps=None, stop_when=None, start_paused=False, **fields):
        """Play one episode and return an ``EpisodeResult``.

        Args:
            seed: passed to ``env.reset``.
            max_steps: end after this many steps (the environment's own limit still applies).
            stop_when: ``callable(session) -> bool`` checked after every step, for example
                ``lambda s: s.captures >= 50``.
            start_paused: show the episode but hold it until ``resume()`` is called (for
                example from a button), such as when participants rate the situation first.
            **fields: extra values stored in the episode's log records (for example a condition).
        """
        if self.quit_requested:
            return None
        observations, infos = self.env.reset(seed=seed)
        self.episode += 1
        self.step, self.team_return, self.captures = 0, 0.0, 0
        self.paused, self._end_reason, self.last_infos, self.last_rewards = False, None, infos, {}
        self._reset_requested = False
        for policy in self._all_policies():
            if hasattr(policy, "reset"):
                policy.reset()
        started = self.now()
        self._episode_started = started
        if self.log is not None:
            self.log.write("episode_start", episode=self.episode, seed=seed, team=self.active_team,
                           entities=self._entities(), **fields)
            self.log.metadata["episodes"].append({"episode": self.episode, "seed": seed, **fields})
        if start_paused:
            self.pause()

        while self.env.agents and self._end_reason is None:
            choice = self._collect_actions(observations)
            if choice is _RESET:
                observations, infos = self.env.reset()
                self._reset_requested = False
                self.last_infos = infos
                self.log_event("reset_board", positions=self._positions(), entities=self._entities())
                continue
            if choice is None:
                break
            actions, sources, reaction_ms = choice
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            common = infos.get("__common__", {})
            team_reward = common.get("team_reward", sum(rewards.values()))
            captures = common.get("captures", [])
            self.team_return += team_reward
            self.captures += len(captures)
            self.last_infos = infos
            self.last_rewards = rewards
            if self.log is not None:
                record = {
                    "episode": self.episode,
                    "step": self.step,
                    "actions": actions,
                    "sources": sources,
                    "reaction_ms": reaction_ms,
                    "team": self.active_team,
                    "rewards": rewards,
                    "team_reward": team_reward,
                    "captures": captures,
                    "terminations": terminations,
                    "truncations": truncations,
                    "positions": self._positions(),
                }
                if self.log.log_state:
                    record["state"] = self.env.state()  # includes messages when communicating
                self.log.write("step", **record)
            self.step += 1
            if max_steps is not None and self.step >= max_steps:
                self._end_reason = "max_steps"
            elif stop_when is not None and stop_when(self):
                self._end_reason = "stop_when"

        reason = self._end_reason or "done"
        self._draw()
        result = EpisodeResult(self.episode, seed, self.step, self.team_return, self.captures, reason,
                               round(self.now() - started, 3))
        if self.log is not None:
            self.log.write("episode_end", **result.__dict__)
        return result

    def ask(self, questionnaire: Questionnaire, responder=None):
        """Show a questionnaire in the window; returns ``{item_id: answer}`` (logged)."""
        if self.quit_requested:
            return {}
        try:
            answers, response_ms = questionnaire.run(
                self.renderer._screen, event_source=self.event_source, responder=responder,
                quit_key=self.quit_key, fps=self.fps, show=not self.headless,
            )
        except QuitRequested:
            self.quit()
            return {}
        if self.log is not None:
            self.log.write("questionnaire", id=questionnaire.id, episode=self.episode, answers=answers,
                           response_ms=response_ms)
        return answers

    def close(self):
        if self.log is not None:
            self.log.write("session_end", episodes=self.episode + 1, quit=self.quit_requested)
            self.log.close()
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -------------------------------------------------------------- internals

    def _all_policies(self):
        policies = list(self.teammates.values())
        for team in self.teams.values():
            policies.extend(team.values())
        return policies

    def _ai_policies(self):
        policies = dict(self.teammates)
        if self.active_team is not None:
            policies.update(self.teams[self.active_team])
        return policies

    def _entities(self):
        """Human-only entities on the board: {id: {"position", "sprite"}}."""
        entities = self.base.entities or {}
        return {
            entity_id: {"position": entity.position, "sprite": Path(str(entity.render_sprite)).name}
            for entity_id, entity in entities.items()
        }

    def _positions(self):
        positions = {agent_id: predator.position for agent_id, predator in self.base.predators.items()}
        positions.update({prey_id: prey.position for prey_id, prey in self.base.prey.items()})
        return positions

    def _collect_actions(self, observations):
        """Wait for the humans (or the timer), then return (actions, sources, reaction_ms)."""
        chosen, reaction_ms = {}, {agent: None for agent in self.humans}
        self._draw()
        shown = self.now()  # reaction times run from the frame appearing
        while True:
            for event in self._events():
                self._handle(event, chosen, reaction_ms, shown)
            if self._end_reason is not None:
                return None
            if self._reset_requested:
                return _RESET
            if self.paused:
                self._tick()
                self._draw()
                shown = self.now()  # reaction times exclude the pause
                continue
            elapsed = self.now() - shown
            if self.step_interval is None:
                if len(chosen) == len(self.humans):
                    break
                if self.turn_timeout is not None and elapsed >= self.turn_timeout:
                    break
            elif elapsed >= self.step_interval:
                break
            self._tick()
            self._draw()

        actions, sources = {}, {}
        for agent in self.humans:
            if agent in chosen:
                actions[agent], sources[agent] = chosen[agent], "human"
            else:
                timed_out = self.step_interval is None
                actions[agent], sources[agent] = STILL, "timeout" if timed_out else "none"
        for agent, policy in self._ai_policies().items():
            actions[agent] = int(policy(observations[agent], agent, self.env))
            sources[agent] = self.active_team if agent not in self.teammates else _policy_name(policy)
        actions = {agent: actions[agent] for agent in self.env.agents}
        return actions, sources, reaction_ms

    def _handle(self, event, chosen, reaction_ms, shown):
        if event.type == pygame.QUIT:
            self.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == self.quit_key:
                self.quit()
            elif event.key == self.pause_key:
                self.toggle_pause()
            elif event.key in self.keys:
                self.log_event("key", key=pygame.key.name(event.key))
                self.keys[event.key](self)
            elif not self.paused:
                for agent, key_map in self.humans.items():
                    if event.key in key_map:
                        action = key_map[event.key]
                        first = agent not in chosen
                        if first or self.step_interval is not None:
                            chosen[agent] = action
                        if first:
                            reaction_ms[agent] = round((self.now() - shown) * 1000, 1)
                        self.log_event("key", agent=agent, key=pygame.key.name(event.key), action=action,
                                       accepted=first or self.step_interval is not None)
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            button = self._buttons_by_label.get(getattr(event.ui_element, "text", None))
            if button is not None and button.label in self._disabled_buttons:
                self.log_event("button", label=button.label, accepted=False)
            elif button is not None:
                self.log_event("button", label=button.label)
                button.on_press(self)

    def _events(self):
        events = list(self.event_source())
        manager = getattr(self.renderer, "_ui_manager", None)
        for event in events:
            if manager is not None and event.type != pygame_gui.UI_BUTTON_PRESSED:
                manager.process_events(event)
        # The UI manager turns clicks into UI_BUTTON_PRESSED events on pygame's queue
        posted = pygame.event.get(pygame_gui.UI_BUTTON_PRESSED)
        if self.event_source is not pygame.event.get:
            pygame.event.clear()  # scripted input: drop the window's own events so the queue never fills
        return events + posted

    def _info_text(self):
        if callable(self.info):
            return str(self.info(self))
        if self.info is not None:
            return str(self.info)
        text = f"Episode {self.episode + 1}   Step {self.step}   Score {self.team_return:g}"
        if self.active_team is not None:
            text += f"\nIn control: {self.active_team}"
        if self.paused:
            text += "   (paused)"
        return text

    def _draw(self):
        self.base.info = self._info_text()
        manager = getattr(self.renderer, "_ui_manager", None)
        if manager is not None:
            manager.update(1.0 / self.fps)
        self.renderer.render()

    def _tick(self):
        self._clock.tick(self.fps)


def _policy_name(policy):
    return getattr(policy, "name", None) or type(policy).__name__
