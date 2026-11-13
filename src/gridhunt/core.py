import copy
import functools
import random

import numpy as np
from gymnasium.spaces import Box, Discrete, Sequence
from pettingzoo import ParallelEnv

from . import helpers
from .agents import Entity, Predator, Prey, Terrain
from .assets import TERRAIN_SPRITE_DICT

STILL = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

REWARD_MODES = ("individual", "team")
PREY_MOTIONS = ("static", "random", "flee")

MOVES = [STILL, UP, DOWN, LEFT, RIGHT]


class GridHunt(ParallelEnv):
    """Grid predator-prey environment with a PettingZoo ``ParallelEnv`` API.

    Use :func:`gridhunt.make` for the registered scenarios; construct this class (or a
    scenario subclass) directly to build a custom one.

    Args:
        grid_size: (width, height) of the grid.
        predators: list of :class:`~gridhunt.Predator` (the controllable agents).
        prey: list of :class:`~gridhunt.Prey` (or a subclass such as ``Stag`` or ``Hare``).
        max_steps: episode length; agents are truncated after this many steps.
        entities: non-capturable objects, drawn by the renderer but never observed by agents
            (for example cues shown only to a human participant).
        render_mode: ``None``, ``"console"``, ``"rgb_array"`` or ``"gui"`` (the last two need
            ``pip install "gridhunt[render]"``).
        observation_mode: one of ``observation_modes``: ``"coords"`` (prey, predator and terrain
            coordinates), ``"flat_coords"`` (prey and predator coordinates), ``"one_hot"``
            (one-hot grids; agents with ``limited_vision`` get an egocentric grid).
        gui_config: buttons, dropdowns and the information bar for ``render_mode="gui"``.
        trainable_prey: ids of prey that are controlled as agents instead of moving by rule.
        terrain_config: terrain types with passability, speed and generation settings.
        observation_encoders: extra features appended to observations: ``"agent_centric"``
            and/or ``"action_mask"``.
        use_action_mask: report each agent's valid moves in ``infos["__common__"]["action_masks"]``.
        agent_role_map, encoder_order: advanced settings for the observation encoders.
        reward_mode: ``"individual"`` (rewarded for own captures) or ``"team"`` (every
            predator receives the team reward).
        prey_motion: ``"static"``, ``"random"`` or ``"flee"`` for non-trainable prey.
        prey_move_probability: probability that a moving prey moves on a given step.
    """

    observation_modes = ("coords", "flat_coords", "one_hot")
    metadata = {"name": "gridhunt", "render_modes": ["console", "rgb_array", "gui"], "is_parallelizable": True}

    def __init__(
        self,
        grid_size,
        predators,
        prey,
        max_steps,
        entities=None,
        render_mode=None,
        action_mode="single",
        observation_mode="coords",
        gui_config=None,
        trainable_prey=None,
        terrain_config=None,
        observation_encoders: list[str] | None = None,
        use_action_mask: bool = False,
        agent_role_map: dict[str, list[str]] | None = None,
        encoder_order: list[str] | None = None,
        reward_mode: str = "individual",
        prey_motion: str = "static",
        prey_move_probability: float = 0.5,
    ):
        self.grid_size = grid_size
        # "individual": a predator is rewarded only for captures it takes part in.
        # "team": every predator receives the team reward (each capture counted once).
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode
        if observation_mode not in self.observation_modes:
            raise ValueError(
                f"observation_mode must be one of {self.observation_modes}, got {observation_mode!r}"
            )
        # How non-trainable prey move: "static" (never), "random" (random direction) or
        # "flee" (away from the nearest predator), each step with prey_move_probability
        if prey_motion not in PREY_MOTIONS:
            raise ValueError(f"prey_motion must be one of {PREY_MOTIONS}, got {prey_motion!r}")
        self.prey_motion = prey_motion
        self.prey_move_probability = prey_move_probability

        self.entities = self._entity_dict(entities)
        self.prey = {
            prey_item.prey_id if isinstance(prey_item, Prey) else prey_item: (
                prey_item if isinstance(prey_item, Prey) else Prey(prey_item)
            )
            for prey_item in prey
        }
        self.predators = {
            predator.predator_id if isinstance(predator, Predator) else predator: (
                predator if isinstance(predator, Predator) else Predator(predator)
            )
            for predator in predators
        }
        self._initial_predators = copy.deepcopy(self.predators)
        self._initial_prey = copy.deepcopy(self.prey)
        self._initial_entities = (
            copy.deepcopy(self.entities) if self.entities is not None else None
        )

        self.predator_ids = set(self.predators.keys())
        self.prey_ids = set(self.prey.keys())

        self.trainable_prey = set(trainable_prey) if trainable_prey else set()
        # Fixed agent list used internally; self._agent_ids follows PettingZoo semantics
        # (all agents after reset(), empty once the episode has ended)
        self._agent_ids = sorted(list(self.predator_ids | self.trainable_prey))
        self.agents = list(self._agent_ids)

        self.rewards_for_one_hot_encoding = {agent: 0 for agent in self._agent_ids}
        self.action_for_one_hot_encoding = {agent: 0 for agent in self._agent_ids}

        self.render_mode = render_mode
        self.action_mode = action_mode
        self.observation_mode = observation_mode

        self.gui_config = gui_config or {}

        self.step_count = 0
        self.score = 0

        self.max_steps = max_steps
        prey_rewards = [abs(prey_item.reward) for prey_item in self.prey.values()]
        self.max_reward = max(prey_rewards) if prey_rewards else 1.0

        # Per-environment random number generators, seeded by reset(seed=...), so
        # seeding one env never affects another env or the training library
        self._np_rng = np.random.RandomState()
        self._py_rng = random.Random()
        # Entities are only drawn for human viewers, so they get their own generator: adding,
        # removing or moving them never changes what the agents experience
        self._entity_rng = random.Random()

        self.terrain_config = terrain_config or {}
        self.terrain_map = {}
        self.terrain = {}
        self.terrain_at = {}

        if self.terrain_config:
            self._initialize_terrain()

        self.observation_encoders = observation_encoders or []
        self.use_action_mask = use_action_mask
        self.encoder_order = encoder_order or self.observation_encoders
        self.agent_role_map = agent_role_map

        valid_encoders = {
            "agent_centric",
            "action_mask",
        }
        invalid_encoders = set(self.observation_encoders) - valid_encoders
        if invalid_encoders:
            raise ValueError(
                f"Invalid observation_encoders: {invalid_encoders}. "
                f"Valid options: {valid_encoders}"
            )

        try:
            self.agent_types = helpers.detect_agent_types(
                self.predator_ids, self.trainable_prey, self.agent_role_map
            )
        except ValueError as e:
            raise ValueError(f"Invalid agent_role_map: {e}") from e

        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be None or one of {self.metadata['render_modes']}, got {render_mode!r}")
        self.renderer = None
        if self.render_mode is not None:
            # Rendering needs pygame (the optional "render" extra); import it only when used.
            try:
                from .rendering import Renderer
            except ImportError as error:
                raise ImportError(
                    f"render_mode={self.render_mode!r} needs the rendering dependencies: "
                    "pip install 'gridhunt[render]'"
                ) from error
            self.renderer = Renderer(env=self, gui_config=self.gui_config)

    @staticmethod
    def _entity_dict(entities):
        if not entities:
            return None
        return {
            entity.entity_id if isinstance(entity, Entity) else entity: (
                entity if isinstance(entity, Entity) else Entity(entity)
            )
            for entity in entities
        }

    def set_entities(self, entities):
        """Replace the human-only entities; they appear at the next reset().

        Entities are drawn by the renderer but are never part of observations, state() or the
        dynamics. Entities without a position are placed on a random cell at every reset.
        """
        self.entities = self._entity_dict(entities)
        self._initial_entities = copy.deepcopy(self.entities)

    def set_prey(self, prey):
        """Replace the prey settings (for example a stag's toxic probability) from the next reset().

        The new prey must have the same ids as the current ones, so observation and state
        spaces stay the same.
        """
        new = {
            item.prey_id if isinstance(item, Prey) else item: item if isinstance(item, Prey) else Prey(item)
            for item in prey
        }
        if set(new) != self.prey_ids:
            raise ValueError(f"set_prey needs the same prey ids {sorted(self.prey_ids)}, got {sorted(new)}")
        self._initial_prey = copy.deepcopy(new)
        self.prey = copy.deepcopy(new)
        rewards = [abs(item.reward) for item in new.values()]
        self.max_reward = max(rewards) if rewards else 1.0

    def render(self):
        """Draw the current state.

        Returns an RGB array of shape (height, width, 3) for ``render_mode="rgb_array"``,
        prints the grid for ``"console"`` and updates the window for ``"gui"``.
        """
        if self.renderer is None:
            return None
        return self.renderer.render()

    def close(self):
        """Close the window and release rendering resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @property
    def possible_agents(self):
        """Return PettingZoo-style agent IDs."""
        return list(self._agent_ids)

    @property
    def agent_ids(self):
        """Return agent IDs."""
        return list(self._agent_ids)

    @property
    def target_ids(self):
        """Return prey IDs."""
        return list(self.prey_ids)

    def _initialize_terrain(self):
        """(Re)generate terrain from config using this env's RNG."""
        self.terrain = {}
        self.terrain_at = {}
        self.terrain_map = helpers.generate_terrain_map(
            self.grid_size, self.terrain_config, rng=self._py_rng
        )

        terrain_id_counter = 0
        for terrain_type, positions in self.terrain_map.items():
            config = self.terrain_config[terrain_type]
            sprite = config.get(
                "sprite",
                TERRAIN_SPRITE_DICT.get(
                    terrain_type, TERRAIN_SPRITE_DICT.get("mountain")
                ),
            )

            for pos in positions:
                terrain_id = f"{terrain_type}_{terrain_id_counter}"
                terrain_id_counter += 1

                terrain_entity = Terrain(
                    terrain_id=terrain_id,
                    terrain_type=terrain_type,
                    position=pos,
                    passability=config.get("passability", 1.0),
                    speed_multiplier=config.get("speed_multiplier", 1.0),
                    console_sprite=config.get("console_sprite", "T"),
                    render_sprite=sprite,
                    status=config.get("status", {}),
                )
                self.terrain[terrain_id] = terrain_entity
                self.terrain_at[pos] = terrain_type

    def _has_procedural_terrain(self):
        """Return whether any terrain type is randomly generated (blob/line)."""
        return any(
            config.get("generation", {}).get("type", "blob") in ("blob", "line")
            for config in self.terrain_config.values()
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_rng.seed(seed)
            self._py_rng.seed(seed)
            self._entity_rng.seed(f"gridhunt-entities-{seed}")

        # New random terrain layout every episode (predefined layouts never change)
        if self.terrain_config and self._has_procedural_terrain():
            self._initialize_terrain()

        self.predators = copy.deepcopy(self._initial_predators)
        self.prey = copy.deepcopy(self._initial_prey)
        self.entities = (
            copy.deepcopy(self._initial_entities)
            if self._initial_entities is not None
            else None
        )

        self.agents = list(self._agent_ids)
        self.step_count = 0
        self.score = 0
        self.rewards_for_one_hot_encoding = {agent: 0 for agent in self._agent_ids}
        self.action_for_one_hot_encoding = {agent: 0 for agent in self._agent_ids}
        valid_coordinates = [
            (x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])
        ]
        used_positions = set()
        for pos, terrain_type in self.terrain_at.items():
            terrain_config = self.terrain_config.get(terrain_type, {})
            if terrain_config.get("passability", 1.0) == 0.0:
                used_positions.add(pos)
        if options:
            if "predator_positions" in options:
                for predator_id, pos in options["predator_positions"].items():
                    if predator_id in self.predators:
                        self.predators[predator_id].position = pos
                        used_positions.add(pos)
            if "prey_positions" in options:
                for prey_id, pos in options["prey_positions"].items():
                    if prey_id in self.prey:
                        self.prey[prey_id].position = pos
                        used_positions.add(pos)
        valid_coordinates = [
            coord for coord in valid_coordinates if coord not in used_positions
        ]
        self._py_rng.shuffle(valid_coordinates)
        for predator in self.predators.values():
            if predator.position is None:
                if not valid_coordinates:
                    raise RuntimeError(
                        "Ran out of coordinates while placing predators."
                    )
                predator.position = valid_coordinates.pop()
            else:
                used_positions.add(predator.position)
        for prey_item in self.prey.values():
            if prey_item.position is None:
                if not valid_coordinates:
                    raise RuntimeError("Ran out of coordinates while placing prey.")
                prey_item.position = valid_coordinates.pop()
            else:
                used_positions.add(prey_item.position)
        if self.entities is not None:
            cells = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
            self._entity_rng.shuffle(cells)
            for entity in self.entities.values():
                if entity.position is None:
                    entity.position = cells.pop()
        observations = self.get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}
        infos["__common__"] = {
            "prey_positions": {
                prey_id: prey_item.position for prey_id, prey_item in self.prey.items()
            }
        }

        return observations, infos

    def step(self, actions):
        self.step_count += 1

        rewards = {agent_id: 0 for agent_id in self._agent_ids}
        infos = {agent_id: {} for agent_id in self._agent_ids}

        for agent_id, action in actions.items():
            if agent_id in self.predators:
                predator = self.predators[agent_id]
                old_pos = predator.position
                new_pos = self._move(old_pos, action, agent=predator)
                predator.position = new_pos

            if agent_id in self.prey:
                prey_item = self.prey[agent_id]
                old_pos = prey_item.position
                new_pos = self._move(old_pos, action, agent=prey_item)
                prey_item.position = new_pos

        captured_prey = []
        team_reward = 0  # sum of capture rewards this step, each capture counted once
        captures = []  # one record per captured prey, taken before any respawn
        for prey_id, prey_item in list(self.prey.items()):
            predators_on_prey = [
                pred_id
                for pred_id, predator in self.predators.items()
                if predator.position == prey_item.position
            ]
            if predators_on_prey:
                total_capture_power = sum(
                    self.predators[pred_id].capture_power
                    for pred_id in predators_on_prey
                )
                if total_capture_power == prey_item.capture_power_required:
                    team_reward += prey_item.reward
                    captures.append(self._capture_record(prey_id, prey_item, overcrowded=False))
                    for pred_id in predators_on_prey:
                        rewards[pred_id] += prey_item.reward
                        infos[pred_id]["capture"] = prey_item.type
                    if prey_id in self.trainable_prey:
                        rewards[prey_id] -= abs(prey_item.reward)
                        infos[prey_id]["captured"] = True
                        captured_prey.append(prey_id)
                    if prey_item.respawn:
                        prey_item.execute_respawn(env=self)
                    else:
                        del self.prey[prey_id]
                elif total_capture_power > prey_item.capture_power_required:
                    team_reward += prey_item.reward - prey_item.overcrowding_penalty
                    captures.append(self._capture_record(prey_id, prey_item, overcrowded=True))
                    for pred_id in predators_on_prey:
                        rewards[pred_id] += (
                            prey_item.reward - prey_item.overcrowding_penalty
                        )
                        infos[pred_id]["overcrowded_capture"] = prey_item.type
                    if prey_id in self.trainable_prey:
                        rewards[prey_id] -= abs(prey_item.reward)
                        infos[prey_id]["captured"] = True
                        captured_prey.append(prey_id)
                    if prey_item.respawn:
                        prey_item.execute_respawn(env=self)
                    else:
                        del self.prey[prey_id]
            else:
                if prey_id in self.trainable_prey and prey_id not in captured_prey:
                    rewards[prey_id] += 0.1

        if self.prey_motion != "static":
            self._move_prey()

        if self.reward_mode == "team":
            for pred_id in self.predators:
                rewards[pred_id] = team_reward

        terminations = {agent_id: False for agent_id in self._agent_ids}
        env_truncation = self.step_count >= self.max_steps
        truncations = {agent_id: env_truncation for agent_id in self._agent_ids}
        if env_truncation or all(terminations.values()):
            self.agents = []

        for agent_id in self._agent_ids:
            if agent_id in actions:
                self.rewards_for_one_hot_encoding[agent_id] = rewards[agent_id]
                self.action_for_one_hot_encoding[agent_id] = actions[agent_id]

        observations = self.get_observations()

        action_masks = None
        if self.use_action_mask:
            action_masks = {}
            for agent_id in self._agent_ids:
                if agent_id in self.predators:
                    agent_pos = self.predators[agent_id].position
                elif agent_id in self.prey:
                    agent_pos = self.prey[agent_id].position
                else:
                    continue

                restricted_terrains = set()
                agent_type = self.agent_types.get(agent_id, 0)
                for terrain_type, config in self.terrain_config.items():
                    restricted_types = config.get("restricted_to", set())
                    if agent_type in restricted_types:
                        restricted_terrains.add(terrain_type)

                try:
                    mask = helpers.create_action_mask(
                        agent_pos,
                        self.grid_size,
                        restricted_terrains=restricted_terrains,
                        terrain_at=self.terrain_at,
                        terrain_config=self.terrain_config,
                    )
                    helpers.validate_action_mask(mask, agent_id)
                    action_masks[agent_id] = mask
                except (ValueError, RuntimeError) as e:
                    raise RuntimeError(
                        f"Failed to generate action mask for agent {agent_id}: {e}"
                    ) from e

        infos["__common__"] = {
            "prey_positions": {
                prey_id: prey_item.position for prey_id, prey_item in self.prey.items()
            }
        }

        if action_masks:
            infos["__common__"]["action_masks"] = action_masks

        infos["__common__"]["team_reward"] = team_reward
        infos["__common__"]["captures"] = captures
        self.score += team_reward

        return observations, rewards, terminations, truncations, infos

    @functools.cache
    def action_space(self, agent):
        if self.action_mode == "single":
            return Discrete(5)
        else:
            return Sequence(Discrete(5))

    @functools.cache
    def observation_space(self, agent):
        """Return agent observation space."""
        num_predators = len(self.predator_ids)
        num_prey = len(self.prey_ids)
        num_terrain_types = len(self.terrain_config)
        grid_x, grid_y = self.grid_size

        # For one_hot mode with limited_vision, adjust grid size
        if self.observation_mode == "one_hot":
            any_limited_vision = any(
                self._has_limited_vision(
                    self.predators.get(agent_id) or self.prey.get(agent_id)
                )
                for agent_id in self.possible_agents
            )
            if any_limited_vision:
                # Use relative vision grid size instead of absolute grid size
                vision_grid_x = (grid_x - 1) * 2 + 1
                vision_grid_y = (grid_y - 1) * 2 + 1
                num_entities = num_predators + num_prey
                num_cells = vision_grid_x * vision_grid_y
                obs_size = (
                    num_entities * num_cells
                    + 1
                    + 5
                    + num_terrain_types * num_cells
                )
                
                # Account for observation encoders
                encoder_size = 0
                if "agent_centric" in self.observation_encoders:
                    encoder_size += 8
                if "action_mask" in self.observation_encoders:
                    encoder_size += 5
                obs_size += encoder_size
                
                return Box(
                    low=-1.0,
                    high=max(grid_x, grid_y),
                    shape=(obs_size,),
                    dtype=np.float32,
                )

        try:
            obs_size = helpers.calculate_observation_size(
                num_predators,
                num_prey,
                num_terrain_types,
                self.grid_size,
                self.observation_mode,
                observation_encoders=self.observation_encoders,
            )
        except ValueError as e:
            raise ValueError(f"Failed to calculate observation size: {e}") from e

        return Box(
            low=-1.0, high=max(grid_x, grid_y), shape=(obs_size,), dtype=np.float32
        )

    def _move_prey(self):
        """Move non-trainable prey after captures are resolved (never onto a predator)."""
        predator_cells = {p.position for p in self.predators.values()}
        for prey_id, prey_item in self.prey.items():
            if prey_id in self.trainable_prey or prey_item.speed <= 0:
                continue
            if self._np_rng.random() >= self.prey_move_probability:
                continue
            options = [
                (action, self._move(prey_item.position, action, agent=prey_item))
                for action in (UP, DOWN, LEFT, RIGHT)
            ]
            options = [(a, pos) for a, pos in options if pos not in predator_cells]
            if not options:
                continue
            if self.prey_motion == "flee":
                if prey_item.position not in predator_cells:
                    options.append((STILL, prey_item.position))  # staying put is allowed
                def nearest_predator(pos):
                    return min(abs(pos[0] - c[0]) + abs(pos[1] - c[1]) for c in predator_cells)
                best = max(nearest_predator(pos) for _, pos in options)
                options = [(a, pos) for a, pos in options if nearest_predator(pos) == best]
            prey_item.position = options[self._np_rng.randint(len(options))][1]

    @staticmethod
    def _capture_record(prey_id, prey_item, overcrowded):
        return {
            "prey": prey_id,
            "type": prey_item.type,
            "toxic": bool(prey_item.status.get("toxic", False)),
            "overcrowded": overcrowded,
        }

    @functools.cached_property
    def state_space(self):
        """Space of state(): see state() for the layout."""
        grid_x, grid_y = self.grid_size
        size = (
            2 * len(self._agent_ids)
            + 4 * len(self.prey_ids)
            + len(self.terrain_config) * grid_x * grid_y
        )
        return Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)

    def state(self):
        """Compact global state for centralised critics (PettingZoo ParallelEnv.state).

        Layout, all values in [0, 1]:
        - per agent (possible_agents order): x, y normalized by the grid size
        - per prey (sorted ids): x, y, present (0 once removed), toxic
        - per terrain type (sorted): a flattened one-hot grid of its cells
        """
        grid_x, grid_y = self.grid_size
        scale_x, scale_y = max(grid_x - 1, 1), max(grid_y - 1, 1)
        parts = []
        for agent_id in self._agent_ids:
            entity = self.predators.get(agent_id) or self.prey.get(agent_id)
            parts += [entity.position[0] / scale_x, entity.position[1] / scale_y]
        for prey_id in sorted(self.prey_ids):
            prey_item = self.prey.get(prey_id)
            if prey_item is None:
                parts += [0.0, 0.0, 0.0, 0.0]
            else:
                parts += [
                    prey_item.position[0] / scale_x,
                    prey_item.position[1] / scale_y,
                    1.0,
                    1.0 if prey_item.status.get("toxic", False) else 0.0,
                ]
        state = np.asarray(parts, dtype=np.float32)
        if self.terrain_config:
            grids = np.zeros((len(self.terrain_config), grid_x, grid_y), dtype=np.float32)
            type_index = {t: i for i, t in enumerate(sorted(self.terrain_config))}
            for (x, y), terrain_type in self.terrain_at.items():
                grids[type_index[terrain_type], x, y] = 1.0
            state = np.concatenate([state, grids.ravel()])
        return state

    def get_observations(self):
        if self.observation_mode == "coords":
            num_prey = len(self.prey_ids)
            num_predators = len(self.predator_ids)
            num_terrain_types = len(self.terrain_config)
            terrain_types_list = list(self.terrain_config.keys())

            obs_size = num_prey * 2 + num_predators * 2 + num_terrain_types * 2
            observation = np.zeros(obs_size, dtype=np.float32)

            idx = 0
            for prey_id in sorted(self.prey_ids):
                if prey_id in self.prey:
                    observation[idx] = self.prey[prey_id].position[0]
                    observation[idx + 1] = self.prey[prey_id].position[1]
                else:
                    observation[idx] = -1
                    observation[idx + 1] = -1
                idx += 2

            for pred_id in sorted(self.predator_ids):
                observation[idx] = self.predators[pred_id].position[0]
                observation[idx + 1] = self.predators[pred_id].position[1]
                idx += 2

            for terrain_type in terrain_types_list:
                positions = list(self.terrain_map.get(terrain_type, []))
                if positions:
                    avg_x = np.mean([p[0] for p in positions])
                    avg_y = np.mean([p[1] for p in positions])
                    observation[idx] = avg_x
                    observation[idx + 1] = avg_y
                else:
                    observation[idx] = -1
                    observation[idx + 1] = -1
                idx += 2

            obs_dict = {}
            for agent in self._agent_ids:
                agent_obs = observation.copy()
                agent_obs = self._apply_encoders(agent, agent_obs)
                obs_dict[agent] = agent_obs
            return obs_dict

        elif self.observation_mode == "flat_coords":
            num_prey = len(self.prey_ids)
            num_predators = len(self.predator_ids)
            num_entities = num_prey + num_predators
            observation = np.zeros((num_entities, 2), dtype=np.float32)
            for i, prey_id in enumerate(sorted(self.prey_ids)):
                if prey_id in self.prey:
                    observation[i] = self.prey[prey_id].position
                else:
                    observation[i] = [-1, -1]
            for i, pred_id in enumerate(sorted(self.predator_ids)):
                observation[num_prey + i] = self.predators[pred_id].position

            obs_dict = {}
            for agent in self._agent_ids:
                agent_obs = observation.copy().flatten()
                agent_obs = self._apply_encoders(agent, agent_obs)
                obs_dict[agent] = agent_obs
            return obs_dict

        elif self.observation_mode == "one_hot":
            rewards_normalized = [
                np.clip(self.rewards_for_one_hot_encoding.get(agent, 0) / self.max_reward, -1.0, 1.0)
                for agent in self.possible_agents
            ]
            
            # Check if any agent uses limited_vision
            any_limited_vision = any(
                self._has_limited_vision(
                    self.predators.get(agent_id) or self.prey.get(agent_id)
                )
                for agent_id in self.possible_agents
            )

            if any_limited_vision:
                # Build per-agent observations with limited/full vision support
                obs_dict = {}
                for agent_idx, agent_id in enumerate(self.possible_agents):
                    agent_obj = (
                        self.predators.get(agent_id)
                        if agent_id in self.predators
                        else self.prey.get(agent_id)
                    )
                    agent_pos = agent_obj.position
                    agent_has_limited_vision = self._has_limited_vision(agent_obj)
                    agent_vision_radius = self._get_agent_vision_radius(agent_obj)

                    entity_grids = []

                    # Encode all predators - use relative positioning for all agents
                    for pred_id in sorted(self.predator_ids):
                        relative_pos = self._get_relative_position(
                            agent_pos, self.predators[pred_id].position
                        )
                        if agent_has_limited_vision:
                            entity_grid = self._get_one_hot_matrix_limited_vision(
                                relative_pos, agent_vision_radius
                            )
                        else:
                            # For agents without limited_vision, use full_vision in relative space
                            entity_grid = self._get_one_hot_matrix_full_vision(
                                relative_pos
                            )
                        entity_grids.append(entity_grid)

                    # Encode prey - use relative positioning for all agents
                    width = (self.grid_size[0] - 1) * 2 + 1
                    height = (self.grid_size[1] - 1) * 2 + 1
                    for prey_id in sorted(self.prey_ids):
                        if prey_id in self.prey:
                            relative_pos = self._get_relative_position(
                                agent_pos, self.prey[prey_id].position
                            )
                            if agent_has_limited_vision:
                                entity_grid = self._get_one_hot_matrix_limited_vision(
                                    relative_pos, agent_vision_radius
                                )
                            else:
                                entity_grid = self._get_one_hot_matrix_full_vision(
                                    relative_pos
                                )
                        else:
                            # Prey captured or dead - use empty grid in relative space
                            entity_grid = np.zeros(width * height, dtype=np.float32)
                        entity_grids.append(entity_grid)

                    # Encode terrain - use relative positioning for all agents
                    terrain_grids = []
                    for terrain_type in sorted(self.terrain_config.keys()):
                        terrain_positions = self.terrain_map.get(terrain_type, [])
                        terrain_grid = np.zeros(width * height, dtype=np.float32)
                        for pos in terrain_positions:
                            relative_pos = self._get_relative_position(
                                agent_pos, pos
                            )
                            if agent_has_limited_vision:
                                # Only include terrain within vision radius
                                if (
                                    abs(relative_pos[0]) <= agent_vision_radius
                                    and abs(relative_pos[1]) <= agent_vision_radius
                                ):
                                    matrix_x = relative_pos[0] + self.grid_size[0] - 1
                                    matrix_y = relative_pos[1] + self.grid_size[1] - 1
                                    if (
                                        0 <= matrix_x < width
                                        and 0 <= matrix_y < height
                                    ):
                                        index = matrix_x * height + matrix_y
                                        terrain_grid[index] = 1.0
                            else:
                                # For agents without limited_vision, include all terrain in relative space
                                matrix_x = relative_pos[0] + self.grid_size[0] - 1
                                matrix_y = relative_pos[1] + self.grid_size[1] - 1
                                if (
                                    0 <= matrix_x < width
                                    and 0 <= matrix_y < height
                                ):
                                    index = matrix_x * height + matrix_y
                                    terrain_grid[index] = 1.0
                        terrain_grids.append(terrain_grid)

                    all_grids = entity_grids + terrain_grids
                    state = (
                        np.concatenate(all_grids)
                        if all_grids
                        else np.array([], dtype=np.float32)
                    )

                    agent_obs = np.concatenate(
                        [
                            state,
                            np.array([rewards_normalized[agent_idx]], dtype=np.float32),
                            self._get_one_hot_action(
                                self.action_for_one_hot_encoding.get(agent_id, 0)
                            ),
                        ]
                    ).astype(np.float32)
                    agent_obs = self._apply_encoders(agent_id, agent_obs)
                    obs_dict[agent_id] = agent_obs
                return obs_dict
            else:
                # Original behavior: shared absolute position grids
                prey_grids = [
                    (
                        self._get_one_hot_matrix(self.prey[prey_id].position)
                        if prey_id in self.prey
                        else np.zeros(
                            self.grid_size[0] * self.grid_size[1], dtype=np.float32
                        )
                    )
                    for prey_id in sorted(self.prey_ids)
                ]
                entity_grids = [
                    self._get_one_hot_matrix(self.predators[pred_id].position)
                    for pred_id in sorted(self.predator_ids)
                ] + prey_grids

                terrain_grids = []
                for terrain_type in sorted(self.terrain_config.keys()):
                    terrain_positions = self.terrain_map.get(terrain_type, [])
                    terrain_grid = self._get_one_hot_terrain_grid(terrain_positions)
                    terrain_grids.append(terrain_grid)

                all_grids = entity_grids + terrain_grids
                state = np.concatenate(all_grids) if all_grids else np.array([])

                obs_dict = {}
                for i, agent in enumerate(self._agent_ids):
                    agent_obs = np.concatenate(
                        [
                            state,
                            np.array([rewards_normalized[i]], dtype=np.float32),
                            self._get_one_hot_action(
                                self.action_for_one_hot_encoding.get(agent, 0)
                            ),
                        ]
                    ).astype(np.float32)
                    agent_obs = self._apply_encoders(agent, agent_obs)
                    obs_dict[agent] = agent_obs
                return obs_dict

    def _is_agent_immune_to_terrain(self, agent):
        """Return whether an agent ignores terrain effects."""
        return getattr(agent, "immune_to_terrain", False)

    def _has_limited_vision(self, agent):
        """Return whether an agent has limited vision enabled."""
        return agent.status.get("limited_vision", False) if agent else False

    def _get_agent_vision_radius(self, agent):
        """Return the vision radius for an agent."""
        return getattr(agent, "limited_vision_radius", 1)

    def _get_one_hot_matrix_limited_vision(self, coords, limited_vision_radius):
        """Encode relative position as one-hot matrix, masked by vision radius.
        
        Returns a flattened one-hot vector with positions outside the vision radius
        set to zero.
        """
        width = (self.grid_size[0] - 1) * 2 + 1
        height = (self.grid_size[1] - 1) * 2 + 1
        one_hot_matrix = np.zeros((width, height), dtype=np.float32)
        if (
            abs(coords[0]) <= limited_vision_radius
            and abs(coords[1]) <= limited_vision_radius
        ):
            matrix_x = coords[0] + self.grid_size[0] - 1
            matrix_y = coords[1] + self.grid_size[1] - 1
            if 0 <= matrix_x < width and 0 <= matrix_y < height:
                one_hot_matrix[matrix_x, matrix_y] = 1.0
        return one_hot_matrix.flatten()

    def _get_one_hot_matrix_full_vision(self, coords):
        """Encode relative position as one-hot matrix without vision masking.
        
        Returns a flattened one-hot vector for any position within the grid range.
        """
        width = (self.grid_size[0] - 1) * 2 + 1
        height = (self.grid_size[1] - 1) * 2 + 1
        one_hot_matrix = np.zeros((width, height), dtype=np.float32)
        matrix_x = coords[0] + self.grid_size[0] - 1
        matrix_y = coords[1] + self.grid_size[1] - 1
        if 0 <= matrix_x < width and 0 <= matrix_y < height:
            one_hot_matrix[matrix_x, matrix_y] = 1.0
        return one_hot_matrix.flatten()

    def _get_relative_position(self, agent_pos, other_pos):
        """Return relative position vector from agent to other entity."""
        return np.array(
            [other_pos[0] - agent_pos[0], other_pos[1] - agent_pos[1]], dtype=np.int32
        )

    def _move(self, position, action, agent=None, speed=None):
        """Move an agent one step with terrain rules."""
        if speed is None and agent is not None:
            speed = agent.speed
        elif speed is None:
            speed = 1
        if action == STILL:
            return position

        x, y = position
        new_x, new_y = x, y

        if action == UP:
            new_y = max(0, y - speed)
        elif action == DOWN:
            new_y = min(self.grid_size[1] - 1, y + speed)
        elif action == LEFT:
            new_x = max(0, x - speed)
        elif action == RIGHT:
            new_x = min(self.grid_size[0] - 1, x + speed)

        new_pos = (new_x, new_y)

        if new_pos in self.terrain_at:
            terrain_type = self.terrain_at[new_pos]
            agent_restricted = getattr(agent, "restricted_terrains", set())
            if terrain_type in agent_restricted:
                return position

        if new_pos in self.terrain_at and not self._is_agent_immune_to_terrain(agent):
            terrain_type = self.terrain_at[new_pos]
            terrain_config = self.terrain_config.get(terrain_type, {})
            passability = terrain_config.get("passability", 1.0)
            speed_multiplier = terrain_config.get("speed_multiplier", 1.0)

            if self._py_rng.random() >= passability:
                return position

            if speed_multiplier < 1.0:
                reduced_speed = max(0, int(speed * speed_multiplier))
                if reduced_speed == 0:
                    return position
                return self._move(position, action, agent=agent, speed=reduced_speed)

        return new_pos

    def _find_random_valid_coord(self):
        """Return a free in-bounds coordinate."""
        occupied_positions = set()

        for pos, terrain_type in self.terrain_at.items():
            if self.terrain_config.get(terrain_type, {}).get("passability", 1.0) == 0.0:
                occupied_positions.add(pos)

        for predator in self.predators.values():
            if predator.position is not None:
                occupied_positions.add(predator.position)

        for prey in self.prey.values():
            if prey.position is not None:
                occupied_positions.add(prey.position)

        for _ in range(100):
            x = self._np_rng.randint(0, self.grid_size[0])
            y = self._np_rng.randint(0, self.grid_size[1])
            if (x, y) not in occupied_positions:
                return (x, y)

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) not in occupied_positions:
                    return (x, y)

        raise RuntimeError("No valid free coordinates left in the grid.")

    def _get_one_hot_matrix(self, coords):
        one_hot_matrix = np.zeros(
            self.grid_size[0] * self.grid_size[1], dtype=np.float32
        )
        index = coords[0] * self.grid_size[1] + coords[1]
        one_hot_matrix[index] = 1.0
        return one_hot_matrix

    def _get_one_hot_action(self, action):
        one_hot_action = np.zeros(5, dtype=np.float32)
        one_hot_action[action] = 1.0
        return one_hot_action

    def _get_one_hot_terrain_grid(self, terrain_positions):
        """Return flattened one-hot terrain grid."""
        one_hot_grid = np.zeros(self.grid_size[0] * self.grid_size[1], dtype=np.float32)
        for pos in terrain_positions:
            index = pos[0] * self.grid_size[1] + pos[1]
            one_hot_grid[index] = 1.0
        return one_hot_grid

    def _apply_encoders(self, agent_id: str, base_obs: np.ndarray) -> np.ndarray:
        """Apply configured observation encoders."""
        if not self.observation_encoders:
            return base_obs

        obs = base_obs.copy()

        for encoder_name in self.encoder_order:
            if encoder_name not in self.observation_encoders:
                continue

            try:
                if encoder_name == "agent_centric":
                    obs = helpers.encode_agent_centric(
                        obs, agent_id, self.agent_types, encoding_size=8
                    )

                elif encoder_name == "action_mask":
                    if agent_id in self.predators:
                        agent_pos = self.predators[agent_id].position
                    elif agent_id in self.prey:
                        agent_pos = self.prey[agent_id].position
                    else:
                        raise RuntimeError(f"Unknown agent: {agent_id}")

                    restricted_terrains = set()
                    agent_type = self.agent_types.get(agent_id, 0)
                    for terrain_type, config in self.terrain_config.items():
                        restricted_types = config.get("restricted_to", set())
                        if agent_type in restricted_types:
                            restricted_terrains.add(terrain_type)

                    mask = helpers.create_action_mask(
                        agent_pos,
                        self.grid_size,
                        restricted_terrains=restricted_terrains,
                        terrain_at=self.terrain_at,
                        terrain_config=self.terrain_config,
                    )
                    obs = np.concatenate([obs, mask]).astype(np.float32)

            except Exception as e:
                raise RuntimeError(
                    f"Error applying encoder '{encoder_name}' to agent {agent_id}: {e}"
                ) from e

        return obs.astype(np.float32)

    def _set_cell(self, grid, pos, symbol):
        x, y = pos
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            grid[x, y] = symbol
