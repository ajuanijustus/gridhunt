
from .assets import (
    BUILDING_SPRITE,
    HARE_SPRITE,
    MAGENTA_HUNTER_SPRITE,
    PIG_SPRITE,
    PLANT_SPRITE,
    STAG_SPRITE,
    TOXIC_STAG_SPRITE,
)


class Predator:
    """Controllable hunter agent."""

    def __init__(
        self,
        predator_id,
        position=None,
        status=None,
        capture_power=1,
        limited_vision_radius=1,
        speed=1,
        render_sprite=MAGENTA_HUNTER_SPRITE,
        immune_to_terrain=False,
        restricted_terrains=None,
    ):
        self.predator_id = predator_id
        self.position = position
        self.capture_power = capture_power
        self.speed = speed
        self.status = status or {}
        self.limited_vision_radius = limited_vision_radius
        self.render_sprite = render_sprite
        self.immune_to_terrain = immune_to_terrain
        self.restricted_terrains = (
            set(restricted_terrains) if restricted_terrains else set()
        )


class Prey:
    """Capturable target."""

    def __init__(
        self,
        prey_id,
        reward,
        position=None,
        prey_type=None,
        respawn=False,
        status=None,
        capture_power_required=1,
        console_sprite="T",
        render_sprite=STAG_SPRITE,
        overcrowding_penalty=0,
        speed=1,
        immune_to_terrain=False,
        restricted_terrains=None,
    ):
        self.prey_id = prey_id
        self.position = position
        self.reward = reward
        self.type = prey_type
        self.status = status or {}
        self.respawn = respawn
        self.capture_power_required = capture_power_required
        self.console_sprite = console_sprite
        self.render_sprite = render_sprite
        self.overcrowding_penalty = overcrowding_penalty
        self.speed = speed
        self.immune_to_terrain = immune_to_terrain
        self.restricted_terrains = (
            set(restricted_terrains) if restricted_terrains else set()
        )

    def execute_respawn(self, env):
        self.position = env._find_random_valid_coord()


class Entity:
    """Non-capturable world object."""

    def __init__(
        self, entity_id, position=None, console_sprite="E", render_sprite=PLANT_SPRITE
    ):
        self.entity_id = entity_id
        self.position = position
        self.console_sprite = console_sprite
        self.render_sprite = render_sprite


class Terrain(Entity):
    """Terrain tile with movement effects."""

    def __init__(
        self,
        terrain_id,
        terrain_type,
        position=None,
        passability=1.0,
        speed_multiplier=1.0,
        console_sprite="T",
        render_sprite=PLANT_SPRITE,
        status=None,
    ):
        super().__init__(
            entity_id=terrain_id,
            position=position,
            console_sprite=console_sprite,
            render_sprite=render_sprite,
        )
        self.terrain_type = terrain_type
        self.passability = passability
        self.speed_multiplier = speed_multiplier
        self.status = status or {}


class Stag(Prey):
    """Cooperative high-value prey."""

    def __init__(self, prey_id, position=None, capture_power_required=2):
        super().__init__(
            prey_id=prey_id,
            reward=5,
            position=position,
            prey_type="stag",
            respawn=True,
            capture_power_required=capture_power_required,
            console_sprite="S",
            render_sprite=STAG_SPRITE,
        )


class Hare(Prey):
    """Solo low-value prey."""

    def __init__(self, prey_id, render_sprite=HARE_SPRITE, speed=1, position=None):
        super().__init__(
            prey_id=prey_id,
            reward=1,
            position=position,
            prey_type="hare",
            respawn=True,
            capture_power_required=1,
            console_sprite="H",
            render_sprite=render_sprite,
            speed=speed,
            overcrowding_penalty=0.5,
        )


class ToxicStag(Prey):
    """Stag variant with toxic/healthy respawn state.

    Each time it appears the stag is toxic with probability ``toxic_probability``. A toxic stag
    that nobody catches moves to a random cell after ``toxic_lifetime`` steps, with its toxicity
    drawn again; a healthy one stays until it is caught.

    toxic_lifetime: steps a toxic stag stays put. An int (the benchmark uses 5), a ``(low, high)``
        pair to draw it uniformly from that range each time the stag turns toxic, or ``None`` for a
        toxic stag that stays until it is caught. A fixed lifetime is a cue in itself: a stag that
        has stayed longer than that must be healthy, so agents without the toxicity flag can infer it.
    """

    def __init__(
        self,
        prey_id,
        position=None,
        toxic_probability=0.5,
        toxic_render_sprite=TOXIC_STAG_SPRITE,
        toxic_lifetime=5,
    ):
        super().__init__(
            prey_id=prey_id,
            reward=5,
            position=position,
            prey_type="stag",
            respawn=True,
            capture_power_required=2,
            console_sprite="S",
            render_sprite=STAG_SPRITE,
        )
        self.toxic_probability = toxic_probability
        # Sprite shown while toxic; pass STAG_SPRITE to hide toxicity from viewers
        self.toxic_render_sprite = toxic_render_sprite
        self.toxic_lifetime = _check_lifetime(toxic_lifetime)
        # Healthy until the environment draws the toxicity from its own seeded RNG on reset
        self.toxic_turns_alive = 0
        self.current_toxic_lifetime = None

    def _reroll_disease_status(self, rng):
        """Re-draw toxicity with the environment's seeded generator."""
        if rng.random() < self.toxic_probability:
            self.status["toxic"] = True
            self.render_sprite = self.toxic_render_sprite
            self.reward = -5
            # A fixed lifetime draws nothing, so the benchmark's random stream is unchanged
            if isinstance(self.toxic_lifetime, tuple):
                low, high = self.toxic_lifetime
                draw = rng.integers if hasattr(rng, "integers") else rng.randint  # Generator or RandomState
                self.current_toxic_lifetime = int(draw(low, high + 1))
            else:
                self.current_toxic_lifetime = self.toxic_lifetime
        else:
            self.status.pop("toxic", None)
            self.render_sprite = STAG_SPRITE
            self.reward = 5
            self.current_toxic_lifetime = None
        self.toxic_turns_alive = 0

    def execute_respawn(self, env):
        self._reroll_disease_status(env._np_rng)
        self.position = env._find_random_valid_coord()


def _check_lifetime(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value) \
                and 1 <= value[0] <= value[1]:
            return (value[0], value[1])
    elif isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    raise ValueError(f"toxic_lifetime must be a positive int, a (low, high) pair of them or None, not {value!r}")


class Building(Prey):
    """Static raid target."""

    def __init__(
        self,
        prey_id,
        position=None,
        reward=1,
        capture_power_required=1,
        trap_probability=0.0,
        trap_penalty=0,
        console_sprite="B",
        render_sprite=BUILDING_SPRITE,
    ):
        super().__init__(
            prey_id=prey_id,
            reward=reward,
            position=position,
            prey_type="building",
            respawn=False,
            capture_power_required=capture_power_required,
            console_sprite=console_sprite,
            render_sprite=render_sprite,
            overcrowding_penalty=0,
            speed=0,
        )
        self.status["trap_probability"] = trap_probability
        self.status["trap_penalty"] = trap_penalty


class Pig(Prey):
    """Pig chase target."""

    def __init__(self, prey_id, render_sprite=None, speed=1, position=None):
        super().__init__(
            prey_id=prey_id,
            reward=1,
            position=position,
            prey_type="pig",
            respawn=True,
            capture_power_required=2,
            console_sprite="P",
            render_sprite=render_sprite or PIG_SPRITE,
            speed=speed,
            overcrowding_penalty=0,
        )
