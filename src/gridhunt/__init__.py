"""GridHunt: heterogeneous, mixed-motive predator-prey environments with a PettingZoo API."""

__version__ = "1.0.0.dev0"

import copy

from .agents import Building, Entity, Hare, Pig, Predator, Stag, ToxicStag
from .assets import (
    BARN_SPRITE,
    BIRD_SPRITE,
    BLUE_HUNTER_SPRITE,
    BOAT_ARCHER_SPRITE,
    BRUISER_SPRITE,
    BUILDING_SPRITE,
    DRONE_SPRITE,
    HOUSE_SPRITE,
    SCOUT_SPRITE,
    VEHICLE_SPRITE,
)
from .wrappers import CommunicationWrapper
from .scenarios import (
    PigChaseScenario,
    RaidScenario,
    StagHuntScenario,
    ToxicStagHuntScenario,
)

CLASSIC_STAG_V0 = {
    "scenario_class": StagHuntScenario,
    "grid_size": (5, 5),
    "max_steps": 1000,
    "observation_mode": "flat_coords",
    "predators": [
        Predator("predator_0", capture_power=1, render_sprite=BLUE_HUNTER_SPRITE),
        Predator("predator_1", capture_power=1),
    ],
    "prey": [Stag("stag_0", capture_power_required=2), Hare("hare_0"), Hare("hare_1")],
    "entities": [],
}

TOXIC_STAG_V0 = {
    "scenario_class": ToxicStagHuntScenario,
    "grid_size": (7, 7),
    "max_steps": 1000,
    "observation_mode": "toxic",
    "predators": [
        Predator(
            "predator_0",
            capture_power=1,
            status={"toxicity_aware": True, "limited_vision": True},
            render_sprite=BLUE_HUNTER_SPRITE,
        ),
        Predator("predator_1", capture_power=1),
        Predator("predator_2", capture_power=1),
    ],
    "prey": [
        ToxicStag("stag_0", toxic_probability=0.5),
        Hare("hare_0"),
        Hare("hare_1"),
    ],
    "entities": [],
}

SPECIALISTS_V0 = {
    "scenario_class": StagHuntScenario,
    "grid_size": (10, 10),
    "max_steps": 1000,
    "observation_mode": "one_hot",
    "predators": [
        Predator(
            "scout_0",
            capture_power=1,
            speed=2,
            limited_vision_radius=10,
            render_sprite=SCOUT_SPRITE,
        ),
        Predator(
            "bruiser_0",
            capture_power=2,
            speed=1,
            limited_vision_radius=2,
            status={"limited_vision": True},
            render_sprite=BRUISER_SPRITE,
        ),
    ],
    "prey": [
        # The Bruiser can capture the stag alone but cannot see it from afar; the
        # fleeing stag has to be found (information interdependence)
        Stag("stag_0", capture_power_required=2),
        Hare("hare_0", speed=2),
        Hare("hare_1", speed=2),
    ],
    "entities": [],
    "prey_motion": "flee",
    "prey_move_probability": 0.5,
}

SWARM_V0 = {
    "scenario_class": StagHuntScenario,
    "grid_size": (23, 23),
    "max_steps": 300,
    "observation_mode": "one_hot",
    "predators": [
        Predator("predator_0", capture_power=1, render_sprite=BLUE_HUNTER_SPRITE)
    ]
    + [Predator(f"predator_{i}", capture_power=1) for i in range(1, 10)],
    "prey": [
        Stag("stag_0", capture_power_required=2),
        Stag("stag_1", capture_power_required=2),
        Stag("stag_2", capture_power_required=2),
        Hare("hare_0"),
        Hare("hare_1"),
        Hare("hare_2"),
        Hare("hare_3"),
        Hare("hare_4"),
    ],
    "entities": [],
}

MILITARY_V0 = {
    "scenario_class": StagHuntScenario,
    "grid_size": (10, 10),
    "max_steps": 1000,
    "observation_mode": "one_hot",
    "predators": [
        Predator(
            "uav_0",
            capture_power=0,
            speed=2,
            limited_vision_radius=15,
            immune_to_terrain=True,
            render_sprite=DRONE_SPRITE,
        ),
        Predator(
            "ground_vehicle_0",
            capture_power=2,
            speed=1,
            limited_vision_radius=3,
            status={"limited_vision": True},
            render_sprite=VEHICLE_SPRITE,
        ),
    ],
    "prey": [
        Stag("stag_0", capture_power_required=2),
        Hare("hare_0", speed=2),
        Hare("hare_1", speed=2),
    ],
    "entities": [],
    "terrain_config": {
        "mountain": {
            "passability": 0.6,
            "speed_multiplier": 1,
            "console_sprite": "M",
            "generation": {
                "type": "blob",
                "count": 1,
                "blob_size": 2,
                "density": 0.4,
                "min_distance": 2,  # Reduced from default 5 for 10x10 grid
            },
        },
        "swamp": {
            "passability": 0.5,
            "speed_multiplier": 1,
            "console_sprite": "S",
            "generation": {
                "type": "blob",
                "count": 1,
                "blob_size": 2,
                "density": 0.1,
                "min_distance": 2,  # Reduced from default 5 for 10x10 grid
            },
        },
        "obstacle": {
            "passability": 0.0,
            "console_sprite": "#",
            "generation": {
                "type": "line",
                "count": 2,
                "length": 6,
                "orientation": "both",
                "min_distance": 2,  # Moved inside generation dict; reduced from implicit default 5
            },
        },
    },
}

RAID_V0 = {
    "scenario_class": RaidScenario,
    "grid_size": (12, 12),
    "max_steps": 1000,
    "observation_mode": "one_hot",
    "predators": [
        Predator("raider_0", capture_power=1, render_sprite=BLUE_HUNTER_SPRITE),
        Predator("raider_1", capture_power=1),
        Predator("raider_2", capture_power=1),
    ],
    "prey": [
        Building(
            "barn_0",
            reward=2,
            capture_power_required=1,
            trap_probability=0.0,
            trap_penalty=0,
            render_sprite=BARN_SPRITE,
        ),
        Building(
            "house_0",
            reward=5,
            capture_power_required=2,
            trap_probability=0.25,
            trap_penalty=-4,
            render_sprite=HOUSE_SPRITE,
        ),
        Building(
            "bank_0",
            reward=20,
            capture_power_required=3,
            trap_probability=0.5,
            trap_penalty=-25,
            render_sprite=BUILDING_SPRITE,
        ),
    ],
    "entities": [],
}

PIG_V0 = {
    "scenario_class": PigChaseScenario,
    "grid_size": (9, 9),
    "max_steps": 1000,
    "observation_mode": "one_hot",
    "predators": [
        Predator(
            "predator_0",
            capture_power=1,
            render_sprite=BLUE_HUNTER_SPRITE,
            position=(2, 2),
        ),
        Predator("predator_1", capture_power=1, position=(6, 6)),
    ],
    "prey": [Pig("pig", position=(4, 4))],
    "entities": [],
    "terrain_config": {
        "obstacle": {
            "passability": 0.0,
            "generation": {
                "type": "predefined",
                "positions": [
                    (1, 1),
                    (2, 1),
                    (3, 1),
                    (4, 1),
                    (5, 1),
                    (6, 1),
                    (7, 1),
                    (1, 7),
                    (2, 7),
                    (3, 7),
                    (4, 7),
                    (5, 7),
                    (6, 7),
                    (7, 7),
                    (0, 3),
                    (0, 4),
                    (0, 5),
                    (8, 3),
                    (8, 4),
                    (8, 5),
                    (1, 2),
                    (1, 3),
                    (1, 5),
                    (1, 6),
                    (7, 2),
                    (7, 3),
                    (7, 5),
                    (7, 6),
                    (3, 3),
                    (5, 3),
                    (3, 5),
                    (5, 5),
                ],
            },
        }
    },
}

LAND_AND_SEA_V0 = {
    "scenario_class": StagHuntScenario,
    "grid_size": (10, 10),
    "max_steps": 1000,
    "observation_mode": "one_hot",
    "predators": [
        Predator(
            "land_predator",
            capture_power=1,
            render_sprite=BLUE_HUNTER_SPRITE,
            position=(8, 5),
            restricted_terrains=["sea"],
        ),
        Predator(
            "sea_predator",
            capture_power=1,
            render_sprite=BOAT_ARCHER_SPRITE,
            position=(1, 5),
            restricted_terrains=["land"],
        ),
    ],
    "prey": [Hare("bird", render_sprite=BIRD_SPRITE)],
    "entities": [],
    "terrain_config": {
        "sea": {
            "passability": 1.0,
            "speed_multiplier": 1.0,
            "console_sprite": "~",
            "generation": {
                "type": "predefined",
                "positions": [
                    *[(x, y) for x in range(0, 3) for y in range(0, 10)],
                    *[(3, y) for y in range(3, 10)],
                    *[(4, y) for y in range(5, 10)],
                    *[(5, y) for y in range(8, 10)],
                ],
            },
        },
        "land": {
            "passability": 1.0,
            "speed_multiplier": 1.0,
            "console_sprite": ".",
            "generation": {
                "type": "predefined",
                "positions": [
                    *[(3, y) for y in range(0, 3)],
                    *[(4, y) for y in range(0, 5)],
                    *[(5, y) for y in range(0, 8)],
                    *[(x, y) for x in range(6, 10) for y in range(0, 10)],
                ],
            },
        },
    },
}

_BENCHMARKS = {
    "GridHunt-ClassicStag-v0": CLASSIC_STAG_V0,
    "GridHunt-ToxicStag-v0": TOXIC_STAG_V0,
    "GridHunt-Specialists-v0": SPECIALISTS_V0,
    "GridHunt-Swarm-v0": SWARM_V0,
    "GridHunt-Military-v0": MILITARY_V0,
    "GridHunt-Raid-v0": RAID_V0,
    "GridHunt-PIG-v0": PIG_V0,
    "GridHunt-LandAndSea-v0": LAND_AND_SEA_V0,
}


def _env_creator(env_config):
    """Create a scenario instance from config."""
    scenario_class = env_config.get("scenario_class", StagHuntScenario)
    return scenario_class(
        grid_size=env_config["grid_size"],
        predators=env_config.get("predators", []),
        prey=env_config.get("prey", []),
        entities=env_config.get("entities", []),
        max_steps=env_config["max_steps"],
        render_mode=env_config.get("render_mode", None),
        observation_mode=env_config.get("observation_mode", "coords"),
        gui_config=env_config.get("gui_config", {}),
        trainable_prey=env_config.get("trainable_prey", None),
        terrain_config=env_config.get("terrain_config", None),
        observation_encoders=env_config.get("observation_encoders", None),
        use_action_mask=env_config.get("use_action_mask", False),
        agent_role_map=env_config.get("agent_role_map", None),
        encoder_order=env_config.get("encoder_order", None),
        reward_mode=env_config.get("reward_mode", "individual"),
        prey_motion=env_config.get("prey_motion", "static"),
        prey_move_probability=env_config.get("prey_move_probability", 0.5),
    )


def make(env_name: str, render_mode: str = None, trainable_prey=None, **kwargs):
    """Instantiate a benchmark environment."""
    if env_name not in _BENCHMARKS:
        raise ValueError(
            f"Environment {env_name} not found. Available envs: {list(_BENCHMARKS.keys())}"
        )
    env_config = copy.deepcopy(_BENCHMARKS[env_name])
    if render_mode is not None:
        env_config["render_mode"] = render_mode
    if trainable_prey is not None:
        env_config["trainable_prey"] = trainable_prey
    env_config.update(kwargs)
    # communication=k adds a k-token message channel (see wrappers.CommunicationWrapper)
    communication = env_config.pop("communication", 0)
    env = _env_creator(env_config)
    return CommunicationWrapper(env, communication) if communication else env
