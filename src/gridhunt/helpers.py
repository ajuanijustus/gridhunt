import random
import warnings
from typing import Any

import numpy as np


def encode_agent_centric(
    base_obs: np.ndarray,
    agent_id: str,
    agent_types: dict[str, int],
    encoding_size: int = 8,
) -> np.ndarray:
    """Append agent-type encoding to an observation."""
    if not isinstance(base_obs, np.ndarray):
        raise TypeError(f"base_obs must be numpy array, got {type(base_obs)}")
    if not agent_types:
        raise ValueError("agent_types dict cannot be empty")

    agent_type_idx = agent_types.get(agent_id, 0)
    num_types = len(set(agent_types.values()))
    agent_encoding = np.zeros(max(num_types, encoding_size), dtype=np.float32)
    agent_encoding[agent_type_idx] = 1.0
    return np.concatenate([base_obs, agent_encoding]).astype(np.float32)


def create_action_mask(
    agent_position: tuple[int, int],
    grid_size: tuple[int, int],
    restricted_terrains: set[str] | None = None,
    terrain_at: dict[tuple[int, int], str] | None = None,
    terrain_config: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Return a 5-action validity mask."""
    if not isinstance(agent_position, tuple) or len(agent_position) != 2:
        raise TypeError(
            f"agent_position must be tuple of length 2, got {agent_position}"
        )
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise TypeError(f"grid_size must be tuple of length 2, got {grid_size}")
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    terrain_at = terrain_at or {}
    terrain_config = terrain_config or {}
    restricted_terrains = restricted_terrains or set()

    mask = np.ones(5, dtype=np.float32)
    x, y = agent_position
    width, height = grid_size

    checks = {
        1: (x, y - 1),  # UP
        2: (x, y + 1),  # DOWN
        3: (x - 1, y),  # LEFT
        4: (x + 1, y),  # RIGHT
    }

    for action, (tx, ty) in checks.items():
        if not (0 <= tx < width and 0 <= ty < height):
            mask[action] = 0.0
            continue
        pos = (tx, ty)
        terrain_type = terrain_at.get(pos)
        if terrain_type is None:
            continue
        if terrain_type in restricted_terrains:
            mask[action] = 0.0
            continue
        if terrain_config.get(terrain_type, {}).get("passability", 1.0) == 0.0:
            mask[action] = 0.0

    return mask


def detect_agent_types(
    predator_ids: set[str],
    trainable_prey: set[str],
    agent_role_map: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """Build agent_id -> type index."""
    agent_types: dict[str, int] = {}
    if agent_role_map:
        for type_idx, (role_id, agent_list) in enumerate(agent_role_map.items()):
            if not isinstance(agent_list, (list, set)):
                raise ValueError(
                    f"agent_role_map['{role_id}'] must be list or set, got {type(agent_list)}"
                )
            for agent_id in agent_list:
                if agent_id not in predator_ids and agent_id not in trainable_prey:
                    raise ValueError(
                        f"agent_role_map references unknown agent '{agent_id}'"
                    )
                agent_types[agent_id] = type_idx
        unassigned = (predator_ids | trainable_prey) - set(agent_types.keys())
        if unassigned:
            raise ValueError(
                f"agent_role_map does not cover all agents. Unassigned: {unassigned}"
            )
        return agent_types

    for agent_id in predator_ids:
        agent_types[agent_id] = 0
    for agent_id in trainable_prey:
        agent_types[agent_id] = 1
    return agent_types


def gather_all_entity_positions(
    predators: dict[str, Any],
    prey: dict[str, Any],
    entities: dict[str, Any] | None = None,
) -> dict[str, tuple[int, int]]:
    """Collect entity positions."""
    positions: dict[str, tuple[int, int]] = {}
    for pred_id, predator in predators.items():
        if hasattr(predator, "position") and predator.position is not None:
            positions[pred_id] = predator.position
    for prey_id, prey_obj in prey.items():
        if hasattr(prey_obj, "position") and prey_obj.position is not None:
            positions[prey_id] = prey_obj.position
    if entities:
        for entity_id, entity_obj in entities.items():
            if hasattr(entity_obj, "position") and entity_obj.position is not None:
                positions[entity_id] = entity_obj.position
    return positions


def validate_action_mask(mask: np.ndarray, agent_id: str) -> None:
    """Validate mask format for policy consumers."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"Agent {agent_id}: action_mask must be numpy array, got {type(mask)}"
        )
    if mask.shape != (5,):
        raise ValueError(
            f"Agent {agent_id}: action_mask shape must be (5,), got {mask.shape}"
        )
    if mask.dtype != np.float32:
        raise ValueError(
            f"Agent {agent_id}: action_mask dtype must be float32, got {mask.dtype}"
        )
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError(
            f"Agent {agent_id}: action_mask values must be 0.0 or 1.0, got {mask}"
        )
    if not np.any(mask == 1.0):
        raise ValueError(
            f"Agent {agent_id}: action_mask has no valid actions (all zeros)"
        )


def calculate_observation_size(
    num_predators: int,
    num_prey: int,
    num_terrain_types: int,
    grid_size: tuple[int, int],
    observation_mode: str,
    observation_encoders: list[str] | None = None,
) -> int:
    """Return flattened observation size for a config."""
    observation_encoders = observation_encoders or []
    grid_x, grid_y = grid_size

    if observation_mode == "coords":
        base_size = (num_prey + num_predators) * 2 + num_terrain_types * 2
    elif observation_mode == "flat_coords":
        base_size = (num_prey + num_predators) * 2
    elif observation_mode == "one_hot":
        num_entities = num_prey + num_predators
        num_cells = grid_x * grid_y
        base_size = num_entities * num_cells + 1 + 5 + num_terrain_types * num_cells
    else:
        raise ValueError(f"Invalid observation_mode: {observation_mode}")

    encoder_size = 0
    if "agent_centric" in observation_encoders:
        encoder_size += 8
    if "action_mask" in observation_encoders:
        encoder_size += 5

    return base_size + encoder_size


def _is_too_close_to_existing(
    center_pos: tuple[int, int],
    existing_positions: set[tuple[int, int]],
    min_distance: int = 2,
) -> bool:
    if not existing_positions:
        return False
    cx, cy = center_pos
    return any(
        abs(cx - ex) + abs(cy - ey) < min_distance for ex, ey in existing_positions
    )


def _generate_terrain_lines(
    grid_size: tuple[int, int],
    count: int = 3,
    length: int = 10,
    orientation: str = "both",
    min_distance: int = 2,
    *,
    rng: random.Random,
) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    width, height = grid_size
    for i in range(count):
        orient = "horizontal" if orientation == "both" and i % 2 == 0 else orientation
        if orientation == "both" and i % 2 == 1:
            orient = "vertical"
        varied_length = max(1, int(length * rng.uniform(0.8, 1.2)))
        placed = False
        for _ in range(5):
            if orient == "horizontal":
                start_x = rng.randint(0, max(0, width - varied_length))
                y = rng.randint(0, height - 1)
                center = (start_x + varied_length // 2, y)
                if _is_too_close_to_existing(center, positions, min_distance):
                    continue
                for x in range(start_x, min(start_x + varied_length, width)):
                    positions.add((x, y))
                placed = True
                break
            start_y = rng.randint(0, max(0, height - varied_length))
            x = rng.randint(0, width - 1)
            center = (x, start_y + varied_length // 2)
            if _is_too_close_to_existing(center, positions, min_distance):
                continue
            for y in range(start_y, min(start_y + varied_length, height)):
                positions.add((x, y))
            placed = True
            break
        if not placed:
            warnings.warn(
                "Could not place a terrain line after retry budget; consider reducing min_distance.",
                RuntimeWarning,
                stacklevel=2,
            )
    return positions


def _generate_terrain_blobs(
    grid_size: tuple[int, int],
    count: int = 5,
    blob_size: int = 3,
    density: float = 0.7,
    min_distance: int = 2,
    *,
    rng: random.Random,
) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    width, height = grid_size
    for _ in range(count):
        placed = False
        for _ in range(5):
            seed = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            if _is_too_close_to_existing(seed, positions, min_distance):
                continue
            queue: list[tuple[int, int]] = [seed]
            visited: set[tuple[int, int]] = set()
            while queue:
                x, y = queue.pop(0)
                if (x, y) in visited or not (0 <= x < width and 0 <= y < height):
                    continue
                visited.add((x, y))
                positions.add((x, y))
                if len(visited) >= blob_size * blob_size:
                    continue
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited and rng.random() < density:
                        queue.append((nx, ny))
            placed = True
            break
        if not placed:
            warnings.warn(
                "Could not place a terrain blob after retry budget; consider reducing min_distance.",
                RuntimeWarning,
                stacklevel=2,
            )
    return positions


def generate_terrain_map(
    grid_size: tuple[int, int],
    terrain_config: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, set[tuple[int, int]]]:
    """Generate terrain positions from generation config.

    rng: the environment's seeded random.Random generator.
    """
    terrain_map: dict[str, set[tuple[int, int]]] = {}
    for terrain_type, config in (terrain_config or {}).items():
        gen_config = config.get("generation", {})
        gen_type = gen_config.get("type", "blob")
        if gen_type == "line":
            positions = _generate_terrain_lines(
                grid_size=grid_size,
                count=gen_config.get("count", 3),
                length=gen_config.get("length", 10),
                orientation=gen_config.get("orientation", "both"),
                min_distance=gen_config.get("min_distance", 5),
                rng=rng,
            )
        elif gen_type == "blob":
            positions = _generate_terrain_blobs(
                grid_size=grid_size,
                count=gen_config.get("count", 5),
                blob_size=gen_config.get("blob_size", 3),
                density=gen_config.get("density", 0.7),
                min_distance=gen_config.get("min_distance", 5),
                rng=rng,
            )
        elif gen_type == "predefined":
            positions = set(gen_config.get("positions", []))
        else:
            positions = set()
        terrain_map[terrain_type] = positions
    return terrain_map
