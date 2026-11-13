import pathlib

STAG_SPRITE = pathlib.Path(__file__).parent / "assets" / "stag_v2.png"
PLANT_SPRITE = pathlib.Path(__file__).parent / "assets" / "plant_no_fruit.png"
TOXIC_STAG_SPRITE = pathlib.Path(__file__).parent / "assets" / "zombie_stag.png"
HARE_SPRITE = pathlib.Path(__file__).parent / "assets" / "hare.png"
BIRD_SPRITE = pathlib.Path(__file__).parent / "assets" / "bird.png"
PIG_SPRITE = pathlib.Path(__file__).parent / "assets" / "pig.png"

HUNTER_SPRITE = pathlib.Path(__file__).parent / "assets" / "hunter.png"
ARCHER_SPRITE = pathlib.Path(__file__).parent / "assets" / "archer.png"
BOAT_ARCHER_SPRITE = pathlib.Path(__file__).parent / "assets" / "boat_archer.png"

BLUE_HUNTER_SPRITE = pathlib.Path(__file__).parent / "assets" / "blue_hunter.png"
MAGENTA_HUNTER_SPRITE = pathlib.Path(__file__).parent / "assets" / "magenta_hunter.png"
YELLOW_HUNTER_SPRITE = pathlib.Path(__file__).parent / "assets" / "yellow_hunter.png"

SPRITE_DICT = {"predator_0": BLUE_HUNTER_SPRITE, "predator_1": MAGENTA_HUNTER_SPRITE}

NO_BERRY_SPRITE = pathlib.Path(__file__).parent / "assets" / "no_berry.png"
RED_BERRY_SPRITE = pathlib.Path(__file__).parent / "assets" / "red_berry.png"
BLUE_BERRY_SPRITE = pathlib.Path(__file__).parent / "assets" / "blue_berry.png"
YELLOW_BERRY_SPRITE = pathlib.Path(__file__).parent / "assets" / "yellow_berry.png"

BRUISER_SPRITE = pathlib.Path(__file__).parent / "assets" / "heavy_hunter.png"
SCOUT_SPRITE = pathlib.Path(__file__).parent / "assets" / "scout.png"

BUILDING_SPRITE = pathlib.Path(__file__).parent / "assets" / "building.png"
HOUSE_SPRITE = pathlib.Path(__file__).parent / "assets" / "house.png"
BARN_SPRITE = pathlib.Path(__file__).parent / "assets" / "barn.png"

DRONE_SPRITE = pathlib.Path(__file__).parent / "assets" / "drone.png"
VEHICLE_SPRITE = pathlib.Path(__file__).parent / "assets" / "vehicle.png"

MOUNTAIN_SPRITE = pathlib.Path(__file__).parent / "assets" / "mountain.png"
SWAMP_SPRITE = pathlib.Path(__file__).parent / "assets" / "swamp.png"
OBSTACLE_SPRITE = pathlib.Path(__file__).parent / "assets" / "obstacle.png"
LAND_SPRITE = pathlib.Path(__file__).parent / "assets" / "land.png"
SEA_SPRITE = pathlib.Path(__file__).parent / "assets" / "water.png"

TERRAIN_SPRITE_DICT = {
    "mountain": MOUNTAIN_SPRITE,
    "swamp": SWAMP_SPRITE,
    "obstacle": OBSTACLE_SPRITE,
    "land": LAND_SPRITE,
    "sea": SEA_SPRITE,
}
