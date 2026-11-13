import os
import sys

import numpy as np
import pygame

try:
    import pygame_gui
except ImportError:  # pragma: no cover
    import warnings

    warnings.warn(
        "pygame_gui is not installed; GUI rendering is unavailable until it is installed.",
        RuntimeWarning,
        stacklevel=2,
    )
    pygame_gui = None
import io

from PIL import Image


class Renderer:
    """Render GridHunt in GUI or console mode."""

    # Text and font configuration
    TEXT_COLOR = (255, 255, 255)
    FONT_NAME = "Arial"
    FONT_SIZE = 20
    TEXT_PADDING = 10

    # UI layout dimensions
    BUTTON_HEIGHT = 30
    BUTTON_PADDING_INTERNAL = 20
    BUTTON_SPACING = 10
    EDGE_PADDING = 10
    ROW_SPACING = 10
    DROPDOWN_DEFAULT_WIDTH = 100
    TILE_SIZE = 32
    GAMEBOARD_COLOR = (255, 185, 137)
    BACKGROUND_COLOR = (20, 20, 20)
    CLEAR = (0, 0, 0, 0)
    GRID_LINE_COLOR = (200, 150, 100, 200)
    SCREEN_SIZE = (750, 750)

    # Grid rendering
    GRID_BOUNDARY_BUFFER = 1

    def __init__(self, env, gui_config: dict | None = None) -> None:
        """Initialize renderer state."""
        self.env = env
        self.gui_config = gui_config or {}
        self.info_bar = self.gui_config.get("info_bar", False)

        # Cache for optimized rendering
        self._sprite_cache = {}
        self._last_score = None
        self._info_text_surface = None
        self._layout = None

        # "gui" draws to a window; "rgb_array" draws the same frame to a hidden window
        if self.env.render_mode in ("gui", "rgb_array"):
            if self.info_bar and pygame_gui is None:
                raise ImportError("pygame_gui is required for the info bar: pip install 'gridhunt[render]'")
            self._setup_pygame()
            self._setup_surfaces()
            if self.info_bar:
                self._setup_info_bar()
            self._draw_grid()

    def _calculate_layout(self) -> dict:
        """
        Calculate GUI element layout dimensions and positions.

        Computes required window width, button positions, and info bar height
        based on button/dropdown configuration and text dimensions.

        Returns:
            Dictionary with layout parameters for use in _setup_info_bar()
        """
        # Number of text lines reserved above the buttons (env.info may contain "\n")
        info_text_lines = max(1, int(self.gui_config.get("info_text_lines", 1)))

        layout = {
            "required_width": self.SCREEN_SIZE[0],
            "text_vertical_space": self.TEXT_PADDING
            + self._text_height * info_text_lines
            + self.TEXT_PADDING,
            "info_bar_height": 0,
        }

        if not self.info_bar:
            return layout

        # Calculate width needed for buttons (widest row when wrapping)
        max_buttons_per_row = self.gui_config.get("buttons_per_row", None)
        x_offset = self.EDGE_PADDING
        row_x_offset = self.EDGE_PADDING
        for i, btn_conf in enumerate(self.gui_config.get("buttons", [])):
            if max_buttons_per_row and i > 0 and i % max_buttons_per_row == 0:
                row_x_offset = self.EDGE_PADDING
            text = btn_conf["text"]
            text_surface = self._font.render(text, True, self.TEXT_COLOR)
            text_width = text_surface.get_width()
            button_width = text_width + self.BUTTON_PADDING_INTERNAL
            row_x_offset += button_width + self.BUTTON_SPACING
            x_offset = max(x_offset, row_x_offset)

        # Check legacy dropdowns with explicit rects
        for dd_conf in self.gui_config.get("dropdowns", []):
            if "rect" in dd_conf:
                rect = pygame.Rect(dd_conf["rect"])
                x_offset = max(x_offset, rect.right + self.EDGE_PADDING)

        layout["required_width"] = max(
            self.SCREEN_SIZE[0], x_offset + self.EDGE_PADDING
        )

        # Calculate info bar height
        buttons = self.gui_config.get("buttons", [])

        if max_buttons_per_row and buttons:
            num_button_rows = (
                len(buttons) + max_buttons_per_row - 1
            ) // max_buttons_per_row
        else:
            num_button_rows = 1 if buttons else 0

        button_rows_height = (
            num_button_rows * (self.BUTTON_HEIGHT + self.ROW_SPACING)
            if num_button_rows > 0
            else 0
        )
        layout["info_bar_height"] = (
            layout["text_vertical_space"] + button_rows_height + self.TEXT_PADDING
        )

        return layout

    def _setup_pygame(self) -> None:
        """Initialize pygame window, surfaces, and display."""
        hidden = self.env.render_mode == "rgb_array"
        if hidden and not pygame.display.get_init() and sys.platform.startswith("linux"):
            # Off-screen frames need no display server (for example on CI machines)
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

        # Calculate game surface dimensions on grid
        self._game_surface_size = (
            self.env.grid_size[0] * self.TILE_SIZE,
            self.env.grid_size[1] * self.TILE_SIZE,
        )

        # Calculate scale factor to fit screen width
        scale_x = self.SCREEN_SIZE[0] // self._game_surface_size[0]
        self._scale = max(1, scale_x)

        # Initialize font and measure text height
        if self.info_bar:
            self._font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)
            sample_text_surface = self._font.render("Score: 0", True, self.TEXT_COLOR)
            self._text_height = sample_text_surface.get_height()
            self._layout = self._calculate_layout()
            self._info_bar_height = self._layout["info_bar_height"]
            required_width = self._layout["required_width"]
        else:
            self._text_height = 0
            self._info_bar_height = 0
            required_width = self.SCREEN_SIZE[0]

        # Calculate final screen size using the full available space
        scaled_game_board_height = self._game_surface_size[1] * self._scale

        # Center game board horizontally and vertically (same offset for uniform padding)
        scaled_game_board_width = self._game_surface_size[0] * self._scale
        self._board_x_offset = (required_width - scaled_game_board_width) // 2
        self._board_y_offset = self._board_x_offset

        self._screen_size = (
            required_width,
            self._board_y_offset + scaled_game_board_height + self._info_bar_height,
        )

        # Position info bar immediately after game board
        self._info_bar_y = self._board_y_offset + scaled_game_board_height

        # Create display and clock
        self._screen = pygame.display.set_mode(self._screen_size, pygame.HIDDEN if hidden else 0)
        self._clock = pygame.time.Clock()
        pygame.display.set_caption("GridHunt")

    def _setup_surfaces(self) -> None:
        """Initialize rendering surface layers: background, grid, and entities."""
        # Game board background on grid
        self._game_surface = pygame.Surface(self._game_surface_size).convert()
        self._game_surface.fill(self.GAMEBOARD_COLOR)

        # Static grid lines layer with boundary buffer for edge lines
        padded_grid_size = (
            self._game_surface_size[0] + self.GRID_BOUNDARY_BUFFER,
            self._game_surface_size[1] + self.GRID_BOUNDARY_BUFFER,
        )
        self._grid_layer = pygame.Surface(padded_grid_size).convert_alpha()
        self._grid_layer.fill(self.CLEAR)

        # Dynamic entity layer for prey, predators, and obstacles
        self._entity_layer = pygame.Surface(self._game_surface_size).convert_alpha()
        self._entity_layer.fill(self.CLEAR)

        # Info bar background
        if self.info_bar:
            self._info_layer = pygame.Surface(
                (self._game_surface_size[0], self._info_bar_height)
            ).convert()
            self._info_layer.fill(self.BACKGROUND_COLOR)

    def _setup_info_bar(self) -> None:
        """Initialize UI manager and create buttons/dropdowns for the info bar."""
        self._ui_manager = pygame_gui.UIManager(self._screen_size)

        # Use pre-calculated layout
        layout = self._layout or {}
        text_vertical_space = layout.get("text_vertical_space", 0)
        max_buttons_per_row = self.gui_config.get("buttons_per_row", None)

        x_offset = self.EDGE_PADDING
        button_row = 0
        buttons_in_row = 0

        # Create buttons
        for btn_conf in self.gui_config.get("buttons", []):
            # Wrap to next row if limit reached
            if max_buttons_per_row and buttons_in_row >= max_buttons_per_row:
                button_row += 1
                buttons_in_row = 0
                x_offset = self.EDGE_PADDING

            text = btn_conf["text"]
            text_surface = self._font.render(text, True, self.TEXT_COLOR)
            button_width = text_surface.get_width() + self.BUTTON_PADDING_INTERNAL
            button_y = text_vertical_space + (
                button_row * (self.BUTTON_HEIGHT + self.ROW_SPACING)
            )

            rect = pygame.Rect(x_offset, button_y, button_width, self.BUTTON_HEIGHT)
            rect.top += self._info_bar_y

            pygame_gui.elements.UIButton(
                relative_rect=rect, text=text, manager=self._ui_manager
            )

            x_offset += button_width + self.BUTTON_SPACING
            buttons_in_row += 1

        # Create dropdowns
        for dd_conf in self.gui_config.get("dropdowns", []):
            if "rect" in dd_conf:
                # Legacy format with explicit rect
                rect = pygame.Rect(dd_conf["rect"])
                rect.top += self._info_bar_y
            else:
                # Dynamic format with row wrapping
                dd_width = dd_conf.get("width", self.DROPDOWN_DEFAULT_WIDTH)

                if max_buttons_per_row and max_buttons_per_row - buttons_in_row < 1:
                    button_row += 1
                    buttons_in_row = 0
                    x_offset = self.EDGE_PADDING

                button_y = text_vertical_space + (
                    button_row * (self.BUTTON_HEIGHT + self.ROW_SPACING)
                )
                rect = pygame.Rect(x_offset, button_y, dd_width, self.BUTTON_HEIGHT)
                rect.top += self._info_bar_y
                x_offset += dd_width + self.BUTTON_SPACING
                buttons_in_row += 1

            pygame_gui.elements.UIDropDownMenu(
                options_list=dd_conf["options"],
                starting_option=dd_conf["options"][0],
                relative_rect=rect,
                manager=self._ui_manager,
            )

    def render(self) -> np.ndarray | None:
        """
        Render one frame of the current environment state.

        Returns the frame as an RGB array for render_mode="rgb_array", and None otherwise.
        The render mode is set when the environment is created.
        """
        if self.env.render_mode is None:
            return None
        if self.env.render_mode == "console":
            self._render_console()
            return None
        self._draw_frame()
        if self.env.render_mode == "rgb_array":
            return self.get_frame()
        pygame.display.flip()
        return None

    def _render_console(self) -> None:
        """
        Render the environment state as an ASCII grid in the terminal.

        Displays a simple text-based view with terrain, prey and predators marked
        with their console sprites.  Useful for debugging or headless operation.
        """
        grid = np.full((self.env.grid_size[0], self.env.grid_size[1]), ".", dtype=str)

        # Place terrain (background)
        if hasattr(self.env, "terrain") and self.env.terrain:
            for terrain_entity in self.env.terrain.values():
                self.env._set_cell(
                    grid, terrain_entity.position, terrain_entity.console_sprite
                )

        # Place prey
        for prey_item in self.env.prey.values():
            self.env._set_cell(grid, prey_item.position, prey_item.console_sprite)

        # Place predators
        for pred_id, predator in self.env.predators.items():
            symbol = pred_id[-1].upper()
            self.env._set_cell(grid, predator.position, symbol)

        # Print grid row by row
        for y in range(self.env.grid_size[1]):
            row = "".join(grid[x, y] for x in range(self.env.grid_size[0]))
            print(row)
        print()

    def _draw_frame(self) -> None:
        """
        Draw the scaled game board, centered, and the optional info bar onto the screen surface.
        """
        # Build complete game board with all layers
        game_board = self._build_game_board()

        # Scale game board ONLY (info bar stays unscaled)
        # Might be worth trying smoothscale() instead of scale for larger grids
        scaled_board = pygame.transform.scale(
            game_board,
            (
                int(self._game_surface_size[0] * self._scale),
                int(self._game_surface_size[1] * self._scale),
            ),
        )

        # Clear screen and draw scaled game board CENTERED horizontally and same offset to top for uniformity
        self._screen.fill(self.BACKGROUND_COLOR)
        self._screen.blit(scaled_board, (self._board_x_offset, self._board_y_offset))

        # Render info bar and UI if enabled
        if self.info_bar:
            self._render_info_bar()
            self._ui_manager.draw_ui(self._screen)

    def _get_limited_vision_predator(self):
        """
        Find first predator with limited_vision status enabled.

        Returns:
            Predator object if found, None otherwise.
        """
        for predator in self.env.predators.values():
            if predator.status.get("limited_vision", False):
                return predator
        return None

    def _create_vision_mask(
        self, agent_pos: tuple, vision_radius: int
    ) -> pygame.Surface:
        """
        Create an opaque mask covering grid cells outside vision_radius.

        Creates a semi-transparent overlay that covers the entire game board,
        with a grid-aligned square transparent area (vision zone) centered on the agent.
        Outside the vision square, the board is darkened; inside, it remains visible.

        Args:
            agent_pos: (x, y) position of the agent whose vision to show
            vision_radius: Radius in tiles for the visible area (creates square zone)

        Returns:
            pygame.Surface: Mask with transparency applied (ready to blit on board)
        """
        # Match padded board size to cover all pixels including boundary buffer
        padded_size = (
            self._game_surface_size[0] + self.GRID_BOUNDARY_BUFFER,
            self._game_surface_size[1] + self.GRID_BOUNDARY_BUFFER,
        )
        mask = pygame.Surface(padded_size).convert_alpha()
        mask.fill((0, 0, 0, 0))  # Start fully transparent

        # Draw opaque overlay covering everything
        mask.fill((20, 20, 20, 254))

        # Clear grid-aligned rectangular area around agent (vision zone stays visible)
        x_min = max(0, (agent_pos[0] - vision_radius) * self.TILE_SIZE)
        x_max = min(padded_size[0], (agent_pos[0] + vision_radius + 1) * self.TILE_SIZE)
        y_min = max(0, (agent_pos[1] - vision_radius) * self.TILE_SIZE)
        y_max = min(padded_size[1], (agent_pos[1] + vision_radius + 1) * self.TILE_SIZE)

        vision_rect = pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)
        pygame.draw.rect(
            mask, (0, 0, 0, 0), vision_rect  # Transparent (reveals visibility)
        )

        return mask

    def _build_game_board(self) -> pygame.Surface:
        """
        Compose the complete game board from background, grid, and entity layers.

        Returns the layered board ready for scaling and display. Grid lines are
        rendered on a slightly padded surface to accommodate boundary lines.

        Returns:
            pygame.Surface: The composed game board with all layers rendered
        """
        self._entity_layer.fill(self.CLEAR)

        # Draw terrain (background layer, drawn first)
        if hasattr(self.env, "terrain") and self.env.terrain:
            for terrain_entity in self.env.terrain.values():
                sprite = self._get_cached_sprite(terrain_entity.render_sprite)
                pos = (
                    terrain_entity.position[0] * self.TILE_SIZE,
                    terrain_entity.position[1] * self.TILE_SIZE,
                )
                self._entity_layer.blit(sprite, pos)

        # Draw static obstacles/background entities
        if self.env.entities is not None:
            for entity in self.env.entities.values():
                sprite = self._get_cached_sprite(entity.render_sprite)
                pos = (
                    entity.position[0] * self.TILE_SIZE,
                    entity.position[1] * self.TILE_SIZE,
                )
                self._entity_layer.blit(sprite, pos)

        # Draw prey
        for prey_item in self.env.prey.values():
            sprite = self._get_cached_sprite(prey_item.render_sprite)
            pos = (
                prey_item.position[0] * self.TILE_SIZE,
                prey_item.position[1] * self.TILE_SIZE,
            )
            self._entity_layer.blit(sprite, pos)

        # Draw predators
        for predator in self.env.predators.values():
            sprite = self._get_cached_sprite(predator.render_sprite)
            pos = (
                predator.position[0] * self.TILE_SIZE,
                predator.position[1] * self.TILE_SIZE,
            )
            self._entity_layer.blit(sprite, pos)

        # Layer composition: background + grid (with boundary buffer) + entities + vision mask
        padded_size = (
            self._game_surface_size[0] + self.GRID_BOUNDARY_BUFFER,
            self._game_surface_size[1] + self.GRID_BOUNDARY_BUFFER,
        )
        board = pygame.Surface(padded_size).convert()
        board.fill(self.GAMEBOARD_COLOR)
        board.blit(self._game_surface, (0, 0))
        board.blit(self._grid_layer, (0, 0))
        board.blit(self._entity_layer, (0, 0))

        # Darken what an agent cannot see. gui_config["vision_mask"]: "auto" (default) masks to the
        # first predator with limited vision, None shows the whole board, an agent id masks to
        # that agent's view
        vision_mask = self.gui_config.get("vision_mask", "auto")
        if vision_mask == "auto":
            limited_vision_predator = self._get_limited_vision_predator()
        elif vision_mask is None:
            limited_vision_predator = None
        else:
            limited_vision_predator = self.env.predators.get(vision_mask)
        if limited_vision_predator:
            vision_mask = self._create_vision_mask(
                limited_vision_predator.position,
                limited_vision_predator.limited_vision_radius,
            )
            board.blit(vision_mask, (0, 0))

        return board

    def _get_info_text_surface(self) -> pygame.Surface:
        """
        Get or regenerate the info text surface with caching.

        Regenerates text only when the score changes, otherwise returns cached surface
        for better performance. Custom info text (if present) is never cached and
        may span several lines separated by "\\n" (reserve space for them with
        gui_config["info_text_lines"]).

        Returns:
            pygame.Surface: The rendered text surface
        """
        if hasattr(self.env, "info") and self.env.info:
            info_text = self.env.info
            current_score = None  # Don't cache custom info
        else:
            info_text = f"Score: {self.env.score}"
            current_score = self.env.score

        # Regenerate text only if score changed (custom info is always regenerated)
        if (
            current_score is None
            or current_score != self._last_score
            or self._info_text_surface is None
        ):
            lines = [
                self._font.render(line, True, self.TEXT_COLOR)
                for line in str(info_text).split("\n")
            ]
            if len(lines) == 1:
                self._info_text_surface = lines[0]
            else:
                surface = pygame.Surface(
                    (
                        max(line.get_width() for line in lines),
                        sum(line.get_height() for line in lines),
                    ),
                    pygame.SRCALPHA,
                )
                y = 0
                for line in lines:
                    surface.blit(line, (0, y))
                    y += line.get_height()
                self._info_text_surface = surface
            self._last_score = current_score

        return self._info_text_surface

    def _render_info_bar(self) -> None:
        """
        Render the info bar with score and UI elements.
        """
        self._screen.blit(self._info_layer, (0, self._info_bar_y))

        text_surface = self._get_info_text_surface()
        text_pos = (self.TEXT_PADDING, self._info_bar_y + self.TEXT_PADDING)
        self._screen.blit(text_surface, text_pos)

    def _get_cached_sprite(self, path: str | os.PathLike) -> pygame.Surface:
        """
        Get sprite surface from cache, loading and caching it if necessary.

        Sprites are cached in memory after first load for performance. All sprites
        are normalized to self.TILE_SIZE x self.TILE_SIZE to ensure they fit within a single
        grid cell. If a sprite file is missing, returns a blank surface as fallback.

        Args:
            path: Path to the sprite image file

        Returns:
            pygame.Surface: The sprite surface (self.TILE_SIZE x self.TILE_SIZE) ready for blitting
        """
        path_str = str(path)
        if path_str not in self._sprite_cache:
            try:
                sprite = self._load_image(path_str)
                # Ensure sprite fits within a single grid cell
                if sprite.get_size() != (self.TILE_SIZE, self.TILE_SIZE):
                    sprite = pygame.transform.scale(
                        sprite, (self.TILE_SIZE, self.TILE_SIZE)
                    )
                self._sprite_cache[path_str] = sprite
            except (FileNotFoundError, pygame.error, OSError) as e:
                # Fallback: return blank surface instead of crashing
                print(f"Warning: Failed to load sprite at {path_str}: {e}")
                self._sprite_cache[path_str] = pygame.Surface(
                    (self.TILE_SIZE, self.TILE_SIZE)
                )

        return self._sprite_cache[path_str]

    def clear_sprite_cache(self) -> None:
        """
        Clear all cached sprite surfaces from memory.

        Use this if assets have changed at runtime and you want to reload them.
        After calling, sprites will be reloaded from disk on next render.
        """
        self._sprite_cache.clear()

    def get_frame(self) -> np.ndarray:
        """
        Export the current frame as an RGB image array.

        Useful for recording gameplay videos, computing vision-based features,
        or integrating with other computer vision pipelines. Returns the current
        screen buffer (including info bar if enabled).

        Returns:
            np.ndarray: RGB image with shape (height, width, 3), uint8 values [0-255]
        """
        return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2))

    def close(self) -> None:
        """
        Clean up and release all pygame resources.

        Call this when shutting down the environment to properly free memory
        and close the pygame window. After calling close(), the renderer
        cannot be used again without reinitializing.
        """
        self.clear_sprite_cache()
        pygame.quit()

    def _draw_grid(self) -> None:
        """
        Draw grid lines on the static grid layer.

        Grid lines are drawn once during initialization and scaled based on the
        zoom level to remain visible at all scales. Line width is proportional
        to the scale factor to maintain visual consistency.
        """
        w, h = self.env.grid_size

        # Scale line width with zoom (thicker at higher zooms, minimum 1px)
        line_width = max(1, int(self._scale * 0.25))

        # Draw vertical lines
        for x in range(0, (w + 1) * self.TILE_SIZE, self.TILE_SIZE):
            pygame.draw.line(
                self._grid_layer,
                self.GRID_LINE_COLOR,
                (x, 0),
                (x, h * self.TILE_SIZE),
                line_width,
            )

        # Draw horizontal lines
        for y in range(0, (h + 1) * self.TILE_SIZE, self.TILE_SIZE):
            pygame.draw.line(
                self._grid_layer,
                self.GRID_LINE_COLOR,
                (0, y),
                (w * self.TILE_SIZE, y),
                line_width,
            )

    def _load_image(self, path: str, clean_image: bool = False) -> pygame.Surface:
        """
        Load an image file as an optimized pygame surface.

        Converts images to alpha-enabled surfaces for efficient blitting.
        Optionally removes ICC profiles from PNG files to suppress warnings.

        Args:
            path: Path to the image file
            clean_image: If True, strips ICC profile from PNG during save (one-time operation)

        Returns:
            pygame.Surface: Optimized surface ready for rendering
        """
        if clean_image:
            with Image.open(path) as img:
                if img.info.get("icc_profile"):
                    img = img.convert("RGBA")
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG", icc_profile=None)
                    img_bytes.seek(0)
                    img.save(path, format="PNG", icc_profile=None)
                    return pygame.image.load(img_bytes).convert_alpha()
                else:
                    return pygame.image.load(str(path)).convert_alpha()
        else:
            return pygame.image.load(path).convert_alpha()
