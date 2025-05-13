import pygame
import random

class Obstacle:
    """
    Represents an obstacle moving back and forth within a single maze cell,
    constrained by the cell's remaining walls.
    """
    DEFAULT_SPEED = 0.75
    DEFAULT_COLOR = (255, 0, 0) # Red
    DEFAULT_SIZE = 7 # Diameter in pixels

    def __init__(self, cell, tile_size, thickness,
                 speed=DEFAULT_SPEED, color=DEFAULT_COLOR, size=DEFAULT_SIZE):
        """
        Initializes an obstacle within a specific cell.

        Args:
            cell (Cell): The maze cell this obstacle belongs to.
            tile_size (int): The pixel dimension of each square tile.
            thickness (int): The pixel thickness of the maze walls.
            speed (int, optional): Movement speed in pixels per frame. Defaults to DEFAULT_SPEED.
            color (tuple, optional): RGB color tuple. Defaults to DEFAULT_COLOR.
            size (int, optional): Diameter of the obstacle in pixels. Defaults to DEFAULT_SIZE.
        """
        self.cell = cell
        self.tile_size = tile_size
        self.thickness = 1
        self.speed = speed
        self.color = color
        self.size = size
        self.radius = self.size // 2

        # Determine movement axis based on missing walls
        self.is_vertical = False
        self.is_horizontal = False
        if not self.cell.walls['top'] and not self.cell.walls['bottom']:
            self.is_horizontal = True
        elif not self.cell.walls['left'] and not self.cell.walls['right']:
            self.is_vertical = True
        else:
            self.speed = 0 # Force stationary

        # Calculate pixel boundaries for the center of the obstacle within the cell
        cell_pixel_x_start = self.cell.x * self.tile_size
        cell_pixel_y_start = self.cell.y * self.tile_size
        cell_pixel_x_end = cell_pixel_x_start + self.tile_size
        cell_pixel_y_end = cell_pixel_y_start + self.tile_size

        # Default coordinates (center of cell), used if stationary
        center_x = cell_pixel_x_start + self.tile_size // 2
        center_y = cell_pixel_y_start + self.tile_size // 2

        if self.is_vertical:
            # Min/Max Y the center can reach
            self.min_coord_pixel = (center_x, cell_pixel_y_start + self.thickness + self.radius)
            self.max_coord_pixel = (center_x, cell_pixel_y_end - self.thickness - self.radius)
            self.current_speed = self.speed
            # Initial position
            self.x = center_x
            self.y = random.uniform(self.min_coord_pixel[1], self.max_coord_pixel[1])

        elif self.is_horizontal:
            # Min/Max X the center can reach
            self.min_coord_pixel = (cell_pixel_x_start + self.thickness + self.radius, center_y)
            self.max_coord_pixel = (cell_pixel_x_end - self.thickness - self.radius, center_y)
            self.current_speed = self.speed
            # Initial position
            self.x = random.uniform(self.min_coord_pixel[0], self.max_coord_pixel[0])
            self.y = center_y
        else: # Stationary
             self.min_coord_pixel = (center_x, center_y)
             self.max_coord_pixel = (center_x, center_y)
             self.current_speed = 0
             self.x = center_x
             self.y = center_y


        # Randomize starting direction if moving
        if self.current_speed != 0 and random.choice([True, False]):
             self.current_speed *= -1

        # Pygame Rect for collision detection and drawing
        self.rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.size, self.size)
        self.rect.center = (int(self.x), int(self.y)) # Ensure rect starts centered

    def move(self):
        """Moves the obstacle within its cell and reverses direction at boundaries."""
        if self.current_speed == 0:
            return # Stationary

        if self.is_vertical:
            self.y += self.current_speed
            # Check boundaries and reverse
            if self.y <= self.min_coord_pixel[1]:
                self.y = self.min_coord_pixel[1] # Clamp position
                self.current_speed *= -1 # Reverse direction
            elif self.y >= self.max_coord_pixel[1]:
                self.y = self.max_coord_pixel[1] # Clamp position
                self.current_speed *= -1 # Reverse direction

        elif self.is_horizontal:
            self.x += self.current_speed
            # Check boundaries and reverse
            if self.x <= self.min_coord_pixel[0]:
                self.x = self.min_coord_pixel[0] # Clamp position
                self.current_speed *= -1 # Reverse direction
            elif self.x >= self.max_coord_pixel[0]:
                self.x = self.max_coord_pixel[0] # Clamp position
                self.current_speed *= -1 # Reverse direction

        # Update rect position (center based)
        self.rect.center = (int(self.x), int(self.y))

    def draw(self, screen):
        """Draws the obstacle on the screen."""
        pygame.draw.circle(screen, self.color, self.rect.center, self.radius)

    def check_collision(self, player_rect):
        """
        Checks if the obstacle collides with the player.
        Args:
            player_rect (pygame.Rect): The rectangle representing the player.
        Returns:
            bool: True if collision occurs, False otherwise.
        """
        return self.rect.colliderect(player_rect)