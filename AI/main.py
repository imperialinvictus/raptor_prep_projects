import pygame, sys
from maze import Maze
from player import Player
from game import Game
from clock import Clock
from obstacle import Obstacle
import random

pygame.init()
pygame.font.init()

class Main():
	TRAP_RED_COLOR = pygame.Color(180, 0, 0)
	MIN_TRAP_INTERVAL = 500
	MAX_TRAP_INTERVAL = 2500
	NUM_TRAPS = 20

	def __init__(self, screen):
		self.screen = screen
		self.font = pygame.font.SysFont("impact", 30)
		self.message_color = pygame.Color("cyan")
		self.running = True
		self.game_over = False
		self.FPS = pygame.time.Clock()
		self.obstacles = [] 
		self.hit_obstacle = False
		self.trap_cells = []


	def instructions(self):
		instructions1 = self.font.render('Use', True, self.message_color)
		instructions2 = self.font.render('Arrow Keys', True, self.message_color)
		instructions3 = self.font.render('to Move', True, self.message_color)
		self.screen.blit(instructions1,(655,300))
		self.screen.blit(instructions2,(610,331))
		self.screen.blit(instructions3,(630,362))

	def create_obstacles(self, maze, num_obstacles, tile_size):
		"""
		Creates a specified number of obstacles in suitable maze cells.
		"""
		self.obstacles = [] # Clear previous obstacles
		suitable_cells = []
		goal_cell_coords = (maze.grid_cells[-1].x, maze.grid_cells[-1].y)

		for cell in maze.grid_cells:
			if (cell.x == 0 and cell.y == 0) or (cell.x == goal_cell_coords[0] and cell.y == goal_cell_coords[1]):
				continue

			is_vertical_passage = not cell.walls['top'] and not cell.walls['bottom'] and cell.walls['left'] and cell.walls['right']
			is_horizontal_passage = not cell.walls['left'] and not cell.walls['right'] and cell.walls['top'] and cell.walls['bottom']

			if is_vertical_passage or is_horizontal_passage:
				suitable_cells.append(cell)

		num_to_create = min(num_obstacles, len(suitable_cells))

		if num_to_create > 0:
			chosen_cells = random.sample(suitable_cells, num_to_create)

			for cell in chosen_cells:
				# Create obstacle with details from the maze and cell
				obstacle = Obstacle(cell, tile_size, maze.thickness)
				self.obstacles.append(obstacle)


	def _select_and_init_traps(self, maze, num_traps, tile_size):
		self.trap_cells = []
		potential_cells = []
		start_cell_coords = (0, 0)
		goal_cell_coords = (maze.grid_cells[-1].x, maze.grid_cells[-1].y)

		for cell in maze.grid_cells:
			if (cell.x, cell.y) != start_cell_coords and (cell.x, cell.y) != goal_cell_coords:
				potential_cells.append(cell)

		num_to_create = min(num_traps, len(potential_cells))

		if num_to_create > 0:
			chosen_cells = random.sample(potential_cells, num_to_create)
			current_time = pygame.time.get_ticks() # Get current time once

			for cell in chosen_cells:
				cell.is_trap = True
				cell.trap_state = random.choice(['original', 'red']) # Start with a random state
				cell.trap_timer_start = current_time
				cell.trap_state_duration = random.randint(self.MIN_TRAP_INTERVAL, self.MAX_TRAP_INTERVAL)
				self.trap_cells.append(cell)

	def _update_traps(self):
		current_time = pygame.time.get_ticks()
		for cell in self.trap_cells:
			if current_time - cell.trap_timer_start > cell.trap_state_duration:
				previous_state = cell.trap_state
				possible_states = ['original', 'red']
				if len(possible_states) > 1 and previous_state in possible_states:
					possible_states.remove(previous_state)

				cell.trap_state = random.choice(possible_states)
				cell.trap_timer_start = current_time
				cell.trap_state_duration = random.randint(self.MIN_TRAP_INTERVAL, self.MAX_TRAP_INTERVAL)


	# draws all configs; maze, player, instructions, obstacles, and time
	def _draw(self, maze, tile, player, game, clock):
		for cell in maze.grid_cells: # Iterate all cells to find traps
			if cell.is_trap:
				cell_rect = pygame.Rect(cell.x * tile, cell.y * tile, tile, tile)
				if cell.trap_state == 'red':
					pygame.draw.rect(self.screen, self.TRAP_RED_COLOR, cell_rect)

		# draw maze walls (cell.draw in cell.py only draws walls)
		[cell.draw(self.screen, tile) for cell in maze.grid_cells]

		# add a goal point to reach
		game.add_goal_point(self.screen)

		# draw every player movement
		player.draw(self.screen)
		player.update() # Player update now uses the modified logic

		for obstacle in self.obstacles:
			if not self.game_over: # Stop moving obstacles when game ends
				obstacle.move()
			
			obstacle.draw(self.screen)

			if obstacle.check_collision(player.rect):
				clock.stop_timer()
				self.hit_obstacle = True

		if not self.hit_obstacle: 
			player_rect_on_grid = player.rect
			for trap_cell_object in self.trap_cells: 
				if trap_cell_object.is_trap and trap_cell_object.trap_state == 'red':
					trap_cell_render_rect = pygame.Rect(trap_cell_object.x * tile - maze.thickness, trap_cell_object.y * tile - maze.thickness, tile, tile)
					if player_rect_on_grid.colliderect(trap_cell_render_rect):
						clock.stop_timer()
						self.hit_obstacle = True # Game over
						break 
						
		# instructions, clock, winning message
		self.instructions()
		if self.game_over:
			clock.stop_timer()
			if not self.hit_obstacle:
				self.screen.blit(game.message(),(610,120))
			else:
				self.screen.blit(game.lose_message(),(610,120))
		else:
			clock.update_timer()
		self.screen.blit(clock.display_timer(), (625,200))

		pygame.display.flip()

	# main game loop
	def main(self, frame_size, tile):
		cols, rows = frame_size[0] // tile, frame_size[-1] // tile
		maze = Maze(cols, rows)
		game = Game(maze.grid_cells[-1], tile)
		player = Player(tile // 3, tile // 3)
		clock = Clock()

		maze.generate_maze()
		self._select_and_init_traps(maze, self.NUM_TRAPS, tile)
		num_obstacles_to_create = 15
		self.create_obstacles(maze, num_obstacles_to_create, tile)

		clock.start_timer()
		while self.running:
			self.screen.fill("gray")
			self.screen.fill( pygame.Color("darkslategray"), (603, 0, 752, 752))
			self._update_traps() 
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
			
			# if keys were pressed still
			if event.type == pygame.KEYDOWN:
				if not self.game_over:
					if event.key == pygame.K_LEFT:
						player.left_pressed = True
					if event.key == pygame.K_RIGHT:
						player.right_pressed = True
					if event.key == pygame.K_UP:
						player.up_pressed = True
					if event.key == pygame.K_DOWN:
						player.down_pressed = True
					player.check_move(tile, maze.grid_cells, maze.thickness)
		
			# if pressed key released
			if event.type == pygame.KEYUP:
				if not self.game_over:
					if event.key == pygame.K_LEFT:
						player.left_pressed = False
					if event.key == pygame.K_RIGHT:
						player.right_pressed = False
					if event.key == pygame.K_UP:
						player.up_pressed = False
					if event.key == pygame.K_DOWN:
						player.down_pressed = False
					player.check_move(tile, maze.grid_cells, maze.thickness)

			if game.is_game_over(player) or self.hit_obstacle:
				self.game_over = True
				player.left_pressed = False
				player.right_pressed = False
				player.up_pressed = False
				player.down_pressed = False

			self._draw(maze, tile, player, game, clock)
			self.FPS.tick(60)


if __name__ == "__main__":
	window_size = (602, 602)
	screen = (window_size[0] + 150, window_size[-1])
	tile_size = 30
	screen = pygame.display.set_mode(screen)
	pygame.display.set_caption("Maze")

	game = Main(screen)
	game.main(window_size, tile_size)