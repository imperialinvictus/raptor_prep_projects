import pygame
import sys
from maze import Maze
from player import Player
from game import Game
from clock import Clock
from obstacle import Obstacle
from rl_agent import RLAgent
import numpy as np
import time
import random
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

class RLTraining:
    def __init__(self, screen, window_size, tile_size):
        self.screen = screen
        self.window_size = window_size
        self.tile_size = tile_size
        self.cols, self.rows = window_size[0] // tile_size, window_size[-1] // tile_size
        self.agent = RLAgent()
        self.episodes = 2000
        self.max_steps = 1000
        self.training = True
        self.visualize_every = 1
        self.show_progress = True
        self.visualization_speed = 30
        self.font = pygame.font.SysFont("Arial", 24)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.rewards_history = []
        self.success_rate = []
        self.exploration_history = []
        self.current_episode = 0
        self.save_path = "trained_agent.pkl"
        self.clock = pygame.time.Clock()
        self.manual_mode = False
        self.MIN_TRAP_INTERVAL = 500
        self.MAX_TRAP_INTERVAL = 2500
        self.NUM_TRAPS = 20
        self.background_training = False
        self.last_visualized_episode = 0
        self.visualization_episode = 0
        self.stats_panel_width = 150

    def train_episode(self):
        """Train a single episode"""
        self.reset_environment()
        state = self.agent.get_state(self.player, self.maze.grid_cells, self.tile_size)
        total_reward = 0
        success = False
        
        for step in range(self.max_steps):
            action = self.agent.choose_action(state)
            movement = self.agent.convert_action_to_movement(action)
            
            self.player.left_pressed = movement['left_pressed']
            self.player.right_pressed = movement['right_pressed']
            self.player.up_pressed = movement['up_pressed']
            self.player.down_pressed = movement['down_pressed']
            
            self.player.check_move(self.tile_size, self.maze.grid_cells, self.maze.thickness)
            self.player.update()
            
            reward = self.agent.get_reward(self.player, self.game, self.maze.grid_cells, self.tile_size, action)
            next_state = self.agent.get_state(self.player, self.maze.grid_cells, self.tile_size)
            
            self.agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if self.game.is_game_over(self.player):
                success = True
                break
        
        return total_reward, success

    def train(self, show_visualization=False):
        """Train the RL agent"""
        running = True
        self.paused = False
        start_time = time.time()
        
        while running and self.current_episode < self.episodes:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_b:
                        self.background_training = not self.background_training
                        print(f"Background training: {'ON' if self.background_training else 'OFF'}")
            
            if not running:
                # Plot stats if quitting early
                if self.current_episode > 0:
                    print("\nTraining interrupted. Plotting results so far...")
                    self.plot_training_results()
                    print("Training plots saved as 'training_results.png'")
                break
            
            if not self.paused:
                # Train a single episode
                self.current_episode += 1
                total_reward, success = self.train_episode()
                
                # Record results
                self.rewards_history.append(total_reward)
                self.success_rate.append(1 if success else 0)
                self.exploration_history.append(self.agent.exploration_rate)
                
                # Print progress
                if self.current_episode % 10 == 0:
                    avg_reward = np.mean(self.rewards_history[-10:])
                    success_rate = np.mean(self.success_rate[-10:]) if self.success_rate else 0
                    elapsed_time = time.time() - start_time
                    episodes_per_second = self.current_episode / elapsed_time
                    print(f"Episode {self.current_episode}/{self.episodes}, "
                          f"Avg Reward (last 10): {avg_reward:.2f}, "
                          f"Success Rate: {success_rate*100:.1f}%, "
                          f"Speed: {episodes_per_second:.1f} episodes/sec")
            
            # Visualize if needed
            if show_visualization:
                self.render_frame()
                pygame.display.flip()
                self.clock.tick(self.visualization_speed)
            
            pygame.event.pump()
        
        if running:
            training_time = time.time() - start_time
            final_success_rate = np.mean(self.success_rate[-100:]) if self.success_rate else 0
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Final success rate: {final_success_rate*100:.1f}%")
            print(f"Average speed: {self.current_episode/training_time:.1f} episodes/second")
            
            # Plot training results
            print("\nGenerating training plots...")
            self.plot_training_results()
            print("Training plots saved as 'training_results.png'")
            
            # Save the trained agent
            try:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.agent, f)
                print(f"Trained agent saved to {self.save_path}")
            except Exception as e:
                print(f"Failed to save trained agent: {e}")

    def test_agent(self, num_episodes=10, demonstration_mode=True, demonstration_speed=1):
        """Test the trained agent with optional demonstration mode"""
        self.training = False
        self.agent.exploration_rate = 0  # No exploration during testing
        running = True
        
        print("\nStarting test phase...")
        if demonstration_mode:
            print("Demonstration mode: ON (Press 'D' to toggle speed, 'SPACE' to pause)")
        
        for episode in range(num_episodes):
            if not running:
                break
                
            self.current_episode = episode + 1
            self.reset_environment()
            state = self.agent.get_state(self.player, self.maze.grid_cells, self.tile_size)
            total_reward = 0
            paused = False
            
            while True:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_SPACE and demonstration_mode:
                            paused = not paused
                
                if not running:
                    break
                
                if paused:
                    # Draw paused state
                    self.render_frame()
                    self.draw_stats(total_reward, self.game.is_game_over(self.player))
                    paused_text = self.big_font.render("PAUSED", True, pygame.Color("red"))
                    text_rect = paused_text.get_rect(center=(self.window_size[0]//2, self.window_size[1]//2))
                    self.screen.blit(paused_text, text_rect)
                    pygame.display.flip()
                    pygame.event.pump()
                    continue
                
                # Choose best action
                action = self.agent.choose_action(state)
                movement = self.agent.convert_action_to_movement(action)
                
                # Apply movement
                self.player.left_pressed = movement['left_pressed']
                self.player.right_pressed = movement['right_pressed']
                self.player.up_pressed = movement['up_pressed']
                self.player.down_pressed = movement['down_pressed']
                
                # Update player
                self.player.check_move(self.tile_size, self.maze.grid_cells, self.maze.thickness)
                self.player.update()
                
                # Get reward
                reward = self.agent.get_reward(self.player, self.game, self.maze.grid_cells, self.tile_size)
                total_reward += reward
                
                # Render
                self.render_frame()
                self.draw_stats(total_reward, self.game.is_game_over(self.player))
                
                # Add demonstration mode indicator
                if demonstration_mode:
                    demo_text = self.font.render("DEMO MODE", True, pygame.Color("yellow"))
                    self.screen.blit(demo_text, (self.window_size[0] + 10, 400))
                
                pygame.display.flip()
                
                # Control demonstration speed
                if demonstration_mode:
                    self.clock.tick(demonstration_speed)
                else:
                    self.clock.tick(60)
                
                pygame.event.pump()
                
                # Check if episode is done
                if self.game.is_game_over(self.player) or self.hit_obstacle:
                    result = "Success" if self.game.is_game_over(self.player) else "Failed"
                    print(f"Test Episode {episode + 1}: {result} (Reward: {total_reward:.2f})")
                    if demonstration_mode:
                        # Pause briefly at the end of each episode
                        pygame.time.wait(1000)
                    break
                
                state = self.agent.get_state(self.player, self.maze.grid_cells, self.tile_size)

    def load_agent(self):
        """Load the trained agent from file"""
        try:
            # Check if file exists and has content
            if not os.path.exists(self.save_path) or os.path.getsize(self.save_path) == 0:
                print("No valid saved agent found.")
                return False
                
            with open(self.save_path, 'rb') as f:
                try:
                    self.agent = pickle.load(f)
                    # Verify the loaded agent has required attributes
                    if not hasattr(self.agent, 'q_table') or not hasattr(self.agent, 'exploration_rate'):
                        print("Saved agent file is corrupted.")
                        return False
                    return True
                except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                    print(f"Error loading saved agent: {e}")
                    # Create backup of corrupted file
                    if os.path.exists(self.save_path):
                        backup_path = f"{self.save_path}.bak"
                        os.rename(self.save_path, backup_path)
                        print(f"Corrupted save file backed up to {backup_path}")
                    return False
        except Exception as e:
            print(f"Unexpected error loading agent: {e}")
            return False

    def reset_environment(self):
        """Reset the game environment for a new episode"""
        self.maze = Maze(self.cols, self.rows)
        self.game = Game(self.maze.grid_cells[-1], self.tile_size)
        self.player = Player(self.tile_size // 3, self.tile_size // 3)
        self.maze.generate_maze()
        self.obstacles = []
        self.hit_obstacle = False
        self.game_over = False
        self.steps = 0
        
        # Get goal position and set it in the agent
        goal_cell = self.maze.grid_cells[-1]
        goal_x, goal_y = goal_cell.x, goal_cell.y
        self.agent.set_maze_dimensions(self.cols, self.rows, goal_x, goal_y)
        
        # Initialize traps like in main game
        self.trap_cells = []
        potential_cells = []
        start_cell_coords = (0, 0)
        goal_cell_coords = (goal_x, goal_y)

        for cell in self.maze.grid_cells:
            if (cell.x, cell.y) != start_cell_coords and (cell.x, cell.y) != goal_cell_coords:
                potential_cells.append(cell)

        num_to_create = min(self.NUM_TRAPS, len(potential_cells))

        if num_to_create > 0:
            chosen_cells = random.sample(potential_cells, num_to_create)
            current_time = pygame.time.get_ticks()

            for cell in chosen_cells:
                cell.is_trap = True
                cell.trap_state = random.choice(['original', 'red'])
                cell.trap_timer_start = current_time
                cell.trap_state_duration = random.randint(self.MIN_TRAP_INTERVAL, self.MAX_TRAP_INTERVAL)
                self.trap_cells.append(cell)
        
        # Initialize obstacles like in main game
        self.create_obstacles()
        
    def create_obstacles(self):
        """Create obstacles in the maze like in main game"""
        self.obstacles = []
        suitable_cells = []
        goal_cell_coords = (self.maze.grid_cells[-1].x, self.maze.grid_cells[-1].y)

        for cell in self.maze.grid_cells:
            if (cell.x == 0 and cell.y == 0) or (cell.x == goal_cell_coords[0] and cell.y == goal_cell_coords[1]):
                continue

            is_vertical_passage = not cell.walls['top'] and not cell.walls['bottom'] and cell.walls['left'] and cell.walls['right']
            is_horizontal_passage = not cell.walls['left'] and not cell.walls['right'] and cell.walls['top'] and cell.walls['bottom']

            if is_vertical_passage or is_horizontal_passage:
                suitable_cells.append(cell)

        num_obstacles = min(15, len(suitable_cells))
        num_to_create = min(num_obstacles, len(suitable_cells))

        if num_to_create > 0:
            chosen_cells = random.sample(suitable_cells, num_to_create)
            for cell in chosen_cells:
                obstacle = Obstacle(cell, self.tile_size, self.maze.thickness)
                self.obstacles.append(obstacle)
                
    def draw_stats(self, episode_reward, success):
        """Draw training statistics on the screen"""
        # Draw background for stats
        stats_rect = pygame.Rect(self.window_size[0], 0, self.stats_panel_width, self.window_size[-1])
        pygame.draw.rect(self.screen, pygame.Color("darkslategray"), stats_rect)
        y = 40
        # Episode label and number on separate lines
        episode_label = self.font.render("Episode:", True, pygame.Color("white"))
        self.screen.blit(episode_label, (self.window_size[0] + 20, y))
        y += 40
        episode_num = self.font.render(str(self.current_episode), True, pygame.Color("yellow"))
        self.screen.blit(episode_num, (self.window_size[0] + 20, y))
        y += 40
        # Mode label and value on separate lines
        mode_label = self.font.render("Mode:", True, pygame.Color("white"))
        self.screen.blit(mode_label, (self.window_size[0] + 20, y))
        y += 40
        mode_text = "Training" if self.training else "Demo"
        mode_value = self.font.render(mode_text, True, pygame.Color("yellow" if not self.training else "white"))
        self.screen.blit(mode_value, (self.window_size[0] + 20, y))

    def plot_training_results(self):
        """Plot only essential training metrics"""
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards with moving average
        episodes = range(1, len(self.rewards_history) + 1)
        window_size = 100
        if len(self.rewards_history) >= window_size:
            moving_avg = np.convolve(self.rewards_history, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            ax1.plot(range(window_size, len(self.rewards_history) + 1), 
                    moving_avg, 'r-', linewidth=2, label='Average Reward')
        ax1.set_title('Learning Progress', pad=20, fontsize=12)
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Reward', fontsize=10)
        ax1.legend(fontsize=10)
        
        # Plot success rate
        if self.success_rate:
            success_rate = np.array(self.success_rate)
            window_size = min(100, len(success_rate))
            moving_avg = np.convolve(success_rate, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            ax2.plot(range(window_size, len(success_rate) + 1), 
                    moving_avg * 100, 'g-', linewidth=2)
            ax2.set_xlabel('Episode', fontsize=10)
            ax2.set_ylabel('Success Rate (%)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print final performance metrics
        final_avg_reward = np.mean(self.rewards_history[-100:])
        final_success_rate = np.mean(self.success_rate[-100:]) * 100
        print(f"\nFinal Performance Metrics:")
        print(f"Average Reward (last 100 episodes): {final_avg_reward:.2f}")
        print(f"Success Rate (last 100 episodes): {final_success_rate:.1f}%")
        
    def render_frame(self):
        """Render a single frame of the game with status in sidebar"""
        # Draw game area
        game_surface = pygame.Surface(self.window_size)
        game_surface.fill("gray")
        
        # Draw maze and game elements
        [cell.draw(game_surface, self.tile_size) for cell in self.maze.grid_cells]
        
        # Update and draw traps
        self.update_traps()
        for cell in self.trap_cells:
            if cell.trap_state == 'red':
                pygame.draw.rect(
                    game_surface,
                    pygame.Color(180, 0, 0),
                    (cell.x * self.tile_size, cell.y * self.tile_size, self.tile_size, self.tile_size),
                    0
                )
        
        # Draw goal and player
        self.game.add_goal_point(game_surface)
        self.player.draw(game_surface)
        
        # Update and draw obstacles
        for obstacle in self.obstacles:
            if not self.game_over:
                obstacle.move()
            obstacle.draw(game_surface)
            if obstacle.check_collision(self.player.rect):
                self.hit_obstacle = True
        
        # Check trap collisions
        if not self.hit_obstacle:
            player_rect_on_grid = self.player.rect
            for trap in self.trap_cells:
                if trap.is_trap and trap.trap_state == 'red':
                    trap_cell_render_rect = pygame.Rect(trap.x * self.tile_size, trap.y * self.tile_size, self.tile_size, self.tile_size)
                    if player_rect_on_grid.colliderect(trap_cell_render_rect):
                        self.hit_obstacle = True
                        break
        
        # Draw game surface to main screen
        self.screen.blit(game_surface, (0, 0))
        
        # Draw sidebar
        sidebar_rect = pygame.Rect(self.window_size[0], 0, self.stats_panel_width, self.window_size[1])
        pygame.draw.rect(self.screen, pygame.Color("darkslategray"), sidebar_rect)
        
        # Draw minimal status info in sidebar with episode numbers on separate lines
        y_offset = 20
        current_ep = str(self.current_episode)
        total_ep = str(self.episodes)
        
        # Draw current episode
        ep_text = self.font.render("Current:", True, pygame.Color("white"))
        self.screen.blit(ep_text, (self.window_size[0] + 10, y_offset))
        y_offset += 30
        
        current_ep_text = self.font.render(current_ep, True, pygame.Color("yellow"))
        self.screen.blit(current_ep_text, (self.window_size[0] + 10, y_offset))
        y_offset += 40
        
        # Draw total episodes
        total_text = self.font.render("Total:", True, pygame.Color("white"))
        self.screen.blit(total_text, (self.window_size[0] + 10, y_offset))
        y_offset += 30
        
        total_ep_text = self.font.render(total_ep, True, pygame.Color("yellow"))
        self.screen.blit(total_ep_text, (self.window_size[0] + 10, y_offset))
        y_offset += 40
        
        # Draw mode information
        mode_text = "BG Mode" if self.background_training else "Normal"
        mode_surface = self.font.render(mode_text, True, pygame.Color("white"))
        self.screen.blit(mode_surface, (self.window_size[0] + 10, y_offset))
        y_offset += 30
        
        if not self.training:
            test_text = self.font.render("Test Mode", True, pygame.Color("white"))
            self.screen.blit(test_text, (self.window_size[0] + 10, y_offset))
            y_offset += 30
        
        if hasattr(self, 'paused') and self.paused:
            pause_text = self.font.render("PAUSED", True, pygame.Color("red"))
            self.screen.blit(pause_text, (self.window_size[0] + 10, y_offset))

    def update_traps(self):
        """Update trap states like in main game"""
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

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    window_size = (800, 800)  # Game window size
    stats_panel_width = 150   # Sidebar width
    screen = (window_size[0] + stats_panel_width, window_size[-1])
    tile_size = 40
    screen = pygame.display.set_mode(screen)
    pygame.display.set_caption("Maze RL Training")
    
    # Create trainer
    trainer = RLTraining(screen, window_size, tile_size)
    trainer.stats_panel_width = stats_panel_width
    
    # Check if trained agent exists
    if trainer.load_agent():
        print("\nLoaded agent. Starting demonstration phase...")
        print("\nRunning demo episodes to visualize agent's performance...")
        trainer.test_agent(num_episodes=5, demonstration_mode=True, demonstration_speed=20)
    else:
        print("\nNo trained agent found. Starting new training...")
        print("Training will run with visualization enabled.")
        print("Controls:")
        print("  Press 'Q' to quit training")
        print("  Press 'SPACE' to pause/resume")
        print("  Press 'B' to toggle background training")
        trainer.train(show_visualization=True)
        print("\nTraining completed. Starting demonstration phase...")
        print("\nRunning demo episodes to visualize agent's performance...")
        trainer.test_agent(num_episodes=5, demonstration_mode=True, demonstration_speed=20)
    
    pygame.quit()
    sys.exit() 