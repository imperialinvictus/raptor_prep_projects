import numpy as np
import random
from collections import defaultdict

def default_q_values():
    """Default Q-values for new states"""
    return np.zeros(4)  # 4 actions: up, right, down, left

class RLAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.997):
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.exploration_rate = exploration_rate  # epsilon
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(default_q_values)  # Using regular function instead of lambda
        self.actions = [0, 1, 2, 3]  # 0: up, 1: right, 2: down, 3: left
        self.cols = None
        self.rows = None
        self.goal_x = None
        self.goal_y = None
        self.start_x = None
        self.start_y = None
        self.best_distance = float('inf')
        self.steps_without_improvement = 0
        self.min_exploration_rate = 0.01  # Minimum exploration rate
        self.last_action = None
        
    def set_maze_dimensions(self, cols, rows, goal_x, goal_y):
        """Set maze dimensions and goal position"""
        self.cols = cols
        self.rows = rows
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.start_x = 0
        self.start_y = 0
        self.best_distance = float('inf')
        self.steps_without_improvement = 0
        
    def get_state(self, player, grid_cells, tile):
        """Convert the current game state into a discrete state representation"""
        current_cell_x = player.x // tile
        current_cell_y = player.y // tile
        current_cell = player.get_current_cell(current_cell_x, current_cell_y, grid_cells)
        
        # Create a state tuple that represents the walls around the current cell
        walls = (
            int(current_cell.walls['top']),
            int(current_cell.walls['right']),
            int(current_cell.walls['bottom']),
            int(current_cell.walls['left'])
        )
        
        # Calculate relative position to goal
        dx = self.goal_x - current_cell_x
        dy = self.goal_y - current_cell_y
        
        # Normalize the direction to goal (using sign to reduce state space)
        goal_direction = (
            np.sign(dx),  # -1: goal is left, 0: same x, 1: goal is right
            np.sign(dy)   # -1: goal is up, 0: same y, 1: goal is down
        )
        
        # Add Manhattan distance to goal (discretized to reduce state space)
        manhattan_distance = abs(dx) + abs(dy)
        distance_category = min(manhattan_distance // 3, 3)  # Categories: 0-2, 3-5, 6-8, 9+
        
        return (walls, goal_direction, distance_category)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(self.actions)
        else:
            # Exploit: choose best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning update formula
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state][action] = new_value
        
        # Decay exploration rate with minimum threshold
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)
    
    def get_reward(self, player, game, grid_cells, tile, action=None):
        """Calculate reward for the current state"""
        current_cell_x = player.x // tile
        current_cell_y = player.y // tile
        current_cell = player.get_current_cell(current_cell_x, current_cell_y, grid_cells)
        
        # Calculate current distance to goal
        current_distance = abs(self.goal_x - current_cell_x) + abs(self.goal_y - current_cell_y)
        
        # Base reward
        reward = -0.1  # Step penalty to encourage efficiency
        
        # Check if player hit a wall
        if (player.x <= current_cell_x * tile + 5 and current_cell.walls['left']) or \
           (player.x >= current_cell_x * tile + tile - 15 and current_cell.walls['right']) or \
           (player.y <= current_cell_y * tile + 5 and current_cell.walls['top']) or \
           (player.y >= current_cell_y * tile + tile - 15 and current_cell.walls['bottom']):
            reward = -1.0  # Penalty for hitting wall
            return reward
        
        # Check for trap collision (only red traps are dangerous)
        if current_cell.is_trap and current_cell.trap_state == 'red':
            reward = -2.0  # Trap penalty
            return reward
        
        # Check if reached goal
        if game.is_game_over(player):
            reward = 20.0  # Goal reward
            return reward
        
        # Reward for getting closer to goal
        if current_distance < self.best_distance:
            reward += 0.5  # Small reward for getting closer
            self.best_distance = current_distance
            self.steps_without_improvement = 0
        elif current_distance > self.best_distance:
            reward -= 0.2  # Small penalty for moving away
            self.steps_without_improvement += 1
        else:
            self.steps_without_improvement += 1
        
        # Additional penalty for staying in the same area too long
        if self.steps_without_improvement > 20:
            reward -= 0.5
            self.steps_without_improvement = 0
        
        # Remove all direction-based and reversal penalties
        self.last_action = action
        
        return reward
    
    def convert_action_to_movement(self, action):
        """Convert action number to player movement"""
        movements = {
            0: {'up_pressed': True, 'down_pressed': False, 'left_pressed': False, 'right_pressed': False},
            1: {'up_pressed': False, 'down_pressed': False, 'left_pressed': False, 'right_pressed': True},
            2: {'up_pressed': False, 'down_pressed': True, 'left_pressed': False, 'right_pressed': False},
            3: {'up_pressed': False, 'down_pressed': False, 'left_pressed': True, 'right_pressed': False}
        }
        return movements[action]

    def reset_episode(self):
        self.best_distance = float('inf')
        self.steps_without_improvement = 0
        self.last_action = None 