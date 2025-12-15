import numpy as np

from resources.pattern import get_game_of_life_pattern

class Grid:

  def __init__(self, grid_size = 100):
    self.grid_size = grid_size
    self.grid = np.empty((grid_size, grid_size))
  
  def initialize_game(self, target_pattern):
    self.grid = get_game_of_life_pattern(target_pattern=target_pattern, grid_size=self.grid_size)
    