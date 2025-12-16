'''
### 1. Conway's Life with Survival Metrics

Implement Conway's Game of Life with a survival tracking system. Each cell has a "fitness score" based on how many generations it and its descendants survive. Create a Flask endpoint that returns grid states and a React visualization showing both the live grid and a heatmap of cumulative survival scores.

**Backend Requirements:**
- NumPy grid representation (0 = dead, 1 = alive)
- Track cell lineage through generations
- Survival score accumulation per cell position
- Flask endpoint: `/api/step` (POST) returns current grid state and survival scores

**Frontend Requirements:**
- Canvas-based grid rendering (black/white cells)
- Overlay heatmap for survival scores (color gradient)
- Play/pause/reset controls
- Generation counter display

**Success Criteria:** Identify which initial patterns (gliders, blinkers, still lifes) accumulate the highest survival scores over 100 generations.

---

'''
import numpy as np
from flask import jsonify, request

from resources.grid_and_pattern import initialize_grid

def initialize(data):
  current_epoch = data['currentEpoch']
  target_pattern = data['targetPattern']
  grid_size = data['gridSize']
  id = data['lastId']
  if id is None:
      id = -1
  grid = initialize_grid(target_pattern=target_pattern, current_epoch=current_epoch, grid_size=grid_size, id=id)
  return jsonify({'grid': grid}), 200

def step(data):
  pass


def safe_slice(grid_np, i, j):
  start_i = max(i - 1, 0)
  end_i = min(i + 1, grid_np.shape[0])
  start_j = max(j-1, 0)
  end_j = min(j+1, grid_np.shape[1])
  
  return grid_np[start_i:end_i, start_j:end_j]

def count_actives_and_max_fitness_around_slice(slice, origin_cell_is_active):
  gol_score = sum(d['is_active'] for d in slice.flat)
  max_fitness = max(d['fitness_score'] for d in slice.ravel())
  
  if origin_cell_is_active:
    gol_score -=1
  
  return {'gol_score': gol_score, 'max_fitness': max_fitness}

def cell_should_be_active_and_max_fitness(grid_np, i, j):
  slice = safe_slice(grid_np, i, j)
  resp = count_actives_and_max_fitness_around_slice(slice=slice, origin_cell_is_active=slice[i,j]['is_active'])
  gol_score = resp['gol_score']
  max_fitness = resp['max_fitness']
  return {'activate': gol_score == 2 or gol_score == 3, 'max_fitness': max_fitness}

def update_grid(grid):
  grid_np = np.array(grid, dtype=dict)
  for i in range(grid_np.shape[0]):
    for j in range(grid_np.shape[1]):
      resp = cell_should_be_active_and_max_fitness(grid_np=grid_np, i=i, j=j)
      grid_np[i, j]['is_active'] = resp['activate']
      max_fitness = resp['max_fitness']
      if grid_np[i, j]['is_active']:
        grid[i, j]['fitness_score'] += max_fitness
      else:
        grid[i, j]['fitness_score'] = 0
  