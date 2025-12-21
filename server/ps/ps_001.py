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
import copy
import numpy as np
from flask import jsonify, request

from resources.grid_and_pattern import initialize_grid

def initialize(data):
  current_epoch = data['currentEpoch']
  target_pattern = data['targetPattern']
  grid_size = data['gridSize']
  grid = initialize_grid(target_pattern=target_pattern, current_epoch=current_epoch, grid_size=grid_size)
  return jsonify({'grid': grid}), 200

def safe_slice(grid_np, i, j):
  start_i = max(i - 1, 0)
  end_i = min(i + 2, grid_np.shape[0])
  start_j = max(j - 1, 0)
  end_j = min(j + 2, grid_np.shape[1])
  
  return grid_np[start_i:end_i, start_j:end_j]

def should_cell_be_active(slice, origin_cell_is_active):
  gol_score = sum(d['is_active'] for d in slice.flat)
  
  if origin_cell_is_active:
    gol_score -=1
  
  activate = False
  if origin_cell_is_active:
    if gol_score == 2 or gol_score == 3:
      activate = True
  else:
    if gol_score == 3:
      activate = True
  
  return activate

def increase_score_for_precedents(slice):
  for i in range(slice.shape[0]):
    for j in range(slice.shape[1]):
      if i == 1 and j == 1:
        continue
      if slice[i, j]['is_active']:
        slice[i, j]['fitness_score'] += 1
  

def update_grid(grid):
  grid_np = np.array(grid, dtype=dict)
  grid_for_update = copy.deepcopy(grid_np)
  for i in range(grid_np.shape[0]):
    for j in range(grid_np.shape[1]):
      slice = safe_slice(grid_np, i, j)
      activate = should_cell_be_active(slice=slice, origin_cell_is_active=grid_np[i,j]['is_active'])
      grid_for_update[i, j]['is_active'] = activate
      if not activate:
        grid_for_update[i, j]['fitness_score'] = 0
        continue
      slice_new = safe_slice(grid_for_update, i, j)
      increase_score_for_precedents(slice_new)
      
  return grid_for_update.tolist()

def step(data):
  grid = data['grid']
  grid = update_grid(grid)
  return jsonify({'grid': grid}), 200
