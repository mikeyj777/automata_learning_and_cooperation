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

from flask import jsonify

from classes.cell import Cell
from classes.grid import Grid
from resources.grid_and_pattern import initialize_grid

def initialize(data):
  current_epoch = data['currentEpoch']
  target_pattern = data['targetPattern']
  grid_size = data['gridSize']
  id = data['lastId']
  if id is None:
      id = -1
  grid = initialize_grid(target_pattern=target_pattern, current_epoch=current_epoch, grid_size=grid_size, id=id)
  return jsonify({'grid': grid}, 200)

def step(data):
  pass
  