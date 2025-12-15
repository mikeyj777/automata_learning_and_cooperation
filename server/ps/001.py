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


from classes.cell import Cell
from classes.grid import Grid

def initialize(data):
  current_epoch = data['currentEpoch']
  target_formation = data['targetFormation']
  grid = Grid(grid_size=100)
  grid.initialize_game(target_pattern=target_formation)
  id = data['id']
  if id is None:
      id = -1
  cells = []
  for i in num_cells:
    id += 1
    cell = Cell(id = 1, current_epoch=current_epoch)