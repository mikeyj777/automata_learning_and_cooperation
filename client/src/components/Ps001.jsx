/*

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

*/

import { React, useEffect, useState, useRef }  from 'react';
import ProblemLayout from './ui/ProblemLayout';

const API_BASE_URL = 'http://localhost:5000/api/'

const Ps001 = ( {isRunning, epoch, resetGrid, setResetGrid, probControlsDict}) => {

  const [grid, setGrid] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  
  const getGridDetails = async (initialize = false) => {
    
    const ep = initialize ? 'initialize_001' : 'step_001';
    const url = API_BASE_URL + ep;
    const { targetPattern, gridSize } = probControlsDict;
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          currentEpoch: 0,
          targetPattern: targetPattern,
          gridSize: gridSize,
          grid:grid,
        })
      });

      if (!response.ok) {
        throw new Error('Failed to initialize grid');
      }

      const data = await response.json();
      setGrid(data.grid);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const drawGrid = () => {
    if (!grid|| !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const gridSize = grid.length;
    const cellSize = canvas.width / gridSize;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw cells
    grid.forEach((row, i) => {
      row.forEach((cell, j) => {
        if (cell.is_active) {
          ctx.fillStyle = '#7fc7ff'; // light blue
          ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
        
        // Optional: draw grid lines for better visibility
        ctx.strokeStyle = '#e0e0e0';
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
      });
    });
  };

  useEffect(() => {
    drawGrid();
  }, [grid])

  useEffect(() => {
    if (isRunning && epoch > 0) {
      getGridDetails(false);
      drawGrid();
    }
  }, [isRunning, epoch])

  // render grid
  const fxnResetGrid = () => {
    getGridDetails(true);
    drawGrid();
  }

  // empty dependency array = run only when component mounts
  useEffect(() => {
    fxnResetGrid();
  }, [])

  // reset grid when reset is pressed
  useEffect(() => {
    if (resetGrid) {
      fxnResetGrid();
      setResetGrid(false);
    }
  }, [resetGrid]);

  useEffect(() => {
    fxnResetGrid();
  }, [probControlsDict])

  // initiate loop (send request to back end) when start is pressed

  

  return (
    <div>
      <canvas 
        ref={canvasRef} 
        width={700} 
        height={700}
        style={{ border: '1px solid #ccc' }}
      />
    </div>
  )

} 

export default Ps001;


