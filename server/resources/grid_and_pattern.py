import numpy as np

def initialize_grid(target_pattern, current_epoch, grid_size=100, id=-1):
    gol_pattern = get_game_of_life_pattern(target_pattern=target_pattern, grid_size=grid_size)
    cell_grid = []
    for i in range(100):
        row = []
        for j in range(100):
            id += 1
            row.append({
                'id': id,
                'coords': [i, j],
                'is_active': bool(gol_pattern[i, j]),
                'birthday': current_epoch,
                'parent': -1,
                'children': [],
                'fitness_score': 0,
            })
        cell_grid.append(row)
    return cell_grid

def get_game_of_life_pattern(target_pattern, grid_size=100):
    """
    Returns a grid with the specified Game of Life pattern.
    
    Args:
        target_pattern (str): Name of the pattern to generate
        grid_size (int): Size of the square grid (default 100)
        
    Returns:
        numpy.ndarray: grid_size x grid_size grid with 0s (inactive) and 1s (active)
    """
    # Initialize empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    
    if target_pattern == "glider":
        pattern = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]
        offset_x, offset_y = 1, 1
        
    elif target_pattern == "gosper_glider_gun":
        pattern = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]
        offset_x, offset_y = 4, 18
        
    elif target_pattern == "pulsar":
        pattern = [
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0]
        ]
        offset_x, offset_y = 6, 6
        
    elif target_pattern == "lwss":
        pattern = [
            [0,1,0,0,1],
            [1,0,0,0,0],
            [1,0,0,0,1],
            [1,1,1,1,0]
        ]
        offset_x, offset_y = 2, 2
        
    elif target_pattern == "blinker":
        pattern = [
            [1,1,1]
        ]
        offset_x, offset_y = 0, 1
        
    elif target_pattern == "toad":
        pattern = [
            [0,1,1,1],
            [1,1,1,0]
        ]
        offset_x, offset_y = 1, 2
        
    elif target_pattern == "beacon":
        pattern = [
            [1,1,0,0],
            [1,1,0,0],
            [0,0,1,1],
            [0,0,1,1]
        ]
        offset_x, offset_y = 2, 2
        
    elif target_pattern == "pentadecathlon":
        pattern = [
            [1,1,1,1,1,1,1,1],
            [1,0,1,1,1,1,0,1],
            [1,1,1,1,1,1,1,1]
        ]
        offset_x, offset_y = 1, 4
        
    elif target_pattern == "r_pentomino":
        pattern = [
            [0,1,1],
            [1,1,0],
            [0,1,0]
        ]
        offset_x, offset_y = 1, 1
        
    elif target_pattern == "diehard":
        pattern = [
            [0,0,0,0,0,0,1,0],
            [1,1,0,0,0,0,0,0],
            [0,1,0,0,0,1,1,1]
        ]
        offset_x, offset_y = 1, 4
        
    elif target_pattern == "acorn":
        pattern = [
            [0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0],
            [1,1,0,0,1,1,1]
        ]
        offset_x, offset_y = 1, 3
        
    else:
        return grid
    
    # Place pattern in center of grid
    pattern = np.array(pattern)
    rows, cols = pattern.shape
    start_row = center - offset_x
    start_col = center - offset_y
    
    grid[start_row:start_row + rows, start_col:start_col + cols] = pattern
    
    return grid