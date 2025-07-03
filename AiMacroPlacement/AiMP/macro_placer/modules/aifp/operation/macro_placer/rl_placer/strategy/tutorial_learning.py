import numpy as np

def get_valid_margin(grid_nums, episode):
    # if episode <= 500:
    #     valid_margin = 0.22
    # elif episode <= 1000:
    #     valid_margin = 0.25
    # else:
    valid_margin = 0.22

    # valid_grids = np.ones((grid_nums, grid_nums), dtype=np.float32)
    obstacle_start_grid = int(grid_nums * valid_margin)
    obstacle_end_grid = grid_nums - obstacle_start_grid - 1

    # for i in range(obstacle_start_grid, obstacle_end_grid + 1):
    #     for j in range(obstacle_start_grid, obstacle_end_grid + 1):
    #         valid_grids[i][j] = 0.0
    # print('grid_nums: ', grid_nums)
    # print('obstacle_start_grid', obstacle_start_grid)
    # print('obstacle_end_grid: ', obstacle_end_grid)
    # valid_grids[obstacle_start_grid: obstacle_end_grid+1, obstacle_start_grid: obstacle_end_grid+1] = 0
    return obstacle_start_grid, obstacle_end_grid