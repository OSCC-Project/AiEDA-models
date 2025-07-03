import numpy as np
import math
from aifp import setting

class HeuristicReward:
    def __init__(self):
        pass

    def dist_to_edge(macro, core, macro_nums):
        ratio = 1.0 / macro_nums

        macro_left = macro.low_x
        macro_right = macro_left + macro.width
        macro_bottom = macro.low_y
        macro_top = macro_bottom + macro.height

        core_left = core.low_x
        core_right = core_left + core.width
        core_bottom = core.low_y
        core_top = core_bottom + core.height

        dists = [macro_left - core_left,
                 core_right - macro_right,
                 macro_bottom - core_bottom,
                 core_top - macro_top]

        dist_to_edge = min(dists)
        assert dist_to_edge >= 0
        dist_to_edge /= (max(core.height, core.width))  # normalize
        dist_to_edge *= ratio
    
    def dist_to_origin_macro(placed_macro, origin_macro, episode_length):
        ratio = 1.0 / episode_length
        dist = ((placed_macro.get_low_x() - origin_macro.get_low_x())**2 + (placed_macro.get_low_y() - origin_macro.get_low_y())) ** 0.5 / origin_macro.get_width()
        return - setting.env_train['reward_scale'] * ratio * dist
