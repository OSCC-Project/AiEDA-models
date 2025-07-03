from aifp import setting
from aifp.utility.operators import scale


class PlacementReward:
    def __init__(self):
        pass

    def minus_wirelength(score_dict, reward_scale):
        evaluator = setting.env_train['evaluator']
        if 'wirelength' in score_dict:
            scaled_wirelength = scale(score_dict['wirelength'], reward_scale)
            return - scaled_wirelength
        else:
            raise ValueError