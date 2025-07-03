class MacroInfo:
    def __init__(
            self,
            name:str='unknown_macro',
            low_x: float = 0,
            low_y: float = 0,
            orient: str = ''):

        self._name = name
        self._low_x = low_x
        self._low_y = low_y
        self._orient = orient

    def set_name(self, name:str):
        self._name = name
    def set_low_x(self, low_x:float):
        self._low_x = low_x
    def set_low_y(self, low_y:float):
        self._low_y = low_y
    def set_orient(self, orient:str):
        self._orient = orient

    def get_name(self):
        return self._name
    def get_low_x(self):
        return self._low_x
    def get_low_y(self):
        return self._low_y
    def get_orient(self):
        return self._orient

    def __str__(self):
        return 'name:{} low_x:{} low_y:{} orient:{}'.format(self._name, self._low_x, self._low_y, self._orient)

    def to_dict(self):
        macro_dict = {
            'name':self._name,
            'low_x':float(self._low_x),
            'low_y':float(self._low_y),
            'orient':self._orient
        }
        return macro_dict

class FPSolution:
    def __init__(self):
        self._design_name = 'unknown_design'
        self._description = ''
        self._valid_flag = False # if fp_solution is valid
        self._score_dict = None
        self._macro_info_dict = dict()

    def set_valid_flag(self, valid_flag:bool):
        self.valid_flag = valid_flag

    def set_design_name(self, design_name:str):
        self._design_name = design_name

    def set_score_dict(self, score_dict:dict):
        self._score_dict = score_dict

    def set_description(self, description:str):
        self._description = description


    def update_macro_info(self, macro_name:str, low_x:float=None, low_y:float=None, orient:str=None):
        if low_x != None:
            self._macro_info_dict[macro_name].set_low_x(low_x)
        if low_y != None:
            self._macro_info_dict[macro_name].set_low_y(low_y)
        if orient != None:
            self._macro_info_dict[macro_name].set_orient(orient)

    def add_macro_info(self, macro_info:MacroInfo):
        self._macro_info_dict[macro_info.get_name()] = macro_info

    def get_design_name(self):
        return self._design_name
    
    def get_description(self):
        return self._description

    def get_score_dict(self):
        return self._score_dict

    def get_valid_flag(self):
        return self._valid_flag

    def get_macro_info(self, macro_name:str):
        if macro_name in self._macro_info_dict:
            return self._macro_info_dict[macro_name]
        else:
            return None

    def get_macro_info_dict(self):
        return self._macro_info_dict

    def __str__(self):
        solution_str = []
        solution_str.append('design_name:{}'.format(self._design_name))
        if self._score_dict != None:
            for score_name, score_value in self._score_dict.items():
                solution_str.append('{}:{}'.format(score_name, score_value))
        for macro_name, macro_info in self._macro_info_dict.items():
            solution_str.append(macro_info.__str__())
        return '\n'.join(solution_str)

    def to_dict(self):
        score_dict = {}
        for k, v in self._score_dict.items():
            score_dict[k] = float(v)
        solution_dict = {
            'design_name':self._design_name,
            'description':self._description,
            'valid_flag':self._valid_flag,
            'score_dict':score_dict,
            'macro_info':[macro_info.to_dict() for macro_name, macro_info in self._macro_info_dict.items()]
        }
        return solution_dict

    def from_dict(self, solution_dict):
        self._design_name = solution_dict['design_name']
        self._description = solution_dict['description']
        self._valid_flag = solution_dict['valid_flag']
        self._score_dict = solution_dict['score_dict']
        self._macro_info_dict = {}
        for macro_info in solution_dict['macro_info']:
            self.add_macro_info(
                MacroInfo(
                name = macro_info['name'],
                low_x = float(macro_info['low_x']),
                low_y = float(macro_info['low_y']),
                orient = macro_info['orient']
                )
            )