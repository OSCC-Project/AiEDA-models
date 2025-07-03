# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class PyInstance:
    def __init__(self):
            self._type = "macro"
            self._status = "unfixed"
            self._name   = ""
            self._orient = ""
            self._origin_orient = self._orient # origin orient from def/lef, not changed, cause pin_offset is not changed..
            self._pin_list = []

            self._index  = -1
            self._grid_x = -1
            self._grid_y = -1
            self._degree = 0

            self._low_x  = 0.0
            self._low_y  = 0.0
            self._width  = 0.0
            self._height = 0.0

            self._halo_extend_left   = 0.0
            self._halo_extend_right  = 0.0
            self._halo_extend_top    = 0.0
            self._halo_extend_bottom = 0.0
    # def __init__(self, inst:aifp_cpp.AifpInstance=None):
    #     if inst == None:
    #         self._type = aifp_cpp.InstanceType.stdcell
    #         self._status = "unfixed"
    #         self._name   = ""
    #         self._orient = ""
    #         self._origin_orient = self._orient # origin orient from def/lef, not changed, cause pin_offset is not changed..
    #         self._pin_list = []

    #         self._index  = -1
    #         self._grid_x = -1
    #         self._grid_y = -1
    #         self._degree = 0

    #         self._low_x  = 0.0
    #         self._low_y  = 0.0
    #         self._width  = 0.0
    #         self._height = 0.0

    #         self._halo_extend_left   = 0.0
    #         self._halo_extend_right  = 0.0
    #         self._halo_extend_top    = 0.0
    #         self._halo_extend_bottom = 0.0
    #     else:
    #         self._type = inst.get_type()
    #         self._status = inst.get_status()
    #         self._name   = inst.get_name()
    #         self._orient = inst.get_orient()
    #         self._origin_orient = self._orient
    #         self._pin_list = inst.get_pin_list()

    #         self._index  = inst.get_index()
    #         self._grid_x = inst.get_grid_x()
    #         self._grid_y = inst.get_grid_y()
    #         self._degree = inst.get_degree()

    #         self._low_x  = inst.get_low_x()
    #         self._low_y  = inst.get_low_y()
    #         self._width  = inst.get_width()
    #         self._height = inst.get_height()

    #         self._halo_extend_left   = inst.get_halo_extend_left()
    #         self._halo_extend_right  = inst.get_halo_extend_right()
    #         self._halo_extend_top    = inst.get_halo_extend_top()
    #         self._halo_extend_bottom = inst.get_halo_extend_bottom()

    # def set_type(self, type:aifp_cpp.InstanceType):
    #     self._type = type
    # def set_status(self, status:aifp_cpp.InstanceStatus):
    #     self._status = status
    def set_type(self, type:str):
        self._type = type
    def set_status(self, status:str):
        self._status = status
    def set_name(self, name:str):
        self._name = name
    def set_origin_orient(self, origin_orient):
        self._origin_orient = origin_orient
    def set_orient(self, orient:str):
        self._orient = orient
    def set_index(self, index:int):
        self._index = index
    def set_grid_x(self, grid_x:int):
        self._grid_x = grid_x
    def set_grid_y(self, grid_y:int):
        self._grid_y = grid_y
    def set_degree(self, degree:int):
        self._degree = degree
    def set_low_x(self, low_x:float):
        self._low_x = low_x
    def set_low_y(self, low_y:float):
        self._low_y = low_y
    def set_width(self, width:float):
        self._width = width
    def set_height(self, height:float):
        self._height = height
    def set_halo_extend_left(self, halo_extend_left:float):
        self._halo_extend_left = halo_extend_left
    def set_halo_extend_right(self, halo_extend_right:float):
        self._halo_extend_right = halo_extend_right
    def set_halo_extend_top(self, halo_extend_top:float):
        self._halo_extend_top = halo_extend_top
    def set_halo_extend_bottom(self, halo_extend_bottom:float):
        self._halo_extend_bottom = halo_extend_bottom


    def get_type(self)->str:
        return self._type
    def get_status(self)->str:
        return self._status

    # def get_type(self)->aifp_cpp.InstanceType:
    #     return self._type
    # def get_status(self)->aifp_cpp.InstanceStatus:
        return self._status
    def get_name(self)->str:
        return self._name
    def get_orient(self)->str:
        return self._orient
    def get_origin_orient(self)->str:
        return self._origin_orient
    def get_pin_list(self):
        return self._pin_list
    def get_index(self)->int:
        return self._index
    def get_grid_x(self)->int:
        return self._grid_x
    def get_grid_y(self)->int:
        return self._grid_y
    def get_degree(self)->int:
        return self._degree
    def get_low_x(self)->float:
        return self._low_x
    def get_low_y(self)->float:
        return self._low_y
    def get_width(self)->float:
        return self._width
    def get_height(self)->float:
        return self._height
    def get_center_x(self)->float:
        return self._low_x + self._width / 2
    def get_center_y(self)->float:
        return self._low_y + self._height / 2

    def get_halo_extend_left(self)->float:
        return self._halo_extend_left
    def get_halo_extend_right(self)->float:
        return self._halo_extend_right
    def get_halo_extend_top(self)->float:
        return self._halo_extend_top
    def get_halo_extend_bottom(self)->float:
        return self._halo_extend_bottom

    def get_halo_low_x(self)->float:
        return self._low_x - self._halo_extend_left
    def get_halo_low_y(self)->float:
        return self._low_y - self._halo_extend_bottom
    def get_halo_width(self)->float:
        return self._width + self._halo_extend_left + self._halo_extend_right
    def get_halo_height(self)->float:
        return self._height + self._halo_extend_bottom + self._halo_extend_top
    def get_halo_center_x(self)->float:
        return self.get_halo_low_x() + self.get_halo_width() / 2
    def get_halo_center_y(self)->float:
        return self.get_halo_low_y() + self.get_halo_height() / 2