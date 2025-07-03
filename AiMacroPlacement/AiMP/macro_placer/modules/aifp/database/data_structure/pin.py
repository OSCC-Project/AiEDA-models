from re import L
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class PyPin:
    def __init__(self):
            # self._low_x = 0.0
            # self._low_y = 0.0
            # self._width = 0.0
            # self._height = 0.0
            self._name = "a-pin"
            self._pin_index = -1
            self._node_index = -1
            self._offset_x = 0
            self._offset_y = 0
    # def __init__(self, pin:aifp_cpp.AifpPin=None):
    #     if  pin == None:
    #         # self._low_x = 0.0
    #         # self._low_y = 0.0
    #         # self._width = 0.0
    #         # self._height = 0.0
    #         self._name = "blockage"
    #         self._pin_index = -1
    #         self._node_index = -1
    #         self._offset_x = 0
    #         self._offset_y = 0

    #     else:
    #         # self._low_x = pin.get_low_x()
    #         # self._low_y = pin.get_low_y()
    #         # self._width = pin.get_width()
    #         # self._height = pin.get_height()
    #         self._name = pin.get_name()
    #         self._pin_index = pin.get_pin_index()
    #         self._node_index = pin.get_node_index()
    #         self._offset_x = pin.get_offset_x()
    #         self._offset_y = pin.get_offset_y()
    
    # def set_low_x(self, low_x:float):
    #     self._low_x = low_x
    # def set_low_y(self, low_y:float):
    #     self._low_y = low_y
    # def set_width(self, width:float):
    #     self._width = width
    # def set_height(self, height:float):
        # self._height = height
    def set_name(self, name:str):
        self._name = name
    def set_pin_index(self, pin_index:int):
        self._pin_index = pin_index
    def set_node_index(self, node_index:int):
        self._node_index = node_index
    def set_offset_x(self, offset_x:float):
        self._offset_x = offset_x
    def set_offset_y(self, offset_y:float):
        self._offset_y = offset_y

    # def get_low_x(self)->float:
    #     return self._low_x
    # def get_low_y(self)->float:
    #     return self._low_y
    # def get_width(self)->float:
    #     return self._width
    # def get_height(self)->float:
        # return self._height
    def get_name(self)->str:
        return self._name
    def get_pin_index(self)->int:
        return self._pin_index
    def get_node_index(self)->int:
        return self._node_index
    def get_offset_x(self)->float:
        return self._offset_x
    def get_offset_y(self)->float:
        return self._offset_y

    # def get_center_x(self)->float:
    #     return self._low_x + self._width / 2
    # def get_center_y(self)->float:
    #     return self._low_y + self._height / 2
    # def get_halo_width(self)->float:
    #     return self.get_width()
    # def get_halo_height(self)->float:
    #     return self.get_height()