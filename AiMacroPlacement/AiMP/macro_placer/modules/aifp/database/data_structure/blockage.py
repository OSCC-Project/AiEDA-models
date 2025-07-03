# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class PyBlockage:
    # def __init__(self, blockage:aifp_cpp.AifpBlockage=None):
    #     if blockage == None:
    #         self._low_x = 0.0
    #         self._low_y = 0.0
    #         self._width = 0.0
    #         self._height = 0.0
    #         self._name = "blockage"
    #     else:
    #         self._low_x = blockage.get_low_x()
    #         self._low_y = blockage.get_low_y()
    #         self._width = blockage.get_width()
    #         self._height = blockage.get_height()
    #         self._name = blockage.get_name()
    def __init__(self):
        self._low_x = 0.0
        self._low_y = 0.0
        self._width = 0.0
        self._height = 0.0
        self._name = "blockage"

    def set_low_x(self, low_x:float):
        self._low_x = low_x
    def set_low_y(self, low_y:float):
        self._low_y = low_y
    def set_width(self, width:float):
        self._width = width
    def set_height(self, height:float):
        self._height = height
    def set_name(self, name:str):
        self._name = name

    def get_low_x(self)->float:
        return self._low_x
    def get_low_y(self)->float:
        return self._low_y
    def get_width(self)->float:
        return self._width
    def get_height(self)->float:
        return self._height
    def get_name(self)->str:
        return self._name
    def get_center_x(self)->float:
        return self._low_x + self._width / 2
    def get_center_y(self)->float:
        return self._low_y + self._height / 2
    def get_halo_width(self)->float:
        return self.get_width()
    def get_halo_height(self)->float:
        return self.get_height()