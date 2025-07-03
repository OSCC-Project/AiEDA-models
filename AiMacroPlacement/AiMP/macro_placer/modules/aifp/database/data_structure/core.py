# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class PyCore:
    def __init__(self):
        self._low_x = 0.0
        self._low_y = 0.0
        self._width = 0.0
        self._height = 0.0
        self._name = "core"
    # def __init__(self, core:aifp_cpp.AifpCore=None):
    #     if core == None:
    #         self._low_x = 0.0
    #         self._low_y = 0.0
    #         self._width = 0.0
    #         self._height = 0.0
    #         self._name = "core"
    #     else:
    #         self._low_x = core.get_low_x()
    #         self._low_y = core.get_low_y()
    #         self._width = core.get_width()
    #         self._height = core.get_height()
    #         self._name = core.get_name()
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