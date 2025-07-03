class BoxSpace():
    def __init__(self, low: int, high: int, shape: tuple):
        self._low = low
        self._high = high
        self._shape = shape
    
    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def shape(self):
        return self._shape
    
    def __str__(self):
        boxspace_str = "box_space.BoxSpace(low={}, high={}, shape={})".format(self._low, self._high, self._shape.__str__())
        return boxspace_str