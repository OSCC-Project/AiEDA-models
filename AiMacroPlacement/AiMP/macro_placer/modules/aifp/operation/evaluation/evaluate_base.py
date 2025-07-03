from abc import abstractmethod

class EvaluateBase:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, macro_list) -> dict:
        pass
