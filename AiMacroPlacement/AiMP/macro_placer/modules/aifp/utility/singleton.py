def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance



if __name__ == '__main__':

    @singleton
    class MyClass:
        def __init__(self, name):
            print('my class initilizing')
            self._name = name

    obj1 = MyClass('a')
    obj2 = MyClass('b')
    print(obj1._name)
    print(obj2._name)

