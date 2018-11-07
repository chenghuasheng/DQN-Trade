import sys


class Const(object):
    class const_error(TypeError): pass
    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise(self.const_error, "Changing const.%s" % key)
            pass
        else:
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.key
        else:
            return None

sys.modules[__name__] = Const()