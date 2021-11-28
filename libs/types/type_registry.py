"""
Metaclass that registers member types.
"""


class Registry(type):
    __member_registry__: dict[str, type] = {}

    def __new__(cls, name, bases, dct, **to_register):
        """
        Registers types under their names or name specified as kwarg.
        """
        x = type.__new__(cls, name, bases, dct)
        if not all(isinstance(x, type) for x in to_register.values()):
            raise ValueError("can register only types")
        cls.__member_registry__.update(to_register)
        return x

    @classmethod
    def get_registry(cls) -> dict[str, type]:
        return cls.__member_registry__
