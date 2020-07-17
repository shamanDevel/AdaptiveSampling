import collections
import copy

class FixedDict(collections.MutableMapping):
    """
    A dictionary with immutable keys 
    (specified in the constructor)
    and mutable values.

    Source: https://stackoverflow.com/a/14816620/1786598
    """

    def __init__(self, data):
        self.__data = copy.deepcopy(data)

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError(k)

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def get_dict(self):
        """Returns a copy of this dictionary as a normal python dict"""
        return copy.deepcopy(self.__data)