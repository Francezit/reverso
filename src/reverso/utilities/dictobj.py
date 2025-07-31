class ConfigurationBase:

    def __init__(self, in_dict: dict = None):
        if in_dict is not None:
            for key, val in in_dict.items():
                self.__setattr__(key, self._converter_deserialize(key, val))

    def _converter_deserialize(self, name: str, value: any):
        return value

    def _converter_serialize(self, name: str, value: any):
        return value

    def cloneConfig(self):
        return type(self)(self.to_dict())

    def to_dict(self):
        temp = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigurationBase):
                newv = v.to_dict()
            elif isinstance(v, list):
                newv = []
                for item in v:
                    if isinstance(item, ConfigurationBase):
                        newv.append(item.to_dict())
                    else:
                        newv.append(self._converter_serialize(k, item))
            else:
                newv = self._converter_serialize(k, v)
            temp[k] = newv
        return temp


class DictObj:

    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key,
                    [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key,
                        DictObj(val) if isinstance(val, dict) else val)
