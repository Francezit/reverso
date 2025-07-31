class HpVector:

    def __init__(self, list: list):
        self.__list = list

    def __getitem__(self, __name):
        return self.__list[__name]

    def __sizeof__(self) -> int:
        return len(self.__list)

    def __len__(self) -> int:
        return len(self.__list)

    def __str__(self) -> str:
        return str(self.__list)

    def __hash__(self) -> int:
        return str(self.__list).__hash__()

    def copy(self):
        return HpVector(self.__list.copy())

    def values(self):
        return self.__list.copy()
