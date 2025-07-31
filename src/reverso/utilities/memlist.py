from typing import IO
import numpy as np
from collections import deque
import os
import struct
import gc
from pympler import asizeof


class MemList():
    _fp: IO
    _size: int
    _fpos: int

    _flag_dim_size: int = 4
    _flag_dim_ord: str = "big"
    _flag_remove_file: bool = True

    _appendCallbackFun = None
    _editCallbackFun = None

    def __init__(self,
                 memFilename: str,
                 serializeFun=None,
                 deserializeFun=None,
                 appendCallbackFun=None,
                 editCallbackFun=None,
                 mode: str = 'w') -> None:

        #mode: 'w' the file is a temp file and it will delete when this object will be dispose
        #      'u' if exists the temp file it will be open and loaded and it will NOT be deleted when this object will be dispose
        #      'r' if exists the temp file it will be open and loaded and it will delete when this object will be dispose

        self._appendCallbackFun = appendCallbackFun
        self._editCallbackFun = editCallbackFun
        if serializeFun is not None:
            self._serialize = serializeFun
        if deserializeFun is not None:
            self._deserialize = deserializeFun

        self._flag_remove_file = mode.find('w') >= 0 or mode.find('r') >= 0

        truncate_mode = mode.find('w') >= 0
        self._fp = open(memFilename, mode='w+b' if truncate_mode else 'r+b')

        loaded_mode = mode.find('u') >= 0 or mode.find('r')
        if loaded_mode:
            self._size = self._readItemSize()
        else:
            self._size = 0
        self._fpos = self._fp.tell()

    @property
    def size(self):
        return self._size

    @property
    def byteMemSize(self):
        return asizeof.asizeof(self)

    @property
    def byteStoreSize(self):
        return self._fpos

    def _serialize(self, obj) -> list:
        b = bytearray(obj, encoding="utf8")
        return list(b)

    def _deserialize(self, bdata: list):
        b = bytearray(bdata)
        return b.decode(encoding="utf8")

    def clear(self):
        self._fp.truncate(0)
        self._fpos = self._fp.tell()
        self._size = 0
        self._fp.flush()

    def replace(self, objs: list):
        self._fp.truncate(0)
        self._size = 0
        self.appends(objs)

    def set(self, index: int, value):
        r = self.sets(indexs=[index], values=[value])
        return r[0]

    def sets(self, indexs: list, values: list, condictionFn=None):
        baseList = self.toList()
        resultList = []
        toBeEdited = []
        for i, index in enumerate(indexs):
            newValue = values[i]
            oldValue = baseList[index]
            if condictionFn is None or condictionFn(index, oldValue, newValue):
                baseList[index] = newValue
                toBeEdited.append((index, newValue, oldValue))
            resultList.append(oldValue)
        self.replace(baseList)
        del baseList

        if len(toBeEdited) > 0 and self._editCallbackFun is not None:
            for data in toBeEdited:
                self._editCallbackFun(*data)
        del toBeEdited
        gc.collect()

        return resultList

    def append(self, obj):
        line = self._serialize(obj)
        linesize = int(len(line))

        self._fp.write(
            linesize.to_bytes(self._flag_dim_size, self._flag_dim_ord))
        self._fp.write(line)
        self._fpos = self._fp.tell()
        self._size = self._size + 1
        self._fp.flush()

    def appends(self, objs: list):
        for obj in objs:
            line = self._serialize(obj)
            linesize = int(len(line))

            self._fp.write(
                linesize.to_bytes(self._flag_dim_size, self._flag_dim_ord))
            self._fp.write(line)

        self._fpos = self._fp.tell()
        self._size = self._size + len(objs)
        self._fp.flush()

    def at(self, index: int):
        return self.gets([index])[0]

    def gets(self, indexs: list):
        for i in range(len(indexs)):
            if indexs[i] < 0:
                indexs[i] = self._size + indexs[i]

        indexs.sort()
        maxIdx = max(indexs)
        minIndx = min(indexs)
        assert minIndx >= 0 and maxIdx < self._size and minIndx <= maxIdx

        self._fp.seek(0, 0)
        lines = []
        i = 0
        while i < self._size:
            bLineSize = self._fp.read(self._flag_dim_size)
            lineSize = int.from_bytes(bLineSize, self._flag_dim_ord)
            bLine = self._fp.read(lineSize)

            if i in indexs:
                line = self._deserialize(bLine)
                lines.append(line)
                del bLineSize, bLine, lineSize
            elif i > maxIdx:
                del bLineSize, bLine, lineSize
                break

            i = i + 1

        self._fp.seek(self._fpos)
        gc.collect()
        return lines

    def _readItemSize(self):
        fp = self._fp
        bsize = os.fstat(fp.fileno()).st_size
        size = 0

        fp.seek(0, 0)
        while fp.tell() < bsize:
            buff = self._fp.read(self._flag_dim_size)
            n = int.from_bytes(buff, self._flag_dim_ord)
            self._fp.seek(n, 1)
            size = size + 1
            del buff
        return size

    def slice(self, startIdx: int = None, endIdx: int = None):
        if startIdx is None:
            startIdx = 0
        if endIdx is None:
            endIdx = self._size
        idxs = list(range(startIdx, endIdx))
        return self.gets(idxs)

    def toList(self):
        idxs = list(range(0, self._size))
        return self.gets(idxs)

    def flush(self):
        self._fp.flush()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, __name):
        return self.at(__name)

    def __setitem__(self, __name, __value):
        self.set(__name, __value)

    def __del__(self):
        memFile = self._fp.name
        self._fp.flush()
        self._fp.close()
        del self._fp
        if self._flag_remove_file:
            os.remove(memFile)
        gc.collect()


class NumpyArrayWrapper():
    _data = None
    _numberComponent: int
    _useMem: bool
    _capacityCacheLastItem: int

    def __init__(self,
                 numberComponents: int,
                 memFilename: str = None,
                 dType: str = "float64",
                 capacityCacheLastItem: int = None) -> None:

        self._numberComponent = numberComponents
        self._capacityCacheLastItem = capacityCacheLastItem

        if memFilename is None:
            self._data = []
            self._useMem = True
            self.appends = lambda x: [
                self._data.append(v) for v in self._checkFormat(x)
            ]
            self.append = lambda x: self._data.append(self._checkFormat(x))
            self.toList = lambda: self._data.copy()
            self.replace = self._replaceList
            self.slice = self._sliceList
            self.tail = lambda: self._checkFormat(self._data.at(-1))
            if capacityCacheLastItem is not None:
                self.getLastItems = self._getLastItemList
        else:

            def serV(nparray: np.ndarray):
                if nparray.dtype.name != dType:
                    nparray = nparray.astype(dType)
                return nparray.tobytes()

            def deserV(byte_array):
                return np.frombuffer(byte_array, dtype=dType)

            self._data = MemList(memFilename=memFilename,
                                 serializeFun=serV,
                                 deserializeFun=deserV)
            self._useMem = False
            self.appends = self._appendsMem
            self.append = self._appendMem
            self.toList = lambda: self._data.toList()
            self.replace = self._replaceMem
            self.slice = lambda sI, eI: self._data.slice(sI, eI)
            self.tail = self._tailMem
            if capacityCacheLastItem is not None:
                self._cacheMem = deque([], maxlen=capacityCacheLastItem)
                self.getLastItems = self._getLastItemMem

    @property
    def byteMemSize(self):
        return asizeof.asizeof(self)

    @property
    def byteStoreSize(self):
        if self._useMem:
            return 0
        else:
            return self._data.byteStoreSize

    @property
    def shape(self):
        return (len(self._data), self._numberComponent)

    @property
    def isInMemory(self):
        return self._useMem

    @property
    def isCacheEnabled(self):
        return self._capacityCacheLastItem is not None

    def __len__(self):
        return len(self._data)

    def __del__(self):
        del self._data
        if self._capacityCacheLastItem is not None:
            del self._cacheMem
        gc.collect()

    def replace(self, objs: list):
        pass

    def appends(self, v: list):
        pass

    def append(self, v: any):
        pass

    def toList(self) -> list:
        pass

    def slice(self, startIndex: int = None, endIndex: int = None) -> list:
        pass

    def getLastItems(self, n: int = None) -> list:
        return None

    def head(self) -> np.ndarray:
        return self._checkFormat(self._data.at(0))

    def tail(self) -> np.ndarray:
        pass

    def clear(self):
        self._data.clear()

    def _getLastItemList(self, n: int = None) -> list:
        v = n if n is not None else self._capacityCacheLastItem
        assert v <= self._capacityCacheLastItem
        return self._data[-v:]

    def _replaceList(self, objs: list):
        objs = self._checkFormat(objs)
        del self._data
        self._data = objs

    def _sliceList(self, startIndex: int = None, endIndex: int = None):
        l: list = self._data
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(l)
        return l[startIndex:endIndex]

    def _tailMem(self):
        if self._capacityCacheLastItem is not None:
            return self._checkFormat(self._cacheMem[-1])
        else:
            return self._checkFormat(self._data.at(-1))

    def _appendsMem(self, x: list):
        x = self._checkFormat(x)
        self._data.appends(x)
        if self._capacityCacheLastItem is not None:
            for v in x:
                self._cacheMem.append(v)

    def _appendMem(self, x):
        x = self._checkFormat(x)
        self._data.append(x)
        if self._capacityCacheLastItem is not None:
            self._cacheMem.append(x)

    def _replaceMem(self, x: list):
        x = self._checkFormat(x)
        self._data.replace(x)
        if self._capacityCacheLastItem is not None:
            self._cacheMem.clear()
            for v in x:
                self._cacheMem.append(v)

    def _getLastItemMem(self, n: int = None):
        v = n if n is not None else self._capacityCacheLastItem
        assert v <= self._capacityCacheLastItem
        l = list(self._cacheMem)
        return l[-v:]

    def _checkFormat(self, input):
        if isinstance(input, list):
            return [np.reshape(x, (1, self._numberComponent)) for x in input]
        elif isinstance(input, np.ndarray):
            if input.size / self._numberComponent == 1:
                return np.reshape(input, (1, self._numberComponent))
            else:
                return [
                    np.reshape(input[i, :], (1, self._numberComponent))
                    for i in range(input.shape[0])
                ]
        else:
            raise Exception("Check not supported")


if __name__ == "__main__":
    import random as rdn

    def serInput(floats: list):
        return b''.join(struct.pack('f', f) for f in floats)

    def desInput(byte_array: list):
        return [
            struct.unpack('f', byte_array[i:i + 4])[0]
            for i in range(0, len(byte_array), 4)
        ]

    filename = "./ciao.txt"
    memlist = MemList(filename, serializeFun=serInput, deserializeFun=desInput)

    for i in range(1000):
        l = [rdn.random() for _ in range(100)]
        memlist.append(l)

    v = memlist.toList()

    l = [rdn.random() for _ in range(100)]
    memlist.append(l)

    v1 = memlist.gets([1, 2, 3, 10])
    del memlist

    mm = NumpyArrayWrapper(100, filename, capacityCacheLastItem=10)
    for i in range(50):
        l = np.array([rdn.random() for _ in range(100)])
        mm.append(l)
        v = mm.getLastItems()
        v1 = mm.getLastItems(5)

    s = mm.slice(1, 10)

    del mm

    pass
