from logging import Logger, INFO
import os, multiprocessing
import json, gc
import psutil
from pympler import asizeof
from cachetools import LRUCache

from .agent import Agent, AgentConfiguration
from .agentbuilder import AgentBuilder


class LRUCache2(LRUCache):
    _useCallback: bool
    _evict: any

    def __init__(self, maxsize, getsizeof=None, evict=None):
        super().__init__(maxsize, getsizeof)
        self._evict = evict
        self._useCallback = evict is not None

    @property
    def useCallback(self):
        return self._useCallback

    def pop(self, key):
        value = super().pop(key)
        if self._useCallback:
            self._evict(key, value)
        return value

    def callbackStatus(self, enable: bool):
        self._useCallback = enable and self._evict is not None


class AgentCache():
    _foldername: str
    _storageCacheCapacityBytes: int
    _memoryCacheCapacityBytes: int
    _isThreadSafe: bool
    _reservedDic: dict
    _logger: Logger
    _loggerLevel: int

    def __init__(self,
                 foldername,
                 storageCacheCapacityMB: float = 10240,
                 memoryCacheCapacityMB: float = None,
                 isThreadSafe: bool = False,
                 logger: Logger = None,
                 loggerLevel: int = INFO):

        if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)
        self._foldername = foldername

        self._storageCacheCapacityBytes = (
            10240 if storageCacheCapacityMB is None else
            storageCacheCapacityMB) * 1024 * 1024
        self._storagecache = LRUCache2(
            maxsize=self._storageCacheCapacityBytes,
            getsizeof=self._getSizeOfCacheItem("storage"),
            evict=self._storageCachePopCallback)

        if memoryCacheCapacityMB is None:
            self._memoryCacheCapacityBytes = max(
                psutil.virtual_memory().available / 4, 2147483648)  #2GB
        else:
            self._memoryCacheCapacityBytes = memoryCacheCapacityMB * 1024 * 1024
        self._memcache = LRUCache2(maxsize=self._memoryCacheCapacityBytes,
                                   getsizeof=self._getSizeOfCacheItem("mem"),
                                   evict=self._memCachePopCallback)

        self._reservedDic = dict()
        self._logger = logger
        self._loggerLevel = loggerLevel

        self._isThreadSafe = isThreadSafe
        if not isThreadSafe:
            self.get = self._get
            self.add = self._add
            self.remove = self._remove
            self.clear = self._clear
            self.setReservation = self._setReservation
            self.isReserved = self._isReserved
            self.getReservedAgents = self._getReservedAgents
            self.clearAllReservations = self._clearAllReservations
            self.hasAgent = self._hasAgent
        else:
            self._lock = multiprocessing.Lock()
            self.get = self._useThreadSafeFunc(self._get)
            self.add = self._useThreadSafeFunc(self._add)
            self.remove = self._useThreadSafeFunc(self._remove)
            self.clear = self._useThreadSafeFunc(self._clear)
            self.setReservation = self._useThreadSafeFunc(self._setReservation)
            self.isReserved = self._useThreadSafeFunc(self._isReserved)
            self.getReservedAgents = self._useThreadSafeFunc(
                self._getReservedAgents)
            self.clearAllReservations = self._useThreadSafeFunc(
                self._clearAllReservations)
            self.hasAgent = self._useThreadSafeFunc(self._hasAgent)

    @property
    def foldername(self) -> str:
        return self._foldername

    @property
    def isMemporyFull(self) -> bool:
        return self._memcache.currsize >= self._memoryCacheCapacityBytes

    @property
    def isStorageFull(self) -> bool:
        return self._storagecache.currsize >= self._storageCacheCapacityBytes

    @property
    def capacityBytes(self) -> int:
        return self._memoryCacheCapacityBytes + self._storageCacheCapacityBytes

    @property
    def busyByte(self) -> int:
        return self._memcache.currsize + self._storagecache.currsize

    @property
    def availableBytes(self) -> int:
        return self.capacityBytes - self.sizeByte

    @property
    def reservedBytes(self) -> int:
        size: int = 0
        for value in self._reservedDic.values():
            size = size + sum([os.path.getsize(xi) for xi in value])
        return size

    @property
    def isThreadSafe(self) -> bool:
        return self._isThreadSafe

    @property
    def cacheSize(self) -> int:
        return len(self._memcache) + len(self._storagecache)

    @property
    def reservedSize(self) -> int:
        return len(self._reservedDic)

    @property
    def agentInfo(self) -> list:
        info = []
        for key in self._memcache.keys():
            info.append({"key": key, "type": "mem"})
        for key in self._storagecache.keys():
            info.append({"key": key, "type": "storage"})
        for key in self._reservedDic.keys():
            info.append({"key": key, "type": "reserved"})
        return info

    def get(self, key: str, removeAgent: bool = False) -> Agent:
        pass

    def remove(self, key: str) -> bool:
        pass

    def add(self, key: str, agent: Agent, setReservation: bool = None) -> bool:
        pass

    def clear(self):
        pass

    def hasAgent(self, key: str) -> bool:
        pass

    def setReservation(self, key: str, status: bool = True) -> bool:
        # se l'agent esiste allora lo riserva per usi futuri, altrimenti ritorna false
        pass

    def isReserved(self, key: str) -> bool:
        pass

    def getReservedAgents(self, clearAllReservations: bool = True) -> list:
        pass

    def clearAllReservations(self):
        pass

    def _isReserved(self, key: str) -> bool:
        return key in self._reservedDic.keys()

    def _hasAgent(self, key: str) -> bool:
        return key in self._memcache.keys() or key in self._storagecache.keys()

    def _setReservation(self, key: str, status: bool = True) -> bool:
        alreadyReserved = self._isReserved(key)
        hasAgent = self._hasAgent(key)

        if hasAgent or alreadyReserved:
            if status is None:
                status = not alreadyReserved

            if status and not alreadyReserved:
                agent = self._memcache.get(key)
                if agent is not None:
                    self._memcache.callbackStatus(False)
                    self._memcache.pop(key)
                    self._memcache.callbackStatus(True)

                    storageItem = self._writeAgentToFiles(
                        agent, [
                            os.path.join(self._foldername,
                                         f"model_{key}_reserved"),
                            os.path.join(self._foldername,
                                         f"meta_{key}_reserved"),
                            os.path.join(self._foldername,
                                         f"config_{key}_reserved")
                        ])
                    del agent
                else:
                    oldStorageItem = self._storagecache.get(key)
                    storageItem = [f"{x}_reserved" for x in oldStorageItem]
                    self._storagecache.callbackStatus(False)
                    self._storagecache.pop(key)
                    self._storagecache.callbackStatus(True)
                    for i, item in enumerate(oldStorageItem):
                        os.rename(item, storageItem[i])
                    del oldStorageItem

                self._reservedDic[key] = storageItem

            elif not status and alreadyReserved:
                storageItem = self._reservedDic[key]
                agent = self._readAgentFromFiles(storageItem)
                self._add(key, agent)

                del self._reservedDic[key]
                for filename in storageItem:
                    os.remove(filename)
                del storageItem
            self._writeLogger("setReservation")
            return True
        else:
            return False

    def _clearAllReservations(self):
        keys = list(self._reservedDic.keys())
        for i in range(len(keys)):
            self._setReservation(keys[i], False)
        self._writeLogger("clearAllReservations")

    def _getReservedAgents(self, clearAllReservations: bool = True) -> list:
        results = []

        for key in self._reservedDic.keys():
            storeItem = self._reservedDic[key]
            agent = self._readAgentFromFiles(storeItem)
            results.append(agent)

        if clearAllReservations:
            self._clearAllReservations()
        return results

    def _get(self, key: str, removeAgent: bool = False) -> Agent:
        agent: Agent = self._memcache.get(key)
        if agent is None:
            storeItem = self._storagecache.get(key)
            if storeItem is not None:
                agent = self._readAgentFromFiles(storeItem)
                self._storagecache.pop(key)
                if not removeAgent:
                    self._memcache[key] = agent
        elif removeAgent:
            self._memcache.callbackStatus(False)
            self._memcache.pop(key)
            self._memcache.callbackStatus(True)

        if agent is not None:
            agent.reset()
            return agent
        else:
            return None

    def _remove(self, key: str) -> bool:
        result: bool = False
        agent: Agent = self._memcache.get(key)
        if agent is not None:
            self._memcache.callbackStatus(False)
            self._memcache.pop(key)
            self._memcache.callbackStatus(True)
            del agent
            result = True
        elif key is self._storagecache.keys():
            self._storagecache.pop(key)
            result = True
        self._writeLogger("remove")
        return result

    def _add(self,
             key: str,
             agent: Agent,
             setReservation: bool = None) -> bool:

        result: bool = False
        existsAgent = self._hasAgent(key)

        if existsAgent and setReservation:
            raise Exception(
                "It is not possibile to set the reservation state in an existing agent in cache"
            )
        elif not existsAgent and setReservation:
            storageItem = self._writeAgentToFiles(agent, [
                os.path.join(self._foldername, f"model_{key}_reserved"),
                os.path.join(self._foldername, f"meta_{key}_reserved"),
                os.path.join(self._foldername, f"config_{key}_reserved")
            ])
            self._reservedDic[key] = storageItem
            result = True
        elif not existsAgent:
            self._memcache[key] = agent
            result = True
        self._writeLogger("add")
        return result

    def _clear(self):
        self._memcache.callbackStatus(False)
        self._storagecache.callbackStatus(False)

        self._memcache.clear()
        for value in self._storagecache.values():
            for filename in value:
                os.remove(filename)
        self._storagecache.clear()

        self._memcache.callbackStatus(True)
        self._storagecache.callbackStatus(True)
        self._writeLogger("clear")

    def _getSizeOfCacheItem(self, typeCache: str):
        if typeCache == "mem":
            return lambda x: asizeof.asizeof(x)
        elif typeCache == "storage":
            return lambda x: sum([os.path.getsize(xi) for xi in x])

    def _writeLogger(self, event: str):
        if self._logger is not None:
            msg = f"[{event}-> mem={len(self._memcache)}({self._memcache.currsize/1024/1024}MB), storage={len(self._storagecache)}({self._storagecache.currsize/1024/1024}MB), reserved={len(self._reservedDic)}({self.reservedBytes/1024/1024}MB)]"
            self._logger.log(level=self._loggerLevel, msg=msg)
            self._logger.debug(f"available:{psutil.virtual_memory().available/1024/1024}MB")

    def _readAgentFromFiles(self, storeItem: list) -> Agent:
        with open(storeItem[2], "r") as fp:
            agentConfig = AgentConfiguration(json.load(fp))
        agentBuilder = AgentBuilder(agentConfig)
        agentBuilder.create()
        agent = agentBuilder.getAgent()
        agent.load(storeItem[0], storeItem[1])
        return agent

    def _writeAgentToFiles(self, agent: Agent, storeItem: list):
        realFilenames = agent.save(modelfilename=storeItem[0],
                                   metadatafilename=storeItem[1])

        configFilename = storeItem[2]
        with open(configFilename, "w") as fp:
            json.dump(agent.getConfiguration().to_dict(), fp)

        return (*realFilenames, configFilename)

    def __setitem__(self, key: str, value):
        assert isinstance(value, Agent)
        self.add(key, value)

    def __getitem__(self, key: str):
        return self.get(key)

    def __sizeof__(self) -> int:
        return self.size

    def __len__(self) -> int:
        return len(self._memcache) + len(self._storagecache)

    def __del__(self):
        self._memcache.callbackStatus(False)
        self._storagecache.callbackStatus(False)

        self._memcache.clear()
        try:
            for value in self._storagecache.values():
                for filename in value:
                    os.remove(filename)
        except Exception as err:
            print(f"ERROR IN AGENT CACHE DISPOSE: {str(err)}")
        self._storagecache.clear()

        try:
            for value in self._reservedDic.values():
                for filename in value:
                    os.remove(filename)
        except Exception as err:
            print(f"ERROR IN AGENT CACHE RESERVED DISPOSE: {str(err)}")
        self._reservedDic.clear()

        del self._reservedDic
        del self._storagecache
        del self._memcache

    def _useThreadSafeFunc(self, fun):

        def lkfun(*args):
            with self._lock:
                return fun(*args)

        return lkfun

    def _memCachePopCallback(self, key: str, value: Agent):
        storageItem = self._writeAgentToFiles(value, [
            os.path.join(self._foldername, f"model_{key}"),
            os.path.join(self._foldername, f"meta_{key}"),
            os.path.join(self._foldername, f"config_{key}")
        ])

        self._storagecache[key] = storageItem
        del value  #TODO potrebbe creare qualche problema
        gc.collect()

    def _storageCachePopCallback(self, key, value: tuple):
        for filename in value:
            os.remove(filename)
