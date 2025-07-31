import numpy as np
import multiprocessing
import gc
import os
import uuid

from pympler import asizeof
from sklearn.linear_model import LinearRegression

from ..utilities import ConfigurationBase, NumpyArrayWrapper
from ..agents import Agent
from .analysis import EnvironmentAnalysis
from .pertubations import SignalPertubationFunction


class EnvironmentState():

    signals: list
    enable_signal: bool
    enable_bounds: bool

    _tick: int
    _numberComponents: int
    _history: NumpyArrayWrapper

    def __init__(self,
                 numberComponents: int,
                 capacityCache: int = None,
                 memFilename: str = None):
        self._numberComponents = numberComponents
        self._history = NumpyArrayWrapper(numberComponents=numberComponents,
                                          memFilename=memFilename,
                                          dType="float32",
                                          capacityCacheLastItem=capacityCache)
        self.signals = []
        self.enable_signal = True
        self.enable_bounds = False

    def __del__(self):
        self._history.clear()
        self.signals.clear()
        del self._history, self.signals

    @property
    def tick(self):
        return self._tick

    @property
    def values(self) -> np.ndarray:
        return self._history.tail()

    @property
    def historySize(self) -> int:
        return len(self._history)

    def resetValues(self, initValues: np.ndarray):
        nobs = initValues.shape[0]
        self._tick = nobs - 1

        self._history.replace(initValues)

    def updateValues(self, newValues: np.ndarray):
        newValues = np.abs(newValues)
        if self.enable_bounds:
            newValues = np.clip(newValues, 0, 1)

        self._history.append(newValues)
        self._tick = self._tick + 1

    def getValue(self, tick: int) -> np.ndarray:
        return self._history[tick]

    def getValues(self,
                  startTick: int = None,
                  endTick: int = None) -> np.ndarray:
        return np.stack(self._history.slice(startTick, endTick))

    def getStateMatrix(self,
                       componentIndex: int,
                       n_step: int = 1,
                       forecastingMode: int = 0) -> np.ndarray:
        n = self._numberComponents
        h = self.historySize

        lastValues = self._history.getLastItems(n_step)
        if len(lastValues) < n_step:
            data = np.zeros((n_step, n))
            data[n_step - h:n_step, :] = np.concatenate(lastValues, axis=0)
        else:
            data = np.concatenate(lastValues, axis=0)

        if forecastingMode == 0:
            mask = np.ones((n, ), dtype=np.bool8)
            mask[componentIndex] = 0
            data = data[:, mask]

        return data

    def copyFrom(self, envState):
        self._tick = envState.tick
        self._history.replace(envState._history.toList())
        # TODO forse si dovrebbero copiare anche le vearibili


class EnvironmentMetadata(ConfigurationBase):
    componentNames: list = None
    datasetName: str = None
    creationDate: str = None
    version: int = None
    trainData: list = []
    testData: list = []


class EnvironmentData(ConfigurationBase):
    initialState: np.ndarray = np.array([])
    useInitialState: bool = False
    maxValueState: np.ndarray = np.array([])

    def _converter_serialize(self, name: str, value: any):
        if name == "initialState":
            return value.tolist() if value is not None else []
        elif name == "maxValueState":
            return value.tolist() if value is not None else []
        return value

    def _converter_deserialize(self, name: str, value: any):
        if name == "initialState":
            return np.array(value)
        elif name == "maxValueState":
            return np.array(value)
        return value


class Environment:
    _agents: list
    _metadata = EnvironmentMetadata
    _envdata: EnvironmentData

    _state: EnvironmentState

    _n_agents: int
    _context_state: EnvironmentState
    _enable_debug: bool
    _enable_upper_bound: bool
    _cache_folder: str

    def __init__(self,
                 agents,
                 metadata: EnvironmentMetadata = None,
                 envdata: EnvironmentData = None,
                 cacheFolder: str = None):

        if metadata is None:
            metadata = EnvironmentMetadata()
        elif not isinstance(metadata, EnvironmentMetadata):
            raise Exception("Metadata is not dict")
        self._metadata = metadata

        if envdata is None:
            envdata = EnvironmentData()
        elif not isinstance(envdata, EnvironmentData):
            raise Exception("envdata is not EnvironmentData")
        self._envdata = envdata

        self._agents = agents
        self._n_agents = len(agents)

        self._enable_debug = False
        self._context_state = None
        self._state = None
        self._cache_folder = cacheFolder

    def analysis(self):
        return EnvironmentAnalysis(data=self.getHistoryState(),
                                   nameComponents=self.componentNames)

    @property
    def mem_size(self):
        return asizeof.asizeof(self)

    @property
    def has_valid_state(self):
        return self._state is not None

    @property
    def is_debug_model(self):
        return self._enable_debug

    @property
    def agentTypes(self):
        return [a.agentType for a in self._agents]

    @property
    def agentMaxSteps(self):
        return max([a.n_step for a in self._agents])

    @property
    def agentConfigurations(self):
        return [a.getConfiguration() for a in self._agents]

    @property
    def currentState(self):
        return self._state.values[0, :]

    @property
    def currentTick(self):
        return self._state.tick

    @property
    def componentNames(self):
        nm = self._metadata.componentNames
        return nm if nm is not None else [
            "Item " + str(i) for i in range(self._n_agents)
        ]

    @property
    def numberComponent(self):
        return self._n_agents

    @property
    def isQuiet(self):
        return self._state.tick == 0

    @property
    def isPertubated(self):
        return len(self._state.signals) > 0

    @property
    def hasInitialState(self):
        return self._envdata.useInitialState

    @property
    def isContextMode(self):
        return hasattr(self,
                       "_contextState") and self._context_state is not None

    @property
    def numberPertubations(self):
        return len(self._state.signals)

    @property
    def usePertubation(self):
        return self._state.enable_signal

    @property
    def agentCoeffs(self):
        return [a.coeff for a in self._agents]

    def enableDebugMode(self, state: bool = None):
        self._enable_debug = state if state is not None else not self._enable_debug

    def getMetadata(self, name: str, default=None):
        try:
            return self._metadata.__getattribute__(name)
        except:
            return default

    def setMetadata(self, name: str, value: object):
        self._metadata.__setattr__(name, value)

    def setAgentCoeffs(self, coeffs: list):
        for i in range(self._n_agents):
            self._agents[i].coeff = coeffs[i]

    def replaceAgent(self, index, agent: Agent):
        self._agents[index] = agent

    def getStateByTick(self, tick: int):
        return self._state.getValue(tick)

    def getHistoryState(self, startTick: int = None, endTick: int = None):
        return self._state.getValues(startTick, endTick)

    def getCurrentValue(self, index: int):
        if hasattr(self._envdata, "maxValueState"):
            return self._state.values[0, index], [
                0, self._envdata.maxValueState[index]
            ]
        else:
            return self._state.values[0, index], [0, 1]

    def storeInitialData(self,
                         initState: np.ndarray = None,
                         maxState: np.ndarray = None):
        if initState is not None:
            initState = np.reshape(initState,
                                   (self.agentMaxSteps, self._n_agents))
            self._envdata.initialState = initState
            self._envdata.useInitialState = True

        if maxState is not None:
            maxState = np.reshape(maxState, (self._n_agents, ))
            self._envdata.maxValueState = maxState

    def runUntil(self, absoluteTick: int, useParallelProcessing=False):
        c = self._state.tick
        e = absoluteTick - c
        assert e >= 0
        if not useParallelProcessing:
            self.nextState(e)
        else:
            self.nextStateParallel(e)

    def runFor(self, relativeTick: int, useParallelProcessing=False):
        assert relativeTick >= 0
        if not useParallelProcessing:
            self.nextState(relativeTick)
        else:
            self.nextStateParallel(relativeTick)

    def runUntilStability(self,
                          step: int = 10,
                          delta: float = 0.2,
                          increaseFactor: float = 1.5):
        assert step > 0
        initTick = self._state.tick
        while True:
            startTick = self._state.tick
            self.nextState(step)
            endTick = self._state.tick

            xTick = np.arange(startTick, endTick, 1).reshape(-1, 1)
            yTick = self.getHistoryState(startTick, endTick)
            coeffs = [
                LinearRegression().fit(xTick, yTick[:, i]).coef_
                for i in range(self._n_agents)
            ]

            score = np.max(np.abs(coeffs))
            if score <= delta:
                break
            else:
                delta = delta * increaseFactor
        return self._state.tick - initTick

    def nextState(self, tick: int = 1):
        if tick == 0:
            return
        assert tick > 0

        envState: EnvironmentState = self._state
        currentSignals = list(
            envState.signals) if envState.enable_signal else []

        for ctick in range(tick):

            stateValues = np.zeros((1, self._n_agents))
            maskState = np.ones((self._n_agents, ), dtype=np.bool8)

            # signal
            if envState.enable_signal:
                removeSignalsIdx = []
                for sidx, item in enumerate(currentSignals):
                    if item["afterTick"] > envState.tick:
                        break
                    elif item["afterTick"] == envState.tick:
                        idx = item["index"]
                        stateValues[0, idx] = item["method"](
                            envState.values[0, :], envState.tick)
                        maskState[idx] = False
                        removeSignalsIdx.append(sidx)
                    else:
                        removeSignalsIdx.append(sidx)
                for rindex in sorted(removeSignalsIdx, reverse=True):
                    del currentSignals[rindex]
                del removeSignalsIdx

            # update state
            for item in self._agents:
                agent: Agent = item
                if maskState[agent.index]:
                    inputData = envState.getStateMatrix(
                        componentIndex=agent.index,
                        n_step=agent.n_step,
                        forecastingMode=agent.mode)
                    outputData = agent.predict(inputData)
                    stateValues[0, agent.index] = outputData * agent.coeff
                    # stateValues[0, agent.index] = min(
                    #    np.abs(outputData * agent.coeff), 1.0)
                    del inputData

            envState.updateValues(stateValues)

            if self._enable_debug:
                print(
                    f"[{str(self.mem_size)}]  {str(envState.tick)}: {', '.join([str(s) for s in envState.values] )}"
                )

            gc.collect()
        pass

    def _agentForecastingTask(self, agent: Agent, enable: bool):
        # NON FUNZIONA
        if enable:
            inputData = self._state.getStateMatrix(componentIndex=agent.index,
                                                   n_step=agent.n_step,
                                                   forecastingMode=agent.mode)
            outputData = agent.predict(inputData)
            return np.abs(outputData * agent.coeff), agent.index
        return None

    def nextStateParallel(self, tick: int = 1):
        # NON FUNZIONA
        if tick == 0:
            return
        assert tick > 0

        envState: EnvironmentState = self._state

        with multiprocessing.Pool() as pool:
            for _ in range(tick):

                stateValues = np.zeros((1, self._n_agents))
                maskState = np.ones((self._n_agents, ), dtype=np.bool8)

                # signal
                if envState.enable_signal:
                    for item in envState.signals:
                        if item["afterTick"] > envState.tick:
                            break
                        elif item["afterTick"] == envState.tick:
                            idx = item["index"]
                            stateValues[0, idx] = item["method"](
                                envState.values[0, :])
                            maskState[idx] = False

                # update state
                taskArgs = [(self._agents[i], maskState[i])
                            for i in range(self._n_agents)]
                for result in pool.starmap(self._agentForecastingTask,
                                           taskArgs):
                    if result is not None:
                        stateValues[0, result[1]] = result[0]

                envState.updateValues(stateValues)
        pass

    def createPertubationFunction(
            self,
            entityIndex: int,
            pertubationFunction: SignalPertubationFunction,
            afterTick: int = None,
            id: str = None):

        if id == None:
            id = f"Pert_{entityIndex}"
        if afterTick is None:
            afterTick = self.currentTick

        _, entityRange = self.getCurrentValue(entityIndex)

        # init pertubation
        def initPertubationFun(v, t):
            pertubationFunction.setInitValue(initTick=t,
                                             initRange=entityRange,
                                             initValue=v[entityIndex])
            return pertubationFunction(t)

        self.createPertubation(entityIndex=entityIndex,
                               value=initPertubationFun,
                               method="custom",
                               afterTick=afterTick,
                               duration=1,
                               id=f"{id}_0")

        # generate pertubations
        def pertubationFn(v, t):
            return pertubationFunction(t)

        for i in range(pertubationFunction.duration - 1):
            afterTick = afterTick + 1
            self.createPertubation(entityIndex=entityIndex,
                                   value=pertubationFn,
                                   method="custom",
                                   afterTick=afterTick,
                                   duration=1,
                                   id=f"{id}_{i+1}")
        pass

    def createPertubation(self,
                          entityIndex: int,
                          value: any,
                          method="exactly",
                          afterTick: int = None,
                          duration: int = 1,
                          id: str = None):

        if method == "add":
            def mfun(x, t): return x[entityIndex] + value
        elif method == "sub":
            def mfun(x, t): return x[entityIndex] - value
        elif method == "mul":
            def mfun(x, t): return x[entityIndex] * value
        elif method == "div":
            def mfun(x, t): return x[entityIndex] / value
        elif method == "custom":
            mfun = value
        else:
            def mfun(x, t): return value

        startTick = afterTick if afterTick is not None else self._state.tick
        for i in range(duration):
            obj = {
                "index": entityIndex,
                "method": mfun,
                "afterTick": startTick + i,
                "id": id
            }
            self._state.signals.append(obj)

        self._state.signals.sort(key=lambda x: x["afterTick"])

    def clearAllPertubations(self):
        self._state.signals.clear()

    def clearPreviousPertubations(self):
        currenttick = self._state.tick
        newsignals = []
        for i in range(len(self._state.signals)):
            if self._state.signals[i]["afterTick"] > currenttick:
                newsignals.append(self._state.signals[i])
        self._state.signals = newsignals

    def disablePertubations(self):
        self._state.enable_signal = False

    def enablePertubations(self):
        self._state.enable_signal = True

    def simulateResponse(self,
                         entityIndex: int,
                         value: np.double,
                         method="exactly",
                         duration: int = 1,
                         offset: int = 1,
                         removeAllPertubation=True):
        self.createPertubation(entityIndex=entityIndex,
                               value=value,
                               method=method,
                               duration=duration)
        initialState = self._state.values
        self.nextState(duration)
        transitionState = self._state.values
        self.nextState(offset)
        finalState = self._state.values
        if removeAllPertubation:
            self.clearAllPertubations()
        return finalState, transitionState, initialState

    def _createEnvState(self):
        id = uuid.uuid4()
        s = EnvironmentState(
            numberComponents=self._n_agents,
            capacityCache=self.agentMaxSteps,
            memFilename=os.path.join(self._cache_folder, f"envState-{id}.bin")
            if self._cache_folder is not None else None)
        return s

    def _duplicateEnvState(self, baseState: EnvironmentState):
        if baseState is None:
            return None

        id = uuid.uuid4()
        s = EnvironmentState(
            numberComponents=self._n_agents,
            capacityCache=self.agentMaxSteps,
            memFilename=os.path.join(self._cache_folder, f"envState-{id}.bin")
            if self._cache_folder is not None else None)
        s.copyFrom(baseState)
        return s

    def resetState(self, initState: np.ndarray = None, steps: int = 1):

        if initState is not None:
            initState = np.reshape(initState, (steps, self._n_agents))
        elif self._envdata.useInitialState:
            initState = self._envdata.initialState
        else:
            initState = np.zeros((1, self._n_agents))

        if self._state is not None:
            del self._state

        self._state = self._createEnvState()
        self._state.resetValues(initState)

        for i in range(self._n_agents):
            self._agents[i].reset()

    def sandboxContext(self, fun):
        bkpmode = hasattr(self, "_contextState")

        if bkpmode:
            tempBackupState = self._context_state
            self._context_state = None

        tempState = self._state
        self._state = self._createEnvState()
        r = fun(self)
        del self._state

        self._state = tempState
        if bkpmode:
            self._context_state = tempBackupState
        return r

    def aquireContext(self):
        assert not self.isContextMode
        self._context_state = self._state
        self._state = self._duplicateEnvState(self._context_state)
        pass

    def commitContext(self):
        assert self.isContextMode
        del self._context_state
        self._context_state = None

    def rollbackContext(self):
        assert self.isContextMode
        del self._state
        self._state = self._context_state
        self._context_state = None
