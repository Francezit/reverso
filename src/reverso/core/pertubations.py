import numpy as np
from ..utilities import ConfigurationBase


class SignalPertubationFunction(ConfigurationBase):

    duration: int

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)

    @property
    def name(self):
        pass

    def setInitValue(self, initTick: int, initValue: float, initRange: list):
        pass

    def getComputeSetting(self, name):
        name = f"_{name}"
        return getattr(self, name) if hasattr(self, name) else None

    def getValue(self, tick):
        pass

    def __call__(self, *args, **kwds):
        return self.getValue(args[0])

    @staticmethod
    def getPertubationFunctionByName(name: str):
        if name == "trapezium":
            return TrapeziumPertubationFunction
        elif name == "instant":
            return InstantPertubationFunction
        else:
            raise Exception("Not supported")

    @staticmethod
    def makePertubationFunction(name: str,
                                config: dict,
                                defaultConfig: dict = None):
        if defaultConfig is None:
            baseConfig = {}
        else:
            baseConfig = defaultConfig.copy()
        baseConfig.update(config)

        clsItem = SignalPertubationFunction.getPertubationFunctionByName(name)
        return clsItem(baseConfig)

    @staticmethod
    def getAvailablePertubationFunctions():
        return ["trapezium", "instant"]


class InstantPertubationFunction(SignalPertubationFunction):
    widthFactor: float = 0.5

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)

    @property
    def name(self):
        return "instant"

    def setInitValue(self, initTick: int, initValue: float,
                        initRange: list):
        entityDeltaMax = initRange[1] - initValue if initValue < initRange[
            1] else 0

        self._basevalue = initValue
        self._startTick = initTick
        self._endTick = self._startTick + self.duration
        self._pertubationWidthMax = self.widthFactor * entityDeltaMax

    def getValue(self, tick):
        if tick >= self._startTick and tick < self._endTick:
            return self._basevalue + self._pertubationWidthMax
        elif tick >= self._endTick:
            return self._basevalue
        else:
            raise Exception(
                "InstantPertubationFunction has not defined in " +
                str(tick))


class TrapeziumPertubationFunction(SignalPertubationFunction):
    maxPeakDuration: int = 2
    widthFactor: float = 0.5

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)

    @property
    def name(self):
        return "trapezium"

    def setInitValue(self, initTick: int, initValue: float, initRange: list):
        entityDeltaMax = initRange[1] - initValue if initValue < initRange[
            1] else 0
        pertubationNotStableSteps = int(
            np.ceil((self.duration - self.maxPeakDuration) / 2))

        pertubationWidthMax = self.widthFactor * entityDeltaMax
        pertubationWidthNormal = pertubationWidthMax / pertubationNotStableSteps

        self._pertubationWidthNormal = pertubationWidthNormal
        self._pertubationWidthMax = pertubationWidthMax

        self._basevalue = initValue
        self._startTick = initTick
        self._peakBeforeTick = initTick + pertubationNotStableSteps
        self._peakAfterTick = self._peakBeforeTick + self.maxPeakDuration
        self._endTick = self._peakAfterTick + pertubationNotStableSteps

    def getValue(self, tick):
        if tick >= self._startTick and tick < self._peakBeforeTick:
            perubationInterval = 1 + (tick - self._startTick)
            pertubationValue = self._basevalue + self._pertubationWidthNormal * perubationInterval
            return pertubationValue
        elif tick >= self._peakBeforeTick and tick < self._peakAfterTick:
            return self._basevalue + self._pertubationWidthMax
        elif tick >= self._peakAfterTick and tick < self._endTick:
            perubationInterval = 1 + (tick - self._peakAfterTick)
            pertubationValue = (self._basevalue + self._pertubationWidthMax
                                ) - (self._pertubationWidthNormal *
                                     perubationInterval)
            return pertubationValue
        else:
            raise Exception(
                "TrapeziumPertubationFunction has not defined in " + str(tick))
