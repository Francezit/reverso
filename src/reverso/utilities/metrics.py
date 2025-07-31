import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import os
from math import ceil, floor


class MetricsInfo:
    _predictedData: np.ndarray
    _targetData: np.ndarray
    _stepToStart: int
    _cache: dict
    _isLightVersion: bool
    _default_metric: str
    _warnings: list
    _global_variables: list
    _useErrorMetricNormalization: bool

    fixStdZero: bool = True
    errorNormalizationFactor: str = "mean"

    def __init__(self,
                 predictedData: np.ndarray,
                 targetData: np.ndarray,
                 defaultMetric: str = "mse",
                 stepToStart: int = None,
                 globalVariables: list = None,
                 useErrorMetricNormalization: bool = False) -> None:
        self._predictedData = predictedData
        self._targetData = targetData
        self._cache = {}
        self._isLightVersion = False
        self._warnings = []
        self._global_variables = globalVariables
        self._default_metric = defaultMetric
        self._useErrorMetricNormalization = useErrorMetricNormalization

        assert predictedData.shape == targetData.shape
        self.setStepToStart(stepToStart)

    @property
    def stepToStart(self):
        return self._stepToStart

    @property
    def hasWarnings(self):
        return len(self._warnings) > 0

    @property
    def hasGlobalVariableFilter(self):
        return self._global_variables is not None

    def setStepToStart(self, value: int = None):
        assert not self._isLightVersion
        if value is None:
            self._stepToStart = 0
        elif value >= 0 and value < self._targetData.shape[0]:
            self._stepToStart = value
        else:
            raise Exception("Value not supported in setStepToStart")

    def getPredictedData(self):
        assert not self._isLightVersion
        return self._predictedData

    def getTargetData(self):
        assert not self._isLightVersion
        return self._targetData

    def getWarnings(self):
        return self._warnings.copy()

    def _addWarning(self, type, msg):
        msg = f"[{type}] {msg}"
        if not msg in self._warnings:
            self._warnings.append(msg)

    def _normalizeErrorMetric(self, error: float, varIndex: int):
        if self._useErrorMetricNormalization:
            dt = self._targetData[:, varIndex]
            dt = dt[np.isfinite(dt)]

            if self.errorNormalizationFactor is None:
                val = np.max(dt)-np.min(dt)
            elif isinstance(self.errorNormalizationFactor, str):
                val = eval(f"np.{self.errorNormalizationFactor}(dt)")
            else:
                val = self.errorNormalizationFactor(dt)

            stdValue = np.std(dt)
            if np.isnan(stdValue) or np.isinf(stdValue) or stdValue == 0:
                self._addWarning("useErrorMetricNormalization",
                                 f"std not valid in {varIndex}-th variable, {stdValue}")

            return error/val
        else:
            return error

    def _normalizeData(self):
        data = self._targetData[self._stepToStart:, :], self._predictedData[
            self._stepToStart:, :]
        if self.fixStdZero:
            for k, item in enumerate(data):
                s: np.ndarray = np.std(item, axis=0)
                for i in range(s.size):
                    if s[i] == 0:
                        item[-1, :] = item[-1, :] + 1e-10
                        self._addWarning(
                            "fixStdZero",
                            f"Found std equals zero in the {i+1}-th component of {'target' if k==0 else 'predicted'} vector"
                        )

        return data

    @property
    def pearsonScore(self) -> np.ndarray:

        def compFun():
            target, predicted = self._normalizeData()
            n = target.shape[1]
            score = np.zeros((n, ))
            for i in range(n):
                score[i] = np.corrcoef(target[:, i], predicted[:, i]).min()
            return np.abs(score)

        return self._checkCache("pearsonScore", compFun)

    @property
    def cosineScore(self) -> np.ndarray:

        def compFun():
            target, predicted = self._normalizeData()
            n = target.shape[1]
            score = np.zeros((n, ))
            for i in range(n):
                score[i] = np.dot(target[:, i], predicted[:, i]) / (
                    np.linalg.norm(target[:, i]) *
                    np.linalg.norm(predicted[:, i]))
            return np.abs(score)

        return self._checkCache("cosineScore", compFun)

    @property
    def mseScore(self) -> np.ndarray:

        def compFun():
            target, predicted = self._normalizeData()
            n = target.shape[1]
            score = np.zeros((n, ))
            for i in range(n):
                e = np.mean(np.square(target[:, i] - predicted[:, i]))
                score[i] = self._normalizeErrorMetric(e, i)
            return score

        return self._checkCache("msescore", compFun)

    @property
    def maeScore(self) -> np.ndarray:

        def compFun():
            target, predicted = self._normalizeData()
            n = target.shape[1]
            score = np.zeros((n, ))
            for i in range(n):
                e = np.mean(np.abs(target[:, i] - predicted[:, i]))
                score[i] = self._normalizeErrorMetric(e, i)
            return score

        return self._checkCache("maescore", compFun)

    @property
    def rmseScore(self) -> np.ndarray:

        def compFun():
            target, predicted = self._normalizeData()
            n = target.shape[1]
            score = np.zeros((n, ))
            for i in range(n):
                e = np.sqrt(np.mean(np.square(target[:, i] - predicted[:, i])))
                score[i] = self._normalizeErrorMetric(e, i)
            return score

        return self._checkCache("rmsescore", compFun)

    @property
    def statProperties(self) -> dict:
        def compFun():
            target, predicted = self._normalizeData()
            return {
                "target": {
                    "mean": np.mean(target, axis=0).tolist(),
                    "std": np.std(target, axis=0).tolist(),
                    "max": np.max(target, axis=0).tolist(),
                    "min": np.min(target, axis=0).tolist(),
                    "median": np.median(target, axis=0).tolist()
                },
                "predicted": {
                    "mean": np.mean(predicted, axis=0).tolist(),
                    "std": np.std(predicted, axis=0).tolist(),
                    "max": np.max(predicted, axis=0).tolist(),
                    "min": np.min(predicted, axis=0).tolist(),
                    "median": np.median(predicted, axis=0).tolist()
                }
            }

        return self._checkCache("statProperties", compFun)

    @property
    def pearsonSimiliarity(self) -> float:
        if self.hasGlobalVariableFilter:
            return np.mean(self.pearsonScore[self._global_variables])
        else:
            return np.mean(self.pearsonScore)

    @property
    def cosineSimiliarity(self) -> float:
        if self.hasGlobalVariableFilter:
            return np.mean(self.cosineScore[self._global_variables])
        else:
            return np.mean(self.cosineScore)

    @property
    def mse(self):
        if self.hasGlobalVariableFilter:
            return np.mean(self.mseScore[self._global_variables])
        else:
            return np.mean(self.mseScore)

    @property
    def mae(self):
        if self.hasGlobalVariableFilter:
            return np.mean(self.maeScore[self._global_variables])
        else:
            return np.mean(self.maeScore)

    @property
    def rmse(self):
        if self.hasGlobalVariableFilter:
            return np.mean(self.rmseScore[self._global_variables])
        else:
            return np.mean(self.rmseScore)

    @property
    def isLightVersion(self):
        return self._isLightVersion

    def to_dic(self):
        res = {
            "MSE": self.mse,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "PearsonSimiliarity": self.pearsonSimiliarity,
            "CosineSimiliarity": self.cosineSimiliarity,
            "PearsonScore": self.pearsonScore.tolist(),
            "CosineScore": self.cosineScore.tolist(),
            "MSEScore": self.mseScore.tolist(),
            "MAEScore": self.maeScore.tolist(),
            "RMSEScore": self.rmseScore.tolist(),
            "StatProperties": self.statProperties
        }
        if len(self._warnings) > 0:
            res["_warnings"] = self._warnings
        if self.hasGlobalVariableFilter:
            res["_global_variables"] = self._global_variables
        return res

    def to_list(self):
        return [
            self.mse, self.mae, self.rmse, self.pearsonSimiliarity,
            self.cosineSimiliarity
        ], ["mse", "mae", "rmse", "pearsonSimiliarity", "cosineSimiliarity"]

    def to_str(self):
        return json.dumps(self.to_dic())

    def getValue(self, metricName: str = None) -> float:
        if metricName is None:
            metricName = self._default_metric
        return self.__getattribute__(metricName)

    def analysis(self,
                 nameComponents: list,
                 foldername: str,
                 includeTargetPlot=True,
                 includePredictedPlot=True,
                 includeSingleComparision=True,
                 includeMergeComparison=True,
                 mergeComparisonCols=2):

        os.makedirs(foldername, exist_ok=True)

        np.savetxt(os.path.join(foldername, "targetData.csv"),
                   self._targetData,
                   fmt='%.10f',
                   delimiter=',')
        np.savetxt(os.path.join(foldername, "predictedData.csv"),
                   self._predictedData,
                   fmt='%.10f',
                   delimiter=',')
        with open(os.path.join(foldername, "metrics.json"), "w") as f:
            json.dump(self.to_dic(), f)

        obs = self._targetData.shape[0]
        ncomps = self._targetData.shape[1]

        plt.close()
        if includeTargetPlot:
            with mpl.rc_context({'lines.linewidth': 1}):
                plt.plot(np.arange(0, obs, dtype=np.int32), self._targetData)
                plt.legend(nameComponents,
                           loc='center right',
                           bbox_to_anchor=(1.25, 0.5))
                plt.xlabel("Time Step")
                plt.ylabel("Values")
                plt.savefig(os.path.join(foldername, "targetPlot.svg"),
                            bbox_inches="tight")
                plt.clf()

        if includePredictedPlot:
            with mpl.rc_context({'lines.linewidth': 1}):
                plt.plot(np.arange(0, obs, dtype=np.int32),
                         self._predictedData)
                plt.legend(nameComponents,
                           loc='center right',
                           bbox_to_anchor=(1.25, 0.5))
                plt.xlabel("Time Step")
                plt.ylabel("Values")
                plt.savefig(os.path.join(foldername, "predictedPlot.svg"),
                            bbox_inches="tight")
                plt.clf()

        if includeSingleComparision:
            with mpl.rc_context({'lines.linewidth': 1}):
                for i in range(ncomps):
                    title = nameComponents[i]
                    plt.plot(
                        np.arange(0, obs, dtype=np.int32),
                        np.vstack([
                            self._targetData[:, i], self._predictedData[:, i]
                        ]).T)
                    plt.legend(["Target", "Predicted"],
                               loc='center right',
                               bbox_to_anchor=(1.25, 0.5))
                    plt.xlabel("Time Step")
                    plt.ylabel("Values")
                    plt.savefig(os.path.join(foldername,
                                             "compare_" + title + ".svg"),
                                bbox_inches="tight")
                    plt.clf()

        if includeMergeComparison:
            with mpl.rc_context({'lines.linewidth': 1}):
                cols = mergeComparisonCols
                rows = int(ceil(ncomps / cols))

                gs = gridspec.GridSpec(rows, cols)
                fig = plt.figure()
                for i in range(ncomps):
                    ax = fig.add_subplot(gs[i])
                    ax.set_title(nameComponents[i], fontsize='small')
                    ax.plot(np.arange(0, obs, dtype=np.int32),
                            self._targetData[:, i],
                            linestyle="dashed")
                    ax.plot(np.arange(0, obs, dtype=np.int32),
                            self._predictedData[:, i],
                            linestyle="solid")
                    # ax.set_xlabel("steps", fontsize='small')
                    # ax.set_ylabel("values", fontsize='small')

                fig.legend(["Target", "Estimated"],
                           loc='lower right',
                           bbox_to_anchor=(1, -0.1),
                           ncol=2,
                           bbox_transform=fig.transFigure)

                fig.tight_layout()
                plt.savefig(os.path.join(foldername, "comparison.svg"),
                            bbox_inches="tight")
                plt.clf()
        plt.close()

    def _checkCache(self, name: str, fun):
        if name in self._cache.keys():
            return self._cache[name]
        elif not self._isLightVersion:
            value = fun()
            self._cache[name] = value
            return value
        else:
            raise Exception("MetricInfo is in light version")

    def clearCache(self):
        if not self._isLightVersion:
            self._cache.clear()
        else:
            raise Exception("MetricInfo is in light version")

    def betterThan(self, other, metricName: str = None):
        if other is None:
            return True
        if metricName is None:
            metricName = self._default_metric

        if isinstance(other, MetricsInfo):
            if metricName == "all":
                return self.mse < other.mse and self.mae < other.mae and self.rmse < other.rmse and self.pearsonSimiliarity > other.pearsonSimiliarity and self.cosineSimiliarity > other.cosineSimiliarity
            else:
                return self.compareValue(self.getValue(metricName),
                                         other.getValue(metricName),
                                         metricName) > 0
        else:
            if metricName == "all":
                return self.mse < other and self.mae < other and self.rmse < other and self.pearsonSimiliarity > other and self.cosineSimiliarity > other
            else:
                return self.compareValue(self.getValue(metricName), other,
                                         metricName) > 0

    def useLightVersion(self, metricsToLoad: list = None):
        if not self._isLightVersion:
            if metricsToLoad is None:
                self.to_list()
            else:
                [self.getValue(name) for name in metricsToLoad]
            del self._predictedData
            del self._targetData
            self._isLightVersion = True

    def save(self, filename: str):
        assert not self._isLightVersion
        data = {
            "pred": self._predictedData.tolist(),
            "target": self._targetData.tolist(),
            "stepToStart": self._stepToStart,
            "defaultMetric": self._default_metric,
            "_warnings": self._warnings,
            "_global_variables": self._global_variables
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @staticmethod
    def compareValue(value1: float, value2: float, metricName: str) -> int:
        if value1 == value2:
            return 0

        if metricName in ("pearsonSimiliarity", "cosineSimiliarity"):
            return 1 if value1 > value2 else -1
        elif metricName in ("mse", "mae", "rmse"):
            return 1 if value1 < value2 else -1

    @staticmethod
    def load(filename: str):
        with open(filename, "r") as f:
            data: dict = json.load(f)

        metric = MetricsInfo(predictedData=data["pred"],
                             targetData=data["target"],
                             defaultMetric=data["defaultMetric"],
                             stepToStart=data["stepToStart"])
        if "warnings" in data.keys():
            metric._warnings = data["_warnings"]
        if "_global_variables" in data.keys():
            metric._global_variables = data["_global_variables"]
        return metric
