from .agents import AgentConfiguration, AgentTrainOption
from .utilities import ConfigurationBase


class EnvironmentEvaluationOption(ConfigurationBase):
    experimentIndexs: list = None
    includeAll: float = 1.0
    defaultMetric: str = "mse"
    useParallelProcessing: bool = False
    useSandbox: bool = True
    excludeInitialValue: bool = False
    evaluationMode: str = None  # "simulation or ts_forecasting"
    forecastPercentage: float = 0.8  # only on ts_forecasting mode


class MakeEnvironmentConfiguration(ConfigurationBase):
    agents: list = []
    trainOption: AgentTrainOption = AgentTrainOption()
    useParallelProcessing: bool = False
    parallelProcesses: int = 4

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)

    def _converter_deserialize(self, name: str, value: any):
        if name == "agents":
            return [AgentConfiguration(x) for x in value]
        elif name == "trainOption":
            return AgentTrainOption(value)
        else:
            return value

    def getAgentConfiguration(self, index: int) -> AgentConfiguration:
        if index >= len(self.agents):
            ct = self.agents[-1].cloneConfig()
            ct.index = index
            return ct
        else:
            return self.agents[index]

    def expandAgents(self,
                     agentIndexs: list,
                     config: AgentConfiguration = None):
        maxIndex = max(agentIndexs)
        minIndex = min(agentIndexs)
        agentIndexs.sort()
        n = len(self.agents)

        if config is None and n > 0:
            config = self.agents[-1].cloneConfig()
        elif config is None and n == 0:
            config = AgentConfiguration()

        if minIndex < n:
            for idx in agentIndexs:
                if idx < n:
                    a = config.cloneConfig()
                    a.index = idx
                    self.agents[idx] = a
                else:
                    break

        if maxIndex >= n:
            for idx in range(maxIndex - n):
                a = config.cloneConfig()
                a.index = n + idx
                self.agents.append(a)

    def getDefaultAgentConfiguration(self):
        if self.agents is None or len(self.agents) == 0:
            return AgentConfiguration()
        else:
            last: AgentConfiguration = self.agents[-1]
            return last.cloneConfig()


class AutomakeEnvironmentConfiguration(ConfigurationBase):
    optimizeMaxIterations: int = 10
    optimizePopulationSize: int = 20
    optimizeProbMutation: float = 0.5
    optimizeProbCrossover: float = 0.7
    optimizeDisplay: bool = False

    agentCacheStorageCapacityMB: float = None
    agentCacheMemoryCapacityMB: float = None

    baseAgentConfig = AgentConfiguration()
    trainOption = AgentTrainOption()
    evalOption = EnvironmentEvaluationOption()
    useParallelProcessing: bool = False
    parallelProcesses: int = 4

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)

    def _converter_deserialize(self, name: str, value: any):
        if name == "baseAgentConfig":
            return AgentConfiguration(value)
        elif name == "trainOption":
            return AgentTrainOption(value)
        elif name == "evalOption":
            return EnvironmentEvaluationOption(value)
        else:
            return value


class ExtractInteractionNetworkOption(ConfigurationBase):
    stabilityEnabled: bool = True
    stabilityWindow: int = 10
    stabilityThreshold: float = 0.2
    stabilityThresholdFactor: float = 1.5

    pertubationType: str = "trapezium"
    pertubationConfig: dict = {}

    thresholdValues: list = None
    thresholdMethod: str = "auto"

    extractImages: bool = False
    extractImagesMargin: int = 0
    extractImageFolder: str = None
    extractImagesYLimit: bool = False

    def __init__(self, in_dict: dict = None):
        super().__init__(in_dict)
