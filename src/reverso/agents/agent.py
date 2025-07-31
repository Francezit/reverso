import json
from logging import Logger
import numpy as np
from ..utilities import ConfigurationBase


class AgentConfiguration(ConfigurationBase):
    index: int
    agentType: str
    n_steps = 3
    forecastingMode: int = 0
    autoSetNStep=False

    def __init__(self, in_dict: dict = None) -> None:
        super().__init__(in_dict)


class AgentTrainOption(ConfigurationBase):
    model_optimize = False
    train_epochs = 10
    train_evaluate_method = "default"
    train_learning_rate = 0.01
    trainingExperiments: list = None
    testExperiments: list = None
    hp_optimizer_crossover = 0.9
    hp_optimizer_mutation = 0.5
    hp_optimizer_generations = 2
    hp_optimizer_population = 10

    def __init__(self, in_dict: dict = None) -> None:
        super().__init__(in_dict)

class Agent:
    index: int
    agentType: str
    n_step: int
    mode: int
    coeff: float

    def __init__(self, config):
        if isinstance(config, AgentConfiguration) or config is dict:
            self.index = config.index
            self.agentType = config.agentType
            self.n_step = config.n_steps
            self.mode = config.forecastingMode
            self.coeff = 1
        elif isinstance(config, Agent):
            self.index = config.index
            self.agentType = config.agentType
            self.n_step = config.n_step
            self.mode = config.mode
            self.coeff = 1
        else:
            raise Exception("type not supported")

    def __del__(self):
        pass

    @property
    def score(self) -> float:
        raise Exception("Not implemented")

    @property
    def uniqueKey(self):
        return f"{self.agentType}_{str(self.index)}"

    def getConfiguration(self):
        a = AgentConfiguration()
        a.index = self.index
        a.agentType = self.agentType
        a.n_steps = self.n_step
        a.forecastingMode = self.mode
        return a

    def predict(self, inputs: np.ndarray) -> np.double:
        raise Exception("Not implemented")

    def reset(self):
        pass

    def train(self, trainset,testset, option: AgentTrainOption, logger: Logger = None):
        raise Exception("Not implemented")

    def evaluate(self, testset, method: str, logger: Logger = None):
        raise Exception("Not implemented")

    def _cloneFrom(self, agent):
        pass

    def _saveModel(self, modelfilename: str):
        pass

    def _loadModel(self, modelfilename: str):
        pass

    def _storeCustomData(self, obj: dict):
        pass

    def _loadCustomData(self, obj: dict):
        pass

    def save(self, modelfilename, metadatafilename):
        modelfilename = self._saveModel(modelfilename)
        with open(metadatafilename, mode="w") as f:
            obj = {"coeff": self.coeff}
            self._storeCustomData(obj)
            json.dump(obj, f)
        return modelfilename, metadatafilename

    def load(self, modelfilename, metadatafilename):
        modelfilename = self._loadModel(modelfilename)
        with open(metadatafilename, mode="r") as f:
            data = json.load(f)

        if "coeff" in data.keys():
            self.coeff = data["coeff"]
        self._loadCustomData(data)
        return modelfilename, metadatafilename
