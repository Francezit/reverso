from logging import Logger, getLogger
from .agent import *
from .nnagent import *
from .cnnagent import *
from .lstmagent import *
from .mlpagent import *
from .dynamicagent import *


def getAllAgentsName():
    return ["CNN", "MLP", "LSTM"]


def cloneAgent(agent: Agent):
    a: Agent = eval("%sAgent" % agent.agentType)(agent)
    a._cloneFrom(agent)
    return a


class AgentBuilder:
    _config: AgentConfiguration
    _agent: Agent
    _logger: Logger

    def __init__(self, config: AgentConfiguration, logger: Logger = None):
        assert config is not None
        self._config = config
        self._logger = logger if logger is not None else getLogger(
            "agent-builder")

    @property
    def config(self):
        return self._config

    def getAgent(self) -> Agent:
        assert self._agent is not None
        return self._agent

    def create(self):
        self._agent = eval("%sAgent(self._config)" %
                           (self._config.agentType.upper()))
        pass

    def training(self, trainset, testset, option: AgentTrainOption = None):
        assert self._agent is not None
        if option is None:
            option = AgentTrainOption()
        self._agent.train(trainset=trainset,
                          testset=testset,
                          option=option,
                          logger=self._logger)
        return self._agent.score
