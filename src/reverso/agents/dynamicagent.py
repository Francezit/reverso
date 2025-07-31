import numpy as np
from logging import Logger

import automatedML as autoML

from .agent import Agent, AgentTrainOption


class DynamicAgent(Agent):
    __model: autoML.ann.ANNModel
    __score: float

    def __init__(self, config):
        super().__init__(config)

    @property
    def score(self) -> float:
        return self.__score

    def predict(self, inputs: np.ndarray) -> np.double:
        return self.__model.predict(inputs)

    def reset(self):
        pass

    def train(self, trainset, testset, option: AgentTrainOption, logger: Logger = None):

        # load datasets
        data = autoML.init_data_container(
            X_train=trainset[0], y_train=trainset[1],
            X_test=testset[0], y_test=testset[1],
            type_of_task=autoML.TypeOfTask.TS_FORECASTING
        )

        # set settings for flat generator
        setting = autoML.flatgenerator.FlatGeneratorSettings()
        setting.layer_classes = [
            "FullyConnectedWithActivation",
            "BatchNormalization",
            "Recurrent",
            "Convolutional",
            "Pooling",
            "Dropout"
        ]
        setting.alg_max_episodes = option.hp_optimizer_generations
        setting.dyann_max_deph_size = 10
        setting.hp_optimization_alg = "ls"
        setting.alg_hp_optimization_prob = 1 if option.model_optimize else 0
        setting.train_settings.epochs = option.train_epochs
        setting.train_settings.loss_function_name = "mean_squared_error"
        setting.train_settings.optimizer = "adam"

        generator = autoML.flatgenerator.FlatGenerator(
            data=data,
            settings=setting,
            logger=logger
        )
        model, metric = generator.generate_iteratively()
        self.__model = model
        self.__score = metric.other_metrics[0]


    def evaluate(self, testset, method: str, logger: Logger = None):
        pass

    def _saveModel(self, modelfilename: str):
        modelfilename = modelfilename + ".zip"
        self.__model.save(modelfilename,)

    def _loadModel(self, modelfilename: str):
        pass