import numpy as np
from deap import base
import random as rdn
from deap import creator
from deap import tools
from deap import algorithms
import tensorflow as tf
from logging import DEBUG, Logger, getLogger

from .agent import Agent, AgentTrainOption

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)


class NNAgent(Agent):
    _model: tf.keras.Model
    _score: float
    _hpVector: list

    @property
    def score(self):
        return self._score

    @property
    def hyperparameters(self):
        return self._hpVector

    @property
    def nnmodel(self):
        return self._model

    def _createModel(self, shapeInput, hpvector: list) -> tf.keras.Model:
        pass

    def _defaultHpvector(self) -> list:
        pass

    def _itemsHpvector(self, index: int) -> list:
        pass

    def _findBestHpvector(self,
                          X_train,
                          y_train,
                          X_test,
                          y_test,
                          crossover=0.9,
                          mutation=0.5,
                          generations=2,
                          population=10,
                          logger: Logger = None):

        nSolution = len(self._defaultHpvector())
        if population % 2 != 0:
            population = population + 1

        def randomSolution():
            l = []
            for i in range(nSolution):
                v = self._itemsHpvector(i)
                index = rdn.randint(0, len(v) - 1)
                l.append(v[index])
            return l

        def objective_function(ind):
            model, loss_train = self._train_internal(X_train=X_train,
                                                     y_train=y_train,
                                                     hpvector=ind,
                                                     epochs=10)
            loss_test, _ = model.evaluate(X_test, y_test)
            if np.isnan(loss_test):
                loss_test = 10000
            del model
            return (loss_test, )

        def mutate_individual(ind):
            a = ind[:]
            idx = rdn.randint(0, nSolution - 1)
            v = self._itemsHpvector(idx)
            a[idx] = v[rdn.randint(0, len(v) - 1) % len(v)]
            return creator.Individual(a),

        def crossover_individuals(ind1, ind2):
            a1 = ind1[:]
            b1 = ind2[:]
            crossover_point = rdn.randint(1, nSolution - 1)
            for i in range(crossover_point):
                a1[i], b1[i] = b1[i], a1[i]

            return creator.Individual(a1), creator.Individual(b1)

        toolbox = base.Toolbox()
        # create an operator that randomly returns a float in the desired range:
        toolbox.register("attrFloat", randomSolution)

        # create an operator that fills up an Individual instance:
        toolbox.register("individualCreator", tools.initIterate,
                         creator.Individual, toolbox.attrFloat)

        # create an operator that generates a list of individuals:
        toolbox.register("populationCreator", tools.initRepeat, list,
                         toolbox.individualCreator)

        # fitness calculation
        toolbox.register("evaluate", objective_function)

        # genetic operators:
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", crossover_individuals)
        toolbox.register("mutate", mutate_individual)

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        pop, logbook = algorithms.eaSimple(
            population=toolbox.populationCreator(n=population),
            toolbox=toolbox,
            cxpb=crossover,
            mutpb=mutation,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=False)

        if logger is None:
            print(str(logbook))
        else:
            logger.debug(str(logbook))

        hpsolution = list(hof.items[0])
        hpsolutionFitness = hof.items[0].fitness.values[0]

        del pop, toolbox, stats, hof

        return hpsolution, hpsolutionFitness

    def _prepareDataset(self, dataset) -> tuple:
        pass

    def train(self,
              trainset,
              testset,
              option: AgentTrainOption,
              logger: Logger = None):

        X_train, y_train, _ = self._prepareDataset(trainset)
        X_test, y_test, _ = self._prepareDataset(testset)

        if option.model_optimize:
            if logger is not None:
                logger.debug(f"[{self.uniqueKey}] HPOptimizer starts")

            hpvector, fitness = self._findBestHpvector(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                crossover=option.hp_optimizer_crossover,
                mutation=option.hp_optimizer_mutation,
                generations=option.hp_optimizer_generations,
                population=option.hp_optimizer_population,
                logger=logger)

            if logger is not None:
                logger.debug(
                    f"[{self.uniqueKey}] HPOptimizer ends with fitness {str(fitness)}"
                )
        else:
            hpvector = self._defaultHpvector()

        model, _ = self._train_internal(X_train=X_train,
                                        y_train=y_train,
                                        hpvector=hpvector,
                                        epochs=option.train_epochs)

        score = self._evaluate_internal(model=model,
                                        X_test=X_test,
                                        y_test=y_test,
                                        method=option.train_evaluate_method)

        if logger is not None:
            logger.debug(
                f"[{self.uniqueKey}] trainScore={str(score)}; hp={str(hpvector)}"
            )

        self._model = model
        self._score = score
        self._hpVector = hpvector

    def evaluate(self, testset, method: str = None, logger: Logger = None):
        X_test, y_test, _ = self._prepareDataset(testset)
        score = self._evaluate_internal(self._model, X_test, y_test, method)
        if logger is not None:
            logger.debug(f"[{self.uniqueKey}] evalScore={str(score)}")

        return score

    def predict(self, input_value):
        input_value = self._adaptInput(input_value)
        input_tensor=tf.convert_to_tensor(input_value, dtype=tf.float32)
        output_tensor= self._model(input_tensor)
        output_value=output_tensor.numpy()
        return output_value

    def reset(self):
        pass

    def _cloneFrom(self, agent):
        sourceModel: tf.keras.Sequential = agent._model
        model = tf.keras.models.clone_model(sourceModel)
        model.set_weights(sourceModel.get_weights())
        self._model = model
        self._score = agent._score

    def _adaptInput(self, value):
        return value

    def _train_internal(self, X_train, y_train, hpvector, epochs):
        shapeInput = X_train[0].shape

        tf.keras.backend.clear_session()
        model = self._createModel(shapeInput, hpvector)

        model.compile(loss='mean_squared_error',
                      optimizer="adam",
                      metrics=['mean_squared_error'])

        history = model.fit(X_train, y_train, epochs=epochs, verbose=False)
        loss = history.history['loss'][-1]
        if np.isnan(loss):
            loss = np.inf
        return model, loss

    def _evaluate_internal(self, model, X_test, y_test, method: str = None):
        val = None
        if (method == None or method == "default" or method=="mse"):
            y_predicted = model(X_test)
            val = np.mean(np.square(y_test-y_predicted), axis=0).mean()
        elif (method == "pearson"):
            y_predicted = model(X_test)
            val = np.abs([np.corrcoef(np.transpose(y_test[:, i]), np.transpose(
                y_predicted[:, i])).min() for i in range(y_test.shape[1])]).mean()
        return val

    def _storeCustomData(self, obj: dict):
        obj["score"] = float(self._score)
        obj["hpVector"] = list(self._hpVector)

    def _loadCustomData(self, obj: dict):
        self._score = obj.get("score")
        self._hpVector = obj.get("hpVector")

    def _saveModel(self, modelfilename: str):
        modelfilename = modelfilename + ".hdf5"
        self._model.save(modelfilename,
                         save_format="hdf5",
                         include_optimizer=False,
                         save_traces=False)
        return modelfilename

    def _loadModel(self, modelfilename: str):
        self._model = tf.keras.models.load_model(modelfilename)
        return modelfilename

    def __str__(self) -> str:
        return f"TYPE: {self.agentType}/n SCORE: {self.score}/n HP: {str(self.hyperparameters)}/n"


class FCNNAgent(NNAgent):
    pass


class RNNAgent(NNAgent):
    pass


class GRUAgent(NNAgent):
    pass


class EncoderDecoderAgent(NNAgent):
    pass
