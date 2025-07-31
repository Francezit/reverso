import json
import os
import multiprocessing
import gc
import sys
import uuid
import numpy as np
import psutil

from datetime import date
from logging import Logger, getLogger, FileHandler, Formatter, StreamHandler
from math import floor, ceil
from cachetools import LRUCache
from zipfile import ZipFile
from sklearn.linear_model import LinearRegression

from .agents import AgentConfiguration, AgentBuilder, getAllAgentsName, AgentTrainOption, AgentCache, cloneAgent
from .core import Environment, EnvironmentMetadata, EnvironmentState, EnvironmentData, InteractionNetwork, SignalPertubationFunction
from .data import DataCollection
from .utilities import MetricsInfo, WorkspaceInfo, callFunctionBeginLogger, callFunctionEndLogger, callFunctionEndErrorLogger, traceFunctionLogger
from .reversoconfig import *


class Reverso:
    _workspace: WorkspaceInfo
    _logger: Logger
    _envs: dict
    _globalTempId: int

    def __init__(self, option: WorkspaceInfo, logger: Logger = None):
        self._workspace = option
        self._envs = {}
        self._globalTempId = self._workspace.get_temp_handler().createSession()

        if logger is None:

            self._logger = getLogger(f"R{option.id_session}")
            self._logger.setLevel(option.__logLevel)

            logFilename = option.get_complete_path("log",
                                                 f"{option.id_session}.txt")
            handler = FileHandler(logFilename)
            formatter = Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

            handler = StreamHandler(sys.stdout)
            formatter = Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        else:
            self._logger = logger

    @staticmethod
    def version():
        return 1.1

    @staticmethod
    def fileType():
        return ".rvs"

    def makeEnvironment(self, option: MakeEnvironmentConfiguration,
                        data: DataCollection) -> Environment:
        """Create a new environment"""

        nentities = data.num_entities

        logId = callFunctionBeginLogger("MakeEnvironment",
                                        self._logger.info,
                                        arg=data.folder,
                                        option=option)

        # prepare arguments for reach call to target function
        tasksResults = []
        taskArgs = [(option.getAgentConfiguration(i), data, option.trainOption)
                    for i in range(nentities)]

        # process
        if option.useParallelProcessing:
            with multiprocessing.Pool(
                    processes=option.parallelProcesses) as pool:
                # call the function for each item in parallel with multiple arguments
                chunksize = ceil(len(taskArgs) / option.parallelProcesses)
                tasksResults = pool.map(self._makeAgentTask, taskArgs,
                                        chunksize)
        else:
            tasksResults = [
                self._makeAgentTask(taskArg) for taskArg in taskArgs
            ]

        # combine agent
        agentScores = []
        agentList = []
        for result in tasksResults:
            agentScores.append(result[1])
            agentList.append(result[0])

        # create agent
        env = self._createEnvironment(agentList=agentList,
                                      trainOption=option.trainOption,
                                      data=data)

        callFunctionEndLogger(logId, self._logger.info)
        return env

    def _createEnvironment(self, agentList: list,
                           trainOption: AgentTrainOption,
                           data: DataCollection):

        metadata = EnvironmentMetadata()
        metadata.componentNames = data.labels
        metadata.datasetName = data.folder
        metadata.testExperiments = trainOption.testExperiments
        metadata.trainingExperiments = trainOption.trainingExperiments
        metadata.creationDate = date.today().strftime("%d/%m/%Y %H:%M:%S")
        metadata.version = Reverso.version()

        env = self._allocateEnvironment(agents=agentList, metadata=metadata)
        return env

    def _makeAgentTask(self, taskData: tuple):
        agentConfig: AgentConfiguration = taskData[0]
        data: DataCollection = taskData[1]
        trainOption: AgentTrainOption = taskData[2]

        if agentConfig.autoSetNStep:
            agentConfig.n_steps = ceil(min(data.num_observations) / 5)

        result = None
        logId = callFunctionBeginLogger(
            "MakeAgent", self._logger.info,
            f"{agentConfig.agentType} {str(agentConfig.index)}")

        try:
            trainset = data.prepare_dataset(
                entityIndex=agentConfig.index,
                experimentIndexs=trainOption.trainingExperiments,
                n_steps=agentConfig.n_steps,
                forecastingMode=agentConfig.forecastingMode)
            testset = data.prepare_dataset(
                entityIndex=agentConfig.index,
                experimentIndexs=trainOption.testExperiments,
                n_steps=agentConfig.n_steps,
                forecastingMode=agentConfig.forecastingMode)

            builder = AgentBuilder(agentConfig, self._logger)
            builder.create()
            score = builder.training(trainset, testset, trainOption)
            result = builder.getAgent(), score

            callFunctionEndLogger(logId, self._logger.info)
        except Exception as err:
            callFunctionEndErrorLogger(logId, str(err), self._logger.exception)
        return result

    def automakeEnvironment(self, option: AutomakeEnvironmentConfiguration,
                            data: DataCollection):
        """Create a new environment"""

        logId = callFunctionBeginLogger("AutomakeEnvironment",
                                        self._logger.info,
                                        arg=data.folder,
                                        option=option)

        if option is None:
            option = AutomakeEnvironmentConfiguration()

        # define temp folder
        backupFolder = self._workspace.get_folder("backup")
        temp = self._workspace.get_temp_handler()
        tempId = temp.createSession()

        # define temp cache
        agentCache = AgentCache(
            foldername=temp.generatePathname(tempId),
            storageCacheCapacityMB=option.agentCacheStorageCapacityMB,
            memoryCacheCapacityMB=option.agentCacheMemoryCapacityMB,
            logger=self._logger.getChild("agentCache"))
        solutionCache = LRUCache(maxsize=option.optimizePopulationSize * 2)

        # define variables
        lenSolution = data.num_entities
        agentTypes = getAllAgentsName()
        bestEnvScore: MetricsInfo = None
        bestEnvFilename: str = None

        def objective(solution, pool):
            solution = [agentTypes[int(i)] for i in np.round(solution)]

            logId = callFunctionBeginLogger("AutomakeEnvironment.Objective",
                                            self._logger.info,
                                            f"solution= {str(solution)}")

            # if it is in cache its fitness is known
            solutionKey = str(solution)
            errorValue = solutionCache.get(solutionKey)
            if errorValue is None:

                nonlocal bestEnvScore
                nonlocal bestEnvFilename

                # compute agent arguments and train it
                taskArgs = []
                for i in range(lenSolution):
                    key = f"{solution[i]}_{str(i)}"
                    if not agentCache.setReservation(key, True):
                        agentConfig = option.baseAgentConfig.cloneConfig()
                        agentConfig.agentType = solution[i]
                        agentConfig.index = i
                        taskArgs.append(
                            (agentConfig, data, option.trainOption))

                # train agent
                if pool is not None:
                    chunksize = max(
                        ceil(len(taskArgs) / option.parallelProcesses), 2)
                    for res in pool.imap_unordered(self._makeAgentTask,
                                                   taskArgs, chunksize):
                        agent = res[0]
                        agentCache.add(key=agent.uniqueKey,
                                       agent=agent,
                                       setReservation=True)
                        print(f"Add {agent.uniqueKey}")
                        del agent
                        pass
                else:
                    for taskArg in taskArgs:
                        agent = self._makeAgentTask(taskArg)[0]
                        agentCache.add(key=agent.uniqueKey,
                                       agent=agent,
                                       setReservation=True)
                        del agent
                        pass

                del taskArgs

                # create environent
                env = self._createEnvironment(
                    agentList=agentCache.getReservedAgents(True),
                    trainOption=option.trainOption,
                    data=data)

                # evaluate
                metric: MetricsInfo = self.evaluateEnvironment(env=env,
                                                               data=data,
                                                               option=option.evalOption,
                                                               isSubrutine=True)
                errorValue = metric.getValue()

                # set best env
                storeEnv = False
                if bestEnvScore is None:
                    bestEnvScore = metric
                    storeEnv = True
                elif metric.betterThan(bestEnvScore):
                    bestEnvScore = metric
                    storeEnv = True
                else:
                    del metric

                if storeEnv:
                    bestEnvFilename = os.path.join(backupFolder,
                                                   "automake_best_env.rvs")
                    self.saveEnvironment(env=env,
                                         filename=bestEnvFilename)
                    traceFunctionLogger(logId, "Backup best environment",
                                        self._logger.info)

                self._dealocateEnvironment(env)
                solutionCache[solutionKey] = errorValue

            callFunctionEndLogger(logId, self._logger.info)
            gc.collect()
            return errorValue

        def gaoptimizer(pool=None):
            from scipy.optimize import differential_evolution
            result = differential_evolution(
                func=lambda x: objective(x, pool),
                bounds=[(0, len(agentTypes) - 1)] * lenSolution,
                strategy='best1bin',
                maxiter=option.optimizeMaxIterations,
                popsize=option.optimizePopulationSize,
                mutation=option.optimizeProbMutation,
                recombination=option.optimizeProbCrossover,
                integrality=[1] * lenSolution,
                disp=option.optimizeDisplay)
            pass

        if option.useParallelProcessing:
            with multiprocessing.Pool(
                    processes=option.parallelProcesses) as taskPool:
                gaoptimizer(taskPool)
        else:
            gaoptimizer()

        del agentCache
        del solutionCache
        temp.removeSession(tempId)

        bestEnv = self.loadEnvironment(bestEnvFilename)
        os.remove(bestEnvFilename)

        callFunctionEndLogger(logId, self._logger.info)
        return bestEnv, bestEnvScore

    def optimizeEnvironment(self, env: Environment, data: DataCollection):
        # ottimizza l'env creato, trova i coefficienti ottimali, alg genetico
        env = self._checkEnvironment(env)
        logId = callFunctionBeginLogger("OptimizeEnvironment",
                                        self._logger.info)

        callFunctionEndLogger(logId, self._logger.info)
        pass

    def evaluateEnvironment(self, env: Environment, data: DataCollection,
                            option: EnvironmentEvaluationOption, isSubrutine: bool = False):

        env = self._checkEnvironment(env)
        logId = callFunctionBeginLogger("EvaluateEnvironment",
                                        self._logger.info,
                                        arg=str(env.agentTypes),
                                        option=option,
                                        includeOption=not isSubrutine)

        if option.evaluationMode is None or option.evaluationMode == "simulation":
            targetData = data.get_average_experiment(
                experimentIndexs=option.experimentIndexs)

            agentStep = env.agentMaxSteps
            initState = targetData[0:agentStep, :]
            nobs = targetData.shape[0]
            maxStep = int(agentStep +
                          floor((nobs - agentStep) * option.includeAll))
            assert maxStep > agentStep

        elif option.evaluationMode == "ts_forecasting":
            obsData = data.concant_experiments(
                experimentIndexs=option.experimentIndexs)

            agentStep = env.agentMaxSteps
            nobs = obsData.shape[0]
            splitIndex = int(
                floor((nobs - agentStep * 2) * option.forecastPercentage))
            assert splitIndex > agentStep

            targetData = obsData[splitIndex:, :]
            initState = obsData[(splitIndex - agentStep):splitIndex, :]
            maxStep = int(agentStep +
                          floor((nobs - splitIndex) * option.includeAll))
            assert maxStep > agentStep

        def testContext(e: Environment):
            e.disablePertubations()
            e.resetState(initState=initState, steps=agentStep)
            e.runUntil(absoluteTick=maxStep - 1)
            return e.getHistoryState()

        if option.useSandbox:
            predictedData = env.sandboxContext(testContext)
        else:
            predictedData = testContext(env)

        metricInfo = MetricsInfo(
            targetData=targetData[:predictedData.shape[0]],
            predictedData=predictedData,
            defaultMetric=option.defaultMetric,
            stepToStart=agentStep if option.excludeInitialValue else None)

        callFunctionEndLogger(logId, self._logger.info)
        return metricInfo

    def extractInteractionNetwork(
            self, env: Environment,
            option: ExtractInteractionNetworkOption) -> InteractionNetwork:

        env = self._checkEnvironment(env)
        logId = callFunctionBeginLogger("ExtractInteractionNetwork",
                                        self._logger.info,
                                        option=option)

        defaultPertubationOption = {"duration": env.agentMaxSteps}

        def processContext(e: Environment):
            # find stability
            e.resetState()
            if option.stabilityEnabled:
                e.runUntilStability(step=option.stabilityWindow,
                                    delta=option.stabilityThreshold,
                                    increaseFactor=option.stabilityThresholdFactor)

            startTick = e.currentTick
            marginTicks = e.agentMaxSteps
            e.nextState(marginTicks)

            # regulatory matrix
            data = np.zeros((e.numberComponent, e.numberComponent))
            for entityIndex in range(e.numberComponent):
                e.aquireContext()

                traceFunctionLogger(
                    logId,
                    f"Start simulation entity: {entityIndex}, mem={psutil.virtual_memory().available/1024/1024}MB",
                    self._logger.info)

                # pertubations
                startPertubationTick = e.currentTick
                pertubationFunction: SignalPertubationFunction = SignalPertubationFunction.makePertubationFunction(
                    name=option.pertubationType,
                    config=option.pertubationConfig,
                    defaultConfig=defaultPertubationOption)
                e.createPertubationFunction(
                    entityIndex=entityIndex,
                    pertubationFunction=pertubationFunction)
                e.nextState(pertubationFunction.duration)
                endPertubationTick = e.currentTick

                e.nextState(marginTicks)
                endTick = e.currentTick

                # plot snapshot
                if option.extractImages:
                    if option.extractImageFolder is None:
                        figFilename = self._workspace.get_complete_path(
                            "outputs",
                            ["network", f"pertubations_gene{entityIndex}.svg"])
                    else:
                        figFilename = os.path.join(
                            option.extractImageFolder,
                            f"pertubations_gene{entityIndex}.svg")

                    e.nextState(option.extractImagesMargin)
                    e.analysis().saveSnapshotImage(figFilename,
                                                   ranges=[
                                                       startTick - option.extractImagesMargin, e.currentTick],
                                                   enable_bounds=option.extractImagesYLimit,
                                                   useYLimit=True)

                # get data
                responseData = e.getHistoryState(startTick=startTick,
                                                 endTick=endTick)

                # process result
                for i in range(e.numberComponent):
                    if entityIndex != i:
                        # rappresenta di quanto una pertubazione del gene entityIndex
                        # implica una pertubazione dell'i esimo gene. Il valore rappresenta quindi il coefficiente angolare di questa variazione,
                        #  piu è estremo piu è presente una correlazione
                        coeff = LinearRegression().fit(
                            responseData[:, entityIndex].reshape(-1, 1),
                            responseData[:, i].reshape(-1, 1)).coef_[0][0]
                    else:
                        dt1 = e.getHistoryState(startTick=startTick,
                                                endTick=startPertubationTick)
                        dt2 = e.getHistoryState(startTick=endPertubationTick,
                                                endTick=endTick)

                        deltaY = dt2[:, entityIndex].mean(
                        ) - dt1[:, entityIndex].mean()
                        deltaX = endPertubationTick - startPertubationTick
                        coeff = deltaY / deltaX

                        del dt1, dt2, deltaX, deltaY
                    data[entityIndex, i] = np.arctan(coeff)
                    del coeff

                # clear data
                e.rollbackContext()
                del responseData
                gc.collect()

                traceFunctionLogger(
                    logId,
                    f"End simulation entity: {entityIndex}, mem={psutil.virtual_memory().available/1024/1024}MB",
                    self._logger.info)
                pass

            return np.matrix(data)

        regMatrix = env.sandboxContext(processContext)
        net = InteractionNetwork(interactionsMatrix=regMatrix,
                                 thresholdValues=option.thresholdValues,
                                 thresholdMethod=option.thresholdMethod,
                                 nodeNames=env.componentNames)

        callFunctionEndLogger(logId, self._logger.info)
        return net

    def getEnvironment(self, envId: str) -> Environment:
        logId = callFunctionBeginLogger("GetEnvironment", self._logger.info)
        env = self._envs[envId]
        callFunctionEndLogger(logId, self._logger.info)
        return env

    def cloneEnvironment(self, env: Environment):
        env = self._checkEnvironment(env)
        metadata = env._metadata.cloneConfig(
        ) if env._metadata is not None else None

        envdata = env._envdata.cloneConfig(
        ) if env._envdata is not None else None

        agents = []
        for a in env._agents:
            agents.append(cloneAgent(a))

        return self._allocateEnvironment(agents=agents,
                                         metadata=metadata,
                                         envdata=envdata)

    def loadEnvironment(self, filename: str) -> Environment:
        """Load an environment from file"""
        logId = callFunctionBeginLogger("LoadEnvironment", self._logger.info,
                                        filename)

        temp = self._workspace.get_temp_handler()
        tempId = temp.createSession()

        def zread(f: ZipFile, name: str) -> dict:
            try:
                zinfo = f.NameToInfo.get(name)
                if zinfo is not None:
                    s = str(f.read(zinfo), encoding="utf-8")
                    return json.loads(s)
            except Exception as zerr:
                self._logger.error(
                    f"Error during zread in '{name}': {str(zerr)}")
            return {}

        with ZipFile(filename, mode='r') as file:
            # state = EnvironmentState(zread(file, "state"))
            metadata = EnvironmentMetadata(zread(file, "metadata"))
            envdata = EnvironmentData(zread(file, "envdata"))
            agentinfo = zread(file, "agentinfo")
            agents = []
            for item in agentinfo:
                modelname = item["model"]
                metaname = item["metadata"]
                index = item["index"]
                config = item["config"]

                pname = temp.generatePathname(tempId)
                fname1 = os.path.join(pname, "agents",
                                      "a" + str(index) + "_model")
                fname2 = os.path.join(pname, "agents",
                                      "a" + str(index) + "_data")
                file.extract(modelname, pname)
                file.extract(metaname, pname)

                agentConfig = AgentConfiguration(config)
                agentBuilder = AgentBuilder(agentConfig)
                agentBuilder.create()
                agent = agentBuilder.getAgent()
                agent.load(fname1, fname2)
                agents.append(agent)

        env = self._allocateEnvironment(agents=agents,
                                        metadata=metadata,
                                        envdata=envdata)

        temp.removeSession(tempId)

        callFunctionEndLogger(logId, self._logger.info)
        return env

    def saveEnvironment(self, env: Environment, filename: str):
        """Save an environment"""
        env = self._checkEnvironment(env)
        logId = callFunctionBeginLogger("SaveEnvironment", self._logger.info,
                                        filename)

        temp = self._workspace.get_temp_handler()
        tempId = temp.createSession()

        targetFilename = None
        if os.path.exists(filename):
            targetFilename = filename
            filename = filename + "_temp"

        try:

            def zwrite(f: ZipFile, name: str, data: dict):
                try:
                    data = bytes(json.dumps(data), encoding="utf-8")
                    f.writestr(name, data)
                except Exception as zerr:
                    self._logger.error(
                        f"Error during zwrite in '{name}': {str(zerr)}")

            with ZipFile(filename, mode='w') as file:
                # zwrite(file, "state",
                #       env._state.to_dict() if env._state is not None else {})
                zwrite(
                    file, "metadata",
                    env._metadata.to_dict()
                    if env._metadata is not None else {})
                zwrite(
                    file, "envdata",
                    env._envdata.to_dict() if env._envdata is not None else {})
                zwrite(file, "rvsapp", {"reverso_version": Reverso.version()})

                agentInfo = []
                for agent in env._agents:
                    fname1 = temp.generateFilename(tempId)
                    fname2 = temp.generateFilename(tempId)
                    fname1, fname2 = agent.save(modelfilename=fname1,
                                                metadatafilename=fname2)

                    modelname = "agents/" + "a" + str(agent.index) + "_model"
                    metaname = "agents/" + "a" + str(agent.index) + "_data"

                    file.write(fname1, modelname)
                    file.write(fname2, metaname)

                    agentInfo.append({
                        "model": modelname,
                        "metadata": metaname,
                        "index": agent.index,
                        "config": agent.getConfiguration().to_dict()
                    })
                zwrite(file, "agentinfo", agentInfo)

            if targetFilename is not None:
                os.remove(targetFilename)
                os.rename(filename, targetFilename)
            callFunctionEndLogger(logId, self._logger.info)
        except Exception as ex:
            os.remove(filename)
            callFunctionEndErrorLogger(logId, str(ex), self._logger.exception)
            raise ex
        finally:
            temp.removeSession(tempId)

    def disposeAllEnvironment(self):
        logId = callFunctionBeginLogger("DisposeAllEnvironment",
                                        self._logger.info)
        self._dealocateAllEnvironments()
        callFunctionEndLogger(logId, self._logger.info)

    def disposeEnvironment(self, env: Environment):
        env = self._checkEnvironment(env)
        logId = callFunctionBeginLogger("DisposeEnvironment",
                                        self._logger.info)
        self._dealocateEnvironment(env)
        callFunctionEndLogger(logId, self._logger.info)

    def _checkEnvironment(self, env: Environment):
        if isinstance(env, Environment):
            return env
        elif isinstance(env, str):
            return self._envs[env]
        else:
            raise Exception("Env not found")

    def _allocateEnvironment(self,
                             agents: list,
                             metadata: EnvironmentMetadata = None,
                             envdata: EnvironmentData = None):
        envId = str(uuid.uuid4())

        env = Environment(
            agents=agents,
            metadata=metadata,
            envdata=envdata,
            cacheFolder=self._workspace.get_temp_handler().generatePathname(
                self._globalTempId))

        env.envId = envId
        self._envs[envId] = env
        return env

    def _dealocateAllEnvironments(self):
        keys = list(self._envs.keys())
        for envId in keys:
            env = self._envs[envId]
            self._dealocateEnvironment(env)
        gc.collect()

    def _dealocateEnvironment(self, env: Environment):
        envId = env.envId
        if env._agents:
            del env._agents
        if env._metadata:
            del env._metadata
        if env._state:
            del env._state
        if env._envdata:
            del env._envdata
        del env
        del self._envs[envId]
        gc.collect()

    def dispose(self):
        self._dealocateAllEnvironments()
        self._workspace.get_temp_handler().removeAllSession()
        del self._envs
        gc.collect()
