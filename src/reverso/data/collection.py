import os
import numpy as np
import pandas as pds

from .experiments import DataExperiment
from ..utilities import getListOfFiles


class DataCollection:
    __collection: list
    __names: list
    __entities: int
    __folder: str

    def __init__(self,
                 experiments: list,
                 names: list,
                 folder: str = None) -> None:
        if isinstance(experiments, DataExperiment):
            experiments = [experiments]
            names = [names]

        n = -1
        for exp in experiments:
            if n < 0:
                n = exp.numberEntities
            assert n == exp.numberEntities
        self.__collection = experiments
        self.__names = names
        self.__entities = n
        self.__folder = folder

    def __len__(self):
        return self.__collection.__len__()

    @property
    def names(self):
        return self.__names

    @property
    def folder(self):
        return self.__folder

    @property
    def labels(self) -> list:
        return self.__collection[0].labels

    @property
    def num_observations(self):
        return [x.numberObservations for x in self.__collection]

    @property
    def num_entities(self) -> int:
        return self.__entities

    @property
    def num_missing_values(self) -> int:
        return sum([sum(x.numberMissingValues) for x in self.__collection])

    @property
    def has_missing_values(self) -> bool:
        for x in self.__collection:
            if sum(x.numberMissingValues) > 0:
                return True
        return False

    @property
    def length(self):
        return self.__collection.__len__()

    def at(self, index) -> DataExperiment:
        return self.__collection[index]

    def get_experiment(self, name: any) -> DataExperiment:
        if isinstance(name, int):
            idx = name
        elif isinstance(name, str):
            idx = self.__names.index(name)
        else:
            raise Exception("Not supported")
        return self.__collection[idx]

    def concant_experiments(self,
                           entities: int = -1,
                           experimentIndexs: list = None) -> np.float64:
        if experimentIndexs == None:
            experimentIndexs = list(range(0, self.length))

        data = self.at(experimentIndexs[0]).getData(entities)
        for i in range(1, len(experimentIndexs)):
            data = np.concatenate(
                (data, self.at(experimentIndexs[i]).getData(entities)))
        return data

    def __ranking(self,
                 data: np.float64,
                 entityIndex: int,
                 method: str = "linear_regression",
                 useNormalization: bool = True) -> np.ndarray:
        n = data.shape[1]
        X = data[:, :]
        Y = data[:, entityIndex]

        scores = self.__correlation(method, useNormalization, n, X, Y)
        if len(scores) < n:
            s = np.ones((n,))
            mask = np.arange(n)
            mask = np.delete(mask, entityIndex)
            s[mask] = scores
            scores = s

        return scores

    def __correlation(self, method: str, useNormalization: bool, n, X: np.ndarray, Y: np.ndarray):
        if method == "spearman":
            from scipy import stats
            r = stats.spearmanr(X, Y).correlation
            v = abs(r[0, 0:n])
            score = np.array(v) / np.sum(v) if useNormalization else np.array(
                v)
        elif method == None or method == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, Y)
            v = abs(model.coef_)
            score = np.array(v) / np.sum(v) if useNormalization else np.array(
                v)
        elif method == "mrmr":
            from mrmr import mrmr_regression
            v = mrmr_regression(pds.DataFrame(X),
                                pds.DataFrame(Y),
                                n,
                                return_scores=True,
                                show_progress=False)
            score = np.array(v[1]) / np.sum(
                v[1]) if useNormalization else np.array(v[1])

        return score

    def get_entity_importance_methods(self):
        return ["spearman", "linear_regression", "mrmr"]

    def get_entity_importance_ranking(self,
                                   experimentIndexs: list = None,
                                   entityIndex: int = None,
                                   method: str = None) -> np.ndarray:
        data = self.concant_experiments(experimentIndexs=experimentIndexs)
        if entityIndex is not None:
            return self.__ranking(data=data,
                                 entityIndex=entityIndex,
                                 method=method)
        else:
            n = self.__entities
            res = np.zeros([n, n])
            for i in range(n):
                res[i, :] = self.__ranking(data=data,
                                          entityIndex=i,
                                          method=method)
            return res

    def copy(self):
        return DataCollection([exp.copy() for exp in self.__collection],
                              self.__names.copy(), self.folder)

    def filter(self,
               entities: list = None,
               experiments: list = None,
               query: dict = None):

        # preliminary check
        if query is not None:
            if isinstance(query, dict):
                keys = query.keys()
                if "entities" in keys:
                    entities = query["entities"]
                elif "exclude_entities" in keys:
                    pass

                if "experiments" in keys:
                    experiments = query["experiments"]
                elif "exclude_experiments" in keys:
                    pass
            elif isinstance(query, list):
                entities = query
            else:
                raise Exception("Query format not valid")

        if entities is None and experiments is None:
            return self.copy()
        elif entities is None and experiments is not None:
            assert len(experiments) > 0
            entities = list(range(self.__entities))
        elif entities is not None and experiments is None:
            assert len(entities) > 0
            experiments = list(range(len(self.__collection)))
        else:
            assert len(experiments) > 0
            assert len(entities) > 0

        # convert lables and entities in the right type for entities
        if isinstance(entities[0], str):
            l = self.labels
            labels = entities.copy()
            entities = [l.index(v) for v in entities]
        elif isinstance(entities[0], int):
            l = self.labels
            labels = [l[i] for i in entities]
        else:
            raise Exception("Format not supported for entities")

        # convert lables and entities in the right type for entities
        if isinstance(experiments[0], str):
            l = self.__collection
            n = self.__names
            names = experiments.copy()
            collection = [l[n.index(v)] for v in experiments]
        elif isinstance(experiments[0], int):
            l = self.__collection
            n = self.__names
            collection = [l[i] for i in experiments]
            names = [n[i] for i in experiments]
        else:
            raise Exception("Format not supported for experiments")

        exps = [
            DataExperiment(v.name, labels, v.times.copy(),
                           v.getData(entities=entities)) for v in collection
        ]
        return DataCollection(exps, names, self.folder)

    def solve_missing_value(self, default_value: float = 0):
        exps = [
            DataExperiment(v.name, v.labels.copy(), v.times.copy(),
                           np.nan_to_num(v.getData(), nan=default_value))
            for v in self.__collection
        ]
        return DataCollection(exps, self.__names.copy(), self.folder)

    def reverse(self,
                data,
                transformMethods: list = None,
                reverseMethods: bool = True):
        dt = self.concant_experiments()
        v_min = np.min(dt, axis=0)
        v_max = np.max(dt, axis=0)
        v_mean = np.mean(dt, axis=0)
        v_std = np.std(dt, axis=0)
        del dt

        if isinstance(transformMethods, str):
            transformMethods = [transformMethods]

        if reverseMethods:
            transformMethods = transformMethods.copy()
            transformMethods.reverse()

        def revtransf(temp_data):
            for method in transformMethods:
                if method is None or method == "minmaxscaling":
                    temp_data = temp_data * (v_max - v_min) + v_min
                elif method == "standardization":
                    temp_data = temp_data * v_std + v_mean
            return temp_data

        if isinstance(data, DataCollection):
            exps = [
                DataExperiment(v.name, v.labels.copy(), v.times.copy(),
                               revtransf(v.getData()))
                for v in data.__collection
            ]
            return DataCollection(exps, data.__names.copy(), data.folder)
        elif isinstance(data, DataExperiment):
            return DataExperiment(data.name, data.labels.copy(),
                                  data.times.copy(), revtransf(data.getData()))
        else:
            return revtransf(data)

    def transform(self, method: str = None):
        if isinstance(method, list):
            outdata = self
            for k in method:
                outdata = outdata.transform(k)
            return outdata
        else:
            dt = self.concant_experiments()
            if method is None or method == "minmaxscaling":
                m = np.min(dt, axis=0)
                M = np.max(dt, axis=0)
                exps = [
                    DataExperiment(v.name, v.labels.copy(), v.times.copy(),
                                   (v.getData() - m) / (M - m))
                    for v in self.__collection
                ]
            elif method == "standardization":
                m = np.mean(dt, axis=0)
                s = np.std(dt, axis=0)
                exps = [
                    DataExperiment(v.name, v.labels.copy(), v.times.copy(),
                                   (v.getData() - m) / s)
                    for v in self.__collection
                ]
            elif method == "abs":
                exps = [
                    DataExperiment(v.name, v.labels.copy(), v.times.copy(),
                                   np.abs(v.getData()))
                    for v in self.__collection
                ]
            elif method == "stdnotzero":
                exps = []
                for exp in self.__collection:
                    expdata = exp.getData()
                    stdvector = exp.std
                    for i in range(exp.numberEntities):
                        if stdvector[i] == 0:
                            expdata[-1, i] = expdata[-1, i] + 1e-15
                    exps.append(
                        DataExperiment(exp.name, exp.labels.copy(),
                                       exp.times.copy(), expdata))
            return DataCollection(exps, self.__names.copy(), self.folder)

    def get_entity_limits(self, experimentIndexs: list = None, useAverage=False):
        if experimentIndexs == None:
            experimentIndexs = list(range(0, self.length))

        if useAverage:
            maxLimit = np.mean([self.at(i).max for i in experimentIndexs], 0)
            minLimit = np.mean([self.at(i).min for i in experimentIndexs], 0)
        else:
            maxLimit = np.max([self.at(i).max for i in experimentIndexs], 0)
            minLimit = np.min([self.at(i).min for i in experimentIndexs], 0)
        return minLimit, maxLimit

    def get_average_experiment(self,
                             entities: int = -1,
                             experimentIndexs: list = None) -> np.float64:
        if experimentIndexs == None:
            experimentIndexs = list(range(0, self.length))

        data = self.at(experimentIndexs[0]).getData(entities)
        if len(experimentIndexs) > 1:
            for i in range(1, len(experimentIndexs)):
                data = data + self.at(experimentIndexs[i]).getData(entities)
            data = data / len(experimentIndexs)
        return data

    # split a univariate sequence into samples
    def __split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix, 0:-1], sequence[end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    # split a multivariate sequence into samples
    def __split_multivariate_sequence(self, sequence, n_steps):
        X, y = list(), list()
        n = int(np.floor(sequence.shape[1] / 2))
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix, 0:n], sequence[end_ix, n:]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def prepare_multivariate_dataset(self,
                                   experimentIndexs: list = None,
                                   n_steps: int = 3):
        if experimentIndexs is None:
            experimentIndexs = range(self.length)
        elif isinstance(experimentIndexs, int):
            experimentIndexs = [experimentIndexs]

        n = self.num_entities
        X, Y = [], []

        for expIdx in experimentIndexs:
            exp = self.at(expIdx)

            # define input sequence
            Xdata = exp.getData()
            in_seq = Xdata
            out_seq = exp.getData()

            # convert to [rows, columns] structure
            in_seq = in_seq.reshape((len(in_seq), n))
            out_seq = out_seq.reshape((len(out_seq), n))

            # horizontally stack columns
            dataset = np.hstack((in_seq, out_seq))

            # convert into input/output
            x, y = self.__split_multivariate_sequence(dataset, n_steps)

            X.append(x)
            Y.append(y)

        return X, Y

    def prepare_dataset(self,
                       entityIndex: int,
                       experimentIndexs: list = None,
                       n_steps: int = 3,
                       forecastingMode: int = 0):
        """
        forecastingMode definisce la modalità da utilizzare per prepare il dataset: 
            0 esclude il valore dell'entità dal dataset finale 
            1 non esclude questo valore
        """

        if experimentIndexs is None:
            experimentIndexs = range(self.length)

        n = self.num_entities

        mask = np.ones([
            self.num_entities,
        ], dtype=np.bool_)
        if forecastingMode == 0:
            mask[entityIndex] = 0
            n = n - 1
        Xentity = np.array(range(self.num_entities))[mask]

        X, Y = [], []

        for expIdx in experimentIndexs:
            exp = self.at(expIdx)

            # define input sequence
            Xdata = exp.getData(Xentity)
            in_seq = Xdata
            out_seq = exp.getData(entityIndex)

            # convert to [rows, columns] structure
            in_seq = in_seq.reshape((len(in_seq), n))
            out_seq = out_seq.reshape((len(out_seq), 1))

            # horizontally stack columns
            dataset = np.hstack((in_seq, out_seq))

            # convert into input/output
            x, y = self.__split_sequence(dataset, n_steps)

            X.append(x)
            Y.append(y)

        return X, Y


def read_data(filename: str,
             sep='\t',
             selectFirstItem: bool = False,
             splitCriteria: any = None):

    def transformConverterFn(value):
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)

    transformConverter = np.vectorize(transformConverterFn)

    datatxt = pds.read_csv(filename, sep=sep)
    basename = os.path.basename(filename).split('.')[0]
    labels = datatxt.columns.values[1:].tolist()

    datalist = []
    datanamelist = []

    if splitCriteria is None:
        nexps, = np.where(datatxt.values[:, 0] == 0)
    elif isinstance(splitCriteria, int):
        nexps, = np.where(datatxt.values[:, 0] == splitCriteria)
    elif isinstance(splitCriteria, float):
        nlen = int(np.floor(datatxt.shape[0] * splitCriteria))
        nexps = np.array([0, nlen], dtype=np.int64)
    elif isinstance(splitCriteria, list) or isinstance(splitCriteria,
                                                       np.ndarray):
        nexps = np.array([0] + splitCriteria, dtype=np.int64)
    else:
        raise Exception("SplitCriteria not supported")

    if len(nexps) > 0:
        i = 0
        ends = False
        while not ends:
            if i < len(nexps) - 1:
                times = datatxt.values[nexps[i]:nexps[i + 1], 0]
                values = datatxt.values[nexps[i]:nexps[i + 1], 1:]
            else:
                times = datatxt.values[nexps[i]:, 0]
                values = datatxt.values[nexps[i]:, 1:]
                ends = True
            times = transformConverter(times)
            values = transformConverter(values)

            i = i + 1
            dataname = basename + "_exp_" + str(i)
            data = DataExperiment(dataname, labels, times, values)

            datalist.append(data)
            datanamelist.append(dataname)
    else:
        times = datatxt.values[:, 0]
        values = datatxt.values[:, 1:]

        times = transformConverter(times)
        values = transformConverter(values)

        data = DataExperiment(basename, labels, times, values)
        datalist.append(data)
        datanamelist.append(basename)

    if selectFirstItem:
        return datalist[0], datanamelist[0]
    else:
        return datalist, datanamelist


def read_datasets(folder: str,
                 onlyTimeSerieFormat: bool = True,
                 sep='\t') -> DataCollection:
    files = getListOfFiles(folder)
    data = []
    names = []
    for fullname in files:
        try:
            if not onlyTimeSerieFormat or fullname.find("timeseries") >= 0:
                dts, nms = read_data(fullname, sep=sep)
                for dt in dts:
                    data.append(dt)
                for nm in nms:
                    names.append(nm)
        except:
            print("Error to load " + fullname)
    return DataCollection(data, names, folder)


def write_dataset(filename: str, data: DataCollection):
    pass
