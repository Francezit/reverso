import matplotlib.pyplot as plt
import numpy as np


class DataExperiment:

    def __init__(self, name, labels, times, values):
        self.__values = values
        self.__labels = labels
        self.__times = times
        self.__name = name
        self.__offset = times[1] - times[0]

    @property
    def labels(self):
        return self.__labels

    @property
    def name(self):
        return self.__name

    @property
    def times(self):
        return self.__times

    @property
    def offset(self):
        return self.__offset

    @property
    def numberEntities(self) -> int:
        return np.size(self.__values, 1)

    @property
    def numberMissingValues(self) -> list:
        return [sum(np.isnan(self.__values[:, i])) for i in range(self.numberEntities)]

    @property
    def numberObservations(self) -> int:
        return np.size(self.__values, 0)

    @property
    def globalSTD(self):
        return self.__values.std()

    @property
    def globalVariance(self):
        return self.__values.var()

    @property
    def globalMean(self):
        return self.__values.mean()

    @property
    def globalMax(self):
        return self.__values.max()

    @property
    def globalMin(self):
        return self.__values.min()

    @property
    def variance(self):
        return np.var(self.__values, axis=0)

    @property
    def correlation(self):
        return np.corrcoef(np.transpose(self.__values))

    @property
    def mean(self):
        return np.mean(self.__values, axis=0)

    @property
    def std(self):
        return np.std(self.__values, axis=0)

    @property
    def max(self):
        return np.max(self.__values, axis=0)

    @property
    def min(self):
        return np.min(self.__values, axis=0)

    def getData(self, entities=-1, start=0, end=-1):
        if end < 0:
            end = self.numberObservations
        if entities is None or (isinstance(entities, int) and entities < 0):
            entities = range(0, self.numberEntities)

        X = self.__values[start:end, entities]
        return X

    def plot(self):
        plt.title(self.__name)
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.plot(self.__values)
        plt.show()

    def toString(self):
        s = ("Entities: " + str(self.numberEntities) + "\n" +
             "Observations: " + str(self.numberObservations) + "\n" +
             "Offset: " + str(self.offset) + "\n" + "Name: " + str(self.name) +
             "\n" + "STD: " + str(self.globalSTD) + "\n" + "Mean: " +
             str(self.globalMean))
        return s

    def copy(self):
        return DataExperiment(self.name, self.labels.copy(), self.times.copy(),
                              self.__values.copy())
