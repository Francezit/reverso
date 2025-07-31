import numpy as np
import tensorflow as tf

from .nnagent import NNAgent


# https://towardsdatascience.com/time-series-forecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc
class MLPAgent(NNAgent):

    def _itemsHpvector(self, index: int):
        if index in [0, 2, 5, 7]:
            return np.arange(50, 1000, 25).tolist()
        elif index in [1, 3, 6, 8]:
            return ["relu", "linear", "elu", "tanh", "glelu", "selu"]
        elif index in [4]:
            return np.arange(0, 1, 0.1).tolist()
        else:
            raise Exception("not valid")

    def _defaultHpvector(self):
        return [3000, 'relu', 1500, 'relu', 0.5, 500, 'relu', 100, 'relu']

    def _createModel(self, shapeInput, hpvector: list):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=shapeInput),
            tf.keras.layers.Dense(hpvector[0], activation=hpvector[1]),
            tf.keras.layers.Dense(hpvector[2], activation=hpvector[3]),
            tf.keras.layers.Dropout(hpvector[4]),
            tf.keras.layers.Dense(hpvector[5], activation=hpvector[6]),
            tf.keras.layers.Dense(hpvector[7], activation=hpvector[8]),
            tf.keras.layers.Dense(1)
        ])
        return model

    def _adaptInput(self, value):
        # input => Nstep x Nentity
        return np.reshape(value, (1, value.size))
        # output => 1 x (Nstep * Nentity)

    def _prepareDataset(self, dataset):
        X, y = dataset[0], dataset[1]
        n_obs = X[0].shape[0]
        n_step = X[0].shape[1]
        n_entity = X[0].shape[2]
        n_exp = len(X)

        # flatten input
        n_input = n_entity * n_step
        X = [x.reshape((n_obs, n_input)) for x in X]

        X = np.reshape(X, [n_obs * n_exp, n_input])
        y = np.reshape(y, [n_obs * n_exp, 1])
        return X, y, [
            n_input,
        ]
