import numpy as np
import tensorflow as tf

from .nnagent import NNAgent


# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/#
class CNNAgent(NNAgent):

    def _itemsHpvector(self, index: int) -> list:
        if index in [1, 4]:
            return np.arange(50, 1000, 25).tolist()
        elif index in [0, 2, 5]:
            return ["relu", "linear", "elu", "tanh", "gelu", "selu"]
        elif index in [3]:
            return np.arange(0, 1, 0.1).tolist()
        else:
            raise Exception("not valid")

    def _defaultHpvector(self):
        return ['relu', 512, 'relu', 0.1, 256, 'relu']

    def _createModel(self, shapeInput, hpvector: list):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=2,
                                   input_shape=shapeInput,
                                   activation=hpvector[0]),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hpvector[1], activation=hpvector[2]),
            tf.keras.layers.Dropout(hpvector[3]),
            tf.keras.layers.Dense(hpvector[4], activation=hpvector[5]),
            tf.keras.layers.Dense(1)
        ])
        return model

    def _adaptInput(self, value):
        # input => Nstep x Nentity
        return np.reshape(value, (1, value.shape[0], value.shape[1]))
        # output => 1 x Nstep x Nentity

    def _prepareDataset(self, dataset):
        X, y = dataset[0], dataset[1]
        n_obs = X[0].shape[0]
        n_step = X[0].shape[1]
        n_entity = X[0].shape[2]
        n_exp = len(X)

        X = np.reshape(X, [n_obs * n_exp, n_step, n_entity])
        y = np.reshape(y, [n_obs * n_exp, 1])
        return X, y, [n_step, n_entity]
