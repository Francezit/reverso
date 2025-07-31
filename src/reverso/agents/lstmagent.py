import numpy as np
import tensorflow as tf

from .nnagent import NNAgent


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
class LSTMAgent(NNAgent):

    def _itemsHpvector(self, index: int) -> list:
        if index == 0:
            return [2, 4, 16, 32, 64, 128, 256, 512, 1024]
        elif index in [2, 5]:
            return np.arange(50, 1000, 25).tolist()
        elif index in [1, 3, 6]:
            return["relu", "linear", "elu", "tanh", "gelu", "selu"]
        elif index in [4]:
            return np.arange(0, 1, 0.1).tolist()
        else:
            raise Exception("not valid")

    def _defaultHpvector(self):
        return [32, 'tanh', 512, 'relu', 0.1, 256, 'relu']

    def _createModel(self, shapeInput, hpvector: list):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=hpvector[0],
                                 input_shape=shapeInput,
                                 activation=hpvector[1]),
            tf.keras.layers.Dense(hpvector[2], activation=hpvector[3]),
            tf.keras.layers.Dropout(hpvector[4]),
            tf.keras.layers.Dense(hpvector[5], activation=hpvector[6]),
            tf.keras.layers.Dense(1)
        ])
        return model

    def _adaptInput(self, value):
        # input => Nstep x Nentity
        value = value.T
        return np.reshape(value, (1, value.shape[0], value.shape[1]))
        # output => 1 x Nentity x Nstep

    def _prepareDataset(self, dataset):
        X, y = dataset[0], dataset[1]
        n_obs = X[0].shape[0]
        n_step = X[0].shape[1]
        n_entity = X[0].shape[2]
        n_exp = len(X)

        # flatten input
        X = np.concatenate(X, axis=0)
        X = np.transpose(X, (0, 2, 1))
        y = np.concatenate(y, axis=0)
        return X, y, [n_entity, n_step]
