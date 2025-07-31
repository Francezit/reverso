__author__ = "Francesco Zito"
__copyright__ = "Copyright 2024, Francesco Zito"
__version__ = "2.0.0"
__email__ = "zitofrancesco@outlook.com"


import sys, os, logging

sys.path.append(os.path.join(__path__[0], "./agents"))
sys.path.append(os.path.join(__path__[0], "./core"))
sys.path.append(os.path.join(__path__[0], "./data"))
sys.path.append(os.path.join(__path__[0], "./utilities"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_AFFINITY"] = "noverbose"

from .reverso import Reverso
from .data import readDatasets, read_data, DataCollection
from .core import Environment, EnvironmentAnalysis, InteractionNetwork, InteractionNetworkHelper, CompareInteractionNetworkResult, SignalPertubationFunction
from .agents import AgentConfiguration, AgentTrainOption, AgentBuilder, getAllAgentsName
from .utilities import MetricsInfo, WorkspaceInfo
from .reversoconfig import AutomakeEnvironmentConfiguration, EnvironmentEvaluationOption, MakeEnvironmentConfiguration, ExtractInteractionNetworkOption

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('h5py').setLevel(logging.ERROR)

import tensorflow as tf

tf.debugging.disable_traceback_filtering()
tf.debugging.experimental.disable_dump_debug_info()
tf.keras.utils.disable_interactive_logging()

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)