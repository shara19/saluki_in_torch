import json
import os
import sys
from collections import defaultdict
from typing import Dict, Union

import numpy as np
from basenji import rnann


class HiddenPrints:
     """Context manager to suppress print statements from Basenji

     This is due to the fact that the basenji rnann module has a print
     statement that clutters stdout
     See line:
     https://github.com/calico/basenji/blob/
     9e1c2e2f5b1b37ad11cfd2a1486d786d356d78a5/basenji/rnann.py#L151
     """

     def __enter__(self):
         """Context manager entry"""
         self._original_stdout = sys.stdout
         sys.stdout = open(os.devnull, "w")

     def __exit__(self, exc_type, exc_val, exc_tb):
         """Context manager exit"""
         sys.stdout.close()
         sys.stdout = self._original_stdout


def get_weights(
     model_file,
     params_file,
) -> Union[Dict, Dict]:
     """Method to get weights from keras file

     Parameters
     ----------
     model_file: str
         Path to saved model to retrieve the weights
     params_file: str
         Path to model hyperparameters

     Returns
     -------
     layer_weights: Dict[Dict]
         Dictionary with key as layer name and weights as different weights
         for said layer. Eg: key - 'linear_1': {'weight_1 : array, 'bias_1: array}

     params_model: Dict
         Dictionary loaded from params_file
     """
     with open(params_file) as params_open:
         params_model = json.load(params_open)["model"]
     with HiddenPrints():
         seqnn_model = rnann.RnaNN(params_model)
     seqnn_model.restore(model_file)
     keras_model = seqnn_model.model
     layer_weights = defaultdict()
     # ignores the first layer because it is a stochastic shift with no weights.
     for layer_idx in range(2, len(keras_model.layers)):
         layer = keras_model.get_layer(index=layer_idx)
         layer_weights[layer.name] = {wt.name: np.array(wt) for wt in layer.weights}
     return layer_weights, params_model