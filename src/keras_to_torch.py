from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging


from src.constants import FEAT_DIM
from src.utils import Loader, get_torch_device

logger = logging.getLogger(__name__)


def regroup_params_gru(
        gru_weight_or_bias: np.array, torch_param: nn.Parameter
) -> nn.Parameter:
    """Rearranges to make keras weights compatible to torch's methods

    Parameters
    ----------
    weight_or_bias_gru: np.array
        GRU weight/bias array pulled from keras model file
    torch_param: nn.Parameyer
        torch's original parameter for the same weight

    Returns
    -------
    regrouped : nn.Parameter
        Keras weight rearranged to make it compatible with torch's GRU method
    """
    [r, z, h] = np.split(gru_weight_or_bias.T, 3, axis=0)
    pre_trained = np.concatenate((z, r, h), axis=0)
    regrouped = nn.Parameter(torch.ones_like(torch_param) * pre_trained)
    return regrouped


def init_weights(layer: nn.Module, weights: Dict):
    """Initializes weights to layers

    Parameters
    ----------
    layer: nn.Module
        Torch layer whose weight needs to be set
    weights: Dict
        dictionary containing keras weights.
        Usually the output of utils.get_weights
    """
    if getattr(layer, "name", None) is not None:
        layer_weights = weights[layer.name]
    if type(layer) == nn.Conv1d:
        layer.weight = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/kernel:0"].transpose())
        )
        if f"{layer.name}/bias:0" in layer_weights:
            layer.bias = nn.Parameter(
                torch.from_numpy(layer_weights[f"{layer.name}/bias:0"].transpose())
            )

    if type(layer) == nn.LayerNorm:
        layer.weight = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/gamma:0"])
        )
        layer.bias = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/beta:0"])
        )

    if type(layer) == nn.GRU:
        gru_weights = list(layer_weights.values())
        layer.weight_ih_l0_reverse = regroup_params_gru(
            np.array(gru_weights[0]), layer.weight_ih_l0_reverse
        )
        layer.weight_hh_l0_reverse = regroup_params_gru(
            np.array(gru_weights[1]), layer.weight_hh_l0_reverse
        )

        bias_shape = torch.stack([layer.bias_ih_l0_reverse, layer.bias_hh_l0_reverse]).T
        bias_weights = regroup_params_gru(np.array(gru_weights[2]), bias_shape).T
        layer.bias_ih_l0_reverse = nn.Parameter(bias_weights[0])
        layer.bias_hh_l0_reverse = nn.Parameter(bias_weights[1])

        layer.weight_ih_l0.requires_grad_(False)
        layer.weight_hh_l0.requires_grad_(False)
        layer.bias_ih_l0.requires_grad_(False)
        layer.bias_hh_l0.requires_grad_(False)

    if type(layer) == nn.BatchNorm1d:
        layer.weight = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/gamma:0"])
        )
        layer.bias = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/beta:0"])
        )
        layer.running_mean = torch.from_numpy(
            layer_weights[f"{layer.name}/moving_mean:0"]
        )
        layer.running_var = torch.from_numpy(
            layer_weights[f"{layer.name}/moving_variance:0"]
        )

    if type(layer) == nn.Linear:
        layer.weight = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/kernel:0"].transpose())
        )
        layer.bias = nn.Parameter(
            torch.from_numpy(layer_weights[f"{layer.name}/bias:0"].transpose())
        )


class SalukiKerasToTorch(nn.Module):
    """Class to create Torch Model from Saluki's Keras weights"""

    def __init__(
            self,
            torch_model_path=None,
            params=None,
            weights=None,
    ):
        """Initializes class

        Parameters
        ----------
        torch_model_path: str, default None
            Path to saved torch model
        params: Dict, default None
            Dictionary loaded from params_file in Saluki's directory
            Usually the output of utils.get_weights
        weights: Dict, default None
            dictionary containing keras weights.
            Usually the output of utils.get_weights
        """
        super().__init__()
        self.device = get_torch_device()

        if not (torch_model_path):
            if not (weights and params):
                raise ValueError("Model path or weights & params should be specified")

        if torch_model_path:
            # load from torch model path
            _model = torch.load(torch_model_path, weights_only=False)
            _model = _model.to(self.device)
            logger.info(f"Model loaded from: {torch_model_path}")
            self.module_list = _model.module_list
        else:
            # load from keras weights and params
            self.module_list = nn.ModuleList()
            self.weights = weights
            self.params = params
            logger.info("Model will be loaded from weights and params")

            for hyperparam, value in self.params.items():
                setattr(self, hyperparam, value)

            self._define_model()
            self._initalize_weights()
            self.module_list.to(self.device)
            self.weights = None  # remove bloat

        # sanity logging
        num_trainable_weights = sum(
            [
                v.numel()
                for k, v in dict(self.module_list.named_parameters()).items()
                if v.requires_grad
            ]
        )
        logger.info(f"Number of trainable weights {num_trainable_weights}")
        is_gpu = next(self.parameters()).is_cuda
        logger.info(f"Model on GPU : {is_gpu}")

    def _define_model(self):
        """Defines model based on Saluki's definition

        Check basenji.rnann for reference
        """
        conv0 = nn.Conv1d(
            in_channels=FEAT_DIM,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            bias=False,
            padding="valid",
        )
        self.module_list.append(conv0)

        for i in range(self.num_layers):
            layer_norm = nn.LayerNorm(self.filters, eps=self.ln_epsilon)
            relu = nn.ReLU()
            conv = nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                bias=True,
                padding="valid",
            )
            dropout = nn.Dropout(p=self.dropout)
            maxpool = nn.MaxPool1d(2)
            self.module_list.extend([layer_norm, relu, conv, dropout, maxpool])

        layer_norm = nn.LayerNorm(self.filters, eps=self.ln_epsilon)
        gru = torch.nn.GRU(
            input_size=self.filters,
            hidden_size=self.filters,
            num_layers=1,
            bidirectional=True,
            bias=True,
            batch_first=True,
        )
        batch_norm = nn.BatchNorm1d(
            num_features=self.filters, momentum=1 - self.bn_momentum, eps=0.001
        )
        self.module_list.extend([layer_norm, relu, gru, batch_norm])

        dense = nn.Linear(self.filters, self.filters)
        batch_norm = nn.BatchNorm1d(
            num_features=self.filters, momentum=1 - self.bn_momentum, eps=0.001
        )
        self.module_list.extend([relu, dense, dropout, batch_norm])

        last = nn.Linear(self.filters, self.num_targets)
        self.module_list.extend([relu, last])

    def _initalize_weights(self):
        """Initalizes model weights to keras saved weights"""
        for torch_layer, keras_layer in zip(self.module_list, self.weights):
            torch_layer.name = keras_layer  # set the layer name as keras
        self.module_list.apply(
            lambda layer: init_weights(
                layer=layer,
                weights=self.weights,
            )
        )

    def layer_forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Helper method to perform layerwise operation

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to perform layer operation
        layer: nn.Module
            Torch layer

        Returns
        -------
        x: torch.Tensor
            Output tensor after layer operation is performed
        """
        if type(layer) == nn.LayerNorm:
            x = x.permute(0, 2, 1)
            x = layer(x)
            x = x.permute(0, 2, 1)
        elif type(layer) == nn.GRU:
            x = x.permute(0, 2, 1)
            _, x = layer(x)
            x = x[1]
        else:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to run the whole network on

        Returns
        -------
        x: torch.Tensor
            Output tensor after forward pass
        """
        for ix, layer in enumerate(self.module_list):
            x = self.layer_forward(x, layer)
        return x

    def predict(self, x):
        """Wrapper around forward with eval mode on"""
        self.eval()
        preds = self(x)
        self.train()
        return preds

    def predict_on_loader(self, loader: Loader) -> pd.DataFrame:
        """Runs prediction on a loader object

        Parameters
        ----------
        loader: Loader
            Loader object to run the prediction on

        Returns
        -------
        preds_df: pd.DataFrame
            Dataframe with transcript_id & prediction
        """
        self.eval()
        predictions, transcript_ids = [], []
        for batch in loader:
            t_ids, encoded_seqs = batch[0], batch[1]
            encoded_seqs = encoded_seqs.to(self.device)
            preds = self.predict(encoded_seqs)
            predictions.extend(preds.squeeze(dim=1).tolist())
            transcript_ids.extend(list(t_ids))
        preds_df = pd.DataFrame(
            data={"transcript_id": transcript_ids, "prediction": predictions}
        )
        self.train()
        return preds_df