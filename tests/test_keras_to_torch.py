from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from basenji import rnann
from torch import nn

from src.basenji_utils import HiddenPrints, get_weights
from src.encoder import SalukiEncoder
from src.keras_to_torch import (
     SalukiKerasToTorch,
     regroup_params_gru,
)
from src.utils import Data, Loader


def test_regroup_params_gru():
     model_file = str(RESOURCES / "model0_best.h5")
     params_file = str(RESOURCES / "params.json")
     weights, params = get_weights(model_file=model_file, params_file=params_file)
     layer = nn.GRU(
         input_size=64,
         hidden_size=64,
         num_layers=1,
         bidirectional=True,
         bias=True,
         batch_first=True,
     )
     torch_wts = [
         layer.weight_ih_l0_reverse,
         layer.weight_hh_l0_reverse,
     ]
     torch_biases = [layer.bias_ih_l0_reverse, layer.bias_hh_l0_reverse]
     gru_key = [name for name in weights if "gru" in name][0]
     for ix, (name, wt) in enumerate(weights[gru_key].items()):
         if "bias" not in name:
             regrouped = regroup_params_gru(wt, torch_wts[ix])
             assert regrouped.shape == torch_wts[ix].shape
         else:
             regrouped_bias = regroup_params_gru(wt, torch.stack(torch_biases).T).T
             assert regrouped_bias[0].shape == torch_biases[0].shape
             assert regrouped_bias[1].shape == torch_biases[1].shape


@pytest.fixture
def get_models(get_weights_and_params):
     weights, params, model_file = get_weights_and_params
     with HiddenPrints():
         keras_model = rnann.RnaNN(params)
     keras_model.restore(model_file)
     torch_model = SalukiKerasToTorch(params=params, weights=weights)
     return keras_model, torch_model


@pytest.fixture
def get_torch_and_keras_layers(get_models):
     keras_model, torch_model = get_models
     # ignoring the first 2 layers of keras model because they are
     # namely - "input layer" and "stochastic shift layer"
     # stochastic shift is created as a util and not a "layer"
     keras_layers = keras_model.model.layers[2:]
     torch_layers = torch_model.module_list
     return keras_layers, torch_layers


@pytest.fixture
def get_encoded_seqs():
     encoder = SalukiEncoder()
     transcripts_path = str(RESOURCES / "finetune_design.csv")
     transcripts = pd.read_csv(transcripts_path)
     transcripts["exon_starts_idx"] = transcripts.exon_starts_idx.apply(eval)
     transcripts["encoded_seq"] = transcripts.apply(
         lambda row: encoder.encode(
             row["sequence"], row["coding_start"], row["exon_starts_idx"]
         )
         .numpy()
         .T,
         axis=1,
     )
     encoded_seqs = np.array(transcripts.encoded_seq.values.tolist())
     torch_encoded_seqs = np.moveaxis(encoded_seqs, [1, 2], [2, 1])
     return encoded_seqs, torch.Tensor(torch_encoded_seqs)


class TestSalukiLoadFromKeras:
     @pytest.mark.parametrize(
         "torch_model_path, weights_arg, params_arg, error_expected",
         (
             [str(RESOURCES / "saluki_in_torch_f0_c0.pt"), False, False, False],
             [None, True, True, False],
             [None, True, False, True],
             [None, False, True, True],
         ),
     )
     def test_init(
         self,
         torch_model_path,
         weights_arg,
         params_arg,
         error_expected,
         get_weights_and_params,
     ):
         weights, params, _ = get_weights_and_params
         if not weights_arg:
             weights = None
         if not params_arg:
             params = None
         if not error_expected:
             model = SalukiKerasToTorch(
                 torch_model_path=torch_model_path, params=params, weights=weights
             )
             assert isinstance(model, SalukiKerasToTorch)
         else:
             with pytest.raises(ValueError):
                 SalukiKerasToTorch(
                     torch_model_path=torch_model_path, params=params, weights=weights
                 )

     def test___define_model(self, get_torch_and_keras_layers):
         keras_layers, torch_layers = get_torch_and_keras_layers
         keras_to_torch_layer_names = {
             "Conv1D": "Conv1d",
             "LayerNormalization": "LayerNorm",
             "ReLU": "ReLU",
             "Dropout": "Dropout",
             "MaxPooling1D": "MaxPool1d",
             "GRU": "GRU",
             "BatchNormalization": "BatchNorm1d",
             "Dense": "Linear",
         }
         assert len(keras_layers) == len(torch_layers)
         for k_layer, t_layer in zip(keras_layers, torch_layers):
             layer_type = [
                 name
                 for name in keras_to_torch_layer_names.keys()
                 if name in str(k_layer)
             ][0]
             assert keras_to_torch_layer_names[layer_type] in str(t_layer)

     def test__initalize_weights(self, get_torch_and_keras_layers):
         keras_layers, torch_layers = get_torch_and_keras_layers
         keras_weights, torch_weights = {}, {}
         for t_layer, k_layer in zip(torch_layers, keras_layers):
             keras_weights[k_layer.name] = k_layer
             torch_weights[k_layer.name] = t_layer

         for layer_name in keras_weights:
             k_weights = keras_weights[layer_name]
             t_weights = torch_weights[layer_name]
             k_sum = sum([np.array(wt).sum() for wt in k_weights.trainable_weights])
             t_sum = sum(
                 [wt.sum().item() for wt in t_weights.parameters() if wt.requires_grad]
             )
             assert round(k_sum, 4) == round(
                 t_sum, 4
             ), f"{layer_name}'s weight is intialized incorrectly"

     def test_layer_forward(
         self, get_encoded_seqs, get_torch_and_keras_layers, get_models
     ):
         keras_model, torch_model = get_models
         keras_layers, torch_layers = get_torch_and_keras_layers
         keras_seqs, torch_seqs = get_encoded_seqs

         k, k_outputs = keras_seqs, {}
         for k_layer in keras_layers:
             k = np.array(k_layer(k))
             k_outputs[k_layer.name] = k

         t, t_outputs = torch_seqs, {}
         for t_layer, (k_name, k_output) in zip(torch_layers, k_outputs.items()):
             t_layer.eval()
             t = torch_model.layer_forward(t, t_layer)
             # just reshaping torch outputs to make similar to keras for comparison
             t_outputs[k_name] = t.reshape(k_output.shape).detach().numpy()
             if t.ndim == 3:
                 t_outputs[k_name] = t.permute(0, 2, 1).detach().numpy()

         assert len(k_outputs) == len(t_outputs)
         for (name, k), t in zip(k_outputs.items(), t_outputs.values()):
             assert np.isclose(t, k, atol=1e-04).all(), f"{name} not the same!"

     @patch.object(SalukiKerasToTorch, "layer_forward")
     def test_forward(
         self, mock_layer_fwd, get_models, get_encoded_seqs, get_torch_and_keras_layers
     ):
         _, torch_model = get_models
         _, torch_seqs = get_encoded_seqs
         _, torch_layers = get_torch_and_keras_layers
         torch_model(torch_seqs)
         assert mock_layer_fwd.call_count == len(torch_layers)

     def test_predict(self, get_models, get_encoded_seqs):
         keras_model, torch_model = get_models
         keras_seqs, torch_seqs = get_encoded_seqs
         k_preds = np.array(
             keras_model.predict(keras_seqs, batch_size=len(keras_seqs), verbose=0)
         )
         t_preds = torch_model.predict(torch_seqs).detach().numpy()
         assert np.isclose(k_preds, t_preds, atol=1e-04).all()

     @patch.object(SalukiKerasToTorch, "predict")
     def test_predict_on_loader(self, mock_predict, get_model):
         transcripts_path = str(RESOURCES / "finetune_design.csv")
         transcripts_df = pd.read_csv(transcripts_path)
         dataset = Data(transcripts_df, transformation=lambda x: x)
         loader = Loader(dataset, batch_size=2, shuffle=False, drop_last=False)
         exp_preds = [
             torch.Tensor([1.0, 1.5]).unsqueeze(dim=1),
             torch.Tensor([2.0, 2.5]).unsqueeze(dim=1),
             torch.Tensor([3.0]).unsqueeze(dim=1),
         ]
         mock_predict.side_effect = exp_preds
         model = get_model
         preds = model.predict_on_loader(loader=loader)
         assert list(preds.columns) == ["transcript_id", "prediction"]
         assert (preds.transcript_id == transcripts_df.transcript_id).all()
         assert list(preds.prediction) == [1, 1.5, 2.0, 2.5, 3]