import os
import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.finetune import SalukiFineTune
from src.utils import Loader


class TestSalukiFineTune:
     def test__init__(self, get_model):
         model = get_model
         assert model.exp_name == "test_run"
         assert isinstance(model.train_loader, Loader)
         assert isinstance(model.val_loader, Loader)
         assert "/tmp/" in model.dev_dir
         assert os.path.isdir(f"{model.dev_dir}/sample_preds")
         assert os.path.isdir(f"{model.dev_dir}/metrics")

     @pytest.mark.parametrize(
         "loss_fn, exp_loss",
         [
             ("mse", "MSELoss"),
             ("bce", "BCEWithLogitsLoss"),
         ],
     )
     def test_init_loss_fn(
         self,
         tmp_path,
         get_weights_and_params,
         get_train_test_val_loaders,
         loss_fn,
         exp_loss,
     ):
         torch_path = str(RESOURCES / "saluki_in_torch_f0_c0.pt")
         train_loader, val_loader, _ = get_train_test_val_loaders
         model = SalukiFineTune(
             torch_model_path=torch_path,
             train_loader=train_loader,
             val_loader=val_loader,
             dev_dir=f"{tmp_path}",
             loss_fn=loss_fn,
         )
         assert exp_loss in str(model.loss)

     @patch.object(SalukiFineTune, "__call__")
     def test_validate(self, mock_call, get_model):
         model = get_model
         exp_preds = torch.Tensor([1.0, 2.0]).unsqueeze(dim=1)
         mock_call.return_value = exp_preds
         exp_labels = np.sqrt(torch.Tensor([4.0, 2.25])).unsqueeze(dim=1)
         val_output = model.validate()
         (
             val_labels_list,
             val_preds_list,
             val_running_loss,
             batch_num,
             val_tx,
         ) = val_output
         exp_mse = ((exp_preds - exp_labels) ** 2).mean()
         exp_tx = ["ENST00000263100.8", "ENST00000337664.9"]
         assert exp_mse == val_running_loss
         assert batch_num == 0
         assert val_labels_list == exp_labels.squeeze().tolist()
         assert val_preds_list == exp_preds.squeeze().tolist()
         assert val_tx == exp_tx

     @patch.object(SalukiFineTune, "validate")
     def test_log_step(self, mock_validate, get_model):
         model = get_model
         exp_val_labels, exp_val_preds, exp_val_loss, exp_val_tx = [], [], [], []
         val_outputs = []
         for epoch in range(4):
             val_labels = [random.uniform(0, 4) for _ in range(10)]
             val_preds = [random.uniform(-4, 4) for _ in range(10)]
             val_tx = [f"val_{i}" for i in range(10)]
             val_running_loss = random.uniform(0, 10)
             val_batch_num = random.randint(0, 10)
             exp_val_labels.append(val_labels)
             exp_val_preds.append(val_preds)
             exp_val_loss.append(val_running_loss / (val_batch_num + 1))
             exp_val_tx.append(val_tx)
             val_outputs.append(
                 (val_labels, val_preds, val_running_loss, val_batch_num, val_tx)
             )
         mock_validate.side_effect = val_outputs

         exp_train_labels, exp_train_preds, exp_train_loss, exp_train_tx = [], [], [], []
         for epoch in range(4):
             train_labels = [random.uniform(0, 4) for _ in range(10)]
             train_preds = [random.uniform(-4, 4) for _ in range(10)]
             train_tx = [f"train_{i}" for i in range(10)]
             running_loss = random.uniform(0, 10)
             batch_num = random.randint(0, 10)
             exp_train_labels.append(train_labels)
             exp_train_preds.append(train_preds)
             exp_train_loss.append(running_loss / (batch_num + 1))
             exp_train_tx.append(train_tx)
             model.log_step(
                 epoch, train_labels, train_preds, running_loss, batch_num, train_tx
             )

         metrics_exp_file = f"{model.dev_dir}/metrics/{model.exp_name}.csv"
         preds_exp_file = f"{model.dev_dir}/sample_preds/{model.exp_name}.csv"
         assert os.path.isfile(metrics_exp_file)
         assert os.path.isfile(preds_exp_file)

         sample_preds = pd.read_csv(preds_exp_file)
         assert exp_val_labels == [eval(i) for i in sample_preds.val_labels]
         assert exp_val_preds == [eval(i) for i in sample_preds.val_preds]
         assert exp_train_labels == [eval(i) for i in sample_preds.train_labels]
         assert exp_train_preds == [eval(i) for i in sample_preds.train_preds]
         assert exp_train_tx == [eval(i) for i in sample_preds.train_ids]
         assert exp_val_tx == [eval(i) for i in sample_preds.val_ids]

         metrics = pd.read_csv(metrics_exp_file)
         exp_metrics = pd.DataFrame(
             data={
                 "train_loss": exp_train_loss,
                 "val_loss": exp_val_loss,
             }
         )
         pd.testing.assert_frame_equal(metrics, exp_metrics)

     @patch.object(SalukiFineTune, "log_step")
     @patch(
         "aiml_rna_stability.saluki_finetune.utils.stochastic_shift",
         side_effect=lambda x: x,
     )
     def test_train_epoch(self, mock_shift, mock_log_step, get_model):
         model = get_model

         def modified_forward(x):
             layer = torch.nn.Linear(12288, 1, bias=False)
             torch.nn.init.constant_(layer.weight, 1.0)
             x = layer(x)
             x = x.sum(axis=1)
             return x

         model.forward = modified_forward

         exp_epoch = random.randint(0, 10)
         seqs = torch.vstack([data[1] for data in model.train_loader])
         # since the modified layer is just a unit matrix, the prediction
         # should be sum of the encoded sequences
         exp_preds = seqs.sum(axis=(1, 2)).tolist()
         exp_labels = [3.0, 5.5]
         exp_tx = ["ENST00000642412.2", "ENST00000248450.9"]
         exp_loss = ((np.array(exp_preds) - np.array(exp_labels)) ** 2).mean()

         model.train_epoch(exp_epoch, log=True)
         (
             epoch,
             labels,
             preds,
             loss,
             batch_num,
             transcripts,
         ) = mock_log_step.call_args_list[0][0]
         assert epoch == exp_epoch
         assert labels == exp_labels
         assert preds == exp_preds
         assert batch_num == 0
         assert loss == exp_loss
         assert transcripts == exp_tx

     @patch.object(SalukiFineTune, "log_step")
     @pytest.mark.parametrize(
         "log, exp_call",
         [
             (True, 1),
             (False, 0),
         ],
     )
     def test_train_epoch_log(self, mock_log_step, get_model, log, exp_call):
         get_model.train_epoch(0, log=log)
         assert mock_log_step.call_count == exp_call

     @patch.object(SalukiFineTune, "train_epoch")
     def test_run_train(self, mock_epoch, get_model):
         model = get_model
         num_epochs = random.randrange(100, 250, 5)
         log_freq = 5
         model.run_train(epochs=num_epochs, log_freq=log_freq, save_model=False)
         assert mock_epoch.call_count == num_epochs
         num_logs = sum([e.kwargs["log"] for e in mock_epoch.call_args_list])
         assert num_epochs // log_freq == num_logs

     @patch.object(SalukiFineTune, "train_epoch")
     def test_save_model(self, mock_epoch, get_model):
         model = get_model
         model.run_train(epochs=6, log_freq=2, save_model=True)
         exp_pt = f"{model.dev_dir}/models/{model.exp_name}.pt"
         assert os.path.isfile(exp_pt)
         saved_model = torch.load(exp_pt, weights_only=False)
         assert isinstance(saved_model, SalukiFineTune)