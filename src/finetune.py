import os
from functools import cached_property
from typing import List

import pandas as pd
import torch.nn
import logging
from tqdm import tqdm

from src.keras_to_torch import SalukiKerasToTorch

logger = logging.getLogger(__name__)

LR_DEFAULT = 0.0001
BETAS_DEFAULT = (0.90, 0.998)
EPOCHS_DEFAULT = 50
LOG_FREQ_DEFAULT = 5


class SalukiFineTune(SalukiKerasToTorch):
    def __init__(self, **kwargs):
        """Initalizes SalukiFineTune

        SalukiFineTune is inherited from SalukiKerasToTorch

        Parameters
        ----------
        kwargs:
            Named arguments that should have the following keywords
                1. train_loader: Pytorch DataLoader for train set
                2. val_loader: Pytorch DataLoader for validation set
                3. params: keras params (see SalukiKerasToTorch)
                4. weights: keras weights (see SalukiKerasToTorch)
            The following arguments are optional
                1. dev_dir
                    path to store training results,
                2. exp_name - default, test_run
                    name of the experiment. files are stored with this name
                3. disable_layers_grad: default (False)
                    See SalukiKerasToTorch for details
        """
        self.train_loader = kwargs.pop("train_loader")
        self.val_loader = kwargs.pop("val_loader")
        self.dev_dir = kwargs.pop("dev_dir", f"/scratch/")
        self.exp_name = kwargs.pop("exp_name", "test_run")
        self.loss_fn = kwargs.pop("loss_fn", "mse")
        logger.info(f"Dev Dir: {self.dev_dir}\tExp Name: {self.exp_name}")
        os.makedirs(f"{self.dev_dir}/sample_preds", exist_ok=True)
        os.makedirs(f"{self.dev_dir}/metrics", exist_ok=True)
        super().__init__(**kwargs)
        self.metrics = {}
        self.sample_preds = {}
        logger.info(
            f"Loss Function: {str(self.loss)}",
        )

    @cached_property
    def optimizer(self):
        """Model's optimizer"""
        return torch.optim.Adam(
            params=self.parameters(), lr=LR_DEFAULT, betas=BETAS_DEFAULT
        )

    @cached_property
    def loss(self):
        """Model Loss"""
        if self.loss_fn == "bce":
            return torch.nn.BCEWithLogitsLoss()
        if self.loss_fn == "mse":
            return torch.nn.MSELoss()
        raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    def train_epoch(self, epoch: int, log: bool = False):
        """Trains a single epoch and conditionally logs results

        Parameters
        ----------
        epoch: int
            Integer indicating the epoch.

        log: bool, default = False
            Boolean to indicate whether to log results
        """
        epoch_running_loss, labels_list, preds_list, tx_list = 0.0, [], [], []
        for batch_num, (t_ids, batch, labels) in enumerate(self.train_loader):
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()  # zero gradients for every batch

            predictions = self(batch)
            batch_loss = self.loss(predictions, labels)

            batch_loss.backward()  # propagate gradients
            self.optimizer.step()  # adjust learning weights

            epoch_running_loss += batch_loss.item()
            labels_list.extend(labels.squeeze().tolist())
            preds_list.extend(predictions.squeeze().tolist())
            tx_list.extend(list(t_ids))
        if log:
            self.log_step(
                epoch, labels_list, preds_list, epoch_running_loss, batch_num, tx_list
            )

    def validate(self):
        """Generates predictions on validations set"""
        self.eval()
        val_running_loss, val_preds_list, val_labels_list, tx_list = 0.0, [], [], []
        for batch_num, (t_ids, batch, labels) in enumerate(self.val_loader):
            batch, labels = batch.to(self.device), labels.to(self.device)

            preds = self(batch)
            val_batch_loss = self.loss(preds, labels)
            val_running_loss += val_batch_loss.item()
            val_labels_list.extend(labels.squeeze().tolist())
            val_preds_list.extend(preds.squeeze().tolist())
            tx_list.extend(list(t_ids))

        self.train()
        return val_labels_list, val_preds_list, val_running_loss, batch_num, tx_list

    def log_step(
            self,
            epoch: int,
            labels_list: List,
            preds_list: List,
            epoch_running_loss: float,
            batch_num: int,
            tx_list: List,
    ):
        """Generates point metrics and logs

        Parameters
        ----------
        epoch: int
            Integer indicating the epoch.
        labels_list: List
            List of label values
        preds_list: List
            List of prediction values
        epoch_running_loss: float
            Cumulative training loss for the epoch
        batch_num: int
            Number of batches in the training loader
        tx_list: List
            List of transcript IDs
        """
        # train metrics
        train_loss = epoch_running_loss / (batch_num + 1)
        # val metrics
        (
            val_labels_list,
            val_preds_list,
            val_running_loss,
            batch_num,
            val_tx_list,
        ) = self.validate()
        val_loss = val_running_loss / (batch_num + 1)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        logger.info(f"epoch: {epoch}: {metrics}")
        self.metrics[epoch] = metrics
        self.sample_preds[epoch] = {
            "val_labels": val_labels_list,
            "val_preds": val_preds_list,
            "train_labels": labels_list,
            "train_preds": preds_list,
            "train_ids": tx_list,
            "val_ids": val_tx_list,
        }
        # store metrics and sample preds
        preds_df = pd.DataFrame(self.sample_preds).T
        preds_df.to_csv(f"{self.dev_dir}/sample_preds/{self.exp_name}.csv", index=False)
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(f"{self.dev_dir}/metrics/{self.exp_name}.csv", index=False)

    def run_train(
            self, epochs=EPOCHS_DEFAULT, log_freq=LOG_FREQ_DEFAULT, save_model=True
    ):
        """Runs training

        Parameters
        ----------
        epochs: int, default EPOCHS_DEFAULT
            The number of epochs to run training
        log_freq: int, default LOG_FREQ_DEFAULT
            Will log every `log_freq` runs
        save_model: bool, default True
            Will save model at {dev_dir}/models/{exp_name}.pt
        """
        for epoch in tqdm(range(epochs)):
            log = False
            if epoch % log_freq == 0:
                log = True
            self.train_epoch(log=log, epoch=epoch)
        if save_model:
            self.save_model()

    def save_model(self):
        """Saves model

        Model is saved at {dev_dir}/models/{exp_name}.pt
        """
        # remove bloat
        self.train_loader = None
        self.val_loader = None
        self.weights = None
        self.metrics = None
        self.sample_preds = None

        # store
        os.makedirs(f"{self.dev_dir}/models", exist_ok=True)
        save_path = f"{self.dev_dir}/models/{self.exp_name}.pt"
        torch.save(self, f"{save_path}")
        logger.info(f"Model saved at {save_path}")