import os
import random
from functools import cached_property
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader, Dataset

from src.encoder import SalukiEncoder

logger = logging.getLogger(__name__)


def stochastic_shift(sequence: torch.tensor):
    """Helper method to shift sequence stochastically

    Based on a random number generated between (-3, 3),
    the sequence is shifted by the value generated.
    If shifted beyond the bounds, it'll be pad with zeros

    Parameters
    ----------
    sequence: torch.tensor
        Sequence to be shifted
    """
    shift = torch.randint(-3, 4, (1,)).item()
    if shift > 0:  # shift right
        shifted = torch.roll(sequence, shift, dims=1)
        shifted[:, :shift] = 0.0
    elif shift < 0:  # shift left
        shifted = torch.roll(sequence, shift, dims=1)
        shifted[:, shift:] = 0.0
    else:
        shifted = sequence
    return shifted


class Data(Dataset):
    """Data class for Saluki Finetuning Model"""

    def __init__(
            self,
            transcripts: pd.DataFrame,
            transformation: Callable = None,
            inference: bool = False,
    ):
        """Initialize class

        Parameters
        ----------
        transcripts: pd.DataFrame
            transcript dataframe containing  following columns
            1. exons_starts_idx
            2. sequence
            3. coding start
            4. label
            5. transcript_id

        transformation: Callable, default=None
            function for label transformation

        inference: bool, default=False
            Creates inference dataset if true
        """
        self.inference = inference
        self.data = transcripts
        self.transformation = transformation
        self.encoder = SalukiEncoder()

        self.t_ids = self.data["transcript_id"]
        self.coding_starts = self.data["coding_start"]
        self.sequences = self.data["sequence"]
        self.splice_sites = self.data["exon_starts_idx"]

        if type(self.data["exon_starts_idx"].iloc[0]) == str:
            self.splice_sites = self.process_splice_sites()

    def process_splice_sites(self):
        """Data pre-process if exon_starts_idx is string"""
        splice_sites = (
            self.data["exon_starts_idx"]
            .apply(lambda x: x.replace("[", "").replace("]", ""))
            .str.split(",")
            .apply(lambda x: [int(i) for i in x if i])
        )
        return splice_sites

    def encoded_seq(self, idx):
        """Returns 6-track encoded sequence"""
        return self.encoder.encode(
            self.sequences[idx], self.coding_starts[idx], self.splice_sites[idx]
        )

    @cached_property
    def labels(self):
        """Half life labels"""
        if self.inference:
            raise ValueError("Inference data doesnt have labels")
        labels = torch.Tensor(self.data["label"]).unsqueeze(1)
        if self.transformation:
            labels = self.transformation(labels)
        return labels

    def __len__(self):
        """Return length of dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at idx"""
        if self.inference:
            return (self.t_ids[idx], self.encoded_seq(idx))
        else:
            return (
                self.data.transcript_id[idx],
                stochastic_shift(self.encoded_seq(idx)),
                self.labels[idx],
            )


class Loader(DataLoader):
    """Class to create a pytorch loader"""

    def __init__(self, dataset, batch_size=64, shuffle=True, drop_last=True):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )


def get_loaders(
        data_path: str = HEK_DESIGN,
        transformation: Callable = None,
        batch_size: int = 64,
        shuffle: bool = True,
) -> Union[Loader, Loader]:
    """Method to get train and validation loaders

    Parameters
    ----------
    data_path: str
        Path to design file. See Data for specs
    transformation: Callable, default=None
        function for label transformation
    batch_size: int, default 64
        Batch size for loader
    shuffle: bool, default True
        Boolean to indicate whether to shuffle samples

    Returns
    -------
    train_loader: Loader
        Dataloader with training dataset
    val_loader
        Dataloader with validation dataset
    """
    logger.info(f"Design path : {data_path}")
    data = pd.read_csv(data_path)
    train_data = data[data["split"] == "train"].reset_index(drop=True)
    train_dataset = Data(train_data, transformation=transformation)
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_data = data[data["split"] == "valid"].reset_index(drop=True)
    val_dataset = Data(val_data, transformation=transformation)
    val_loader = Loader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    test_data = data[data["split"] == "test"].reset_index(drop=True)
    test_dataset = Data(test_data, transformation=transformation)
    test_loader = Loader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


def get_torch_device() -> torch.device:
    """Assign torch device to GPU if available, otherwise returns CPU as device.

    Returns
    -------
    device : torch.device
        Available PyTorch device. ("cuda:x" for GPU, "cpu" for CPU)
    """
    if torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            device_ordinal = int(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            device_ordinal = torch.cuda.current_device()
        device = torch.device("cuda", device_ordinal)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def set_random_seeds(random_seed):
    """Sets random seeds"""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)