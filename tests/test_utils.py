import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.basenji_utils import get_weights
from src.utils import Data, get_loaders, stochastic_shift


@patch("aiml_rna_stability.saluki_finetune.encoder.torch.randint")
def test_stochastic_shift_right(mock_random):
    mock_random.return_value = torch.tensor(3)
    sequence = torch.tensor([[1, 2, 3, 4, 5]] * 6)
    shifted_seq = stochastic_shift(sequence=sequence)
    expected = torch.tensor([[0, 0, 0, 1, 2]] * 6)
    assert (expected == shifted_seq).all()


@patch("aiml_rna_stability.saluki_finetune.encoder.torch.randint")
def test_stochastic_shift_left(mock_random):
    mock_random.return_value = torch.tensor(-3)
    sequence = torch.tensor([[1, 2, 3, 4, 5]] * 6)
    shifted_seq = stochastic_shift(sequence=sequence)
    expected = torch.tensor([[4, 5, 0, 0, 0]] * 6)
    assert (expected == shifted_seq).all()


class TestData:
    def test__init__(self):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts)
        assert len(dataset) == 5
        assert len(dataset.t_ids) == 5
        assert len(dataset.coding_starts) == 5
        assert len(dataset.sequences) == 5
        assert len(dataset.splice_sites) == 5
        assert len(dataset.labels) == 5
        assert dataset.encoded_seq(0).shape == (6, 12288)

    @pytest.mark.parametrize(
        "transformation_fn, labels",
        [
            (torch.sqrt, [1, 4]),
            (
                    torch.log,
                    [
                        2.718281828459045,
                        7.38905609893065,
                    ],
            ),
            (None, [1, 2]),
        ],
    )
    def test_labels(self, transformation_fn, labels):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        transcripts = transcripts[:2]
        transcripts["label"] = labels
        dataset = Data(transcripts=transcripts, transformation=transformation_fn)
        expected = torch.Tensor([1.00, 2.00]).unsqueeze(1)
        assert torch.isclose(dataset.labels, expected).all()
        assert torch.isclose(dataset.labels[1], expected[1]).all()

    @pytest.mark.parametrize("exons_type_list", [False, True])
    def test_encoded_seq(self, exons_type_list):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path, skiprows=[1, 3, 4])
        if exons_type_list:
            transcripts["exon_starts_idx"] = transcripts["exon_starts_idx"].apply(eval)
        dataset = Data(
            transcripts=transcripts, inference=True
        )  # turn off stochastic_shift
        assert np.array(dataset.encoded_seq(0)).shape == (6, 12288)
        exp_track_sums = [2.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        assert dataset[0][1].sum(dim=1).tolist() == exp_track_sums
        exp_track_sums = [1.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        assert dataset[1][1].sum(dim=1).tolist() == exp_track_sums

    @patch(
        "aiml_rna_stability.saluki_finetune.utils.stochastic_shift",
        side_effect=lambda x: x,
    )
    def test_iter(self, mock_shift):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts)
        actual_transcript_ids = []
        for i in range(len(dataset)):
            iter_data = dataset[i]
            assert len(iter_data) == 3
            transcript_id, seq, label = iter_data
            assert type(transcript_id) == str
            assert seq.shape == (6, 12288)
            assert type(label) == torch.Tensor
            actual_transcript_ids.append(transcript_id)
        assert list(transcripts.transcript_id) == actual_transcript_ids
        assert mock_shift.call_count == len(dataset)


class TestInferenceData(unittest.TestCase):
    def test__init__(self):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts, inference=True)
        assert len(dataset) == 5
        assert len(dataset.t_ids) == 5
        assert len(dataset.coding_starts) == 5
        assert len(dataset.sequences) == 5
        assert len(dataset.splice_sites) == 5
        assert dataset.encoded_seq(0).shape == (6, 12288)

    def test_labels(self):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts, inference=True)
        with self.assertRaises(ValueError):
            dataset.labels

    def test_encoded_seq(self):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts, inference=True)
        assert dataset.encoded_seq(0).shape == (6, 12288)

    @patch("aiml_rna_stability.saluki_finetune.utils.stochastic_shift")
    def test_iter(self, mock_shift):
        transcripts_path = str(RESOURCES / "finetune_design.csv")
        transcripts = pd.read_csv(transcripts_path)
        dataset = Data(transcripts=transcripts, inference=True)
        actual_transcript_ids = []
        for i in range(len(dataset)):
            iter_data = dataset[i]
            assert len(iter_data) == 2
            transcript_id, seq = iter_data
            assert type(transcript_id) == str
            assert seq.shape == (6, 12288)
            actual_transcript_ids.append(transcript_id)
        assert list(transcripts.transcript_id) == actual_transcript_ids
        assert mock_shift.call_count == 0


def test_get_loaders():
    transcripts_path = str(RESOURCES / "finetune_design.csv")
    train_loader, val_loader, test_loader = get_loaders(
        data_path=transcripts_path,
    )
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 1
    assert "RandomSampler" in str(train_loader.sampler)
    assert "RandomSampler" in str(val_loader.sampler)
    assert train_loader.batch_size == 64
    assert val_loader.batch_size == 64


@hpc_only
def test_get_weights():
    model_file = str(RESOURCES / "model0_best.h5")
    params_file = str(RESOURCES / "params.json")
    weights, params = get_weights(model_file=model_file, params_file=params_file)
    assert len(weights) == 41  # there should be 41 layers
    assert len(params) == 13