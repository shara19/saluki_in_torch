from random import choice

import pytest
import torch

from src.encoder import (
     CODON_LEN,
     MAXLEN,
     SalukiEncoder,
     get_coding_stop,
)
from src.constants import FEAT_DIM


@pytest.fixture
def sample_seq():
     codonstring = "TCAGGCTTGACT"
     sequence = codonstring * 10 + "T"
     code_start = 29
     code_stop = 59
     splice_sites = [12, 40, 90, 111]
     # add start codon
     sequence = sequence[:code_start] + "ATG" + sequence[code_start + 3 :]
     # add stop codon
     sequence = sequence[:code_stop] + "TAA" + sequence[code_stop + 3 :]
     return sequence, code_start, code_stop, splice_sites


@pytest.mark.parametrize(
     "code_stop_criterion",
     [
         "code_stop_exists",  # code stop exists
         "no_code_stop",  # code stop doesn't exist, defaults to len of sequence
         "long_sequence_code_stop",  # code stop exists beyond MAXLEN
     ],
)
def test_get_code_stop(sample_seq, code_stop_criterion):
     sequence, code_start, code_stop, splice_sites = sample_seq
     if code_stop_criterion == "code_stop_exists":  # code stop exists
         exp_code_stop = code_stop
         actual_code_stop = get_coding_stop(sequence, code_start)
         assert sequence[actual_code_stop : actual_code_stop + 3] == "TAA"

     # code stop doesn't exist, defaults to len of sequence
     if code_stop_criterion == "no_code_stop":
         sequence = (
             sequence[:59] + "GGC" + sequence[59 + CODON_LEN :]
         )  # remove stop codon
         actual_code_stop = get_coding_stop(sequence, code_start)
         exp_code_stop = len(sequence) - 1

     # code stop exists beyond MAXLEN
     if code_stop_criterion == "long_sequence_code_stop":
         updated_seq = sequence[:59] + "GGC" + sequence[59 + 3 :]  # remove stop codon
         updated_seq = updated_seq[:-1]
         updated_seq = updated_seq * 120  # make real large sequence

         # move code_stop > MAXLEN
         exp_code_stop = code_start + 3 * 4500
         assert exp_code_stop > MAXLEN
         sequence = (
             updated_seq[:exp_code_stop] + "TAA" + updated_seq[exp_code_stop + 3 :]
         )
         actual_code_stop = get_coding_stop(sequence, code_start)

     assert actual_code_stop == exp_code_stop
     encoder = SalukiEncoder()
     encoded_seq = encoder.encode(sequence, code_start, splice_sites)
     encoded_seq_code_stop = (
         torch.where(encoded_seq[-2, :].squeeze() == 1.0)[0].max().item()
     )
     if code_stop_criterion != "long_sequence_code_stop":
         assert abs(encoded_seq_code_stop - actual_code_stop) <= 3
     if code_stop_criterion == "long_sequence_code_stop":
         assert encoded_seq_code_stop == actual_code_stop - (
             len(sequence) - MAXLEN
         )  # left shift idx


class TestSalukiEncoder:
     def test_encode_seq_padding_ok(self):
         sample_seq = "TATGCAGTAA"
         encoder = SalukiEncoder()
         encoded_seq = encoder.encode(
             sequence=sample_seq, coding_start=1, splice_sites=[3]
         )
         assert encoded_seq.shape == (FEAT_DIM, MAXLEN)
         unpadded_encoding = torch.zeros((6, len(sample_seq)))
         unpadded_encoding[0, :] = torch.tensor([0, 1, 0, 0, 0, 1, 0, 0, 1, 1])  # A
         unpadded_encoding[1, :] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # C
         unpadded_encoding[2, :] = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])  # G
         unpadded_encoding[3, :] = torch.tensor([1, 0, 1, 0, 0, 0, 0, 1, 0, 0])  # T
         unpadded_encoding[4, :] = torch.tensor(
             [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
         )  # codon start
         unpadded_encoding[5, :] = torch.tensor(
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
         )  # splice site
         assert (encoded_seq[:, 0 : len(sample_seq) :] == unpadded_encoding).all()
         assert encoded_seq[:, len(sample_seq) :].sum() == 0

     def test_encode_seq_long_ok(self):
         sample_seq = "".join(choice("ATGC") for _ in range(12288 + 500))
         code_start = 12288 + 500 - 30
         sample_seq = sample_seq[:code_start] + "ATG" + sample_seq[code_start + 3 :]
         encoder = SalukiEncoder()
         encoded_seq = encoder.encode(
             sequence=sample_seq, coding_start=code_start, splice_sites=[0, 12300, 12305]
         )
         assert encoded_seq.shape == (FEAT_DIM, MAXLEN)
         nt_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
         inv_map = {v: k for k, v in nt_to_idx.items()}
         actual_seq = "".join(
             [inv_map[i] for i in torch.argmax(encoded_seq[0:4, :], dim=0).tolist()]
         )
         # left shift nucleotide
         assert actual_seq == sample_seq[500:]  # 5' 500nt are truncated

         # left shift codon track
         encoded_seq_code_start = (
             torch.where(encoded_seq[-2, :].squeeze() == 1.0)[0].min().item()
         )
         assert encoded_seq_code_start == 12258
         code_stop = get_coding_stop(sequence=sample_seq, coding_start=code_start)
         assert encoded_seq[-2, :].sum() == (code_stop - code_start + 3) // 3

         # left shift splice-site track
         assert encoded_seq[-1, :].sum() == 2
         ss_1 = torch.where(encoded_seq[-1, :].squeeze() == 1.0)[0].min().item()
         ss_2 = torch.where(encoded_seq[-1, :].squeeze() == 1.0)[0].max().item()
         assert ss_1 == 12300 - ((12288 + 500) - MAXLEN)
         assert ss_2 == 12305 - ((12288 + 500) - MAXLEN)

     def test_encode_seq_nucleotide_ok(self):
         sample_seq = "ATGC"
         encoder = SalukiEncoder()
         encoded_seq = encoder.encode(sample_seq, 0, [])
         trailing_zeros = [0] * (MAXLEN - 4)
         assert all(encoded_seq[0, :] == torch.tensor([1, 0, 0, 0] + trailing_zeros))
         assert all(encoded_seq[1, :] == torch.tensor([0, 0, 0, 1] + trailing_zeros))
         assert all(encoded_seq[2, :] == torch.tensor([0, 0, 1, 0] + trailing_zeros))
         assert all(encoded_seq[3, :] == torch.tensor([0, 1, 0, 0] + trailing_zeros))

     def test_encode_seq_codon_start_ok(self, sample_seq):
         sequence, code_start, code_stop, splice_sites = sample_seq
         encoder = SalukiEncoder()
         encoded_seq = encoder.encode(
             sequence=sequence, coding_start=code_start, splice_sites=[]
         )
         codon_starts_idx = torch.where(encoded_seq[4, :].squeeze() == 1)[0].tolist()
         assert codon_starts_idx == [i for i in range(code_start, code_stop + 1, 3)]

     def test_encode_seq_splice_site_ok(self, sample_seq):
         sequence, code_start, _, splice_sites = sample_seq
         encoder = SalukiEncoder()
         encoded_seq = encoder.encode(
             sequence=sequence, coding_start=code_start, splice_sites=splice_sites
         )
         splice_starts_idx = torch.where(encoded_seq[5, :].squeeze() == 1.0)[0].tolist()
         assert splice_starts_idx == splice_sites
