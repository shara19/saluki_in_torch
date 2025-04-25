import torch

from src.constants import  CODON_LEN, MAXLEN, STOP_CODONS


def get_coding_stop(sequence, coding_start):
    """Helper method to get coding stop index from sequence and coding_start"""
    coding_stop = len(sequence) - 1  # default stop codon
    coding_seq = sequence[coding_start:]
    for ix in range(0, len(coding_seq), CODON_LEN):
        if coding_seq[ix: ix + CODON_LEN] in STOP_CODONS:
            coding_stop = coding_start + ix
            break
    return coding_stop


class SalukiEncoder:
    """Saluki Encoder Class"""

    def __init__(self):
        self.lookup_table = torch.zeros((256, 4))
        self.lookup_table[ord("A")] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.lookup_table[ord("C")] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        self.lookup_table[ord("G")] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self.lookup_table[ord("T")] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.lookup_table[ord("U")] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.lookup_table[ord("N")] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.lookup_table[ord(".")] = torch.tensor([0.25, 0.25, 0.25, 0.25])

    def encode(self, sequence, coding_start, splice_sites) -> torch.Tensor:
        """Encodes the sequence in a matrix of MAXLEN length

        The matrix is 6 dimensional one-hot encoders.
        Out of which, 4 are used to indicate the presence of the nucleotides (ACGU),
        one to indicate each codon start,
        one to indicate the existence of a splice start site.

        Parameters
        ----------
        sequence: str
            Transcript CDNA  5' UTR, exons and 3' UTR concatenated together in order

        coding_start: int
            Position in the sequence where coding begins.
            Note expects zero-based numbering

        splice_sites:
            Positions of 5' splice-sites. Note expects zero-based numbering

        Returns
        -------
        encoder: Array of shape (6, 12288)
        """
        # nuc emb tracks
        input_tokens = [ord(c) for c in sequence[-MAXLEN:].upper()]
        nuc_emb = self.lookup_table[input_tokens].T

        # codon track
        coding_stop = get_coding_stop(sequence, coding_start)
        codon_track = torch.zeros(1, len(sequence))
        codon_track[:, list(range(coding_start, coding_stop + 1, 3))] = 1.0
        codon_track = codon_track[:, -MAXLEN:]

        # splice site track
        ss_track = torch.zeros(1, len(sequence))
        ss_track[:, splice_sites] = 1.0
        ss_track = ss_track[:, -MAXLEN:]

        onehot = torch.vstack([nuc_emb, codon_track, ss_track])

        # padding
        seq_len = len(sequence[-MAXLEN:])
        onehot = torch.nn.functional.pad(onehot, (0, max(MAXLEN - seq_len, 0)), value=0)
        return onehot