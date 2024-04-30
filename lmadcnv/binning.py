import math

import numpy as np
import pysam
from Bio import Seq, SeqIO, SeqUtils


class RDdata:
    pos: np.ndarray  # the index of bin
    RD: np.ndarray  # the RD of bin

    def __init__(self, num_of_bin: int) -> None:
        self.pos = np.arange(num_of_bin)
        self.RD = np.zeros(num_of_bin)

    def __len__(self) -> int:
        return len(self.RD)


class GCData:
    valid: np.ndarray
    GC: np.ndarray  # GC per thousand (GCcount / bp_per_bin * 1000)
    bp_per_bin: int

    def __init__(self, fasta: Seq.Seq, bp_per_bin: int) -> None:
        self.bp_per_bin = bp_per_bin

        num_of_bin = len(fasta) // bp_per_bin
        self.valid = np.full(num_of_bin, True)
        self.GC = np.full(num_of_bin, 0)

        for i in range(num_of_bin):
            cur = fasta[i * bp_per_bin : (i + 1) * bp_per_bin]
            self.valid[i] = all(dp in "AGCTSWagctsw" for dp in cur)
            # self.valid[i] = cur.count("N") < len(cur) * 1.0
            self.GC[i] = round(SeqUtils.gc_fraction(cur) * 1000)

    def limit(self, length: int):
        self.valid = self.valid[:length]
        self.GC = self.GC[:length]


def binning(
    bam_path: str, fa_path: dict[str, str], bp_per_bin: int = 1000
) -> dict[str, RDdata]:
    """Divide the DNA sequence into bins according to a certain length and calculate the RD value of each bin.

    Parameters
    ----------
    bam_path: str
    The path of bam file.

    fa_path: dict[str, str]
    The path of reference.

    bp_per_bin: int
    The length of each bin.

    Notes
    -----
    The name in `fa_path` should appear in the information corresponding to the bam file, otherwise an exception will be thrown.
    """
    RDs: dict[str, RDdata] = {}
    GCs: dict[str, GCData] = {}
    with pysam.AlignmentFile(bam_path, "rb", ignore_truncation=True) as samfile:
        refs = {
            key: value for key, value in fa_path.items() if key in samfile.references
        }
        if not refs:
            raise ValueError("Valid reference name is not found")

        # calculate GC
        for chr, fasta in refs.items():
            expect_len = samfile.get_reference_length(chr)
            num_of_bin = expect_len // bp_per_bin
            fasta = SeqIO.read(fasta, "fasta").seq
            actual_len = len(fasta)
            if expect_len > actual_len:
                raise ValueError(
                    f"{chr}'s reference length is not matched! expect {expect_len}, actual {actual_len}"
                )
            # elif expect_len < actual_len:
            #     warnings.warn(
            #         f"{chr}'s reference length is not matched! expect {expect_len}, actual {actual_len}"
            #     )
            GCs[chr] = GCData(fasta, bp_per_bin)
            RDs[chr] = RDdata(num_of_bin)
            GCs[chr].limit(num_of_bin)

        # count read
        for read in samfile:
            if read.is_secondary:
                continue
            idx = read.reference_start // bp_per_bin
            chr = read.reference_name
            if chr in refs and idx < len(RDs[chr].RD):
                RDs[chr].RD[idx] += 1

    def gc_correct(RD: np.ndarray, Gc: np.ndarray):
        bin_count = np.bincount(Gc)
        global_rd_ave = np.mean(RD)
        for i in range(len(RD)):
            if bin_count[Gc[i]] < 2:
                continue
            mean = np.mean(RD[Gc == Gc[i]])
            if not math.isclose(mean, 0.0):
                RD[i] *= global_rd_ave / mean
        return RD

    # correct RD
    for chr, rd in RDs.items():
        # delete invalid data
        rd.RD = rd.RD[GCs[chr].valid]
        rd.pos = rd.pos[GCs[chr].valid]
        rd.RD /= bp_per_bin  # read count -> read depth
        rd.RD = gc_correct(rd.RD, GCs[chr].GC[GCs[chr].valid])

        # or fill invalid data
        # left_indices = np.maximum.accumulate(
        #     np.where(GCs[chr].valid, np.arange(len(rd.RD)), 0)
        # )
        # right_indices = np.minimum.accumulate(
        #     np.where(
        #         np.flip(GCs[chr].valid),
        #         np.arange(len(rd.RD) - 1, -1, -1),
        #         len(rd.RD) - 1,
        #     )
        # )[::-1]
        # invalid_idx = ~GCs[chr].valid
        # rd.RD[invalid_idx] = (
        #     rd.RD[left_indices[invalid_idx]] + rd.RD[right_indices[invalid_idx]]
        # ) / 2
        # rd.RD /= bp_per_bin  # read count -> read depth
        # rd.RD = gc_correct(rd.RD, GCs[chr].GC)

    return RDs
