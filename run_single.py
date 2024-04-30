# This is a simple entry file that allows you to directly specify command-line
# parameters to use this tool. However, it only supports processing one
# reference sequence at a time.
# If you have more complex requirements, you can check the details in readme

import argparse

from lmadcnv.binning import binning
from lmadcnv.lmad import LMADCNV

parser = argparse.ArgumentParser()
parser.add_argument("bam", help="the bam file path")
parser.add_argument("contig", help="the reference name in bam file")
parser.add_argument("fa", help="the reference fasta file path")
args = parser.parse_args()

# generate RD profile
data = binning(args.bam, {args.contig: args.fa})

# LMADCNV
CNVs = LMADCNV(data[args.contig])
print(CNVs)
