<h1 align="center">
LMADCNV
</h1>
<p align="center">
 A CNV Detection Method Based on Local Features and MAD for NGS Data
</p>

## Getting Started

### Dependency Package

- Python >= 3.9
  - numpy
  - pandas
  - scikit-image
  - scipy
  - biopython
  - pysam
  - pyod
  - pythresh
  - rpy2
- R
  - DNAcopy

### Simple usage

Once you have installed the required dependencies, if you wish to detect CNV regions in a sample based on a BAM alignment file and a reference sequence FASTA file, you can use our provided "run_single" file. Simply specify the files using command-line arguments, and you will quickly obtain the desired results.

Example command:

```shell
python run_single.py xxx/test.bam chr21 xxx/chr21.fa
```

Running `python run_single.py --help` provides the following usage instructions:
```shell
usage: run_single.py [-h] bam contig fa

positional arguments:
  bam         the bam file path
  contig      the reference name in bam file
  fa          the reference fasta file path

options:
  -h, --help  show this help message and exit
```

Tips: The provided contig (the name of the reference sequence) must be accurate and correspond to the contigs in the BAM file. Otherwise, the tool will be unable to determine which chromosome you intend to detect.

## Details

### File Structure

```shell
├── lmadcnv         # lmadcnv package
│   ├── binning.py  # preprocessing input files into RD profiles. API: binning
│   └── lmad.py     # main code of the method. API: LMADCNV
├── README.md       # readme
└── run_single.py   # entry file for simple use
```

### API

`binning(bam_path, fa_path, bp_per_bin) -> RDdata`
```python
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
```

`LMADCNV(data, *, bp_per_bin) -> DataFrame`
```python
"""Copy number variations detection.

Parameters
----------
data: RDdata
Generated by `binning` function

bp_per_bin: int
The length of each bin

Returns
-------
result: pd.DataFrame, title is (`start`, `end`, `type`), and the `type` contains (gain|loss)
"""
```

Tips: `binning` supports concurrent processing of multiple reference sequences. You simply need to pass the corresponding contig and FASTA file into the fa_path parameter in dictionary format. This allows you to obtain read depth profile for multiple chromosomes while only reading the BAM file once.

### Input File

- Sequence alignment file in Bam format. (`.bam`)
- Reference sequence file in fasta format. (`.fa` or `.fasta`)

### Output Format

To facilitate further processing of the results, we provide the return value of `LMADCNV` in the format of a `pandas.DataFrame`. If you print it directly, you will see results similar to the following:

```shell
# pd.Dataframe, eg:
       start       end  type
0   32000001  32010000  gain
1   32500001  32510000  gain
2   36000001  36020000  gain
3   36500001  36520000  gain
# ...
```

Each row represents a detected CNV region. The first column is the ID, the second and third columns denote the start and end positions of the region, and the fourth column represents the type of copy number variation.

If you want to save the results to a file, you can use the `to_csv` method from pandas in your code, or you can directly print it in the shell and save it to a file by using redirection.
