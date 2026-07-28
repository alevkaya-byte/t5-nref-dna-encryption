# Datasets and provenance

This document records the exact fixed inputs used by the T5 comparison regimes
and the genome-encryption scale experiments. All distributed files contain only
uppercase `A`, `C`, `G`, and `T` characters.

Source accessions and sequence identities were re-verified on 2026-07-28.

## Experimental mapping

| Regime or experiment | Input | Data use |
|---|---|---|
| R1 | `real_dna_1m.txt` | Trained real-sequence comparison |
| R1-ext | `real-ext_dna_1m.txt` | Trained external-real-sequence comparison |
| R2 | `synthetic_dna_1m.txt` | Trained synthetic-sequence comparison |
| R3 / T5-NREF | None | Training-free and reference-free generation |
| Genome encryption | `ds_*.txt` | Scale-specific canonical DNA inputs |

T5-NREF does not read any of the real or synthetic sequence files during model
construction or generation.

## R1: real sequence

- Organism: *Escherichia coli* str. K-12 substr. MG1655
- Source: NCBI Nucleotide
- Versioned accession: [U00096.3](https://www.ncbi.nlm.nih.gov/nuccore/U00096.3?report=fasta)
- Source length: 4,641,652 bases
- Processing: remove FASTA formatting, retain the canonical sequence, and take
  the first 1,000,000 bases
- Distributed input: `real_dna_1m.txt`

The supplied full record was checked against U00096.3 and matched exactly after
FASTA line breaks were removed.

## R1-ext: external real sequence

- Organism: *Bacillus subtilis* subsp. *subtilis* str. 168
- Source: NCBI Nucleotide
- Versioned accession: [NC_000964.3](https://www.ncbi.nlm.nih.gov/nuccore/NC_000964.3?report=fasta)
- Source length: 4,215,606 bases
- Processing: retain the canonical sequence and take the first 1,000,000 bases
- Distributed input: `real-ext_dna_1m.txt`

The supplied full record was checked against NC_000964.3 and matched exactly.

## R2: synthetic sequence

`synthetic_dna_1m.txt` is the fixed 1,000,000-base artificial DNA sequence
generated locally with Biopython for the R2 comparison. The exact fixed file is
distributed so that the reported experiment does not depend on regenerating a
new random sequence.

The file contains:

| Base | Count |
|---|---:|
| A | 250,010 |
| C | 250,341 |
| G | 250,092 |
| T | 249,557 |

## Genome-encryption source and preprocessing

- Organism: *Tribolium castaneum* strain Georgia GA2
- Record: linkage group LG4, whole-genome shotgun sequence
- Source: NCBI Nucleotide
- Versioned accession: [CM000279.2](https://www.ncbi.nlm.nih.gov/nuccore/CM000279.2?report=fasta)
- Source record length: 12,290,766 positions
- Canonical `A/C/G/T` bases retained: 11,632,613
- Non-canonical/ambiguous positions removed: 658,153

The canonical sequence was formed as:

```python
canonical = "".join(
    base for base in fasta_sequence.upper()
    if base in "ACGT"
)
```

`ds_tam.txt` is this complete canonical sequence. Each scale-specific file is
an exact prefix of `ds_tam.txt`:

| File | Canonical bases |
|---|---:|
| `ds_1kb.txt` | 1,000 |
| `ds_10kb.txt` | 10,000 |
| `ds_100kb.txt` | 100,000 |
| `ds_1mb.txt` | 1,000,000 |
| `ds_5mb.txt` | 5,000,000 |
| `ds_tam.txt` | 11,632,613 |

CM000279.2 is one linkage-group/chromosome record, not the complete
multi-chromosome *T. castaneum* genome.

## SHA-256 checksums

| File | SHA-256 |
|---|---|
| `real_dna_1m.txt` | `d51bbf6b29a23f043b0d7d7e7d7a816d8971a63d351f70e4e091aecacd32cfb8` |
| `real-ext_dna_1m.txt` | `98ec6d3f9ea2f7297c8a34f462d3313051cb38556d024c9eb63f3026dc5c9063` |
| `synthetic_dna_1m.txt` | `b10e60da853d1388be49755766146abd50caecd4de8e2c69c9ebd9fbea3bacf0` |
| `ds_1kb.txt` | `703f735264f34f0c0be5f9eb4d27452ce6bad3fbbdb9a2ad398709d0f1fa544a` |
| `ds_10kb.txt` | `eda1c7de96b5d34d0811084245312e80e4c7a5321392a5e3bec4590ea44ab1a5` |
| `ds_100kb.txt` | `194705bc58a9744a2924bbfaf881f7b76e0c693b106d1cf578ca2721615f8dbe` |
| `ds_1mb.txt` | `fa19c252e751f5bfe6d3afb5de1fe102159606fea61d453bbd25d33fb2a6aa10` |
| `ds_5mb.txt` | `56faae0c3c94b4e45741c098d873368650cda8c5ac4237028b793a8717d635dc` |
| `ds_tam.txt` | `9802dbd8ddade1f43e186fd4f99ef5e100d481321534bb0af6f9da9cd369953a` |

The two full bacterial source copies are not duplicated in this repository,
because their versioned NCBI records are public and the exact 1,000,000-base
experimental prefixes are distributed.
