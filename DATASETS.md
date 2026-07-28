# Dataset layout and provenance

Input DNA files must contain canonical `A`, `C`, `G`, and `T` bases. FASTA
headers and non-canonical symbols are handled only where explicitly documented
by a script.

## Reference and comparison inputs

| Expected name | Role | Expected canonical length |
|---|---|---:|
| `real_dna_1m.txt` | R1 reference/training sequence | 1,000,000 bases |
| `real-ext_dna_1m.txt` | R1-ext reference/training sequence; default input for `T5_train.py` | 1,000,000 bases |
| `synthetic_dna_1m.txt` | R2 synthetic reference/training sequence | 1,000,000 bases |

The k-mer script also recognizes several historical aliases, but the names
above should be used in the public repository.

## Genome-encryption inputs

The experiment scripts recognize the following scale-specific files:

| Expected name | Intended scale |
|---|---:|
| `ds_1kb.txt` | 1 KB-scale subset |
| `ds_10kb.txt` | 10 KB-scale subset |
| `ds_100kb.txt` | 100 KB-scale subset |
| `ds_1mb.txt` | 1 MB-scale subset |
| `ds_5mb.txt` | 5 MB-scale subset |
| `ds_tam.txt` | Complete processed sequence |

These files should be derived by a documented deterministic preprocessing
procedure. Do not describe their size from the filename alone; report the
actual canonical base count and byte count.

## Multi-stream run directories

`t5_multi_stream_independence.py` expects 10 runs for each regime:

```text
T5_R1.1 ... T5_R1.10
T5_R1-ext.1 ... T5_R1-ext.10
T5_R2.1 ... T5_R2.10
T5_R3.1 ... T5_R3.10
```

Each directory should contain a matched metadata/DNA/bit bundle:

```text
<run-name>.json
<run-name>.dna.txt
<run-name>.bits.txt
```

The expected generated lengths are 500,000 DNA bases and 1,000,000 bits per
run.

## Required provenance record

Before publishing data, document the following for every source or derived
file:

| Field | Required information |
|---|---|
| Source | Database, repository, or generation procedure |
| Identifier | Accession number, DOI, or stable URL |
| Retrieval date | Date on which the source was downloaded |
| License/terms | Redistribution permission or citation requirement |
| Preprocessing | Filtering, canonicalization, slicing, and random seeds |
| Canonical length | Number of retained `A/C/G/T` bases |
| File size | Exact number of bytes |
| SHA-256 | Checksum of the distributed file |

Large generated-stream collections are better distributed through a versioned
GitHub release or an archival service such as Zenodo, with the persistent link
and checksums recorded here.
