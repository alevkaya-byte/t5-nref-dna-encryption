# T5-NREF DNA Encryption

Research code and reproducibility materials for the T5-NREF controlled,
DNA-local genomic encryption framework.

## Scope

The main T5-NREF component is **training-free and reference-free**: it uses no
pretrained weights, optimization epochs, or reference-genome corpus.
`T5_train.py` is included only as a trained comparison baseline and is not part
of the proposed T5-NREF encryption method.

The term *T5-inspired* refers to the encoder-decoder Transformer structure used
by the implementation. It does not mean that pretrained Google T5 weights are
used.

## Files

| File | Purpose |
|---|---|
| `T5_noref.py` | Main training-free, reference-free DNA generator |
| `T5_train.py` | Trained comparison baseline for the R1/R1-ext/R2 regimes |
| `t5_noref_ablation_multiseed.py` | Multi-seed component ablation for T5-NREF |
| `t5_multi_stream_independence.py` | Pairwise independence analysis across generated streams |
| `t5_kmer_leakage.py` | Exact long-k-mer memorization/leakage analysis |
| `nist_sp800_22.py` | NIST SP 800-22-style statistical test suite |
| `nist_sp800_90b.py` | SP 800-90B-oriented entropy and IID diagnostics |
| `genome_encrypt.py` | T5-NREF DNA-SPD genome encryption |
| `genome_decrypt.py` | Authenticated genome decryption |
| `genome_integrity.py` | SHA-256, BRR, and BCR integrity verification |
| `genome_supplementary_metrics.py` | Supplementary entropy, k-mer, ACF, and comparison metrics |
| `genome_security_tests.py` | Avalanche, key/nonce sensitivity, CPA/KPA-style, and tamper tests |
| `genome_encryption_ablation.py` | Encryption-layer ablation |
| `genome_fault_robustness.py` | Corruption, wrong-key, metadata, and source-error robustness |

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch can run on CPU. A compatible CUDA installation is optional.

## Input data

The scripts use paths relative to the repository directory. Place the required
input files beside the scripts, using the names documented in
[`DATASETS.md`](DATASETS.md). Dataset names can also be changed in the
user-settings block near the top of each script.

Do not commit a real secret key. `genome_encrypt.py` creates
`master_key_128.txt` locally when it is absent; this file is excluded by
`.gitignore`.

## T5-NREF generation

```bash
python T5_noref.py
```

The default run generates 500,000 DNA bases (1,000,000 bits) and stores raw DNA,
bit, rule, binary, and metadata outputs in `outputs/`.

The full multi-seed ablation is computationally expensive:

```bash
python t5_noref_ablation_multiseed.py
```

## Trained comparison baseline

Before running the baseline, set `REAL_PATH`, `OUT_TAG`, and `RUN_SEED` in
`T5_train.py` for the intended regime:

```bash
python T5_train.py
```

This script is a comparator; it must not be described as part of the
training-free T5-NREF method.

## Statistical tests

Pass a generated bit-stream file to each test script:

```bash
python nist_sp800_22.py outputs/<generated-stream>.bits.txt
python nist_sp800_90b.py outputs/<generated-stream>.bits.txt
```

For the multi-stream analyses, prepare the run directories described in
`DATASETS.md`, then run:

```bash
python t5_multi_stream_independence.py
python t5_kmer_leakage.py
```

## Encryption workflow

Place the selected canonical DNA input in the repository directory. The main
scripts default to `ds_tam.txt`; the security and ablation scripts default to
`ds_5mb.txt`.

Run the core workflow in this order:

```bash
python genome_encrypt.py
python genome_decrypt.py
python genome_integrity.py
python genome_supplementary_metrics.py
```

Run the additional security experiments after the baseline encryption files
have been created:

```bash
python genome_security_tests.py
python genome_encryption_ablation.py
python genome_fault_robustness.py
```

## Reproducibility notes

- Record the Python, PyTorch, NumPy, SciPy, hardware, seed, and device details
  used for manuscript experiments.
- Keep the raw JSON/CSV metadata for reported runs in a release or an external
  archival record when the output collection is too large for Git.
- Publish SHA-256 checksums and provenance for every input dataset.
- The code uses deterministic execution settings where applicable, but exact
  timing depends on hardware and software versions.

## Security notice

This repository is research software and has not undergone a production
cryptographic audit. Do not use it to protect sensitive or clinical genomic
data.
