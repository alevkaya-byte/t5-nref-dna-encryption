

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
import zlib

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    import psutil

    HAVE_PSUTIL = True

except Exception:
    psutil = None
    HAVE_PSUTIL = False


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(
    __file__
).resolve().parent


PLAIN_FILENAME = "ds_5mb.txt"


DATASET_STEM = Path(
    PLAIN_FILENAME
).stem


CIPHER_FILENAME = (
    f"cipher_{DATASET_STEM}.txt"
)


META_FILENAME = (
    f"meta_{DATASET_STEM}.json"
)


MASTER_KEY_FILENAME = (
    "master_key_128.txt"
)


T5_MODULE_FILENAME = (
    "T5_noref.py"
)


ENCRYPTION_CODE_FILENAME = (
    "genome_encrypt.py"
)


SECURITY_CODE_FILENAME = (
    "genome_security_tests.py"
)


OUTPUT_DIRNAME = (
    f"encryption_ablation_{DATASET_STEM}"
)


DEVICE = "cpu"


# 1 kb için 100 uygundur.
# Büyük dosyalarda 20 veya 10 yapılabilir.
AVALANCHE_TRIALS = 100


CPA_CHANGE_RATIO = 0.10


# T5 yeniden üretilmez.
# Yalnız şifreleme çekirdeği tekrar edilir.
TIMING_REPEATS = 100


TEST_RANDOM_SEED = 2026062201


SAVE_CIPHERS = True


REQUIRE_BASELINE_MATCH = True


REQUIRE_EXACT_RECOVERY = True


# =============================================================================
# DNA SABİTLERİ
# =============================================================================

DNA = "ACGT"


ASCII_TO_CODE = np.full(
    256,
    255,
    dtype=np.uint8
)


for _index, _base in enumerate(
    DNA
):

    ASCII_TO_CODE[
        ord(
            _base
        )
    ] = _index


BITCOUNT_2BIT = np.array(
    [
        0,
        1,
        1,
        2
    ],
    dtype=np.uint8
)


# =============================================================================
# ABLATION VARYANTLARI
# =============================================================================

@dataclass(
    frozen=True
)
class Variant:

    code: str

    description: str

    sub: bool

    perm: bool

    diff: bool

    xor: bool

    needed_domains: Tuple[
        str,
        ...
    ]


VARIANTS = (
    Variant(
        "E0_full",

        "SUB + PERM + DIFF + XOR",

        True,
        True,
        True,
        True,

        (
            "KS",
            "SUB",
            "PERM",
            "DIFF"
        )
    ),

    Variant(
        "E1_no_substitution",

        "PERM + DIFF + XOR",

        False,
        True,
        True,
        True,

        (
            "KS",
            "PERM",
            "DIFF"
        )
    ),

    Variant(
        "E2_no_permutation",

        "SUB + DIFF + XOR",

        True,
        False,
        True,
        True,

        (
            "KS",
            "SUB",
            "DIFF"
        )
    ),

    Variant(
        "E3_no_diffusion",

        "SUB + PERM + XOR",

        True,
        True,
        False,
        True,

        (
            "KS",
            "SUB",
            "PERM"
        )
    ),

    Variant(
        "E4_xor_only",

        "Yalnız T5 DNA-XOR",

        False,
        False,
        False,
        True,

        (
            "KS",
        )
    ),

    Variant(
        "E5_spd_no_xor",

        "SUB + PERM + DIFF",

        True,
        True,
        True,
        False,

        (
            "SUB",
            "PERM",
            "DIFF"
        )
    )
)


# =============================================================================
# MODÜL VE DOSYA İŞLEMLERİ
# =============================================================================

def load_module(
    path: Path,
    name: str
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    specification = (
        importlib.util
        .spec_from_file_location(
            name,
            path
        )
    )


    if (
        specification is None
        or specification.loader is None
    ):

        raise ImportError(
            path
        )


    module = (
        importlib.util
        .module_from_spec(
            specification
        )
    )


    sys.modules[
        name
    ] = module


    specification.loader.exec_module(
        module
    )


    return module


def read_json(
    path: Path
) -> dict:

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        value = json.load(
            file
        )


    if not isinstance(
        value,
        dict
    ):

        raise ValueError(
            "JSON kökü nesne olmalıdır."
        )


    return value


def write_json(
    path: Path,
    value: dict
) -> None:

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2
        )

        file.write(
            "\n"
        )


def validate_api(
    encryption,
    security
) -> None:

    encryption_names = (
        "SCHEME",

        "read_dna",

        "validate_master_key",

        "load_t5_module",

        "dna_to_ints",

        "ints_to_dna",

        "substitute",

        "inverse_substitute",

        "make_permutation",

        "permute",

        "inverse_permute",

        "diffuse",

        "inverse_diffuse",

        "dna_xor"
    )


    security_names = (
        "build_session_material",
    )


    missing = [
        f"enc.{name}"
        for name in encryption_names
        if not hasattr(
            encryption,
            name
        )
    ]


    missing += [
        f"sec.{name}"
        for name in security_names
        if not hasattr(
            security,
            name
        )
    ]


    if missing:

        raise AttributeError(
            "Eksik API: "
            + ", ".join(
                missing
            )
        )


# =============================================================================
# VARYANT ŞİFRELEME / DEŞİFRELEME
# =============================================================================

def encrypt_block(
    encryption,
    plain: str,
    keystream: str,
    control,
    variant: Variant
) -> str:

    values = encryption.dna_to_ints(
        plain
    )


    if variant.sub:

        values = encryption.substitute(
            values,
            control.substitution
        )


    if variant.perm:

        permutation = encryption.make_permutation(
            len(
                values
            ),
            control.permutation_seed
        )


        values = encryption.permute(
            values,
            permutation
        )


    if variant.diff:

        values = encryption.diffuse(
            values,
            control
        )


    if variant.xor:

        values = encryption.dna_xor(
            values,

            encryption.dna_to_ints(
                keystream
            )
        )


    return encryption.ints_to_dna(
        values
    )


def decrypt_block(
    encryption,
    cipher: str,
    keystream: str,
    control,
    variant: Variant
) -> str:

    values = encryption.dna_to_ints(
        cipher
    )


    if variant.xor:

        values = encryption.dna_xor(
            values,

            encryption.dna_to_ints(
                keystream
            )
        )


    if variant.diff:

        values = encryption.inverse_diffuse(
            values,
            control
        )


    if variant.perm:

        permutation = encryption.make_permutation(
            len(
                values
            ),
            control.permutation_seed
        )


        values = encryption.inverse_permute(
            values,
            permutation
        )


    if variant.sub:

        values = encryption.inverse_substitute(
            values,
            control.substitution
        )


    return encryption.ints_to_dna(
        values
    )


def transform(
    encryption,
    text: str,
    material,
    variant: Variant,
    decrypt: bool = False
) -> str:

    parts: List[
        str
    ] = []


    for chunk in material.chunks:

        start = int(
            chunk.start
        )


        length = int(
            chunk.length
        )


        current = text[
            start:
            start + length
        ]


        number_of_blocks = math.ceil(
            length
            / material.spd_block_bases
        )


        if len(
            chunk.controls
        ) != number_of_blocks:

            raise ValueError(
                "Kontrol sayısı blok sayısıyla uyuşmuyor."
            )


        for block_index in range(
            number_of_blocks
        ):

            local_start = (
                block_index
                * material.spd_block_bases
            )


            local_end = min(
                local_start
                + material.spd_block_bases,
                length
            )


            if decrypt:

                output = decrypt_block(
                    encryption,

                    current[
                        local_start:
                        local_end
                    ],

                    chunk.keystream[
                        local_start:
                        local_end
                    ],

                    chunk.controls[
                        block_index
                    ],

                    variant
                )


            else:

                output = encrypt_block(
                    encryption,

                    current[
                        local_start:
                        local_end
                    ],

                    chunk.keystream[
                        local_start:
                        local_end
                    ],

                    chunk.controls[
                        block_index
                    ],

                    variant
                )


            parts.append(
                output
            )


    result = "".join(
        parts
    )


    if len(
        result
    ) != len(
        text
    ):

        raise RuntimeError(
            "Dönüşüm uzunluğu hatalı."
        )


    return result


# =============================================================================
# HAMMING VE MUTASYONLAR
# =============================================================================

def base_difference(
    first: str,
    second: str
) -> float:

    if len(
        first
    ) != len(
        second
    ):

        raise ValueError(
            "Uzunluklar eşit olmalıdır."
        )


    return (
        100.0
        * sum(
            first_base
            != second_base
            for first_base, second_base
            in zip(
                first,
                second
            )
        )
        / max(
            len(
                first
            ),
            1
        )
    )


def bit_difference(
    first: str,
    second: str
) -> float:

    if len(
        first
    ) != len(
        second
    ):

        raise ValueError(
            "Uzunluklar eşit olmalıdır."
        )


    first_codes = ASCII_TO_CODE[
        np.frombuffer(
            first.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    second_codes = ASCII_TO_CODE[
        np.frombuffer(
            second.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    differing_bits = int(
        BITCOUNT_2BIT[
            np.bitwise_xor(
                first_codes,
                second_codes
            )
        ].sum()
    )


    return (
        100.0
        * differing_bits
        / max(
            2
            * len(
                first
            ),
            1
        )
    )


def numeric_summary(
    values: Sequence[
        float
    ]
) -> dict:

    return {
        "count": len(
            values
        ),

        "min": float(
            min(
                values
            )
        ),

        "mean": float(
            statistics.fmean(
                values
            )
        ),

        "median": float(
            statistics.median(
                values
            )
        ),

        "max": float(
            max(
                values
            )
        ),

        "sample_std": (
            float(
                statistics.stdev(
                    values
                )
            )
            if len(
                values
            ) > 1
            else 0.0
        )
    }


def other_base(
    base: str,
    random_generator: random.Random
) -> str:

    return random_generator.choice(
        [
            candidate
            for candidate in DNA
            if candidate != base
        ]
    )


def single_mutations(
    plain: str,
    trials: int,
    random_generator: random.Random
):

    output = []


    for _ in range(
        trials
    ):

        position = random_generator.randrange(
            len(
                plain
            )
        )


        output.append(
            (
                position,

                other_base(
                    plain[
                        position
                    ],
                    random_generator
                )
            )
        )


    return output


def ratio_mutation(
    plain: str,
    ratio: float,
    random_generator: random.Random
):

    number_to_change = min(
        len(
            plain
        ),

        max(
            1,

            int(
                round(
                    len(
                        plain
                    )
                    * ratio
                )
            )
        )
    )


    positions = sorted(
        random_generator.sample(
            range(
                len(
                    plain
                )
            ),
            number_to_change
        )
    )


    output = list(
        plain
    )


    for position in positions:

        output[
            position
        ] = other_base(
            output[
                position
            ],
            random_generator
        )


    return (
        "".join(
            output
        ),
        positions
    )


def block_bounds(
    material,
    position: int
) -> Tuple[
    int,
    int
]:

    for chunk in material.chunks:

        start = int(
            chunk.start
        )


        end = (
            start
            + int(
                chunk.length
            )
        )


        if (
            start
            <= position
            < end
        ):

            local_position = (
                position
                - start
            )


            block_index = (
                local_position
                // material.spd_block_bases
            )


            block_start = (
                start
                + block_index
                * material.spd_block_bases
            )


            block_end = min(
                block_start
                + material.spd_block_bases,
                end
            )


            return (
                block_start,
                block_end
            )


    raise ValueError(
        "Pozisyon bulunamadı."
    )


# =============================================================================
# AVALANCHE
# =============================================================================

def avalanche_test(
    encryption,
    plain: str,
    reference_cipher: str,
    material,
    variant: Variant,
    mutations
) -> dict:

    global_base_values = []

    global_bit_values = []

    local_base_values = []

    local_bit_values = []

    changed_cipher_bases = []


    for position, new_base in mutations:

        changed_plain = (
            plain[
                :position
            ]
            + new_base
            + plain[
                position + 1:
            ]
        )


        changed_cipher = transform(
            encryption,
            changed_plain,
            material,
            variant
        )


        block_start, block_end = block_bounds(
            material,
            position
        )


        global_base_values.append(
            base_difference(
                reference_cipher,
                changed_cipher
            )
        )


        global_bit_values.append(
            bit_difference(
                reference_cipher,
                changed_cipher
            )
        )


        local_base_values.append(
            base_difference(
                reference_cipher[
                    block_start:
                    block_end
                ],

                changed_cipher[
                    block_start:
                    block_end
                ]
            )
        )


        local_bit_values.append(
            bit_difference(
                reference_cipher[
                    block_start:
                    block_end
                ],

                changed_cipher[
                    block_start:
                    block_end
                ]
            )
        )


        changed_cipher_bases.append(
            float(
                sum(
                    first_base
                    != second_base
                    for first_base, second_base
                    in zip(
                        reference_cipher,
                        changed_cipher
                    )
                )
            )
        )


    number_of_bases = len(
        plain
    )


    return {
        "trials": len(
            mutations
        ),

        "changed_cipher_bases": numeric_summary(
            changed_cipher_bases
        ),

        "global_base_pct": numeric_summary(
            global_base_values
        ),

        "global_bit_pct": numeric_summary(
            global_bit_values
        ),

        "affected_block_base_pct": numeric_summary(
            local_base_values
        ),

        "affected_block_bit_pct": numeric_summary(
            local_bit_values
        ),

        "position_preserving_reference": {
            "changed_bases": 1,

            "global_base_pct": (
                100.0
                / number_of_bases
            ),

            "global_bit_pct_range": [
                100.0
                / (
                    2
                    * number_of_bases
                ),

                100.0
                / number_of_bases
            ]
        }
    }


# =============================================================================
# CPA
# =============================================================================

def cpa_test(
    encryption,
    plain: str,
    reference_cipher: str,
    material,
    variant: Variant,
    changed_plain: str,
    changed_positions
) -> dict:

    changed_cipher = transform(
        encryption,
        changed_plain,
        material,
        variant
    )


    return {
        "changed_plaintext_bases": len(
            changed_positions
        ),

        "plaintext_base_pct": base_difference(
            plain,
            changed_plain
        ),

        "plaintext_bit_pct": bit_difference(
            plain,
            changed_plain
        ),

        "ciphertext_base_pct": base_difference(
            reference_cipher,
            changed_cipher
        ),

        "ciphertext_bit_pct": bit_difference(
            reference_cipher,
            changed_cipher
        ),

        "same_nonce_and_session_material": True
    }


# =============================================================================
# CIPHERTEXT METRİKLERİ
# =============================================================================

def maximum_homopolymer(
    sequence: str
) -> int:

    if not sequence:

        return 0


    maximum = 1

    current = 1


    for index in range(
        1,
        len(
            sequence
        )
    ):

        if (
            sequence[
                index
            ]
            == sequence[
                index - 1
            ]
        ):

            current += 1


            maximum = max(
                maximum,
                current
            )


        else:

            current = 1


    return maximum


def kmer_array(
    sequence: str,
    k: int
) -> np.ndarray:

    if len(
        sequence
    ) < k:

        return np.zeros(
            4 ** k,
            dtype=np.int64
        )


    codes = ASCII_TO_CODE[
        np.frombuffer(
            sequence.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    number_of_windows = (
        len(
            sequence
        )
        - k
        + 1
    )


    indices = np.zeros(
        number_of_windows,
        dtype=np.int64
    )


    for offset in range(
        k
    ):

        indices = (
            indices
            * 4
            + codes[
                offset:
                offset
                + number_of_windows
            ]
        )


    return np.bincount(
        indices,
        minlength=4 ** k
    )[
        :4 ** k
    ].astype(
        np.int64
    )


def js_divergence_bits(
    first: np.ndarray,
    second: np.ndarray
) -> float:

    midpoint = (
        0.5
        * (
            first
            + second
        )
    )


    def kl_divergence(
        left,
        right
    ):

        mask = (
            (left > 0)
            & (right > 0)
        )


        return float(
            np.sum(
                left[
                    mask
                ]
                * np.log2(
                    left[
                        mask
                    ]
                    / right[
                        mask
                    ]
                )
            )
        )


    return (
        0.5
        * kl_divergence(
            first,
            midpoint
        )
        + 0.5
        * kl_divergence(
            second,
            midpoint
        )
    )


def kmer_metrics(
    counts: np.ndarray
) -> dict:

    total = int(
        counts.sum()
    )


    probabilities = (
        counts.astype(
            np.float64
        )
        / max(
            total,
            1
        )
    )


    uniform = np.full(
        len(
            counts
        ),
        1.0
        / len(
            counts
        )
    )


    expected = (
        total
        / len(
            counts
        )
        if total
        else 0.0
    )


    chi_square = (
        float(
            np.sum(
                (
                    counts
                    - expected
                ) ** 2
                / expected
            )
        )
        if expected
        else 0.0
    )


    return {
        "windows": total,

        "max_abs_deviation": float(
            np.max(
                np.abs(
                    probabilities
                    - uniform
                )
            )
        ),

        "js_divergence_bits": js_divergence_bits(
            probabilities,
            uniform
        ),

        "chi_square": (
            chi_square
        )
    }


def acf_metrics(
    sequence: str,
    max_lag: int = 20
) -> dict:

    codes = ASCII_TO_CODE[
        np.frombuffer(
            sequence.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    max_lag = min(
        max_lag,
        max(
            0,
            len(
                codes
            )
            - 1
        )
    )


    per_base = {}


    global_maximum = 0.0


    for code, base in enumerate(
        DNA
    ):

        values = (
            codes
            == code
        ).astype(
            np.float64
        )


        mean = (
            float(
                values.mean()
            )
            if len(
                values
            )
            else 0.0
        )


        variance = (
            mean
            * (
                1.0
                - mean
            )
        )


        correlations = []


        if variance > 0:

            centered = (
                values
                - mean
            )


            for lag in range(
                1,
                max_lag
                + 1
            ):

                covariance = float(
                    np.dot(
                        centered[
                            lag:
                        ],
                        centered[
                            :-lag
                        ]
                    )
                    / (
                        len(
                            values
                        )
                        - lag
                    )
                )


                correlations.append(
                    covariance
                    / variance
                )


        else:

            correlations = [
                0.0
            ] * max_lag


        current_maximum = max(
            (
                abs(
                    value
                )
                for value in correlations
            ),
            default=0.0
        )


        global_maximum = max(
            global_maximum,
            current_maximum
        )


        per_base[
            base
        ] = {
            "lag1": (
                float(
                    correlations[
                        0
                    ]
                )
                if correlations
                else 0.0
            ),

            "max_abs": float(
                current_maximum
            )
        }


    return {
        "max_lag": max_lag,

        "per_base": per_base,

        "max_abs_all": (
            global_maximum
        )
    }


def pack_rule0(
    sequence: str
) -> bytes:

    codes = ASCII_TO_CODE[
        np.frombuffer(
            sequence.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    bits = np.empty(
        2
        * len(
            codes
        ),
        dtype=np.uint8
    )


    bits[
        0::2
    ] = (
        codes
        >> 1
    ) & 1


    bits[
        1::2
    ] = (
        codes
        & 1
    )


    return np.packbits(
        bits,
        bitorder="big"
    ).tobytes()


def ciphertext_metrics(
    cipher: str
):

    codes = ASCII_TO_CODE[
        np.frombuffer(
            cipher.encode(
                "ascii"
            ),
            dtype=np.uint8
        )
    ]


    counts = np.bincount(
        codes,
        minlength=4
    )[
        :4
    ].astype(
        np.int64
    )


    probabilities = (
        counts.astype(
            np.float64
        )
        / max(
            len(
                cipher
            ),
            1
        )
    )


    nonzero = (
        probabilities > 0
    )


    entropy = float(
        -np.sum(
            probabilities[
                nonzero
            ]
            * np.log2(
                probabilities[
                    nonzero
                ]
            )
        )
    )


    kmer_arrays: Dict[
        int,
        np.ndarray
    ] = {}


    kmer_results = {}


    for k in (
        1,
        2,
        3
    ):

        current_array = kmer_array(
            cipher,
            k
        )


        kmer_arrays[
            k
        ] = current_array


        kmer_results[
            str(
                k
            )
        ] = kmer_metrics(
            current_array
        )


    packed = pack_rule0(
        cipher
    )


    return (
        {
            "length": len(
                cipher
            ),

            "sha256": hashlib.sha256(
                cipher.encode(
                    "ascii"
                )
            ).hexdigest(),

            "counts": {
                base: int(
                    counts[
                        index
                    ]
                )
                for index, base
                in enumerate(
                    DNA
                )
            },

            "probabilities": {
                base: float(
                    probabilities[
                        index
                    ]
                )
                for index, base
                in enumerate(
                    DNA
                )
            },

            "gc_ratio": float(
                (
                    counts[
                        1
                    ]
                    + counts[
                        2
                    ]
                )
                / max(
                    len(
                        cipher
                    ),
                    1
                )
            ),

            "entropy_bits_per_base": (
                entropy
            ),

            "entropy_deficit": (
                2.0
                - entropy
            ),

            "max_homopolymer": maximum_homopolymer(
                cipher
            ),

            "acf": acf_metrics(
                cipher
            ),

            "kmer": (
                kmer_results
            ),

            "compression_ratio": (
                len(
                    zlib.compress(
                        packed,
                        9
                    )
                )
                / max(
                    len(
                        packed
                    ),
                    1
                )
            )
        },

        kmer_arrays
    )


# =============================================================================
# PERFORMANS VE KAYIT
# =============================================================================

def time_core(
    encryption,
    plain: str,
    material,
    variant: Variant
):

    times = []


    last_cipher = ""


    process = (
        psutil.Process(
            os.getpid()
        )
        if HAVE_PSUTIL
        else None
    )


    cpu_start = (
        process.cpu_times().user
        + process.cpu_times().system
        if process
        else time.process_time()
    )


    for _ in range(
        max(
            1,
            TIMING_REPEATS
        )
    ):

        start = time.perf_counter()


        last_cipher = transform(
            encryption,
            plain,
            material,
            variant
        )


        times.append(
            time.perf_counter()
            - start
        )


    cpu_end = (
        process.cpu_times().user
        + process.cpu_times().system
        if process
        else time.process_time()
    )


    timing = numeric_summary(
        times
    )


    return (
        last_cipher,

        {
            "repeats": (
                TIMING_REPEATS
            ),

            "wall_seconds": (
                timing
            ),

            "cpu_seconds_total": float(
                cpu_end
                - cpu_start
            ),

            "throughput_from_mean": (
                len(
                    plain
                )
                / max(
                    timing[
                        "mean"
                    ],
                    1e-12
                )
            ),

            "throughput_from_median": (
                len(
                    plain
                )
                / max(
                    timing[
                        "median"
                    ],
                    1e-12
                )
            ),

            "scope": (
                "Encryption core only; "
                "T5 generation excluded."
            )
        }
    )


def save_kmers(
    output_directory: Path,
    code: str,
    arrays: Dict[
        int,
        np.ndarray
    ]
):

    files = []


    for k, counts in arrays.items():

        path = (
            output_directory
            / f"{code}.k{k}.csv"
        )


        total = int(
            counts.sum()
        )


        with path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow(
                [
                    "kmer",
                    "count",
                    "probability"
                ]
            )


            labels = [
                "".join(
                    symbols
                )
                for symbols
                in itertools.product(
                    DNA,
                    repeat=k
                )
            ]


            for label, count in zip(
                labels,
                counts.tolist()
            ):

                writer.writerow(
                    [
                        label,

                        count,

                        (
                            count
                            / total
                            if total
                            else 0.0
                        )
                    ]
                )


        files.append(
            path.name
        )


    return files


# =============================================================================
# ANA ÇALIŞMA
# =============================================================================

def main() -> None:

    plain_path = (
        BASE_DIR
        / PLAIN_FILENAME
    )


    cipher_path = (
        BASE_DIR
        / CIPHER_FILENAME
    )


    metadata_path = (
        BASE_DIR
        / META_FILENAME
    )


    key_path = (
        BASE_DIR
        / MASTER_KEY_FILENAME
    )


    t5_path = (
        BASE_DIR
        / T5_MODULE_FILENAME
    )


    encryption_path = (
        BASE_DIR
        / ENCRYPTION_CODE_FILENAME
    )


    security_path = (
        BASE_DIR
        / SECURITY_CODE_FILENAME
    )


    output_directory = (
        BASE_DIR
        / OUTPUT_DIRNAME
    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"[PLAIN]      {plain_path}"
    )


    print(
        f"[CIPHER]     {cipher_path}"
    )


    print(
        f"[META]       {metadata_path}"
    )


    print(
        f"[KEY]        {key_path}"
    )


    print(
        f"[T5]         {t5_path}"
    )


    print(
        f"[ENC]        {encryption_path}"
    )


    print(
        f"[SECURITY]   {security_path}"
    )


    print(
        f"[OUTPUT]     {output_directory}"
    )


    total_start = time.perf_counter()


    encryption = load_module(
        encryption_path,
        "enc_ablation_api"
    )


    security = load_module(
        security_path,
        "security_ablation_api"
    )


    validate_api(
        encryption,
        security
    )


    t5_backend = encryption.load_t5_module(
        t5_path
    )


    plain = encryption.read_dna(
        plain_path
    )


    existing_cipher = encryption.read_dna(
        cipher_path
    )


    master_key = encryption.read_dna(
        key_path
    )


    encryption.validate_master_key(
        master_key
    )


    metadata = read_json(
        metadata_path
    )


    if str(
        metadata[
            "scheme"
        ]
    ) != encryption.SCHEME:

        raise ValueError(
            "SCHEME uyuşmuyor."
        )


    if int(
        metadata[
            "input"
        ][
            "canonical_plain_bases"
        ]
    ) != len(
        plain
    ):

        raise ValueError(
            "Plaintext uzunluğu metadata ile uyuşmuyor."
        )


    if len(
        existing_cipher
    ) != len(
        plain
    ):

        raise ValueError(
            "Ciphertext uzunluğu uyuşmuyor."
        )


    if not bool(
        metadata[
            "encryption"
        ][
            "xor_enabled"
        ]
    ):

        raise ValueError(
            "Baseline şifrelemede XOR açık olmalıdır."
        )


    nonce = bytes.fromhex(
        str(
            metadata[
                "session"
            ][
                "nonce_hex"
            ]
        )
    )


    block_size = int(
        metadata[
            "encryption"
        ][
            "spd_block_bases"
        ]
    )


    chunk_size = int(
        metadata[
            "encryption"
        ][
            "t5_chunk_bases"
        ]
    )


    print(
        "\n[SESSION] Aynı T5 oturum materyali "
        "yeniden üretiliyor..."
    )


    material = security.build_session_material(
        enc=encryption,

        t5_backend=t5_backend,

        master_key=master_key,

        nonce=nonce,

        number_of_bases=len(
            plain
        ),

        spd_block_bases=block_size,

        t5_chunk_bases=chunk_size,

        xor_enabled=True,

        device=DEVICE
    )


    print(
        f"[SESSION] model_seed="
        f"{material.model_seed}"
    )


    print(
        f"[SESSION] setup="
        f"{material.setup_seconds:.6f} s"
    )


    print(
        f"[SESSION] generation="
        f"{material.generation_seconds:.6f} s"
    )


    random_generator = random.Random(
        TEST_RANDOM_SEED
    )


    mutations = single_mutations(
        plain,
        AVALANCHE_TRIALS,
        random_generator
    )


    (
        cpa_plain,
        cpa_positions
    ) = ratio_mutation(
        plain,
        CPA_CHANGE_RATIO,
        random_generator
    )


    results = []


    full_cipher: Optional[
        str
    ] = None


    for number, variant in enumerate(
        VARIANTS,
        1
    ):

        print(
            "\n"
            + "=" * 88
        )


        print(
            f"[{number}/{len(VARIANTS)}] "
            f"{variant.code} — "
            f"{variant.description}"
        )


        print(
            "=" * 88
        )


        (
            cipher,
            timing
        ) = time_core(
            encryption,
            plain,
            material,
            variant
        )


        recovered = transform(
            encryption,
            cipher,
            material,
            variant,
            decrypt=True
        )


        recovery_ok = (
            recovered
            == plain
        )


        baseline_match = (
            cipher
            == existing_cipher
        )


        if (
            REQUIRE_EXACT_RECOVERY
            and not recovery_ok
        ):

            raise RuntimeError(
                f"{variant.code}: "
                f"exact recovery başarısız."
            )


        if variant.code == "E0_full":

            full_cipher = cipher


            if (
                REQUIRE_BASELINE_MATCH
                and not baseline_match
            ):

                raise RuntimeError(
                    "E0_full mevcut ciphertext'i üretemedi."
                )


        if full_cipher is None:

            raise RuntimeError(
                "E0_full ilk sırada olmalıdır."
            )


        (
            metrics,
            kmer_arrays
        ) = ciphertext_metrics(
            cipher
        )


        avalanche_result = avalanche_test(
            encryption,
            plain,
            cipher,
            material,
            variant,
            mutations
        )


        cpa_result = cpa_test(
            encryption,
            plain,
            cipher,
            material,
            variant,
            cpa_plain,
            cpa_positions
        )


        files = {}


        if SAVE_CIPHERS:

            target = (
                output_directory
                / (
                    f"{variant.code}_"
                    f"{DATASET_STEM}.cipher.txt"
                )
            )


            target.write_text(
                cipher,
                encoding="ascii"
            )


            files[
                "cipher"
            ] = target.name


        files[
            "kmer_csv"
        ] = save_kmers(
            output_directory,
            variant.code,
            kmer_arrays
        )


        result = {
            "variant": (
                variant.code
            ),

            "description": (
                variant.description
            ),

            "layers": {
                "sub": (
                    variant.sub
                ),

                "perm": (
                    variant.perm
                ),

                "diff": (
                    variant.diff
                ),

                "xor": (
                    variant.xor
                )
            },

            "needed_domains_if_optimized": list(
                variant.needed_domains
            ),

            "same_full_material_used": (
                True
            ),

            "baseline_cipher_match": (
                baseline_match
            ),

            "exact_recovery": (
                recovery_ok
            ),

            "metrics": (
                metrics
            ),

            "plain_cipher_base_pct": base_difference(
                plain,
                cipher
            ),

            "plain_cipher_bit_pct": bit_difference(
                plain,
                cipher
            ),

            "difference_from_full_base_pct": base_difference(
                full_cipher,
                cipher
            ),

            "difference_from_full_bit_pct": bit_difference(
                full_cipher,
                cipher
            ),

            "avalanche": (
                avalanche_result
            ),

            "cpa": (
                cpa_result
            ),

            "timing": (
                timing
            ),

            "files": (
                files
            )
        }


        results.append(
            result
        )


        print(
            f"Exact recovery              : "
            f"{recovery_ok}"
        )


        print(
            f"Baseline cipher match       : "
            f"{baseline_match}"
        )


        print(
            f"A/C/G/T                     : "
            f"{metrics['probabilities']}"
        )


        print(
            f"Entropy                     : "
            f"{metrics['entropy_bits_per_base']:.6f}"
        )


        print(
            f"GC                          : "
            f"{metrics['gc_ratio']:.6f}"
        )


        print(
            f"Max homopolymer             : "
            f"{metrics['max_homopolymer']}"
        )


        print(
            f"Max |ACF|                   : "
            f"{metrics['acf']['max_abs_all']:.6f}"
        )


        print(
            "Avalanche global base       : "
            f"{avalanche_result['global_base_pct']['min']:.6f} / "
            f"{avalanche_result['global_base_pct']['mean']:.6f} / "
            f"{avalanche_result['global_base_pct']['max']:.6f}"
        )


        print(
            "Avalanche affected block    : "
            f"{avalanche_result['affected_block_base_pct']['min']:.6f} / "
            f"{avalanche_result['affected_block_base_pct']['mean']:.6f} / "
            f"{avalanche_result['affected_block_base_pct']['max']:.6f}"
        )


        print(
            f"CPA cipher base/bit         : "
            f"{cpa_result['ciphertext_base_pct']:.6f} / "
            f"{cpa_result['ciphertext_bit_pct']:.6f}"
        )


        print(
            f"Core median time            : "
            f"{timing['wall_seconds']['median']:.9f} s"
        )


        print(
            f"Core median throughput      : "
            f"{timing['throughput_from_median']:.2f} base/s"
        )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    report = {
        "analysis": (
            "T5-NREF DNA-SPD "
            "encryption architecture ablation"
        ),

        "dataset": (
            PLAIN_FILENAME
        ),

        "bases": len(
            plain
        ),

        "nonce_hex": (
            nonce.hex()
        ),

        "block_size": (
            block_size
        ),

        "chunk_size": (
            chunk_size
        ),

        "model_seed": (
            material.model_seed
        ),

        "model_fingerprint": (
            material.model_fingerprint
        ),

        "t5_setup_seconds": (
            material.setup_seconds
        ),

        "t5_material_generation_seconds": (
            material.generation_seconds
        ),

        "design": {
            "same_plaintext": (
                True
            ),

            "same_master_key": (
                True
            ),

            "same_nonce": (
                True
            ),

            "same_ks_sub_perm_diff": (
                True
            ),

            "avalanche_trials": (
                AVALANCHE_TRIALS
            ),

            "cpa_change_ratio": (
                CPA_CHANGE_RATIO
            ),

            "timing_repeats": (
                TIMING_REPEATS
            ),

            "core_timing_excludes_t5": (
                True
            )
        },

        "variants": (
            results
        ),

        "total_seconds": (
            total_seconds
        ),

        "notes": [
            (
                "E3 ve E4 diffusion içermediği için "
                "tek-baz değişimi yerel kalmalıdır."
            ),

            (
                "E0 ve E5 aynı plaintext çifti için aynı "
                "Hamming yayılımını vermelidir; "
                "son XOR bijektif maskedir."
            ),

            (
                "E4 önceki pozisyon-koruyucu "
                "DNA-XOR mimarisinin baseline'ıdır."
            ),

            (
                "Final XOR katkısı yalnız avalanche ile değil "
                "key/nonce sensitivity ve masking ile de "
                "yorumlanmalıdır."
            )
        ]
    }


    json_path = (
        output_directory
        / (
            f"encryption_ablation_"
            f"{DATASET_STEM}.json"
        )
    )


    write_json(
        json_path,
        report
    )


    csv_path = (
        output_directory
        / (
            f"encryption_ablation_"
            f"{DATASET_STEM}.csv"
        )
    )


    fields = [
        "variant",

        "description",

        "sub",

        "perm",

        "diff",

        "xor",

        "exact_recovery",

        "baseline_match",

        "entropy",

        "gc",

        "max_hp",

        "max_acf",

        "k1_js",

        "k2_js",

        "k3_js",

        "plain_cipher_base",

        "plain_cipher_bit",

        "from_full_base",

        "from_full_bit",

        "av_global_base_mean",

        "av_global_bit_mean",

        "av_block_base_mean",

        "av_block_bit_mean",

        "cpa_cipher_base",

        "cpa_cipher_bit",

        "core_median_seconds",

        "core_median_base_per_second"
    ]


    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )


        writer.writeheader()


        for result in results:

            metrics = result[
                "metrics"
            ]


            avalanche_result = result[
                "avalanche"
            ]


            cpa_result = result[
                "cpa"
            ]


            timing = result[
                "timing"
            ]


            writer.writerow({
                "variant": (
                    result[
                        "variant"
                    ]
                ),

                "description": (
                    result[
                        "description"
                    ]
                ),

                "sub": (
                    result[
                        "layers"
                    ][
                        "sub"
                    ]
                ),

                "perm": (
                    result[
                        "layers"
                    ][
                        "perm"
                    ]
                ),

                "diff": (
                    result[
                        "layers"
                    ][
                        "diff"
                    ]
                ),

                "xor": (
                    result[
                        "layers"
                    ][
                        "xor"
                    ]
                ),

                "exact_recovery": (
                    result[
                        "exact_recovery"
                    ]
                ),

                "baseline_match": (
                    result[
                        "baseline_cipher_match"
                    ]
                ),

                "entropy": (
                    metrics[
                        "entropy_bits_per_base"
                    ]
                ),

                "gc": (
                    metrics[
                        "gc_ratio"
                    ]
                ),

                "max_hp": (
                    metrics[
                        "max_homopolymer"
                    ]
                ),

                "max_acf": (
                    metrics[
                        "acf"
                    ][
                        "max_abs_all"
                    ]
                ),

                "k1_js": (
                    metrics[
                        "kmer"
                    ][
                        "1"
                    ][
                        "js_divergence_bits"
                    ]
                ),

                "k2_js": (
                    metrics[
                        "kmer"
                    ][
                        "2"
                    ][
                        "js_divergence_bits"
                    ]
                ),

                "k3_js": (
                    metrics[
                        "kmer"
                    ][
                        "3"
                    ][
                        "js_divergence_bits"
                    ]
                ),

                "plain_cipher_base": (
                    result[
                        "plain_cipher_base_pct"
                    ]
                ),

                "plain_cipher_bit": (
                    result[
                        "plain_cipher_bit_pct"
                    ]
                ),

                "from_full_base": (
                    result[
                        "difference_from_full_base_pct"
                    ]
                ),

                "from_full_bit": (
                    result[
                        "difference_from_full_bit_pct"
                    ]
                ),

                "av_global_base_mean": (
                    avalanche_result[
                        "global_base_pct"
                    ][
                        "mean"
                    ]
                ),

                "av_global_bit_mean": (
                    avalanche_result[
                        "global_bit_pct"
                    ][
                        "mean"
                    ]
                ),

                "av_block_base_mean": (
                    avalanche_result[
                        "affected_block_base_pct"
                    ][
                        "mean"
                    ]
                ),

                "av_block_bit_mean": (
                    avalanche_result[
                        "affected_block_bit_pct"
                    ][
                        "mean"
                    ]
                ),

                "cpa_cipher_base": (
                    cpa_result[
                        "ciphertext_base_pct"
                    ]
                ),

                "cpa_cipher_bit": (
                    cpa_result[
                        "ciphertext_bit_pct"
                    ]
                ),

                "core_median_seconds": (
                    timing[
                        "wall_seconds"
                    ][
                        "median"
                    ]
                ),

                "core_median_base_per_second": (
                    timing[
                        "throughput_from_median"
                    ]
                )
            })


    print(
        "\n"
        + "=" * 88
    )


    print(
        "ŞİFRELEME MİMARİSİ ABLATION TAMAMLANDI"
    )


    print(
        "=" * 88
    )


    print(
        f"JSON : {json_path}"
    )


    print(
        f"CSV  : {csv_path}"
    )


    print(
        f"Süre : {total_seconds:.6f} s"
    )


if __name__ == "__main__":

    main()
