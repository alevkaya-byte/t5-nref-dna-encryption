
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
import zlib

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


try:
    import psutil

    HAVE_PSUTIL = True

except Exception:
    psutil = None
    HAVE_PSUTIL = False


try:
    from scipy.stats import chisquare, t as student_t

    HAVE_SCIPY = True

except Exception:
    chisquare = None
    student_t = None
    HAVE_SCIPY = False


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(
    __file__
).resolve().parent


T5_MODULE_FILENAME = (
    "T5_noref.py"
)


OUTPUT_DIRNAME = (
    "t5_noref_ablation_outputs"
)


# Nihai çoklu-seed ablation tasarımı.
# Bilimsel olarak önerilen ayar: 100_000 baz × 5 ortak seed.
# 500_000 baz × 5 seed çalıştırmak istersen yalnızca bu değeri 500_000 yap.
OUTPUT_BASES = 500_000


DEVICE = "cpu"


# Her satır bağımsız bir koşumu temsil eder. Aynı satırdaki model, prompt ve
# generation seed'leri o koşumdaki A0–A8 varyantlarının tamamında aynıdır.
# Böylece varyantlar eşleştirilmiş olarak karşılaştırılır.
RUN_SEED_TRIPLETS = (
    (2026062101, 2026062102, 2026062103),
    (2026062201, 2026062202, 2026062203),
    (2026062301, 2026062302, 2026062303),
    (2026062401, 2026062402, 2026062403),
    (2026062501, 2026062502, 2026062503),
)


# ACF için en büyük gecikme
MAX_ACF_LAG = 20


# k-mer uzunlukları
K_VALUES = (
    1,
    2,
    3
)


VARIANTS = (
    "A0_full",

    "A1_no_logit_calibration",

    "A2_no_uniform_mix",

    "A3_no_soft_balance",

    "A4_no_homopolymer",

    "A5_no_bigram_trigram",

    "A6_no_multilag",

    "A7_raw_t5_only",

    "A8_uniform_no_t5",
)


# DNA ve bit akışlarını kaydet
SAVE_STREAM_FILES = True


# k-mer sayımlarını CSV olarak kaydet
SAVE_KMER_CSV = True


# =============================================================================
# SABİTLER
# =============================================================================

DNA = "ACGT"


DNA_ASCII = np.array(
    [
        ord("A"),
        ord("C"),
        ord("G"),
        ord("T"),
    ],
    dtype=np.uint8,
)


ASCII_TO_CODE = np.full(
    256,
    255,
    dtype=np.uint8,
)


for index, base in enumerate(
    DNA
):

    ASCII_TO_CODE[
        ord(base)
    ] = index


# 2-bit XOR sonuçlarının Hamming ağırlıkları
BITCOUNT_2BIT = np.array(
    [
        0,
        1,
        1,
        2,
    ],
    dtype=np.uint8,
)


# =============================================================================
# T5 MODÜLÜNÜ YÜKLEME
# =============================================================================

def load_module(
    path: Path
):

    if not path.exists():

        raise FileNotFoundError(
            f"T5 modülü bulunamadı: {path}"
        )


    spec = importlib.util.spec_from_file_location(
        "t5_noref_ablation_backend",
        path,
    )


    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"T5 modülü yüklenemedi: {path}"
        )


    module = importlib.util.module_from_spec(
        spec
    )


    sys.modules[
        spec.name
    ] = module


    spec.loader.exec_module(
        module
    )


    return module


def validate_module(
    module
) -> None:

    required = (
        "T5NoRefConfig",

        "T5NoReferenceDNA",

        "seed_all",

        "make_generator",

        "generate_dna",

        "model_fingerprint",

        "ENABLE_BLOCK_LOGIT_CALIBRATION",

        "MODEL_UNIFORM_MIX",

        "ENABLE_SOFT_BALANCE",

        "HOMOPOLYMER_MAX",

        "RECENT_LAG_DAMP",

        "RECENT_LAG_LOG_PENALTIES",

        "MAX_RECENT_LAG",

        "ENABLE_BIGRAM",

        "ENABLE_TRIMER",
    )


    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name,
        )
    ]


    if missing:

        raise AttributeError(
            "T5_noref.py içinde eksik bileşenler: "
            + ", ".join(
                missing
            )
        )


# =============================================================================
# T5 AYARLARINI SAKLAMA VE GERİ YÜKLEME
# =============================================================================

ABLATION_GLOBALS = (
    "ENABLE_BLOCK_LOGIT_CALIBRATION",

    "MODEL_UNIFORM_MIX",

    "ENABLE_SOFT_BALANCE",

    "HOMOPOLYMER_MAX",

    "RECENT_LAG_DAMP",

    "RECENT_LAG_LOG_PENALTIES",

    "MAX_RECENT_LAG",

    "ENABLE_BIGRAM",

    "ENABLE_TRIMER",
)


def capture_settings(
    module
) -> dict:

    output = {}


    for name in ABLATION_GLOBALS:

        value = getattr(
            module,
            name,
        )


        if isinstance(
            value,
            dict,
        ):

            value = dict(
                value
            )


        output[
            name
        ] = value


    return output


def restore_settings(
    module,
    settings: dict
) -> None:

    for name, value in settings.items():

        if isinstance(
            value,
            dict,
        ):

            value = dict(
                value
            )


        setattr(
            module,
            name,
            value,
        )


def disable_multilag(
    module
) -> None:

    module.RECENT_LAG_DAMP = {}


    module.RECENT_LAG_LOG_PENALTIES = ()


    module.MAX_RECENT_LAG = 0


# =============================================================================
# ABLATION VARYANTLARI
# =============================================================================

def apply_variant(
    module,
    variant: str,
    original_settings: dict
) -> dict:

    restore_settings(
        module,
        original_settings,
    )


    descriptions = {
        "A0_full": (
            "Full T5-NREF"
        ),

        "A1_no_logit_calibration": (
            "Block-logit calibration removed"
        ),

        "A2_no_uniform_mix": (
            "Uniform probability mixture removed"
        ),

        "A3_no_soft_balance": (
            "Local/global soft balance removed"
        ),

        "A4_no_homopolymer": (
            "Homopolymer constraint removed"
        ),

        "A5_no_bigram_trigram": (
            "Bigram/trigram corrections removed"
        ),

        "A6_no_multilag": (
            "Multi-lag decorrelation removed"
        ),

        "A7_raw_t5_only": (
            "Raw T5 logits plus categorical sampling only"
        ),

        "A8_uniform_no_t5": (
            "Uniform A/C/G/T sampling without T5"
        ),
    }


    if variant == "A1_no_logit_calibration":

        module.ENABLE_BLOCK_LOGIT_CALIBRATION = False


    elif variant == "A2_no_uniform_mix":

        module.MODEL_UNIFORM_MIX = 0.0


    elif variant == "A3_no_soft_balance":

        module.ENABLE_SOFT_BALANCE = False


    elif variant == "A4_no_homopolymer":

        module.HOMOPOLYMER_MAX = None


    elif variant == "A5_no_bigram_trigram":

        module.ENABLE_BIGRAM = False

        module.ENABLE_TRIMER = False


    elif variant == "A6_no_multilag":

        disable_multilag(
            module
        )


    elif variant == "A7_raw_t5_only":

        module.ENABLE_BLOCK_LOGIT_CALIBRATION = False

        module.MODEL_UNIFORM_MIX = 0.0

        module.ENABLE_SOFT_BALANCE = False

        module.HOMOPOLYMER_MAX = None

        module.ENABLE_BIGRAM = False

        module.ENABLE_TRIMER = False


        disable_multilag(
            module
        )


    elif variant not in descriptions:

        raise ValueError(
            f"Bilinmeyen varyant: {variant}"
        )


    return {
        "description": (
            descriptions[
                variant
            ]
        ),

        "enable_block_logit_calibration": bool(
            module.ENABLE_BLOCK_LOGIT_CALIBRATION
        ),

        "model_uniform_mix": float(
            module.MODEL_UNIFORM_MIX
        ),

        "enable_soft_balance": bool(
            module.ENABLE_SOFT_BALANCE
        ),

        "homopolymer_max": (
            None
            if module.HOMOPOLYMER_MAX is None
            else int(
                module.HOMOPOLYMER_MAX
            )
        ),

        "enable_bigram": bool(
            module.ENABLE_BIGRAM
        ),

        "enable_trimer": bool(
            module.ENABLE_TRIMER
        ),

        "recent_lag_damp": {
            str(
                lag
            ): float(
                damping
            )
            for lag, damping
            in module.RECENT_LAG_DAMP.items()
        },
    }


# =============================================================================
# T5 DNA ÜRETİMİ
# =============================================================================

def generate_t5(
    module,
    model,
    prompt: torch.LongTensor,
    output_length: int,
    generation_seed: int,
) -> str:

    generator = module.make_generator(
        generation_seed,
        DEVICE,
    )


    (
        dna_sequence,
        _rules,
        _dynamic_bits,
    ) = module.generate_dna(
        model=model,

        start_tokens=(
            prompt.clone()
        ),

        output_length=(
            output_length
        ),

        device=DEVICE,

        generator=generator,
    )


    if len(
        dna_sequence
    ) != output_length:

        raise RuntimeError(
            "DNA uzunluğu hatalı: "
            f"{len(dna_sequence)} != {output_length}"
        )


    return dna_sequence


# =============================================================================
# T5 OLMADAN UNIFORM KONTROL
# =============================================================================

def generate_uniform_no_t5(
    output_length: int,
    generation_seed: int,
) -> str:

    generator = torch.Generator(
        device=DEVICE
    )


    generator.manual_seed(
        generation_seed
    )


    codes = torch.randint(
        low=0,

        high=4,

        size=(
            output_length,
        ),

        generator=generator,

        device=DEVICE,

        dtype=torch.int64,
    )


    codes_numpy = (
        codes
        .cpu()
        .numpy()
        .astype(
            np.uint8,
            copy=False,
        )
    )


    return bytes(
        DNA_ASCII[
            codes_numpy
        ]
    ).decode(
        "ascii"
    )


# =============================================================================
# DNA → SABİT RULE-0 BİTLERİ
# =============================================================================

def dna_to_rule0_bits(
    dna_sequence: str
) -> bytes:

    mapping = {
        ord("A"): b"00",

        ord("C"): b"01",

        ord("G"): b"10",

        ord("T"): b"11",
    }


    output = bytearray()


    for byte in dna_sequence.encode(
        "ascii"
    ):

        output.extend(
            mapping[
                byte
            ]
        )


    return bytes(
        output
    )


def pack_ascii_bits(
    bits_ascii: bytes
) -> bytes:

    if not bits_ascii:

        return b""


    bit_array = (
        np.frombuffer(
            bits_ascii,
            dtype=np.uint8,
        )
        - ord("0")
    )


    return np.packbits(
        bit_array,
        bitorder="big",
    ).tobytes()



# =============================================================================
# SABİT RULE-0 BİT ÖZETİ
# =============================================================================

def bit_summary(
    bits_ascii: bytes
) -> dict:

    if not bits_ascii:
        return {
            "length_bits": 0,
            "p_one": None,
            "monobit_p": None,
            "runs_p": None,
        }

    bits = (
        np.frombuffer(
            bits_ascii,
            dtype=np.uint8,
        )
        - ord("0")
    ).astype(
        np.uint8,
        copy=False,
    )

    n = int(
        bits.size
    )

    p_one = float(
        bits.mean()
    )

    signed_sum = int(
        np.sum(
            2 * bits.astype(
                np.int64
            )
            - 1
        )
    )

    s_obs = (
        abs(
            signed_sum
        )
        / math.sqrt(
            n
        )
    )

    monobit_p = float(
        math.erfc(
            s_obs
            / math.sqrt(
                2.0
            )
        )
    )

    tau = (
        2.0
        / math.sqrt(
            n
        )
    )

    if abs(
        p_one
        - 0.5
    ) >= tau:
        runs_p = 0.0

    else:
        number_of_runs = int(
            1
            + np.count_nonzero(
                bits[
                    1:
                ]
                != bits[
                    :-1
                ]
            )
        )

        numerator = abs(
            number_of_runs
            - 2.0
            * n
            * p_one
            * (
                1.0
                - p_one
            )
        )

        denominator = (
            2.0
            * math.sqrt(
                2.0
                * n
            )
            * p_one
            * (
                1.0
                - p_one
            )
        )

        runs_p = float(
            math.erfc(
                numerator
                / max(
                    denominator,
                    1e-300,
                )
            )
        )

    return {
        "length_bits": n,
        "p_one": p_one,
        "monobit_p": monobit_p,
        "runs_p": runs_p,
    }


# =============================================================================
# HOMOPOLİMER
# =============================================================================

def maximum_homopolymer(
    dna_sequence: str
) -> int:

    if not dna_sequence:

        return 0


    maximum = 1

    current = 1


    for index in range(
        1,
        len(
            dna_sequence
        ),
    ):

        if (
            dna_sequence[
                index
            ]
            == dna_sequence[
                index - 1
            ]
        ):

            current += 1


            maximum = max(
                maximum,
                current,
            )


        else:

            current = 1


    return maximum


# =============================================================================
# k-MER
# =============================================================================

def all_kmers(
    k: int
) -> List[str]:

    return [
        "".join(
            symbols
        )
        for symbols
        in itertools.product(
            DNA,
            repeat=k,
        )
    ]


def kmer_count_array(
    dna_sequence: str,
    k: int
) -> np.ndarray:

    number_of_categories = (
        4 ** k
    )


    if len(
        dna_sequence
    ) < k:

        return np.zeros(
            number_of_categories,
            dtype=np.int64,
        )


    ascii_values = np.frombuffer(
        dna_sequence.encode(
            "ascii"
        ),
        dtype=np.uint8,
    )


    codes = ASCII_TO_CODE[
        ascii_values
    ]


    number_of_windows = (
        len(
            dna_sequence
        )
        - k
        + 1
    )


    indices = np.zeros(
        number_of_windows,
        dtype=np.int64,
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
        minlength=number_of_categories,
    )[
        :number_of_categories
    ].astype(
        np.int64
    )


def js_divergence(
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
        p: np.ndarray,
        q: np.ndarray
    ) -> float:

        mask = (
            (p > 0.0)
            & (q > 0.0)
        )


        return float(
            np.sum(
                p[
                    mask
                ]
                * np.log2(
                    p[
                        mask
                    ]
                    / q[
                        mask
                    ]
                )
            )
        )


    return (
        0.5
        * kl_divergence(
            first,
            midpoint,
        )
        + 0.5
        * kl_divergence(
            second,
            midpoint,
        )
    )


def summarize_kmer(
    counts: np.ndarray
) -> dict:

    total = int(
        counts.sum()
    )


    categories = len(
        counts
    )


    if total == 0:

        return {
            "total_windows": 0,

            "max_abs_deviation_from_uniform": None,

            "js_divergence_from_uniform_bits": None,

            "chi_square_uniform": None,

            "p_value": None,
        }


    probabilities = (
        counts.astype(
            np.float64
        )
        / total
    )


    uniform = np.full(
        categories,
        1.0
        / categories,
    )


    expected = np.full(
        categories,
        total
        / categories,
    )


    chi_square_value = float(
        np.sum(
            (
                counts
                - expected
            ) ** 2
            / expected
        )
    )


    p_value = (
        float(
            chisquare(
                counts,
                expected,
            ).pvalue
        )
        if HAVE_SCIPY
        else None
    )


    return {
        "total_windows": (
            total
        ),

        "max_abs_deviation_from_uniform": float(
            np.max(
                np.abs(
                    probabilities
                    - uniform
                )
            )
        ),

        "js_divergence_from_uniform_bits": (
            js_divergence(
                probabilities,
                uniform,
            )
        ),

        "chi_square_uniform": (
            chi_square_value
        ),

        "p_value": (
            p_value
        ),
    }


# =============================================================================
# ACF
# =============================================================================

def acf_summary(
    dna_sequence: str,
    max_lag: int
) -> dict:

    ascii_values = np.frombuffer(
        dna_sequence.encode(
            "ascii"
        ),
        dtype=np.uint8,
    )


    codes = ASCII_TO_CODE[
        ascii_values
    ]


    effective_lag = min(
        max_lag,
        max(
            0,
            len(
                codes
            )
            - 1,
        ),
    )


    output = {
        "per_base": {},

        "maximum_absolute_lag1_all": 0.0,

        "maximum_absolute_acf_all": 0.0,
    }


    for code, base in enumerate(
        DNA
    ):

        indicator = (
            codes
            == code
        ).astype(
            np.float64
        )


        mean = (
            float(
                indicator.mean()
            )
            if len(
                indicator
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


        values = []


        if variance > 0.0:

            centered = (
                indicator
                - mean
            )


            for lag in range(
                1,
                effective_lag
                + 1,
            ):

                covariance = float(
                    np.dot(
                        centered[
                            lag:
                        ],

                        centered[
                            :-lag
                        ],
                    )
                    / (
                        len(
                            centered
                        )
                        - lag
                    )
                )


                values.append(
                    covariance
                    / variance
                )


        else:

            values = [
                0.0
            ] * effective_lag


        maximum_absolute = max(
            (
                abs(
                    value
                )
                for value
                in values
            ),
            default=0.0,
        )


        output[
            "per_base"
        ][
            base
        ] = {
            "lag_1": (
                float(
                    values[
                        0
                    ]
                )
                if values
                else 0.0
            ),

            "maximum_absolute_lag_1_to_L": float(
                maximum_absolute
            ),
        }


        output[
            "maximum_absolute_lag1_all"
        ] = max(
            output[
                "maximum_absolute_lag1_all"
            ],

            abs(
                float(
                    values[
                        0
                    ]
                )
            )
            if values
            else 0.0,
        )


        output[
            "maximum_absolute_acf_all"
        ] = max(
            output[
                "maximum_absolute_acf_all"
            ],

            maximum_absolute,
        )


    return output


# =============================================================================
# DNA ANALİZİ
# =============================================================================

def analyze_dna(
    dna_sequence: str
) -> Tuple[
    dict,
    Dict[int, np.ndarray],
]:

    ascii_values = np.frombuffer(
        dna_sequence.encode(
            "ascii"
        ),
        dtype=np.uint8,
    )


    codes = ASCII_TO_CODE[
        ascii_values
    ]


    counts = np.bincount(
        codes,
        minlength=4,
    )[
        :4
    ].astype(
        np.int64
    )


    probabilities = (
        counts.astype(
            np.float64
        )
        / len(
            dna_sequence
        )
    )


    nonzero = (
        probabilities > 0.0
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


    gc_ratio = float(
        (
            counts[
                1
            ]
            + counts[
                2
            ]
        )
        / len(
            dna_sequence
        )
    )


    kmer_arrays = {}

    kmer_statistics = {}


    for k in K_VALUES:

        current_counts = kmer_count_array(
            dna_sequence,
            k,
        )


        kmer_arrays[
            k
        ] = current_counts


        kmer_statistics[
            str(
                k
            )
        ] = summarize_kmer(
            current_counts
        )


    bits_ascii = dna_to_rule0_bits(
        dna_sequence
    )


    bit_statistics = bit_summary(
        bits_ascii
    )


    packed_bits = pack_ascii_bits(
        bits_ascii
    )


    compressed = zlib.compress(
        packed_bits,
        level=9,
    )


    compression_ratio = (
        len(
            compressed
        )
        / max(
            len(
                packed_bits
            ),
            1,
        )
    )


    acf_statistics = acf_summary(
        dna_sequence,
        MAX_ACF_LAG,
    )


    metrics = {
        "length_bases": (
            len(
                dna_sequence
            )
        ),

        "sha256": hashlib.sha256(
            dna_sequence.encode(
                "ascii"
            )
        ).hexdigest(),

        "base_counts": {
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

        "base_probabilities": {
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

        "gc_ratio": (
            gc_ratio
        ),

        "entropy_bits_per_base": (
            entropy
        ),

        "entropy_deficit": float(
            2.0
            - entropy
        ),

        "maximum_homopolymer": (
            maximum_homopolymer(
                dna_sequence
            )
        ),

        "compression_ratio_packed_rule0": float(
            compression_ratio
        ),

        "bit_rule0": (
            bit_statistics
        ),

        "kmer": (
            kmer_statistics
        ),

        "acf": (
            acf_statistics
        ),
    }


    return (
        metrics,
        kmer_arrays,
    )


# =============================================================================
# İKİ DNA DİZİSİ ARASINDAKİ FARK
# =============================================================================

def compare_sequences(
    first: str,
    second: str
) -> dict:

    first_ascii = np.frombuffer(
        first.encode(
            "ascii"
        ),
        dtype=np.uint8,
    )


    second_ascii = np.frombuffer(
        second.encode(
            "ascii"
        ),
        dtype=np.uint8,
    )


    if len(
        first_ascii
    ) != len(
        second_ascii
    ):

        raise ValueError(
            "Ablation akış uzunlukları eşit değil."
        )


    first_codes = ASCII_TO_CODE[
        first_ascii
    ]


    second_codes = ASCII_TO_CODE[
        second_ascii
    ]


    base_differences = int(
        np.count_nonzero(
            first_ascii
            != second_ascii
        )
    )


    xor_values = np.bitwise_xor(
        first_codes,
        second_codes,
    )


    bit_differences = int(
        BITCOUNT_2BIT[
            xor_values
        ].sum()
    )


    length = len(
        first_ascii
    )


    return {
        "base_difference_percent": (
            100.0
            * base_differences
            / length
        ),

        "bit_difference_percent_2bit": (
            100.0
            * bit_differences
            / (
                2
                * length
            )
        ),
    }


# =============================================================================
# DOSYA KAYITLARI
# =============================================================================

def save_streams(
    output_directory: Path,
    variant: str,
    dna_sequence: str
) -> dict:

    dna_path = (
        output_directory
        / f"{variant}.dna.txt"
    )


    bits_path = (
        output_directory
        / f"{variant}.rule0.bits.txt"
    )


    binary_path = (
        output_directory
        / f"{variant}.rule0.bin"
    )


    bits_ascii = dna_to_rule0_bits(
        dna_sequence
    )


    dna_path.write_text(
        dna_sequence,
        encoding="ascii",
    )


    bits_path.write_bytes(
        bits_ascii
    )


    binary_path.write_bytes(
        pack_ascii_bits(
            bits_ascii
        )
    )


    return {
        "dna": (
            dna_path.name
        ),

        "rule0_bits_ascii": (
            bits_path.name
        ),

        "rule0_bits_packed": (
            binary_path.name
        ),
    }


def save_kmer_csv(
    output_directory: Path,
    variant: str,
    arrays: Dict[int, np.ndarray]
) -> List[str]:

    output_files = []


    for k, counts in arrays.items():

        path = (
            output_directory
            / f"{variant}.k{k}.csv"
        )


        total = int(
            counts.sum()
        )


        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:

            writer = csv.writer(
                handle
            )


            writer.writerow(
                [
                    "kmer",
                    "count",
                    "probability",
                ]
            )


            for label, count in zip(
                all_kmers(
                    k
                ),

                counts.tolist(),
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
                        ),
                    ]
                )


        output_files.append(
            path.name
        )


    return output_files


# =============================================================================
# ANA AKIŞ
# =============================================================================

def t_critical_95(
    sample_size: int
) -> float:

    if sample_size <= 1:
        return 0.0

    if HAVE_SCIPY and student_t is not None:
        return float(
            student_t.ppf(
                0.975,
                df=(
                    sample_size
                    - 1
                ),
            )
        )

    fallback = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }

    return fallback.get(
        sample_size,
        1.96,
    )


def summarize_values(
    values: List[float]
) -> dict:

    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None
            and np.isfinite(
                float(
                    value
                )
            )
        ],
        dtype=np.float64,
    )

    n = int(
        clean.size
    )

    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "minimum": None,
            "maximum": None,
        }

    mean = float(
        clean.mean()
    )

    sd = float(
        clean.std(
            ddof=1
        )
    ) if n > 1 else 0.0

    half_width = (
        t_critical_95(
            n
        )
        * sd
        / math.sqrt(
            n
        )
        if n > 1
        else 0.0
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": float(
            mean
            - half_width
        ),
        "ci95_high": float(
            mean
            + half_width
        ),
        "minimum": float(
            clean.min()
        ),
        "maximum": float(
            clean.max()
        ),
    }


def flatten_result(
    run_index: int,
    seeds: dict,
    result: dict
) -> dict:

    metrics = result[
        "metrics"
    ]

    difference = result[
        "difference_from_full"
    ]

    return {
        "run_index": run_index,
        "model_seed": seeds[
            "model_seed"
        ],
        "prompt_seed": seeds[
            "prompt_seed"
        ],
        "generation_seed": seeds[
            "generation_seed"
        ],
        "variant": result[
            "variant"
        ],
        "description": result[
            "settings"
        ][
            "description"
        ],
        "gc_ratio": metrics[
            "gc_ratio"
        ],
        "entropy_bits_per_base": metrics[
            "entropy_bits_per_base"
        ],
        "maximum_homopolymer": metrics[
            "maximum_homopolymer"
        ],
        "bit_p_one_rule0": metrics[
            "bit_rule0"
        ][
            "p_one"
        ],
        "monobit_p_rule0": metrics[
            "bit_rule0"
        ][
            "monobit_p"
        ],
        "runs_p_rule0": metrics[
            "bit_rule0"
        ][
            "runs_p"
        ],
        "maximum_abs_lag1": metrics[
            "acf"
        ][
            "maximum_absolute_lag1_all"
        ],
        "maximum_abs_acf_lag1_to_20": metrics[
            "acf"
        ][
            "maximum_absolute_acf_all"
        ],
        "k1_max_deviation": metrics[
            "kmer"
        ][
            "1"
        ][
            "max_abs_deviation_from_uniform"
        ],
        "k1_js": metrics[
            "kmer"
        ][
            "1"
        ][
            "js_divergence_from_uniform_bits"
        ],
        "k2_max_deviation": metrics[
            "kmer"
        ][
            "2"
        ][
            "max_abs_deviation_from_uniform"
        ],
        "k2_js": metrics[
            "kmer"
        ][
            "2"
        ][
            "js_divergence_from_uniform_bits"
        ],
        "k3_max_deviation": metrics[
            "kmer"
        ][
            "3"
        ][
            "max_abs_deviation_from_uniform"
        ],
        "k3_js": metrics[
            "kmer"
        ][
            "3"
        ][
            "js_divergence_from_uniform_bits"
        ],
        "compression_ratio": metrics[
            "compression_ratio_packed_rule0"
        ],
        "wall_seconds": result[
            "performance"
        ][
            "wall_seconds"
        ],
        "cpu_seconds": result[
            "performance"
        ][
            "cpu_seconds"
        ],
        "rss_delta_mb": result[
            "performance"
        ][
            "rss_delta_mb"
        ],
        "base_per_second": result[
            "performance"
        ][
            "base_per_second"
        ],
        "base_difference_from_full_percent": (
            None
            if difference is None
            else difference[
                "base_difference_percent"
            ]
        ),
        "bit_difference_from_full_percent": (
            None
            if difference is None
            else difference[
                "bit_difference_percent_2bit"
            ]
        ),
    }


def aggregate_per_variant(
    per_run_rows: List[dict]
) -> List[dict]:

    metric_names = (
        "gc_ratio",
        "entropy_bits_per_base",
        "maximum_homopolymer",
        "bit_p_one_rule0",
        "monobit_p_rule0",
        "runs_p_rule0",
        "maximum_abs_lag1",
        "maximum_abs_acf_lag1_to_20",
        "k1_max_deviation",
        "k1_js",
        "k2_max_deviation",
        "k2_js",
        "k3_max_deviation",
        "k3_js",
        "compression_ratio",
        "wall_seconds",
        "cpu_seconds",
        "rss_delta_mb",
        "base_per_second",
        "base_difference_from_full_percent",
        "bit_difference_from_full_percent",
    )

    aggregate_rows = []

    for variant in VARIANTS:
        current = [
            row
            for row in per_run_rows
            if row[
                "variant"
            ] == variant
        ]

        if not current:
            continue

        for metric_name in metric_names:
            summary = summarize_values(
                [
                    row.get(
                        metric_name
                    )
                    for row in current
                ]
            )

            aggregate_rows.append({
                "variant": variant,
                "description": current[
                    0
                ][
                    "description"
                ],
                "metric": metric_name,
                **summary,
            })

    return aggregate_rows


def write_csv(
    path: Path,
    rows: List[dict]
) -> None:

    if not rows:
        return

    fields = list(
        rows[
            0
        ].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:

    module_path = (
        BASE_DIR
        / T5_MODULE_FILENAME
    )

    output_directory = (
        BASE_DIR
        / OUTPUT_DIRNAME
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[T5_MODULE]    {module_path}"
    )
    print(
        f"[OUTPUT_DIR]   {output_directory}"
    )
    print(
        f"[OUTPUT_BASES] {OUTPUT_BASES:,}"
    )
    print(
        f"[RUN_COUNT]    {len(RUN_SEED_TRIPLETS)}"
    )
    print(
        f"[DEVICE]       {DEVICE}"
    )

    module = load_module(
        module_path
    )

    validate_module(
        module
    )

    original_settings = capture_settings(
        module
    )

    process = (
        psutil.Process(
            os.getpid()
        )
        if HAVE_PSUTIL
        else None
    )

    all_run_reports = []
    per_run_rows = []
    total_start = time.perf_counter()

    for run_index, seed_triplet in enumerate(
        RUN_SEED_TRIPLETS,
        start=1,
    ):
        (
            model_seed,
            prompt_seed,
            generation_seed,
        ) = seed_triplet

        seeds = {
            "model_seed": int(
                model_seed
            ),
            "prompt_seed": int(
                prompt_seed
            ),
            "generation_seed": int(
                generation_seed
            ),
        }

        run_directory = (
            output_directory
            / f"run_{run_index:02d}_seed_{generation_seed}"
        )
        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\n"
            + "#" * 92
        )
        print(
            f"RUN {run_index}/{len(RUN_SEED_TRIPLETS)} | "
            f"model={model_seed} | prompt={prompt_seed} | "
            f"generation={generation_seed}"
        )
        print(
            "#" * 92
        )

        restore_settings(
            module,
            original_settings,
        )

        module.seed_all(
            model_seed
        )

        config = module.T5NoRefConfig()

        model = (
            module
            .T5NoReferenceDNA(
                config
            )
            .to(
                DEVICE
            )
            .eval()
        )

        model_fingerprint = module.model_fingerprint(
            model
        )

        prompt_generator = module.make_generator(
            prompt_seed,
            DEVICE,
        )

        prompt = torch.randint(
            low=0,
            high=4,
            size=(
                1,
                int(
                    config.source_len
                ),
            ),
            generator=prompt_generator,
            device=DEVICE,
            dtype=torch.long,
        )

        run_results = []
        full_sequence: Optional[
            str
        ] = None
        run_start = time.perf_counter()

        for variant_number, variant in enumerate(
            VARIANTS,
            start=1,
        ):
            print(
                "\n"
                + "=" * 88
            )
            print(
                f"[RUN {run_index} | {variant_number}/{len(VARIANTS)}] "
                f"{variant}"
            )
            print(
                "=" * 88
            )

            settings = apply_variant(
                module,
                variant,
                original_settings,
            )

            gc.collect()

            if (
                DEVICE.startswith(
                    "cuda"
                )
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            rss_start = (
                process.memory_info().rss
                if process
                else None
            )

            cpu_start = (
                (
                    process.cpu_times().user
                    + process.cpu_times().system
                )
                if process
                else time.process_time()
            )

            wall_start = time.perf_counter()

            if variant == "A8_uniform_no_t5":
                dna_sequence = generate_uniform_no_t5(
                    OUTPUT_BASES,
                    generation_seed,
                )
            else:
                dna_sequence = generate_t5(
                    module,
                    model,
                    prompt,
                    OUTPUT_BASES,
                    generation_seed,
                )

            if (
                DEVICE.startswith(
                    "cuda"
                )
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize()

            wall_seconds = (
                time.perf_counter()
                - wall_start
            )

            cpu_end = (
                (
                    process.cpu_times().user
                    + process.cpu_times().system
                )
                if process
                else time.process_time()
            )

            rss_end = (
                process.memory_info().rss
                if process
                else None
            )

            (
                metrics,
                kmer_arrays,
            ) = analyze_dna(
                dna_sequence
            )

            if variant == "A0_full":
                full_sequence = dna_sequence
                difference_from_full = None
            else:
                if full_sequence is None:
                    raise RuntimeError(
                        "A0_full ilk varyant olarak çalışmalıdır."
                    )

                difference_from_full = compare_sequences(
                    full_sequence,
                    dna_sequence,
                )

            files = {}

            if SAVE_STREAM_FILES:
                files.update(
                    save_streams(
                        run_directory,
                        variant,
                        dna_sequence,
                    )
                )

            if SAVE_KMER_CSV:
                files[
                    "kmer_csv"
                ] = save_kmer_csv(
                    run_directory,
                    variant,
                    kmer_arrays,
                )

            result = {
                "run_index": run_index,
                "seeds": seeds,
                "variant": variant,
                "settings": settings,
                "metrics": metrics,
                "difference_from_full": difference_from_full,
                "performance": {
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": (
                        cpu_end
                        - cpu_start
                    ),
                    "rss_delta_mb": (
                        None
                        if rss_start is None
                        or rss_end is None
                        else (
                            rss_end
                            - rss_start
                        )
                        / (
                            1024 ** 2
                        )
                    ),
                    "base_per_second": (
                        OUTPUT_BASES
                        / max(
                            wall_seconds,
                            1e-12,
                        )
                    ),
                },
                "files": files,
            }

            run_results.append(
                result
            )

            per_run_rows.append(
                flatten_result(
                    run_index,
                    seeds,
                    result,
                )
            )

            print(
                f"GC oranı                   : "
                f"{metrics['gc_ratio']:.6f}"
            )
            print(
                f"Entropi                    : "
                f"{metrics['entropy_bits_per_base']:.6f} bit/baz"
            )
            print(
                f"Maksimum homopolimer       : "
                f"{metrics['maximum_homopolymer']}"
            )
            print(
                f"Rule-0 bit p(1)            : "
                f"{metrics['bit_rule0']['p_one']:.6f}"
            )
            print(
                f"Rule-0 Monobit / Runs p    : "
                f"{metrics['bit_rule0']['monobit_p']:.6g} / "
                f"{metrics['bit_rule0']['runs_p']:.6g}"
            )
            print(
                f"Maksimum |lag-1 ACF|       : "
                f"{metrics['acf']['maximum_absolute_lag1_all']:.6f}"
            )
            print(
                f"Maksimum |ACF| (1–{MAX_ACF_LAG}) : "
                f"{metrics['acf']['maximum_absolute_acf_all']:.6f}"
            )
            print(
                f"Üretim süresi              : "
                f"{wall_seconds:.6f} s"
            )
            print(
                f"Throughput                 : "
                f"{result['performance']['base_per_second']:.2f} baz/s"
            )

            for k in K_VALUES:
                current_stats = metrics[
                    "kmer"
                ][
                    str(
                        k
                    )
                ]

                print(
                    f"k={k} Δmax / JS              : "
                    f"{current_stats['max_abs_deviation_from_uniform']:.6f} / "
                    f"{current_stats['js_divergence_from_uniform_bits']:.6f}"
                )

            if difference_from_full is not None:
                print(
                    f"Full'e göre baz farkı      : "
                    f"{difference_from_full['base_difference_percent']:.6f}%"
                )
                print(
                    f"Full'e göre 2-bit farkı    : "
                    f"{difference_from_full['bit_difference_percent_2bit']:.6f}%"
                )

            del dna_sequence
            gc.collect()

        restore_settings(
            module,
            original_settings,
        )

        run_wall_seconds = (
            time.perf_counter()
            - run_start
        )

        run_report = {
            "run_index": run_index,
            "seeds": seeds,
            "model_fingerprint_sha256": model_fingerprint,
            "output_bases_per_variant": OUTPUT_BASES,
            "variants": run_results,
            "run_wall_seconds": run_wall_seconds,
        }

        run_json_path = (
            run_directory
            / "run_summary.json"
        )
        run_json_path.write_text(
            json.dumps(
                run_report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        all_run_reports.append(
            run_report
        )

        del model
        del prompt
        gc.collect()

        if (
            DEVICE.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    restore_settings(
        module,
        original_settings,
    )

    aggregate_rows = aggregate_per_variant(
        per_run_rows
    )

    per_run_csv_path = (
        output_directory
        / "t5_noref_ablation_per_run.csv"
    )
    aggregate_csv_path = (
        output_directory
        / "t5_noref_ablation_aggregate.csv"
    )
    final_json_path = (
        output_directory
        / "t5_noref_ablation_multiseed_summary.json"
    )

    write_csv(
        per_run_csv_path,
        per_run_rows,
    )
    write_csv(
        aggregate_csv_path,
        aggregate_rows,
    )

    total_wall_seconds = (
        time.perf_counter()
        - total_start
    )

    final_report = {
        "analysis": "BMC T5-NREF multi-seed generator ablation",
        "scope": (
            "T5 generator and constraint ablation; "
            "not SPD encryption-layer ablation"
        ),
        "t5_module": T5_MODULE_FILENAME,
        "output_bases_per_variant_per_run": OUTPUT_BASES,
        "run_count": len(
            RUN_SEED_TRIPLETS
        ),
        "device": DEVICE,
        "seed_triplets": [
            {
                "model_seed": int(
                    values[
                        0
                    ]
                ),
                "prompt_seed": int(
                    values[
                        1
                    ]
                ),
                "generation_seed": int(
                    values[
                        2
                    ]
                ),
            }
            for values in RUN_SEED_TRIPLETS
        ],
        "rule0_mapping": {
            "A": "00",
            "C": "01",
            "G": "10",
            "T": "11",
        },
        "runs": all_run_reports,
        "aggregate": aggregate_rows,
        "total_wall_seconds": total_wall_seconds,
    }

    final_json_path.write_text(
        json.dumps(
            final_report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 92
    )
    print(
        "T5-NREF ÇOKLU-SEED ABLATION TAMAMLANDI"
    )
    print(
        "=" * 92
    )
    print(
        f"Koşum-bazlı CSV : {per_run_csv_path}"
    )
    print(
        f"Toplu özet CSV  : {aggregate_csv_path}"
    )
    print(
        f"Nihai JSON      : {final_json_path}"
    )
    print(
        f"Toplam süre     : {total_wall_seconds:.6f} s"
    )
    print(
        "NIST/Diehard için her run klasöründeki "
        "*.rule0.bits.txt veya *.rule0.bin dosyalarını kullan."
    )
    print(
        "=" * 92
    )


if __name__ == "__main__":
    main()
