
import hashlib
import itertools
import json
import math
import os
import time

from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


try:
    from scipy.stats import chi2

    HAVE_SCIPY = True

except Exception:
    chi2 = None
    HAVE_SCIPY = False


try:
    import psutil

    HAVE_PSUTIL = True

except Exception:
    psutil = None
    HAVE_PSUTIL = False


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# Sadece bu adı değiştir:
#
# ds_1kb.txt
# ds_10kb.txt
# ds_100kb.txt
# ds_1mb.txt
# ds_5mb.txt
# ds_tam.txt
#
ORIGINAL_FILENAME = "ds_tam.txt"


DATASET_STEM = Path(
    ORIGINAL_FILENAME
).stem


CIPHER_FILENAME = (
    f"cipher_{DATASET_STEM}.txt"
)


DECRYPTED_FILENAME = (
    f"decrypted_{DATASET_STEM}.txt"
)


ENCRYPTION_METADATA_FILENAME = (
    f"meta_{DATASET_STEM}.json"
)


DECRYPTION_METADATA_FILENAME = (
    f"decrypt_meta_{DATASET_STEM}.json"
)


REPORT_FILENAME = (
    f"genom_ekler_{DATASET_STEM}.json"
)


CSV_PREFIX = (
    f"genom_ekler_{DATASET_STEM}"
)


# Analiz edilecek k-mer uzunlukları
K_VALUES: Tuple[int, ...] = (
    1,
    2,
    3
)


# ACF için en büyük gecikme
MAX_LAG = 20


# Büyük DNA dosyaları bu uzunlukta parçalar hâlinde okunur.
STREAM_CHUNK_BASES = 1_000_000


# ACF, ilk bu kadar baz üzerinde hesaplanır.
# 1 kb, 10 kb ve 100 kb dosyalarında dosyanın tamamı kullanılır.
ACF_SAMPLE_BASES = 1_000_000


# k-mer CSV dosyalarını oluştur.
SAVE_KMER_CSV = True


# Deşifrelenmiş dosya varsa onu da analiz et.
ANALYZE_DECRYPTED = True


# =============================================================================
# SABİTLER
# =============================================================================

DNA = "ACGT"


DNA_BYTES = {
    ord("A"),
    ord("C"),
    ord("G"),
    ord("T")
}


ASCII_TO_CODE = np.full(
    256,
    255,
    dtype=np.uint8
)


ASCII_TO_CODE[
    ord("A")
] = 0


ASCII_TO_CODE[
    ord("C")
] = 1


ASCII_TO_CODE[
    ord("G")
] = 2


ASCII_TO_CODE[
    ord("T")
] = 3


# XOR sonucu 0–3 için 2-bit Hamming ağırlığı:
#
# 0 = 00 → 0 bit farklı
# 1 = 01 → 1 bit farklı
# 2 = 10 → 1 bit farklı
# 3 = 11 → 2 bit farklı
#
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
# DOSYA OKUMA
# =============================================================================

def iter_dna_chunks(
    path: Path,
    chunk_bases: int
) -> Iterator[bytes]:
    """
    FASTA başlıklarını atar.

    Yalnızca A/C/G/T bazlarını tutar.

    DNA dizisini sabit büyüklükte parçalar hâlinde döndürür.
    """

    if chunk_bases <= 0:

        raise ValueError(
            "chunk_bases pozitif olmalıdır."
        )


    if not path.exists():

        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}"
        )


    buffer = bytearray()


    with path.open(
        "rb"
    ) as handle:

        for line in handle:

            if line.startswith(
                b">"
            ):

                continue


            for byte in line.upper():

                if byte in DNA_BYTES:

                    buffer.append(
                        byte
                    )


                    if (
                        len(buffer)
                        == chunk_bases
                    ):

                        yield bytes(
                            buffer
                        )

                        buffer.clear()


    if buffer:

        yield bytes(
            buffer
        )


def read_json_optional(
    path: Path
) -> Optional[dict]:

    if not path.exists():

        return None


    with path.open(
        "r",
        encoding="utf-8"
    ) as handle:

        data = json.load(
            handle
        )


    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            f"JSON kökü nesne olmalıdır: "
            f"{path}"
        )


    return data


# =============================================================================
# HOMOPOLİMER
# =============================================================================

def finalize_run(
    histograms: List[Counter],
    base_code: int,
    run_length: int
) -> None:

    if run_length > 0:

        histograms[
            base_code
        ][
            run_length
        ] += 1


def histogram_percentile(
    histogram: Counter,
    q: float
) -> float:

    total = sum(
        histogram.values()
    )


    if total == 0:

        return 0.0


    target = (
        q
        * total
    )


    cumulative = 0


    for length in sorted(
        histogram
    ):

        cumulative += (
            histogram[
                length
            ]
        )


        if cumulative >= target:

            return float(
                length
            )


    return float(
        max(
            histogram
        )
    )


def summarize_runs(
    histograms: List[Counter]
) -> dict:

    result: Dict[
        str,
        dict
    ] = {}


    combined = Counter()


    for base_code, base in enumerate(
        DNA
    ):

        histogram = (
            histograms[
                base_code
            ]
        )


        combined.update(
            histogram
        )


        count = sum(
            histogram.values()
        )


        total_length = sum(
            length * frequency
            for length, frequency
            in histogram.items()
        )


        result[
            base
        ] = {
            "run_count": int(
                count
            ),

            "maximum_run": int(
                max(
                    histogram
                )
                if histogram
                else 0
            ),

            "mean_run": (
                float(
                    total_length
                    / count
                )
                if count
                else 0.0
            ),

            "p95_run": (
                histogram_percentile(
                    histogram,
                    0.95
                )
            ),

            "runs_ge_6": int(
                sum(
                    frequency
                    for length, frequency
                    in histogram.items()
                    if length >= 6
                )
            )
        }


    count = sum(
        combined.values()
    )


    total_length = sum(
        length * frequency
        for length, frequency
        in combined.items()
    )


    result[
        "_all"
    ] = {
        "run_count": int(
            count
        ),

        "maximum_run": int(
            max(
                combined
            )
            if combined
            else 0
        ),

        "mean_run": (
            float(
                total_length
                / count
            )
            if count
            else 0.0
        ),

        "p95_run": (
            histogram_percentile(
                combined,
                0.95
            )
        ),

        "runs_ge_6": int(
            sum(
                frequency
                for length, frequency
                in combined.items()
                if length >= 6
            )
        )
    }


    return result


# =============================================================================
# k-MER
# =============================================================================

def update_kmer_counts(
    counts: np.ndarray,
    carry: np.ndarray,
    codes: np.ndarray,
    k: int
) -> np.ndarray:

    combined = (
        np.concatenate(
            (
                carry,
                codes
            )
        )
        if carry.size
        else codes
    )


    windows = (
        combined.size
        - k
        + 1
    )


    if windows > 0:

        indices = np.zeros(
            windows,
            dtype=np.int64
        )


        for offset in range(
            k
        ):

            indices = (
                indices * 4
                + combined[
                    offset:
                    offset + windows
                ]
            )


        counts += np.bincount(
            indices,
            minlength=4 ** k
        )[
            :4 ** k
        ]


    if k == 1:

        return np.empty(
            0,
            dtype=np.uint8
        )


    keep = min(
        k - 1,
        combined.size
    )


    return combined[
        -keep:
    ].copy()


def js_divergence_bits(
    p: np.ndarray,
    q: np.ndarray
) -> float:

    midpoint = (
        0.5
        * (p + q)
    )


    def kl_divergence(
        first: np.ndarray,
        second: np.ndarray
    ) -> float:

        mask = (
            (first > 0.0)
            & (second > 0.0)
        )


        return float(
            np.sum(
                first[
                    mask
                ]
                * (
                    np.log2(
                        first[
                            mask
                        ]
                    )
                    - np.log2(
                        second[
                            mask
                        ]
                    )
                )
            )
        )


    return (
        0.5
        * kl_divergence(
            p,
            midpoint
        )
        + 0.5
        * kl_divergence(
            q,
            midpoint
        )
    )


def summarize_kmers(
    counts: np.ndarray,
    k: int
) -> dict:

    total = int(
        counts.sum()
    )


    categories = int(
        counts.size
    )


    degrees_of_freedom = (
        categories
        - 1
    )


    if total == 0:

        return {
            "k": k,

            "total_windows": 0,

            "max_abs_deviation_from_uniform": None,

            "js_divergence_from_uniform_bits": None,

            "chi_square_uniform": None,

            "degrees_of_freedom": (
                degrees_of_freedom
            ),

            "p_value": None
        }


    probabilities = (
        counts.astype(
            np.float64
        )
        / total
    )


    uniform = np.full(
        categories,
        1.0 / categories
    )


    expected = (
        total
        / categories
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


    return {
        "k": k,

        "total_windows": (
            total
        ),

        "ideal_probability": float(
            1.0
            / categories
        ),

        "minimum_probability": float(
            probabilities.min()
        ),

        "maximum_probability": float(
            probabilities.max()
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
            js_divergence_bits(
                probabilities,
                uniform
            )
        ),

        "chi_square_uniform": (
            chi_square_value
        ),

        "degrees_of_freedom": (
            degrees_of_freedom
        ),

        "p_value": (
            float(
                chi2.sf(
                    chi_square_value,
                    degrees_of_freedom
                )
            )
            if HAVE_SCIPY
            else None
        )
    }


def save_kmer_csv(
    path: Path,
    counts: np.ndarray,
    k: int
) -> None:

    labels = [
        "".join(
            characters
        )
        for characters in itertools.product(
            DNA,
            repeat=k
        )
    ]


    total = int(
        counts.sum()
    )


    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as handle:

        handle.write(
            "kmer,count,probability\n"
        )


        for label, count in zip(
            labels,
            counts.tolist()
        ):

            probability = (
                count
                / total
                if total
                else 0.0
            )


            handle.write(
                f"{label},"
                f"{count},"
                f"{probability:.12f}\n"
            )


# =============================================================================
# ACF
# =============================================================================

def calculate_acf(
    sample_codes: np.ndarray,
    max_lag: int
) -> dict:

    number_of_bases = int(
        sample_codes.size
    )


    result = {
        "bases_analyzed": (
            number_of_bases
        ),

        "maximum_lag": int(
            max_lag
        ),

        "per_base": {},

        "maximum_absolute_acf_all": 0.0
    }


    if (
        number_of_bases < 2
        or max_lag <= 0
    ):

        return result


    effective_lag = min(
        max_lag,
        number_of_bases - 1
    )


    global_maximum = 0.0


    for base_code, base in enumerate(
        DNA
    ):

        indicator = (
            sample_codes
            == base_code
        ).astype(
            np.float64
        )


        mean = float(
            indicator.mean()
        )


        variance = (
            mean
            * (1.0 - mean)
        )


        values: List[
            float
        ] = []


        if variance == 0.0:

            values = [
                0.0
            ] * effective_lag


        else:

            centered = (
                indicator
                - mean
            )


            for lag in range(
                1,
                effective_lag + 1
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
                        number_of_bases
                        - lag
                    )
                )


                values.append(
                    covariance
                    / variance
                )


        base_maximum = max(
            (
                abs(value)
                for value in values
            ),
            default=0.0
        )


        global_maximum = max(
            global_maximum,
            base_maximum
        )


        result[
            "per_base"
        ][
            base
        ] = {
            "lag_1": (
                float(
                    values[0]
                )
                if values
                else 0.0
            ),

            "maximum_absolute_lag_1_to_L": float(
                base_maximum
            ),

            "values": [
                float(value)
                for value in values
            ]
        }


    result[
        "maximum_absolute_acf_all"
    ] = float(
        global_maximum
    )


    return result


# =============================================================================
# TANILAMA
# =============================================================================

def diagnostic_verdict(
    length: int,
    gc_ratio: float,
    homopolymer: dict,
    kmer: Dict[str, dict],
    acf: dict
) -> dict:

    issues: List[
        str
    ] = []


    gc_limit = max(
        0.02,

        4.0
        * math.sqrt(
            0.25
            / max(
                length,
                1
            )
        )
    )


    if (
        abs(
            gc_ratio
            - 0.5
        )
        > gc_limit
    ):

        issues.append(
            "GC sapması uyarlamalı sınırı aşıyor: "
            f"|GC-0.5|={abs(gc_ratio - 0.5):.6f}, "
            f"sınır={gc_limit:.6f}."
        )


    run_limit = max(
        6,

        int(
            math.ceil(
                math.log(
                    max(
                        length,
                        2
                    ),
                    4
                )
            )
        )
        + 3
    )


    maximum_run = int(
        homopolymer[
            "_all"
        ][
            "maximum_run"
        ]
    )


    if maximum_run > run_limit:

        issues.append(
            "Maksimum homopolimer sınırı aşıyor: "
            f"gözlenen={maximum_run}, "
            f"sınır={run_limit}."
        )


    for k_text, summary in kmer.items():

        windows = int(
            summary.get(
                "total_windows",
                0
            )
        )


        deviation = summary.get(
            "max_abs_deviation_from_uniform"
        )


        if (
            windows == 0
            or deviation is None
        ):

            continue


        k = int(
            k_text
        )


        probability = (
            1.0
            / (4 ** k)
        )


        limit = max(
            0.005,

            4.0
            * math.sqrt(
                probability
                * (
                    1.0
                    - probability
                )
                / windows
            )
        )


        if float(
            deviation
        ) > limit:

            issues.append(
                f"k={k} uniformluk sapması "
                "sınırı aşıyor: "
                f"gözlenen={float(deviation):.6f}, "
                f"sınır={limit:.6f}."
            )


    acf_bases = int(
        acf.get(
            "bases_analyzed",
            0
        )
    )


    maximum_acf = float(
        acf.get(
            "maximum_absolute_acf_all",
            0.0
        )
    )


    if acf_bases > 1:

        limit = max(
            0.03,

            4.0
            / math.sqrt(
                acf_bases
            )
        )


        if maximum_acf > limit:

            issues.append(
                "Maksimum |ACF| sınırı aşıyor: "
                f"gözlenen={maximum_acf:.6f}, "
                f"sınır={limit:.6f}."
            )


    return {
        "status": (
            "PASS"
            if not issues
            else "FLAG"
        ),

        "issues": (
            issues
        ),

        "note": (
            "Tanımlayıcı kalite kontrolüdür; "
            "tek başına kriptografik "
            "güvenlik kanıtı değildir."
        )
    }


# =============================================================================
# TEK DOSYA ANALİZİ
# =============================================================================

def analyze_file(
    path: Path,
    label: str
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"{label} dosyası "
            f"bulunamadı: {path}"
        )


    analysis_start = (
        time.perf_counter()
    )


    canonical_hash = (
        hashlib.sha256()
    )


    base_counts = np.zeros(
        4,
        dtype=np.int64
    )


    kmer_counts = {
        k: np.zeros(
            4 ** k,
            dtype=np.int64
        )
        for k in K_VALUES
    }


    carries = {
        k: np.empty(
            0,
            dtype=np.uint8
        )
        for k in K_VALUES
    }


    run_histograms: List[
        Counter
    ] = [
        Counter()
        for _ in range(4)
    ]


    pending_base: Optional[
        int
    ] = None


    pending_length = 0


    acf_buffer = bytearray()


    total_bases = 0


    for chunk in iter_dna_chunks(
        path,
        STREAM_CHUNK_BASES
    ):

        canonical_hash.update(
            chunk
        )


        total_bases += len(
            chunk
        )


        if (
            len(
                acf_buffer
            )
            < ACF_SAMPLE_BASES
        ):

            remaining = (
                ACF_SAMPLE_BASES
                - len(
                    acf_buffer
                )
            )


            acf_buffer.extend(
                chunk[
                    :remaining
                ]
            )


        codes = ASCII_TO_CODE[
            np.frombuffer(
                chunk,
                dtype=np.uint8
            )
        ]


        base_counts += np.bincount(
            codes,
            minlength=4
        )[
            :4
        ]


        for k in K_VALUES:

            carries[
                k
            ] = update_kmer_counts(
                kmer_counts[
                    k
                ],

                carries[
                    k
                ],

                codes,

                k
            )


        if codes.size:

            boundaries = (
                np.flatnonzero(
                    codes[
                        1:
                    ]
                    != codes[
                        :-1
                    ]
                )
                + 1
            )


            starts = np.concatenate(
                (
                    np.array(
                        [0]
                    ),

                    boundaries
                )
            )


            ends = np.concatenate(
                (
                    boundaries,

                    np.array(
                        [
                            codes.size
                        ]
                    )
                )
            )


            run_bases = (
                codes[
                    starts
                ]
            )


            run_lengths = (
                ends
                - starts
            ).astype(
                np.int64
            )


            if pending_base is not None:

                if (
                    int(
                        run_bases[0]
                    )
                    == pending_base
                ):

                    run_lengths[
                        0
                    ] += pending_length


                else:

                    finalize_run(
                        run_histograms,
                        pending_base,
                        pending_length
                    )


            for (
                base_code,
                run_length
            ) in zip(
                run_bases[
                    :-1
                ],

                run_lengths[
                    :-1
                ]
            ):

                finalize_run(
                    run_histograms,

                    int(
                        base_code
                    ),

                    int(
                        run_length
                    )
                )


            pending_base = int(
                run_bases[
                    -1
                ]
            )


            pending_length = int(
                run_lengths[
                    -1
                ]
            )


    if pending_base is not None:

        finalize_run(
            run_histograms,
            pending_base,
            pending_length
        )


    if total_bases == 0:

        raise ValueError(
            f"{path} içinde geçerli "
            "A/C/G/T bulunamadı."
        )


    probabilities = (
        base_counts.astype(
            np.float64
        )
        / total_bases
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
            base_counts[1]
            + base_counts[2]
        )
        / total_bases
    )


    homopolymer = summarize_runs(
        run_histograms
    )


    kmer_summary: Dict[
        str,
        dict
    ] = {}


    for k in K_VALUES:

        kmer_summary[
            str(k)
        ] = summarize_kmers(
            kmer_counts[
                k
            ],

            k
        )


        if SAVE_KMER_CSV:

            save_kmer_csv(
                BASE_DIR
                / (
                    f"{CSV_PREFIX}_"
                    f"{label.lower()}_"
                    f"k{k}.csv"
                ),

                kmer_counts[
                    k
                ],

                k
            )


    acf_codes = ASCII_TO_CODE[
        np.frombuffer(
            bytes(
                acf_buffer
            ),
            dtype=np.uint8
        )
    ]


    acf = calculate_acf(
        acf_codes,
        MAX_LAG
    )


    summary = {
        "label": (
            label
        ),

        "file": (
            path.name
        ),

        "length_bases": int(
            total_bases
        ),

        "canonical_sha256": (
            canonical_hash.hexdigest()
        ),

        "base_counts": {
            base: int(
                base_counts[
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

        "shannon_entropy_bits_per_base": (
            entropy
        ),

        "ideal_entropy_bits_per_base": (
            2.0
        ),

        "entropy_deficit_from_ideal": float(
            2.0
            - entropy
        ),

        "homopolymer": (
            homopolymer
        ),

        "kmer": (
            kmer_summary
        ),

        "acf": (
            acf
        ),

        "analysis_wall_seconds": float(
            time.perf_counter()
            - analysis_start
        )
    }


    summary[
        "diagnostic_verdict"
    ] = diagnostic_verdict(
        total_bases,
        gc_ratio,
        homopolymer,
        kmer_summary,
        acf
    )


    return summary


# =============================================================================
# İKİ DNA DOSYASINI KARŞILAŞTIRMA
# =============================================================================

def compare_files(
    first_path: Path,
    second_path: Path
) -> dict:

    first_iterator = (
        iter_dna_chunks(
            first_path,
            STREAM_CHUNK_BASES
        )
    )


    second_iterator = (
        iter_dna_chunks(
            second_path,
            STREAM_CHUNK_BASES
        )
    )


    first_length = 0

    second_length = 0

    matching_bases = 0

    differing_bits = 0


    for first, second in itertools.zip_longest(
        first_iterator,
        second_iterator,
        fillvalue=b""
    ):

        first_length += len(
            first
        )


        second_length += len(
            second
        )


        overlap = min(
            len(
                first
            ),

            len(
                second
            )
        )


        if overlap:

            first_ascii = np.frombuffer(
                first[
                    :overlap
                ],
                dtype=np.uint8
            )


            second_ascii = np.frombuffer(
                second[
                    :overlap
                ],
                dtype=np.uint8
            )


            matching_bases += int(
                np.count_nonzero(
                    first_ascii
                    == second_ascii
                )
            )


            xor_values = np.bitwise_xor(
                ASCII_TO_CODE[
                    first_ascii
                ],

                ASCII_TO_CODE[
                    second_ascii
                ]
            )


            differing_bits += int(
                BITCOUNT_2BIT[
                    xor_values
                ].sum()
            )


        differing_bits += (
            2
            * abs(
                len(
                    first
                )
                - len(
                    second
                )
            )
        )


    comparison_length = max(
        first_length,
        second_length
    )


    differing_bases = (
        comparison_length
        - matching_bases
    )


    total_bits = (
        2
        * comparison_length
    )


    return {
        "first_bases": int(
            first_length
        ),

        "second_bases": int(
            second_length
        ),

        "length_match": (
            first_length
            == second_length
        ),

        "matching_bases": int(
            matching_bases
        ),

        "differing_bases": int(
            differing_bases
        ),

        "base_difference_percent": (
            100.0
            * differing_bases
            / comparison_length
            if comparison_length
            else 0.0
        ),

        "base_similarity_percent": (
            100.0
            * matching_bases
            / comparison_length
            if comparison_length
            else 100.0
        ),

        "differing_bits_2bit": int(
            differing_bits
        ),

        "bit_difference_percent_2bit": (
            100.0
            * differing_bits
            / total_bits
            if total_bits
            else 0.0
        ),

        "bit_similarity_percent_2bit": (
            100.0
            * (
                total_bits
                - differing_bits
            )
            / total_bits
            if total_bits
            else 100.0
        )
    }


# =============================================================================
# EKRAN RAPORU
# =============================================================================

def print_summary(
    summary: dict
) -> None:

    print(
        "\n"
        + "-" * 78
    )


    print(
        f"{summary['label']} — "
        f"{summary['file']}"
    )


    print(
        "-" * 78
    )


    print(
        f"Baz sayısı                    : "
        f"{summary['length_bases']:,}"
    )


    print(
        f"Baz sayımları                 : "
        f"{summary['base_counts']}"
    )


    print(
        f"Baz olasılıkları              : "
        f"{summary['base_probabilities']}"
    )


    print(
        f"GC oranı                      : "
        f"{summary['gc_ratio']:.6f}"
    )


    print(
        f"Shannon entropisi             : "
        f"{summary['shannon_entropy_bits_per_base']:.6f} "
        f"bit/baz"
    )


    print(
        f"İdealden entropi farkı        : "
        f"{summary['entropy_deficit_from_ideal']:.6f} "
        f"bit/baz"
    )


    print(
        f"Maksimum homopolimer          : "
        f"{summary['homopolymer']['_all']['maximum_run']}"
    )


    for k_text, statistics in summary[
        "kmer"
    ].items():

        print(
            f"k={k_text} Δmax / JS           : "
            f"{statistics['max_abs_deviation_from_uniform']:.6f} / "
            f"{statistics['js_divergence_from_uniform_bits']:.6f} "
            f"bit"
        )


    print(
        f"ACF analiz baz sayısı         : "
        f"{summary['acf']['bases_analyzed']:,}"
    )


    print(
        f"Maksimum |ACF|                : "
        f"{summary['acf']['maximum_absolute_acf_all']:.6f}"
    )


    print(
        f"Tanılama durumu               : "
        f"{summary['diagnostic_verdict']['status']}"
    )


    for issue in summary[
        "diagnostic_verdict"
    ][
        "issues"
    ]:

        print(
            f"  - {issue}"
        )


def print_comparison(
    title: str,
    comparison: dict
) -> None:

    print(
        "\n"
        + title
    )


    print(
        f"Uzunluk eşleşmesi             : "
        f"{comparison['length_match']}"
    )


    print(
        f"Baz farkı                     : "
        f"{comparison['base_difference_percent']:.6f}%"
    )


    print(
        f"2-bit farkı                   : "
        f"{comparison['bit_difference_percent_2bit']:.6f}%"
    )


# =============================================================================
# ANA AKIŞ
# =============================================================================

def main() -> None:

    plain_path = (
        BASE_DIR
        / ORIGINAL_FILENAME
    )


    cipher_path = (
        BASE_DIR
        / CIPHER_FILENAME
    )


    decrypted_path = (
        BASE_DIR
        / DECRYPTED_FILENAME
    )


    encryption_metadata_path = (
        BASE_DIR
        / ENCRYPTION_METADATA_FILENAME
    )


    decryption_metadata_path = (
        BASE_DIR
        / DECRYPTION_METADATA_FILENAME
    )


    report_path = (
        BASE_DIR
        / REPORT_FILENAME
    )


    print(
        f"[PLAIN]      "
        f"{plain_path}"
    )


    print(
        f"[CIPHER]     "
        f"{cipher_path}"
    )


    print(
        f"[DECRYPTED]  "
        f"{decrypted_path}"
    )


    print(
        f"[REPORT]     "
        f"{report_path}"
    )


    if not plain_path.exists():

        raise FileNotFoundError(
            f"Plaintext bulunamadı: "
            f"{plain_path}"
        )


    if not cipher_path.exists():

        raise FileNotFoundError(
            f"Ciphertext bulunamadı: "
            f"{cipher_path}"
        )


    process = (
        psutil.Process(
            os.getpid()
        )
        if HAVE_PSUTIL
        else None
    )


    rss_start = (
        process.memory_info().rss
        if process
        else None
    )


    cpu_start = (
        process.cpu_times().user
        + process.cpu_times().system
        if process
        else None
    )


    wall_start = (
        time.perf_counter()
    )


    summaries = {
        "plaintext": analyze_file(
            plain_path,
            "PLAINTEXT"
        ),

        "ciphertext": analyze_file(
            cipher_path,
            "CIPHERTEXT"
        )
    }


    if (
        ANALYZE_DECRYPTED
        and decrypted_path.exists()
    ):

        summaries[
            "decrypted"
        ] = analyze_file(
            decrypted_path,
            "DECRYPTED"
        )


    comparisons = {
        "plaintext_vs_ciphertext": (
            compare_files(
                plain_path,
                cipher_path
            )
        )
    }


    if "decrypted" in summaries:

        comparisons[
            "plaintext_vs_decrypted"
        ] = compare_files(
            plain_path,
            decrypted_path
        )


    encryption_metadata = (
        read_json_optional(
            encryption_metadata_path
        )
    )


    decryption_metadata = (
        read_json_optional(
            decryption_metadata_path
        )
    )


    wall_seconds = (
        time.perf_counter()
        - wall_start
    )


    cpu_end = (
        process.cpu_times().user
        + process.cpu_times().system
        if process
        else None
    )


    rss_end = (
        process.memory_info().rss
        if process
        else None
    )


    report = {
        "analysis": (
            "BMC T5-NREF DNA-SPD "
            "supplementary genomic metrics"
        ),

        "dataset": (
            DATASET_STEM
        ),

        "settings": {
            "k_values": list(
                K_VALUES
            ),

            "maximum_lag": (
                MAX_LAG
            ),

            "stream_chunk_bases": (
                STREAM_CHUNK_BASES
            ),

            "acf_sample_bases_limit": (
                ACF_SAMPLE_BASES
            ),

            "save_kmer_csv": (
                SAVE_KMER_CSV
            ),

            "scipy_available": (
                HAVE_SCIPY
            )
        },

        "summaries": (
            summaries
        ),

        "comparisons": (
            comparisons
        ),

        "metadata_context": {
            "encryption_metadata_available": (
                encryption_metadata
                is not None
            ),

            "decryption_metadata_available": (
                decryption_metadata
                is not None
            ),

            "scheme": (
                encryption_metadata.get(
                    "scheme"
                )
                if encryption_metadata
                else None
            ),

            "nonce_hex": (
                encryption_metadata.get(
                    "session",
                    {}
                ).get(
                    "nonce_hex"
                )
                if encryption_metadata
                else None
            ),

            "ciphertext_hmac_verified": (
                decryption_metadata.get(
                    "integrity",
                    {}
                ).get(
                    "ciphertext_hmac_verified"
                )
                if decryption_metadata
                else None
            )
        },

        "performance": {
            "total_wall_seconds": float(
                wall_seconds
            ),

            "cpu_seconds": (
                None
                if (
                    cpu_start is None
                    or cpu_end is None
                )
                else float(
                    cpu_end
                    - cpu_start
                )
            ),

            "rss_start_mb": (
                None
                if rss_start is None
                else (
                    rss_start
                    / (1024.0 ** 2)
                )
            ),

            "rss_end_mb": (
                None
                if rss_end is None
                else (
                    rss_end
                    / (1024.0 ** 2)
                )
            ),

            "rss_delta_mb": (
                None
                if (
                    rss_start is None
                    or rss_end is None
                )
                else (
                    rss_end
                    - rss_start
                )
                / (1024.0 ** 2)
            )
        }
    }


    with report_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2
        )


        handle.write(
            "\n"
        )


    print(
        "\n"
        + "=" * 82
    )


    print(
        "BMC T5-NREF DNA-SPD "
        "EK GENOMİK METRİKLER"
    )


    print(
        "=" * 82
    )


    for summary in summaries.values():

        print_summary(
            summary
        )


    print(
        "\n"
        + "-" * 78
    )


    print(
        "DOSYALAR ARASI "
        "KARŞILAŞTIRMALAR"
    )


    print(
        "-" * 78
    )


    print_comparison(
        "PLAINTEXT vs CIPHERTEXT",

        comparisons[
            "plaintext_vs_ciphertext"
        ]
    )


    if (
        "plaintext_vs_decrypted"
        in comparisons
    ):

        print_comparison(
            "PLAINTEXT vs DECRYPTED",

            comparisons[
                "plaintext_vs_decrypted"
            ]
        )


    print(
        "\n"
        + "-" * 78
    )


    print(
        "ÇIKTILAR"
    )


    print(
        "-" * 78
    )


    print(
        f"JSON raporu                    : "
        f"{report_path}"
    )


    print(
        f"Toplam analiz süresi           : "
        f"{wall_seconds:.6f} s"
    )


    if SAVE_KMER_CSV:

        print(
            f"k-mer CSV öneki                : "
            f"{CSV_PREFIX}_..."
        )


    print(
        "=" * 82
    )


if __name__ == "__main__":

    main()
