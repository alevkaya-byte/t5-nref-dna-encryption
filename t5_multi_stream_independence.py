
import csv
import itertools
import json
import math
import statistics
import sys

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


try:
    from scipy.stats import t as student_t

    HAVE_SCIPY = True

except Exception:
    student_t = None
    HAVE_SCIPY = False


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


N_STREAMS = 10


EXPECTED_DNA_BASES = 500_000


EXPECTED_BITS = 1_000_000


# True yapılırsa her rejimde 45 çiftin tamamı ekrana ve TXT'ye yazılır.
# Çok uzun çıktı istemezsen False yapabilirsin.
# Pair sonuçları her durumda CSV dosyasına kaydedilir.
PRINT_PAIR_DETAILS = True


# Bir koşum veya dosya eksikse çalışmayı durdurur.
STRICT_MODE = True


# Akış dosyalarının neredeyse tamamen hedef alfabeden oluşması beklenir.
MIN_CONTENT_PURITY = 0.995


# Görsellerdeki gerçek klasör isimleri
REGIMES = (
    (
        "R1",
        "T5_R1.{index}",
    ),

    (
        "R1-ext",
        "T5_R1-ext.{index}",
    ),

    (
        "R2",
        "T5_R2.{index}",
    ),

    (
        "R3",
        "T5_R3.{index}",
    ),
)


OUT_TEXT = (
    BASE_DIR
    / "t5_multi_stream_independence.txt"
)


OUT_PAIRS_CSV = (
    BASE_DIR
    / "t5_multi_stream_independence_pairs.csv"
)


OUT_SUMMARY_CSV = (
    BASE_DIR
    / "t5_multi_stream_independence_summary.csv"
)


OUT_JSON = (
    BASE_DIR
    / "t5_multi_stream_independence.json"
)


OUT_SELECTED_FILES_CSV = (
    BASE_DIR
    / "t5_multi_stream_selected_files.csv"
)


# =============================================================================
# SABİTLER
# =============================================================================

DNA_ALPHABET = "ACGT"


DNA_LUT = np.full(
    256,
    255,
    dtype=np.uint8,
)


for code, base in enumerate(
    DNA_ALPHABET
):

    DNA_LUT[
        ord(base)
    ] = code

    DNA_LUT[
        ord(
            base.lower()
        )
    ] = code


WHITESPACE_BYTES = np.array(
    [
        9,
        10,
        13,
        32,
    ],
    dtype=np.uint8,
)


# =============================================================================
# VERİ SINIFLARI
# =============================================================================

@dataclass(
    frozen=True
)
class FileProbe:

    path: str

    size_bytes: int

    non_whitespace_count: int

    dna_count: int

    dna_purity: float

    bit_count: int

    bit_purity: float


@dataclass(
    frozen=True
)
class StreamFiles:

    regime: str

    run_index: int

    folder: str

    dna_path: str

    bits_path: str

    dna_length: int

    bit_length: int

    detection_method: str


@dataclass
class StreamData:

    label: str

    run_index: int

    dna_path: Path

    bits_path: Path

    dna: np.ndarray

    bits: np.ndarray

    dna_probabilities: np.ndarray

    bit_one_probability: float


# =============================================================================
# EKRAN + DOSYA ÇIKTISI
# =============================================================================

class Tee:

    def __init__(
        self,
        *files
    ):

        self.files = files


    def write(
        self,
        data: str
    ) -> None:

        for file in self.files:

            file.write(
                data
            )


    def flush(
        self
    ) -> None:

        for file in self.files:

            file.flush()


# =============================================================================
# DOSYA İÇERİĞİNİ TANIMA
# =============================================================================

def probe_text_file(
    path: Path
) -> FileProbe:

    raw = path.read_bytes()


    if not raw:

        return FileProbe(
            path=str(
                path
            ),

            size_bytes=0,

            non_whitespace_count=0,

            dna_count=0,

            dna_purity=0.0,

            bit_count=0,

            bit_purity=0.0,
        )


    values = np.frombuffer(
        raw,
        dtype=np.uint8,
    )


    whitespace_mask = np.isin(
        values,
        WHITESPACE_BYTES,
    )


    non_whitespace_count = int(
        np.count_nonzero(
            ~whitespace_mask
        )
    )


    upper_values = values.copy()


    lower_mask = (
        (
            upper_values
            >= ord("a")
        )
        & (
            upper_values
            <= ord("z")
        )
    )


    upper_values[
        lower_mask
    ] -= 32


    dna_mask = np.isin(
        upper_values,

        np.array(
            [
                ord("A"),
                ord("C"),
                ord("G"),
                ord("T"),
            ],
            dtype=np.uint8,
        ),
    )


    bit_mask = (
        (
            values
            == ord("0")
        )
        | (
            values
            == ord("1")
        )
    )


    dna_count = int(
        np.count_nonzero(
            dna_mask
        )
    )


    bit_count = int(
        np.count_nonzero(
            bit_mask
        )
    )


    denominator = max(
        non_whitespace_count,
        1,
    )


    return FileProbe(
        path=str(
            path
        ),

        size_bytes=len(
            raw
        ),

        non_whitespace_count=(
            non_whitespace_count
        ),

        dna_count=(
            dna_count
        ),

        dna_purity=(
            dna_count
            / denominator
        ),

        bit_count=(
            bit_count
        ),

        bit_purity=(
            bit_count
            / denominator
        ),
    )


def validate_dna_candidate(
    path: Path,
    probe_cache: Dict[
        Path,
        FileProbe
    ]
) -> bool:

    if path not in probe_cache:

        probe_cache[
            path
        ] = probe_text_file(
            path
        )


    probe = probe_cache[
        path
    ]


    return (
        probe.dna_count
        == EXPECTED_DNA_BASES
        and probe.dna_purity
        >= MIN_CONTENT_PURITY
    )


def validate_bits_candidate(
    path: Path,
    probe_cache: Dict[
        Path,
        FileProbe
    ]
) -> bool:

    if path not in probe_cache:

        probe_cache[
            path
        ] = probe_text_file(
            path
        )


    probe = probe_cache[
        path
    ]


    return (
        probe.bit_count
        == EXPECTED_BITS
        and probe.bit_purity
        >= MIN_CONTENT_PURITY
    )


# =============================================================================
# AYNI KÖKE AİT JSON + DNA + BITS PAKETİNİ BULMA
# =============================================================================

def metadata_bundle_candidates(
    folder: Path,
    probe_cache: Dict[
        Path,
        FileProbe
    ]
) -> List[
    Tuple[
        Path,
        Path,
        Path
    ]
]:

    output: List[
        Tuple[
            Path,
            Path,
            Path
        ]
    ] = []


    json_files = sorted(
        folder.glob(
            "*.json"
        ),

        key=lambda path: (
            path.stat().st_mtime
        ),

        reverse=True,
    )


    for metadata_path in json_files:

        base_path = metadata_path.with_suffix(
            ""
        )


        dna_path = Path(
            str(
                base_path
            )
            + ".dna.txt"
        )


        bits_path = Path(
            str(
                base_path
            )
            + ".bits.txt"
        )


        if not (
            dna_path.is_file()
            and bits_path.is_file()
        ):

            continue


        if not validate_dna_candidate(
            dna_path,
            probe_cache,
        ):

            continue


        if not validate_bits_candidate(
            bits_path,
            probe_cache,
        ):

            continue


        output.append(
            (
                metadata_path,
                dna_path,
                bits_path,
            )
        )


    return output


# =============================================================================
# YEDEK OTOMATİK DOSYA SEÇİMİ
# =============================================================================

def dna_filename_score(
    path: Path
) -> int:

    name = path.name.lower()


    score = 0


    if name.endswith(
        ".dna.txt"
    ):

        score += 1000


    if ".dna." in name:

        score += 500


    if "dna_500k" in name:

        score += 100


    if (
        "softbalance" in name
        or "continuouspos" in name
    ):

        score += 40


    if "bits" in name:

        score -= 1000


    if "rules" in name:

        score -= 1000


    if "analysis" in name:

        score -= 1000


    if "dna_1m" in name:

        score -= 200


    return score


def bits_filename_score(
    path: Path
) -> int:

    name = path.name.lower()


    score = 0


    if name.endswith(
        ".bits.txt"
    ):

        score += 1000


    if ".bits." in name:

        score += 500


    if (
        "1mbits" in name
        or "1m_bits" in name
    ):

        score += 100


    if (
        "softbalance" in name
        or "continuouspos" in name
    ):

        score += 40


    if "rules" in name:

        score -= 1000


    if "analysis" in name:

        score -= 1000


    if (
        ".dna." in name
        or "dna_500k" in name
    ):

        score -= 1000


    return score


def fallback_content_detection(
    folder: Path,
    probe_cache: Dict[
        Path,
        FileProbe
    ]
) -> Tuple[
    Path,
    Path
]:

    text_files = [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            == ".txt"
        )
    ]


    dna_candidates = [
        path
        for path in text_files
        if validate_dna_candidate(
            path,
            probe_cache,
        )
    ]


    bit_candidates = [
        path
        for path in text_files
        if validate_bits_candidate(
            path,
            probe_cache,
        )
    ]


    if not dna_candidates:

        raise FileNotFoundError(
            f"{folder}: "
            f"{EXPECTED_DNA_BASES:,} bazlık "
            f"DNA çıktısı bulunamadı."
        )


    if not bit_candidates:

        raise FileNotFoundError(
            f"{folder}: "
            f"{EXPECTED_BITS:,} bitlik "
            f"bit çıktısı bulunamadı."
        )


    dna_candidates.sort(
        key=lambda path: (
            dna_filename_score(
                path
            ),

            path.stat().st_mtime,
        ),

        reverse=True,
    )


    bit_candidates.sort(
        key=lambda path: (
            bits_filename_score(
                path
            ),

            path.stat().st_mtime,
        ),

        reverse=True,
    )


    return (
        dna_candidates[
            0
        ],

        bit_candidates[
            0
        ],
    )


# =============================================================================
# HER KOŞUM İÇİN DOSYA SEÇİMİ
# =============================================================================

def discover_stream_files(
    regime: str,
    folder_template: str,
    run_index: int
) -> StreamFiles:

    folder = (
        BASE_DIR
        / folder_template.format(
            index=run_index
        )
    )


    if not folder.is_dir():

        raise FileNotFoundError(
            f"Koşum klasörü bulunamadı: "
            f"{folder}"
        )


    probe_cache: Dict[
        Path,
        FileProbe
    ] = {}


    bundles = metadata_bundle_candidates(
        folder,
        probe_cache,
    )


    if bundles:

        (
            _metadata_path,
            dna_path,
            bits_path,
        ) = bundles[
            0
        ]


        detection_method = (
            "matching_json_dna_bits_bundle"
        )


    else:

        (
            dna_path,
            bits_path,
        ) = fallback_content_detection(
            folder,
            probe_cache,
        )


        detection_method = (
            "content_and_filename_fallback"
        )


    if dna_path not in probe_cache:

        probe_cache[
            dna_path
        ] = probe_text_file(
            dna_path
        )


    if bits_path not in probe_cache:

        probe_cache[
            bits_path
        ] = probe_text_file(
            bits_path
        )


    dna_probe = probe_cache[
        dna_path
    ]


    bits_probe = probe_cache[
        bits_path
    ]


    return StreamFiles(
        regime=regime,

        run_index=run_index,

        folder=str(
            folder
        ),

        dna_path=str(
            dna_path
        ),

        bits_path=str(
            bits_path
        ),

        dna_length=(
            dna_probe.dna_count
        ),

        bit_length=(
            bits_probe.bit_count
        ),

        detection_method=(
            detection_method
        ),
    )


# =============================================================================
# DNA VE BİT DOSYALARINI OKUMA
# =============================================================================

def read_dna_codes(
    path: Path
) -> np.ndarray:

    raw = path.read_bytes()


    values = np.frombuffer(
        raw,
        dtype=np.uint8,
    )


    codes = DNA_LUT[
        values
    ]


    valid = (
        codes
        != 255
    )


    output = codes[
        valid
    ].astype(
        np.uint8,
        copy=True,
    )


    if output.size != EXPECTED_DNA_BASES:

        raise ValueError(
            f"{path}: DNA uzunluğu "
            f"{output.size:,}; "
            f"beklenen "
            f"{EXPECTED_DNA_BASES:,}."
        )


    return output


def read_bit_array(
    path: Path
) -> np.ndarray:

    raw = path.read_bytes()


    values = np.frombuffer(
        raw,
        dtype=np.uint8,
    )


    valid = (
        (
            values
            == ord("0")
        )
        | (
            values
            == ord("1")
        )
    )


    output = (
        values[
            valid
        ]
        - ord("0")
    ).astype(
        np.uint8,
        copy=True,
    )


    if output.size != EXPECTED_BITS:

        raise ValueError(
            f"{path}: bit uzunluğu "
            f"{output.size:,}; "
            f"beklenen "
            f"{EXPECTED_BITS:,}."
        )


    return output


def load_stream(
    selected: StreamFiles
) -> StreamData:

    dna_path = Path(
        selected.dna_path
    )


    bits_path = Path(
        selected.bits_path
    )


    dna = read_dna_codes(
        dna_path
    )


    bits = read_bit_array(
        bits_path
    )


    dna_counts = np.bincount(
        dna,
        minlength=4,
    )[
        :4
    ].astype(
        np.float64
    )


    dna_probabilities = (
        dna_counts
        / dna.size
    )


    bit_one_probability = float(
        bits.mean()
    )


    return StreamData(
        label=(
            f"{selected.regime}."
            f"{selected.run_index}"
        ),

        run_index=(
            selected.run_index
        ),

        dna_path=(
            dna_path
        ),

        bits_path=(
            bits_path
        ),

        dna=(
            dna
        ),

        bits=(
            bits
        ),

        dna_probabilities=(
            dna_probabilities
        ),

        bit_one_probability=(
            bit_one_probability
        ),
    )


# =============================================================================
# GÜVEN ARALIĞI VE P-DEĞERİ
# =============================================================================

def wilson_ci(
    p_hat: float,
    n: int,
    z: float = 1.96
) -> Tuple[
    float,
    float
]:

    if n <= 0:

        return (
            float("nan"),
            float("nan"),
        )


    denominator = (
        1.0
        + z * z / n
    )


    center = (
        p_hat
        + z * z / (
            2.0
            * n
        )
    )


    adjustment = z * math.sqrt(
        (
            p_hat
            * (
                1.0
                - p_hat
            )
            + z * z
            / (
                4.0
                * n
            )
        )
        / n
    )


    return (
        (
            center
            - adjustment
        )
        / denominator,

        (
            center
            + adjustment
        )
        / denominator,
    )


def two_sided_normal_p_value(
    p_hat: float,
    p0: float,
    n: int
) -> float:

    if n <= 0:

        return float(
            "nan"
        )


    if not (
        0.0
        < p0
        < 1.0
    ):

        return float(
            "nan"
        )


    standard_error = math.sqrt(
        p0
        * (
            1.0
            - p0
        )
        / n
    )


    if standard_error <= 0.0:

        return float(
            "nan"
        )


    z_score = (
        p_hat
        - p0
    ) / standard_error


    return math.erfc(
        abs(
            z_score
        )
        / math.sqrt(
            2.0
        )
    )


def format_p_value(
    value: float
) -> str:

    if math.isnan(
        value
    ):

        return "N/A"


    if value == 0.0:

        return "<1e-300"


    if value < 0.001:

        return f"{value:.3e}"


    return f"{value:.6f}"


# =============================================================================
# TANIMLAYICI ÖZET
# =============================================================================

def descriptive_summary(
    values: Sequence[
        float
    ]
) -> dict:

    data = [
        float(
            value
        )
        for value in values
    ]


    if not data:

        return {
            "count": 0,

            "mean": None,

            "sample_sd": None,

            "minimum": None,

            "maximum": None,
        }


    return {
        "count": len(
            data
        ),

        "mean": float(
            statistics.fmean(
                data
            )
        ),

        "sample_sd": (
            float(
                statistics.stdev(
                    data
                )
            )
            if len(
                data
            ) > 1
            else 0.0
        ),

        "minimum": float(
            min(
                data
            )
        ),

        "maximum": float(
            max(
                data
            )
        ),
    }


# =============================================================================
# KOŞUM-SEVİYELİ JACKKNIFE GÜVEN ARALIĞI
# =============================================================================

def jackknife_ci_over_streams(
    pair_records: Sequence[
        dict
    ],
    number_of_streams: int,
    metric_key: str
) -> dict:

    all_values = [
        float(
            record[
                metric_key
            ]
        )
        for record in pair_records
    ]


    theta = float(
        statistics.fmean(
            all_values
        )
    )


    leave_one_out: List[
        float
    ] = []


    for excluded_index in range(
        1,
        number_of_streams
        + 1,
    ):

        retained = [
            float(
                record[
                    metric_key
                ]
            )
            for record in pair_records
            if (
                record[
                    "run_i"
                ]
                != excluded_index
                and record[
                    "run_j"
                ]
                != excluded_index
            )
        ]


        leave_one_out.append(
            float(
                statistics.fmean(
                    retained
                )
            )
        )


    leave_one_mean = float(
        statistics.fmean(
            leave_one_out
        )
    )


    standard_error = math.sqrt(
        (
            number_of_streams
            - 1
        )
        / number_of_streams
        * sum(
            (
                value
                - leave_one_mean
            ) ** 2
            for value in leave_one_out
        )
    )


    if HAVE_SCIPY:

        critical = float(
            student_t.ppf(
                0.975,

                df=(
                    number_of_streams
                    - 1
                ),
            )
        )


    else:

        # df = 9 için yaklaşık t kritik değeri
        critical = 2.262157


    return {
        "estimate": (
            theta
        ),

        "standard_error": float(
            standard_error
        ),

        "ci95_low": float(
            theta
            - critical
            * standard_error
        ),

        "ci95_high": float(
            theta
            + critical
            * standard_error
        ),

        "method": (
            "leave-one-stream-out jackknife"
        ),
    }


# =============================================================================
# BİT PEARSON KORELASYONU
# =============================================================================

def bit_pearson(
    first: np.ndarray,
    second: np.ndarray
) -> float:

    first_float = first.astype(
        np.float64,
        copy=False,
    )


    second_float = second.astype(
        np.float64,
        copy=False,
    )


    first_centered = (
        first_float
        - first_float.mean()
    )


    second_centered = (
        second_float
        - second_float.mean()
    )


    denominator = math.sqrt(
        float(
            np.dot(
                first_centered,
                first_centered,
            )
        )
        * float(
            np.dot(
                second_centered,
                second_centered,
            )
        )
    )


    if denominator <= 0.0:

        return 0.0


    return float(
        np.dot(
            first_centered,
            second_centered,
        )
        / denominator
    )


# =============================================================================
# MUTUAL INFORMATION
# =============================================================================

def mutual_information_from_joint(
    joint_counts: np.ndarray
) -> float:

    joint = joint_counts.astype(
        np.float64
    )


    total = float(
        joint.sum()
    )


    if total <= 0.0:

        return 0.0


    joint /= total


    row_probabilities = joint.sum(
        axis=1,
        keepdims=True,
    )


    column_probabilities = joint.sum(
        axis=0,
        keepdims=True,
    )


    independent = (
        row_probabilities
        * column_probabilities
    )


    mask = (
        (
            joint
            > 0.0
        )
        & (
            independent
            > 0.0
        )
    )


    return float(
        np.sum(
            joint[
                mask
            ]
            * np.log2(
                joint[
                    mask
                ]
                / independent[
                    mask
                ]
            )
        )
    )


# =============================================================================
# İKİ AKIŞI KARŞILAŞTIRMA
# =============================================================================

def compare_pair(
    regime: str,
    first: StreamData,
    second: StreamData
) -> dict:

    # -------------------------------------------------------------------------
    # Bit alanı
    # -------------------------------------------------------------------------

    bit_n = min(
        first.bits.size,
        second.bits.size,
    )


    first_bits = first.bits[
        :bit_n
    ]


    second_bits = second.bits[
        :bit_n
    ]


    bit_differences = int(
        np.count_nonzero(
            first_bits
            != second_bits
        )
    )


    bit_hamming = (
        bit_differences
        / bit_n
    )


    (
        bit_ci_low,
        bit_ci_high,
    ) = wilson_ci(
        bit_hamming,
        bit_n,
    )


    p1 = float(
        first_bits.mean()
    )


    p2 = float(
        second_bits.mean()
    )


    bit_expected_marginal = (
        p1
        * (
            1.0
            - p2
        )
        + (
            1.0
            - p1
        )
        * p2
    )


    bit_joint_index = (
        first_bits.astype(
            np.int16,
            copy=False,
        )
        * 2
        + second_bits.astype(
            np.int16,
            copy=False,
        )
    )


    bit_joint = np.bincount(
        bit_joint_index,
        minlength=4,
    ).reshape(
        2,
        2,
    )


    # -------------------------------------------------------------------------
    # DNA alanı
    # -------------------------------------------------------------------------

    dna_n = min(
        first.dna.size,
        second.dna.size,
    )


    first_dna = first.dna[
        :dna_n
    ]


    second_dna = second.dna[
        :dna_n
    ]


    dna_differences = int(
        np.count_nonzero(
            first_dna
            != second_dna
        )
    )


    dna_mismatch = (
        dna_differences
        / dna_n
    )


    (
        dna_ci_low,
        dna_ci_high,
    ) = wilson_ci(
        dna_mismatch,
        dna_n,
    )


    first_dna_probabilities = (
        np.bincount(
            first_dna,
            minlength=4,
        )[
            :4
        ]
        / dna_n
    )


    second_dna_probabilities = (
        np.bincount(
            second_dna,
            minlength=4,
        )[
            :4
        ]
        / dna_n
    )


    dna_expected_marginal = float(
        1.0
        - np.dot(
            first_dna_probabilities,
            second_dna_probabilities,
        )
    )


    dna_joint_index = (
        first_dna.astype(
            np.int16,
            copy=False,
        )
        * 4
        + second_dna.astype(
            np.int16,
            copy=False,
        )
    )


    dna_joint = np.bincount(
        dna_joint_index,
        minlength=16,
    ).reshape(
        4,
        4,
    )


    return {
        "regime": (
            regime
        ),

        "stream_i": (
            first.label
        ),

        "stream_j": (
            second.label
        ),

        "run_i": (
            first.run_index
        ),

        "run_j": (
            second.run_index
        ),

        # ---------------------------------------------------------------------
        # Bit alanı
        # ---------------------------------------------------------------------

        "bit_n": int(
            bit_n
        ),

        "bit_differences": (
            bit_differences
        ),

        "bit_hamming": float(
            bit_hamming
        ),

        "bit_ci95_low": float(
            bit_ci_low
        ),

        "bit_ci95_high": float(
            bit_ci_high
        ),

        "bit_p_vs_0_5": float(
            two_sided_normal_p_value(
                bit_hamming,
                0.5,
                bit_n,
            )
        ),

        "bit_expected_from_marginals": float(
            bit_expected_marginal
        ),

        "bit_residual_from_marginals": float(
            bit_hamming
            - bit_expected_marginal
        ),

        "bit_p_vs_marginals": float(
            two_sided_normal_p_value(
                bit_hamming,
                bit_expected_marginal,
                bit_n,
            )
        ),

        "bit_pearson": float(
            bit_pearson(
                first_bits,
                second_bits,
            )
        ),

        "bit_mutual_information_bits": float(
            mutual_information_from_joint(
                bit_joint
            )
        ),

        # ---------------------------------------------------------------------
        # DNA alanı
        # ---------------------------------------------------------------------

        "dna_n": int(
            dna_n
        ),

        "dna_differences": (
            dna_differences
        ),

        "dna_mismatch": float(
            dna_mismatch
        ),

        "dna_ci95_low": float(
            dna_ci_low
        ),

        "dna_ci95_high": float(
            dna_ci_high
        ),

        "dna_p_vs_0_75": float(
            two_sided_normal_p_value(
                dna_mismatch,
                0.75,
                dna_n,
            )
        ),

        "dna_expected_from_marginals": float(
            dna_expected_marginal
        ),

        "dna_residual_from_marginals": float(
            dna_mismatch
            - dna_expected_marginal
        ),

        "dna_p_vs_marginals": float(
            two_sided_normal_p_value(
                dna_mismatch,
                dna_expected_marginal,
                dna_n,
            )
        ),

        "dna_mutual_information_bits": float(
            mutual_information_from_joint(
                dna_joint
            )
        ),
    }


# =============================================================================
# BİR REJİMİ ANALİZ ETME
# =============================================================================

def analyze_regime(
    regime: str,
    folder_template: str
) -> Tuple[
    dict,
    List[dict],
    List[StreamFiles],
]:

    print(
        "\n"
        + "=" * 100
    )


    print(
        f"{regime}: "
        f"10 bağımsız T5 akışı"
    )


    print(
        "=" * 100
    )


    selected_files: List[
        StreamFiles
    ] = []


    for run_index in range(
        1,
        N_STREAMS
        + 1,
    ):

        selected = discover_stream_files(
            regime,
            folder_template,
            run_index,
        )


        selected_files.append(
            selected
        )


        print(
            f"[SELECT] "
            f"{regime}.{run_index}:"
        )


        print(
            f"         DNA    = "
            f"{Path(selected.dna_path).name} "
            f"({selected.dna_length:,} baz)"
        )


        print(
            f"         BITS   = "
            f"{Path(selected.bits_path).name} "
            f"({selected.bit_length:,} bit)"
        )


        print(
            f"         METHOD = "
            f"{selected.detection_method}"
        )


    if (
        STRICT_MODE
        and len(
            selected_files
        )
        != N_STREAMS
    ):

        raise RuntimeError(
            f"{regime}: "
            f"{N_STREAMS} akış bekleniyordu; "
            f"{len(selected_files)} bulundu."
        )


    streams = [
        load_stream(
            selected
        )
        for selected in selected_files
    ]


    expected_pairs = (
        len(
            streams
        )
        * (
            len(
                streams
            )
            - 1
        )
        // 2
    )


    print(
        f"\n[INFO] Toplanan akış = "
        f"{len(streams)}"
    )


    print(
        f"[INFO] Beklenen çift = "
        f"{expected_pairs}"
    )


    pair_records: List[
        dict
    ] = []


    for (
        first,
        second
    ) in itertools.combinations(
        streams,
        2,
    ):

        record = compare_pair(
            regime,
            first,
            second,
        )


        pair_records.append(
            record
        )


        if PRINT_PAIR_DETAILS:

            print(
                f"\nPAIR "
                f"{first.label} "
                f"vs "
                f"{second.label}"
            )


            print(
                f"  Bit Hamming             = "
                f"{record['bit_hamming']:.6f}"
            )


            print(
                f"  Bit Pearson             = "
                f"{record['bit_pearson']:+.6f}"
            )


            print(
                f"  Bit MI                  = "
                f"{record['bit_mutual_information_bits']:.6e} bit"
            )


            print(
                f"  Bit beklenen (marjinal) = "
                f"{record['bit_expected_from_marginals']:.6f}"
            )


            print(
                f"  DNA mismatch            = "
                f"{record['dna_mismatch']:.6f}"
            )


            print(
                f"  DNA beklenen (uniform)  = "
                f"0.750000"
            )


            print(
                f"  DNA beklenen (marjinal) = "
                f"{record['dna_expected_from_marginals']:.6f}"
            )


            print(
                f"  DNA MI                  = "
                f"{record['dna_mutual_information_bits']:.6e} bit"
            )


    if (
        STRICT_MODE
        and len(
            pair_records
        )
        != expected_pairs
    ):

        raise RuntimeError(
            f"{regime}: "
            f"{expected_pairs} çift bekleniyordu; "
            f"{len(pair_records)} bulundu."
        )


    # =========================================================================
    # POOLED BİT SONUCU
    # =========================================================================

    bit_total_n = int(
        sum(
            record[
                "bit_n"
            ]
            for record in pair_records
        )
    )


    bit_total_differences = int(
        sum(
            record[
                "bit_differences"
            ]
            for record in pair_records
        )
    )


    bit_pooled = (
        bit_total_differences
        / bit_total_n
    )


    bit_pooled_ci = wilson_ci(
        bit_pooled,
        bit_total_n,
    )


    # =========================================================================
    # POOLED DNA SONUCU
    # =========================================================================

    dna_total_n = int(
        sum(
            record[
                "dna_n"
            ]
            for record in pair_records
        )
    )


    dna_total_differences = int(
        sum(
            record[
                "dna_differences"
            ]
            for record in pair_records
        )
    )


    dna_pooled = (
        dna_total_differences
        / dna_total_n
    )


    dna_pooled_ci = wilson_ci(
        dna_pooled,
        dna_total_n,
    )


    # =========================================================================
    # PAIRWISE LİSTELER
    # =========================================================================

    pair_bit_hamming = [
        record[
            "bit_hamming"
        ]
        for record in pair_records
    ]


    pair_dna_mismatch = [
        record[
            "dna_mismatch"
        ]
        for record in pair_records
    ]


    pair_bit_pearson = [
        record[
            "bit_pearson"
        ]
        for record in pair_records
    ]


    pair_bit_mi = [
        record[
            "bit_mutual_information_bits"
        ]
        for record in pair_records
    ]


    pair_dna_mi = [
        record[
            "dna_mutual_information_bits"
        ]
        for record in pair_records
    ]


    pair_dna_expected_marginal = [
        record[
            "dna_expected_from_marginals"
        ]
        for record in pair_records
    ]


    pair_dna_residual_marginal = [
        record[
            "dna_residual_from_marginals"
        ]
        for record in pair_records
    ]


    summary = {
        "regime": (
            regime
        ),

        "n_streams": len(
            streams
        ),

        "n_pairs": len(
            pair_records
        ),

        "stream_lengths": {
            "dna_bases_each": (
                EXPECTED_DNA_BASES
            ),

            "bits_each": (
                EXPECTED_BITS
            ),
        },

        # ---------------------------------------------------------------------
        # Bit alanı
        # ---------------------------------------------------------------------

        "bit_domain": {
            "theoretical_uniform_reference": (
                0.5
            ),

            "pooled_total_n": (
                bit_total_n
            ),

            "pooled_differences": (
                bit_total_differences
            ),

            "pooled_hamming": float(
                bit_pooled
            ),

            "pooled_wilson_ci95_low": float(
                bit_pooled_ci[
                    0
                ]
            ),

            "pooled_wilson_ci95_high": float(
                bit_pooled_ci[
                    1
                ]
            ),

            "pooled_p_vs_0_5": float(
                two_sided_normal_p_value(
                    bit_pooled,
                    0.5,
                    bit_total_n,
                )
            ),

            "pairwise_hamming": descriptive_summary(
                pair_bit_hamming
            ),

            "pairwise_hamming_jackknife_ci": (
                jackknife_ci_over_streams(
                    pair_records,
                    len(
                        streams
                    ),
                    "bit_hamming",
                )
            ),

            "pairwise_pearson": descriptive_summary(
                pair_bit_pearson
            ),

            "pairwise_mutual_information_bits": (
                descriptive_summary(
                    pair_bit_mi
                )
            ),
        },

        # ---------------------------------------------------------------------
        # DNA alanı
        # ---------------------------------------------------------------------

        "dna_domain": {
            "theoretical_uniform_reference": (
                0.75
            ),

            "pooled_total_n": (
                dna_total_n
            ),

            "pooled_differences": (
                dna_total_differences
            ),

            "pooled_mismatch": float(
                dna_pooled
            ),

            "pooled_wilson_ci95_low": float(
                dna_pooled_ci[
                    0
                ]
            ),

            "pooled_wilson_ci95_high": float(
                dna_pooled_ci[
                    1
                ]
            ),

            "pooled_p_vs_0_75": float(
                two_sided_normal_p_value(
                    dna_pooled,
                    0.75,
                    dna_total_n,
                )
            ),

            "pairwise_mismatch": descriptive_summary(
                pair_dna_mismatch
            ),

            "pairwise_mismatch_jackknife_ci": (
                jackknife_ci_over_streams(
                    pair_records,
                    len(
                        streams
                    ),
                    "dna_mismatch",
                )
            ),

            "pairwise_expected_from_marginals": (
                descriptive_summary(
                    pair_dna_expected_marginal
                )
            ),

            "pairwise_residual_from_marginals": (
                descriptive_summary(
                    pair_dna_residual_marginal
                )
            ),

            "pairwise_mutual_information_bits": (
                descriptive_summary(
                    pair_dna_mi
                )
            ),
        },

        "notes": [
            (
                "Pooled Wilson aralıkları ve p-değerleri, "
                "önceki Entropy analiziyle karşılaştırılabilirlik "
                "için korunmuştur."
            ),

            (
                "Aynı akış birden fazla çiftte kullanıldığı için "
                "koşum-seviyeli leave-one-stream-out jackknife "
                "güven aralıkları da hesaplanmıştır."
            ),

            (
                "DNA için baz-kompozisyonuna göre düzeltilmiş "
                "beklenti 1-sum_b p_i(b)p_j(b) olarak raporlanmıştır."
            ),
        ],
    }


    # =========================================================================
    # REJİM ÖZETİNİ EKRANA YAZMA
    # =========================================================================

    print(
        "\n"
        + "-" * 100
    )


    print(
        f"{regime} FINAL SUMMARY"
    )


    print(
        "-" * 100
    )


    print(
        f"Bit pooled Hamming       : "
        f"{bit_pooled:.6f}"
    )


    print(
        f"Bit pooled %95 CI        : "
        f"[{bit_pooled_ci[0]:.6f}, "
        f"{bit_pooled_ci[1]:.6f}]"
    )


    print(
        f"Bit p vs 0.5             : "
        f"{format_p_value(summary['bit_domain']['pooled_p_vs_0_5'])}"
    )


    print(
        f"Bit Pearson ort. ± SD    : "
        f"{summary['bit_domain']['pairwise_pearson']['mean']:+.6f} "
        f"± "
        f"{summary['bit_domain']['pairwise_pearson']['sample_sd']:.6f}"
    )


    print(
        f"Bit MI ort. ± SD         : "
        f"{summary['bit_domain']['pairwise_mutual_information_bits']['mean']:.6e} "
        f"± "
        f"{summary['bit_domain']['pairwise_mutual_information_bits']['sample_sd']:.6e}"
    )


    print(
        f"DNA pooled mismatch      : "
        f"{dna_pooled:.6f}"
    )


    print(
        f"DNA pooled %95 CI        : "
        f"[{dna_pooled_ci[0]:.6f}, "
        f"{dna_pooled_ci[1]:.6f}]"
    )


    print(
        f"DNA p vs 0.75            : "
        f"{format_p_value(summary['dna_domain']['pooled_p_vs_0_75'])}"
    )


    print(
        f"DNA marjinal beklenti    : "
        f"{summary['dna_domain']['pairwise_expected_from_marginals']['mean']:.6f}"
    )


    print(
        f"DNA marjinal residual    : "
        f"{summary['dna_domain']['pairwise_residual_from_marginals']['mean']:+.6e}"
    )


    print(
        f"DNA MI ort. ± SD         : "
        f"{summary['dna_domain']['pairwise_mutual_information_bits']['mean']:.6e} "
        f"± "
        f"{summary['dna_domain']['pairwise_mutual_information_bits']['sample_sd']:.6e}"
    )


    return (
        summary,
        pair_records,
        selected_files,
    )


# =============================================================================
# SEÇİLEN DOSYALARI CSV'YE YAZMA
# =============================================================================

def write_selected_files_csv(
    selected_files: Sequence[
        StreamFiles
    ]
) -> None:

    fields = [
        "regime",

        "run_index",

        "folder",

        "dna_path",

        "bits_path",

        "dna_length",

        "bit_length",

        "detection_method",
    ]


    with OUT_SELECTED_FILES_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )


        writer.writeheader()


        for selected in selected_files:

            writer.writerow(
                asdict(
                    selected
                )
            )


# =============================================================================
# 180 ÇİFTİN TAMAMINI CSV'YE YAZMA
# =============================================================================

def write_pairs_csv(
    pair_records: Sequence[
        dict
    ]
) -> None:

    if not pair_records:

        return


    fields = list(
        pair_records[
            0
        ].keys()
    )


    with OUT_PAIRS_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )


        writer.writeheader()


        writer.writerows(
            pair_records
        )


# =============================================================================
# ANA ÖZET TABLOSU İÇİN DÜZLEŞTİRME
# =============================================================================

def flatten_summary_for_csv(
    summary: dict
) -> dict:

    bit = summary[
        "bit_domain"
    ]


    dna = summary[
        "dna_domain"
    ]


    return {
        "regime": (
            summary[
                "regime"
            ]
        ),

        "n_streams": (
            summary[
                "n_streams"
            ]
        ),

        "n_pairs": (
            summary[
                "n_pairs"
            ]
        ),

        # ---------------------------------------------------------------------
        # Bit alanı
        # ---------------------------------------------------------------------

        "bit_total_n": (
            bit[
                "pooled_total_n"
            ]
        ),

        "bit_pooled_hamming": (
            bit[
                "pooled_hamming"
            ]
        ),

        "bit_pooled_ci95_low": (
            bit[
                "pooled_wilson_ci95_low"
            ]
        ),

        "bit_pooled_ci95_high": (
            bit[
                "pooled_wilson_ci95_high"
            ]
        ),

        "bit_p_vs_0_5": (
            bit[
                "pooled_p_vs_0_5"
            ]
        ),

        "bit_pair_mean": (
            bit[
                "pairwise_hamming"
            ][
                "mean"
            ]
        ),

        "bit_pair_sd": (
            bit[
                "pairwise_hamming"
            ][
                "sample_sd"
            ]
        ),

        "bit_jackknife_ci95_low": (
            bit[
                "pairwise_hamming_jackknife_ci"
            ][
                "ci95_low"
            ]
        ),

        "bit_jackknife_ci95_high": (
            bit[
                "pairwise_hamming_jackknife_ci"
            ][
                "ci95_high"
            ]
        ),

        "bit_pearson_mean": (
            bit[
                "pairwise_pearson"
            ][
                "mean"
            ]
        ),

        "bit_pearson_sd": (
            bit[
                "pairwise_pearson"
            ][
                "sample_sd"
            ]
        ),

        "bit_mi_mean": (
            bit[
                "pairwise_mutual_information_bits"
            ][
                "mean"
            ]
        ),

        "bit_mi_sd": (
            bit[
                "pairwise_mutual_information_bits"
            ][
                "sample_sd"
            ]
        ),

        # ---------------------------------------------------------------------
        # DNA alanı
        # ---------------------------------------------------------------------

        "dna_total_n": (
            dna[
                "pooled_total_n"
            ]
        ),

        "dna_pooled_mismatch": (
            dna[
                "pooled_mismatch"
            ]
        ),

        "dna_pooled_ci95_low": (
            dna[
                "pooled_wilson_ci95_low"
            ]
        ),

        "dna_pooled_ci95_high": (
            dna[
                "pooled_wilson_ci95_high"
            ]
        ),

        "dna_p_vs_0_75": (
            dna[
                "pooled_p_vs_0_75"
            ]
        ),

        "dna_pair_mean": (
            dna[
                "pairwise_mismatch"
            ][
                "mean"
            ]
        ),

        "dna_pair_sd": (
            dna[
                "pairwise_mismatch"
            ][
                "sample_sd"
            ]
        ),

        "dna_jackknife_ci95_low": (
            dna[
                "pairwise_mismatch_jackknife_ci"
            ][
                "ci95_low"
            ]
        ),

        "dna_jackknife_ci95_high": (
            dna[
                "pairwise_mismatch_jackknife_ci"
            ][
                "ci95_high"
            ]
        ),

        "dna_expected_marginal_mean": (
            dna[
                "pairwise_expected_from_marginals"
            ][
                "mean"
            ]
        ),

        "dna_residual_marginal_mean": (
            dna[
                "pairwise_residual_from_marginals"
            ][
                "mean"
            ]
        ),

        "dna_mi_mean": (
            dna[
                "pairwise_mutual_information_bits"
            ][
                "mean"
            ]
        ),

        "dna_mi_sd": (
            dna[
                "pairwise_mutual_information_bits"
            ][
                "sample_sd"
            ]
        ),
    }


# =============================================================================
# ANA ÖZET CSV
# =============================================================================

def write_summary_csv(
    summaries: Sequence[
        dict
    ]
) -> None:

    rows = [
        flatten_summary_for_csv(
            summary
        )
        for summary in summaries
    ]


    if not rows:

        return


    fields = list(
        rows[
            0
        ].keys()
    )


    with OUT_SUMMARY_CSV.open(
        "w",
        encoding="utf-8-sig",
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


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main() -> None:

    print(
        "T5 MULTI-STREAM INDEPENDENCE ANALYSIS"
    )


    print(
        f"Çalışma zamanı            : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    print(
        f"Ana klasör                : "
        f"{BASE_DIR}"
    )


    print(
        f"Her rejimdeki akış sayısı : "
        f"{N_STREAMS}"
    )


    print(
        f"Her rejimdeki çift sayısı : "
        f"{N_STREAMS * (N_STREAMS - 1) // 2}"
    )


    print(
        f"Beklenen DNA uzunluğu     : "
        f"{EXPECTED_DNA_BASES:,}"
    )


    print(
        f"Beklenen bit uzunluğu     : "
        f"{EXPECTED_BITS:,}"
    )


    print(
        f"Pair ayrıntıları          : "
        f"{PRINT_PAIR_DETAILS}"
    )


    all_summaries: List[
        dict
    ] = []


    all_pairs: List[
        dict
    ] = []


    all_selected_files: List[
        StreamFiles
    ] = []


    for (
        regime,
        folder_template
    ) in REGIMES:

        (
            summary,
            pair_records,
            selected_files,
        ) = analyze_regime(
            regime,
            folder_template,
        )


        all_summaries.append(
            summary
        )


        all_pairs.extend(
            pair_records
        )


        all_selected_files.extend(
            selected_files
        )


    write_selected_files_csv(
        all_selected_files
    )


    write_pairs_csv(
        all_pairs
    )


    write_summary_csv(
        all_summaries
    )


    report = {
        "analysis": (
            "T5 multi-stream independence "
            "and DNA-domain mismatch"
        ),

        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "base_directory": str(
            BASE_DIR
        ),

        "configuration": {
            "n_streams_per_regime": (
                N_STREAMS
            ),

            "expected_pairs_per_regime": (
                N_STREAMS
                * (
                    N_STREAMS
                    - 1
                )
                // 2
            ),

            "expected_dna_bases": (
                EXPECTED_DNA_BASES
            ),

            "expected_bits": (
                EXPECTED_BITS
            ),

            "minimum_content_purity": (
                MIN_CONTENT_PURITY
            ),

            "strict_mode": (
                STRICT_MODE
            ),
        },

        "selected_files": [
            asdict(
                selected
            )
            for selected in all_selected_files
        ],

        "summaries": (
            all_summaries
        ),

        "pair_records": (
            all_pairs
        ),

        "interpretation_notes": [
            (
                "Bit Hamming değerlerinin 0.5'e yakın olması, "
                "dengeli bağımsız bit akışlarıyla uyumludur."
            ),

            (
                "DNA mismatch değerlerinin 0.75'e yakın olması, "
                "uniform ve bağımsız dört sembollü akışlarla uyumludur."
            ),

            (
                "DNA kompozisyonları uniform değilse, "
                "1-sum_b p_i(b)p_j(b) biçimindeki düzeltilmiş beklenti "
                "yalnız 0.75 referansından daha uygun olabilir."
            ),

            (
                "Sıfıra yakın Pearson korelasyonu ve mutual information, "
                "düşük ikili bağımlılığı destekler; ancak biçimsel "
                "kriptografik güvenlik kanıtı değildir."
            ),
        ],
    }


    with OUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )


        handle.write(
            "\n"
        )


    # =========================================================================
    # SON KISA TABLO
    # =========================================================================

    print(
        "\n"
        + "=" * 100
    )


    print(
        "FINAL COMPACT TABLE"
    )


    print(
        "=" * 100
    )


    for summary in all_summaries:

        bit = summary[
            "bit_domain"
        ]


        dna = summary[
            "dna_domain"
        ]


        print(
            f"{summary['regime']:7s} | "
            f"Bit="
            f"{bit['pooled_hamming']:.6f} "
            f"["
            f"{bit['pooled_wilson_ci95_low']:.6f}, "
            f"{bit['pooled_wilson_ci95_high']:.6f}"
            f"] | "
            f"DNA="
            f"{dna['pooled_mismatch']:.6f} "
            f"["
            f"{dna['pooled_wilson_ci95_low']:.6f}, "
            f"{dna['pooled_wilson_ci95_high']:.6f}"
            f"] | "
            f"DNA beklenen(marjinal)="
            f"{dna['pairwise_expected_from_marginals']['mean']:.6f}"
        )


    print(
        "\nKaydedilen dosyalar:"
    )


    print(
        f"  TXT raporu        : "
        f"{OUT_TEXT}"
    )


    print(
        f"  Seçilen dosyalar  : "
        f"{OUT_SELECTED_FILES_CSV}"
    )


    print(
        f"  Pair sonuçları    : "
        f"{OUT_PAIRS_CSV}"
    )


    print(
        f"  Ana özet tablosu  : "
        f"{OUT_SUMMARY_CSV}"
    )


    print(
        f"  JSON raporu       : "
        f"{OUT_JSON}"
    )


if __name__ == "__main__":

    original_stdout = sys.stdout


    with OUT_TEXT.open(
        "w",
        encoding="utf-8",
    ) as text_handle:

        sys.stdout = Tee(
            original_stdout,
            text_handle,
        )


        try:

            main()


        finally:

            sys.stdout = (
                original_stdout
            )
