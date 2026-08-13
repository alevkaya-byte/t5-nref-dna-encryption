from __future__ import annotations

import argparse
import hashlib
import io
import math
import platform
import random
import sys
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy
from scipy.fft import fft
from scipy.special import erfc, gammaincc, hyp1f1
from scipy.stats import norm


ALPHA = 0.01
VERSION = "audit-v4-clean"
VERIFIED_ENVIRONMENT = "Python 3.12.13 | NumPy 2.3.5 | SciPy 1.17.0"
MIN_BASIC_BITS = 100
MIN_DFT_BITS = 1_000
MIN_STANDARD_BITS = 1_000_000


@dataclass(frozen=True)
class TestResult:
    """Tek bir p-değerini ve uygulanabilirlik durumunu taşır."""

    name: str
    p_value: Optional[float]
    detail: str = ""

    def status(self, alpha: float = ALPHA) -> str:
        if self.p_value is None:
            return "NOT APPLICABLE"
        if not math.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0:
            return "INVALID"
        return "PASS" if self.p_value >= alpha else "FAIL"


# =============================================================================
# Girdi, raporlama ve provenance
# =============================================================================

def validate_bits(bits: str, *, allow_empty: bool = False) -> None:
    if not bits and not allow_empty:
        raise ValueError("Bit dizisi boş olamaz.")
    invalid = set(bits) - {"0", "1"}
    if invalid:
        shown = ", ".join(repr(value) for value in sorted(invalid)[:8])
        raise ValueError(f"Bit dizisi yalnızca '0' ve '1' içermelidir; geçersiz: {shown}")


def read_bits_from_file(
    path: str | Path,
    bit_count: Optional[int] = None,
) -> str:
    text = Path(path).read_text(encoding="utf-8")
    bits = "".join(text.split())
    validate_bits(bits)
    if bit_count is not None:
        if bit_count <= 0:
            raise ValueError("--n pozitif bir tam sayı olmalıdır.")
        if len(bits) < bit_count:
            raise ValueError(
                f"Dosyada {len(bits):,} bit var; istenen {bit_count:,} bit yok."
            )
        bits = bits[:bit_count]
    return bits


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits_sha256(bits: str) -> str:
    return hashlib.sha256(bits.encode("ascii")).hexdigest()


def print_provenance(path: str | Path, bits: str) -> None:
    print("=" * 78)
    print("SP 800-22 Rev. 1a — TEK BİT AKIŞI TANI ÇALIŞTIRICISI")
    print("=" * 78)
    print(f"Dosya           : {Path(path)}")
    print(f"Analiz bit sayısı: {len(bits):,}")
    print(f"Dosya SHA-256   : {file_sha256(path)}")
    print(f"Analiz SHA-256  : {bits_sha256(bits)}")
    print(f"Kod SHA-256     : {file_sha256(Path(__file__))}")
    print(f"Kod sürümü      : {VERSION}")
    print(f"Python          : {platform.python_version()}")
    print(f"NumPy           : {np.__version__}")
    print(f"SciPy           : {scipy.__version__}")
    print(f"Platform        : {platform.platform()}")
    print(f"Doğrulanan ortam: {VERIFIED_ENVIRONMENT}")
    if (
        platform.python_version() != "3.12.13"
        or np.__version__ != "2.3.5"
        or scipy.__version__ != "1.17.0"
    ):
        print("UYARI: Paket ortamı doğrulanan sürümlerden farklı; --self-test çalıştırın.")
    print("UYARI: Tek-akış p-değerleri kriptografik güvenlik kanıtı değildir.")
    print("=" * 78)


def print_result(result: TestResult, alpha: float = ALPHA) -> None:
    if result.p_value is None:
        p_text = "N/A"
    elif math.isfinite(result.p_value):
        p_text = format_p_value(result.p_value)
    else:
        p_text = repr(result.p_value)
    detail = f" | {result.detail}" if result.detail else ""
    print(f"{result.name}: p={p_text} -> {result.status(alpha)}{detail}")


def format_p_value(p_value: float) -> str:
    """Karar sınırının iki tarafını görünür tutacak p-değeri gösterimi."""
    return f"{p_value:.12g}"


# =============================================================================
# 1. Frequency (Monobit)
# =============================================================================

def nist_monobit(bits: str) -> Optional[float]:
    validate_bits(bits)
    n = len(bits)
    if n < MIN_BASIC_BITS:
        return None
    total = 2 * bits.count("1") - n
    return float(erfc(abs(total) / math.sqrt(2.0 * n)))


# =============================================================================
# 2. Frequency within a Block
# =============================================================================

def nist_block_frequency(bits: str, block_size: int = 128) -> Optional[float]:
    validate_bits(bits)
    if block_size <= 0:
        raise ValueError("Blok boyutu pozitif olmalıdır.")
    if len(bits) < MIN_BASIC_BITS or block_size < 20:
        return None
    number_of_blocks = len(bits) // block_size
    if number_of_blocks == 0:
        return None

    chi_squared = 0.0
    for index in range(number_of_blocks):
        block = bits[index * block_size:(index + 1) * block_size]
        proportion = block.count("1") / block_size
        chi_squared += 4.0 * block_size * (proportion - 0.5) ** 2
    return float(gammaincc(number_of_blocks / 2.0, chi_squared / 2.0))


# =============================================================================
# 3. Runs
# =============================================================================

def nist_runs(bits: str) -> Optional[float]:
    validate_bits(bits)
    n = len(bits)
    if n < MIN_BASIC_BITS:
        return None
    proportion = bits.count("1") / n
    if abs(proportion - 0.5) > 2.0 / math.sqrt(n):
        return None

    observed_runs = 1 + sum(
        bits[index] != bits[index - 1]
        for index in range(1, n)
    )
    numerator = abs(observed_runs - 2.0 * n * proportion * (1.0 - proportion))
    denominator = 2.0 * math.sqrt(2.0 * n) * proportion * (1.0 - proportion)
    return float(erfc(numerator / denominator))


# =============================================================================
# 4. Longest Run of Ones in a Block
# =============================================================================

def _longest_one_run(block: str) -> int:
    longest = 0
    current = 0
    for bit in block:
        if bit == "1":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def nist_longest_run_ones(bits: str) -> Optional[float]:
    validate_bits(bits)
    n = len(bits)
    if n < 128:
        return None
    if n < 6272:
        block_size = 8
        degrees = 3
        boundaries = (1, 2, 3)
        probabilities = (0.21484375, 0.3671875, 0.23046875, 0.1875)
    elif n < 750000:
        block_size = 128
        degrees = 5
        boundaries = (4, 5, 6, 7, 8)
        probabilities = (
            0.1174035788, 0.242955959, 0.249363483,
            0.17517706, 0.102701071, 0.112398847,
        )
    else:
        block_size = 10000
        degrees = 6
        boundaries = (10, 11, 12, 13, 14, 15)
        probabilities = (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727)

    number_of_blocks = n // block_size
    counts = np.zeros(degrees + 1, dtype=np.int64)
    for index in range(number_of_blocks):
        block = bits[index * block_size:(index + 1) * block_size]
        longest = _longest_one_run(block)
        if longest <= boundaries[0]:
            category = 0
        elif longest >= boundaries[-1] + 1:
            category = degrees
        else:
            category = longest - boundaries[0]
        counts[category] += 1

    expected = number_of_blocks * np.asarray(probabilities, dtype=float)
    chi_squared = float(np.sum((counts - expected) ** 2 / expected))
    return float(gammaincc(degrees / 2.0, chi_squared / 2.0))


# =============================================================================
# 5. Binary Matrix Rank
# =============================================================================

def gf2_rank(matrix: np.ndarray) -> int:
    """GF(2) üzerinde bağımsız satır/sütun pivotlu Gauss eliminasyonu."""
    array = np.asarray(matrix, dtype=np.uint8).copy() & 1
    if array.ndim != 2:
        raise ValueError("Matris iki boyutlu olmalıdır.")
    rows, columns = array.shape
    rank = 0
    for column in range(columns):
        candidates = np.flatnonzero(array[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            array[[rank, pivot], :] = array[[pivot, rank], :]
        rows_to_clear = np.flatnonzero(array[:, column])
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        if rows_to_clear.size:
            array[rows_to_clear, :] ^= array[rank, :]
        rank += 1
        if rank == min(rows, columns):
            break
    return rank


class BinaryMatrix:
    """Eski çağrılarla uyumluluk için güvenli GF(2) rank sarmalayıcısı."""

    def __init__(self, matrix: np.ndarray, rows: int, columns: int):
        array = np.asarray(matrix, dtype=np.uint8)
        if array.shape != (rows, columns):
            raise ValueError("Bildirilen ve gerçek matris boyutları uyuşmuyor.")
        self.A = array.copy()

    def compute_rank(self, verbose: bool = False) -> int:
        rank = gf2_rank(self.A)
        if verbose:
            print(f"GF(2) rank={rank}\n{self.A}")
        return rank


def binary_matrix_rank_probability(rows: int, columns: int, rank: int) -> float:
    if not 0 <= rank <= min(rows, columns):
        return 0.0
    probability = 2.0 ** (rank * (rows + columns - rank) - rows * columns)
    for index in range(rank):
        numerator = (
            (1.0 - 2.0 ** (index - rows))
            * (1.0 - 2.0 ** (index - columns))
        )
        denominator = 1.0 - 2.0 ** (index - rank)
        probability *= numerator / denominator
    return probability


def nist_binary_matrix_rank(bits: str, q: int = 32) -> Optional[float]:
    validate_bits(bits)
    if q < 2:
        raise ValueError("q en az 2 olmalıdır.")
    block_length = q * q
    number_of_blocks = len(bits) // block_length
    if number_of_blocks < 38:
        return None

    counts = np.zeros(3, dtype=np.int64)
    for index in range(number_of_blocks):
        block = bits[index * block_length:(index + 1) * block_length]
        matrix = (
            np.frombuffer(block.encode("ascii"), dtype=np.uint8) - ord("0")
        ).reshape(q, q)
        rank = gf2_rank(matrix)
        counts[0 if rank == q else 1 if rank == q - 1 else 2] += 1

    p_full = binary_matrix_rank_probability(q, q, q)
    p_minus_one = binary_matrix_rank_probability(q, q, q - 1)
    probabilities = np.array(
        [p_full, p_minus_one, 1.0 - p_full - p_minus_one],
        dtype=float,
    )
    expected = number_of_blocks * probabilities
    chi_squared = float(np.sum((counts - expected) ** 2 / expected))
    return float(math.exp(-chi_squared / 2.0))


# =============================================================================
# 6. Discrete Fourier Transform (Spectral)
# =============================================================================

def nist_spectral(bits: str) -> Optional[float]:
    validate_bits(bits)
    n = len(bits)
    if n < MIN_DFT_BITS:
        return None
    numeric = np.frombuffer(bits.encode("ascii"), dtype=np.uint8)
    plus_minus_one = np.where(numeric == ord("1"), 1.0, -1.0)
    magnitudes = np.abs(fft(plus_minus_one)[: n // 2])
    threshold = math.sqrt(math.log(1.0 / 0.05) * n)
    expected_below = 0.95 * n / 2.0
    observed_below = float(np.sum(magnitudes < threshold))
    normalized = (
        (observed_below - expected_below)
        / math.sqrt(n * 0.95 * 0.05 / 4.0)
    )
    return float(erfc(abs(normalized) / math.sqrt(2.0)))


# =============================================================================
# 7. Non-overlapping Template Matching
# =============================================================================

def is_nonperiodic_template(pattern: str) -> bool:
    """NIST template dosyalarındaki unbordered/non-periodic koşul."""
    validate_bits(pattern)
    return all(pattern[:size] != pattern[-size:] for size in range(1, len(pattern)))


def generate_nonperiodic_templates(pattern_size: int = 9) -> list[str]:
    if not 2 <= pattern_size <= 16:
        raise ValueError("Şablon boyutu bu uygulamada 2..16 aralığında olmalıdır.")
    return [
        f"{value:0{pattern_size}b}"
        for value in range(2 ** pattern_size)
        if is_nonperiodic_template(f"{value:0{pattern_size}b}")
    ]


def _count_nonoverlapping(block: str, pattern: str) -> int:
    count = 0
    start = 0
    while True:
        position = block.find(pattern, start)
        if position < 0:
            return count
        count += 1
        start = position + len(pattern)


def nist_non_overlapping_template(
    bits: str,
    pattern: str = "000000001",
    number_of_blocks: int = 8,
) -> Optional[float]:
    validate_bits(bits)
    validate_bits(pattern)
    if not is_nonperiodic_template(pattern):
        raise ValueError("Non-overlap şablonu non-periodic (unbordered) olmalıdır.")
    if number_of_blocks <= 0:
        raise ValueError("Blok sayısı pozitif olmalıdır.")
    if len(bits) < MIN_STANDARD_BITS or number_of_blocks > 100:
        return None
    block_size = len(bits) // number_of_blocks
    pattern_size = len(pattern)
    if pattern_size == 0 or pattern_size > block_size:
        return None

    counts = np.array(
        [
            _count_nonoverlapping(
                bits[index * block_size:(index + 1) * block_size],
                pattern,
            )
            for index in range(number_of_blocks)
        ],
        dtype=float,
    )
    mean = (block_size - pattern_size + 1) / (2.0 ** pattern_size)
    variance = block_size * (
        2.0 ** (-pattern_size)
        - (2.0 * pattern_size - 1.0) * 2.0 ** (-2 * pattern_size)
    )
    if variance <= 0.0:
        return None
    chi_squared = float(np.sum((counts - mean) ** 2 / variance))
    return float(gammaincc(number_of_blocks / 2.0, chi_squared / 2.0))


def nist_non_overlapping_templates(
    bits: str,
    pattern_size: int = 9,
    number_of_blocks: int = 8,
) -> list[tuple[str, Optional[float]]]:
    validate_bits(bits)
    return [
        (
            pattern,
            nist_non_overlapping_template(bits, pattern, number_of_blocks),
        )
        for pattern in generate_nonperiodic_templates(pattern_size)
    ]


# =============================================================================
# 8. Overlapping Template Matching
# =============================================================================

def _overlap_probability(category: int, eta: float) -> float:
    if category == 0:
        return float(math.exp(-eta))
    return float(
        eta
        * math.exp(-2.0 * eta)
        * 2.0 ** (-category)
        * hyp1f1(category + 1, 2, eta)
    )


def nist_overlapping_template(
    bits: str,
    pattern_size: int = 9,
    block_size: int = 1032,
) -> Optional[float]:
    validate_bits(bits)
    if pattern_size <= 0 or block_size <= 0:
        raise ValueError("Şablon ve blok boyutu pozitif olmalıdır.")
    if pattern_size > block_size:
        return None
    if len(bits) < MIN_STANDARD_BITS:
        return None
    number_of_blocks = len(bits) // block_size
    if number_of_blocks == 0:
        return None

    pattern = "1" * pattern_size
    eta = (block_size - pattern_size + 1) / (2.0 ** pattern_size) / 2.0
    probabilities = [_overlap_probability(category, eta) for category in range(5)]
    probabilities.append(1.0 - sum(probabilities))
    counts = np.zeros(6, dtype=np.int64)

    for index in range(number_of_blocks):
        block = bits[index * block_size:(index + 1) * block_size]
        hits = 0
        start = 0
        while True:
            position = block.find(pattern, start)
            if position < 0:
                break
            hits += 1
            start = position + 1
        counts[min(hits, 5)] += 1

    expected = number_of_blocks * np.asarray(probabilities, dtype=float)
    if np.any(expected <= 0.0):
        return None
    chi_squared = float(np.sum((counts - expected) ** 2 / expected))
    return float(gammaincc(5.0 / 2.0, chi_squared / 2.0))


# =============================================================================
# 9. Maurer's Universal Statistical
# =============================================================================

UNIVERSAL_THRESHOLDS = (
    (6, 387840), (7, 904960), (8, 2068480), (9, 4654080),
    (10, 10342400), (11, 22753280), (12, 49643520),
    (13, 107560960), (14, 231669760), (15, 496435200),
    (16, 1059061760),
)
UNIVERSAL_EXPECTED = (
    0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656,
    8.1764248, 9.1723243, 10.170032, 11.168765, 12.168070,
    13.167693, 14.167488, 15.167379,
)
UNIVERSAL_VARIANCE = (
    0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311,
    3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421,
)


def nist_universal(bits: str) -> Optional[float]:
    validate_bits(bits)
    n = len(bits)
    pattern_size = 5
    for candidate, threshold in UNIVERSAL_THRESHOLDS:
        if n >= threshold:
            pattern_size = candidate
    if pattern_size < 6:
        return None

    initialization_blocks = 10 * (2 ** pattern_size)
    total_blocks = n // pattern_size
    test_blocks = total_blocks - initialization_blocks
    if test_blocks <= 0:
        return None

    last_seen = np.zeros(2 ** pattern_size, dtype=np.int64)
    accumulated = 0.0
    for index in range(total_blocks):
        block = bits[index * pattern_size:(index + 1) * pattern_size]
        value = int(block, 2)
        one_based_index = index + 1
        if index < initialization_blocks:
            last_seen[value] = one_based_index
        else:
            distance = one_based_index - int(last_seen[value])
            accumulated += math.log2(distance)
            last_seen[value] = one_based_index

    phi = accumulated / test_blocks
    correction = (
        0.7 - 0.8 / pattern_size
        + (4.0 + 32.0 / pattern_size)
        * test_blocks ** (-3.0 / pattern_size) / 15.0
    )
    sigma = correction * math.sqrt(
        UNIVERSAL_VARIANCE[pattern_size] / test_blocks
    )
    argument = abs(phi - UNIVERSAL_EXPECTED[pattern_size]) / (math.sqrt(2.0) * sigma)
    return float(erfc(argument))


# =============================================================================
# 10. Linear Complexity
# =============================================================================

def berlekamp_massey(block: str) -> int:
    validate_bits(block, allow_empty=True)
    n = len(block)
    if n == 0:
        return 0
    sequence = [int(bit) for bit in block]
    connection = [0] * n
    backup = [0] * n
    connection[0] = 1
    backup[0] = 1
    complexity = 0
    last_update = -1

    for position in range(n):
        discrepancy = sequence[position]
        for offset in range(1, complexity + 1):
            discrepancy ^= connection[offset] & sequence[position - offset]
        if discrepancy == 0:
            continue

        previous = connection.copy()
        shift = position - last_update
        for index in range(n - shift):
            connection[index + shift] ^= backup[index]
        if complexity <= position // 2:
            complexity = position + 1 - complexity
            last_update = position
            backup = previous
    return complexity


def nist_linear_complexity(bits: str, block_size: int = 500) -> Optional[float]:
    validate_bits(bits)
    if block_size <= 0:
        raise ValueError("Blok boyutu pozitif olmalıdır.")
    if len(bits) < MIN_STANDARD_BITS or not 500 <= block_size <= 5000:
        return None
    number_of_blocks = len(bits) // block_size
    if number_of_blocks < 200:
        return None

    probabilities = np.array(
        [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833],
        dtype=float,
    )
    sign_for_mean = -1 if (block_size + 1) % 2 == 0 else 1
    mean = (
        block_size / 2.0
        + (9.0 + sign_for_mean) / 36.0
        - (block_size / 3.0 + 2.0 / 9.0) * (2.0 ** (-block_size))
    )
    sign = 1 if block_size % 2 == 0 else -1
    counts = np.zeros(7, dtype=np.int64)

    for index in range(number_of_blocks):
        block = bits[index * block_size:(index + 1) * block_size]
        statistic = sign * (berlekamp_massey(block) - mean) + 2.0 / 9.0
        if statistic <= -2.5:
            category = 0
        elif statistic <= -1.5:
            category = 1
        elif statistic <= -0.5:
            category = 2
        elif statistic <= 0.5:
            category = 3
        elif statistic <= 1.5:
            category = 4
        elif statistic <= 2.5:
            category = 5
        else:
            category = 6
        counts[category] += 1

    expected = number_of_blocks * probabilities
    chi_squared = float(np.sum((counts - expected) ** 2 / expected))
    return float(gammaincc(3.0, chi_squared / 2.0))


# =============================================================================
# 11. Serial
# =============================================================================

def _serial_psi(bits: str, pattern_size: int) -> float:
    if pattern_size <= 0:
        return 0.0
    n = len(bits)
    extended = bits + bits[:pattern_size - 1]
    counts = np.zeros(2 ** pattern_size, dtype=np.int64)
    for index in range(n):
        counts[int(extended[index:index + pattern_size], 2)] += 1
    return float(np.sum(counts ** 2) * (2 ** pattern_size) / n - n)


def nist_serial(bits: str, pattern_size: int = 16) -> tuple[Optional[float], Optional[float]]:
    validate_bits(bits)
    if pattern_size < 2:
        raise ValueError("Serial örüntü boyutu en az 2 olmalıdır.")
    if pattern_size >= math.floor(math.log2(len(bits))) - 2:
        return None, None
    psi_m = _serial_psi(bits, pattern_size)
    psi_m1 = _serial_psi(bits, pattern_size - 1)
    psi_m2 = _serial_psi(bits, pattern_size - 2)
    delta_1 = psi_m - psi_m1
    delta_2 = psi_m - 2.0 * psi_m1 + psi_m2
    p_1 = gammaincc(2 ** (pattern_size - 2), delta_1 / 2.0)
    p_2 = gammaincc(2 ** (pattern_size - 3), delta_2 / 2.0)
    return float(p_1), float(p_2)


# =============================================================================
# 12. Approximate Entropy
# =============================================================================

def nist_approximate_entropy(bits: str, pattern_size: int = 10) -> Optional[float]:
    validate_bits(bits)
    if pattern_size <= 0:
        raise ValueError("Örüntü boyutu pozitif olmalıdır.")
    n = len(bits)
    if pattern_size >= math.floor(math.log2(n)) - 5:
        return None

    return _approximate_entropy_core(bits, pattern_size)


def _approximate_entropy_core(bits: str, pattern_size: int) -> float:
    """Uygulanabilirliği dışarıda denetlenmiş Approximate Entropy çekirdeği."""
    n = len(bits)

    # m+1 dairesel pencereler için ilk m bit mutlaka eklenmelidir.
    extended = bits + bits[:pattern_size]
    phi_values = []
    for size in (pattern_size, pattern_size + 1):
        counts = Counter(extended[index:index + size] for index in range(n))
        phi_values.append(
            sum(count * math.log(count / n) for count in counts.values()) / n
        )
    approximate_entropy = phi_values[0] - phi_values[1]
    chi_squared = 2.0 * n * (math.log(2.0) - approximate_entropy)
    return float(gammaincc(2 ** (pattern_size - 1), chi_squared / 2.0))


# =============================================================================
# 13. Cumulative Sums
# =============================================================================

def nist_cumulative_sums(bits: str, mode: int = 0) -> Optional[float]:
    validate_bits(bits)
    if mode not in (0, 1):
        raise ValueError("mode yalnızca 0 (ileri) veya 1 (geri) olabilir.")
    if len(bits) < MIN_BASIC_BITS:
        return None
    data = bits if mode == 0 else bits[::-1]
    walk = np.cumsum(
        np.where(
            np.frombuffer(data.encode("ascii"), dtype=np.uint8) == ord("1"),
            1,
            -1,
        ),
        dtype=np.int64,
    )
    maximum = int(np.max(np.abs(walk)))
    if maximum == 0:
        return 1.0

    n = len(bits)
    root_n = math.sqrt(n)
    first_sum = 0.0
    start = math.floor((-n / maximum + 1.0) / 4.0)
    end = math.floor((n / maximum - 1.0) / 4.0)
    for index in range(start, end + 1):
        first_sum += (
            norm.cdf((4 * index + 1) * maximum / root_n)
            - norm.cdf((4 * index - 1) * maximum / root_n)
        )

    second_sum = 0.0
    start = math.floor((-n / maximum - 3.0) / 4.0)
    end = math.floor((n / maximum - 1.0) / 4.0)
    for index in range(start, end + 1):
        second_sum += (
            norm.cdf((4 * index + 3) * maximum / root_n)
            - norm.cdf((4 * index + 1) * maximum / root_n)
        )
    return float(1.0 - first_sum + second_sum)


# =============================================================================
# 14–15. Random Excursions yardımcıları
# =============================================================================

def random_walk(bits: str) -> np.ndarray:
    validate_bits(bits)
    numeric = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
    return np.cumsum(2 * numeric.astype(np.int64) - 1, dtype=np.int64)


def random_walk_cycles(walk: np.ndarray) -> list[np.ndarray]:
    if walk.size == 0:
        return []
    zero_positions = np.flatnonzero(walk == 0)
    cycles = list(np.split(walk, zero_positions + 1))
    if cycles and cycles[-1].size == 0:
        cycles.pop()
    return cycles


def excursions_minimum_cycles(n: int) -> float:
    return max(0.005 * math.sqrt(n), 500.0)


def _excursion_probability(visits: int, state: int) -> float:
    absolute_state = abs(state)
    if visits == 0:
        return 1.0 - 1.0 / (2.0 * absolute_state)
    if visits >= 5:
        return (
            1.0 / (2.0 * absolute_state)
            * (1.0 - 1.0 / (2.0 * absolute_state)) ** 4
        )
    return (
        1.0 / (4.0 * state * state)
        * (1.0 - 1.0 / (2.0 * absolute_state)) ** (visits - 1)
    )


# =============================================================================
# 14. Random Excursions
# =============================================================================

def nist_random_excursions(
    bits: str,
) -> Optional[list[tuple[int, float, float]]]:
    walk = random_walk(bits)
    if len(bits) < MIN_STANDARD_BITS:
        return None
    cycles = random_walk_cycles(walk)
    number_of_cycles = len(cycles)
    if number_of_cycles < excursions_minimum_cycles(len(bits)):
        return None

    states = (-4, -3, -2, -1, 1, 2, 3, 4)
    results = []
    for state in states:
        categories = np.zeros(6, dtype=np.int64)
        for cycle in cycles:
            visits = int(np.sum(cycle == state))
            categories[min(visits, 5)] += 1
        probabilities = np.array(
            [_excursion_probability(visits, state) for visits in range(6)],
            dtype=float,
        )
        expected = number_of_cycles * probabilities
        chi_squared = float(np.sum((categories - expected) ** 2 / expected))
        p_value = float(gammaincc(2.5, chi_squared / 2.0))
        results.append((state, chi_squared, p_value))
    return results


# =============================================================================
# 15. Random Excursions Variant
# =============================================================================

def nist_random_excursions_variant(
    bits: str,
) -> Optional[list[tuple[int, int, float]]]:
    walk = random_walk(bits)
    if len(bits) < MIN_STANDARD_BITS:
        return None
    number_of_cycles = len(random_walk_cycles(walk))
    if number_of_cycles < excursions_minimum_cycles(len(bits)):
        return None

    results = []
    for state in list(range(-9, 0)) + list(range(1, 10)):
        visits = int(np.sum(walk == state))
        denominator = math.sqrt(
            2.0 * number_of_cycles * (4.0 * abs(state) - 2.0)
        )
        p_value = float(erfc(abs(visits - number_of_cycles) / denominator))
        results.append((state, visits, p_value))
    return results


# =============================================================================
# Tüm testleri çalıştırma
# =============================================================================

def run_suite(
    path: str | Path,
    block_sizes: Iterable[int] = (128,),
    rank_q: int = 32,
    nonover_pattern_size: int = 9,
    nonover_blocks: int = 8,
    overlap_pattern_size: int = 9,
    overlap_block_size: int = 1032,
    linear_block_size: int = 500,
    serial_pattern_size: int = 16,
    approximate_entropy_pattern_size: int = 10,
    alpha: float = ALPHA,
    bit_count: Optional[int] = None,
) -> list[TestResult]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 0 ile 1 arasında olmalıdır.")
    bits = read_bits_from_file(path, bit_count)
    print_provenance(path, bits)
    n = len(bits)
    results: list[TestResult] = []

    results.append(TestResult(
        "[01] Frequency (Monobit)",
        nist_monobit(bits),
        f"n={n:,}; ones={bits.count('1'):,}",
    ))
    for block_size in block_sizes:
        p_value = nist_block_frequency(bits, block_size)
        number_of_blocks = n // block_size
        results.append(TestResult(
            f"[02] Block Frequency (M={block_size})",
            p_value,
            f"N={number_of_blocks:,}; discarded={n % block_size:,}",
        ))
    proportion = bits.count("1") / n
    results.append(TestResult(
        "[03] Runs",
        nist_runs(bits),
        f"pi={proportion:.12g}; monobit prerequisite",
    ))
    longest_block_size = 8 if n < 6272 else 128 if n < 750000 else 10000
    results.append(TestResult(
        "[04] Longest Run of Ones",
        nist_longest_run_ones(bits),
        f"M={longest_block_size}; N={n // longest_block_size:,}; "
        f"discarded={n % longest_block_size:,}",
    ))
    results.append(TestResult(
        f"[05] Binary Matrix Rank ({rank_q}x{rank_q})",
        nist_binary_matrix_rank(bits, rank_q),
        f"N={n // (rank_q * rank_q):,}; discarded={n % (rank_q * rank_q):,}",
    ))
    results.append(TestResult(
        "[06] Discrete Fourier Transform",
        nist_spectral(bits),
        f"n={n:,}; NIST minimum=1,000",
    ))

    nonover = nist_non_overlapping_templates(
        bits,
        nonover_pattern_size,
        nonover_blocks,
    )
    for pattern, p_value in nonover:
        results.append(TestResult(
            f"[07] Non-overlap Template ({pattern})",
            p_value,
            f"m={nonover_pattern_size}; N={nonover_blocks}; "
            f"M={n // nonover_blocks:,}; discarded={n % nonover_blocks:,}",
        ))

    results.append(TestResult(
        f"[08] Overlap Template (m={overlap_pattern_size}, M={overlap_block_size})",
        nist_overlapping_template(bits, overlap_pattern_size, overlap_block_size),
        f"N={n // overlap_block_size:,}; discarded={n % overlap_block_size:,}",
    ))
    universal_l = max(
        (candidate for candidate, threshold in UNIVERSAL_THRESHOLDS if n >= threshold),
        default=None,
    )
    if universal_l is None:
        universal_detail = "n < 387,840"
    else:
        universal_q = 10 * (2 ** universal_l)
        universal_k = n // universal_l - universal_q
        universal_detail = (
            f"L={universal_l}; Q={universal_q:,}; K={universal_k:,}; "
            f"discarded={n % universal_l:,}"
        )
    results.append(TestResult(
        "[09] Universal (Maurer)",
        nist_universal(bits),
        universal_detail,
    ))
    results.append(TestResult(
        f"[10] Linear Complexity (M={linear_block_size})",
        nist_linear_complexity(bits, linear_block_size),
        f"N={n // linear_block_size:,}; discarded={n % linear_block_size:,}",
    ))
    serial_first, serial_second = nist_serial(bits, serial_pattern_size)
    serial_detail = f"n={n:,}; m={serial_pattern_size}"
    results.append(TestResult("[11a] Serial Δ1", serial_first, serial_detail))
    results.append(TestResult("[11b] Serial Δ2", serial_second, serial_detail))
    results.append(TestResult(
        f"[12] Approximate Entropy (m={approximate_entropy_pattern_size})",
        nist_approximate_entropy(bits, approximate_entropy_pattern_size),
        f"n={n:,}; m={approximate_entropy_pattern_size}",
    ))
    results.append(TestResult(
        "[13a] Cumulative Sums (forward)",
        nist_cumulative_sums(bits, 0),
        f"n={n:,}",
    ))
    results.append(TestResult(
        "[13b] Cumulative Sums (backward)",
        nist_cumulative_sums(bits, 1),
        f"n={n:,}",
    ))

    observed_cycles = len(random_walk_cycles(random_walk(bits)))
    cycle_detail = (
        f"J={observed_cycles:,}; required J>="
        f"{excursions_minimum_cycles(n):.3f}; n>=1,000,000"
    )
    excursions = nist_random_excursions(bits)
    if excursions is None:
        for state in (-4, -3, -2, -1, 1, 2, 3, 4):
            results.append(TestResult(
                f"[14] Random Excursions (state={state:+d})",
                None,
                cycle_detail,
            ))
    else:
        for state, chi_squared, p_value in excursions:
            results.append(TestResult(
                f"[14] Random Excursions (state={state:+d})",
                p_value,
                f"chi²={chi_squared:.6f}; {cycle_detail}",
            ))

    variant = nist_random_excursions_variant(bits)
    if variant is None:
        for state in list(range(-9, 0)) + list(range(1, 10)):
            results.append(TestResult(
                f"[15] Random Excursions Variant (state={state:+d})",
                None,
                cycle_detail,
            ))
    else:
        for state, visits, p_value in variant:
            results.append(TestResult(
                f"[15] Random Excursions Variant (state={state:+d})",
                p_value,
                f"visits={visits}; {cycle_detail}",
            ))

    for result in results:
        print_result(result, alpha)

    statuses = Counter(result.status(alpha) for result in results)
    print("-" * 78)
    print(
        "ÖZET | "
        f"PASS={statuses['PASS']} | FAIL={statuses['FAIL']} | "
        f"N/A={statuses['NOT APPLICABLE']} | INVALID={statuses['INVALID']}"
    )
    print(
        "NOT: Çok sayıda p-değeri birlikte üretildiğinden, tek bir FAIL sonucu "
        "tek başına üretecin başarısız olduğuna karar vermek için yeterli değildir."
    )
    return results


# =============================================================================
# Yerleşik bilimsel doğrulama testleri
# =============================================================================

def _brute_force_gf2_rank(matrix: np.ndarray) -> int:
    row_values = []
    for row in matrix:
        value = 0
        for bit in row:
            value = (value << 1) | int(bit)
        row_values.append(value)
    span = {0}
    for row_value in row_values:
        span.update(value ^ row_value for value in tuple(span))
    return len(span).bit_length() - 1


def _approximate_entropy_oracle(bits: str, pattern_size: int) -> float:
    n = len(bits)
    phi = []
    for size in (pattern_size, pattern_size + 1):
        counts = Counter(
            "".join(bits[(start + offset) % n] for offset in range(size))
            for start in range(n)
        )
        phi.append(sum(count * math.log(count / n) for count in counts.values()) / n)
    chi_squared = 2.0 * n * (math.log(2.0) - (phi[0] - phi[1]))
    return float(gammaincc(2 ** (pattern_size - 1), chi_squared / 2.0))


def _assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


# NIST STS 2.1.2'nin AlgorithmTesting/data.pi, ilk 1,000,000 bit için
# altı ondalıkla yayımladığı 188 birinci-seviye p-değeri. Sıra, STS sırasıdır.
OFFICIAL_STS_PI_EXPECTED = tuple(float(value) for value in """
0.578211 0.380615 0.419268 0.024390 0.083553 0.010186
0.165757 0.382326 0.156875 0.874722 0.581720 0.589575
0.783509 0.624977 0.639322 0.985135 0.288901 0.194427
0.037993 0.265240 0.832686 0.588049 0.409602 0.138110
0.896209 0.929268 0.792044 0.643830 0.270787 0.390738
0.059570 0.181126 0.052244 0.958331 0.912935 0.236531
0.557389 0.595216 0.349108 0.058277 0.315421 0.998656
0.782297 0.626872 0.696020 0.502599 0.045332 0.521905
0.123232 0.384250 0.754650 0.882317 0.778652 0.730925
0.374502 0.103953 0.662572 0.306478 0.657473 0.670630
0.264868 0.395330 0.193497 0.630072 0.288980 0.547465
0.635052 0.484592 0.947644 0.964763 0.744847 0.578795
0.189071 0.024996 0.911318 0.540118 0.145127 0.097345
0.282688 0.354112 0.165757 0.701427 0.539889 0.769862
0.622845 0.510756 0.730925 0.821815 0.662572 0.871170
0.598040 0.455770 0.183003 0.937696 0.880992 0.123153
0.623792 0.715708 0.921678 0.100917 0.045251 0.560980
0.521226 0.429096 0.899681 0.525415 0.416446 0.828509
0.896791 0.760198 0.294489 0.233616 0.632206 0.308219
0.699665 0.349561 0.183059 0.260438 0.650711 0.361381
0.242863 0.451423 0.920656 0.101084 0.860930 0.159428
0.267553 0.557389 0.338800 0.040447 0.849891 0.209872
0.863192 0.939493 0.959616 0.264497 0.722113 0.975779
0.889875 0.152587 0.693548 0.074951 0.505721 0.646084
0.680214 0.421826 0.808279 0.660319 0.058379 0.302607
0.817966 0.005302 0.359532 0.354112 0.296897 0.669012
0.255475 0.143005 0.034354 0.361595 0.628308 0.663369
0.279235 0.639439 0.268428 0.613106 0.844143 0.794540
0.790685 0.627278 0.995094 0.926985 0.854948 0.657527
0.760966 0.687364 0.864963 0.650024 0.760966 0.509815
0.714432 0.954795 0.708635 0.806410 0.945155 0.932760
0.911398 1.000000
""".split())

OFFICIAL_STS_PI_BITS_SHA256 = (
    "417539f94f4d570b0f5c9e655b2e7d5cbb652028bf28a31add155d73a560a34d"
)


def _read_official_sts_pi_bits(path: str | Path) -> str:
    """Haricî STS ``data.pi`` dosyasının ilk 1,000,000 bitini doğrula."""

    bits = read_bits_from_file(path, MIN_STANDARD_BITS)
    if bits_sha256(bits) != OFFICIAL_STS_PI_BITS_SHA256:
        raise AssertionError(
            "Verilen dosyanın ilk 1,000,000 biti resmî NIST STS 2.1.2 "
            "data.pi örneğiyle eşleşmiyor."
        )
    return bits


def _official_sts_vector(bits: str) -> tuple[float, ...]:
    """STS 2.1.2'nin varsayılan 188 p-değerini aynı sırada üret."""
    if len(bits) != MIN_STANDARD_BITS:
        raise ValueError("Resmî STS oracle tam 1,000,000 bit gerektirir.")
    serial = nist_serial(bits, 16)
    excursions = nist_random_excursions(bits)
    variant = nist_random_excursions_variant(bits)
    if excursions is None or variant is None:
        raise AssertionError("Resmî fixture için excursion testleri N/A olamaz.")
    values: list[float] = [
        nist_monobit(bits),
        nist_block_frequency(bits, 128),
        nist_runs(bits),
        nist_longest_run_ones(bits),
        nist_binary_matrix_rank(bits, 32),
        nist_spectral(bits),
    ]
    values.extend(p_value for _, p_value in nist_non_overlapping_templates(bits, 9, 8))
    values.extend([
        nist_overlapping_template(bits, 9, 1032),
        nist_universal(bits),
        nist_linear_complexity(bits, 500),
        serial[0], serial[1],
        nist_approximate_entropy(bits, 10),
        nist_cumulative_sums(bits, 0),
        nist_cumulative_sums(bits, 1),
    ])
    values.extend(p_value for _, _, p_value in excursions)
    values.extend(p_value for _, _, p_value in variant)
    if len(values) != 188 or any(value is None for value in values):
        raise AssertionError(f"STS vektör boyutu/geçerliliği yanlış: {len(values)}")
    return tuple(float(value) for value in values)


def run_self_tests(reference_data: Optional[str | Path] = None) -> None:
    checks = 0

    # 0) Negatif girdi: bütün dış test API'leri geçersiz karakteri reddeder.
    invalid_calls = (
        lambda: nist_monobit("A"),
        lambda: nist_block_frequency("X", 20),
        lambda: nist_runs("2"),
        lambda: nist_longest_run_ones("N"),
        lambda: nist_binary_matrix_rank("x", 32),
        lambda: nist_spectral("?"),
        lambda: nist_non_overlapping_template("A", "000000001", 8),
        lambda: nist_overlapping_template("A"),
        lambda: nist_universal("A"),
        lambda: nist_linear_complexity("A"),
        lambda: nist_serial("A"),
        lambda: nist_approximate_entropy("A"),
        lambda: nist_cumulative_sums("A"),
        lambda: nist_random_excursions("A"),
        lambda: nist_random_excursions_variant("A"),
        lambda: berlekamp_massey("A"),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Bir dış test API'si geçersiz biti kabul etti.")

    # 1) Exhaustive 3x3 GF(2) rank: 512/512 matris bağımsız oracle ile aynı.
    for encoded in range(2 ** 9):
        matrix = np.fromiter(
            (int(bit) for bit in f"{encoded:09b}"),
            dtype=np.uint8,
        ).reshape(3, 3)
        if gf2_rank(matrix) != _brute_force_gf2_rank(matrix):
            raise AssertionError(f"GF(2) rank uyuşmazlığı: {encoded:09b}")
    checks += 512

    # 2) Elle hesaplanabilir 2x2 rank dağılımı.
    _assert_close(binary_matrix_rank_probability(2, 2, 2), 6 / 16)
    _assert_close(binary_matrix_rank_probability(2, 2, 1), 9 / 16)
    _assert_close(binary_matrix_rank_probability(2, 2, 0), 1 / 16)
    checks += 3

    # 3) Approximate Entropy, bağımsız modüler-indeks oracle'ı.
    vectors = (
        "00101101011100101",
        "010101110001011001101",
        "1110001010010110100011011",
    )
    for bits in vectors:
        for pattern_size in (2, 3, 4):
            actual = _approximate_entropy_core(bits, pattern_size)
            expected = _approximate_entropy_oracle(bits, pattern_size)
            _assert_close(actual, expected)
            checks += 1

    # 4) Random-walk sınırı: sıfırda biten yürüyüşe fazladan cycle eklenmez.
    alternating = "10" * 500
    cycles = random_walk_cycles(random_walk(alternating))
    if len(cycles) != 500:
        raise AssertionError(f"Cycle sayısı 500 olmalıydı, bulunan={len(cycles)}")
    if nist_random_excursions(alternating) is not None:
        raise AssertionError("1,000,000 bitten kısa Excursions N/A olmalı.")
    if nist_random_excursions_variant(alternating) is not None:
        raise AssertionError("1,000,000 bitten kısa Variant N/A olmalı.")
    checks += 3

    # 5) NIST m=9 non-periodic template sayısı 148 olmalı.
    templates = generate_nonperiodic_templates(9)
    template_digest = hashlib.sha256("\n".join(templates).encode("ascii")).hexdigest()
    if (
        len(templates) != 148
        or len(set(templates)) != 148
        or template_digest != "2c4d118b09d3eca849863941961d1373c501b488d8c40dc8ab2f365037cfda7c"
    ):
        raise AssertionError(f"m=9 şablon sayısı yanlış: {len(templates)}")
    checks += 3

    # 6) Berlekamp-Massey basit oracles.
    if berlekamp_massey("0" * 64) != 0:
        raise AssertionError("Sıfır dizisinin linear complexity değeri 0 olmalı.")
    if berlekamp_massey("1" * 64) != 1:
        raise AssertionError("Sabit bir dizinin linear complexity değeri 1 olmalı.")
    if berlekamp_massey("01" * 32) != 2:
        raise AssertionError("Alternating dizinin linear complexity değeri 2 olmalı.")
    if berlekamp_massey("") != 0:
        raise AssertionError("Boş dizinin linear complexity değeri 0 olmalı.")
    checks += 4

    # 7) Beş sabit seed: aynı girdi aynı sonuç, p-değerleri sonlu ve [0,1].
    previous_seed_values = None
    for seed in range(5):
        generator = random.Random(seed)
        bits = "".join(str(generator.getrandbits(1)) for _ in range(16_384))

        def diagnostics() -> tuple[Optional[float], ...]:
            serial = nist_serial(bits, 5)
            return (
                nist_monobit(bits),
                nist_block_frequency(bits, 64),
                nist_runs(bits),
                nist_binary_matrix_rank(bits, 8),
                serial[0], serial[1],
                nist_approximate_entropy(bits, 5),
                nist_cumulative_sums(bits, 0),
                nist_cumulative_sums(bits, 1),
            )

        first = diagnostics()
        second = diagnostics()
        if first != second:
            raise AssertionError(f"Determinism hatası, seed={seed}")
        for p_value in first:
            if p_value is None or not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise AssertionError(f"Geçersiz p-değeri, seed={seed}: {p_value!r}")
        if previous_seed_values is not None and first == previous_seed_values:
            raise AssertionError("Farklı seed'ler beklenmedik biçimde tümüyle aynı çıktı verdi.")
        previous_seed_values = first
        checks += len(first) * 2

    # 8) NIST'in belgelediği minimum/önerilen boyutların altında yanlış PASS yok.
    undersized_results = (
        nist_monobit("0"),
        nist_block_frequency("0", 1),
        nist_runs("0"),
        nist_longest_run_ones("0" * 127),
        nist_binary_matrix_rank("0" * 1024, 32),
        nist_spectral("00"),
        nist_non_overlapping_template("0" * 72, "000000001", 8),
        nist_overlapping_template("0" * 1032),
        nist_universal("0" * 1000),
        nist_linear_complexity("0" * 1000, 500),
        *nist_serial("0" * 32, 16),
        nist_approximate_entropy("0" * 11, 10),
        nist_cumulative_sums("0"),
        nist_random_excursions("10" * 500),
        nist_random_excursions_variant("10" * 500),
    )
    if any(value is not None for value in undersized_results):
        raise AssertionError(f"Kısa girdide yanlış uygulanabilirlik: {undersized_results!r}")
    checks += len(undersized_results)

    # Runs önkoşulunun eşitlik sınırı STS 2.1.2 ile aynıdır: '>' kullanılır.
    if nist_runs("1" * 160 + "0" * 96) is None:
        raise AssertionError("Runs eşitlik sınırı yanlışlıkla N/A oldu.")
    checks += 1

    # Periodic şablon, non-overlap API'sinde sessizce kabul edilmez.
    try:
        nist_non_overlapping_template("0" * 100, "000000000", 8)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Periodic non-overlap şablonu reddedilmedi.")

    with tempfile.TemporaryDirectory(prefix="nist22_badparam_") as directory:
        bit_path = Path(directory) / "bits.txt"
        bit_path.write_text("01" * 50, encoding="utf-8")
        try:
            with redirect_stdout(io.StringIO()):
                run_suite(bit_path, block_sizes=(0,))
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Sıfır Block Frequency boyutu ValueError üretmedi.")

    # 9) Karar, ekranda yuvarlanarak belirsizleştirilmez.
    below = TestResult("below", 0.0099996)
    above = TestResult("above", 0.0100004)
    if below.status() != "FAIL" or above.status() != "PASS":
        raise AssertionError("Alpha sınırı kararı yanlış.")
    if format_p_value(below.p_value) == format_p_value(above.p_value):
        raise AssertionError("Alpha sınırının iki tarafı aynı metinle gösteriliyor.")
    checks += 3

    # 10) Dosya prefix seçimi ve sabit 188-result şekli.
    with tempfile.TemporaryDirectory(prefix="nist22_selftest_") as directory:
        bit_path = Path(directory) / "bits.txt"
        bit_path.write_text("01" * 60, encoding="utf-8")
        prefix = read_bits_from_file(bit_path, 100)
        if len(prefix) != 100 or prefix != "01" * 50:
            raise AssertionError("--n/prefix seçimi yanlış.")
        try:
            read_bits_from_file(bit_path, 121)
        except ValueError:
            pass
        else:
            raise AssertionError("Dosyada olmayan prefix uzunluğu kabul edildi.")
        with redirect_stdout(io.StringIO()):
            short_results = run_suite(bit_path, bit_count=100)
        if len(short_results) != 188:
            raise AssertionError(f"Sonuç vektörü 188 değil: {len(short_results)}")
        checks += 3

    # 11) İsteğe bağlı haricî oracle: STS 2.1.2 data.pi üzerinde 188/188.
    official_oracle_checked = reference_data is not None
    if official_oracle_checked:
        print("Resmî NIST STS 2.1.2 oracle karşılaştırması çalışıyor (188 p-değeri)...")
        official_bits = _read_official_sts_pi_bits(reference_data)
        actual_values = _official_sts_vector(official_bits)
        if len(OFFICIAL_STS_PI_EXPECTED) != 188:
            raise AssertionError("Resmî beklenen p-değeri vektörü 188 değil.")
        for index, (actual, expected) in enumerate(
            zip(actual_values, OFFICIAL_STS_PI_EXPECTED),
            start=1,
        ):
            if abs(actual - expected) > 0.5000001e-6:
                raise AssertionError(
                    f"STS oracle uyuşmazlığı #{index}: actual={actual:.12g}, "
                    f"official_6dp={expected:.6f}"
                )
            checks += 1

    print("=" * 78)
    print(f"SELF-TEST BAŞARILI: {checks} doğrulama geçti.")
    print("- Exhaustive 3x3 GF(2) rank oracle: PASS (512/512)")
    print("- Approximate Entropy circular oracle: PASS")
    print("- Random Excursions cycle/state sınırları: PASS")
    print("- NIST m=9 template envanteri: PASS (148/148)")
    print("- Berlekamp-Massey basit oracles: PASS")
    print("- Beş-seed deterministik regresyon: PASS")
    print("- Kısa/uygunsuz girdilerde N/A ve negatif API testleri: PASS")
    print("- Sabit sonuç şekli ve --n prefix sınırları: PASS")
    if official_oracle_checked:
        print("- Haricî resmî NIST STS 2.1.2 data.pi oracle: PASS (188/188, 6 ondalık)")
    else:
        print(
            "- Haricî resmî data.pi oracle: ÇALIŞTIRILMADI "
            "(--reference-data verilmedi)"
        )
    print("=" * 78)


# =============================================================================
# Komut satırı
# =============================================================================

def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SP 800-22 Rev. 1a tek-bit-akışı tanı paketi",
    )
    parser.add_argument(
        "bit_file",
        nargs="?",
        help="Yalnız 0/1 içeren bit akışı dosyası",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Yerleşik oracle, sınır ve deterministik regresyon testlerini çalıştır",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        metavar="DATA_PI",
        help=(
            "--self-test sırasında resmî STS 2.1.2 data.pi dosyasını haricî "
            "oracle olarak kullan"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Anlamlılık düzeyi (varsayılan: 0.01)",
    )
    parser.add_argument(
        "--n",
        dest="bit_count",
        type=int,
        help=(
            "Dosyanın yalnız ilk N bitini analiz et; resmî data.pi karşılaştırması "
            "için --n 1000000 kullanılır"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    arguments = parse_arguments(argv)
    if not arguments.self_test and not arguments.bit_file:
        print("Girdi verilmedi; hiçbir NIST testi çalıştırılmadı.")
        print("Yerleşik doğrulama için:  TEMİZ_SON_NIST.py --self-test")
        print("Bir bit akışı için:       TEMİZ_SON_NIST.py uretilen.bits.txt")
        return 0
    if arguments.reference_data is not None and not arguments.self_test:
        print("HATA: --reference-data yalnızca --self-test ile kullanılabilir.", file=sys.stderr)
        return 2
    try:
        if arguments.self_test:
            run_self_tests(arguments.reference_data)
        if arguments.bit_file:
            run_suite(
                arguments.bit_file,
                alpha=arguments.alpha,
                bit_count=arguments.bit_count,
            )
    except (AssertionError, OSError, ValueError) as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
