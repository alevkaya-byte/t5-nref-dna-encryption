

"""
BMC Bioinformatics – T5-NREF DNA-SPD genom bütünlük doğrulaması.

Amaç:
- Orijinal DNA ile deşifrelenmiş DNA'yı A/C/G/T-kanonik biçimde karşılaştırır.
- SHA-256, baz düzeyi birebir eşleşme, Base Recovery Rate (BRR) ve
  DNA'nın doğal 2-bit gösterimi üzerinden Bit Correction Rate (BCR) hesaplar.
- Şifreleme/deşifreleme metadata dosyalarının birbiriyle tutarlılığını denetler.
- Büyük dosyaları belleğe tamamen almadan parçalı olarak işler.

DNA 2-bit eşlemesi:
    A = 00
    C = 01
    G = 10
    T = 11

Not:
Orijinal FASTA başlıkları, satır sonları ve A/C/G/T dışındaki karakterler
kanonik bütünlük hesabına dahil edilmez. Böylece test, dosya biçimini değil,
gerçek genomik dizinin kayıpsız geri kazanımını ölçer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

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

BASE_DIR = Path(__file__).resolve().parent


# Yalnızca bu dosya adını değiştirmen yeterlidir:
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
    f"integrity_{DATASET_STEM}.json"
)


# Büyük dosyalar bu uzunlukta kanonik DNA parçalarıyla işlenir.
COMPARE_CHUNK_BASES = 1_000_000


# Bütünlük başarısızsa raporu yazdıktan sonra hata üretir.
STRICT_MODE = True


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


BASE_TO_CODE = {
    ord("A"): 0,
    ord("C"): 1,
    ord("G"): 2,
    ord("T"): 3
}


CODE_TO_BASE = [
    "A",
    "C",
    "G",
    "T"
]


# ASCII karakterlerini 0–3 DNA değerlerine dönüştürmek için tablo
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


# XOR sonucu 0–3 için 2-bit Hamming ağırlığı
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
# YARDIMCI İŞLEVLER
# =============================================================================

def optional_float(
    value: Optional[float],
    digits: int = 6
) -> str:

    return (
        "N/A"
        if value is None
        else f"{value:.{digits}f}"
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
            f"JSON kök yapısı nesne olmalıdır: "
            f"{path}"
        )

    return data


def sha256_raw_file(
    path: Path,
    chunk_bytes: int = 4 * 1024 * 1024
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                chunk_bytes
            )

            if not block:

                break

            digest.update(
                block
            )

    return digest.hexdigest()


def count_raw_lines(
    path: Path,
    chunk_bytes: int = 4 * 1024 * 1024
) -> int:

    line_count = 0

    saw_any = False

    last_byte = None

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                chunk_bytes
            )

            if not block:

                break

            saw_any = True

            line_count += block.count(
                b"\n"
            )

            last_byte = block[
                -1
            ]

    if (
        saw_any
        and last_byte != 10
    ):

        line_count += 1

    return line_count


def iter_canonical_dna_chunks(
    path: Path,
    chunk_bases: int
) -> Iterator[bytes]:
    """
    FASTA başlıklarını atar, yalnız A/C/G/T karakterlerini tutar ve
    sabit büyüklükte kanonik DNA parçaları üretir.
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

        for raw_line in handle:

            if raw_line.startswith(
                b">"
            ):

                continue

            for byte in raw_line.upper():

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


def sequence_entropy_from_counts(
    counts: np.ndarray
) -> float:

    total = int(
        counts.sum()
    )

    if total <= 0:

        return 0.0

    probabilities = (
        counts.astype(
            np.float64
        )
        / float(total)
    )

    nonzero = (
        probabilities > 0.0
    )

    return float(
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


def gc_ratio_from_counts(
    counts: np.ndarray
) -> float:

    total = int(
        counts.sum()
    )

    if total <= 0:

        return float(
            "nan"
        )

    return float(
        (
            int(counts[1])
            + int(counts[2])
        )
        / total
    )


def counts_to_dict(
    counts: np.ndarray
) -> Dict[str, int]:

    return {
        CODE_TO_BASE[index]: int(
            counts[index]
        )
        for index in range(4)
    }


def get_nested(
    data: Optional[dict],
    path: Tuple[str, ...]
):

    current = data

    for key in path:

        if (
            not isinstance(
                current,
                dict
            )
            or key not in current
        ):

            return None

        current = current[
            key
        ]

    return current


# =============================================================================
# KANONİK DNA BÜTÜNLÜK KARŞILAŞTIRMASI
# =============================================================================

def compare_canonical_dna(
    original_path: Path,
    decrypted_path: Path,
    chunk_bases: int
) -> dict:

    original_iterator = (
        iter_canonical_dna_chunks(
            original_path,
            chunk_bases
        )
    )

    decrypted_iterator = (
        iter_canonical_dna_chunks(
            decrypted_path,
            chunk_bases
        )
    )


    original_sha256 = (
        hashlib.sha256()
    )

    decrypted_sha256 = (
        hashlib.sha256()
    )


    original_counts = np.zeros(
        4,
        dtype=np.int64
    )

    decrypted_counts = np.zeros(
        4,
        dtype=np.int64
    )


    original_length = 0

    decrypted_length = 0


    matching_bases = 0

    bit_mismatches = 0


    first_mismatch_position_1based: Optional[
        int
    ] = None


    global_position = 0


    for (
        original_chunk,
        decrypted_chunk
    ) in itertools.zip_longest(
        original_iterator,
        decrypted_iterator,
        fillvalue=b""
    ):

        original_sha256.update(
            original_chunk
        )

        decrypted_sha256.update(
            decrypted_chunk
        )


        original_length += len(
            original_chunk
        )

        decrypted_length += len(
            decrypted_chunk
        )


        if original_chunk:

            original_codes = ASCII_TO_CODE[
                np.frombuffer(
                    original_chunk,
                    dtype=np.uint8
                )
            ]

            original_counts += (
                np.bincount(
                    original_codes,
                    minlength=4
                )[:4]
            )

        else:

            original_codes = np.empty(
                0,
                dtype=np.uint8
            )


        if decrypted_chunk:

            decrypted_codes = ASCII_TO_CODE[
                np.frombuffer(
                    decrypted_chunk,
                    dtype=np.uint8
                )
            ]

            decrypted_counts += (
                np.bincount(
                    decrypted_codes,
                    minlength=4
                )[:4]
            )

        else:

            decrypted_codes = np.empty(
                0,
                dtype=np.uint8
            )


        overlap = min(
            len(original_chunk),
            len(decrypted_chunk)
        )


        if overlap > 0:

            original_overlap_bytes = (
                np.frombuffer(
                    original_chunk[
                        :overlap
                    ],
                    dtype=np.uint8
                )
            )

            decrypted_overlap_bytes = (
                np.frombuffer(
                    decrypted_chunk[
                        :overlap
                    ],
                    dtype=np.uint8
                )
            )


            equal_mask = (
                original_overlap_bytes
                == decrypted_overlap_bytes
            )


            matching_bases += int(
                np.count_nonzero(
                    equal_mask
                )
            )


            if (
                first_mismatch_position_1based
                is None

                and not bool(
                    np.all(
                        equal_mask
                    )
                )
            ):

                first_local_mismatch = int(
                    np.flatnonzero(
                        ~equal_mask
                    )[0]
                )

                first_mismatch_position_1based = (
                    global_position
                    + first_local_mismatch
                    + 1
                )


            original_overlap_codes = (
                original_codes[
                    :overlap
                ]
            )

            decrypted_overlap_codes = (
                decrypted_codes[
                    :overlap
                ]
            )


            xor_values = (
                np.bitwise_xor(
                    original_overlap_codes,
                    decrypted_overlap_codes
                )
            )


            bit_mismatches += int(
                BITCOUNT_2BIT[
                    xor_values
                ].sum()
            )


        length_difference = abs(
            len(original_chunk)
            - len(decrypted_chunk)
        )


        if length_difference > 0:

            # Eksik veya fazla her DNA bazı,
            # iki bitlik kayıp olarak cezalandırılır.
            bit_mismatches += (
                2
                * length_difference
            )

            if (
                first_mismatch_position_1based
                is None
            ):

                first_mismatch_position_1based = (
                    global_position
                    + overlap
                    + 1
                )


        global_position += max(
            len(original_chunk),
            len(decrypted_chunk)
        )


    comparison_length = max(
        original_length,
        decrypted_length
    )


    base_mismatches = (
        comparison_length
        - matching_bases
    )


    base_recovery_rate = (
        100.0

        if comparison_length == 0

        else (
            100.0
            * matching_bases
            / comparison_length
        )
    )


    total_bits = (
        2
        * comparison_length
    )


    matching_bits = (
        total_bits
        - bit_mismatches
    )


    bit_correction_rate = (
        100.0

        if total_bits == 0

        else (
            100.0
            * matching_bits
            / total_bits
        )
    )


    original_hash = (
        original_sha256.hexdigest()
    )

    decrypted_hash = (
        decrypted_sha256.hexdigest()
    )


    length_match = (
        original_length
        == decrypted_length
    )


    hash_match = (
        original_hash
        == decrypted_hash
    )


    exact_match = (
        length_match
        and hash_match
        and base_mismatches == 0
        and bit_mismatches == 0
    )


    return {
        "exact_match": (
            exact_match
        ),

        "length_match": (
            length_match
        ),

        "sha256_match": (
            hash_match
        ),

        "original_bases": (
            original_length
        ),

        "decrypted_bases": (
            decrypted_length
        ),

        "original_canonical_sha256": (
            original_hash
        ),

        "decrypted_canonical_sha256": (
            decrypted_hash
        ),

        "matching_bases": (
            matching_bases
        ),

        "base_mismatches": (
            base_mismatches
        ),

        "base_recovery_rate_percent": (
            base_recovery_rate
        ),

        "total_dna_bits_2bit": (
            total_bits
        ),

        "matching_dna_bits_2bit": (
            matching_bits
        ),

        "bit_mismatches_2bit": (
            bit_mismatches
        ),

        "bit_correction_rate_percent": (
            bit_correction_rate
        ),

        "first_mismatch_position_1based": (
            first_mismatch_position_1based
        ),

        "original_base_counts": (
            counts_to_dict(
                original_counts
            )
        ),

        "decrypted_base_counts": (
            counts_to_dict(
                decrypted_counts
            )
        ),

        "original_gc_ratio": (
            gc_ratio_from_counts(
                original_counts
            )
        ),

        "decrypted_gc_ratio": (
            gc_ratio_from_counts(
                decrypted_counts
            )
        ),

        "original_symbol_entropy_bits_per_base": (
            sequence_entropy_from_counts(
                original_counts
            )
        ),

        "decrypted_symbol_entropy_bits_per_base": (
            sequence_entropy_from_counts(
                decrypted_counts
            )
        ),

        "dna_bit_mapping": {
            "A": "00",
            "C": "01",
            "G": "10",
            "T": "11"
        }
    }


# =============================================================================
# METADATA TUTARLILIK KONTROLLERİ
# =============================================================================

def check_metadata_consistency(
    encryption_metadata: Optional[dict],
    decryption_metadata: Optional[dict],
    canonical_result: dict
) -> dict:

    encryption_nonce = get_nested(
        encryption_metadata,
        (
            "session",
            "nonce_hex"
        )
    )


    decryption_nonce = get_nested(
        decryption_metadata,
        (
            "session",
            "nonce_hex"
        )
    )


    encryption_bases = get_nested(
        encryption_metadata,
        (
            "input",
            "canonical_plain_bases"
        )
    )


    recovered_bases = get_nested(
        decryption_metadata,
        (
            "decryption",
            "recovered_bases"
        )
    )


    decrypt_recovered_sha256 = get_nested(
        decryption_metadata,
        (
            "integrity",
            "recovered_plaintext_sha256"
        )
    )


    ciphertext_hmac_verified = get_nested(
        decryption_metadata,
        (
            "integrity",
            "ciphertext_hmac_verified"
        )
    )


    encryption_scheme = get_nested(
        encryption_metadata,
        (
            "scheme",
        )
    )


    decryption_scheme = get_nested(
        decryption_metadata,
        (
            "scheme",
        )
    )


    return {
        "encryption_metadata_available": (
            encryption_metadata
            is not None
        ),

        "decryption_metadata_available": (
            decryption_metadata
            is not None
        ),

        "scheme_match": (
            None

            if encryption_scheme is None
            or decryption_scheme is None

            else (
                encryption_scheme
                == decryption_scheme
            )
        ),

        "nonce_match": (
            None

            if encryption_nonce is None
            or decryption_nonce is None

            else (
                encryption_nonce
                == decryption_nonce
            )
        ),

        "encryption_length_matches_original": (
            None

            if encryption_bases is None

            else (
                int(
                    encryption_bases
                )
                == int(
                    canonical_result[
                        "original_bases"
                    ]
                )
            )
        ),

        "decryption_length_matches_recovered": (
            None

            if recovered_bases is None

            else (
                int(
                    recovered_bases
                )
                == int(
                    canonical_result[
                        "decrypted_bases"
                    ]
                )
            )
        ),

        "decryption_metadata_sha256_matches_file": (
            None

            if decrypt_recovered_sha256 is None

            else (
                str(
                    decrypt_recovered_sha256
                ).lower()
                == str(
                    canonical_result[
                        "decrypted_canonical_sha256"
                    ]
                ).lower()
            )
        ),

        "ciphertext_hmac_verified_during_decryption": (
            ciphertext_hmac_verified
        )
    }


# =============================================================================
# ANA BÜTÜNLÜK TESTİ
# =============================================================================

def run_integrity_test(
    *,
    original_path: Path,
    decrypted_path: Path,
    encryption_metadata_path: Path,
    decryption_metadata_path: Path,
    report_path: Path,
    chunk_bases: int,
    strict_mode: bool
) -> dict:

    if not original_path.exists():

        raise FileNotFoundError(
            f"Orijinal DNA dosyası "
            f"bulunamadı: {original_path}"
        )


    if not decrypted_path.exists():

        raise FileNotFoundError(
            f"Deşifrelenmiş DNA dosyası "
            f"bulunamadı: {decrypted_path}"
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
        if process is not None
        else None
    )


    cpu_start = (
        sum(
            process.cpu_times()[:2]
        )
        if process is not None
        else None
    )


    wall_start = (
        time.perf_counter()
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


    canonical_result = (
        compare_canonical_dna(
            original_path,
            decrypted_path,
            chunk_bases
        )
    )


    metadata_result = (
        check_metadata_consistency(
            encryption_metadata,
            decryption_metadata,
            canonical_result
        )
    )


    raw_original_sha256 = (
        sha256_raw_file(
            original_path
        )
    )


    raw_decrypted_sha256 = (
        sha256_raw_file(
            decrypted_path
        )
    )


    wall_seconds = (
        time.perf_counter()
        - wall_start
    )


    cpu_end = (
        sum(
            process.cpu_times()[:2]
        )
        if process is not None
        else None
    )


    rss_end = (
        process.memory_info().rss
        if process is not None
        else None
    )


    result = {
        "analysis": (
            "BMC-T5-NREF-DNA-SPD "
            "canonical genomic integrity"
        ),

        "input_files": {
            "original": (
                original_path.name
            ),

            "decrypted": (
                decrypted_path.name
            ),

            "encryption_metadata": (
                encryption_metadata_path.name
            ),

            "decryption_metadata": (
                decryption_metadata_path.name
            )
        },

        "canonical_dna_integrity": (
            canonical_result
        ),

        "metadata_consistency": (
            metadata_result
        ),

        "raw_file_diagnostics": {
            "note": (
                "Raw file equality is formatting-sensitive "
                "and is not the primary genomic integrity criterion."
            ),

            "original_file_bytes": (
                original_path.stat().st_size
            ),

            "decrypted_file_bytes": (
                decrypted_path.stat().st_size
            ),

            "original_raw_sha256": (
                raw_original_sha256
            ),

            "decrypted_raw_sha256": (
                raw_decrypted_sha256
            ),

            "raw_sha256_match": (
                raw_original_sha256
                == raw_decrypted_sha256
            ),

            "original_raw_line_count": (
                count_raw_lines(
                    original_path
                )
            ),

            "decrypted_raw_line_count": (
                count_raw_lines(
                    decrypted_path
                )
            )
        },

        "performance": {
            "chunk_bases": (
                chunk_bases
            ),

            "wall_seconds": (
                wall_seconds
            ),

            "cpu_seconds": (
                None

                if cpu_start is None
                or cpu_end is None

                else (
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

                if rss_start is None
                or rss_end is None

                else (
                    rss_end
                    - rss_start
                )
                / (1024.0 ** 2)
            )
        }
    }


    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with report_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2
        )

        handle.write(
            "\n"
        )


    print_report(
        result,
        report_path
    )


    if (
        strict_mode
        and not canonical_result[
            "exact_match"
        ]
    ):

        raise RuntimeError(
            "Bütünlük doğrulaması başarısız: "
            "orijinal ve deşifrelenmiş kanonik DNA "
            "birebir eşleşmiyor."
        )


    return result


# =============================================================================
# RAPORLAMA
# =============================================================================

def print_report(
    result: dict,
    report_path: Path
) -> None:

    canonical = (
        result[
            "canonical_dna_integrity"
        ]
    )


    metadata = (
        result[
            "metadata_consistency"
        ]
    )


    performance = (
        result[
            "performance"
        ]
    )


    print(
        "\n"
        + "=" * 82
    )


    print(
        "BMC T5-NREF DNA-SPD "
        "GENOM BÜTÜNLÜK DOĞRULAMASI"
    )


    print(
        "=" * 82
    )


    print(
        f"Orijinal dosya                 : "
        f"{result['input_files']['original']}"
    )


    print(
        f"Deşifrelenmiş dosya            : "
        f"{result['input_files']['decrypted']}"
    )


    print(
        "\n--- KANONİK DNA BÜTÜNLÜĞÜ ---"
    )


    print(
        f"Orijinal baz sayısı            : "
        f"{canonical['original_bases']:,}"
    )


    print(
        f"Deşifrelenmiş baz sayısı       : "
        f"{canonical['decrypted_bases']:,}"
    )


    print(
        f"Uzunluk eşleşmesi              : "
        f"{canonical['length_match']}"
    )


    print(
        f"SHA-256 eşleşmesi              : "
        f"{canonical['sha256_match']}"
    )


    print(
        f"Birebir DNA eşleşmesi          : "
        f"{canonical['exact_match']}"
    )


    print(
        f"Baz uyuşmazlığı                : "
        f"{canonical['base_mismatches']:,}"
    )


    print(
        f"Base Recovery Rate (BRR)       : "
        f"{canonical['base_recovery_rate_percent']:.6f}%"
    )


    print(
        f"2-bit uyuşmayan bit            : "
        f"{canonical['bit_mismatches_2bit']:,}"
    )


    print(
        f"Bit Correction Rate (BCR)      : "
        f"{canonical['bit_correction_rate_percent']:.6f}%"
    )


    print(
        f"İlk uyuşmazlık konumu          : "
        f"{canonical['first_mismatch_position_1based']}"
    )


    print(
        f"Orijinal canonical SHA-256     : "
        f"{canonical['original_canonical_sha256']}"
    )


    print(
        f"Deşifre canonical SHA-256      : "
        f"{canonical['decrypted_canonical_sha256']}"
    )


    print(
        "\n--- METADATA / HMAC TUTARLILIĞI ---"
    )


    print(
        f"Şema eşleşmesi                 : "
        f"{metadata['scheme_match']}"
    )


    print(
        f"Nonce eşleşmesi                : "
        f"{metadata['nonce_match']}"
    )


    print(
        f"Şifreleme uzunluğu tutarlı     : "
        f"{metadata['encryption_length_matches_original']}"
    )


    print(
        f"Deşifreleme uzunluğu tutarlı   : "
        f"{metadata['decryption_length_matches_recovered']}"
    )


    print(
        f"Deşifre metadata SHA tutarlı   : "
        f"{metadata['decryption_metadata_sha256_matches_file']}"
    )


    print(
        f"Ciphertext HMAC doğrulandı     : "
        f"{metadata['ciphertext_hmac_verified_during_decryption']}"
    )


    print(
        "\n--- ANALİZ PERFORMANSI ---"
    )


    print(
        f"Bütünlük analiz wall time      : "
        f"{performance['wall_seconds']:.6f} s"
    )


    print(
        f"Bütünlük analiz CPU time       : "
        f"{optional_float(performance['cpu_seconds'])} s"
    )


    print(
        f"RAM değişimi                   : "
        f"{optional_float(performance['rss_delta_mb'], 3)} MB"
    )


    print(
        "\n--- SONUÇ ---"
    )


    if canonical[
        "exact_match"
    ]:

        print(
            "PASS — Orijinal ve deşifrelenmiş "
            "kanonik DNA tam olarak aynıdır."
        )

    else:

        print(
            "FAIL — Orijinal ve deşifrelenmiş "
            "kanonik DNA arasında fark vardır."
        )


    print(
        f"Bütünlük raporu                : "
        f"{report_path}"
    )


    print(
        "=" * 82
    )


# =============================================================================
# DOĞRUDAN ÇALIŞTIRMA
# =============================================================================

if __name__ == "__main__":

    original_path = (
        BASE_DIR
        / ORIGINAL_FILENAME
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
        f"[ORIGINAL]    {original_path}"
    )


    print(
        f"[DECRYPTED]   {decrypted_path}"
    )


    print(
        f"[ENC_META]    {encryption_metadata_path}"
    )


    print(
        f"[DEC_META]    {decryption_metadata_path}"
    )


    run_integrity_test(
        original_path=(
            original_path
        ),

        decrypted_path=(
            decrypted_path
        ),

        encryption_metadata_path=(
            encryption_metadata_path
        ),

        decryption_metadata_path=(
            decryption_metadata_path
        ),

        report_path=(
            report_path
        ),

        chunk_bases=(
            COMPARE_CHUNK_BASES
        ),

        strict_mode=(
            STRICT_MODE
        )
    )

