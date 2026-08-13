
import copy
import csv
import hashlib
import hmac
import importlib.util
import json
import random
import sys
import time

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import List, Sequence


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# İlk koşum
PLAIN_FILENAME = "ds_5mb.txt"


# İkinci koşumda yukarıdaki satırı kapatıp bunu aç:
# PLAIN_FILENAME = "ds_1mb.txt"


MASTER_KEY_FILENAME = "master_key_128.txt"

T5_MODULE_FILENAME = "T5_noref.py"

ENCRYPTION_CODE_FILENAME = (
    "genome_encrypt.py"
)

DECRYPTION_CODE_FILENAME = (
    "genome_decrypt.py"
)


DEVICE = "cpu"


# False bırakılması önerilir.
#
# False:
# Bütün 15 vaka doğrulanır.
# Seçilmiş altı vaka ayrıca gerçek decrypt_genome ile çalıştırılır.
#
# True:
# On beş vakanın tamamı gerçek decrypt_genome ile çalıştırılır.
RUN_FULL_DECRYPT_FOR_ALL_CASES = False


# Şifrelemeden önce oluşmuş kaynak/base-call hatası:
# hatalı girdi -> şifreleme -> deşifreleme round-trip testi
RUN_SOURCE_ERROR_ROUNDTRIP = True


# Oluşturulan bozuk cipher/meta/key girdileri saklansın mı?
KEEP_MUTATED_INPUT_FILES = False


TEST_RANDOM_SEED = 2026062301


# False modunda gerçek decrypt_genome ile ayrıca çalıştırılacak vakalar
FULL_DECRYPT_CASE_CODES = {
    "C1_cipher_substitution_1",
    "C3_cipher_insertion_1",
    "C4_cipher_deletion_1",
    "C5_cipher_truncation",
    "K1_key_one_base",
    "M2_metadata_nonce",
}


DNA = "ACGT"


# =============================================================================
# OTOMATİK DOSYA YOLLARI
# =============================================================================

STEM = Path(
    PLAIN_FILENAME
).stem


PLAIN_PATH = (
    BASE_DIR
    / PLAIN_FILENAME
)


CIPHER_PATH = (
    BASE_DIR
    / f"cipher_{STEM}.txt"
)


META_PATH = (
    BASE_DIR
    / f"meta_{STEM}.json"
)


MASTER_KEY_PATH = (
    BASE_DIR
    / MASTER_KEY_FILENAME
)


T5_PATH = (
    BASE_DIR
    / T5_MODULE_FILENAME
)


ENC_PATH = (
    BASE_DIR
    / ENCRYPTION_CODE_FILENAME
)


DEC_PATH = (
    BASE_DIR
    / DECRYPTION_CODE_FILENAME
)


OUT_DIR = (
    BASE_DIR
    / f"fault_robustness_{STEM}"
)


CASES_DIR = (
    OUT_DIR
    / "cases"
)


SOURCE_DIR = (
    OUT_DIR
    / "source_error_roundtrip"
)


OUT_JSON = (
    OUT_DIR
    / f"fault_robustness_{STEM}.json"
)


OUT_CSV = (
    OUT_DIR
    / f"fault_robustness_{STEM}.csv"
)


OUT_TXT = (
    OUT_DIR
    / f"fault_robustness_{STEM}.txt"
)


# =============================================================================
# GENEL YARDIMCILAR
# =============================================================================

def load_module(
    path: Path,
    name: str,
):
    if not path.is_file():

        raise FileNotFoundError(
            f"Python dosyası bulunamadı: {path}"
        )


    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )


    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Modül yüklenemedi: {path}"
        )


    module = importlib.util.module_from_spec(
        spec
    )


    sys.modules[
        name
    ] = module


    spec.loader.exec_module(
        module
    )


    return module


def read_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        value = json.load(
            handle
        )


    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            f"JSON kökü nesne olmalıdır: {path}"
        )


    return value


def write_json(
    path: Path,
    value: dict,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
        )


        handle.write(
            "\n"
        )


def write_dna(
    path: Path,
    sequence: str,
) -> None:

    path.write_text(
        sequence,
        encoding="ascii",
    )


def sha256_text(
    sequence: str,
) -> str:

    return hashlib.sha256(
        sequence.encode(
            "ascii"
        )
    ).hexdigest()


def next_base(
    base: str,
) -> str:
    """
    A -> C
    C -> G
    G -> T
    T -> A
    """

    return DNA[
        (
            DNA.index(
                base
            )
            + 1
        )
        % 4
    ]


def mutate_positions(
    sequence: str,
    positions: Sequence[int],
):

    values = list(
        sequence
    )


    details = []


    for position in positions:

        old_base = values[
            position
        ]


        new_base = next_base(
            old_base
        )


        values[
            position
        ] = new_base


        details.append(
            {
                "position": int(
                    position
                ),

                "old_base": (
                    old_base
                ),

                "new_base": (
                    new_base
                ),
            }
        )


    return (
        "".join(
            values
        ),

        details,
    )


def flip_hex(
    value: str,
) -> str:

    if not value:

        return value


    replacement = (
        "0"
        if value[
            0
        ].lower()
        != "0"
        else "1"
    )


    return (
        replacement
        + value[
            1:
        ]
    )


# =============================================================================
# GERÇEK SİSTEMLE AYNI HMAC-SHA256
# =============================================================================

def compute_hmac(
    enc,
    master_key: str,
    metadata: dict,
    cipher: str,
) -> str:
    """
    HMAC kapsamı:

    scheme
    + nonce
    + number_of_bases
    + spd_block_bases
    + t5_chunk_bases
    + xor_enabled
    + ciphertext
    """

    if (
        str(
            metadata[
                "scheme"
            ]
        )
        != enc.SCHEME
    ):

        raise ValueError(
            "Scheme uyuşmuyor."
        )


    number_of_bases = int(
        metadata[
            "input"
        ][
            "canonical_plain_bases"
        ]
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


    spd_block_bases = int(
        metadata[
            "encryption"
        ][
            "spd_block_bases"
        ]
    )


    t5_chunk_bases = int(
        metadata[
            "encryption"
        ][
            "t5_chunk_bases"
        ]
    )


    xor_enabled = bool(
        metadata[
            "encryption"
        ][
            "xor_enabled"
        ]
    )


    enc.validate_master_key(
        master_key
    )


    master_key_bytes = enc.pack_dna_2bit(
        master_key
    )


    authentication_key = enc.kdf(
        master_key_bytes,
        nonce,
        "AUTHENTICATION",
    )


    authentication = hmac.new(
        authentication_key,
        digestmod=hashlib.sha256,
    )


    authentication.update(
        enc.SCHEME.encode(
            "ascii"
        )
    )


    authentication.update(
        nonce
    )


    authentication.update(
        number_of_bases.to_bytes(
            8,
            "big",
        )
    )


    authentication.update(
        spd_block_bases.to_bytes(
            8,
            "big",
        )
    )


    authentication.update(
        t5_chunk_bases.to_bytes(
            8,
            "big",
        )
    )


    authentication.update(
        bytes(
            [
                1
                if xor_enabled
                else 0
            ]
        )
    )


    authentication.update(
        cipher.encode(
            "ascii"
        )
    )


    return authentication.hexdigest()


# =============================================================================
# BÜTÜN VAKALAR İÇİN ÖN DOĞRULAMA
# =============================================================================

def lightweight_check(
    enc,
    dec,
    cipher: str,
    metadata: dict,
    key: str,
    baseline_metadata: dict,
) -> dict:
    """
    Bu kontrol yaklaşık veya sahte bir kontrol değildir.

    Deşifreleme kodunun kullandığı:

    - metadata ayrıştırmasını
    - key doğrulamasını
    - uzunluk kontrolünü
    - aynı HMAC-SHA256 formülünü

    uygular.
    """

    try:

        parsed = dec.parse_encryption_metadata(
            metadata
        )


        dec.validate_master_key(
            key
        )


        expected_bases = int(
            parsed[
                "number_of_bases"
            ]
        )


        if len(
            cipher
        ) != expected_bases:

            return {
                "rejected": True,

                "stage": (
                    "length_validation"
                ),

                "reason": (
                    f"metadata={expected_bases}, "
                    f"ciphertext={len(cipher)}"
                ),
            }


        baseline_fingerprint = (
            baseline_metadata
            .get(
                "t5_noref",
                {},
            )
            .get(
                "model_fingerprint_sha256"
            )
        )


        observed_fingerprint = (
            metadata
            .get(
                "t5_noref",
                {},
            )
            .get(
                "model_fingerprint_sha256"
            )
        )


        if (
            baseline_fingerprint is not None
            and observed_fingerprint is not None
            and str(
                baseline_fingerprint
            ).lower()
            != str(
                observed_fingerprint
            ).lower()
        ):

            return {
                "rejected": True,

                "stage": (
                    "metadata_fingerprint_validation"
                ),

                "reason": (
                    "Model fingerprint alanı "
                    "değiştirilmiş."
                ),
            }


        computed_hmac = compute_hmac(
            enc,
            key,
            metadata,
            cipher,
        )


        expected_hmac = str(
            metadata[
                "integrity"
            ][
                "ciphertext_hmac_sha256"
            ]
        ).lower()


        if not hmac.compare_digest(
            computed_hmac,
            expected_hmac,
        ):

            return {
                "rejected": True,

                "stage": (
                    "hmac_authentication"
                ),

                "reason": (
                    "Hesaplanan HMAC metadata "
                    "etiketiyle uyuşmuyor."
                ),

                "computed_hmac": (
                    computed_hmac
                ),

                "expected_hmac": (
                    expected_hmac
                ),
            }


        return {
            "rejected": False,

            "stage": (
                "authentication_passed"
            ),

            "reason": (
                "Reddetme nedeni bulunmadı."
            ),

            "computed_hmac": (
                computed_hmac
            ),

            "expected_hmac": (
                expected_hmac
            ),
        }


    except Exception as error:

        return {
            "rejected": True,

            "stage": (
                "metadata_or_key_validation"
            ),

            "reason": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


# =============================================================================
# CIPHERTEXT BOZULMALARI
# =============================================================================

def c_sub_1(
    cipher,
    metadata,
    key,
    rng,
):

    mutated, details = mutate_positions(
        cipher,
        [
            len(
                cipher
            )
            // 2
        ],
    )


    return (
        mutated,
        copy.deepcopy(
            metadata
        ),
        key,
        {
            "changes": (
                details
            )
        },
    )


def c_sub_5(
    cipher,
    metadata,
    key,
    rng,
):

    positions = sorted(
        rng.sample(
            range(
                len(
                    cipher
                )
            ),
            5,
        )
    )


    mutated, details = mutate_positions(
        cipher,
        positions,
    )


    return (
        mutated,
        copy.deepcopy(
            metadata
        ),
        key,
        {
            "changes": (
                details
            )
        },
    )


def c_insert(
    cipher,
    metadata,
    key,
    rng,
):

    position = len(
        cipher
    ) // 2


    return (
        cipher[
            :position
        ]
        + "A"
        + cipher[
            position:
        ],

        copy.deepcopy(
            metadata
        ),

        key,

        {
            "position": (
                position
            ),

            "inserted_base": (
                "A"
            ),
        },
    )


def c_delete(
    cipher,
    metadata,
    key,
    rng,
):

    position = len(
        cipher
    ) // 2


    return (
        cipher[
            :position
        ]
        + cipher[
            position
            + 1:
        ],

        copy.deepcopy(
            metadata
        ),

        key,

        {
            "position": (
                position
            ),

            "deleted_base": (
                cipher[
                    position
                ]
            ),
        },
    )


def c_truncate(
    cipher,
    metadata,
    key,
    rng,
):

    removed_bases = min(
        128,
        max(
            1,
            len(
                cipher
            )
            // 1000,
        ),
    )


    return (
        cipher[
            :-removed_bases
        ],

        copy.deepcopy(
            metadata
        ),

        key,

        {
            "removed_tail_bases": (
                removed_bases
            )
        },
    )


def c_transpose(
    cipher,
    metadata,
    key,
    rng,
):

    segment_length = min(
        64,
        max(
            4,
            len(
                cipher
            )
            // 100,
        ),
    )


    first_start = len(
        cipher
    ) // 5


    second_start = min(
        3
        * len(
            cipher
        )
        // 5,

        len(
            cipher
        )
        - segment_length,
    )


    first_segment = cipher[
        first_start:
        first_start
        + segment_length
    ]


    second_segment = cipher[
        second_start:
        second_start
        + segment_length
    ]


    values = list(
        cipher
    )


    values[
        first_start:
        first_start
        + segment_length
    ] = second_segment


    values[
        second_start:
        second_start
        + segment_length
    ] = first_segment


    mutated = "".join(
        values
    )


    if mutated == cipher:

        mutated, details = mutate_positions(
            cipher,
            [
                first_start
            ],
        )


        information = {
            "fallback": (
                "single_substitution"
            ),

            "changes": (
                details
            ),
        }


    else:

        information = {
            "segment_length": (
                segment_length
            ),

            "first_start": (
                first_start
            ),

            "second_start": (
                second_start
            ),
        }


    return (
        mutated,
        copy.deepcopy(
            metadata
        ),
        key,
        information,
    )


# =============================================================================
# MASTER KEY BOZULMALARI
# =============================================================================

def k_one_base(
    cipher,
    metadata,
    key,
    rng,
):

    mutated, details = mutate_positions(
        key,
        [
            len(
                key
            )
            // 2
        ],
    )


    return (
        cipher,
        copy.deepcopy(
            metadata
        ),
        mutated,
        {
            "changes": (
                details
            )
        },
    )


def k_independent(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = "".join(
        next_base(
            base
        )
        for base in key
    )


    return (
        cipher,
        copy.deepcopy(
            metadata
        ),
        mutated,
        {
            "changed_key_bases": sum(
                first
                != second
                for first, second
                in zip(
                    key,
                    mutated,
                )
            )
        },
    )


# =============================================================================
# METADATA BOZULMALARI
# =============================================================================

def m_length(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = int(
        mutated[
            "input"
        ][
            "canonical_plain_bases"
        ]
    )


    mutated[
        "input"
    ][
        "canonical_plain_bases"
    ] = (
        old_value
        + 1
    )


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                old_value
                + 1
            ),
        },
    )


def m_nonce(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = str(
        mutated[
            "session"
        ][
            "nonce_hex"
        ]
    )


    new_value = flip_hex(
        old_value
    )


    mutated[
        "session"
    ][
        "nonce_hex"
    ] = new_value


    return (
        cipher,
        mutated,
        key,
        {
            "old_nonce": (
                old_value
            ),

            "new_nonce": (
                new_value
            ),
        },
    )


def m_spd(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = int(
        mutated[
            "encryption"
        ][
            "spd_block_bases"
        ]
    )


    new_value = (
        old_value
        // 2
        if old_value
        // 2
        >= 128
        else old_value
        * 2
    )


    mutated[
        "encryption"
    ][
        "spd_block_bases"
    ] = new_value


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                new_value
            ),
        },
    )


def m_chunk(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = int(
        mutated[
            "encryption"
        ][
            "t5_chunk_bases"
        ]
    )


    spd_block_bases = int(
        mutated[
            "encryption"
        ][
            "spd_block_bases"
        ]
    )


    new_value = max(
        spd_block_bases,
        old_value
        // 2,
    )


    if new_value == old_value:

        new_value = (
            old_value
            + spd_block_bases
        )


    mutated[
        "encryption"
    ][
        "t5_chunk_bases"
    ] = new_value


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                new_value
            ),
        },
    )


def m_xor(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = bool(
        mutated[
            "encryption"
        ][
            "xor_enabled"
        ]
    )


    mutated[
        "encryption"
    ][
        "xor_enabled"
    ] = not old_value


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                not old_value
            ),
        },
    )


def m_fingerprint(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = str(
        mutated[
            "t5_noref"
        ][
            "model_fingerprint_sha256"
        ]
    )


    new_value = flip_hex(
        old_value
    )


    mutated[
        "t5_noref"
    ][
        "model_fingerprint_sha256"
    ] = new_value


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                new_value
            ),
        },
    )


def m_hmac(
    cipher,
    metadata,
    key,
    rng,
):

    mutated = copy.deepcopy(
        metadata
    )


    old_value = str(
        mutated[
            "integrity"
        ][
            "ciphertext_hmac_sha256"
        ]
    )


    new_value = flip_hex(
        old_value
    )


    mutated[
        "integrity"
    ][
        "ciphertext_hmac_sha256"
    ] = new_value


    return (
        cipher,
        mutated,
        key,
        {
            "old": (
                old_value
            ),

            "new": (
                new_value
            ),
        },
    )


# =============================================================================
# TEST VAKALARI
# =============================================================================

CASES = (
    (
        "C1_cipher_substitution_1",
        "ciphertext",
        "1 baz substitution",
        "HMAC mismatch",
        c_sub_1,
    ),

    (
        "C2_cipher_substitution_5",
        "ciphertext",
        "5 baz substitution",
        "HMAC mismatch",
        c_sub_5,
    ),

    (
        "C3_cipher_insertion_1",
        "ciphertext",
        "1 baz insertion",
        "Length mismatch",
        c_insert,
    ),

    (
        "C4_cipher_deletion_1",
        "ciphertext",
        "1 baz deletion",
        "Length mismatch",
        c_delete,
    ),

    (
        "C5_cipher_truncation",
        "ciphertext",
        "Truncation",
        "Length mismatch",
        c_truncate,
    ),

    (
        "C6_cipher_transposition",
        "ciphertext",
        "Segment transposition",
        "HMAC mismatch",
        c_transpose,
    ),

    (
        "K1_key_one_base",
        "master_key",
        "Master key tek baz değişimi",
        "Fingerprint/HMAC mismatch",
        k_one_base,
    ),

    (
        "K2_key_independent",
        "master_key",
        "Tamamen farklı master key",
        "Fingerprint/HMAC mismatch",
        k_independent,
    ),

    (
        "M1_metadata_length",
        "metadata",
        "Baz sayısı değişimi",
        "Length mismatch",
        m_length,
    ),

    (
        "M2_metadata_nonce",
        "metadata",
        "Nonce değişimi",
        "Fingerprint/HMAC mismatch",
        m_nonce,
    ),

    (
        "M3_metadata_spd",
        "metadata",
        "SPD blok boyutu değişimi",
        "HMAC mismatch",
        m_spd,
    ),

    (
        "M4_metadata_chunk",
        "metadata",
        "T5 chunk boyutu değişimi",
        "HMAC mismatch",
        m_chunk,
    ),

    (
        "M5_metadata_xor",
        "metadata",
        "XOR bayrağı değişimi",
        "HMAC mismatch",
        m_xor,
    ),

    (
        "M6_metadata_fingerprint",
        "metadata",
        "Model fingerprint değişimi",
        "Fingerprint mismatch",
        m_fingerprint,
    ),

    (
        "M7_metadata_hmac",
        "metadata",
        "HMAC etiketi değişimi",
        "HMAC mismatch",
        m_hmac,
    ),
)


# =============================================================================
# TEST DOSYALARINI OLUŞTURMA
# =============================================================================

def create_case_files(
    directory: Path,
    code: str,
    cipher: str,
    metadata: dict,
    key: str,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    cipher_path = (
        directory
        / f"{code}_cipher.txt"
    )


    metadata_path = (
        directory
        / f"{code}_meta.json"
    )


    key_path = (
        directory
        / f"{code}_key.txt"
    )


    write_dna(
        cipher_path,
        cipher,
    )


    write_json(
        metadata_path,
        metadata,
    )


    write_dna(
        key_path,
        key,
    )


    return (
        cipher_path,
        metadata_path,
        key_path,
    )


# =============================================================================
# GERÇEK decrypt_genome ÇAĞRISI
# =============================================================================

def run_real_decryption(
    dec,
    directory: Path,
    code: str,
    cipher_path: Path,
    metadata_path: Path,
    key_path: Path,
) -> dict:

    output_path = (
        directory
        / f"{code}_recovered.txt"
    )


    decryption_metadata_path = (
        directory
        / f"{code}_decrypt_meta.json"
    )


    temporary_output_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )


    for path in (
        output_path,
        decryption_metadata_path,
        temporary_output_path,
    ):

        if path.exists():

            path.unlink()


    captured_stdout = StringIO()


    start = time.perf_counter()


    try:

        with redirect_stdout(
            captured_stdout
        ):

            dec.decrypt_genome(
                cipher_path=(
                    cipher_path
                ),

                encryption_metadata_path=(
                    metadata_path
                ),

                master_key_path=(
                    key_path
                ),

                t5_module_path=(
                    T5_PATH
                ),

                output_path=(
                    output_path
                ),

                decryption_metadata_path=(
                    decryption_metadata_path
                ),

                device=(
                    DEVICE
                ),

                overwrite=True,
            )


        return {
            "executed": True,

            "rejected": False,

            "exception_type": None,

            "exception_message": None,

            "output_created": (
                output_path.exists()
            ),

            "temporary_output_left": (
                temporary_output_path.exists()
            ),

            "elapsed_seconds": (
                time.perf_counter()
                - start
            ),

            "stdout": (
                captured_stdout.getvalue()
            ),
        }


    except Exception as error:

        return {
            "executed": True,

            "rejected": True,

            "exception_type": (
                type(
                    error
                ).__name__
            ),

            "exception_message": str(
                error
            ),

            "output_created": (
                output_path.exists()
            ),

            "temporary_output_left": (
                temporary_output_path.exists()
            ),

            "elapsed_seconds": (
                time.perf_counter()
                - start
            ),

            "stdout": (
                captured_stdout.getvalue()
            ),
        }


# =============================================================================
# ŞİFRELEME ÖNCESİ KAYNAK HATASI ROUND-TRIP
# =============================================================================

def source_error_roundtrip(
    enc,
    dec,
    original: str,
    baseline_metadata: dict,
) -> dict:

    SOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    erroneous, changes = mutate_positions(
        original,
        [
            len(
                original
            )
            // 2
        ],
    )


    plain_path = (
        SOURCE_DIR
        / f"source_error_{STEM}.txt"
    )


    cipher_path = (
        SOURCE_DIR
        / f"cipher_source_error_{STEM}.txt"
    )


    metadata_path = (
        SOURCE_DIR
        / f"meta_source_error_{STEM}.json"
    )


    recovered_path = (
        SOURCE_DIR
        / f"recovered_source_error_{STEM}.txt"
    )


    decryption_metadata_path = (
        SOURCE_DIR
        / f"decrypt_meta_source_error_{STEM}.json"
    )


    write_dna(
        plain_path,
        erroneous,
    )


    spd_block_bases = int(
        baseline_metadata[
            "encryption"
        ][
            "spd_block_bases"
        ]
    )


    t5_chunk_bases = int(
        baseline_metadata[
            "encryption"
        ][
            "t5_chunk_bases"
        ]
    )


    xor_enabled = bool(
        baseline_metadata[
            "encryption"
        ][
            "xor_enabled"
        ]
    )


    encryption_start = time.perf_counter()


    encryption_result = enc.encrypt_genome(
        plain_path=(
            plain_path
        ),

        master_key_path=(
            MASTER_KEY_PATH
        ),

        t5_module_path=(
            T5_PATH
        ),

        cipher_path=(
            cipher_path
        ),

        metadata_path=(
            metadata_path
        ),

        spd_block_bases=(
            spd_block_bases
        ),

        t5_chunk_bases=(
            t5_chunk_bases
        ),

        device=(
            DEVICE
        ),

        xor_enabled=(
            xor_enabled
        ),

        self_check=False,

        overwrite=True,
    )


    encryption_seconds = (
        time.perf_counter()
        - encryption_start
    )


    decryption_start = time.perf_counter()


    decryption_result = dec.decrypt_genome(
        cipher_path=(
            cipher_path
        ),

        encryption_metadata_path=(
            metadata_path
        ),

        master_key_path=(
            MASTER_KEY_PATH
        ),

        t5_module_path=(
            T5_PATH
        ),

        output_path=(
            recovered_path
        ),

        decryption_metadata_path=(
            decryption_metadata_path
        ),

        device=(
            DEVICE
        ),

        overwrite=True,
    )


    decryption_seconds = (
        time.perf_counter()
        - decryption_start
    )


    recovered = enc.read_dna(
        recovered_path
    )


    recovered_equals_erroneous = (
        recovered
        == erroneous
    )


    recovered_equals_original = (
        recovered
        == original
    )


    differences_from_original = sum(
        first
        != second
        for first, second
        in zip(
            recovered,
            original,
        )
    )


    base_recovery_rate = (
        100.0
        * sum(
            first
            == second
            for first, second
            in zip(
                recovered,
                erroneous,
            )
        )
        / len(
            erroneous
        )
    )


    passed = (
        recovered_equals_erroneous
        and not recovered_equals_original
        and differences_from_original
        == 1
        and abs(
            base_recovery_rate
            - 100.0
        )
        < 1e-12
    )


    return {
        "performed": True,

        "mutation": (
            changes[
                0
            ]
        ),

        "original_sha256": (
            sha256_text(
                original
            )
        ),

        "erroneous_input_sha256": (
            sha256_text(
                erroneous
            )
        ),

        "recovered_sha256": (
            sha256_text(
                recovered
            )
        ),

        "recovered_equals_erroneous_input": (
            recovered_equals_erroneous
        ),

        "recovered_equals_original_input": (
            recovered_equals_original
        ),

        "differences_from_original": (
            differences_from_original
        ),

        "base_recovery_rate_vs_erroneous_input_percent": (
            base_recovery_rate
        ),

        "encryption_elapsed_seconds": (
            encryption_seconds
        ),

        "decryption_elapsed_seconds": (
            decryption_seconds
        ),

        "encryption_metadata": (
            encryption_result
        ),

        "decryption_metadata": (
            decryption_result
        ),

        "passed": (
            passed
        ),

        "interpretation": (
            "Şifreleme öncesindeki base-call hatası "
            "kayıpsız korundu; sistem hata düzeltme "
            "veya variant calling yapmamaktadır."
        ),
    }


# =============================================================================
# CSV RAPORU
# =============================================================================

def write_csv(
    records: Sequence[dict],
) -> None:

    fields = [
        "code",

        "category",

        "description",

        "expected_rejection",

        "lightweight_rejected",

        "lightweight_stage",

        "real_decrypt_executed",

        "real_decrypt_rejected",

        "exception_type",

        "exception_message",

        "output_created",

        "temporary_output_left",

        "final_rejected",

        "passed",

        "elapsed_seconds",
    ]


    with OUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )


        writer.writeheader()


        for record in records:

            lightweight = record[
                "lightweight"
            ]


            real_decrypt = record[
                "real_decrypt"
            ]


            writer.writerow(
                {
                    "code": (
                        record[
                            "code"
                        ]
                    ),

                    "category": (
                        record[
                            "category"
                        ]
                    ),

                    "description": (
                        record[
                            "description"
                        ]
                    ),

                    "expected_rejection": (
                        record[
                            "expected_rejection"
                        ]
                    ),

                    "lightweight_rejected": (
                        lightweight[
                            "rejected"
                        ]
                    ),

                    "lightweight_stage": (
                        lightweight[
                            "stage"
                        ]
                    ),

                    "real_decrypt_executed": (
                        real_decrypt[
                            "executed"
                        ]
                    ),

                    "real_decrypt_rejected": (
                        real_decrypt[
                            "rejected"
                        ]
                    ),

                    "exception_type": (
                        real_decrypt[
                            "exception_type"
                        ]
                    ),

                    "exception_message": (
                        real_decrypt[
                            "exception_message"
                        ]
                    ),

                    "output_created": (
                        real_decrypt[
                            "output_created"
                        ]
                    ),

                    "temporary_output_left": (
                        real_decrypt[
                            "temporary_output_left"
                        ]
                    ),

                    "final_rejected": (
                        record[
                            "final_rejected"
                        ]
                    ),

                    "passed": (
                        record[
                            "passed"
                        ]
                    ),

                    "elapsed_seconds": (
                        real_decrypt[
                            "elapsed_seconds"
                        ]
                    ),
                }
            )


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main() -> None:

    total_start = time.perf_counter()


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    CASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    required_paths = (
        PLAIN_PATH,

        CIPHER_PATH,

        META_PATH,

        MASTER_KEY_PATH,

        T5_PATH,

        ENC_PATH,

        DEC_PATH,
    )


    missing_paths = [
        path
        for path in required_paths
        if not path.is_file()
    ]


    if missing_paths:

        raise FileNotFoundError(
            "Eksik dosyalar:\n"
            + "\n".join(
                str(
                    path
                )
                for path in missing_paths
            )
        )


    enc = load_module(
        ENC_PATH,
        "bmc_fault_enc",
    )


    dec = load_module(
        DEC_PATH,
        "bmc_fault_dec",
    )


    required_encryption_api = (
        "SCHEME",

        "read_dna",

        "validate_master_key",

        "pack_dna_2bit",

        "kdf",

        "encrypt_genome",
    )


    for name in required_encryption_api:

        if not hasattr(
            enc,
            name,
        ):

            raise AttributeError(
                "Şifreleme modülünde "
                f"eksik bileşen: {name}"
            )


    required_decryption_api = (
        "SCHEME",

        "parse_encryption_metadata",

        "validate_master_key",

        "decrypt_genome",
    )


    for name in required_decryption_api:

        if not hasattr(
            dec,
            name,
        ):

            raise AttributeError(
                "Deşifreleme modülünde "
                f"eksik bileşen: {name}"
            )


    if (
        enc.SCHEME
        != dec.SCHEME
    ):

        raise ValueError(
            "Şifreleme/deşifreleme "
            "SCHEME değerleri uyuşmuyor."
        )


    plain = enc.read_dna(
        PLAIN_PATH
    )


    cipher = enc.read_dna(
        CIPHER_PATH
    )


    key = enc.read_dna(
        MASTER_KEY_PATH
    )


    metadata = read_json(
        META_PATH
    )


    enc.validate_master_key(
        key
    )


    if len(
        plain
    ) != int(
        metadata[
            "input"
        ][
            "canonical_plain_bases"
        ]
    ):

        raise ValueError(
            "Plaintext uzunluğu metadata "
            "ile uyuşmuyor."
        )


    if len(
        cipher
    ) != len(
        plain
    ):

        raise ValueError(
            "Baseline cipher/plain "
            "uzunluğu uyuşmuyor."
        )


    expected_hmac = str(
        metadata[
            "integrity"
        ][
            "ciphertext_hmac_sha256"
        ]
    ).lower()


    computed_hmac = compute_hmac(
        enc,
        key,
        metadata,
        cipher,
    )


    baseline_hmac_passed = (
        hmac.compare_digest(
            expected_hmac,
            computed_hmac,
        )
    )


    if not baseline_hmac_passed:

        raise RuntimeError(
            "Baseline HMAC başarısız. "
            "Plain/cipher/meta/key dosyaları "
            "aynı oturuma ait olmayabilir."
        )


    print(
        "DNA-SPD HATA / BOZULMA "
        "DAYANIKLILIK TESTİ"
    )


    print(
        f"[DATASET]        {PLAIN_PATH}"
    )


    print(
        f"[BASES]          {len(plain):,}"
    )


    print(
        "[BASELINE HMAC]  PASS"
    )


    print(
        f"[FULL ALL CASES] "
        f"{RUN_FULL_DECRYPT_FOR_ALL_CASES}"
    )


    random_generator = random.Random(
        TEST_RANDOM_SEED
    )


    records: List[dict] = []


    for (
        case_index,
        (
            code,
            category,
            description,
            expected_rejection,
            builder,
        ),
    ) in enumerate(
        CASES,
        start=1,
    ):

        print(
            "\n"
            + "=" * 90
        )


        print(
            f"[{case_index}/{len(CASES)}] "
            f"{code} — {description}"
        )


        print(
            "=" * 90
        )


        (
            mutated_cipher,
            mutated_metadata,
            mutated_key,
            mutation_details,
        ) = builder(
            cipher,
            metadata,
            key,
            random_generator,
        )


        case_directory = (
            CASES_DIR
            / code
        )


        (
            case_cipher_path,
            case_metadata_path,
            case_key_path,
        ) = create_case_files(
            case_directory,
            code,
            mutated_cipher,
            mutated_metadata,
            mutated_key,
        )


        lightweight_result = lightweight_check(
            enc,
            dec,
            mutated_cipher,
            mutated_metadata,
            mutated_key,
            metadata,
        )


        run_real_decrypt = (
            RUN_FULL_DECRYPT_FOR_ALL_CASES
            or code
            in FULL_DECRYPT_CASE_CODES
        )


        if run_real_decrypt:

            real_decrypt_result = (
                run_real_decryption(
                    dec,
                    case_directory,
                    code,
                    case_cipher_path,
                    case_metadata_path,
                    case_key_path,
                )
            )


        else:

            real_decrypt_result = {
                "executed": False,

                "rejected": None,

                "exception_type": None,

                "exception_message": None,

                "output_created": None,

                "temporary_output_left": None,

                "elapsed_seconds": 0.0,

                "stdout": "",
            }


        if real_decrypt_result[
            "executed"
        ]:

            final_rejected = bool(
                real_decrypt_result[
                    "rejected"
                ]
            )


            output_safe = (
                not bool(
                    real_decrypt_result[
                        "output_created"
                    ]
                )
                and not bool(
                    real_decrypt_result[
                        "temporary_output_left"
                    ]
                )
            )


        else:

            final_rejected = bool(
                lightweight_result[
                    "rejected"
                ]
            )


            output_safe = True


        passed = (
            final_rejected
            and output_safe
        )


        records.append(
            {
                "code": (
                    code
                ),

                "category": (
                    category
                ),

                "description": (
                    description
                ),

                "expected_rejection": (
                    expected_rejection
                ),

                "mutation_details": (
                    mutation_details
                ),

                "lightweight": (
                    lightweight_result
                ),

                "real_decrypt": (
                    real_decrypt_result
                ),

                "final_rejected": (
                    final_rejected
                ),

                "output_safe": (
                    output_safe
                ),

                "passed": (
                    passed
                ),
            }
        )


        print(
            "Ön doğrulamada reddedildi : "
            f"{lightweight_result['rejected']}"
        )


        print(
            "Reddetme aşaması          : "
            f"{lightweight_result['stage']}"
        )


        print(
            "Gerçek decrypt çalıştı mı : "
            f"{real_decrypt_result['executed']}"
        )


        if real_decrypt_result[
            "executed"
        ]:

            print(
                "Gerçek decrypt reddetti mi: "
                f"{real_decrypt_result['rejected']}"
            )


            print(
                "Hata tipi                 : "
                f"{real_decrypt_result['exception_type']}"
            )


            print(
                "Hata mesajı               : "
                f"{real_decrypt_result['exception_message']}"
            )


            print(
                "Nihai çıktı oluştu mu      : "
                f"{real_decrypt_result['output_created']}"
            )


            print(
                "Geçici çıktı kaldı mı      : "
                f"{real_decrypt_result['temporary_output_left']}"
            )


        print(
            "VAKA SONUCU                : "
            f"{'PASS' if passed else 'FAIL'}"
        )


        if not KEEP_MUTATED_INPUT_FILES:

            for path in (
                case_cipher_path,
                case_metadata_path,
                case_key_path,
            ):

                if path.exists():

                    path.unlink()


    # =========================================================================
    # ŞİFRELEME ÖNCESİ KAYNAK HATASI
    # =========================================================================

    if RUN_SOURCE_ERROR_ROUNDTRIP:

        print(
            "\n"
            + "=" * 90
        )


        print(
            "SOURCE-ERROR ROUND-TRIP"
        )


        print(
            "=" * 90
        )


        source_error_result = (
            source_error_roundtrip(
                enc,
                dec,
                plain,
                metadata,
            )
        )


        print(
            "Recovered == erroneous input : "
            f"{source_error_result['recovered_equals_erroneous_input']}"
        )


        print(
            "Recovered == original input  : "
            f"{source_error_result['recovered_equals_original_input']}"
        )


        print(
            "BCR vs erroneous input       : "
            f"{source_error_result['base_recovery_rate_vs_erroneous_input_percent']:.6f}%"
        )


        print(
            "SOURCE-ERROR SONUCU          : "
            f"{'PASS' if source_error_result['passed'] else 'FAIL'}"
        )


    else:

        source_error_result = {
            "performed": False,

            "passed": None,
        }


    all_faults_passed = all(
        record[
            "passed"
        ]
        for record in records
    )


    overall_passed = (
        all_faults_passed
        and (
            bool(
                source_error_result[
                    "passed"
                ]
            )
            if source_error_result[
                "performed"
            ]
            else True
        )
    )


    report = {
        "analysis": (
            "T5-NREF DNA-SPD fault, corruption, "
            "wrong-key and source-error robustness"
        ),

        "dataset": {
            "plain_file": str(
                PLAIN_PATH
            ),

            "cipher_file": str(
                CIPHER_PATH
            ),

            "metadata_file": str(
                META_PATH
            ),

            "number_of_bases": len(
                plain
            ),

            "plain_sha256": (
                sha256_text(
                    plain
                )
            ),

            "cipher_sha256": (
                sha256_text(
                    cipher
                )
            ),
        },

        "configuration": {
            "device": (
                DEVICE
            ),

            "run_full_decrypt_for_all_cases": (
                RUN_FULL_DECRYPT_FOR_ALL_CASES
            ),

            "selected_real_decrypt_cases_when_false": (
                sorted(
                    FULL_DECRYPT_CASE_CODES
                )
            ),

            "run_source_error_roundtrip": (
                RUN_SOURCE_ERROR_ROUNDTRIP
            ),

            "test_random_seed": (
                TEST_RANDOM_SEED
            ),
        },

        "baseline": {
            "expected_hmac": (
                expected_hmac
            ),

            "computed_hmac": (
                computed_hmac
            ),

            "passed": (
                baseline_hmac_passed
            ),
        },

        "fault_cases": (
            records
        ),

        "source_error_roundtrip": (
            source_error_result
        ),

        "summary": {
            "number_of_fault_cases": len(
                records
            ),

            "passed_fault_cases": sum(
                int(
                    record[
                        "passed"
                    ]
                )
                for record in records
            ),

            "all_fault_cases_passed": (
                all_faults_passed
            ),

            "source_error_roundtrip_passed": (
                source_error_result[
                    "passed"
                ]
            ),

            "overall_passed": (
                overall_passed
            ),
        },

        "total_elapsed_seconds": (
            time.perf_counter()
            - total_start
        ),
    }


    write_json(
        OUT_JSON,
        report,
    )


    write_csv(
        records
    )


    text_lines = [
        "DNA-SPD FAULT / CORRUPTION ROBUSTNESS SUMMARY",

        "=" * 82,

        f"Dataset: {STEM}",

        f"Bases: {len(plain):,}",

        (
            "Baseline HMAC: "
            + (
                "PASS"
                if baseline_hmac_passed
                else "FAIL"
            )
        ),

        "",

        (
            "Code | Category | Rejected | "
            "Output-safe | Result"
        ),

        "-" * 82,
    ]


    for record in records:

        text_lines.append(
            f"{record['code']} | "
            f"{record['category']} | "
            f"{record['final_rejected']} | "
            f"{record['output_safe']} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )


    text_lines.extend(
        [
            "",

            (
                "Source-error round-trip: "
                + (
                    "PASS"
                    if source_error_result.get(
                        "passed"
                    )
                    else (
                        "NOT RUN"
                        if not source_error_result.get(
                            "performed"
                        )
                        else "FAIL"
                    )
                )
            ),

            (
                "Overall: "
                + (
                    "PASS"
                    if overall_passed
                    else "FAIL"
                )
            ),

            "",

            f"JSON: {OUT_JSON}",

            f"CSV : {OUT_CSV}",
        ]
    )


    OUT_TXT.write_text(
        "\n".join(
            text_lines
        )
        + "\n",

        encoding="utf-8",
    )


    print(
        "\n"
        + "=" * 90
    )


    print(
        "HATA / BOZULMA TESTİ TAMAMLANDI"
    )


    print(
        "=" * 90
    )


    print(
        "Fault cases : "
        f"{sum(int(record['passed']) for record in records)}"
        f"/{len(records)} PASS"
    )


    print(
        "Source error: "
        f"{source_error_result.get('passed')}"
    )


    print(
        "Overall     : "
        f"{'PASS' if overall_passed else 'FAIL'}"
    )


    print(
        f"JSON        : {OUT_JSON}"
    )


    print(
        f"CSV         : {OUT_CSV}"
    )


    print(
        f"TXT         : {OUT_TXT}"
    )


if __name__ == "__main__":

    main()
