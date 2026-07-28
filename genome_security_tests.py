# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:51:33 2026

@author: Alev Kaya
"""


# -*- coding: utf-8 -*-
"""
BMC Bioinformatics – T5-NREF DNA-SPD Güvenlik-2 testleri.

Bu kod, önceki Entropy çalışmasındaki DNA-yerel güvenlik testlerini
T5-NREF kontrollü şifreleme mimarisine uyarlar.

Testler:
1. Plaintext sensitivity / avalanche
2. Master-key sensitivity
3. Nonce sensitivity
4. CPA-style chosen-plaintext divergence
5. KPA-style independent-key divergence
6. Ciphertext fault propagation
7. HMAC tamper detection

DNA-yerel ölçüm:
    A = 00
    C = 01
    G = 10
    T = 11

Bu nedenle:
- Bağımsız DNA dizileri için beklenen baz farkı yaklaşık %75'tir.
- 2-bit gösterimde beklenen bit farkı yaklaşık %50'dir.

Önemli:
- Plaintext avalanche testinde tek baz değişimi yalnız ilgili SPD bloğuna
  yayılır. Bu nedenle hem tüm dizi üzerindeki GLOBAL fark hem de değişen
  bloğa ait LOCAL fark ayrıca raporlanır.
- CPA testi aynı nonce/kontrol akışı altında kontrollü bir tanılama testidir.
  Gerçek kullanımda aynı master key ile nonce tekrarlanmamalıdır.
- KPA-style sonucu gerçek bir anahtar-kurtarma saldırısı değildir; aynı
  plaintext'in bağımsız anahtarlarda ürettiği ciphertext ayrışmasını ölçer.
- Fault testi, HMAC kontrolü bilinçli olarak atlanarak yayılımı ölçer.
  Normal deşifreleme kodu değiştirilmiş ciphertext'i HMAC aşamasında reddeder.
- Bu deneyler ampirik güvenlik göstergeleridir; biçimsel güvenlik kanıtı değildir.
"""



import gc
import hashlib
import hmac
import importlib.util
import json
import math
import random
import statistics
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# Yalnızca veri seti adını değiştir:
#
# ds_1kb.txt
# ds_10kb.txt
# ds_100kb.txt
# ds_1mb.txt
# ds_5mb.txt
# ds_tam.txt
#
PLAIN_FILENAME = "ds_5mb.txt"


DATASET_STEM = Path(
    PLAIN_FILENAME
).stem


CIPHER_FILENAME = (
    f"cipher_{DATASET_STEM}.txt"
)


ENCRYPTION_METADATA_FILENAME = (
    f"meta_{DATASET_STEM}.json"
)


MASTER_KEY_FILENAME = (
    "master_key_128.txt"
)


T5_MODULE_FILENAME = (
    "T5_noref.py"
)


# Son kullandığın şifreleme kodunun dosya adı
ENCRYPTION_CODE_FILENAME = (
    "genome_encrypt.py"
)


OUTPUT_JSON_FILENAME = (
    f"security2_t5_{DATASET_STEM}.json"
)


DEVICE = "cpu"


# T5 modeli, master key veya nonce değiştiğinde yeniden oluşturulur.
# Bu nedenle key ve nonce denemeleri daha maliyetlidir.
AVALANCHE_TRIALS = 100

KEY_SENSITIVITY_TRIALS = 10

NONCE_SENSITIVITY_TRIALS = 10

KPA_STYLE_TRIALS = 10


# CPA: plaintext bazlarının %10'u değiştirilir.
CPA_CHANGE_RATIO = 0.10


# Fault: ciphertext içinde 5 baz değiştirilir.
FAULT_BASES = 5


# Test pozisyonlarını tekrarlanabilir seçmek için kullanılır.
# T5 kriptografik üretimini belirlemez.
TEST_RANDOM_SEED = 1337


# Mevcut ciphertext'in aynı key ve nonce ile yeniden üretildiğini doğrular.
VERIFY_BASELINE_REPRODUCTION = True


# =============================================================================
# SABİTLER
# =============================================================================

DNA = "ACGT"


BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


# 0=00, 1=01, 2=10, 3=11 için bit sayıları
BITCOUNT_2BIT = [
    0,
    1,
    1,
    2
]


# =============================================================================
# VERİ SINIFLARI
# =============================================================================

@dataclass
class ChunkMaterial:

    start: int

    length: int

    keystream: str

    controls: list


@dataclass
class SessionMaterial:

    nonce: bytes

    spd_block_bases: int

    t5_chunk_bases: int

    xor_enabled: bool

    chunks: List[ChunkMaterial]

    model_seed: int

    model_fingerprint: str

    setup_seconds: float

    generation_seconds: float


# =============================================================================
# DOSYA VE MODÜL İŞLEMLERİ
# =============================================================================

def read_json(
    path: Path
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: "
            f"{path}"
        )


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


def write_json(
    path: Path,
    data: dict
) -> None:

    with path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2
        )

        handle.write(
            "\n"
        )


def load_python_module(
    path: Path,
    module_name: str
):

    if not path.exists():

        raise FileNotFoundError(
            f"Python modülü bulunamadı: "
            f"{path}"
        )


    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )


    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Python modülü yüklenemedi: "
            f"{path}"
        )


    module = importlib.util.module_from_spec(
        spec
    )


    sys.modules[
        module_name
    ] = module


    spec.loader.exec_module(
        module
    )


    return module


def validate_required_encryption_api(
    enc
) -> None:

    required = (
        "SCHEME",
        "SUB_CONTROL_BASES",
        "PERM_CONTROL_BASES",
        "DIFF_CONTROL_BASES",
        "read_dna",
        "validate_master_key",
        "pack_dna_2bit",
        "kdf",
        "load_t5_module",
        "create_session_model",
        "generate_domain_dna",
        "block_control",
        "encrypt_block",
        "decrypt_block_for_self_check"
    )


    missing = [
        name
        for name in required
        if not hasattr(
            enc,
            name
        )
    ]


    if missing:

        raise AttributeError(
            "Şifreleme modülünde gerekli "
            "bileşenler eksik: "
            + ", ".join(
                missing
            )
        )


# =============================================================================
# DNA FARK ÖLÇÜLERİ
# =============================================================================

def hamming_base_percent(
    first: str,
    second: str
) -> float:

    comparison_length = max(
        len(first),
        len(second),
        1
    )


    overlap = min(
        len(first),
        len(second)
    )


    differences = sum(
        first[index]
        != second[index]
        for index in range(
            overlap
        )
    )


    differences += abs(
        len(first)
        - len(second)
    )


    return (
        100.0
        * differences
        / comparison_length
    )


def hamming_bit_percent_2bit(
    first: str,
    second: str
) -> float:

    comparison_length = max(
        len(first),
        len(second),
        1
    )


    overlap = min(
        len(first),
        len(second)
    )


    differing_bits = 0


    for index in range(
        overlap
    ):

        xor_value = (
            BASE_TO_INT[
                first[index]
            ]
            ^ BASE_TO_INT[
                second[index]
            ]
        )


        differing_bits += (
            BITCOUNT_2BIT[
                xor_value
            ]
        )


    differing_bits += (
        2
        * abs(
            len(first)
            - len(second)
        )
    )


    return (
        100.0
        * differing_bits
        / (
            2
            * comparison_length
        )
    )


def similarity_base_percent(
    first: str,
    second: str
) -> float:

    return (
        100.0
        - hamming_base_percent(
            first,
            second
        )
    )


def similarity_bit_percent_2bit(
    first: str,
    second: str
) -> float:

    return (
        100.0
        - hamming_bit_percent_2bit(
            first,
            second
        )
    )


def numeric_summary(
    values: Sequence[float]
) -> dict:

    if not values:

        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
            "sample_std": None
        }


    return {
        "count": len(
            values
        ),

        "minimum": float(
            min(
                values
            )
        ),

        "mean": float(
            statistics.fmean(
                values
            )
        ),

        "maximum": float(
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
            if len(values) > 1
            else 0.0
        )
    }


def flip_to_other_base(
    base: str,
    rng: random.Random
) -> str:

    choices = [
        candidate
        for candidate in DNA
        if candidate != base
    ]


    return rng.choice(
        choices
    )


def mutate_one_plain_base(
    plain: str,
    rng: random.Random
) -> Tuple[str, int, str, str]:

    position = rng.randrange(
        len(plain)
    )


    old_base = plain[
        position
    ]


    new_base = flip_to_other_base(
        old_base,
        rng
    )


    mutated = (
        plain[
            :position
        ]
        + new_base
        + plain[
            position + 1:
        ]
    )


    return (
        mutated,
        position,
        old_base,
        new_base
    )


def mutate_plain_ratio(
    plain: str,
    ratio: float,
    rng: random.Random
) -> Tuple[str, List[int]]:

    ratio = max(
        0.0,
        min(
            1.0,
            float(
                ratio
            )
        )
    )


    number_to_change = max(
        1,
        int(
            round(
                len(plain)
                * ratio
            )
        )
    )


    number_to_change = min(
        number_to_change,
        len(plain)
    )


    positions = sorted(
        rng.sample(
            range(
                len(plain)
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
        ] = flip_to_other_base(
            output[
                position
            ],
            rng
        )


    return (
        "".join(
            output
        ),
        positions
    )


def mutate_master_key_one_base(
    key: str,
    rng: random.Random
) -> Tuple[str, int, str, str]:

    position = rng.randrange(
        len(key)
    )


    old_base = key[
        position
    ]


    new_base = flip_to_other_base(
        old_base,
        rng
    )


    mutated = (
        key[
            :position
        ]
        + new_base
        + key[
            position + 1:
        ]
    )


    return (
        mutated,
        position,
        old_base,
        new_base
    )


def random_master_key(
    length: int,
    rng: random.Random
) -> str:

    return "".join(
        rng.choice(
            DNA
        )
        for _ in range(
            length
        )
    )


def mutate_nonce_one_bit(
    nonce: bytes,
    rng: random.Random
) -> Tuple[bytes, int]:

    bit_position = rng.randrange(
        len(nonce)
        * 8
    )


    byte_index = (
        bit_position
        // 8
    )


    bit_index = (
        bit_position
        % 8
    )


    output = bytearray(
        nonce
    )


    output[
        byte_index
    ] ^= (
        1
        << bit_index
    )


    return (
        bytes(
            output
        ),
        bit_position
    )


def inject_cipher_faults(
    cipher: str,
    number_of_faults: int,
    rng: random.Random
) -> Tuple[str, List[int]]:

    if not cipher:

        raise ValueError(
            "Ciphertext boş olamaz."
        )


    number_of_faults = min(
        max(
            1,
            int(
                number_of_faults
            )
        ),
        len(cipher)
    )


    positions = sorted(
        rng.sample(
            range(
                len(cipher)
            ),
            number_of_faults
        )
    )


    output = list(
        cipher
    )


    for position in positions:

        output[
            position
        ] = flip_to_other_base(
            output[
                position
            ],
            rng
        )


    return (
        "".join(
            output
        ),
        positions
    )


# =============================================================================
# T5-NREF OTURUM MATERYALİ
# =============================================================================

def build_session_material(
    *,
    enc,
    t5_backend,
    master_key: str,
    nonce: bytes,
    number_of_bases: int,
    spd_block_bases: int,
    t5_chunk_bases: int,
    xor_enabled: bool,
    device: str
) -> SessionMaterial:

    master_key_bytes = (
        enc.pack_dna_2bit(
            master_key
        )
    )


    setup_start = (
        time.perf_counter()
    )


    (
        model,
        config,
        model_seed,
        model_fingerprint
    ) = enc.create_session_model(
        t5_backend,
        master_key_bytes,
        nonce,
        device
    )


    setup_seconds = (
        time.perf_counter()
        - setup_start
    )


    generation_start = (
        time.perf_counter()
    )


    chunks: List[
        ChunkMaterial
    ] = []


    processed_bases = 0

    global_block_counter = 0

    chunk_counter = 0


    while (
        processed_bases
        < number_of_bases
    ):

        chunk_length = min(
            t5_chunk_bases,
            number_of_bases
            - processed_bases
        )


        blocks_in_chunk = math.ceil(
            chunk_length
            / spd_block_bases
        )


        keystream = (
            enc.generate_domain_dna(
                module=t5_backend,

                model=model,

                config=config,

                master_key=(
                    master_key
                ),

                master_key_bytes=(
                    master_key_bytes
                ),

                nonce=nonce,

                label="KS",

                counter=(
                    chunk_counter
                ),

                output_length=(
                    chunk_length
                ),

                device=device
            )
        )


        substitution_stream = (
            enc.generate_domain_dna(
                module=t5_backend,

                model=model,

                config=config,

                master_key=(
                    master_key
                ),

                master_key_bytes=(
                    master_key_bytes
                ),

                nonce=nonce,

                label="SUB",

                counter=(
                    chunk_counter
                ),

                output_length=(
                    blocks_in_chunk
                    * enc.SUB_CONTROL_BASES
                ),

                device=device
            )
        )


        permutation_stream = (
            enc.generate_domain_dna(
                module=t5_backend,

                model=model,

                config=config,

                master_key=(
                    master_key
                ),

                master_key_bytes=(
                    master_key_bytes
                ),

                nonce=nonce,

                label="PERM",

                counter=(
                    chunk_counter
                ),

                output_length=(
                    blocks_in_chunk
                    * enc.PERM_CONTROL_BASES
                ),

                device=device
            )
        )


        diffusion_stream = (
            enc.generate_domain_dna(
                module=t5_backend,

                model=model,

                config=config,

                master_key=(
                    master_key
                ),

                master_key_bytes=(
                    master_key_bytes
                ),

                nonce=nonce,

                label="DIFF",

                counter=(
                    chunk_counter
                ),

                output_length=(
                    blocks_in_chunk
                    * enc.DIFF_CONTROL_BASES
                ),

                device=device
            )
        )


        controls = []


        for local_block in range(
            blocks_in_chunk
        ):

            controls.append(
                enc.block_control(
                    nonce=nonce,

                    block_counter=(
                        global_block_counter
                        + local_block
                    ),

                    sub_dna=(
                        substitution_stream[
                            local_block
                            * enc.SUB_CONTROL_BASES:
                            (
                                local_block
                                + 1
                            )
                            * enc.SUB_CONTROL_BASES
                        ]
                    ),

                    perm_dna=(
                        permutation_stream[
                            local_block
                            * enc.PERM_CONTROL_BASES:
                            (
                                local_block
                                + 1
                            )
                            * enc.PERM_CONTROL_BASES
                        ]
                    ),

                    diff_dna=(
                        diffusion_stream[
                            local_block
                            * enc.DIFF_CONTROL_BASES:
                            (
                                local_block
                                + 1
                            )
                            * enc.DIFF_CONTROL_BASES
                        ]
                    )
                )
            )


        chunks.append(
            ChunkMaterial(
                start=(
                    processed_bases
                ),

                length=(
                    chunk_length
                ),

                keystream=(
                    keystream
                ),

                controls=(
                    controls
                )
            )
        )


        processed_bases += (
            chunk_length
        )


        global_block_counter += (
            blocks_in_chunk
        )


        chunk_counter += 1


    generation_seconds = (
        time.perf_counter()
        - generation_start
    )


    del model

    gc.collect()


    return SessionMaterial(
        nonce=nonce,

        spd_block_bases=(
            spd_block_bases
        ),

        t5_chunk_bases=(
            t5_chunk_bases
        ),

        xor_enabled=(
            xor_enabled
        ),

        chunks=(
            chunks
        ),

        model_seed=int(
            model_seed
        ),

        model_fingerprint=str(
            model_fingerprint
        ),

        setup_seconds=float(
            setup_seconds
        ),

        generation_seconds=float(
            generation_seconds
        )
    )


def encrypt_with_material(
    enc,
    plain: str,
    material: SessionMaterial
) -> str:

    output_parts: List[
        str
    ] = []


    for chunk in material.chunks:

        plain_chunk = plain[
            chunk.start:
            chunk.start
            + chunk.length
        ]


        chunk_parts: List[
            str
        ] = []


        for (
            local_block,
            control
        ) in enumerate(
            chunk.controls
        ):

            local_start = (
                local_block
                * material.spd_block_bases
            )


            local_end = min(
                local_start
                + material.spd_block_bases,
                chunk.length
            )


            chunk_parts.append(
                enc.encrypt_block(
                    plain_chunk[
                        local_start:
                        local_end
                    ],

                    chunk.keystream[
                        local_start:
                        local_end
                    ],

                    control,

                    xor_enabled=(
                        material.xor_enabled
                    )
                )
            )


        output_parts.append(
            "".join(
                chunk_parts
            )
        )


    ciphertext = "".join(
        output_parts
    )


    if len(ciphertext) != len(
        plain
    ):

        raise RuntimeError(
            "Üretilen ciphertext "
            "uzunluğu hatalı."
        )


    return ciphertext


def decrypt_with_material(
    enc,
    cipher: str,
    material: SessionMaterial
) -> str:

    output_parts: List[
        str
    ] = []


    for chunk in material.chunks:

        cipher_chunk = cipher[
            chunk.start:
            chunk.start
            + chunk.length
        ]


        chunk_parts: List[
            str
        ] = []


        for (
            local_block,
            control
        ) in enumerate(
            chunk.controls
        ):

            local_start = (
                local_block
                * material.spd_block_bases
            )


            local_end = min(
                local_start
                + material.spd_block_bases,
                chunk.length
            )


            chunk_parts.append(
                enc.decrypt_block_for_self_check(
                    cipher_chunk[
                        local_start:
                        local_end
                    ],

                    chunk.keystream[
                        local_start:
                        local_end
                    ],

                    control,

                    xor_enabled=(
                        material.xor_enabled
                    )
                )
            )


        output_parts.append(
            "".join(
                chunk_parts
            )
        )


    recovered = "".join(
        output_parts
    )


    if len(recovered) != len(
        cipher
    ):

        raise RuntimeError(
            "Deşifrelenen DNA "
            "uzunluğu hatalı."
        )


    return recovered


# =============================================================================
# HMAC DOĞRULAMASI
# =============================================================================

def compute_ciphertext_hmac(
    *,
    enc,
    master_key: str,
    nonce: bytes,
    cipher: str,
    spd_block_bases: int,
    t5_chunk_bases: int,
    xor_enabled: bool
) -> str:

    master_key_bytes = (
        enc.pack_dna_2bit(
            master_key
        )
    )


    authentication_key = (
        enc.kdf(
            master_key_bytes,
            nonce,
            "AUTHENTICATION"
        )
    )


    authentication = hmac.new(
        authentication_key,
        digestmod=hashlib.sha256
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
        len(cipher).to_bytes(
            8,
            "big"
        )
    )


    authentication.update(
        spd_block_bases.to_bytes(
            8,
            "big"
        )
    )


    authentication.update(
        t5_chunk_bases.to_bytes(
            8,
            "big"
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
# TEST 1: PLAINTEXT SENSITIVITY / AVALANCHE
# =============================================================================

def test_plaintext_avalanche(
    *,
    enc,
    plain: str,
    reference_cipher: str,
    material: SessionMaterial,
    trials: int,
    rng: random.Random
) -> dict:

    global_base_values: List[
        float
    ] = []


    global_bit_values: List[
        float
    ] = []


    local_base_values: List[
        float
    ] = []


    local_bit_values: List[
        float
    ] = []


    trial_details = []


    for trial_index in range(
        trials
    ):

        (
            mutated_plain,
            position,
            old_base,
            new_base
        ) = mutate_one_plain_base(
            plain,
            rng
        )


        mutated_cipher = encrypt_with_material(
            enc,
            mutated_plain,
            material
        )


        global_base = hamming_base_percent(
            reference_cipher,
            mutated_cipher
        )


        global_bit = hamming_bit_percent_2bit(
            reference_cipher,
            mutated_cipher
        )


        block_start = (
            position
            // material.spd_block_bases
        ) * material.spd_block_bases


        block_end = min(
            block_start
            + material.spd_block_bases,
            len(plain)
        )


        local_base = hamming_base_percent(
            reference_cipher[
                block_start:
                block_end
            ],

            mutated_cipher[
                block_start:
                block_end
            ]
        )


        local_bit = hamming_bit_percent_2bit(
            reference_cipher[
                block_start:
                block_end
            ],

            mutated_cipher[
                block_start:
                block_end
            ]
        )


        global_base_values.append(
            global_base
        )


        global_bit_values.append(
            global_bit
        )


        local_base_values.append(
            local_base
        )


        local_bit_values.append(
            local_bit
        )


        trial_details.append({
            "trial": (
                trial_index
                + 1
            ),

            "changed_position_1based": (
                position
                + 1
            ),

            "old_base": (
                old_base
            ),

            "new_base": (
                new_base
            ),

            "affected_block_start_1based": (
                block_start
                + 1
            ),

            "affected_block_end_1based": (
                block_end
            ),

            "global_base_difference_percent": (
                global_base
            ),

            "global_bit_difference_percent_2bit": (
                global_bit
            ),

            "affected_block_base_difference_percent": (
                local_base
            ),

            "affected_block_bit_difference_percent_2bit": (
                local_bit
            )
        })


    return {
        "trials": (
            trials
        ),

        "changed_plain_bases_per_trial": (
            1
        ),

        "global_base_difference_percent": (
            numeric_summary(
                global_base_values
            )
        ),

        "global_bit_difference_percent_2bit": (
            numeric_summary(
                global_bit_values
            )
        ),

        "affected_block_base_difference_percent": (
            numeric_summary(
                local_base_values
            )
        ),

        "affected_block_bit_difference_percent_2bit": (
            numeric_summary(
                local_bit_values
            )
        ),

        "interpretation": (
            "Tek baz değişiminin etkisi SPD blok sınırları içindedir. "
            "Büyük verilerde global yüzde blok/dosya oranı nedeniyle "
            "düşebilir; yerel blok yüzdesi diffusion davranışını daha "
            "doğrudan gösterir."
        ),

        "details": (
            trial_details
        )
    }


# =============================================================================
# TEST 2: MASTER-KEY SENSITIVITY
# =============================================================================

def test_master_key_sensitivity(
    *,
    enc,
    t5_backend,
    plain: str,
    reference_cipher: str,
    master_key: str,
    nonce: bytes,
    spd_block_bases: int,
    t5_chunk_bases: int,
    xor_enabled: bool,
    trials: int,
    device: str,
    rng: random.Random
) -> dict:

    base_values: List[
        float
    ] = []


    bit_values: List[
        float
    ] = []


    trial_details = []


    total_setup_seconds = 0.0

    total_generation_seconds = 0.0


    for trial_index in range(
        trials
    ):

        (
            mutated_key,
            position,
            old_base,
            new_base
        ) = mutate_master_key_one_base(
            master_key,
            rng
        )


        material = build_session_material(
            enc=enc,

            t5_backend=t5_backend,

            master_key=(
                mutated_key
            ),

            nonce=nonce,

            number_of_bases=(
                len(plain)
            ),

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            device=device
        )


        mutated_cipher = encrypt_with_material(
            enc,
            plain,
            material
        )


        base_difference = hamming_base_percent(
            reference_cipher,
            mutated_cipher
        )


        bit_difference = hamming_bit_percent_2bit(
            reference_cipher,
            mutated_cipher
        )


        base_values.append(
            base_difference
        )


        bit_values.append(
            bit_difference
        )


        total_setup_seconds += (
            material.setup_seconds
        )


        total_generation_seconds += (
            material.generation_seconds
        )


        trial_details.append({
            "trial": (
                trial_index
                + 1
            ),

            "changed_key_position_1based": (
                position
                + 1
            ),

            "old_base": (
                old_base
            ),

            "new_base": (
                new_base
            ),

            "base_difference_percent": (
                base_difference
            ),

            "bit_difference_percent_2bit": (
                bit_difference
            ),

            "model_seed": (
                material.model_seed
            ),

            "model_fingerprint_sha256": (
                material.model_fingerprint
            )
        })


        del material

        gc.collect()


    return {
        "trials": (
            trials
        ),

        "changed_master_key_bases_per_trial": (
            1
        ),

        "base_difference_percent": (
            numeric_summary(
                base_values
            )
        ),

        "bit_difference_percent_2bit": (
            numeric_summary(
                bit_values
            )
        ),

        "expected_for_independent_ciphertexts": {
            "base_difference_percent": (
                75.0
            ),

            "bit_difference_percent_2bit": (
                50.0
            )
        },

        "t5_setup_seconds_total": (
            total_setup_seconds
        ),

        "t5_generation_seconds_total": (
            total_generation_seconds
        ),

        "details": (
            trial_details
        )
    }


# =============================================================================
# TEST 3: NONCE SENSITIVITY
# =============================================================================

def test_nonce_sensitivity(
    *,
    enc,
    t5_backend,
    plain: str,
    reference_cipher: str,
    master_key: str,
    nonce: bytes,
    spd_block_bases: int,
    t5_chunk_bases: int,
    xor_enabled: bool,
    trials: int,
    device: str,
    rng: random.Random
) -> dict:

    base_values: List[
        float
    ] = []


    bit_values: List[
        float
    ] = []


    trial_details = []


    total_setup_seconds = 0.0

    total_generation_seconds = 0.0


    used_nonce_hex = {
        nonce.hex()
    }


    for trial_index in range(
        trials
    ):

        while True:

            (
                mutated_nonce,
                bit_position
            ) = mutate_nonce_one_bit(
                nonce,
                rng
            )


            if (
                mutated_nonce.hex()
                not in used_nonce_hex
            ):

                used_nonce_hex.add(
                    mutated_nonce.hex()
                )

                break


        material = build_session_material(
            enc=enc,

            t5_backend=t5_backend,

            master_key=(
                master_key
            ),

            nonce=(
                mutated_nonce
            ),

            number_of_bases=(
                len(plain)
            ),

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            device=device
        )


        mutated_cipher = encrypt_with_material(
            enc,
            plain,
            material
        )


        base_difference = hamming_base_percent(
            reference_cipher,
            mutated_cipher
        )


        bit_difference = hamming_bit_percent_2bit(
            reference_cipher,
            mutated_cipher
        )


        base_values.append(
            base_difference
        )


        bit_values.append(
            bit_difference
        )


        total_setup_seconds += (
            material.setup_seconds
        )


        total_generation_seconds += (
            material.generation_seconds
        )


        trial_details.append({
            "trial": (
                trial_index
                + 1
            ),

            "changed_nonce_bit_0based": (
                bit_position
            ),

            "mutated_nonce_hex": (
                mutated_nonce.hex()
            ),

            "base_difference_percent": (
                base_difference
            ),

            "bit_difference_percent_2bit": (
                bit_difference
            ),

            "model_seed": (
                material.model_seed
            ),

            "model_fingerprint_sha256": (
                material.model_fingerprint
            )
        })


        del material

        gc.collect()


    return {
        "trials": (
            trials
        ),

        "changed_nonce_bits_per_trial": (
            1
        ),

        "base_difference_percent": (
            numeric_summary(
                base_values
            )
        ),

        "bit_difference_percent_2bit": (
            numeric_summary(
                bit_values
            )
        ),

        "expected_for_independent_ciphertexts": {
            "base_difference_percent": (
                75.0
            ),

            "bit_difference_percent_2bit": (
                50.0
            )
        },

        "t5_setup_seconds_total": (
            total_setup_seconds
        ),

        "t5_generation_seconds_total": (
            total_generation_seconds
        ),

        "details": (
            trial_details
        )
    }


# =============================================================================
# TEST 4: CPA-STYLE CHOSEN-PLAINTEXT DIVERGENCE
# =============================================================================

def test_cpa_style(
    *,
    enc,
    plain: str,
    reference_cipher: str,
    material: SessionMaterial,
    change_ratio: float,
    rng: random.Random
) -> dict:

    (
        mutated_plain,
        changed_positions
    ) = mutate_plain_ratio(
        plain,
        change_ratio,
        rng
    )


    mutated_cipher = encrypt_with_material(
        enc,
        mutated_plain,
        material
    )


    return {
        "controlled_same_nonce_and_same_session_material": (
            True
        ),

        "requested_plaintext_change_ratio": (
            change_ratio
        ),

        "changed_plaintext_bases": (
            len(
                changed_positions
            )
        ),

        "changed_positions_1based": [
            position
            + 1
            for position in changed_positions
        ],

        "plaintext_base_difference_percent": (
            hamming_base_percent(
                plain,
                mutated_plain
            )
        ),

        "plaintext_bit_difference_percent_2bit": (
            hamming_bit_percent_2bit(
                plain,
                mutated_plain
            )
        ),

        "ciphertext_base_difference_percent": (
            hamming_base_percent(
                reference_cipher,
                mutated_cipher
            )
        ),

        "ciphertext_bit_difference_percent_2bit": (
            hamming_bit_percent_2bit(
                reference_cipher,
                mutated_cipher
            )
        ),

        "note": (
            "Aynı nonce yalnız karşılaştırmayı izole etmek için kontrollü "
            "olarak kullanılmıştır. Gerçek şifreleme oturumlarında nonce "
            "benzersiz olmalıdır."
        )
    }


# =============================================================================
# TEST 5: KPA-STYLE INDEPENDENT-KEY DIVERGENCE
# =============================================================================

def test_kpa_style(
    *,
    enc,
    t5_backend,
    plain: str,
    reference_cipher: str,
    master_key_length: int,
    nonce: bytes,
    spd_block_bases: int,
    t5_chunk_bases: int,
    xor_enabled: bool,
    trials: int,
    device: str,
    rng: random.Random
) -> dict:

    base_values: List[
        float
    ] = []


    bit_values: List[
        float
    ] = []


    trial_details = []


    total_setup_seconds = 0.0

    total_generation_seconds = 0.0


    for trial_index in range(
        trials
    ):

        second_key = random_master_key(
            master_key_length,
            rng
        )


        material = build_session_material(
            enc=enc,

            t5_backend=t5_backend,

            master_key=(
                second_key
            ),

            nonce=nonce,

            number_of_bases=(
                len(plain)
            ),

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            device=device
        )


        second_cipher = encrypt_with_material(
            enc,
            plain,
            material
        )


        base_difference = hamming_base_percent(
            reference_cipher,
            second_cipher
        )


        bit_difference = hamming_bit_percent_2bit(
            reference_cipher,
            second_cipher
        )


        base_values.append(
            base_difference
        )


        bit_values.append(
            bit_difference
        )


        total_setup_seconds += (
            material.setup_seconds
        )


        total_generation_seconds += (
            material.generation_seconds
        )


        trial_details.append({
            "trial": (
                trial_index
                + 1
            ),

            "base_difference_percent": (
                base_difference
            ),

            "bit_difference_percent_2bit": (
                bit_difference
            ),

            "second_key_sha256": (
                hashlib.sha256(
                    second_key.encode(
                        "ascii"
                    )
                ).hexdigest()
            )
        })


        del material

        gc.collect()


    return {
        "trials": (
            trials
        ),

        "base_difference_percent": (
            numeric_summary(
                base_values
            )
        ),

        "bit_difference_percent_2bit": (
            numeric_summary(
                bit_values
            )
        ),

        "expected_for_independent_ciphertexts": {
            "base_difference_percent": (
                75.0
            ),

            "bit_difference_percent_2bit": (
                50.0
            )
        },

        "t5_setup_seconds_total": (
            total_setup_seconds
        ),

        "t5_generation_seconds_total": (
            total_generation_seconds
        ),

        "note": (
            "Bu ölçüm anahtar-kurtarma başarısını değil, bilinen aynı "
            "plaintext altında bağımsız master key'lerin oluşturduğu "
            "ciphertext ayrışmasını gösterir."
        ),

        "details": (
            trial_details
        )
    }


# =============================================================================
# TEST 6–7: FAULT PROPAGATION VE HMAC TAMPER DETECTION
# =============================================================================

def test_fault_and_hmac(
    *,
    enc,
    plain: str,
    reference_cipher: str,
    material: SessionMaterial,
    master_key: str,
    expected_hmac: str,
    number_of_faults: int,
    rng: random.Random
) -> dict:

    (
        tampered_cipher,
        positions
    ) = inject_cipher_faults(
        reference_cipher,
        number_of_faults,
        rng
    )


    # Yalnız kontrollü fault propagation analizi için HMAC atlanır.
    recovered_without_hmac = decrypt_with_material(
        enc,
        tampered_cipher,
        material
    )


    original_computed_hmac = (
        compute_ciphertext_hmac(
            enc=enc,

            master_key=(
                master_key
            ),

            nonce=(
                material.nonce
            ),

            cipher=(
                reference_cipher
            ),

            spd_block_bases=(
                material.spd_block_bases
            ),

            t5_chunk_bases=(
                material.t5_chunk_bases
            ),

            xor_enabled=(
                material.xor_enabled
            )
        )
    )


    tampered_computed_hmac = (
        compute_ciphertext_hmac(
            enc=enc,

            master_key=(
                master_key
            ),

            nonce=(
                material.nonce
            ),

            cipher=(
                tampered_cipher
            ),

            spd_block_bases=(
                material.spd_block_bases
            ),

            t5_chunk_bases=(
                material.t5_chunk_bases
            ),

            xor_enabled=(
                material.xor_enabled
            )
        )
    )


    original_hmac_verified = (
        hmac.compare_digest(
            original_computed_hmac.lower(),
            expected_hmac.lower()
        )
    )


    tampered_hmac_verified = (
        hmac.compare_digest(
            tampered_computed_hmac.lower(),
            expected_hmac.lower()
        )
    )


    return {
        "injected_cipher_faults": (
            len(
                positions
            )
        ),

        "fault_positions_1based": [
            position
            + 1
            for position in positions
        ],

        "recovered_base_similarity_percent_without_hmac": (
            similarity_base_percent(
                plain,
                recovered_without_hmac
            )
        ),

        "recovered_bit_similarity_percent_2bit_without_hmac": (
            similarity_bit_percent_2bit(
                plain,
                recovered_without_hmac
            )
        ),

        "recovered_base_difference_percent_without_hmac": (
            hamming_base_percent(
                plain,
                recovered_without_hmac
            )
        ),

        "recovered_bit_difference_percent_2bit_without_hmac": (
            hamming_bit_percent_2bit(
                plain,
                recovered_without_hmac
            )
        ),

        "expected_hmac_sha256": (
            expected_hmac
        ),

        "original_computed_hmac_sha256": (
            original_computed_hmac
        ),

        "tampered_computed_hmac_sha256": (
            tampered_computed_hmac
        ),

        "original_hmac_verified": (
            original_hmac_verified
        ),

        "tampered_hmac_verified": (
            tampered_hmac_verified
        ),

        "normal_decryption_expected_action": (
            "REJECT"
            if not tampered_hmac_verified
            else "UNEXPECTED_ACCEPT"
        ),

        "note": (
            "Normal deşifreleme, değiştirilmiş ciphertext'i HMAC "
            "doğrulamasında reddeder. Geri kazanım benzerliği yalnız "
            "fault yayılımını ölçmek için HMAC atlanarak hesaplanmıştır."
        )
    }


# =============================================================================
# EKRAN RAPORU
# =============================================================================

def print_min_mean_max(
    label: str,
    summary: dict
) -> None:

    print(
        f"{label:<36}: "
        f"{summary['minimum']:.6f} / "
        f"{summary['mean']:.6f} / "
        f"{summary['maximum']:.6f}"
    )


def print_report(
    report: dict
) -> None:

    baseline = (
        report[
            "baseline"
        ]
    )


    avalanche = (
        report[
            "plaintext_avalanche"
        ]
    )


    key_sensitivity = (
        report[
            "master_key_sensitivity"
        ]
    )


    nonce_sensitivity = (
        report[
            "nonce_sensitivity"
        ]
    )


    cpa = (
        report[
            "cpa_style"
        ]
    )


    kpa = (
        report[
            "kpa_style"
        ]
    )


    fault = (
        report[
            "fault_and_hmac"
        ]
    )


    print(
        "\n"
        + "=" * 88
    )


    print(
        "BMC T5-NREF DNA-SPD "
        "GÜVENLİK-2 TESTLERİ"
    )


    print(
        "=" * 88
    )


    print(
        f"Plaintext                       : "
        f"{report['files']['plain']}"
    )


    print(
        f"Ciphertext                      : "
        f"{report['files']['cipher']}"
    )


    print(
        f"DNA bazı                        : "
        f"{report['meta']['plain_bases']:,}"
    )


    print(
        f"SPD blok boyutu                 : "
        f"{report['meta']['spd_block_bases']:,}"
    )


    print(
        f"Nonce                           : "
        f"{report['meta']['nonce_hex']}"
    )


    print(
        "\n--- BASELINE DOĞRULAMA ---"
    )


    print(
        f"Mevcut ciphertext yeniden üretildi: "
        f"{baseline['ciphertext_reproduction_match']}"
    )


    print(
        f"Model seed eşleşmesi            : "
        f"{baseline['model_seed_match']}"
    )


    print(
        f"Model fingerprint eşleşmesi     : "
        f"{baseline['model_fingerprint_match']}"
    )


    print(
        f"Ciphertext HMAC eşleşmesi       : "
        f"{baseline['hmac_match']}"
    )


    print(
        "\n--- 1. PLAINTEXT SENSITIVITY / AVALANCHE ---"
    )


    print_min_mean_max(
        "Global baz % min/mean/max",

        avalanche[
            "global_base_difference_percent"
        ]
    )


    print_min_mean_max(
        "Global 2-bit % min/mean/max",

        avalanche[
            "global_bit_difference_percent_2bit"
        ]
    )


    print_min_mean_max(
        "Etkilenen blok baz %",

        avalanche[
            "affected_block_base_difference_percent"
        ]
    )


    print_min_mean_max(
        "Etkilenen blok 2-bit %",

        avalanche[
            "affected_block_bit_difference_percent_2bit"
        ]
    )


    print(
        "\n--- 2. MASTER-KEY SENSITIVITY ---"
    )


    print_min_mean_max(
        "Baz % min/mean/max",

        key_sensitivity[
            "base_difference_percent"
        ]
    )


    print_min_mean_max(
        "2-bit % min/mean/max",

        key_sensitivity[
            "bit_difference_percent_2bit"
        ]
    )


    print(
        "\n--- 3. NONCE SENSITIVITY ---"
    )


    print_min_mean_max(
        "Baz % min/mean/max",

        nonce_sensitivity[
            "base_difference_percent"
        ]
    )


    print_min_mean_max(
        "2-bit % min/mean/max",

        nonce_sensitivity[
            "bit_difference_percent_2bit"
        ]
    )


    print(
        "\n--- 4. CPA-STYLE ---"
    )


    print(
        f"Plaintext baz farkı             : "
        f"{cpa['plaintext_base_difference_percent']:.6f}%"
    )


    print(
        f"Plaintext 2-bit farkı           : "
        f"{cpa['plaintext_bit_difference_percent_2bit']:.6f}%"
    )


    print(
        f"Ciphertext baz farkı            : "
        f"{cpa['ciphertext_base_difference_percent']:.6f}%"
    )


    print(
        f"Ciphertext 2-bit farkı          : "
        f"{cpa['ciphertext_bit_difference_percent_2bit']:.6f}%"
    )


    print(
        "\n--- 5. KPA-STYLE BAĞIMSIZ KEY ---"
    )


    print_min_mean_max(
        "Baz % min/mean/max",

        kpa[
            "base_difference_percent"
        ]
    )


    print_min_mean_max(
        "2-bit % min/mean/max",

        kpa[
            "bit_difference_percent_2bit"
        ]
    )


    print(
        "\n--- 6–7. FAULT VE HMAC ---"
    )


    print(
        f"Enjekte edilen fault            : "
        f"{fault['injected_cipher_faults']}"
    )


    print(
        f"Recovered baz benzerliği        : "
        f"{fault['recovered_base_similarity_percent_without_hmac']:.6f}%"
    )


    print(
        f"Recovered 2-bit benzerliği      : "
        f"{fault['recovered_bit_similarity_percent_2bit_without_hmac']:.6f}%"
    )


    print(
        f"Orijinal HMAC doğrulandı        : "
        f"{fault['original_hmac_verified']}"
    )


    print(
        f"Bozulmuş HMAC doğrulandı        : "
        f"{fault['tampered_hmac_verified']}"
    )


    print(
        f"Normal deşifreleme kararı       : "
        f"{fault['normal_decryption_expected_action']}"
    )


    print(
        "\n--- SÜRE ---"
    )


    print(
        f"Toplam test süresi              : "
        f"{report['timing']['total_wall_seconds']:.6f} s"
    )


    print(
        f"JSON raporu                     : "
        f"{report['files']['output_json']}"
    )


    print(
        "=" * 88
    )


# =============================================================================
# ANA AKIŞ
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
        / ENCRYPTION_METADATA_FILENAME
    )


    master_key_path = (
        BASE_DIR
        / MASTER_KEY_FILENAME
    )


    t5_module_path = (
        BASE_DIR
        / T5_MODULE_FILENAME
    )


    encryption_code_path = (
        BASE_DIR
        / ENCRYPTION_CODE_FILENAME
    )


    output_json_path = (
        BASE_DIR
        / OUTPUT_JSON_FILENAME
    )


    print(
        f"[PLAIN]       "
        f"{plain_path}"
    )


    print(
        f"[CIPHER]      "
        f"{cipher_path}"
    )


    print(
        f"[META]        "
        f"{metadata_path}"
    )


    print(
        f"[MASTER_KEY]  "
        f"{master_key_path}"
    )


    print(
        f"[T5_MODULE]   "
        f"{t5_module_path}"
    )


    print(
        f"[ENC_CODE]    "
        f"{encryption_code_path}"
    )


    print(
        f"[DEVICE]      "
        f"{DEVICE}"
    )


    total_start = (
        time.perf_counter()
    )


    enc = load_python_module(
        encryption_code_path,
        "bmc_t5_spd_encrypt_api"
    )


    validate_required_encryption_api(
        enc
    )


    t5_backend = enc.load_t5_module(
        t5_module_path
    )


    plain = enc.read_dna(
        plain_path
    )


    actual_cipher = enc.read_dna(
        cipher_path
    )


    master_key = enc.read_dna(
        master_key_path
    )


    enc.validate_master_key(
        master_key
    )


    metadata = read_json(
        metadata_path
    )


    scheme = str(
        metadata[
            "scheme"
        ]
    )


    if scheme != enc.SCHEME:

        raise ValueError(
            f"Şema uyumsuzluğu: "
            f"metadata={scheme}, "
            f"şifreleme kodu={enc.SCHEME}"
        )


    metadata_bases = int(
        metadata[
            "input"
        ][
            "canonical_plain_bases"
        ]
    )


    if metadata_bases != len(
        plain
    ):

        raise ValueError(
            "Plaintext uzunluğu metadata "
            "ile uyuşmuyor."
        )


    if len(actual_cipher) != len(
        plain
    ):

        raise ValueError(
            "Ciphertext ve plaintext "
            "uzunluğu uyuşmuyor."
        )


    nonce = bytes.fromhex(
        metadata[
            "session"
        ][
            "nonce_hex"
        ]
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


    expected_hmac = str(
        metadata[
            "integrity"
        ][
            "ciphertext_hmac_sha256"
        ]
    )


    rng = random.Random(
        TEST_RANDOM_SEED
    )


    print(
        "\n[BASELINE] Aynı oturum "
        "materyali yeniden üretiliyor..."
    )


    baseline_material = (
        build_session_material(
            enc=enc,

            t5_backend=t5_backend,

            master_key=(
                master_key
            ),

            nonce=nonce,

            number_of_bases=(
                len(plain)
            ),

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            device=DEVICE
        )
    )


    reproduced_cipher = (
        encrypt_with_material(
            enc,
            plain,
            baseline_material
        )
    )


    reproduced_hmac = (
        compute_ciphertext_hmac(
            enc=enc,

            master_key=(
                master_key
            ),

            nonce=nonce,

            cipher=(
                reproduced_cipher
            ),

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            )
        )
    )


    metadata_model_seed = int(
        metadata[
            "t5_noref"
        ][
            "model_seed"
        ]
    )


    metadata_fingerprint = str(
        metadata[
            "t5_noref"
        ][
            "model_fingerprint_sha256"
        ]
    )


    baseline = {
        "ciphertext_reproduction_match": (
            reproduced_cipher
            == actual_cipher
        ),

        "model_seed_match": (
            baseline_material.model_seed
            == metadata_model_seed
        ),

        "model_fingerprint_match": (
            baseline_material
            .model_fingerprint
            .lower()
            == metadata_fingerprint
            .lower()
        ),

        "hmac_match": (
            hmac.compare_digest(
                reproduced_hmac.lower(),
                expected_hmac.lower()
            )
        ),

        "setup_seconds": (
            baseline_material
            .setup_seconds
        ),

        "generation_seconds": (
            baseline_material
            .generation_seconds
        )
    }


    if VERIFY_BASELINE_REPRODUCTION:

        if not all(
            (
                baseline[
                    "ciphertext_reproduction_match"
                ],

                baseline[
                    "model_seed_match"
                ],

                baseline[
                    "model_fingerprint_match"
                ],

                baseline[
                    "hmac_match"
                ]
            )
        ):

            raise RuntimeError(
                "Baseline yeniden üretimi başarısız. "
                "Doğru şifreleme kodu, master key, "
                "T5 modülü, ciphertext ve metadata "
                "birlikte kullanılmalıdır."
            )


    print(
        "[1/6] Plaintext avalanche testi..."
    )


    avalanche = test_plaintext_avalanche(
        enc=enc,

        plain=plain,

        reference_cipher=(
            actual_cipher
        ),

        material=(
            baseline_material
        ),

        trials=(
            AVALANCHE_TRIALS
        ),

        rng=rng
    )


    print(
        "[2/6] Master-key sensitivity testi..."
    )


    key_sensitivity = (
        test_master_key_sensitivity(
            enc=enc,

            t5_backend=t5_backend,

            plain=plain,

            reference_cipher=(
                actual_cipher
            ),

            master_key=(
                master_key
            ),

            nonce=nonce,

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            trials=(
                KEY_SENSITIVITY_TRIALS
            ),

            device=DEVICE,

            rng=rng
        )
    )


    print(
        "[3/6] Nonce sensitivity testi..."
    )


    nonce_sensitivity = (
        test_nonce_sensitivity(
            enc=enc,

            t5_backend=t5_backend,

            plain=plain,

            reference_cipher=(
                actual_cipher
            ),

            master_key=(
                master_key
            ),

            nonce=nonce,

            spd_block_bases=(
                spd_block_bases
            ),

            t5_chunk_bases=(
                t5_chunk_bases
            ),

            xor_enabled=(
                xor_enabled
            ),

            trials=(
                NONCE_SENSITIVITY_TRIALS
            ),

            device=DEVICE,

            rng=rng
        )
    )


    print(
        "[4/6] CPA-style testi..."
    )


    cpa_style = test_cpa_style(
        enc=enc,

        plain=plain,

        reference_cipher=(
            actual_cipher
        ),

        material=(
            baseline_material
        ),

        change_ratio=(
            CPA_CHANGE_RATIO
        ),

        rng=rng
    )


    print(
        "[5/6] KPA-style "
        "bağımsız-key testi..."
    )


    kpa_style = test_kpa_style(
        enc=enc,

        t5_backend=t5_backend,

        plain=plain,

        reference_cipher=(
            actual_cipher
        ),

        master_key_length=(
            len(master_key)
        ),

        nonce=nonce,

        spd_block_bases=(
            spd_block_bases
        ),

        t5_chunk_bases=(
            t5_chunk_bases
        ),

        xor_enabled=(
            xor_enabled
        ),

        trials=(
            KPA_STYLE_TRIALS
        ),

        device=DEVICE,

        rng=rng
    )


    print(
        "[6/6] Fault propagation "
        "ve HMAC testi..."
    )


    fault_and_hmac = test_fault_and_hmac(
        enc=enc,

        plain=plain,

        reference_cipher=(
            actual_cipher
        ),

        material=(
            baseline_material
        ),

        master_key=(
            master_key
        ),

        expected_hmac=(
            expected_hmac
        ),

        number_of_faults=(
            FAULT_BASES
        ),

        rng=rng
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    report = {
        "analysis": (
            "BMC T5-NREF DNA-SPD "
            "Security-2"
        ),

        "research_prototype": (
            True
        ),

        "files": {
            "plain": (
                plain_path.name
            ),

            "cipher": (
                cipher_path.name
            ),

            "encryption_metadata": (
                metadata_path.name
            ),

            "master_key": (
                master_key_path.name
            ),

            "t5_module": (
                t5_module_path.name
            ),

            "encryption_code": (
                encryption_code_path.name
            ),

            "output_json": (
                output_json_path.name
            )
        },

        "meta": {
            "scheme": (
                enc.SCHEME
            ),

            "plain_bases": (
                len(plain)
            ),

            "master_key_bases": (
                len(master_key)
            ),

            "nonce_hex": (
                nonce.hex()
            ),

            "spd_block_bases": (
                spd_block_bases
            ),

            "t5_chunk_bases": (
                t5_chunk_bases
            ),

            "xor_enabled": (
                xor_enabled
            ),

            "dna_bit_mapping": {
                "A": "00",
                "C": "01",
                "G": "10",
                "T": "11"
            },

            "independent_sequence_expectation": {
                "base_difference_percent": (
                    75.0
                ),

                "bit_difference_percent_2bit": (
                    50.0
                )
            },

            "test_random_seed": (
                TEST_RANDOM_SEED
            )
        },

        "baseline": (
            baseline
        ),

        "plaintext_avalanche": (
            avalanche
        ),

        "master_key_sensitivity": (
            key_sensitivity
        ),

        "nonce_sensitivity": (
            nonce_sensitivity
        ),

        "cpa_style": (
            cpa_style
        ),

        "kpa_style": (
            kpa_style
        ),

        "fault_and_hmac": (
            fault_and_hmac
        ),

        "timing": {
            "total_wall_seconds": (
                total_seconds
            )
        },

        "interpretation_limits": [
            (
                "Baz ve 2-bit Hamming "
                "ölçümleri ayrı raporlanmalıdır."
            ),

            (
                "Plaintext avalanche etkisi "
                "SPD blok sınırları içinde "
                "değerlendirilmelidir."
            ),

            (
                "KPA-style testi gerçek "
                "anahtar kurtarma saldırısı "
                "değildir."
            ),

            (
                "Ampirik sonuçlar biçimsel "
                "kriptografik güvenlik "
                "kanıtı değildir."
            )
        ]
    }


    write_json(
        output_json_path,
        report
    )


    print_report(
        report
    )


if __name__ == "__main__":

    main()
