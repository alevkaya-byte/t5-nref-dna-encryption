# -*- coding: utf-8 -*-
"""
BMC Bioinformatics – T5-NREF kontrollü DNA-yerel genom şifreleme.

Akış:
128-baz master key + nonce + counter + görev etiketi
-> T5-NREF KS/SUB/PERM/DIFF çıktıları
-> substitution -> permutation -> çift yönlü Z4 diffusion -> DNA-XOR
-> yalnız A/C/G/T ciphertext.

Master key dosyası yoksa ilk çalıştırmada otomatik oluşturulur ve aynı dosya
sonraki şifreleme/deşifreleme işlemleri için korunur. Master key tam uzunluklu
anahtar olarak periyodik biçimde tekrarlanmaz; T5-NREF her veri parçası için
counter-separated keystream üretir.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import itertools
import json
import math
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch

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

# Şifrelenecek DNA dosyası.
# Sonraki ölçeklerde yalnızca bu adı değiştir:
# ds_10kb.txt, ds_100kb.txt, ds_1mb.txt,
# ds_5mb.txt, ds_tam.txt
INPUT_FILENAME = "ds_tam.txt"

# Daha önce oluşturulan T5-NREF kodu
T5_MODULE_FILENAME = "T5_noref.py"

# Dosya yoksa otomatik olarak üretilecek 128 bazlık gizli anahtar
MASTER_KEY_FILENAME = "master_key_128.txt"

# Çıktı isimleri girdi adına göre otomatik oluşur
INPUT_STEM = Path(INPUT_FILENAME).stem

CIPHER_FILENAME = f"cipher_{INPUT_STEM}.txt"
METADATA_FILENAME = f"meta_{INPUT_STEM}.json"

# "cpu" veya "cuda"
DEVICE = "cpu"

# SPD blok boyutu
SPD_BLOCK_BASES = 4096

# Büyük veriler bu uzunlukta T5 parçalarına ayrılır
T5_CHUNK_BASES = 500_000

# Son kuaterner DNA-XOR katmanı
XOR_ENABLED = True

# İlk pilotta True.
# Nihai performans deneylerinde False yapılmalıdır.
SELF_CHECK = False

# Daha önceki çıktının üzerine yaz
OVERWRITE = True


# =============================================================================
# SABİTLER VE VERİ SINIFLARI
# =============================================================================

SCHEME = "BMC-T5-NREF-DNA-SPD-v3"

DNA = "ACGT"
DNA_SET = set(DNA)

BASE_TO_INT = {
    base: index
    for index, base in enumerate(DNA)
}

MASTER_KEY_BASES = 128
NONCE_BYTES = 16

# Her SPD bloğu için üretilecek kısa T5 kontrol DNA uzunlukları
SUB_CONTROL_BASES = 8
PERM_CONTROL_BASES = 32
DIFF_CONTROL_BASES = 16

# A/C/G/T üzerinde mümkün olan 24 birebir substitution tablosu
SUBSTITUTIONS: Tuple[
    Tuple[int, int, int, int], ...
] = tuple(
    itertools.permutations(
        range(4)
    )
)


@dataclass
class PhaseMetrics:

    wall_seconds: float
    cpu_seconds: Optional[float]

    rss_start_mb: Optional[float]
    rss_end_mb: Optional[float]
    rss_delta_mb: Optional[float]

    peak_rss_mb: Optional[float]
    peak_rss_delta_mb: Optional[float]

    peak_gpu_allocated_mb: Optional[float]


@dataclass
class MetricsAccumulator:

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    cpu_available: bool = True

    max_peak_rss_mb: Optional[float] = None
    max_peak_rss_delta_mb: Optional[float] = None
    max_peak_gpu_allocated_mb: Optional[float] = None

    def add(
        self,
        metric: PhaseMetrics
    ) -> None:

        self.wall_seconds += (
            metric.wall_seconds
        )

        if metric.cpu_seconds is None:

            self.cpu_available = False

        else:

            self.cpu_seconds += (
                metric.cpu_seconds
            )

        if metric.peak_rss_mb is not None:

            self.max_peak_rss_mb = (
                metric.peak_rss_mb
                if self.max_peak_rss_mb is None
                else max(
                    self.max_peak_rss_mb,
                    metric.peak_rss_mb
                )
            )

        if metric.peak_rss_delta_mb is not None:

            self.max_peak_rss_delta_mb = (
                metric.peak_rss_delta_mb
                if self.max_peak_rss_delta_mb is None
                else max(
                    self.max_peak_rss_delta_mb,
                    metric.peak_rss_delta_mb
                )
            )

        if metric.peak_gpu_allocated_mb is not None:

            self.max_peak_gpu_allocated_mb = (
                metric.peak_gpu_allocated_mb
                if self.max_peak_gpu_allocated_mb is None
                else max(
                    self.max_peak_gpu_allocated_mb,
                    metric.peak_gpu_allocated_mb
                )
            )

    def to_dict(
        self
    ) -> dict:

        return {
            "wall_seconds": (
                self.wall_seconds
            ),
            "cpu_seconds": (
                self.cpu_seconds
                if self.cpu_available
                else None
            ),
            "max_peak_rss_mb": (
                self.max_peak_rss_mb
            ),
            "max_peak_rss_delta_mb": (
                self.max_peak_rss_delta_mb
            ),
            "max_peak_gpu_allocated_mb": (
                self.max_peak_gpu_allocated_mb
            )
        }


@dataclass(frozen=True)
class BlockControl:

    substitution_id: int
    substitution: Tuple[int, int, int, int]

    permutation_seed: bytes

    diffusion_iv_forward: int
    diffusion_key_forward: int

    diffusion_iv_backward: int
    diffusion_key_backward: int


# =============================================================================
# PERFORMANS ÖLÇÜMÜ
# =============================================================================

class PeakRSSSampler:

    def __init__(
        self,
        interval_seconds: float = 0.01
    ):

        self.interval_seconds = max(
            0.001,
            float(interval_seconds)
        )

        self._stop = threading.Event()

        self._thread: Optional[
            threading.Thread
        ] = None

        self._process = (
            psutil.Process(
                os.getpid()
            )
            if HAVE_PSUTIL
            else None
        )

        self.start_rss: Optional[int] = None
        self.end_rss: Optional[int] = None
        self.peak_rss: Optional[int] = None

    def _run(
        self
    ) -> None:

        assert self._process is not None

        while not self._stop.is_set():

            try:

                rss = int(
                    self._process
                    .memory_info()
                    .rss
                )

                self.peak_rss = (
                    rss
                    if self.peak_rss is None
                    else max(
                        self.peak_rss,
                        rss
                    )
                )

            except Exception:

                pass

            self._stop.wait(
                self.interval_seconds
            )

    def __enter__(
        self
    ):

        if self._process is not None:

            self.start_rss = int(
                self._process
                .memory_info()
                .rss
            )

            self.peak_rss = (
                self.start_rss
            )

            self._thread = (
                threading.Thread(
                    target=self._run,
                    daemon=True
                )
            )

            self._thread.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb
    ):

        if self._process is not None:

            self._stop.set()

            if self._thread is not None:

                self._thread.join(
                    timeout=1.0
                )

            self.end_rss = int(
                self._process
                .memory_info()
                .rss
            )

            self.peak_rss = max(
                self.peak_rss or 0,
                self.end_rss
            )


def _cpu_now() -> Optional[float]:

    if not HAVE_PSUTIL:

        return None

    times = (
        psutil.Process(
            os.getpid()
        )
        .cpu_times()
    )

    return float(
        times.user
        + times.system
    )


def _cuda_sync(
    device: str
) -> None:

    if (
        device.startswith("cuda")
        and torch.cuda.is_available()
    ):

        torch.cuda.synchronize()


def measure(
    function,
    *,
    device: str
):

    use_cuda = (
        device.startswith("cuda")
        and torch.cuda.is_available()
    )

    if use_cuda:

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

    cpu_start = _cpu_now()

    with PeakRSSSampler() as sampler:

        wall_start = (
            time.perf_counter()
        )

        result = function()

        _cuda_sync(
            device
        )

        wall_seconds = (
            time.perf_counter()
            - wall_start
        )

    cpu_end = _cpu_now()

    def to_mb(
        value: Optional[int]
    ) -> Optional[float]:

        return (
            None
            if value is None
            else value
            / (1024.0 ** 2)
        )

    rss_start_mb = to_mb(
        sampler.start_rss
    )

    rss_end_mb = to_mb(
        sampler.end_rss
    )

    peak_rss_mb = to_mb(
        sampler.peak_rss
    )

    peak_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / (1024.0 ** 2)
        if use_cuda
        else None
    )

    metric = PhaseMetrics(
        wall_seconds=wall_seconds,

        cpu_seconds=(
            None
            if cpu_start is None
            or cpu_end is None
            else cpu_end
            - cpu_start
        ),

        rss_start_mb=rss_start_mb,
        rss_end_mb=rss_end_mb,

        rss_delta_mb=(
            None
            if rss_start_mb is None
            or rss_end_mb is None
            else rss_end_mb
            - rss_start_mb
        ),

        peak_rss_mb=peak_rss_mb,

        peak_rss_delta_mb=(
            None
            if rss_start_mb is None
            or peak_rss_mb is None
            else peak_rss_mb
            - rss_start_mb
        ),

        peak_gpu_allocated_mb=(
            peak_gpu_mb
        )
    )

    return (
        result,
        metric
    )


# =============================================================================
# DNA DOSYA İŞLEMLERİ VE MASTER KEY
# =============================================================================

def read_dna(
    path: Path
) -> str:

    if not path.exists():

        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}"
        )

    output = bytearray()

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            if line.startswith(">"):

                continue

            for character in line.upper():

                if character in DNA_SET:

                    output.append(
                        ord(character)
                    )

    if not output:

        raise ValueError(
            f"Geçerli A/C/G/T dizisi "
            f"bulunamadı: {path}"
        )

    return output.decode(
        "ascii"
    )


def count_dna_bases(
    path: Path
) -> int:

    if not path.exists():

        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}"
        )

    total = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            if not line.startswith(">"):

                total += sum(
                    character in DNA_SET
                    for character
                    in line.upper()
                )

    if total <= 0:

        raise ValueError(
            f"Geçerli A/C/G/T dizisi "
            f"bulunamadı: {path}"
        )

    return total


def iter_dna_chunks(
    path: Path,
    chunk_bases: int
) -> Iterator[str]:

    buffer = bytearray()

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            if line.startswith(">"):

                continue

            for character in line.upper():

                if character in DNA_SET:

                    buffer.append(
                        ord(character)
                    )

                    if (
                        len(buffer)
                        == chunk_bases
                    ):

                        yield buffer.decode(
                            "ascii"
                        )

                        buffer.clear()

    if buffer:

        yield buffer.decode(
            "ascii"
        )


def infer_expected_bases_from_name(
    path: Path
) -> Optional[int]:

    match = re.fullmatch(
        r"ds_(\d+)(kb|mb)",
        path.stem.lower()
    )

    if match is None:

        return None

    value = int(
        match.group(1)
    )

    unit = match.group(2)

    return value * (
        1_000
        if unit == "kb"
        else 1_000_000
    )


def validate_master_key(
    master_key: str
) -> None:

    if (
        len(master_key)
        != MASTER_KEY_BASES
    ):

        raise ValueError(
            f"Master key tam "
            f"{MASTER_KEY_BASES} baz "
            f"olmalıdır; "
            f"mevcut={len(master_key)}."
        )

    if any(
        base not in DNA_SET
        for base in master_key
    ):

        raise ValueError(
            "Master key yalnız "
            "A/C/G/T içermelidir."
        )


def load_or_create_master_key(
    path: Path
) -> Tuple[str, bool]:

    if path.exists():

        key = read_dna(
            path
        )

        validate_master_key(
            key
        )

        return (
            key,
            False
        )

    key = "".join(
        secrets.choice(
            DNA
        )
        for _ in range(
            MASTER_KEY_BASES
        )
    )

    path.write_text(
        key + "\n",
        encoding="ascii"
    )

    try:

        os.chmod(
            path,
            0o600
        )

    except Exception:

        pass

    return (
        key,
        True
    )


def dna_to_ints(
    sequence: str
) -> List[int]:

    return [
        BASE_TO_INT[base]
        for base in sequence
    ]


def ints_to_dna(
    values: Sequence[int]
) -> str:

    return "".join(
        DNA[int(value) & 3]
        for value in values
    )


def pack_dna_2bit(
    sequence: str
) -> bytes:

    output = bytearray(
        math.ceil(
            len(sequence) / 4
        )
    )

    output_index = 0
    accumulator = 0
    used = 0

    for base in sequence:

        accumulator = (
            accumulator << 2
        ) | BASE_TO_INT[base]

        used += 1

        if used == 4:

            output[
                output_index
            ] = accumulator

            output_index += 1
            accumulator = 0
            used = 0

    if used:

        output[
            output_index
        ] = accumulator << (
            2 * (4 - used)
        )

    return bytes(
        output
    )


# =============================================================================
# DOMAIN SEPARATION / KDF
# =============================================================================

def _lp(
    payload: bytes
) -> bytes:

    return (
        len(payload)
        .to_bytes(
            4,
            "big"
        )
        + payload
    )


def kdf(
    master_key_bytes: bytes,
    nonce: bytes,
    label: str,
    *,
    counter: Optional[int] = None,
    extra: bytes = b""
) -> bytes:

    message = bytearray()

    message.extend(
        _lp(
            SCHEME.encode(
                "ascii"
            )
        )
    )

    message.extend(
        _lp(
            label.encode(
                "utf-8"
            )
        )
    )

    message.extend(
        _lp(
            nonce
        )
    )

    if counter is not None:

        if counter < 0:

            raise ValueError(
                "Counter negatif olamaz."
            )

        message.extend(
            counter.to_bytes(
                8,
                "big"
            )
        )

    message.extend(
        _lp(
            extra
        )
    )

    return hmac.new(
        master_key_bytes,
        bytes(message),
        hashlib.sha256
    ).digest()


def seed_from_digest(
    digest: bytes
) -> int:

    return (
        int.from_bytes(
            digest[:8],
            "big"
        )
        & ((1 << 63) - 1)
    )


def expand_base4_mask(
    master_key_bytes: bytes,
    nonce: bytes,
    label: str,
    counter: int,
    length: int
) -> List[int]:

    output: List[int] = []

    inner_counter = 0

    while len(output) < length:

        digest = kdf(
            master_key_bytes,
            nonce,
            f"PROMPT-MASK:{label}",
            counter=counter,
            extra=(
                inner_counter
                .to_bytes(
                    8,
                    "big"
                )
            )
        )

        for byte in digest:

            output.extend(
                (
                    (byte >> 6) & 3,
                    (byte >> 4) & 3,
                    (byte >> 2) & 3,
                    byte & 3
                )
            )

            if len(output) >= length:

                break

        inner_counter += 1

    return output[:length]


def build_prompt(
    master_key: str,
    master_key_bytes: bytes,
    nonce: bytes,
    label: str,
    counter: int,
    source_len: int,
    device: str
) -> torch.LongTensor:

    key_values = dna_to_ints(
        master_key
    )

    if (
        source_len
        != len(key_values)
    ):

        key_values = [
            key_values[
                index
                % len(key_values)
            ]
            for index in range(
                source_len
            )
        ]

    mask = expand_base4_mask(
        master_key_bytes,
        nonce,
        label,
        counter,
        source_len
    )

    prompt = [
        key_values[index]
        ^ mask[index]
        for index in range(
            source_len
        )
    ]

    return torch.tensor(
        prompt,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)


# =============================================================================
# T5-NREF BACKEND
# =============================================================================

def load_t5_module(
    module_path: Path
):

    if not module_path.exists():

        raise FileNotFoundError(
            f"T5 modülü bulunamadı: "
            f"{module_path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "bmc_t5_noref_backend",
            module_path
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"T5 modülü yüklenemedi: "
            f"{module_path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    required = (
        "T5NoRefConfig",
        "T5NoReferenceDNA",
        "seed_all",
        "make_generator",
        "generate_dna",
        "model_fingerprint"
    )

    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name
        )
    ]

    if missing:

        raise AttributeError(
            "T5 modülünde eksik "
            "bileşenler: "
            + ", ".join(
                missing
            )
        )

    return module


def create_session_model(
    module,
    master_key_bytes: bytes,
    nonce: bytes,
    device: str
):

    model_seed = (
        seed_from_digest(
            kdf(
                master_key_bytes,
                nonce,
                "MODEL-INITIALIZATION"
            )
        )
    )

    module.seed_all(
        model_seed
    )

    config = (
        module.T5NoRefConfig()
    )

    model = (
        module
        .T5NoReferenceDNA(
            config
        )
        .to(
            device
        )
        .eval()
    )

    fingerprint = (
        module.model_fingerprint(
            model
        )
    )

    return (
        model,
        config,
        model_seed,
        fingerprint
    )


def generate_domain_dna(
    *,
    module,
    model,
    config,
    master_key: str,
    master_key_bytes: bytes,
    nonce: bytes,
    label: str,
    counter: int,
    output_length: int,
    device: str
) -> str:

    if output_length <= 0:

        return ""

    prompt = build_prompt(
        master_key,
        master_key_bytes,
        nonce,
        label,
        counter,
        int(
            config.source_len
        ),
        device
    )

    generation_seed = (
        seed_from_digest(
            kdf(
                master_key_bytes,
                nonce,
                f"GENERATION:{label}",
                counter=counter
            )
        )
    )

    generator = (
        module.make_generator(
            generation_seed,
            device
        )
    )

    dna, _rules, _bits = (
        module.generate_dna(
            model=model,
            start_tokens=prompt,
            output_length=int(
                output_length
            ),
            device=device,
            generator=generator
        )
    )

    if (
        len(dna)
        != output_length
    ):

        raise RuntimeError(
            f"{label}/{counter} "
            f"uzunluğu hatalı."
        )

    return dna


# =============================================================================
# SUBSTITUTION / PERMUTATION / DIFFUSION / DNA-XOR
# =============================================================================

def control_digest(
    nonce: bytes,
    label: str,
    block_counter: int,
    dna_control: str
) -> bytes:

    digest = hashlib.sha256()

    digest.update(
        _lp(
            SCHEME.encode(
                "ascii"
            )
        )
    )

    digest.update(
        _lp(
            label.encode(
                "ascii"
            )
        )
    )

    digest.update(
        _lp(
            nonce
        )
    )

    digest.update(
        block_counter.to_bytes(
            8,
            "big"
        )
    )

    digest.update(
        _lp(
            dna_control.encode(
                "ascii"
            )
        )
    )

    return digest.digest()


def block_control(
    *,
    nonce: bytes,
    block_counter: int,
    sub_dna: str,
    perm_dna: str,
    diff_dna: str
) -> BlockControl:

    sub_hash = control_digest(
        nonce,
        "SUB",
        block_counter,
        sub_dna
    )

    perm_hash = control_digest(
        nonce,
        "PERM",
        block_counter,
        perm_dna
    )

    diff_hash = control_digest(
        nonce,
        "DIFF",
        block_counter,
        diff_dna
    )

    substitution_id = (
        int.from_bytes(
            sub_hash[:4],
            "big"
        )
        % len(
            SUBSTITUTIONS
        )
    )

    return BlockControl(
        substitution_id=(
            substitution_id
        ),

        substitution=(
            SUBSTITUTIONS[
                substitution_id
            ]
        ),

        permutation_seed=(
            perm_hash
        ),

        diffusion_iv_forward=(
            diff_hash[0] & 3
        ),

        diffusion_key_forward=(
            diff_hash[1] & 3
        ),

        diffusion_iv_backward=(
            diff_hash[2] & 3
        ),

        diffusion_key_backward=(
            diff_hash[3] & 3
        )
    )


class HashCounterRNG:

    def __init__(
        self,
        seed: bytes
    ):

        self.seed = bytes(
            seed
        )

        self.counter = 0

        self.buffer = (
            bytearray()
        )

    def _refill(
        self
    ) -> None:

        self.buffer.extend(
            hashlib.sha256(
                self.seed
                + self.counter
                .to_bytes(
                    8,
                    "big"
                )
            ).digest()
        )

        self.counter += 1

    def uint64(
        self
    ) -> int:

        while len(self.buffer) < 8:

            self._refill()

        value = int.from_bytes(
            self.buffer[:8],
            "big"
        )

        del self.buffer[:8]

        return value

    def randbelow(
        self,
        upper: int
    ) -> int:

        if upper <= 0:

            raise ValueError(
                "upper pozitif olmalıdır."
            )

        limit = (
            (1 << 64)
            - (
                (1 << 64)
                % upper
            )
        )

        while True:

            value = self.uint64()

            if value < limit:

                return value % upper


def make_permutation(
    length: int,
    seed: bytes
) -> List[int]:

    permutation = list(
        range(length)
    )

    rng = HashCounterRNG(
        seed
    )

    for index in range(
        length - 1,
        0,
        -1
    ):

        swap_index = (
            rng.randbelow(
                index + 1
            )
        )

        (
            permutation[index],
            permutation[swap_index]
        ) = (
            permutation[swap_index],
            permutation[index]
        )

    return permutation


def substitute(
    values: Sequence[int],
    table: Sequence[int]
) -> List[int]:

    return [
        int(
            table[value]
        )
        for value in values
    ]


def inverse_substitute(
    values: Sequence[int],
    table: Sequence[int]
) -> List[int]:

    inverse = [
        0,
        0,
        0,
        0
    ]

    for source, target in enumerate(
        table
    ):

        inverse[
            int(target)
        ] = int(source)

    return [
        inverse[value]
        for value in values
    ]


def permute(
    values: Sequence[int],
    permutation: Sequence[int]
) -> List[int]:

    return [
        int(
            values[source]
        )
        for source in permutation
    ]


def inverse_permute(
    values: Sequence[int],
    permutation: Sequence[int]
) -> List[int]:

    output = [
        0
    ] * len(values)

    for (
        output_index,
        source_index
    ) in enumerate(
        permutation
    ):

        output[
            source_index
        ] = int(
            values[
                output_index
            ]
        )

    return output


def diffuse(
    values: Sequence[int],
    control: BlockControl
) -> List[int]:

    if not values:

        return []

    forward = [
        0
    ] * len(values)

    previous = (
        control
        .diffusion_iv_forward
    )

    for index, value in enumerate(
        values
    ):

        tweak = (
            control
            .diffusion_key_forward
            + (index & 3)
        ) & 3

        current = (
            int(value)
            + previous
            + tweak
        ) & 3

        forward[index] = current

        previous = current

    output = [
        0
    ] * len(values)

    next_value = (
        control
        .diffusion_iv_backward
    )

    for reverse_offset, index in enumerate(
        range(
            len(values) - 1,
            -1,
            -1
        )
    ):

        tweak = (
            control
            .diffusion_key_backward
            + (
                reverse_offset
                & 3
            )
        ) & 3

        current = (
            forward[index]
            + next_value
            + tweak
        ) & 3

        output[index] = current

        next_value = current

    return output


def inverse_diffuse(
    values: Sequence[int],
    control: BlockControl
) -> List[int]:

    if not values:

        return []

    forward = [
        0
    ] * len(values)

    for reverse_offset, index in enumerate(
        range(
            len(values) - 1,
            -1,
            -1
        )
    ):

        next_value = (
            control
            .diffusion_iv_backward
            if index
            == len(values) - 1
            else int(
                values[index + 1]
            )
        )

        tweak = (
            control
            .diffusion_key_backward
            + (
                reverse_offset
                & 3
            )
        ) & 3

        forward[index] = (
            int(values[index])
            - next_value
            - tweak
        ) & 3

    output = [
        0
    ] * len(values)

    for index in range(
        len(values)
    ):

        previous = (
            control
            .diffusion_iv_forward
            if index == 0
            else forward[
                index - 1
            ]
        )

        tweak = (
            control
            .diffusion_key_forward
            + (index & 3)
        ) & 3

        output[index] = (
            forward[index]
            - previous
            - tweak
        ) & 3

    return output


def dna_xor(
    values: Sequence[int],
    keystream: Sequence[int]
) -> List[int]:

    if (
        len(values)
        != len(keystream)
    ):

        raise ValueError(
            "DNA-XOR uzunlukları "
            "eşit olmalıdır."
        )

    return [
        int(value)
        ^ int(key_value)
        for value, key_value
        in zip(
            values,
            keystream
        )
    ]


def encrypt_block(
    plain_dna: str,
    keystream_dna: str,
    control: BlockControl,
    *,
    xor_enabled: bool
) -> str:

    values = substitute(
        dna_to_ints(
            plain_dna
        ),
        control.substitution
    )

    permutation = make_permutation(
        len(values),
        control.permutation_seed
    )

    values = permute(
        values,
        permutation
    )

    values = diffuse(
        values,
        control
    )

    if xor_enabled:

        values = dna_xor(
            values,
            dna_to_ints(
                keystream_dna
            )
        )

    return ints_to_dna(
        values
    )


def decrypt_block_for_self_check(
    cipher_dna: str,
    keystream_dna: str,
    control: BlockControl,
    *,
    xor_enabled: bool
) -> str:

    values = dna_to_ints(
        cipher_dna
    )

    if xor_enabled:

        values = dna_xor(
            values,
            dna_to_ints(
                keystream_dna
            )
        )

    values = inverse_diffuse(
        values,
        control
    )

    permutation = make_permutation(
        len(values),
        control.permutation_seed
    )

    values = inverse_permute(
        values,
        permutation
    )

    values = inverse_substitute(
        values,
        control.substitution
    )

    return ints_to_dna(
        values
    )


# =============================================================================
# ANA ŞİFRELEME
# =============================================================================

def encrypt_genome(
    *,
    plain_path: Path,
    master_key_path: Path,
    t5_module_path: Path,
    cipher_path: Path,
    metadata_path: Path,
    spd_block_bases: int,
    t5_chunk_bases: int,
    device: str,
    xor_enabled: bool,
    self_check: bool,
    overwrite: bool
) -> dict:

    if spd_block_bases < 128:

        raise ValueError(
            "SPD blok boyutu "
            "en az 128 baz olmalıdır."
        )

    if (
        t5_chunk_bases
        < spd_block_bases
    ):

        raise ValueError(
            "T5 chunk boyutu "
            "SPD blok boyutundan "
            "küçük olamaz."
        )

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA istendi ancak "
            "CUDA kullanılamıyor."
        )

    if (
        cipher_path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            f"Çıktı zaten var: "
            f"{cipher_path}"
        )

    if (
        metadata_path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            f"Metadata zaten var: "
            f"{metadata_path}"
        )

    observed_start = (
        time.perf_counter()
    )

    # -------------------------------------------------------------------------
    # Girdi ve master key
    # -------------------------------------------------------------------------

    read_start = (
        time.perf_counter()
    )

    number_of_bases = (
        count_dna_bases(
            plain_path
        )
    )

    expected_bases = (
        infer_expected_bases_from_name(
            plain_path
        )
    )

    if (
        expected_bases is not None
        and number_of_bases
        != expected_bases
    ):

        raise ValueError(
            f"{plain_path.name} için "
            f"{expected_bases:,} baz "
            f"bekleniyordu; "
            f"{number_of_bases:,} "
            f"geçerli baz bulundu."
        )

    (
        master_key,
        master_key_created
    ) = load_or_create_master_key(
        master_key_path
    )

    master_key_bytes = (
        pack_dna_2bit(
            master_key
        )
    )

    io_read_seconds = (
        time.perf_counter()
        - read_start
    )

    # Her koşumda yeni nonce
    nonce = secrets.token_bytes(
        NONCE_BYTES
    )

    # -------------------------------------------------------------------------
    # T5 setup
    # -------------------------------------------------------------------------

    def setup_t5():

        module = load_t5_module(
            t5_module_path
        )

        (
            model,
            config,
            model_seed,
            fingerprint
        ) = create_session_model(
            module,
            master_key_bytes,
            nonce,
            device
        )

        return (
            module,
            model,
            config,
            model_seed,
            fingerprint
        )

    (
        module,
        model,
        config,
        model_seed,
        fingerprint
    ), setup_metric = measure(
        setup_t5,
        device=device
    )

  
    full_t5_chunks, tail_bases = divmod(
        number_of_bases,
        t5_chunk_bases
        )
    number_of_spd_blocks = (
        full_t5_chunks
        * math.ceil(
            t5_chunk_bases
            / spd_block_bases
            )
        + (
            math.ceil(
                tail_bases
                / spd_block_bases
                )
            if tail_bases > 0
            else 0
            )
        )
    
    
    number_of_t5_chunks = (
        math.ceil(
            number_of_bases
            / t5_chunk_bases
        )
    )

    keygen_accumulator = (
        MetricsAccumulator()
    )

    encryption_accumulator = (
        MetricsAccumulator()
    )

    self_check_accumulator = (
        MetricsAccumulator()
    )

    substitution_histogram = [
        0
    ] * len(
        SUBSTITUTIONS
    )

    io_cipher_write_seconds = 0.0

    # -------------------------------------------------------------------------
    # Ciphertext HMAC
    # -------------------------------------------------------------------------

    authentication_key = kdf(
        master_key_bytes,
        nonce,
        "AUTHENTICATION"
    )

    authentication = hmac.new(
        authentication_key,
        digestmod=hashlib.sha256
    )

    authentication.update(
        SCHEME.encode(
            "ascii"
        )
    )

    authentication.update(
        nonce
    )

    authentication.update(
        number_of_bases.to_bytes(
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

    temporary_cipher_path = (
        cipher_path.with_suffix(
            cipher_path.suffix
            + ".tmp"
        )
    )

    global_spd_counter = 0
    processed_bases = 0

    with temporary_cipher_path.open(
        "w",
        encoding="ascii",
        newline=""
    ) as output_handle:

        for (
            chunk_counter,
            plain_chunk
        ) in enumerate(
            iter_dna_chunks(
                plain_path,
                t5_chunk_bases
            )
        ):

            chunk_length = len(
                plain_chunk
            )

            blocks_in_chunk = math.ceil(
                chunk_length
                / spd_block_bases
            )

            chunk_spd_counter_start = (
                global_spd_counter
            )

            # -------------------------------------------------------------
            # T5-NREF tam uzunluklu KS ve kısa kontrol akışları
            # -------------------------------------------------------------

            def generate_material():

                keystream = (
                    generate_domain_dna(
                        module=module,
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
                    generate_domain_dna(
                        module=module,
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
                            * SUB_CONTROL_BASES
                        ),

                        device=device
                    )
                )

                permutation_stream = (
                    generate_domain_dna(
                        module=module,
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
                            * PERM_CONTROL_BASES
                        ),

                        device=device
                    )
                )

                diffusion_stream = (
                    generate_domain_dna(
                        module=module,
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
                            * DIFF_CONTROL_BASES
                        ),

                        device=device
                    )
                )

                return (
                    keystream,
                    substitution_stream,
                    permutation_stream,
                    diffusion_stream
                )

            (
                keystream,
                substitution_stream,
                permutation_stream,
                diffusion_stream
            ), keygen_metric = measure(
                generate_material,
                device=device
            )

            keygen_accumulator.add(
                keygen_metric
            )

            def get_control(
                local_block: int
            ) -> BlockControl:

                return block_control(
                    nonce=nonce,

                    block_counter=(
                        chunk_spd_counter_start
                        + local_block
                    ),

                    sub_dna=(
                        substitution_stream[
                            local_block
                            * SUB_CONTROL_BASES:
                            (
                                local_block + 1
                            )
                            * SUB_CONTROL_BASES
                        ]
                    ),

                    perm_dna=(
                        permutation_stream[
                            local_block
                            * PERM_CONTROL_BASES:
                            (
                                local_block + 1
                            )
                            * PERM_CONTROL_BASES
                        ]
                    ),

                    diff_dna=(
                        diffusion_stream[
                            local_block
                            * DIFF_CONTROL_BASES:
                            (
                                local_block + 1
                            )
                            * DIFF_CONTROL_BASES
                        ]
                    )
                )

            # -------------------------------------------------------------
            # SPD + DNA-XOR şifreleme çekirdeği
            # -------------------------------------------------------------

            def encrypt_chunk():

                cipher_parts: List[str] = []

                for local_block in range(
                    blocks_in_chunk
                ):

                    local_start = (
                        local_block
                        * spd_block_bases
                    )

                    local_end = min(
                        local_start
                        + spd_block_bases,
                        chunk_length
                    )

                    control = get_control(
                        local_block
                    )

                    substitution_histogram[
                        control.substitution_id
                    ] += 1

                    cipher_parts.append(
                        encrypt_block(
                            plain_chunk[
                                local_start:
                                local_end
                            ],

                            keystream[
                                local_start:
                                local_end
                            ],

                            control,

                            xor_enabled=(
                                xor_enabled
                            )
                        )
                    )

                return "".join(
                    cipher_parts
                )

            (
                cipher_chunk,
                encryption_metric
            ) = measure(
                encrypt_chunk,
                device="cpu"
            )

            encryption_accumulator.add(
                encryption_metric
            )

            # -------------------------------------------------------------
            # İç tersinirlik kontrolü
            # Bu süre şifreleme çekirdeğine dâhil değildir.
            # -------------------------------------------------------------

            if self_check:

                def verify_chunk():

                    recovered_parts: List[str] = []

                    for local_block in range(
                        blocks_in_chunk
                    ):

                        local_start = (
                            local_block
                            * spd_block_bases
                        )

                        local_end = min(
                            local_start
                            + spd_block_bases,
                            chunk_length
                        )

                        recovered_parts.append(
                            decrypt_block_for_self_check(
                                cipher_chunk[
                                    local_start:
                                    local_end
                                ],

                                keystream[
                                    local_start:
                                    local_end
                                ],

                                get_control(
                                    local_block
                                ),

                                xor_enabled=(
                                    xor_enabled
                                )
                            )
                        )

                    recovered_chunk = "".join(
                        recovered_parts
                    )

                    if (
                        recovered_chunk
                        != plain_chunk
                    ):

                        raise RuntimeError(
                            "Tersinirlik kontrolü "
                            "başarısız: "
                            f"chunk={chunk_counter}"
                        )

                    return True

                _, self_check_metric = measure(
                    verify_chunk,
                    device="cpu"
                )

                self_check_accumulator.add(
                    self_check_metric
                )

            # -------------------------------------------------------------
            # Ciphertext yazımı
            # -------------------------------------------------------------

            write_start = (
                time.perf_counter()
            )

            output_handle.write(
                cipher_chunk
            )

            authentication.update(
                cipher_chunk.encode(
                    "ascii"
                )
            )

            io_cipher_write_seconds += (
                time.perf_counter()
                - write_start
            )

            global_spd_counter += (
                blocks_in_chunk
            )

            processed_bases += (
                chunk_length
            )

            print(
                f"[CHUNK "
                f"{chunk_counter + 1}/"
                f"{number_of_t5_chunks}] "
                f"bases={chunk_length:,} | "
                f"keygen="
                f"{keygen_metric.wall_seconds:.3f}s | "
                f"encrypt="
                f"{encryption_metric.wall_seconds:.6f}s"
            )

    if (
        processed_bases
        != number_of_bases
    ):

        raise RuntimeError(
            f"İşlenen baz sayısı hatalı: "
            f"beklenen={number_of_bases:,}, "
            f"işlenen={processed_bases:,}"
        )

    os.replace(
        temporary_cipher_path,
        cipher_path
    )

    # -------------------------------------------------------------------------
    # Toplam süreler ve throughput
    # -------------------------------------------------------------------------

    setup_seconds = (
        setup_metric.wall_seconds
    )

    keygen_seconds = (
        keygen_accumulator
        .wall_seconds
    )

    encryption_seconds = (
        encryption_accumulator
        .wall_seconds
    )

    compute_total_seconds = (
        setup_seconds
        + keygen_seconds
        + encryption_seconds
    )

    keygen_throughput = (
        number_of_bases
        / keygen_seconds
        if keygen_seconds > 0
        else float("inf")
    )

    encryption_throughput = (
        number_of_bases
        / encryption_seconds
        if encryption_seconds > 0
        else float("inf")
    )

    end_to_end_throughput = (
        number_of_bases
        / compute_total_seconds
        if compute_total_seconds > 0
        else float("inf")
    )

    total_compute_cpu_seconds = None

    if (
        setup_metric.cpu_seconds
        is not None

        and
        keygen_accumulator
        .cpu_available

        and
        encryption_accumulator
        .cpu_available
    ):

        total_compute_cpu_seconds = (
            setup_metric.cpu_seconds
            + keygen_accumulator.cpu_seconds
            + encryption_accumulator.cpu_seconds
        )

    peak_rss_values = [
        value
        for value in (
            setup_metric.peak_rss_mb,

            keygen_accumulator
            .max_peak_rss_mb,

            encryption_accumulator
            .max_peak_rss_mb
        )
        if value is not None
    ]

    peak_gpu_values = [
        value
        for value in (
            setup_metric
            .peak_gpu_allocated_mb,

            keygen_accumulator
            .max_peak_gpu_allocated_mb,

            encryption_accumulator
            .max_peak_gpu_allocated_mb
        )
        if value is not None
    ]

    metadata = {
        "scheme": SCHEME,

        "research_prototype": True,

        "input": {
            "plain_file": (
                plain_path.name
            ),

            "canonical_plain_bases": (
                number_of_bases
            ),

            "expected_bases_from_filename": (
                expected_bases
            )
        },

        "master_key": {
            "file": (
                master_key_path.name
            ),

            "bases": (
                len(master_key)
            ),

            "created_this_run": (
                master_key_created
            ),

            "stored_in_ciphertext": False,
            "stored_in_metadata": False,

            "periodic_extension": False
        },

        "session": {
            "nonce_hex": (
                nonce.hex()
            ),

            "nonce_bytes": (
                len(nonce)
            ),

            "nonce_must_not_repeat_with_same_master_key": (
                True
            ),

            "domain_labels": [
                "KS",
                "SUB",
                "PERM",
                "DIFF"
            ]
        },

        "t5_noref": {
            "module_file": (
                t5_module_path.name
            ),

            "trained": False,

            "uses_reference_genome": (
                False
            ),

            "model_seed": int(
                model_seed
            ),

            "model_fingerprint_sha256": (
                fingerprint
            ),

            "config": asdict(
                config
            ),

            "full_length_keystream_bases": (
                number_of_bases
            ),

            "generation_mode": (
                "counter_separated_"
                "chunked_full_length_"
                "keystream"
            )
        },

        "encryption": {
            "pipeline": [
                "dynamic_substitution",
                "key_dependent_permutation",
                "bidirectional_diffusion_Z4",
                "quaternary_DNA_XOR_masking"
            ],

            "spd_block_bases": (
                spd_block_bases
            ),

            "t5_chunk_bases": (
                t5_chunk_bases
            ),

            "number_of_spd_blocks": (
                number_of_spd_blocks
            ),

            "number_of_t5_chunks": (
                number_of_t5_chunks
            ),

            "xor_enabled": (
                xor_enabled
            ),

            "substitution_family_size": (
                len(
                    SUBSTITUTIONS
                )
            ),

            "substitution_usage_histogram": (
                substitution_histogram
            ),

            "self_check_enabled": (
                self_check
            ),

            "self_check_passed": (
                True
                if self_check
                else None
            )
        },

        "integrity": {
            "ciphertext_hmac_sha256": (
                authentication.hexdigest()
            ),

            "tag_external_to_dna_ciphertext": (
                True
            )
        },

        "performance": {
            "scope": {
                "t5_setup": (
                    "module load + "
                    "deterministic session "
                    "model construction"
                ),

                "key_generation": (
                    "all KS/SUB/PERM/DIFF "
                    "T5 outputs"
                ),

                "encryption_core": (
                    "substitution + "
                    "permutation + diffusion "
                    "+ DNA-XOR"
                ),

                "compute_total": (
                    "t5_setup + "
                    "key_generation + "
                    "encryption_core"
                ),

                "self_check": (
                    "separate; excluded "
                    "from encryption_core"
                ),

                "file_io": (
                    "reported separately"
                )
            },

            "io_read_seconds": (
                io_read_seconds
            ),

            "io_cipher_write_seconds": (
                io_cipher_write_seconds
            ),

            "t5_setup": asdict(
                setup_metric
            ),

            "key_generation": (
                keygen_accumulator
                .to_dict()
            ),

            "encryption_core": (
                encryption_accumulator
                .to_dict()
            ),

            "self_check": (
                self_check_accumulator
                .to_dict()
                if self_check
                else None
            ),

            "compute_total_seconds": (
                compute_total_seconds
            ),

            "compute_total_cpu_seconds": (
                total_compute_cpu_seconds
            ),

            "overall_peak_rss_mb": (
                max(peak_rss_values)
                if peak_rss_values
                else None
            ),

            "overall_peak_gpu_allocated_mb": (
                max(peak_gpu_values)
                if peak_gpu_values
                else None
            ),

            "key_generation_throughput_base_per_second": (
                keygen_throughput
            ),

            "key_generation_throughput_base_per_millisecond": (
                keygen_throughput
                / 1000.0
            ),

            "encryption_core_throughput_base_per_second": (
                encryption_throughput
            ),

            "encryption_core_throughput_base_per_millisecond": (
                encryption_throughput
                / 1000.0
            ),

            "end_to_end_compute_throughput_base_per_second": (
                end_to_end_throughput
            ),

            "end_to_end_compute_throughput_base_per_millisecond": (
                end_to_end_throughput
                / 1000.0
            )
        }
    }

    # Metadata yazımı
    metadata_write_start = (
        time.perf_counter()
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2
        )

        handle.write(
            "\n"
        )

    metadata_write_seconds = (
        time.perf_counter()
        - metadata_write_start
    )

    observed_total_seconds = (
        time.perf_counter()
        - observed_start
    )

    metadata[
        "performance"
    ][
        "io_metadata_write_seconds"
    ] = metadata_write_seconds

    metadata[
        "performance"
    ][
        "observed_total_wall_seconds"
    ] = observed_total_seconds

    metadata[
        "performance"
    ][
        "observed_total_throughput_base_per_second"
    ] = (
        number_of_bases
        / observed_total_seconds
        if observed_total_seconds > 0
        else float("inf")
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2
        )

        handle.write(
            "\n"
        )

    return metadata


# =============================================================================
# RAPOR
# =============================================================================

def format_optional(
    value: Optional[float],
    digits: int = 6
) -> str:

    return (
        "N/A"
        if value is None
        else f"{value:.{digits}f}"
    )


def print_report(
    metadata: dict,
    cipher_path: Path,
    metadata_path: Path
) -> None:

    performance = (
        metadata["performance"]
    )

    key_generation = (
        performance[
            "key_generation"
        ]
    )

    encryption = (
        performance[
            "encryption_core"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BMC T5-NREF DNA-SPD "
        "ŞİFRELEME TAMAMLANDI"
    )

    print(
        "=" * 80
    )

    print(
        f"Girdi dosyası                  : "
        f"{metadata['input']['plain_file']}"
    )

    print(
        f"Gerçek DNA bazı                : "
        f"{metadata['input']['canonical_plain_bases']:,}"
    )

    print(
        f"Master key dosyası             : "
        f"{metadata['master_key']['file']}"
    )

    print(
        f"Master key yeni oluşturuldu    : "
        f"{metadata['master_key']['created_this_run']}"
    )

    print(
        f"Nonce                           : "
        f"{metadata['session']['nonce_hex']}"
    )

    print(
        "\n--- SÜRE / CPU ---"
    )

    print(
        f"T5 setup wall                  : "
        f"{performance['t5_setup']['wall_seconds']:.6f} s"
    )

    print(
        f"T5 setup CPU                   : "
        f"{format_optional(performance['t5_setup']['cpu_seconds'])} s"
    )

    print(
        f"T5 key/control generation wall : "
        f"{key_generation['wall_seconds']:.6f} s"
    )

    print(
        f"T5 key/control generation CPU  : "
        f"{format_optional(key_generation['cpu_seconds'])} s"
    )

    print(
        f"Encryption core wall           : "
        f"{encryption['wall_seconds']:.6f} s"
    )

    print(
        f"Encryption core wall           : "
        f"{encryption['wall_seconds'] * 1000.0:.6f} ms"
    )

    print(
        f"Encryption core CPU            : "
        f"{format_optional(encryption['cpu_seconds'])} s"
    )

    print(
        f"Compute total                  : "
        f"{performance['compute_total_seconds']:.6f} s"
    )

    print(
        f"Compute total CPU              : "
        f"{format_optional(performance['compute_total_cpu_seconds'])} s"
    )

    print(
        "\n--- RAM / GPU ---"
    )

    print(
        f"T5 keygen peak RSS             : "
        f"{format_optional(key_generation['max_peak_rss_mb'], 3)} MB"
    )

    print(
        f"Encryption peak RSS            : "
        f"{format_optional(encryption['max_peak_rss_mb'], 3)} MB"
    )

    print(
        f"Overall peak RSS               : "
        f"{format_optional(performance['overall_peak_rss_mb'], 3)} MB"
    )

    print(
        f"Overall peak GPU allocated     : "
        f"{format_optional(performance['overall_peak_gpu_allocated_mb'], 3)} MB"
    )

    print(
        "\n--- VERİMLİLİK ---"
    )

    print(
        f"KeyGen throughput              : "
        f"{performance['key_generation_throughput_base_per_second']:.2f} base/s"
    )

    print(
        f"Encryption throughput          : "
        f"{performance['encryption_core_throughput_base_per_second']:.2f} base/s"
    )

    print(
        f"Encryption throughput          : "
        f"{performance['encryption_core_throughput_base_per_millisecond']:.6f} base/ms"
    )

    print(
        f"End-to-end compute throughput  : "
        f"{performance['end_to_end_compute_throughput_base_per_second']:.2f} base/s"
    )

    print(
        f"Observed total incl. I/O/check : "
        f"{performance['observed_total_wall_seconds']:.6f} s"
    )

    if (
        metadata[
            "encryption"
        ][
            "self_check_enabled"
        ]
    ):

        print(
            f"Self-check wall (ayrı)         : "
            f"{performance['self_check']['wall_seconds']:.6f} s"
        )

        print(
            f"Self-check sonucu              : "
            f"{metadata['encryption']['self_check_passed']}"
        )

    print(
        "\n--- ÇIKTILAR ---"
    )

    print(
        f"Ciphertext                     : "
        f"{cipher_path}"
    )

    print(
        f"Metadata                       : "
        f"{metadata_path}"
    )

    print(
        "=" * 80
    )


# =============================================================================
# DOĞRUDAN ÇALIŞTIRMA
# =============================================================================

if __name__ == "__main__":

    plain_path = (
        BASE_DIR
        / INPUT_FILENAME
    )

    master_key_path = (
        BASE_DIR
        / MASTER_KEY_FILENAME
    )

    t5_module_path = (
        BASE_DIR
        / T5_MODULE_FILENAME
    )

    cipher_path = (
        BASE_DIR
        / CIPHER_FILENAME
    )

    metadata_path = (
        BASE_DIR
        / METADATA_FILENAME
    )

    print(
        f"[BASE_DIR]   {BASE_DIR}"
    )

    print(
        f"[PLAIN]      {plain_path}"
    )

    print(
        f"[T5_MODULE]  {t5_module_path}"
    )

    print(
        f"[MASTER_KEY] {master_key_path}"
    )

    print(
        f"[DEVICE]     {DEVICE}"
    )

    print(
        f"[SELF_CHECK] {SELF_CHECK}"
    )

    metadata = encrypt_genome(
        plain_path=plain_path,

        master_key_path=(
            master_key_path
        ),

        t5_module_path=(
            t5_module_path
        ),

        cipher_path=(
            cipher_path
        ),

        metadata_path=(
            metadata_path
        ),

        spd_block_bases=(
            SPD_BLOCK_BASES
        ),

        t5_chunk_bases=(
            T5_CHUNK_BASES
        ),

        device=DEVICE,

        xor_enabled=(
            XOR_ENABLED
        ),

        self_check=(
            SELF_CHECK
        ),

        overwrite=(
            OVERWRITE
        )
    )

    print_report(
        metadata,
        cipher_path,
        metadata_path
    )