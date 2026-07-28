# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:07:46 2026

@author: Alev Kaya
"""

# -*- coding: utf-8 -*-

"""
T5_train_soft_balance.py
R1 / R1-ext / R2 için T5-inspired encoder-decoder DNA-PRNG.

- Her koşumda 25 adet 1.000-baz segment seçilir: 20 eğitim, 5 doğrulama.
- Eğitim pencereleri segment sınırlarını geçmez; model 128 bazlık blokları paralel üretir.
- 500.000 DNA bazı ve 1.000.000 bit üretilir.
- Sekiz dinamik DNA-bit kodlama kuralı kullanılır.
- SHAKE kullanılmaz; ham model çıktısı değerlendirilir.
- Örnekleme, eğitim, DataLoader ve üretim tohumları RUN_SEED'den
  ayrı ayrı türetilir.
- Eski remaining/steps_left tam-kota dengelemesi kullanılmaz.
- Bunun yerine sonlu kayan pencere + zayıf global geri beslemeli,
  üstten sınırlandırılmış yumuşak baz dengelemesi uygulanır.

R1-ext ve R2 için yalnızca REAL_PATH ve OUT_TAG değerlerini değiştirin.
"""

import hashlib
import json
import math
import os
import random
import re
import time
import zlib
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


try:
    import psutil

    HAVE_PSUTIL = True

except Exception:
    psutil = None
    HAVE_PSUTIL = False


try:
    from scipy.stats import binomtest, chisquare

    HAVE_SCIPY = True

except Exception:
    HAVE_SCIPY = False


# =============================================================================
# AYARLAR
# =============================================================================

try:
    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

except NameError:
    BASE_DIR = os.getcwd()


REAL_PATH = os.path.join(
    BASE_DIR,
    "real-ext_dna_1m.txt"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

OUT_TAG = (
    "t5_real_softbalance_v1_trimer_bigram"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


TARGET_BITS = 1_000_000
TARGET_BASES = TARGET_BITS // 2

EPOCHS = 1
BATCH = 256
LR = 3e-4


# Kontrollü ilk koşum için sabit seed.
RUN_SEED: Optional[int] = None


SAMPLE_MODE = "scattered"
SCATTER_CHUNK_LEN = 1000
SCATTER_NUM_CHUNKS = 25


# T5-inspired blok-paralel encoder-decoder
SOURCE_LEN = 128
TARGET_BLOCK = 128

D_MODEL = 128
N_HEAD = 4

N_ENCODER_LAYER = 3
N_DECODER_LAYER = 3

D_FF = 512
DROPOUT = 0.10


# =============================================================================
# KOTASIZ, SINIRLANDIRILMIŞ YUMUŞAK DENGELEME
# =============================================================================

ENABLE_SOFT_BALANCE = True

GC_TARGET = 0.50

BALANCE_WINDOW = 4096
BALANCE_WARMUP = 512

LOCAL_BALANCE_GAIN = 4.0
LOCAL_BALANCE_CLAMP = 0.25

GLOBAL_BALANCE_GAIN = 3.0
GLOBAL_BALANCE_CLAMP = 0.10

TOTAL_BALANCE_CLAMP = 0.30


# =============================================================================
# DİĞER ÜRETİM KISITLARI
# =============================================================================

HOMOPOLYMER_MAX = 5

LAG1_DAMP = 0.030

RULE_TEMP = 3.9

ENABLE_TRIMER = True
TRIMER_ALPHA = 0.26
TRIMER_CLAMP = 0.40
TRIMER_WARMUP = 1280

ENABLE_BIGRAM = True
BIGRAM_ALPHA = 0.06
BIGRAM_CLAMP = 0.25
BIGRAM_WARMUP = 512


TAIL_WINDOWS = [
    128,
    256,
    512,
    1000,
    5000,
    10000,
]


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if torch.cuda.is_available():
    torch.set_float32_matmul_precision(
        "high"
    )


# =============================================================================
# DNA VE SEKİZ KODLAMA KURALI
# =============================================================================

DNA = [
    "A",
    "C",
    "G",
    "T",
]


VOCAB = {
    base: index
    for index, base in enumerate(DNA)
}


INV_VOCAB = {
    index: base
    for base, index in VOCAB.items()
}


MASK_ID = 4
VOCAB_SIZE = 5


ENC_RULES: List[Dict[str, str]] = [
    {
        "A": "00",
        "C": "01",
        "G": "10",
        "T": "11",
    },
    {
        "A": "11",
        "C": "01",
        "G": "10",
        "T": "00",
    },
    {
        "A": "00",
        "C": "10",
        "G": "01",
        "T": "11",
    },
    {
        "A": "11",
        "C": "10",
        "G": "01",
        "T": "00",
    },
    {
        "A": "01",
        "C": "00",
        "G": "11",
        "T": "10",
    },
    {
        "A": "10",
        "C": "00",
        "G": "11",
        "T": "01",
    },
    {
        "A": "01",
        "C": "11",
        "G": "00",
        "T": "10",
    },
    {
        "A": "10",
        "C": "11",
        "G": "00",
        "T": "01",
    },
]


RULE_MATS = np.zeros(
    (
        8,
        4,
        2,
    ),
    dtype=np.float32,
)


for rule_id, rule in enumerate(
    ENC_RULES
):
    for base, bits in rule.items():

        base_id = VOCAB[
            base
        ]

        RULE_MATS[
            rule_id,
            base_id,
            0,
        ] = float(
            bits[0] == "1"
        )

        RULE_MATS[
            rule_id,
            base_id,
            1,
        ] = float(
            bits[1] == "1"
        )


RULE_MATS_T = torch.tensor(
    RULE_MATS,
    dtype=torch.float32,
)


# =============================================================================
# TOHUMLAMA
# =============================================================================

MAX_TORCH_SEED = (
    (1 << 63) - 1
)


def derive_seed(
    master_seed: int,
    label: str,
) -> int:

    payload = (
        f"{int(master_seed)}::{label}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        seed
        & MAX_TORCH_SEED
    )


def set_training_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed % (2**32)
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def make_generator(
    seed: int,
    device: str = "cpu",
) -> torch.Generator:

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        int(seed)
    )

    return generator


# =============================================================================
# VERİ OKUMA VE SEGMENT ÖRNEKLEME
# =============================================================================

def read_fasta_or_txt(
    path: str,
) -> str:

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
            and not line.startswith(">")
        ]

    sequence = "".join(
        lines
    ).upper()

    sequence = "".join(
        character
        for character in sequence
        if character in VOCAB
    )

    if not sequence:
        raise ValueError(
            "Dosyada geçerli A/C/G/T dizisi bulunamadı."
        )

    return sequence


def sample_25k_segments(
    full_sequence: str,
    mode: str,
    generator: torch.Generator,
) -> Tuple[
    List[int],
    List[str],
]:

    required_length = 25_000

    if len(full_sequence) < required_length:
        raise ValueError(
            f"Kaynak en az {required_length:,} baz olmalıdır; "
            f"mevcut uzunluk={len(full_sequence):,}."
        )

    mode = mode.lower().strip()

    if mode == "contiguous":

        start = int(
            torch.randint(
                0,
                (
                    len(full_sequence)
                    - required_length
                    + 1
                ),
                (1,),
                generator=generator,
            ).item()
        )

        starts = [
            (
                start
                + chunk_index
                * SCATTER_CHUNK_LEN
            )
            for chunk_index in range(
                SCATTER_NUM_CHUNKS
            )
        ]

        segments = [
            full_sequence[
                segment_start:
                segment_start
                + SCATTER_CHUNK_LEN
            ]
            for segment_start in starts
        ]

        print(
            f"[SAMPLE] contiguous: "
            f"start={start}, "
            f"{SCATTER_NUM_CHUNKS} x "
            f"{SCATTER_CHUNK_LEN} = "
            f"{sum(len(segment) for segment in segments):,}"
        )

        return (
            starts,
            segments,
        )

    if mode != "scattered":
        raise ValueError(
            "SAMPLE_MODE yalnızca "
            "'scattered' veya "
            "'contiguous' olabilir."
        )

    if (
        SCATTER_CHUNK_LEN
        * SCATTER_NUM_CHUNKS
        != required_length
    ):
        raise ValueError(
            "SCATTER_CHUNK_LEN x "
            "SCATTER_NUM_CHUNKS değeri "
            "25.000 olmalıdır."
        )

    grid_starts = torch.arange(
        0,
        (
            len(full_sequence)
            - SCATTER_CHUNK_LEN
            + 1
        ),
        SCATTER_CHUNK_LEN,
    )

    if (
        grid_starts.numel()
        < SCATTER_NUM_CHUNKS
    ):
        raise ValueError(
            "Kaynak scattered örnekleme "
            "için yeterli değildir."
        )

    permutation = torch.randperm(
        grid_starts.numel(),
        generator=generator,
    )

    starts = [
        int(value)
        for value in (
            grid_starts[
                permutation[
                    :SCATTER_NUM_CHUNKS
                ]
            ]
            .sort()
            .values
            .tolist()
        )
    ]

    segments = [
        full_sequence[
            segment_start:
            segment_start
            + SCATTER_CHUNK_LEN
        ]
        for segment_start in starts
    ]

    if any(
        len(segment)
        != SCATTER_CHUNK_LEN
        for segment in segments
    ):
        raise RuntimeError(
            "Örneklenen segmentlerden biri "
            "eksik uzunluktadır."
        )

    print(
        f"[SAMPLE] scattered: "
        f"{SCATTER_NUM_CHUNKS} x "
        f"{SCATTER_CHUNK_LEN} = "
        f"{sum(len(segment) for segment in segments):,}"
    )

    return (
        starts,
        segments,
    )


# =============================================================================
# SEGMENT-GÜVENLİ DATASET
# =============================================================================

class SegmentedDNASeq2SeqDataset(
    Dataset
):

    def __init__(
        self,
        segments: Sequence[str],
        source_len: int,
        target_len: int,
    ):

        self.source_len = int(
            source_len
        )

        self.target_len = int(
            target_len
        )

        self.total_length = (
            self.source_len
            + self.target_len
        )

        self.segment_tokens: List[
            torch.Tensor
        ] = []

        self.index_map: List[
            Tuple[int, int]
        ] = []

        for (
            segment_index,
            segment,
        ) in enumerate(
            segments
        ):

            tokens = torch.tensor(
                [
                    VOCAB[base]
                    for base in segment
                ],
                dtype=torch.long,
            )

            if (
                len(tokens)
                < self.total_length
            ):
                raise ValueError(
                    "Bir segment source_len + "
                    "target_len değerinden kısadır."
                )

            self.segment_tokens.append(
                tokens
            )

            window_count = (
                len(tokens)
                - self.total_length
                + 1
            )

            for offset in range(
                window_count
            ):
                self.index_map.append(
                    (
                        segment_index,
                        offset,
                    )
                )

        if not self.index_map:
            raise ValueError(
                "Segment-güvenli dataset "
                "geçerli pencere üretmedi."
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.index_map
        )

    def __getitem__(
        self,
        index: int,
    ):

        (
            segment_index,
            offset,
        ) = self.index_map[
            index
        ]

        tokens = self.segment_tokens[
            segment_index
        ]

        source = tokens[
            offset:
            offset + self.source_len
        ]

        target = tokens[
            offset + self.source_len:
            offset + self.total_length
        ]

        decoder_query = torch.full(
            (
                self.target_len,
            ),
            MASK_ID,
            dtype=torch.long,
        )

        return (
            source,
            decoder_query,
            target,
        )


# =============================================================================
# T5-INSPIRED ENCODER-DECODER
# =============================================================================

@dataclass
class T5Config:

    vocab_size: int = VOCAB_SIZE

    d_model: int = D_MODEL
    n_head: int = N_HEAD

    n_encoder_layer: int = (
        N_ENCODER_LAYER
    )

    n_decoder_layer: int = (
        N_DECODER_LAYER
    )

    d_ff: int = D_FF

    source_len: int = SOURCE_LEN
    target_len: int = TARGET_BLOCK

    dropout: float = DROPOUT

    lambda_rule: float = 0.2
    beta_uniform: float = 0.2
    tau_rule: float = 0.25


class T5DNA(
    nn.Module
):

    def __init__(
        self,
        config: T5Config,
    ):

        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.source_position_embedding = (
            nn.Embedding(
                config.source_len,
                config.d_model,
            )
        )

        self.target_position_embedding = (
            nn.Embedding(
                config.target_len,
                config.d_model,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        decoder_layer = (
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=(
                config.n_encoder_layer
            ),
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=(
                config.n_decoder_layer
            ),
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.base_head = nn.Linear(
            config.d_model,
            4,
            bias=False,
        )

        self.rule_head = nn.Linear(
            config.d_model,
            8,
            bias=False,
        )

    def forward(
        self,
        source,
        decoder_query,
        targets=None,
    ):

        batch_size = source.size(
            0
        )

        source_positions = (
            torch.arange(
                self.config.source_len,
                device=source.device,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        target_positions = (
            torch.arange(
                self.config.target_len,
                device=source.device,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        source_hidden = (
            self.token_embedding(
                source
            )
            + self.source_position_embedding(
                source_positions
            )
        )

        decoder_hidden = (
            self.token_embedding(
                decoder_query
            )
            + self.target_position_embedding(
                target_positions
            )
        )

        encoder_memory = self.encoder(
            source_hidden
        )

        hidden_states = self.decoder(
            decoder_hidden,
            encoder_memory,
        )

        hidden_states = self.final_norm(
            hidden_states
        )

        base_logits = self.base_head(
            hidden_states
        )

        rule_logits = self.rule_head(
            hidden_states
        )

        loss = None

        if targets is not None:

            base_loss = F.cross_entropy(
                base_logits.reshape(
                    -1,
                    4,
                ),
                targets.reshape(
                    -1
                ),
            )

            with torch.no_grad():

                base_probabilities = (
                    torch.softmax(
                        base_logits,
                        dim=-1,
                    )
                )

                rule_matrices = (
                    RULE_MATS_T.to(
                        base_probabilities.device
                    )
                )

                expected_bits = torch.einsum(
                    "btk,rkj->btrj",
                    base_probabilities,
                    rule_matrices,
                )

                scores = (
                    1.0
                    - (
                        torch.abs(
                            expected_bits[
                                ...,
                                0,
                            ]
                            - 0.5
                        )
                        + torch.abs(
                            expected_bits[
                                ...,
                                1,
                            ]
                            - 0.5
                        )
                    )
                    / 2.0
                )

                rule_targets = torch.softmax(
                    (
                        scores
                        / self.config.tau_rule
                    ),
                    dim=-1,
                )

            rule_log_probabilities = (
                torch.log_softmax(
                    rule_logits,
                    dim=-1,
                )
            )

            rule_cross_entropy = -(
                rule_targets
                * rule_log_probabilities
            ).sum(
                dim=-1
            ).mean()

            rule_probabilities = (
                torch.softmax(
                    rule_logits,
                    dim=-1,
                )
            )

            uniform_probabilities = (
                torch.full_like(
                    rule_probabilities,
                    1.0 / 8.0,
                )
            )

            uniform_kl = (
                rule_probabilities
                * (
                    torch.log(
                        rule_probabilities
                        + 1e-12
                    )
                    - torch.log(
                        uniform_probabilities
                    )
                )
            ).sum(
                dim=-1
            ).mean()

            loss = (
                base_loss
                + self.config.lambda_rule
                * rule_cross_entropy
                + self.config.beta_uniform
                * uniform_kl
            )

        return (
            base_logits,
            rule_logits,
            loss,
        )


# =============================================================================
# EĞİTİM
# =============================================================================

@dataclass
class TrainOutput:

    model: T5DNA
    validation_bits_per_base: float


def calculate_validation_bits_per_base(
    model: T5DNA,
    validation_loader: DataLoader,
    device: str,
) -> float:

    model.eval()

    negative_log_likelihood = 0.0
    number_of_tokens = 0

    with torch.no_grad():

        for (
            source,
            decoder_query,
            target,
        ) in validation_loader:

            source = source.to(
                device
            )

            decoder_query = decoder_query.to(
                device
            )

            target = target.to(
                device
            )

            (
                base_logits,
                _,
                _,
            ) = model(
                source,
                decoder_query,
            )

            loss = F.cross_entropy(
                base_logits.reshape(
                    -1,
                    4,
                ),
                target.reshape(
                    -1
                ),
                reduction="sum",
            )

            negative_log_likelihood += (
                loss.item()
            )

            number_of_tokens += (
                target.numel()
            )

    return float(
        (
            negative_log_likelihood
            / max(
                1,
                number_of_tokens,
            )
        )
        / math.log(2)
    )


def train_model(
    train_segments: Sequence[str],
    validation_segments: Sequence[str],
    config: T5Config,
    device: str,
    loader_seed: int,
) -> TrainOutput:

    model = T5DNA(
        config
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    train_dataset = (
        SegmentedDNASeq2SeqDataset(
            train_segments,
            config.source_len,
            config.target_len,
        )
    )

    validation_dataset = (
        SegmentedDNASeq2SeqDataset(
            validation_segments,
            config.source_len,
            config.target_len,
        )
    )

    if (
        len(train_dataset) == 0
        or len(validation_dataset) == 0
    ):
        raise ValueError(
            "Eğitim veya doğrulama "
            "veri kümesi boş oluştu."
        )

    print(
        f"[WINDOWS] "
        f"train={len(train_dataset):,}  "
        f"validation={len(validation_dataset):,}"
    )

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        loader_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=loader_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    best_validation_bits_per_base = (
        float("inf")
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"T5 train epoch "
                f"{epoch + 1}"
            ),
        )

        for (
            source,
            decoder_query,
            target,
        ) in progress_bar:

            source = source.to(
                device
            )

            decoder_query = decoder_query.to(
                device
            )

            target = target.to(
                device
            )

            (
                _,
                _,
                loss,
            ) = model(
                source,
                decoder_query,
                target,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            progress_bar.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )

        validation_bits_per_base = (
            calculate_validation_bits_per_base(
                model,
                validation_loader,
                device,
            )
        )

        best_validation_bits_per_base = min(
            best_validation_bits_per_base,
            validation_bits_per_base,
        )

        print(
            f"[VAL] bits/base="
            f"{validation_bits_per_base:.6f}, "
            f"best="
            f"{best_validation_bits_per_base:.6f}"
        )

    return TrainOutput(
        model=model,
        validation_bits_per_base=(
            best_validation_bits_per_base
        ),
    )


# =============================================================================
# ÜRETİM YARDIMCILARI
# =============================================================================

def calculate_target_probabilities(
    gc_target: float,
) -> Dict[str, float]:

    if not (
        0.0
        <= gc_target
        <= 1.0
    ):
        raise ValueError(
            "GC_TARGET 0 ile 1 arasında olmalıdır."
        )

    return {
        "A": (
            1.0 - gc_target
        ) / 2.0,
        "C": (
            gc_target
        ) / 2.0,
        "G": (
            gc_target
        ) / 2.0,
        "T": (
            1.0 - gc_target
        ) / 2.0,
    }


def calculate_soft_balance_bias(
    recent_counts: torch.Tensor,
    recent_length: int,
    global_counts: torch.Tensor,
    generated_length: int,
    target_vector: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:

    zero_bias = torch.zeros(
        4,
        device=target_vector.device,
        dtype=dtype,
    )

    if not ENABLE_SOFT_BALANCE:
        return zero_bias

    if (
        generated_length
        < BALANCE_WARMUP
        or recent_length <= 0
    ):
        return zero_bias

    local_rates = (
        recent_counts.to(
            dtype=dtype
        )
        / float(
            recent_length
        )
    )

    local_error = (
        target_vector.to(
            dtype=dtype
        )
        - local_rates
    )

    local_bias = torch.clamp(
        (
            LOCAL_BALANCE_GAIN
            * local_error
        ),
        min=-LOCAL_BALANCE_CLAMP,
        max=LOCAL_BALANCE_CLAMP,
    )

    if generated_length > 0:

        global_rates = (
            global_counts.to(
                dtype=dtype
            )
            / float(
                generated_length
            )
        )

        global_error = (
            target_vector.to(
                dtype=dtype
            )
            - global_rates
        )

        global_bias = torch.clamp(
            (
                GLOBAL_BALANCE_GAIN
                * global_error
            ),
            min=-GLOBAL_BALANCE_CLAMP,
            max=GLOBAL_BALANCE_CLAMP,
        )

    else:
        global_bias = zero_bias

    total_bias = torch.clamp(
        (
            local_bias
            + global_bias
        ),
        min=-TOTAL_BALANCE_CLAMP,
        max=TOTAL_BALANCE_CLAMP,
    )

    return total_bias


# =============================================================================
# DNA ÜRETİMİ
# =============================================================================

@torch.inference_mode()
def generate_dna(
    model: T5DNA,
    start_tokens: torch.LongTensor,
    output_length: int,
    device: str,
    generator: torch.Generator,
) -> Tuple[
    str,
    List[int],
    str,
]:

    model.eval()

    config = model.config

    context = start_tokens.to(
        device
    )

    dna_output: List[str] = []
    rule_output: List[int] = []
    bit_output: List[str] = []

    target = calculate_target_probabilities(
        GC_TARGET
    )

    target_vector = torch.tensor(
        [
            target[base]
            for base in DNA
        ],
        device=device,
        dtype=torch.float32,
    )

    global_counts = torch.zeros(
        4,
        dtype=torch.long,
        device=device,
    )

    recent_counts = torch.zeros(
        4,
        dtype=torch.long,
        device=device,
    )

    recent_window: Deque[int] = (
        deque()
    )

    last_base: Optional[int] = None
    run_length = 0

    previous_1: Optional[int] = None
    previous_2: Optional[int] = None

    bigram_counts = torch.zeros(
        16,
        dtype=torch.long,
        device=device,
    )

    trigram_counts = torch.zeros(
        64,
        dtype=torch.long,
        device=device,
    )

    number_of_bigrams = 0
    number_of_trigrams = 0

    epsilon = 1e-6

    rule_matrices = RULE_MATS_T.to(
        device
    )

    progress_bar = tqdm(
        total=output_length,
        desc="T5 generate",
        unit="base",
    )

    while (
        len(dna_output)
        < output_length
    ):

        block_length = min(
            config.target_len,
            (
                output_length
                - len(dna_output)
            ),
        )

        decoder_query = torch.full(
            (
                1,
                config.target_len,
            ),
            MASK_ID,
            dtype=torch.long,
            device=device,
        )

        (
            base_logits_block,
            rule_logits_block,
            _,
        ) = model(
            context,
            decoder_query,
        )

        new_block: List[int] = []

        for position in range(
            block_length
        ):

            step = len(
                dna_output
            )

            log_probabilities = (
                torch.log_softmax(
                    base_logits_block[
                        0,
                        position,
                    ],
                    dim=-1,
                )
            )

            balance_bias = (
                calculate_soft_balance_bias(
                    recent_counts=(
                        recent_counts
                    ),
                    recent_length=(
                        len(
                            recent_window
                        )
                    ),
                    global_counts=(
                        global_counts
                    ),
                    generated_length=(
                        step
                    ),
                    target_vector=(
                        target_vector
                    ),
                    dtype=(
                        log_probabilities.dtype
                    ),
                )
            )

            log_probabilities = (
                log_probabilities
                + balance_bias
            )

            if (
                last_base is not None
                and LAG1_DAMP > 0.0
            ):
                log_probabilities[
                    last_base
                ] += math.log(
                    max(
                        epsilon,
                        (
                            1.0
                            - LAG1_DAMP
                        ),
                    )
                )

            if (
                ENABLE_TRIMER
                and previous_2 is not None
                and previous_1 is not None
                and step >= TRIMER_WARMUP
            ):

                expected = max(
                    1.0,
                    (
                        number_of_trigrams
                        / 64.0
                    ),
                )

                for base_id in range(
                    4
                ):

                    index = (
                        previous_2 << 4
                        | previous_1 << 2
                        | base_id
                    )

                    observed_ratio = (
                        (
                            float(
                                trigram_counts[
                                    index
                                ].item()
                            )
                            + 1.0
                        )
                        / (
                            expected
                            + 1.0
                        )
                    )

                    bias = (
                        -TRIMER_ALPHA
                        * math.log(
                            max(
                                epsilon,
                                observed_ratio,
                            )
                        )
                    )

                    log_probabilities[
                        base_id
                    ] += max(
                        -TRIMER_CLAMP,
                        min(
                            TRIMER_CLAMP,
                            bias,
                        ),
                    )

            if (
                ENABLE_BIGRAM
                and previous_1 is not None
                and step >= BIGRAM_WARMUP
            ):

                expected = max(
                    1.0,
                    (
                        number_of_bigrams
                        / 16.0
                    ),
                )

                for base_id in range(
                    4
                ):

                    index = (
                        previous_1 << 2
                        | base_id
                    )

                    observed_ratio = (
                        (
                            float(
                                bigram_counts[
                                    index
                                ].item()
                            )
                            + 1.0
                        )
                        / (
                            expected
                            + 1.0
                        )
                    )

                    bias = (
                        -BIGRAM_ALPHA
                        * math.log(
                            max(
                                epsilon,
                                observed_ratio,
                            )
                        )
                    )

                    log_probabilities[
                        base_id
                    ] += max(
                        -BIGRAM_CLAMP,
                        min(
                            BIGRAM_CLAMP,
                            bias,
                        ),
                    )

            if (
                HOMOPOLYMER_MAX
                is not None
                and last_base is not None
                and run_length
                >= HOMOPOLYMER_MAX
            ):
                log_probabilities[
                    last_base
                ] = -1e9

            base_probabilities = (
                torch.softmax(
                    log_probabilities,
                    dim=-1,
                )
            )

            if (
                not torch.isfinite(
                    base_probabilities
                ).all()
                or float(
                    base_probabilities
                    .sum()
                    .item()
                )
                <= 0.0
            ):
                raise RuntimeError(
                    f"Geçersiz baz olasılığı "
                    f"oluştu; step={step}."
                )

            base_id = int(
                torch.multinomial(
                    base_probabilities,
                    1,
                    generator=generator,
                ).item()
            )

            expected_bits = torch.einsum(
                "f,rfj->rj",
                base_probabilities,
                rule_matrices,
            )

            scores = (
                1.0
                - (
                    torch.abs(
                        expected_bits[
                            :,
                            0,
                        ]
                        - 0.5
                    )
                    + torch.abs(
                        expected_bits[
                            :,
                            1,
                        ]
                        - 0.5
                    )
                )
                / 2.0
            )

            rule_probabilities = (
                torch.softmax(
                    (
                        rule_logits_block[
                            0,
                            position,
                        ]
                        / RULE_TEMP
                        + scores
                    ),
                    dim=-1,
                )
            )

            rule_id = int(
                torch.multinomial(
                    rule_probabilities,
                    1,
                    generator=generator,
                ).item()
            )

            base_character = (
                INV_VOCAB[
                    base_id
                ]
            )

            dna_output.append(
                base_character
            )

            rule_output.append(
                rule_id
            )

            bit_output.append(
                ENC_RULES[
                    rule_id
                ][
                    base_character
                ]
            )

            new_block.append(
                base_id
            )

            global_counts[
                base_id
            ] += 1

            if (
                len(recent_window)
                >= BALANCE_WINDOW
            ):

                removed_base = (
                    recent_window.popleft()
                )

                recent_counts[
                    removed_base
                ] -= 1

            recent_window.append(
                base_id
            )

            recent_counts[
                base_id
            ] += 1

            if previous_1 is not None:

                bigram_index = (
                    previous_1 << 2
                    | base_id
                )

                bigram_counts[
                    bigram_index
                ] += 1

                number_of_bigrams += 1

            if (
                previous_2 is not None
                and previous_1 is not None
            ):

                trigram_index = (
                    previous_2 << 4
                    | previous_1 << 2
                    | base_id
                )

                trigram_counts[
                    trigram_index
                ] += 1

                number_of_trigrams += 1

            if (
                last_base is None
                or base_id != last_base
            ):

                last_base = base_id
                run_length = 1

            else:
                run_length += 1

            previous_2 = previous_1
            previous_1 = base_id

            progress_bar.update(
                1
            )

        new_block_tensor = torch.tensor(
            new_block,
            dtype=torch.long,
            device=device,
        ).unsqueeze(
            0
        )

        context = torch.cat(
            [
                context,
                new_block_tensor,
            ],
            dim=1,
        )[
            :,
            -config.source_len:
        ]

    progress_bar.close()

    dna_sequence = "".join(
        dna_output
    )

    bit_sequence = "".join(
        bit_output
    )

    if (
        len(dna_sequence)
        != output_length
    ):
        raise RuntimeError(
            f"DNA uzunluğu hatalı: "
            f"{len(dna_sequence):,}."
        )

    if (
        len(bit_sequence)
        != output_length * 2
    ):
        raise RuntimeError(
            f"Bit uzunluğu hatalı: "
            f"{len(bit_sequence):,}."
        )

    return (
        dna_sequence,
        rule_output,
        bit_sequence,
    )


# =============================================================================
# ANALİZ
# =============================================================================

def analyze_generated_output(
    dna_sequence: str,
    bit_sequence: str,
) -> Dict[str, object]:

    counts = {
        base: dna_sequence.count(
            base
        )
        for base in DNA
    }

    number_of_bases = len(
        dna_sequence
    )

    entropy = -sum(
        (
            counts[base]
            / number_of_bases
        )
        * math.log2(
            counts[base]
            / number_of_bases
        )
        for base in DNA
        if counts[base] > 0
    )

    results: Dict[
        str,
        object,
    ] = {
        "len_bases": number_of_bases,
        "len_bits": len(bit_sequence),
        "counts": counts,
        "entropy_bits_per_base": entropy,
        "p_one": (
            bit_sequence.count("1")
            / len(bit_sequence)
        ),
    }

    if HAVE_SCIPY:

        results[
            "monobit_p"
        ] = float(
            binomtest(
                bit_sequence.count(
                    "1"
                ),
                len(
                    bit_sequence
                ),
                p=0.5,
            ).pvalue
        )

        results[
            "chi2_p"
        ] = float(
            chisquare(
                [
                    counts[base]
                    for base in DNA
                ],
                [
                    number_of_bases
                    / 4.0
                ]
                * 4,
            ).pvalue
        )

    else:
        results["monobit_p"] = None
        results["chi2_p"] = None

    return results


def calculate_tail_summary(
    dna_sequence: str,
    windows: List[int],
) -> Dict[
    str,
    Dict[str, object],
]:

    summary: Dict[
        str,
        Dict[str, object],
    ] = {}

    for window_size in windows:

        actual_size = min(
            window_size,
            len(dna_sequence),
        )

        if actual_size <= 0:
            continue

        segment = dna_sequence[
            -actual_size:
        ]

        counts = {
            base: segment.count(
                base
            )
            for base in DNA
        }

        rates = {
            base: (
                counts[base]
                / actual_size
            )
            for base in DNA
        }

        summary[
            str(window_size)
        ] = {
            "actual_size": actual_size,
            "counts": counts,
            "rates": rates,
            "gc_rate": (
                rates["C"]
                + rates["G"]
            ),
        }

    return summary


def print_tail_summary(
    tail_summary: Dict[
        str,
        Dict[str, object],
    ],
) -> None:

    print(
        "\n--- TAIL BASE DISTRIBUTION ---"
    )

    for (
        window_key,
        item,
    ) in tail_summary.items():

        rates = item[
            "rates"
        ]

        print(
            f"last "
            f"{int(window_key):>6,} bases: "
            f"A={100 * rates['A']:.3f}%  "
            f"C={100 * rates['C']:.3f}%  "
            f"G={100 * rates['G']:.3f}%  "
            f"T={100 * rates['T']:.3f}%  "
            f"GC={100 * item['gc_rate']:.3f}%"
        )


def pack_bits_to_bytes(
    bit_sequence: str,
) -> bytes:

    output = bytearray()

    accumulator = 0
    number_of_bits = 0

    for character in bit_sequence:

        accumulator = (
            accumulator << 1
            | (
                1
                if character == "1"
                else 0
            )
        )

        number_of_bits += 1

        if number_of_bits == 8:

            output.append(
                accumulator
            )

            accumulator = 0
            number_of_bits = 0

    if number_of_bits > 0:

        output.append(
            accumulator
            << (
                8
                - number_of_bits
            )
        )

    return bytes(
        output
    )


# =============================================================================
# ANA AKIŞ
# =============================================================================

if __name__ == "__main__":

    print(
        "[DEVICE]",
        DEVICE,
    )

    print(
        "[OUT_DIR]",
        OUT_DIR,
    )

    print(
        "[BALANCE_MODE] "
        "bounded local-window "
        "+ weak global feedback"
    )

    seed_text = os.getenv(
        "RUN_SEED"
    )

    if (
        seed_text
        and seed_text.strip()
    ):

        try:
            run_seed = (
                int(
                    seed_text,
                    0,
                )
                & MAX_TORCH_SEED
            )

        except ValueError as error:
            raise ValueError(
                f"RUN_SEED geçerli bir "
                f"tam sayı değil: "
                f"{seed_text}"
            ) from error

    elif RUN_SEED is not None:

        run_seed = (
            int(RUN_SEED)
            & MAX_TORCH_SEED
        )

    else:

        run_seed = (
            int.from_bytes(
                os.urandom(8),
                byteorder="little",
                signed=False,
            )
            & MAX_TORCH_SEED
        )

    sample_seed = derive_seed(
        run_seed,
        "sampling",
    )

    training_seed = derive_seed(
        run_seed,
        "model_and_training",
    )

    loader_seed = derive_seed(
        run_seed,
        "dataloader",
    )

    generation_seed = derive_seed(
        run_seed,
        "generation",
    )

    print(
        "[RUN_SEED]",
        run_seed,
    )

    print(
        "[SAMPLE_SEED]",
        sample_seed,
    )

    print(
        "[TRAINING_SEED]",
        training_seed,
    )

    print(
        "[LOADER_SEED]",
        loader_seed,
    )

    print(
        "[GENERATION_SEED]",
        generation_seed,
    )

    sample_generator = make_generator(
        sample_seed,
        "cpu",
    )

    generation_generator = make_generator(
        generation_seed,
        DEVICE,
    )

    set_training_seed(
        training_seed
    )

    full_sequence = read_fasta_or_txt(
        REAL_PATH
    )

    if (
        len(full_sequence)
        < 100_000
    ):
        raise ValueError(
            f"Dosya çok kısa: "
            f"{REAL_PATH}; "
            f"uzunluk="
            f"{len(full_sequence):,}, "
            f"en az 100.000 olmalıdır."
        )

    (
        sampled_starts,
        sampled_segments,
    ) = sample_25k_segments(
        full_sequence,
        SAMPLE_MODE,
        sample_generator,
    )

    train_segments = (
        sampled_segments[
            :20
        ]
    )

    validation_segments = (
        sampled_segments[
            20:25
        ]
    )

    train_starts = (
        sampled_starts[
            :20
        ]
    )

    validation_starts = (
        sampled_starts[
            20:25
        ]
    )

    train_base_count = sum(
        len(segment)
        for segment in train_segments
    )

    validation_base_count = sum(
        len(segment)
        for segment in validation_segments
    )

    print(
        f"[SPLIT] "
        f"train={train_base_count:,} baz / "
        f"{len(train_segments)} segment, "
        f"val={validation_base_count:,} baz / "
        f"{len(validation_segments)} segment"
    )

    print(
        "[TRAIN_STARTS]",
        train_starts,
    )

    print(
        "[VAL_STARTS]",
        validation_starts,
    )

    config = T5Config()

    training_start = (
        time.perf_counter()
    )

    trained_model = train_model(
        train_segments,
        validation_segments,
        config,
        DEVICE,
        loader_seed,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    start_tokens = torch.randint(
        0,
        4,
        (
            1,
            config.source_len,
        ),
        generator=generation_generator,
        device=DEVICE,
    )

    if HAVE_PSUTIL:

        process = psutil.Process(
            os.getpid()
        )

        cpu_start = (
            process.cpu_times().user
            + process.cpu_times().system
        )

        ram_start = (
            process.memory_info().rss
        )

    else:

        process = None
        cpu_start = time.process_time()
        ram_start = None

    generation_start = (
        time.perf_counter()
    )

    (
        dna_sequence,
        rules,
        bit_sequence,
    ) = generate_dna(
        trained_model.model,
        start_tokens,
        TARGET_BASES,
        DEVICE,
        generation_generator,
    )

    generation_wall_seconds = (
        time.perf_counter()
        - generation_start
    )

    if (
        HAVE_PSUTIL
        and process is not None
    ):

        cpu_end = (
            process.cpu_times().user
            + process.cpu_times().system
        )

        generation_cpu_seconds = (
            cpu_end
            - cpu_start
        )

        ram_delta_mb = (
            (
                process.memory_info().rss
                - ram_start
            )
            / (
                1024
                * 1024
            )
        )

    else:

        generation_cpu_seconds = (
            time.process_time()
            - cpu_start
        )

        ram_delta_mb = None

    statistics = analyze_generated_output(
        dna_sequence,
        bit_sequence,
    )

    tail_summary = calculate_tail_summary(
        dna_sequence,
        TAIL_WINDOWS,
    )

    efficiency = (
        len(dna_sequence)
        / max(
            generation_wall_seconds,
            1e-12,
        )
    )

    maximum_homopolymer = max(
        (
            len(
                match.group(0)
            )
            for match in re.finditer(
                r"(A+|C+|G+|T+)",
                dna_sequence,
            )
        ),
        default=0,
    )

    rule_counts = Counter(
        rules
    )

    print(
        "\n--- T5 CORE PERFORMANCE ---"
    )

    print(
        f"Train time="
        f"{training_seconds:.3f}s  "
        f"Generate time="
        f"{generation_wall_seconds:.3f}s  "
        f"CPU="
        f"{generation_cpu_seconds:.3f}s"
    )

    ram_text = (
        f"{ram_delta_mb:.3f} MB"
        if ram_delta_mb is not None
        else "N/A"
    )

    print(
        f"RAM delta="
        f"{ram_text}  "
        f"Efficiency="
        f"{efficiency:.2f} base/s"
    )

    print(
        "\n--- ANALYSIS ---"
    )

    print(
        f"bases="
        f"{len(dna_sequence):,}  "
        f"bits="
        f"{len(bit_sequence):,}  "
        f"entropy="
        f"{statistics['entropy_bits_per_base']:.6f}"
    )

    print(
        f"p(1)="
        f"{statistics['p_one']:.6f}  "
        f"monobit_p="
        f"{statistics['monobit_p']}  "
        f"chi2_p="
        f"{statistics['chi2_p']}"
    )

    print(
        "counts=",
        statistics[
            "counts"
        ],
    )

    print(
        "max homopolymer=",
        maximum_homopolymer,
    )

    compression_ratio = (
        len(
            zlib.compress(
                bit_sequence.encode(
                    "ascii"
                ),
                9,
            )
        )
        / len(
            bit_sequence
        )
    )

    print(
        "compression ratio=",
        compression_ratio,
    )

    print_tail_summary(
        tail_summary
    )

    print(
        "\n--- RULE USAGE ---"
    )

    for rule_id in range(
        8
    ):
        print(
            f"R{rule_id}: "
            f"{rule_counts.get(rule_id, 0)}"
        )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    base_path = os.path.join(
        OUT_DIR,
        (
            f"{OUT_TAG}_"
            f"seed{run_seed}_"
            f"{timestamp}"
        ),
    )

    dna_path = (
        base_path
        + ".dna.txt"
    )

    bits_path = (
        base_path
        + ".bits.txt"
    )

    rules_path = (
        base_path
        + ".rules.txt"
    )

    metadata_path = (
        base_path
        + ".json"
    )

    with open(
        dna_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            dna_sequence
        )

    with open(
        bits_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            bit_sequence
        )

    with open(
        rules_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            " ".join(
                map(
                    str,
                    rules,
                )
            )
        )

    raw_bytes = pack_bits_to_bytes(
        bit_sequence
    )

    keystream_path = os.path.join(
        OUT_DIR,
        (
            f"t5_keystream_"
            f"seed{run_seed}_"
            f"{timestamp}.bin"
        ),
    )

    key_path = os.path.join(
        OUT_DIR,
        (
            f"t5_key_"
            f"seed{run_seed}_"
            f"{timestamp}.hex"
        ),
    )

    with open(
        keystream_path,
        "wb",
    ) as file:
        file.write(
            raw_bytes[
                :64 * 1024
            ]
        )

    with open(
        key_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            raw_bytes[
                :32
            ].hex()
            + "\n"
        )

    metadata = {
        "mode": (
            "t5_trained_real_"
            "constraint_aware_"
            "soft_balance"
        ),

        "architecture": (
            "T5-inspired "
            "non-autoregressive "
            "block-parallel "
            "encoder-decoder"
        ),

        "seeds": {
            "run_seed": int(
                run_seed
            ),
            "sample_seed": int(
                sample_seed
            ),
            "training_seed": int(
                training_seed
            ),
            "loader_seed": int(
                loader_seed
            ),
            "generation_seed": int(
                generation_seed
            ),
        },

        "sampling": {
            "mode": SAMPLE_MODE,
            "scatter_chunk_len": (
                SCATTER_CHUNK_LEN
            ),
            "scatter_num_chunks": (
                SCATTER_NUM_CHUNKS
            ),
            "train_bases": int(
                train_base_count
            ),
            "validation_bases": int(
                validation_base_count
            ),
            "train_segment_starts": [
                int(value)
                for value in train_starts
            ],
            "validation_segment_starts": [
                int(value)
                for value in validation_starts
            ],
            "segment_safe_windows": True,
        },

        "validation_bits_per_base": (
            trained_model
            .validation_bits_per_base
        ),

        "dataset_windows": {
            "train": int(
                len(
                    SegmentedDNASeq2SeqDataset(
                        train_segments,
                        config.source_len,
                        config.target_len,
                    )
                )
            ),
            "validation": int(
                len(
                    SegmentedDNASeq2SeqDataset(
                        validation_segments,
                        config.source_len,
                        config.target_len,
                    )
                )
            ),
        },

        "train_seconds": (
            training_seconds
        ),

        "generation_wall_seconds": (
            generation_wall_seconds
        ),

        "generation_cpu_seconds": (
            generation_cpu_seconds
        ),

        "ram_delta_mb": (
            ram_delta_mb
        ),

        "efficiency_base_per_second": (
            efficiency
        ),

        "rule_counts": {
            str(rule_id): (
                rule_counts.get(
                    rule_id,
                    0,
                )
            )
            for rule_id in range(
                8
            )
        },

        "analysis": statistics,

        "tail_summary": (
            tail_summary
        ),

        "model": asdict(
            config
        ),

        "data_integrity": {
            "segment_boundaries_preserved": True,
            "cross_segment_windows": False,
        },

        "constraints": {
            "balance_mode": (
                "bounded_local_window_"
                "plus_weak_global_feedback"
            ),
            "enable_soft_balance": (
                ENABLE_SOFT_BALANCE
            ),
            "gc_target": (
                GC_TARGET
            ),
            "balance_window": (
                BALANCE_WINDOW
            ),
            "balance_warmup": (
                BALANCE_WARMUP
            ),
            "local_balance_gain": (
                LOCAL_BALANCE_GAIN
            ),
            "local_balance_clamp": (
                LOCAL_BALANCE_CLAMP
            ),
            "global_balance_gain": (
                GLOBAL_BALANCE_GAIN
            ),
            "global_balance_clamp": (
                GLOBAL_BALANCE_CLAMP
            ),
            "total_balance_clamp": (
                TOTAL_BALANCE_CLAMP
            ),
            "homopolymer_max": (
                HOMOPOLYMER_MAX
            ),
            "lag1_damp": (
                LAG1_DAMP
            ),
            "enable_trimer": (
                ENABLE_TRIMER
            ),
            "trimer_alpha": (
                TRIMER_ALPHA
            ),
            "trimer_clamp": (
                TRIMER_CLAMP
            ),
            "trimer_warmup": (
                TRIMER_WARMUP
            ),
            "enable_bigram": (
                ENABLE_BIGRAM
            ),
            "bigram_alpha": (
                BIGRAM_ALPHA
            ),
            "bigram_clamp": (
                BIGRAM_CLAMP
            ),
            "bigram_warmup": (
                BIGRAM_WARMUP
            ),
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    fixed_dna_path = os.path.join(
        OUT_DIR,
        "t5_real_dna_500k.txt",
    )

    fixed_bits_path = os.path.join(
        OUT_DIR,
        "t5_real_dna_1m.txt",
    )

    with open(
        fixed_dna_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            dna_sequence
        )

    with open(
        fixed_bits_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            bit_sequence
        )

    print(
        f"\n[SAVED] dna       -> "
        f"{dna_path}"
    )

    print(
        f"[SAVED] bits      -> "
        f"{bits_path}"
    )

    print(
        f"[SAVED] rules     -> "
        f"{rules_path}"
    )

    print(
        f"[SAVED] meta      -> "
        f"{metadata_path}"
    )

    print(
        f"[SAVED] keystream -> "
        f"{keystream_path}"
    )

    print(
        f"[SAVED] key       -> "
        f"{key_path}"
    )

    print(
        f"[SAVED] fixed     -> "
        f"{fixed_dna_path}, "
        f"{fixed_bits_path}"
    )