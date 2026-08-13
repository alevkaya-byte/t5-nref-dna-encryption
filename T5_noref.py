import hashlib
import json
import math
import os
import random
import re
import time
import zlib

from array import array
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# =============================================================================
# İSTEĞE BAĞLI PAKETLER
# =============================================================================

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
# DETERMINİSTİK ÇALIŞMA
# =============================================================================

DETERMINISTIC = True
FORCE_CPU = False
DISABLE_TF32 = True

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if DISABLE_TF32:
    torch.set_float32_matmul_precision(
        "highest"
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

if torch.cuda.is_available():

    try:
        torch.backends.cuda.enable_flash_sdp(
            False
        )

        torch.backends.cuda.enable_mem_efficient_sdp(
            False
        )

        torch.backends.cuda.enable_math_sdp(
            True
        )

    except Exception:
        pass


# =============================================================================
# DOSYA YOLLARI
# =============================================================================

try:
    BASE_DIR = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

except NameError:
    BASE_DIR = os.getcwd()


OUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

OUT_TAG = (
    "t5_noref_v3_continuouspos_"
    "contextquery_multilag"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =============================================================================
# GENEL AYARLAR
# =============================================================================

TARGET_BITS = 1_000_000
TARGET_BASES = TARGET_BITS // 2

USE_SHAKE = False

KEYSTREAM_BYTES = 64 * 1024
KEY_BYTES = 32


# =============================================================================
# T5-INSPIRED MİMARİ
# =============================================================================

SOURCE_LEN = 128
TARGET_BLOCK = 128

D_MODEL = 128
N_HEAD = 4

N_ENCODER_LAYER = 3
N_DECODER_LAYER = 3

D_FF = 512
DROPOUT = 0.0


# =============================================================================
# BLOK LOGİT KALİBRASYONU
# =============================================================================

ENABLE_BLOCK_LOGIT_CALIBRATION = True

BLOCK_LOGIT_SCALE = 0.70
BLOCK_LOGIT_CLAMP = 1.25

MODEL_UNIFORM_MIX = 0.15


# =============================================================================
# KOTASIZ SINIRLANDIRILMIŞ YUMUŞAK DENGELEME
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
# DNA-YEREL KISITLAR
# =============================================================================

HOMOPOLYMER_MAX = 5


# =============================================================================
# KISA MENZİLLİ ÇOKLU-LAG DEKORELASYON
# =============================================================================
#
# Son 1-5 konumdaki aynı baz tercihlerini sınırlandırır.
# Baz kotası değildir ve kalan çıktı uzunluğuna bağlı değildir.
#

RECENT_LAG_DAMP = {
    1: 0.32,
    2: 0.25,
    3: 0.18,
    4: 0.14,
    5: 0.12
}

MAX_RECENT_LAG = max(
    RECENT_LAG_DAMP
)

# Logaritmik cezalar yalnızca bir kez hesaplanır.
# Bu yalnızca hız optimizasyonudur; yöntemi ve olasılıkları değiştirmez.
RECENT_LAG_LOG_PENALTIES = tuple(
    (
        lag,
        math.log(
            max(
                1e-6,
                1.0 - damp
            )
        )
    )
    for lag, damp in RECENT_LAG_DAMP.items()
)


# =============================================================================
# BIGRAM / TRIGRAM DÜZELTMESİ
# =============================================================================

ENABLE_TRIMER = True

TRIMER_ALPHA = 0.26
TRIMER_CLAMP = 0.40
TRIMER_WARMUP = 1280


ENABLE_BIGRAM = True

BIGRAM_ALPHA = 0.06
BIGRAM_CLAMP = 0.25
BIGRAM_WARMUP = 512


# =============================================================================
# DİNAMİK DNA-BİT KURALI
# =============================================================================

RULE_TEMP = 4.0
RECENT_RULE_WINDOW = 8


# =============================================================================
# RAPORLAMA
# =============================================================================

TAIL_WINDOWS = [
    128,
    256,
    512,
    1000,
    5000,
    10000
]


DEVICE = (
    "cpu"
    if FORCE_CPU
    else (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
)


# =============================================================================
# DNA ALFABESİ
# =============================================================================

DNA = [
    "A",
    "C",
    "G",
    "T"
]

VOCAB = {
    base: index
    for index, base in enumerate(
        DNA
    )
}

INV_VOCAB = {
    index: base
    for base, index in VOCAB.items()
}

VOCAB_SIZE = 4


# =============================================================================
# SEKİZ DNA-BİT KODLAMA KURALI
# =============================================================================

ENC_RULES: List[
    Dict[str, str]
] = [
    {
        "A": "00",
        "C": "01",
        "G": "10",
        "T": "11"
    },
    {
        "A": "11",
        "C": "01",
        "G": "10",
        "T": "00"
    },
    {
        "A": "00",
        "C": "10",
        "G": "01",
        "T": "11"
    },
    {
        "A": "11",
        "C": "10",
        "G": "01",
        "T": "00"
    },
    {
        "A": "01",
        "C": "00",
        "G": "11",
        "T": "10"
    },
    {
        "A": "10",
        "C": "00",
        "G": "11",
        "T": "01"
    },
    {
        "A": "01",
        "C": "11",
        "G": "00",
        "T": "10"
    },
    {
        "A": "10",
        "C": "11",
        "G": "00",
        "T": "01"
    }
]


RULE_MATS = np.zeros(
    (
        8,
        4,
        2
    ),
    dtype=np.float32
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
            0
        ] = float(
            bits[0] == "1"
        )

        RULE_MATS[
            rule_id,
            base_id,
            1
        ] = float(
            bits[1] == "1"
        )


RULE_MATS_T = torch.tensor(
    RULE_MATS,
    dtype=torch.float32
)


# =============================================================================
# TOHUMLAMA
# =============================================================================

MAX_TORCH_SEED = (
    (1 << 63) - 1
)


def derive_seed(
    master_seed: int,
    label: str
) -> int:

    digest = hashlib.sha256(
        f"{int(master_seed)}::{label}"
        .encode(
            "utf-8"
        )
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False
        )
        & MAX_TORCH_SEED
    )


def seed_all(
    seed: int
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
    device: str
) -> torch.Generator:

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        int(
            seed
        )
    )

    return generator


# =============================================================================
# DEVAMLI MUTLAK SİNÜZOİDAL KONUM KODLAMASI
# =============================================================================

class ContinuousSinusoidalPosition(
    nn.Module
):

    def __init__(
        self,
        d_model: int
    ):

        super().__init__()

        self.d_model = int(
            d_model
        )

        inverse_frequency = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float64
            )
            * (
                -math.log(
                    10000.0
                )
                / d_model
            )
        )

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False
        )

    def forward(
        self,
        start_position: int,
        sequence_length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:

        positions = torch.arange(
            int(
                start_position
            ),
            int(
                start_position
                + sequence_length
            ),
            device=device,
            dtype=torch.float64
        )

        frequencies = torch.outer(
            positions,
            self.inverse_frequency.to(
                device=device
            )
        )

        position_encoding = torch.zeros(
            (
                sequence_length,
                self.d_model
            ),
            device=device,
            dtype=torch.float64
        )

        position_encoding[
            :,
            0::2
        ] = torch.sin(
            frequencies
        )

        position_encoding[
            :,
            1::2
        ] = torch.cos(
            frequencies
        )

        return (
            position_encoding
            .to(
                dtype=dtype
            )
            .unsqueeze(
                0
            )
            .expand(
                batch_size,
                -1,
                -1
            )
        )


# =============================================================================
# T5-INSPIRED NO-REFERENCE MODEL
# =============================================================================

@dataclass
class T5NoRefConfig:

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

    source_len: int = (
        SOURCE_LEN
    )

    target_len: int = (
        TARGET_BLOCK
    )

    dropout: float = DROPOUT


class T5NoReferenceDNA(
    nn.Module
):

    def __init__(
        self,
        config: T5NoRefConfig
    ):

        super().__init__()

        self.config = config

        self.token_embedding = (
            nn.Embedding(
                config.vocab_size,
                config.d_model
            )
        )

        self.position_encoding = (
            ContinuousSinusoidalPosition(
                config.d_model
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
                norm_first=True
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
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=(
                    config.n_encoder_layer
                )
            )
        )

        self.decoder = (
            nn.TransformerDecoder(
                decoder_layer,
                num_layers=(
                    config.n_decoder_layer
                )
            )
        )

        self.target_position_projection = (
            nn.Linear(
                config.d_model,
                config.d_model,
                bias=False
            )
        )

        self.context_summary_projection = (
            nn.Linear(
                config.d_model,
                config.d_model,
                bias=False
            )
        )

        self.final_norm = (
            nn.LayerNorm(
                config.d_model
            )
        )

        self.base_head = (
            nn.Linear(
                config.d_model,
                4,
                bias=False
            )
        )

        self.initialize_no_reference_symmetry()

    def initialize_no_reference_symmetry(
        self
    ) -> None:

        with torch.no_grad():

            embedding_weights = (
                self.token_embedding.weight
            )

            embedding_weights.sub_(
                embedding_weights.mean(
                    dim=0,
                    keepdim=True
                )
            )

            embedding_weights.div_(
                embedding_weights.norm(
                    dim=1,
                    keepdim=True
                ).clamp_min(
                    1e-8
                )
            )

            embedding_weights.mul_(
                math.sqrt(
                    self.config.d_model
                )
            )

            nn.init.orthogonal_(
                self.target_position_projection.weight
            )

            nn.init.orthogonal_(
                self.context_summary_projection.weight
            )

            random_basis = torch.randn(
                (
                    self.config.d_model,
                    3
                ),
                device=(
                    self.base_head.weight.device
                ),
                dtype=(
                    self.base_head.weight.dtype
                )
            )

            orthogonal_basis, _ = (
                torch.linalg.qr(
                    random_basis,
                    mode="reduced"
                )
            )

            tetrahedron_vertices = (
                torch.tensor(
                    [
                        [
                            1.0,
                            1.0,
                            1.0
                        ],
                        [
                            1.0,
                            -1.0,
                            -1.0
                        ],
                        [
                            -1.0,
                            1.0,
                            -1.0
                        ],
                        [
                            -1.0,
                            -1.0,
                            1.0
                        ]
                    ],
                    device=(
                        self.base_head.weight.device
                    ),
                    dtype=(
                        self.base_head.weight.dtype
                    )
                )
                / math.sqrt(
                    3.0
                )
            )

            symmetric_weights = (
                tetrahedron_vertices
                @ orthogonal_basis.transpose(
                    0,
                    1
                )
            )

            self.base_head.weight.copy_(
                symmetric_weights
            )

    def forward(
        self,
        source_tokens: torch.LongTensor,
        source_absolute_start: int,
        target_absolute_start: int
    ) -> torch.Tensor:

        batch_size = source_tokens.size(
            0
        )

        model_dtype = (
            self.token_embedding.weight.dtype
        )

        source_positions = (
            self.position_encoding(
                start_position=(
                    source_absolute_start
                ),
                sequence_length=(
                    self.config.source_len
                ),
                batch_size=(
                    batch_size
                ),
                device=(
                    source_tokens.device
                ),
                dtype=(
                    model_dtype
                )
            )
        )

        source_hidden = (
            self.token_embedding(
                source_tokens
            )
            + source_positions
        )

        encoder_memory = (
            self.encoder(
                source_hidden
            )
        )

        target_positions = (
            self.position_encoding(
                start_position=(
                    target_absolute_start
                ),
                sequence_length=(
                    self.config.target_len
                ),
                batch_size=(
                    batch_size
                ),
                device=(
                    source_tokens.device
                ),
                dtype=(
                    model_dtype
                )
            )
        )

        context_summary = (
            encoder_memory.mean(
                dim=1,
                keepdim=True
            )
        )

        decoder_query = (
            self.target_position_projection(
                target_positions
            )
            + self.context_summary_projection(
                context_summary
            )
        )

        decoder_hidden = (
            self.decoder(
                decoder_query,
                encoder_memory
            )
        )

        decoder_hidden = (
            self.final_norm(
                decoder_hidden
            )
        )

        return self.base_head(
            decoder_hidden
        )


# =============================================================================
# BLOK LOGİT KALİBRASYONU
# =============================================================================

def calibrate_block_logits(
    block_logits: torch.Tensor
) -> torch.Tensor:

    if not ENABLE_BLOCK_LOGIT_CALIBRATION:
        return block_logits

    centered_logits = (
        block_logits
        - block_logits.mean(
            dim=1,
            keepdim=True
        )
    )

    column_rms = torch.sqrt(
        torch.mean(
            centered_logits
            * centered_logits,
            dim=1,
            keepdim=True
        )
        + 1e-6
    )

    normalized_logits = (
        centered_logits
        / column_rms
    )

    normalized_logits = (
        normalized_logits
        - normalized_logits.mean(
            dim=-1,
            keepdim=True
        )
    )

    normalized_logits = (
        normalized_logits
        * BLOCK_LOGIT_SCALE
    )

    return torch.clamp(
        normalized_logits,
        min=-BLOCK_LOGIT_CLAMP,
        max=BLOCK_LOGIT_CLAMP
    )


# =============================================================================
# YUMUŞAK DENGELEME
# =============================================================================

def calculate_target_probabilities(
    gc_target: float
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
            1.0
            - gc_target
        ) / 2.0,
        "C": (
            gc_target
        ) / 2.0,
        "G": (
            gc_target
        ) / 2.0,
        "T": (
            1.0
            - gc_target
        ) / 2.0
    }


def calculate_soft_balance_bias(
    recent_counts: torch.Tensor,
    recent_length: int,
    global_counts: torch.Tensor,
    generated_length: int,
    target_vector: torch.Tensor,
    dtype: torch.dtype
) -> torch.Tensor:

    zero_bias = torch.zeros(
        4,
        device=(
            target_vector.device
        ),
        dtype=dtype
    )

    if (
        not ENABLE_SOFT_BALANCE
        or generated_length
        < BALANCE_WARMUP
        or recent_length <= 0
    ):

        return zero_bias

    target = target_vector.to(
        dtype=dtype
    )

    local_rates = (
        recent_counts.to(
            dtype=dtype
        )
        / float(
            recent_length
        )
    )

    global_rates = (
        global_counts.to(
            dtype=dtype
        )
        / float(
            max(
                1,
                generated_length
            )
        )
    )

    local_bias = torch.clamp(
        LOCAL_BALANCE_GAIN
        * (
            target
            - local_rates
        ),
        min=-LOCAL_BALANCE_CLAMP,
        max=LOCAL_BALANCE_CLAMP
    )

    global_bias = torch.clamp(
        GLOBAL_BALANCE_GAIN
        * (
            target
            - global_rates
        ),
        min=-GLOBAL_BALANCE_CLAMP,
        max=GLOBAL_BALANCE_CLAMP
    )

    return torch.clamp(
        local_bias
        + global_bias,
        min=-TOTAL_BALANCE_CLAMP,
        max=TOTAL_BALANCE_CLAMP
    )


# =============================================================================
# KATEGORİK ÖRNEKLEME
# =============================================================================

def sample_categorical(
    probabilities: torch.Tensor,
    generator: torch.Generator
) -> int:

    if not torch.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Örnekleme olasılıklarında NaN/Inf oluştu."
        )

    probability_sum = (
        probabilities.sum()
    )

    if float(
        probability_sum.item()
    ) <= 0.0:

        raise RuntimeError(
            "Örnekleme olasılıklarının toplamı sıfır."
        )

    normalized_probabilities = (
        probabilities
        / probability_sum
    )

    return int(
        torch.multinomial(
            normalized_probabilities,
            num_samples=1,
            generator=generator
        ).item()
    )


# =============================================================================
# ADAPTİF DNA-BİT KURALI
# =============================================================================

def select_rule_adaptive(
    base_probabilities: torch.Tensor,
    recent_rules: Deque[int],
    generator: torch.Generator,
    rule_matrices: torch.Tensor
) -> int:

    expected_bits = torch.einsum(
        "f,rfj->rj",
        base_probabilities,
        rule_matrices
    )

    clipped_bits = expected_bits.clamp(
        1e-6,
        1.0 - 1e-6
    )

    entropy_scores = -(
        clipped_bits
        * torch.log2(
            clipped_bits
        )
        + (
            1.0
            - clipped_bits
        )
        * torch.log2(
            1.0
            - clipped_bits
        )
    ).sum(
        dim=1
    )

    recent_rule_counts = torch.zeros(
        8,
        device=(
            base_probabilities.device
        ),
        dtype=(
            base_probabilities.dtype
        )
    )

    for rule_id in recent_rules:

        recent_rule_counts[
            int(
                rule_id
            )
        ] += 1.0

    diversity_scores = (
        1.0
        / (
            1.0
            + recent_rule_counts
        )
    )

    diversity_scores = (
        diversity_scores
        / diversity_scores.mean()
    )

    rule_probabilities = (
        torch.softmax(
            (
                entropy_scores
                * diversity_scores
            )
            / max(
                1e-4,
                RULE_TEMP
            ),
            dim=-1
        )
    )

    return sample_categorical(
        rule_probabilities,
        generator
    )


# =============================================================================
# DNA ÜRETİMİ
# =============================================================================

@torch.inference_mode()
def generate_dna(
    model: T5NoReferenceDNA,
    start_tokens: torch.LongTensor,
    output_length: int,
    device: str,
    generator: torch.Generator
) -> Tuple[
    str,
    array,
    bytes
]:

    model.eval()

    config = model.config

    context = start_tokens.to(
        device
    )

    context_absolute_start = 0

    dna_output = bytearray()
    bit_output = bytearray()

    rule_output = array(
        "B"
    )

    target_probabilities = (
        calculate_target_probabilities(
            GC_TARGET
        )
    )

    target_vector = torch.tensor(
        [
            target_probabilities[
                base
            ]
            for base in DNA
        ],
        device=device,
        dtype=torch.float32
    )

    global_counts = torch.zeros(
        4,
        dtype=torch.long,
        device=device
    )

    recent_counts = torch.zeros(
        4,
        dtype=torch.long,
        device=device
    )

    recent_window: Deque[int] = (
        deque()
    )

    recent_rules: Deque[int] = (
        deque(
            maxlen=RECENT_RULE_WINDOW
        )
    )

    recent_bases: Deque[int] = (
        deque(
            maxlen=MAX_RECENT_LAG
        )
    )

    last_base: Optional[int] = None
    run_length = 0

    previous_1: Optional[int] = None
    previous_2: Optional[int] = None

    bigram_counts = torch.zeros(
        16,
        dtype=torch.long,
        device=device
    )

    trigram_counts = torch.zeros(
        64,
        dtype=torch.long,
        device=device
    )

    number_of_bigrams = 0
    number_of_trigrams = 0

    rule_matrices = RULE_MATS_T.to(
        device=device
    )

    progress_bar = tqdm(
        total=output_length,
        desc="T5 no-ref v3 generate",
        unit="base"
    )

    while (
        len(
            dna_output
        )
        < output_length
    ):

        block_length = min(
            config.target_len,
            output_length
            - len(
                dna_output
            )
        )

        target_absolute_start = (
            context_absolute_start
            + config.source_len
        )

        raw_block_logits = model(
            source_tokens=context,
            source_absolute_start=(
                context_absolute_start
            ),
            target_absolute_start=(
                target_absolute_start
            )
        )

        calibrated_block_logits = (
            calibrate_block_logits(
                raw_block_logits
            )
        )

        new_block: List[int] = []

        for position in range(
            block_length
        ):

            step = len(
                dna_output
            )

            step_logits = (
                calibrated_block_logits[
                    0,
                    position
                ].clone()
            )

            step_logits += (
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
                        step_logits.dtype
                    )
                )
            )

            # -------------------------------------------------------------
            # Kısa menzilli çoklu-lag dekorrelasyon
            # -------------------------------------------------------------
            #
            # Hazır logaritmik cezalar kullanılır.
            # Bu bölüm her baz adımında yeniden hesaplama yapmaz.
            #

            for lag, log_penalty in RECENT_LAG_LOG_PENALTIES:

                if len(
                    recent_bases
                ) >= lag:

                    base_at_lag = recent_bases[
                        -lag
                    ]

                    step_logits[
                        base_at_lag
                    ] += log_penalty

            # -------------------------------------------------------------
            # Trigram düzeltmesi
            # -------------------------------------------------------------

            if (
                ENABLE_TRIMER
                and previous_2 is not None
                and previous_1 is not None
                and step
                >= TRIMER_WARMUP
            ):

                expected_trigram_count = max(
                    1.0,
                    number_of_trigrams
                    / 64.0
                )

                for base_id in range(
                    4
                ):

                    trigram_index = (
                        (
                            previous_2
                            << 4
                        )
                        | (
                            previous_1
                            << 2
                        )
                        | base_id
                    )

                    observed_ratio = (
                        (
                            float(
                                trigram_counts[
                                    trigram_index
                                ].item()
                            )
                            + 1.0
                        )
                        / (
                            expected_trigram_count
                            + 1.0
                        )
                    )

                    trigram_bias = (
                        -TRIMER_ALPHA
                        * math.log(
                            max(
                                1e-6,
                                observed_ratio
                            )
                        )
                    )

                    step_logits[
                        base_id
                    ] += max(
                        -TRIMER_CLAMP,
                        min(
                            TRIMER_CLAMP,
                            trigram_bias
                        )
                    )

            # -------------------------------------------------------------
            # Bigram düzeltmesi
            # -------------------------------------------------------------

            if (
                ENABLE_BIGRAM
                and previous_1 is not None
                and step
                >= BIGRAM_WARMUP
            ):

                expected_bigram_count = max(
                    1.0,
                    number_of_bigrams
                    / 16.0
                )

                for base_id in range(
                    4
                ):

                    bigram_index = (
                        previous_1
                        << 2
                    ) | base_id

                    observed_ratio = (
                        (
                            float(
                                bigram_counts[
                                    bigram_index
                                ].item()
                            )
                            + 1.0
                        )
                        / (
                            expected_bigram_count
                            + 1.0
                        )
                    )

                    bigram_bias = (
                        -BIGRAM_ALPHA
                        * math.log(
                            max(
                                1e-6,
                                observed_ratio
                            )
                        )
                    )

                    step_logits[
                        base_id
                    ] += max(
                        -BIGRAM_CLAMP,
                        min(
                            BIGRAM_CLAMP,
                            bigram_bias
                        )
                    )

            # -------------------------------------------------------------
            # Homopolimer maskesi
            # -------------------------------------------------------------

            allowed_mask = torch.ones(
                4,
                device=device,
                dtype=torch.bool
            )

            if (
                HOMOPOLYMER_MAX is not None
                and last_base is not None
                and run_length
                >= HOMOPOLYMER_MAX
            ):

                allowed_mask[
                    last_base
                ] = False

            model_probabilities = (
                torch.softmax(
                    step_logits,
                    dim=-1
                )
            )

            model_probabilities = (
                model_probabilities
                * allowed_mask.to(
                    dtype=(
                        model_probabilities.dtype
                    )
                )
            )

            if float(
                model_probabilities
                .sum()
                .item()
            ) <= 0.0:

                model_probabilities = (
                    allowed_mask.to(
                        dtype=(
                            model_probabilities.dtype
                        )
                    )
                )

            model_probabilities = (
                model_probabilities
                / model_probabilities.sum()
            )

            uniform_allowed = (
                allowed_mask.to(
                    dtype=(
                        model_probabilities.dtype
                    )
                )
            )

            uniform_allowed = (
                uniform_allowed
                / uniform_allowed.sum()
            )

            base_probabilities = (
                (
                    1.0
                    - MODEL_UNIFORM_MIX
                )
                * model_probabilities
                + MODEL_UNIFORM_MIX
                * uniform_allowed
            )

            base_probabilities = (
                base_probabilities
                / base_probabilities.sum()
            )

            base_id = sample_categorical(
                base_probabilities,
                generator
            )

            rule_id = select_rule_adaptive(
                base_probabilities=(
                    base_probabilities
                ),
                recent_rules=(
                    recent_rules
                ),
                generator=(
                    generator
                ),
                rule_matrices=(
                    rule_matrices
                )
            )

            base_character = INV_VOCAB[
                base_id
            ]

            bit_pair = ENC_RULES[
                rule_id
            ][
                base_character
            ]

            dna_output.append(
                ord(
                    base_character
                )
            )

            bit_output.extend(
                bit_pair.encode(
                    "ascii"
                )
            )

            rule_output.append(
                rule_id
            )

            recent_rules.append(
                rule_id
            )

            new_block.append(
                base_id
            )

            # Seçilen baz yalnızca sonraki adımları etkiler.
            recent_bases.append(
                base_id
            )

            # -------------------------------------------------------------
            # Baz sayaçları
            # -------------------------------------------------------------

            global_counts[
                base_id
            ] += 1

            if (
                len(
                    recent_window
                )
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

            # -------------------------------------------------------------
            # Bigram / trigram sayaçları
            # -------------------------------------------------------------

            if previous_1 is not None:

                bigram_index = (
                    previous_1
                    << 2
                ) | base_id

                bigram_counts[
                    bigram_index
                ] += 1

                number_of_bigrams += 1

            if (
                previous_2 is not None
                and previous_1 is not None
            ):

                trigram_index = (
                    (
                        previous_2
                        << 4
                    )
                    | (
                        previous_1
                        << 2
                    )
                    | base_id
                )

                trigram_counts[
                    trigram_index
                ] += 1

                number_of_trigrams += 1

            # -------------------------------------------------------------
            # Run-length
            # -------------------------------------------------------------

            if (
                last_base is None
                or base_id
                != last_base
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
            device=device
        ).unsqueeze(
            0
        )

        context = torch.cat(
            [
                context,
                new_block_tensor
            ],
            dim=1
        )[
            :,
            -config.source_len:
        ]

        context_absolute_start += (
            block_length
        )

    progress_bar.close()

    dna_sequence = dna_output.decode(
        "ascii"
    )

    bit_sequence = bytes(
        bit_output
    )

    if (
        len(
            dna_sequence
        )
        != output_length
    ):

        raise RuntimeError(
            f"DNA uzunluğu hatalı: "
            f"{len(dna_sequence):,}"
        )

    if (
        len(
            bit_sequence
        )
        != 2
        * output_length
    ):

        raise RuntimeError(
            f"Bit uzunluğu hatalı: "
            f"{len(bit_sequence):,}"
        )

    return (
        dna_sequence,
        rule_output,
        bit_sequence
    )


# =============================================================================
# ANALİZ
# =============================================================================

def entropy_from_counts(
    counts: Dict[str, int]
) -> float:

    total = sum(
        counts.values()
    )

    if total <= 0:
        return 0.0

    entropy = 0.0

    for base in DNA:

        count = counts[
            base
        ]

        if count <= 0:
            continue

        probability = (
            count
            / total
        )

        entropy -= (
            probability
            * math.log2(
                probability
            )
        )

    return float(
        entropy
    )


def analyze_outputs(
    dna_sequence: str,
    bit_sequence: bytes
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

    number_of_bits = len(
        bit_sequence
    )

    number_of_ones = (
        bit_sequence.count(
            ord(
                "1"
            )
        )
    )

    results: Dict[str, object] = {
        "len_bases": (
            number_of_bases
        ),
        "len_bits": (
            number_of_bits
        ),
        "counts": (
            counts
        ),
        "entropy_bits_per_base": (
            entropy_from_counts(
                counts
            )
        ),
        "p_one": (
            number_of_ones
            / max(
                1,
                number_of_bits
            )
        )
    }

    if HAVE_SCIPY:

        results[
            "monobit_p"
        ] = float(
            binomtest(
                number_of_ones,
                number_of_bits,
                p=0.5
            ).pvalue
        )

        results[
            "chi2_p"
        ] = float(
            chisquare(
                [
                    counts[
                        base
                    ]
                    for base in DNA
                ],
                [
                    number_of_bases
                    / 4.0
                ] * 4
            ).pvalue
        )

    else:

        results[
            "monobit_p"
        ] = None

        results[
            "chi2_p"
        ] = None

    return results


def calculate_tail_summary(
    dna_sequence: str
) -> Dict[str, object]:

    summary: Dict[
        str,
        object
    ] = {}

    for window_size in TAIL_WINDOWS:

        actual_size = min(
            int(
                window_size
            ),
            len(
                dna_sequence
            )
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
                counts[
                    base
                ]
                / actual_size
            )
            for base in DNA
        }

        summary[
            str(
                window_size
            )
        ] = {
            "actual_size": (
                actual_size
            ),
            "counts": (
                counts
            ),
            "rates": (
                rates
            ),
            "gc_rate": (
                rates[
                    "C"
                ]
                + rates[
                    "G"
                ]
            )
        }

    return summary


def print_tail_summary(
    summary: Dict[str, object]
) -> None:

    print(
        "\n--- TAIL BASE DISTRIBUTION ---"
    )

    for (
        window_size,
        item
    ) in summary.items():

        rates = item[
            "rates"
        ]

        print(
            f"last "
            f"{int(window_size):>6,} bases: "
            f"A="
            f"{100 * rates['A']:.3f}%  "
            f"C="
            f"{100 * rates['C']:.3f}%  "
            f"G="
            f"{100 * rates['G']:.3f}%  "
            f"T="
            f"{100 * rates['T']:.3f}%  "
            f"GC="
            f"{100 * item['gc_rate']:.3f}%"
        )


# =============================================================================
# BİT PAKETLEME VE SHAKE
# =============================================================================

def pack_bits(
    bit_sequence: bytes
) -> bytes:

    output = bytearray()

    accumulator = 0
    number_of_bits = 0

    for character in bit_sequence:

        accumulator = (
            accumulator
            << 1
        ) | (
            1
            if character
            == ord(
                "1"
            )
            else 0
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


def model_fingerprint(
    model: nn.Module
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        tensor
    ) in model.state_dict().items():

        digest.update(
            name.encode(
                "utf-8"
            )
            + b"\x00"
        )

        digest.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode(
                "ascii"
            )
            + b"\x00"
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "ascii"
            )
            + b"\x00"
        )

        digest.update(
            tensor.detach()
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def shake_extract(
    bit_sequence: bytes,
    run_seed: int,
    number_of_bytes: int,
    domain: bytes,
    fingerprint_hex: str
) -> bytes:

    shake = hashlib.shake_256()

    shake.update(
        domain
        + b"\x00"
    )

    shake.update(
        int(
            run_seed
        ).to_bytes(
            8,
            byteorder="big",
            signed=False
        )
    )

    shake.update(
        bytes.fromhex(
            fingerprint_hex
        )
    )

    shake.update(
        len(
            bit_sequence
        ).to_bytes(
            8,
            byteorder="big",
            signed=False
        )
    )

    shake.update(
        pack_bits(
            bit_sequence
        )
    )

    return shake.digest(
        number_of_bytes
    )


# =============================================================================
# ANA AKIŞ
# =============================================================================

if __name__ == "__main__":

    print(
        "[DEVICE]",
        DEVICE
    )

    print(
        "[OUT_DIR]",
        OUT_DIR
    )

    print(
        "[MODE] "
        "T5 no-reference v3 / no-training"
    )

    print(
        "[POSITION] "
        "continuous absolute sinusoidal"
    )

    print(
        "[DECODER_QUERY] "
        "absolute target position "
        "+ encoder context summary"
    )

    print(
        "[BASE_HEAD] "
        "symmetric tetrahedral initialization"
    )

    print(
        "[LOGIT_CALIBRATION] "
        "block-column centering "
        "+ RMS normalization"
    )

    print(
        "[BALANCE_MODE] "
        "bounded local-window "
        "+ weak global feedback"
    )

    print(
        "[DECORRELATION] "
        "multi-lag 1-5 bounded penalties"
    )

    run_seed = (
        int.from_bytes(
            os.urandom(
                8
            ),
            byteorder="little",
            signed=False
        )
        & MAX_TORCH_SEED
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
                    0
                )
                & MAX_TORCH_SEED
            )

        except ValueError as error:

            raise ValueError(
                f"RUN_SEED geçerli bir "
                f"tam sayı değil: "
                f"{seed_text}"
            ) from error

    model_seed = derive_seed(
        run_seed,
        "model_initialization"
    )

    prompt_seed = derive_seed(
        run_seed,
        "initial_prompt"
    )

    generation_seed = derive_seed(
        run_seed,
        "generation"
    )

    print(
        "[RUN_SEED]",
        run_seed
    )

    print(
        "[MODEL_SEED]",
        model_seed
    )

    print(
        "[PROMPT_SEED]",
        prompt_seed
    )

    print(
        "[GENERATION_SEED]",
        generation_seed
    )

    seed_all(
        model_seed
    )

    config = T5NoRefConfig()

    model = (
        T5NoReferenceDNA(
            config
        )
        .to(
            DEVICE
        )
        .eval()
    )

    fingerprint = model_fingerprint(
        model
    )

    print(
        "[MODEL_FINGERPRINT]",
        fingerprint[
            :16
        ]
        + "..."
    )

    prompt_generator = make_generator(
        prompt_seed,
        DEVICE
    )

    generation_generator = make_generator(
        generation_seed,
        DEVICE
    )

    start_tokens = torch.randint(
        0,
        4,
        (
            1,
            config.source_len
        ),
        generator=(
            prompt_generator
        ),
        device=DEVICE,
        dtype=torch.long
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
        bit_sequence
    ) = generate_dna(
        model=model,
        start_tokens=(
            start_tokens
        ),
        output_length=(
            TARGET_BASES
        ),
        device=(
            DEVICE
        ),
        generator=(
            generation_generator
        )
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
            process.memory_info().rss
            - ram_start
        ) / (
            1024
            * 1024
        )

    else:

        generation_cpu_seconds = (
            time.process_time()
            - cpu_start
        )

        ram_delta_mb = None

    statistics = analyze_outputs(
        dna_sequence,
        bit_sequence
    )

    tail_summary = calculate_tail_summary(
        dna_sequence
    )

    efficiency = (
        len(
            dna_sequence
        )
        / max(
            generation_wall_seconds,
            1e-12
        )
    )

    maximum_homopolymer = max(
        (
            len(
                match.group(
                    0
                )
            )
            for match in re.finditer(
                r"(A+|C+|G+|T+)",
                dna_sequence
            )
        ),
        default=0
    )

    compression_ratio = (
        len(
            zlib.compress(
                bit_sequence,
                level=9
            )
        )
        / max(
            1,
            len(
                bit_sequence
            )
        )
    )

    rule_counts = Counter(
        rules
    )

    print(
        "\n--- T5 NO-REF V3 CORE PERFORMANCE ---"
    )

    print(
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
        ]
    )

    print(
        "max homopolymer=",
        maximum_homopolymer
    )

    print(
        "compression ratio=",
        compression_ratio
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
        f"{OUT_TAG}_{timestamp}"
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
        encoding="utf-8"
    ) as file:

        file.write(
            dna_sequence
        )

    with open(
        bits_path,
        "wb"
    ) as file:

        file.write(
            bit_sequence
        )

    with open(
        rules_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            " ".join(
                str(
                    int(
                        rule
                    )
                )
                for rule in rules
            )
        )

    packed_bits = pack_bits(
        bit_sequence
    )

    if USE_SHAKE:

        keystream = shake_extract(
            bit_sequence=(
                bit_sequence
            ),
            run_seed=(
                run_seed
            ),
            number_of_bytes=(
                KEYSTREAM_BYTES
            ),
            domain=(
                b"T5-NOREF-V3/core-v1"
            ),
            fingerprint_hex=(
                fingerprint
            )
        )

        key_256 = shake_extract(
            bit_sequence=(
                bit_sequence
            ),
            run_seed=(
                run_seed
            ),
            number_of_bytes=(
                KEY_BYTES
            ),
            domain=(
                b"T5-NOREF-V3/key-v1"
            ),
            fingerprint_hex=(
                fingerprint
            )
        )

    else:

        keystream = packed_bits[
            :KEYSTREAM_BYTES
        ]

        key_256 = packed_bits[
            :KEY_BYTES
        ]

    keystream_path = os.path.join(
        OUT_DIR,
        f"t5_noref_v3_keystream_"
        f"{timestamp}.bin"
    )

    key_path = os.path.join(
        OUT_DIR,
        f"t5_noref_v3_key_"
        f"{timestamp}.hex"
    )

    with open(
        keystream_path,
        "wb"
    ) as file:

        file.write(
            keystream
        )

    with open(
        key_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            key_256.hex()
            + "\n"
        )

    metadata = {
        "mode": (
            "t5_noref_v3_"
            "continuous_position_"
            "context_query_"
            "multilag"
        ),
        "trained": False,
        "uses_reference_data": False,
        "architecture": (
            "T5-inspired no-reference "
            "block-parallel encoder-decoder"
        ),
        "position_encoding": (
            "continuous_absolute_sinusoidal"
        ),
        "decoder_query": (
            "absolute_target_position_"
            "plus_encoder_context_summary"
        ),
        "base_head_initialization": (
            "symmetric_tetrahedral"
        ),
        "device": (
            DEVICE
        ),
        "torch_version": (
            torch.__version__
        ),
        "use_shake": (
            USE_SHAKE
        ),
        "model_fingerprint_sha256": (
            fingerprint
        ),
        "seeds": {
            "run_seed": int(
                run_seed
            ),
            "model_seed": int(
                model_seed
            ),
            "prompt_seed": int(
                prompt_seed
            ),
            "generation_seed": int(
                generation_seed
            )
        },
        "performance": {
            "wall_seconds": (
                generation_wall_seconds
            ),
            "cpu_seconds": (
                generation_cpu_seconds
            ),
            "ram_delta_mb": (
                ram_delta_mb
            ),
            "base_per_second": (
                efficiency
            )
        },
        "analysis": (
            statistics
        ),
        "tail_summary": (
            tail_summary
        ),
        "max_homopolymer": (
            maximum_homopolymer
        ),
        "compression_ratio": (
            compression_ratio
        ),
        "rule_counts": {
            str(
                rule_id
            ): int(
                rule_counts.get(
                    rule_id,
                    0
                )
            )
            for rule_id in range(
                8
            )
        },
        "model": asdict(
            config
        ),
        "logit_calibration": {
            "enabled": (
                ENABLE_BLOCK_LOGIT_CALIBRATION
            ),
            "block_logit_scale": (
                BLOCK_LOGIT_SCALE
            ),
            "block_logit_clamp": (
                BLOCK_LOGIT_CLAMP
            ),
            "model_uniform_mix": (
                MODEL_UNIFORM_MIX
            )
        },
        "constraints": {
            "balance_mode": (
                "bounded_local_window_"
                "plus_weak_global_feedback"
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
            "recent_lag_damp": {
                str(
                    lag
                ): float(
                    damp
                )
                for lag, damp
                in RECENT_LAG_DAMP.items()
            },
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
            "rule_selection": (
                "adaptive_bit_entropy_"
                "plus_recent_rule_diversity"
            )
        }
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    fixed_dna_path = os.path.join(
        OUT_DIR,
        "t5_noref_dna_500k.txt"
    )

    fixed_bits_path = os.path.join(
        OUT_DIR,
        "t5_noref_dna_1m.txt"
    )

    with open(
        fixed_dna_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            dna_sequence
        )

    with open(
        fixed_bits_path,
        "wb"
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

