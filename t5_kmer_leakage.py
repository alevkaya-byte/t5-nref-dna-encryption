# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:09:13 2026

@author: Alev Kaya
"""


# -*- coding: utf-8 -*-
"""
BMC Bioinformatics – T5 exact-match long k-mer memorization/leakage control.

Amaç
----
R1, R1-ext ve R2 rejimlerinin her birinde:

- 1.000.000 bazlık ilgili referans/eğitim DNA dizisini,
- 10 bağımsız koşumda üretilmiş 500.000 bazlık T5 DNA akışlarıyla

k = 32, 48 ve 64 için karşılaştırır.

Birincil ölçüm:
    Üretilen akıştaki her k-mer penceresinin, ilgili referans/eğitim
    dizisindeki herhangi bir k-mer ile doğrudan tam eşleşip eşleşmediği.

Ek ölçüm:
    Üretilen k-mer'in reverse-complement biçiminin referansta bulunup
    bulunmadığı ayrıca raporlanır. Bu ek ölçüm, Entropy çalışmasındaki
    doğrudan eşleşme sonucunun yerine geçmez; destekleyici tanılamadır.

R3:
    Eğitim veya harici referans korpusu kullanmadığı için klasik
    memorization/leakage analizine dahil edilmez ve "Not applicable"
    olarak raporlanır.

Beklenen klasör yapısı
----------------------
    T5_R1.1 ... T5_R1.10
    T5_R1-ext.1 ... T5_R1-ext.10
    T5_R2.1 ... T5_R2.10

Kod, daha önce oluşturulan:
    t5_multi_stream_selected_files.csv

dosyası mevcutsa aynı 30 üretilen DNA dosyasını doğrudan kullanır.
Bu CSV yoksa her koşum klasöründe 500.000 bazlık *.dna.txt dosyasını
otomatik bulur.

Referans dosyaları
------------------
R1 için öncelikli ad:
    real_dna_1m.txt

R1-ext için öncelikli ad:
    real-ext_dna_1m.txt

R2 için desteklenen yaygın adlar:
    synthetic_dna_1m.txt
    sentetic_dna_1m.txt
    sentetik_dna_1m.txt
    yapaydna_1m.txt
    yapay_dna_1m.txt

Dosya adların farklıysa yalnız REGIMES içindeki "reference_candidates"
listesine gerçek adı eklemen yeterlidir.

Çıktılar
--------
    t5_kmer_leakage_results.txt
    t5_kmer_leakage_per_stream.csv
    t5_kmer_leakage_summary.csv
    t5_kmer_leakage_selected_files.csv
    t5_kmer_leakage.json

Bilimsel sınır
--------------
Sıfır exact-match hit bulunması, test edilen k değerleri altında doğrudan
uzun-fragman kopyalanmasına karşı kanıt sağlar. Tüm olası ezberleme veya
veri sızıntısı biçimlerinin bulunmadığını matematiksel olarak kanıtlamaz.
"""



import csv
import json
import statistics
import sys
import time

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple


# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


N_STREAMS = 10


EXPECTED_REFERENCE_BASES = 1_000_000


EXPECTED_GENERATED_BASES = 500_000


K_VALUES: Tuple[int, ...] = (
    32,
    48,
    64,
)


# Entropy çalışmasıyla karşılaştırılabilir birincil ölçüm doğrudan eşleşmedir.
# Reverse-complement sonucu destekleyici olarak ayrıca verilir.
CHECK_REVERSE_COMPLEMENT = True


# Herhangi bir hit bulunursa tanılama amacıyla en fazla bu kadar örnek saklanır.
MAX_EXAMPLE_HITS = 20


# Dosya veya uzunluk problemi varsa çalışmayı durdur.
STRICT_MODE = True


# Referans dizisinin tam 1.000.000 baz olmasını zorunlu tut.
# Referansın biraz farklı uzunlukta ise False yapabilirsin; gerçek uzunluk
# rapora yine yazılır.
STRICT_REFERENCE_LENGTH = True


# Üretilen her T5 DNA akışının tam 500.000 baz olmasını zorunlu tut.
STRICT_GENERATED_LENGTH = True


# Daha önce tamamlanan multi-stream analizinin seçtiği dosyaları kullan.
MULTI_STREAM_SELECTED_CSV = (
    BASE_DIR
    / "t5_multi_stream_selected_files.csv"
)


REGIMES = (
    {
        "name": "R1",
        "folder_template": "T5_R1.{index}",
        "reference_candidates": (
            "real_dna_1m.txt",
            "real-DNA_1m.txt",
            "t5_real_dna_1m.txt",
        ),
    },

    {
        "name": "R1-ext",
        "folder_template": "T5_R1-ext.{index}",
        "reference_candidates": (
            "real-ext_dna_1m.txt",
            "real_ext_dna_1m.txt",
            "B_subtilis_168_clean_1M.txt",
            "B_subtilis_168_clean_1m.txt",
        ),
    },

    {
        "name": "R2",
        "folder_template": "T5_R2.{index}",
        "reference_candidates": (
            "synthetic_dna_1m.txt",
            "sentetic_dna_1m.txt",
            "sentetik_dna_1m.txt",
            "yapaydna_1m.txt",
            "yapay_dna_1m.txt",
            "t5_synthetic_dna_1m.txt",
        ),
    },
)


OUT_TEXT = (
    BASE_DIR
    / "t5_kmer_leakage_results.txt"
)


OUT_PER_STREAM_CSV = (
    BASE_DIR
    / "t5_kmer_leakage_per_stream.csv"
)


OUT_SUMMARY_CSV = (
    BASE_DIR
    / "t5_kmer_leakage_summary.csv"
)


OUT_SELECTED_FILES_CSV = (
    BASE_DIR
    / "t5_kmer_leakage_selected_files.csv"
)


OUT_JSON = (
    BASE_DIR
    / "t5_kmer_leakage.json"
)


# =============================================================================
# SABİTLER
# =============================================================================

DNA = "ACGT"


BASE_TO_CODE = {
    ord("A"): 0,
    ord("C"): 1,
    ord("G"): 2,
    ord("T"): 3,
}


CODE_TO_BASE = (
    "A",
    "C",
    "G",
    "T",
)


# =============================================================================
# VERİ SINIFLARI
# =============================================================================

@dataclass(
    frozen=True
)
class SelectedStream:

    regime: str

    run_index: int

    folder: str

    dna_path: str

    dna_length: int

    selection_method: str


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
# DNA OKUMA
# =============================================================================

def read_dna_codes(
    path: Path
) -> bytes:
    """
    FASTA başlıklarını atar.

    Yalnız A/C/G/T karakterlerini 0,1,2,3 kodlarına dönüştürür.

    Dönen bytes nesnesinde:
        A=0, C=1, G=2, T=3
    """

    if not path.is_file():

        raise FileNotFoundError(
            f"DNA dosyası bulunamadı: {path}"
        )


    output = bytearray()


    with path.open(
        "rb"
    ) as handle:

        for line in handle:

            stripped = line.lstrip()


            if stripped.startswith(
                b">"
            ):

                continue


            for byte in line.upper():

                code = BASE_TO_CODE.get(
                    byte
                )


                if code is not None:

                    output.append(
                        code
                    )


    if not output:

        raise ValueError(
            f"Dosyada A/C/G/T bulunamadı: {path}"
        )


    return bytes(
        output
    )


def probe_dna_length(
    path: Path
) -> int:

    return len(
        read_dna_codes(
            path
        )
    )


# =============================================================================
# REFERANS DOSYASINI BULMA
# =============================================================================

def resolve_reference_file(
    candidate_names: Sequence[str]
) -> Path:

    attempted = []


    for name in candidate_names:

        path = (
            BASE_DIR
            / name
        )


        attempted.append(
            str(
                path
            )
        )


        if path.is_file():

            return path


    raise FileNotFoundError(
        "Referans/eğitim DNA dosyası bulunamadı.\n"
        "Denenen yollar:\n  "
        + "\n  ".join(
            attempted
        )
        + "\nREGIMES içindeki reference_candidates listesine "
        "gerçek dosya adını ekle."
    )


# =============================================================================
# MULTI-STREAM CSV'DEN ÜRETİLEN DOSYALARI ALMA
# =============================================================================

def load_streams_from_multi_stream_csv(
    regime: str
) -> List[
    SelectedStream
]:

    if not MULTI_STREAM_SELECTED_CSV.is_file():

        return []


    selected: List[
        SelectedStream
    ] = []


    with MULTI_STREAM_SELECTED_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(
            handle
        )


        required_columns = {
            "regime",
            "run_index",
            "dna_path",
        }


        if not required_columns.issubset(
            set(
                reader.fieldnames
                or []
            )
        ):

            return []


        for row in reader:

            if (
                str(
                    row.get(
                        "regime",
                        ""
                    )
                ).strip()
                != regime
            ):

                continue


            try:

                run_index = int(
                    row[
                        "run_index"
                    ]
                )


            except Exception:

                continue


            dna_path = Path(
                row[
                    "dna_path"
                ]
            )


            if not dna_path.is_file():

                continue


            reported_length = row.get(
                "dna_length",
                ""
            )


            try:

                dna_length = int(
                    reported_length
                )


            except Exception:

                dna_length = probe_dna_length(
                    dna_path
                )


            selected.append(
                SelectedStream(
                    regime=regime,

                    run_index=run_index,

                    folder=str(
                        dna_path.parent
                    ),

                    dna_path=str(
                        dna_path
                    ),

                    dna_length=(
                        dna_length
                    ),

                    selection_method=(
                        "t5_multi_stream_selected_files.csv"
                    ),
                )
            )


    selected.sort(
        key=lambda item: (
            item.run_index
        )
    )


    if len(
        selected
    ) == N_STREAMS:

        return selected


    return []


# =============================================================================
# KLASÖRDEN OTOMATİK ÜRETİLEN DNA DOSYASI BULMA
# =============================================================================

def candidate_score(
    path: Path
) -> int:

    name = path.name.lower()


    score = 0


    if name.endswith(
        ".dna.txt"
    ):

        score += 1000


    if ".dna." in name:

        score += 300


    if (
        "softbalance" in name
        or "continuouspos" in name
    ):

        score += 100


    if "500k" in name:

        score += 20


    if "1m" in name:

        score -= 100


    if "analysis" in name:

        score -= 1000


    if "rules" in name:

        score -= 1000


    if "bits" in name:

        score -= 1000


    return score


def discover_generated_dna_in_folder(
    folder: Path
) -> Tuple[
    Path,
    int,
    str
]:

    if not folder.is_dir():

        raise FileNotFoundError(
            f"Koşum klasörü bulunamadı: {folder}"
        )


    # Önce JSON + aynı köklü .dna.txt paketini ara.
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


        if not dna_path.is_file():

            continue


        length = probe_dna_length(
            dna_path
        )


        if (
            not STRICT_GENERATED_LENGTH
            or length
            == EXPECTED_GENERATED_BASES
        ):

            return (
                dna_path,
                length,
                "matching_json_dna_bundle",
            )


    # Yedek olarak bütün TXT dosyalarında içerik/uzunluk kontrolü yap.
    candidates: List[
        Tuple[
            int,
            float,
            Path,
            int
        ]
    ] = []


    for path in folder.glob(
        "*.txt"
    ):

        if not path.is_file():

            continue


        try:

            length = probe_dna_length(
                path
            )


        except Exception:

            continue


        if (
            STRICT_GENERATED_LENGTH
            and length
            != EXPECTED_GENERATED_BASES
        ):

            continue


        if (
            not STRICT_GENERATED_LENGTH
            and length
            <= 0
        ):

            continue


        candidates.append(
            (
                candidate_score(
                    path
                ),

                path.stat().st_mtime,

                path,

                length,
            )
        )


    if not candidates:

        raise FileNotFoundError(
            f"{folder}: uygun üretilen DNA dosyası bulunamadı."
        )


    candidates.sort(
        reverse=True
    )


    (
        _score,
        _modified,
        selected_path,
        selected_length,
    ) = candidates[
        0
    ]


    return (
        selected_path,
        selected_length,
        "content_and_filename_fallback",
    )


def discover_generated_streams(
    regime: str,
    folder_template: str
) -> List[
    SelectedStream
]:

    selected_from_csv = load_streams_from_multi_stream_csv(
        regime
    )


    if selected_from_csv:

        return selected_from_csv


    selected: List[
        SelectedStream
    ] = []


    for run_index in range(
        1,
        N_STREAMS
        + 1
    ):

        folder = (
            BASE_DIR
            / folder_template.format(
                index=run_index
            )
        )


        (
            dna_path,
            dna_length,
            method,
        ) = discover_generated_dna_in_folder(
            folder
        )


        selected.append(
            SelectedStream(
                regime=regime,

                run_index=run_index,

                folder=str(
                    folder
                ),

                dna_path=str(
                    dna_path
                ),

                dna_length=(
                    dna_length
                ),

                selection_method=(
                    method
                ),
            )
        )


    return selected


# =============================================================================
# 2-BİT EXACT k-MER KODLAMA
# =============================================================================

def iter_rolling_kmer_codes(
    sequence: bytes,
    k: int
) -> Iterator[
    Tuple[
        int,
        int
    ]
]:
    """
    Her pencere için:

        forward_code,
        reverse_complement_code

    döndürür.

    Kodlama:
        A=00, C=01, G=10, T=11

    Python int kullanıldığı için k=64 için 128-bit exact kodlama yapılır;
    hash çakışması veya yaklaşık eşleşme yoktur.
    """

    if k <= 0:

        raise ValueError(
            "k pozitif olmalıdır."
        )


    if len(
        sequence
    ) < k:

        return


    mask = (
        1
        << (
            2
            * k
        )
    ) - 1


    high_shift = (
        2
        * (
            k
            - 1
        )
    )


    forward_code = 0

    reverse_complement_code = 0


    for index, base_code in enumerate(
        sequence
    ):

        forward_code = (
            (
                forward_code
                << 2
            )
            | base_code
        ) & mask


        reverse_complement_code = (
            reverse_complement_code
            >> 2
        ) | (
            (
                3
                - base_code
            )
            << high_shift
        )


        if index >= (
            k
            - 1
        ):

            yield (
                forward_code,
                reverse_complement_code,
            )


def decode_kmer(
    code: int,
    k: int
) -> str:

    output = [
        "A"
    ] * k


    value = int(
        code
    )


    for position in range(
        k - 1,
        -1,
        -1
    ):

        output[
            position
        ] = CODE_TO_BASE[
            value
            & 3
        ]


        value >>= 2


    return "".join(
        output
    )


# =============================================================================
# REFERANS k-MER SETİ
# =============================================================================

def build_reference_kmer_set(
    reference: bytes,
    k: int
) -> Tuple[
    set,
    int
]:

    total_windows = max(
        0,
        len(
            reference
        )
        - k
        + 1
    )


    reference_set = {
        forward_code
        for (
            forward_code,
            _reverse_complement_code
        ) in iter_rolling_kmer_codes(
            reference,
            k
        )
    }


    return (
        reference_set,
        total_windows,
    )


# =============================================================================
# TEK ÜRETİLEN AKIŞI ANALİZ ETME
# =============================================================================

def analyze_generated_stream(
    *,
    regime: str,
    run_index: int,
    dna_path: Path,
    generated: bytes,
    reference_set: set,
    reference_unique_kmers: int,
    k: int
) -> dict:

    total_windows = max(
        0,
        len(
            generated
        )
        - k
        + 1
    )


    direct_hits = 0

    reverse_complement_hits = 0

    any_orientation_hits = 0


    unique_direct_reference_codes = set()

    unique_reverse_reference_codes = set()

    unique_generated_windows_with_any_hit = set()


    example_direct_codes: List[
        int
    ] = []


    example_reverse_generated_codes: List[
        int
    ] = []


    for (
        forward_code,
        reverse_complement_code
    ) in iter_rolling_kmer_codes(
        generated,
        k
    ):

        direct_match = (
            forward_code
            in reference_set
        )


        reverse_match = (
            CHECK_REVERSE_COMPLEMENT
            and reverse_complement_code
            in reference_set
        )


        if direct_match:

            direct_hits += 1


            unique_direct_reference_codes.add(
                forward_code
            )


            if len(
                example_direct_codes
            ) < MAX_EXAMPLE_HITS:

                example_direct_codes.append(
                    forward_code
                )


        if reverse_match:

            reverse_complement_hits += 1


            unique_reverse_reference_codes.add(
                reverse_complement_code
            )


            if len(
                example_reverse_generated_codes
            ) < MAX_EXAMPLE_HITS:

                example_reverse_generated_codes.append(
                    forward_code
                )


        if (
            direct_match
            or reverse_match
        ):

            any_orientation_hits += 1


            unique_generated_windows_with_any_hit.add(
                forward_code
            )


    direct_hit_rate = (
        direct_hits
        / total_windows
        if total_windows
        else 0.0
    )


    reverse_hit_rate = (
        reverse_complement_hits
        / total_windows
        if total_windows
        else 0.0
    )


    any_hit_rate = (
        any_orientation_hits
        / total_windows
        if total_windows
        else 0.0
    )


    # Uniform dört-sembollü bir kontrol altında yaklaşık beklenen doğrudan hit.
    expected_direct_hits_uniform = (
        total_windows
        * reference_unique_kmers
        / (
            4 ** k
        )
    )


    return {
        "regime": (
            regime
        ),

        "run_index": (
            run_index
        ),

        "generated_file": str(
            dna_path
        ),

        "generated_bases": len(
            generated
        ),

        "k": (
            k
        ),

        "generated_windows": (
            total_windows
        ),

        "reference_unique_kmers": (
            reference_unique_kmers
        ),

        "direct_hits": (
            direct_hits
        ),

        "direct_hit_rate": float(
            direct_hit_rate
        ),

        "unique_direct_reference_kmers_hit": len(
            unique_direct_reference_codes
        ),

        "reverse_complement_checked": (
            CHECK_REVERSE_COMPLEMENT
        ),

        "reverse_complement_hits": (
            reverse_complement_hits
        ),

        "reverse_complement_hit_rate": float(
            reverse_hit_rate
        ),

        "unique_reverse_reference_kmers_hit": len(
            unique_reverse_reference_codes
        ),

        "any_orientation_hits": (
            any_orientation_hits
        ),

        "any_orientation_hit_rate": float(
            any_hit_rate
        ),

        "unique_generated_windows_with_any_hit": len(
            unique_generated_windows_with_any_hit
        ),

        "expected_direct_hits_uniform_approx": float(
            expected_direct_hits_uniform
        ),

        "direct_zero_hit_upper95_rule_of_three": (
            float(
                3.0
                / total_windows
            )
            if (
                total_windows > 0
                and direct_hits == 0
            )
            else None
        ),

        "example_direct_kmers": [
            decode_kmer(
                code,
                k
            )
            for code in example_direct_codes
        ],

        "example_generated_kmers_with_reverse_complement_hit": [
            decode_kmer(
                code,
                k
            )
            for code in example_reverse_generated_codes
        ],
    }


# =============================================================================
# BİR REJİMİ ANALİZ ETME
# =============================================================================

def analyze_regime(
    regime: dict
) -> Tuple[
    dict,
    List[dict],
    List[SelectedStream]
]:

    name = str(
        regime[
            "name"
        ]
    )


    folder_template = str(
        regime[
            "folder_template"
        ]
    )


    reference_path = resolve_reference_file(
        regime[
            "reference_candidates"
        ]
    )


    reference = read_dna_codes(
        reference_path
    )


    if (
        STRICT_REFERENCE_LENGTH
        and len(
            reference
        )
        != EXPECTED_REFERENCE_BASES
    ):

        raise ValueError(
            f"{name}: referans uzunluğu "
            f"{len(reference):,}; beklenen "
            f"{EXPECTED_REFERENCE_BASES:,}. "
            f"Dosya: {reference_path}"
        )


    selected_streams = discover_generated_streams(
        name,
        folder_template
    )


    if (
        STRICT_MODE
        and len(
            selected_streams
        )
        != N_STREAMS
    ):

        raise RuntimeError(
            f"{name}: {N_STREAMS} akış bekleniyordu; "
            f"{len(selected_streams)} bulundu."
        )


    generated_streams: List[
        Tuple[
            SelectedStream,
            bytes
        ]
    ] = []


    print(
        "\n"
        + "=" * 100
    )


    print(
        f"{name} EXACT-MATCH LONG k-MER CONTROL"
    )


    print(
        "=" * 100
    )


    print(
        f"Reference file   : "
        f"{reference_path}"
    )


    print(
        f"Reference bases  : "
        f"{len(reference):,}"
    )


    for selected in selected_streams:

        path = Path(
            selected.dna_path
        )


        generated = read_dna_codes(
            path
        )


        if (
            STRICT_GENERATED_LENGTH
            and len(
                generated
            )
            != EXPECTED_GENERATED_BASES
        ):

            raise ValueError(
                f"{name}.{selected.run_index}: "
                f"üretilen DNA uzunluğu "
                f"{len(generated):,}; beklenen "
                f"{EXPECTED_GENERATED_BASES:,}. "
                f"Dosya: {path}"
            )


        generated_streams.append(
            (
                selected,
                generated,
            )
        )


        print(
            f"[SELECT] {name}.{selected.run_index}: "
            f"{path.name} "
            f"({len(generated):,} baz) "
            f"[{selected.selection_method}]"
        )


    per_stream_records: List[
        dict
    ] = []


    k_summaries: List[
        dict
    ] = []


    for k in K_VALUES:

        build_start = time.perf_counter()


        (
            reference_set,
            reference_windows,
        ) = build_reference_kmer_set(
            reference,
            k
        )


        build_seconds = (
            time.perf_counter()
            - build_start
        )


        print(
            "\n"
            + "-" * 100
        )


        print(
            f"{name} | k={k}"
        )


        print(
            "-" * 100
        )


        print(
            f"Reference windows       : "
            f"{reference_windows:,}"
        )


        print(
            f"Reference unique k-mers : "
            f"{len(reference_set):,}"
        )


        print(
            f"Reference-set build     : "
            f"{build_seconds:.6f} s"
        )


        current_records: List[
            dict
        ] = []


        for (
            selected,
            generated
        ) in generated_streams:

            stream_start = time.perf_counter()


            record = analyze_generated_stream(
                regime=name,

                run_index=(
                    selected.run_index
                ),

                dna_path=Path(
                    selected.dna_path
                ),

                generated=generated,

                reference_set=reference_set,

                reference_unique_kmers=len(
                    reference_set
                ),

                k=k,
            )


            record[
                "analysis_seconds"
            ] = float(
                time.perf_counter()
                - stream_start
            )


            current_records.append(
                record
            )


            per_stream_records.append(
                record
            )


            print(
                f"{name}.{selected.run_index:02d} | "
                f"direct={record['direct_hits']} | "
                f"RC={record['reverse_complement_hits']} | "
                f"any={record['any_orientation_hits']} | "
                f"time={record['analysis_seconds']:.4f} s"
            )


        total_windows = int(
            sum(
                record[
                    "generated_windows"
                ]
                for record in current_records
            )
        )


        total_direct_hits = int(
            sum(
                record[
                    "direct_hits"
                ]
                for record in current_records
            )
        )


        total_reverse_hits = int(
            sum(
                record[
                    "reverse_complement_hits"
                ]
                for record in current_records
            )
        )


        total_any_hits = int(
            sum(
                record[
                    "any_orientation_hits"
                ]
                for record in current_records
            )
        )


        direct_rates = [
            float(
                record[
                    "direct_hit_rate"
                ]
            )
            for record in current_records
        ]


        reverse_rates = [
            float(
                record[
                    "reverse_complement_hit_rate"
                ]
            )
            for record in current_records
        ]


        streams_with_direct_hits = sum(
            int(
                record[
                    "direct_hits"
                ]
                > 0
            )
            for record in current_records
        )


        streams_with_reverse_hits = sum(
            int(
                record[
                    "reverse_complement_hits"
                ]
                > 0
            )
            for record in current_records
        )


        pooled_direct_rate = (
            total_direct_hits
            / total_windows
            if total_windows
            else 0.0
        )


        pooled_reverse_rate = (
            total_reverse_hits
            / total_windows
            if total_windows
            else 0.0
        )


        pooled_any_rate = (
            total_any_hits
            / total_windows
            if total_windows
            else 0.0
        )


        summary = {
            "regime": (
                name
            ),

            "reference_file": str(
                reference_path
            ),

            "reference_bases": len(
                reference
            ),

            "k": (
                k
            ),

            "reference_windows": (
                reference_windows
            ),

            "reference_unique_kmers": len(
                reference_set
            ),

            "number_of_generated_streams": len(
                current_records
            ),

            "generated_bases_each": (
                EXPECTED_GENERATED_BASES
            ),

            "total_generated_windows": (
                total_windows
            ),

            "total_direct_hits": (
                total_direct_hits
            ),

            "pooled_direct_hit_rate": float(
                pooled_direct_rate
            ),

            "streams_with_direct_hits": (
                streams_with_direct_hits
            ),

            "direct_rate_mean": float(
                statistics.fmean(
                    direct_rates
                )
            ),

            "direct_rate_sample_sd": (
                float(
                    statistics.stdev(
                        direct_rates
                    )
                )
                if len(
                    direct_rates
                ) > 1
                else 0.0
            ),

            "direct_zero_hit_upper95_rule_of_three": (
                float(
                    3.0
                    / total_windows
                )
                if (
                    total_windows > 0
                    and total_direct_hits == 0
                )
                else None
            ),

            "reverse_complement_checked": (
                CHECK_REVERSE_COMPLEMENT
            ),

            "total_reverse_complement_hits": (
                total_reverse_hits
            ),

            "pooled_reverse_complement_hit_rate": float(
                pooled_reverse_rate
            ),

            "streams_with_reverse_complement_hits": (
                streams_with_reverse_hits
            ),

            "reverse_rate_mean": float(
                statistics.fmean(
                    reverse_rates
                )
            ),

            "reverse_rate_sample_sd": (
                float(
                    statistics.stdev(
                        reverse_rates
                    )
                )
                if len(
                    reverse_rates
                ) > 1
                else 0.0
            ),

            "total_any_orientation_hits": (
                total_any_hits
            ),

            "pooled_any_orientation_hit_rate": float(
                pooled_any_rate
            ),

            "expected_total_direct_hits_uniform_approx": float(
                total_windows
                * len(
                    reference_set
                )
                / (
                    4 ** k
                )
            ),

            "reference_set_build_seconds": float(
                build_seconds
            ),

            "primary_result": (
                "No direct exact-match long k-mer overlap detected"
                if total_direct_hits == 0
                else "Direct exact-match long k-mer overlap detected"
            ),
        }


        k_summaries.append(
            summary
        )


        print(
            "\nPooled direct hits       : "
            f"{total_direct_hits}"
        )


        print(
            "Pooled direct hit rate   : "
            f"{pooled_direct_rate:.12e}"
        )


        print(
            "Pooled RC hits           : "
            f"{total_reverse_hits}"
        )


        print(
            "Pooled any-orient. hits  : "
            f"{total_any_hits}"
        )


        print(
            "Primary interpretation   : "
            f"{summary['primary_result']}"
        )


        del reference_set


    regime_summary = {
        "regime": (
            name
        ),

        "reference_file": str(
            reference_path
        ),

        "reference_bases": len(
            reference
        ),

        "number_of_streams": len(
            selected_streams
        ),

        "generated_bases_each": (
            EXPECTED_GENERATED_BASES
        ),

        "k_summaries": (
            k_summaries
        ),

        "all_direct_hit_totals_zero": all(
            item[
                "total_direct_hits"
            ]
            == 0
            for item in k_summaries
        ),

        "interpretation": (
            "No direct exact-match long-fragment copying detected "
            "for k=32,48,64 under the tested streams."
            if all(
                item[
                    "total_direct_hits"
                ]
                == 0
                for item in k_summaries
            )
            else (
                "At least one direct exact-match long k-mer was detected; "
                "the matched examples and source files should be reviewed."
            )
        ),
    }


    return (
        regime_summary,
        per_stream_records,
        selected_streams,
    )


# =============================================================================
# CSV YAZMA
# =============================================================================

def write_selected_files_csv(
    selected_streams: Sequence[
        SelectedStream
    ]
) -> None:

    fields = [
        "regime",
        "run_index",
        "folder",
        "dna_path",
        "dna_length",
        "selection_method",
    ]


    with OUT_SELECTED_FILES_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields
        )


        writer.writeheader()


        for selected in selected_streams:

            writer.writerow(
                asdict(
                    selected
                )
            )


def write_per_stream_csv(
    records: Sequence[
        dict
    ]
) -> None:

    if not records:

        return


    excluded = {
        "example_direct_kmers",
        "example_generated_kmers_with_reverse_complement_hit",
    }


    fields = [
        key
        for key in records[
            0
        ].keys()
        if key not in excluded
    ]


    with OUT_PER_STREAM_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields
        )


        writer.writeheader()


        for record in records:

            writer.writerow({
                key: value
                for key, value in record.items()
                if key in fields
            })


def flatten_summary_rows(
    regime_summaries: Sequence[
        dict
    ]
) -> List[
    dict
]:

    rows: List[
        dict
    ] = []


    for regime_summary in regime_summaries:

        for item in regime_summary[
            "k_summaries"
        ]:

            rows.append({
                "regime": (
                    item[
                        "regime"
                    ]
                ),

                "reference_file": (
                    item[
                        "reference_file"
                    ]
                ),

                "reference_bases": (
                    item[
                        "reference_bases"
                    ]
                ),

                "k": (
                    item[
                        "k"
                    ]
                ),

                "reference_windows": (
                    item[
                        "reference_windows"
                    ]
                ),

                "reference_unique_kmers": (
                    item[
                        "reference_unique_kmers"
                    ]
                ),

                "n_streams": (
                    item[
                        "number_of_generated_streams"
                    ]
                ),

                "generated_bases_each": (
                    item[
                        "generated_bases_each"
                    ]
                ),

                "total_generated_windows": (
                    item[
                        "total_generated_windows"
                    ]
                ),

                "total_direct_hits": (
                    item[
                        "total_direct_hits"
                    ]
                ),

                "pooled_direct_hit_rate": (
                    item[
                        "pooled_direct_hit_rate"
                    ]
                ),

                "streams_with_direct_hits": (
                    item[
                        "streams_with_direct_hits"
                    ]
                ),

                "direct_zero_hit_upper95_rule_of_three": (
                    item[
                        "direct_zero_hit_upper95_rule_of_three"
                    ]
                ),

                "total_reverse_complement_hits": (
                    item[
                        "total_reverse_complement_hits"
                    ]
                ),

                "pooled_reverse_complement_hit_rate": (
                    item[
                        "pooled_reverse_complement_hit_rate"
                    ]
                ),

                "total_any_orientation_hits": (
                    item[
                        "total_any_orientation_hits"
                    ]
                ),

                "pooled_any_orientation_hit_rate": (
                    item[
                        "pooled_any_orientation_hit_rate"
                    ]
                ),

                "expected_total_direct_hits_uniform_approx": (
                    item[
                        "expected_total_direct_hits_uniform_approx"
                    ]
                ),

                "primary_result": (
                    item[
                        "primary_result"
                    ]
                ),
            })


    rows.append({
        "regime": "R3",
        "reference_file": "",
        "reference_bases": "",
        "k": "N/A",
        "reference_windows": "",
        "reference_unique_kmers": "",
        "n_streams": 10,
        "generated_bases_each": EXPECTED_GENERATED_BASES,
        "total_generated_windows": "",
        "total_direct_hits": "N/A",
        "pooled_direct_hit_rate": "N/A",
        "streams_with_direct_hits": "N/A",
        "direct_zero_hit_upper95_rule_of_three": "N/A",
        "total_reverse_complement_hits": "N/A",
        "pooled_reverse_complement_hit_rate": "N/A",
        "total_any_orientation_hits": "N/A",
        "pooled_any_orientation_hit_rate": "N/A",
        "expected_total_direct_hits_uniform_approx": "N/A",
        "primary_result": (
            "Not applicable: R3 uses no external "
            "training/reference corpus"
        ),
    })


    return rows


def write_summary_csv(
    regime_summaries: Sequence[
        dict
    ]
) -> None:

    rows = flatten_summary_rows(
        regime_summaries
    )


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
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields
        )


        writer.writeheader()


        writer.writerows(
            rows
        )


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main() -> None:

    total_start = time.perf_counter()


    print(
        "T5 EXACT-MATCH LONG k-MER MEMORIZATION / LEAKAGE CONTROL"
    )


    print(
        f"Run time                  : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


    print(
        f"Base directory            : "
        f"{BASE_DIR}"
    )


    print(
        f"k values                  : "
        f"{K_VALUES}"
    )


    print(
        f"Streams per regime        : "
        f"{N_STREAMS}"
    )


    print(
        f"Expected reference bases  : "
        f"{EXPECTED_REFERENCE_BASES:,}"
    )


    print(
        f"Expected generated bases  : "
        f"{EXPECTED_GENERATED_BASES:,}"
    )


    print(
        f"Reverse-complement check  : "
        f"{CHECK_REVERSE_COMPLEMENT}"
    )


    print(
        f"Multi-stream file index   : "
        f"{MULTI_STREAM_SELECTED_CSV}"
    )


    regime_summaries: List[
        dict
    ] = []


    all_per_stream_records: List[
        dict
    ] = []


    all_selected_streams: List[
        SelectedStream
    ] = []


    for regime in REGIMES:

        (
            regime_summary,
            per_stream_records,
            selected_streams,
        ) = analyze_regime(
            regime
        )


        regime_summaries.append(
            regime_summary
        )


        all_per_stream_records.extend(
            per_stream_records
        )


        all_selected_streams.extend(
            selected_streams
        )


    write_selected_files_csv(
        all_selected_streams
    )


    write_per_stream_csv(
        all_per_stream_records
    )


    write_summary_csv(
        regime_summaries
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    report = {
        "analysis": (
            "T5 exact-match long k-mer memorization/leakage control"
        ),

        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "configuration": {
            "base_directory": str(
                BASE_DIR
            ),

            "n_streams_per_regime": (
                N_STREAMS
            ),

            "expected_reference_bases": (
                EXPECTED_REFERENCE_BASES
            ),

            "expected_generated_bases": (
                EXPECTED_GENERATED_BASES
            ),

            "k_values": list(
                K_VALUES
            ),

            "check_reverse_complement": (
                CHECK_REVERSE_COMPLEMENT
            ),

            "strict_mode": (
                STRICT_MODE
            ),

            "strict_reference_length": (
                STRICT_REFERENCE_LENGTH
            ),

            "strict_generated_length": (
                STRICT_GENERATED_LENGTH
            ),

            "multi_stream_selected_csv": str(
                MULTI_STREAM_SELECTED_CSV
            ),
        },

        "regime_summaries": (
            regime_summaries
        ),

        "per_stream_records": (
            all_per_stream_records
        ),

        "selected_streams": [
            asdict(
                selected
            )
            for selected in all_selected_streams
        ],

        "r3": {
            "status": (
                "Not applicable"
            ),

            "reason": (
                "R3 does not use an external training or reference corpus."
            ),
        },

        "interpretation_notes": [
            (
                "The primary result is the direct exact-match k-mer count, "
                "which is directly comparable with the Entropy analysis."
            ),

            (
                "Reverse-complement overlap is reported as a supplementary "
                "DNA-aware diagnostic."
            ),

            (
                "A zero direct-hit result supports the absence of direct "
                "long-fragment copying under the tested k values."
            ),

            (
                "Zero exact-match hits do not constitute a formal proof "
                "that every possible form of memorization is absent."
            ),
        ],

        "total_seconds": float(
            total_seconds
        ),
    }


    with OUT_JSON.open(
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
        + "=" * 100
    )


    print(
        "FINAL COMPACT TABLE"
    )


    print(
        "=" * 100
    )


    print(
        "Regime | k | Direct hits | RC hits | Any orientation | Result"
    )


    print(
        "-" * 100
    )


    for regime_summary in regime_summaries:

        for item in regime_summary[
            "k_summaries"
        ]:

            print(
                f"{item['regime']:6s} | "
                f"{item['k']:2d} | "
                f"{item['total_direct_hits']:11d} | "
                f"{item['total_reverse_complement_hits']:7d} | "
                f"{item['total_any_orientation_hits']:15d} | "
                f"{item['primary_result']}"
            )


    print(
        "R3    | N/A | N/A         | N/A     | N/A             | "
        "Not applicable: no training/reference corpus"
    )


    print(
        "\nKaydedilen dosyalar:"
    )


    print(
        f"  TXT raporu       : "
        f"{OUT_TEXT}"
    )


    print(
        f"  Seçilen dosyalar : "
        f"{OUT_SELECTED_FILES_CSV}"
    )


    print(
        f"  Akış sonuçları   : "
        f"{OUT_PER_STREAM_CSV}"
    )


    print(
        f"  Ana özet CSV     : "
        f"{OUT_SUMMARY_CSV}"
    )


    print(
        f"  JSON raporu      : "
        f"{OUT_JSON}"
    )


    print(
        f"  Toplam süre      : "
        f"{total_seconds:.6f} s"
    )


if __name__ == "__main__":

    original_stdout = sys.stdout


    with OUT_TEXT.open(
        "w",
        encoding="utf-8"
    ) as text_handle:

        sys.stdout = Tee(
            original_stdout,
            text_handle
        )


        try:

            main()


        finally:

            sys.stdout = (
                original_stdout
            )
