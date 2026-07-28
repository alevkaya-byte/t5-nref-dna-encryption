# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:27:05 2025

@author: Alev Kaya
"""

# nist_minisuite.py
# NIST mini-suite:
#  1) Monobit
#  2) Block Frequency
#  3) Runs
#  4) Longest Run of Ones in a Block
#  5) Binary Matrix Rank  (senin sınıf ve p-değeri yaklaşımınla)
#  6) Spectral (DFT)
#  7) Non-overlapping Template Matching  (senin formüllerinle)
#  8) Overlapping Template Matching      (senin formüllerinle)
#  9) Universal Statistical (Maurer)     (senin formüllerinle)
# 10) Linear Complexity                  (senin formüllerinle)
# 11) Serial Test (m, m-1, m-2)
# 12) Approximate Entropy (ApEn)
# 13) Cumulative Sums (forward & backward)
# 14) Random Excursions                   (senin mantığınla)
# 15) Random Excursions Variant           (senin mantığınla)

import sys
import math
import copy
from typing import Iterable
import numpy as np
from collections import Counter
from scipy.special import erfc, gammaincc, hyp1f1
from scipy.fft import fft
from scipy.stats import norm

ALPHA = 0.01
VERBOSE = True

# --------- ortak yardımcılar ---------
def read_bits_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        bits = f.read().replace(" ", "").replace("\n", "")
    if not bits:
        raise ValueError("Dosya boş.")
    if not set(bits).issubset({"0", "1"}):
        raise ValueError("Dosya yalnızca '0' ve '1' içermelidir.")
    return bits

def print_result(tag: str, p: float, alpha: float = ALPHA):
    status = "PASS" if (p is not None and p >= alpha) else "FAIL"
    ptxt = "None" if p is None else f"{p:.6f}"
    print(f"{tag} p={ptxt} -> {status}")

# --------- [NIST-1] Monobit ---------
def nist_monobit(bits: str) -> float:
    n = len(bits)
    s = 0
    for ch in bits:
        s += 1 if ch == "1" else -1
    sobs = s / math.sqrt(n)
    return erfc(abs(sobs) / math.sqrt(2.0))

# --------- [NIST-2] Block Frequency ---------
def nist_block_frequency(bits: str, M: int = 128) -> float:
    n = len(bits)
    if M <= 0:
        raise ValueError("Blok boyutu M > 0 olmalı.")
    N = n // M
    if N == 0:
        return 0.0
    chi_sum = 0.0
    idx = 0
    for _ in range(N):
        block = bits[idx:idx+M]
        pi = block.count("1") / M
        chi_sum += (pi - 0.5) ** 2
        idx += M
    chi_squared = 4.0 * M * chi_sum
    return gammaincc(N / 2.0, chi_squared / 2.0)

# --------- [NIST-3] Runs ---------
def nist_runs(bits: str, verbose: bool = False) -> float:
    n = len(bits)
    pi = bits.count("1") / n
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        if verbose:
            print("Run Test yapılmadı: Monobit koşulu sağlanmadı (|pi-0.5| >= tau).")
        return 0.0
    vObs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            vObs += 1
    p_val = erfc(abs(vObs - (2*n*pi*(1-pi))) / (2*math.sqrt(2*n)*pi*(1-pi)))
    if verbose:
        print("Run Test DEBUG BEGIN:")
        print(f"n={n}, tau={tau}, pi={pi}, vObs={vObs}, p={p_val}")
        print("DEBUG END.")
    return p_val

# --------- [NIST-4] Longest Run of Ones in a Block ---------
def nist_longest_run_ones(bits: str, verbose: bool = False) -> float:
    n = len(bits)
    if n < 128:
        return 0.0
    if n < 6272:
        k, m = 3, 8
        v_values = [1, 2, 3, 4]
        pi_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif n < 750000:
        k, m = 5, 128
        v_values = [4, 5, 6, 7, 8, 9]
        pi_values = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        k, m = 6, 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    N = n // m
    if N == 0:
        return 0.0
    freqs = np.zeros(k + 1, dtype=float)
    idx = 0
    for _ in range(N):
        block = bits[idx:idx+m]
        idx += m
        max_run = 0
        run = 0
        for b in block:
            if b == "1":
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        if max_run < v_values[0]:
            freqs[0] += 1
        for j in range(k):
            if max_run == v_values[j]:
                freqs[j] += 1
        if max_run > v_values[k-1]:
            freqs[k] += 1
    xObs = 0.0
    for i in range(len(freqs)):
        exp_i = N * pi_values[i]
        xObs += (freqs[i] - exp_i) ** 2 / exp_i
    p_val = gammaincc(k / 2.0, xObs / 2.0)
    if verbose:
        print("Longest Run DEBUG BEGIN:")
        print(f"n={n}, m={m}, N={N}, freqs={freqs}, xObs={xObs}, p={p_val}")
        print("DEBUG END.")
    return p_val

# --------- [NIST-5] Binary Matrix Rank (senin sınıfınla) ---------
class BinaryMatrix:
    def __init__(self, matrix, rows, cols):
        self.M = rows
        self.Q = cols
        self.A = matrix.astype(np.uint8, copy=True)
        self.m = min(rows, cols)

    def compute_rank(self, verbose=False):
        if verbose:
            print("Başlangıç Matrisi:\n", self.A)
        # ileri eliminasyon
        for i in range(self.m):
            if self.A[i, i] == 0:
                row_swap = self.find_unit_element_swap(i)
                if row_swap == -1:
                    continue
                self.A[i, :] ^= self.A[row_swap, :]
            for j in range(i + 1, self.M):
                if self.A[j, i] == 1:
                    self.A[j, :] ^= self.A[i, :]
        if verbose:
            print("İleri Eliminasyon Sonrası:\n", self.A)
        # geri eliminasyon
        for i in range(self.m - 1, -1, -1):
            if self.A[i, i] == 1:
                for j in range(i - 1, -1, -1):
                    if self.A[j, i] == 1:
                        self.A[j, :] ^= self.A[i, :]
        if verbose:
            print("Geri Eliminasyon Sonrası:\n", self.A)
        return self.determine_rank()

    def find_unit_element_swap(self, i):
        for row in range(i + 1, self.M):
            if self.A[row, i] == 1:
                return row
        return -1

    def determine_rank(self):
        return int(np.sum(np.any(self.A == 1, axis=1)))

def nist_binary_matrix_rank(bits: str, q: int = 32, verbose: bool = False) -> float:
    n = len(bits)
    block_size = q * q
    num_blocks = n // block_size
    if num_blocks == 0:
        if verbose:
            print("Yetersiz veri bloğu!")
        return 0.0
    counts = np.zeros(3, dtype=float)  # [rank=q, rank=q-1, diğer]
    idx = 0
    for _ in range(num_blocks):
        block_data = bits[idx:idx+block_size]
        idx += block_size
        block = np.frombuffer(block_data.encode("ascii"), dtype=np.uint8) - 48
        block = block.reshape((q, q)).astype(np.uint8, copy=False)
        r = BinaryMatrix(block, q, q).compute_rank(False)
        if r == q:       counts[0] += 1
        elif r == q - 1: counts[1] += 1
        else:            counts[2] += 1
    # senin teorik olasılıklar
    p0 = np.prod([1.0 - 1.0 / (2 ** i) for i in range(1, q + 1)])
    p1 = 2.0 * p0
    p2 = 1.0 - p0 - p1
    piks = np.array([p0, p1, p2], dtype=float)
    expected = num_blocks * piks
    chi_sq = np.sum((counts - expected) ** 2 / expected)
    p_val = math.exp(-chi_sq / 2.0)  # senin kullandığın yaklaşım
    if verbose:
        print("Binary Matrix Rank DEBUG BEGIN:")
        print(f"q={q}, num_blocks={num_blocks}, counts={counts}, expected={expected}, chi^2={chi_sq}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-6] Spectral (DFT) (senin formülünle) ---------
def nist_spectral(bits: str, verbose: bool = False) -> float:
    n = len(bits)
    if n < 16:
        return 0.0
    plus_minus_one = np.where(
        np.frombuffer(bits.encode("ascii"), dtype=np.uint8) == ord("1"),
        1, -1
    ).astype(float, copy=False)
    s = fft(plus_minus_one)
    modulus = np.abs(s[: n // 2])
    tau = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * (n / 2.0)
    N1 = float(np.sum(modulus < tau))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p_val = erfc(abs(d) / math.sqrt(2.0))
    if verbose:
        print("DFT (Spectral) DEBUG BEGIN:")
        print(f"n={n}, tau={tau}, N0={N0}, N1={N1}, d={d}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-7] Non-overlapping Template Matching ---------
def nist_non_overlapping_template(bits: str,
                                  pattern: str = "000000001",
                                  num_blocks: int = 8,
                                  verbose: bool = False) -> float:
    n = len(bits)
    m = len(pattern)
    if num_blocks <= 0 or m <= 0:
        return 0.0
    block_size = n // num_blocks
    if block_size == 0 or m > block_size:
        return 0.0
    pattern_counts = np.zeros(num_blocks, dtype=float)
    for i in range(num_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        j = 0
        while j <= block_size - m:
            if block[j:j+m] == pattern:
                pattern_counts[i] += 1
                j += m
            else:
                j += 1
    mean = (block_size - m + 1) / (2 ** m)
    var = block_size * ((1 / (2 ** m)) - ((2*m - 1) / (2 ** (2*m))))
    if var <= 0:
        return 0.0
    chi_sq = float(np.sum((pattern_counts - mean) ** 2 / var))
    p_val = gammaincc(num_blocks / 2.0, chi_sq / 2.0)
    if verbose:
        print("Non-overlapping Template DEBUG BEGIN:")
        print(f"n={n}, m={m}, blocks={num_blocks}, block_size={block_size}")
        print(f"counts={pattern_counts}, mean={mean}, var={var}, chi^2={chi_sq}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-8] Overlapping Template Matching ---------
def _overlap_prob(u: int, x: float) -> float:
    if u == 0:
        return float(np.exp(-x))
    return float(x * np.exp(-2.0 * x) * (2.0 ** -u) * hyp1f1(u + 1, 2, x))

def nist_overlapping_template(bits: str,
                              pattern_size: int = 9,
                              block_size: int = 1032,
                              verbose: bool = False) -> float:
    n = len(bits)
    if pattern_size <= 0 or block_size <= 0:
        return 0.0
    pattern = "1" * pattern_size
    num_blocks = n // block_size
    if num_blocks == 0:
        return 0.0
    lam = float(block_size - pattern_size + 1) / (2 ** pattern_size)
    eta = lam / 2.0
    piks = [_overlap_prob(u, eta) for u in range(5)]
    s = float(np.sum(piks))
    piks.append(max(0.0, 1.0 - s))  # 5+ kategorisi
    counts = np.zeros(6, dtype=float)
    for i in range(num_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        hits = 0
        for j in range(0, block_size - pattern_size + 1):
            if block[j:j+pattern_size] == pattern:
                hits += 1
        if hits <= 4:
            counts[hits] += 1
        else:
            counts[5] += 1
    expected = num_blocks * np.array(piks, dtype=float)
    expected = np.where(expected < 1e-12, 1e-12, expected)
    chi_sq = float(np.sum((counts - expected) ** 2 / expected))
    p_val = gammaincc(5.0 / 2.0, chi_sq / 2.0)
    if verbose:
        print("Overlapping Template DEBUG BEGIN:")
        print(f"n={n}, m={pattern_size}, block_size={block_size}, blocks={num_blocks}")
        print(f"eta={eta}, piks={piks}, counts={counts}, chi^2={chi_sq}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-9] Universal Statistical (Maurer) ---------
def nist_universal(bits: str, verbose: bool = False) -> float:
    n = len(bits)
    thresholds = [387840, 904960, 2068480, 4654080, 10342400, 22753280,
                  49643520, 107560960, 231669760, 496435200, 1059061760]
    pattern_size = 5 + sum(n >= t for t in thresholds)  # m
    if not (5 < pattern_size < 16):
        if verbose:
            print("Universal: Uygun m aralığı bulunamadı.")
        return 0.0

    ones = "1" * pattern_size
    num_ints = int(ones, 2)  # 2^m - 1
    vobs = np.zeros(num_ints + 1)

    num_blocks = n // pattern_size
    init_bits = 10 * (2 ** pattern_size)
    test_blocks = num_blocks - init_bits
    if test_blocks <= 0:
        if verbose:
            print("Universal: Yeterli test blok yok.")
        return 0.0

    c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * (test_blocks ** (-3 / pattern_size)) / 15
    variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
    expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
    sigma = c * math.sqrt(variance[pattern_size] / (test_blocks + 1e-10))

    cumsum = 0.0
    for i in range(num_blocks):
        block = bits[i*pattern_size:(i+1)*pattern_size]
        try:
            int_rep = int(block, 2)
        except ValueError:
            int_rep = 0
        if i < init_bits:
            vobs[int_rep] = i + 1
        else:
            initial = vobs[int_rep]
            vobs[int_rep] = i + 1
            cumsum += math.log(i - initial + 1, 2)

    phi = float(cumsum / test_blocks)
    stat = abs(phi - expected[pattern_size]) / (math.sqrt(2) * sigma)
    p_val = erfc(stat)
    if verbose:
        print("Universal DEBUG BEGIN:")
        print(f"n={n}, m={pattern_size}, init={init_bits}, test_blocks={test_blocks}")
        print(f"phi={phi}, expected={expected[pattern_size]}, sigma={sigma}, stat={stat}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-10] Linear Complexity (senin formülünle) ---------
def berlekamp_massey(block_data: str) -> int:
    n = len(block_data)
    c = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    c[0] = 1
    b[0] = 1
    L = 0
    m = -1
    i = 0
    int_data = (np.frombuffer(block_data.encode("ascii"), dtype=np.uint8) - 48).astype(int)
    while i < n:
        if L > 0:
            v = int_data[i - L:i][::-1]
            cc = c[1:L + 1]
            d = (int_data[i] + int(np.dot(v, cc) % 2)) % 2
        else:
            d = int_data[i] % 2
        if d == 1:
            temp = c.copy()
            p = np.zeros(n, dtype=int)
            for j in range(L):
                if b[j] == 1:
                    p[j + i - m] = 1
            c = (c + p) % 2
            if L <= i // 2:
                L_new = i + 1 - L
                m = i
                b = temp
                L = L_new
        i += 1
    return int(L)

# --- 1) Linear Complexity olasılıkları ---
def nist_linear_complexity(bits: str, block_size: int = 500, verbose: bool = False) -> float:
    dof = 6
    # Eski: piks = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    piks = [1/96, 1/32, 1/8, 1/2, 1/4, 1/16, 1/48]  # NIST tam değerleri
    t2 = (block_size / 3.0 + 2.0 / 9.0) / (2 ** block_size)
    mean = 0.5 * block_size + (1.0 / 36.0) * (9 + (-1) ** (block_size + 1)) - t2

    num_blocks = len(bits) // block_size
    if num_blocks <= 1:
        if verbose:
            print("Linear Complexity: Geçersiz veri ya da yetersiz blok sayısı.")
        return 0.0

    blocks = [bits[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    complexities = [berlekamp_massey(block) for block in blocks]
    t = [-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9.0) for chunk in complexities]

    vg = np.histogram(t, bins=[-1e12, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1e12])[0][::-1]
    chi_squared = sum(((vg[ii] - num_blocks * piks[ii]) ** 2) / (num_blocks * piks[ii]) for ii in range(7))
    p_val = gammaincc(dof / 2.0, chi_squared / 2.0)

    if verbose:
        print("Linear Complexity DEBUG BEGIN:")
        print(f"blocks={num_blocks}, block_size={block_size}")
        print(f"mean={mean}, counts={vg}, chi^2={chi_squared}, p={p_val}")
        print("DEBUG END.")
    return float(p_val)

# --------- [NIST-11] Serial Test (senin mantığınla) ---------
def nist_serial(bits: str, pattern_length: int = 16, verbose: bool = False):
    n = len(bits)
    if pattern_length <= 0 or n == 0:
        return 0.0, 0.0
    ext = bits + bits[:pattern_length - 1]

    vobs = []
    for i in range(3):
        max_pattern = 2 ** (pattern_length - i)
        vobs.append(np.zeros(max_pattern))

    for i in range(n):
        for j in range(3):
            mlen = pattern_length - j
            patt = ext[i:i + mlen]
            vobs[j][int(patt, 2)] += 1

    sums = np.zeros(3)
    for i in range(3):
        sums[i] = np.sum(vobs[i] ** 2) * (2 ** (pattern_length - i)) / n - n

    nabla_01 = sums[0] - sums[1]
    nabla_02 = sums[0] - 2.0 * sums[1] + sums[2]

    p_value_01 = gammaincc((2 ** (pattern_length - 1)) / 2.0, nabla_01 / 2.0)
    p_value_02 = gammaincc((2 ** (pattern_length - 2)) / 2.0, nabla_02 / 2.0)

    if verbose:
        print('Serial Test DEBUG BEGIN:')
        print(f"\tLength of input: {n}")
        print(f"\tψ values: {sums}")
        print(f"\tNabla values: {nabla_01}, {nabla_02}")
        print(f"\tP-Value 01: {p_value_01}")
        print(f"\tP-Value 02: {p_value_02}")
        print('DEBUG END.')

    return float(p_value_01), float(p_value_02)

# --------- [NIST-12] Approximate Entropy (senin mantığınla) ---------
def nist_approximate_entropy(bits: str, pattern_length: int = 10, verbose: bool = False) -> float:
    n = len(bits)
    if pattern_length <= 0 or n == 0:
        return 0.0
    ext = bits + bits[:pattern_length - 1]

    vobs_01 = Counter()
    vobs_02 = Counter()
    for i in range(n):
        vobs_01[ext[i:i + pattern_length]] += 1
        vobs_02[ext[i:i + pattern_length + 1]] += 1

    sum_01 = sum(v * math.log(v / n) for v in vobs_01.values() if v > 0)
    sum_02 = sum(v * math.log(v / n) for v in vobs_02.values() if v > 0)
    ape = sum_01 / n - sum_02 / n

    xObs = 2.0 * n * (math.log(2.0) - ape)
    p_value = gammaincc(2 ** (pattern_length - 1), xObs / 2.0)

    if verbose:
        print('Approximate Entropy DEBUG BEGIN:')
        print(f"\tLength of input: {n}")
        print(f"\tm={pattern_length}, ApEn={ape}, xObs={xObs}, p={p_value}")
        print('DEBUG END.')

    return float(p_value)

# --------- [NIST-13] Cumulative Sums (senin mantığınla) ---------
def nist_cumulative_sums(bits: str, mode: int = 0, verbose: bool = False) -> float:
    n = len(bits)
    if n == 0:
        return 0.0
    data = bits if mode == 0 else bits[::-1]
    counts = np.zeros(n)
    for i, ch in enumerate(data):
        sub = 1 if ch == '1' else -1
        counts[i] = counts[i - 1] + sub if i > 0 else sub
    z = np.max(np.abs(counts))
    if z == 0:
        return 1.0  # tamamen dengeli ise p=1

    start = int(np.floor(0.25 * np.floor(-n / z + 1)))
    end = int(np.floor(0.25 * np.floor(n / z - 1)))
    terms_one = []
    for k in range(start, end + 1):
        sub = norm.cdf((4 * k - 1) * z / math.sqrt(n))
        terms_one.append(norm.cdf((4 * k + 1) * z / math.sqrt(n)) - sub)

    start = int(np.floor(0.25 * np.floor(-n / z - 3)))
    end = int(np.floor(0.25 * np.floor(n / z) - 1))
    terms_two = []
    for k in range(start, end + 1):
        sub = norm.cdf((4 * k + 1) * z / math.sqrt(n))
        terms_two.append(norm.cdf((4 * k + 3) * z / math.sqrt(n)) - sub)

    p_value = 1.0 - np.sum(np.array(terms_one)) + np.sum(np.array(terms_two))

    if verbose:
        print('Cumulative Sums Test DEBUG BEGIN:')
        print(f"\tLength of input: {n}")
        print(f"\tMode: {mode}")
        print(f"\tValue of z: {z}")
        print(f"\tP-Value: {p_value}")
        print('DEBUG END.')

    return float(p_value)

# --------- [NIST-14] Random Excursions (senin mantığınla) ---------
def _rex_pi_value(k: int, x: int) -> float:
    ax = abs(x)
    if k == 0:
        return 1.0 - 1.0 / (2.0 * ax)
    elif k >= 5:
        return (1.0 / (2.0 * ax)) * ((1.0 - 1.0 / (2.0 * ax)) ** 4)
    else:
        return (1.0 / (4.0 * (x * x))) * ((1.0 - 1.0 / (2.0 * ax)) ** (k - 1))

def nist_random_excursions(bits: str, verbose: bool = False):
    n = len(bits)
    if n == 0:
        return None
    x = np.where((np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48) == 1, 1.0, -1.0)
    cs = np.cumsum(x)
    cs = np.concatenate(([0.0], cs, [0.0]))
    positions = np.where(cs == 0)[0]
    if len(positions) < 2:
        if verbose:
            print("Random Excursions: yeterli cycle yok (J=0).")
        return None

    cycles = [cs[positions[i]:positions[i+1]+1] for i in range(len(positions)-1)]
    num_cycles = len(cycles)

    x_values = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=int)

    # her cycle için state sayıları
    state_count = []
    for cyc in cycles:
        state_count.append([int(np.sum(cyc == s)) for s in x_values])
    state_count = np.clip(np.array(state_count).T, 0, 5)  # shape: 8 x J

    # su[k][state] = 'k kez görülme' sayısı; sonra transpose -> 8 x 6
    su = []
    for k in range(6):
        su.append([(row == k).sum() for row in state_count])
    su = np.array(su).T  # 8 x 6

    # teorik olasılıklar
    pi = [[_rex_pi_value(k, int(s)) for k in range(6)] for s in x_values]
    inner = num_cycles * np.array(pi)  # 8 x 6

    xObs = np.sum((su - inner) ** 2 / inner, axis=1)
    p_values = [gammaincc(2.5, val / 2.0) for val in xObs]

    if verbose:
        print("Random Excursions DEBUG BEGIN:")
        print(f"\tLength: {n}, J (num_cycles): {num_cycles}")
        for idx, s in enumerate(x_values):
            print(f"\tstate={s:>+2}  xObs={xObs[idx]:.6f}  p={p_values[idx]:.6f}")
        print("DEBUG END.")

    labels = ['-4', '-3', '-2', '-1', '+1', '+2', '+3', '+4']
    result = []
    for i, p in enumerate(p_values):
        result.append((labels[i], int(x_values[i]), float(xObs[i]), float(p), bool(p >= 0.01)))
    return result

# --------- [NIST-15] Random Excursions Variant (senin mantığınla) ---------
def nist_random_excursions_variant(bits: str, verbose: bool = False):
    n = len(bits)
    if n == 0:
        return None
    int_data = (np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48).astype(float)
    sum_int = 2.0 * int_data - 1.0
    cs = np.cumsum(sum_int)

    # states: all unique levels with |level| <= 9
    unique_levels = sorted(set(cs.tolist()))
    index = [lvl for lvl in unique_levels if abs(lvl) <= 9]
    if not index:
        if verbose:
            print("Random Excursions Variant: uygun state yok.")
        return None

    # frequency list for each state
    li_data = [[lvl, int(np.sum(cs == lvl))] for lvl in index]

    # j = freq(0) + 1
    j = 0
    for lvl, cnt in li_data:
        if lvl == 0:
            j = cnt
            break
    j = j + 1

    p_values = []
    for lvl in index:
        if lvl != 0:
            den = math.sqrt(2.0 * j * (4.0 * abs(lvl) - 2.0))
            freq = 0
            for x_lvl, cnt in li_data:
                if x_lvl == lvl:
                    freq = cnt
                    break
            p_values.append((lvl, freq, erfc(abs(freq - j) / den)))

    if verbose:
        print("Random Excursions Variant DEBUG BEGIN:")
        print(f"\tLength: {n}, j={j}")
        for lvl, freq, pv in p_values:
            print(f"\tstate={lvl:>+2}  count={freq}  p={pv:.6f}")
        print("DEBUG END.")

    states = [('+' + str(int(v)) if v > 0 else str(int(v))) for v, _, _ in p_values]
    result = []
    for i, (lvl, freq, pv) in enumerate(p_values):
        result.append((states[i], int(lvl), int(freq), float(pv), bool(pv >= 0.01)))
    return result

# --------- Koşum ---------
def run_suite(path: str,
              block_sizes: Iterable[int] = (128, 1000),
              rank_q: int = 32,
              nonover_pattern: str = "000000001",
              nonover_blocks: int = 8,
              over_m: int = 9,
              over_block_size: int = 1032,
              lincomp_block: int = 500,
              serial_m: int = 16,
              apen_m: int = 10,
              verbose: bool = VERBOSE):
    bits = read_bits_from_file(path)

    # [1] Monobit
    print_result("[NIST-1] Monobit", nist_monobit(bits))

    # [2] Block Frequency
    for M in block_sizes:
        print_result(f"[NIST-2] BlockFreq (M={M})", nist_block_frequency(bits, M=M))

    # [3] Runs
    print_result("[NIST-3] Runs", nist_runs(bits, verbose=verbose))

    # [4] Longest Run of Ones in a Block
    print_result("[NIST-4] Longest-Run-of-Ones", nist_longest_run_ones(bits, verbose=verbose))

    # [5] Binary Matrix Rank
    print_result(f"[NIST-5] Binary Matrix Rank ({rank_q}x{rank_q})",
                 nist_binary_matrix_rank(bits, q=rank_q, verbose=verbose))

    # [6] DFT (Spectral)
    print_result("[NIST-6] DFT (Spectral)", nist_spectral(bits, verbose=verbose))

    # [7] Non-overlapping Template Matching
    print_result(f"[NIST-7] Non-overlap Template (pattern='{nonover_pattern}', K={nonover_blocks})",
                 nist_non_overlapping_template(bits, pattern=nonover_pattern,
                                               num_blocks=nonover_blocks, verbose=verbose))

    # [8] Overlapping Template Matching
    print_result(f"[NIST-8] Overlap Template (m={over_m}, M={over_block_size})",
                 nist_overlapping_template(bits, pattern_size=over_m,
                                           block_size=over_block_size, verbose=verbose))

    # [9] Universal Statistical (Maurer)
    print_result("[NIST-9] Universal (Maurer)", nist_universal(bits, verbose=verbose))

    # [10] Linear Complexity
    print_result(f"[NIST-10] Linear Complexity (block={lincomp_block})",
                 nist_linear_complexity(bits, block_size=lincomp_block, verbose=verbose))

    # [11] Serial Test -> iki p-değeri
    p01, p02 = nist_serial(bits, pattern_length=serial_m, verbose=verbose)
    print_result(f"[NIST-11] Serial (m={serial_m})  ψ_m - ψ_m-1", p01)
    print_result(f"[NIST-11] Serial (m={serial_m})  ψ_m - 2ψ_m-1 + ψ_m-2", p02)

    # [12] Approximate Entropy
    print_result(f"[NIST-12] Approximate Entropy (m={apen_m})",
                 nist_approximate_entropy(bits, pattern_length=apen_m, verbose=verbose))

    # [13] Cumulative Sums (forward & backward)
    print_result("[NIST-13] CumuSums (forward)",  nist_cumulative_sums(bits, mode=0, verbose=verbose))
    print_result("[NIST-13] CumuSums (backward)", nist_cumulative_sums(bits, mode=1, verbose=verbose))

    # [14] Random Excursions (8 state ayrı ayrı)
    rex = nist_random_excursions(bits, verbose=verbose)
    if not rex:
        print("[NIST-14] Random Excursions: Not applicable (insufficient zero-crossing cycles).")
    else:
        for label, x_val, x_obs, p, ok in rex:
            print_result(f"[NIST-14] Random Excursions (state={label})", p)

    # [15] Random Excursions Variant (|state|<=9 için)
    rev = nist_random_excursions_variant(bits, verbose=verbose)
    if not rev:
        print("[NIST-15] Random Excursions Variant: Not applicable (no eligible states).")
    else:
        for label, lvl, count, p, ok in rev:
            print_result(f"[NIST-15] Random Excursions Variant (state={label})", p)

if __name__ == "__main__":
    # Kullanım:
    #   python nist_minisuite.py DNA_ModC_1Mbitv.txt
    default_path = "fff.txt"
    path = sys.argv[1] if len(sys.argv) >= 2 else default_path

    BLOCK_SIZES = (128, 1000)
    RANK_Q = 32
    NONOVER_PATTERN = "000000001"
    NONOVER_BLOCKS = 8
    OVER_M = 9
    OVER_BLOCK_SIZE = 1032
    LINCOMP_BLOCK = 500
    SERIAL_M = 16
    APEN_M = 10

    print(f"== NIST mini-suite ==\nDosya: {path}")
    run_suite(path,
              block_sizes=BLOCK_SIZES,
              rank_q=RANK_Q,
              nonover_pattern=NONOVER_PATTERN,
              nonover_blocks=NONOVER_BLOCKS,
              over_m=OVER_M,
              over_block_size=OVER_BLOCK_SIZE,
              lincomp_block=LINCOMP_BLOCK,
              serial_m=SERIAL_M,
              apen_m=APEN_M,
              verbose=VERBOSE)
