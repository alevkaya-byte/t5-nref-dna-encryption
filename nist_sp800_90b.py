

import sys, math, zlib
from typing import List, Tuple, Optional
import numpy as np

# ---------- Yardımcılar ----------
def read_bits(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().replace(" ", "").replace("\n", "")
    if not s or not set(s).issubset({"0","1"}):
        raise ValueError("Dosya yalnızca '0' ve '1' içermelidir.")
    return s

def pack_bits_to_bytes(bitstr: str) -> bytes:
    if not bitstr: return b""
    pad = (-len(bitstr)) % 8
    if pad: bitstr += "0"*pad
    return int(bitstr, 2).to_bytes(len(bitstr)//8, "big")

def freq01(bits: str) -> Tuple[int,int,float]:
    n1 = bits.count("1")
    n = len(bits)
    return n-n1, n1, n1/n

def safe_log2(x: float) -> float:
    return -1e9 if x <= 0 else math.log(x, 2)

# ---------- Min-entropy tahminleri ----------
def mcv_min_entropy(bits: str) -> float:
    n0, n1, p1 = freq01(bits)
    pmax = max(n1, n0) / len(bits)
    return -safe_log2(pmax)

def renyi2_per_bit(bits: str) -> float:
    # Collision/Rényi-2 entropi (bit başına)
    n0, n1, p1 = freq01(bits)
    p0 = 1 - p1
    c = p0*p0 + p1*p1
    return -safe_log2(c)

def ttuple_min_entropy(bits: str, t_vals: List[int] = [1,2,3,4,5,6]) -> Tuple[float, int]:
    n = len(bits)
    best = float("inf"); best_t = 1
    for t in t_vals:
        if n < t: break
        counts = {}
        win = n - t + 1
        for i in range(win):
            s = bits[i:i+t]
            counts[s] = counts.get(s, 0) + 1
        pmax = max(counts.values()) / win
        h = -safe_log2(pmax) / t
        if h < best:
            best = h; best_t = t
    return best, best_t

def markov_min_entropy(bits: str, laplace: float = 1e-6) -> float:
    # 1. dereceden Markov zinciri: H_inf ≈ - Σ_pi log2(max_j P(j|i))
    # pi: durağan dağılım (tahmini), P: geçiş olasılıkları
    if len(bits) < 2: return mcv_min_entropy(bits)
    T = np.zeros((2,2), dtype=np.float64)  # rows: prev, cols: next
    b = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48
    a = b[:-1]; c = b[1:]
    for i,j in zip(a,c):
        T[i,j] += 1
    # Laplace düzeltme
    T += laplace
    P = T / T.sum(axis=1, keepdims=True)
    # Durağan dağılım ~ satır toplamları normlanmış (iyi yaklaşım)
    row = T.sum(axis=1)
    pi = row / row.sum()
    # Per-bit min-entropy Markov
    h = 0.0
    for i in range(2):
        pmax = float(P[i].max())
        h += float(pi[i]) * (-safe_log2(pmax))
    return h

def compression_lower_bound(bits: str) -> float:
    # Bit->byte paketle, sonra zlib; kaba alt-sınır: H >= (8*compressed_bytes)/n_bits
    raw = pack_bits_to_bytes(bits)
    comp = zlib.compress(raw, level=9)
    nbits = len(bits)
    return min(1.0, (len(comp)*8) / max(1, nbits))

# ---------- IID kestirimi ----------
def runs_pvalue(bits: str) -> float:
    n = len(bits)
    pi = bits.count("1") / n
    if n == 0: return 1.0
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0
    # runs say
    v = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            v += 1
    # NIST Runs p-değeri
    num = abs(v - 2*n*pi*(1-pi))
    den = 2*math.sqrt(2*n)*pi*(1-pi)
    from math import erfc
    return erfc(num/den)

def markov_mutual_info(bits: str) -> float:
    # I(X_t ; X_{t-1}) (bit)  -- 2x2 tablo
    if len(bits) < 2: return 0.0
    b = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48
    a = b[:-1]; c = b[1:]
    joint = np.zeros((2,2), dtype=np.float64)
    for i,j in zip(a,c):
        joint[i,j] += 1
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    eps = 1e-12
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pxy = joint[i,j]
            if pxy > 0:
                mi += pxy * (math.log(pxy/(px[i,0]*py[0,j]+eps), 2))
    return float(mi)

def iid_judgement(bits: str) -> Tuple[bool, dict]:
    # Basit karar: Monobit ve Runs p>=0.01, I(X_t;X_{t-1}) < 0.005 bit => "muhtemelen IID"
    from math import erfc
    n = len(bits)
    p1 = bits.count("1")/n
    # Monobit p
    s = sum(1 if ch=="1" else -1 for ch in bits)
    sobs = s / math.sqrt(n)
    p_mono = erfc(abs(sobs)/math.sqrt(2.0))
    p_runs = runs_pvalue(bits)
    imi = markov_mutual_info(bits)
    decision = (p_mono >= 0.01) and (p_runs >= 0.01) and (imi < 0.005)
    return decision, {"p_monobit": p_mono, "p_runs": p_runs, "I_markov_bits": imi}

# ---------- Health Tests (SP 800-90B çevrim-içi testlerine benzer) ----------
def rct_threshold(n: int, alpha: float = 1e-6) -> int:
    # Yaklaşık: Pr(max-run >= r) <= n * 2^{-r}  -> r >= ceil(log2(n/alpha))
    return int(math.ceil(math.log2(max(1.0, n/alpha))))

def repetition_count_test(bits: str, alpha: float = 1e-6) -> Tuple[bool, int, int]:
    n = len(bits)
    thr = rct_threshold(n, alpha=alpha)
    max_run = 0
    run = 1
    for i in range(1, n):
        if bits[i] == bits[i-1]:
            run += 1
            if run > max_run: max_run = run
        else:
            run = 1
    return (max_run < thr), max_run, thr

def binom_sf_ge(k: int, n: int, p: float = 0.5) -> float:
    # P[X >= k] için log-binom toplayıcı (iki uçlu için aşağıda katlanacağız)
    # Sayısal kararlılık için log-uzay
    from math import lgamma, log, exp
    def logC(n,k): return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
    s = -1e300
    for i in range(k, n+1):
        t = logC(n,i) + i*math.log(p) + (n-i)*math.log(1-p)
        s = t if s < -1e200 else s + math.log1p(math.exp(t-s))
    return 0.0 if s < -1e200 else math.exp(s)

def adaptive_proportion_test(bits: str, W: int = 512, alpha: float = 1e-6) -> Tuple[bool, int, Tuple[int,int], float]:
    # Her pencere için #1 say; en uç pencerenin iki-uçlu binom p-değerine bak ve Bonferroni düzelt
    n = len(bits)
    if n < W: 
        return True, 0, (0,0), 1.0
    ones = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48
    worst_p = 1.0; worst_k = 0; worst_idx = 0
    m = n // W
    for w in range(m):
        seg = ones[w*W:(w+1)*W]
        k = int(seg.sum())
        # iki uçlu p = 2 * min(P[X >= k], P[X <= k]) (simetriyle)
        upper = binom_sf_ge(k, W, 0.5)
        lower = binom_sf_ge(W-k, W, 0.5)  # P[X <= k] = P[W-X >= W-k]
        p = 2.0 * min(upper, lower)
        if p < worst_p:
            worst_p = p; worst_k = k; worst_idx = w
    # Bonferroni: m pencerede alpha/m
    passed = worst_p >= (alpha / max(1, m))
    return passed, worst_idx, (worst_k, W), worst_p

# ---------- Ana ----------
def main(path: str):
    bits = read_bits(path)
    n = len(bits)
    n0, n1, p1 = freq01(bits)
    print(f"== SP 800-90B (esinli) ==\nDosya: {path}\nNbit={n:,}  p(1)={p1:.6f}")

    # Entropi tahminleri
    h_mcv = mcv_min_entropy(bits)
    h_r2  = renyi2_per_bit(bits)
    h_markov = markov_min_entropy(bits)
    h_t, tbest = ttuple_min_entropy(bits, [1,2,3,4,5,6])
    h_comp = compression_lower_bound(bits)

    print("\n-- Min-Entropy (bit başına) --")
    print(f"MCV           : {h_mcv:.6f}")
    print(f"t-tuple (best): {h_t:.6f}  (t={tbest})")
    print(f"Markov(1)     : {h_markov:.6f}")
    print(f"Collision H2  : {h_r2:.6f}  (karşılaştırma amaçlı)")
    print(f"Compression LB: {h_comp:.6f}  (zlib alt-sınır)")

    # IID kestirimi
    iid_ok, det = iid_judgement(bits)
    print("\n-- IID kestirimi (heuristic) --")
    print(f"Monobit p     : {det['p_monobit']:.6g}")
    print(f"Runs p        : {det['p_runs']:.6g}")
    print(f"I(X_t;X_t-1)  : {det['I_markov_bits']:.6f} bits")
    print(f"Karar         : {'Muhtemelen IID' if iid_ok else 'Non-IID belirtisi var'}")

    # Health tests
    rct_ok, max_run, thr = repetition_count_test(bits, alpha=1e-6)
    apt_ok, widx, (k,W), pmin = adaptive_proportion_test(bits, W=512, alpha=1e-6)

    print("\n-- Health Tests (SP 800-90B ruhu) --")
    print(f"RCT : max_run={max_run}  eşik≈{thr}  -> {'PASS' if rct_ok else 'FAIL'}")
    print(f"APT : en kötü pencere #{widx}  k={k}/{W}  p(two-sided)={pmin:.3g}  -> {'PASS' if apt_ok else 'FAIL'}")

    # Özet alt-sınır (mühafazakâr): MCV vs t-tuple vs Markov içinden en küçüğü
    h_min_lb = min(h_mcv, h_t, h_markov, 1.0)
    print("\n== Alt-sınır özet ==")
    print(f"H_min (bit) ≥ {h_min_lb:.6f}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>=2 else "outputs/t5_noref_dna_1m.txt"
    main(path)
