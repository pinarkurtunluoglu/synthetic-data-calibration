"""
Tek bir dogruluk kaynagi: butun fazlarin ciktilarini birlestirir,
orneklem tabanini ve kesin permutasyon p-degerlerini hesaplar.
Cikti: numbers_of_record.json + ekrana okunabilir ozet.
"""
import json, itertools, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import KNNImputer

WD = "/home/claude/tez/run"; os.chdir(WD)
J = {f: json.load(open(f + ".json")) for f in
     ["phase_ab", "phase_c", "phase_d", "phase_e", "phase_f", "phase_g"]}

# ---------- kohort buyuklukleri -------------------------------------------
def tamponade_sizes():
    raw = pd.read_parquet("patient_matrix_24h.parquet")
    sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
    T = "mortality"
    d = raw.copy(); d["subject_id"] = sub["subject_id"].values; d[T] = sub[T].values
    OUT = {"hospital_expire_flag", "mortality"}
    FE = [c for c in d.columns if c not in ({"subject_id", T} | OUT)]
    pt = d.groupby("subject_id")[T].max().reset_index()
    r = np.random.default_rng(42); ids = []
    for lab in [0, 1]:
        a = pt[pt[T] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * .25)])
    m = d["subject_id"].isin(set(ids)); tr, te = d[~m], d[m]
    return dict(n_adm=len(d), n_pat=d.subject_id.nunique(), n_feat=len(FE),
                mortality=float(d[T].mean()),
                tr_adm=len(tr), tr_pat=tr.subject_id.nunique(),
                te_adm=len(te), te_pat=te.subject_id.nunique(),
                members=int((tr[T] == 1).sum()), nonmembers=int((te[T] == 1).sum()),
                n_synth=int((tr[T] == 0).sum()))

def aortic_sizes():
    tr = pd.read_parquet("aortic_train.parquet"); te = pd.read_parquet("aortic_test.parquet")
    T = "mortality"
    FE = [c for c in tr.columns if c not in {"subject_id", T}]
    return dict(n_feat=len(FE), tr_adm=len(tr), te_adm=len(te),
                members=int((tr[T] == 1).sum()), nonmembers=int((te[T] == 1).sum()),
                n_synth=int((tr[T] == 0).sum()))

S = {"tamponad": tamponade_sizes(), "aort": aortic_sizes()}

# ---------- KS orneklem tabani --------------------------------------------
# iki bagimsiz surekli ornek icin beklenen KS'in dagilimdan bagimsiz olcegi
def ks_floor(n, m):
    return 0.8687 * np.sqrt((n + m) / (n * m))

S["tamponad"]["ks_floor_in"] = ks_floor(S["tamponad"]["members"], S["tamponad"]["n_synth"])
S["tamponad"]["ks_floor_out"] = ks_floor(S["tamponad"]["nonmembers"], S["tamponad"]["n_synth"])
S["aort"]["ks_floor_in"] = ks_floor(S["aort"]["members"], S["aort"]["n_synth"])
S["aort"]["ks_floor_out"] = ks_floor(S["aort"]["nonmembers"], S["aort"]["n_synth"])

# ---------- kesin permutasyon testi ---------------------------------------
def exact_spearman(x, y):
    """kucuk n icin tum permutasyonlar uzerinden kesin p"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho0 = spearmanr(x, y).statistic
    n = len(x)
    cnt = 0; tot = 0
    for p in itertools.permutations(range(n)):
        r = spearmanr(x, y[list(p)]).statistic
        tot += 1
        if abs(r) >= abs(rho0) - 1e-12: cnt += 1
    return float(rho0), cnt / tot, tot

ARCH = ["SMOTE", "ADASYN", "TabDDPM", "CTGAN", "TVAE", "Gaussian copula", "Vanilla GAN"]

CORR = {}
for cohort, src in [("tamponad", J["phase_d"]["res"]), ("aort", J["phase_e"]["res"])]:
    res = {k: src[k] for k in ARCH if k in src}
    names = list(res)
    nn = [res[a]["nn"] for a in names]
    dom = [res[a]["dom"] for a in names]
    out = {}
    for lab, key in [("ks_in", "ks_in"), ("dup", "dup"), ("ratio", "ratio")]:
        x = [res[a][key] for a in names]
        r1, p1, tot = exact_spearman(x, nn)
        r2, p2, _ = exact_spearman(x, dom)
        out[lab] = {"vs_nn": {"rho": r1, "p_exact": p1},
                    "vs_dom": {"rho": r2, "p_exact": p2}, "n": len(names), "perms": tot}
    CORR[cohort] = out

OUT = {"sizes": S, "correlations": CORR,
       "tamponade_main": J["phase_ab"], "utility": J["phase_c"]["C1"],
       "dp": J["phase_c"]["C2"], "dp_meta": J["phase_c"]["meta"],
       "proximity_tamponade": J["phase_d"]["res"], "proximity_aortic": J["phase_e"]["res"],
       "detection": J["phase_f"], "budget": J["phase_g"]}
json.dump(OUT, open("numbers_of_record.json", "w"), indent=1)

# ---------- ekran ozeti ----------------------------------------------------
def f(x, d=3): return f"{x:.{d}f}"
print("=" * 78)
print("KOHORT BUYUKLUKLERI")
for k, v in S.items():
    print(f"  {k}: {v}")
print()
print("KS ORNEKLEM TABANI (dagilimdan bagimsiz)")
for k in S:
    print(f"  {k}: egitim-ici {f(S[k]['ks_floor_in'])}  egitim-disi {f(S[k]['ks_floor_out'])}")
print()
print("=" * 78)
print("TAMPONAD ANA TABLO (10 tohum)")
print(f"{'uretec':18s} {'KS_ic':>7s} {'KS_dis':>7s} {'EYK':>7s} {'kayit %95':>18s} "
      f"{'DOMIAS':>7s} {'yakinlik':>9s} {'yakin-kopya':>12s}")
ab, pd_ = J["phase_ab"], J["phase_d"]["res"]
for g in ["Copy (ceiling)", "SMOTE", "ADASYN", "TabDDPM", "Gaussian copula", "TVAE",
          "Vanilla GAN", "CTGAN", "Noise (floor)"]:
    a = ab[g]; p = pd_[g]
    ci = f"[{f(a['nn_rec_ci'][0])}, {f(a['nn_rec_ci'][1])}]"
    print(f"{g:18s} {f(a['ks_in']):>7s} {f(a['ks_out']):>7s} {f(a['nn']):>7s} {ci:>18s} "
          f"{f(a['dom']):>7s} {f(p['ratio']):>9s} {f(p['dup']):>12s}")
print()
print("guven araligi genisleme (kayit / tohum):")
w = [( (ab[g]['nn_rec_ci'][1]-ab[g]['nn_rec_ci'][0]) /
       max(1e-9, ab[g]['nn_seed_ci'][1]-ab[g]['nn_seed_ci'][0]) ) for g in ab]
print(f"  medyan {np.median(w):.1f}x, ortalama {np.mean(w):.1f}x")
print()
print("=" * 78)
print("FAYDA (PR-AUC, gercek sinama kumesi)")
c1 = J["phase_c"]["C1"]
rows = sorted([(v["prauc"], k) for k, v in c1.items() if k != "_baseline_no_synth"], reverse=True)
print(f"  {'taban cizgisi (sentetik yok)':30s} {f(c1['_baseline_no_synth']['prauc'])}")
for v, k in rows:
    print(f"  {k:30s} {f(v)}")
print()
print("=" * 78)
print("EGITIM BUTCESI (difuzyon)")
for k in ["TabDDPM (500)", "TabDDPM (3000)", "TabDDPM (10000)"]:
    g = J["phase_g"][k]
    print(f"  {k:18s} KS {f(g['ks'])}  EYK {f(g['nn'])}  yakin-kopya {f(g['dup'])}")
print()
print("=" * 78)
print("KORELASYONLAR (7 mimari, kesin permutasyon testi)")
for c in CORR:
    print(f"  {c}:")
    for lab in CORR[c]:
        a = CORR[c][lab]
        print(f"    {lab:7s} vs EYK  rho {a['vs_nn']['rho']:+.3f} p {a['vs_nn']['p_exact']:.4f}"
              f"   | vs DOMIAS rho {a['vs_dom']['rho']:+.3f} p {a['vs_dom']['p_exact']:.4f}")
print()
print("=" * 78)
print("DP butceleri")
for k, v in J["phase_c"]["C2"].items():
    print(f"  {k:10s} epsilon {v['epsilon']:.1f}  KS_ic {f(v['ks_in'])}  EYK {f(v['nn'])}")
print(f"  meta: {J['phase_c']['meta']}")
print()
print("=" * 78)
print("AYIRT EDICI TESTI (0.5 = ayirt edilemiyor)")
for c in ["tamponad", "aort"]:
    print(f"  {c}:")
    for g, v in J["phase_f"][c].items():
        print(f"    {g:18s} AUC {f(v['detection_auc'])} (sd {f(v['detection_sd'])})")
print("\nyazildi: numbers_of_record.json")
