"""
Faz K: dorduncu hakem turunun istedigi kosular.

(1) FAYDA, on bolme uzerinden eşli fark ve aralik.
    Ayrica mekanizma kontrolu: ayni gurultu satirlari "sag kalan" etiketiyle
    eklenirse kazanc suruyor mu? Suruyorsa etiket degil duzenlilestirme etkisi.

(2) Gurultu ureticisini bolme basina 50 kez cekip ortalamasini almak.
    Farkin SD'si dusuyor mu? Dusmuyorsa oneri "analitik null bandi yeterli"
    diye sadelesir.

(3) Tablo 6'daki 800 adimli kirpmasiz referans icin PR-AUC.

Cikti: phase_k.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

os.chdir("/home/claude/run")
SPLITS = list(range(10))
N_SYNTH = 479
OUTK = {}

raw = pd.read_parquet("patient_matrix_24h.parquet")
sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
TARGET = "mortality"
D = raw.copy(); D["subject_id"] = sub["subject_id"].values; D[TARGET] = sub[TARGET].values
OUT = {"hospital_expire_flag", "mortality"}
for c in [c for c in raw.columns if c not in OUT and not pd.api.types.is_numeric_dtype(D[c])]:
    v = sorted(D[c].dropna().unique()); D[c] = D[c].map({v[0]: 0.0, v[1]: 1.0})
FEATS = [c for c in D.columns if c not in ({"subject_id", TARGET} | OUT)]
D[FEATS] = D[FEATS].astype("float64")
PT = D.groupby("subject_id")[TARGET].max().reset_index()


def make_split(seed):
    r = np.random.default_rng(seed); ids = []
    for lab in [0, 1]:
        a = PT[PT[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * .25)])
    m = D["subject_id"].isin(set(ids))
    tr, te = D[~m].copy(), D[m].copy()
    imp = KNNImputer(n_neighbors=10)
    tr[FEATS] = imp.fit_transform(tr[FEATS]); te[FEATS] = imp.transform(te[FEATS])
    return tr, te


def noise_rows(members, seed, n=N_SYNTH):
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: r.uniform(members[c].min(), members[c].max(), n) for c in FEATS})


def prauc(Xtr, ytr, Xte, yte, seed, balanced=False):
    clf = HistGradientBoostingClassifier(
        random_state=seed, class_weight=("balanced" if balanced else None))
    clf.fit(Xtr, ytr)
    return float(average_precision_score(yte, clf.predict_proba(Xte)[:, 1]))


# ================= (1) fayda, on bolme, esli fark + etiket kontrolu =========
print("=" * 74)
print("(1) FAYDA: on bolme, esli fark ve etiket kontrolu", flush=True)
rows = []
for sp in SPLITS:
    tr, te = make_split(sp)
    members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
    Xtr, ytr = tr[FEATS].values, tr[TARGET].values
    Xte, yte = te[FEATS].values, te[TARGET].values
    syn = noise_rows(members, sp)

    base = prauc(Xtr, ytr, Xte, yte, sp)
    bal = prauc(Xtr, ytr, Xte, yte, sp, balanced=True)
    # gurultu satirlari OLEN etiketiyle (makaledeki protokol)
    Xd = np.vstack([Xtr, syn.values]); yd = np.r_[ytr, np.ones(len(syn))]
    died = prauc(Xd, yd, Xte, yte, sp)
    # ayni satirlar SAG KALAN etiketiyle (hakemin onerdigi kontrol)
    Xs = np.vstack([Xtr, syn.values]); ys = np.r_[ytr, np.zeros(len(syn))]
    surv = prauc(Xs, ys, Xte, yte, sp)

    rows.append({"split": sp, "baseline": base, "balanced": bal,
                 "noise_deceased": died, "noise_survivor": surv})
    print(f"   bolme {sp}: taban {base:.3f} | agirlikli {bal:.3f} | "
          f"gurultu-olen {died:.3f} | gurultu-sagkalan {surv:.3f}", flush=True)

A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "split"}
def ci(x):
    b = np.array([np.mean(np.random.default_rng(i).choice(x, len(x))) for i in range(4000)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

summ = {}
for k, v in A.items():
    lo, hi = ci(v)
    summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci": [lo, hi]}
d_died = A["noise_deceased"] - A["baseline"]
d_surv = A["noise_survivor"] - A["baseline"]
d_bal = A["balanced"] - A["baseline"]
for lab, d in [("noise_deceased_minus_baseline", d_died),
               ("noise_survivor_minus_baseline", d_surv),
               ("balanced_minus_baseline", d_bal)]:
    lo, hi = ci(d)
    summ[lab] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci": [lo, hi]}
OUTK["utility_splits"] = {"rows": rows, "summary": summ}
json.dump(OUTK, open("phase_k.json", "w"), indent=1)

print("\n   ozet (on bolme, esli):")
for k in ["baseline", "balanced", "noise_deceased", "noise_survivor"]:
    s_ = summ[k]; print(f"     {k:18s} {s_['mean']:.3f}  %95 [{s_['ci'][0]:.3f}, {s_['ci'][1]:.3f}]")
for k in ["noise_deceased_minus_baseline", "noise_survivor_minus_baseline",
          "balanced_minus_baseline"]:
    s_ = summ[k]; print(f"     {k:32s} {s_['mean']:+.3f}  %95 [{s_['ci'][0]:+.3f}, {s_['ci'][1]:+.3f}]")

# ================= (2) gurultu ureticisini 50 kez ortala ====================
print("\n" + "=" * 74)
print("(2) gurultu referansini 50 cekimle ortalamak", flush=True)
from scipy.stats import rankdata
from scipy.stats import norm as _norm

def copula_rows(members, seed):
    rng = np.random.default_rng(seed); X = members[FEATS].values.astype("float64")
    n, d = X.shape
    u = np.column_stack([(rankdata(X[:, j]) - .5) / n for j in range(d)])
    z = _norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(d)
    us = _norm.cdf(rng.multivariate_normal(np.zeros(d), cov, size=N_SYNTH))
    return pd.DataFrame(np.column_stack(
        [np.quantile(X[:, j], np.clip(us[:, j], 0, 1)) for j in range(d)]), columns=FEATS)

res2 = []
for sp in SPLITS:
    tr, te = make_split(sp)
    members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
    nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
    SC = StandardScaler().fit(members); M = SC.transform(members); Xn = SC.transform(nonmembers)
    y = np.r_[np.ones(len(M)), np.zeros(len(Xn))]

    def auc(syn):
        k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
        return float(roc_auc_score(y, -np.r_[k.kneighbors(M)[0].ravel(),
                                             k.kneighbors(Xn)[0].ravel()]))
    one = auc(noise_rows(members, sp))
    many = float(np.mean([auc(noise_rows(members, 1000 * sp + j)) for j in range(50)]))
    cop = auc(copula_rows(members, sp))
    res2.append({"split": sp, "noise_1": one, "noise_50": many, "copula": cop})
    print(f"   bolme {sp}: gurultu(1) {one:.3f} | gurultu(50 ort) {many:.3f} | kopula {cop:.3f}",
          flush=True)

n1 = np.array([r["noise_1"] for r in res2]); n50 = np.array([r["noise_50"] for r in res2])
cp = np.array([r["copula"] for r in res2])
OUTK["noise_averaging"] = {
    "rows": res2,
    "copula_sd": float(cp.std(ddof=1)),
    "diff_vs_noise1_sd": float((cp - n1).std(ddof=1)),
    "diff_vs_noise50_sd": float((cp - n50).std(ddof=1)),
    "noise1_sd": float(n1.std(ddof=1)), "noise50_sd": float(n50.std(ddof=1)),
    "noise50_mean": float(n50.mean())}
r2 = OUTK["noise_averaging"]
print(f"\n   kopulanin kendi SD'si          {r2['copula_sd']:.3f}")
print(f"   farkin SD'si (tek cekim)       {r2['diff_vs_noise1_sd']:.3f}")
print(f"   farkin SD'si (50 cekim ort.)   {r2['diff_vs_noise50_sd']:.3f}")
print(f"   gurultu SD: tek {r2['noise1_sd']:.3f} -> 50 ort {r2['noise50_sd']:.3f} "
      f"(ortalama {r2['noise50_mean']:.3f})")
json.dump(OUTK, open("phase_k.json", "w"), indent=1)
print("\nbitti", flush=True)
