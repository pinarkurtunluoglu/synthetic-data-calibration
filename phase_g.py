"""
Faz G: "derin modelleri yeterince egitmediniz" itirazinin testi.

Her derin model birkac egitim butcesinde calistirilir. Soru:
herhangi bir yapilandirma, kopulanin sadakatine, taban duzeyinde
sizinti ile ulasabiliyor mu?

Cikti: phase_g.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SEEDS = [0, 1, 2]

raw = pd.read_parquet("patient_matrix_24h.parquet")
sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
TARGET = "mortality"
d = raw.copy(); d["subject_id"] = sub["subject_id"].values; d[TARGET] = sub[TARGET].values
OUT = {"hospital_expire_flag", "mortality"}
for c in [c for c in raw.columns if c not in OUT and not pd.api.types.is_numeric_dtype(d[c])]:
    v = sorted(d[c].dropna().unique()); d[c] = d[c].map({v[0]: 0.0, v[1]: 1.0})
FEATS = [c for c in d.columns if c not in ({"subject_id", TARGET} | OUT)]
d[FEATS] = d[FEATS].astype("float64")
pt = d.groupby("subject_id")[TARGET].max().reset_index()
r = np.random.default_rng(42); ids = []
for lab in [0, 1]:
    a = pt[pt[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
    ids.extend(a[:int(len(a) * .25)])
m = d["subject_id"].isin(set(ids)); tr, te = d[~m].copy(), d[m].copy()
imp = KNNImputer(n_neighbors=10)
tr[FEATS] = imp.fit_transform(tr[FEATS]); te[FEATS] = imp.transform(te[FEATS])
members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
N_SYNTH = int((tr[TARGET] == 0).sum())
SC = StandardScaler().fit(members); M = SC.transform(members)
_nn = NearestNeighbors(n_neighbors=2).fit(M); dr = _nn.kneighbors(M)[0][:, 1]
D_MED, D_P5 = float(np.median(dr)), float(np.percentile(dr, 5))
print(f"uye {len(members)} | sentetik {N_SYNTH}", flush=True)

from ctgan import CTGAN, TVAE
T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def g_ctgan(seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    m = CTGAN(epochs=epochs, verbose=False); m.fit(members); return m.sample(N_SYNTH)[FEATS]

def g_tvae(seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TVAE(epochs=epochs); m.fit(members); return m.sample(N_SYNTH)[FEATS]

def g_ddpm(seed, steps):
    torch.manual_seed(seed); sc = StandardScaler().fit(members)
    x0 = torch.tensor(sc.transform(members), dtype=torch.float32); dd = x0.shape[1]
    net = nn.Sequential(nn.Linear(dd + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dd))
    f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        t = torch.randint(0, T, (len(x0),)); nz = torch.randn_like(x0); a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * nz
        opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
    with torch.no_grad():
        x = torch.randn(N_SYNTH, dd)
        for t in reversed(range(T)):
            e = f(x, torch.full((N_SYNTH,), t))
            x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
            if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
    return pd.DataFrame(sc.inverse_transform(x.numpy()), columns=FEATS)

def g_copula(seed, _=None):
    rng = np.random.default_rng(seed); X = members.values.astype(float); n, k = X.shape
    u = np.column_stack([(rankdata(X[:, j]) - .5) / n for j in range(k)])
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)); cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(k)
    zs = rng.multivariate_normal(np.zeros(k), cov, size=N_SYNTH); us = norm.cdf(zs)
    return pd.DataFrame(np.column_stack([np.quantile(X[:, j], np.clip(us[:, j], 0, 1))
                                         for j in range(k)]), columns=FEATS)

def g_noise(seed, _=None):
    rg = np.random.default_rng(seed)
    return pd.DataFrame({c: rg.uniform(members[c].min(), members[c].max(), N_SYNTH) for c in FEATS})

def mean_ks(syn): return float(np.mean([ks_2samp(syn[c], members[c]).statistic for c in FEATS]))
def nn_auc(syn):
    k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
    dm = k.kneighbors(M)[0].ravel(); dn = k.kneighbors(SC.transform(nonmembers))[0].ravel()
    return float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))
def dup_rate(syn):
    k = NearestNeighbors(n_neighbors=1).fit(M)
    return float(np.mean(k.kneighbors(SC.transform(syn))[0].ravel() < D_P5))
def det_auc(syn, seed):
    n = min(len(members), len(syn))
    rs = members.sample(n, random_state=seed); ss = syn.sample(n, random_state=seed)
    X = np.vstack([rs.values, ss.values]); y = np.r_[np.ones(n), np.zeros(n)]
    sk = StratifiedKFold(5, shuffle=True, random_state=seed); p = np.zeros(len(y))
    for a, b in sk.split(X, y):
        p[b] = HistGradientBoostingClassifier(random_state=seed).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return float(roc_auc_score(y, p))

CONFIGS = [("CTGAN", g_ctgan, e) for e in (100, 300, 1000)] + \
          [("TVAE", g_tvae, e) for e in (100, 300, 1000)] + \
          [("TabDDPM", g_ddpm, s) for s in (500, 3000, 10000)] + \
          [("Gauss kopulasi", g_copula, None), ("Gurultu (taban)", g_noise, None)]

res = {}; t0 = time.time()
for name, fn, budget in CONFIGS:
    key = f"{name}" + (f" ({budget})" if budget else "")
    acc = {k: [] for k in ["ks", "det", "nn", "dup"]}
    for s in SEEDS:
        try:
            syn = fn(s, budget)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  {key} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50: continue
        acc["ks"].append(mean_ks(syn)); acc["det"].append(det_auc(syn, s))
        acc["nn"].append(nn_auc(syn)); acc["dup"].append(dup_rate(syn))
    if not acc["ks"]: continue
    res[key] = {k: float(np.mean(v)) for k, v in acc.items()}
    res[key]["det_dist"] = abs(res[key]["det"] - 0.5)
    r_ = res[key]
    print(f"  {key:22s} KS {r_['ks']:.3f} | ayirt-uzaklik {r_['det_dist']:.3f} | "
          f"sizinti {r_['nn']:.3f} | yakin-kopya {r_['dup']:.3f}", flush=True)
    json.dump(res, open("phase_g.json", "w"), indent=1)

print(f"\nsure {(time.time()-t0)/60:.1f} dk", flush=True)
