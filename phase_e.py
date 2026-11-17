"""
Faz E: yakinlik olcutunun ikinci kohortta (aort diseksiyonu) replikasyonu.
Veri zaten bolunmus ve doldurulmus halde geliyor.
Cikti: phase_e.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, spearmanr, norm, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SEEDS = list(range(10))

tr = pd.read_parquet("aortic_train.parquet")
te = pd.read_parquet("aortic_test.parquet")
TARGET = "mortality"
FEATS = [c for c in tr.columns if c not in {"subject_id", TARGET}]
for d in (tr, te):
    for c in FEATS:
        if not pd.api.types.is_numeric_dtype(d[c]):
            v = sorted(d[c].dropna().unique()); d[c] = d[c].map({v[0]: 0.0, v[1]: 1.0})
    # nullable Int64 gibi tipler .values uzerinde object dtype uretiyor
    d[FEATS] = d[FEATS].astype("float64")

members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
ref_pop = pd.concat([tr[tr[TARGET] == 0], te[te[TARGET] == 0]])[FEATS]
N_SYNTH = int((tr[TARGET] == 0).sum())
print(f"uye {len(members)} | uye-olmayan {len(nonmembers)} | sentetik {N_SYNTH} | "
      f"ozellik {len(FEATS)}", flush=True)

SC = StandardScaler().fit(members); M = SC.transform(members)
_nn = NearestNeighbors(n_neighbors=2).fit(M)
d_real = _nn.kneighbors(M)[0][:, 1]
D_MED = float(np.median(d_real)); D_P5 = float(np.percentile(d_real, 5))
print(f"uyeler arasi EYK: medyan {D_MED:.3f}, 5. yuzdelik {D_P5:.3f}", flush=True)

from imblearn.over_sampling import SMOTE, ADASYN
from ctgan import CTGAN, TVAE

def _pick(syn, seed):
    i = np.random.default_rng(seed).choice(len(syn), N_SYNTH, replace=len(syn) < N_SYNTH)
    return pd.DataFrame(syn[i], columns=FEATS)

def g_smote(seed):
    X, y = tr[FEATS].values, tr[TARGET].values
    sm = SMOTE(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
               k_neighbors=min(5, int((y == 1).sum()) - 1))
    Xr, _ = sm.fit_resample(X, y); return _pick(Xr[len(X):], seed)

def g_adasyn(seed):
    X, y = tr[FEATS].values, tr[TARGET].values
    ad = ADASYN(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                n_neighbors=min(5, int((y == 1).sum()) - 1))
    Xr, _ = ad.fit_resample(X, y); return _pick(Xr[len(X):], seed)

def g_gan(seed, epochs=2000, z=16):
    torch.manual_seed(seed); sc = StandardScaler().fit(members)
    X = torch.tensor(sc.transform(members), dtype=torch.float32); d = X.shape[1]
    G = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d))
    D = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(.5, .999))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(.5, .999)); bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        f = G(torch.randn(len(X), z)); od.zero_grad()
        (bce(D(X), torch.ones(len(X), 1)) + bce(D(f.detach()), torch.zeros(len(X), 1))).backward(); od.step()
        og.zero_grad(); bce(D(f), torch.ones(len(X), 1)).backward(); og.step()
    with torch.no_grad(): o = G(torch.randn(N_SYNTH, z)).numpy()
    return pd.DataFrame(sc.inverse_transform(o), columns=FEATS)

def g_ctgan(seed, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    m = CTGAN(epochs=epochs, verbose=False); m.fit(members); return m.sample(N_SYNTH)[FEATS]

def g_tvae(seed, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TVAE(epochs=epochs); m.fit(members); return m.sample(N_SYNTH)[FEATS]

T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def g_ddpm(seed, epochs=3000):
    torch.manual_seed(seed); sc = StandardScaler().fit(members)
    x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
    net = nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))
    f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        t = torch.randint(0, T, (len(x0),)); nz = torch.randn_like(x0); a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * nz
        opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
    with torch.no_grad():
        x = torch.randn(N_SYNTH, d)
        for t in reversed(range(T)):
            e = f(x, torch.full((N_SYNTH,), t))
            x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
            if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
    return pd.DataFrame(sc.inverse_transform(x.numpy()), columns=FEATS)

def g_copula(seed):
    rng = np.random.default_rng(seed)
    X = members[FEATS].values; n, d = X.shape
    u = np.column_stack([(rankdata(X[:, j]) - 0.5) / n for j in range(d)])
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(d)
    zs = rng.multivariate_normal(np.zeros(d), cov, size=N_SYNTH)
    us = norm.cdf(zs)
    out = np.column_stack([np.quantile(X[:, j], np.clip(us[:, j], 0, 1)) for j in range(d)])
    return pd.DataFrame(out, columns=FEATS)

def g_copy(seed):
    i = np.random.default_rng(seed).choice(len(members), N_SYNTH, replace=True)
    return members.iloc[i].reset_index(drop=True)

def g_noise(seed):
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: r.uniform(members[c].min(), members[c].max(), N_SYNTH) for c in FEATS})

GENS = {"Copy (ceiling)": g_copy, "SMOTE": g_smote, "ADASYN": g_adasyn, "TabDDPM": g_ddpm,
        "CTGAN": g_ctgan, "TVAE": g_tvae, "Gaussian copula": g_copula,
        "Vanilla GAN": g_gan, "Noise (floor)": g_noise}

def proximity(syn):
    S = SC.transform(syn[FEATS])
    k = NearestNeighbors(n_neighbors=1).fit(M)
    d = k.kneighbors(S)[0].ravel()
    return float(np.median(d) / D_MED), float(np.mean(d < D_P5))

def nn_auc(syn):
    k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
    dm = k.kneighbors(M)[0].ravel(); dn = k.kneighbors(SC.transform(nonmembers))[0].ravel()
    return float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))

def dom_auc(syn):
    kr = KernelDensity(bandwidth=1.0).fit(SC.transform(ref_pop))
    ks = KernelDensity(bandwidth=1.0).fit(SC.transform(syn))
    sm = ks.score_samples(M) - kr.score_samples(M)
    Xn = SC.transform(nonmembers); sn = ks.score_samples(Xn) - kr.score_samples(Xn)
    return float(roc_auc_score(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn]))

def mean_ks(syn, real):
    return float(np.mean([ks_2samp(syn[c], real[c]).statistic for c in FEATS]))

res = {}; t0 = time.time()
for name, fn in GENS.items():
    acc = {k: [] for k in ["ratio", "dup", "nn", "dom", "ks_in", "ks_out"]}
    for s in SEEDS:
        try:
            syn = fn(s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  {name} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50: continue
        r_, d_ = proximity(syn)
        acc["ratio"].append(r_); acc["dup"].append(d_)
        acc["nn"].append(nn_auc(syn)); acc["dom"].append(dom_auc(syn))
        acc["ks_in"].append(mean_ks(syn, members)); acc["ks_out"].append(mean_ks(syn, nonmembers))
    if not acc["ratio"]: continue
    res[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    r = res[name]
    print(f"  {name:17s} yakinlik {r['ratio']:.3f} | yakin-kopya {r['dup']:.3f} | "
          f"KS {r['ks_in']:.3f} | NN {r['nn']:.3f} | DOM {r['dom']:.3f}", flush=True)

json.dump({"res": res, "d_med": D_MED, "d_p5": D_P5}, open("phase_e.json", "w"), indent=1)

ARCH = [a for a in ["SMOTE", "ADASYN", "TabDDPM", "CTGAN", "TVAE", "Gaussian copula", "Vanilla GAN"] if a in res]
print("\n" + "=" * 72)
print(f"AORT KOHORTU, ONGORU GUCU ({len(ARCH)} mimari)")
nnv = [res[a]["nn"] for a in ARCH]; dmv = [res[a]["dom"] for a in ARCH]
for lab, key in [("yakinlik orani", "ratio"), ("yakin-kopya orani", "dup"), ("KS_ic (sadakat)", "ks_in")]:
    x = [res[a][key] for a in ARCH]
    r1, p1 = spearmanr(x, nnv); r2, p2 = spearmanr(x, dmv)
    print(f"  {lab:20s} vs NN rho={r1:+.3f} p={p1:.4f} | vs DOMIAS rho={r2:+.3f} p={p2:.4f}")
print(f"\nsure {(time.time()-t0)/60:.1f} dk")
