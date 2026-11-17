"""
Faz A: kayit duzeyinde guven araliklari (6 jenerator, 10 tohum)
Faz B: uc yeni mimari (TVAE, ADASYN, Gauss kopulasi) -> toplam 7 mimari
Cikti: phase_ab.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, spearmanr, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"
SEEDS = list(range(10))
N_SYNTH, NBOOT = 479, 2000
RNG = np.random.default_rng(0)
os.chdir(WD)

raw = pd.read_parquet("patient_matrix_24h.parquet")
sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
TARGET = "mortality"
df = raw.copy(); df["subject_id"] = sub["subject_id"].values; df[TARGET] = sub[TARGET].values
OUT = {"hospital_expire_flag", "mortality"}
for c in [c for c in raw.columns if c not in OUT and not pd.api.types.is_numeric_dtype(df[c])]:
    v = sorted(df[c].dropna().unique()); df[c] = df[c].map({v[0]: 0.0, v[1]: 1.0}).astype(float)
FEATS = [c for c in df.columns if c not in ({"subject_id", TARGET} | OUT)]

def split(d, frac=.25, seed=42):
    pt = d.groupby("subject_id")[TARGET].max().reset_index()
    r = np.random.default_rng(seed); ids = []
    for lab in [0, 1]:
        a = pt[pt[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * frac)])
    m = d["subject_id"].isin(set(ids)); return d[~m].copy(), d[m].copy()

tr, te = split(df)
imp = KNNImputer(n_neighbors=10)
tr[FEATS] = imp.fit_transform(tr[FEATS]); te[FEATS] = imp.transform(te[FEATS])
members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
ref_pop = pd.concat([tr[tr[TARGET] == 0], te[te[TARGET] == 0]])[FEATS]
print(f"uye {len(members)} | uye-olmayan {len(nonmembers)} | ozellik {len(FEATS)}", flush=True)

# ---------------- jeneratorler ----------------
from imblearn.over_sampling import SMOTE, ADASYN
from ctgan import CTGAN, TVAE

def g_smote(seed):
    X, y = tr[FEATS].values, tr[TARGET].values
    sm = SMOTE(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
               k_neighbors=min(5, int((y == 1).sum()) - 1))
    Xr, _ = sm.fit_resample(X, y); syn = Xr[len(X):]
    i = np.random.default_rng(seed).choice(len(syn), N_SYNTH, replace=len(syn) < N_SYNTH)
    return pd.DataFrame(syn[i], columns=FEATS)

def g_adasyn(seed):
    X, y = tr[FEATS].values, tr[TARGET].values
    ad = ADASYN(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                n_neighbors=min(5, int((y == 1).sum()) - 1))
    Xr, _ = ad.fit_resample(X, y); syn = Xr[len(X):]
    i = np.random.default_rng(seed).choice(len(syn), N_SYNTH, replace=len(syn) < N_SYNTH)
    return pd.DataFrame(syn[i], columns=FEATS)

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

def g_ddpm(seed, T=200, epochs=3000):
    torch.manual_seed(seed); sc = StandardScaler().fit(members)
    x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
    b = torch.linspace(1e-4, .02, T); al = 1 - b; ab = torch.cumprod(al, 0)
    net = nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))
    f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        t = torch.randint(0, T, (len(x0),)); nz = torch.randn_like(x0); a = ab[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * nz
        opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
    with torch.no_grad():
        x = torch.randn(N_SYNTH, d)
        for t in reversed(range(T)):
            e = f(x, torch.full((N_SYNTH,), t))
            x = (x - (1 - al[t]) / (1 - ab[t]).sqrt() * e) / al[t].sqrt()
            if t > 0: x = x + b[t].sqrt() * torch.randn_like(x)
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

GENS = {"Copy (ceiling)": g_copy, "SMOTE": g_smote, "ADASYN": g_adasyn,
        "TabDDPM": g_ddpm, "CTGAN": g_ctgan, "TVAE": g_tvae,
        "Gaussian copula": g_copula, "Vanilla GAN": g_gan, "Noise (floor)": g_noise}

# ---------------- olcutler ----------------
def mean_ks(syn, real):
    return float(np.mean([ks_2samp(syn[c], real[c]).statistic for c in FEATS]))

def nn_scores(syn):
    sc = StandardScaler().fit(members)
    k = NearestNeighbors(n_neighbors=1).fit(sc.transform(syn))
    return (-k.kneighbors(sc.transform(members))[0].ravel(),
            -k.kneighbors(sc.transform(nonmembers))[0].ravel())

def dom_scores(syn):
    sc = StandardScaler().fit(members)
    kr = KernelDensity(bandwidth=1.0).fit(sc.transform(ref_pop))
    ks = KernelDensity(bandwidth=1.0).fit(sc.transform(syn))
    Xm, Xn = sc.transform(members), sc.transform(nonmembers)
    return (ks.score_samples(Xm) - kr.score_samples(Xm),
            ks.score_samples(Xn) - kr.score_samples(Xn))

def auc(sm, sn):
    return float(roc_auc_score(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn]))

def boot_records(sm, sn):
    out = np.empty(NBOOT)
    for i in range(NBOOT):
        a = sm[RNG.integers(0, len(sm), len(sm))]
        b = sn[RNG.integers(0, len(sn), len(sn))]
        out[i] = auc(a, b)
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]

def boot_seeds(vals):
    vals = np.asarray(vals)
    draws = [float(np.mean(RNG.choice(vals, len(vals)))) for _ in range(NBOOT)]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]

# ---------------- calistir ----------------
res = {}; t0 = time.time()
for name, fn in GENS.items():
    ks_in, ks_out, nn_s, dom_s = [], [], [], []
    pm_nn, pn_nn, pm_dm, pn_dm = [], [], [], []
    for s in SEEDS:
        try:
            syn = fn(s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  {name} seed {s} HATA: {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50:
            print(f"  {name} seed {s}: az gecerli kayit, atlandi", flush=True); continue
        ks_in.append(mean_ks(syn, members)); ks_out.append(mean_ks(syn, nonmembers))
        m1, n1 = nn_scores(syn); m2, n2 = dom_scores(syn)
        nn_s.append(auc(m1, n1)); dom_s.append(auc(m2, n2))
        pm_nn.append(m1); pn_nn.append(n1); pm_dm.append(m2); pn_dm.append(n2)
    if not ks_in:
        print(f"  {name}: sonuc yok", flush=True); continue
    nm, nn_p = np.mean(pm_nn, 0), np.mean(pn_nn, 0)
    dm, dn_p = np.mean(pm_dm, 0), np.mean(pn_dm, 0)
    res[name] = {
        "n_seeds": len(ks_in),
        "ks_in": float(np.mean(ks_in)), "ks_out": float(np.mean(ks_out)),
        "nn": float(np.mean(nn_s)), "dom": float(np.mean(dom_s)),
        "nn_seed_ci": boot_seeds(nn_s), "nn_rec_ci": boot_records(nm, nn_p),
        "dom_seed_ci": boot_seeds(dom_s), "dom_rec_ci": boot_records(dm, dn_p),
    }
    r = res[name]
    print(f"  {name:17s} KSin {r['ks_in']:.3f} KSout {r['ks_out']:.3f} "
          f"NN {r['nn']:.3f} tohum({r['nn_seed_ci'][0]:.3f}-{r['nn_seed_ci'][1]:.3f}) "
          f"KAYIT({r['nn_rec_ci'][0]:.3f}-{r['nn_rec_ci'][1]:.3f}) DOM {r['dom']:.3f}",
          flush=True)

json.dump(res, open("phase_ab.json", "w"), indent=1)

ARCH = ["SMOTE", "ADASYN", "TabDDPM", "CTGAN", "TVAE", "Gaussian copula", "Vanilla GAN"]
ARCH = [a for a in ARCH if a in res]
ksi = [res[a]["ks_in"] for a in ARCH]; kso = [res[a]["ks_out"] for a in ARCH]
nnv = [res[a]["nn"] for a in ARCH]; dmv = [res[a]["dom"] for a in ARCH]
gap = [o - i for i, o in zip(ksi, kso)]
print("\n" + "=" * 70)
print(f"KORELASYON ({len(ARCH)} mimari): " + ", ".join(ARCH))
for lab, x in [("KS_ic", ksi), ("KS_dis", kso), ("acik", gap)]:
    r1, p1 = spearmanr(x, nnv); r2, p2 = spearmanr(x, dmv)
    print(f"  {lab:7s} vs NN rho={r1:+.3f} p={p1:.4f} | vs DOMIAS rho={r2:+.3f} p={p2:.4f}")
ws = float(np.mean([res[a]["nn_seed_ci"][1] - res[a]["nn_seed_ci"][0] for a in res]))
wr = float(np.mean([res[a]["nn_rec_ci"][1] - res[a]["nn_rec_ci"][0] for a in res]))
print(f"\nortalama GA genisligi: tohum {ws:.4f} | kayit {wr:.4f} ({wr/ws:.1f} kat)")
print(f"sure {(time.time()-t0)/60:.1f} dk")
