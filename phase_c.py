"""
Faz C: duzeltilmis boru hatti uzerinde eksik olcumler.
  C1 - 7 mimari + 2 referans icin W1, korelasyon farki, PR-AUC, duyarlilik
  C2 - DP-TabDDPM: Poisson alt-ornekleme + RDP hesaplayici ile gercek (eps, delta)
Cikti: phase_c.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SEEDS = list(range(10)); DP_SEEDS = [0, 1, 2]
N_SYNTH = 479
SIGMAS = [0.5, 1.0, 2.0]; CLIP = 1.0; STEPS = 800; BATCH = 32; DELTA = 1e-5

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
Xte, yte = te[FEATS].values, te[TARGET].values
print(f"uye {len(members)} | uye-olmayan {len(nonmembers)} | test {len(te)}", flush=True)

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

T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def make_net(d):
    return nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))

def g_ddpm(seed, epochs=3000):
    torch.manual_seed(seed); sc = StandardScaler().fit(members)
    x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
    net = make_net(d); f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        t = torch.randint(0, T, (len(x0),)); nz = torch.randn_like(x0); a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * nz
        opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
    return sample_ddpm(f, x0.shape[1], sc)

def sample_ddpm(f, d, sc):
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

# ---------------- olcutler ----------------
rng_w = {c: (tr[c].max() - tr[c].min()) or 1.0 for c in FEATS}
C_real = tr[FEATS].corr().values

def w1_norm(syn, real):
    return float(np.mean([wasserstein_distance(syn[c], real[c]) / rng_w[c] for c in FEATS]))

def corr_diff(syn):
    C = syn[FEATS].corr().values
    return float(np.linalg.norm(np.nan_to_num(C) - C_real, ord="fro"))

def utility(syn, seed):
    Xa = np.vstack([tr[FEATS].values, syn[FEATS].values])
    ya = np.r_[tr[TARGET].values, np.ones(len(syn))]
    clf = HistGradientBoostingClassifier(random_state=seed).fit(Xa, ya)
    p = clf.predict_proba(Xte)[:, 1]
    return (float(average_precision_score(yte, p)),
            float(roc_auc_score(yte, p)),
            float(recall_score(yte, (p >= .5).astype(int))))

def nn_auc(syn):
    sc = StandardScaler().fit(members)
    k = NearestNeighbors(n_neighbors=1).fit(sc.transform(syn))
    dm = k.kneighbors(sc.transform(members))[0].ravel()
    dn = k.kneighbors(sc.transform(nonmembers))[0].ravel()
    return float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))

def dom_auc(syn):
    sc = StandardScaler().fit(members)
    kr = KernelDensity(bandwidth=1.0).fit(sc.transform(ref_pop))
    ks = KernelDensity(bandwidth=1.0).fit(sc.transform(syn))
    Xm, Xn = sc.transform(members), sc.transform(nonmembers)
    sm = ks.score_samples(Xm) - kr.score_samples(Xm); sn = ks.score_samples(Xn) - kr.score_samples(Xn)
    return float(roc_auc_score(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn]))

def mean_ks(syn, real):
    return float(np.mean([ks_2samp(syn[c], real[c]).statistic for c in FEATS]))

# ---------------- C1 ----------------
res = {"C1": {}, "C2": {}}
t0 = time.time()
print("\n--- C1: 9 jenerator, tam olcut seti ---", flush=True)
for name, fn in GENS.items():
    acc = {k: [] for k in ["ks_in", "ks_out", "w1", "corr", "prauc", "rocauc", "recall", "nn", "dom"]}
    for s in SEEDS:
        try:
            syn = fn(s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  {name} s{s} HATA {type(e).__name__}", flush=True); continue
        if len(syn) < 50: continue
        acc["ks_in"].append(mean_ks(syn, members)); acc["ks_out"].append(mean_ks(syn, nonmembers))
        acc["w1"].append(w1_norm(syn, members)); acc["corr"].append(corr_diff(syn))
        pr, ro, rc = utility(syn, s)
        acc["prauc"].append(pr); acc["rocauc"].append(ro); acc["recall"].append(rc)
        acc["nn"].append(nn_auc(syn)); acc["dom"].append(dom_auc(syn))
    if not acc["ks_in"]: continue
    res["C1"][name] = {k: float(np.mean(v)) for k, v in acc.items()}
    res["C1"][name]["n_seeds"] = len(acc["ks_in"])
    r = res["C1"][name]
    print(f"  {name:17s} KS {r['ks_in']:.3f}/{r['ks_out']:.3f} W1 {r['w1']:.3f} "
          f"corr {r['corr']:.2f} PR {r['prauc']:.3f} NN {r['nn']:.3f} DOM {r['dom']:.3f}", flush=True)

# taban cizgisi: sentetik veri olmadan
clf = HistGradientBoostingClassifier(random_state=0).fit(tr[FEATS].values, tr[TARGET].values)
p = clf.predict_proba(Xte)[:, 1]
res["C1"]["_baseline_no_synth"] = {
    "prauc": float(average_precision_score(yte, p)),
    "rocauc": float(roc_auc_score(yte, p)),
    "recall": float(recall_score(yte, (p >= .5).astype(int)))}
print("  TABAN (sentetiksiz) PR %.3f ROC %.3f" %
      (res["C1"]["_baseline_no_synth"]["prauc"], res["C1"]["_baseline_no_synth"]["rocauc"]), flush=True)

# ---------------- C2: DP ----------------
print("\n--- C2: DP, Poisson alt-ornekleme + RDP hesaplayici ---", flush=True)
from opacus.accountants import RDPAccountant
N = len(members); Q = BATCH / N

def epsilon_for(sigma):
    acc = RDPAccountant()
    for _ in range(STEPS): acc.step(noise_multiplier=sigma, sample_rate=Q)
    return float(acc.get_epsilon(delta=DELTA))

EPS = {s: epsilon_for(s) for s in SIGMAS}
for s in SIGMAS: print(f"  sigma={s} -> epsilon={EPS[s]:.2f} (delta={DELTA}, q={Q:.4f}, adim={STEPS})", flush=True)

def train_dp(sigma, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(members)
    x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
    net = make_net(d); f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3); params = list(net.parameters())
    for _ in range(STEPS):
        idx = np.nonzero(rng.random(N) < Q)[0]
        if len(idx) == 0: continue
        xb = x0[idx]; t = torch.randint(0, T, (len(idx),)); nz = torch.randn_like(xb)
        a = abar[t].unsqueeze(1); xt = a.sqrt() * xb + (1 - a).sqrt() * nz
        accg = [torch.zeros_like(p) for p in params]
        for j in range(len(idx)):
            net.zero_grad()
            li = F.mse_loss(f(xt[j:j+1], t[j:j+1]), nz[j:j+1])
            gi = torch.autograd.grad(li, params)
            gnorm = torch.sqrt(sum((g ** 2).sum() for g in gi))
            scale = min(1.0, CLIP / (gnorm.item() + 1e-6))
            for A, g in zip(accg, gi): A += g * scale
        opt.zero_grad()
        for p_, A in zip(params, accg):
            p_.grad = (A + torch.randn_like(A) * sigma * CLIP) / max(1, len(idx))
        opt.step()
    return sample_ddpm(f, x0.shape[1], sc)

for sg in SIGMAS:
    acc = {k: [] for k in ["ks_in", "ks_out", "nn", "dom", "prauc", "recall"]}
    for s in DP_SEEDS:
        syn = train_dp(sg, s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        if len(syn) < 50: print(f"  sigma={sg} s{s} gecersiz", flush=True); continue
        acc["ks_in"].append(mean_ks(syn, members)); acc["ks_out"].append(mean_ks(syn, nonmembers))
        acc["nn"].append(nn_auc(syn)); acc["dom"].append(dom_auc(syn))
        pr, ro, rc = utility(syn, s); acc["prauc"].append(pr); acc["recall"].append(rc)
    if not acc["ks_in"]: continue
    res["C2"][f"sigma={sg}"] = {k: float(np.mean(v)) for k, v in acc.items()}
    res["C2"][f"sigma={sg}"]["epsilon"] = EPS[sg]
    r = res["C2"][f"sigma={sg}"]
    print(f"  sigma={sg} eps={EPS[sg]:.1f} KS {r['ks_in']:.3f}/{r['ks_out']:.3f} "
          f"NN {r['nn']:.3f} DOM {r['dom']:.3f} PR {r['prauc']:.3f}", flush=True)

res["meta"] = {"delta": DELTA, "q": Q, "steps": STEPS, "clip": CLIP,
               "n_members": int(N), "n_nonmembers": int(len(nonmembers))}
json.dump(res, open("phase_c.json", "w"), indent=1)
print(f"\nsure {(time.time()-t0)/60:.1f} dk", flush=True)
