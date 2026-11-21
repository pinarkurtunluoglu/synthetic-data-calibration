"""
Faz L: duzeltilmis kosulari ANA TABLOYA tasimak icin gereken sayilar.

(1) Tablo 2'nin duzeltilmis hali: CTGAN ve TVAE kategorik sutun bildirilmis ve
    kucuk batch ile; saldiri olcegi hem uye kumesinden hem referans
    populasyondan; bootstrap araligi on tohumun ortalamasi uzerinden
    (boylece nokta tahmin araligin icinde kalir).

(2) Aort kohortu 669 sentetik kayitla (dengeleme sayisi), 743 yerine.

(3) Tablo 6'daki 800 adimli kirpmasiz referans icin PR-AUC.

Cikti: phase_l.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

os.chdir("/home/claude/run")
T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
SEEDS = list(range(10)); NBOOT = 2000
OUTL = {}

# ---------------- tamponad ----------------
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
r = np.random.default_rng(42); ids = []
for lab in [0, 1]:
    a = PT[PT[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
    ids.extend(a[:int(len(a) * .25)])
m = D["subject_id"].isin(set(ids)); TR, TE = D[~m].copy(), D[m].copy()
imp = KNNImputer(n_neighbors=10)
TR[FEATS] = imp.fit_transform(TR[FEATS]); TE[FEATS] = imp.transform(TE[FEATS])
MEM = TR[TR[TARGET] == 1][FEATS].reset_index(drop=True)
NON = TE[TE[TARGET] == 1][FEATS].reset_index(drop=True)
REF = pd.concat([TR[TR[TARGET] == 0], TE[TE[TARGET] == 0]])[FEATS]
N_SYNTH = 479
CATS = [c for c in FEATS if D[c].dropna().nunique() <= 2]
print(f"uye {len(MEM)} | uye-olmayan {len(NON)} | kategorik {CATS}", flush=True)

SC_MEM = StandardScaler().fit(MEM); SC_PUB = StandardScaler().fit(REF)
M_MEM, N_MEM = SC_MEM.transform(MEM), SC_MEM.transform(NON)
dr = NearestNeighbors(n_neighbors=2).fit(M_MEM).kneighbors(M_MEM)[0][:, 1]
D_MED, D_P5 = float(np.median(dr)), float(np.percentile(dr, 5))
KR = KernelDensity(bandwidth=1.0).fit(SC_MEM.transform(REF))

from imblearn.over_sampling import SMOTE, ADASYN
from ctgan import CTGAN, TVAE


def build(members, tr, n_synth, feats):
    def pick(arr, seed):
        i = np.random.default_rng(seed).choice(len(arr), n_synth, replace=len(arr) < n_synth)
        return pd.DataFrame(arr[i], columns=feats)

    def g_smote(seed):
        X, y = tr[feats].values, tr[TARGET].values
        sm = SMOTE(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                   k_neighbors=min(5, int((y == 1).sum()) - 1))
        return pick(sm.fit_resample(X, y)[0][len(X):], seed)

    def g_adasyn(seed):
        X, y = tr[feats].values, tr[TARGET].values
        ad = ADASYN(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                    n_neighbors=min(5, int((y == 1).sum()) - 1))
        return pick(ad.fit_resample(X, y)[0][len(X):], seed)

    cats = [c for c in feats if members[c].dropna().nunique() <= 2]

    def g_ctgan(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mo = CTGAN(epochs=300, batch_size=30, pac=1, verbose=False)
        mo.fit(members, discrete_columns=cats); return mo.sample(n_synth)[feats]

    def g_tvae(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mo = TVAE(epochs=300, batch_size=30)
        mo.fit(members, discrete_columns=cats); return mo.sample(n_synth)[feats]

    def g_ddpm(seed, steps=3000, poisson=False, clip=None):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        sc = StandardScaler().fit(members)
        x0 = torch.tensor(sc.transform(members), dtype=torch.float32); dd = x0.shape[1]
        net = nn.Sequential(nn.Linear(dd + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dd))
        f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        Nm = len(members); Q = 32 / Nm
        for _ in range(steps):
            idx = np.nonzero(rng.random(Nm) < Q)[0] if poisson else np.arange(Nm)
            if len(idx) == 0: continue
            xb = x0[idx]; t = torch.randint(0, T, (len(idx),)); nz = torch.randn_like(xb)
            a = abar[t].unsqueeze(1); xt = a.sqrt() * xb + (1 - a).sqrt() * nz
            opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
        with torch.no_grad():
            x = torch.randn(n_synth, dd)
            for t in reversed(range(T)):
                e = f(x, torch.full((n_synth,), t))
                x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
                if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
        return pd.DataFrame(sc.inverse_transform(x.numpy()), columns=feats)

    def g_gan(seed, epochs=2000, z=16):
        torch.manual_seed(seed); sc = StandardScaler().fit(members)
        X = torch.tensor(sc.transform(members), dtype=torch.float32); dd = X.shape[1]
        G = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, dd))
        Dm = nn.Sequential(nn.Linear(dd, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
        og = torch.optim.Adam(G.parameters(), 2e-4, betas=(.5, .999))
        od = torch.optim.Adam(Dm.parameters(), 2e-4, betas=(.5, .999)); bce = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            fk = G(torch.randn(len(X), z)); od.zero_grad()
            (bce(Dm(X), torch.ones(len(X), 1)) +
             bce(Dm(fk.detach()), torch.zeros(len(X), 1))).backward(); od.step()
            og.zero_grad(); bce(Dm(fk), torch.ones(len(X), 1)).backward(); og.step()
        with torch.no_grad(): o = G(torch.randn(n_synth, z)).numpy()
        return pd.DataFrame(sc.inverse_transform(o), columns=feats)

    def g_copula(seed):
        rng = np.random.default_rng(seed); X = members[feats].values.astype("float64")
        n, d = X.shape
        u = np.column_stack([(rankdata(X[:, j]) - .5) / n for j in range(d)])
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(d)
        us = norm.cdf(rng.multivariate_normal(np.zeros(d), cov, size=n_synth))
        return pd.DataFrame(np.column_stack(
            [np.quantile(X[:, j], np.clip(us[:, j], 0, 1)) for j in range(d)]), columns=feats)

    def g_copy(seed):
        i = np.random.default_rng(seed).choice(len(members), n_synth, replace=True)
        return members.iloc[i].reset_index(drop=True)

    def g_noise(seed):
        rr = np.random.default_rng(seed)
        return pd.DataFrame({c: rr.uniform(members[c].min(), members[c].max(), n_synth)
                             for c in feats})

    return {"Copy (ceiling)": g_copy, "SMOTE": g_smote, "ADASYN": g_adasyn,
            "Diffusion": g_ddpm, "Gaussian copula": g_copula, "TVAE": g_tvae,
            "Vanilla GAN": g_gan, "CTGAN": g_ctgan, "Noise (floor)": g_noise}


# ============ (1) duzeltilmis Tablo 2 ============
print("\n(1) duzeltilmis ana tablo", flush=True)
G = build(MEM, TR, N_SYNTH, FEATS)
RNG = np.random.default_rng(0)
res1 = {}
for name, fn in G.items():
    acc = {k: [] for k in ["ks_in", "ks_out", "nn_mem", "nn_pub", "dom", "ratio", "dup"]}
    pooled_m, pooled_n = [], []
    for s in SEEDS:
        try:
            syn = fn(s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"   {name} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50: continue
        acc["ks_in"].append(float(np.mean([ks_2samp(syn[c], MEM[c]).statistic for c in FEATS])))
        acc["ks_out"].append(float(np.mean([ks_2samp(syn[c], NON[c]).statistic for c in FEATS])))
        k1 = NearestNeighbors(n_neighbors=1).fit(SC_MEM.transform(syn))
        dm = k1.kneighbors(M_MEM)[0].ravel(); dn = k1.kneighbors(N_MEM)[0].ravel()
        yy = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
        acc["nn_mem"].append(float(roc_auc_score(yy, -np.r_[dm, dn])))
        pooled_m.append(-dm); pooled_n.append(-dn)
        k2 = NearestNeighbors(n_neighbors=1).fit(SC_PUB.transform(syn))
        acc["nn_pub"].append(float(roc_auc_score(
            yy, -np.r_[k2.kneighbors(SC_PUB.transform(MEM))[0].ravel(),
                       k2.kneighbors(SC_PUB.transform(NON))[0].ravel()])))
        ks_ = KernelDensity(bandwidth=1.0).fit(SC_MEM.transform(syn))
        sm_ = ks_.score_samples(M_MEM) - KR.score_samples(M_MEM)
        sn_ = ks_.score_samples(N_MEM) - KR.score_samples(N_MEM)
        acc["dom"].append(float(roc_auc_score(yy, np.r_[sm_, sn_])))
        kk = NearestNeighbors(n_neighbors=1).fit(M_MEM)
        dsyn = kk.kneighbors(SC_MEM.transform(syn))[0].ravel()
        acc["ratio"].append(float(np.median(dsyn) / D_MED))
        acc["dup"].append(float(np.mean(dsyn < D_P5)))
    if not acc["ks_in"]: continue
    # bootstrap: her yinelemede ON TOHUMUN ORTALAMASI alinir
    SM = np.array(pooled_m); SN = np.array(pooled_n)  # (seed, n)
    boots = []
    for _ in range(NBOOT):
        im = RNG.integers(0, SM.shape[1], SM.shape[1])
        inn = RNG.integers(0, SN.shape[1], SN.shape[1])
        per = [roc_auc_score(np.r_[np.ones(len(im)), np.zeros(len(inn))],
                             np.r_[SM[j][im], SN[j][inn]]) for j in range(SM.shape[0])]
        boots.append(np.mean(per))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res1[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    res1[name]["nn_ci"] = [float(lo), float(hi)]
    rr = res1[name]
    print(f"   {name:17s} KS {rr['ks_in']:.3f} | NN {rr['nn_mem']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] | NN(ref) {rr['nn_pub']:.3f} | dup {rr['dup']:.3f}", flush=True)
OUTL["table2_corrected"] = res1
json.dump(OUTL, open("phase_l.json", "w"), indent=1)

# ============ (3) 800 adimli kirpmasiz referans PR-AUC ============
print("\n(3) 800 adimli kirpmasiz referans icin PR-AUC", flush=True)
Xte, yte = TE[FEATS].values, TE[TARGET].values
pr = []
for s in SEEDS:
    syn = G["Diffusion"](s, steps=800, poisson=True)
    syn = syn[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
    Xa = np.vstack([TR[FEATS].values, syn.values]); ya = np.r_[TR[TARGET].values, np.ones(len(syn))]
    clf = HistGradientBoostingClassifier(random_state=s).fit(Xa, ya)
    pr.append(float(average_precision_score(yte, clf.predict_proba(Xte)[:, 1])))
OUTL["unconstrained_800_prauc"] = {"mean": float(np.mean(pr)), "sd": float(np.std(pr))}
print(f"   PR-AUC {np.mean(pr):.3f} (sd {np.std(pr):.3f})", flush=True)
json.dump(OUTL, open("phase_l.json", "w"), indent=1)

# ============ (2) aort, 669 kayit ============
print("\n(2) aort kohortu, 669 sentetik kayit", flush=True)
tra = pd.read_parquet("aortic_train.parquet"); tea = pd.read_parquet("aortic_test.parquet")
FA = [c for c in tra.columns if c not in {"subject_id", TARGET}]
for d_ in (tra, tea):
    for c in FA:
        if not pd.api.types.is_numeric_dtype(d_[c]):
            v = sorted(d_[c].dropna().unique()); d_[c] = d_[c].map({v[0]: 0.0, v[1]: 1.0})
    d_[FA] = d_[FA].astype("float64")
MEMA = tra[tra[TARGET] == 1][FA].reset_index(drop=True)
NONA = tea[tea[TARGET] == 1][FA].reset_index(drop=True)
REFA = pd.concat([tra[tra[TARGET] == 0], tea[tea[TARGET] == 0]])[FA]
NA = int((tra[TARGET] == 0).sum()) - int((tra[TARGET] == 1).sum())
print(f"   uye {len(MEMA)} | uye-olmayan {len(NONA)} | sentetik {NA}", flush=True)
SCA = StandardScaler().fit(MEMA); MA = SCA.transform(MEMA); NAx = SCA.transform(NONA)
dra = NearestNeighbors(n_neighbors=2).fit(MA).kneighbors(MA)[0][:, 1]
DMEDA, DP5A = float(np.median(dra)), float(np.percentile(dra, 5))
KRA = KernelDensity(bandwidth=1.0).fit(SCA.transform(REFA))
GA = build(MEMA, tra, NA, FA)
res2 = {}
for name, fn in GA.items():
    acc = {k: [] for k in ["ks_in", "ks_out", "nn", "dom", "ratio", "dup"]}
    for s in SEEDS:
        try:
            syn = fn(s)[FA].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"   {name} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50: continue
        acc["ks_in"].append(float(np.mean([ks_2samp(syn[c], MEMA[c]).statistic for c in FA])))
        acc["ks_out"].append(float(np.mean([ks_2samp(syn[c], NONA[c]).statistic for c in FA])))
        k1 = NearestNeighbors(n_neighbors=1).fit(SCA.transform(syn))
        dm = k1.kneighbors(MA)[0].ravel(); dn = k1.kneighbors(NAx)[0].ravel()
        yy = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
        acc["nn"].append(float(roc_auc_score(yy, -np.r_[dm, dn])))
        ks_ = KernelDensity(bandwidth=1.0).fit(SCA.transform(syn))
        acc["dom"].append(float(roc_auc_score(yy, np.r_[
            ks_.score_samples(MA) - KRA.score_samples(MA),
            ks_.score_samples(NAx) - KRA.score_samples(NAx)])))
        kk = NearestNeighbors(n_neighbors=1).fit(MA)
        ds = kk.kneighbors(SCA.transform(syn))[0].ravel()
        acc["ratio"].append(float(np.median(ds) / DMEDA)); acc["dup"].append(float(np.mean(ds < DP5A)))
    if not acc["ks_in"]: continue
    res2[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    rr = res2[name]
    print(f"   {name:17s} KS {rr['ks_in']:.3f} | NN {rr['nn']:.3f} | DOM {rr['dom']:.3f} "
          f"| ratio {rr['ratio']:.3f} | dup {rr['dup']:.3f}", flush=True)
OUTL["aortic_669"] = {"n_synth": NA, "res": res2}
json.dump(OUTL, open("phase_l.json", "w"), indent=1)
print("\nbitti", flush=True)
