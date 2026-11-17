"""
Faz H: tekrarli bolme (hakem madde 6).

Simdiye kadar bolme tohumu 42'de sabitti, sadece uretici tohumu degisiyordu.
Yani butun sonuclar tek bir 93 uye / 30 uye-olmayan ayrimina dayaniyordu.
Burada bolmenin kendisi 10 kez degistiriliyor; her bolmede hat bastan
kuruluyor (imputer dahil) ve her uretec bir kez calistiriliyor.

Asil soru: gurultu tabaninin 0.5'in uzerinde cikmasi bu bolmeye mi ozgu?

Cikti: phase_h.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SPLIT_SEEDS = list(range(10))     # bolme tohumlari: asil degisken bu
N_SYNTH = 479                     # diger fazlarla ayni
T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

# ---------------- veri ----------------
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


def make_split(split_seed):
    """hasta duzeyinde tabakali bolme + sadece egitimde ogrenilen imputer"""
    r = np.random.default_rng(split_seed); ids = []
    for lab in [0, 1]:
        a = PT[PT[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * .25)])
    m = D["subject_id"].isin(set(ids))
    tr, te = D[~m].copy(), D[m].copy()
    imp = KNNImputer(n_neighbors=10)
    tr[FEATS] = imp.fit_transform(tr[FEATS]); te[FEATS] = imp.transform(te[FEATS])
    return tr, te


def build(members, tr):
    from imblearn.over_sampling import SMOTE, ADASYN
    from ctgan import CTGAN, TVAE

    def pick(arr, seed):
        i = np.random.default_rng(seed).choice(len(arr), N_SYNTH, replace=len(arr) < N_SYNTH)
        return pd.DataFrame(arr[i], columns=FEATS)

    def g_smote(seed):
        X, y = tr[FEATS].values, tr[TARGET].values
        k = min(5, int((y == 1).sum()) - 1)
        sm = SMOTE(sampling_strategy={1: int((y == 0).sum())}, random_state=seed, k_neighbors=k)
        Xr, _ = sm.fit_resample(X, y); return pick(Xr[len(X):], seed)

    def g_adasyn(seed):
        X, y = tr[FEATS].values, tr[TARGET].values
        k = min(5, int((y == 1).sum()) - 1)
        ad = ADASYN(sampling_strategy={1: int((y == 0).sum())}, random_state=seed, n_neighbors=k)
        Xr, _ = ad.fit_resample(X, y); return pick(Xr[len(X):], seed)

    def g_ctgan(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = CTGAN(epochs=300, verbose=False); m.fit(members); return m.sample(N_SYNTH)[FEATS]

    def g_tvae(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = TVAE(epochs=300); m.fit(members); return m.sample(N_SYNTH)[FEATS]

    def g_ddpm(seed, steps=3000):
        torch.manual_seed(seed); sc = StandardScaler().fit(members)
        x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
        net = nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))
        f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for _ in range(steps):
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

    def g_gan(seed, epochs=2000, z=16):
        torch.manual_seed(seed); sc = StandardScaler().fit(members)
        X = torch.tensor(sc.transform(members), dtype=torch.float32); d = X.shape[1]
        G = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d))
        Dm = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
        og = torch.optim.Adam(G.parameters(), 2e-4, betas=(.5, .999))
        od = torch.optim.Adam(Dm.parameters(), 2e-4, betas=(.5, .999)); bce = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            fk = G(torch.randn(len(X), z)); od.zero_grad()
            (bce(Dm(X), torch.ones(len(X), 1)) +
             bce(Dm(fk.detach()), torch.zeros(len(X), 1))).backward(); od.step()
            og.zero_grad(); bce(Dm(fk), torch.ones(len(X), 1)).backward(); og.step()
        with torch.no_grad(): o = G(torch.randn(N_SYNTH, z)).numpy()
        return pd.DataFrame(sc.inverse_transform(o), columns=FEATS)

    def g_copula(seed):
        rng = np.random.default_rng(seed); X = members[FEATS].values.astype("float64")
        n, d = X.shape
        u = np.column_stack([(rankdata(X[:, j]) - .5) / n for j in range(d)])
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(d)
        us = norm.cdf(rng.multivariate_normal(np.zeros(d), cov, size=N_SYNTH))
        return pd.DataFrame(np.column_stack(
            [np.quantile(X[:, j], np.clip(us[:, j], 0, 1)) for j in range(d)]), columns=FEATS)

    def g_copy(seed):
        i = np.random.default_rng(seed).choice(len(members), N_SYNTH, replace=True)
        return members.iloc[i].reset_index(drop=True)

    def g_noise(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame({c: r.uniform(members[c].min(), members[c].max(), N_SYNTH)
                             for c in FEATS})

    return {"Copy (ceiling)": g_copy, "SMOTE": g_smote, "ADASYN": g_adasyn, "TabDDPM": g_ddpm,
            "Gaussian copula": g_copula, "TVAE": g_tvae, "Vanilla GAN": g_gan,
            "CTGAN": g_ctgan, "Noise (floor)": g_noise}


res = {}
t0 = time.time()
for sp in SPLIT_SEEDS:
    tr, te = make_split(sp)
    members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
    nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
    ref_pop = pd.concat([tr[tr[TARGET] == 0], te[te[TARGET] == 0]])[FEATS]
    SC = StandardScaler().fit(members); M = SC.transform(members)
    dr = NearestNeighbors(n_neighbors=2).fit(M).kneighbors(M)[0][:, 1]
    D_P5 = float(np.percentile(dr, 5))
    kr = KernelDensity(bandwidth=1.0).fit(SC.transform(ref_pop))
    Xn = SC.transform(nonmembers)

    def nn_auc(syn):
        k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
        dm = k.kneighbors(M)[0].ravel(); dn = k.kneighbors(Xn)[0].ravel()
        return float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))

    def dom_auc(syn):
        ks_ = KernelDensity(bandwidth=1.0).fit(SC.transform(syn))
        sm = ks_.score_samples(M) - kr.score_samples(M)
        sn = ks_.score_samples(Xn) - kr.score_samples(Xn)
        return float(roc_auc_score(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn]))

    def dup(syn):
        k = NearestNeighbors(n_neighbors=1).fit(M)
        return float(np.mean(k.kneighbors(SC.transform(syn))[0].ravel() < D_P5))

    GENS = build(members, tr)
    print(f"\n=== bolme {sp} | uye {len(members)} uye-olmayan {len(nonmembers)} ===", flush=True)
    for name, fn in GENS.items():
        try:
            syn = fn(sp)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  {name} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50: continue
        row = {"ks_in": float(np.mean([ks_2samp(syn[c], members[c]).statistic for c in FEATS])),
               "nn": nn_auc(syn), "dom": dom_auc(syn), "dup": dup(syn),
               "n_members": int(len(members)), "n_nonmembers": int(len(nonmembers))}
        res.setdefault(name, []).append(row)
        print(f"  {name:17s} KS {row['ks_in']:.3f} NN {row['nn']:.3f} "
              f"DOM {row['dom']:.3f} dup {row['dup']:.3f}", flush=True)
    json.dump(res, open("phase_h.json", "w"), indent=1)

print("\n" + "=" * 76)
print("BOLMELER ARASI OZET (10 bolme)")
print(f"{'uretec':17s} {'NN ort':>7s} {'NN sd':>6s} {'NN min':>7s} {'NN max':>7s} {'KS ort':>7s}")
for name, rows in res.items():
    nn = np.array([r["nn"] for r in rows]); ks = np.array([r["ks_in"] for r in rows])
    print(f"{name:17s} {nn.mean():7.3f} {nn.std():6.3f} {nn.min():7.3f} {nn.max():7.3f} {ks.mean():7.3f}")
print(f"\nsure {(time.time()-t0)/60:.1f} dk", flush=True)
