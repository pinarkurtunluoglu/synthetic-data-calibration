"""
Faz F: iddianin kirmizi takim testi.

ITIRAZ: "KS marjinal dagilimlara bakiyor, Gauss kopulasi zaten marjinalleri
birebir kuruyor. Kopulayi kendi optimize ettigi olcutle olcup kazandi
demissiniz."

TEST: kopulanin optimize ETMEDIGI bir sadakat olcutu -> ayirt edici testi.
Gercek uye kayitlari ile sentetik kayitlari ayirmaya calisan bir
siniflandirici egit. AUC 0.5'e yakinsa ayirt edemiyor, yani veri gercekten
gercekci. Bu olcut ortak yapiya bakar, marjinallere degil.

Ikinci olcut: ciftli bagimlilik hatasi (karsilikli bilgi matrisi farki).
Kopula korelasyonu kuruyor ama dogrusal olmayan bagimliliklari kurmuyor.

Iki kohortta da calistirilir.
Cikti: phase_f.json
"""
import os, json, time, warnings, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SEEDS = list(range(5))

def load_tamponade():
    raw = pd.read_parquet("patient_matrix_24h.parquet")
    sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
    T = "mortality"
    d = raw.copy(); d["subject_id"] = sub["subject_id"].values; d[T] = sub[T].values
    OUT = {"hospital_expire_flag", "mortality"}
    for c in [c for c in raw.columns if c not in OUT and not pd.api.types.is_numeric_dtype(d[c])]:
        v = sorted(d[c].dropna().unique()); d[c] = d[c].map({v[0]: 0.0, v[1]: 1.0})
    F_ = [c for c in d.columns if c not in ({"subject_id", T} | OUT)]
    d[F_] = d[F_].astype("float64")
    pt = d.groupby("subject_id")[T].max().reset_index()
    r = np.random.default_rng(42); ids = []
    for lab in [0, 1]:
        a = pt[pt[T] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * .25)])
    m = d["subject_id"].isin(set(ids)); tr, te = d[~m].copy(), d[m].copy()
    imp = KNNImputer(n_neighbors=10)
    tr[F_] = imp.fit_transform(tr[F_]); te[F_] = imp.transform(te[F_])
    return tr, te, F_, T

def load_aortic():
    tr = pd.read_parquet("aortic_train.parquet"); te = pd.read_parquet("aortic_test.parquet")
    T = "mortality"; F_ = [c for c in tr.columns if c not in {"subject_id", T}]
    for d in (tr, te):
        for c in F_:
            if not pd.api.types.is_numeric_dtype(d[c]):
                v = sorted(d[c].dropna().unique()); d[c] = d[c].map({v[0]: 0.0, v[1]: 1.0})
        d[F_] = d[F_].astype("float64")
    return tr, te, F_, T

T_DIFF = 200
betas = torch.linspace(1e-4, .02, T_DIFF); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def build(members, FEATS, N_SYNTH, tr, TARGET):
    from imblearn.over_sampling import SMOTE, ADASYN
    from ctgan import CTGAN, TVAE

    def pick(syn, seed):
        i = np.random.default_rng(seed).choice(len(syn), N_SYNTH, replace=len(syn) < N_SYNTH)
        return pd.DataFrame(syn[i], columns=FEATS)

    def g_smote(seed):
        X, y = tr[FEATS].values, tr[TARGET].values
        sm = SMOTE(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                   k_neighbors=min(5, int((y == 1).sum()) - 1))
        Xr, _ = sm.fit_resample(X, y); return pick(Xr[len(X):], seed)

    def g_adasyn(seed):
        X, y = tr[FEATS].values, tr[TARGET].values
        ad = ADASYN(sampling_strategy={1: int((y == 0).sum())}, random_state=seed,
                    n_neighbors=min(5, int((y == 1).sum()) - 1))
        Xr, _ = ad.fit_resample(X, y); return pick(Xr[len(X):], seed)

    def g_ctgan(seed, epochs=300):
        torch.manual_seed(seed); np.random.seed(seed)
        m = CTGAN(epochs=epochs, verbose=False); m.fit(members); return m.sample(N_SYNTH)[FEATS]

    def g_tvae(seed, epochs=300):
        torch.manual_seed(seed); np.random.seed(seed)
        m = TVAE(epochs=epochs); m.fit(members); return m.sample(N_SYNTH)[FEATS]

    def g_ddpm(seed, epochs=3000):
        torch.manual_seed(seed); sc = StandardScaler().fit(members)
        x0 = torch.tensor(sc.transform(members), dtype=torch.float32); d = x0.shape[1]
        net = nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))
        f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T_DIFF], 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for _ in range(epochs):
            t = torch.randint(0, T_DIFF, (len(x0),)); nz = torch.randn_like(x0); a = abar[t].unsqueeze(1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * nz
            opt.zero_grad(); F.mse_loss(f(xt, t), nz).backward(); opt.step()
        with torch.no_grad():
            x = torch.randn(N_SYNTH, d)
            for t in reversed(range(T_DIFF)):
                e = f(x, torch.full((N_SYNTH,), t))
                x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
                if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
        return pd.DataFrame(sc.inverse_transform(x.numpy()), columns=FEATS)

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

    def g_copula(seed):
        rng = np.random.default_rng(seed)
        X = members[FEATS].values.astype("float64"); n, d = X.shape
        u = np.column_stack([(rankdata(X[:, j]) - 0.5) / n for j in range(d)])
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(d)
        zs = rng.multivariate_normal(np.zeros(d), cov, size=N_SYNTH)
        us = norm.cdf(zs)
        out = np.column_stack([np.quantile(X[:, j], np.clip(us[:, j], 0, 1)) for j in range(d)])
        return pd.DataFrame(out, columns=FEATS)

    def g_noise(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame({c: r.uniform(members[c].min(), members[c].max(), N_SYNTH) for c in FEATS})

    return {"SMOTE": g_smote, "ADASYN": g_adasyn, "TabDDPM": g_ddpm, "CTGAN": g_ctgan,
            "TVAE": g_tvae, "Gaussian copula": g_copula, "Vanilla GAN": g_gan,
            "Noise (floor)": g_noise}

def detection_auc(real, syn, seed):
    """gercek vs sentetik ayirt edici testi; 0.5 = ayirt edilemiyor"""
    n = min(len(real), len(syn))
    rs = real.sample(n, random_state=seed) if len(real) > n else real
    ss = syn.sample(n, random_state=seed) if len(syn) > n else syn
    X = np.vstack([rs.values, ss.values]); y = np.r_[np.ones(len(rs)), np.zeros(len(ss))]
    sk = StratifiedKFold(5, shuffle=True, random_state=seed); p = np.zeros(len(y))
    for a, b in sk.split(X, y):
        m = HistGradientBoostingClassifier(random_state=seed).fit(X[a], y[a])
        p[b] = m.predict_proba(X[b])[:, 1]
    return float(roc_auc_score(y, p))

def mi_matrix(d, cols):
    k = len(cols); M = np.zeros((k, k))
    for i in range(k):
        mi = mutual_info_regression(d[cols].values, d[cols[i]].values, random_state=0)
        M[i] = mi
    return M

def mi_error(real, syn, cols):
    A = mi_matrix(real, cols); B = mi_matrix(syn, cols)
    np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    return float(np.linalg.norm(A - B, ord="fro"))

out = {}
for cohort, loader in [("tamponad", load_tamponade), ("aort", load_aortic)]:
    tr, te, FEATS, TARGET = loader()
    members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
    N_SYNTH = int((tr[TARGET] == 0).sum())
    GENS = build(members, FEATS, N_SYNTH, tr, TARGET)
    print(f"\n=== {cohort.upper()} | uye {len(members)} | sentetik {N_SYNTH} ===", flush=True)
    out[cohort] = {}
    for name, fn in GENS.items():
        det, ks, mie = [], [], []
        for s in SEEDS:
            try:
                syn = fn(s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
            except Exception as e:
                print(f"  {name} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
            if len(syn) < 50: continue
            det.append(detection_auc(members, syn, s))
            ks.append(float(np.mean([ks_2samp(syn[c], members[c]).statistic for c in FEATS])))
            mie.append(mi_error(members, syn, FEATS))
        if not det: continue
        out[cohort][name] = {"detection_auc": float(np.mean(det)),
                             "detection_sd": float(np.std(det)),
                             "ks_in": float(np.mean(ks)),
                             "mi_err": float(np.mean(mie)), "n": len(det)}
        r = out[cohort][name]
        print(f"  {name:17s} ayirt-edici AUC {r['detection_auc']:.3f} (sd {r['detection_sd']:.3f}) | "
              f"KS {r['ks_in']:.3f} | karsilikli-bilgi hatasi {r['mi_err']:.3f}", flush=True)

json.dump(out, open("phase_f.json", "w"), indent=1)
print("\nNOT: ayirt-edici AUC 0.5'e ne kadar yakinsa sentetik veri o kadar ayirt edilemez.", flush=True)
