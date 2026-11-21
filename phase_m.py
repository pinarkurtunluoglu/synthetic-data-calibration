"""
Faz M: besinci hakem turu.

(1) Tablo 3'u duzeltilmis yapilandirmayla yeniden kos (CTGAN/TVAE kategorik
    sutun bildirilmis, batch 30). Ayni kosuda her bolmede her ureticinin
    FAYDASINI da olc, boylece "her gercek uretecten buyuk" iddiasi on bolmede
    gosterilebilsin.

(2) Gurultu ureticisinin BOLME ICI standart sapmasi: 50 cekimin kendi
    sapmasi. Analitik null SD 0.061 ile karsilastirilacak.

(3) Aort kohortunda fayda analizi: taban, sinif agirligi, gurultu-olen,
    gurultu-sagkalan, on bolme.

(4) "Olen hastalar daha bozuk" iddiasini destekleyecek tek sayi:
    olen ve sag kalan kayitlarin egitim merkezine ortalama uzakligi.

Cikti: phase_m.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

os.chdir("/home/claude/run")
SPLITS = list(range(10))
T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
OUTM = {}


def prep(df, target="mortality"):
    feats = [c for c in df.columns if c not in {"subject_id", target}]
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            v = sorted(df[c].dropna().unique()); df[c] = df[c].map({v[0]: 0.0, v[1]: 1.0})
    df[feats] = df[feats].astype("float64")
    return df, feats


TARGET = "mortality"
raw = pd.read_parquet("patient_matrix_24h.parquet")
sub = pd.read_parquet("matrix_imputed_with_subject.parquet")
D = raw.copy(); D["subject_id"] = sub["subject_id"].values; D[TARGET] = sub[TARGET].values
D = D.drop(columns=[c for c in ["hospital_expire_flag"] if c in D.columns])
D, FEATS = prep(D)
PT = D.groupby("subject_id")[TARGET].max().reset_index()


def split_of(df, pt, seed, feats):
    r = np.random.default_rng(seed); ids = []
    for lab in [0, 1]:
        a = pt[pt[TARGET] == lab]["subject_id"].to_numpy(copy=True); r.shuffle(a)
        ids.extend(a[:int(len(a) * .25)])
    m = df["subject_id"].isin(set(ids))
    tr, te = df[~m].copy(), df[m].copy()
    imp = KNNImputer(n_neighbors=10)
    tr[feats] = imp.fit_transform(tr[feats]); te[feats] = imp.transform(te[feats])
    return tr, te


def noise_rows(members, seed, n, feats):
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: r.uniform(members[c].min(), members[c].max(), n) for c in feats})


def prauc(Xtr, ytr, Xte, yte, seed, balanced=False):
    clf = HistGradientBoostingClassifier(
        random_state=seed, class_weight=("balanced" if balanced else None)).fit(Xtr, ytr)
    return float(average_precision_score(yte, clf.predict_proba(Xte)[:, 1]))


from imblearn.over_sampling import SMOTE, ADASYN
from ctgan import CTGAN, TVAE


def build(members, tr, n_synth, feats):
    cats = [c for c in feats if members[c].dropna().nunique() <= 2]

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

    def g_ctgan(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = CTGAN(epochs=300, batch_size=30, pac=1, verbose=False)
        m.fit(members, discrete_columns=cats); return m.sample(n_synth)[feats]

    def g_tvae(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = TVAE(epochs=300, batch_size=30)
        m.fit(members, discrete_columns=cats); return m.sample(n_synth)[feats]

    def g_ddpm(seed, steps=3000):
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
        return noise_rows(members, seed, n_synth, feats)

    return {"Copy (ceiling)": g_copy, "SMOTE": g_smote, "ADASYN": g_adasyn,
            "Diffusion": g_ddpm, "Gaussian copula": g_copula, "TVAE": g_tvae,
            "Vanilla GAN": g_gan, "CTGAN": g_ctgan, "Noise (floor)": g_noise}


# ============ (1)+(2) tamponad: on bolme, duzeltilmis yapilandirma ============
print("=" * 74)
print("(1) Tablo 3 duzeltilmis yapilandirma + her uretecin faydasi", flush=True)
rows, noise_within = [], []
t0 = time.time()
for sp in SPLITS:
    tr, te = split_of(D, PT, sp, FEATS)
    members = tr[tr[TARGET] == 1][FEATS].reset_index(drop=True)
    nonmembers = te[te[TARGET] == 1][FEATS].reset_index(drop=True)
    n_syn = int((tr[TARGET] == 0).sum()) - int((tr[TARGET] == 1).sum())
    SC = StandardScaler().fit(members); M = SC.transform(members); Xn = SC.transform(nonmembers)
    yy = np.r_[np.ones(len(M)), np.zeros(len(Xn))]
    dr = NearestNeighbors(n_neighbors=2).fit(M).kneighbors(M)[0][:, 1]
    P5 = float(np.percentile(dr, 5))
    Xtr, ytr = tr[FEATS].values, tr[TARGET].values
    Xte, yte = te[FEATS].values, te[TARGET].values
    base = prauc(Xtr, ytr, Xte, yte, sp)

    def auc_of(syn):
        k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
        return float(roc_auc_score(yy, -np.r_[k.kneighbors(M)[0].ravel(),
                                              k.kneighbors(Xn)[0].ravel()]))
    # bolme ici gurultu sapmasi (50 cekim)
    draws = [auc_of(noise_rows(members, 1000 * sp + j, n_syn, FEATS)) for j in range(50)]
    noise_within.append(float(np.std(draws, ddof=1)))

    G = build(members, tr, n_syn, FEATS)
    rec = {"split": sp, "n_synth": n_syn, "baseline": base}
    for name, fn in G.items():
        try:
            syn = fn(sp)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"   {name} HATA {type(e).__name__}", flush=True); continue
        if len(syn) < 50: continue
        kk = NearestNeighbors(n_neighbors=1).fit(M)
        ds = kk.kneighbors(SC.transform(syn))[0].ravel()
        Xa = np.vstack([Xtr, syn.values]); ya = np.r_[ytr, np.ones(len(syn))]
        rec[name] = {"nn": auc_of(syn),
                     "ks_in": float(np.mean([ks_2samp(syn[c], members[c]).statistic for c in FEATS])),
                     "dup": float(np.mean(ds < P5)),
                     "prauc": prauc(Xa, ya, Xte, yte, sp)}
    rows.append(rec)
    print(f"   bolme {sp} bitti ({(time.time()-t0)/60:.1f} dk)", flush=True)
    json.dump({"tamponade_splits": rows, "noise_within_sd": noise_within}, open("phase_m.json", "w"), indent=1)

OUTM["tamponade_splits"] = rows
OUTM["noise_within_sd"] = {"per_split": noise_within,
                           "mean": float(np.mean(noise_within))}
print(f"\n   bolme ici gurultu SD'si: ortalama {np.mean(noise_within):.3f} "
      f"(aralik {min(noise_within):.3f}-{max(noise_within):.3f})", flush=True)

GN = [k for k in rows[0] if k not in ("split", "n_synth", "baseline")]
print(f"\n   {'uretec':17s} {'NN ort':>7s} {'sd':>6s} {'KS':>6s} {'dup':>6s} {'PR-AUC':>7s} {'fark':>7s}")
base = np.array([r["baseline"] for r in rows])
summary = {}
for g in GN:
    nn = np.array([r[g]["nn"] for r in rows if g in r])
    ks = np.array([r[g]["ks_in"] for r in rows if g in r])
    dp = np.array([r[g]["dup"] for r in rows if g in r])
    pr = np.array([r[g]["prauc"] for r in rows if g in r])
    d = pr - base[:len(pr)]
    bs = np.array([np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(4000)])
    summary[g] = {"nn": float(nn.mean()), "nn_sd": float(nn.std(ddof=1)),
                  "nn_min": float(nn.min()), "nn_max": float(nn.max()),
                  "ks_in": float(ks.mean()), "dup": float(dp.mean()),
                  "prauc": float(pr.mean()), "prauc_diff": float(d.mean()),
                  "prauc_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    s_ = summary[g]
    print(f"   {g:17s} {s_['nn']:7.3f} {s_['nn_sd']:6.3f} {s_['ks_in']:6.3f} {s_['dup']:6.3f} "
          f"{s_['prauc']:7.3f} {s_['prauc_diff']:+7.3f} [{s_['prauc_ci'][0]:+.3f},{s_['prauc_ci'][1]:+.3f}]")
print(f"   {'taban':17s} {'':7s} {'':6s} {'':6s} {'':6s} {base.mean():7.3f}")
OUTM["tamponade_summary"] = summary
OUTM["tamponade_baseline"] = float(base.mean())
json.dump(OUTM, open("phase_m.json", "w"), indent=1)

# ============ (3) aort kohortunda fayda ============
print("\n" + "=" * 74)
print("(3) aort kohortunda fayda", flush=True)
ta = pd.read_parquet("aortic_train.parquet"); te_ = pd.read_parquet("aortic_test.parquet")
AO = pd.concat([ta, te_], ignore_index=True)
AO, FA = prep(AO)
PTA = AO.groupby("subject_id")[TARGET].max().reset_index()
arows = []
for sp in SPLITS:
    tr, te = split_of(AO, PTA, sp, FA)
    members = tr[tr[TARGET] == 1][FA].reset_index(drop=True)
    n_syn = int((tr[TARGET] == 0).sum()) - int((tr[TARGET] == 1).sum())
    Xtr, ytr = tr[FA].values, tr[TARGET].values
    Xte, yte = te[FA].values, te[TARGET].values
    syn = noise_rows(members, sp, n_syn, FA)
    b = prauc(Xtr, ytr, Xte, yte, sp)
    bw = prauc(Xtr, ytr, Xte, yte, sp, balanced=True)
    dd = prauc(np.vstack([Xtr, syn.values]), np.r_[ytr, np.ones(len(syn))], Xte, yte, sp)
    ss = prauc(np.vstack([Xtr, syn.values]), np.r_[ytr, np.zeros(len(syn))], Xte, yte, sp)
    arows.append({"split": sp, "baseline": b, "balanced": bw,
                  "noise_deceased": dd, "noise_survivor": ss})
    print(f"   bolme {sp}: taban {b:.3f} agirlikli {bw:.3f} olen {dd:.3f} sagkalan {ss:.3f}", flush=True)
AA = {k: np.array([r[k] for r in arows]) for k in arows[0] if k != "split"}
asum = {}
for k in ["balanced", "noise_deceased", "noise_survivor"]:
    d = AA[k] - AA["baseline"]
    bs = np.array([np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(4000)])
    asum[k] = {"mean": float(AA[k].mean()), "diff": float(d.mean()),
               "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
asum["baseline"] = {"mean": float(AA["baseline"].mean())}
OUTM["aortic_utility"] = {"rows": arows, "summary": asum}
print("\n   aort ozeti:")
for k, v in asum.items():
    if "diff" in v:
        print(f"     {k:18s} {v['mean']:.3f}  fark {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    else:
        print(f"     {k:18s} {v['mean']:.3f}")

# ============ (4) olen hastalar gercekten daha mi uc degerlerde ============
print("\n" + "=" * 74)
print("(4) olen ve sag kalan kayitlarin egitim merkezine uzakligi", flush=True)
for lab, df, feats in [("tamponad", D, FEATS), ("aort", AO, FA)]:
    tr, te = split_of(df, df.groupby("subject_id")[TARGET].max().reset_index(), 42, feats)
    sc = StandardScaler().fit(tr[feats])
    Z = sc.transform(tr[feats]); dist = np.linalg.norm(Z, axis=1)
    y = tr[TARGET].values
    print(f"   {lab}: olen {dist[y==1].mean():.2f} | sag kalan {dist[y==0].mean():.2f}")
    OUTM.setdefault("centre_distance", {})[lab] = {
        "deceased": float(dist[y == 1].mean()), "survivor": float(dist[y == 0].mean())}
json.dump(OUTM, open("phase_m.json", "w"), indent=1)
print("\nbitti", flush=True)
