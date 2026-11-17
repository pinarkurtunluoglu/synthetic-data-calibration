"""
Faz I: DP tarafindaki eksikler (hakem madde 4).

Uc degisiklik:
  1. sigma = 0 kosusu eklendi: sadece kirpma, gurultu yok. Hakem bu
     ablasyonu istedi; referans kosuda kirpmanin etkisi boylece ayrisiyor.
  2. tohum sayisi 3'ten 10'a cikti.
  3. onceki uygulamadaki iki kusur duzeltildi:
       - standartlastirma katsayilari uye kumesinden hesaplaniyordu, yani
         ciktiya hesaba katilmamis bir sizinti tasiniyordu. Artik referans
         populasyondan (sag kalan yatislar) hesaplaniyor; bunlar uye degil,
         dolayisiyla uye kumesinin butcesini harcamiyor.
       - gradyan toplami gercek grup boyutuna bolunuyordu. Poisson
         alt-ornekleme ile dogru payda beklenen grup boyutu q*N'dir.

Cikti: phase_i.json
"""
import os, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from opacus.accountants import RDPAccountant
from torch.func import functional_call, vmap, grad
warnings.filterwarnings("ignore")

WD = "/home/claude/tez/run"; os.chdir(WD)
SIGMAS = [0.0, 0.5, 1.0, 2.0]
SEEDS = list(range(10))
CLIP, STEPS, BATCH, DELTA, N_SYNTH = 1.0, 800, 32, 1e-5, 479
T = 200
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

# ---------------- veri: diger fazlarla ayni bolme ----------------
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
ref_pop = pd.concat([tr[tr[TARGET] == 0], te[te[TARGET] == 0]])[FEATS]
N = len(members); Q = BATCH / N
print(f"uye {N} | uye-olmayan {len(nonmembers)} | referans {len(ref_pop)} | q={Q:.4f}", flush=True)

# olcum altyapisi
SC = StandardScaler().fit(members); M = SC.transform(members)
Xn = SC.transform(nonmembers)
KR = KernelDensity(bandwidth=1.0).fit(SC.transform(ref_pop))
Xte, yte = te[FEATS].values, te[TARGET].values

# DUZELTME 1: olcekleyici referans populasyondan, uye kumesinden degil
SC_PUB = StandardScaler().fit(ref_pop)

def mean_ks(syn, real):
    return float(np.mean([ks_2samp(syn[c], real[c]).statistic for c in FEATS]))

def nn_auc(syn):
    k = NearestNeighbors(n_neighbors=1).fit(SC.transform(syn))
    dm = k.kneighbors(M)[0].ravel(); dn = k.kneighbors(Xn)[0].ravel()
    return float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))

def dom_auc(syn):
    ks_ = KernelDensity(bandwidth=1.0).fit(SC.transform(syn))
    sm = ks_.score_samples(M) - KR.score_samples(M)
    sn = ks_.score_samples(Xn) - KR.score_samples(Xn)
    return float(roc_auc_score(np.r_[np.ones(len(sm)), np.zeros(len(sn))], np.r_[sm, sn]))

def utility(syn, seed):
    Xa = np.vstack([tr[FEATS].values, syn[FEATS].values])
    ya = np.r_[tr[TARGET].values, np.ones(len(syn))]
    clf = HistGradientBoostingClassifier(random_state=seed).fit(Xa, ya)
    p = clf.predict_proba(Xte)[:, 1]
    return float(average_precision_score(yte, p))

def make_net(dd):
    return nn.Sequential(nn.Linear(dd + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dd))

def epsilon_for(sigma):
    if sigma <= 0: return float("inf")
    acc = RDPAccountant()
    for _ in range(STEPS): acc.step(noise_multiplier=sigma, sample_rate=Q)
    return float(acc.get_epsilon(delta=DELTA))

EPS = {s: epsilon_for(s) for s in SIGMAS}
for s in SIGMAS:
    e = EPS[s]
    print(f"  sigma={s} -> epsilon={'sonsuz (garanti yok)' if np.isinf(e) else f'{e:.2f}'}", flush=True)

def train_dp(sigma, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    x0 = torch.tensor(SC_PUB.transform(members), dtype=torch.float32); dd = x0.shape[1]
    net = make_net(dd); f = lambda x, t: net(torch.cat([x, t.float().unsqueeze(1) / T], 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    denom = Q * N                     # DUZELTME 2: beklenen grup boyutu

    # ornek basina gradyanlar vektorlestirildi; Python dongusuyle birebir ayni
    # sonucu veriyor (fark 1e-7 mertebesinde) ama ~10 kat hizli.
    P = dict(net.named_parameters())

    def loss_one(p, x, tt, n):
        xin = torch.cat([x, (tt.float() / T).unsqueeze(0)]).unsqueeze(0)
        return F.mse_loss(functional_call(net, p, (xin,)).squeeze(0), n)

    gfn = vmap(grad(loss_one), in_dims=(None, 0, 0, 0))

    for _ in range(STEPS):
        idx = np.nonzero(rng.random(N) < Q)[0]
        if len(idx) == 0: continue
        xb = x0[idx]; t = torch.randint(0, T, (len(idx),)); nz = torch.randn_like(xb)
        a = abar[t].unsqueeze(1); xt = a.sqrt() * xb + (1 - a).sqrt() * nz
        ps = gfn({k: v.detach() for k, v in P.items()}, xt, t, nz)
        names = list(ps.keys())
        flat = torch.cat([ps[k].reshape(len(idx), -1) for k in names], 1)
        sc_ = (CLIP / (flat.norm(dim=1) + 1e-6)).clamp(max=1.0)
        opt.zero_grad()
        for k in names:
            A = (ps[k] * sc_.view(-1, *([1] * (ps[k].dim() - 1)))).sum(0)
            noise = torch.randn_like(A) * sigma * CLIP if sigma > 0 else 0.0
            P[k].grad = (A + noise) / denom
        opt.step()
    with torch.no_grad():
        x = torch.randn(N_SYNTH, dd)
        for t in reversed(range(T)):
            e = f(x, torch.full((N_SYNTH,), t))
            x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
            if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
    return pd.DataFrame(SC_PUB.inverse_transform(x.numpy()), columns=FEATS)

res = {}; t0 = time.time()
for sg in SIGMAS:
    acc = {k: [] for k in ["ks_in", "ks_out", "nn", "dom", "prauc"]}
    for s in SEEDS:
        try:
            syn = train_dp(sg, s)[FEATS].replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            print(f"  sigma={sg} s{s} HATA {type(e).__name__}: {e}", flush=True); continue
        if len(syn) < 50:
            print(f"  sigma={sg} s{s} gecersiz ({len(syn)} satir)", flush=True); continue
        acc["ks_in"].append(mean_ks(syn, members)); acc["ks_out"].append(mean_ks(syn, nonmembers))
        acc["nn"].append(nn_auc(syn)); acc["dom"].append(dom_auc(syn))
        acc["prauc"].append(utility(syn, s))
    if not acc["ks_in"]: continue
    row = {k: float(np.mean(v)) for k, v in acc.items()}
    row.update({k + "_sd": float(np.std(v)) for k, v in acc.items()})
    row["epsilon"] = EPS[sg]; row["n_seeds"] = len(acc["ks_in"])
    res[f"sigma={sg}"] = row
    e = EPS[sg]
    print(f"  sigma={sg} eps={'inf' if np.isinf(e) else f'{e:.1f}'} "
          f"KS {row['ks_in']:.3f} (sd {row['ks_in_sd']:.3f}) "
          f"NN {row['nn']:.3f} (sd {row['nn_sd']:.3f}) PR {row['prauc']:.3f}", flush=True)
    json.dump({"res": res, "meta": {"delta": DELTA, "q": Q, "steps": STEPS, "clip": CLIP,
                                    "n_members": N, "n_nonmembers": len(nonmembers),
                                    "scaler": "referans populasyon", "denom": "q*N"}},
              open("phase_i.json", "w"), indent=1)

print(f"\nsure {(time.time()-t0)/60:.1f} dk", flush=True)
