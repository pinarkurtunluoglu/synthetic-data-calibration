"""
Ornek basina gradyanlari vektorlestirme denemesi.
Eski hal: her adimda len(idx) kez autograd.grad cagrisi (Python dongusu).
Yeni hal: vmap(grad(...)) ile tek cagri.
Once dogrulugu, sonra hizi olculuyor.
"""
import time, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad
warnings.filterwarnings("ignore")

torch.manual_seed(0)
T, CLIP, d, B = 200, 1.0, 14, 32
betas = torch.linspace(1e-4, .02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def make_net(dd):
    return nn.Sequential(nn.Linear(dd + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dd))

net = make_net(d)
xb = torch.randn(B, d); t = torch.randint(0, T, (B,)); nz = torch.randn(B, d)

# ---------- eski yol ----------
def old_way(net, xb, t, nz):
    params = list(net.parameters())
    f = lambda x, tt: net(torch.cat([x, tt.float().unsqueeze(1) / T], 1))
    accg = [torch.zeros_like(p) for p in params]
    for j in range(len(xb)):
        net.zero_grad()
        li = F.mse_loss(f(xb[j:j+1], t[j:j+1]), nz[j:j+1])
        gi = torch.autograd.grad(li, params)
        gnorm = torch.sqrt(sum((g ** 2).sum() for g in gi))
        scale = min(1.0, CLIP / (gnorm.item() + 1e-6))
        for A, g in zip(accg, gi): A += g * scale
    return accg

# ---------- yeni yol ----------
P = {k: v.detach() for k, v in net.named_parameters()}

def loss_one(p, x, tt, n):
    xin = torch.cat([x, (tt.float() / T).unsqueeze(0)]).unsqueeze(0)
    pred = functional_call(net, p, (xin,))
    return F.mse_loss(pred.squeeze(0), n)

gfn = vmap(grad(loss_one), in_dims=(None, 0, 0, 0))

def new_way(xb, t, nz):
    ps = gfn(P, xb, t, nz)                      # her biri (B, ...)
    names = list(ps.keys())
    flat = torch.cat([ps[k].reshape(len(xb), -1) for k in names], 1)
    sc = (CLIP / (flat.norm(dim=1) + 1e-6)).clamp(max=1.0)
    return {k: (ps[k] * sc.view(-1, *([1] * (ps[k].dim() - 1)))).sum(0) for k in names}

# ---------- dogruluk ----------
a = old_way(net, xb, t, nz)
b = new_way(xb, t, nz)
names = [k for k, _ in net.named_parameters()]
md = max(float((a[i] - b[names[i]]).abs().max()) for i in range(len(names)))
print(f"en buyuk fark: {md:.3e}")

# ---------- hiz ----------
t0 = time.time()
for _ in range(20): old_way(net, xb, t, nz)
t_old = (time.time() - t0) / 20
t0 = time.time()
for _ in range(20): new_way(xb, t, nz)
t_new = (time.time() - t0) / 20
print(f"eski {t_old*1000:.1f} ms/adim | yeni {t_new*1000:.1f} ms/adim | hizlanma {t_old/t_new:.1f}x")
print(f"800 adim: eski {t_old*800:.0f} sn -> yeni {t_new*800:.0f} sn")
print(f"40 kosu : eski {t_old*800*40/60:.0f} dk -> yeni {t_new*800*40/60:.0f} dk")
