"""Verify ULA invariant-measure mean-bias theory.

Checks:
 (A) Quadratic target: ULA stationary mean is exactly mu, variance 1/(1-eta/2).
 (B) Nonquadratic 1D target U = x^2/2 + eps*sin(2x) (globally strongly convex):
     first-order mean bias  E_{pi_eta}[x] - E_pi[x] = -(eta/4) Cov_pi(x, U'') + O(eta^2)
     and the modified-density prediction pi_eta ~ exp(-(U - (eta/4)U'^2 + (eta/2)U'')).
"""
import numpy as np

rng = np.random.default_rng(20260713)
eps = 0.2

def Up(x):  return x + 2.0 * eps * np.cos(2.0 * x)
def U(x):   return 0.5 * x * x + eps * np.sin(2.0 * x)
def Upp(x): return 1.0 - 4.0 * eps * np.sin(2.0 * x)

# ---------- (A) quadratic sanity ----------
eta = 0.2
n = 200_000
x = np.zeros(n)
for t in range(4000):
    x = x - eta * x + np.sqrt(2 * eta) * rng.standard_normal(n)
m_emp, v_emp = x.mean(), x.var()
print(f"[quadratic eta={eta}] mean={m_emp:+.5f} (exact 0), var={v_emp:.5f} (exact {1/(1-eta/2):.5f})")

# ---------- (B) quadrature references ----------
xg = np.linspace(-12.0, 12.0, 1_200_001)
w = np.exp(-(U(xg) - U(xg).min()))
w /= w.sum()
m0 = (xg * w).sum()
EUpp = (Upp(xg) * w).sum()
cov_x_DU = ((xg - m0) * (Upp(xg) - EUpp) * w).sum()
stein = -8.0 * eps * np.exp(-2.0)   # leading-order Stein value of Cov(x, U'') under N(0,1)
print(f"[target] E_pi[x] = {m0:+.6f},  Cov_pi(x,U'') = {cov_x_DU:+.6f}  (Stein leading order {stein:+.6f})")

hdr = f"{'eta':>5} | {'MC mean':>10} {'+-se':>8} | {'mod-dens mean':>13} | {'1st-order pred':>14} | {'bias MC':>9} {'bias th':>9}"
print(hdr); print("-" * len(hdr))
for eta in (0.05, 0.10, 0.20):
    # modified-density quadrature
    Ue = U(xg) - 0.25 * eta * Up(xg) ** 2 + 0.5 * eta * Upp(xg)
    we = np.exp(-(Ue - Ue.min())); we /= we.sum()
    m_mod = (xg * we).sum()
    # first-order formula
    m_th = m0 - 0.25 * eta * cov_x_DU
    # MC ULA
    chains, burn, keep = 250_000, 6000, 8000
    x = np.zeros(chains)
    for t in range(burn):
        x = x - eta * Up(x) + np.sqrt(2 * eta) * rng.standard_normal(chains)
    acc = np.zeros(chains)
    for t in range(keep):
        x = x - eta * Up(x) + np.sqrt(2 * eta) * rng.standard_normal(chains)
        acc += x
    pm = acc / keep
    m_mc, se = pm.mean(), pm.std(ddof=1) / np.sqrt(chains)
    print(f"{eta:5.2f} | {m_mc:+.6f} {se:8.1e} | {m_mod:+.6f}     | {m_th:+.6f}      | {m_mc-m0:+.2e} {m_th-m0:+.2e}")
