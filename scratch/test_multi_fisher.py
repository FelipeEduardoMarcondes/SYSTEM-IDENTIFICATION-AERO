import os, sys, time
import numpy as np
import casadi as cs

dir_path = os.path.abspath('sysid/notebooks/identificação-collab/IDENT-FELIPE')
sys.path.append(dir_path)
sys.path.append(os.getcwd())
from aerodata import readData
from ident_greybox_prof import C_g

R7 = 'data/experimentos/coleta_dados/RODADA-7/'
TR = (10.0, 13.0)
TRAIN_NAMES = [
    'aprbs-45-3_0904_20-01.csv',
    'chirp-45-amp45_0904_20-15.csv',
    'multi-seno-45-050Hz_0904_20-29.csv',
    'seq-degraus-45-3_0904_20-48.csv',
]

def load_exp(path, trim_start, trim_end):
    res = readData(dataset_name=path, decimar=1, return_ref=True)
    y_deg, u, t = res[0], res[1], res[2]
    i0 = np.searchsorted(t, t[0] + trim_start)
    i1 = np.searchsorted(t, t[-1] - trim_end) if trim_end > 0 else len(t)
    y = np.deg2rad(y_deg[i0:i1])
    t = t[i0:i1] - t[i0]
    dt = float(np.mean(np.diff(t)))
    v = np.gradient(np.convolve(y, np.ones(5)/5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    return dict(name=os.path.basename(path), y=y, u=u[i0:i1].astype(float), t=t, dt=dt, v0=(y[1]-y[0])/dt)

train = [load_exp(R7 + f, *TR) for f in TRAIN_NAMES]

k_static = 5.39995e-4
u0_fixed = 11.42
tau_m_fixed = 0.08

th_free = cs.SX.sym('th_free', 3)
c1 = th_free[0]
c2 = th_free[1]
Fc = th_free[2]
c3 = -k_static * c1

x = cs.SX.sym('x', 3)
u_sym = cs.SX.sym('u')
h_sym = cs.SX.sym('h')

def f(x_, u_):
    th_pos, v, a = x_[0], x_[1], x_[2]
    thrust = cs.fmax(a - u0_fixed, 0)**2
    fric = -Fc * cs.tanh(50 * v)
    acc = c1 * cs.sin(th_pos) + c2 * v + fric + c3 * thrust
    a_dot = (u_ - a) / tau_m_fixed
    return cs.vertcat(v, acc, a_dot)

k1 = f(x, u_sym)
k2 = f(x + h_sym/2 * k1, u_sym)
k3 = f(x + h_sym/2 * k2, u_sym)
k4 = f(x + h_sym * k3, u_sym)
RK4 = cs.Function('rk4_sim', [x, u_sym, th_free, h_sym],
                  [x + h_sym/6 * (k1 + 2*k2 + 2*k3 + k4)])

F_total = np.zeros((3, 3))
r_total_sq = 0.0
n_total = 0

th_test = [-8.34, -0.4, 0.05]

for exp in train:
    N = len(exp['y'])
    n_total += N
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    name_clean = exp['name'][:5].replace('-', '_')
    sim = RK4.mapaccum(f'sim_{name_clean}_{N}', N - 1)
    X = sim(x0, cs.DM(exp['u'][:-1]).T, cs.repmat(th_free, 1, N - 1), exp['dt'])
    y_hat = cs.vertcat(exp['y'][0], X[0, :].T)
    r = cs.DM(exp['y']) - y_hat
    J = cs.jacobian(r, th_free)
    F = cs.mtimes(J.T, J)
    F_fn = cs.Function('F_fn', [th_free], [F, r])
    F_val, r_val = F_fn(th_test)
    F_total += np.array(F_val)
    r_total_sq += np.sum(np.array(r_val)**2)

s2 = r_total_sq / (n_total - 3)
cov = s2 * np.linalg.pinv(F_total)
std = np.sqrt(np.maximum(np.diag(cov), 1e-16))
corr = cov / np.outer(std, std)

names = ['c1', 'c2', 'Fc']
print("\nMulti-Experimento - Matriz de Correlação:")
print("     " + "".join(f"{n:>8s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>5s}" + "".join(f"{corr[i, j]:8.3f}" for j in range(3)))

print("\nDesvios-Padrão estimados:")
for i, n in enumerate(names):
    rel = std[i] / abs(th_test[i]) * 100
    print(f"  {n:>5s} = {th_test[i]:10.5e} ± {std[i]:10.5e} ({rel:5.2f}%)")
