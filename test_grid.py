import os, glob, sys, json
import numpy as np
import casadi as cs

root_dir = os.path.abspath("C:/Users/vicio/Documents/SYSTEM-IDENTIFICATION-AERO-main")
sys.path.append(root_dir)
from aerodata import readData

TRAIN_NAMES = [
    "multi-seno-45-050Hz_0904_20-29.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "aprbs-45-1_0904_19-54.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "aprbs-60-1_0904_20-04.csv",
]
VAL_NAMES = [
    "multi-seno-45-030Hz_0904_20-23.csv",
    "chirp-45-amp35_0904_20-13.csv",
    "seq-degraus-45-2_0904_20-45.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
    "chirp-60-amp40_0904_20-16.csv",
    "seq-degraus-60-2_0904_20-54.csv",
]

def load_dataset(fname, trim=(10.0, 13.0)):
    fpath = os.path.join(root_dir, "data/experimentos/coleta_dados/RODADA-7", fname)
    y_deg, u, t = readData(fpath)[:3]
    i0 = np.searchsorted(t, t[0] + trim[0])
    i1 = np.searchsorted(t, t[-1] - trim[1])
    y = np.deg2rad(y_deg[i0:i1])
    u = u[i0:i1].astype(float)
    t = t[i0:i1] - t[i0]
    dt = float(np.mean(np.diff(t)))
    v0 = (y[1] - y[0]) / dt
    v = np.gradient(np.convolve(y, np.ones(5)/5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    return dict(name=fname.replace('.csv',''), y=y, u=u, t=t, dt=dt, v0=v0, v=v)

train = [load_dataset(f) for f in TRAIN_NAMES]
val = [load_dataset(f) for f in VAL_NAMES]

# Build CasADi RK4
def build_rk4(u0_val=11.26, tau_m_val=0.08):
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    h = cs.SX.sym('h')
    th = cs.SX.sym('th', 5)
    def f(x_, u_):
        th_pos, vel, act = x_[0], x_[1], x_[2]
        thrust = cs.fmax(act - u0_val, 0)**2
        fric = -th[3] * cs.tanh(50.0 * vel)
        aero = -th[4] * vel * cs.fabs(vel)
        acc = th[0] * cs.sin(th_pos) + th[1] * vel + aero + fric + th[2] * thrust
        adot = (u_ - act) / tau_m_val
        return cs.vertcat(vel, acc, adot)
    k1 = f(x, u)
    k2 = f(x + h/2 * k1, u)
    k3 = f(x + h/2 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return cs.Function('rk4', [x, u, th, h], [x_next])

rk4_fn = build_rk4()

def evaluate_th(th, datasets, label=""):
    print(f"\n--- Avaliando {label}: th={th} ---")
    c1, c2, c3, Fc, ca = th
    fits, rmses = [], []
    for exp in datasets:
        N = len(exp['y'])
        dt = exp['dt']
        y_sim = np.zeros(N)
        y_sim[0] = exp['y'][0]
        v = exp['v0']
        a = exp['u'][0]
        th_pos = exp['y'][0]
        for k in range(N - 1):
            uk = exp['u'][k]
            thrust = max(a - 11.26, 0.0)**2
            fric = -Fc * np.tanh(50.0 * v)
            aero = -ca * v * abs(v)
            acc = c1 * np.sin(th_pos) + c2 * v + aero + fric + c3 * thrust
            adot = (uk - a) / 0.08
            
            # Simple Euler / RK4
            k1_p, k1_v, k1_a = v, acc, adot
            
            p2 = th_pos + 0.5*dt*k1_p
            v2 = v + 0.5*dt*k1_v
            a2 = a + 0.5*dt*k1_a
            t2 = max(a2 - 11.26, 0.0)**2
            acc2 = c1*np.sin(p2) + c2*v2 - ca*v2*abs(v2) - Fc*np.tanh(50*v2) + c3*t2
            adot2 = (uk - a2)/0.08
            
            p3 = th_pos + 0.5*dt*v2
            v3 = v + 0.5*dt*acc2
            a3 = a + 0.5*dt*adot2
            t3 = max(a3 - 11.26, 0.0)**2
            acc3 = c1*np.sin(p3) + c2*v3 - ca*v3*abs(v3) - Fc*np.tanh(50*v3) + c3*t3
            adot3 = (uk - a3)/0.08
            
            p4 = th_pos + dt*v3
            v4 = v + dt*acc3
            a4 = a + dt*adot3
            t4 = max(a4 - 11.26, 0.0)**2
            acc4 = c1*np.sin(p4) + c2*v4 - ca*v4*abs(v4) - Fc*np.tanh(50*v4) + c3*t4
            adot4 = (uk - a4)/0.08
            
            th_pos += dt/6 * (k1_p + 2*v2 + 2*v3 + v4)
            v += dt/6 * (k1_v + 2*acc2 + 2*acc3 + acc4)
            a += dt/6 * (k1_a + 2*adot2 + 2*adot3 + adot4)
            y_sim[k+1] = th_pos
            
        denom = np.linalg.norm(exp['y'] - np.mean(exp['y']))
        fit = 100.0 * (1 - np.linalg.norm(exp['y'] - y_sim) / (denom + 1e-9))
        rmse = np.rad2deg(np.sqrt(np.mean((exp['y'] - y_sim)**2)))
        fits.append(fit)
        rmses.append(rmse)
        print(f"  {exp['name']:25s} | FIT: {fit:6.1f}% | RMSE: {rmse:5.2f}° | Max: {np.rad2deg(np.max(np.abs(y_sim))):5.1f}°")
    print(f"  --> Media FIT: {np.mean(fits):.1f}%, RMSE: {np.mean(rmses):.2f}° | Mediana: {np.median(fits):.1f}%")
    return fits, rmses

# Evaluate grid over c_aero and c2
for ca in [0.40, 0.45, 0.50, 0.55, 0.60]:
    for c2 in [-0.20, -0.30, -0.40]:
        for c3 in [0.0039, 0.0041, 0.0043]:
            th = [-8.34, c2, c3, 0.18, ca]
            # quick check on seq-degraus-60-2 and chirp-45-amp35
            # let's run on full val
            evaluate_th(th, val, f"ca={ca}, c2={c2}, c3={c3}")
