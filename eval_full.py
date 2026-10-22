import os, glob, sys, json
import numpy as np

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
    return dict(name=fname.replace('.csv',''), y=y, u=u, t=t, dt=dt, v0=v0)

train = [load_dataset(f) for f in TRAIN_NAMES]
val = [load_dataset(f) for f in VAL_NAMES]

def simulate(th, exp, L_obs=0.0, u0=11.26, tau_m=0.08):
    c1, c2, c3, Fc, ca = th
    dt = exp['dt']
    N = len(exp['y'])
    y_sim = np.zeros(N)
    y_sim[0] = exp['y'][0]
    v = exp['v0']
    a = exp['u'][0]
    th_pos = exp['y'][0]
    
    for k in range(N - 1):
        uk = exp['u'][k]
        y_meas = exp['y'][k]
        
        def f(st):
            p, vel, act = st
            thrust = max(act - u0, 0.0)**2
            fric = -Fc * np.tanh(50.0 * vel)
            aero = -ca * vel * abs(vel)
            acc = c1 * np.sin(p) + c2 * vel + aero + fric + c3 * thrust
            adot = (uk - act) / tau_m
            obs = L_obs * (y_meas - p)
            return np.array([vel + obs, acc, adot])
            
        st = np.array([th_pos, v, a])
        k1 = f(st)
        k2 = f(st + 0.5 * dt * k1)
        k3 = f(st + 0.5 * dt * k2)
        k4 = f(st + dt * k3)
        st_next = st + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        th_pos, v, a = st_next
        y_sim[k+1] = th_pos
        
    denom = np.linalg.norm(exp['y'] - np.mean(exp['y']))
    fit = 100.0 * (1 - np.linalg.norm(exp['y'] - y_sim) / (denom + 1e-9))
    rmse = np.rad2deg(np.sqrt(np.mean((exp['y'] - y_sim)**2)))
    return fit, rmse

th_opt = [-8.34, -0.40, 0.0043, 0.18, 0.60]

print("="*80)
print("AVALIAÇÃO EM MALHA ABERTA PURA (FREE-RUN)")
print("="*80)
for label, dataset in [("TREINO", train), ("VALIDACAO", val)]:
    print(f"\n--- {label} ---")
    fits, rmses = [], []
    for exp in dataset:
        fit, rmse = simulate(th_opt, exp, L_obs=0.0)
        fits.append(fit)
        rmses.append(rmse)
        print(f"  {exp['name']:28s} | FIT: {fit:6.1f}% | RMSE: {rmse:5.2f}°")
    print(f"  Média {label}: FIT = {np.mean(fits):.1f}%, RMSE = {np.mean(rmses):.2f}°")

print("\n" + "="*80)
print("AVALIAÇÃO COM OBSERVADOR (L=2.0)")
print("="*80)
for label, dataset in [("TREINO", train), ("VALIDACAO", val)]:
    print(f"\n--- {label} ---")
    fits, rmses = [], []
    for exp in dataset:
        fit, rmse = simulate(th_opt, exp, L_obs=2.0)
        fits.append(fit)
        rmses.append(rmse)
        print(f"  {exp['name']:28s} | FIT: {fit:6.1f}% | RMSE: {rmse:5.2f}°")
    print(f"  Média {label}: FIT = {np.mean(fits):.1f}%, RMSE = {np.mean(rmses):.2f}°")
