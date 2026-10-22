import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Parametros fisicos v5 (mecanica regularizada)
c1 = -7.241572256248335
c2 = -0.034705431566244116
c3 = 0.003838173551740177
Fc = 0.12545756917619713
c_aero = 0.26828819941082166
u0 = 11.263721602969127
tau_m = 0.08
Ts = 0.01
nx = 3
N = 150

def f_plant(x, u):
    th, v, a = x
    thrust = max(a - u0, 0.0)**2
    fric = -Fc * np.tanh(50.0 * v)
    aero = -c_aero * v * abs(v)
    acc = c1 * np.sin(th) + c2 * v + aero + fric + c3 * thrust
    a_dot = (u - a) / tau_m
    return np.array([v, acc, a_dot])

def rk4_step(x, u):
    k1 = f_plant(x, u)
    k2 = f_plant(x + Ts / 2.0 * k1, u)
    k3 = f_plant(x + Ts / 2.0 * k2, u)
    k4 = f_plant(x + Ts * k3, u)
    return x + Ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def u_eq(th_deg):
    if th_deg <= 0.0: return 0.0
    val = -c1 * np.sin(np.radians(th_deg)) / c3
    return float(min(u0 + np.sqrt(val), 80.0)) if val > 0 else 0.0

MPC_DIR = r"c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP004_greybox_v5_mpc\2_mpc_python"
cache_dataset = os.path.join(MPC_DIR, "dataset_mpc_greybox_v5.npz")
model_path = os.path.join(MPC_DIR, "best_ann_greybox_v5.pth")

loaded = np.load(cache_dataset)
P_data = loaded['P_data']
U_data = loaded['U_data']

X_train_full, X_test, y_train_full, y_test = train_test_split(P_data, U_data, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

class MPCApproximator(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(MPCApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = nx + N
model = MPCApproximator(in_dim=input_dim, out_dim=1)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

Kp, Ki, Kd = 0.5793, 0.6647, 0.2

def run_simulation(x2ref_custom, use_res_integrator=True, ki_res=0.20):
    steps_c = len(x2ref_custom)
    tvec_c = np.arange(steps_c) * Ts
    x2ref_full_c = np.concatenate([x2ref_custom, np.full(N, x2ref_custom[-1])])

    x_curr = np.array([np.radians(x2ref_custom[0]), 0.0, u_eq(x2ref_custom[0])])
    y_sim_c = []
    u_sim_c = []

    u_i_c = u_eq(x2ref_custom[0])
    u_i_res_c = 0.0
    was_ann_c = False
    e_1_c = 0.0
    y_1_c = x2ref_custom[0]
    y_at_c = x2ref_custom[0]
    u_cmd = u_eq(x2ref_custom[0])

    T_PID_START = 5.0
    T_PID_END = 5.0

    for k in range(steps_c):
        t_c = k * Ts
        r_c = x2ref_custom[k]

        if t_c < T_PID_START or t_c >= (steps_c * Ts - T_PID_END):
            if was_ann_c:
                # Bumpless transfer: RNA -> PID
                erro_tr = r_c - y_at_c
                u_p_tr = Kp * erro_tr
                u_d_tr = -(Kd / Ts) * (y_at_c - y_1_c)
                u_i_c = u_cmd - u_p_tr - u_d_tr
                was_ann_c = False

            erro = r_c - y_at_c
            u_p = Kp * erro
            u_i_c = u_i_c + Ki * (Ts / 2.0) * (erro + e_1_c)
            u_d = -(Kd / Ts) * (y_at_c - y_1_c)
            u_calc = u_p + u_i_c + u_d
            u_cmd = float(np.clip(u_calc, -10.0, 80.0))
            if u_calc != u_cmd:
                u_i_c -= (u_calc - u_cmd)
            e_1_c = erro
            y_1_c = y_at_c
        else:
            if not was_ann_c:
                u_i_res_c = 0.0
                was_ann_c = True

            st_deg = [x_curr[0] * 180.0 / np.pi, x_curr[1] * 180.0 / np.pi, x_curr[2]]
            feat = np.concatenate([st_deg, x2ref_full_c[k : k + N]])
            feat_sc = scaler.transform(feat.reshape(1, -1))
            with torch.no_grad():
                u_nn = float(model(torch.tensor(feat_sc, dtype=torch.float32)).item())

            erro = r_c - y_at_c
            if use_res_integrator:
                u_i_res_c += ki_res * (Ts / 2.0) * (erro + e_1_c)
                u_i_res_c = float(np.clip(u_i_res_c, -10.0, 10.0))
                u_calc = u_nn + u_i_res_c
            else:
                u_calc = u_nn

            u_cmd = float(np.clip(u_calc, 0.0, 80.0))

            # Rastreamento PID no fundo
            u_p_tr = Kp * erro
            u_d_tr = -(Kd / Ts) * (y_at_c - y_1_c)
            u_i_c = u_cmd - u_p_tr - u_d_tr
            e_1_c = erro
            y_1_c = y_at_c

        x_curr = rk4_step(x_curr, u_cmd)
        y_at_c = x_curr[0] * 180.0 / np.pi
        y_sim_c.append(y_at_c)
        u_sim_c.append(u_cmd)

    y_sim_c = np.array(y_sim_c)
    mask = (tvec_c >= 5.0) & (tvec_c < steps_c * Ts - 5.0)
    rmse_ann = np.sqrt(np.mean((y_sim_c[mask] - x2ref_custom[mask])**2))
    rmse_total = np.sqrt(np.mean((y_sim_c - x2ref_custom)**2))
    return rmse_total, rmse_ann, y_sim_c

df_val1 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python.csv"))
df_val2 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python_val2_degraus.csv"))
df_val3 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python_val3_senoidal.csv"))

print("=== COMPARAÇÃO: SEM INTEGRADOR vs COM INTEGRADOR RESIDUAL (Ki_res=0.20) ===")
for name, df in [("Val1 Mista", df_val1), ("Val2 Degraus", df_val2), ("Val3 Senoidal", df_val3)]:
    r_tot_no, r_ann_no, _ = run_simulation(df['referencia'].values, use_res_integrator=False)
    r_tot_wi, r_ann_wi, _ = run_simulation(df['referencia'].values, use_res_integrator=True, ki_res=0.20)
    print(f"{name:15s} | SEM INTEGR.: RMSE Total = {r_tot_no:5.2f}°, RMSE RNA = {r_ann_no:5.2f}° | COM INTEGR.: RMSE Total = {r_tot_wi:5.2f}°, RMSE RNA = {r_ann_wi:5.2f}°")
