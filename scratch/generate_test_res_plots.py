import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
ENABLE_RESIDUAL_INTEGRATOR = True
KI_RES = 0.20
UI_RES_MAX = 10.0

def run_test_and_plot(x2ref_custom, name, out_png):
    steps_c = len(x2ref_custom)
    tvec_c = np.arange(steps_c) * Ts

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
    ann_end_idx = int(round((steps_c * Ts - T_PID_END) / Ts))

    # Cria horizonte sem antecipação espúria além do fim da janela da RNA
    x2ref_full_c = np.copy(x2ref_custom)
    for i in range(ann_end_idx - N, ann_end_idx):
        if i >= 0:
            # Não antecipa o retorno ao PID: mantém o patamar atual
            pass
    # Adiciona padding
    x2ref_full_c = np.concatenate([x2ref_custom, np.full(N, x2ref_custom[-1])])

    for k in range(steps_c):
        t_c = k * Ts
        r_c = x2ref_custom[k]

        if t_c < T_PID_START or t_c >= (steps_c * Ts - T_PID_END):
            if was_ann_c:
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
            
            # Horizonte preditivo seguro: antes de 70s, não deixa antecipar o salto de pouso
            ref_horizon = np.zeros(N)
            for j in range(N):
                idx_fut = k + j
                if idx_fut < ann_end_idx:
                    ref_horizon[j] = x2ref_custom[idx_fut]
                else:
                    ref_horizon[j] = x2ref_custom[ann_end_idx - 1] # Mantém patamar de voo até o handover

            feat = np.concatenate([st_deg, ref_horizon])
            feat_sc = scaler.transform(feat.reshape(1, -1))
            with torch.no_grad():
                u_nn = float(model(torch.tensor(feat_sc, dtype=torch.float32)).item())

            erro = r_c - y_at_c
            if ENABLE_RESIDUAL_INTEGRATOR:
                u_i_res_c += KI_RES * (Ts / 2.0) * (erro + e_1_c)
                u_i_res_c = float(np.clip(u_i_res_c, -UI_RES_MAX, UI_RES_MAX))
                u_calc = u_nn + u_i_res_c
            else:
                u_calc = u_nn

            u_cmd = float(np.clip(u_calc, 0.0, 80.0))

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
    rmse_tot = np.sqrt(np.mean((y_sim_c - x2ref_custom)**2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})
    ax1.plot(tvec_c, x2ref_custom, 'k--', label='Referência')
    ax1.plot(tvec_c, y_sim_c, 'b-', label=f'Saída Grey-Box v5 + Integrador Residual (RMSE RNA = {rmse_ann:.2f}°)')
    ax1.axvspan(0, T_PID_START, color='gray', alpha=0.15, label='Controle PID')
    ax1.axvspan(steps_c * Ts - T_PID_END, steps_c * Ts, color='gray', alpha=0.15)
    ax1.set_ylabel('Ângulo [°]')
    ax1.set_title(f'Python SIL Validation v5 - {name} (Com Integrador Residual e Transição Suave)')
    ax1.legend(loc='lower right' if 'Degraus' not in name else 'upper right')
    ax1.grid(True)

    ax2.plot(tvec_c, u_sim_c, 'b-', label='Controle u [%]')
    ax2.axhline(0, color='gray', ls='--', lw=0.7)
    ax2.axhline(80, color='gray', ls='--', lw=0.7)
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel('Controle u [%]')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"Salvo {out_png}: RMSE Total={rmse_tot:.2f}°, RMSE RNA={rmse_ann:.2f}°")

df_val1 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python.csv"))
df_val2 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python_val2_degraus.csv"))
df_val3 = pd.read_csv(os.path.join(MPC_DIR, "simulacao_python_val3_senoidal.csv"))

run_test_and_plot(df_val1['referencia'].values, "Validação 1 (Mista)", os.path.join(MPC_DIR, "test_res_val1.png"))
run_test_and_plot(df_val2['referencia'].values, "Validação 2 (Degraus)", os.path.join(MPC_DIR, "test_res_val2.png"))
run_test_and_plot(df_val3['referencia'].values, "Validação 3 (Senoidal)", os.path.join(MPC_DIR, "test_res_val3.png"))
