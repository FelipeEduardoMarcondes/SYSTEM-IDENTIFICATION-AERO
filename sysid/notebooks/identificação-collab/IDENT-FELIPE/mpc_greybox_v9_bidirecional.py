# %% [markdown]
# # MPC - Modelo Grey-Box v9 Bidirecional Completo (-60° a +130°)
# 
# Este script implementa o pipeline completo de Controle Preditivo Baseado em Modelo Não-Linear (NMPC)
# e aproximação por Rede Neural (ANN) para o Modelo Grey-Box v9 com **Atuação Bidirecional Completa**:
#   * Faixa de Operação Estendida: **-60° a +130°**
#   * Controle Bidirecional Real: $u \in [-80.0\%, +80.0\%]$ (ESC 3D: 1000 us = -100%, 1500 us = 0%, 2000 us = +100%)
#   * Estados (nx = 3): [theta (rad), theta_dot (rad/s), a_motor (%)]
#   * Equações Diferenciais da Física (v9 Bidirecional):
#       theta' = v
#       v'     = c1*sin(theta) - k_th*theta + c2*v - ca*v*|v| - Fc*tanh(50*v) + c3*sign(a)*(max(|a| - u0, 0))^p
#       a'     = (u - a) / tau_m
#   * Dinâmica do atuador: atraso de 1ª ordem tau_m ~ 113 ms
#   * Empuxo com expoente não-linear real p ~ 1.759 e zona morta simétrica u0 ~ 11.85%
#   * Freio aerodinâmico quadrático: ca ~ 0.0193 rad^-1
#   * Rigidez elástica de torção do chicote elétrico: k_th ~ 1.565 rad/s^2
#   * Restrições físicas: u in [-80, +80]%

# %%
# =============================================================================
# 1. Imports, Diretórios e Definições de Amostragem
# =============================================================================
import os
import sys
import time
import json
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from casadi import SX, MX, DM, Function, nlpsol, vertcat, fmax, tanh, sin, sqrt, fabs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Suporte UTF-8 no console Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ── Experimento ativo (Bidirecional) ─────────────────────────────────────────
EXP_ID = os.environ.get("EXP_ID", "EXP006_greybox_v9_bidirecional")
EXP_OUTPUT_DIR = os.path.join(root_dir, "data", "experimentos_novos", EXP_ID)
IDENT_DIR   = os.path.join(EXP_OUTPUT_DIR, "1_identificacao")
MPC_DIR     = os.path.join(EXP_OUTPUT_DIR, "2_mpc_python")
SIL_DIR     = os.path.join(EXP_OUTPUT_DIR, "3_sil_stm32")
REAL_DIR    = os.path.join(EXP_OUTPUT_DIR, "4_aeropendulo")
PLOTS_DIR   = MPC_DIR
SINAIS_REF_DIR = os.path.join(root_dir, "data", "sinais_referencia")

for _d in [IDENT_DIR, MPC_DIR, SIL_DIR, REAL_DIR, SINAIS_REF_DIR]:
    os.makedirs(_d, exist_ok=True)

# Parâmetros de amostragem
Ts = 0.01  # Período de amostragem: 10 ms (100 Hz, nativo da bancada)
nx = 3     # Estados: [theta (rad), omega (rad/s), a_motor (%)]

# %%
# =============================================================================
# 2. Carregamento dos Parâmetros e Modelo Grey-Box v9 no CasADi
# =============================================================================
json_path = os.path.join(current_dir, "modelo_greybox_v9.json")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        mod_data = json.load(f)
    
    m_key = "v9_otimizado"
    if m_key not in mod_data.get("models", {}):
        m_key = "v9_completo_r1" if "v9_completo_r1" in mod_data.get("models", {}) else list(mod_data["models"].keys())[0]
    
    p_dict = mod_data["models"][m_key]["params"]
    c1 = float(p_dict["c1"])
    c2 = float(p_dict["c2"])
    c3 = float(p_dict["c3"])
    ca = float(p_dict.get("ca", 0.019310))
    Fc = float(p_dict.get("Fc", 0.255820))
    u0 = float(p_dict.get("u0", 11.852171))
    tau_m = float(p_dict.get("tau_m", 0.113127))
    p_thrust = float(p_dict.get("p", 1.759243))
    k_th = float(p_dict.get("k_th", 1.565398))
    print(f"Parâmetros carregados de '{json_path}' [{m_key}]:")
else:
    # Valores nominais calibrados do Grey-Box v9
    c1 = -6.858314
    c2 = -0.331455
    c3 = 0.012412
    ca = 0.019310
    Fc = 0.255820
    u0 = 11.852171
    tau_m = 0.113127
    p_thrust = 1.759243
    k_th = 1.565398
    print("Utilizando parâmetros padrão do Modelo Grey-Box v9:")

print(f"  c1 = {c1:.5f} rad/s^2 | k_th = {k_th:.5f} rad/s^2 | c2 = {c2:.5f} s^-1")
print(f"  ca = {ca:.5f} rad^-1  | Fc = {Fc:.5f} rad/s^2   | c3 = {c3:.6e}")
print(f"  u0 = {u0:.3f}%        | tau_m = {tau_m*1000:.1f} ms  | p = {p_thrust:.4f}")

with open(os.path.join(IDENT_DIR, "parametros_greybox_v9_bidirecional.json"), "w", encoding="utf-8") as f:
    json.dump({
        "c1": c1, "c2": c2, "c3": c3, "ca": ca, "Fc": Fc,
        "u0": u0, "tau_m": tau_m, "p": p_thrust, "k_th": k_th,
        "u_min": -80.0, "u_max": 80.0, "th_min_deg": -60.0, "th_max_deg": 130.0
    }, f, indent=2)

# Regularização C^inf para derivadas exatas no IPOPT
eps_dz = 0.5   # Regularização suave da zona morta [%]
eps_dir = 0.1  # Regularização da transição de sinal

def f_dyn_mpc(x_vec, u_val):
    th = x_vec[0]
    v  = x_vec[1]
    a  = x_vec[2]

    # Empuxo bidirecional suave: dir * eff^p
    a_abs = sqrt(a**2 + 1e-4)
    z = a_abs - u0
    smooth_eff = 0.5 * (z + sqrt(z**2 + eps_dz**2))
    thrust_mag = smooth_eff ** p_thrust
    direction = a / sqrt(a**2 + eps_dir**2)
    thrust = direction * thrust_mag

    # Arrasto quadrático e atrito seco
    aero_quad = -ca * v * sqrt(v**2 + 1e-6)
    fric = -Fc * tanh(50.0 * v)

    # Dinâmica angular v9 completa
    acc = c1 * sin(th) - k_th * th + c2 * v + aero_quad + fric + c3 * thrust
    a_dot = (u_val - a) / fmax(tau_m, 0.01)
    return vertcat(v, acc, a_dot)

# Modelo da planta física (com sinal estrito)
def f_dyn_plant(x_vec, u_val):
    th = x_vec[0]
    v  = x_vec[1]
    a  = x_vec[2]

    a_abs = sqrt(a**2 + 1e-6)
    eff = fmax(a_abs - u0, 0.0)
    thrust_mag = eff ** p_thrust
    direction = a / a_abs
    thrust = direction * thrust_mag

    aero_quad = -ca * v * sqrt(v**2 + 1e-6)
    fric = -Fc * tanh(50.0 * v)

    acc = c1 * sin(th) - k_th * th + c2 * v + aero_quad + fric + c3 * thrust
    a_dot = (u_val - a) / fmax(tau_m, 0.01)
    return vertcat(v, acc, a_dot)

def make_integrator(dyn_func, name):
    x_s = SX.sym('x', nx)
    u_s = SX.sym('u')
    k1 = dyn_func(x_s, u_s)
    k2 = dyn_func(x_s + Ts / 2.0 * k1, u_s)
    k3 = dyn_func(x_s + Ts / 2.0 * k2, u_s)
    k4 = dyn_func(x_s + Ts * k3, u_s)
    x_nxt = x_s + Ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Function(name, [x_s, u_s], [x_nxt, x_nxt[0]], ['x0', 'p'], ['xf', 'yk'])

F_mpc = make_integrator(f_dyn_mpc, 'F_mpc')
F_plant = make_integrator(f_dyn_plant, 'F_plant')

# Função analítica exata de equilíbrio estático bidirecional:
# c3 * sign(u) * (|u| - u0)^p = -c1 * sin(theta) + k_th * theta
def u_equilibrium(th_deg):
    th_rad = np.radians(th_deg)
    restoring = -c1 * np.sin(th_rad) + k_th * th_rad
    if abs(restoring) < 1e-6:
        return 0.0
    sign_restoring = np.sign(restoring)
    val = abs(restoring) / c3
    u_mag = u0 + (val ** (1.0 / p_thrust))
    u_val = sign_restoring * u_mag
    return float(np.clip(u_val, -80.0, 80.0))

print("\nVerificação dos Pontos de Equilíbrio Estático Bidirecional:")
for test_angle in [-60.0, -45.0, -30.0, 0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 130.0]:
    print(f"  Theta = {test_angle:+6.1f}°  -->  u_eq = {u_equilibrium(test_angle):+6.2f}% PWM")

# %%
# =============================================================================
# 3. Setup do NMPC Ótimo Bidirecional (CasADi + IPOPT)
# =============================================================================
N = 100  # Horizonte preditivo (N * Ts = 100 * 0.01 s = 1.0 s à frente)

data = {
    'Ts': Ts,
    'x0': np.zeros(nx),
    'u_min': np.array([-80.0]),    # Limite reverso (-80%)
    'u_max': np.array([80.0]),     # Limite positivo (+80%)
    'u_guess': np.array([0.0]),
    'x_guess': np.zeros(nx),
    'tol': 1e-5,
}

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []
J = 0

xk_param = SX.sym('xk_param', nx)
Pref = SX.sym('Pref', N)
u_prev_param = SX.sym('u_prev_param', 1)

# Estado inicial no horizonte: x0 == xk_param
xk = SX.sym('x0', nx)
w.append(xk)
lbw.extend([-np.inf] * nx)
ubw.extend([np.inf] * nx)
w0.extend(data['x_guess'])

g.append(xk - xk_param)
lbg.extend([0.0] * nx)
ubg.extend([0.0] * nx)

for k in range(N):
    uk = SX.sym(f'u_{k}', 1)
    w.append(uk)
    lbw.append(float(data['u_min'][0]))
    ubw.append(float(data['u_max'][0]))
    w0.append(float(data['u_guess'][0]))

    Fk = F_mpc(x0=xk, p=uk)
    xnext = Fk['xf']
    yk = Fk['yk'] * 180.0 / np.pi

    if k == 0:
        du = uk - u_prev_param
    else:
        du = uk - u_prev
    u_prev = uk

    # Custo: Rastreamento angular (500) + esforço de controle (0.1) + suavidade de controle (200)
    J = J + 500.0 * (yk - Pref[k])**2 + 0.1 * uk**2 + 200.0 * du**2

    xk = SX.sym(f'x_{k+1}', nx)
    w.append(xk)
    lbw.extend([-np.inf] * nx)
    ubw.extend([np.inf] * nx)
    w0.extend(data['x_guess'])

    g.append(xk - xnext)
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)

w = vertcat(*w)
lbw = np.array(lbw, dtype=np.float64)
ubw = np.array(ubw, dtype=np.float64)
w0 = np.array(w0, dtype=np.float64)
g = vertcat(*g)
lbg = np.array(lbg, dtype=np.float64)
ubg = np.array(ubg, dtype=np.float64)

nlp = {'x': w, 'g': g, 'f': J, 'p': vertcat(xk_param, Pref, u_prev_param)}
solver = nlpsol('solver', 'ipopt', nlp, {
    'ipopt.print_level': 0,
    'ipopt.tol': data['tol'],
    'ipopt.max_iter': 40,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.warm_start_bound_push': 1e-4,
    'ipopt.warm_start_mult_bound_push': 1e-4,
    'ipopt.mu_init': 1e-3,
    'print_time': 0
})

def build_guess(x_curr, ref_seq):
    """Constrói chute inicial analítico para warm-start robusto do IPOPT."""
    w_g = []
    w_g.extend(x_curr)
    for k in range(N):
        r_deg = ref_seq[k]
        u_eq = u_equilibrium(r_deg)
        w_g.append(u_eq)
        w_g.extend([np.radians(r_deg), 0.0, u_eq])
    return np.array(w_g)

# %%
# =============================================================================
# 4. Geração do Sinal de Treino Rico em Faixa Total (-60° a +130°)
# =============================================================================
np.random.seed(42)

HOLD_TIME = 4.0   # Patamares estáticos entre manobras [s]
f_max = 0.50      # Frequência máxima de excitação dinâmica [Hz]

# --- BLOCO 1: Nível DC = 0° (Ancoragem em Repouso e Transição Inicial) ---
hold_0 = np.full(int(round(HOLD_TIME / Ts)), 0.0)

# --- BLOCO 2: Dinâmica no Quadrante Negativo (DC = -30°, Excursão: -60° a 0°) ---
chirp_dur_neg = 16.0
n_ch_neg = int(round(chirp_dur_neg / Ts))
t_ch_neg = np.arange(n_ch_neg) * Ts
f0_ch, f1_ch = 0.05, f_max
phase_ch_neg = 2.0 * np.pi * (f0_ch * t_ch_neg + ((f1_ch - f0_ch) / (2.0 * chirp_dur_neg)) * t_ch_neg**2)
chirp_neg = -30.0 + 30.0 * np.sin(phase_ch_neg)  # -60° a 0°

ms_dur_neg = 50.0
n_ms_neg = int(round(ms_dur_neg / Ts))
t_ms_neg = np.arange(n_ms_neg) * Ts
df_neg = 1.0 / ms_dur_neg
freqs_neg = np.arange(df_neg, f_max + 1e-9, df_neg)
phases_neg = np.random.uniform(0, 2 * np.pi, len(freqs_neg))
ms_neg = np.sum([np.sin(2 * np.pi * f * t_ms_neg + ph) for f, ph in zip(freqs_neg, phases_neg)], axis=0)
ms_neg = (ms_neg / np.max(np.abs(ms_neg))) * 30.0 - 30.0  # -60° a 0°

n_steps_neg = 20
step_pieces_neg = []
for _ in range(n_steps_neg):
    S = np.random.uniform(-60.0, 0.0)
    dur = np.random.uniform(2.5, 4.0)
    step_pieces_neg.append(np.full(int(round(dur / Ts)), S))
steps_neg = np.concatenate(step_pieces_neg)

# --- BLOCO 3: Dinâmica no Nível Intermediário (DC = 45°, Excursão: 0° a 90°) ---
hold_45 = np.full(int(round(HOLD_TIME / Ts)), 45.0)
chirp_dur_45 = 16.0
n_ch_45 = int(round(chirp_dur_45 / Ts))
t_ch_45 = np.arange(n_ch_45) * Ts
phase_ch_45 = 2.0 * np.pi * (f0_ch * t_ch_45 + ((f1_ch - f0_ch) / (2.0 * chirp_dur_45)) * t_ch_45**2)
chirp_45 = 45.0 + 45.0 * np.sin(phase_ch_45)  # 0° a 90°

ms_dur_45 = 50.0
n_ms_45 = int(round(ms_dur_45 / Ts))
t_ms_45 = np.arange(n_ms_45) * Ts
df_45 = 1.0 / ms_dur_45
freqs_45 = np.arange(df_45, f_max + 1e-9, df_45)
phases_45 = np.random.uniform(0, 2 * np.pi, len(freqs_45))
ms_45 = np.sum([np.sin(2 * np.pi * f * t_ms_45 + ph) for f, ph in zip(freqs_45, phases_45)], axis=0)
ms_45 = (ms_45 / np.max(np.abs(ms_45))) * 45.0 + 45.0  # 0° a 90°

n_steps_45 = 20
step_pieces_45 = []
for _ in range(n_steps_45):
    S = np.random.uniform(0.0, 90.0)
    dur = np.random.uniform(2.5, 4.0)
    step_pieces_45.append(np.full(int(round(dur / Ts)), S))
steps_45 = np.concatenate(step_pieces_45)

# --- BLOCO 4: Dinâmica no Nível Elevado (DC = 65°, Excursão: 0° a 130°) ---
hold_65 = np.full(int(round(HOLD_TIME / Ts)), 65.0)
chirp_dur_65 = 16.0
n_ch_65 = int(round(chirp_dur_65 / Ts))
t_ch_65 = np.arange(n_ch_65) * Ts
phase_ch_65 = 2.0 * np.pi * (f0_ch * t_ch_65 + ((f1_ch - f0_ch) / (2.0 * chirp_dur_65)) * t_ch_65**2)
chirp_65 = 65.0 + 65.0 * np.sin(phase_ch_65)  # 0° a 130°

ms_dur_65 = 50.0
n_ms_65 = int(round(ms_dur_65 / Ts))
t_ms_65 = np.arange(n_ms_65) * Ts
df_65 = 1.0 / ms_dur_65
freqs_65 = np.arange(df_65, f_max + 1e-9, df_65)
phases_65 = np.random.uniform(0, 2 * np.pi, len(freqs_65))
ms_65 = np.sum([np.sin(2 * np.pi * f * t_ms_65 + ph) for f, ph in zip(freqs_65, phases_65)], axis=0)
ms_65 = (ms_65 / np.max(np.abs(ms_65))) * 65.0 + 65.0  # 0° a 130°

n_steps_65 = 20
step_pieces_65 = []
for _ in range(n_steps_65):
    S = np.random.uniform(0.0, 130.0)
    dur = np.random.uniform(2.5, 4.0)
    step_pieces_65.append(np.full(int(round(dur / Ts)), S))
steps_65 = np.concatenate(step_pieces_65)

# --- BLOCO 5: Cruzamentos Amplos Transversais (-60° a +130°) ---
# Chirp amplo cruzando o zero (-55° a +125°)
chirp_dur_wide = 20.0
n_ch_wide = int(round(chirp_dur_wide / Ts))
t_ch_wide = np.arange(n_ch_wide) * Ts
phase_ch_wide = 2.0 * np.pi * (f0_ch * t_ch_wide + ((f1_ch - f0_ch) / (2.0 * chirp_dur_wide)) * t_ch_wide**2)
chirp_wide = 35.0 + 90.0 * np.sin(phase_ch_wide)  # -55° a +125°

# Degraus amplos e desafiadores alternando extremos
broad_levels = [-50.0, 60.0, -30.0, 110.0, -60.0, 130.0, 0.0, -45.0, 80.0, -20.0, 120.0, 0.0]
broad_pieces = [np.full(int(round(3.5 / Ts)), lvl) for lvl in broad_levels]
steps_wide = np.concatenate(broad_pieces)

# --- Montagem da Sequência Completa de Treinamento ---
x2ref_train = np.clip(np.concatenate([
    hold_0,
    chirp_neg, ms_neg, steps_neg,
    hold_0,
    chirp_45, ms_45, steps_45,
    hold_65,
    chirp_65, ms_65, steps_65,
    hold_0,
    chirp_wide, steps_wide,
    hold_0
]), -60.0, 130.0)

steps_train = len(x2ref_train)
tvec_train = np.arange(steps_train) * Ts

print(f"\nSequência de Treino Bidirecional Gerada:")
print(f"  Total de passos: {steps_train} ({steps_train*Ts:.1f} segundos)")
print(f"  Mínimo: {x2ref_train.min():.1f}° | Máximo: {x2ref_train.max():.1f}°")

plt.figure(figsize=(14, 4))
plt.plot(tvec_train, x2ref_train, 'k-', lw=1.2, label='Referência Bidirecional (-60° a +130°)')
plt.axhline(0.0, color='gray', ls='--', lw=0.8, label='Zero / Repouso')
plt.axhline(-60.0, color='red', ls=':', lw=0.9, label='Limite Inferior (-60°)')
plt.axhline(130.0, color='blue', ls=':', lw=0.9, label='Limite Superior (+130°)')
plt.xlabel('Tempo [s]')
plt.ylabel('Ângulo [°]')
plt.title(f'Bidirectional Training Trajectory (-60° to +130° | {steps_train*Ts:.1f} s)', fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "referencia_treino_bidirecional.png"), dpi=130)
plt.close()

# %%
# =============================================================================
# 5. Simulação do MPC Ótimo sobre o Sinal Rico e Coleta do Dataset
# =============================================================================
RECOMPUTE_MPC = globals().get('RECOMPUTE_MPC', True)
dataset_cache_file = os.path.join(MPC_DIR, "dataset_mpc_treino_bidirecional.npz")

if os.path.exists(dataset_cache_file) and not RECOMPUTE_MPC:
    print(f"\n[CACHE] Carregando dataset NMPC do arquivo: {dataset_cache_file} ...")
    loaded_data = np.load(dataset_cache_file)
    P_data = loaded_data['P_data']
    U_data = loaded_data['U_data']
    ysim_train = list(loaded_data['ysim_train'])
    usim_train = list(loaded_data['usim_train'])
    xsim_train = loaded_data['xsim_train']
    sim_steps_train = len(ysim_train)
    print(f"[CACHE] Dataset carregado com sucesso! ({len(P_data)} amostras)\n")
else:
    xsim_train = np.zeros((nx, 1))
    xsim_train[:, 0] = [np.radians(x2ref_train[0]), 0.0, u_equilibrium(x2ref_train[0])]
    ysim_train = []
    usim_train = []
    u_prev_sim = u_equilibrium(x2ref_train[0])
    w0_val = build_guess(xsim_train[:, 0], x2ref_train[:N])

    sim_steps_train = steps_train - N
    print(f"\nSimulando MPC sobre sinal rico para geração do dataset ({sim_steps_train} passos)...")

    for k in tqdm(range(sim_steps_train), desc="Coleta Dataset MPC Bidirecional"):
        ref_window = x2ref_train[k : k + N]
        pval = np.concatenate([xsim_train[:, -1], ref_window, [u_prev_sim]])

        # Atualiza chute inicial quando há grandes degraus
        if abs(ref_window[0] - xsim_train[0, -1] * 180.0 / np.pi) > 15.0 or abs(u_prev_sim) < 1.0:
            w0_val = build_guess(xsim_train[:, -1], ref_window)

        sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            w0_retry = build_guess(xsim_train[:, -1], ref_window)
            sol = solver(x0=w0_retry, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)

        w_opt = sol['x'].full().flatten()
        u_opt = float(w_opt[nx])

        # Injeção de ruído exploratório na planta física (+- 3.0%)
        u_applied = np.random.normal(u_opt, 3.0)
        u_applied = np.clip(u_applied, data['u_min'][0], data['u_max'][0])

        sim_step = F_plant(x0=xsim_train[:, -1], p=u_applied)
        xk1 = sim_step['xf'].full().flatten()
        yk = sim_step['yk'].full().item() * 180.0 / np.pi

        xsim_train = np.c_[xsim_train, xk1]
        usim_train.append(u_opt)  # Target para a ANN é o controle ótimo limpo
        ysim_train.append(yk)

        # Shifting do horizonte
        w_shifted = np.zeros_like(w_opt)
        w_shifted[:nx] = xk1
        for i in range(N - 1):
            w_shifted[3 + i*4] = w_opt[3 + (i+1)*4]
            w_shifted[4 + i*4 : 4 + i*4 + 3] = w_opt[4 + (i+1)*4 : 4 + (i+1)*4 + 3]
        w_shifted[3 + (N-1)*4] = w_opt[3 + (N-1)*4]
        w_shifted[4 + (N-1)*4 : 4 + (N-1)*4 + 3] = w_opt[4 + (N-1)*4 : 4 + (N-1)*4 + 3]
        w0_val = w_shifted
        u_prev_sim = u_applied

    # Montagem dos tensores de entrada (Features = [theta, omega, a, r(k)...r(k+N-1)])
    P_list = []
    for k in range(sim_steps_train):
        st_deg = [xsim_train[0, k] * 180.0 / np.pi, xsim_train[1, k] * 180.0 / np.pi, xsim_train[2, k]]
        feat = np.concatenate([st_deg, x2ref_train[k : k + N]])
        P_list.append(feat)

    P_data = np.array(P_list)
    U_data = np.array(usim_train).reshape(-1, 1)

    np.savez_compressed(dataset_cache_file, P_data=P_data, U_data=U_data, xsim_train=xsim_train,
                        ysim_train=np.array(ysim_train), usim_train=np.array(usim_train))
    print(f"[CACHE] Dataset NMPC salvo em cache: {dataset_cache_file}\n")

print(f"Dataset pronto: P_data shape = {P_data.shape}, U_data shape = {U_data.shape}")

# Gráficos da coleta de treino
tvec_sim_train = tvec_train[:sim_steps_train]
plt.figure(figsize=(14, 5))
plt.plot(tvec_sim_train, ysim_train, color='#1f77b4', lw=1.3, label='Saída do Aeropêndulo y(t)')
plt.step(tvec_sim_train, x2ref_train[:sim_steps_train], where='post', color='k', ls='--', label='Referência r(t)')
plt.axhline(0.0, color='gray', ls=':', lw=0.8)
plt.ylabel('Ângulo [°]')
plt.xlabel('Tempo [s]')
plt.title('Bidirectional NMPC Control - Training Tracking (-60° to +130°)', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_treino_rastreamento_bidirecional.png"), dpi=130)
plt.close()

plt.figure(figsize=(14, 4))
plt.step(tvec_sim_train, usim_train, where='post', color='r', lw=1.1, label='Controle Ótimo u [%]')
plt.axhline(0.0, color='k', ls='--', lw=0.8)
plt.axhline(80.0, color='gray', ls=':')
plt.axhline(-80.0, color='gray', ls=':')
plt.ylabel('Controle [%]')
plt.xlabel('Tempo [s]')
plt.title('Bidirectional NMPC Control - Optimal Effort u(t)', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_treino_controle_bidirecional.png"), dpi=130)
plt.close()

# %%
# =============================================================================
# 6. Criação e Treinamento Rigoroso da Rede Neural (PyTorch MLP)
# =============================================================================
input_dim = nx + N  # 3 estados + 100 referências futuras = 103 entradas

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

scaler = StandardScaler()
P_scaled = scaler.fit_transform(P_data)

# Divisão estratificada dos dados: 80% treino, 10% validação, 10% teste
X_train_val, X_test, y_train_val, y_test = train_test_split(P_scaled, U_data, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)

tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
tensor_X_val   = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val   = torch.tensor(y_val, dtype=torch.float32)
tensor_X_test  = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test  = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), batch_size=64, shuffle=True)

model = MPCApproximator(in_dim=input_dim, out_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

epochs = 180
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
best_weights = copy.deepcopy(model.state_dict())

print(f"\nTreinando Rede Neural Rigorosa (MLP {input_dim} -> 128 -> 128 -> 1) por {epochs} épocas...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    with torch.no_grad():
        v_out = model(tensor_X_val)
        v_loss = criterion(v_out, tensor_y_val).item()
    val_loss_history.append(v_loss)
    scheduler.step(v_loss)

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_weights = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
        lr_curr = optimizer.param_groups[0]['lr']
        print(f"  Época {epoch+1:3d}/{epochs} | Treino: {train_loss:.5f} | Val: {v_loss:.5f} | lr: {lr_curr:.1e}")

# Restaura os melhores pesos observados
model.load_state_dict(best_weights)
print(f"  -> Treinamento concluído. Melhor Val Loss: {best_val_loss:.5f}")

torch.save(model.state_dict(), os.path.join(MPC_DIR, "best_ann_greybox_v9_bidirecional.pth"))

# Gráfico de Perda
plt.figure(figsize=(8, 4))
plt.plot(train_loss_history, label='Perda Treino (MSE)', color='#1f77b4')
plt.plot(val_loss_history, label='Perda Validação (MSE)', color='#ff7f0e')
plt.title("Loss Curve - Bidirectional MPC Approximator", fontweight='bold')
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ann_loss_bidirecional.png"), dpi=130)
plt.close()

# Avaliação R2 (Treino, Validação, Teste)
model.eval()
with torch.no_grad():
    y_tr_pred = model(tensor_X_train).numpy()
    y_va_pred = model(tensor_X_val).numpy()
    y_te_pred = model(tensor_X_test).numpy()

r2_tr = r2_score(y_train, y_tr_pred)
r2_va = r2_score(y_val, y_va_pred)
r2_te = r2_score(y_test, y_te_pred)

print(f"\nAvaliação de Desempenho R²:")
print(f"  R² Treino    : {r2_tr:.5f}")
print(f"  R² Validação : {r2_va:.5f}")
print(f"  R² Teste     : {r2_te:.5f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sets = [(y_train, y_tr_pred, 'Treino', '#1f77b4', axes[0], r2_tr),
        (y_val, y_va_pred, 'Validação', '#2ca02c', axes[1], r2_va),
        (y_test, y_te_pred, 'Teste', '#ff7f0e', axes[2], r2_te)]

for y_true, y_pred, name, col, ax, r2_val in sets:
    ax.scatter(y_true, y_pred, alpha=0.25, color=col, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_title(f"Conjunto de {name}\nR²: {r2_val:.4f}", fontweight='bold')
    ax.set_xlabel("Controle Ótimo u [%]")
    ax.set_ylabel("Controle Predito pela ANN u [%]")
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ann_r2_bidirecional.png"), dpi=130)
plt.close()

# %%
# =============================================================================
# 7. Exportação dos Pesos para C Puro (ann_weights_bidirecional.h)
# =============================================================================
def export_ann_to_c(mod, sc, filename, guard_name="ANN_WEIGHTS_BIDIRECIONAL_H"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"#ifndef {guard_name}\n#define {guard_name}\n\n")
        f.write("/* Pesos da Rede Neural e Simulador Fisico Grey-Box v9 Bidirecional (-60 a +130 deg) */\n")
        f.write("#define MODEL_TYPE_GREYBOX_V9 1\n")
        f.write("#define MODEL_TYPE_GREYBOX_V9_BIDIRECIONAL 1\n")
        f.write(f"#define MPC_HORIZON {N}\n")
        f.write(f"#define NN_INPUT_DIM {input_dim}\n")
        f.write("#define NY_MODEL 1\n")
        f.write("#define NU_MODEL 1\n")
        f.write(f"#define NX_SYSTEM {nx}\n\n")

        f.write("/* Buffers unificados para telemetria e SIL */\n")
        f.write("static float sil_y_hist[NY_MODEL] = {0};\n")
        f.write("static float sil_u_hist[NU_MODEL] = {0};\n\n")

        f.write('#include "model_physical.h"\n\n')

        # Scaler
        f.write(f"const float scaler_mean[{len(sc.mean_)}] = {{\n  ")
        f.write(", ".join([f"{val:.8f}f" for val in sc.mean_]))
        f.write("\n};\n\n")

        f.write(f"const float scaler_scale[{len(sc.scale_)}] = {{\n  ")
        f.write(", ".join([f"{val:.8f}f" for val in sc.scale_]))
        f.write("\n};\n\n")

        # Pesos
        state = mod.state_dict()
        layer = 0
        for name, tensor in state.items():
            arr = tensor.cpu().numpy()
            if "weight" in name:
                flat = arr.flatten()
                f.write(f"// Layer {layer} Weights: shape {arr.shape}\n")
                f.write(f"const float W{layer}[{len(flat)}] = {{\n  ")
                f.write(", ".join([f"{val:.8f}f" for val in flat]))
                f.write("\n};\n\n")
            elif "bias" in name:
                f.write(f"// Layer {layer} Biases: shape {arr.shape}\n")
                f.write(f"const float b{layer}[{len(arr)}] = {{\n  ")
                f.write(", ".join([f"{val:.8f}f" for val in arr]))
                f.write("\n};\n\n")
                layer += 1

        f.write("static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }\n\n")
        f.write("static float ann_predict(float *input) {\n")
        f.write("    float norm_in[NN_INPUT_DIM];\n")
        f.write("    for (int i = 0; i < NN_INPUT_DIM; i++) {\n")
        f.write("        norm_in[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];\n")
        f.write("    }\n\n")
        f.write("    float h1[128];\n")
        f.write("    for (int i = 0; i < 128; i++) {\n")
        f.write("        float sum = b0[i];\n")
        f.write("        for (int j = 0; j < NN_INPUT_DIM; j++) {\n")
        f.write("            sum += W0[i * NN_INPUT_DIM + j] * norm_in[j];\n")
        f.write("        }\n")
        f.write("        h1[i] = relu(sum);\n")
        f.write("    }\n\n")
        f.write("    float h2[128];\n")
        f.write("    for (int i = 0; i < 128; i++) {\n")
        f.write("        float sum = b1[i];\n")
        f.write("        for (int j = 0; j < 128; j++) {\n")
        f.write("            sum += W1[i * 128 + j] * h1[j];\n")
        f.write("        }\n")
        f.write("        h2[i] = relu(sum);\n")
        f.write("    }\n\n")
        f.write("    float out = b2[0];\n")
        f.write("    for (int i = 0; i < 128; i++) {\n")
        f.write("        out += W2[i] * h2[i];\n")
        f.write("    }\n\n")
        f.write("    // Saturacao estrita bidirecional [-80%, +80%]\n")
        f.write("    if (out > 80.0f) out = 80.0f;\n")
        f.write("    if (out < -80.0f) out = -80.0f;\n")
        f.write("    return out;\n")
        f.write(f"}}\n\n#endif // {guard_name}\n")

def export_model_physical_c(filename):
    content = f'''/*
 * model_physical_bidirecional.h
 * =============================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v9 Bidirecional
 * (Arrasto Aerodinamico -ca*w*|w| + Rigidez Torsional -k_th*th + Empuxo Bidirecional +-80%)
 */

#ifndef MODEL_PHYSICAL_BIDIRECIONAL_H
#define MODEL_PHYSICAL_BIDIRECIONAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define GB_C1          ({c1:.6f}f)
#define GB_K_TH        ({k_th:.6f}f)
#define GB_C2          ({c2:.6f}f)
#define GB_CA          ({ca:.6f}f)
#define GB_FC          ({Fc:.6f}f)
#define GB_C3          ({c3:.8f}f)
#define GB_P_THRUST    ({p_thrust:.6f}f)
#define GB_U0          ({u0:.4f}f)
#define GB_TAU_M       ({tau_m:.6f}f)

static float sil_th_rad = 0.0f;
static float sil_v_rad  = 0.0f;
static float sil_a_pct  = 0.0f;

static inline float u_equilibrium(float th_deg)
{{
    float th_rad = th_deg * ((float)M_PI / 180.0f);
    float restoring = -GB_C1 * sinf(th_rad) + GB_K_TH * th_rad;
    if (fabsf(restoring) < 1e-5f) return 0.0f;
    float dir = (restoring > 0.0f) ? 1.0f : -1.0f;
    float u_val = GB_U0 + powf(fabsf(restoring) / GB_C3, 1.0f / GB_P_THRUST);
    if (u_val > 80.0f) u_val = 80.0f;
    return dir * u_val;
}}

static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{{
    float u_motor = (u_cmd > 80.0f) ? 80.0f : ((u_cmd < -80.0f) ? -80.0f : u_cmd);
    float a_abs = fabsf(a);
    float z = a_abs - GB_U0;
    float thrust = 0.0f;
    if (z > 0.0f) {{
        float dir = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f);
        thrust = dir * powf(z, GB_P_THRUST);
    }}
    float aero = -GB_CA * v * fabsf(v);
    float fric = -GB_FC * tanhf(50.0f * v);

    *dth = v;
    *dv  = GB_C1 * sinf(th) - GB_K_TH * th + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / GB_TAU_M;
}}

static inline float simulate_physical_plant(float u_cmd)
{{
    const float dt = 0.01f;
    const float h  = dt / 2.0f;
    float th = sil_th_rad;
    float v  = sil_v_rad;
    float a  = sil_a_pct;

    for (int sub = 0; sub < 2; sub++) {{
        float k1_th, k1_v, k1_a;
        phys_derivs(th, v, a, u_cmd, &k1_th, &k1_v, &k1_a);

        float k2_th, k2_v, k2_a;
        phys_derivs(th + 0.5f * h * k1_th, v + 0.5f * h * k1_v, a + 0.5f * h * k1_a, u_cmd, &k2_th, &k2_v, &k2_a);

        float k3_th, k3_v, k3_a;
        phys_derivs(th + 0.5f * h * k2_th, v + 0.5f * h * k2_v, a + 0.5f * h * k2_a, u_cmd, &k3_th, &k3_v, &k3_a);

        float k4_th, k4_v, k4_a;
        phys_derivs(th + h * k3_th, v + h * k3_v, a + h * k3_a, u_cmd, &k4_th, &k4_v, &k4_a);

        th += (h / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);
        v  += (h / 6.0f) * (k1_v  + 2.0f * k2_v  + 2.0f * k3_v  + k4_v);
        a  += (h / 6.0f) * (k1_a  + 2.0f * k2_a  + 2.0f * k3_a  + k4_a);
    }}

    sil_th_rad = th;
    sil_v_rad  = v;
    sil_a_pct  = a;

    return th * (180.0f / (float)M_PI);
}}

#endif /* MODEL_PHYSICAL_BIDIRECIONAL_H */
'''
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

export_ann_to_c(model, scaler, os.path.join(SIL_DIR, "ann_weights_bidirecional.h"), guard_name="ANN_WEIGHTS_BIDIRECIONAL_H")
export_model_physical_c(os.path.join(SIL_DIR, "model_physical_bidirecional.h"))
print(f"Arquivos C gerados com sucesso em: {SIL_DIR}")

# Exporta tambem diretamente para o firmware STM32 fisico
stm32_inc = os.path.join(root_dir, "firmware", "stm32-fisico", "Core", "Inc")
if os.path.exists(stm32_inc):
    export_ann_to_c(model, scaler, os.path.join(stm32_inc, "ann_weights.h"), guard_name="ANN_WEIGHTS_H")
    print(f"[FIRMWARE] Pesos bidirecionais exportados com sucesso para: {stm32_inc}")

# %%
# =============================================================================
# 8. Validação SIL (Software-in-the-Loop) - Bateria Completa (-60° a +130°)
# =============================================================================
def run_sil_evaluation(x2ref_seq, val_name, filename_prefix):
    n_steps = len(x2ref_seq)
    tvec_seq = np.arange(n_steps) * Ts
    x2ref_full = np.concatenate([x2ref_seq, np.full(N, x2ref_seq[-1])])

    # Inicialização nos pontos de equilíbrio
    xs = np.zeros((nx, 1))
    xs[:, 0] = [np.radians(x2ref_seq[0]), 0.0, u_equilibrium(x2ref_seq[0])]
    y_sim = []
    u_sim = []
    t_inf_list = []

    # Ganhos PID nominal para trechos de transição
    Kp, Ki, Kd = 0.5793, 0.6647, 0.2
    u_i_pid = u_equilibrium(x2ref_seq[0])
    u_i_res = 0.0
    e_1 = 0.0
    y_1 = x2ref_seq[0]
    y_curr = x2ref_seq[0]
    was_ann = False

    T_PID_START = 4.0
    T_PID_END = 4.0

    model.eval()
    for k in tqdm(range(n_steps), desc=f"SIL {val_name}"):
        t_c = tvec_seq[k]
        r_c = x2ref_seq[k]
        t_inf = 0.0

        if t_c < T_PID_START or t_c >= (n_steps * Ts - T_PID_END):
            # Controle PID Nominal
            if was_ann:
                erro_tr = r_c - y_curr
                u_p_tr = Kp * erro_tr
                u_d_tr = -(Kd / Ts) * (y_curr - y_1)
                u_i_pid = u_sim[-1] - u_p_tr - u_d_tr
                was_ann = False

            erro = r_c - y_curr
            u_p = Kp * erro
            u_i_pid = u_i_pid + Ki * (Ts / 2.0) * (erro + e_1)
            u_d = -(Kd / Ts) * (y_curr - y_1)
            u_cmd = float(np.clip(u_p + u_i_pid + u_d, -80.0, 80.0))
            e_1 = erro
            y_1 = y_curr
        else:
            # Controle por Rede Neural (NMPC Bidirecional)
            if not was_ann:
                u_i_res = 0.0
                was_ann = True

            st_deg = [xs[0, -1] * 180.0 / np.pi, xs[1, -1] * 180.0 / np.pi, xs[2, -1]]
            feat = np.concatenate([st_deg, x2ref_full[k : k + N]])
            feat_sc = scaler.transform(feat.reshape(1, -1))

            t0 = time.perf_counter()
            with torch.no_grad():
                u_nn = float(model(torch.tensor(feat_sc, dtype=torch.float32)).item())
            t_inf = (time.perf_counter() - t0) * 1000.0

            # Integrador residual suave (Ki = 0.15) para rejeição de offset
            erro = r_c - y_curr
            u_i_res = float(np.clip(u_i_res + 0.15 * (Ts / 2.0) * (erro + e_1), -10.0, 10.0))
            u_cmd = float(np.clip(u_nn + u_i_res, -80.0, 80.0))

            # Mantém PID sincronizado em background
            u_p_tr = Kp * erro
            u_d_tr = -(Kd / Ts) * (y_curr - y_1)
            u_i_pid = u_cmd - u_p_tr - u_d_tr
            e_1 = erro
            y_1 = y_curr

        # Simulação física
        sim_step = F_plant(x0=xs[:, -1], p=u_cmd)
        xk1 = sim_step['xf'].full().flatten()
        y_curr = sim_step['yk'].full().item() * 180.0 / np.pi

        xs = np.c_[xs, xk1]
        y_sim.append(y_curr)
        u_sim.append(u_cmd)
        t_inf_list.append(t_inf)

    y_sim = np.array(y_sim)
    u_sim = np.array(u_sim)
    rmse = float(np.sqrt(np.mean((y_sim - x2ref_seq)**2)))
    print(f"Validação '{val_name}' Concluída! RMSE = {rmse:.2f}°")

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(tvec_seq, x2ref_seq, 'k--', label='Referência r(t)')
    plt.plot(tvec_seq, y_sim, color='#1f77b4', lw=1.5, label=f'Saída do Aeropêndulo y(t) (RMSE = {rmse:.2f}°)')
    plt.axhline(0.0, color='gray', ls=':', lw=0.8)
    plt.axvspan(0, T_PID_START, color='gray', alpha=0.15, label='Modo PID')
    plt.axvspan(n_steps * Ts - T_PID_END, n_steps * Ts, color='gray', alpha=0.15)
    plt.ylabel('Ângulo [°]')
    plt.xlabel('Tempo [s]')
    plt.title(f'SIL Validation: {val_name} (-60° to +130°)', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}_angulo.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(tvec_seq, u_sim, 'r-', lw=1.1, label='Controle u [%]')
    plt.axhline(0.0, color='k', ls='--', lw=0.8)
    plt.axhline(80.0, color='gray', ls=':')
    plt.axhline(-80.0, color='gray', ls=':')
    plt.ylabel('Controle [%]')
    plt.xlabel('Tempo [s]')
    plt.title(f'Control Effort: {val_name}', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}_controle.png"), dpi=130)
    plt.close()

    # Exporta CSV
    df = pd.DataFrame({
        'tempo_ms': (tvec_seq * 1000).astype(int),
        'angulo_deg': y_sim,
        'u_pct': u_sim,
        'referencia': x2ref_seq,
        'tempo_inf_ms': t_inf_list
    })
    df.to_csv(os.path.join(MPC_DIR, f"simulacao_{filename_prefix}.csv"), index=False)
    return rmse

# Teste 1: Degraus Aleatórios em Faixa Total (-60° a +130°)
np.random.seed(123)
n_val1 = int(round(75.0 / Ts))
t_val1 = np.arange(n_val1) * Ts
ref_val1 = np.zeros(n_val1)
idx = int(round(4.0 / Ts))
ref_val1[:idx] = 0.0
while idx < n_val1 - int(round(4.0 / Ts)):
    dur = int(round(np.random.uniform(4.0, 7.0) / Ts))
    if idx + dur > n_val1 - int(round(4.0 / Ts)):
        dur = (n_val1 - int(round(4.0 / Ts))) - idx
    ref_val1[idx : idx + dur] = np.random.uniform(-55.0, 125.0)
    idx += dur
ref_val1[idx:] = 0.0

run_sil_evaluation(ref_val1, "Degraus Aleatorios (-60 a +130)", "val_bidirecional_degraus")

# Teste 2: Senoide Suave Ampla Cruzando Zero
n_val2 = int(round(60.0 / Ts))
t_val2 = np.arange(n_val2) * Ts
ref_val2 = 35.0 + 85.0 * np.sin(2.0 * np.pi * 0.08 * t_val2)  # Varia de -50° a +120°
ref_val2[:int(round(4.0 / Ts))] = 0.0
ref_val2[-int(round(4.0 / Ts)):] = 0.0

run_sil_evaluation(ref_val2, "Senoide Ampla Cruzando Zero", "val_bidirecional_senoide")

print("\n" + "="*80)
print(" PIPELINE GREY-BOX v9 BIDIRECIONAL (-60° a +130°) FINALIZADO COM SUCESSO! ")
print(f" Arquivos e plots salvos em: {EXP_OUTPUT_DIR}")
print("="*80)

