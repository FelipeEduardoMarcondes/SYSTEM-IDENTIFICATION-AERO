# %% [markdown]
# # MPC - Modelo Grey-Box v9 Completo (Aeropêndulo / 1/4-Drone com Arrasto Quadrático e Rigidez do Cabo)
# 
# Este script implementa o pipeline completo de Controle Preditivo Baseado em Modelo Não-Linear (NMPC)
# e aproximação por Rede Neural (ANN) para o Modelo Grey-Box v9 de 3ª Ordem:
#   * Estados (nx = 3): [theta (rad), theta_dot (rad/s), a_motor (%)]
#   * Equações Diferenciais da Física (v9 Completo):
#       theta' = v
#       v'     = c1*sin(theta) - k_th*theta + c2*v - ca*v*|v| - Fc*tanh(50*v) + c3*(max(a - u0, 0))^p
#       a'     = (u - a) / tau_m
#   * Dinâmica do atuador: atraso de 1ª ordem tau_m ~ 113 ms
#   * Não-linearidade estática de empuxo real: expoente p ~ 1.759 (hélice carenada)
#   * Zona morta real do ESC/motor: u0 ~ 11.85%
#   * Freio aerodinâmico quadrático: ca ~ 0.0193 rad^-1 (-ca*v*|v|)
#   * Rigidez elástica de torção do chicote/cabo no pivô: k_th ~ 1.565 rad/s^2 (-k_th*theta)
#   * Restrições físicas: u in [0, 80]%
#
# Segue o fluxograma padronizado (IEEE / PUCPR / ABNT):
#   1. Configuração de diretórios e experimento (EXP_ID)
#   2. Carregamento do modelo v9 (modelo_greybox_v9.json) no CasADi
#   2b. Validação em malha aberta (free-run sobre multisseno experimental)
#   3. Setup do NMPC ótimo (Multiple Shooting + IPOPT)
#   4. Teste preliminar do MPC em degraus simples
#   5. Geração de sinal rico de treino com ancoragem em 0° (0°, 45° e 60°)
#   6. Simulação do MPC ótimo e coleta do dataset de controle (com ruído exploratório)
#   7. Criação, treinamento e avaliação da Rede Neural (PyTorch MLP + R2 plots)
#   8. Exportação dos pesos da rede para firmware C (ann_weights.h)
#   9. Simulação SIL Python completa (PID -> ANN -> PID) com transferência bumpless
#   10. Exportação dos CSVs para interface Sim-to-Real e comparação STM32
#   11. Baterias extras de validação SIL (Degraus aleatórios e Senoidal suave)

# %%
# =============================================================================
# 1. Imports, Diretórios e Definições de Amostragem
# =============================================================================
import os
import sys
import time
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from casadi import SX, MX, DM, Function, nlpsol, vertcat, fmax, tanh, sin, sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Garante suporte UTF-8 no Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from aerodata import readData

# ── Experimento ativo ────────────────────────────────────────────────────────
EXP_ID = os.environ.get("EXP_ID", "EXP005_greybox_v9_mpc")
EXP_OUTPUT_DIR = os.path.join(root_dir, "data", "experimentos_novos", EXP_ID)
IDENT_DIR   = os.path.join(EXP_OUTPUT_DIR, "1_identificacao")
MPC_DIR     = os.path.join(EXP_OUTPUT_DIR, "2_mpc_python")
SIL_DIR     = os.path.join(EXP_OUTPUT_DIR, "3_sil_stm32")
REAL_DIR    = os.path.join(EXP_OUTPUT_DIR, "4_aeropendulo")
PLOTS_DIR   = MPC_DIR
COLETAS_DIR = os.path.join(root_dir, "data", "coletas")
SINAIS_REF_DIR = os.path.join(root_dir, "data", "sinais_referencia")

for _d in [IDENT_DIR, MPC_DIR, SIL_DIR, REAL_DIR, SINAIS_REF_DIR]:
    os.makedirs(_d, exist_ok=True)

# Parâmetros de amostragem
Ts = 0.01  # Base de tempo: 10 ms (100 Hz, nativo da bancada)
nx = 3     # Estados: [theta (rad), omega (rad/s), a_motor (%)]

# %%
# =============================================================================
# 2. Carregamento dos Parâmetros e Modelo Grey-Box v9 no CasADi
# =============================================================================
json_path = os.path.join(current_dir, "modelo_greybox_v9.json")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        mod_data = json.load(f)
    
    # Seleção do modelo v9 ótimo
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

# Salva uma cópia dos parâmetros no diretório do experimento
with open(os.path.join(IDENT_DIR, "parametros_greybox_v9.json"), "w", encoding="utf-8") as f:
    json.dump({
        "c1": c1, "c2": c2, "c3": c3, "ca": ca, "Fc": Fc,
        "u0": u0, "tau_m": tau_m, "p": p_thrust, "k_th": k_th
    }, f, indent=2)

# Modelo preditivo interno do MPC (com suavização C^inf para derivadas exatas no IPOPT)
eps_dz = 0.5  # Regularização da zona morta [%]

def f_dyn_mpc(x_vec, u_val):
    th = x_vec[0]
    v  = x_vec[1]
    a  = x_vec[2]

    # Empuxo suave com expoente real p_thrust
    z = a - u0
    smooth_pos = 0.5 * (z + sqrt(z**2 + eps_dz**2))
    thrust = smooth_pos ** p_thrust

    # Arrasto aerodinâmico quadrático (-ca * v * |v|) e atrito seco
    aero_quad = -ca * v * sqrt(v**2 + 1e-6)
    fric = -Fc * tanh(50.0 * v)

    # Equação v9 com rigidez elástica do cabo elétrico (-k_th * th)
    acc = c1 * sin(th) - k_th * th + c2 * v + aero_quad + fric + c3 * thrust
    a_dot = (u_val - a) / fmax(tau_m, 0.01)
    return vertcat(v, acc, a_dot)

# Modelo físico exato da planta (para simulação fidedigna em malha fechada)
def f_dyn_plant(x_vec, u_val):
    th = x_vec[0]
    v  = x_vec[1]
    a  = x_vec[2]

    u_eff = fmax(a - u0, 0.0)
    thrust = (sqrt(u_eff**2 + 1e-4)) ** p_thrust
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
F = F_plant  # Alias padrão para simulação

# Função analítica exata de equilíbrio estático da planta física v9:
# c3 * (u_eq - u0)^p = -c1 * sin(theta) + k_th * theta
def u_equilibrium(th_deg):
    if th_deg <= 0.0:
        return 0.0  # Em repouso (<= 0°), motor totalmente desligado
    th_rad = np.radians(th_deg)
    restoring = -c1 * np.sin(th_rad) + k_th * th_rad
    if restoring > 0:
        val = restoring / c3
        return float(min(u0 + (val ** (1.0 / p_thrust)), 80.0))
    return 0.0

# %%
# =============================================================================
# 2b. Validação em Malha Aberta (Free-Run sobre Dados Experimentais)
# =============================================================================
TRIM_START_SEC, TRIM_END_SEC, DECIMATION = 10.0, 13.0, 1

def load_processed(name, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=DECIMATION):
    y, u, t, ref = readData(dataset_name=name, return_ref=True)
    t_start_target = t[0] + trim_start
    t_end_target = t[-1] - trim_end
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
    sl = slice(idx_start, idx_end, decimation)
    return u[sl], y[sl], t[sl], ref[sl] if len(ref) > 0 else np.full_like(t[sl], np.nan)

ms_candidates = [
    'data/experimentos/coleta_dados/RODADA-7/multi-seno-45-030Hz_0904_20-23.csv',
    'data/coletas/RODADA-7_20260904/multi-seno-45-040Hz_0904_20-26.csv',
    'RODADA-7/multi-seno-45-030Hz_0904_20-23.csv'
]
ms_file = None
for cand in ms_candidates:
    full_cand = os.path.join(root_dir, cand) if not os.path.isabs(cand) else cand
    if os.path.exists(full_cand):
        ms_file = cand
        break

if ms_file is not None:
    print(f"\nExecutando validação em malha aberta (free-run 100 Hz) sobre '{ms_file}'...")
    u_ms, y_ms, t_ms, ref_ms = load_processed(ms_file)
    x_state = np.array([np.radians(y_ms[0]), 0.0, u_ms[0]])
    y_casadi = np.zeros(len(y_ms))

    for k in range(len(y_ms)):
        res = F_plant(x0=x_state, p=u_ms[k])
        x_state = res['xf'].full().flatten()
        y_casadi[k] = float(res['yk']) * 180.0 / np.pi

    err_fr = y_ms - y_casadi
    rmse_fr = float(np.sqrt(np.mean(err_fr ** 2)))
    denom_fr = float(np.linalg.norm(y_ms - np.mean(y_ms)))
    fit_fr = float(100.0 * (1.0 - np.linalg.norm(err_fr) / denom_fr)) if denom_fr > 1e-6 else 0.0
    print(f"  Free-run CasADi v9: FIT = {fit_fr:.1f}% | RMSE = {rmse_fr:.3f}°")

    plt.figure(figsize=(12, 5))
    plt.plot(t_ms - t_ms[0], y_ms, 'k', label='Medição Experimental (Bancada Real)', lw=1.5)
    plt.plot(t_ms - t_ms[0], y_casadi, '--', color='#2ca02c', label=f'CasADi Grey-Box v9 Free-Run (FIT={fit_fr:.1f}%, RMSE={rmse_fr:.2f}°)', lw=1.6)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ângulo [°]')
    plt.title(f'CasADi Grey-Box v9 Model vs. Measured Multisine (FIT = {fit_fr:.1f}%, RMSE = {rmse_fr:.3f}°)', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "casadi_greybox_v9_freerun_multisine.png"), dpi=130)
    plt.show(block=False)
    plt.pause(0.1)

# %%
# =============================================================================
# 3. Setup do NMPC Ótimo (CasADi + IPOPT)
# =============================================================================
N = 100  # Horizonte preditivo (N * Ts = 100 * 0.01 s = 1.0 s à frente)

data = {
    'Ts': Ts,
    'x0': np.zeros(nx),
    'u_min': np.array([0.0]),      # Sem reversão espúria de empuxo
    'u_max': np.array([80.0]),     # Teto seguro da bancada (80%)
    'u_guess': np.array([15.0]),
    'x_guess': np.zeros(nx),
    'tol': 1e-5,
}

def vcat(lst):
    return vertcat(*[DM(x) if not hasattr(x, 'is_symbolic') else x for x in lst])

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

    # Custo: rastreamento angular (1000) + esforço de controle (0.1) + suavidade (50)
    J = J + 500 * (yk - Pref[k])**2 + 0.1 * uk**2 + 200.0 * du**2

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
    """Constrói chute inicial warm-start a partir do equilíbrio das referências."""
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
# 4. Geração de Referência Simples (Degraus) e Teste Preliminar do MPC
# =============================================================================
step_duration = 3.5
samples_per_step = int(round(step_duration / Ts))
step_levels = np.arange(30, 80, 10)  # 30, 40, 50, 60, 70 graus

t_total = len(step_levels) * step_duration
steps = len(step_levels) * samples_per_step

tvec = np.arange(0, t_total, Ts)
x2ref = np.zeros(steps)
for i, level in enumerate(step_levels):
    x2ref[i*samples_per_step : (i+1)*samples_per_step] = level

x2ref_full = np.concatenate([x2ref, np.full(N, 70.0)])

# Simulação MPC simples
xsim = np.zeros((nx, 1))
ysim = []
usim = []
asim = []
tvec_sim = []
dtvec = []
u_prev_sim = 0.0

w0_val = build_guess(xsim[:, 0], x2ref_full[:N])

print(f"\nSimulando MPC (Modelo Grey-Box v9) em degraus simples por {steps} passos ({t_total:.1f} s)...")
for k in tqdm(range(steps), desc="MPC Teste Degraus"):
    t = k * Ts
    ref_window = x2ref_full[k : k + N]

    pval = np.concatenate([xsim[:, -1], ref_window, [u_prev_sim]])

    if u_prev_sim < u0 and ref_window[0] > 2.0:
        w0_val = build_guess(xsim[:, -1], ref_window)

    tic = time.perf_counter()
    sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        w0_retry = build_guess(xsim[:, -1], ref_window)
        sol = solver(x0=w0_retry, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    dtvec.append(time.perf_counter() - tic)

    w_opt = sol['x'].full().flatten()
    u_opt = float(w_opt[nx])

    sim_step = F_plant(x0=xsim[:, -1], p=u_opt)
    xk1 = sim_step['xf'].full().flatten()
    yk = sim_step['yk'].full().item() * 180.0 / np.pi

    xsim = np.c_[xsim, xk1]
    usim.append(u_opt)
    ysim.append(yk)
    asim.append(float(xk1[2]))
    tvec_sim.append(t)

    # Shifting do horizonte
    w_shifted = np.zeros_like(w_opt)
    w_shifted[:nx] = xk1
    for i in range(N - 1):
        w_shifted[3 + i*4] = w_opt[3 + (i+1)*4]
        w_shifted[4 + i*4 : 4 + i*4 + 3] = w_opt[4 + (i+1)*4 : 4 + (i+1)*4 + 3]
    w_shifted[3 + (N-1)*4] = w_opt[3 + (N-1)*4]
    w_shifted[4 + (N-1)*4 : 4 + (N-1)*4 + 3] = w_opt[4 + (N-1)*4 : 4 + (N-1)*4 + 3]
    w0_val = w_shifted
    u_prev_sim = u_opt

dtvec = np.array(dtvec)
print(f"Tempo médio IPOPT: {dtvec.mean()*1000:.2f} ms por passo (Máx: {dtvec.max()*1000:.2f} ms)")

# %%
# =============================================================================
# 5. Plotagem dos Resultados do MPC Simples
# =============================================================================
plt.figure(figsize=(12, 5))
plt.plot(tvec_sim, ysim, color='#1f77b4', linewidth=2, label='Saída Grey-Box v9 (y)')
plt.step(tvec_sim, x2ref[:len(tvec_sim)], where='post', color='k', linestyle='--', label='Referência r(t)')
plt.ylabel('Ângulo [°]')
plt.xlabel('Tempo [s]')
plt.title('Grey-Box v9 MPC Control - Angle Tracking (Simple Steps)', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_simples_rastreamento.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(12, 4))
plt.step(tvec_sim, usim, where='post', color='r', label='Controle u [%]')
plt.plot(tvec_sim, asim, 'm--', label='Ativação do Motor a [%]')
plt.axhline(u0, color='orange', ls=':', label=f'Zona Morta u0={u0:.1f}%')
plt.ylabel('Entrada [%]')
plt.xlabel('Tempo [s]')
plt.title('Grey-Box v9 MPC Control - Control Effort', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_simples_controle.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(12, 3))
plt.plot(tvec_sim, dtvec * 1000.0, 'g-', lw=1.2)
plt.axhline(dtvec.mean() * 1000.0, color='k', ls=':', label=f'Média: {dtvec.mean()*1000:.2f} ms')
plt.grid(True)
plt.xlabel('Tempo [s]')
plt.ylabel('Tempo de Resolução [ms]')
plt.title('MPC Computational Time (IPOPT)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_simples_tempo_computacional.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 6. Geração do Sinal Rico de Treino da Rede Neural (Exatamente como mpc_v4.py)
# =============================================================================
np.random.seed(42)

HOLD_TIME = 4.0       # holds iniciais, intermediários e finais [s]
f_max = 0.50          # Frequência máxima de excitação: 0.50 Hz

# --- 1. Dinâmica em Nível DC = 45° (Excursão: 0° a 90°, i.e., 45° ± 45°) ---
# 1.1 Chirp linear de varredura completa (0.05 Hz a 0.50 Hz)
chirp_dur_45 = 16.0
n_ch_45 = int(round(chirp_dur_45 / Ts))
t_ch_45 = np.arange(n_ch_45) * Ts
f0_ch, f1_ch = 0.05, f_max
phase_ch_45 = 2.0 * np.pi * (f0_ch * t_ch_45 + ((f1_ch - f0_ch) / (2.0 * chirp_dur_45)) * t_ch_45**2)
chirp_45 = 45.0 + 45.0 * np.sin(phase_ch_45)  # 0° a 90°

# 1.2 Multisseno com superposição harmônica (todas frequências simultâneas até 0.50 Hz)
ms_duration_45 = 60.0
n_ms_45 = int(round(ms_duration_45 / Ts))
t_ms_45 = np.arange(n_ms_45) * Ts
df_45 = 1.0 / ms_duration_45
freqs_45 = np.arange(df_45, f_max + 1e-9, df_45)
phases_45 = np.random.uniform(0, 2 * np.pi, len(freqs_45))
ms_45 = np.sum([np.sin(2 * np.pi * f * t_ms_45 + ph) for f, ph in zip(freqs_45, phases_45)], axis=0)
ms_45 = ms_45 / np.max(np.abs(ms_45)) * 45.0 + 45.0  # 0° a 90°

# 1.3 Degraus aleatórios em torno de 45° (Excursão completa 0° a 90°)
n_steps_45 = 25
step_pieces_45 = []
for _ in range(n_steps_45):
    S = np.random.uniform(0.0, 90.0)
    dur = np.random.uniform(2.5, 4.0)
    step_pieces_45.append(np.full(int(round(dur / Ts)), S))
steps_45 = np.concatenate(step_pieces_45)

# --- 2. Dinâmica em Nível DC = 60° (Excursão: 0° a 120°, i.e., 60° ± 60°) ---
# 2.1 Chirp linear de varredura completa (0.05 Hz a 0.50 Hz)
chirp_dur_60 = 16.0
n_ch_60 = int(round(chirp_dur_60 / Ts))
t_ch_60 = np.arange(n_ch_60) * Ts
phase_ch_60 = 2.0 * np.pi * (f0_ch * t_ch_60 + ((f1_ch - f0_ch) / (2.0 * chirp_dur_60)) * t_ch_60**2)
chirp_60 = 60.0 + 60.0 * np.sin(phase_ch_60)  # 0° a 120°

# 2.2 Multisseno com superposição harmônica (fmax = 0.50 Hz)
ms_duration_60 = 60.0
n_ms_60 = int(round(ms_duration_60 / Ts))
t_ms_60 = np.arange(n_ms_60) * Ts
df_60 = 1.0 / ms_duration_60
freqs_60 = np.arange(df_60, f_max + 1e-9, df_60)
phases_60 = np.random.uniform(0, 2 * np.pi, len(freqs_60))
ms_60 = np.sum([np.sin(2 * np.pi * f * t_ms_60 + ph) for f, ph in zip(freqs_60, phases_60)], axis=0)
ms_60 = ms_60 / np.max(np.abs(ms_60)) * 60.0 + 60.0  # 0° a 120°

# 2.3 Degraus aleatórios em torno de 60° (Excursão máxima 0° a 120°)
n_steps_60 = 25
step_pieces_60 = []
for _ in range(n_steps_60):
    S = np.random.uniform(0.0, 120.0)
    dur = np.random.uniform(2.5, 4.0)
    step_pieces_60.append(np.full(int(round(dur / Ts)), S))
steps_60 = np.concatenate(step_pieces_60)

# --- Montagem da Sequência Completa de Treinamento ---
hold_45 = np.full(int(round(HOLD_TIME / Ts)), 45.0)
hold_60 = np.full(int(round(HOLD_TIME / Ts)), 60.0)

x2ref_train = np.clip(np.concatenate([
    hold_45, chirp_45, ms_45, steps_45, 
    hold_60, chirp_60, ms_60, steps_60, 
    hold_45
]), 0.0, 120.0)

steps_train = len(x2ref_train)
tvec_train = np.arange(steps_train) * Ts

plt.figure(figsize=(13, 3.5))
plt.plot(tvec_train, x2ref_train, 'k-', lw=1.2, label='Referência de Treino')
plt.axhline(0.0, color='gray', ls='--', lw=0.8)
plt.axhline(45.0, color='gray', ls=':', lw=1)
plt.axhline(60.0, color='gray', ls=':', lw=1)
plt.xlabel('Tempo [s]')
plt.ylabel('Ângulo [°]')
plt.title(f'Training Reference (Multisine fmax={f_max} Hz + Random Steps in 0°, 45° and 60° | Total: {steps_train*Ts:.1f} s)', fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "referencia_treino_rica.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 7. Simulação do MPC sobre o Sinal Rico e Coleta do Dataset (com Cache .npz)
# =============================================================================
RECOMPUTE_MPC = globals().get('RECOMPUTE_MPC', True)
dataset_cache_file = os.path.join(MPC_DIR, "dataset_mpc_treino.npz")

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

    for k in tqdm(range(sim_steps_train), desc="Coleta Dataset MPC"):
        ref_window = x2ref_train[k : k + N]
        pval = np.concatenate([xsim_train[:, -1], ref_window, [u_prev_sim]])

        if u_prev_sim < u0 and ref_window[0] > 2.0:
            w0_val = build_guess(xsim_train[:, -1], ref_window)

        sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            w0_retry = build_guess(xsim_train[:, -1], ref_window)
            sol = solver(x0=w0_retry, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
        w_opt = sol['x'].full().flatten()
        u_opt = float(w_opt[nx])

        # Ruído exploratório de 3.0% aplicado à planta (idêntico ao mpc_v4.py)
        u_applied = np.random.normal(u_opt, 3.0)
        u_applied = np.clip(u_applied, data['u_min'][0], data['u_max'][0])

        sim_step = F_plant(x0=xsim_train[:, -1], p=u_applied)
        xk1 = sim_step['xf'].full().flatten()
        yk = sim_step['yk'].full().item() * 180.0 / np.pi

        xsim_train = np.c_[xsim_train, xk1]
        usim_train.append(u_opt)  # Target para a ANN é o controle ótimo livre de ruído
        ysim_train.append(yk)

        # Shifting
        w_shifted = np.zeros_like(w_opt)
        w_shifted[:nx] = xk1
        for i in range(N - 1):
            w_shifted[3 + i*4] = w_opt[3 + (i+1)*4]
            w_shifted[4 + i*4 : 4 + i*4 + 3] = w_opt[4 + (i+1)*4 : 4 + (i+1)*4 + 3]
        w_shifted[3 + (N-1)*4] = w_opt[3 + (N-1)*4]
        w_shifted[4 + (N-1)*4 : 4 + (N-1)*4 + 3] = w_opt[4 + (N-1)*4 : 4 + (N-1)*4 + 3]
        w0_val = w_shifted
        u_prev_sim = u_applied

    # Construção dos tensores para a Rede Neural:
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

# Plot do rastreamento no treino
tvec_sim_train = tvec_train[:sim_steps_train]
plt.figure(figsize=(12, 5))
plt.plot(tvec_sim_train, ysim_train, color='#1f77b4', lw=1.5, label='Saída Grey-Box v9 (y)')
plt.step(tvec_sim_train, x2ref_train[:sim_steps_train], where='post', color='k', ls='--', label='Referência r(t)')
plt.ylabel('Ângulo [°]')
plt.xlabel('Tempo [s]')
plt.title('Grey-Box v9 MPC Control - Training Reference Tracking', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_treino_rastreamento.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(12, 4))
plt.step(tvec_sim_train, usim_train, where='post', color='r', label='Controle Ótimo u [%]')
plt.ylabel('Controle [%]')
plt.xlabel('Tempo [s]')
plt.title('Grey-Box v9 MPC Control - Training Control Input', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mpc_treino_controle.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 8. Criação e Treinamento da Rede Neural (PyTorch)
# =============================================================================
input_dim = nx + N

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

X_train_val, X_test, y_train_val, y_test = train_test_split(P_scaled, U_data, test_size=0.1, random_state=42)
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

import copy
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

epochs = 160
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
best_weights = copy.deepcopy(model.state_dict())

print(f"\nTreinando Rede Neural (MLP {input_dim} -> 128 -> 128 -> 1) por {epochs} épocas...")
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

# Restaura os melhores pesos observados na validação (idêntico a mpc_v4.py)
model.load_state_dict(best_weights)
print(f"  -> Treinamento concluído. Melhor Val Loss: {best_val_loss:.5f}")

# Salva modelo treinado
torch.save(model.state_dict(), os.path.join(MPC_DIR, "best_ann_greybox_v9.pth"))

# Gráfico de Perda
plt.figure(figsize=(8, 4))
plt.plot(train_loss_history, label='Perda Treino (MSE)', color='#1f77b4')
plt.plot(val_loss_history, label='Perda Validação (MSE)', color='#ff7f0e')
plt.title("Training and Validation Loss - MPC Grey-Box v9 Approximator", fontweight='bold')
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ann_loss_history.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# Avaliação R2 (Treino, Validação, Teste)
model.eval()
with torch.no_grad():
    y_tr_pred = model(tensor_X_train).numpy()
    y_va_pred = model(tensor_X_val).numpy()
    y_te_pred = model(tensor_X_test).numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sets = [(y_train, y_tr_pred, 'Treino', '#1f77b4', axes[0]),
        (y_val, y_va_pred, 'Validação', '#2ca02c', axes[1]),
        (y_test, y_te_pred, 'Teste', '#ff7f0e', axes[2])]

for y_true, y_pred, name, col, ax in sets:
    r2 = r2_score(y_true, y_pred)
    ax.scatter(y_true, y_pred, alpha=0.3, color=col)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_title(f"Conjunto de {name}\nR²: {r2:.4f}", fontweight='bold')
    ax.set_xlabel("Controle Ótimo u [%]")
    ax.set_ylabel("Controle Predito pela ANN u [%]")
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ann_r2_scores.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 9. Exportação dos Pesos para C (ann_weights.h para Firmware STM32 - stm32-fisico)
# =============================================================================
def export_ann_to_c(mod, sc, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n")
        f.write("/* Pesos da Rede Neural e Simulador Fisico treinados para o Modelo Grey-Box v9 */\n")
        f.write("#define MODEL_TYPE_GREYBOX_V9 1\n")
        f.write(f"#define MPC_HORIZON {N}\n")
        f.write(f"#define NN_INPUT_DIM {input_dim}\n")
        f.write("#define NY_MODEL 1\n")
        f.write("#define NU_MODEL 1\n")
        f.write(f"#define NX_SYSTEM {nx}\n\n")

        f.write("/* Buffers de historico unificados para telemetria e SIL */\n")
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

        # Pesos PyTorch
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
        f.write("    if (out > 80.0f) out = 80.0f;\n")
        f.write("    if (out < -10.0f) out = -10.0f;\n")
        f.write("    return out;\n")
        f.write("}\n\n#endif // ANN_WEIGHTS_H\n")


def export_model_physical_c(filename):
    content = f'''/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v9 Completo
 * (Arrasto Aerodinamico Quadratico -ca*w*|w| + Rigidez Torsional do Chicote -k_th*th + Dinamica de 1a Ordem tau_m)
 * Identificado em: ident_greybox_prof_multi_v9.py (Modelo Selecionado: v9_otimizado)
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem):
 *   d(theta)/dt = omega
 *   d(omega)/dt = c1*sin(theta) - k_th*theta + c2*omega - ca*omega*|omega| - Fc*tanh(50*omega) + c3*(max(a - u0, 0))^p
 *   d(a)/dt     = (u_motor - a) / tau_m
 *
 * Parametros identificados (v9_otimizado):
 *   c1    = {c1:.6f}f rad/s^2   (torque gravitacional normalizado)
 *   k_th  = {k_th:.6f}f s^-2       (rigidez torsional elastica do chicote no pivo)
 *   c2    = {c2:.6f}f s^-1      (amortecimento viscoso mecanico)
 *   ca    = {ca:.6f}f rad^-1     (coeficiente de arrasto quadratico aerodinamico)
 *   Fc    = {Fc:.6f}f rad/s^2    (atrito seco de Coulomb no mancal)
 *   c3    = {c3:.8f}f            (ganho de empuxo nao-linear)
 *   p     = {p_thrust:.6f}f            (expoente de empuxo aerodinamico da helice)
 *   u0    = {u0:.4f}f %         (zona morta da turbina/ESC)
 *   tau_m = {tau_m:.6f}f s          (constante de tempo de resposta do motor)
 *
 * Integrador numerico: Runge-Kutta 4a Ordem (RK4) com 2 subpassos de 5 ms a cada ciclo de 10 ms (100 Hz).
 */

#ifndef MODEL_PHYSICAL_H
#define MODEL_PHYSICAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Parametros fisicos do modelo Grey-Box v9 (v9_otimizado) ─────────────── */
#ifndef GB_C1
#define GB_C1          ({c1:.6f}f)   /* rad/s^2 - torque gravitacional normalizado */
#endif

#ifndef GB_K_TH
#define GB_K_TH        ({k_th:.6f}f)   /* s^-2    - rigidez torsional do chicote no pivo */
#endif

#ifndef GB_C2
#define GB_C2          ({c2:.6f}f)   /* s^-1    - amortecimento viscoso mecanico */
#endif

#ifndef GB_CA
#define GB_CA          ({ca:.6f}f)   /* rad^-1  - coeficiente de arrasto quadratico */
#endif

#ifndef GB_FC
#define GB_FC          ({Fc:.6f}f)   /* rad/s^2 - atrito seco de Coulomb no mancal */
#endif

#ifndef GB_C3
#define GB_C3          ({c3:.8f}f)  /* ganho de empuxo nao-linear */
#endif

#ifndef GB_P_THRUST
#define GB_P_THRUST    ({p_thrust:.6f}f)   /* expoente de empuxo nao-linear */
#endif

#ifndef GB_U0
#define GB_U0          ({u0:.4f}f)    /* %       - zona morta da turbina/ESC */
#endif

#ifndef GB_TAU_M
#define GB_TAU_M       ({tau_m:.6f}f)   /* s       - constante de tempo do atuador */
#endif

/* ── Estados internos da planta fisica no modo SIL ──────────────────────── */
static float sil_th_rad = 0.0f; /* Posicao angular (rad, 0 = repouso vertical para baixo) */
static float sil_v_rad  = 0.0f; /* Velocidade angular (rad/s) */
static float sil_a_pct  = 0.0f; /* Estado dinamico do atuador/turbina (%) */

/*
 * Calculo do controle PWM de equilibrio estatico u_eq(theta_deg)
 * Balanceamento de momento: c1*sin(th) - k_th*th + c3*(u - u0)^p = 0
 */
static inline float u_equilibrium(float th_deg)
{{
    if (th_deg <= 0.0f) return 0.0f;
    float th_rad = th_deg * ((float)M_PI / 180.0f);
    float restoring = -GB_C1 * sinf(th_rad) + GB_K_TH * th_rad;
    if (restoring <= 0.0f) return 0.0f;
    float u_val = GB_U0 + powf(restoring / GB_C3, 1.0f / GB_P_THRUST);
    if (u_val > 80.0f) u_val = 80.0f;
    return u_val;
}}

/*
 * Derivadas continuas do modelo fisico Grey-Box v9
 */
static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{{
    float u_motor = (u_cmd > 0.0f) ? ((u_cmd > 80.0f) ? 80.0f : u_cmd) : 0.0f;
    float z = a - GB_U0;
    float thrust = 0.0f;
    if (z > 0.0f) {{
        thrust = powf(z, GB_P_THRUST);
    }}
    float aero = -GB_CA * v * fabsf(v);
    float fric = -GB_FC * tanhf(50.0f * v);

    *dth = v;
    *dv  = GB_C1 * sinf(th) - GB_K_TH * th + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / GB_TAU_M;
}}

/*
 * Simulador de Planta Fisica Grey-Box v9 (RK4 a 100 Hz com 2 subpassos de 5 ms)
 */
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
        phys_derivs(th + 0.5f * h * k1_th,
                    v  + 0.5f * h * k1_v,
                    a  + 0.5f * h * k1_a,
                    u_cmd, &k2_th, &k2_v, &k2_a);

        float k3_th, k3_v, k3_a;
        phys_derivs(th + 0.5f * h * k2_th,
                    v  + 0.5f * h * k2_v,
                    a  + 0.5f * h * k2_a,
                    u_cmd, &k3_th, &k3_v, &k3_a);

        float k4_th, k4_v, k4_a;
        phys_derivs(th + h * k3_th,
                    v  + h * k3_v,
                    a  + h * k3_a,
                    u_cmd, &k4_th, &k4_v, &k4_a);

        th += (h / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);
        v  += (h / 6.0f) * (k1_v  + 2.0f * k2_v  + 2.0f * k3_v  + k4_v);
        a  += (h / 6.0f) * (k1_a  + 2.0f * k2_a  + 2.0f * k3_a  + k4_a);
    }}

    sil_th_rad = th;
    sil_v_rad  = v;
    sil_a_pct  = a;

    float y_deg = th * (180.0f / (float)M_PI);
    sil_y_hist[0] = y_deg;
    return y_deg;
}}

/* Alias para compatibilidade com rotinas legadas de simulacao */
static inline float simulate_narx(float u_cmd)
{{
    return simulate_physical_plant(u_cmd);
}}

#endif /* MODEL_PHYSICAL_H */
'''
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# 1. Exporta cópia local no diretório do experimento (SIL)
ann_c_file = os.path.join(SIL_DIR, "ann_weights.h")
export_ann_to_c(model, scaler, ann_c_file)
export_model_physical_c(os.path.join(SIL_DIR, "model_physical.h"))
print(f"Arquivos C gerados em (SIL backup): {SIL_DIR}")

# 2. Exporta SEMPRE para o firmware correto: stm32-fisico
stm32_fisico_inc = os.path.join(root_dir, "firmware", "stm32-fisico", "Core", "Inc")
if os.path.exists(stm32_fisico_inc):
    export_ann_to_c(model, scaler, os.path.join(stm32_fisico_inc, "ann_weights.h"))
    export_model_physical_c(os.path.join(stm32_fisico_inc, "model_physical.h"))
    print(f"[FIRMWARE] Exportado com SUCESSO para firmware físico: {stm32_fisico_inc}")

# %%
# 10. Geração do Sinal de Validação 1 (Custom: PID -> ANN -> PID)
# =============================================================================
np.random.seed(99)
T_PID_START = 5.0
T_MULTISINE = 30.0
T_CHIRP = 15.0
T_STEPS = 20.0
T_PID_END = 5.0
t_total_val = T_PID_START + T_MULTISINE + T_CHIRP + T_STEPS + T_PID_END
steps_val = int(round(t_total_val / Ts))
tvec_val = np.arange(steps_val) * Ts

x2ref_val = np.zeros(steps_val)
idx = 0

# 1. PID Inicial (45°)
n_pid_start = int(round(T_PID_START / Ts))
x2ref_val[idx : idx + n_pid_start] = 45.0
idx += n_pid_start

# 2. Multisseno (45 +/- 15°)
n_ms = int(round(T_MULTISINE / Ts))
f_max_val = 0.35
df_v = 1.0 / T_MULTISINE
freqs_v = np.arange(df_v, f_max_val + 1e-9, df_v)
phases_v = np.random.uniform(0, 2 * np.pi, len(freqs_v))
t_ms_v = np.arange(n_ms) * Ts
ms_v = np.sum([np.sin(2*np.pi*f*t_ms_v + ph) for f, ph in zip(freqs_v, phases_v)], axis=0)
ms_v = ms_v / np.max(np.abs(ms_v)) * 15.0 + 45.0
x2ref_val[idx : idx + n_ms] = ms_v
idx += n_ms

# 3. Chirp (0.05 a 0.4 Hz)
n_ch = int(round(T_CHIRP / Ts))
t_ch = np.arange(n_ch) * Ts
f0, f1 = 0.05, 0.4
ch_phase = 2.0 * np.pi * (f0 * t_ch + ((f1 - f0) / (2.0 * T_CHIRP)) * t_ch**2)
ch = 15.0 * np.sin(ch_phase) + 45.0
x2ref_val[idx : idx + n_ch] = ch
idx += n_ch

# 4. Degraus
n_st = int(round(T_STEPS / Ts))
step_lvls = [30.0, 60.0, 45.0, 65.0]
p_len = n_st // len(step_lvls)
for lvl in step_lvls:
    x2ref_val[idx : idx + p_len] = lvl
    idx += p_len
if idx < steps_val - int(round(T_PID_END / Ts)):
    x2ref_val[idx : steps_val - int(round(T_PID_END / Ts))] = 45.0

# 5. PID Final (45°)
x2ref_val[steps_val - int(round(T_PID_END / Ts)) :] = 45.0

x2ref_full_val = np.concatenate([x2ref_val, np.full(N, 45.0)])

plt.figure(figsize=(14, 4))
plt.plot(tvec_val, x2ref_val, 'k-', lw=1.5, label='Referência de Validação')
plt.axvline(T_PID_START, color='gray', ls=':')
plt.axvline(T_PID_START + T_MULTISINE, color='gray', ls=':')
plt.axvline(T_PID_START + T_MULTISINE + T_CHIRP, color='gray', ls=':')
plt.axvline(t_total_val - T_PID_END, color='gray', ls=':')
plt.xlabel('Tempo [s]')
plt.ylabel('Ângulo [°]')
plt.title('Custom Validation Reference: PID -> Multisine -> Chirp -> Steps -> PID', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "referencia_validacao_1.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 11. Simulação Python SIL da Validação 1 (PID -> ANN -> PID com Integrador Residual)
# =============================================================================
Kp, Ki, Kd = 0.5793, 0.6647, 0.2
u_i_pid = u_equilibrium(45.0)
u_i_res = 0.0
e_1 = 0.0
y_1 = 45.0
was_ann_mode = False
u_cmd = u_equilibrium(45.0)

ENABLE_RESIDUAL_INTEGRATOR = True
Ki_res = 0.20

xs_val = np.zeros((nx, 1))
xs_val[:, 0] = [np.radians(45.0), 0.0, u_equilibrium(45.0)]

y_sim_val = []
u_sim_val = []
t_inf_sim = []
y_atual = 45.0

model.eval()
print(f"\nExecutando Python SIL (PID -> RNA [Integrador Residual: {ENABLE_RESIDUAL_INTEGRATOR}, Bumpless Transfer] -> PID)...")

for k in tqdm(range(steps_val), desc="Python SIL Val 1"):
    t_curr = tvec_val[k]
    r_curr = x2ref_val[k]
    t_inf_step = 0.0

    if t_curr < T_PID_START or t_curr >= (t_total_val - T_PID_END):
        # --- Controle PID Nominal ---
        if was_ann_mode:
            # Bumpless transfer: RNA -> PID
            erro_tr = r_curr - y_atual
            u_p_tr = Kp * erro_tr
            u_d_tr = -(Kd / Ts) * (y_atual - y_1)
            u_i_pid = u_cmd - u_p_tr - u_d_tr
            was_ann_mode = False

        erro = r_curr - y_atual
        u_p = Kp * erro
        u_i_pid = u_i_pid + Ki * (Ts / 2.0) * (erro + e_1)
        u_d = -(Kd / Ts) * (y_atual - y_1)
        u_calc = u_p + u_i_pid + u_d
        u_cmd = float(np.clip(u_calc, -10.0, 80.0))
        if u_calc != u_cmd:
            u_i_pid -= (u_calc - u_cmd)
        e_1 = erro
        y_1 = y_atual
    else:
        # --- Controle por Rede Neural (Aproximador NMPC) ---
        if not was_ann_mode:
            # Bumpless transfer: PID -> RNA
            u_i_res = 0.0
            was_ann_mode = True

        st_deg = [xs_val[0, -1] * 180.0 / np.pi, xs_val[1, -1] * 180.0 / np.pi, xs_val[2, -1]]
        feat = np.concatenate([st_deg, x2ref_full_val[k : k + N]])
        feat_scaled = scaler.transform(feat.reshape(1, -1))

        t0_inf = time.perf_counter()
        with torch.no_grad():
            u_nn = float(model(torch.tensor(feat_scaled, dtype=torch.float32)).item())
        t_inf_step = (time.perf_counter() - t0_inf) * 1000.0

        erro = r_curr - y_atual
        if ENABLE_RESIDUAL_INTEGRATOR:
            u_i_res = u_i_res + Ki_res * (Ts / 2.0) * (erro + e_1)
            u_i_res = float(np.clip(u_i_res, -10.0, 10.0))
            u_calc = u_nn + u_i_res
            u_cmd = float(np.clip(u_calc, data['u_min'][0], data['u_max'][0]))
        else:
            u_cmd = float(np.clip(u_nn, data['u_min'][0], data['u_max'][0]))

        # Sincronização do integrador PID para retorno suave
        u_p_tr = Kp * erro
        u_d_tr = -(Kd / Ts) * (y_atual - y_1)
        u_i_pid = u_cmd - u_p_tr - u_d_tr
        e_1 = erro
        y_1 = y_atual

    # Avança dinâmica da planta física real
    sim_step = F_plant(x0=xs_val[:, -1], p=u_cmd)
    xk1 = sim_step['xf'].full().flatten()
    y_atual = sim_step['yk'].full().item() * 180.0 / np.pi

    xs_val = np.c_[xs_val, xk1]
    y_sim_val.append(y_atual)
    u_sim_val.append(u_cmd)
    t_inf_sim.append(t_inf_step)

y_sim_val = np.array(y_sim_val)
u_sim_val = np.array(u_sim_val)
rmse_sil1 = float(np.sqrt(np.mean((y_sim_val - x2ref_val)**2)))
print(f"Validação 1 SIL Concluída! RMSE = {rmse_sil1:.2f}°")

# Plot Validação 1
plt.figure(figsize=(14, 5))
plt.plot(tvec_val, x2ref_val, 'k--', label='Referência r(t)')
plt.plot(tvec_val, y_sim_val, color='#1f77b4', label=f'Saída do Aeropêndulo y(t) (RMSE = {rmse_sil1:.2f}°)', lw=1.5)
plt.axvspan(0, T_PID_START, color='gray', alpha=0.15, label='Controle PID')
plt.axvspan(t_total_val - T_PID_END, t_total_val, color='gray', alpha=0.15)
plt.ylabel('Ângulo [°]')
plt.xlabel('Tempo [s]')
plt.title('Python SIL Validation 1 (PID -> ANN MPC -> PID)', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sil_validacao_1_angulo.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(14, 4))
plt.plot(tvec_val, u_sim_val, 'r-', label='Comando de Controle u [%]', lw=1.2)
plt.axvspan(0, T_PID_START, color='gray', alpha=0.15)
plt.axvspan(t_total_val - T_PID_END, t_total_val, color='gray', alpha=0.15)
plt.ylabel('Controle [%]')
plt.xlabel('Tempo [s]')
plt.title('Python SIL Validation 1 - Control Effort', fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sil_validacao_1_controle.png"), dpi=130)
plt.show(block=False); plt.pause(0.1)

# %%
# =============================================================================
# 12. Exportação de Dados CSV para Interface e Comparação STM32
# =============================================================================
df_export1 = pd.DataFrame({
    'tempo_ms': (tvec_val * 1000).astype(int),
    'angulo_deg': y_sim_val,
    'u_pct': u_sim_val,
    'referencia': x2ref_val,
    'tempo_inf_ms': t_inf_sim
})
csv_py = os.path.join(MPC_DIR, "simulacao_python.csv")
df_export1.to_csv(csv_py, index=False)

# Exportações para interface Sim-to-Real
sim_csv = os.path.join(SINAIS_REF_DIR, "simulacao_mpc.csv")
df_export1.to_csv(sim_csv, index=False)
ref_csv = os.path.join(SINAIS_REF_DIR, "referencia_mpc.csv")
pd.DataFrame({'tempo_s': df_export1['tempo_ms'] / 1000.0, 'referencia_deg': x2ref_val}).to_csv(ref_csv, index=False)

# Exporta dados do treino
df_mpc_tr = pd.DataFrame({
    'tempo_ms': (tvec_train[:sim_steps_train] * 1000).astype(int),
    'angulo_deg': ysim_train,
    'u_pct': usim_train,
    'referencia': x2ref_train[:sim_steps_train]
})
df_mpc_tr.to_csv(os.path.join(MPC_DIR, "simulacao_mpc_treino.csv"), index=False)

# Waveform arbitrária para GUI
df_wave = pd.DataFrame({'tempo_s': tvec_train[:sim_steps_train], 'referencia_deg': x2ref_train[:sim_steps_train]})
df_wave.to_csv(os.path.join(SINAIS_REF_DIR, "wave_mpc_treino.csv"), index=False)
print("CSVs de simulação e interface exportados com sucesso!")

# %%
# =============================================================================
# 13. Testes Extras de Validação SIL (Degraus e Senoidal Suave)
# =============================================================================
def run_sil_validation(x2ref_custom, name, filename_prefix):
    steps_c = len(x2ref_custom)
    tvec_c = np.arange(steps_c) * Ts
    x2ref_full_c = np.concatenate([x2ref_custom, np.full(N, x2ref_custom[-1])])

    xs_c = np.zeros((nx, 1))
    xs_c[:, 0] = [np.radians(x2ref_custom[0]), 0.0, u_equilibrium(x2ref_custom[0])]
    y_sim_c = []
    u_sim_c = []

    u_i_c = u_equilibrium(x2ref_custom[0])
    e_1_c = 0.0
    y_1_c = x2ref_custom[0]
    y_at_c = x2ref_custom[0]

    for k in tqdm(range(steps_c), desc=f"SIL {name}"):
        t_c = k * Ts
        r_c = x2ref_custom[k]

        if t_c < T_PID_START or t_c >= (steps_c * Ts - T_PID_END):
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
            st_deg = [xs_c[0, -1] * 180.0 / np.pi, xs_c[1, -1] * 180.0 / np.pi, xs_c[2, -1]]
            feat = np.concatenate([st_deg, x2ref_full_c[k : k + N]])
            feat_sc = scaler.transform(feat.reshape(1, -1))
            with torch.no_grad():
                u_nn = float(model(torch.tensor(feat_sc, dtype=torch.float32)).item())
            u_cmd = float(np.clip(u_nn, data['u_min'][0], data['u_max'][0]))

            erro = r_c - y_at_c
            e_1_c = erro
            y_1_c = y_at_c
            u_i_c = u_cmd - (Kp * erro) - (-(Kd / Ts) * (y_at_c - y_1_c))

        sim_step = F_plant(x0=xs_c[:, -1], p=u_cmd)
        xk1 = sim_step['xf'].full().flatten()
        y_at_c = sim_step['yk'].full().item() * 180.0 / np.pi

        xs_c = np.c_[xs_c, xk1]
        y_sim_c.append(y_at_c)
        u_sim_c.append(u_cmd)

    y_sim_c = np.array(y_sim_c)
    rmse_c = float(np.sqrt(np.mean((y_sim_c - x2ref_custom)**2)))
    print(f"Validação '{name}' concluída! RMSE = {rmse_c:.2f}°")

    plt.figure(figsize=(14, 5))
    plt.plot(tvec_c, x2ref_custom, 'k--', label='Referência')
    plt.plot(tvec_c, y_sim_c, color='#1f77b4', label=f'Saída y(t) (RMSE = {rmse_c:.2f}°)', lw=1.5)
    plt.axvspan(0, T_PID_START, color='gray', alpha=0.15, label='PID')
    plt.axvspan(steps_c * Ts - T_PID_END, steps_c * Ts, color='gray', alpha=0.15)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ângulo [°]')
    plt.title(f'Python SIL Validation - {name}', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}_angulo.png"), dpi=130)
    plt.show(block=False); plt.pause(0.1)

    df_out = pd.DataFrame({
        'tempo_ms': (tvec_c * 1000).astype(int),
        'angulo_deg': y_sim_c,
        'u_pct': u_sim_c,
        'referencia': x2ref_custom
    })
    df_out.to_csv(os.path.join(MPC_DIR, f"simulacao_python_{filename_prefix}.csv"), index=False)
    pd.DataFrame({'tempo_s': tvec_c, 'referencia_deg': x2ref_custom}).to_csv(os.path.join(SINAIS_REF_DIR, f"referencia_{filename_prefix}.csv"), index=False)

# --- VAL 2: Degraus Aleatórios ---
np.random.seed(101)
x2ref_val2 = np.zeros(steps_val)
idx2 = 0
x2ref_val2[idx2 : idx2 + n_pid_start] = 45.0
idx2 += n_pid_start
while idx2 < steps_val - int(round(T_PID_END / Ts)):
    dur = int(round(np.random.uniform(5.0, 12.0) / Ts))
    if idx2 + dur > steps_val - int(round(T_PID_END / Ts)):
        dur = (steps_val - int(round(T_PID_END / Ts))) - idx2
    x2ref_val2[idx2 : idx2 + dur] = np.random.uniform(20.0, 80.0)
    idx2 += dur
x2ref_val2[steps_val - int(round(T_PID_END / Ts)) :] = 45.0

run_sil_validation(x2ref_val2, "Degraus Aleatórios", "val2_degraus")

# --- VAL 3: Senoidal Suave ---
x2ref_val3 = np.zeros(steps_val)
x2ref_val3[:n_pid_start] = 45.0
n_mid = steps_val - n_pid_start - int(round(T_PID_END / Ts))
t_mid = np.arange(n_mid) * Ts
f_max_s = 0.1
df_s = 1.0 / (n_mid * Ts)
freqs_s = np.arange(df_s, f_max_s + 1e-9, df_s)
phases_s = np.random.uniform(0, 2 * np.pi, len(freqs_s))
ms_s = np.sum([np.sin(2*np.pi*f*t_mid + ph) for f, ph in zip(freqs_s, phases_s)], axis=0)
ms_s = ms_s / np.max(np.abs(ms_s)) * 30.0 + 50.0
x2ref_val3[n_pid_start : steps_val - int(round(T_PID_END / Ts))] = ms_s
x2ref_val3[steps_val - int(round(T_PID_END / Ts)) :] = 45.0

run_sil_validation(x2ref_val3, "Senoidal Suave", "val3_senoidal")

print("\n" + "="*80)
print(f" PIPELINE COMPLETO DO MODELO GREY-BOX v9 CONCLUÍDO COM SUCESSO! ")
print(f" Resultados, gráficos e CSVs disponíveis em: {EXP_OUTPUT_DIR}")
print("="*80)
