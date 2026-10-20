# %% [markdown]
# # MPC - NARX
# 
# This script runs an optimal MPC (using casadi) for the 1/4 drone case study, using a NARX model identified.

# %% [markdown]
# ## Dataset Generation
# 
# Using a rich combination of sine, step, and chirp signals.

# %%

# %%
# 1. Imports and Definitions
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from tqdm.auto import tqdm
from casadi import SX, MX, DM, Function, nlpsol, vertcat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# sysid library (used to load the measured 1/4 drone datasets)
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ── Experimento ativo — EDITAR AQUI para cada novo experimento ──────
EXP_ID = os.environ.get("EXP_ID", "EXP002_legado_narx-mpc")
EXP_OUTPUT_DIR = os.path.join(root_dir, "data", "experimentos_novos", EXP_ID)
IDENT_DIR   = os.path.join(EXP_OUTPUT_DIR, "1_identificacao")
MPC_DIR     = os.path.join(EXP_OUTPUT_DIR, "2_mpc_python")
SIL_DIR     = os.path.join(EXP_OUTPUT_DIR, "3_sil_stm32")
REAL_DIR    = os.path.join(EXP_OUTPUT_DIR, "4_aeropendulo")
PLOTS_DIR   = MPC_DIR
COLETAS_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados")
SINAIS_REF_DIR = os.path.join(root_dir, "data", "sinais_referencia")
for _d in [IDENT_DIR, MPC_DIR, SIL_DIR, REAL_DIR]:
    os.makedirs(_d, exist_ok=True)

# Configuração de estilo acadêmico (IEEE / ABNT / PUCPR)
def setup_academic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.85",
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

setup_academic_style()

from aerodata import readData

Ts = 0.01  # Sampling time in seconds (DECIMATION=5 -> 5 * 10ms = 50ms)

# =============================================================================
# CONTROLE DE EXECUÇÃO / CACHE:
# - Se False: carrega o dataset de 32.547 amostras já calculado (.npz) em < 1s
# - Se True: reexecuta a otimização CasADi IPOPT por ~23 minutos e regrava o cache
RECOMPUTE_MPC = False
# =============================================================================

import json
try:
    with open(os.path.join(current_dir, 'narx_model.json'), 'r') as f:
        model_data = json.load(f)
        ny_model = model_data['ny']
        nu_model = model_data['nu']
        NARX_TERMS = model_data['terms']
        NARX_THETA = model_data['theta']
except FileNotFoundError:
    print("Run narx.py first to generate narx_model.json")
    import sys; sys.exit(1)

nx = ny_model + nu_model

x = SX.sym('x', nx)
y_syms = [x[i] for i in range(ny_model)]              # y(k-1) ... y(k-ny)
u_syms = [x[ny_model + i] for i in range(nu_model)]   # u(k-1) ... u(k-nu)
u_sym = SX.sym('u')

def _factor(tok):
    v, lag = re.match(r'([yu])\(k-(\d+)\)', tok).groups()
    return (y_syms if v == 'y' else u_syms)[int(lag) - 1]

def _term_expr(name):
    if name == 'constant':
        return SX(1.0)
    e = SX(1.0)
    for tok in re.findall(r'[yu]\(k-\d+\)', name):
        e = e * _factor(tok)
    return e

# y(k) = sum_i theta_i * term_i
y_k = SX(0)
for _name, _th in zip(NARX_TERMS, NARX_THETA):
    y_k = y_k + _th * _term_expr(_name)

y_next = vertcat(y_k, x[0:ny_model - 1])
u_next = vertcat(u_sym, x[ny_model:ny_model + nu_model - 1])
x_next = vertcat(y_next, u_next)

F = Function('F', [x, u_sym], [x_next, y_k], ['x0', 'p'], ['xf', 'yk'])

# %%
# 2b. Open-loop validation: free-run the CasADi state-space over the measured
#     multisine data (same selection used for identification) and compare.
TRIM_START_SEC, TRIM_END_SEC, DECIMATION = 10.0, 13.0, 1

def load_processed(name, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=DECIMATION):
    """Load a dataset from aerodata, trim by fixed time margins, and decimate. Returns u, y, t, ref."""
    y, u, t, ref = readData(dataset_name=name, return_ref=True)
    t_start_target = t[0] + trim_start
    t_end_target = t[-1] - trim_end
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    if idx_start >= idx_end: idx_start, idx_end = 0, len(t)
    sl = slice(idx_start, idx_end, decimation)
    t_c = t[sl] - t[sl][0]
    return u[sl], y[sl], t_c, ref[sl] if len(ref) > 0 else np.full_like(t_c, np.nan)

u_ms, y_ms, t_ms, ref_ms = load_processed('RODADA-7/multi-seno-45-030Hz_0904_20-23.csv')
ml = max(ny_model, nu_model)

# initial state from measured history: [y(k-1..k-ny), u(k-1..k-nu)] at k=ml
x_state = np.concatenate([y_ms[ml-1::-1][:ny_model], u_ms[ml-1::-1][:nu_model]])
y_casadi = np.zeros(len(y_ms))
y_casadi[:ml] = y_ms[:ml]

for k in range(ml, len(y_ms)):
    res = F(x0=x_state, p=u_ms[k])
    x_state = np.array(res['xf']).flatten()
    y_casadi[k] = float(res['yk'])

err = y_ms[ml:] - y_casadi[ml:]
rmse = np.sqrt(np.mean(err ** 2))
mae = np.mean(np.abs(err))
denom = np.linalg.norm(y_ms[ml:] - np.mean(y_ms[ml:]))
fit = 100.0 * (1.0 - np.linalg.norm(err) / denom) if denom > 1e-6 else 0.0

print(f"\n[CasADi Open-Loop Validation] RMSE: {rmse:.3f}° | MAE: {mae:.3f}° | FIT: {fit:.1f}%")

# Gráfico em padrão acadêmico (Subplots duplos sincronizados: Ângulo e Controle)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

ax1.plot(t_ms, y_ms, color='#111111', lw=1.1, label=r'Medido $y(k)$')
ax1.plot(t_ms, y_casadi, color='#d62728', ls='--', lw=1.3, label=r'CasADi NARX Free-Run $\hat{y}(k)$')
if not np.all(np.isnan(ref_ms)):
    ax1.plot(t_ms, ref_ms, color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Referência $r(k)$')

ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', framealpha=0.95)
ax1.set_xlim(0, t_ms[-1])

# Caixa com métricas
metric_text = (f"RMSE = {rmse:.3f}" + r"$^\circ$" + "\n"
               f"MAE = {mae:.3f}" + r"$^\circ$" + "\n"
               f"FIT = {fit:.1f}\\%")
ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.8', alpha=0.92))

ax2.plot(t_ms, u_ms, color='#1f77b4', lw=1.0, label=r'Excitação Multisseno $u(k)$')
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel(r'Controle $u$ [\%]')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()

# Salva em alta resolução (PNG 300 DPI e PDF vetorial)
target_dirs = [
    PLOTS_DIR,
    os.path.join(current_dir, 'plots'),
    os.path.join(root_dir, 'tcc', 'img', 'resultados'),
    os.path.join(root_dir, 'img', 'resultados')
]

for d in target_dirs:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "casadi_narx_freerun.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, "casadi_narx_freerun.pdf"), format='pdf', bbox_inches='tight')

print(f"  -> Gráfico CasADi salvo: casadi_narx_freerun (.png e .pdf) em {len(target_dirs)} pastas.")
plt.show(block=False)
plt.pause(0.1)

# %%
# 3. MPC Setup
N = 300
data = {
    'Ts': Ts,
    'x0': np.zeros(nx),
    'u_min': np.array([0.0]),
    'u_max': np.array([100.0]),
    'u_guess': np.array([0.0]),
    'x_guess': np.zeros(nx),
    'tol': 1e-8,
}

def vcat(lst):
    return vertcat(*[DM(x) if not hasattr(x, 'is_symbolic') else x for x in lst])

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []
J = 0

xk_param = MX.sym('xk_param', nx)
Pref = MX.sym('Pref', N)

xk = MX.sym('x0', nx)
w.append(xk)
lbw.append(np.full(nx, -np.inf))
ubw.append(np.full(nx, np.inf))
w0.append(data['x_guess'])

g.append(xk - xk_param)
lbg.append(np.zeros(nx))
ubg.append(np.zeros(nx))

for k in range(N):
    uk = MX.sym(f'u_{k}', 1)
    w.append(uk)
    lbw.append(data['u_min'])
    ubw.append(data['u_max'])
    w0.append(data['u_guess'])

    Fk = F(x0=xk, p=uk)
    xnext = Fk['xf']
    yk = Fk['yk']

    if k == 0:
        du = uk - xk[ny_model]
    else:
        du = uk - u_prev
    u_prev = uk

    J = J + 1e3 * (yk - Pref[k])**2 + 0.1 * uk**2 + 10000.0 * du**2

    xk = MX.sym(f'x_{k+1}', nx)
    w.append(xk)
    lbw.append(np.full(nx, -np.inf))
    ubw.append(np.full(nx, np.inf))
    w0.append(data['x_guess'])

    g.append(xk - xnext)
    lbg.append(np.zeros(nx))
    ubg.append(np.zeros(nx))

w = vertcat(*w)
lbw = vcat(lbw)
ubw = vcat(ubw)
w0 = vcat(w0)
g = vertcat(*g)
lbg = vcat(lbg)
ubg = vcat(ubg)

nlp = {'x': w, 'g': g, 'f': J, 'p': vertcat(xk_param, Pref)}
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

# %%
# 4. Reference Sequence Generation
# Sequence of 5s each, from 30 to 70 with step of 10
step_duration = 5.0
samples_per_step = int(round(step_duration / Ts))
step_levels = np.arange(30, 80, 10)  # 30, 40, 50, 60, 70

t_total = len(step_levels) * step_duration
steps = len(step_levels) * samples_per_step

tvec = np.arange(0, t_total, Ts)
x2ref = np.zeros(steps)

for i, level in enumerate(step_levels):
    x2ref[i*samples_per_step : (i+1)*samples_per_step] = level

# Padding for MPC horizon
x2ref_full = np.concatenate([x2ref, np.full(N, 70.0)])

# %%
# 5. Simulation Loop
xsim = np.zeros((nx, 1))
ysim = []
usim = []
tvec_sim = []
dtvec = []

w0_val = np.zeros(w.shape[0])

for k in tqdm(range(steps), desc="MPC Simulation"):
    t = k * Ts
    ref_window = x2ref_full[k : k + N]

    pval = np.concatenate([xsim[:, -1], ref_window])

    tic = time.perf_counter()
    sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    dtvec.append(time.perf_counter() - tic)

    w_opt = sol['x'].full().flatten()
    u_opt = w_opt[nx] # The first u after the initial x0

    # Open Loop Simulation using NARX model F
    sim_step = F(x0=xsim[:, -1], p=u_opt)
    xk1 = sim_step['xf'].full().flatten()
    yk = sim_step['yk'].full().item()

    xsim = np.c_[xsim, xk1]
    usim.append(u_opt)
    ysim.append(yk)
    tvec_sim.append(t)

    # Warm start
    w0_val = w_opt

# %%
# 6. Plotting Results
plt.figure(figsize=(12, 6))
plt.plot(tvec_sim, ysim, 'b-', linewidth=2, label='NARX Output (y)')
plt.step(tvec_sim, x2ref[:len(tvec_sim)], where='post', color='k', linestyle='--', label='Reference')
plt.ylabel('Angle (deg)')
plt.xlabel('Time (s)')
plt.title('NARX MPC Control - Drone Angle Tracking')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "".join([c if c.isalnum() else "_" for c in (plt.gca().get_title() or str(id(plt.gcf())))]) + ".png")); plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(12, 4))
plt.step(tvec_sim, usim, where='post', color='r', label='Control Input (u_pct)')
plt.ylabel('Input (u_pct)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "".join([c if c.isalnum() else "_" for c in (plt.gca().get_title() or str(id(plt.gcf())))]) + ".png")); plt.show(block=False); plt.pause(0.1)

plt.figure(figsize=(12, 3))
plt.plot(tvec_sim, dtvec)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Solve Time (s)')
plt.title('MPC Computational Time')
plt.savefig(os.path.join(PLOTS_DIR, "".join([c if c.isalnum() else "_" for c in (plt.gca().get_title() or str(id(plt.gcf())))]) + ".png")); plt.show(block=False); plt.pause(0.1)

# %%
# 7. Training Reference Signal Generation (DC 45° & 60°, Faixa 0° a 120°, f_max = 0.50 Hz)
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
# [Hold 45°] + [Chirp 45°] + [MS 45°] + [Degraus 45°] +
# [Hold 60°] + [Chirp 60°] + [MS 60°] + [Degraus 60°] +
# [Hold 45°]
hold_45 = np.full(int(round(HOLD_TIME / Ts)), 45.0)
hold_60 = np.full(int(round(HOLD_TIME / Ts)), 60.0)

x2ref_train = np.clip(np.concatenate([
    hold_45, chirp_45, ms_45, steps_45, 
    hold_60, chirp_60, ms_60, steps_60, 
    hold_45
]), 0.0, 120.0)

steps_train = len(x2ref_train)
tvec_train = np.arange(steps_train) * Ts

fig_ref, ax_ref = plt.subplots(figsize=(10.0, 3.4))
ax_ref.plot(tvec_train, x2ref_train, color='#1f77b4', lw=1.2, label=r'Referência de Treino $r(t)$')
ax_ref.axhline(45.0, color='gray', ls=':', lw=0.9, alpha=0.8, label=r'Nível DC $45^\circ$ ($0^\circ \!-\! 90^\circ$)')
ax_ref.axhline(60.0, color='black', ls='--', lw=0.9, alpha=0.6, label=r'Nível DC $60^\circ$ ($0^\circ \!-\! 120^\circ$)')
ax_ref.set_xlabel('Tempo [s]')
ax_ref.set_ylabel(r'Ângulo $\theta_{\text{ref}}$ [$^\circ$]')
ax_ref.set_title(f'Trajetória de Treinamento NMPC (DC $45^\\circ$ e $60^\\circ$, Faixa $0^\\circ$ a $120^\\circ$, $f_{{\\max}} = {f_max:.2f}$ Hz, Total: {steps_train*Ts:.0f} s)')
ax_ref.set_xlim(0, tvec_train[-1])
ax_ref.set_ylim(-2.0, 125.0)
ax_ref.grid(True, linestyle='--', alpha=0.5)
ax_ref.legend(loc='upper right', framealpha=0.95, fontsize=8.5)
plt.tight_layout()

# Salva a referência de treinamento
for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
         os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
         os.path.join(root_dir, 'img', 'resultados')]:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig_ref.savefig(os.path.join(d, "mpc_referencia_treino.png"), dpi=300, bbox_inches='tight')
        fig_ref.savefig(os.path.join(d, "mpc_referencia_treino.pdf"), format='pdf', bbox_inches='tight')

print(f"  -> Gráfico de referência de treino salvo: mpc_referencia_treino (.png e .pdf)")
plt.show(block=False)
plt.pause(0.1)

# %%
# 8. Simulation Loop (Rich Signal Dataset Collection)
# -----------------------------------------------------------------------------
# FLAG DE RECOMPUTAÇÃO DO MPC:
# - Se False: carrega o dataset de 32.547 amostras instantaneamente do cache (.npz)
# - Se True: executa a otimização CasADi IPOPT por ~23 minutos e regrava o cache
RECOMPUTE_MPC = globals().get('RECOMPUTE_MPC', False)
# -----------------------------------------------------------------------------

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
    ysim_train = []
    usim_train = []
    w0_val = np.zeros(w.shape[0])

    print("Simulating MPC over Rich Signal for Dataset...")
    sim_steps_train = steps_train - N
    for k in tqdm(range(sim_steps_train), desc="MPC Dataset Collection"):
        ref_window = x2ref_train[k : k + N]
        pval = np.concatenate([xsim_train[:, -1], ref_window])

        sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)

        w_opt = sol['x'].full().flatten()
        u_opt = w_opt[nx]

        # Add exploratory noise to the applied control input (NEW)
        u_applied = np.random.normal(u_opt, 3.0)  # NEW: standard deviation of 3.0% control noise
        u_applied = np.clip(u_applied, data['u_min'][0], data['u_max'][0])  # NEW: respect bounds

        sim_step = F(x0=xsim_train[:, -1], p=u_applied)  # NEW: apply noisy control to model
        xk1 = sim_step['xf'].full().flatten()
        yk = sim_step['yk'].full().item()

        xsim_train = np.c_[xsim_train, xk1]
        usim_train.append(u_opt)  # Target remains the optimal control (NEW)
        ysim_train.append(yk)
        w0_val = w_opt

    # Collect Dataset for ANN
    P_data = np.array([np.concatenate([xsim_train[:, k], x2ref_train[k : k + N]]) for k in range(sim_steps_train)])
    U_data = np.array(usim_train).reshape(-1, 1)

    np.savez_compressed(dataset_cache_file, P_data=P_data, U_data=U_data, xsim_train=xsim_train, ysim_train=np.array(ysim_train), usim_train=np.array(usim_train))
    print(f"[CACHE] Dataset NMPC salvo em cache: {dataset_cache_file}\n")


# --- Plotting MPC Results (Formato Acadêmico Padronizado: mpc_treino) ---
tvec_sim_train = tvec_train[:sim_steps_train]
fig_mpc, (ax_m1, ax_m2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                       gridspec_kw={'height_ratios': [2.2, 1.2]})

ax_m1.step(tvec_sim_train, x2ref_train[:sim_steps_train], where='post', color='#111111', linestyle='--', lw=1.1, label=r'Referência $r(t)$')
ax_m1.plot(tvec_sim_train, ysim_train, color='#1f77b4', lw=1.2, label=r'Saída MPC Nominal $y(t)$')
ax_m1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax_m1.set_title('Rastreamento Nominal da Referência de Treinamento pelo NMPC', fontsize=9.5, fontweight='bold', pad=4)
ax_m1.grid(True, linestyle='--', alpha=0.5)
ax_m1.legend(loc='upper right', framealpha=0.95)
ax_m1.set_xlim(0, tvec_sim_train[-1])

ax_m2.plot(tvec_sim_train, usim_train, color='#d62728', lw=1.0, label=r'Ação de Controle Ótima $u^*(t)$')
ax_m2.set_xlabel('Tempo [s]')
ax_m2.set_ylabel(r'Controle $u$ [\%]')
ax_m2.grid(True, linestyle='--', alpha=0.5)
ax_m2.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()

# Salva em todas as pastas do projeto
for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
         os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
         os.path.join(root_dir, 'img', 'resultados')]:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig_mpc.savefig(os.path.join(d, "mpc_treino.png"), dpi=300, bbox_inches='tight')
        fig_mpc.savefig(os.path.join(d, "mpc_treino.pdf"), format='pdf', bbox_inches='tight')

print(f"  -> Gráfico de rastreamento do MPC salvo: mpc_treino (.png e .pdf)")
plt.show(block=False)
plt.pause(0.1)

# %%
# 9. ANN Creation and Training
# Define the MLP (increased capacity for nx+N=35 inputs)
class MPCApproximator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MPCApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Normalize inputs ONLY using StandardScaler to match robust template
scaler = StandardScaler()
P_scaled = scaler.fit_transform(P_data)

# Split data into train, validation, and test sets to monitor overfitting and evaluate generalization
X_train_val, X_test, y_train_val, y_test = train_test_split(P_scaled, U_data, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)

tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
tensor_X_val = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val = torch.tensor(y_val, dtype=torch.float32)
tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# model = MPCApproximator(input_dim=nx + N, output_dim=1)
model = MPCApproximator(input_dim=nx + N, output_dim=1)
criterion = nn.MSELoss()
import copy
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

epochs = 160
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
best_weights = copy.deepcopy(model.state_dict())

print("Treinando Rede Neural (Aproximador NMPC)...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(tensor_X_val)
        val_loss = criterion(val_outputs, tensor_y_val).item()
    val_loss_history.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
        lr_curr = optimizer.param_groups[0]['lr']
        print(f"  Epoca {epoch+1:03d}/{epochs} | Treino: {train_loss:.5f} | Val: {val_loss:.5f} | lr: {lr_curr:.1e}")

# Restaura os melhores pesos observados na validacao
model.load_state_dict(best_weights)
print(f"  -> Treinamento concluido. Melhor Val Loss: {best_val_loss:.5f}")

# Grafico de convergencia das perdas
fig_loss, ax_loss = plt.subplots(figsize=(6.8, 3.4))
ax_loss.plot(train_loss_history, label='Perda Treino (MSE)', color='#1f77b4', lw=1.2)
ax_loss.plot(val_loss_history, label='Perda Validacao (MSE)', color='#ff7f0e', lw=1.2)
ax_loss.set_title("Evolucao da Perda de Treinamento da Rede Neural", fontsize=9.5, fontweight='bold', pad=4)
ax_loss.set_xlabel("Epoca")
ax_loss.set_ylabel(r"MSE de Controle $u^*$ [$\%^2$]")
ax_loss.set_yscale('log')
ax_loss.legend(loc='upper right', framealpha=0.95)
ax_loss.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
         os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
         os.path.join(root_dir, 'img', 'resultados')]:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig_loss.savefig(os.path.join(d, "ann_loss_evolution.png"), dpi=300, bbox_inches='tight')
        fig_loss.savefig(os.path.join(d, "ann_loss_evolution.pdf"), format='pdf', bbox_inches='tight')

plt.show(block=False)
plt.pause(0.1)

# Avaliacao em Treino, Validacao e Teste
model.eval()
with torch.no_grad():
    y_train_pred = model(tensor_X_train).numpy()
    y_val_pred = model(tensor_X_val).numpy()
    y_test_pred = model(tensor_X_test).numpy()

r2_tr = r2_score(y_train, y_train_pred)
r2_va = r2_score(y_val, y_val_pred)
r2_te = r2_score(y_test, y_test_pred)

fig_sc, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)

axes[0].scatter(y_train, y_train_pred, alpha=0.25, color='#1f77b4', s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=1.2)
axes[0].set_title(f"Treino ($R^2 = {r2_tr:.4f}$)", fontsize=9.0)
axes[0].set_xlabel(r"Controle NMPC $u^*$ [\%]")
axes[0].set_ylabel(r"Predicao RNA $\hat{u}$ [\%]")
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].scatter(y_val, y_val_pred, alpha=0.35, color='#2ca02c', s=12)
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=1.2)
axes[1].set_title(f"Validacao ($R^2 = {r2_va:.4f}$)", fontsize=9.0)
axes[1].set_xlabel(r"Controle NMPC $u^*$ [\%]")
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].scatter(y_test, y_test_pred, alpha=0.35, color='#ff7f0e', s=12)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1.2)
axes[2].set_title(f"Teste Cego ($R^2 = {r2_te:.4f}$)", fontsize=9.0)
axes[2].set_xlabel(r"Controle NMPC $u^*$ [\%]")
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
         os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
         os.path.join(root_dir, 'img', 'resultados')]:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig_sc.savefig(os.path.join(d, "ann_regression_fit.png"), dpi=300, bbox_inches='tight')
        fig_sc.savefig(os.path.join(d, "ann_regression_fit.pdf"), format='pdf', bbox_inches='tight')

print(f"  -> Graficos de ajuste da RNA salvos: ann_loss_evolution e ann_regression_fit (.png e .pdf)")
plt.show(block=False)
plt.pause(0.1)

# (Optional) We can also export ANN weights as before
def export_ann_to_c(model, scaler, filename="ann_weights.h", narx_terms=None, narx_theta=None, ny=15, nu=15):
    with open(filename, "w") as f:
        f.write("#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n")

        # 1. StandardScaler (mean e scale)
        f.write(f"const float scaler_mean[{len(scaler.mean_)}] = {{")
        f.write(", ".join([f"{val}f" for val in scaler.mean_]))
        f.write("};\n")

        f.write(f"const float scaler_scale[{len(scaler.scale_)}] = {{")
        f.write(", ".join([f"{val}f" for val in scaler.scale_]))
        f.write("};\n\n")

        # 2. PyTorch Weights
        state_dict = model.state_dict()
        layer_idx = 0
        for name, tensor in state_dict.items():
            np_arr = tensor.cpu().numpy()
            if "weight" in name:
                f.write(f"// Layer {layer_idx} Weights: shape {np_arr.shape}\n")
                # Flatten the 2D matrix (PyTorch linear weights are [out_features, in_features])
                flat = np_arr.flatten()
                f.write(f"const float W{layer_idx}[{len(flat)}] = {{")
                f.write(", ".join([f"{val}f" for val in flat]))
                f.write("};\n")
            elif "bias" in name:
                f.write(f"// Layer {layer_idx} Biases: shape {np_arr.shape}\n")
                f.write(f"const float b{layer_idx}[{len(np_arr)}] = {{")
                f.write(", ".join([f"{val}f" for val in np_arr]))
                f.write("};\n\n")
                layer_idx += 1

        # 3. Dynamic SIL C generation
        if narx_terms is not None and narx_theta is not None:
            f.write(f"#define NY_MODEL {ny}\n")
            f.write(f"#define NU_MODEL {nu}\n")
            f.write(f"static float sil_y_hist[NY_MODEL] = {{0}};\n")
            f.write(f"static float sil_u_hist[NU_MODEL] = {{0}};\n\n")
            
            f.write("static float simulate_narx(float u_atual) {\n")
            f.write("    for (int i = NU_MODEL - 1; i > 0; i--) sil_u_hist[i] = sil_u_hist[i-1];\n")
            f.write("    sil_u_hist[0] = u_atual;\n\n")
            f.write("    float y_k = ")
            for i, (term, th) in enumerate(zip(narx_terms, narx_theta)):
                import re
                if term == 'constant':
                    c_term = "1.0f"
                else:
                    factors = []
                    for v, lag in re.findall(r'([yu])\(k-(\d+)\)', term):
                        if v == 'y':
                            factors.append(f"sil_y_hist[{int(lag)-1}]")
                        else:
                            factors.append(f"sil_u_hist[{int(lag)-1}]")
                    c_term = "(" + " * ".join(factors) + ")"
                if i > 0:
                    f.write("\n              + ")
                f.write(f"{th}f * {c_term}")
            f.write(";\n\n")
            f.write("    for (int i = NY_MODEL - 1; i > 0; i--) sil_y_hist[i] = sil_y_hist[i-1];\n")
            f.write("    sil_y_hist[0] = y_k;\n")
            f.write("    return y_k;\n")
            f.write("}\n\n")

            input_dim = model.net[0].in_features
            f.write("static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }\n\n")
            f.write("static float ann_predict(float *input) {\n")
            f.write("    float h1[128] = {0};\n")
            f.write("    float h2[128] = {0};\n")
            f.write("    float out = 0.0f;\n\n")
            f.write(f"    float norm_input[{input_dim}];\n")
            f.write(f"    for (int i = 0; i < {input_dim}; i++) {{\n")
            f.write("        norm_input[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];\n")
            f.write("    }\n\n")
            f.write("    for (int i = 0; i < 128; i++) {\n")
            f.write("        float sum = b0[i];\n")
            f.write(f"        for (int j = 0; j < {input_dim}; j++) {{\n")
            f.write(f"            sum += W0[i * {input_dim} + j] * norm_input[j];\n")
            f.write("        }\n")
            f.write("        h1[i] = relu(sum);\n")
            f.write("    }\n\n")
            f.write("    for (int i = 0; i < 128; i++) {\n")
            f.write("        float sum = b1[i];\n")
            f.write("        for (int j = 0; j < 128; j++) {\n")
            f.write("            sum += W1[i * 128 + j] * h1[j];\n")
            f.write("        }\n")
            f.write("        h2[i] = relu(sum);\n")
            f.write("    }\n\n")
            f.write("    out = b2[0];\n")
            f.write("    for (int i = 0; i < 128; i++) {\n")
            f.write("        out += W2[i] * h2[i];\n")
            f.write("    }\n\n")
            f.write("    if (out > 80.0f) out = 80.0f;\n")
            f.write("    if (out < -10.0f) out = -10.0f;\n")
            f.write("    return out;\n")
            f.write("}\n")

        f.write("\n#endif\n")

ann_filename = os.path.join(SIL_DIR, 'ann_weights.h')
export_ann_to_c(model, scaler, filename=ann_filename, narx_terms=NARX_TERMS, narx_theta=NARX_THETA, ny=ny_model, nu=nu_model)
print(f"Arquivo {ann_filename} gerado com sucesso!")

stm_inc_dir = os.path.join(root_dir, "firmware", "stm32-sil", "Core", "Inc")
if os.path.exists(stm_inc_dir):
    stm_ann_filename = os.path.join(stm_inc_dir, 'ann_weights.h')
    export_ann_to_c(model, scaler, filename=stm_ann_filename, narx_terms=NARX_TERMS, narx_theta=NARX_THETA, ny=ny_model, nu=nu_model)
    print(f"Arquivo {stm_ann_filename} exportado direto para o firmware STM32 com sucesso!")

# %%
# 10. Validation Test Signal Generation (Custom: PID -> ANN -> PID)
np.random.seed(99)
T_PID_START = 5.0
T_MULTISINE = 30.0
T_CHIRP = 15.0
T_STEPS = 20.0
T_PID_END = 5.0
t_total_val = T_PID_START + T_MULTISINE + T_CHIRP + T_STEPS + T_PID_END
steps_val = int(t_total_val / Ts)
tvec_val = np.arange(steps_val) * Ts

x2ref_val = np.zeros(steps_val)
idx = 0
# PID START
n_pid_start = int(T_PID_START/Ts)
x2ref_val[idx : idx + n_pid_start] = 45.0
idx += n_pid_start

# MULTISINE
n_ms = int(T_MULTISINE/Ts)
f_max_val = 0.4
df = 1.0/T_MULTISINE
freqs = np.arange(df, f_max_val+1e-9, df)
phases = np.random.uniform(0, 2*np.pi, len(freqs))
t_ms = np.arange(n_ms)*Ts
ms = np.sum([np.sin(2*np.pi*f*t_ms + ph) for f,ph in zip(freqs, phases)], axis=0)
ms = ms - ms[0]  # Conexão contínua com 45 deg no instante da transição
ms = ms / np.max(np.abs(ms)) * 25.0 + 45.0
x2ref_val[idx : idx + n_ms] = ms
idx += n_ms

# CHIRP
n_ch = int(T_CHIRP/Ts)
t_ch = np.arange(n_ch)*Ts
f0, f1 = 0.05, 0.6
ch = np.sin(2*np.pi * (f0*t_ch + (f1-f0)/(2*T_CHIRP)*t_ch**2))
ch = ch * 20.0 + 45.0
x2ref_val[idx : idx + n_ch] = ch
idx += n_ch

# STEPS
n_st = int(T_STEPS/Ts)
n_pieces = 5
piece_len = n_st // n_pieces
st = np.zeros(n_st)
for i in range(n_pieces):
    S = np.random.uniform(20.0, 70.0)
    st[i*piece_len : (i+1)*piece_len] = S
if n_st > piece_len*n_pieces:
    st[piece_len*n_pieces:] = S
x2ref_val[idx : idx + n_st] = st
idx += n_st

# PID END
n_pid_end = steps_val - idx
x2ref_val[idx : steps_val] = 45.0

x2ref_full_val = np.concatenate([x2ref_val, np.full(N, 45.0)])

# %%
# 11. Simulation Loop (Python SIL generation com opção de Ação Integral Residual)
xs = np.zeros((nx, 1))
y_atual = 0.0
y_sim = []
u_sim = []

# Ganhos PID nominal para as fases de decolagem e pouso
Kp, Ki, Kd = 0.5793, 0.6647, 0.2
u_i_pid = 0.0
u_i_res = 0.0
e_1 = 0.0
y_1 = 0.0
was_ann_mode = False
u_opt = 0.0

# Configuracao do Integrador Residual para a RNA:
# Quando ativado, trimma offsets estáticos e distúrbios contínuos sem desestabilizar
ENABLE_RESIDUAL_INTEGRATOR = True
Ki_res = 0.20

print(f"Executando Python SIL (PID -> RNA [Integrador Residual: {ENABLE_RESIDUAL_INTEGRATOR}, Bumpless Transfer] -> PID)...")
from tqdm import tqdm
import torch
import pandas as pd

for k in tqdm(range(steps_val), desc="Python SIL Simulation"):
    t = k * Ts
    r_curr = x2ref_val[k]
    
    if t < T_PID_START or t >= t_total_val - T_PID_END:
        # --- Controle PID Nominal ---
        if was_ann_mode:
            # Bumpless transfer: RNA -> PID
            # Pré-carrega o integrador do PID com o esforço exato do último instante da rede
            erro_tr = r_curr - y_atual
            u_p_tr = Kp * erro_tr
            u_d_tr = -(Kd / Ts) * (y_atual - y_1)
            u_i_pid = u_opt - u_p_tr - u_d_tr
            was_ann_mode = False

        erro = r_curr - y_atual
        u_p = Kp * erro
        u_i_pid = u_i_pid + Ki * (Ts / 2.0) * (erro + e_1)
        u_d = -(Kd / Ts) * (y_atual - y_1)
        u_calc = u_p + u_i_pid + u_d
        u_opt = float(np.clip(u_calc, -10.0, 80.0))
        if u_calc != u_opt:
            u_i_pid -= (u_calc - u_opt)
        e_1 = erro; y_1 = y_atual
    else:
        # --- Controle por Rede Neural (Aproximador NMPC) ---
        if not was_ann_mode:
            # Bumpless transfer: PID -> RNA
            # Reseta o integrador residual; a rede assume com seu valor nominal ótimo sem solavancos
            u_i_res = 0.0
            was_ann_mode = True

        ref_window = x2ref_full_val[k : k + N]
        pval = np.concatenate([xs[:, -1], ref_window])
        pval_scaled = scaler.transform(pval.reshape(1, -1))
        with torch.no_grad():
            u_ann = float(model(torch.tensor(pval_scaled, dtype=torch.float32)).item())
        
        erro = r_curr - y_atual
        if ENABLE_RESIDUAL_INTEGRATOR:
            u_i_res = u_i_res + Ki_res * (Ts / 2.0) * (erro + e_1)
            u_i_res = float(np.clip(u_i_res, -10.0, 10.0))  # Anti-windup suave
            u_calc = u_ann + u_i_res
            u_opt = float(np.clip(u_calc, data["u_min"][0], data["u_max"][0]))
        else:
            u_opt = float(np.clip(u_ann, data["u_min"][0], data["u_max"][0]))
            
        # Sincronização do integrador PID no fundo para retorno suave
        u_p_tr = Kp * erro
        u_d_tr = -(Kd / Ts) * (y_atual - y_1)
        u_i_pid = u_opt - u_p_tr - u_d_tr
        e_1 = erro; y_1 = y_atual

    res = F(x0=xs[:, -1], p=u_opt)
    xf_new = res["xf"].full().flatten()
    
    if np.any(np.isnan(xf_new)) or np.any(np.isinf(xf_new)) or np.max(np.abs(xf_new)) > 1e6:
        print(f"\nWarning: Simulation diverged at step {k} (t={t:.2f}s). Filling remaining steps with NaN.")
        remaining = steps_val - k
        y_sim.extend([np.nan] * remaining)
        u_sim.extend([np.nan] * remaining)
        break

    xs = np.c_[xs, xf_new]
    y_atual = float(res["yk"])
    y_sim.append(y_atual)
    u_sim.append(u_opt)

# --- Plotagem Padronizada Acadêmica (IEEE / ABNT) ---
fig_sil, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                       gridspec_kw={'height_ratios': [2.2, 1.2]})

# Subplot 1: Rastreamento Angular
ax_s1.plot(tvec_val, x2ref_val, color='#111111', linestyle='--', lw=1.1, label=r'Referência $r(t)$')
ax_s1.plot(tvec_val, y_sim, color='#1f77b4', lw=1.2, label=r'Saída SIL $y(t)$')
ax_s1.axvspan(0, T_PID_START, color='#7f7f7f', alpha=0.15, label='Transição PID')
ax_s1.axvspan(t_total_val - T_PID_END, t_total_val, color='#7f7f7f', alpha=0.15)
ax_s1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
title_str = 'Validação SIL Python (PID $\\rightarrow$ RNA MPC' + (' + Int. Residual' if ENABLE_RESIDUAL_INTEGRATOR else '') + ' $\\rightarrow$ PID)'
ax_s1.set_title(title_str, fontsize=9.5, fontweight='bold', pad=4)
ax_s1.grid(True, linestyle='--', alpha=0.5)
ax_s1.legend(loc='upper right', framealpha=0.95)
ax_s1.set_xlim(0, tvec_val[-1])

# Subplot 2: Esforço de Controle
ax_s2.plot(tvec_val, u_sim, color='#d62728', lw=1.0, label=r'Ação de Controle $u(t)$')
ax_s2.axvspan(0, T_PID_START, color='#7f7f7f', alpha=0.15)
ax_s2.axvspan(t_total_val - T_PID_END, t_total_val, color='#7f7f7f', alpha=0.15)
ax_s2.set_xlabel('Tempo [s]')
ax_s2.set_ylabel(r'Controle $u$ [\%]')
ax_s2.grid(True, linestyle='--', alpha=0.5)
ax_s2.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()

for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
         os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
         os.path.join(root_dir, 'img', 'resultados')]:
    if d and (os.path.exists(d) or d == PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
        fig_sil.savefig(os.path.join(d, "sil_validacao_completa.png"), dpi=300, bbox_inches='tight')
        fig_sil.savefig(os.path.join(d, "sil_validacao_completa.pdf"), format='pdf', bbox_inches='tight')

print("  -> Gráfico de validação SIL salvo: sil_validacao_completa (.png e .pdf)")
plt.show(block=False)
plt.pause(0.1)

# %%
# 12. Export to CSV for STM comparison
# Salvando na nova estrutura de experimentos
df_export = pd.DataFrame({"tempo_ms": (tvec_val * 1000).astype(int), "angulo_deg": y_sim, "u_pct": u_sim, "referencia": x2ref_val})
csv_filename = os.path.join(MPC_DIR, "simulacao_python.csv")
df_export.to_csv(csv_filename, index=False)
print(f"\nSimulation data exported to {csv_filename}!")

# Exportar para sinais_referencia (usado pela interface sim-to-real)
os.makedirs(SINAIS_REF_DIR, exist_ok=True)
sim_filename = os.path.join(SINAIS_REF_DIR, "simulacao_mpc.csv")
df_export.to_csv(sim_filename, index=False)
ref_filename = os.path.join(SINAIS_REF_DIR, "referencia_mpc.csv")
df_ref = pd.DataFrame({'tempo_s': df_export['tempo_ms'] / 1000.0, 'referencia_deg': x2ref_val})
df_ref.to_csv(ref_filename, index=False)
print(f"Auto Sim-to-Real files exported to {SINAIS_REF_DIR}")

df_mpc_train = pd.DataFrame({"tempo_ms": (tvec_train[:sim_steps_train] * 1000).astype(int), "angulo_deg": ysim_train, "u_pct": usim_train, "referencia": x2ref_train[:sim_steps_train]})
csv_mpc_train = os.path.join(MPC_DIR, "simulacao_mpc_treino.csv")
df_mpc_train.to_csv(csv_mpc_train, index=False)

df_mpc_sim = pd.DataFrame({"tempo_ms": (np.array(tvec_sim) * 1000).astype(int), "angulo_deg": ysim, "u_pct": usim, "referencia": x2ref[:len(tvec_sim)]})
csv_mpc_sim = os.path.join(MPC_DIR, "simulacao_mpc_simples.csv")
df_mpc_sim.to_csv(csv_mpc_sim, index=False)

# Export wave for GUI (Sinal Arbitrario) → sinais_referencia
df_wave = pd.DataFrame({"tempo_s": tvec_train[:sim_steps_train], "referencia_deg": x2ref_train[:sim_steps_train]})
csv_wave = os.path.join(SINAIS_REF_DIR, "wave_mpc_treino.csv")
df_wave.to_csv(csv_wave, index=False)
print(f"Wave para GUI (Sinal Arbitrario) exportada para {csv_wave}!")

print("Exportacoes de CSV concluidas!")

# =====================================================================
# 13. Testes Extras de Validação (Degraus e Senoidal Suave)
# =====================================================================
def run_sil_validation(x2ref_custom, name, filename_prefix):
    xs_val = np.zeros((nx, 1))
    y_atual_val = 0.0
    y_sim_val = []
    u_sim_val = []
    u_i_pid_val = 0.0
    u_i_res_val = 0.0
    e_1_val = 0.0
    y_1_val = 0.0
    was_ann_val = False
    u_opt = 0.0

    steps_custom = len(x2ref_custom)
    tvec_custom = np.arange(steps_custom) * Ts
    x2ref_full_custom = np.concatenate([x2ref_custom, np.full(N, 45.0)])

    print(f"\nRunning Python SIL Validation ({name})...")
    for k in tqdm(range(steps_custom), desc=f"SIL {name}"):
        t = k * Ts
        r_curr = x2ref_custom[k]
        
        if t < T_PID_START or t >= (steps_custom * Ts) - T_PID_END:
            # PID Nominal
            if was_ann_val:
                # Bumpless transfer: RNA -> PID
                erro_tr = r_curr - y_atual_val
                u_p_tr = Kp * erro_tr
                u_d_tr = -(Kd / Ts) * (y_atual_val - y_1_val)
                u_i_pid_val = u_opt - u_p_tr - u_d_tr
                was_ann_val = False

            erro = r_curr - y_atual_val
            u_p = Kp * erro
            u_i_pid_val += Ki * (Ts / 2.0) * (erro + e_1_val)
            u_d = -(Kd / Ts) * (y_atual_val - y_1_val)
            u_calc = u_p + u_i_pid_val + u_d
            u_opt = float(np.clip(u_calc, -10.0, 80.0))
            if u_calc != u_opt:
                u_i_pid_val -= (u_calc - u_opt)
            e_1_val = erro; y_1_val = y_atual_val
        else:
            # ANN
            if not was_ann_val:
                # Bumpless transfer: PID -> RNA
                u_i_res_val = 0.0
                was_ann_val = True

            ref_window = x2ref_full_custom[k : k + N]
            pval = np.concatenate([xs_val[:, -1], ref_window])
            pval_scaled = scaler.transform(pval.reshape(1, -1))
            with torch.no_grad():
                u_ann = float(model(torch.tensor(pval_scaled, dtype=torch.float32)).item())
            erro = r_curr - y_atual_val
            if ENABLE_RESIDUAL_INTEGRATOR:
                u_i_res_val += Ki_res * (Ts / 2.0) * (erro + e_1_val)
                u_i_res_val = float(np.clip(u_i_res_val, -10.0, 10.0))
                u_calc = u_ann + u_i_res_val
                u_opt = float(np.clip(u_calc, data["u_min"][0], data["u_max"][0]))
            else:
                u_opt = float(np.clip(u_ann, data["u_min"][0], data["u_max"][0]))

            # Tracking mode do integrador PID para retorno suave
            u_p_tr = Kp * erro
            u_d_tr = -(Kd / Ts) * (y_atual_val - y_1_val)
            u_i_pid_val = u_opt - u_p_tr - u_d_tr
            e_1_val = erro; y_1_val = y_atual_val
            
        res = F(x0=xs_val[:, -1], p=u_opt)
        xf_new = res["xf"].full().flatten()
        
        if np.any(np.isnan(xf_new)) or np.any(np.isinf(xf_new)) or np.max(np.abs(xf_new)) > 1e6:
            print(f"Warning: Diverged at step {k}")
            break
            
        xs_val = np.c_[xs_val, xf_new]
        y_atual_val = float(res["yk"])
        y_sim_val.append(y_atual_val)
        u_sim_val.append(u_opt)
        
    tvec_plot = tvec_custom[:len(y_sim_val)]
    fig_v, (ax_v1, ax_v2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                         gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    ax_v1.plot(tvec_plot, x2ref_custom[:len(y_sim_val)], color='#111111', linestyle='--', lw=1.1, label=r'Referência $r(t)$')
    ax_v1.plot(tvec_plot, y_sim_val, color='#1f77b4', lw=1.2, label=r'Saída $y(t)$')
    ax_v1.axvspan(0, T_PID_START, color='#7f7f7f', alpha=0.15, label='PID')
    ax_v1.axvspan(steps_custom*Ts - T_PID_END, steps_custom*Ts, color='#7f7f7f', alpha=0.15)
    ax_v1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax_v1.set_title(f'Validação SIL - {name}', fontsize=9.5, fontweight='bold', pad=4)
    ax_v1.grid(True, linestyle='--', alpha=0.5)
    ax_v1.legend(loc='upper right', framealpha=0.95)
    ax_v1.set_xlim(0, tvec_plot[-1])

    ax_v2.plot(tvec_plot, u_sim_val, color='#d62728', lw=1.0, label=r'Controle $u(t)$')
    ax_v2.axvspan(0, T_PID_START, color='#7f7f7f', alpha=0.15)
    ax_v2.axvspan(steps_custom*Ts - T_PID_END, steps_custom*Ts, color='#7f7f7f', alpha=0.15)
    ax_v2.set_xlabel('Tempo [s]')
    ax_v2.set_ylabel(r'Controle $u$ [\%]')
    ax_v2.grid(True, linestyle='--', alpha=0.5)
    ax_v2.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    for d in [PLOTS_DIR, os.path.join(current_dir, 'plots'), 
             os.path.join(root_dir, 'tcc', 'img', 'resultados'), 
             os.path.join(root_dir, 'img', 'resultados')]:
        if d and (os.path.exists(d) or d == PLOTS_DIR):
            os.makedirs(d, exist_ok=True)
            fig_v.savefig(os.path.join(d, f"{filename_prefix}_plot.png"), dpi=300, bbox_inches='tight')
            fig_v.savefig(os.path.join(d, f"{filename_prefix}_plot.pdf"), format='pdf', bbox_inches='tight')
    plt.show(block=False); plt.pause(0.1)

    df_export = pd.DataFrame({"tempo_ms": (tvec_custom[:len(y_sim_val)] * 1000).astype(int), 
                              "angulo_deg": y_sim_val, "u_pct": u_sim_val, 
                              "referencia": x2ref_custom[:len(y_sim_val)]})
    df_export.to_csv(os.path.join(MPC_DIR, f"simulacao_python_{filename_prefix}.csv"), index=False)
    
    df_ref = pd.DataFrame({'tempo_s': df_export['tempo_ms'] / 1000.0, 'referencia_deg': x2ref_custom[:len(y_sim_val)]})
    df_ref.to_csv(os.path.join(SINAIS_REF_DIR, f"referencia_{filename_prefix}.csv"), index=False)
    print(f"Exported {filename_prefix} data.")

# --- VAL 2: Degraus ---
np.random.seed(101)
x2ref_val2 = np.zeros(steps_val)
idx = 0
x2ref_val2[idx : idx + n_pid_start] = 45.0
idx += n_pid_start
while idx < steps_val - n_pid_end:
    dur = int(round(np.random.uniform(5.0, 15.0) / Ts))
    if idx + dur > steps_val - n_pid_end:
        dur = (steps_val - n_pid_end) - idx
    x2ref_val2[idx : idx + dur] = np.random.uniform(20.0, 110.0)
    idx += dur
x2ref_val2[steps_val - n_pid_end : steps_val] = 45.0

run_sil_validation(x2ref_val2, "Degraus Aleatorios", "val2_degraus")

# --- VAL 3: Senoidal Suave ---
x2ref_val3 = np.zeros(steps_val)
x2ref_val3[0 : n_pid_start] = 45.0
n_mid = steps_val - n_pid_start - n_pid_end
t_mid = np.arange(n_mid) * Ts
f_max = 0.1
df_val = 1.0 / (n_mid * Ts)
freqs = np.arange(df_val, f_max + 1e-9, df_val)
phases = np.random.uniform(0, 2 * np.pi, len(freqs))
ms = np.sum([np.sin(2*np.pi*f*t_mid + ph) for f, ph in zip(freqs, phases)], axis=0)
ms = ms - ms[0]  # Garante partida suave em 45 deg sem degrau
ms = ms / np.max(np.abs(ms)) * 40.0 + 45.0
x2ref_val3[n_pid_start : steps_val - n_pid_end] = ms
x2ref_val3[steps_val - n_pid_end : steps_val] = 45.0

run_sil_validation(x2ref_val3, "Senoidal Suave", "val3_senoidal")

plt.show()
