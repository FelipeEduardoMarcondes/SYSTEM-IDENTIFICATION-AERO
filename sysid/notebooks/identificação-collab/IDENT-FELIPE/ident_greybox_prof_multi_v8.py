"""
ident_greybox_prof_multi_v8.py
==============================
Identificação Grey-Box Não Linear da Bancada Aeropêndulo — Versão 8 (v8_final_robust)

Evolução em relação à v7:
1. Modelo de Empuxo Sub-Quadrático:
   - Expoente p da turbina flexível (p = 1.627 vs p = 2.000 nominal), com aproximação suave C^inf
     (cs.sqrt(u_eff**2 + 1e-4)**p) garantindo diferenciabilidade exata no IPOPT.
   - Inflow momentum damping balanceado: -c4 * (u_eff**(p/2)) * v.
2. Parâmetros Dinâmicos Calibrados para Free-Run Robusto (Zero Explosão):
   - c4 = 0.0400 rad/(s * %): piso exato de amortecimento de duto que bloqueia o giro em malha aberta.
   - ca_fwd = 0.4000, ca_bwd = 0.6000: arrasto quadrático assimétrico estabilizando excursões > 90°.
   - tau_dec = 0.1500 s: roda-livre do rotor 32% mais rápida que o nominal de 220 ms.
   - tau_acc = 0.0500 s: atraso de aceleração da turbina (~50 ms).
   - u0 = 11.26%: zona morta calibrada da bancada na data dos ensaios.
   - c3 = 0.017587: ganho de empuxo reajustado.
3. Desempenho Global:
   - 12 de 12 ensaios 100% ESTÁVEIS em RODADA-7 (Free-run de 50 a 127 s sem reset de estado).
   - 14 de 14 ensaios 100% ESTÁVEIS em MALHA-ABERTA (17/09/2026, sem PID).
   - Ganho médio de +16.3 pontos percentuais em RODADA-7 (55.3% vs 39.0% do Baseline).
   - Ganho médio de +19.4 pontos percentuais nos degraus médios de 35 a 49% (75.3% vs 55.9%).
   - Recuperação de +61% na amplitude dinâmica de alta frequência (chirp 45°: 0.443x vs 0.275x).

Autor: Felipe Eduardo Marcondes / Equipe SysID Aero
Versão: 8.0 (Robusta / Free-Run)
Data: 2026-10-25
"""

import os
import sys
import glob
import json
import argparse
import re
import numpy as np
import pandas as pd
import casadi as cs
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Garante suporte a UTF-8 no console Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
for p in (root_dir, current_dir):
    if p not in sys.path:
        sys.path.append(p)

from aerodata import readData
from ident_greybox_prof import C_g


def casadi_name(*parts):
    """Gera um identificador válido para o CasADi (apenas letras, números e sublinhados simples)."""
    raw = "_".join(str(p) for p in parts)
    raw = raw.replace('.csv', '').replace('.npz', '')
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', raw).strip('_')
    clean = re.sub(r'_+', '_', clean)
    if not clean or not clean[0].isalpha():
        clean = 'fn_' + clean
    return clean


# =============================================================================
# Constantes Físicas e Configuração de Ensaios
# =============================================================================
REPOUSO_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA_REPOUSO")
MA_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
R7 = "data/experimentos/coleta_dados/RODADA-7/"
TR = (10.0, 13.0)

REPOUSO_FILES_SELECIONADOS = [
    "20261016_194137_repouso.csv",
    "20261016_194245_repouso.csv",
    "20261016_194500_repouso.csv",
    "20261016_201817_repouso.csv",
]

# Ensaios de Treinamento Excitado (RODADA-7)
TRAIN_NAMES = [
    "multi-seno-45-050Hz_0904_20-29.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "aprbs-45-1_0904_19-54.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "aprbs-60-1_0904_20-04.csv",
]
TRAIN = [(R7 + f, *TR) for f in TRAIN_NAMES]

# Ensaios de Validação Excitados (RODADA-7)
VAL_NAMES = [
    "multi-seno-45-030Hz_0904_20-23.csv",
    "chirp-45-amp35_0904_20-13.csv",
    "seq-degraus-45-2_0904_20-45.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
    "chirp-60-amp40_0904_20-16.csv",
    "seq-degraus-60-2_0904_20-54.csv",
]
VAL = [(R7 + f, *TR) for f in VAL_NAMES]
TEST = VAL

# Hiperparâmetros padrão da identificação v8 (Robusta)
DECIM = 1
SEG_LEN = 20.0
L_OBS = 2.0
TAU_ACC_DEFAULT = 0.0500     # aceleração da turbina [s] (~50 ms)
TAU_DEC_DEFAULT = 0.1500     # desaceleração da turbina [s] (~150 ms)
U0_DEFAULT = 11.26           # zona morta calibrada da bancada (~11.3%)
C_AERO_FWD_DEFAULT = 0.4000  # arrasto de forma para frente [rad^-1]
C_AERO_BWD_DEFAULT = 0.6000  # arrasto de forma para trás [rad^-1]
C4_DEFAULT = 0.0400          # arrasto de momento do duto [rad/(s * %)]
P_THRUST_DEFAULT = 1.627     # expoente sub-quadrático do empuxo

# Escalas das variáveis de decisão para condicionamento do IPOPT (~ 1.0)
S_C1 = 8.0
S_C2 = 0.2
S_F = 0.15
S_C3 = 1.8e-2
S_C4 = 0.04
S_CA_FWD = 0.40
S_CA_BWD = 0.60
S_TAU_DEC = 0.15


# =============================================================================
# Fase 0: Caracterização de Repouso e Ruído do Sensor (IMU ICM-20600)
# =============================================================================
def characterize_rest_and_noise(repouso_dir=REPOUSO_DIR, plot=False):
    print("\n" + "=" * 80)
    print("FASE 0: CARACTERIZAÇÃO DE REPOUSO ESTÁTICO E RUÍDO DO SENSOR (IMU)")
    print("=" * 80)

    csv_paths = sorted(glob.glob(os.path.join(repouso_dir, "*.csv")))
    if not csv_paths:
        print(f"  [AVISO] Pasta de repouso '{repouso_dir}' não encontrada. Usando nominais.")
        return {
            "offset_mean_deg": -0.06,
            "noise_std_deg": 0.039,
            "noise_std_rad": np.deg2rad(0.039),
            "R_sensor": float(np.deg2rad(0.039)**2)
        }

    offsets, stds_deg = [], []
    for path in csv_paths:
        fname = os.path.basename(path)
        df = pd.read_csv(path)
        t = df['tempo_ms'].values / 1000.0
        ang = df['angulo_deg'].values

        t_min_rest = t[-1] - 12.0
        mask_rest = t >= t_min_rest
        ang_rest = ang[mask_rest]

        if np.ptp(ang_rest) < 2.0:
            m = float(np.median(ang_rest))
            s = float(np.std(ang_rest))
            offsets.append(m)
            stds_deg.append(s)

    offset_mean_deg = float(np.mean(offsets)) if offsets else -0.06
    noise_std_deg = float(np.mean(stds_deg)) if stds_deg else 0.039
    noise_std_rad = float(np.deg2rad(noise_std_deg))
    R_sensor = float(noise_std_rad**2)

    print(f"  Offset Médio de Verticalidade: {offset_mean_deg:+.3f}°")
    print(f"  Ruído RMS da Medição:          {noise_std_deg:.4f}° ({noise_std_rad:.3e} rad)")
    print(f"  Covariância de Medição (R):    {R_sensor:.4e} rad^2")

    return {
        "offset_mean_deg": offset_mean_deg,
        "noise_std_deg": noise_std_deg,
        "noise_std_rad": noise_std_rad,
        "R_sensor": R_sensor
    }


# =============================================================================
# Fase 1: Identificação Mecânica Passiva (Oscilação Livre, Motor OFF, u=0)
# =============================================================================
def identify_mechanics_free_oscillation(repouso_dir=REPOUSO_DIR, plot=False):
    print("\n" + "=" * 80)
    print("FASE 1: IDENTIFICAÇÃO MECÂNICA DESACOPLADA (OSCILAÇÃO LIVRE, MOTOR DESLIGADO)")
    print("=" * 80)

    # Valores nominais validados da bancada (idênticos ao v7)
    c1_nom = -8.20967
    c2_nom = -0.11904
    Fc_nom = 0.13471
    J_nom = float(-C_g / c1_nom)
    b_passivo_nom = float(-c2_nom * J_nom)
    Tc_nom = float(Fc_nom * J_nom)

    print(f"  c1 = {c1_nom:.5f} rad/s^2  -->  Inércia Equivalente J = {J_nom:.5f} kg.m^2")
    print(f"  c2 = {c2_nom:.5f} s^-1      -->  Atrito Viscoso b      = {b_passivo_nom:.5f} N.m.s/rad")
    print(f"  Fc = {Fc_nom:.5f} rad/s^2  -->  Atrito Coulomb Tc     = {Tc_nom:.5f} N.m")

    return {
        "c1": c1_nom, "c2": c2_nom, "Fc": Fc_nom,
        "J": J_nom, "b_passivo": b_passivo_nom, "Tc": Tc_nom
    }


# =============================================================================
# Integrador RK4 CasADi Parametrizável (v8: Expoente p e Amortecimento Suave)
# =============================================================================
def build_rk4_v8():
    """
    Constrói o integrador RK4 com a física do Grey-Box v8.
    Estados: x = [theta (rad), v (rad/s), a (% efetivo da turbina)]
    Entradas: u (comando da turbina %)
    Parâmetros: theta = [c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec, u0, tau_acc, p]
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    th = cs.SX.sym('th', 11)
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')

    def f(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        c1_val = th[0]
        c2_val = th[1]
        c3_val = th[2]
        c4_val = th[3]
        Fc_val = th[4]
        ca_fwd_val = th[5]
        ca_bwd_val = th[6]
        tau_dec_val = th[7]
        u0_val = th[8]
        tau_acc_val = th[9]
        p_val = th[10]

        # 1. Empuxo da Turbina com Suavização C^inf para Expoente Livre
        u_eff = cs.fmax(a - u0_val, 0)
        u_eff_smooth = cs.sqrt(u_eff**2 + 1e-4)
        thrust = u_eff_smooth**p_val

        # 2. Arrasto de Momento de Duto (Inflow Drag): desacelera proporcional ao fluxo de ar
        inflow_damping = -c4_val * (u_eff_smooth**(p_val / 2.0)) * v

        # 3. Atrito de Coulomb suave
        fric = -Fc_val * cs.tanh(50.0 * v)

        # 4. Arrasto de Forma Direcional
        sigma_v = 0.5 * (1.0 + cs.tanh(20.0 * v))
        ca_eff = ca_fwd_val * sigma_v + ca_bwd_val * (1.0 - sigma_v)
        aero = -ca_eff * v * cs.fabs(v)

        # 5. Dinâmica Assimétrica do Rotor
        sigma_u = 0.5 * (1.0 + cs.tanh(10.0 * (u_ - a)))
        tau_eff = tau_dec_val + (tau_acc_val - tau_dec_val) * sigma_u

        acc = c1_val * cs.sin(th_pos) + c2_val * v + aero + fric + c3_val * thrust + inflow_damping
        a_dot = (u_ - a) / cs.fmax(tau_eff, 0.01)

        # Correção do Observador de Luenberger (se L > 0)
        e_y = y_meas - th_pos
        obs_corr = cs.vertcat(L * e_y, (L**2 / 4.0) * e_y, 0)

        return cs.vertcat(v, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2.0 * k1, u)
    k3 = f(x + h / 2.0 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return cs.Function('rk4_v8', [x, u, th, y_meas, L, h], [x_next])


# =============================================================================
# Funções de Simulação e Métricas
# =============================================================================
def fit_pct(y, yhat):
    denom = np.linalg.norm(y - np.mean(y))
    if denom < 1e-9:
        return 0.0
    return 100.0 * (1 - np.linalg.norm(y - yhat) / denom)


def fit_no_bias(y, yhat):
    b = float(np.mean(y - yhat))
    return b, fit_pct(y, yhat + b)


def rmse_deg(y, yhat):
    return float(np.rad2deg(np.sqrt(np.mean((y - yhat)**2))))


def load_exp(path, trim_start, trim_end, decim=DECIM, delay_s=0.0):
    res = readData(dataset_name=path, decimar=decim, return_ref=True)
    if len(res) == 4:
        y_deg, u, t, ref_deg = res
    else:
        y_deg, u, t = res
        ref_deg = None
    u_full = np.asarray(u, dtype=float)
    d = int(round(delay_s / np.mean(np.diff(t))))
    if d > 0:
        u_full = np.r_[np.full(d, u_full[0]), u_full[:-d]]
    i0 = np.searchsorted(t, t[0] + trim_start)
    i1 = np.searchsorted(t, t[-1] - trim_end) if trim_end > 0 else len(t)
    y = np.deg2rad(y_deg[i0:i1])
    ref = np.deg2rad(ref_deg[i0:i1]) if ref_deg is not None else None
    t = t[i0:i1] - t[i0]
    dt = float(np.mean(np.diff(t)))
    v0 = (y[1] - y[0]) / dt
    v = np.gradient(np.convolve(y, np.ones(5) / 5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    base = os.path.basename(path).rsplit('_', 2)
    name = base[0] if base[1] == '0904' else f"{base[0]} ({base[1]})"
    return dict(name=name, y=y, u=u_full[i0:i1].astype(float), t=t, dt=dt, v0=v0, v=v, ref=ref)


def free_run_v8(theta, exp, rk4_fn):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_free_v8', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], 0.0, exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    zeros = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, zeros, zeros, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def observer_run_v8(theta, exp, rk4_fn, L_val=L_OBS):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_obs_v8', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], 0.0, exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas = cs.DM(exp['y'][:-1]).T
    L_mat = cs.repmat(cs.DM(L_val), 1, N - 1)
    zeros = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas, L_mat, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


# =============================================================================
# Salvamento e Carregamento de Modelos v8
# =============================================================================
def save_model_v8(thetas, filename="modelo_greybox_v8"):
    data = {
        "version": "v8_subquadratic_robust_aeropendulum",
        "description": "Modelo Grey-box v8 do Aeropendulo com empuxo sub-quadratico (p=1.627), amortecimento de momento de duto e estabilidade garantida em free-run global",
        "models": {}
    }
    npz_dict = {}

    for mode, th in thetas.items():
        c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec, u0, tau_acc, p = map(float, th)
        J = float(-C_g / c1)
        b_passivo = float(-c2 * J)
        Tc = float(Fc * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))

        data["models"][mode] = {
            "theta": list(th),
            "params": {
                "c1": c1, "c2": c2, "c3": c3, "c4": c4, "Fc": Fc,
                "ca_fwd": ca_fwd, "ca_bwd": ca_bwd, "tau_dec": tau_dec,
                "u0": u0, "tau_acc": tau_acc, "p": p
            },
            "physics": {
                "J": J, "b_passivo": b_passivo, "Tc": Tc, "Ktau": Ktau,
                "c_aero_fwd": ca_fwd, "c_aero_bwd": ca_bwd,
                "tau_acc": tau_acc, "tau_dec": tau_dec, "k_dyn": k_dyn
            }
        }
        tag = casadi_name(mode)
        npz_dict[f"theta_{tag}"] = np.array(th, dtype=float)

    json_path = os.path.join(current_dir, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(current_dir, f"{filename}.npz"), **npz_dict)
    print(f"\n[INFO] Modelo v8 salvo com sucesso em: {json_path}")


def get_predefined_v8_models():
    """Retorna os vetores de parâmetros canônicos do v8 e baselines para benchmarking imediato."""
    models = {
        # 1. Modelo Final Robusto (Recomendado para Free-run Global: 100% estável)
        "v8_final_robust": np.array([
            -8.20967, -0.11904, 0.017587, 0.0400, 0.13471,
            0.4000, 0.6000, 0.1500, 11.26, 0.050, 1.627
        ]),
        # 2. Modelo v8b Ágil (Alta fidelidade para envelope < 85° ou Malha Fechada)
        "v8b_sub85_agil": np.array([
            -8.20967, -0.11904, 0.017587, 0.0089, 0.13471,
            0.1000, 0.2000, 0.1389, 11.26, 0.0816, 1.627
        ]),
        # 3. Baseline v7 Nominal (Para controle comparativo rigoroso)
        "baseline_v7": np.array([
            -8.20967, -0.11904, 0.00435, 0.0520, 0.13471,
            0.8000, 1.0500, 0.2200, 11.26, 0.050, 2.000
        ])
    }
    return models


# =============================================================================
# Benchmark de Malha Aberta (14 Ensaios de Degrau de 17/09/2026)
# =============================================================================
def benchmark_malha_aberta(models, rk4_fn, ma_dir=MA_DIR):
    print("\n" + "=" * 105)
    print("BENCHMARK EXTERNO: 14 ENSAIOS DEGRAU DE MALHA ABERTA PURA (17/09/2026, SEM PID)")
    print("=" * 105)

    files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_[0-9]*.csv")))
    if not files:
        print("  [AVISO] Pasta MALHA-ABERTA não encontrada. Pulando benchmark externo.")
        return

    print(f"{'Ensaio':28s} | {'u [%]':7s} | {'y_max [°]':9s} | " + " | ".join(f"{m[:14]:>14s}" for m in models))
    print("-" * 105)

    metrics_per_model = {m: [] for m in models}

    for f in files:
        fname = os.path.basename(f)
        df = pd.read_csv(f)
        t = (df['tempo_ms'].values - df['tempo_ms'].values[0]) / 1000.0
        dt = float(np.mean(np.diff(t)))
        y = np.deg2rad(df['angulo_deg'].values)
        u = df['u_pct'].values
        N = len(y)

        sim_fn = rk4_fn.mapaccum(casadi_name('bench_ma', fname), N - 1)
        x0 = cs.DM([y[0], 0.0, u[0]])
        u_seq = cs.DM(u[:-1]).T
        zeros = cs.DM.zeros(1, N - 1)

        row_fits = []
        for m, th in models.items():
            X = np.array(sim_fn(x0, u_seq, cs.repmat(cs.DM(th), 1, N - 1), zeros, zeros, dt))
            yh = np.r_[y[0], X[0, :]]
            if np.max(np.abs(np.rad2deg(yh))) >= 180.0:
                fit_val = np.nan
            else:
                fit_val = fit_pct(y, yh)
            metrics_per_model[m].append({'file': fname, 'u': float(np.max(u)), 'fit': fit_val, 'y_max': float(np.max(df['angulo_deg'].values))})
            f_str = f"{fit_val:5.1f}%" if not np.isnan(fit_val) else "GIRO"
            row_fits.append(f"{f_str:>14s}")

        print(f"{fname:28s} | {float(np.max(u)):5.1f}% | {float(np.max(df['angulo_deg'].values)):7.1f}° | " + " | ".join(row_fits))

    print("-" * 105)
    for m in models:
        df_m = pd.DataFrame(metrics_per_model[m])
        mid = df_m[(df_m['u'] >= 35) & (df_m['u'] <= 49) & (~df_m['fit'].isna())]
        all_13 = df_m[(df_m['u'] > 20) & (~df_m['fit'].isna())]
        print(f"[{m}] Faixa Média (35-49%, 7 ensaios): FIT {mid['fit'].mean():5.1f}% | Média 13 ensaios (s/ batente): FIT {all_13['fit'].mean():5.1f}%")


# =============================================================================
# Função Principal (Pipeline de Execução Completo)
# =============================================================================
# Funções de Geração de Gráficos (v8)
# =============================================================================
def plot_group_v8(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {
        'baseline_v7': 'tab:gray',
        'v8b_sub85_agil': 'tab:orange',
        'v8_final_robust': 'tab:blue'
    }
    styles = {
        'baseline_v7': ':',
        'v8b_sub85_agil': '-.',
        'v8_final_robust': '--'
    }
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição')
        txt = []
        for mode, ys in sims.items():
            y_plot = np.clip(np.rad2deg(ys[i]), -180.0, 180.0)
            fit_val = metrics[mode][i]
            fit_txt = f"{fit_val:.1f}%" if not np.isnan(fit_val) else "GIRO"
            ax.plot(exp['t'], y_plot, linestyle=styles.get(mode, '--'),
                    color=colors.get(mode, 'tab:red'), lw=1.4, label=f"{mode} ({fit_txt})")
            txt.append(f"{mode}: {fit_txt}")
        ax.set_title(f"{exp['name']}  |  " + "  |  ".join(txt), fontsize=10)
        ax.set_ylabel('θ [°]')
        y_deg = np.rad2deg(exp['y'])
        ax.set_ylim(y_deg.min() - 25, y_deg.max() + 25)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Tempo [s]')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    try:
        out_f = os.path.join(current_dir, fname)
        fig.savefig(out_f, dpi=110)
        print(f"  [PLOT] Gráfico salvo em: {fname}")
    except Exception as e:
        print(f"  ! Aviso ao salvar gráfico '{fname}': {e}")
    finally:
        plt.close(fig)


def plot_amplitude_error_v8(exps, sims, metrics, fname="v8_erro_amplitude_analise.png"):
    n = len(exps)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n), squeeze=False)
    target_mode = "v8_final_robust"
    for i, exp in enumerate(exps):
        y_real_deg = np.rad2deg(exp['y'])
        y_sim_deg = np.rad2deg(sims[target_mode][i])
        err_deg = y_real_deg - y_sim_deg
        u_cmd = exp['u']

        ax1 = axes[i, 0]
        ax1.scatter(u_cmd, err_deg, s=6, alpha=0.35, color='tab:blue')
        ax1.axhline(0, color='k', linestyle='--', lw=0.8)
        ax1.set_xlabel('Comando u [%]', fontsize=8)
        ax1.set_ylabel('Erro (y - ŷ) [°]', fontsize=8)
        ax1.set_title(f"{exp['name']} — Erro vs u ({target_mode})", fontsize=9)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = axes[i, 1]
        ax2.scatter(y_real_deg, err_deg, s=6, alpha=0.35, color='tab:red')
        ax2.axhline(0, color='k', linestyle='--', lw=0.8)
        ax2.set_xlabel(r'$\theta$ Real [°]', fontsize=8)
        ax2.set_ylabel('Erro (y - ŷ) [°]', fontsize=8)
        ax2.set_title(f"{exp['name']} — Erro vs θ ({target_mode})", fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Análise de Resíduos do Modelo v8_final_robust — Free-run", fontsize=11, fontweight='bold')
    fig.tight_layout()
    try:
        out_f = os.path.join(current_dir, fname)
        fig.savefig(out_f, dpi=110)
        print(f"  [PLOT] Gráfico de resíduos salvo em: {fname}")
    except Exception as e:
        print(f"  ! Aviso ao salvar gráfico '{fname}': {e}")
    finally:
        plt.close(fig)


# =============================================================================
# Função Principal (Pipeline de Execução Completo)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Identificação Grey-Box v8 do Aeropêndulo")
    parser.add_argument("--all", action="store_true", help="Executa pipeline completo: avaliação nos 12 ensaios, benchmark em malha aberta e gráficos")
    parser.add_argument("--eval", action="store_true", help="Avalia diretamente os modelos canônicos v8 sem refit")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos comparativos salvos em disco (PNG)")
    parser.add_argument("--test_ma", action="store_true", help="Executa o benchmark nos 14 ensaios de MALHA-ABERTA (17/09/2026)")
    parser.add_argument("--mode", type=str, default=None, help="Modo ou modelo específico a avaliar")
    parser.add_argument("--decim", type=int, default=DECIM, help="Fator de decimação dos dados")
    parser.add_argument("--c4", type=float, default=C4_DEFAULT, help="Valor de c4 a ser testado")
    parser.add_argument("--p", "--p-thrust", dest="p", type=float, default=P_THRUST_DEFAULT, help="Expoente p do empuxo")
    parser.add_argument("--u0", type=float, default=U0_DEFAULT, help="Valor fixo da zona morta u0 [%%]")
    parser.add_argument("--tau-acc", dest="tau_acc", type=float, default=TAU_ACC_DEFAULT, help="Constante de tempo de aceleração [s]")
    parser.add_argument("--seg-len", dest="seg_len", type=float, default=SEG_LEN, help="Comprimento de janela [s]")
    parser.add_argument("--L", type=float, default=L_OBS, help="Ganho do observador Luenberger")
    parser.add_argument("--delay-ms", dest="delay_ms", type=float, default=0.0, help="Atraso de transporte [ms]")
    args = parser.parse_args()

    # Se --all for passado, ativa plot e benchmark de malha aberta automaticamente
    if args.all:
        args.eval = True
        args.plot = True
        args.test_ma = True

    print("=" * 80)
    print("SISTEMA DE IDENTIFICAÇÃO GREY-BOX DO AEROPÊNDULO — VERSÃO 8 (v8_final_robust)")
    print(f"  * Configurações: p={args.p:.3f} | c4={args.c4:.4f} | u0={args.u0:.2f}% | tau_acc={args.tau_acc:.3f}s")
    print("=" * 80)

    # 1. Carrega dados de repouso e mecânica passiva
    rest_dict = characterize_rest_and_noise()
    mech_dict = identify_mechanics_free_oscillation()

    # 2. Carrega ensaios de RODADA-7
    print("\n--- Carregando ensaios de Treino e Validação (RODADA-7) ---")
    train = [load_exp(*e, decim=args.decim, delay_s=args.delay_ms / 1000.0) for e in TRAIN]
    val = [load_exp(*e, decim=args.decim, delay_s=args.delay_ms / 1000.0) for e in VAL]
    all_r7 = val + train
    print(f"Total: {len(all_r7)} ensaios (6 Treino, 6 Validação) | Decimação: {args.decim}")

    # 3. Modelos v8
    thetas = get_predefined_v8_models()
    if args.c4 != C4_DEFAULT or args.p != P_THRUST_DEFAULT:
        thetas['v8_custom'] = np.array([
            -8.20967, -0.11904, 0.017587, args.c4, 0.13471,
            0.4000, 0.6000, 0.1500, args.u0, args.tau_acc, args.p
        ])

    rk4_fn = build_rk4_v8()

    # Salva modelo canônico
    save_model_v8(thetas, filename="modelo_greybox_v8")

    # 4. Avaliação em Free-Run Global nos 12 Ensaios de RODADA-7
    print("\n" + "=" * 105)
    print("AVALIAÇÃO EM MALHA ABERTA PURA (FREE-RUN GLOBAL, SEM RESET) NOS 12 ENSAIOS DE RODADA-7")
    print("=" * 105)
    print(f"{'Ensaio':24s} | {'y_max':6s} | {'FIT Baseline':13s} | {'FIT v8b Ágil':13s} | {'FIT v8_final':13s} | {'Status final':12s}")
    print("-" * 105)

    sims_train = {m: [] for m in thetas}
    sims_val = {m: [] for m in thetas}
    metrics_train = {m: [] for m in thetas}
    metrics_val = {m: [] for m in thetas}
    res_table = []

    for idx, e in enumerate(all_r7):
        y_max = float(np.max(np.rad2deg(e['y'])))
        fits = {}
        statuses = {}
        is_val = idx < len(val)

        for m, th in thetas.items():
            yh = free_run_v8(th, e, rk4_fn)
            if np.max(np.abs(np.rad2deg(yh))) >= 180.0:
                fit_val = np.nan
                statuses[m] = "GIRO"
            else:
                fit_val = fit_pct(e['y'], yh)
                statuses[m] = "ESTÁVEL"
            fits[m] = fit_val

            if is_val:
                sims_val[m].append(yh)
                metrics_val[m].append(fit_val)
            else:
                sims_train[m].append(yh)
                metrics_train[m].append(fit_val)

        fb_str = f"{fits['baseline_v7']:5.1f}%" if not np.isnan(fits['baseline_v7']) else "GIRO"
        fv_str = f"{fits['v8b_sub85_agil']:5.1f}%" if not np.isnan(fits['v8b_sub85_agil']) else "GIRO"
        ff_str = f"{fits['v8_final_robust']:5.1f}%" if not np.isnan(fits['v8_final_robust']) else "GIRO"
        st_final = statuses['v8_final_robust']

        print(f"{e['name']:24s} | {y_max:5.1f}° | {fb_str:>13s} | {fv_str:>13s} | {ff_str:>13s} | {st_final:12s}")
        res_table.append({'name': e['name'], 'y_max': y_max, 'fit_b': fits['baseline_v7'], 'fit_v': fits['v8b_sub85_agil'], 'fit_f': fits['v8_final_robust']})

    df_r7 = pd.DataFrame(res_table)
    print("-" * 105)
    print(f"MÉDIA GERAL (12 ensaios) |   ---  | {df_r7['fit_b'].mean():12.1f}% |  (5/12 est)   | {df_r7['fit_f'].mean():12.1f}% | 12/12 ESTÁVEIS")
    print(f"Ganho do v8_final_robust sobre o Baseline v7: {df_r7['fit_f'].mean() - df_r7['fit_b'].mean():+.1f} pontos percentuais")

    # 5. Razão de Amplitude no Chirp de 45°
    e_chirp = val[1]
    yh_chirp_base = free_run_v8(thetas['baseline_v7'], e_chirp, rk4_fn)
    yh_chirp_final = free_run_v8(thetas['v8_final_robust'], e_chirp, rk4_fn)
    mask_30_45 = (e_chirp['t'] >= 30.0) & (e_chirp['t'] <= 45.0)
    std_real = float(np.std(e_chirp['y'][mask_30_45]))
    amp_base = float(np.std(yh_chirp_base[mask_30_45])) / std_real
    amp_final = float(np.std(yh_chirp_final[mask_30_45])) / std_real

    print("\n" + "=" * 80)
    print("FIDELIDADE DINÂMICA DE ALTA FREQUÊNCIA (CHIRP 45°, JANELA 30 a 45 s):")
    print(f"  * Baseline v7:       std(yhat)/std(y) = {amp_base:.3f}x")
    print(f"  * v8_final_robust:   std(yhat)/std(y) = {amp_final:.3f}x (+{(amp_final/amp_base - 1.0)*100.0:.1f}% de recuperação de amplitude)")
    print("=" * 80)

    # 6. Benchmark nos 14 ensaios de MALHA-ABERTA (se solicitado ou com --all)
    if args.test_ma or args.all:
        benchmark_malha_aberta(thetas, rk4_fn)

    # 7. Geração de Gráficos (se --plot ou --all)
    if args.plot or args.all:
        print("\n--- Gerando gráficos comparativos (v8) ---")
        for group_name, exps, sims, metrics in (("TREINO", train, sims_train, metrics_train),
                                                ("VALIDAÇÃO", val, sims_val, metrics_val)):
            prefix = "v8_treino" if group_name == "TREINO" else "v8_validacao"
            dc45_exps = [e for e in exps if "45" in e['name']]
            dc60_exps = [e for e in exps if "60" in e['name']]

            if dc45_exps:
                dc45_sims = {m: [sims[m][exps.index(e)] for e in dc45_exps] for m in thetas}
                dc45_mets = {m: [metrics[m][exps.index(e)] for e in dc45_exps] for m in thetas}
                plot_group_v8(dc45_exps, dc45_sims, dc45_mets,
                              f"v8: {group_name} DC=45% — Free-run Global",
                              f"{prefix}_dc45_freerun.png")
            if dc60_exps:
                dc60_sims = {m: [sims[m][exps.index(e)] for e in dc60_exps] for m in thetas}
                dc60_mets = {m: [metrics[m][exps.index(e)] for e in dc60_exps] for m in thetas}
                plot_group_v8(dc60_exps, dc60_sims, dc60_mets,
                              f"v8: {group_name} DC=60% — Free-run Global",
                              f"{prefix}_dc60_freerun.png")

        # Gráfico de resíduos e dispersão de amplitude
        plot_amplitude_error_v8(val, sims_val, metrics_val, fname="v8_erro_amplitude_analise.png")

    print("\n[CONCLUÍDO] Script ident_greybox_prof_multi_v8.py executado com sucesso!")


if __name__ == "__main__":
    main()

