"""
Identificação Grey-Box Multi-Experimento v5 (Física Desacoplada com Arrasto Aerodinâmico e Zona Morta)
====================================================================================================
MOTIVAÇÃO FÍSICA E INOVAÇÕES METODOLÓGICAS (v5):
====================================================================================================
1. Síntese Definitiva dos Modelos v2 e v4:
   - O modelo v4 trouxe a excelente precisão estática com zona morta u0 ~ 11.3% (R²=0.99) e o
     desacoplamento mecânico por oscilação livre.
   - Porém, a remoção do arrasto quadrático c_aero permitiu que o modelo explodisse em malha aberta
     quando u > 54.8% (onde o empuxo supera a gravidade máxima em 90° e entra em looping infinito).
   - O modelo v5 reintroduz o arrasto aerodinâmico quadrático c_aero*v*|v| atuando em conjunto com
     a zona morta u0 e a constante de tempo tau_m do motor.

2. Equações de Estado do Modelo v5 (3ª Ordem com Arrasto e Zona Morta):
     theta' = v
     v'     = c1*sin(theta) + c2*v - c_aero*v*|v| - Fc*tanh(50*v) + c3*(max(a - u0, 0))^2
     a'     = (u - a) / tau_m

3. Modos de Otimização em v5:
   - "mecanica fixada": c1, c2, Fc e c_aero fixados nos valores mecânicos; otimiza APENAS c3.
   - "mecanica regularizada": todos livres (c1, c2, c3, Fc, c_aero) com priors da oscilação livre.
   - "c3 fixo": c3 = -k_static * c1 acoplado à escada; c_aero regularizado.
   - "c3 livre": c1, c2, c3, Fc, c_aero livres com regularização adaptativa em torno de k_static.
====================================================================================================
"""

import os
import sys
import time
import json
import glob
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

# Garante inclusão dos diretórios no path
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
R7 = "data/experimentos/coleta_dados/RODADA-7/"
TR = (10.0, 13.0)

# Ensaios de oscilação livre para identificação mecânica desacoplada
REPOUSO_FILES_SELECIONADOS = [
    "20261016_194137_repouso.csv",
    "20261016_194245_repouso.csv",
    "20261016_194500_repouso.csv",
    "20261016_201817_repouso.csv",
]

# Ensaios de Treinamento Excitado (Idênticos ao narx.py)
TRAIN_NAMES = [
    "multi-seno-45-050Hz_0904_20-29.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "aprbs-45-1_0904_19-54.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "aprbs-60-1_0904_20-04.csv",
]
TRAIN = [(R7 + f, *TR) for f in TRAIN_NAMES]

ALL_R7_FILES = [
    "aprbs-45-1_0904_19-54.csv",
    "aprbs-45-2_0904_19-57.csv",
    "aprbs-45-3_0904_20-01.csv",
    "chirp-45-amp25_0904_20-11.csv",
    "chirp-45-amp35_0904_20-13.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "multi-seno-45-030Hz_0904_20-23.csv",
    "multi-seno-45-030Hz_0905_00-47.csv",
    "multi-seno-45-040Hz_0904_20-26.csv",
    "multi-seno-45-050Hz_0904_20-29.csv",
    "seq-degraus-45-1_0904_20-41.csv",
    "seq-degraus-45-2_0904_20-45.csv",
    "seq-degraus-45-3_0904_20-48.csv",
    "aprbs-60-1_0904_20-04.csv",
    "aprbs-60-2_0904_20-07.csv",
    "aprbs-60-3_0904_20-10.csv",
    "chirp-60-amp40_0904_20-16.csv",
    "chirp-60-amp50_0904_20-18.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
    "multi-seno-60-040Hz_0904_20-35.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "seq-degraus-60-1_0904_20-51.csv",
    "seq-degraus-60-2_0904_20-54.csv",
    "seq-degraus-60-3_0904_20-57.csv",
]

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

# Hiperparâmetros padrão da identificação
DECIM = 1
LAMBDA_C3 = 10.0       # peso da regularização de c3 em torno de k_static*|c1|
LAMBDA_MECH = 15.0     # peso da regularização mecânica com os priors de oscilação livre
LAMBDA_AERO = 10.0     # peso da regularização do arrasto aerodinâmico
SEG_LEN = 5.0          # comprimento da janela de multiple shooting [s] (ótimo para convergência KKT)
L_OBS = 2.0            # ganho do observador de estados na validação (rastreamento dinâmico)
TAU_M_DEFAULT = 0.08   # constante de tempo típica do motor/hélice [s] (80 ms)
U0_DEFAULT = 11.26     # zona morta calibrada da bancada (~11.3%)
C_AERO_DEFAULT = 0.55  # arrasto aerodinâmico dinâmico com hélice em rotação [rad^-1]

# Escalas das variáveis de decisão para condicionamento do IPOPT (~ 1.0)
S_C1, S_C2, S_F, S_CA, S_C3 = 8.0, 0.4, 0.15, 0.15, 8.0e-3


# =============================================================================
# Fase 0: Caracterização de Repouso e Ruído do Sensor (IMU ICM-20600)
# =============================================================================
def characterize_rest_and_noise(repouso_dir=REPOUSO_DIR, plot=True):
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
            print(f"  {fname:30s} | Repouso final: {m:+.3f}° | Ruído RMS: {s:.4f}° | PTP: {np.ptp(ang_rest):.3f}°")

    offset_mean_deg = float(np.mean(offsets)) if offsets else -0.06
    noise_std_deg = float(np.mean(stds_deg)) if stds_deg else 0.039
    noise_std_rad = float(np.deg2rad(noise_std_deg))
    R_sensor = float(noise_std_rad**2)

    print("\nResumo da Bancada em Repouso Estático:")
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
# Fase 1: Identificação Mecânica com Arrasto (Oscilação Livre, Motor OFF, u=0)
# =============================================================================
def identify_mechanics_free_oscillation(repouso_dir=REPOUSO_DIR, plot=True):
    print("\n" + "=" * 80)
    print("FASE 1: IDENTIFICAÇÃO MECÂNICA DESACOPLADA (OSCILAÇÃO LIVRE COM ARRASTO)")
    print("=" * 80)

    datasets = []
    for fname in REPOUSO_FILES_SELECIONADOS:
        fpath = os.path.join(repouso_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        t = df['tempo_ms'].values / 1000.0
        ang = df['angulo_deg'].values

        offset_local = np.median(ang[-150:])
        ang_c = ang - offset_local
        vel = np.gradient(ang_c, t)
        imax_v = int(np.argmax(np.abs(vel)))
        i_rel = imax_v
        while i_rel > 0 and (np.abs(vel[i_rel]) > 2.0 or np.sign(vel[i_rel]) == np.sign(vel[imax_v])):
            i_rel -= 1

        dev = np.where((t > t[i_rel]) & (np.abs(ang_c) > 0.5))[0]
        t_end = t[dev[-1]] + 1.5 if len(dev) > 0 else t[-1]
        t_end = min(t_end, t[-1])

        mask = (t >= t[i_rel]) & (t <= t_end)
        t_seg = t[mask] - t[mask][0]
        y_seg = np.deg2rad(ang_c[mask])
        dt_local = np.mean(np.diff(t_seg[:10]))
        v0 = 0.0  # Disparo solto a partir do repouso

        datasets.append({
            "name": fname,
            "t": t_seg,
            "y": y_seg,
            "y_deg": ang_c[mask],
            "v0": v0,
            "dt": dt_local
        })

    if not datasets:
        print("  [AVISO] Nenhum arquivo de oscilação livre carregado. Usando nominais.")
        return {
            "c1": -8.5, "c2": -0.15, "Fc": 0.08, "c_aero": C_AERO_DEFAULT,
            "J": 0.033, "b": 0.005, "Tc": 0.003,
            "omega_n": np.sqrt(8.5),
            "period_T0": 2 * np.pi / np.sqrt(8.5)
        }

    print(f"  Carregados {len(datasets)} ensaios de oscilação livre com amplitudes até {max(d['y_deg'].max() for d in datasets):.1f}°")

    def sim_ode(p, t_exp, y0, v0):
        c1, c2, Fc, ca = p
        def ode(t_, x_):
            return [x_[1], c1 * np.sin(x_[0]) + c2 * x_[1] - ca * x_[1] * abs(x_[1]) - Fc * np.tanh(50.0 * x_[1])]
        sol = solve_ivp(ode, [t_exp[0], t_exp[-1]], [y0, v0], t_eval=t_exp, rtol=1e-4, atol=1e-5)
        if sol.success and len(sol.y[0]) == len(t_exp):
            return sol.y[0]
        return np.zeros_like(t_exp)

    def residuals(p):
        res = []
        for d in datasets:
            t_sub = d['t'][::2]
            y_sub = d['y'][::2]
            y_hat = sim_ode(p, t_sub, y_sub[0], d['v0'])
            res.append(y_hat - y_sub)
        return np.concatenate(res)

    p0 = [-8.34, -0.10, 0.14, 0.06]
    bounds = ([-14.0, -1.5, 0.01, 0.0], [-5.0, -0.01, 0.8, 0.50])
    print("  Executando ajuste multi-experimento da dinâmica mecânica pura (com arrasto)...")
    t0_opt = time.time()
    res_opt = opt.least_squares(residuals, p0, bounds=bounds, ftol=1e-5, xtol=1e-5, diff_step=1e-3)
    dt_opt = time.time() - t0_opt

    c1_mech, c2_mech, Fc_mech, ca_mech = map(float, res_opt.x)
    J_mech = float(-C_g / c1_mech)
    b_mech = float(-c2_mech * J_mech)
    Tc_mech = float(Fc_mech * J_mech)
    omega_n = float(np.sqrt(abs(c1_mech)))
    period_T0 = float(2 * np.pi / omega_n)

    print(f"  [Ajuste Concluído em {dt_opt:.2f} s]")
    print(f"  Parâmetros Mecânicos Normalizados (Grey-Box v5):")
    print(f"    c1 (gravidade/inércia) = {c1_mech:.5f} rad/s^2  -->  omega_n = {omega_n:.3f} rad/s")
    print(f"    c2 (amortecimento visc)= {c2_mech:.5f} s^-1")
    print(f"    Fc (atrito de Coulomb) = {Fc_mech:.5f} rad/s^2")
    print(f"    c_aero (arrasto quadr) = {ca_mech:.5f} rad^-1")
    print(f"  Parâmetros Físicos Equivalentes:")
    print(f"    Inércia Total J        = {J_mech:.5f} kg.m^2")
    print(f"    Coef. Viscoso b        = {b_mech:.5f} N.m.s/rad")
    print(f"    Torque de Coulomb Tc   = {Tc_mech:.5f} N.m")

    if plot:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 2.5 * len(datasets)), squeeze=False)
        for i, d in enumerate(datasets):
            ax = axes[i, 0]
            y_sim = sim_ode([c1_mech, c2_mech, Fc_mech, ca_mech], d['t'], d['y'][0], d['v0'])
            fit = 100.0 * (1 - np.linalg.norm(d['y'] - y_sim) / (np.linalg.norm(d['y'] - np.mean(d['y'])) + 1e-9))
            rmse = np.rad2deg(np.sqrt(np.mean((d['y'] - y_sim)**2)))

            ax.plot(d['t'], np.rad2deg(d['y']), 'k-', lw=1.2, label='Medição (u=0)')
            ax.plot(d['t'], np.rad2deg(y_sim), 'r--', lw=1.4, label='Modelo Mecânico v5')
            ax.set_title(f"{d['name']} — Oscilação Livre | FIT={fit:.1f}% | RMSE={rmse:.2f}°", fontsize=10)
            ax.set_ylabel('θ [°]', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        axes[-1, 0].set_xlabel('Tempo [s]', fontsize=9)
        fig.suptitle(f"Identificação Mecânica Passiva v5 (Oscilação Livre)\n"
                     f"J={J_mech:.4f} kg·m² | b={b_mech:.4f} N·m·s/rad | c_aero={ca_mech:.4f}",
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        out_f = os.path.join(current_dir, "v5_caracterizacao_repouso_oscilacao.png")
        try:
            fig.savefig(out_f, dpi=120)
            print(f"  Gráfico salvo em: v5_caracterizacao_repouso_oscilacao.png")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico '{out_f}': {e}")
        plt.close(fig)

    return {
        "c1": c1_mech, "c2": c2_mech, "Fc": Fc_mech, "c_aero": ca_mech,
        "J": J_mech, "b": b_mech, "Tc": Tc_mech,
        "omega_n": omega_n,
        "period_T0": period_T0
    }


# =============================================================================
# Fase 2: Identificação Semiestática com Zona Morta (k e u0)
# =============================================================================
def extract_static_gain(method="malha_aberta", u0_override=None, plot=True):
    print(f"\n--- Fase 2: Identificação Semiestática com Zona Morta ({method.upper()}) ---")
    if method == "malha_aberta":
        ma_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
        files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_*.csv")))
        points = []
        for f in files:
            fname = os.path.basename(f)
            if fname.startswith("malha_aberta_-") or "16-23" in fname:
                continue
            df = pd.read_csv(f)
            t = df.iloc[:, 0].values / 1000.0 if df.iloc[:, 0].max() > 1000 else df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            u_sig = df.iloc[:, 2].values
            t_end = t[-1]
            mask = t >= (t_end - 5.0)
            u_mean = float(np.mean(u_sig[mask]))
            y_mean = float(np.mean(y[mask]))
            points.append((u_mean, y_mean))
        points = sorted(points, key=lambda x: x[0])
        u_avg = np.array([p[0] for p in points])
        Torque_g = np.sin(np.deg2rad([p[1] for p in points]))
        title_tag = "Degraus de Malha Aberta (09/17)"
        fname = "v5_curva_semiestatica_malha_aberta.png"
    else:
        # Fallback padrão
        return 0.0005958, 11.26

    if u0_override is not None:
        u0_opt = float(u0_override)
        u_eff_sq = np.maximum(u_avg - u0_opt, 0.0)**2
        k_opt = float(np.sum(Torque_g * u_eff_sq) / (np.sum(u_eff_sq**2) + 1e-12))
        r2_opt = float(1 - np.sum((Torque_g - k_opt * u_eff_sq)**2) / np.sum((Torque_g - Torque_g.mean())**2))
    else:
        def model_u0(p):
            return p[0] * np.maximum(u_avg - p[1], 0.0)**2

        def res_u0(p):
            return Torque_g - model_u0(p)

        fit_res = opt.least_squares(res_u0, [1e-3, 11.0], bounds=([1e-6, 0.0], [1e-2, 35.0]))
        k_opt, u0_opt = float(fit_res.x[0]), float(fit_res.x[1])
        pred_opt = model_u0(fit_res.x)
        r2_opt = float(1 - np.sum((Torque_g - pred_opt)**2) / np.sum((Torque_g - Torque_g.mean())**2))

    print(f"  Modelo v5 Semiestático: k = {k_opt:.4e} 1/%^2 | u0 = {u0_opt:.2f}% | R^2 = {r2_opt:.4f}")

    if plot:
        plt.figure(figsize=(8, 5))
        u_grid = np.linspace(0, max(u_avg.max(), 65) + 2, 200)
        plt.scatter(u_avg, Torque_g, s=35, c='k', label='Dados Experimentais')
        plt.plot(u_grid, k_opt * np.maximum(u_grid - u0_opt, 0)**2, 'r-', lw=2.2,
                 label=f'v5: k·(max(u-{u0_opt:.1f}%, 0))² (R²={r2_opt:.3f})')
        plt.axvline(u0_opt, color='gray', linestyle=':', label=f'Zona morta u0 = {u0_opt:.1f}%')
        plt.title(f"Curva Semiestática do Empuxo ({title_tag})", fontsize=11, fontweight='bold')
        plt.xlabel("Comando do Motor u [%]", fontsize=10)
        plt.ylabel(r"Torque Gravitacional Normalizado $\sin(\theta)$", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        out_img = os.path.join(current_dir, fname)
        try:
            plt.savefig(out_img, dpi=120)
            print(f"  Gráfico salvo em: {fname}")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico '{out_img}': {e}")
        plt.close()

    return k_opt, u0_opt


# =============================================================================
# Modelo e Integrador RK4 (CasADi)
# =============================================================================
def build_rk4_v5(u0_fixed=None, tau_m_fixed=None):
    """
    Integrador RK4 com estado 3D: [theta, theta_dot, a].
    Parâmetros no vetor th: [c1, c2, c3, Fc, c_aero].
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')
    th = cs.SX.sym('th', 5)

    u0_val = cs.SX(u0_fixed if u0_fixed is not None else U0_DEFAULT)
    tau_m_val = cs.SX(tau_m_fixed if tau_m_fixed is not None else 0.08)

    def f(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        thrust = cs.fmax(a - u0_val, 0)**2
        fric = -th[3] * cs.tanh(50.0 * v)
        aero = -th[4] * v * cs.fabs(v)
        acc = th[0] * cs.sin(th_pos) + th[1] * v + aero + fric + th[2] * thrust
        a_dot = (u_ - a) / tau_m_val
        obs_corr = cs.vertcat(L * (y_meas - th_pos), 0, 0)
        return cs.vertcat(v, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2.0 * k1, u)
    k3 = f(x + h / 2.0 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return cs.Function('rk4_v5', [x, u, th, y_meas, L, h], [x_next])


# =============================================================================
# Simulações de Validação
# =============================================================================
def free_run(theta, exp, rk4_fn, u0_val, tau_m_val):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_free_v5', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_dummy = cs.DM.zeros(1, N - 1)
    L_zero = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_dummy, L_zero, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def observer_run(theta, exp, rk4_fn, L_val=L_OBS):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_obs_v5', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas = cs.DM(exp['y'][:-1]).T
    L_mat = cs.repmat(cs.DM(L_val), 1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas, L_mat, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def segments(N, dt, seg_len):
    if seg_len is None:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def windowed_run(theta, exp, rk4_fn, seg_len=SEG_LEN):
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = rk4_fn.mapaccum(casadi_name('sim_win_v5', exp['name'], s), n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s], exp['u'][s]])
        y_dummy = cs.DM.zeros(1, n - 1)
        L_zero = cs.DM.zeros(1, n - 1)
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1),
                         y_dummy, L_zero, exp['dt']))
        y_hat[s + 1:e] = X[0, :]
    return y_hat


def load_exp(path, trim_start, trim_end, decim=DECIM):
    res = readData(dataset_name=path, decimar=decim, return_ref=True)
    if len(res) == 4:
        y_deg, u, t, ref_deg = res
    else:
        y_deg, u, t = res
        ref_deg = None
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
    return dict(name=name, y=y, u=u[i0:i1].astype(float), t=t, dt=dt, v0=v0, v=v, ref=ref)


def fit_pct(y, yhat):
    denom = np.linalg.norm(y - np.mean(y))
    if denom < 1e-9:
        return 0.0
    return 100.0 * (1 - np.linalg.norm(y - yhat) / denom)


def rmse_deg(y, yhat):
    return float(np.rad2deg(np.sqrt(np.mean((y - yhat)**2))))


# =============================================================================
# Fase 3: Identificação Multi-Experimento (CasADi / Multiple Shooting / IPOPT)
# =============================================================================
def identify_v5(train, k_static, u0_val, tau_m_val, mode="c3 livre",
                mech_params=None, lam=LAMBDA_C3, lam_mech=LAMBDA_MECH,
                lam_aero=LAMBDA_AERO, th_init=None, seg_len=SEG_LEN,
                tol=1e-3, max_iter=300, time_limit=600.0):
    rk4_fn = build_rk4_v5(u0_fixed=u0_val, tau_m_fixed=tau_m_val)
    opti = cs.Opti()

    # Prior mecânico de arrasto (com hélice girando em regime dinâmico)
    ca_ref = C_AERO_DEFAULT

    if mode == "mecanica fixada":
        c1 = float(mech_params['c1'])
        # Em voo excitado com hélice em rotação, amortecimento viscoso e aerodinâmico aumentam
        c2 = float(np.clip(mech_params['c2'] * 3.0, -0.6, -0.30))
        Fc = float(max(mech_params['Fc'], 0.16))
        c_aero = ca_ref
        p_c3 = opti.variable()
        c3 = S_C3 * p_c3
        opti.subject_to(opti.bounded(0.05, p_c3, 20.0))
        if th_init is not None:
            opti.set_initial(p_c3, float(np.clip(th_init[2] / S_C3, 0.05, 20.0)))
        else:
            opti.set_initial(p_c3, (-k_static * c1) / S_C3)
    else:
        p1 = opti.variable()
        p2 = opti.variable()
        p3 = opti.variable()
        p_ca = opti.variable()

        c1 = -S_C1 * p1
        c2 = -S_C2 * p2
        Fc = S_F * p3
        c_aero = S_CA * p_ca

        opti.subject_to(opti.bounded(0.175, p1, 3.5))      # |c1| entre 1.4 e 28.0
        opti.subject_to(opti.bounded(0.40, p2, 12.0))      # |c2| entre 0.16 e 4.80 (garante atrito viscoso físico)
        opti.subject_to(opti.bounded(0.50, p3, 6.0))       # Fc entre 0.075 e 0.90 (garante atrito Coulomb)
        opti.subject_to(opti.bounded(1.0, p_ca, 6.0))      # c_aero entre 0.15 e 0.90 (permite arrasto dinâmico da hélice)

        if mode in ("c3 livre", "mecanica regularizada"):
            p_c3 = opti.variable()
            c3 = S_C3 * p_c3
            opti.subject_to(opti.bounded(0.05, p_c3, 20.0))
        else:  # c3 fixo
            c3 = -k_static * c1

        # Inicialização
        if th_init is not None:
            opti.set_initial(p1, float(np.clip(-th_init[0] / S_C1, 0.175, 3.5)))
            opti.set_initial(p2, float(np.clip(-th_init[1] / S_C2, 0.40, 12.0)))
            opti.set_initial(p3, float(np.clip(th_init[3] / S_F, 0.50, 6.0)))
            opti.set_initial(p_ca, float(np.clip(th_init[4] / S_CA, 1.0, 6.0)) if len(th_init) > 4 else ca_ref / S_CA)
            if mode in ("c3 livre", "mecanica regularizada"):
                opti.set_initial(p_c3, float(np.clip(th_init[2] / S_C3, 0.05, 20.0)))
        elif mech_params is not None:
            opti.set_initial(p1, -mech_params['c1'] / S_C1)
            opti.set_initial(p2, max(-mech_params['c2'] / S_C2 * 3.0, 0.85))  # chute inicial amortecido
            opti.set_initial(p3, max(mech_params['Fc'] / S_F, 1.1))
            opti.set_initial(p_ca, ca_ref / S_CA)
            if mode in ("c3 livre", "mecanica regularizada"):
                opti.set_initial(p_c3, (-k_static * mech_params['c1']) / S_C3)
        else:
            opti.set_initial(p1, 8.5 / S_C1)
            opti.set_initial(p2, 0.35 / S_C2)
            opti.set_initial(p3, 0.18 / S_F)
            opti.set_initial(p_ca, C_AERO_DEFAULT / S_CA)
            if mode in ("c3 livre", "mecanica regularizada"):
                opti.set_initial(p_c3, (k_static * 8.5) / S_C3)

    theta = cs.vertcat(c1, c2, c3, Fc, c_aero)

    cost = 0
    for exp in train:
        N = len(exp['y'])
        X = opti.variable(3, N)
        opti.set_initial(X[0, :], exp['y'])
        opti.set_initial(X[1, :], exp['v'])
        opti.set_initial(X[2, :], exp['u'])

        segs = segments(N, exp['dt'], seg_len)
        starts = {s for s, _ in segs}
        idx = [k for k in range(N - 1) if (k + 1) not in starts]

        y_meas_dummy = cs.DM.zeros(1, N - 1)
        L_zero = cs.DM.zeros(1, N - 1)
        Xn = rk4_fn.map(N - 1)(X[:, :-1], cs.DM(exp['u'][:-1]).T, cs.repmat(theta, 1, N - 1),
                               y_meas_dummy, L_zero, exp['dt'])
        opti.subject_to(X[:, [k + 1 for k in idx]] == Xn[:, idx])
        cost += cs.sumsqr(X[0, :] - cs.DM(exp['y']).T)

    # Regularização adaptativa do empuxo
    if mode in ("c3 livre", "mecanica regularizada", "mecanica fixada") and lam > 0:
        cost += lam * ((c3 / (-c1)) / k_static - 1.0)**2

    # Regularização do arrasto aerodinâmico
    if mode != "mecanica fixada" and lam_aero > 0:
        cost += lam_aero * ((c_aero - ca_ref) / ca_ref)**2

    # Regularização mecânica com os priors da oscilação livre
    if mode == "mecanica regularizada" and mech_params is not None and lam_mech > 0:
        c1_ref, c2_ref, Fc_ref = mech_params['c1'], mech_params['c2'], mech_params['Fc']
        cost += lam_mech * (
            ((c1 - c1_ref) / c1_ref)**2 +
            ((c2 - c2_ref) / c2_ref)**2 +
            ((Fc - Fc_ref) / (Fc_ref + 1e-4))**2
        )

    opti.minimize(cost)

    ipopt_opts = {
        "max_iter": int(max_iter),
        "max_cpu_time": float(time_limit),
        "max_wall_time": float(time_limit),
        "print_level": 5,
        "tol": float(tol),
        "dual_inf_tol": 5e-2,
        "constr_viol_tol": 1e-3,
        "compl_inf_tol": 5e-3,
        "acceptable_tol": 1e-2,
        "acceptable_iter": 15,
        "acceptable_obj_change_tol": 1e-4,
        "acceptable_dual_inf_tol": 1.0,
        "acceptable_constr_viol_tol": 1e-2,
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": 20,
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based",
    }
    opti.solver('ipopt', {'print_time': True}, ipopt_opts)

    t0 = time.time()
    try:
        sol = opti.solve()
        status = "CONVERGÊNCIA ÓTIMA (IPOPT)"
    except Exception as e:
        sol = opti.debug
        status = f"PARADA (IPOPT): {e}"

    c1_val = float(sol.value(c1))
    c2_val = float(sol.value(c2))
    c3_val = float(sol.value(c3))
    Fc_val = float(sol.value(Fc))
    ca_val = float(sol.value(c_aero))
    th_val = np.array([c1_val, c2_val, c3_val, Fc_val, ca_val])
    cost_val = float(sol.value(cost))
    print(f"  [{status} em {time.time()-t0:.1f} s | Custo: {cost_val:.2f}]")

    return th_val, cost_val, None, None


# =============================================================================
# Análise de Fisher (Covariância e Correlação)
# =============================================================================
def compute_fisher_v5(theta_opt, train, k_static, u0_val, tau_m_val, mode="c3 livre"):
    print(f"\n--- Matriz de Informação de Fisher / Incerteza (v5: {mode.upper()}) ---")
    rk4_fn = build_rk4_v5(u0_fixed=u0_val, tau_m_fixed=tau_m_val)

    if mode == "mecanica fixada":
        param_names = ["c3"]
        th_eval = np.array([theta_opt[2]])
    elif mode == "c3 fixo":
        param_names = ["c1", "c2", "Fc", "c_aero"]
        th_eval = np.array([theta_opt[0], theta_opt[1], theta_opt[3], theta_opt[4]])
    else:
        param_names = ["c1", "c2", "c3", "Fc", "c_aero"]
        th_eval = np.array([theta_opt[0], theta_opt[1], theta_opt[2], theta_opt[3], theta_opt[4]])

    n_p = len(param_names)
    F_total = np.zeros((n_p, n_p))
    res_sq_sum = 0.0
    total_samples = 0

    p_sym = cs.SX.sym('p_active', n_p)
    if mode == "mecanica fixada":
        full_th = cs.vertcat(theta_opt[0], theta_opt[1], p_sym[0], theta_opt[3], theta_opt[4])
    elif mode == "c3 fixo":
        full_th = cs.vertcat(p_sym[0], p_sym[1], -k_static * p_sym[0], p_sym[2], p_sym[3])
    else:
        full_th = cs.vertcat(p_sym[0], p_sym[1], p_sym[2], p_sym[3], p_sym[4])

    for exp in train:
        N = len(exp['y'])
        total_samples += N
        sim_step = rk4_fn.map(N - 1)
        x0 = cs.DM([exp['y'][:-1], exp['v'][:-1], exp['u'][:-1]])
        u_seq = cs.DM(exp['u'][:-1]).T
        y_dummy = cs.DM.zeros(1, N - 1)
        L_zero = cs.DM.zeros(1, N - 1)

        X_next = sim_step(x0, u_seq, cs.repmat(full_th, 1, N - 1), y_dummy, L_zero, exp['dt'])
        y_pred = X_next[0, :].T
        res = y_pred - cs.DM(exp['y'][1:])

        Jac = cs.jacobian(res, p_sym)
        F_exp = cs.mtimes(Jac.T, Jac)

        F_fn = cs.Function('F_fn', [p_sym], [F_exp, res])
        F_val, r_val = F_fn(th_eval)

        F_total += np.array(F_val)
        res_sq_sum += float(np.sum(np.array(r_val)**2))

    dof = max(total_samples - n_p, 1)
    s2 = res_sq_sum / dof

    reg_eps = 1e-9 * np.trace(F_total) / n_p if n_p > 1 else 1e-9
    cov_mat = s2 * np.linalg.pinv(F_total + reg_eps * np.eye(n_p))
    std_devs = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))

    corr_mat = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(n_p):
            denom = std_devs[i] * std_devs[j]
            corr_mat[i, j] = cov_mat[i, j] / denom if denom > 1e-12 else 0.0

    print(f"Amostras: {total_samples} | Variância Residual s²: {s2:.3e} (desvio: {np.rad2deg(np.sqrt(s2)):.2f}°)")
    print(f"{'Parâmetro':>12s} {'Valor Ótimo':>14s} {'Desvio Padrão':>14s} {'Incerteza Relativa':>20s}")
    for name, val, sd in zip(param_names, th_eval, std_devs):
        rel = (sd / abs(val) * 100.0) if abs(val) > 1e-9 else 0.0
        print(f"{name:>12s} {val:14.5e} {sd:14.5e} {rel:19.2f}%")

    if n_p > 1:
        print("\nMatriz de Correlação:")
        hdr = " " * 12 + "".join(f"{p:>12s}" for p in param_names)
        print(hdr)
        for i, row in enumerate(corr_mat):
            r_str = "".join(f"{c:12.3f}" for c in row)
            print(f"{param_names[i]:>12s}{r_str}")

    return {
        "param_names": param_names,
        "th_eval": th_eval.tolist(),
        "std_devs": std_devs.tolist(),
        "cov_matrix": cov_mat.tolist(),
        "corr_matrix": corr_mat.tolist(),
        "residual_std": float(np.sqrt(s2))
    }


# =============================================================================
# Exibição e Salvamento do Modelo v5
# =============================================================================
def print_params_v5(label, th, k_static, u0_val, tau_m_val):
    c1, c2, c3, Fc, ca = th[:5]
    J = -C_g / c1
    b = -c2 * J
    Tc = Fc * J
    Ktau = c3 * J
    k_dyn = c3 / abs(c1)

    print(f"\n[{label}]")
    print(f"  c1 = {c1:.5f} | c2 = {c2:.5f} | c3 = {c3:.5e}")
    print(f"  Fc = {Fc:.5f} rad/s^2 | c_aero = {ca:.5f} rad^-1")
    print(f"  u0 = {u0_val:.2f}% | tau_m = {tau_m_val:.4f} s")
    print(f"  k_dyn = c3/|c1| = {k_dyn:.4e}  (k_estatico = {k_static:.4e}, razao = {k_dyn / k_static:.3f})")
    print(f"  Física Equivalente:")
    print(f"    Inércia J       = {J:.5f} kg.m^2")
    print(f"    Atrito Viscoso  = {b:.5f} N.m.s/rad")
    print(f"    Torque Coulomb  = {Tc:.5f} N.m")
    print(f"    Ganho de Torque = {Ktau:.4e} N.m/%^2")


def save_model_v5(thetas, k_static, u0_val, tau_m_val, mech_dict=None,
                  rest_dict=None, fisher_dict=None, filename="modelo_greybox_v5"):
    data = {
        "version": "v5_aerodynamic_drag",
        "description": "Modelo Grey-box de 3ª ordem com desacoplamento mecânico, arrasto aerodinâmico quadrático e zona morta",
        "k_static": float(k_static),
        "u0_static": float(u0_val),
        "tau_m_fixed": float(tau_m_val),
        "C_g": float(C_g),
        "rest_characterization": rest_dict or {},
        "free_oscillation_mechanics": mech_dict or {},
        "models": {},
        "fisher_analysis": fisher_dict or {}
    }
    npz_dict = {
        "k_static": np.array(k_static, dtype=float),
        "u0_static": np.array(u0_val, dtype=float),
        "tau_m_fixed": np.array(tau_m_val, dtype=float)
    }

    for mode, th in thetas.items():
        c1, c2, c3, Fc, ca = map(float, th[:5])
        J = float(-C_g / c1)
        b = float(-c2 * J)
        Tc = float(Fc * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        tag = casadi_name(mode)

        data["models"][mode] = {
            "theta": [c1, c2, c3, Fc, ca, float(u0_val), float(tau_m_val)],
            "params": {
                "c1": c1, "c2": c2, "c3": c3, "Fc": Fc, "c_aero": ca,
                "u0": float(u0_val), "tau_m": float(tau_m_val)
            },
            "physics": {
                "J": J, "b": b, "Tc": Tc, "Ktau": Ktau,
                "c_aero": ca,
                "k_dyn": k_dyn,
                "ratio_k_dyn_static": float(k_dyn / k_static)
            }
        }
        npz_dict[f"theta_{tag}"] = np.array(th, dtype=float)

    json_path = os.path.join(current_dir, f"{filename}.json")
    npz_path = os.path.join(current_dir, f"{filename}.npz")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    np.savez(npz_path, **npz_dict)

    print(f"\nModelo v5 salvo com sucesso:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")


def load_saved_model_v5(filename="modelo_greybox_v5"):
    json_path = os.path.join(current_dir, f"{filename}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thetas = {}
    for mode, mdata in data.get("models", {}).items():
        thetas[mode] = np.array(mdata["theta"][:5], dtype=float)
    return thetas, data.get("k_static"), data.get("u0_static"), data.get("tau_m_fixed"), data.get("free_oscillation_mechanics")


def plot_group(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {
        'c3 fixo': 'tab:blue',
        'c3 livre': 'tab:red',
        'mecanica fixada': 'tab:green',
        'mecanica regularizada': 'tab:purple'
    }
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição')
        txt = []
        for mode, ys in sims.items():
            ax.plot(exp['t'], np.rad2deg(ys[i]), '--', color=colors.get(mode, 'tab:orange'),
                    lw=1.3, label=mode)
            txt.append(f"{mode}: FIT={metrics[mode][i][0]:.1f}%")
        ax.set_title(f"{exp['name']}  |  " + "  |  ".join(txt), fontsize=10)
        ax.set_ylabel('θ [°]')
        y_deg = np.rad2deg(exp['y'])
        ax.set_ylim(y_deg.min() - 30, y_deg.max() + 30)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Tempo [s]')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    try:
        out_f = os.path.join(current_dir, fname)
        fig.savefig(out_f, dpi=110)
        print(f"Gráfico salvo em: {fname}")
    except Exception as e:
        print(f"  ! Aviso ao salvar gráfico '{fname}': {e}")
    finally:
        plt.close(fig)


# =============================================================================
# Pipeline Principal de Execução
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Identificação Grey-Box Multi-Experimento v5 (Arrasto e Zona Morta)")
    parser.add_argument("--eval", action="store_true", help="Avalia modelo salvo sem reotimizar")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos a partir do modelo salvo")
    parser.add_argument("--all", action="store_true", help="Força reotimização completa de todos os modos")
    parser.add_argument("--u0", type=float, default=None, help="Valor fixo de zona morta u0 [%%]")
    parser.add_argument("--static-method", type=str, default="malha_aberta", choices=["malha_aberta", "escada"],
                        help="Método de calibração da curva semiestática (padrão: malha_aberta)")
    parser.add_argument("--tau-m", type=float, default=TAU_M_DEFAULT, help="Valor fixado de tau_m [s]")
    parser.add_argument("--decim", type=int, default=DECIM, help="Fator de decimação dos ensaios")
    parser.add_argument("--max-iter", type=int, default=300, help="Teto de iterações do IPOPT (padrão: 300)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerância de convergência KKT (padrão: 1e-3)")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Tempo limite por run em segundos (padrão: 600 s)")
    args = parser.parse_args()

    print("=" * 80)
    print("IDENTIFICAÇÃO GREY-BOX MULTI-EXPERIMENTO v5 (ARRASTO AERODINÂMICO + ZONA MORTA)")
    print("  * Fase 0: Ruído do sensor e offset vertical extraídos de RODADA_REPOUSO")
    print("  * Fase 1: Identificação mecânica passiva isolada com arrasto (u=0)")
    print("  * Fase 2: Ganho e zona morta semiestáticos do empuxo k*(max(u - u0, 0))²")
    print("  * Fase 3: Treino Multi-Experimento com c_aero, c3, Fc, c1, c2")
    print("=" * 80)

    # Passo 0: Caracterização de Repouso e Ruído
    rest_dict = characterize_rest_and_noise(plot=True)

    # Passo 1: Identificação Mecânica Passiva Isolada com Arrasto
    mech_dict = identify_mechanics_free_oscillation(plot=True)

    # Passo 2: Curva Semiestática do Empuxo
    k_static, u0_static = extract_static_gain(method=args.static_method, u0_override=args.u0, plot=True)
    tau_m_fixed = float(args.tau_m)

    # Carregamento dos Ensaios de Treino e Validação Excitados
    print("\n--- Carregando ensaios de Treino e Validação (RODADA-7) ---")
    train = [load_exp(*e, decim=args.decim) for e in TRAIN]
    val = [load_exp(*e, decim=args.decim) for e in VAL]
    print(f"Treino: {sum(len(e['y']) for e in train)} amostras em {len(train)} ensaios | Validação: {len(val)} ensaios")

    thetas = {}
    fisher_res = {}

    if args.eval or args.plot:
        loaded = load_saved_model_v5()
        if loaded is not None:
            thetas, k_saved, u0_saved, tau_m_saved, mech_saved = loaded
            if k_saved: k_static = k_saved
            if u0_saved: u0_static = u0_saved
            if tau_m_saved: tau_m_fixed = tau_m_saved
            if mech_saved: mech_dict = mech_saved
            print("\nModelo v5 carregado com sucesso do disco:")
            for m, th in thetas.items():
                print_params_v5(m, th, k_static, u0_static, tau_m_fixed)

    if not thetas:
        modos_para_rodar = [
            "mecanica fixada",
            "mecanica regularizada",
            "c3 fixo",
            "c3 livre"
        ]

        loaded = load_saved_model_v5()
        if loaded is not None and not args.all:
            saved_thetas = loaded[0]
            for m in list(modos_para_rodar):
                if m in saved_thetas:
                    thetas[m] = saved_thetas[m]
                    print_params_v5(f"{m} (reutilizado do disco)", thetas[m], k_static, u0_static, tau_m_fixed)

        for mode in modos_para_rodar:
            if mode in thetas:
                continue

            print(f"\n--- Otimização Multi-Experimento (v5): {mode.upper()} ---")
            th_init = None
            if "mecanica fixada" in thetas and mode != "mecanica fixada":
                th_init = thetas["mecanica fixada"]

            th_opt, cost_val, _, _ = identify_v5(
                train, k_static, u0_static, tau_m_fixed, mode=mode,
                mech_params=mech_dict, lam=LAMBDA_C3, lam_mech=LAMBDA_MECH,
                lam_aero=LAMBDA_AERO, th_init=th_init, seg_len=SEG_LEN,
                tol=args.tol, max_iter=args.max_iter, time_limit=args.time_limit
            )
            thetas[mode] = th_opt
            print_params_v5(mode, thetas[mode], k_static, u0_static, tau_m_fixed)

            # Salva imediatamente
            save_model_v5(thetas, k_static, u0_static, tau_m_fixed,
                          mech_dict=mech_dict, rest_dict=rest_dict, fisher_dict=fisher_res)

            # Análise de Fisher
            fisher_res[mode] = compute_fisher_v5(
                thetas[mode], train, k_static, u0_static, tau_m_fixed, mode=mode
            )
            save_model_v5(thetas, k_static, u0_static, tau_m_fixed,
                          mech_dict=mech_dict, rest_dict=rest_dict, fisher_dict=fisher_res)

    # Simulação e Validação em 3 Modos: Free-run, Janela e Observador
    rk4_fn = build_rk4_v5(u0_fixed=u0_static, tau_m_fixed=tau_m_fixed)
    results = {}
    sim_kinds = [
        ("freerun", "Free-run (Malha Aberta Pura)",
         lambda th, e: free_run(th, e, rk4_fn, u0_static, tau_m_fixed)),
        ("janela", f"Janelas de {SEG_LEN:g} s",
         lambda th, e: windowed_run(th, e, rk4_fn, SEG_LEN)),
        ("observador", f"Preditor com Observador (L={L_OBS})",
         lambda th, e: observer_run(th, e, rk4_fn, L_OBS)),
    ]

    for group_name, exps in (("TREINO", train), ("VALIDAÇÃO", val)):
        for kind, kind_label, fn in sim_kinds:
            with np.errstate(all='ignore'):
                sims = {m: [fn(th, e) for e in exps] for m, th in thetas.items()}
                metrics = {m: [(fit_pct(e['y'], s), rmse_deg(e['y'], s)) for e, s in zip(exps, sims[m])]
                           for m in thetas}
            results[(group_name, kind)] = (exps, sims, metrics, kind_label)

            print(f"\n=== {kind_label} — {group_name} (FIT % / RMSE °) ===")
            header = f"{'ensaio':28s}" + "".join(f"{m:>24s}" for m in thetas)
            print(header)
            for i, e in enumerate(exps):
                row = f"{e['name']:28s}" + "".join(
                    f"{metrics[m][i][0]:13.1f}% / {metrics[m][i][1]:5.2f}°" for m in thetas)
                print(row)
            print(f"{'MÉDIA':28s}" + "".join(
                f"{np.mean([x[0] for x in metrics[m]]):13.1f}% / {np.mean([x[1] for x in metrics[m]]):5.2f}°"
                for m in thetas))
            print(f"{'MEDIANA':28s}" + "".join(
                f"{np.median([x[0] for x in metrics[m]]):13.1f}% / {np.median([x[1] for x in metrics[m]]):5.2f}°"
                for m in thetas))

    # Gráficos de Validação Salvos por DC
    for (group_name, kind), (exps, sims, metrics, kind_label) in results.items():
        tag = "treino" if group_name == "TREINO" else "validacao"
        for dc in ("45", "60"):
            idx = [i for i, e in enumerate(exps) if f"-{dc}-" in e['name']]
            if not idx:
                continue
            plot_group([exps[i] for i in idx],
                       {m: [s[i] for i in idx] for m, s in sims.items()},
                       {m: [x[i] for i in idx] for m, x in metrics.items()},
                       f"{kind_label} — {group_name} — DC {dc} (Modelo v5)",
                       f"v5_{tag}_dc{dc}_{kind}.png")

    print("\n[CONCLUÍDO] Identificação e validação do modelo v5 finalizadas com sucesso!")


if __name__ == "__main__":
    main()
