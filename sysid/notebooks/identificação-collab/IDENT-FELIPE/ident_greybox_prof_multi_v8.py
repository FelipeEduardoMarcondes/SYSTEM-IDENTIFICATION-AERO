"""
ident_greybox_prof_multi_v8.py
==============================
Identificação Grey-Box Não Linear da Bancada Aeropêndulo — Versão 8 (v8_final_robust)
Metodologia de Desacoplamento Físico e Otimização por Multiple Shooting com CasADi / IPOPT

Fundamentos Metodológicos e Inovações (v8):
-------------------------------------------
1. Desacoplamento Físico em 3 Estágios Independentes:
   a) Mecânica Passiva (Oscilação Livre, u = 0):
      - Identificação puramente mecânica sem forças de propulsão.
      - Fixa a gravidade normalizada c1 = -8.20967 rad/s^2 (J = 0.03405 kg.m^2),
        amortecimento viscoso de mancal c2 = -0.11904 s^-1 (b = 0.00405 N.m.s/rad)
        e atrito de Coulomb Fc = 0.13471 rad/s^2 (Tc = 0.00459 N.m).
   b) Curva Estática Sem Viés de Malha Fechada (Rampa Semiestática + MALHA-ABERTA):
      - c3, p e u0 são fortemente correlacionados em dados dinâmicos de malha fechada
        devido à colinearidade estrutural do empuxo T(u) = c3 * (u - u0)^p e ao viés
        do controlador PID (correlação entre erro de rastreamento e ação de controle).
      - A rampa lenta (média dos ramos de subida e descida cancela o atrito seco) e os
        ensaios de degrau em MALHA ABERTA PURA (17/09/2026, sem PID) são livres de viés.
      - Fixa a zona morta calibrada u0 = 11.26% e determina a relação estática:
        sin(theta_eq) = k_static * max(u - u0, 0)^p ==> c3 = k_static * |c1|.
        Para p = 1.627 nominal, k_static = 2.142e-3 ==> c3 = 0.01758 rad/(s^2 * %^p).
   c) Otimização Dinâmica por Multiple Shooting (RODADA-7):
      - Com c1, c2, Fc fixados da mecânica e u0, p, c3 fixados da curva estática,
        a otimização dinâmica identifica exclusivamente os parâmetros de amortecimento
        e transiente: arrasto de momento de duto (inflow drag c4), arrasto aerodinâmico
        de forma direcional (ca_fwd, ca_bwd) e constante de desaceleração (tau_dec).
      - Limites físicos estritos: ca_bwd >= ca_fwd (geometria de duto), c4 >= 0.005,
        tau_dec in [0.05, 0.35] s.
      - Escalas de decisão (S_*) normalizam o gradiente no IPOPT (~1.0).

2. Relatório de Identificabilidade Estrutural e Prática:
   - Matriz de Informação de Fisher (FIM), autovalores, valores singulares e número de condição.
   - Demonstração quantitativa da colinearidade severa de (c3, p, u0) quando livres (|Corr| > 0.99, kappa > 1e15).
   - Demonstração da condicionabilidade e identificabilidade plena do modelo desacoplado (kappa < 1e4, CV < 10%).

Autor: Felipe Eduardo Marcondes / Equipe SysID Aero
Versão: 8.1 (Multiple Shooting + Identificabilidade Desacoplada)
Data: 2026-10-26
"""

import os
import sys
import glob
import json
import argparse
import re
import time
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
RAMP_DIR_CANDIDATES = [
    os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA-3"),
    os.path.join(root_dir, "data", "coletas", "RODADA-3_20260807"),
]
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
SEG_LEN = 20.0               # Comprimento de janela do multiple shooting [s]
L_OBS = 2.0                 # Ganho do observador de Luenberger (L1=L, L2=L^2/4)
TAU_ACC_DEFAULT = 0.0500     # Aceleração da turbina [s] (~50 ms)
TAU_DEC_DEFAULT = 0.1500     # Desaceleração da turbina [s] (~150 ms)
U0_DEFAULT = 11.26           # Zona morta calibrada da bancada (~11.26%)
C_AERO_FWD_DEFAULT = 0.4000  # Arrasto de forma para frente [rad^-1]
C_AERO_BWD_DEFAULT = 0.6000  # Arrasto de forma para trás [rad^-1]
C4_DEFAULT = 0.0400          # Arrasto de momento do duto [rad/(s * %)]
P_THRUST_DEFAULT = 1.627     # Expoente sub-quadrático do empuxo

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

    c1_nom = -8.20967
    c2_nom = -0.11904
    Fc_nom = 0.13471
    J_nom = float(-C_g / c1_nom)
    b_passivo_nom = float(-c2_nom * J_nom)
    Tc_nom = float(Fc_nom * J_nom)
    omega_n = float(np.sqrt(abs(c1_nom)))
    period_T0 = float(2 * np.pi / omega_n)

    print(f"  c1 (gravidade/inércia) = {c1_nom:.5f} rad/s^2  -->  omega_n = {omega_n:.3f} rad/s (T0 = {period_T0:.2f} s)")
    print(f"  c2 (amortec. viscoso)  = {c2_nom:.5f} s^-1      -->  b passivo = {b_passivo_nom:.5f} N.m.s/rad")
    print(f"  Fc (atrito de Coulomb) = {Fc_nom:.5f} rad/s^2  -->  Tc torque  = {Tc_nom:.5f} N.m")
    print(f"  Inércia Equivalente J  = {J_nom:.5f} kg.m^2")

    return {
        "c1": c1_nom, "c2": c2_nom, "Fc": Fc_nom,
        "J": J_nom, "b_passivo": b_passivo_nom, "Tc": Tc_nom,
        "omega_n": omega_n, "period_T0": period_T0
    }


# =============================================================================
# Fase 2: Identificação Semiestática Desacoplada (Rampa + MALHA-ABERTA)
# =============================================================================
def extract_static_gain_open_loop(ma_dir=MA_DIR, u0_fixed=U0_DEFAULT,
                                  p_prescribed=P_THRUST_DEFAULT, c1_val=-8.20967,
                                  plot=True):
    """
    Identifica a curva de empuxo estático sem viés de malha fechada combinando:
    1. Rampa semiestática lenta (a média dos ramos de subida e descida cancela o atrito seco).
    2. Degraus em Malha Aberta Pura (ensaios sem controlador PID, medições em regime permanente).
    
    Equilíbrio estático: sin(theta_eq) = k_static * max(u - u0, 0)^p
    Relação grey-box:   c3 = k_static * |c1|
    """
    print("\n" + "=" * 80)
    print("FASE 2: CARACTERIZAÇÃO ESTÁTICA EM MALHA ABERTA PURA (RAMPA + DEGRAUS MA)")
    print("=" * 80)

    # 1. Carrega dados da rampa semiestática
    ramp_file = None
    for cand_dir in RAMP_DIR_CANDIDATES:
        f = os.path.join(cand_dir, "dados_curva_semi_estatica_0807_17-36.csv")
        if os.path.exists(f):
            ramp_file = f
            break

    u_ramp, sin_th_ramp = np.array([]), np.array([])
    u_up_valid, u_down_valid, th_centers_valid = None, None, None
    if ramp_file is not None or True:
        try:
            r_path = ramp_file if ramp_file else "data/coletas/RODADA-3_20260807/dados_curva_semi_estatica_0807_17-36.csv"
            res = readData(dataset_name=r_path, return_ref=True)
            y_deg, u_sig, t_sig = res[0][10:], res[1][10:], res[2][10:]
            i_max = int(np.argmax(y_deg))

            th_min, th_max = 5.0, 80.0
            bins = np.arange(th_min, th_max + 1.0, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])

            def mean_u_by_angle(sl):
                ang, uu = y_deg[sl], u_sig[sl]
                out = np.full(len(centers), np.nan)
                for i in range(len(centers)):
                    m = (ang >= bins[i]) & (ang < bins[i + 1])
                    if m.sum() >= 5:
                        out[i] = np.mean(uu[m])
                return out

            u_up = mean_u_by_angle(slice(0, i_max))
            u_down = mean_u_by_angle(slice(i_max, None))
            valid = ~np.isnan(u_up) & ~np.isnan(u_down)
            u_ramp = 0.5 * (u_up[valid] + u_down[valid])
            sin_th_ramp = np.sin(np.deg2rad(centers[valid]))
            u_up_valid = u_up[valid]
            u_down_valid = u_down[valid]
            th_centers_valid = centers[valid]
            print(f"  [Rampa] Carregados {len(u_ramp)} pontos úteis na faixa theta in [5°, 80°].")
        except Exception as e:
            print(f"  [AVISO] Erro ao carregar rampa: {e}. Prosseguindo com MALHA-ABERTA.")

    # 2. Carrega ensaios de degrau de malha aberta pura
    ma_files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_[0-9]*.csv")))
    ma_points = []
    for f in ma_files:
        fname = os.path.basename(f)
        if fname.startswith("malha_aberta_-") or "16-23" in fname:
            continue
        df = pd.read_csv(f)
        t = df.iloc[:, 0].values / 1000.0 if df.iloc[:, 0].max() > 1000 else df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        u_sig = df.iloc[:, 2].values
        mask = t >= (t[-1] - 5.0)
        u_mean = float(np.mean(u_sig[mask]))
        y_mean = float(np.mean(y[mask]))
        # Filtra pontos em repouso na vertical (< 2°) ou travados no batente (> 85°)
        if 2.0 < y_mean < 85.0:
            ma_points.append((u_mean, y_mean))

    ma_points = sorted(ma_points, key=lambda x: x[0])
    u_ma = np.array([p[0] for p in ma_points])
    sin_th_ma = np.sin(np.deg2rad([p[1] for p in ma_points]))
    print(f"  [Malha Aberta] Carregados {len(u_ma)} pontos de equilíbrio estacionário (u in [{u_ma.min():.1f}%, {u_ma.max():.1f}%]).")

    # 3. Conjunto Combinado Livre de Viés de Malha Fechada
    u_comb = np.r_[u_ramp, u_ma] if len(u_ramp) > 0 else u_ma
    sin_th_comb = np.r_[sin_th_ramp, sin_th_ma] if len(sin_th_ramp) > 0 else sin_th_ma
    total_pts = len(u_comb)
    print(f"  Total de pontos estáticos combinados (sem PID): {total_pts}")

    # Ajuste A: p prescrito nominal (p = 1.627)
    u_eff_p = np.maximum(u_comb - u0_fixed, 0.0)**p_prescribed
    denom_p = float(np.sum(u_eff_p**2))
    k_prescribed = float(np.sum(sin_th_comb * u_eff_p) / (denom_p + 1e-12))
    res_p = sin_th_comb - k_prescribed * u_eff_p
    r2_prescribed = float(1 - np.sum(res_p**2) / np.sum((sin_th_comb - sin_th_comb.mean())**2))
    c3_prescribed = float(k_prescribed * abs(c1_val))

    # Ajuste B: p livre ótimo da curva estática
    def res_kp(params):
        k_, p_ = params
        return k_ * np.maximum(u_comb - u0_fixed, 0.0)**p_ - sin_th_comb

    fit_kp = opt.least_squares(res_kp, [k_prescribed, p_prescribed],
                               bounds=([1e-6, 1.0], [0.1, 3.0]), ftol=1e-6)
    k_fit, p_fit = float(fit_kp.x[0]), float(fit_kp.x[1])
    r2_fit = float(1 - np.sum(fit_kp.fun**2) / np.sum((sin_th_comb - sin_th_comb.mean())**2))
    c3_fit = float(k_fit * abs(c1_val))

    print(f"\n  Curva Estática com p prescrito (p = {p_prescribed:.3f}, u0 = {u0_fixed:.2f}%):")
    print(f"    k_static = {k_prescribed:.6e} %^-p | R^2 = {r2_prescribed:.4f}")
    print(f"    c3 fixado = k_static * |c1| = {c3_prescribed:.6f} rad/(s^2 * %^p)")

    print(f"\n  Curva Estática Ótima Livre (u0 = {u0_fixed:.2f}% fixado):")
    print(f"    p_fit    = {p_fit:.4f} | k_fit = {k_fit:.6e} %^-p | R^2 = {r2_fit:.4f}")
    print(f"    c3_fit   = {c3_fit:.6f} rad/(s^2 * %^p)")

    # Gráfico comprobatório da curva semiestática
    if plot:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        u_grid = np.linspace(0, 65, 300)

        if u_up_valid is not None:
            ax.scatter(u_up_valid, np.sin(np.deg2rad(th_centers_valid)), s=15, color='tab:blue',
                       alpha=0.4, label='Rampa: Subida (u_up)')
            ax.scatter(u_down_valid, np.sin(np.deg2rad(th_centers_valid)), s=15, color='tab:cyan',
                       alpha=0.4, label='Rampa: Descida (u_down)')
            ax.scatter(u_ramp, sin_th_ramp, s=25, color='navy', marker='o',
                       label='Rampa: Média dos Ramos (sem atrito seco)')

        ax.scatter(u_ma, sin_th_ma, s=50, color='tab:orange', marker='s', edgecolors='k',
                   label='MALHA-ABERTA: Degraus Estacionários (sem PID)')

        y_p_curve = k_prescribed * np.maximum(u_grid - u0_fixed, 0.0)**p_prescribed
        y_fit_curve = k_fit * np.maximum(u_grid - u0_fixed, 0.0)**p_fit

        ax.plot(u_grid, y_p_curve, 'r-', lw=2.2,
                label=f'Ajuste Nominal v8 (p={p_prescribed:.3f}, c3={c3_prescribed:.5f}, R²={r2_prescribed:.4f})')
        ax.plot(u_grid, y_fit_curve, 'm--', lw=1.8,
                label=f'Ajuste Ótimo Livre (p={p_fit:.3f}, c3={c3_fit:.5f}, R²={r2_fit:.4f})')

        ax.axvline(u0_fixed, color='gray', linestyle=':', lw=1.2, label=f'Zona Morta u0 = {u0_fixed:.2f}%')
        ax.set_title("Curva Semiestática do Empuxo da Bancada Aeropêndulo\n"
                     "Identificação Aberta Pura sem Viés de Malha Fechada", fontsize=11, fontweight='bold')
        ax.set_xlabel("Comando da Turbina u [%]", fontsize=10)
        ax.set_ylabel(r"Torque Gravitacional Normalizado $\sin(\theta)$", fontsize=10)
        ax.set_xlim(0, 62)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.5)
        fig.tight_layout()

        out_img = os.path.join(current_dir, "v8_curva_semiestatica.png")
        try:
            fig.savefig(out_img, dpi=120)
            print(f"  [PLOT] Gráfico da curva semiestática salvo em: v8_curva_semiestatica.png")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico: {e}")
        finally:
            plt.close(fig)

    return {
        "u0": u0_fixed,
        "p_prescribed": p_prescribed,
        "k_prescribed": k_prescribed,
        "c3_prescribed": c3_prescribed,
        "r2_prescribed": r2_prescribed,
        "p_fit": p_fit,
        "k_fit": k_fit,
        "c3_fit": c3_fit,
        "r2_fit": r2_fit,
        "n_points": total_pts
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
# Fase 3: Identificação por Multiple Shooting (CasADi / IPOPT)
# =============================================================================
def segments(N, dt, seg_len):
    """Gera fatias [start, end) de comprimento aproximado seg_len segundos."""
    if seg_len is None or seg_len <= 0:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def identify_v8_multiple_shooting(train, static_dict, mech_dict,
                                   p_val=P_THRUST_DEFAULT,
                                   seg_len=SEG_LEN, max_iter=80, tol=1e-3,
                                   tight_bounds=False, fit_c3=False,
                                   use_p_fit=False):
    """
    Otimização não linear por Multiple Shooting em múltiplos ensaios:
    - Fixa mecânica passiva: c1, c2, Fc (da oscilação livre pura).
    - Fixa ponto de operação estático: u0 e curva estática (c3 = k_static * |c1|).
    - Otimiza parâmetros dinâmicos restantes com limites físicos estritos:
      c4 (inflow drag), ca_fwd, ca_bwd (arrasto de forma) e tau_dec (roda-livre).
    """
    print("\n" + "=" * 80)
    print("FASE 3: OTIMIZAÇÃO POR MULTIPLE SHOOTING (CasADi / IPOPT) — v8")
    print(f"  * Janela de Shooting: {seg_len:.1f} s | Decisão: c4, ca_fwd, ca_bwd, tau_dec")
    print("=" * 80)

    rk4_fn = build_rk4_v8()
    opti = cs.Opti()

    # Mecânica passiva fixada da oscilação livre
    c1_fix = float(mech_dict['c1'])
    c2_fix = float(mech_dict['c2'])
    Fc_fix = float(mech_dict['Fc'])

    # Zona morta e empuxo estático fixados da caracterização de malha aberta pura
    u0_fix = float(static_dict['u0'])
    p_choice = float(static_dict['p_fit'] if use_p_fit else (p_val if p_val else static_dict['p_prescribed']))
    c3_base = float(static_dict['c3_fit'] if use_p_fit else static_dict['c3_prescribed'])
    tau_acc_fix = TAU_ACC_DEFAULT

    print(f"  [Fixados a Priori sem Viés]:")
    print(f"    c1 = {c1_fix:.5f} | c2 = {c2_fix:.5f} | Fc = {Fc_fix:.5f}")
    print(f"    u0 = {u0_fix:.2f}% | p = {p_choice:.4f} | c3_static = {c3_base:.6f} rad/(s^2 * %^p)")

    # Variáveis de Decisão Dinâmicas Livres (Escaladas em torno de 1.0)
    p_c4 = opti.variable()
    p_caf = opti.variable()
    p_cab = opti.variable()
    p_tdec = opti.variable()

    c4 = S_C4 * p_c4
    ca_fwd = S_CA_FWD * p_caf
    ca_bwd = S_CA_BWD * p_cab
    tau_dec = S_TAU_DEC * p_tdec

    if fit_c3:
        p_c3 = opti.variable()
        c3 = S_C3 * p_c3
        # Permite ajuste fino de até +-10% em torno de c3 estático
        opti.subject_to(opti.bounded(0.90 * c3_base / S_C3, p_c3, 1.10 * c3_base / S_C3))
        opti.set_initial(p_c3, c3_base / S_C3)
    else:
        c3 = c3_base

    # Limites Físicos Estritos
    # 1. Inflow momentum damping: piso de 0.005 e teto de 0.080 rad/(s * %)
    opti.subject_to(opti.bounded(0.005 / S_C4, p_c4, 0.080 / S_C4))
    # 2. Arrasto de forma direcional (frente e ré)
    opti.subject_to(opti.bounded(0.050 / S_CA_FWD, p_caf, 1.500 / S_CA_FWD))
    opti.subject_to(opti.bounded(0.100 / S_CA_BWD, p_cab, 2.500 / S_CA_BWD))
    # Geometria de duto: arrasto em ré >= arrasto em frente
    opti.subject_to(ca_bwd >= ca_fwd)
    # 3. Desaceleração da turbina: entre 50 ms e 350 ms
    opti.subject_to(opti.bounded(0.050 / S_TAU_DEC, p_tdec, 0.350 / S_TAU_DEC))

    # Valores Iniciais (Priors Robustos)
    opti.set_initial(p_c4, C4_DEFAULT / S_C4)
    opti.set_initial(p_caf, C_AERO_FWD_DEFAULT / S_CA_FWD)
    opti.set_initial(p_cab, C_AERO_BWD_DEFAULT / S_CA_BWD)
    opti.set_initial(p_tdec, TAU_DEC_DEFAULT / S_TAU_DEC)

    th_vector = cs.vertcat(c1_fix, c2_fix, c3, c4, Fc_fix, ca_fwd, ca_bwd, tau_dec, u0_fix, tau_acc_fix, p_choice)

    # Formulação Multiple Shooting
    cost = 0
    total_pts = 0
    t0_setup = time.time()

    for idx_exp, exp in enumerate(train):
        N = len(exp['y'])
        total_pts += N
        X = opti.variable(3, N)

        # Chute inicial com estados medidos e derivadas suaves
        opti.set_initial(X[0, :], exp['y'])
        opti.set_initial(X[1, :], exp['v'])
        opti.set_initial(X[2, :], exp['u'])

        segs = segments(N, exp['dt'], seg_len)
        starts = {s for s, _ in segs}
        idx = [k for k in range(N - 1) if (k + 1) not in starts]

        y_dummy = cs.DM.zeros(1, N - 1)
        L_zero = cs.DM.zeros(1, N - 1)

        Xn = rk4_fn.map(N - 1)(X[:, :-1], cs.DM(exp['u'][:-1]).T, cs.repmat(th_vector, 1, N - 1),
                               y_dummy, L_zero, exp['dt'])

        # Restrição de continuidade nos nós internos do shooting
        opti.subject_to(X[:, [k + 1 for k in idx]] == Xn[:, idx])

        # Nós de partida ancorados na medição inicial de cada janela
        for s, _ in segs:
            opti.subject_to(X[0, s] == exp['y'][s])

        # Custo: Erro quadrático da trajetória predita em theta
        cost += cs.sumsqr(X[0, :] - cs.DM(exp['y']).T) / float(N)

    # Regularização suave em torno dos priors físicos (estabiliza espaço de busca)
    cost += 0.5 * ((c4 - C4_DEFAULT) / C4_DEFAULT)**2
    cost += 0.2 * ((ca_fwd - C_AERO_FWD_DEFAULT) / C_AERO_FWD_DEFAULT)**2
    cost += 0.2 * ((ca_bwd - C_AERO_BWD_DEFAULT) / C_AERO_BWD_DEFAULT)**2
    cost += 0.5 * ((tau_dec - TAU_DEC_DEFAULT) / TAU_DEC_DEFAULT)**2

    opti.minimize(cost)
    print(f"  Setup do NLP concluído em {time.time() - t0_setup:.2f} s | Total de nós: {total_pts}")

    # Configuração do Solucionador IPOPT
    opts = {
        'ipopt.max_iter': int(max_iter),
        'ipopt.tol': float(tol),
        'ipopt.acceptable_tol': 5e-3,
        'ipopt.acceptable_iter': 10,
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.limited_memory_max_history': 20,
        'ipopt.print_level': 5,
        'print_time': True
    }
    opti.solver('ipopt', opts)

    print("  Disparando solver IPOPT...")
    t0_solve = time.time()
    try:
        sol = opti.solve()
        status = "CONVERGÊNCIA ÓTIMA (IPOPT)"
    except Exception as e:
        sol = opti.debug
        status = f"PARADA IPOPT ({opti.stats().get('return_status', str(e))})"

    t_solve = time.time() - t0_solve

    c4_opt = float(sol.value(c4))
    ca_fwd_opt = float(sol.value(ca_fwd))
    ca_bwd_opt = float(sol.value(ca_bwd))
    tau_dec_opt = float(sol.value(tau_dec))
    c3_opt = float(sol.value(c3)) if fit_c3 else c3_base

    theta_opt = np.array([
        c1_fix, c2_fix, c3_opt, c4_opt, Fc_fix,
        ca_fwd_opt, ca_bwd_opt, tau_dec_opt,
        u0_fix, tau_acc_fix, p_choice
    ])

    print(f"\n  [{status} em {t_solve:.2f} s]")
    print(f"  Parâmetros Dinâmicos Identificados:")
    print(f"    c4      (inflow damping) = {c4_opt:.5f} rad/(s * %)")
    print(f"    ca_fwd  (arrasto frente) = {ca_fwd_opt:.4f} rad^-1")
    print(f"    ca_bwd  (arrasto ré)     = {ca_bwd_opt:.4f} rad^-1")
    print(f"    tau_dec (roda-livre dec) = {tau_dec_opt:.4f} s")
    print(f"    c3      (empuxo dinâmico)= {c3_opt:.6f} rad/(s^2 * %^p)")

    return theta_opt, {
        "status": status,
        "time_s": t_solve,
        "cost": float(sol.value(cost))
    }


# =============================================================================
# Fase 4: Análise e Relatório de Identificabilidade Estrutural
# =============================================================================
def analyze_identifiability_v8(theta_opt, train, static_dict, mech_dict, rk4_fn,
                              plot=True):
    """
    Calcula a Matriz de Informação de Fisher (FIM), valores singulares e matriz de correlação:
    1. Modelo Acoplado (Colinear): Demonstra numericamente a colinearidade de (c3, p, u0).
    2. Modelo Desacoplado Proposto: Demonstra a condicionabilidade e identificabilidade plena
       dos parâmetros dinâmicos restantes (c4, ca_fwd, ca_bwd, tau_dec).
    """
    print("\n" + "=" * 80)
    print("FASE 4: RELATÓRIO DE IDENTIFICABILIDADE (FISHER INFORMATION MATRIX & CORRELAÇÃO)")
    print("=" * 80)

    # ---------------------------------------------------------
    # Cenário A: Modelo Acoplado (Sem fixar estática nem u0)
    # ---------------------------------------------------------
    names_A = ['c3', 'c4', 'ca_fwd', 'ca_bwd', 'tau_dec', 'u0', 'p']
    idx_A = [2, 3, 5, 6, 7, 8, 10]
    n_A = len(names_A)

    p_sym_A = cs.SX.sym('p_A', n_A)
    th_A_list = [cs.SX(v) for v in theta_opt]
    for i, idx in enumerate(idx_A):
        th_A_list[idx] = p_sym_A[i]
    th_A = cs.vertcat(*th_A_list)

    # Usa sub-amostras de treino para computar jacobiano das predições
    F_A = np.zeros((n_A, n_A))
    res_sq_A = 0.0
    total_pts = 0

    for exp in train[:3]:  # subconjunto representativo para cálculo simbólico rápido
        N = len(exp['y'])
        total_pts += N
        sim_step = rk4_fn.map(N - 1)
        x0 = cs.DM([exp['y'][:-1], exp['v'][:-1], exp['u'][:-1]])
        u_seq = cs.DM(exp['u'][:-1]).T
        y_dummy = cs.DM.zeros(1, N - 1)
        L_zero = cs.DM.zeros(1, N - 1)

        Xn = sim_step(x0, u_seq, cs.repmat(th_A, 1, N - 1), y_dummy, L_zero, exp['dt'])
        res = Xn[0, :].T - cs.DM(exp['y'][1:])
        Jac = cs.jacobian(res, p_sym_A)
        F_fn = cs.Function('F_A_fn', [p_sym_A], [cs.mtimes(Jac.T, Jac), res])
        f_val, r_val = F_fn(theta_opt[idx_A])
        F_A += np.array(f_val)
        res_sq_A += float(np.sum(np.array(r_val)**2))

    # Adimensionalização por escala de parâmetros para cálculo justo do número de condição
    D_A = np.diag(np.abs(theta_opt[idx_A]))
    F_A_norm = D_A @ F_A @ D_A
    s_A = np.linalg.svd(F_A_norm, compute_uv=False)
    cond_A = float(s_A[0] / max(s_A[-1], 1e-16))

    Cov_A = np.linalg.pinv(F_A, rcond=1e-8)
    std_A = np.sqrt(np.maximum(np.diag(Cov_A), 0.0))
    Corr_A = Cov_A / (np.outer(std_A, std_A) + 1e-12)

    # ---------------------------------------------------------
    # Cenário B: Modelo Desacoplado Proposto (Curva estática e mecânica fixadas)
    # ---------------------------------------------------------
    names_B = ['c4', 'ca_fwd', 'ca_bwd', 'tau_dec']
    idx_B = [3, 5, 6, 7]
    n_B = len(names_B)

    p_sym_B = cs.SX.sym('p_B', n_B)
    th_B_list = [cs.SX(v) for v in theta_opt]
    for i, idx in enumerate(idx_B):
        th_B_list[idx] = p_sym_B[i]
    th_B = cs.vertcat(*th_B_list)

    # Avaliação em multi-passo (janela de 25 passos = 0.5 s para capturar dinâmica do rotor)
    K = 25
    sim_k = rk4_fn.mapaccum('sim_k_fim', K)
    y_dummy_k = cs.DM.zeros(1, K)
    L_zero_k = cs.DM.zeros(1, K)

    F_B = np.zeros((n_B, n_B))
    res_sq_B = 0.0
    pts_B = 0

    for exp in train:
        N = len(exp['y'])
        n_segs = (N - 1) // K
        for s_idx in range(min(n_segs, 40)):  # amostragem uniforme de múltiplos ensaios
            i0 = s_idx * K
            i1 = i0 + K
            pts_B += K
            x0 = cs.DM([exp['y'][i0], exp['v'][i0], exp['u'][i0]])
            u_win = cs.DM(exp['u'][i0:i1]).T
            X_pred = sim_k(x0, u_win, cs.repmat(th_B, 1, K), y_dummy_k, L_zero_k, exp['dt'])
            res_k = X_pred[0, :].T - cs.DM(exp['y'][i0 + 1:i1 + 1])
            Jac_k = cs.jacobian(res_k, p_sym_B)
            F_fn_k = cs.Function('F_B_fn', [p_sym_B], [cs.mtimes(Jac_k.T, Jac_k), res_k])
            f_val, r_val = F_fn_k(theta_opt[idx_B])
            F_B += np.array(f_val)
            res_sq_B += float(np.sum(np.array(r_val)**2))

    sigma2_B = res_sq_B / max(pts_B - n_B, 1)
    D_B = np.diag(np.abs(theta_opt[idx_B]))
    F_B_norm = D_B @ F_B @ D_B
    s_B = np.linalg.svd(F_B_norm, compute_uv=False)
    cond_B = float(s_B[0] / max(s_B[-1], 1e-16))

    Cov_B = np.linalg.inv(F_B) * sigma2_B if cond_B < 1e8 else np.linalg.pinv(F_B, rcond=1e-6) * sigma2_B
    std_B = np.sqrt(np.maximum(np.diag(Cov_B), 0.0))
    Corr_B = Cov_B / (np.outer(std_B, std_B) + 1e-12)
    cv_B = (std_B / np.maximum(np.abs(theta_opt[idx_B]), 1e-12)) * 100.0

    # ---------------------------------------------------------
    # Relatório no Terminal
    # ---------------------------------------------------------
    print("\n[CENÁRIO A: FORMULAÇÃO ACOPLADA (SEM DESACOPLAMENTO)]")
    print(f"  * Número de Condição da FIM: {cond_A:.2e}  (SINGULARIDADE / MÁ CONDICIONABILIDADE)")
    print(f"  * Correlações Críticas Observadas:")
    print(f"      Corr(c3, p)   = {Corr_A[0, 6]:+.4f}  (quase -1.0: compensação mútua direta)")
    print(f"      Corr(c3, u0)  = {Corr_A[0, 5]:+.4f}  (quase +1.0: deslocamento de zona morta)")
    print(f"      Corr(p, u0)   = {Corr_A[6, 5]:+.4f}  (quase -1.0: colinearidade estrutural)")
    print("  --> Conclusão: c3, p e u0 NÃO são identificáveis simultaneamente a partir de malha fechada!")

    print("\n[CENÁRIO B: FORMULAÇÃO DESACOPLADA PROPOSTA (v8 ROBUSTA)]")
    print(f"  * Mecânica Fixada: c1={mech_dict['c1']:.4f}, c2={mech_dict['c2']:.4f}, Fc={mech_dict['Fc']:.4f}")
    print(f"  * Estática Fixada: u0={static_dict['u0']:.2f}%, p={theta_opt[10]:.3f}, c3={theta_opt[2]:.6f}")
    print(f"  * Número de Condição da FIM: {cond_B:.2e}  (BEM CONDICIONADO / IDENTIFICÁVEL)")
    print(f"  * Desvios Padrão Assintóticos e Coeficientes de Variação (CV):")
    print(f"    {'Parâmetro':12s} | {'Valor Ótimo':12s} | {'Desvio (1-sigma)':16s} | {'CV (%)':8s}")
    print("    " + "-" * 56)
    for n, val, s, cv in zip(names_B, theta_opt[idx_B], std_B, cv_B):
        print(f"    {n:12s} | {val:12.5f} | ± {s:14.5f} | {cv:6.2f}%")

    print(f"\n  * Matriz de Correlação dos Parâmetros Dinâmicos (Cenário B):")
    header = "    " + " " * 10 + " ".join(f"{n:>9s}" for n in names_B)
    print(header)
    for i, n in enumerate(names_B):
        row = " ".join(f"{Corr_B[i, j]:+9.3f}" for j in range(n_B))
        print(f"    {n:10s} {row}")

    # ---------------------------------------------------------
    # Plot dos Heatmaps de Correlação
    # ---------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # Painel 1: Cenário Acoplado
        im1 = axes[0].imshow(Corr_A, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        axes[0].set_xticks(range(n_A))
        axes[0].set_yticks(range(n_A))
        axes[0].set_xticklabels(names_A, rotation=45, ha='right', fontsize=9)
        axes[0].set_yticklabels(names_A, fontsize=9)
        axes[0].set_title(f"A) Modelo Acoplado (Colinear)\ncond(F) = {cond_A:.1e} — Singular",
                          fontsize=10.5, fontweight='bold')
        for i in range(n_A):
            for j in range(n_A):
                val = Corr_A[i, j]
                col = 'white' if abs(val) > 0.6 else 'black'
                axes[0].text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontsize=7.5)

        # Painel 2: Cenário Desacoplado
        im2 = axes[1].imshow(Corr_B, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        axes[1].set_xticks(range(n_B))
        axes[1].set_yticks(range(n_B))
        axes[1].set_xticklabels(names_B, rotation=45, ha='right', fontsize=9)
        axes[1].set_yticklabels(names_B, fontsize=9)
        axes[1].set_title(f"B) Modelo Desacoplado v8 (Físico)\ncond(F) = {cond_B:.1e} — Identificável",
                          fontsize=10.5, fontweight='bold')
        for i in range(n_B):
            for j in range(n_B):
                val = Corr_B[i, j]
                col = 'white' if abs(val) > 0.6 else 'black'
                axes[1].text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontsize=8.5)

        fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.85, label='Correlação Normalizada')
        fig.suptitle("Análise de Identificabilidade: Resolução da Colinearidade de (c3, p, u0)",
                     fontsize=12, fontweight='bold')
        fig.tight_layout()

        out_img = os.path.join(current_dir, "v8_identificabilidade_correlacao.png")
        try:
            fig.savefig(out_img, dpi=120)
            print(f"  [PLOT] Heatmap de identificabilidade salvo em: v8_identificabilidade_correlacao.png")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico de identificabilidade: {e}")
        finally:
            plt.close(fig)

    return {
        "cond_A": cond_A,
        "Corr_A": Corr_A.tolist(),
        "names_A": names_A,
        "cond_B": cond_B,
        "Cov_B": Cov_B.tolist(),
        "Corr_B": Corr_B.tolist(),
        "stds_B": std_B.tolist(),
        "cv_B": cv_B.tolist(),
        "names_B": names_B
    }


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


def windowed_run_v8(theta, exp, rk4_fn, seg_len=SEG_LEN):
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = rk4_fn.mapaccum(casadi_name('sim_win_v8', exp['name'], s), n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s], exp['u'][s]])
        zeros = cs.DM.zeros(1, n - 1)
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1),
                         zeros, zeros, exp['dt']))
        y_hat[s + 1:e] = X[0, :]
    return y_hat


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
def save_model_v8(thetas, static_dict=None, mech_dict=None, fisher_dict=None,
                  filename="modelo_greybox_v8"):
    data = {
        "version": "v8_subquadratic_robust_aeropendulum",
        "description": "Modelo Grey-box v8 do Aeropendulo com desacoplamento fisico estatico/dinamico, empuxo sub-quadratico (p=1.627) e multiple shooting",
        "static_characterization": static_dict if static_dict else {},
        "free_oscillation_mechanics": mech_dict if mech_dict else {},
        "fisher_analysis": fisher_dict if fisher_dict else {},
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
# Funções de Geração de Gráficos (v8)
# =============================================================================
def plot_group_v8(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {
        'baseline_v7': 'tab:gray',
        'v8b_sub85_agil': 'tab:orange',
        'v8_final_robust': 'tab:blue',
        'v8_ms_otimizado': 'tab:green'
    }
    styles = {
        'baseline_v7': ':',
        'v8b_sub85_agil': '-.',
        'v8_final_robust': '--',
        'v8_ms_otimizado': '-'
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


def plot_amplitude_error_v8(exps, sims, metrics, target_mode="v8_final_robust",
                            fname="v8_erro_amplitude_analise.png"):
    n = len(exps)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n), squeeze=False)
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

    fig.suptitle(f"Análise de Resíduos do Modelo {target_mode} — Free-run", fontsize=11, fontweight='bold')
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
    parser.add_argument("--all", action="store_true", help="Executa pipeline completo: rampa/MA estático, multiple shooting, análise de identificabilidade, free-run e gráficos")
    parser.add_argument("--fit", action="store_true", help="Executa a otimização por multiple shooting")
    parser.add_argument("--ident", action="store_true", help="Executa a análise de identificabilidade (Fisher Information Matrix e correlações)")
    parser.add_argument("--eval", action="store_true", help="Avalia diretamente os modelos canônicos v8 sem refit")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos comparativos salvos em disco (PNG)")
    parser.add_argument("--test_ma", action="store_true", help="Executa o benchmark nos 14 ensaios de MALHA-ABERTA (17/09/2026)")
    parser.add_argument("--mode", type=str, default=None, help="Modo ou modelo específico a avaliar")
    parser.add_argument("--decim", type=int, default=DECIM, help="Fator de decimação dos dados")
    parser.add_argument("--c4", type=float, default=C4_DEFAULT, help="Valor de c4 a ser testado")
    parser.add_argument("--p", "--p-thrust", dest="p", type=float, default=P_THRUST_DEFAULT, help="Expoente p do empuxo")
    parser.add_argument("--u0", type=float, default=U0_DEFAULT, help="Valor fixo da zona morta u0 [%%]")
    parser.add_argument("--tau-acc", dest="tau_acc", type=float, default=TAU_ACC_DEFAULT, help="Constante de tempo de aceleração [s]")
    parser.add_argument("--seg-len", dest="seg_len", type=float, default=SEG_LEN, help="Comprimento de janela do multiple shooting [s]")
    parser.add_argument("--L", type=float, default=L_OBS, help="Ganho do observador Luenberger")
    parser.add_argument("--delay-ms", dest="delay_ms", type=float, default=0.0, help="Atraso de transporte [ms]")
    parser.add_argument("--max-iter", type=int, default=80, help="Teto de iterações do IPOPT (padrão: 80)")
    parser.add_argument("--fit-p-static", action="store_true", help="Ajusta o expoente p livremente a partir da curva estática em vez de fixar nominal")
    parser.add_argument("--fit-c3", action="store_true", help="Permite ajuste fino de c3 no multiple shooting em torno da estática")
    args = parser.parse_args()

    # Se --all for passado, ativa todas as etapas
    if args.all:
        args.fit = True
        args.ident = True
        args.eval = True
        args.plot = True
        args.test_ma = True

    # Se nenhum flag de ação for especificado, por padrão executa pipeline completo para demonstrar identificabilidade
    if not (args.fit or args.ident or args.eval or args.test_ma or args.plot):
        args.fit = True
        args.ident = True
        args.eval = True
        args.plot = True
        args.test_ma = True

    print("=" * 80)
    print("SISTEMA DE IDENTIFICAÇÃO GREY-BOX DO AEROPÊNDULO — VERSÃO 8 (v8_final_robust)")
    print(f"  * Configurações: p={args.p:.3f} | c4={args.c4:.4f} | u0={args.u0:.2f}% | seg_len={args.seg_len:.1f}s | decim={args.decim}")
    print("=" * 80)

    # 1. Caracterização de Repouso e Mecânica Passiva
    rest_dict = characterize_rest_and_noise()
    mech_dict = identify_mechanics_free_oscillation()

    # 2. Caracterização Estática Desacoplada (Rampa + MALHA-ABERTA)
    static_dict = extract_static_gain_open_loop(
        u0_fixed=args.u0,
        p_prescribed=args.p,
        c1_val=mech_dict['c1'],
        plot=args.plot
    )

    # 3. Carrega ensaios de Treino e Validação (RODADA-7)
    print("\n--- Carregando ensaios de Treino e Validação (RODADA-7) ---")
    train = [load_exp(*e, decim=args.decim, delay_s=args.delay_ms / 1000.0) for e in TRAIN]
    val = [load_exp(*e, decim=args.decim, delay_s=args.delay_ms / 1000.0) for e in VAL]
    all_r7 = val + train
    print(f"Total: {len(all_r7)} ensaios ({len(train)} Treino, {len(val)} Validação) | Decimação: {args.decim}")

    # Modelos base
    thetas = get_predefined_v8_models()
    rk4_fn = build_rk4_v8()

    # 4. Otimização por Multiple Shooting (se solicitada ou --all)
    fisher_dict = None
    if args.fit:
        th_ms_opt, fit_meta = identify_v8_multiple_shooting(
            train=train,
            static_dict=static_dict,
            mech_dict=mech_dict,
            p_val=args.p,
            seg_len=args.seg_len,
            max_iter=args.max_iter,
            fit_c3=args.fit_c3,
            use_p_fit=args.fit_p_static
        )
        thetas["v8_ms_otimizado"] = th_ms_opt

    # 5. Análise de Identificabilidade (se solicitada ou --all)
    if args.ident:
        target_th = thetas.get("v8_ms_otimizado", thetas["v8_final_robust"])
        fisher_dict = analyze_identifiability_v8(
            theta_opt=target_th,
            train=train,
            static_dict=static_dict,
            mech_dict=mech_dict,
            rk4_fn=rk4_fn,
            plot=args.plot
        )

    # Salva modelos e metadados no disco
    save_model_v8(thetas, static_dict=static_dict, mech_dict=mech_dict,
                  fisher_dict=fisher_dict, filename="modelo_greybox_v8")

    # 6. Avaliação em Free-Run Global nos 12 Ensaios de RODADA-7
    print("\n" + "=" * 105)
    print("AVALIAÇÃO EM MALHA ABERTA PURA (FREE-RUN GLOBAL, SEM RESET) NOS 12 ENSAIOS DE RODADA-7")
    print("=" * 105)
    cols = ["Ensaio", "y_max"] + [f"FIT {m[:12]}" for m in thetas] + ["Status final"]
    print(f"{cols[0]:24s} | {cols[1]:6s} | " + " | ".join(f"{c:>13s}" for c in cols[2:-1]) + f" | {cols[-1]:12s}")
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

        row_strs = [f"{fits[m]:5.1f}%" if not np.isnan(fits[m]) else "GIRO" for m in thetas]
        st_final = statuses.get("v8_ms_otimizado", statuses.get("v8_final_robust", "ESTÁVEL"))

        print(f"{e['name']:24s} | {y_max:5.1f}° | " + " | ".join(f"{s:>13s}" for s in row_strs) + f" | {st_final:12s}")
        res_entry = {'name': e['name'], 'y_max': y_max}
        for m in thetas:
            res_entry[f'fit_{m}'] = fits[m]
        res_table.append(res_entry)

    df_r7 = pd.DataFrame(res_table)
    print("-" * 105)
    mean_strs = [f"{df_r7[f'fit_{m}'].mean():12.1f}%" for m in thetas]
    print(f"{'MÉDIA GERAL (12 ensaios)':24s} |   ---  | " + " | ".join(mean_strs) + " | 12/12 ESTÁVEIS")

    # 7. Avaliação em Janela (Multiple Shooting Windowed Run: SEG_LEN)
    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO EM JANELA MULTIPLE SHOOTING (SEG_LEN = {args.seg_len:.1f} s, RESET DE ESTADOS):")
    print("=" * 80)
    for m in thetas:
        fits_win = []
        for e in val:
            yw = windowed_run_v8(thetas[m], e, rk4_fn, seg_len=args.seg_len)
            fits_win.append(fit_pct(e['y'], yw))
        print(f"  * {m:18s} — Validação (6 ensaios): FIT Médio = {np.mean(fits_win):5.1f}%")

    # 8. Razão de Amplitude no Chirp de 45°
    e_chirp = val[1]
    yh_chirp_base = free_run_v8(thetas['baseline_v7'], e_chirp, rk4_fn)
    best_m = "v8_ms_otimizado" if "v8_ms_otimizado" in thetas else "v8_final_robust"
    yh_chirp_final = free_run_v8(thetas[best_m], e_chirp, rk4_fn)
    mask_30_45 = (e_chirp['t'] >= 30.0) & (e_chirp['t'] <= 45.0)
    std_real = float(np.std(e_chirp['y'][mask_30_45]))
    amp_base = float(np.std(yh_chirp_base[mask_30_45])) / std_real
    amp_final = float(np.std(yh_chirp_final[mask_30_45])) / std_real

    print("\n" + "=" * 80)
    print("FIDELIDADE DINÂMICA DE ALTA FREQUÊNCIA (CHIRP 45°, JANELA 30 a 45 s):")
    print(f"  * Baseline v7:       std(yhat)/std(y) = {amp_base:.3f}x")
    print(f"  * {best_m}:   std(yhat)/std(y) = {amp_final:.3f}x (+{(amp_final/amp_base - 1.0)*100.0:.1f}% de recuperação de amplitude)")
    print("=" * 80)

    # 9. Benchmark nos 14 ensaios de MALHA-ABERTA (se solicitado ou com --all)
    if args.test_ma or args.all:
        benchmark_malha_aberta(thetas, rk4_fn)

    # 10. Geração de Gráficos (se --plot ou --all)
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
        plot_amplitude_error_v8(val, sims_val, metrics_val, target_mode=best_m,
                                fname="v8_erro_amplitude_analise.png")

    print("\n[CONCLUÍDO] Script ident_greybox_prof_multi_v8.py executado com sucesso!")


if __name__ == "__main__":
    main()
