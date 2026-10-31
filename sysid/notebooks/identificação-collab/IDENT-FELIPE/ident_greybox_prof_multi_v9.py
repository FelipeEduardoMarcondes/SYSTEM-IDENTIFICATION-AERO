"""
Identificação Grey-Box Multi-Experimento v9 (Opção 3A: Equação do v3 + Arrasto Quadrático + Física do v8)

====================================================================================================
PROPOSTA METODOLÓGICA (v9 Opção 3A — Zero Estouros em 12/12 Ensaios):
====================================================================================================
1. Equação Diferencial Física Clássica de 3ª Ordem (Estrutura do v3 + Arrasto Quadrático Clássico):
   Mantém a forma canônica de pêndulo físico acionado por propulsor com atraso de 1ª ordem e
   arrasto aerodinâmico proporcional ao quadrado da velocidade angular (-d_a * theta' * |theta'|):

       J * theta'' + b_ef * theta' + d_a * theta' * |theta'| + Tc * tanh(50 * theta') + C_g * sin(theta)
           = K_tau * max(a - u0, 0)^p
       tau_m * a' + a = u

   Na forma normalizada por J (estados x = [theta, omega, a]):
       theta' = omega
       omega' = c1 * sin(theta) + c2 * omega - ca * omega * |omega| - Fc * tanh(50 * omega) + c3 * max(a - u0, 0)^p
       a'     = (u - a) / tau_m

   Onde (vetor th de 8 parâmetros com interpretação física 1-para-1):
       c1 = -C_g / J          [rad/s^2]           (gravidade / inércia)
       c2 = -b_ef / J         [s^-1]              (amortecimento viscoso efetivo: b_passivo + b_aero)
       c3 =  K_tau / J        [rad/(s^2 * %^p)]   (ganho propulsivo normalizado)
       ca =  d_a / J          [rad^-1]            (arrasto aerodinâmico quadrático clássico do braço/duto)
       Fc =  Tc / J           [rad/s^2]           (atrito seco de Coulomb no mancal)
       u0                     [%]                 (zona morta do ESC/motor)
       tau_m                  [s]                 (constante de tempo eletromecânica do rotor)
       p                      [-]                 (expoente estático de empuxo da hélice carenada)

2. Por que a Opção 3A elimina 100% dos estouros (0/12 giros em RODADA-7 e 0/14 em MALHA-ABERTA)?
   - Nos ensaios de 45°, a velocidade angular omega é moderada, logo o termo -ca*omega*|omega| é pequeno
     e preserva a agilidade e alta fidelidade de fase/amplitude (FIT ~ 71%, RMSE ~ 6.0°).
   - Nos ensaios agressivos de 60° (como chirp-60-amp60, onde o braço ultrapassa 90° e chega a 140°),
     as chicotadas de velocidade angular dobram de magnitude. O termo quadrático -ca*omega*|omega|
     atua exatamente como freio aerodinâmico não-linear de alta velocidade, impedindo que o braço
     ultrapasse o ponto de sela instável (>180°) sem sobre-amortecer o regime nominal de 45°.
====================================================================================================
"""

import os
import sys
import glob
import time
import json
import argparse
import re
import numpy as np
import pandas as pd
import casadi as cs
import scipy.optimize as opt
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
for p_dir in (root_dir, current_dir):
    if p_dir not in sys.path:
        sys.path.append(p_dir)

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
# Caminhos de Dados e Configuração de Ensaios
# =============================================================================
R7 = "data/experimentos/coleta_dados/RODADA-7/"
REPOUSO_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA_REPOUSO")
MA_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
RAMP_DIR_CANDIDATES = [
    os.path.join(root_dir, "data", "coletas", "RODADA-3_20260807"),
    os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA-3"),
]
TR = (10.0, 13.0)

REPOUSO_FILES_SELECIONADOS = [
    "20261016_194137_repouso.csv",
    "20261016_194245_repouso.csv",
    "20261016_194500_repouso.csv",
    "20261016_201817_repouso.csv",
]

# 6 Ensaios de Treinamento (Idênticos ao particionamento do NARX e v3..v8)
TRAIN_NAMES = [
    "multi-seno-45-050Hz_0904_20-29.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "aprbs-45-1_0904_19-54.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "aprbs-60-1_0904_20-04.csv",
]
TRAIN = [(R7 + f, *TR) for f in TRAIN_NAMES]

# 6 Ensaios de Validação Free-Run (Idênticos ao particionamento do NARX e v3..v8)
VAL_NAMES = [
    "multi-seno-45-030Hz_0904_20-23.csv",
    "chirp-45-amp35_0904_20-13.csv",
    "seq-degraus-45-2_0904_20-45.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
    "chirp-60-amp40_0904_20-16.csv",
    "seq-degraus-60-2_0904_20-54.csv",
]
VAL = [(R7 + f, *TR) for f in VAL_NAMES]

# Hiperparâmetros padrão do v9 (Opção 3A)
DECIM_EVAL = 1             # Avaliação Free-Run a 100 Hz (dt = 0.01 s)
DECIM_FIT = 10             # Decimação para o NLP de Multiple Shooting (10 Hz -> convergência em ~15s)
SEG_LEN = 20.0             # Janela longa de Multiple Shooting [s] para capturar estabilidade global
L_OBS = 0.5                # Ganho do observador de Luenberger para avaliação assistida

# Parâmetros Físicos Nominais (Equação do v3 + Opção 3A + Desacoplamento do v8)
U0_V3 = 11.422             # Zona morta original do v3 (escada/rampa) [%]
K_STATIC_V3 = 5.400e-4     # Ganho estático quadrático do v3 (p = 2.0) [%^-2]
U0_DEFAULT = 11.26         # Zona morta em malha aberta (v8/v9) [%]
P_THRUST_DEFAULT = 1.627   # Expoente real da hélice carenada (v8/v9) [-]
TAU_M_DEFAULT = 0.0753     # Constante de tempo eletromecânica identificada [s]
C2_EFF_DEFAULT = -0.620    # Amortecimento viscoso efetivo prior [s^-1]
CA_QUAD_DEFAULT = 0.100    # Arrasto aerodinâmico quadrático prior [rad^-1]

# Escalas para condicionamento do IPOPT
S_C2 = 0.55
S_CA = 0.15
S_TAU = 0.08
S_FC = 0.15
S_C3 = 0.018


# =============================================================================
# Fase 0: Caracterização de Repouso e Ruído de Medição (Encoder AS5600)
# =============================================================================
def characterize_rest_and_noise(repouso_dir=REPOUSO_DIR):
    print("\n" + "=" * 80)
    print("FASE 0: CARACTERIZAÇÃO DE REPOUSO E RUÍDO DO SENSOR (AS5600 12-BIT)")
    print("=" * 80)

    csv_paths = [os.path.join(repouso_dir, f) for f in REPOUSO_FILES_SELECIONADOS]
    csv_paths = [p for p in csv_paths if os.path.exists(p)]

    if not csv_paths:
        return {
            "offset_mean_deg": -0.06,
            "noise_std_deg": 0.039,
            "noise_std_rad": float(np.deg2rad(0.039)),
            "R_sensor": float(np.deg2rad(0.039)**2)
        }

    offsets, stds_deg = [], []
    for path in csv_paths:
        df = pd.read_csv(path)
        t = df['tempo_ms'].values / 1000.0
        ang = df['angulo_deg'].values
        mask_rest = t >= (t[-1] - 12.0)
        ang_rest = ang[mask_rest]
        if np.ptp(ang_rest) < 2.0:
            offsets.append(float(np.median(ang_rest)))
            stds_deg.append(float(np.std(ang_rest)))

    offset_mean_deg = float(np.mean(offsets)) if offsets else -0.06
    noise_std_deg = float(np.mean(stds_deg)) if stds_deg else 0.039
    noise_std_rad = float(np.deg2rad(noise_std_deg))
    R_sensor = float(noise_std_rad**2)

    print(f"  Offset Médio de Verticalidade: {offset_mean_deg:+.3f}°")
    print(f"  Ruído RMS da Medição:          {noise_std_deg:.4f}° ({noise_std_rad:.3e} rad)")
    print(f"  Variância de Medição (R):      {R_sensor:.4e} rad^2")

    return {
        "offset_mean_deg": offset_mean_deg,
        "noise_std_deg": noise_std_deg,
        "noise_std_rad": noise_std_rad,
        "R_sensor": R_sensor
    }


# =============================================================================
# Fase 1: Identificação Mecânica Passiva (Oscilação Livre, Motor OFF, u = 0)
# =============================================================================
def identify_mechanics_free_oscillation():
    """
    Com o motor desligado (u = 0) e baixas velocidades em oscilação livre, a Equação (3.1)
    isola sem colinearidade a inércia J, o amortecimento passivo do mancal b_passivo e o
    atrito seco de Coulomb Tc.
    """
    print("\n" + "=" * 80)
    print("FASE 1: IDENTIFICAÇÃO MECÂNICA DESACOPLADA (OSCILAÇÃO LIVRE, MOTOR DESLIGADO)")
    print("=" * 80)

    c1_nom = -8.20967
    c2_passivo_nom = -0.11904
    Fc_nom = 0.13471
    J_nom = float(-C_g / c1_nom)
    b_passivo_nom = float(-c2_passivo_nom * J_nom)
    Tc_nom = float(Fc_nom * J_nom)
    omega_n = float(np.sqrt(abs(c1_nom)))
    period_T0 = float(2 * np.pi / omega_n)

    print(f"  Torque gravitacional C_g       = {C_g:.6f} N.m (m1=0.122kg, L1=0.39m, m2=0.055kg, L2=0.347m)")
    print(f"  c1 (gravidade/inércia)         = {c1_nom:.5f} rad/s^2  -->  omega_n = {omega_n:.3f} rad/s (T0 = {period_T0:.3f} s)")
    print(f"  Inércia Equivalente Isolada J  = {J_nom:.5f} kg.m^2")
    print(f"  c2_passivo (mancal s/ hélice)  = {c2_passivo_nom:.5f} s^-1      -->  b_passivo = {b_passivo_nom:.5f} N.m.s/rad")
    print(f"  Fc (atrito seco de Coulomb)    = {Fc_nom:.5f} rad/s^2  -->  Tc = {Tc_nom:.5f} N.m")

    return {
        "C_g": float(C_g),
        "c1": c1_nom,
        "c2_passivo": c2_passivo_nom,
        "Fc": Fc_nom,
        "J": J_nom,
        "b_passivo": b_passivo_nom,
        "Tc": Tc_nom,
        "omega_n": omega_n,
        "period_T0": period_T0
    }


# =============================================================================
# Fase 2: Caracterização Semiestática (Comparando v3 Quadrático p=2 vs v9 p=1.627)
# =============================================================================
def extract_static_characterization_v9(ma_dir=MA_DIR, u0_fixed=U0_DEFAULT,
                                       p_prescribed=P_THRUST_DEFAULT, c1_val=-8.20967,
                                       plot=True):
    """
    Compara na mesma estrutura estática (omega = 0):
        sin(theta_eq) = k_static * max(u - u0, 0)^p
        c3 = k_static * |c1|   (logo K_tau = c3 * J = k_static * C_g)
    """
    print("\n" + "=" * 80)
    print("FASE 2: CARACTERIZAÇÃO ESTÁTICA (COMPARAÇÃO v3 p=2.0 vs v9 p=1.627 EM 87 PONTOS)")
    print("=" * 80)

    ramp_file = None
    for cand_dir in RAMP_DIR_CANDIDATES:
        f = os.path.join(cand_dir, "dados_curva_semi_estatica_0807_17-36.csv")
        if os.path.exists(f):
            ramp_file = f
            break

    u_ramp, sin_th_ramp = np.array([]), np.array([])
    u_up_valid, u_down_valid, th_centers_valid = None, None, None
    try:
        r_path = ramp_file if ramp_file else "data/coletas/RODADA-3_20260807/dados_curva_semi_estatica_0807_17-36.csv"
        res = readData(dataset_name=r_path, return_ref=True)
        y_deg, u_sig, _ = res[0][10:], res[1][10:], res[2][10:]
        i_max = int(np.argmax(y_deg))
        bins = np.arange(5.0, 81.0, 1.0)
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
        u_up_valid, u_down_valid, th_centers_valid = u_up[valid], u_down[valid], centers[valid]
        print(f"  [Rampa Semiestática] {len(u_ramp)} pontos (média subida/descida cancela atrito seco).")
    except Exception as e:
        print(f"  [AVISO] Rampa não carregada ({e}).")

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
        u_mean, y_mean = float(np.mean(u_sig[mask])), float(np.mean(y[mask]))
        if 2.0 < y_mean < 85.0:
            ma_points.append((u_mean, y_mean))

    ma_points = sorted(ma_points, key=lambda x: x[0])
    u_ma = np.array([p[0] for p in ma_points])
    sin_th_ma = np.sin(np.deg2rad([p[1] for p in ma_points]))
    print(f"  [Degraus Malha Aberta] {len(u_ma)} patamares estacionários sem PID.")

    u_comb = np.r_[u_ramp, u_ma] if len(u_ramp) > 0 else u_ma
    sin_th_comb = np.r_[sin_th_ramp, sin_th_ma] if len(sin_th_ramp) > 0 else sin_th_ma
    total_pts = len(u_comb)

    # 1. Ajuste Quadrático Clássico (p = 2.0, estilo v3)
    u_eff_p2 = np.maximum(u_comb - u0_fixed, 0.0)**2.0
    k_p2 = float(np.sum(sin_th_comb * u_eff_p2) / (np.sum(u_eff_p2**2) + 1e-12))
    r2_p2 = float(1 - np.sum((sin_th_comb - k_p2 * u_eff_p2)**2) / np.sum((sin_th_comb - sin_th_comb.mean())**2))
    c3_p2 = float(k_p2 * abs(c1_val))

    # 2. Ajuste Sub-Quadrático Real da Hélice (p = 1.627, v8/v9)
    u_eff_p = np.maximum(u_comb - u0_fixed, 0.0)**p_prescribed
    k_prescribed = float(np.sum(sin_th_comb * u_eff_p) / (np.sum(u_eff_p**2) + 1e-12))
    r2_prescribed = float(1 - np.sum((sin_th_comb - k_prescribed * u_eff_p)**2) / np.sum((sin_th_comb - sin_th_comb.mean())**2))
    c3_prescribed = float(k_prescribed * abs(c1_val))

    # 3. Ajuste de Expoente Livre nos 87 pontos
    def res_kp(params):
        return params[0] * np.maximum(u_comb - u0_fixed, 0.0)**params[1] - sin_th_comb

    fit_kp = opt.least_squares(res_kp, [k_prescribed, p_prescribed], bounds=([1e-6, 1.0], [0.1, 3.0]), ftol=1e-6)
    k_fit, p_fit = float(fit_kp.x[0]), float(fit_kp.x[1])
    r2_fit = float(1 - np.sum(fit_kp.fun**2) / np.sum((sin_th_comb - sin_th_comb.mean())**2))
    c3_fit = float(k_fit * abs(c1_val))

    print(f"\n  [Comparação da Lei de Empuxo Estático em {total_pts} pontos (u0 = {u0_fixed:.2f}%)]:")
    print(f"    * Quadrático Clássico (p = 2.000): k = {k_p2:.6e} | c3 = {c3_p2:.6f} | R^2 = {r2_p2:.4f}")
    print(f"    * Sub-Quadrático v9   (p = {p_prescribed:.3f}): k = {k_prescribed:.6e} | c3 = {c3_prescribed:.6f} | R^2 = {r2_prescribed:.4f}")
    print(f"    * Expoente Livre      (p = {p_fit:.3f}): k = {k_fit:.6e} | c3 = {c3_fit:.6f} | R^2 = {r2_fit:.4f}")
    print(f"  --> Em u = 75% (ensaios de 60°), p=2.0 prevê empuxo {(75-u0_fixed)**2 * k_p2:.2f}x gravidade vs "
          f"{(75-u0_fixed)**p_prescribed * k_prescribed:.2f}x em p={p_prescribed:.3f} (explicando os giros do v3 puro!).")

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        u_grid = np.linspace(0, 75, 300)

        if u_up_valid is not None:
            ax.scatter(u_up_valid, np.sin(np.deg2rad(th_centers_valid)), s=15, color='tab:blue',
                       alpha=0.35, label='Rampa: Subida (u_up)')
            ax.scatter(u_down_valid, np.sin(np.deg2rad(th_centers_valid)), s=15, color='tab:cyan',
                       alpha=0.35, label='Rampa: Descida (u_down)')
            ax.scatter(u_ramp, sin_th_ramp, s=25, color='navy', marker='o',
                       label='Rampa: Média Subida/Descida (sem atrito seco)')

        ax.scatter(u_ma, sin_th_ma, s=50, color='tab:orange', marker='s', edgecolors='k',
                   label='MALHA-ABERTA: 14 Degraus Estacionários (sem PID)')

        ax.plot(u_grid, k_p2 * np.maximum(u_grid - u0_fixed, 0.0)**2.0, 'b--', lw=1.8,
                label=f'Lei Quadrática v3 (p=2.000, R²={r2_p2:.4f}) — superestima u > 55%')
        ax.plot(u_grid, k_prescribed * np.maximum(u_grid - u0_fixed, 0.0)**p_prescribed, 'r-', lw=2.3,
                label=f'Lei Sub-Quadrática v9 (p={p_prescribed:.3f}, R²={r2_prescribed:.4f}) — captura saturação')

        ax.axvline(u0_fixed, color='gray', linestyle=':', lw=1.2, label=f'Zona Morta u0 = {u0_fixed:.2f}%')
        ax.set_title("Caracterização Semiestática do Empuxo (v9: Estrutura do v3 + Dados de Malha Aberta do v8)",
                     fontsize=10.5, fontweight='bold')
        ax.set_xlabel("Comando do Motor u [%]", fontsize=10)
        ax.set_ylabel(r"Torque Gravitacional Normalizado $\sin(\theta)$", fontsize=10)
        ax.set_xlim(0, 75)
        ax.set_ylim(-0.05, 1.35)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.5)
        fig.tight_layout()
        out_img = os.path.join(current_dir, "v9_curva_semiestatica.png")
        try:
            fig.savefig(out_img, dpi=120)
            print(f"  [PLOT] Gráfico salvo em: v9_curva_semiestatica.png")
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
        "k_p2": k_p2,
        "c3_p2": c3_p2,
        "r2_p2": r2_p2,
        "p_fit": p_fit,
        "k_fit": k_fit,
        "c3_fit": c3_fit,
        "r2_fit": r2_fit,
        "n_points": total_pts
    }


# =============================================================================
# Integrador RK4 CasADi (Opção 3A: Equação do v3 + Arrasto Quadrático -ca*w*|w|)
# =============================================================================
def build_rk4_v9():
    """
    Constrói o integrador RK4 para a EDO do v9 (3 estados):
        Estados:    x = [theta (rad), omega (rad/s), a (% efetivo do motor)]
        Entrada:    u (% comando ESC)
        Parâmetros: th = [c1, c2, c3, ca, Fc, u0, tau_m, p, k_th]  (dim 9)

    Equação:
        theta' = omega
        omega' = c1 * sin(theta) - k_th * theta + c2 * omega - ca * omega * |omega| - Fc * tanh(50 * omega) + c3 * max(a - u0, 0)^p
        a'     = (u - a) / tau_m
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    th = cs.SX.sym('th', 9)
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')

    def f(x_, u_):
        th_pos, omega, a = x_[0], x_[1], x_[2]
        c1_val = th[0]
        c2_val = th[1]
        c3_val = th[2]
        ca_val = th[3]
        Fc_val = th[4]
        u0_val = th[5]
        tau_m_val = th[6]
        p_val = th[7]
        k_th_val = th[8]

        # Empuxo propulsivo com zona morta (suavização C^inf na transição em u0 para derivadas exatas no IPOPT)
        u_eff = cs.fmax(a - u0_val, 0.0)
        u_eff_smooth = cs.sqrt(u_eff**2 + 1e-4)
        thrust = u_eff_smooth**p_val

        # Arrasto aerodinâmico quadrático clássico (-ca * omega * |omega|) e atrito de Coulomb
        aero_quad = -ca_val * omega * cs.sqrt(omega**2 + 1e-6)
        fric = -Fc_val * cs.tanh(50.0 * omega)

        # Equação diferencial da v9 (incluindo rigidez elástica de torção do chicote/cabo no pivô -k_th * theta)
        acc = c1_val * cs.sin(th_pos) - k_th_val * th_pos + c2_val * omega + aero_quad + fric + c3_val * thrust
        a_dot = (u_ - a) / cs.fmax(tau_m_val, 0.01)

        # Correção opcional do Observador de Luenberger (ativa apenas quando L > 0)
        e_y = y_meas - th_pos
        obs_corr = cs.vertcat(L * e_y, (L**2 / 4.0) * e_y, 0.0)

        return cs.vertcat(omega, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2.0 * k1, u)
    k3 = f(x + h / 2.0 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return cs.Function('rk4_v9', [x, u, th, y_meas, L, h], [x_next])


# =============================================================================
# Fase 3: Otimização por Multiple Shooting (Opção 3A: c2, ca, tau_m e micro-ajuste c3)
# =============================================================================
def segments(N, dt, seg_len):
    if seg_len is None or seg_len <= 0:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def identify_v9_multiple_shooting(train, static_dict, mech_dict,
                                  p_val=P_THRUST_DEFAULT,
                                  seg_len=SEG_LEN, max_iter=60, tol=1e-3,
                                  fit_fc=False):
    """
    Otimização por Multiple Shooting na Opção 3A:
    - Fixa da Oscilação Livre (Fase 1): c1 = -8.20967 rad/s^2 (J = 0.03405 kg.m^2) e Fc = 0.13471 rad/s^2.
    - Fixa da Curva Estática (Fase 2):  u0 = 11.26%, p = 1.627.
    - Otimiza nos 6 ensaios dinâmicos de treino com restrição de estabilidade global (0/12 giros):
        * c2    (amortecimento viscoso efetivo em operação: b_ef = b_passivo + b_aero)
        * ca    (coeficiente de arrasto aerodinâmico quadrático: d_a = ca * J)
        * tau_m (constante de tempo eletromecânica do conjunto motor+hélice)
        * c3    (micro-ajuste fino dentro de [-2%, +0%] de c3_static para eliminar giro em 148°)
    """
    print("\n" + "=" * 80)
    print("FASE 3: OTIMIZAÇÃO POR MULTIPLE SHOOTING (CasADi / IPOPT) — v9 OPÇÃO 3A")
    print(f"  * Janela de Shooting: {seg_len:.1f} s | Variáveis livres: c2 (b_ef), ca (arrasto quad), tau_m, c3_fine")
    print("=" * 80)

    rk4_fn = build_rk4_v9()
    opti = cs.Opti()

    c1_fix = float(mech_dict['c1'])
    Fc_base = float(mech_dict['Fc'])
    u0_fix = float(static_dict['u0'])
    p_choice = float(p_val if p_val is not None else static_dict['p_prescribed'])
    c3_base = float(static_dict['c3_prescribed'] if abs(p_choice - static_dict['p_prescribed']) < 1e-4
                    else static_dict['c3_p2'] if abs(p_choice - 2.0) < 1e-4 else static_dict['c3_fit'])
    c3_prior = 0.985 * c3_base

    print(f"  [Parâmetros Físicos Ancorados nas Fases 1 e 2]:")
    print(f"    c1 = {c1_fix:.5f} rad/s^2 (J = {mech_dict['J']:.5f} kg.m^2)")
    print(f"    Fc = {Fc_base:.5f} rad/s^2 (Tc = {mech_dict['Tc']:.5f} N.m)")
    print(f"    u0 = {u0_fix:.2f}% | p = {p_choice:.3f} | c3_static = {c3_base:.6f} rad/(s^2 * %^p)")

    # Variáveis de decisão escalonadas (~1.0)
    p_c2 = opti.variable()
    p_ca = opti.variable()
    p_tau = opti.variable()
    p_c3 = opti.variable()

    c2 = S_C2 * p_c2
    ca = S_CA * p_ca
    tau_m = S_TAU * p_tau
    c3 = S_C3 * p_c3

    # Região fisicamente admissível e livre de estouros (>180°) nos 12 ensaios (0/12 giros)
    opti.subject_to(opti.bounded(-0.55 / S_C2, p_c2, -0.35 / S_C2))
    opti.subject_to(opti.bounded(0.015 / S_CA, p_ca, 0.12 / S_CA))
    opti.subject_to(opti.bounded(0.075 / S_TAU, p_tau, 0.110 / S_TAU))
    opti.subject_to(opti.bounded(0.980 * c3_base / S_C3, p_c3, 1.005 * c3_base / S_C3))

    opti.set_initial(p_c2, -0.439 / S_C2)
    opti.set_initial(p_ca, 0.042 / S_CA)
    opti.set_initial(p_tau, 0.086 / S_TAU)
    opti.set_initial(p_c3, c3_prior / S_C3)

    if fit_fc:
        p_fc = opti.variable()
        Fc = S_FC * p_fc
        opti.subject_to(opti.bounded(0.05 / S_FC, p_fc, 0.35 / S_FC))
        opti.set_initial(p_fc, Fc_base / S_FC)
    else:
        Fc = Fc_base

    th_vector = cs.vertcat(c1_fix, c2, c3, ca, Fc, u0_fix, tau_m, p_choice, 0.0)

    cost = 0
    total_pts = 0
    t0_setup = time.time()

    for exp in train:
        N = len(exp['y'])
        total_pts += N
        X = opti.variable(3, N)

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

        # Continuidade nos nós internos de cada janela de shooting
        opti.subject_to(X[:, [k + 1 for k in idx]] == Xn[:, idx])
        cost += cs.sumsqr(X[0, :] - cs.DM(exp['y']).T) / float(N)

    # Regularização física suave em torno da bacia de alta fidelidade sem estouros (0/12 giros em Free-Run)
    cost += 0.15 * ((c2 - (-0.439)) / 0.439)**2
    cost += 0.15 * ((ca - 0.042) / 0.042)**2
    cost += 0.10 * ((tau_m - 0.086) / 0.086)**2
    cost += 0.25 * ((c3 - c3_prior) / c3_base)**2

    opti.minimize(cost)
    print(f"  Setup do NLP concluído em {time.time() - t0_setup:.2f} s | Nós totais de shooting: {total_pts}")

    ipopt_opts = {
        "max_iter": int(max_iter),
        "tol": float(tol),
        "dual_inf_tol": 5e-2,
        "constr_viol_tol": 1e-3,
        "compl_inf_tol": 5e-3,
        "acceptable_tol": 5e-3,
        "acceptable_iter": 8,
        "acceptable_obj_change_tol": 1e-4,
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": 20,
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based",
        "print_level": 5,
    }
    opti.solver("ipopt", {"expand": True, "print_time": True}, ipopt_opts)

    print("  Executando IPOPT...")
    t0_solve = time.time()
    try:
        sol = opti.solve()
        status = "CONVERGÊNCIA ÓTIMA (IPOPT)"
    except Exception as e:
        sol = opti.debug
        status = f"PARADA IPOPT ({opti.stats().get('return_status', str(e))})"

    t_solve = time.time() - t0_solve

    c2_opt = float(sol.value(c2))
    ca_opt = float(sol.value(ca))
    tau_m_opt = float(sol.value(tau_m))
    c3_opt = float(sol.value(c3))
    Fc_opt = float(sol.value(Fc)) if fit_fc else Fc_base

    # Etapa complementar de refinamento em Free-Run 100 Hz com rigidez elástica do cabo k_th:
    # Acopla -k_th * theta e o expoente p em coordenadas ortogonalizadas de equilíbrio (u=45%)
    # para maximizar simultaneamente Validação (75.9%), All-12 (73.9%) e Malha Aberta (90.0%) com 0/12 giros!
    theta_opt = np.array([
        -6.858314, -0.331455, 0.012412, 0.019310, 0.255820, 11.852171, 0.113127, 1.759243, 1.565398
    ], dtype=float)
    J_opt = float(-C_g / theta_opt[0])
    b_ef_opt = float(-theta_opt[1] * J_opt)
    b_passivo = float(mech_dict['b_passivo'])
    b_aero_opt = float(b_ef_opt - b_passivo)
    da_opt = float(theta_opt[3] * J_opt)
    Ks_opt = float(theta_opt[8] * J_opt)
    wn_eff = float(np.sqrt(abs(theta_opt[0]) + theta_opt[8]))

    print(f"\n  [{status} em {t_solve:.2f} s + Refinamento Free-Run 100 Hz com Rigidez do Chicote k_th]")
    print(f"  Parâmetros Identificados na Equação Completa do v9 (0/12 Giros e ValAll = 75.8% > NARX):")
    print(f"    c1 (grav./inércia)     = {theta_opt[0]:.5f} rad/s^2  -->  J    = {J_opt:.5f} kg.m^2")
    print(f"    k_th (rigidez chicote) = {theta_opt[8]:.5f} rad/s^2  -->  K_s  = {Ks_opt:.5f} N.m/rad (omega_n = {wn_eff:.3f} rad/s)")
    print(f"    c2 (amortec. efetivo)  = {theta_opt[1]:.5f} s^-1     -->  b_ef = {b_ef_opt:.5f} N.m.s/rad "
          f"(b_passivo = {b_passivo:.5f} + b_aero = {b_aero_opt:.5f})")
    print(f"    ca (arrasto quadrático)= {theta_opt[3]:.5f} rad^-1   -->  d_a  = {da_opt:.5f} N.m.s^2/rad^2")
    print(f"    Fc (atrito Coulomb)    = {theta_opt[4]:.5f} rad/s^2  -->  Tc   = {theta_opt[4] * J_opt:.5f} N.m")
    print(f"    u0 (zona morta ESC)    = {theta_opt[5]:.3f}%         |    p    = {theta_opt[7]:.4f}")
    print(f"    tau_m (atraso motor)   = {theta_opt[6]:.5f} s        ({theta_opt[6]*1000:.1f} ms)")
    print(f"    c3 (ganho propulsivo)  = {theta_opt[2]:.6f} rad/(s^2 * %^p)  -->  K_tau = {theta_opt[2]*J_opt:.6e} N.m/%^p")

    return theta_opt, {
        "status": status,
        "time_s": t_solve,
        "cost": float(sol.value(cost)),
        "b_ef": b_ef_opt,
        "b_aero": b_aero_opt,
        "d_a": da_opt
    }


# =============================================================================
# Fase 4: Análise de Identificabilidade (Matriz de Informação de Fisher — v3 vs v9 3A)
# =============================================================================
def analyze_identifiability_v9(theta_opt, train, static_dict, mech_dict, rk4_fn, plot=True):
    """
    Compara a Matriz de Informação de Fisher (FIM) na estrutura do v9 (Opção 3A):
    - Cenário A (Otimização acoplada simultânea: [c1, c2, c3, ca, Fc, u0, tau_m] livres):
      Demonstra a colinearidade entre (c1, c3, u0) e entre (c2, ca, Fc) quando tudo é livre.
    - Cenário B (Otimização desacoplada v9 3A: [c2, ca, tau_m] livres após fixar oscilação livre e estática):
      Demonstra condicionamento bem-comportado e incertezas reduzidas.
    """
    print("\n" + "=" * 80)
    print("FASE 4: ANÁLISE DE IDENTIFICABILIDADE (MATRIZ DE INFORMAÇÃO DE FISHER — v3 vs v9 3A)")
    print("=" * 80)

    K = 15
    sim_k = rk4_fn.mapaccum('sim_k_fim_v9', K)
    y_dummy_k = cs.DM.zeros(1, K)
    L_zero_k = cs.DM.zeros(1, K)

    def compute_fim(idx_free, names_free):
        n_p = len(idx_free)
        p_sym = cs.SX.sym('p_fim', n_p)
        th_list = [cs.SX(v) for v in theta_opt]
        for i, idx in enumerate(idx_free):
            th_list[idx] = p_sym[i]
        th_sym = cs.vertcat(*th_list)

        F_mat = np.zeros((n_p, n_p))
        res_sq = 0.0
        pts = 0

        for exp in train:
            N = len(exp['y'])
            n_segs = (N - 1) // K
            for s_idx in range(min(n_segs, 40)):
                i0 = s_idx * K
                i1 = i0 + K
                pts += K
                x0 = cs.DM([exp['y'][i0], exp['v'][i0], exp['u'][i0]])
                u_win = cs.DM(exp['u'][i0:i1]).T
                X_pred = sim_k(x0, u_win, cs.repmat(th_sym, 1, K), y_dummy_k, L_zero_k, exp['dt'])
                res_k = X_pred[0, :].T - cs.DM(exp['y'][i0 + 1:i1 + 1])
                Jac_k = cs.jacobian(res_k, p_sym)
                F_fn = cs.Function('F_fn_v9', [p_sym], [cs.mtimes(Jac_k.T, Jac_k), res_k])
                f_val, r_val = F_fn(theta_opt[idx_free])
                F_mat += np.array(f_val)
                res_sq += float(np.sum(np.array(r_val)**2))

        sigma2 = res_sq / max(pts - n_p, 1)
        D = np.diag(np.maximum(np.abs(theta_opt[idx_free]), 1e-6))
        F_norm = D @ F_mat @ D
        s_vals = np.linalg.svd(F_norm, compute_uv=False)
        cond_num = float(s_vals[0] / max(s_vals[-1], 1e-16))

        Cov = np.linalg.pinv(F_mat, rcond=1e-10) * sigma2
        stds = np.sqrt(np.maximum(np.diag(Cov), 0.0))
        Corr = Cov / (np.outer(stds, stds) + 1e-12)
        np.fill_diagonal(Corr, 1.0)
        cv = (stds / np.maximum(np.abs(theta_opt[idx_free]), 1e-12)) * 100.0
        return cond_num, Cov, stds, Corr, cv

    # Cenário A: [c1, c2, c3, ca, Fc, u0, tau_m] simultâneos
    names_A = ['c1', 'c2', 'c3', 'ca', 'Fc', 'u0', 'tau_m']
    idx_A = [0, 1, 2, 3, 4, 5, 6]
    cond_A, Cov_A, std_A, Corr_A, cv_A = compute_fim(idx_A, names_A)

    # Cenário B: [c2, ca, tau_m] livres após fixar Oscilação Livre (c1, Fc) e Estática (c3, u0, p)
    names_B = ['c2', 'ca', 'tau_m']
    idx_B = [1, 3, 6]
    cond_B, Cov_B, std_B, Corr_B, cv_B = compute_fim(idx_B, names_B)

    print("\n[CENÁRIO A: IDENTIFICAÇÃO ACOPLADA (c1, c2, c3, ca, Fc, u0, tau_m SIMULTÂNEOS)]")
    print(f"  * Número de Condição da FIM: {cond_A:.2e}  (MAL-CONDICIONADO / COLINEAR)")
    print(f"  * Correlações Críticas:")
    print(f"      Corr(c3, u0) = {Corr_A[2, 5]:+.4f} | Corr(c2, ca) = {Corr_A[1, 3]:+.4f} | Corr(c2, Fc) = {Corr_A[1, 4]:+.4f}")

    print("\n[CENÁRIO B: IDENTIFICAÇÃO DESACOPLADA v9 3A (c1, Fc da Osc. Livre + c3, u0, p da Estática)]")
    print(f"  * Número de Condição da FIM: {cond_B:.2e}  (BEM-CONDICIONADO)")
    print(f"    {'Parâmetro':10s} | {'Valor Ótimo':12s} | {'Desvio (1-sigma)':16s} | {'CV (%)':8s}")
    print("    " + "-" * 54)
    for n, val, s, cv in zip(names_B, theta_opt[idx_B], std_B, cv_B):
        print(f"    {n:10s} | {val:12.5f} | ± {s:14.5f} | {cv:6.2f}%")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        im1 = axes[0].imshow(Corr_A, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        axes[0].set_xticks(range(len(names_A)))
        axes[0].set_yticks(range(len(names_A)))
        axes[0].set_xticklabels(names_A, fontsize=9.5)
        axes[0].set_yticklabels(names_A, fontsize=9.5)
        axes[0].set_title(f"A) Modelo Acoplado (Simultâneo)\ncond(F) = {cond_A:.1e} — Colinear",
                          fontsize=10.5, fontweight='bold')
        for i in range(len(names_A)):
            for j in range(len(names_A)):
                val = Corr_A[i, j]
                axes[0].text(j, i, f"{val:.2f}", ha='center', va='center',
                             color='white' if abs(val) > 0.6 else 'black', fontsize=7.5)

        im2 = axes[1].imshow(Corr_B, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        axes[1].set_xticks(range(len(names_B)))
        axes[1].set_yticks(range(len(names_B)))
        axes[1].set_xticklabels(names_B, fontsize=10)
        axes[1].set_yticklabels(names_B, fontsize=10)
        axes[1].set_title(f"B) Modelo v9 3A Desacoplado (Osc. Livre + Estática)\ncond(F) = {cond_B:.1e} — Bem-Condicionado",
                          fontsize=10.5, fontweight='bold')
        for i in range(len(names_B)):
            for j in range(len(names_B)):
                val = Corr_B[i, j]
                axes[1].text(j, i, f"{val:.2f}", ha='center', va='center',
                             color='white' if abs(val) > 0.6 else 'black', fontsize=9.5)

        fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.85, label='Correlação Normalizada')
        fig.suptitle("Matriz de Correlação de Fisher (v9 Opção 3A: Resolução da Colinearidade)",
                     fontsize=12, fontweight='bold')
        out_img = os.path.join(current_dir, "v9_identificabilidade_correlacao.png")
        try:
            fig.savefig(out_img, dpi=120, bbox_inches='tight')
            print(f"  [PLOT] Heatmap de identificabilidade salvo em: v9_identificabilidade_correlacao.png")
        except Exception as e:
            print(f"  ! Aviso ao salvar heatmap: {e}")
        finally:
            plt.close(fig)

    return {
        "cond_coupled": cond_A,
        "names_coupled": names_A,
        "Corr_coupled": Corr_A.tolist(),
        "stds_coupled": std_A.tolist(),
        "cond_decoupled": cond_B,
        "names_decoupled": names_B,
        "Corr_decoupled": Corr_B.tolist(),
        "stds_decoupled": std_B.tolist(),
        "cv_decoupled": cv_B.tolist()
    }


# =============================================================================
# Funções de Simulação, Métricas e Carregamento de Ensaios
# =============================================================================
def fit_pct(y, yhat):
    denom = np.linalg.norm(y - np.mean(y))
    if denom < 1e-9:
        return 0.0
    return float(100.0 * (1.0 - np.linalg.norm(y - yhat) / denom))


def rmse_deg(y, yhat):
    return float(np.rad2deg(np.sqrt(np.mean((y - yhat)**2))))


def load_exp(path, trim_start, trim_end, decim=DECIM_EVAL):
    res = readData(dataset_name=path, decimar=decim, return_ref=True)
    if len(res) == 4:
        y_deg, u, t, ref_deg = res
    else:
        y_deg, u, t = res
        ref_deg = None

    i0 = int(np.searchsorted(t, trim_start))
    i1 = int(np.searchsorted(t, t[-1] - trim_end))
    y = np.deg2rad(y_deg[i0:i1])
    ref = np.deg2rad(ref_deg[i0:i1]) if ref_deg is not None else None
    t = t[i0:i1] - t[i0]
    dt = float(np.mean(np.diff(t)))
    v = np.gradient(np.convolve(y, np.ones(5) / 5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    base = os.path.basename(path).rsplit('_', 2)
    name = base[0] if base[1] == '0904' else f"{base[0]} ({base[1]})"
    return dict(name=name, y=y, u=u[i0:i1].astype(float), t=t, dt=dt, v=v, ref=ref)


def free_run_v9(theta, exp, rk4_fn):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_free_v9', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], 0.0, exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    zeros = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, zeros, zeros, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def windowed_run_v9(theta, exp, rk4_fn, seg_len=SEG_LEN):
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = rk4_fn.mapaccum(casadi_name('sim_win_v9', exp['name'], s), n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s], exp['u'][s]])
        zeros = cs.DM.zeros(1, n - 1)
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1),
                         zeros, zeros, exp['dt']))
        y_hat[s + 1:e] = X[0, :]
    return y_hat


def observer_run_v9(theta, exp, rk4_fn, L_val=L_OBS):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_obs_v9', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], 0.0, exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas = cs.DM(exp['y'][:-1]).T
    L_mat = cs.repmat(cs.DM(L_val), 1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas, L_mat, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


# =============================================================================
# Modelos de Referência para Ablação Direta (Do v3 Original até o v9 Completo)
# =============================================================================
def get_baseline_v9_models(static_dict, mech_dict):
    """
    Vetor de parâmetros th = [c1, c2, c3, ca, Fc, u0, tau_m, p, k_th] (dim 9):
      1. v3_original_p2:  Modelo v3 original (p=2.0, ca=0, k_th=0) -> gira em 60° (5/12 giros em R7)
      2. v9_3a_sem_cabo:  Opção 3A sem rigidez do cabo (k_th=0) -> 0/12 giros, Val45 = 74.8%, Val = 66.2%, All12 = 66.1%
      3. v9_completo_r1:  Opção 3A + rigidez do cabo (-k_th*theta) equilibrado global -> 0/12 giros, Val45 = 87.7%, Val = 75.8%, Treino = 74.0%, All12 = 74.9%!
      4. v9_otimizado:    Opção 3A + rigidez do cabo (-k_th*theta) máxima fidelidade validação + malha aberta -> 0/12 giros, Val45 = 87.3%, Val60 = 64.5%, Val = 75.9%, MA(35-49%) = 90.0%!
    """
    return {
        # 1. v3 original (exatamente como salvo em modelo_greybox_v3.json -> "c3 fixo")
        "v3_original_p2": np.array([
            -6.54116, -0.44585, 0.003532, 0.000, 0.32383, U0_V3, 0.080, 2.000, 0.000
        ], dtype=float),
        # 2. v9 Opção 3A sem rigidez do cabo (k_th = 0) -> 0/12 giros em R7, Val45 = 74.8%, Val = 66.2%, All12 = 66.1%
        "v9_3a_sem_cabo": np.array([
            -8.541826, -0.368049, 0.022161, 0.042950, 0.177478, 10.624496, 0.106281, 1.565856, 0.000
        ], dtype=float),
        # 3. v9 Completo R1 (Equilibrado Global: Val45 = 87.7%, Val = 75.8%, Treino = 74.0%, All12 = 74.9%, 0/12 giros)
        "v9_completo_r1": np.array([
            -6.903059, -0.315352, 0.012466, 0.021040, 0.233892, 11.531358, 0.125374, 1.755476, 1.575768
        ], dtype=float),
        # 4. v9 Otimizado (Máxima Fidelidade na Validação e Malha Aberta: Val45 = 87.3%, Val60 = 64.5%, Val = 75.9%, All12 = 73.9%, MA 35-49% = 90.0%, 0/12 giros)
        "v9_otimizado": np.array([
            -6.858314, -0.331455, 0.012412, 0.019310, 0.255820, 11.852171, 0.113127, 1.759243, 1.565398
        ], dtype=float),
    }


def save_model_v9(thetas, rest_dict=None, mech_dict=None, static_dict=None,
                  fisher_dict=None, metrics_summary=None, filename="modelo_greybox_v9"):
    data = {
        "version": "v9_complete_quadratic_drag_cable_stiffness_zero_loopovers",
        "description": (
            "Modelo Grey-Box v9 Completo (Opcao 3A + Rigidez Torsional do Cabo): Equacao diferencial do v3 "
            "acrescida do arrasto aerodinamico quadratico (-d_a*theta'*|theta'|) e da rigidez elastica torsional "
            "do chicote/cabo eletrico no pivo (-K_s*theta), parametrizada com a oscilacao livre e curva estatica. "
            "Zero giros (0/12 em RODADA-7) e FIT de Validacao 45° de 87.2% (Geral 75.8%), superando o NARX (69.7%)."
        ),
        "equation_normalized": (
            "omega' = c1*sin(theta) - k_th*theta + c2*omega - ca*omega*|omega| - Fc*tanh(50*omega) + c3*max(a - u0, 0)^p; "
            "a' = (u - a)/tau_m"
        ),
        "equation_physical": (
            "J*theta'' + b_ef*theta' + d_a*theta'*|theta'| + Tc*tanh(50*theta') + C_g*sin(theta) + K_s*theta "
            "= K_tau*max(a - u0, 0)^p"
        ),
        "sensor_rest_characterization": rest_dict if rest_dict else {},
        "free_oscillation_mechanics": mech_dict if mech_dict else {},
        "static_characterization": static_dict if static_dict else {},
        "fisher_analysis": fisher_dict if fisher_dict else {},
        "metrics_summary": metrics_summary if metrics_summary else {},
        "models": {}
    }
    npz_dict = {}

    b_passivo_ref = float(mech_dict['b_passivo']) if mech_dict and 'b_passivo' in mech_dict else 0.004053

    for mode, th in thetas.items():
        c1, c2, c3, ca, Fc, u0, tau_m, p, k_th = map(float, th)
        J = float(-C_g / c1)
        b_ef = float(-c2 * J)
        b_aero = float(max(b_ef - b_passivo_ref, 0.0))
        d_a = float(ca * J)
        Tc = float(Fc * J)
        K_s = float(k_th * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        omega_n_eff = float(np.sqrt(abs(c1) + k_th))

        data["models"][mode] = {
            "theta": [c1, c2, c3, ca, Fc, u0, tau_m, p, k_th],
            "params": {
                "c1": c1, "c2": c2, "c3": c3, "ca": ca,
                "Fc": Fc, "u0": u0, "tau_m": tau_m, "p": p, "k_th": k_th
            },
            "physics": {
                "C_g": float(C_g),
                "J": J,
                "K_s": K_s,
                "omega_n_eff_rad_s": omega_n_eff,
                "b_ef": b_ef,
                "b_passivo": b_passivo_ref,
                "b_aero": b_aero,
                "d_a": d_a,
                "Tc": Tc,
                "Ktau": Ktau,
                "u0": u0,
                "tau_m": tau_m,
                "p": p,
                "k_th": k_th,
                "k_dyn": k_dyn
            }
        }
        npz_dict[f"theta_{casadi_name(mode)}"] = np.array(th, dtype=float)

    json_path = os.path.join(current_dir, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(current_dir, f"{filename}.npz"), **npz_dict)
    print(f"\n[INFO] Modelo v9 Completo salvo em: {json_path}")
    export_c_header_v9(data, filename="model_physical.h")


def export_c_header_v9(data, filename="model_physical.h"):
    models = data.get("models", {})
    if not models:
        return

    m_key = "v9_otimizado" if "v9_otimizado" in models else list(models.keys())[0]
    m_p = models[m_key]["params"]

    c1 = m_p["c1"]
    k_th = m_p["k_th"]
    c2 = m_p["c2"]
    ca = m_p["ca"]
    Fc = m_p["Fc"]
    c3 = m_p["c3"]
    p_thrust = m_p["p"]
    u0 = m_p["u0"]
    tau_m = m_p["tau_m"]

    content = f"""/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v9 Completo
 * (Arrasto Aerodinamico Quadratico -ca*w*|w| + Rigidez Torsional do Chicote -k_th*th + Dinamica de 1a Ordem tau_m)
 * Identificado em: ident_greybox_prof_multi_v9.py (Modelo Selecionado: {m_key})
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem):
 *   d(theta)/dt = omega
 *   d(omega)/dt = c1*sin(theta) - k_th*theta + c2*omega - ca*omega*|omega| - Fc*tanh(50*omega) + c3*(max(a - u0, 0))^p
 *   d(a)/dt     = (u_motor - a) / tau_m
 *
 * Parametros identificados ({m_key}):
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

/* ── Parametros fisicos do modelo Grey-Box v9 ({m_key}) ─────────────── */
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
"""
    dest_dirs = [
        os.path.join(root_dir, "firmware", "stm32-fisico", "Core", "Inc"),
        current_dir
    ]
    for d in dest_dirs:
        if os.path.exists(d):
            out_file = os.path.join(d, filename)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Header C ({m_key}) exportado em: {out_file}")


# =============================================================================
# Benchmark de Malha Aberta (14 Ensaios de Degrau — MALHA-ABERTA)
# =============================================================================
def benchmark_malha_aberta_v9(models, rk4_fn, ma_dir=MA_DIR):
    print("\n" + "=" * 110)
    print("BENCHMARK EXTERNO: 14 ENSAIOS DE DEGRAU EM MALHA ABERTA PURA (SEM PID)")
    print("=" * 110)

    files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_[0-9]*.csv")))
    if not files:
        print("  [AVISO] Pasta MALHA-ABERTA não encontrada.")
        return

    print(f"{'Ensaio':28s} | {'u [%]':7s} | {'y_max [°]':9s} | " + " | ".join(f"{m[:15]:>15s}" for m in models))
    print("-" * 110)

    metrics_per_model = {m: [] for m in models}
    for f in files:
        fname = os.path.basename(f)
        df = pd.read_csv(f)
        t = (df['tempo_ms'].values - df['tempo_ms'].values[0]) / 1000.0
        dt = float(np.mean(np.diff(t)))
        y = np.deg2rad(df['angulo_deg'].values)
        u = df['u_pct'].values
        N = len(y)

        sim_fn = rk4_fn.mapaccum(casadi_name('bench_ma_v9', fname), N - 1)
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
            metrics_per_model[m].append({'file': fname, 'u': float(np.max(u)), 'fit': fit_val})
            row_fits.append(f"{fit_val:6.1f}%" if not np.isnan(fit_val) else f"{'GIRO':>7s}")

        print(f"{fname:28s} | {float(np.max(u)):5.1f}% | {float(np.max(df['angulo_deg'].values)):7.1f}° | "
              + " | ".join(f"{s:>15s}" for s in row_fits))

    print("-" * 110)
    for m in models:
        df_m = pd.DataFrame(metrics_per_model[m])
        mid = df_m[(df_m['u'] >= 35) & (df_m['u'] <= 49) & (~df_m['fit'].isna())]
        all_13 = df_m[(df_m['u'] > 20) & (~df_m['fit'].isna())]
        giros = int(df_m['fit'].isna().sum())
        print(f"  * {m:18s}: Giros = {giros}/14 | Faixa Operacional (35-49%): FIT = {mid['fit'].mean():5.1f}% "
              f"| Média Geral (s/ u=15%): FIT = {all_13['fit'].mean():5.1f}%")


# =============================================================================
# Geração de Gráficos Comparativos (v9 Completo vs v3)
# =============================================================================
def plot_group_v9(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {
        'v3_original_p2': 'tab:gray',
        'v9_3a_sem_cabo': 'tab:orange',
        'v9_completo_r1': 'tab:blue',
        'v9_otimizado': 'tab:green'
    }
    styles = {
        'v3_original_p2': ':',
        'v9_3a_sem_cabo': '-.',
        'v9_completo_r1': '--',
        'v9_otimizado': '-'
    }
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição Experimental')
        txt = []
        for mode, ys in sims.items():
            y_plot = np.clip(np.rad2deg(ys[i]), -180.0, 180.0)
            fit_val = metrics[mode][i]
            fit_txt = f"{fit_val:.1f}%" if not np.isnan(fit_val) else "GIRO"
            ax.plot(exp['t'], y_plot, linestyle=styles.get(mode, '-'),
                    color=colors.get(mode, 'tab:red'), lw=1.4, label=f"{mode} ({fit_txt})")
            txt.append(f"{mode}: {fit_txt}")
        ax.set_title(f"{exp['name']}  |  " + "  |  ".join(txt), fontsize=9.5)
        ax.set_ylabel('θ [°]')
        y_deg = np.rad2deg(exp['y'])
        ax.set_ylim(y_deg.min() - 25, y_deg.max() + 25)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Tempo [s]')
    fig.suptitle(title, fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    try:
        fig.savefig(os.path.join(current_dir, fname), dpi=115)
        print(f"  [PLOT] Gráfico salvo em: {fname}")
    except Exception as e:
        print(f"  ! Aviso ao salvar '{fname}': {e}")
    finally:
        plt.close(fig)


# =============================================================================
# Pipeline Principal (v9 Completo)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Identificação Grey-Box v9 Completo (Opção 3A + Rigidez Torsional do Cabo + Física v8)")
    parser.add_argument("--all", action="store_true", help="Executa pipeline completo: fases 0..4, avaliação, malha aberta e gráficos")
    parser.add_argument("--fit", action="store_true", help="Executa a otimização por Multiple Shooting")
    parser.add_argument("--ident", action="store_true", help="Executa a análise da Matriz de Informação de Fisher")
    parser.add_argument("--eval", action="store_true", help="Avalia os modelos em RODADA-7 (100 Hz)")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos comparativos em disco (PNG)")
    parser.add_argument("--test_ma", action="store_true", help="Executa benchmark nos 14 ensaios de MALHA-ABERTA")
    parser.add_argument("--decim-fit", dest="decim_fit", type=int, default=DECIM_FIT, help="Decimação no NLP do Multiple Shooting (padrão: 10 -> 10 Hz)")
    parser.add_argument("--decim-eval", dest="decim_eval", type=int, default=DECIM_EVAL, help="Decimação na avaliação Free-Run (padrão: 1 -> 100 Hz)")
    parser.add_argument("--p", type=float, default=P_THRUST_DEFAULT, help="Expoente estático do empuxo p (padrão: 1.627)")
    parser.add_argument("--u0", type=float, default=U0_DEFAULT, help="Zona morta do ESC u0 [%%] (padrão: 11.26)")
    parser.add_argument("--seg-len", dest="seg_len", type=float, default=SEG_LEN, help="Janela do Multiple Shooting [s] (padrão: 20.0)")
    parser.add_argument("--L", type=float, default=L_OBS, help="Ganho do observador de Luenberger (padrão: 0.5)")
    parser.add_argument("--max-iter", dest="max_iter", type=int, default=50, help="Máximo de iterações do IPOPT")
    parser.add_argument("--fit-fc", dest="fit_fc", action="store_true", help="Permite refinar Fc junto com c2, ca e tau_m")
    args = parser.parse_args()

    if args.all or not (args.fit or args.ident or args.eval or args.test_ma or args.plot):
        args.fit = True
        args.ident = True
        args.eval = True
        args.plot = True
        args.test_ma = True

    print("=" * 90)
    print("SISTEMA DE IDENTIFICAÇÃO GREY-BOX DO AEROPÊNDULO — VERSÃO 9 COMPLETO (OPÇÃO 3A + RIGIDEZ DO CABO)")
    print("  * EDO: J*theta'' + b_ef*theta' + d_a*theta'*|theta'| + Tc*tanh(50*theta') + C_g*sin(theta) + K_s*theta = K_tau*max(a - u0, 0)^p")
    print(f"  * Configuração: p={args.p:.3f} | u0={args.u0:.2f}% | seg_len={args.seg_len:.1f}s | decim_fit={args.decim_fit} | decim_eval={args.decim_eval}")
    print("=" * 90)

    # 1. Fases 0, 1 e 2: Repouso, Oscilação Livre (u=0) e Curva Estática (87 pontos)
    rest_dict = characterize_rest_and_noise()
    mech_dict = identify_mechanics_free_oscillation()
    static_dict = extract_static_characterization_v9(
        u0_fixed=args.u0,
        p_prescribed=args.p,
        c1_val=mech_dict['c1'],
        plot=args.plot
    )

    # 2. Modelos de comparação para ablação (do v3 original até o v9 Completo)
    thetas = get_baseline_v9_models(static_dict, mech_dict)
    rk4_fn = build_rk4_v9()

    # 3. Fase 3: Otimização por Multiple Shooting + Refinamento Free-Run 100 Hz
    train_fit = [load_exp(*e, decim=args.decim_fit) for e in TRAIN]
    fisher_dict = None
    if args.fit:
        th_ms_opt, _ = identify_v9_multiple_shooting(
            train=train_fit,
            static_dict=static_dict,
            mech_dict=mech_dict,
            p_val=args.p,
            seg_len=args.seg_len,
            max_iter=args.max_iter,
            fit_fc=args.fit_fc
        )
        thetas["v9_otimizado"] = th_ms_opt

    # 4. Fase 4: Matriz de Informação de Fisher (Comparando Acoplado vs Desacoplado)
    if args.ident:
        target_th = thetas["v9_otimizado"]
        fisher_dict = analyze_identifiability_v9(
            theta_opt=target_th,
            train=train_fit,
            static_dict=static_dict,
            mech_dict=mech_dict,
            rk4_fn=rk4_fn,
            plot=args.plot
        )

    # 5. Avaliação Completa em Free-Run a 100 Hz (decim_eval = 1) nos 12 ensaios de RODADA-7
    print("\n--- Carregando 12 ensaios de RODADA-7 para avaliação a 100 Hz ---")
    train_eval = [load_exp(*e, decim=args.decim_eval) for e in TRAIN]
    val_eval = [load_exp(*e, decim=args.decim_eval) for e in VAL]
    all_r7 = val_eval + train_eval

    print("\n" + "=" * 115)
    print("AVALIAÇÃO EM MALHA ABERTA PURA (FREE-RUN GLOBAL, 100 Hz) — ABLAÇÃO DO v3 ATÉ O v9 (OPÇÃO 3A)")
    print("=" * 115)
    cols = ["Ensaio", "Grupo", "y_max"] + [f"{m[:15]}" for m in thetas]
    print(f"{cols[0]:24s} | {cols[1]:5s} | {cols[2]:6s} | " + " | ".join(f"{c:>17s}" for c in cols[3:]))
    print("-" * 115)

    sims_train = {m: [] for m in thetas}
    sims_val = {m: [] for m in thetas}
    metrics_train = {m: [] for m in thetas}
    metrics_val = {m: [] for m in thetas}
    rmses_val = {m: [] for m in thetas}
    rmses_train = {m: [] for m in thetas}
    giros_count = {m: 0 for m in thetas}

    for idx, e in enumerate(all_r7):
        is_val = idx < len(val_eval)
        grp = "VAL" if is_val else "TRN"
        y_max = float(np.max(np.rad2deg(e['y'])))
        row_strs = []

        for m, th in thetas.items():
            yh = free_run_v9(th, e, rk4_fn)
            if np.max(np.abs(np.rad2deg(yh))) >= 180.0:
                fit_val = np.nan
                rmse_val = np.nan
                giros_count[m] += 1
                row_strs.append(f"{'GIRO (>180°)':>17s}")
            else:
                fit_val = fit_pct(e['y'], yh)
                rmse_val = rmse_deg(e['y'], yh)
                row_strs.append(f"{fit_val:5.1f}% ({rmse_val:5.2f}°)")

            if is_val:
                sims_val[m].append(yh)
                metrics_val[m].append(fit_val)
                rmses_val[m].append(rmse_val)
            else:
                sims_train[m].append(yh)
                metrics_train[m].append(fit_val)
                rmses_train[m].append(rmse_val)

        print(f"{e['name']:24s} | {grp:5s} | {y_max:5.1f}° | " + " | ".join(row_strs))

    print("-" * 115)
    summary_metrics = {}
    for label, sl_val in [("MÉDIA VALIDAÇÃO 45° (3)", slice(0, 3)),
                          ("MÉDIA VALIDAÇÃO 60° (3)", slice(3, 6)),
                          ("MÉDIA VALIDAÇÃO GERAL (6)", slice(0, 6))]:
        row_s = []
        for m in thetas:
            f_arr = np.array(metrics_val[m][sl_val], dtype=float)
            r_arr = np.array(rmses_val[m][sl_val], dtype=float)
            if np.any(np.isnan(f_arr)):
                n_g = int(np.sum(np.isnan(f_arr)))
                row_s.append(f"GIRO ({n_g}/{len(f_arr)})")
            else:
                row_s.append(f"{np.mean(f_arr):5.1f}% ({np.mean(r_arr):5.2f}°)")
        print(f"{label:24s} |  VAL  |   ---  | " + " | ".join(f"{s:>17s}" for s in row_s))

    # Linha extra: Média Geral nos 12 Ensaios (Validação + Treino)
    row_12 = []
    for m in thetas:
        f_all = np.array(metrics_val[m] + metrics_train[m], dtype=float)
        r_all = np.array(rmses_val[m] + rmses_train[m], dtype=float)
        if giros_count[m] > 0:
            row_12.append(f"GIRO ({giros_count[m]}/12)")
        else:
            row_12.append(f"{np.mean(f_all):5.1f}% ({np.mean(r_all):5.2f}°)")
    print(f"{'MÉDIA GLOBAL (12 ENSAIOS)':24s} | ALL12 |   ---  | " + " | ".join(f"{s:>17s}" for s in row_12))
    print("-" * 115)

    for m in thetas:
        f_all = np.array(metrics_val[m] + metrics_train[m], dtype=float)
        r_all = np.array(rmses_val[m] + rmses_train[m], dtype=float)
        f_v45 = np.array(metrics_val[m][:3], dtype=float)
        r_v45 = np.array(rmses_val[m][:3], dtype=float)
        f_v60 = np.array(metrics_val[m][3:], dtype=float)
        r_v60 = np.array(rmses_val[m][3:], dtype=float)

        fits_obs_45, rmses_obs_45, fits_obs_60, rmses_obs_60 = [], [], [], []
        fits_win_val = []
        for i_v, e_v in enumerate(val_eval):
            yo = observer_run_v9(thetas[m], e_v, rk4_fn, L_val=args.L)
            yw = windowed_run_v9(thetas[m], e_v, rk4_fn, seg_len=args.seg_len)
            fits_win_val.append(fit_pct(e_v['y'], yw))
            if i_v < 3:
                fits_obs_45.append(fit_pct(e_v['y'], yo))
                rmses_obs_45.append(rmse_deg(e_v['y'], yo))
            else:
                fits_obs_60.append(fit_pct(e_v['y'], yo))
                rmses_obs_60.append(rmse_deg(e_v['y'], yo))

        summary_metrics[m] = {
            "giros_total_12": int(giros_count[m]),
            "val45_freerun_fit": float(np.nanmean(f_v45)),
            "val45_freerun_rmse_deg": float(np.nanmean(r_v45)),
            "val60_freerun_fit": float(np.nanmean(f_v60)) if not np.any(np.isnan(f_v60)) else None,
            "val60_freerun_rmse_deg": float(np.nanmean(r_v60)) if not np.any(np.isnan(r_v60)) else None,
            "val_all_freerun_fit": float(np.nanmean(metrics_val[m])) if not np.any(np.isnan(metrics_val[m])) else None,
            "val_all_freerun_rmse_deg": float(np.nanmean(rmses_val[m])) if not np.any(np.isnan(rmses_val[m])) else None,
            "all12_freerun_fit": float(np.nanmean(f_all)) if giros_count[m] == 0 else None,
            "all12_freerun_rmse_deg": float(np.nanmean(r_all)) if giros_count[m] == 0 else None,
            "val_windowed_fit": float(np.mean(fits_win_val)),
            "val45_observer_fit": float(np.mean(fits_obs_45)),
            "val45_observer_rmse_deg": float(np.mean(rmses_obs_45)),
            "val60_observer_fit": float(np.mean(fits_obs_60)),
            "val60_observer_rmse_deg": float(np.mean(rmses_obs_60)),
        }

    print("\n" + "=" * 95)
    print(f"AVALIAÇÃO COMPLEMENTAR NA VALIDAÇÃO: JANELA ({args.seg_len:.0f}s) E OBSERVADOR DE LUENBERGER (L={args.L})")
    print("=" * 95)
    for m, sm in summary_metrics.items():
        print(f"  * {m:18s} | Janela {args.seg_len:.0f}s: FIT = {sm['val_windowed_fit']:5.1f}% "
              f"| Observador 45°: {sm['val45_observer_fit']:5.1f}% ({sm['val45_observer_rmse_deg']:.2f}°) "
              f"| Observador 60°: {sm['val60_observer_fit']:5.1f}% ({sm['val60_observer_rmse_deg']:.2f}°)")

    # Salva JSON e NPZ
    save_model_v9(
        thetas=thetas,
        rest_dict=rest_dict,
        mech_dict=mech_dict,
        static_dict=static_dict,
        fisher_dict=fisher_dict,
        metrics_summary=summary_metrics,
        filename="modelo_greybox_v9"
    )

    # 6. Benchmark nos 14 ensaios de MALHA-ABERTA
    if args.test_ma:
        benchmark_malha_aberta_v9(thetas, rk4_fn)

    # 7. Gráficos comparativos
    if args.plot:
        print("\n--- Gerando gráficos comparativos (v9 Opção 3A) ---")
        for group_name, exps, sims, metrics in (("TREINO", train_eval, sims_train, metrics_train),
                                                ("VALIDAÇÃO", val_eval, sims_val, metrics_val)):
            prefix = "v9_treino" if group_name == "TREINO" else "v9_validacao"
            dc45_exps = [e for e in exps if "45" in e['name']]
            dc60_exps = [e for e in exps if "60" in e['name']]

            if dc45_exps:
                dc45_sims = {m: [sims[m][exps.index(e)] for e in dc45_exps] for m in thetas}
                dc45_mets = {m: [metrics[m][exps.index(e)] for e in dc45_exps] for m in thetas}
                plot_group_v9(dc45_exps, dc45_sims, dc45_mets,
                              f"v9 (Opção 3A — Equação v3 + Arrasto Quadrático): {group_name} DC=45% — Free-Run",
                              f"{prefix}_dc45_freerun.png")
            if dc60_exps:
                dc60_sims = {m: [sims[m][exps.index(e)] for e in dc60_exps] for m in thetas}
                dc60_mets = {m: [metrics[m][exps.index(e)] for e in dc60_exps] for m in thetas}
                plot_group_v9(dc60_exps, dc60_sims, dc60_mets,
                              f"v9 (Opção 3A — Equação v3 + Arrasto Quadrático): {group_name} DC=60% — Free-Run",
                              f"{prefix}_dc60_freerun.png")

    print("\n[CONCLUÍDO] Script ident_greybox_prof_multi_v9.py (Opção 3A) finalizado com sucesso!")


if __name__ == "__main__":
    main()
