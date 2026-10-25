"""
Identificação Grey-Box Multi-Experimento v6 (Dinâmica Assimétrica de Turbina EDF e Arrasto do Duto)
====================================================================================================
MOTIVAÇÃO FÍSICA E INOVAÇÕES METODOLÓGICAS (v6):
====================================================================================================
1. Dinâmica de Turbina Elétrica com Duto (Electric Ducted Fan - EDF):
   - Acelerando (u > a): O motor brushless injeta torque máximo elétrico contra a inércia do rotor,
     acelerando muito rápido (tau_acc ~ 0.04 a 0.07 s).
   - Desacelerando (u < a): O ESC padrão não possui frenagem regenerativa ativa (active braking).
     O rotor desacelera por roda-livre e atrito com o ar interno do duto (tau_dec ~ 0.15 a 0.35 s),
     sendo significativamente mais lento.
   - Isso elimina a clássica defasagem na descida de degraus e chirps observada no modelo v5.

2. Arrasto Aerodinâmico Assimétrico do Duto (Forward vs Backward):
   - Movimento para frente (v > 0): A boca arredondada do duto capta ar suavemente (c_aero_fwd).
   - Movimento para trás (v < 0): O duto atua como um obstáculo e freio aerodinâmico turbulento (c_aero_bwd).
   - O arrasto é modelado suavemente com transição via tanh:
       c_aero(v) = c_aero_fwd * sigma_v + c_aero_bwd * (1 - sigma_v)

3. Observador Luenberger Completo de 2ª Ordem na Validação:
   - Em vez de corrigir apenas o ângulo (L1), agora corrige simultaneamente o ângulo e a velocidade angular:
       L1 = L,  L2 = L^2 / 4.0 (alocação de polos criticamente amortecida).
   - Estimação de velocidade sem atraso de fase, permitindo almejar FIT ~ 90% com estabilidade total.

4. Modos de Otimização em v6:
   - "turbina fixada" (Recomendado): Mecânica passiva fixada da oscilação livre; calibra c3,
     c_aero_fwd, c_aero_bwd e tau_dec (máxima robustez e estabilidade sem divergência).
   - "turbina regularizada": Todos os parâmetros livres com regularização penalizando afastamento dos priors.
   - "assimetria motriz": Arrasto simétrico, com assimetria focada no motor (tau_acc != tau_dec).
   - "simetrico v5": Reprodução da física simétrica do v5 para comparação direta na mesma tabela.
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

# Hiperparâmetros padrão da identificação v6
DECIM = 1
LAMBDA_C3 = 10.0         # peso da regularização de c3 em torno de k_static*|c1|
LAMBDA_MECH = 15.0       # peso da regularização mecânica com os priors de oscilação livre
LAMBDA_AERO = 10.0       # peso da regularização do arrasto aerodinâmico
LAMBDA_MOTOR = 8.0       # peso da regularização da desaceleração da turbina
SEG_LEN = 20.0            # comprimento da janela de multiple shooting [s]
L_OBS = 2.0              # ganho do observador de estados na validação (L1=L, L2=L^2/4)
TAU_ACC_DEFAULT = 0.05   # constante de tempo de aceleração da turbina [s] (~60 ms)
TAU_DEC_DEFAULT = 0.20   # constante de tempo de desaceleração (roda-livre) [s] (~200 ms)
U0_DEFAULT = 11.26       # zona morta calibrada da bancada (~11.3%)
C_AERO_FWD_DEFAULT = 0.85 # arrasto para frente (boca do duto suave) [rad^-1]
C_AERO_BWD_DEFAULT = 1.35 # arrasto para trás (duto como obstáculo) [rad^-1]

# Escalas das variáveis de decisão para condicionamento do IPOPT (~ 1.0)
S_C1 = 8.0
S_C2 = 0.4
S_F = 0.15
S_CA_FWD = 0.20
S_CA_BWD = 0.20
S_C3 = 8.0e-3
S_TAU_DEC = 0.20


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
        v0 = 0.0

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
            "c1": -8.21, "c2": -0.15, "Fc": 0.08, "c_aero": 0.95,
            "J": 0.034, "b": 0.005, "Tc": 0.003,
            "omega_n": np.sqrt(8.21),
            "period_T0": 2 * np.pi / np.sqrt(8.21)
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

    p0 = [-8.34, -0.12, 0.16, 0.08]
    bounds = ([-14.0, -2.5, 0.01, 0.0], [-5.0, -0.01, 1.5, 1.0])
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
    print(f"  Parâmetros Mecânicos Normalizados (Grey-Box v6):")
    print(f"    c1 (gravidade/inércia) = {c1_mech:.5f} rad/s^2  -->  omega_n = {omega_n:.3f} rad/s")
    print(f"    c2 (amortecimento visc)= {c2_mech:.5f} s^-1")
    print(f"    Fc (atrito de Coulomb) = {Fc_mech:.5f} rad/s^2")
    print(f"    c_aero (arrasto repouso)= {ca_mech:.5f} rad^-1")
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
            ax.plot(d['t'], np.rad2deg(y_sim), 'r--', lw=1.4, label='Modelo Mecânico v6')
            ax.set_title(f"{d['name']} — Oscilação Livre | FIT={fit:.1f}% | RMSE={rmse:.2f}°", fontsize=10)
            ax.set_ylabel('θ [°]', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        axes[-1, 0].set_xlabel('Tempo [s]', fontsize=9)
        fig.suptitle(f"Identificação Mecânica Passiva v6 (Oscilação Livre)\n"
                     f"J={J_mech:.4f} kg·m² | b={b_mech:.4f} N·m·s/rad | c_aero={ca_mech:.4f}",
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        out_f = os.path.join(current_dir, "v6_caracterizacao_repouso_oscilacao.png")
        try:
            fig.savefig(out_f, dpi=120)
            print(f"  Gráfico salvo em: v6_caracterizacao_repouso_oscilacao.png")
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
        fname = "v6_curva_semiestatica_malha_aberta.png"
    else:
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

    print(f"  Modelo v6 Semiestático: k = {k_opt:.4e} 1/%^2 | u0 = {u0_opt:.2f}% | R^2 = {r2_opt:.4f}")

    if plot:
        plt.figure(figsize=(8, 5))
        u_grid = np.linspace(0, max(u_avg.max(), 65) + 2, 200)
        plt.scatter(u_avg, Torque_g, s=35, c='k', label='Dados Experimentais')
        plt.plot(u_grid, k_opt * np.maximum(u_grid - u0_opt, 0)**2, 'r-', lw=2.2,
                 label=f'v6: k·(max(u-{u0_opt:.1f}%, 0))² (R²={r2_opt:.3f})')
        plt.axvline(u0_opt, color='gray', linestyle=':', label=f'Zona morta u0 = {u0_opt:.1f}%')
        plt.title(f"Curva Semiestática do Empuxo ({title_tag})", fontsize=11, fontweight='bold')
        plt.xlabel("Comando da Turbina u [%]", fontsize=10)
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
# Modelo e Integrador RK4 da Turbina EDF (CasADi)
# =============================================================================
def build_rk4_v6(u0_fixed=None, tau_acc_fixed=None):
    """
    Integrador RK4 com estado 3D: [theta, v, a].
    Vetor de parâmetros th (7 elementos):
      th[0]: c1         - Torque gravitacional normalizado
      th[1]: c2         - Amortecimento viscoso
      th[2]: c3         - Ganho de empuxo da turbina
      th[3]: Fc         - Atrito de Coulomb no mancal
      th[4]: c_aero_fwd - Arrasto aerodinâmico para frente (v > 0)
      th[5]: c_aero_bwd - Arrasto aerodinâmico para trás (v < 0, duto reverso)
      th[6]: tau_dec    - Constante de tempo de desaceleração (roda-livre da turbina)
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')
    th = cs.SX.sym('th', 7)

    u0_val = cs.SX(u0_fixed if u0_fixed is not None else U0_DEFAULT)
    tau_acc_val = cs.SX(tau_acc_fixed if tau_acc_fixed is not None else TAU_ACC_DEFAULT)

    def f(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        thrust = cs.fmax(a - u0_val, 0)**2
        fric = -th[3] * cs.tanh(50.0 * v)

        # Arrasto direcional do duto da turbina
        # sigma_v -> 1 se v > 0 (frente), sigma_v -> 0 se v < 0 (trás)
        sigma_v = 0.5 * (1.0 + cs.tanh(20.0 * v))
        ca_eff = th[4] * sigma_v + th[5] * (1.0 - sigma_v)
        aero = -ca_eff * v * cs.fabs(v)

        # Dinâmica assimétrica de rotação do rotor da turbina
        # sigma_u -> 1 se u > a (aceleração motriz forte), sigma_u -> 0 se u < a (desaceleração passiva lenta)
        sigma_u = 0.5 * (1.0 + cs.tanh(10.0 * (u_ - a)))
        tau_eff = th[6] + (tau_acc_val - th[6]) * sigma_u

        acc = th[0] * cs.sin(th_pos) + th[1] * v + aero + fric + th[2] * thrust
        a_dot = (u_ - a) / cs.fmax(tau_eff, 0.01)

        # Observador de estados Luenberger de 2ª ordem (criticamente amortecido)
        e_y = y_meas - th_pos
        obs_corr = cs.vertcat(L * e_y, (L**2 / 4.0) * e_y, 0)
        return cs.vertcat(v, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2.0 * k1, u)
    k3 = f(x + h / 2.0 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return cs.Function('rk4_v6', [x, u, th, y_meas, L, h], [x_next])


# =============================================================================
# Funções de Simulação e Validação
# =============================================================================
def free_run(theta, exp, rk4_fn, u0_val, tau_acc_val):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_free_v6', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_dummy = cs.DM.zeros(1, N - 1)
    L_zero = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_dummy, L_zero, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def observer_run(theta, exp, rk4_fn, L_val=L_OBS):
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_obs_v6', exp['name'], N), N - 1)
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
        sim = rk4_fn.mapaccum(casadi_name('sim_win_v6', exp['name'], s), n - 1)
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
# Fase 3: Identificação Multi-Experimento da Turbina EDF (CasADi / IPOPT)
# =============================================================================
def identify_v6(train, k_static, u0_val, tau_acc_val, mode="turbina fixada",
                mech_params=None, lam=LAMBDA_C3, lam_mech=LAMBDA_MECH,
                lam_aero=LAMBDA_AERO, lam_motor=LAMBDA_MOTOR, th_init=None,
                seg_len=SEG_LEN, tol=1e-3, max_iter=300, time_limit=600.0):
    rk4_fn = build_rk4_v6(u0_fixed=u0_val, tau_acc_fixed=tau_acc_val)
    opti = cs.Opti()

    # Parâmetros de referência
    ca_fwd_ref = C_AERO_FWD_DEFAULT
    ca_bwd_ref = C_AERO_BWD_DEFAULT
    tau_dec_ref = TAU_DEC_DEFAULT

    if mode == "turbina fixada":
        # Congela rigidamente a mecânica passiva calibrada
        c1 = float(mech_params['c1'])
        c2 = float(np.clip(mech_params['c2'] * 5.75, -0.75, -0.68)) 
        Fc = float(max(mech_params['Fc'] * 2.0, 0.30))

        # Variáveis livres da Turbina EDF
        p_c3 = opti.variable()
        p_ca_fwd = opti.variable()
        p_ca_bwd = opti.variable()
        p_tdec = opti.variable()

        c3 = S_C3 * p_c3
        c_aero_fwd = S_CA_FWD * p_ca_fwd
        c_aero_bwd = S_CA_BWD * p_ca_bwd
        tau_dec = S_TAU_DEC * p_tdec

        # Limites físicos (teto seguro anti-looping c3 <= 0.00428 e dissipação de duto)
        opti.subject_to(opti.bounded(0.0030 / S_C3, p_c3, 0.00428 / S_C3))
        opti.subject_to(opti.bounded(0.70 / S_CA_FWD, p_ca_fwd, 2.00 / S_CA_FWD))
        opti.subject_to(opti.bounded(0.85 / S_CA_BWD, p_ca_bwd, 3.50 / S_CA_BWD))
        opti.subject_to(opti.bounded(0.08 / S_TAU_DEC, p_tdec, 0.120 / S_TAU_DEC))

        # Arrasto de ré maior ou igual ao de frente (geometria de duto)
        opti.subject_to(c_aero_bwd >= c_aero_fwd)

        # Inicialização
        opti.set_initial(p_c3, 0.00425 / S_C3)
        opti.set_initial(p_ca_fwd, 0.85 / S_CA_FWD)
        opti.set_initial(p_ca_bwd, 0.95 / S_CA_BWD)
        opti.set_initial(p_tdec, 0.115 / S_TAU_DEC)

    elif mode == "assimetria motriz":
        # Assimetria apenas no motor; arrasto simétrico
        c1 = float(mech_params['c1'])
        c2 = float(np.clip(mech_params['c2'] * 5.75, -0.75, -0.68)) 
        Fc = float(max(mech_params['Fc'] * 2.0, 0.30))

        p_c3 = opti.variable()
        p_ca = opti.variable()
        p_tdec = opti.variable()

        c3 = S_C3 * p_c3
        c_aero_fwd = S_CA_FWD * p_ca
        c_aero_bwd = c_aero_fwd  # simétrico
        tau_dec = S_TAU_DEC * p_tdec

        opti.subject_to(opti.bounded(0.0030 / S_C3, p_c3, 0.00428 / S_C3))
        opti.subject_to(opti.bounded(0.50 / S_CA_FWD, p_ca, 2.50 / S_CA_FWD))
        opti.subject_to(opti.bounded(0.08 / S_TAU_DEC, p_tdec, 0.40 / S_TAU_DEC))

        opti.set_initial(p_c3, 0.00420 / S_C3)
        opti.set_initial(p_ca, 0.95 / S_CA_FWD)
        opti.set_initial(p_tdec, tau_dec_ref / S_TAU_DEC)

    elif mode == "simetrico v5":
        # Linha de base v5 pura: tau_dec = tau_acc e c_aero simétrico
        c1 = float(mech_params['c1'])
        c2 = float(np.clip(mech_params['c2'] * 5.75, -0.75, -0.68)) 
        Fc = float(max(mech_params['Fc'] * 2.0, 0.30))
        c_aero_fwd = 0.95
        c_aero_bwd = 0.95
        tau_dec = tau_acc_val

        p_c3 = opti.variable()
        c3 = S_C3 * p_c3
        opti.subject_to(opti.bounded(0.0030 / S_C3, p_c3, 0.00428 / S_C3))
        opti.set_initial(p_c3, 0.00425 / S_C3)

    else:  # "turbina regularizada"
        p1 = opti.variable()
        p2 = opti.variable()
        p3 = opti.variable()
        p_c3 = opti.variable()
        p_ca_fwd = opti.variable()
        p_ca_bwd = opti.variable()
        p_tdec = opti.variable()

        c1 = -S_C1 * p1
        c2 = -S_C2 * p2
        Fc = S_F * p3
        c3 = S_C3 * p_c3
        c_aero_fwd = S_CA_FWD * p_ca_fwd
        c_aero_bwd = S_CA_BWD * p_ca_bwd
        tau_dec = S_TAU_DEC * p_tdec

        opti.subject_to(opti.bounded(0.85, p1, 1.25))       # |c1| entre 6.8 e 10.0 rad/s^2
        opti.subject_to(opti.bounded(1.85, p2, 5.0))        # |c2| entre 0.74 e 2.0 s^-1
        opti.subject_to(opti.bounded(1.80, p3, 5.0))        # Fc entre 0.27 e 0.75 rad/s^2
        opti.subject_to(opti.bounded(0.0032 / S_C3, p_c3, 0.00435 / S_C3))
        opti.subject_to(opti.bounded(0.40 / S_CA_FWD, p_ca_fwd, 2.00 / S_CA_FWD))
        opti.subject_to(opti.bounded(0.40 / S_CA_BWD, p_ca_bwd, 3.50 / S_CA_BWD))
        opti.subject_to(opti.bounded(0.08 / S_TAU_DEC, p_tdec, 0.40 / S_TAU_DEC))
        opti.subject_to(c_aero_bwd >= c_aero_fwd)

        opti.set_initial(p1, 8.21 / S_C1)
        opti.set_initial(p2, 0.89 / S_C2)
        opti.set_initial(p3, 0.30 / S_F)
        opti.set_initial(p_c3, 0.00418 / S_C3)
        opti.set_initial(p_ca_fwd, ca_fwd_ref / S_CA_FWD)
        opti.set_initial(p_ca_bwd, ca_bwd_ref / S_CA_BWD)
        opti.set_initial(p_tdec, tau_dec_ref / S_TAU_DEC)

    theta = cs.vertcat(c1, c2, c3, Fc, c_aero_fwd, c_aero_bwd, tau_dec)

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

    # Regularização de desaceleração do motor da turbina
    if mode in ("turbina fixada", "turbina regularizada", "assimetria motriz") and lam_motor > 0:
        cost += lam_motor * ((tau_dec - tau_dec_ref) / tau_dec_ref)**2

    # Regularização aerodinâmica do duto
    if mode in ("turbina fixada", "turbina regularizada") and lam_aero > 0:
        cost += lam_aero * (
            ((c_aero_fwd - ca_fwd_ref) / ca_fwd_ref)**2 +
            ((c_aero_bwd - ca_bwd_ref) / ca_bwd_ref)**2
        )

    # Regularização mecânica
    if mode == "turbina regularizada" and mech_params is not None and lam_mech > 0:
        c1_ref = float(mech_params['c1'])
        cost += lam_mech * (
            ((c1 - c1_ref) / c1_ref)**2 +
            0.5 * ((c2 - (-0.89)) / 0.89)**2 +
            0.5 * ((Fc - 0.30) / 0.30)**2
        )
        k_dyn_target = 0.00051
        cost += lam * ((c3 / (-c1) - k_dyn_target) / k_dyn_target)**2

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
    ca_fwd_val = float(sol.value(c_aero_fwd))
    ca_bwd_val = float(sol.value(c_aero_bwd))
    tau_dec_val = float(sol.value(tau_dec))

    th_val = np.array([c1_val, c2_val, c3_val, Fc_val, ca_fwd_val, ca_bwd_val, tau_dec_val])
    cost_val = float(sol.value(cost))
    print(f"  [{status} em {time.time()-t0:.1f} s | Custo: {cost_val:.2f}]")

    return th_val, cost_val, None, None


# =============================================================================
# Análise de Fisher (Covariância e Incerteza v6)
# =============================================================================
def compute_fisher_v6(theta_opt, train, k_static, u0_val, tau_acc_val, mode="turbina fixada"):
    print(f"\n--- Matriz de Informação de Fisher / Incerteza (v6: {mode.upper()}) ---")
    rk4_fn = build_rk4_v6(u0_fixed=u0_val, tau_acc_fixed=tau_acc_val)

    if mode == "turbina fixada":
        param_names = ["c3", "ca_fwd", "ca_bwd", "tau_dec"]
        indices = [2, 4, 5, 6]
    elif mode == "assimetria motriz":
        param_names = ["c3", "ca_fwd", "tau_dec"]
        indices = [2, 4, 6]
    elif mode == "simetrico v5":
        param_names = ["c3"]
        indices = [2]
    else:
        param_names = ["c1", "c2", "c3", "Fc", "ca_fwd", "ca_bwd", "tau_dec"]
        indices = list(range(7))

    n_p = len(param_names)
    th_eval = theta_opt[indices]
    F_total = np.zeros((n_p, n_p))
    res_sq_sum = 0.0
    total_samples = 0

    p_sym = cs.SX.sym('p_active', n_p)
    full_th_list = list(theta_opt)
    for p_name, idx in zip(param_names, indices):
        p_idx = param_names.index(p_name)
        full_th_list[idx] = p_sym[p_idx]
    if mode == "assimetria motriz":
        full_th_list[5] = p_sym[1]  # ca_bwd = ca_fwd

    full_th = cs.vertcat(*full_th_list)

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

    sigma2_res = res_sq_sum / max(total_samples - n_p, 1)
    try:
        Cov = np.linalg.inv(F_total) * sigma2_res
        stds = np.sqrt(np.maximum(np.diag(Cov), 0.0))
        D_inv = np.diag(1.0 / np.maximum(stds, 1e-12))
        Corr = D_inv @ Cov @ D_inv
        cv_pct = (stds / np.maximum(np.abs(th_eval), 1e-12)) * 100.0

        print(f"  Variância Residual (sigma^2): {sigma2_res:.4e} rad^2  (RMSE 1-passo: {np.rad2deg(np.sqrt(sigma2_res)):.3f}°)")
        print(f"  {'Parâmetro':15s} | {'Valor Ótimo':14s} | {'Desvio Pad (1-sigma)':20s} | {'CV (%)':8s}")
        print("  " + "-" * 65)
        for name, val, s, cv in zip(param_names, th_eval, stds, cv_pct):
            print(f"  {name:15s} | {val:14.5e} | ± {s:18.5e} | {cv:6.2f}%")
        return {"Cov": Cov.tolist(), "Corr": Corr.tolist(), "stds": stds.tolist(), "cv_pct": cv_pct.tolist()}
    except np.linalg.LinAlgError:
        print("  ! Aviso: Matriz de Fisher singular ou mal-condicionada.")
        return {}


# =============================================================================
# Exibição e Salvamento dos Modelos
# =============================================================================
def print_params_v6(label, th, k_static, u0_val, tau_acc_val):
    c1, c2, c3, Fc, ca_fwd, ca_bwd, tau_dec = map(float, th[:7])
    J = float(-C_g / c1)
    b = float(-c2 * J)
    Tc = float(Fc * J)
    Ktau = c3 * J
    k_dyn = c3 / abs(c1)

    print(f"\n[{label}]")
    print(f"  c1 = {c1:.5f} | c2 = {c2:.5f} | c3 = {c3:.5e}")
    print(f"  Fc = {Fc:.5f} rad/s^2")
    print(f"  Arrasto Duto:  c_aero_fwd = {ca_fwd:.4f} rad^-1 | c_aero_bwd = {ca_bwd:.4f} rad^-1  (Razão bwd/fwd: {ca_bwd/max(ca_fwd, 1e-6):.2f})")
    print(f"  Motor Turbina: tau_acc = {tau_acc_val:.4f} s | tau_dec = {tau_dec:.4f} s  (Razão dec/acc: {tau_dec/max(tau_acc_val, 1e-6):.2f})")
    print(f"  Zona Morta u0 = {u0_val:.2f}% | k_dyn = {k_dyn:.4e} (k_static = {k_static:.4e}, ratio = {k_dyn / k_static:.3f})")
    print(f"  Física Equivalente: J = {J:.5f} kg.m^2 | b = {b:.5f} N.m.s/rad | Tc = {Tc:.5f} N.m | Ktau = {Ktau:.4e} N.m/%^2")


def save_model_v6(thetas, k_static, u0_val, tau_acc_val, mech_dict=None,
                  rest_dict=None, fisher_dict=None, filename="modelo_greybox_v6"):
    data = {
        "version": "v6_asymmetric_edf_turbine",
        "description": "Modelo Grey-box de 3ª ordem da Turbina EDF com arrasto direcional de duto e dinâmica assimétrica de rotação",
        "k_static": float(k_static),
        "u0_static": float(u0_val),
        "tau_acc_fixed": float(tau_acc_val),
        "C_g": float(C_g),
        "rest_characterization": rest_dict or {},
        "free_oscillation_mechanics": mech_dict or {},
        "models": {},
        "fisher_analysis": fisher_dict or {}
    }
    npz_dict = {
        "k_static": np.array(k_static, dtype=float),
        "u0_static": np.array(u0_val, dtype=float),
        "tau_acc_fixed": np.array(tau_acc_val, dtype=float)
    }

    for mode, th in thetas.items():
        c1, c2, c3, Fc, ca_fwd, ca_bwd, tau_dec = map(float, th[:7])
        J = float(-C_g / c1)
        b = float(-c2 * J)
        Tc = float(Fc * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        tag = casadi_name(mode)

        data["models"][mode] = {
            "theta": [c1, c2, c3, Fc, ca_fwd, ca_bwd, tau_dec, float(u0_val), float(tau_acc_val)],
            "params": {
                "c1": c1, "c2": c2, "c3": c3, "Fc": Fc,
                "c_aero_fwd": ca_fwd, "c_aero_bwd": ca_bwd,
                "tau_dec": tau_dec, "u0": float(u0_val), "tau_acc": float(tau_acc_val)
            },
            "physics": {
                "J": J, "b": b, "Tc": Tc, "Ktau": Ktau,
                "c_aero_fwd": ca_fwd, "c_aero_bwd": ca_bwd,
                "tau_acc": float(tau_acc_val), "tau_dec": tau_dec,
                "k_dyn": k_dyn, "ratio_k_dyn_static": float(k_dyn / k_static)
            }
        }
        npz_dict[f"theta_{tag}"] = np.array(th, dtype=float)

    json_path = os.path.join(current_dir, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Modelo v6 salvo em: {json_path}")

    npz_path = os.path.join(current_dir, f"{filename}.npz")
    np.savez(npz_path, **npz_dict)
    print(f"  Pesos numéricos salvos em: {npz_path}")

    # Exporta Header C para STM32
    export_c_header_v6(data, filename="model_physical.h")


def export_c_header_v6(data, filename="model_physical.h"):
    base_dir = root_dir
    models = data.get("models", {})
    if not models:
        return

    m_key = "turbina fixada" if "turbina fixada" in models else list(models.keys())[0]
    m_p = models[m_key]["params"]

    c1 = m_p["c1"]
    c2 = m_p["c2"]
    c3 = m_p["c3"]
    Fc = m_p["Fc"]
    ca_fwd = m_p["c_aero_fwd"]
    ca_bwd = m_p["c_aero_bwd"]
    tau_acc = m_p["tau_acc"]
    tau_dec = m_p["tau_dec"]
    u0 = m_p["u0"]

    content = f"""/*
 * model_physical.h
 * ================
 * Simulador de Planta Fisica do Aeropendulo baseado no Modelo Grey-Box v6 (Turbina EDF Assimetrica)
 * Identificado em: ident_greybox_prof_multi_v6.py (Modelo Selecionado: {m_key})
 *
 * Equacoes de estado nao-lineares em tempo continuo (3a ordem com assimetria de turbina):
 *   d(theta)/dt = v
 *   d(v)/dt     = c1*sin(theta) + c2*v - c_aero(v)*v*|v| - Fc*tanh(50*v) + c3*(max(a - u0, 0))^2
 *   d(a)/dt     = (u_motor - a) / tau_eff(u_motor, a)
 *
 * Onde:
 *   c_aero(v) = c_aero_fwd * sigma_v + c_aero_bwd * (1 - sigma_v)    [Arrasto Direcional de Duto]
 *   tau_eff   = tau_dec + (tau_acc - tau_dec) * sigma_u              [Dinamica de Rotacao Assimetrica]
 *
 * Integrador numerico: Runge-Kutta 4a Ordem (RK4) com 2 subpassos de 5 ms a cada ciclo de 10 ms (100 Hz).
 */

#ifndef MODEL_PHYSICAL_H
#define MODEL_PHYSICAL_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Parametros fisicos do modelo Grey-Box v6 (Turbina EDF) ──────────────── */
#ifndef GB_C1
#define GB_C1          ({c1:.6f}f)   /* rad/s^2 - torque gravitacional normalizado */
#endif

#ifndef GB_C2
#define GB_C2          ({c2:.6f}f)   /* s^-1    - amortecimento viscoso mecanico */
#endif

#ifndef GB_C3
#define GB_C3          ({c3:.8f}f)  /* rad/(s^2 * %^2) - ganho de empuxo quadratico */
#endif

#ifndef GB_FC
#define GB_FC          ({Fc:.6f}f)    /* rad/s^2 - atrito seco de Coulomb no mancal */
#endif

#ifndef GB_CA_FWD
#define GB_CA_FWD      ({ca_fwd:.6f}f)    /* rad^-1  - arrasto com duto avancando */
#endif

#ifndef GB_CA_BWD
#define GB_CA_BWD      ({ca_bwd:.6f}f)    /* rad^-1  - arrasto com duto recuando */
#endif

#ifndef GB_TAU_ACC
#define GB_TAU_ACC     ({tau_acc:.4f}f)      /* s       - constante de aceleracao motriz */
#endif

#ifndef GB_TAU_DEC
#define GB_TAU_DEC     ({tau_dec:.4f}f)      /* s       - constante de desaceleracao roda-livre */
#endif

#ifndef GB_U0
#define GB_U0          ({u0:.4f}f)     /* %       - zona morta da turbina/esc */
#endif

/* ── Estados internos da planta fisica no modo SIL ──────────────────────── */
static float sil_th_rad = 0.0f; /* Posicao angular (rad, 0 = repouso vertical para baixo) */
static float sil_v_rad  = 0.0f; /* Velocidade angular (rad/s) */
static float sil_a_pct  = 0.0f; /* Estado dinamico do atuador/turbina (%) */

/*
 * Derivadas continuas do modelo fisico Grey-Box v6
 */
static inline void phys_derivs(float th, float v, float a, float u_cmd,
                               float *dth, float *dv, float *da)
{{
    float u_motor = (u_cmd > 0.0f) ? ((u_cmd > 100.0f) ? 100.0f : u_cmd) : 0.0f;
    float z = a - GB_U0;
    float thrust = (z > 0.0f) ? (z * z) : 0.0f;
    float fric   = -GB_FC * tanhf(50.0f * v);

    /* Arrasto direcional do duto da turbina */
    float sigma_v = 0.5f * (1.0f + tanhf(20.0f * v));
    float ca_eff  = GB_CA_FWD * sigma_v + GB_CA_BWD * (1.0f - sigma_v);
    float aero    = -ca_eff * v * fabsf(v);

    /* Dinamica assimetrica de rotacao do rotor da turbina */
    float sigma_u = 0.5f * (1.0f + tanhf(10.0f * (u_motor - a)));
    float tau_eff = GB_TAU_DEC + (GB_TAU_ACC - GB_TAU_DEC) * sigma_u;

    *dth = v;
    *dv  = GB_C1 * sinf(th) + GB_C2 * v + aero + fric + GB_C3 * thrust;
    *da  = (u_motor - a) / (tau_eff > 0.01f ? tau_eff : 0.01f);
}}

/*
 * Simulador de Planta Fisica Grey-Box v6 (RK4 a 100 Hz com 2 subpassos de 5 ms)
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

    return th * (180.0f / (float)M_PI);
}}

static inline float simulate_narx(float u_cmd)
{{
    return simulate_physical_plant(u_cmd);
}}

#endif /* MODEL_PHYSICAL_H */
"""
    dest_dirs = [
        os.path.join(base_dir, "firmware", "stm32-fisico", "Core", "Inc"),
        os.path.join(base_dir, "firmware", "stm32-sil", "Core", "Inc"),
        current_dir
    ]
    for d in dest_dirs:
        if os.path.exists(d):
            out_file = os.path.join(d, "model_physical.h")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Header C exportado em: {out_file}")


def load_saved_model_v6(filename="modelo_greybox_v6"):
    json_path = os.path.join(current_dir, f"{filename}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thetas = {}
    for mode, mdata in data.get("models", {}).items():
        thetas[mode] = np.array(mdata["theta"][:7], dtype=float)
    return thetas, data.get("k_static"), data.get("u0_static"), data.get("tau_acc_fixed"), data.get("free_oscillation_mechanics")


def plot_group_v6(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {
        'simetrico v5': 'tab:gray',
        'assimetria motriz': 'tab:blue',
        'turbina fixada': 'tab:green',
        'turbina regularizada': 'tab:purple'
    }
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição')
        txt = []
        for mode, ys in sims.items():
            ax.plot(exp['t'], np.rad2deg(ys[i]), '--', color=colors.get(mode, 'tab:orange'),
                    lw=1.3, label=mode)
            txt.append(f"{mode}: {metrics[mode][i][0]:.1f}%")
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
# Pipeline Principal de Execução v6
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Identificação Grey-Box Multi-Experimento v6 (Turbina EDF Assimétrica)")
    parser.add_argument("--eval", action="store_true", help="Avalia modelo salvo sem reotimizar")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos a partir do modelo salvo")
    parser.add_argument("--all", action="store_true", help="Força reotimização completa de todos os modos")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["turbina fixada", "turbina regularizada", "assimetria motriz", "simetrico v5"],
                        help="Treina/reotimiza apenas o modo especificado")
    parser.add_argument("--u0", type=float, default=None, help="Valor fixo de zona morta u0 [%%]")
    parser.add_argument("--static-method", type=str, default="malha_aberta", choices=["malha_aberta", "escada"],
                        help="Método de calibração da curva semiestática")
    parser.add_argument("--tau-acc", type=float, default=TAU_ACC_DEFAULT, help="Constante de aceleração da turbina [s]")
    parser.add_argument("--L", type=float, default=L_OBS, help="Ganho do observador Luenberger (padrão: 2.0)")
    parser.add_argument("--decim", type=int, default=DECIM, help="Fator de decimação dos ensaios")
    parser.add_argument("--max-iter", type=int, default=300, help="Teto de iterações do IPOPT (padrão: 300)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerância de convergência KKT")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Tempo limite por run em segundos")
    args = parser.parse_args()

    print("=" * 80)
    print("IDENTIFICAÇÃO GREY-BOX MULTI-EXPERIMENTO v6 (TURBINA EDF ASSIMÉTRICA + ARRASTO DE DUTO)")
    print("  * Fase 0: Ruído do sensor e offset vertical extraídos de RODADA_REPOUSO")
    print("  * Fase 1: Identificação mecânica passiva isolada com arrasto (u=0)")
    print("  * Fase 2: Ganho e zona morta semiestáticos do empuxo k*(max(u - u0, 0))²")
    print("  * Fase 3: Treino Multi-Experimento com c3, c_aero_fwd, c_aero_bwd, tau_dec")
    print("=" * 80)

    # Passo 0: Caracterização de Repouso e Ruído
    rest_dict = characterize_rest_and_noise(plot=True)

    # Passo 1: Identificação Mecânica Passiva Isolada com Arrasto
    mech_dict = identify_mechanics_free_oscillation(plot=True)

    # Passo 2: Curva Semiestática do Empuxo
    k_static, u0_static = extract_static_gain(method=args.static_method, u0_override=args.u0, plot=True)
    tau_acc_fixed = float(args.tau_acc)

    # Carregamento dos Ensaios de Treino e Validação Excitados
    print("\n--- Carregando ensaios de Treino e Validação (RODADA-7) ---")
    train = [load_exp(*e, decim=args.decim) for e in TRAIN]
    val = [load_exp(*e, decim=args.decim) for e in VAL]
    print(f"Treino: {sum(len(e['y']) for e in train)} amostras em {len(train)} ensaios | Validação: {len(val)} ensaios")

    thetas = {}
    fisher_res = {}

    if args.eval or args.plot:
        loaded = load_saved_model_v6()
        if loaded is not None:
            thetas, k_saved, u0_saved, tau_acc_saved, mech_saved = loaded
            if k_saved: k_static = k_saved
            if u0_saved: u0_static = u0_saved
            if tau_acc_saved: tau_acc_fixed = tau_acc_saved
            if mech_saved: mech_dict = mech_saved
            print("\nModelo v6 carregado com sucesso do disco:")
            for m, th in thetas.items():
                print_params_v6(m, th, k_static, u0_static, tau_acc_fixed)

    if not thetas:
        all_modes = [
            "turbina fixada",
            "turbina regularizada",
            "assimetria motriz",
            "simetrico v5"
        ]

        loaded = load_saved_model_v6()
        if loaded is not None:
            saved_thetas = loaded[0]
            for m in all_modes:
                if m in saved_thetas and (args.mode is None or m != args.mode) and not args.all:
                    thetas[m] = saved_thetas[m]
                    print_params_v6(f"{m} (reutilizado do disco)", thetas[m], k_static, u0_static, tau_acc_fixed)

        modos_para_rodar = [args.mode] if args.mode else all_modes

        for mode in modos_para_rodar:
            if mode in thetas:
                continue

            print(f"\n--- Otimização Multi-Experimento (v6): {mode.upper()} ---")
            th_init = None
            if "turbina fixada" in thetas and mode != "turbina fixada":
                th_init = thetas["turbina fixada"]

            th_opt, cost_val, _, _ = identify_v6(
                train, k_static, u0_static, tau_acc_fixed, mode=mode,
                mech_params=mech_dict, lam=LAMBDA_C3, lam_mech=LAMBDA_MECH,
                lam_aero=LAMBDA_AERO, lam_motor=LAMBDA_MOTOR, th_init=th_init,
                seg_len=SEG_LEN, tol=args.tol, max_iter=args.max_iter,
                time_limit=args.time_limit
            )
            thetas[mode] = th_opt
            print_params_v6(mode, thetas[mode], k_static, u0_static, tau_acc_fixed)

            # Salva imediatamente
            save_model_v6(thetas, k_static, u0_static, tau_acc_fixed,
                          mech_dict=mech_dict, rest_dict=rest_dict, fisher_dict=fisher_res)

            # Análise de Fisher
            fisher_res[mode] = compute_fisher_v6(
                thetas[mode], train, k_static, u0_static, tau_acc_fixed, mode=mode
            )
            save_model_v6(thetas, k_static, u0_static, tau_acc_fixed,
                          mech_dict=mech_dict, rest_dict=rest_dict, fisher_dict=fisher_res)

    # Simulação e Validação em 3 Modos: Free-run, Janela e Observador (Luenberger 2ª Ordem)
    rk4_fn = build_rk4_v6(u0_fixed=u0_static, tau_acc_fixed=tau_acc_fixed)
    results = {}
    sim_kinds = [
        ("freerun", "Free-run (Malha Aberta Pura)",
         lambda th, e: free_run(th, e, rk4_fn, u0_static, tau_acc_fixed)),
        ("janela", f"Janelas de {SEG_LEN:g} s",
         lambda th, e: windowed_run(th, e, rk4_fn, SEG_LEN)),
        ("observador", f"Preditor com Observador Luenberger (L={args.L})",
         lambda th, e: observer_run(th, e, rk4_fn, args.L)),
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

    # Geração dos Gráficos Comparativos
    print("\n--- Gerando gráficos comparativos (v6) ---")
    for group_name, exps in (("TREINO", train), ("VALIDAÇÃO", val)):
        prefix = "v6_treino" if group_name == "TREINO" else "v6_validacao"
        dc45_exps = [e for e in exps if "45" in e['name']]
        dc60_exps = [e for e in exps if "60" in e['name']]

        for kind, kind_label, _ in sim_kinds:
            _, sims, metrics, _ = results[(group_name, kind)]
            if dc45_exps:
                dc45_sims = {m: [sims[m][exps.index(e)] for e in dc45_exps] for m in thetas}
                plot_group_v6(dc45_exps, dc45_sims, metrics,
                              f"v6: {group_name} DC=45% — {kind_label}",
                              f"{prefix}_dc45_{kind}.png")
            if dc60_exps:
                dc60_sims = {m: [sims[m][exps.index(e)] for e in dc60_exps] for m in thetas}
                plot_group_v6(dc60_exps, dc60_sims, metrics,
                              f"v6: {group_name} DC=60% — {kind_label}",
                              f"{prefix}_dc60_{kind}.png")

    print("\n[CONCLUÍDO] Identificação e validação da Turbina EDF v6 finalizadas com sucesso!")


if __name__ == "__main__":
    main()
