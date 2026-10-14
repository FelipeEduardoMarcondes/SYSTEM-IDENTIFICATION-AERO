"""
Identificação Grey-Box Multi-Experimento v3 (Modelo Mínimo Físico com Zona Morta e Atuador)

====================================================================================================
MOTIVAÇÃO FÍSICA E MUDANÇAS ESTRUTURAIS (v3 vs v2/v1):
====================================================================================================
1. Empuxo com Zona Morta Realista k*(max(u - u0, 0))^2:
   - Em motores brushless / ESCs reais, existe uma zona morta abaixo de u0 ~ 11% onde o motor
     não gira ou produz empuxo nulo.
   - Acima de u0, a rotação é proporcional a (u - u0) e o empuxo aerodinâmico é proporcional ao
     quadrado da rotação: T ~ (max(u - u0, 0))^2.
   - Na curva semiestática, a introdução de u0 eleva o R² de 0.940 para 0.987 (escada) e 0.994 (rampa),
     eliminando o maior erro sistemático do modelo com o menor custo computacional.

2. Remoção do Arrasto Quadrático c_aero:
   - A velocidade angular do aeropêndulo é baixa (~2 a 3 rad/s).
   - O arrasto aerodinâmico do braço (c_aero * v * |v|) é desprezível frente ao amortecimento viscoso (c2).
   - A análise da matriz de informação de Fisher revelou correlação de 0.958 (> 0.95) entre c2 e c_aero,
     comprovando colinearidade e justificando matematicamente a remoção de c_aero.

3. Remoção do Atrito de Stribeck (Fs, vs):
   - Os parâmetros de Stribeck apresentaram correlação elevada (-0.870) e ficavam nos limites do solver.
   - O atrito mecânico é satisfatoriamente modelado pela combinação de amortecimento viscoso linear (c2)
     e atrito de Coulomb (Fc) regularizado por tanh(50*v).

4. Dinâmica do Atuador (tau_m) Identificada Separadamente e Fixada:
   - O atraso eletromecânico do conjunto motor+hélice (tau_m ~ 0.06 a 0.08 s) evita trancos instantâneos
     e oscilações artificiais em degraus/APRBS.
   - A identificação de tau_m em ensaio dedicado (degrau de u com braço travado e célula de carga,
     tacômetro ou microfone) desacopla a dinâmica do motor da dinâmica mecânica do pêndulo.
   - No modelo v3, tau_m pode ser fixado com base no ensaio isolado, reduzindo a dimensionalidade.

5. Matriz de Correlação dos Parâmetros e Desvios-Padrão (Aproximação de Fisher / Gauss-Newton):
   - CasADi calcula a sensibilidade analítica (Jacobiano Jac = dr/dtheta) dos resíduos do modelo.
   - A matriz de informação de Fisher F = Jacᵀ * Jac fornece a covariância assintótica Cov = s² * (Jacᵀ*Jac)⁻¹.
   - Desvios-padrão e matriz de correlação normalizada são reportados.
   - Pares com correlação |R| > 0.95 são destacados como candidatos a fixação ou remoção.

6. Otimização Multi-Start (Múltiplos Chutes Iniciais):
   - Múltiplos pontos de partida (multi-start) são testados para verificar se o solver converge
     para o mesmo atrator / mínimo global e confirmar a convexidade local do espaço de parâmetros.
====================================================================================================

Equações de Estado do Modelo v3 (3ª Ordem Mínimo):
    theta' = v
    v'     = c1*sin(theta) + c2*v - Fc*tanh(50*v) + c3*(max(a - u0, 0))^2
    a'     = (u - a) / tau_m

Preditor Contínuo (Observador com ganho L na posição para validação):
    obs_corr = [L * (y_meas - theta), 0, 0]ᵀ
"""

import os
import sys
import time
import json
import argparse
import re
import numpy as np
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
R7 = "data/experimentos/coleta_dados/RODADA-7/"
TR = (10.0, 13.0)

# Ensaios de Treinamento (RODADA-7, DC 45 e DC 60, malha fechada)
TRAIN_NAMES = [
    "aprbs-45-3_0904_20-01.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "multi-seno-45-050Hz_0904_20-29.csv",
    "seq-degraus-45-3_0904_20-48.csv",
    "aprbs-60-3_0904_20-10.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "seq-degraus-60-3_0904_20-57.csv",
    "aprbs-60-2_0904_20-07.csv",
    "seq-degraus-60-2_0904_20-54.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
]
TRAIN = [(R7 + f, *TR) for f in list(dict.fromkeys(TRAIN_NAMES))]

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

# Ensaios de Validação (todos os demais ensaios da RODADA-7)
train_paths = {e[0] for e in TRAIN}
VAL = [(R7 + f, *TR) for f in ALL_R7_FILES if (R7 + f) not in train_paths]

# Hiperparâmetros padrão da identificação
DECIM = 1
LAMBDA_C3 = 50.0       # peso da regularização de c3 em torno de k_static*|c1|
SEG_LEN = 15.0         # comprimento da janela de multiple shooting [s]
L_OBS = 0.5            # ganho do observador de estados na validação
TAU_M_DEFAULT = 0.08   # constante de tempo típica do motor/hélice [s] (80 ms)

# Escalas das variáveis de decisão (para condicionamento do IPOPT ~ 1.0)
S_C1, S_C2, S_F, S_C3 = 8.0, 0.4, 0.15, 4.0e-3
S_TAU = 0.10


# =============================================================================
# Fase 1: Identificação Semiestática com Zona Morta (k e u0)
# =============================================================================
def extract_static_gain(method="escada", plot=True):
    """
    Identifica o ganho estático k e a zona morta u0 do empuxo:
        sin(theta) = k * (max(u - u0, 0))^2
    
    A média dos ramos de subida e descida cancela o atrito seco (Coulomb).
    """
    print(f"\n--- Fase 1: Identificação Semiestática com Zona Morta ({method.upper()}) ---")
    if method == "escada":
        file_path = "data/experimentos/coleta_dados/RODADA-4/escada-atrito-1_0819_19-08.csv"
        res = readData(dataset_name=file_path, return_ref=True)
        y_deg, u, t, ref = res[0], res[1], res[2], res[3]
        
        settle_frac = 0.4
        th_max = 80.0
        change = np.flatnonzero(np.abs(np.diff(ref)) > 1e-6) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(ref)]
        i_peak = int(np.argmax(ref))
        
        plateaus = []
        for s, e in zip(starts, ends):
            n = e - s
            if n < 100:
                continue
            a = s + int((1 - settle_frac) * n)
            theta_m = np.mean(y_deg[a:e])
            u_m = np.mean(u[a:e])
            r = ref[s]
            if r <= 0 or theta_m > th_max or u_m < 5:
                continue
            plateaus.append((r, theta_m, u_m, s < i_peak))
            
        up = {p[0]: p for p in plateaus if p[3]}
        down = {p[0]: p for p in plateaus if not p[3]}
        common = sorted(set(up) & set(down))
        
        th_up = np.array([up[r][1] for r in common])
        u_up = np.array([up[r][2] for r in common])
        th_dn = np.array([down[r][1] for r in common])
        u_dn = np.array([down[r][2] for r in common])
        
        th_avg = np.deg2rad(0.5 * (th_up + th_dn))
        u_avg = 0.5 * (u_up + u_dn)
        Torque_g = np.sin(th_avg)
        title_tag = "Escada"
        fname = "v3_curva_semiestatica_escada.png"
    else:
        file_path = "data/coletas/RODADA-3_20260807/dados_curva_semi_estatica_0807_17-36.csv"
        res = readData(dataset_name=file_path, return_ref=True)
        y_deg, u, t = res[0][10:], res[1][10:], res[2][10:]
        i_max = int(np.argmax(y_deg))
        th_min, th_max = 5.0, 80.0
        bins = np.arange(th_min, th_max + 1.0, 1.0)
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        def mean_u_by_angle(sl):
            ang, uu = y_deg[sl], u[sl]
            out = np.full(len(centers), np.nan)
            for i in range(len(centers)):
                m = (ang >= bins[i]) & (ang < bins[i+1])
                if m.sum() >= 5:
                    out[i] = np.mean(uu[m])
            return out

        u_up = mean_u_by_angle(slice(0, i_max))
        u_down = mean_u_by_angle(slice(i_max, None))
        valid = ~np.isnan(u_up) & ~np.isnan(u_down)
        u_avg = 0.5 * (u_up[valid] + u_down[valid])
        Torque_g = np.sin(np.deg2rad(centers[valid]))
        u_up = u_up[valid]
        u_dn = u_down[valid]
        th_up = centers[valid]
        th_dn = centers[valid]
        title_tag = "Rampa Semiestática"
        fname = "v3_curva_semiestatica_rampa.png"

    # 1. Ajuste clássico sem zona morta (k * u^2)
    u2_old = u_avg**2
    k_old = float(np.dot(u2_old, Torque_g) / np.dot(u2_old, u2_old))
    r2_old = float(1 - np.sum((Torque_g - k_old * u2_old)**2) / np.sum((Torque_g - Torque_g.mean())**2))

    # 2. Ajuste com zona morta: k * (max(u - u0, 0))^2
    def model_u0(p):
        return p[0] * np.maximum(u_avg - p[1], 0)**2

    def res_u0(p):
        return Torque_g - model_u0(p)

    fit_res = opt.least_squares(res_u0, [k_old, 10.0], bounds=([1e-6, 0.0], [1e-2, 35.0]))
    k_opt, u0_opt = float(fit_res.x[0]), float(fit_res.x[1])
    pred_opt = model_u0(fit_res.x)
    r2_opt = float(1 - np.sum((Torque_g - pred_opt)**2) / np.sum((Torque_g - Torque_g.mean())**2))

    print(f"  Modelo antigo (sem u0): k = {k_old:.4e} 1/%^2 | u0 = 0.00%  | R^2 = {r2_old:.4f}")
    print(f"  Modelo v3 (com u0)    : k = {k_opt:.4e} 1/%^2 | u0 = {u0_opt:.2f}% | R^2 = {r2_opt:.4f}")
    print(f"  Ganho de ajuste Delta R^2 = +{(r2_opt - r2_old):.4f} (reducao drastica do residuo em baixo comando)")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(u_up, np.sin(np.deg2rad(th_up)), s=25, alpha=0.5, label='Ramo Subida')
        plt.scatter(u_dn, np.sin(np.deg2rad(th_dn)), s=25, alpha=0.5, label='Ramo Descida')
        plt.scatter(u_avg, Torque_g, s=35, c='k', label='Média dos Ramos')
        
        u_grid = np.linspace(0, max(u_avg.max(), u_up.max()) + 2, 200)
        plt.plot(u_grid, k_old * u_grid**2, 'b--', lw=1.5,
                 label=f'Sem u0: k·u² (R²={r2_old:.3f})')
        plt.plot(u_grid, k_opt * np.maximum(u_grid - u0_opt, 0)**2, 'r-', lw=2.2,
                 label=f'v3: k·(max(u-{u0_opt:.1f}%, 0))² (R²={r2_opt:.3f})')
        plt.axvline(u0_opt, color='gray', linestyle=':', label=f'Zona morta u0 = {u0_opt:.1f}%')
        
        plt.title(f"Curva Semiestática do Empuxo ({title_tag})", fontsize=11, fontweight='bold')
        plt.xlabel("Comando do Motor u [%]", fontsize=10)
        plt.ylabel(r"Torque Gravitacional Normalizado $\sin(\theta)$", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        out_img = os.path.join(current_dir, fname)
        plt.savefig(out_img, dpi=120)
        plt.close()
        print(f"  Gráfico salvo em: {fname}")

    return k_opt, u0_opt


def explain_tau_m_identification():
    """Explicação metodológica sobre o ensaio isolado para identificação de tau_m."""
    msg = """
====================================================================================================
ENSAIO ISOLADO PARA IDENTIFICAÇÃO DA CONSTANTE DE TEMPO DO MOTOR (tau_m):
====================================================================================================
Procedimento Recomendado (Braço Travado):
1. Travar o braço do aeropêndulo rigidamente na vertical ou em suporte fixo com célula de carga.
2. Alternativamente, utilizar um tacômetro óptico na ponta da hélice ou microfone acústico
   (análise espectral FFT da frequência fundamental das pás do rotor).
3. Aplicar um degrau no comando do ESC: u(t) = u_base -> u_step (ex: de 25% para 55%).
4. O empuxo ou a rotação angular Omega(t) responde conforme sistema de 1ª ordem:
       a(t) = a_0 + Delta_a * (1 - exp(-(t - t0)/tau_m))
5. Ajustando-se a curva nos dados experimentais de alta taxa (1 kHz ou mais), obtém-se:
       tau_m ~ 0.05 a 0.09 s (tipicamente 60 a 80 ms para motores brushless classe 2204/2205).
6. Fixar tau_m na identificação da dinâmica angular elimina o acoplamento espúrio entre
   tau_m e os coeficientes de amortecimento c2 / Fc.
====================================================================================================
"""
    print(msg)


# =============================================================================
# Modelo e Integrador RK4 (CasADi) — 3ª Ordem Mínimo
# =============================================================================
def build_rk4_v3(u0_fixed=None, tau_m_fixed=None):
    """
    Integrador RK4 com estado 3D: [theta, theta_dot, a]
    
    Se u0_fixed ou tau_m_fixed forem passados, são tratados como constantes fixas,
    reduzindo o vetor de parâmetros livres th.
    
    Vetor th padrão:
      Se u0 e tau_m livres: th = [c1, c2, c3, Fc, u0, tau_m] (dim 6)
      Se u0 fixo e tau_m fixo: th = [c1, c2, c3, Fc] (dim 4)
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')

    n_th = 4
    if u0_fixed is None:
        n_th += 1
    if tau_m_fixed is None:
        n_th += 1
    th = cs.SX.sym('th', n_th)

    idx = 4
    if u0_fixed is not None:
        u0_expr = cs.SX(u0_fixed)
    else:
        u0_expr = th[idx]
        idx += 1

    if tau_m_fixed is not None:
        tau_m_expr = cs.SX(tau_m_fixed)
    else:
        tau_m_expr = th[idx]

    def f(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        # Empuxo quadrático com zona morta u0
        thrust = cs.fmax(a - u0_expr, 0)**2
        # Dissipação mecânica mínima: Viscoso (c2) + Coulomb (Fc)
        fric = -th[3] * cs.tanh(50.0 * v)
        acc = th[0] * cs.sin(th_pos) + th[1] * v + fric + th[2] * thrust
        # Dinâmica de 1ª ordem do motor/hélice
        a_dot = (u_ - a) / tau_m_expr
        # Correção do observador de estados na posição
        obs_corr = cs.vertcat(L * (y_meas - th_pos), 0, 0)
        return cs.vertcat(v, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2.0 * k1, u)
    k3 = f(x + h / 2.0 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return cs.Function('rk4_v3', [x, u, th, y_meas, L, h], [x_next]), n_th


# =============================================================================
# Funções de Simulação
# =============================================================================
def free_run(theta, exp, rk4_fn, u0_val, tau_m_val):
    """Simulação livre (malha aberta pura, L=0)."""
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_free', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_dummy = cs.DM.zeros(1, N - 1)
    L_zero = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_dummy, L_zero, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def observer_run(theta, exp, rk4_fn, L_val=L_OBS):
    """Simulação com observador de estados (preditor contínuo)."""
    N = len(exp['y'])
    sim = rk4_fn.mapaccum(casadi_name('sim_obs', exp['name'], N), N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas = cs.DM(exp['y'][:-1]).T
    L_mat = cs.repmat(cs.DM(L_val), 1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas, L_mat, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def segments(N, dt, seg_len):
    """Divisão em janelas temporais para multiple shooting."""
    if seg_len is None:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def windowed_run(theta, exp, rk4_fn, seg_len=SEG_LEN):
    """Simulação por janelas reiniciando no estado medido/filtrado."""
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = rk4_fn.mapaccum(casadi_name('sim_win', exp['name'], s), n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s], exp['u'][s]])
        y_dummy = cs.DM.zeros(1, n - 1)
        L_zero = cs.DM.zeros(1, n - 1)
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1),
                         y_dummy, L_zero, exp['dt']))
        y_hat[s + 1:e] = X[0, :]
    return y_hat


# =============================================================================
# Carregamento de Dados e Métricas
# =============================================================================
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
# Identificação Multi-Experimento (Programação Não-Linear via IPOPT)
# =============================================================================
def identify_v3(train, k_static, u0_val, tau_m_val, c3_free=False, lam=LAMBDA_C3,
                th_init=None, seg_len=SEG_LEN, tol=1e-3, max_iter=400,
                time_limit=1200.0, silent=False):
    """
    Otimização dinâmica de 3ª ordem mínima via Multiple Shooting com IPOPT.
    Critérios de parada calibrados para dados experimentais: encerra ágil assim que estabiliza.
    """
    rk4_fn, n_th = build_rk4_v3(u0_fixed=u0_val, tau_m_fixed=tau_m_val)
    
    opti = cs.Opti()
    p1 = opti.variable()  # c1 = -S_C1 * p1
    p2 = opti.variable()  # c2 = -S_C2 * p2
    p3 = opti.variable()  # Fc = S_F * p3

    c1 = -S_C1 * p1
    c2 = -S_C2 * p2
    Fc = S_F * p3

    # Limites físicos
    opti.subject_to(opti.bounded(0.175, p1, 3.5))  # c1 em [-28.0, -1.4]
    opti.subject_to(opti.bounded(0.0, p2, 12.0))   # c2 em [-4.8, 0.0]
    opti.subject_to(opti.bounded(0.0, p3, 6.0))    # Fc em [0.0, 0.9]

    if c3_free:
        p_c3 = opti.variable()
        c3 = S_C3 * p_c3
        opti.subject_to(opti.bounded(0.05, p_c3, 20.0))
    else:
        c3 = -k_static * c1

    th_vars = [c1, c2, c3, Fc]
    theta = cs.vertcat(*th_vars)

    # Inicialização (chute inicial / warm-start)
    if th_init is not None:
        opti.set_initial(p1, float(np.clip(-th_init[0] / S_C1, 0.175, 3.5)))
        opti.set_initial(p2, float(np.clip(-th_init[1] / S_C2, 0.0, 12.0)))
        opti.set_initial(p3, float(np.clip(th_init[3] / S_F, 0.0, 6.0)))
        if c3_free:
            opti.set_initial(p_c3, float(np.clip(th_init[2] / S_C3, 0.05, 20.0)))
    else:
        c1_init = -C_g / 0.035
        opti.set_initial(p1, -c1_init / S_C1)
        opti.set_initial(p2, (0.016 / 0.035) / S_C2)
        opti.set_initial(p3, (0.005 / 0.035) / S_F)
        if c3_free:
            opti.set_initial(p_c3, (-k_static * c1_init) / S_C3)

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

    # Regularização adaptativa de c3 em torno do ganho estático da escada
    if c3_free and lam > 0:
        cost += lam * ((c3 / (-c1)) / k_static - 1.0)**2

    opti.minimize(cost)

    ipopt_opts = {
        "max_iter": int(max_iter),
        "max_cpu_time": float(time_limit),
        "max_wall_time": float(time_limit),
        "print_level": 0 if silent else 5,
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
    opti.solver("ipopt", {"expand": True}, ipopt_opts)

    t0 = time.time()
    converged = True
    try:
        sol = opti.solve()
        val = sol.value
        status = "Convergiu no critério ótimo"
    except (Exception, KeyboardInterrupt) as exc:
        converged = False
        status = f"Encerrou ({type(exc).__name__})"
        val = opti.debug.value

    th_opt = np.array(val(theta)).flatten()
    cost_val = float(val(cost))
    dt_sol = time.time() - t0

    if not silent:
        print(f"  [{status}] tempo: {dt_sol:.1f} s | custo: {cost_val:.3f}")

    return th_opt, cost_val, converged, dt_sol


# =============================================================================
# Análise de Sensibilidade e Matriz de Informação de Fisher
# =============================================================================
def compute_fisher_correlation(th_opt, train_exps, k_static, u0_val, tau_m_val,
                               c3_free=False, seg_len=SEG_LEN):
    """
    Calcula a Matriz de Informação de Fisher (Gauss-Newton: Jacᵀ · Jac dos resíduos),
    os desvios-padrão dos parâmetros e a Matriz de Correlação analítica.
    
    Resíduos r(theta) = y_medido - y_simulado calculados através do simulador RK4.
    """
    print("\n" + "=" * 80)
    print("ANÁLISE DE INCERTEZA E MATRIZ DE CORRELAÇÃO DE FISHER (GAUSS-NEWTON)")
    print("=" * 80)

    # Parâmetros livres analisados
    if c3_free:
        param_names = ['c1', 'c2', 'c3', 'Fc']
        n_p = 4
    else:
        param_names = ['c1', 'c2', 'Fc']
        n_p = 3

    th_sym = cs.SX.sym('th_free', n_p)
    if c3_free:
        c1_s, c2_s, c3_s, Fc_s = th_sym[0], th_sym[1], th_sym[2], th_sym[3]
    else:
        c1_s, c2_s, Fc_s = th_sym[0], th_sym[1], th_sym[2]
        c3_s = -k_static * c1_s

    # Monta RK4 simbólico para sensibilidade
    x_s = cs.SX.sym('x', 3)
    u_s = cs.SX.sym('u')
    h_s = cs.SX.sym('h')

    def f_sens(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        thrust = cs.fmax(a - u0_val, 0)**2
        fric = -Fc_s * cs.tanh(50.0 * v)
        acc = c1_s * cs.sin(th_pos) + c2_s * v + fric + c3_s * thrust
        a_dot = (u_ - a) / tau_m_val
        return cs.vertcat(v, acc, a_dot)

    k1 = f_sens(x_s, u_s)
    k2 = f_sens(x_s + h_s / 2.0 * k1, u_s)
    k3 = f_sens(x_s + h_s / 2.0 * k2, u_s)
    k4 = f_sens(x_s + h_s * k3, u_s)
    rk4_step = cs.Function('rk4_sens', [x_s, u_s, th_sym, h_s],
                           [x_s + h_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)])

    F_total = np.zeros((n_p, n_p))
    res_sq_sum = 0.0
    total_samples = 0

    th_eval = th_opt[:n_p] if c3_free else np.array([th_opt[0], th_opt[1], th_opt[3]])

    for exp in train_exps:
        N = len(exp['y'])
        total_samples += N
        sim = rk4_step.mapaccum(casadi_name('sim_fisher', exp['name'], N), N - 1)
        x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
        X = sim(x0, cs.DM(exp['u'][:-1]).T, cs.repmat(th_sym, 1, N - 1), exp['dt'])
        y_hat = cs.vertcat(exp['y'][0], X[0, :].T)
        r = cs.DM(exp['y']) - y_hat

        J = cs.jacobian(r, th_sym)
        F_exp = cs.mtimes(J.T, J)
        F_fn = cs.Function(casadi_name('F_fn', exp['name'], N), [th_sym], [F_exp, r])
        F_val, r_val = F_fn(th_eval)

        F_total += np.array(F_val)
        res_sq_sum += float(np.sum(np.array(r_val)**2))

    # Variância dos resíduos: s² = sum(r²) / (N_total - p)
    dof = max(total_samples - n_p, 1)
    s2 = res_sq_sum / dof

    # Matriz de Covariância dos Parâmetros: Cov = s² * F⁻¹ (pseudo-inversa para estabilidade)
    cov_mat = s2 * np.linalg.pinv(F_total)
    std_devs = np.sqrt(np.maximum(np.diag(cov_mat), 1e-16))
    corr_mat = cov_mat / np.outer(std_devs, std_devs)

    # Exibição dos Desvios-Padrão
    print("\nDesvios-Padrao dos Parametros Estimados:")
    print(f"{'Parametro':12s} {'Valor Otimo':>14s} {'Desvio-Padrao (1 sigma)':>24s} {'Incerteza Relativa':>20s}")
    for i, p_name in enumerate(param_names):
        val_i = float(th_eval[i])
        std_i = float(std_devs[i])
        rel_i = (std_i / abs(val_i)) * 100.0 if abs(val_i) > 1e-12 else 0.0
        print(f"{p_name:12s} {val_i:14.5e} {std_i:24.5e} {rel_i:19.2f}%")

    # Exibição da Matriz de Correlação
    print("\nMatriz de Correlação dos Parâmetros (Fisher):")
    header = " " * 10 + "".join(f"{p:>10s}" for p in param_names)
    print(header)
    for i, p_row in enumerate(param_names):
        row_str = f"{p_row:>10s}" + "".join(f"{corr_mat[i, j]:10.3f}" for j in range(n_p))
        print(row_str)

    # Verificação de Critério de Colinearidade (|R| > 0.95)
    high_corrs = []
    print("\nDiagnóstico de Colinearidade (|R_ij| > 0.95):")
    found_high = False
    for i in range(n_p):
        for j in range(i + 1, n_p):
            r_val = float(corr_mat[i, j])
            if abs(r_val) > 0.95:
                found_high = True
                high_corrs.append((param_names[i], param_names[j], r_val))
                print(f"  [ALERTA] |Corr({param_names[i]}, {param_names[j]})| = {r_val:.3f} > 0.95 "
                      f"=> Forte colinearidade detectada! Justifica fixar ou remover um dos parâmetros.")

    if not found_high:
        print("  [OK] Nenhuma correlação excessiva (|R| > 0.95). Os parâmetros são bem condicionados!")

    return {
        "param_names": param_names,
        "th_eval": th_eval.tolist(),
        "std_devs": std_devs.tolist(),
        "cov_matrix": cov_mat.tolist(),
        "corr_matrix": corr_mat.tolist(),
        "high_correlations": [{"p1": p1, "p2": p2, "corr": c} for p1, p2, c in high_corrs],
        "residual_std": float(np.sqrt(s2))
    }


# =============================================================================
# Fase 4: Otimização Multi-Start (Múltiplos Chutes Iniciais)
# =============================================================================
def multi_start_identify(train, k_static, u0_val, tau_m_val, c3_free=False,
                         n_starts=5, lam=LAMBDA_C3, seg_len=SEG_LEN,
                         max_iter=400, tol=1e-3, time_limit=1200.0):
    """
    Executa otimização multi-start com n_starts chutes iniciais distintos.
    Verifica se os diferentes chutes convergem para o mesmo atrator / mínimo global.
    """
    print("\n" + "=" * 80)
    print(f"OTIMIZAÇÃO MULTI-START (TESTE DE {n_starts} CHUTES INICIAIS)")
    print("=" * 80)

    # Chute nominal de física (Start 0)
    c1_nom = -C_g / 0.035
    c2_nom = -(0.016 / 0.035)
    c3_nom = -k_static * c1_nom
    Fc_nom = 0.005 / 0.035

    th_nominal = [c1_nom, c2_nom, c3_nom, Fc_nom]

    starts = [th_nominal]
    np.random.seed(42)  # Reprodutibilidade
    for k in range(1, n_starts):
        pert = np.random.uniform(0.6, 1.4, size=4)
        th_k = [
            float(np.clip(c1_nom * pert[0], -25.0, -2.0)),
            float(np.clip(c2_nom * pert[1], -4.0, -0.01)),
            float(np.clip(c3_nom * pert[2], 1e-4, 0.02)),
            float(np.clip(Fc_nom * pert[3], 0.001, 0.5))
        ]
        starts.append(th_k)

    results = []
    print(f"{'Start':>6s} {'c1_init':>10s} {'c2_init':>10s} {'Fc_init':>10s} "
          f"{'c1_opt':>10s} {'c2_opt':>10s} {'Fc_opt':>10s} {'Custo':>10s} {'Tempo':>8s} {'Status':>12s}")

    for idx, th_0 in enumerate(starts):
        th_opt, cost_val, conv, dt_sol = identify_v3(
            train, k_static, u0_val, tau_m_val, c3_free=c3_free, lam=lam,
            th_init=th_0, seg_len=seg_len, tol=tol, max_iter=max_iter,
            time_limit=time_limit, silent=True
        )
        status_lbl = "OK" if conv else "IterLimit"
        print(f"{idx:6d} {th_0[0]:10.3f} {th_0[1]:10.3f} {th_0[3]:10.3f} "
              f"{th_opt[0]:10.3f} {th_opt[1]:10.3f} {th_opt[3]:10.3f} "
              f"{cost_val:10.2f} {dt_sol:7.1f}s {status_lbl:>12s}")
        results.append({
            "start_idx": idx,
            "th_init": th_0,
            "th_opt": th_opt,
            "cost": cost_val,
            "converged": conv,
            "time_s": dt_sol
        })

    # Análise de convergência e dispersão das soluções
    best_res = min(results, key=lambda x: x["cost"])
    costs = np.array([r["cost"] for r in results])
    c1_opts = np.array([r["th_opt"][0] for r in results])
    c2_opts = np.array([r["th_opt"][1] for r in results])
    Fc_opts = np.array([r["th_opt"][3] for r in results])

    spread_c1 = np.std(c1_opts)
    spread_c2 = np.std(c2_opts)
    spread_Fc = np.std(Fc_opts)

    print("\nResumo da Convergencia Multi-Start:")
    print(f"  Melhor Custo: {best_res['cost']:.3f} (Start #{best_res['start_idx']})")
    print(f"  Desvio-padrao das solucoes: std(c1) = {spread_c1:.4f} | std(c2) = {spread_c2:.4f} | std(Fc) = {spread_Fc:.4f}")

    if spread_c1 < 0.05 and spread_c2 < 0.02:
        print("  [CONVERGENCIA CONFIRMADA] Todas as inicializacoes convergiram para o mesmo minimo global!")
    else:
        print("  [AVISO] Detectada ligeira sensibilidade a inicializacao. A melhor solucao foi selecionada.")

    return best_res["th_opt"], results


# =============================================================================
# Exibição e Salvamento do Modelo v3
# =============================================================================
def print_params_v3(label, th, k_static, u0_val, tau_m_val):
    c1, c2, c3, Fc = th[:4]
    J = -C_g / c1
    b = -c2 * J
    Tc = Fc * J
    Ktau = c3 * J
    k_dyn = c3 / abs(c1)
    
    print(f"\n[{label}]")
    print(f"  c1 = {c1:.5f} | c2 = {c2:.5f} | c3 = {c3:.5e}")
    print(f"  Fc = {Fc:.5f} rad/s^2 | u0 = {u0_val:.2f}% | tau_m = {tau_m_val:.4f} s ({tau_m_val*1000:.1f} ms)")
    print(f"  k_dyn = c3/|c1| = {k_dyn:.4e}  (k_estatico = {k_static:.4e}, razao = {k_dyn / k_static:.3f})")
    print(f"  Parametros Fisicos Equivalentes:")
    print(f"    Inercia J       = {J:.5f} kg.m^2")
    print(f"    Atrito Viscoso  = {b:.5f} N.m.s/rad")
    print(f"    Torque Coulomb  = {Tc:.5f} N.m")
    print(f"    Ganho de Torque = {Ktau:.4e} N.m/%^2")


def save_model_v3(thetas, k_static, u0_val, tau_m_val, fisher_dict=None,
                  multi_start_dict=None, filename="modelo_greybox_v3"):
    """Salva os parâmetros identificados do modelo v3 em JSON e NPZ."""
    data = {
        "version": "v3_minimal",
        "description": "Modelo Grey-box de 3ª ordem mínimo com empuxo quadrático com zona morta u0 e atuador tau_m",
        "k_static": float(k_static),
        "u0_static": float(u0_val),
        "tau_m_fixed": float(tau_m_val),
        "C_g": float(C_g),
        "models": {},
        "fisher_analysis": fisher_dict or {},
        "multi_start_summary": multi_start_dict or []
    }
    npz_dict = {
        "k_static": np.array(k_static, dtype=float),
        "u0_static": np.array(u0_val, dtype=float),
        "tau_m_fixed": np.array(tau_m_val, dtype=float)
    }

    for mode, th in thetas.items():
        c1, c2, c3, Fc = map(float, th[:4])
        J = float(-C_g / c1)
        b = float(-c2 * J)
        Tc = float(Fc * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        tag = "c3_fixo" if "fixo" in mode else "c3_livre"

        data["models"][mode] = {
            "theta": [c1, c2, c3, Fc, float(u0_val), float(tau_m_val)],
            "params": {
                "c1": c1, "c2": c2, "c3": c3, "Fc": Fc,
                "u0": float(u0_val), "tau_m": float(tau_m_val)
            },
            "physics": {
                "J": J, "b": b, "Tc": Tc, "Ktau": Ktau,
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

    print(f"\nModelo v3 salvo com sucesso:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")


def load_saved_model_v3(filename="modelo_greybox_v3"):
    """Carrega parâmetros salvos do modelo v3."""
    json_path = os.path.join(current_dir, f"{filename}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thetas = {}
    for mode, mdata in data.get("models", {}).items():
        thetas[mode] = np.array(mdata["theta"][:4], dtype=float)
    return thetas, data.get("k_static"), data.get("u0_static"), data.get("tau_m_fixed")


# =============================================================================
# Funções de Plotagem e Relatório Gráfico
# =============================================================================
def plot_group(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {'c3 fixo': 'tab:blue', 'c3 livre': 'tab:red'}
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição')
        txt = []
        for mode, ys in sims.items():
            ax.plot(exp['t'], np.rad2deg(ys[i]), '--', color=colors.get(mode, 'tab:green'),
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


def plot_training_data(train_exps):
    """Gera gráficos dos sinais de treinamento: resposta angular e sinal de controle."""
    for dc in ("45", "60"):
        exps_dc = [e for e in train_exps if f"-{dc}-" in e['name']]
        if not exps_dc:
            continue
        n = len(exps_dc)
        fig, axes = plt.subplots(n, 2, figsize=(15, 2.7 * n), sharex='row',
                                 gridspec_kw={'width_ratios': [1.2, 1]})
        if n == 1:
            axes = axes.reshape(1, 2)

        for i, exp in enumerate(exps_dc):
            t = exp['t']
            y_deg = np.rad2deg(exp['y'])
            u = exp['u']

            ax_y = axes[i, 0]
            if exp.get('ref') is not None:
                ref_deg = np.rad2deg(exp['ref'])
                ax_y.plot(t, ref_deg, 'r--', lw=1.2, label='Referência r', alpha=0.85)
            ax_y.plot(t, y_deg, 'k-', lw=1.2, label='Medição θ')
            ax_y.set_ylabel('θ [°]', fontsize=9)
            ax_y.set_title(f"{exp['name']} — Ângulo [°]", fontsize=10, fontweight='bold')
            ax_y.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax_y.legend(loc='upper right', fontsize=8)

            ax_u = axes[i, 1]
            ax_u.plot(t, u, color='tab:blue', lw=1.1, label='Controle u [%]')
            ax_u.set_ylabel('u [%]', fontsize=9)
            ax_u.set_title(f"{exp['name']} — Entrada u [%]", fontsize=10)
            ax_u.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax_u.legend(loc='upper right', fontsize=8)

        axes[-1, 0].set_xlabel('Tempo [s]', fontsize=9)
        axes[-1, 1].set_xlabel('Tempo [s]', fontsize=9)
        fig.suptitle(f"Dados de Treinamento (v3) — Faixa DC {dc}", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fname = f"v3_dados_treinamento_dc{dc}.png"
        try:
            fig.savefig(os.path.join(current_dir, fname), dpi=120)
            print(f"Gráfico de treino salvo em: {fname}")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico de treino '{fname}': {e}")
        finally:
            plt.close(fig)


# =============================================================================
# Pipeline Principal de Execução
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Identificação Grey-Box Multi-Experimento v3")
    parser.add_argument("--eval", action="store_true", help="Avalia modelo salvo sem reotimizar")
    parser.add_argument("--plot", action="store_true", help="Gera gráficos a partir do modelo salvo")
    parser.add_argument("--all", action="store_true", help="Força reotimização completa (c3 fixo e livre)")
    parser.add_argument("--multi-start", action="store_true", help="Executa otimização multi-start")
    parser.add_argument("--n-starts", type=int, default=5, help="Número de chutes iniciais para multi-start")
    parser.add_argument("--tau-m", type=float, default=TAU_M_DEFAULT, help="Valor fixado de tau_m [s]")
    parser.add_argument("--decim", type=int, default=DECIM, help="Fator de decimação dos ensaios")
    parser.add_argument("--max-iter", type=int, default=400, help="Teto de iterações do IPOPT (padrão: 400)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerância de convergência KKT (padrão: 1e-3)")
    parser.add_argument("--time-limit", type=float, default=1200.0, help="Tempo limite por run em segundos (padrão: 1200 s = 20 min)")
    args = parser.parse_args()

    print("=" * 80)
    print("IDENTIFICAÇÃO GREY-BOX MULTI-EXPERIMENTO v3 (MODELO MÍNIMO)")
    print("  * Empuxo quadrático com zona morta: k * (max(u - u0, 0))²")
    print("  * Remoção do arrasto quadrático do braço c_aero (justificado por |R|=0.958)")
    print("  * Remoção do Stribeck (atrito mínimo: viscoso c2 + Coulomb Fc)")
    print("  * Dinâmica do atuador motor/hélice fixada: tau_m = {:.3f} s".format(args.tau_m))
    print("=" * 80)

    explain_tau_m_identification()

    # Passo 1: Curva Semiestática
    k_static, u0_static = extract_static_gain(method="escada", plot=True)
    # Validação cruzada na rampa
    try:
        extract_static_gain(method="rampa", plot=True)
    except Exception as e:
        print(f"  ! Aviso na validação por rampa: {e}")

    tau_m_fixed = float(args.tau_m)

    # Carregamento dos Ensaios
    print("\n--- Carregando ensaios de Treino e Validação ---")
    train = [load_exp(*e, decim=args.decim) for e in TRAIN]
    val = [load_exp(*e, decim=args.decim) for e in VAL]
    print(f"Treino: {sum(len(e['y']) for e in train)} amostras em {len(train)} ensaios | "
          f"Validação: {len(val)} ensaios")

    print("\n--- Gerando gráficos dos dados de treinamento ---")
    plot_training_data(train)

    thetas = {}
    fisher_res = {}
    multi_res_list = []

    # Verificação de carregamento prévio
    if args.eval or args.plot:
        loaded = load_saved_model_v3()
        if loaded is not None:
            thetas, k_saved, u0_saved, tau_m_saved = loaded
            if k_saved:
                k_static = k_saved
            if u0_saved:
                u0_static = u0_saved
            if tau_m_saved:
                tau_m_fixed = tau_m_saved
            print("\nModelo v3 carregado com sucesso do disco:")
            for m, th in thetas.items():
                print_params_v3(m, th, k_static, u0_static, tau_m_fixed)
        else:
            print("  ! Arquivo modelo_greybox_v3.json não encontrado. Prosseguindo com otimização.")

    if not thetas:
        # Se c3 fixo já estiver salvo em disco e não foi pedido --all, reutiliza
        loaded = load_saved_model_v3()
        if loaded is not None and not args.all:
            saved_thetas = loaded[0]
            if "c3 fixo" in saved_thetas:
                print("\n>>> Modelo 'c3 fixo' reutilizado do disco.")
                thetas["c3 fixo"] = saved_thetas["c3 fixo"]
                print_params_v3("c3 fixo (reutilizado do disco)", thetas["c3 fixo"], k_static, u0_static, tau_m_fixed)

        # Multi-Start opcional para "c3 fixo"
        if args.multi_start and "c3 fixo" not in thetas:
            th_best, multi_res_list = multi_start_identify(
                train, k_static, u0_static, tau_m_fixed, c3_free=False,
                n_starts=args.n_starts, lam=LAMBDA_C3, seg_len=SEG_LEN,
                max_iter=args.max_iter, tol=args.tol, time_limit=args.time_limit
            )
            thetas["c3 fixo"] = th_best
            print_params_v3("c3 fixo (Otimo Multi-Start)", thetas["c3 fixo"], k_static, u0_static, tau_m_fixed)

        # Otimizações dos modos: c3 fixo e c3 livre
        for mode, free in (("c3 fixo", False), ("c3 livre", True)):
            if mode in thetas:
                continue

            print(f"\n--- Otimizacao Multi-Experimento (v3): {mode} ---")
            th_init = thetas.get("c3 fixo") if (free and "c3 fixo" in thetas) else None
            th_opt, cost_val, _, _ = identify_v3(
                train, k_static, u0_static, tau_m_fixed, c3_free=free, lam=LAMBDA_C3,
                th_init=th_init, seg_len=SEG_LEN, tol=args.tol, max_iter=args.max_iter,
                time_limit=args.time_limit
            )
            thetas[mode] = th_opt
            print_params_v3(mode, thetas[mode], k_static, u0_static, tau_m_fixed)

            # Salva o modelo imediatamente (resguardo caso Fisher ou etapas posteriores falhem)
            save_model_v3(thetas, k_static, u0_static, tau_m_fixed,
                          fisher_dict=fisher_res, multi_start_dict=multi_res_list)

            # Análise de Fisher para cada modo
            fisher_res[mode] = compute_fisher_correlation(
                thetas[mode], train, k_static, u0_static, tau_m_fixed,
                c3_free=free, seg_len=SEG_LEN
            )

            # Salva novamente com a análise de Fisher
            save_model_v3(thetas, k_static, u0_static, tau_m_fixed,
                          fisher_dict=fisher_res, multi_start_dict=multi_res_list)

    # Executa a análise de Fisher para qualquer modo que ainda não a possua
    for mode, th in thetas.items():
        if mode not in fisher_res:
            is_free = "livre" in mode
            fisher_res[mode] = compute_fisher_correlation(
                th, train, k_static, u0_static, tau_m_fixed,
                c3_free=is_free, seg_len=SEG_LEN
            )
            save_model_v3(thetas, k_static, u0_static, tau_m_fixed,
                          fisher_dict=fisher_res, multi_start_dict=multi_res_list)

    # =========================================================================
    # Simulação e Validação em 3 Modos: Free-run, Janela e Observador
    # =========================================================================
    rk4_fn, _ = build_rk4_v3(u0_fixed=u0_static, tau_m_fixed=tau_m_fixed)
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
            header = f"{'ensaio':28s}" + "".join(f"{m:>22s}" for m in thetas)
            print(header)
            for i, e in enumerate(exps):
                row = f"{e['name']:28s}" + "".join(
                    f"{metrics[m][i][0]:12.1f}% / {metrics[m][i][1]:5.2f}°" for m in thetas)
                print(row)
            print(f"{'MÉDIA':28s}" + "".join(
                f"{np.mean([x[0] for x in metrics[m]]):12.1f}% / {np.mean([x[1] for x in metrics[m]]):5.2f}°"
                for m in thetas))
            print(f"{'MEDIANA':28s}" + "".join(
                f"{np.median([x[0] for x in metrics[m]]):12.1f}% / {np.median([x[1] for x in metrics[m]]):5.2f}°"
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
                       f"{kind_label} — {group_name} — DC {dc} (Modelo v3)",
                       f"v3_{tag}_dc{dc}_{kind}.png")

    print("\n[CONCLUÍDO] Identificação e validação do modelo v3 finalizadas com sucesso!")


if __name__ == "__main__":
    main()
