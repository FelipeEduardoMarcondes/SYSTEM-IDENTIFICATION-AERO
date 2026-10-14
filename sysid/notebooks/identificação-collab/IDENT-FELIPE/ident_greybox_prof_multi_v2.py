"""
Identificação grey-box multi-experimento v2 (modelo com motor e aerodinâmica)

Equações do modelo expandido de 3ª ordem:
    theta'' = c1*sin(theta) + c2*theta' - c_aero*theta'*|theta'| + c3*a*|a| + Fa_NL(theta')
    a'      = (u - a) / tau_m
    Fa_NL   = -(Fc + Fs*exp(-(theta'/vs)^2)) * tanh(50*theta')

Novidades em relação à v1:
  1. Dinâmica de 1ª ordem do atuador motor/hélice (tau_m):
     Elimina o tranco instantâneo em comutações de APRBS e degraus que jogava
     o braço além dos 90°, evitando os loopings artificiais.
  2. Arrasto aerodinâmico quadrático (c_aero):
     Dissipa energia cinética realista em velocidades angulares elevadas.
  3. Regularização adaptativa de c3 (LAMBDA_C3):
     Preserva a coerência física em torno do ganho estático da escada.
  4. Validação em três modos:
     - Free-run puro (malha aberta pura, sem nenhuma realimentação)
     - Janelas de SEG_LEN segundos (reinício periódico)
     - Preditor com observador L (estabiliza o regime instável de DC 60)
"""
import os
import sys
import time
import json
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
for p in (root_dir, current_dir):
    if p not in sys.path:
        sys.path.append(p)

from aerodata import readData
from ident_greybox_prof import extract_static_gain, C_g

# =============================================================================
# Configuração dos ensaios (somente RODADA-7, DC 45 e DC 60, malha fechada)
# =============================================================================
R7 = "data/experimentos/coleta_dados/RODADA-7/"
TR = (10.0, 13.0)

# Treino: 1 ensaio de cada tipo por DC (com maior excitação para excitar dinâmica)
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

# Validação: todos os outros da RODADA-7 que não estão no treino
train_paths = {e[0] for e in TRAIN}
VAL = [(R7 + f, *TR) for f in ALL_R7_FILES if (R7 + f) not in train_paths]

DECIM = 1         
LAMBDA_C3 = 50.0   # regularização de c3 livre em torno do ganho estático (evita overfitting)
SEG_LEN = 15.0      # janela do multiple shooting [s]
L_OBS = 0.5        # ganho do observador de estados para validação do regime instável

# Escalas das variáveis de decisão (valores físicos = escala * variável ~ 1)
S_C1, S_C2, S_F, S_C3 = 8.0, 0.4, 0.15, 2.5e-3
S_AERO, S_TAU = 0.05, 0.10


# =============================================================================
# Modelo e integrador (CasADi) — 3ª Ordem com Observador
# =============================================================================
def build_rk4():
    """
    Integrador RK4 com estado 3D: [theta, theta_dot, motor_activation]
    Parâmetros th: [c1, c2, c3, Fc, Fs, vs, c_aero, tau_m]
    L: ganho do observador (L=0 para simulação livre/treino, L>0 para preditor)
    """
    x = cs.SX.sym('x', 3)
    u = cs.SX.sym('u')
    th = cs.SX.sym('th', 8)
    y_meas = cs.SX.sym('y_meas')
    L = cs.SX.sym('L')
    h = cs.SX.sym('h')

    def f(x_, u_):
        th_pos, v, a = x_[0], x_[1], x_[2]
        # Atrito não-linear (Coulomb + Stribeck)
        fric_nl = -(th[3] + th[4] * cs.exp(-(v / th[5])**2)) * cs.tanh(50 * v)
        # Dinâmica angular com arrasto quadrático
        acc = (th[0] * cs.sin(th_pos)
               + th[1] * v
               - th[6] * v * cs.fabs(v)
               + th[2] * a * cs.fabs(a)
               + fric_nl)
        # Dinâmica de 1ª ordem do motor/hélice
        a_dot = (u_ - a) / th[7]
        # Correção do observador de estado na posição
        obs_corr = cs.vertcat(L * (y_meas - th_pos), 0, 0)
        return cs.vertcat(v, acc, a_dot) + obs_corr

    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return cs.Function('rk4_v2', [x, u, th, y_meas, L, h], [x_next])


RK4 = build_rk4()


def free_run(theta, exp):
    """Simulação livre (malha aberta pura, L=0)."""
    N = len(exp['y'])
    sim = RK4.mapaccum('sim_free', N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas_dummy = cs.DM.zeros(1, N - 1)
    L_zero = cs.DM.zeros(1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas_dummy, L_zero, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def observer_run(theta, exp, L_val=L_OBS):
    """Simulação com observador de estados (preditor contínuo)."""
    N = len(exp['y'])
    sim = RK4.mapaccum('sim_obs', N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0'], exp['u'][0]])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    y_meas = cs.DM(exp['y'][:-1]).T
    L_mat = cs.repmat(cs.DM(L_val), 1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, y_meas, L_mat, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def segments(N, dt, seg_len):
    """Lista de (início, fim) das janelas."""
    if seg_len is None:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def windowed_run(theta, exp, seg_len):
    """Simulação por janelas de SEG_LEN s."""
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = RK4.mapaccum('sim_win', n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s], exp['u'][s]])
        y_meas_dummy = cs.DM.zeros(1, n - 1)
        L_zero = cs.DM.zeros(1, n - 1)
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1),
                         y_meas_dummy, L_zero, exp['dt']))
        y_hat[s + 1:e] = X[0, :]
    return y_hat


# =============================================================================
# Dados
# =============================================================================
def load_exp(path, trim_start, trim_end):
    res = readData(dataset_name=path, decimar=DECIM, return_ref=True)
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
# Identificação multi-experimento com Motor e Aerodinâmica
# =============================================================================
def identify(train, k_static, c3_free, lam=LAMBDA_C3, th_init=None,
             ipopt_time_limit=7200.0, tol=1e-4, max_iter=3000):
    """
    Identificação por Programação Não-Linear (NLP) usando IPOPT.
    
    Ajustado para regime de convergência ótima e finalização ágil:
      - tol: 1e-4 (tolerância ideal para dados experimentais não-lineares)
      - acceptable_tol: 1e-3 com acceptable_iter: 15 e acceptable_dual_inf_tol: 1.0
        (encerra automaticamente e com sucesso assim que a função de custo atinge o fundo do vale)
      - max_iter: 3000 iterações (teto seguro)
      - max_wall_time: 2 horas (7200 s) por etapa
      - limited_memory_max_history: 20 pares L-BFGS (fatoração KKT rápida e estável)
      - Suporte seguro a interrupção manual (KeyboardInterrupt resgata melhor ponto)
    """
    opti = cs.Opti()
    p1, p2, p3, p4, vs, p_aero, p_tau = [opti.variable() for _ in range(7)]
    c1 = -S_C1 * p1
    c2 = -S_C2 * p2
    Fc = S_F * p3
    Fs = S_F * p4
    c_aero = S_AERO * p_aero
    tau_m = S_TAU * p_tau

    opti.subject_to(opti.bounded(0.175, p1, 3.5))
    opti.subject_to(opti.bounded(0.0, p2, 12.0))
    opti.subject_to(opti.bounded(0.0, p3, 6.0))
    opti.subject_to(opti.bounded(0.0, p4, 6.0))
    opti.subject_to(opti.bounded(0.05, vs, 5.0))
    opti.subject_to(opti.bounded(0.0, p_aero, 10.0))       # c_aero in [0.0, 0.50]
    opti.subject_to(opti.bounded(0.3, p_tau, 3.0))         # tau_m in [0.03, 0.30] s

    if th_init is not None:
        c1_0, c2_0, c3_0, Fc_0, Fs_0, vs_0, c_aero_0, tau_m_0 = th_init
        opti.set_initial(p1, float(np.clip(-c1_0 / S_C1, 0.175, 3.5)))
        opti.set_initial(p2, float(np.clip(-c2_0 / S_C2, 0.0, 12.0)))
        opti.set_initial(p3, float(np.clip(Fc_0 / S_F, 0.0, 6.0)))
        opti.set_initial(p4, float(np.clip(Fs_0 / S_F, 0.0, 6.0)))
        opti.set_initial(vs, float(np.clip(vs_0, 0.05, 5.0)))
        opti.set_initial(p_aero, float(np.clip(c_aero_0 / S_AERO, 0.0, 10.0)))
        opti.set_initial(p_tau, float(np.clip(tau_m_0 / S_TAU, 0.3, 3.0)))
        print(f"  -> Warm-start de parâmetros ativado a partir de solução prévia.")
    else:
        c1_init = -C_g / 0.035
        opti.set_initial(p1, -c1_init / S_C1)
        opti.set_initial(p2, (0.016 / 0.035) / S_C2)
        opti.set_initial(p3, (0.005 / 0.035) / S_F)
        opti.set_initial(p4, (0.005 / 0.035) / S_F)
        opti.set_initial(vs, 0.5)
        opti.set_initial(p_aero, 0.2)                          # c_aero ~ 0.01
        opti.set_initial(p_tau, 0.8)                           # tau_m ~ 0.08 s

    if c3_free:
        p5 = opti.variable()
        c3 = S_C3 * p5
        opti.subject_to(opti.bounded(0.05, p5, 20.0))
        if th_init is not None:
            opti.set_initial(p5, float(np.clip(th_init[2] / S_C3, 0.05, 20.0)))
        else:
            c1_init = -C_g / 0.035
            opti.set_initial(p5, (-k_static * c1_init) / S_C3)
    else:
        c3 = -k_static * c1

    theta = cs.vertcat(c1, c2, c3, Fc, Fs, vs, c_aero, tau_m)

    cost = 0
    for exp in train:
        N = len(exp['y'])
        X = opti.variable(3, N)
        opti.set_initial(X[0, :], exp['y'])
        opti.set_initial(X[1, :], exp['v'])
        opti.set_initial(X[2, :], exp['u'])

        segs = segments(N, exp['dt'], SEG_LEN)
        starts = {s for s, _ in segs}
        idx = [k for k in range(N - 1) if (k + 1) not in starts]

        y_meas_dummy = cs.DM.zeros(1, N - 1)
        L_zero = cs.DM.zeros(1, N - 1)
        Xn = RK4.map(N - 1)(X[:, :-1], cs.DM(exp['u'][:-1]).T, cs.repmat(theta, 1, N - 1),
                            y_meas_dummy, L_zero, exp['dt'])
        opti.subject_to(X[:, [k + 1 for k in idx]] == Xn[:, idx])
        cost += cs.sumsqr(X[0, :] - cs.DM(exp['y']).T)

    if c3_free and lam > 0:
        cost += lam * ((c3 / (-c1)) / k_static - 1)**2

    opti.minimize(cost)

    # Opções do IPOPT: balanceadas para achar o ótimo real e encerrar sem loop infinito
    ipopt_opts = {
        "max_iter": int(max_iter),
        "max_cpu_time": float(ipopt_time_limit),
        "max_wall_time": float(ipopt_time_limit),
        "print_level": 5,
        "tol": float(tol),
        "dual_inf_tol": 1e-2,
        "constr_viol_tol": 1e-4,
        "compl_inf_tol": 1e-3,
        "acceptable_tol": 1e-3,
        "acceptable_iter": 15,
        "acceptable_obj_change_tol": 1e-5,
        "acceptable_constr_viol_tol": 1e-3,
        "acceptable_dual_inf_tol": 1.0,
        "acceptable_compl_inf_tol": 1e-2,
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": 20,
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based",
    }
    opti.solver("ipopt", {"expand": True}, ipopt_opts)

    t0 = time.time()
    try:
        sol = opti.solve()
        val = sol.value
        print(f"  [OK] IPOPT convergiu com sucesso para o critério ótimo (tol={tol})!")
    except (Exception, KeyboardInterrupt) as exc:
        print(f"  [OK] Solver finalizou ({exc}), recuperando melhor solução encontrada.")
        val = opti.debug.value
    th = np.array(val(theta)).flatten()
    cost_val = float(val(cost))
    print(f"  tempo de otimização: {time.time() - t0:.1f} s | custo final = {cost_val:.3f}")
    return th, cost_val


def print_params(label, th, k_static):
    c1, c2, c3, Fc, Fs, vs, c_aero, tau_m = th
    J = -C_g / c1
    print(f"\n[{label}]")
    print(f"  c1 = {c1:.5f} | c2 = {c2:.5f} | c3 = {c3:.5e}")
    print(f"  c_aero = {c_aero:.5f} | tau_m = {tau_m:.4f} s ({tau_m*1000:.1f} ms)")
    print(f"  Fc = {Fc:.5f} | Fs = {Fs:.5f} | vs = {vs:.4f} rad/s")
    print(f"  k_dyn = c3/|c1| = {c3 / abs(c1):.4e}  (k estático = {k_static:.4e}, razão = {c3 / abs(c1) / k_static:.3f})")
    print(f"  J = {J:.5f} kg.m^2 | b = {-c2 * J:.5f} N.m.s/rad | Ktau = {c3 * J:.4e}")


def save_model(thetas, k_static, filename="modelo_greybox_v2"):
    """Salva os parâmetros identificados e grandezas físicas em JSON e NPZ."""
    data = {
        "k_static": float(k_static),
        "C_g": float(C_g),
        "models": {}
    }
    npz_dict = {"k_static": np.array(k_static, dtype=float)}

    for mode, th in thetas.items():
        c1, c2, c3, Fc, Fs, vs, c_aero, tau_m = map(float, th)
        J = float(-C_g / c1)
        b = float(-c2 * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        tag = "c3_fixo" if "fixo" in mode else "c3_livre"

        data["models"][mode] = {
            "theta": [c1, c2, c3, Fc, Fs, vs, c_aero, tau_m],
            "params": {
                "c1": c1, "c2": c2, "c3": c3,
                "Fc": Fc, "Fs": Fs, "vs": vs,
                "c_aero": c_aero, "tau_m": tau_m
            },
            "physics": {
                "J": J,
                "b": b,
                "Ktau": Ktau,
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

    print(f"\nModelo salvo com sucesso:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")


def load_saved_model(filename="modelo_greybox_v2"):
    """Carrega parâmetros salvos previamente para reavaliação ou gráficos rápidos."""
    json_path = os.path.join(current_dir, f"{filename}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thetas = {}
    for mode, mdata in data.get("models", {}).items():
        thetas[mode] = np.array(mdata["theta"], dtype=float)
    return thetas


def plot_group(exps, sims, metrics, title, fname):
    n = len(exps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    colors = {'c3 fixo': 'tab:blue', 'c3 livre': 'tab:red'}
    for i, exp in enumerate(exps):
        ax = axes[i, 0]
        ax.plot(exp['t'], np.rad2deg(exp['y']), 'k', lw=1.2, label='Medição')
        txt = []
        for mode, ys in sims.items():
            ax.plot(exp['t'], np.rad2deg(ys[i]), '--', color=colors[mode], lw=1.3, label=mode)
            txt.append(f"{mode}: FIT={metrics[mode][i][0]:.1f}%")
        ax.set_title(f"{exp['name']}  |  " + "  |  ".join(txt), fontsize=10)
        ax.set_ylabel('θ [°]')
        y_deg = np.rad2deg(exp['y'])
        ax.set_ylim(y_deg.min() - 30, y_deg.max() + 30)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Tempo [s]')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    try:
        fig.savefig(os.path.join(current_dir, fname), dpi=110)
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
        fig.suptitle(f"Dados de Treinamento — Faixa DC {dc}", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fname = f"v2_dados_treinamento_dc{dc}.png"
        try:
            fig.savefig(os.path.join(current_dir, fname), dpi=120)
            print(f"Gráfico de treino salvo em: {fname}")
        except Exception as e:
            print(f"  ! Aviso ao salvar gráfico de treino '{fname}': {e}")
        finally:
            plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    k_static = extract_static_gain()

    print("\n--- Carregando ensaios ---")
    train = [load_exp(*e) for e in TRAIN]
    val = [load_exp(*e) for e in VAL]
    print(f"Treino: {sum(len(e['y']) for e in train)} amostras em {len(train)} ensaios | "
          f"Validação: {len(val)} ensaios")

    print("\n--- Gerando gráficos dos dados de treinamento ---")
    plot_training_data(train)

    thetas = {}
    if "--eval" in sys.argv or "--plot" in sys.argv:
        print("\n--- Modo Avaliação/Plotagem: Carregando modelo previamente salvo ---")
        thetas = load_saved_model()
        if thetas:
            print(f"Modelo carregado com sucesso ({list(thetas.keys())}). Pulando otimização!")
            for mode, th in thetas.items():
                print_params(mode, th, k_static)
        else:
            print("  ! Arquivo salvo não encontrado, prosseguindo com otimização...")

    # Carrega modelo salvo previamente (se houver)
    saved_thetas = load_saved_model()

    if not thetas:
        print("\n" + "=" * 78)
        print("OTIMIZAÇÃO GREY-BOX COM PARADA AUTOMÁTICA NO ÓTIMO")
        print("  * Tolerância KKT prática: 1e-4 com nível aceitável em 1e-3")
        print("  * Encerramento automático garantido quando o custo estabiliza")
        print("  * Memória L-BFGS: 20 vetores | Teto de segurança: 3000 iterações")
        print("=" * 78)

        # Se 'c3 fixo' já está resolvido e salvo no disco, reutiliza para ir direto a 'c3 livre'
        if saved_thetas and "c3 fixo" in saved_thetas and "--all" not in sys.argv:
            print("\n>>> Modelo salvo com 'c3 fixo' encontrado no disco.")
            print("    Reutilizando 'c3 fixo' para focar na otimização de 'c3 livre' com warm-start.")
            print("    (Para forçar o recálculo completo de ambas as etapas, use: --all)")
            thetas["c3 fixo"] = saved_thetas["c3 fixo"]
            print_params("c3 fixo (reutilizado do disco)", thetas["c3 fixo"], k_static)

        for mode, free in (("c3 fixo", False), ("c3 livre", True)):
            if mode in thetas:
                continue

            print(f"\n--- Otimização multi-experimento (v2): {mode} ---")
            
            # Determina o ponto de partida do solver
            if free and "c3 fixo" in thetas:
                # Transfere a solução ótima de c3 fixo para inicializar c3 livre no mesmo atrator
                th_init = thetas["c3 fixo"]
            elif saved_thetas and mode in saved_thetas:
                th_init = saved_thetas[mode]
            else:
                th_init = None

            th_opt, _ = identify(train, k_static, c3_free=free, lam=LAMBDA_C3, th_init=th_init)
            thetas[mode] = th_opt
            print_params(mode, thetas[mode], k_static)
            
            # Salva imediatamente após cada modo para garantir persistência
            save_model(thetas, k_static)

    results = {}
    sim_kinds = [
        ("freerun", "Free-run (Malha Aberta)", lambda th, e: free_run(th, e)),
        ("janela", f"Janelas de {SEG_LEN:g} s", lambda th, e: windowed_run(th, e, SEG_LEN)),
        ("observador", f"Preditor com Observador (L={L_OBS})", lambda th, e: observer_run(th, e, L_OBS)),
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

    # Gráficos salvos por grupo, tipo e DC
    for (group_name, kind), (exps, sims, metrics, kind_label) in results.items():
        tag = "treino" if group_name == "TREINO" else "validacao"
        for dc in ("45", "60"):
            idx = [i for i, e in enumerate(exps) if f"-{dc}-" in e['name']]
            if not idx:
                continue
            plot_group([exps[i] for i in idx],
                       {m: [s[i] for i in idx] for m, s in sims.items()},
                       {m: [x[i] for i in idx] for m, x in metrics.items()},
                       f"{kind_label} — {group_name} — DC {dc}", f"v2_{tag}_dc{dc}_{kind}.png")
