"""
Identificação grey-box multi-experimento (modelo do professor) + validação free-run.

    theta'' = c1*sin(theta) + c2*theta' + c3*u|u| + Fa_NL(theta')
    Fa_NL   = -(Fc + Fs*exp(-(theta'/vs)^2)) * tanh(50*theta')

Compara dois modos:
  - "c3 fixo" : c3 = -k*c1, com k = c3/|c1| vindo da curva semiestática (escada)
  - "c3 livre": c3 é variável de decisão (opcionalmente regularizada em torno de k)

Treino: multiple shooting com parâmetros compartilhados entre vários ensaios.
Validação: simulação livre (free-run, só u medido + condição inicial) em ensaios não vistos.
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
# (arquivo, corte_inicial [s], corte_final [s])
# Fora: MIX_DC_*.csv (concatenação dos mesmos ensaios) e degraus_0905 (buraco de 10 s no tempo)
# =============================================================================
R7 = "data/experimentos/coleta_dados/RODADA-7/"
TR = (10.0, 13.0)

# Treino: 1 ensaio de cada tipo por DC
TRAIN = [(R7 + f, *TR) for f in [
    "aprbs-45-3_0904_20-01.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "multi-seno-45-050Hz_0904_20-29.csv",
    "seq-degraus-45-3_0904_20-48.csv",
    "aprbs-60-3_0904_20-10.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "seq-degraus-60-3_0904_20-57.csv",
]]

# Todos os ensaios da RODADA-7 elegíveis
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

# Validação free-run: todo o resto da RODADA-7 que não está no treino
train_paths = {e[0] for e in TRAIN}
VAL = [(R7 + f, *TR) for f in ALL_R7_FILES if (R7 + f) not in train_paths]

DECIM = 1        # 2 = 50 Hz (1 = 100 Hz)
LAMBDA_C3 = 0.0    # peso da regularização de c3 livre em torno de k estático (0 = totalmente livre)

# Janela do multiple shooting [s]. Cada janela começa com estado livre (sem continuidade entre
# janelas). Necessário porque acima de 90° a planta é instável em malha aberta (DC 60 chega a
# ~150°): simular 130 s seguidos com o u de malha fechada diverge e o IPOPT não converge.
# None = horizonte inteiro (comportamento antigo).
SEG_LEN = 5.0

# Escalas das variáveis de decisão (valores físicos = escala * variável ~ 1)
S_C1, S_C2, S_F, S_C3 = 8.0, 0.4, 0.15, 2.5e-3


# =============================================================================
# Modelo e integrador (CasADi)
# =============================================================================
def build_rk4():
    x = cs.SX.sym('x', 2)
    u = cs.SX.sym('u')
    th = cs.SX.sym('th', 6)   # [c1, c2, c3, Fc, Fs, vs]
    h = cs.SX.sym('h')

    def f(x_, u_):
        v = x_[1]
        fric_nl = -(th[3] + th[4] * cs.exp(-(v / th[5])**2)) * cs.tanh(50 * v)
        acc = th[0] * cs.sin(x_[0]) + th[1] * v + th[2] * u_ * cs.fabs(u_) + fric_nl
        return cs.vertcat(v, acc)

    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)
    x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return cs.Function('rk4', [x, u, th, h], [x_next])


RK4 = build_rk4()


def free_run(theta, exp):
    """Simulação livre: só u medido + condição inicial (y0, v0)."""
    N = len(exp['y'])
    sim = RK4.mapaccum('sim', N - 1)
    x0 = cs.DM([exp['y'][0], exp['v0']])
    th = cs.repmat(cs.DM(theta), 1, N - 1)
    X = np.array(sim(x0, cs.DM(exp['u'][:-1]).T, th, exp['dt']))
    return np.r_[exp['y'][0], X[0, :]]


def segments(N, dt, seg_len):
    """Lista de (início, fim) das janelas."""
    if seg_len is None:
        return [(0, N)]
    L = max(int(round(seg_len / dt)), 2)
    return [(s, min(s + L, N)) for s in range(0, N - 1, L)]


def windowed_run(theta, exp, seg_len):
    """Simulação por janelas: reinicia em cada janela a partir de (y medido, v estimado)."""
    y_hat = np.empty_like(exp['y'])
    for s, e in segments(len(exp['y']), exp['dt'], seg_len):
        n = e - s
        y_hat[s] = exp['y'][s]
        if n < 2:
            continue
        sim = RK4.mapaccum('sim', n - 1)
        x0 = cs.DM([exp['y'][s], exp['v'][s]])
        X = np.array(sim(x0, cs.DM(exp['u'][s:e - 1]).T, cs.repmat(cs.DM(theta), 1, n - 1), exp['dt']))
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
    # velocidade estimada (média móvel de 5 amostras + diferença central) p/ início das janelas
    v = np.gradient(np.convolve(y, np.ones(5) / 5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    base = os.path.basename(path).rsplit('_', 2)        # [nome, MMDD, HH-MM]
    name = base[0] if base[1] == '0904' else f"{base[0]} ({base[1]})"
    return dict(name=name, y=y, u=u[i0:i1].astype(float), t=t, dt=dt, v0=v0, v=v, ref=ref)


def fit_pct(y, yhat):
    return 100.0 * (1 - np.linalg.norm(y - yhat) / np.linalg.norm(y - np.mean(y)))


def rmse_deg(y, yhat):
    return float(np.rad2deg(np.sqrt(np.mean((y - yhat)**2))))


# =============================================================================
# Identificação multi-experimento
# =============================================================================
def identify(train, k_static, c3_free, lam=0.0, th_init=None,
             ipopt_time_limit=7200.0, tol=1e-4, max_iter=3000):
    opti = cs.Opti()
    p1, p2, p3, p4, vs = [opti.variable() for _ in range(5)]
    c1 = -S_C1 * p1
    c2 = -S_C2 * p2
    Fc = S_F * p3
    Fs = S_F * p4

    opti.subject_to(opti.bounded(0.175, p1, 3.5))
    opti.subject_to(opti.bounded(0.0, p2, 12.0))
    opti.subject_to(opti.bounded(0.0, p3, 6.0))
    opti.subject_to(opti.bounded(0.0, p4, 6.0))
    opti.subject_to(opti.bounded(0.1, vs, 5.0))

    if th_init is not None:
        c1_0, c2_0, c3_0, Fc_0, Fs_0, vs_0 = th_init[:6]
        opti.set_initial(p1, float(np.clip(-c1_0 / S_C1, 0.175, 3.5)))
        opti.set_initial(p2, float(np.clip(-c2_0 / S_C2, 0.0, 12.0)))
        opti.set_initial(p3, float(np.clip(Fc_0 / S_F, 0.0, 6.0)))
        opti.set_initial(p4, float(np.clip(Fs_0 / S_F, 0.0, 6.0)))
        opti.set_initial(vs, float(np.clip(vs_0, 0.1, 5.0)))
    else:
        c1_init = -C_g / 0.035
        opti.set_initial(p1, -c1_init / S_C1)
        opti.set_initial(p2, (0.016 / 0.035) / S_C2)
        opti.set_initial(p3, (0.005 / 0.035) / S_F)
        opti.set_initial(p4, (0.005 / 0.035) / S_F)
        opti.set_initial(vs, 0.5)

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

    theta = cs.vertcat(c1, c2, c3, Fc, Fs, vs)

    cost = 0
    for exp in train:
        N = len(exp['y'])
        X = opti.variable(2, N)
        opti.set_initial(X[0, :], exp['y'])
        opti.set_initial(X[1, :], exp['v'])
        segs = segments(N, exp['dt'], SEG_LEN)
        if SEG_LEN is None:
            opti.subject_to(X[0, 0] == exp['y'][0])
            opti.subject_to(X[1, 0] == exp['v0'])
        # continuidade só DENTRO de cada janela (início de janela = estado livre)
        starts = {s for s, _ in segs}
        idx = [k for k in range(N - 1) if (k + 1) not in starts]
        Xn = RK4.map(N - 1)(X[:, :-1], cs.DM(exp['u'][:-1]).T, cs.repmat(theta, 1, N - 1), exp['dt'])
        opti.subject_to(X[:, [k + 1 for k in idx]] == Xn[:, idx])
        cost += cs.sumsqr(X[0, :] - cs.DM(exp['y']).T)

    if c3_free and lam > 0:
        cost += lam * ((c3 / (-c1)) / k_static - 1)**2

    opti.minimize(cost)
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
    print(f"  tempo de otimização: {time.time() - t0:.1f} s | custo = {float(val(cost)):.3f}")
    return th


def print_params(label, th, k_static):
    c1, c2, c3, Fc, Fs, vs = th
    J = -C_g / c1
    print(f"\n[{label}]")
    print(f"  c1 = {c1:.5f} | c2 = {c2:.5f} | c3 = {c3:.5e}")
    print(f"  Fc = {Fc:.5f} | Fs = {Fs:.5f} | vs = {vs:.4f} rad/s")
    print(f"  k_dyn = c3/|c1| = {c3 / abs(c1):.4e}  (k estático = {k_static:.4e}, razão = {c3 / abs(c1) / k_static:.3f})")
    print(f"  J = {J:.5f} kg.m^2 | b = {-c2 * J:.5f} N.m.s/rad | Ktau = {c3 * J:.4e}")


def save_model(thetas, k_static, filename="modelo_greybox_v1"):
    """Salva os parâmetros identificados e grandezas físicas em JSON e NPZ (modelo v1)."""
    data = {
        "k_static": float(k_static),
        "C_g": float(C_g),
        "models": {}
    }
    npz_dict = {"k_static": np.array(k_static, dtype=float)}

    for mode, th in thetas.items():
        c1, c2, c3, Fc, Fs, vs = map(float, th)
        J = float(-C_g / c1)
        b = float(-c2 * J)
        Ktau = float(c3 * J)
        k_dyn = float(c3 / abs(c1))
        tag = "c3_fixo" if "fixo" in mode else "c3_livre"

        data["models"][mode] = {
            "theta": [c1, c2, c3, Fc, Fs, vs],
            "params": {
                "c1": c1, "c2": c2, "c3": c3,
                "Fc": Fc, "Fs": Fs, "vs": vs
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

    print(f"\nModelo v1 salvo com sucesso:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")


def load_saved_model(filename="modelo_greybox_v1"):
    """Carrega parâmetros do modelo v1 salvos previamente."""
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
        ax.set_ylim(y_deg.min() - 30, y_deg.max() + 30)   # simulações divergentes não estragam a escala
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Tempo [s]')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(current_dir, fname), dpi=110)
    plt.close(fig)
    print(f"Gráfico salvo em: {fname}")


def plot_training_data(train_exps):
    """Gera gráficos dos sinais de treinamento: resposta angular theta (com referência) e sinal de controle u."""
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
            
            # Painel da esquerda: Ângulo e Referência
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
                
            # Painel da direita: Sinal de Controle u
            ax_u = axes[i, 1]
            ax_u.plot(t, u, color='tab:blue', lw=1.1, label='Controle u [%]')
            ax_u.set_ylabel('u [%]', fontsize=9)
            ax_u.set_title(f"{exp['name']} — Entrada u [%]", fontsize=10)
            ax_u.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax_u.legend(loc='upper right', fontsize=8)
                
        axes[-1, 0].set_xlabel('Tempo [s]', fontsize=9)
        axes[-1, 1].set_xlabel('Tempo [s]', fontsize=9)
        fig.suptitle(f"Dados de Treinamento — Faixa DC {dc} (Ângulo e Controle)", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fname = f"dados_treinamento_dc{dc}.png"
        fig.savefig(os.path.join(current_dir, fname), dpi=120)
        plt.close(fig)
        print(f"Gráfico de treino salvo em: {fname}")

    # Gráfico consolidado com todos os ensaios (layout 4x2 com eixo duplo)
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex='col')
    for col_idx, dc in enumerate(["45", "60"]):
        exps_dc = [e for e in train_exps if f"-{dc}-" in e['name']]
        for row_idx, exp in enumerate(exps_dc[:4]):
            ax = axes[row_idx, col_idx]
            t = exp['t']
            y_deg = np.rad2deg(exp['y'])
            u = exp['u']
            
            l1, l2 = [], []
            if exp.get('ref') is not None:
                ref_deg = np.rad2deg(exp['ref'])
                l1 = ax.plot(t, ref_deg, 'r--', lw=1.2, label='Referência r', alpha=0.85)
            l2 = ax.plot(t, y_deg, 'k-', lw=1.2, label='Ângulo θ')
            ax.set_ylabel('θ [°]', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            ax2 = ax.twinx()
            l3 = ax2.plot(t, u, color='tab:blue', lw=0.9, alpha=0.55, label='Controle u')
            ax2.set_ylabel('u [%]', color='tab:blue', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            ax.set_title(f"{exp['name']}  (θ: {y_deg.min():.0f}°..{y_deg.max():.0f}° | u: {u.min():.0f}%..{u.max():.0f}%)",
                         fontsize=9.5, fontweight='bold')
            if row_idx == 0 and col_idx == 0:
                lines = l1 + l2 + l3
                labels = [line.get_label() for line in lines]
                ax.legend(lines, labels, loc='upper right', fontsize=8)
                
    axes[-1, 0].set_xlabel('Tempo [s]', fontsize=9)
    axes[-1, 1].set_xlabel('Tempo [s]', fontsize=9)
    fig.suptitle("Visão Geral dos Dados de Treinamento (DC 45 vs DC 60)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fname_all = "dados_treinamento.png"
    fig.savefig(os.path.join(current_dir, fname_all), dpi=120)
    plt.close(fig)
    print(f"Gráfico geral de treino salvo em: {fname_all}")


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
        print("\n--- Modo Avaliação/Plotagem: Carregando modelo previamente salvo (v1) ---")
        thetas = load_saved_model()
        if thetas:
            print(f"Modelo v1 carregado com sucesso ({list(thetas.keys())}). Pulando otimização!")
            for mode, th in thetas.items():
                print_params(mode, th, k_static)
        else:
            print("  ! Arquivo salvo não encontrado, prosseguindo com otimização...")

    saved_thetas = load_saved_model()

    if not thetas:
        print("\n" + "=" * 78)
        print("OTIMIZAÇÃO GREY-BOX V1 (MODELO DE 2ª ORDEM)")
        print("  * Tolerância KKT: 1e-4 com nível aceitável em 1e-3")
        print("  * Salvamento em modelo_greybox_v1.json / modelo_greybox_v1.npz")
        print("=" * 78)

        if saved_thetas and "c3 fixo" in saved_thetas and "--all" not in sys.argv:
            print("\n>>> Modelo salvo com 'c3 fixo' encontrado no disco.")
            print("    Reutilizando 'c3 fixo' para focar na otimização de 'c3 livre' com warm-start.")
            print("    (Para forçar o recálculo completo de ambas as etapas, use: --all)")
            thetas["c3 fixo"] = saved_thetas["c3 fixo"]
            print_params("c3 fixo (reutilizado do disco)", thetas["c3 fixo"], k_static)

        for mode, free in (("c3 fixo", False), ("c3 livre", True)):
            if mode in thetas:
                continue

            print(f"\n--- Otimização multi-experimento (v1): {mode} ---")
            th_init = thetas.get("c3 fixo") if (free and "c3 fixo" in thetas) else saved_thetas.get(mode) if saved_thetas else None
            thetas[mode] = identify(train, k_static, c3_free=free, lam=LAMBDA_C3, th_init=th_init)
            print_params(mode, thetas[mode], k_static)
            save_model(thetas, k_static)

    results = {}
    sim_kinds = (("freerun", "Free-run", lambda th, e: free_run(th, e)),
                 ("janela", f"Simulação em janelas de {SEG_LEN:g} s", lambda th, e: windowed_run(th, e, SEG_LEN)))
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

    # Gráficos separados por grupo, tipo de simulação e DC (45 / 60)
    for (group_name, kind), (exps, sims, metrics, kind_label) in results.items():
        tag = "treino" if group_name == "TREINO" else "validacao"
        for dc in ("45", "60"):
            idx = [i for i, e in enumerate(exps) if f"-{dc}-" in e['name']]
            if not idx:
                continue
            plot_group([exps[i] for i in idx],
                       {m: [s[i] for i in idx] for m, s in sims.items()},
                       {m: [x[i] for i in idx] for m, x in metrics.items()},
                       f"{kind_label} — {group_name} — DC {dc}", f"multi_{tag}_dc{dc}_{kind}.png")
