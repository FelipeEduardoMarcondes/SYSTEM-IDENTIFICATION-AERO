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
R7 = "data/coletas/RODADA-7_20260904/"
TR = (10.0, 13.0)

# Treino: 1 ensaio de cada tipo por DC
TRAIN = [(R7 + f, *TR) for f in [
    "aprbs-45-1_0904_19-54.csv",
    "chirp-45-amp35_0904_20-13.csv",
    "multi-seno-45-040Hz_0904_20-26.csv",
    "seq-degraus-45-1_0904_20-41.csv",
    "aprbs-60-1_0904_20-04.csv",
    "chirp-60-amp50_0904_20-18.csv",
    "multi-seno-60-040Hz_0904_20-35.csv",
    "seq-degraus-60-1_0904_20-51.csv",
]]

# Validação free-run: todo o resto da RODADA-7
VAL = [(R7 + f, *TR) for f in [
    "aprbs-45-2_0904_19-57.csv",
    "aprbs-45-3_0904_20-01.csv",
    "chirp-45-amp25_0904_20-11.csv",
    "chirp-45-amp45_0904_20-15.csv",
    "multi-seno-45-030Hz_0904_20-23.csv",
    "multi-seno-45-030Hz_0905_00-47.csv",
    "multi-seno-45-050Hz_0904_20-29.csv",
    "seq-degraus-45-2_0904_20-45.csv",
    "seq-degraus-45-3_0904_20-48.csv",
    "aprbs-60-2_0904_20-07.csv",
    "aprbs-60-3_0904_20-10.csv",
    "chirp-60-amp40_0904_20-16.csv",
    "chirp-60-amp60_0904_20-20.csv",
    "multi-seno-60-030Hz_0904_20-32.csv",
    "multi-seno-60-050Hz_0904_20-38.csv",
    "seq-degraus-60-2_0904_20-54.csv",
    "seq-degraus-60-3_0904_20-57.csv",
]]

DECIM = 2          # 2 = 50 Hz (1 = 100 Hz)
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
    y_deg, u, t = readData(dataset_name=path, decimar=DECIM)
    i0 = np.searchsorted(t, t[0] + trim_start)
    i1 = np.searchsorted(t, t[-1] - trim_end) if trim_end > 0 else len(t)
    y = np.deg2rad(y_deg[i0:i1])
    t = t[i0:i1] - t[i0]
    dt = float(np.mean(np.diff(t)))
    v0 = (y[1] - y[0]) / dt
    # velocidade estimada (média móvel de 5 amostras + diferença central) p/ início das janelas
    v = np.gradient(np.convolve(y, np.ones(5) / 5, mode='same'), dt)
    v[:3], v[-3:] = np.gradient(y, dt)[:3], np.gradient(y, dt)[-3:]
    base = os.path.basename(path).rsplit('_', 2)        # [nome, MMDD, HH-MM]
    name = base[0] if base[1] == '0904' else f"{base[0]} ({base[1]})"
    return dict(name=name, y=y, u=u[i0:i1].astype(float), t=t, dt=dt, v0=v0, v=v)


def fit_pct(y, yhat):
    return 100.0 * (1 - np.linalg.norm(y - yhat) / np.linalg.norm(y - np.mean(y)))


def rmse_deg(y, yhat):
    return float(np.rad2deg(np.sqrt(np.mean((y - yhat)**2))))


# =============================================================================
# Identificação multi-experimento
# =============================================================================
def identify(train, k_static, c3_free, lam=0.0):
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
    opti.solver("ipopt", {"expand": True},
                {"max_iter": 500, "max_cpu_time": 600.0, "max_wall_time": 600.0, "print_level": 3,
                 "tol": 1e-4, "hessian_approximation": "limited-memory"})

    t0 = time.time()
    try:
        sol = opti.solve()
        val = sol.value
    except Exception:
        print("  ! IPOPT não convergiu, usando a última iteração.")
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


# =============================================================================
if __name__ == "__main__":
    k_static = extract_static_gain()

    print("\n--- Carregando ensaios ---")
    train = [load_exp(*e) for e in TRAIN]
    val = [load_exp(*e) for e in VAL]
    print(f"Treino: {sum(len(e['y']) for e in train)} amostras em {len(train)} ensaios | "
          f"Validação: {len(val)} ensaios")

    thetas = {}
    for mode, free in (("c3 fixo", False), ("c3 livre", True)):
        print(f"\n--- Otimização multi-experimento: {mode} ---")
        thetas[mode] = identify(train, k_static, c3_free=free, lam=LAMBDA_C3)

    for mode, th in thetas.items():
        print_params(mode, th, k_static)

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
