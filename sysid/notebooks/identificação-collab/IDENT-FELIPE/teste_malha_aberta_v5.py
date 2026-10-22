"""
Teste do Modelo Grey-Box v5 em Malha Aberta (Free-Run nos degraus de MALHA-ABERTA)
==================================================================================
Para cada modelo salvo em modelo_greybox_v5.json:
  1. Free-run (sem correção por medição) em todos os degraus positivos de malha aberta.
  2. Métricas: FIT, RMSE, erro de regime, pico (overshoot) e detecção de "looping" (|θ| > 180°).
  3. Mapa estático: equilíbrio do modelo  sin(θ_eq) = c3/|c1| * max(u-u0,0)^2  vs regime real.
  4. Margem de looping: o pêndulo dá volta se o pico ultrapassar o equilíbrio instável 180° - θ_eq.

Uso:  python teste_malha_aberta_v5.py
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
MA_DIR = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
MODEL_JSON = os.path.join(current_dir, "modelo_greybox_v5.json")
T_REGIME = 5.0  # últimos segundos usados como regime permanente


# -----------------------------------------------------------------------------
# Modelo v5 (mesma equação de ident_greybox_prof_multi_v5.build_rk4_v5)
# -----------------------------------------------------------------------------
def f_v5(x, u, th, u0, tau_m):
    c1, c2, c3, Fc, ca = th
    pos, v, a = x
    thrust = max(a - u0, 0.0) ** 2
    acc = c1 * np.sin(pos) + c2 * v - ca * v * abs(v) - Fc * np.tanh(50.0 * v) + c3 * thrust
    return np.array([v, acc, (u - a) / tau_m])


def free_run(th, u, y0, dt, u0, tau_m):
    x = np.array([y0, 0.0, u[0]])
    out = np.empty(len(u))
    out[0] = y0
    for k in range(len(u) - 1):
        k1 = f_v5(x, u[k], th, u0, tau_m)
        k2 = f_v5(x + dt / 2 * k1, u[k], th, u0, tau_m)
        k3 = f_v5(x + dt / 2 * k2, u[k], th, u0, tau_m)
        k4 = f_v5(x + dt * k3, u[k], th, u0, tau_m)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = x[0]
    return out


def equilibrium_deg(th, u, u0):
    s = th[2] / abs(th[0]) * max(u - u0, 0.0) ** 2
    return np.nan if s > 1 else np.rad2deg(np.arcsin(s))


def fit_pct(y, yh):
    d = np.linalg.norm(y - y.mean())
    return 100 * (1 - np.linalg.norm(y - yh) / d) if d > 1e-9 else 0.0


# -----------------------------------------------------------------------------
def load_open_loop():
    files = sorted(glob.glob(os.path.join(MA_DIR, "malha_aberta_*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("malha_aberta_-")]
    files += [os.path.join(MA_DIR, "degraus_0917_15-33.csv")]
    exps = []
    for f in files:
        df = pd.read_csv(f)
        t = df.tempo_ms.values / 1000.0
        t = t - t[0]
        y = np.deg2rad(df.angulo_deg.values)
        u = np.clip(df.u_pct.values.astype(float), 0, None)
        name = os.path.basename(f).replace(".csv", "")
        exps.append(dict(name=name, t=t, y=y, u=u, dt=float(np.median(np.diff(t)))))
    return exps


def main():
    with open(MODEL_JSON, encoding="utf-8") as fh:
        model = json.load(fh)
    u0 = model["u0_static"]
    tau_m = model["tau_m_fixed"]
    k_static = model["k_static"]
    thetas = {m: np.array(d["theta"][:5]) for m, d in model["models"].items()}
    exps = load_open_loop()
    print(f"u0 = {u0:.2f}% | tau_m = {tau_m:.3f} s | k_static = {k_static:.4e}")
    print(f"{len(exps)} ensaios de malha aberta | modelos: {list(thetas)}\n")

    sims = {m: [] for m in thetas}
    rows = []
    for e in exps:
        mask_r = e['t'] >= e['t'][-1] - T_REGIME
        y_ss = np.rad2deg(e['y'][mask_r].mean())
        y_pk = np.rad2deg(e['y'].max())
        u_step = e['u'].max()
        for m, th in thetas.items():
            with np.errstate(all='ignore'):
                yh = free_run(th, e['u'], e['y'][0], e['dt'], u0, tau_m)
            sims[m].append(yh)
            yh_deg = np.rad2deg(yh)
            loop = bool(np.nanmax(np.abs(yh_deg)) > 180 or not np.all(np.isfinite(yh)))
            th_eq = equilibrium_deg(th, u_step, u0)
            rows.append(dict(
                ensaio=e['name'], modelo=m, u=u_step,
                fit=fit_pct(e['y'], yh) if not loop else np.nan,
                rmse=np.rad2deg(np.sqrt(np.mean((e['y'] - yh) ** 2))) if not loop else np.nan,
                ss_real=y_ss, ss_mod=yh_deg[mask_r].mean() if not loop else np.nan,
                eq_mod=th_eq, pk_real=y_pk, pk_mod=np.nanmax(yh_deg) if not loop else np.nan,
                limite_loop=180 - th_eq if np.isfinite(th_eq) else np.nan,
                LOOP=loop))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    for m in thetas:
        d = df[df.modelo == m].drop(columns="modelo")
        print("=" * 110)
        print(f"MODELO: {m}   (k_dyn/k_static = {thetas[m][2] / abs(thetas[m][0]) / k_static:.3f})")
        print("=" * 110)
        print(d.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
        ok = d[~d.LOOP]
        print(f"  -> Looping em {int(d.LOOP.sum())}/{len(d)} ensaios | FIT médio (sem loop) = {ok.fit.mean():.1f}% | "
              f"erro de regime médio = {np.mean(np.abs(ok.ss_mod - ok.ss_real)):.2f}°\n")
    df.to_csv(os.path.join(current_dir, "v5_teste_malha_aberta.csv"), index=False)

    # ---------------- Gráfico 1: free-run por ensaio ----------------
    colors = {'c3 fixo': 'tab:blue', 'c3 livre': 'tab:red',
              'mecanica fixada': 'tab:green', 'mecanica regularizada': 'tab:purple'}
    n = len(exps)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 2.8 * nrow), squeeze=False)
    for i, e in enumerate(exps):
        ax = axes[i // ncol, i % ncol]
        ax.plot(e['t'], np.rad2deg(e['y']), 'k', lw=1.3, label='Medição')
        for m in thetas:
            ax.plot(e['t'], np.clip(np.rad2deg(sims[m][i]), -200, 400), '--',
                    color=colors.get(m, 'tab:orange'), lw=1.1, label=m)
        ax.set_title(f"{e['name']} (u={e['u'].max():.0f}%)", fontsize=9)
        ax.set_ylim(-20, max(140, np.rad2deg(e['y']).max() + 20))
        ax.axhline(90, color='gray', ls=':', lw=0.8)
        ax.grid(True, ls=':', alpha=0.5)
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.suptitle("Modelo v5 — Free-run nos degraus de MALHA ABERTA", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(current_dir, "v5_teste_malha_aberta_freerun.png"), dpi=100)
    plt.close(fig)

    # ---------------- Gráfico 2: mapa estático ----------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    d0 = df[df.modelo == list(thetas)[0]]
    ax[0].scatter(d0.u, d0.ss_real, c='k', s=40, zorder=5, label='Regime real')
    ax[1].scatter(d0.u, d0.pk_real, c='k', s=40, zorder=5, label='Pico real')
    ug = np.linspace(0, 60, 300)
    ax[0].plot(ug, [np.rad2deg(np.arcsin(min(k_static * max(x - u0, 0) ** 2, 1))) for x in ug],
               'k:', label='k_static (curva estática)')
    for m, th in thetas.items():
        c = colors.get(m, 'tab:orange')
        eq = np.array([equilibrium_deg(th, x, u0) for x in ug])
        ax[0].plot(ug, eq, color=c, label=f'{m} (equilíbrio)')
        ax[1].plot(ug, 180 - eq, color=c, ls=':', lw=1)
        dm = df[(df.modelo == m)]
        ax[1].scatter(dm.u, dm.pk_mod, color=c, marker='x', s=40, label=f'{m} pico')
        dl = dm[dm.LOOP]
        ax[1].scatter(dl.u, np.full(len(dl), 175), color=c, marker='v', s=70)
    ax[0].set(xlabel='u [%]', ylabel='θ regime [°]', title='Mapa estático: equilíbrio do modelo vs real')
    ax[1].set(xlabel='u [%]', ylabel='θ pico [°]', ylim=(0, 185),
              title='Pico do degrau  (pontilhado = limite de looping 180°-θeq, ▼ = deu volta)')
    for a in ax:
        a.grid(True, ls=':', alpha=0.5)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(current_dir, "v5_teste_malha_aberta_estatico.png"), dpi=110)
    plt.close(fig)
    print("Gráficos: v5_teste_malha_aberta_freerun.png, v5_teste_malha_aberta_estatico.png")
    print("Tabela:   v5_teste_malha_aberta.csv")


if __name__ == "__main__":
    main()
