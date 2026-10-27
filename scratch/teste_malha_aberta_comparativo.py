"""
Teste Comparativo de Simulação Livre (Free-Run) nos Degraus de Malha Aberta
Modelos Avaliados:
  1. Modelo v1 (Professor original: ident_greybox_prof_multi.py - c3 livre e c3 fixo)
  2. Modelo v7 (ident_greybox_prof_multi_v7.py - turbina fixada com momentum drag)
  3. Modelo v8 (ident_greybox_prof_multi_v8.py - v8_final_robust com p=1.627)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
ma_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
ident_dir = os.path.join(root_dir, "sysid", "notebooks", "identificação-collab", "IDENT-FELIPE")

# ==============================================================================
# 1. Carregamento dos Modelos
# ==============================================================================
with open(os.path.join(ident_dir, "modelo_greybox_v1.json"), "r", encoding="utf-8") as f:
    mod_v1 = json.load(f)

with open(os.path.join(ident_dir, "modelo_greybox_v7.json"), "r", encoding="utf-8") as f:
    mod_v7 = json.load(f)

with open(os.path.join(ident_dir, "modelo_greybox_v8.json"), "r", encoding="utf-8") as f:
    mod_v8 = json.load(f)

# Parâmetros v1
th_v1_livre = mod_v1["models"]["c3 livre"]["theta"]  # [c1, c2, c3, Fc, Fs, vs]
th_v1_fixo = mod_v1["models"]["c3 fixo"]["theta"]

# Parâmetros v7
th_v7 = mod_v7["models"]["turbina fixada"]["theta"]
u0_v7 = mod_v7["u0_static"]
tau_acc_v7 = mod_v7["tau_acc_fixed"]

# Parâmetros v8
th_v8 = mod_v8["models"]["v8_final_robust"]["theta"]

# ==============================================================================
# 2. Funções de Simulação RK4 Numérica Pura (NumPy rápido)
# ==============================================================================
def sim_v1(th, u, y0, dt):
    # th: [c1, c2, c3, Fc, Fs, vs]
    c1, c2, c3, Fc, Fs, vs = th
    pos = y0
    v = 0.0
    N = len(u)
    out = np.empty(N)
    out[0] = pos
    
    def f(p_, v_, u_):
        fric = -(Fc + Fs * np.exp(-(v_ / max(vs, 1e-4))**2)) * np.tanh(50.0 * v_)
        acc = c1 * np.sin(p_) + c2 * v_ + c3 * u_ * abs(u_) + fric
        return v_, acc

    for k in range(N - 1):
        uk = u[k]
        k1_p, k1_v = f(pos, v, uk)
        k2_p, k2_v = f(pos + dt/2.0 * k1_p, v + dt/2.0 * k1_v, uk)
        k3_p, k3_v = f(pos + dt/2.0 * k2_p, v + dt/2.0 * k2_v, uk)
        k4_p, k4_v = f(pos + dt * k3_p, v + dt * k3_v, uk)
        
        pos += dt/6.0 * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p)
        v += dt/6.0 * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        out[k + 1] = pos
        if abs(pos) > 10.0:  # divergiu
            out[k + 1:] = np.nan
            break
    return out

def sim_v7(th, u, y0, dt, u0, tau_acc):
    # th: [c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec, ...]
    c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec = th[:8]
    pos = y0
    v = 0.0
    a = u[0]
    N = len(u)
    out = np.empty(N)
    out[0] = pos
    
    def f(p_, v_, a_, u_):
        u_eff = max(a_ - u0, 0.0)
        thrust = u_eff**2.0
        fric = -Fc * np.tanh(50.0 * v_)
        sigma_v = 0.5 * (1.0 + np.tanh(20.0 * v_))
        ca_eff = ca_fwd * sigma_v + ca_bwd * (1.0 - sigma_v)
        aero = -ca_eff * v_ * abs(v_)
        inflow = -c4 * u_eff * v_
        
        sigma_u = 0.5 * (1.0 + np.tanh(10.0 * (u_ - a_)))
        tau_eff = tau_dec + (tau_acc - tau_dec) * sigma_u
        
        acc = c1 * np.sin(p_) + c2 * v_ + aero + fric + c3 * thrust + inflow
        a_dot = (u_ - a_) / max(tau_eff, 0.01)
        return v_, acc, a_dot

    for k in range(N - 1):
        uk = u[k]
        k1_p, k1_v, k1_a = f(pos, v, a, uk)
        k2_p, k2_v, k2_a = f(pos + dt/2.0 * k1_p, v + dt/2.0 * k1_v, a + dt/2.0 * k1_a, uk)
        k3_p, k3_v, k3_a = f(pos + dt/2.0 * k2_p, v + dt/2.0 * k2_v, a + dt/2.0 * k2_a, uk)
        k4_p, k4_v, k4_a = f(pos + dt * k3_p, v + dt * k3_v, a + dt * k3_a, uk)
        
        pos += dt/6.0 * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p)
        v += dt/6.0 * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        a += dt/6.0 * (k1_a + 2.0*k2_a + 2.0*k3_a + k4_a)
        out[k + 1] = pos
        if abs(pos) > 10.0:  # divergiu / looping infinito
            out[k + 1:] = np.nan
            break
    return out

def sim_v8(th, u, y0, dt):
    # th: [c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec, u0, tau_acc, p]
    c1, c2, c3, c4, Fc, ca_fwd, ca_bwd, tau_dec, u0, tau_acc, p = th
    pos = y0
    v = 0.0
    a = u[0]
    N = len(u)
    out = np.empty(N)
    out[0] = pos
    
    def f(p_, v_, a_, u_):
        u_eff = max(a_ - u0, 0.0)
        u_eff_smooth = np.sqrt(u_eff**2 + 1e-4)
        thrust = u_eff_smooth**p
        inflow = -c4 * (u_eff_smooth**(p / 2.0)) * v_
        fric = -Fc * np.tanh(50.0 * v_)
        sigma_v = 0.5 * (1.0 + np.tanh(20.0 * v_))
        ca_eff = ca_fwd * sigma_v + ca_bwd * (1.0 - sigma_v)
        aero = -ca_eff * v_ * abs(v_)
        
        sigma_u = 0.5 * (1.0 + np.tanh(10.0 * (u_ - a_)))
        tau_eff = tau_dec + (tau_acc - tau_dec) * sigma_u
        
        acc = c1 * np.sin(p_) + c2 * v_ + aero + fric + c3 * thrust + inflow
        a_dot = (u_ - a_) / max(tau_eff, 0.01)
        return v_, acc, a_dot

    for k in range(N - 1):
        uk = u[k]
        k1_p, k1_v, k1_a = f(pos, v, a, uk)
        k2_p, k2_v, k2_a = f(pos + dt/2.0 * k1_p, v + dt/2.0 * k1_v, a + dt/2.0 * k1_a, uk)
        k3_p, k3_v, k3_a = f(pos + dt/2.0 * k2_p, v + dt/2.0 * k2_v, a + dt/2.0 * k2_a, uk)
        k4_p, k4_v, k4_a = f(pos + dt * k3_p, v + dt * k3_v, a + dt * k3_a, uk)
        
        pos += dt/6.0 * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p)
        v += dt/6.0 * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        a += dt/6.0 * (k1_a + 2.0*k2_a + 2.0*k3_a + k4_a)
        out[k + 1] = pos
        if abs(pos) > 10.0:
            out[k + 1:] = np.nan
            break
    return out

# ==============================================================================
# 3. Carregamento dos Ensaios de Malha Aberta
# ==============================================================================
files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_*.csv")))
# Desconsidera negativos que bugaram
files = [f for f in files if not os.path.basename(f).startswith("malha_aberta_-")]
# Adiciona o ensaio de degraus múltiplos
if os.path.exists(os.path.join(ma_dir, "degraus_0917_15-33.csv")):
    files.append(os.path.join(ma_dir, "degraus_0917_15-33.csv"))

exps = []
for f in files:
    df = pd.read_csv(f)
    t = df['tempo_ms'].values / 1000.0
    t = t - t[0]
    y = np.deg2rad(df['angulo_deg'].values)
    u = np.clip(df['u_pct'].values.astype(float), 0, None)
    name = os.path.basename(f).replace('.csv', '')
    dt = float(np.median(np.diff(t)))
    exps.append(dict(name=name, t=t, y=y, u=u, dt=dt))

print(f"Total de ensaios de malha aberta válidos carregados: {len(exps)}")

# ==============================================================================
# 4. Execução da Simulação Livre para Todos os Modelos
# ==============================================================================
models = {
    "v1 (c3 livre)": lambda u, y0, dt: sim_v1(th_v1_livre, u, y0, dt),
    "v1 (c3 fixo)":  lambda u, y0, dt: sim_v1(th_v1_fixo, u, y0, dt),
    "v7 (turbina fixada)": lambda u, y0, dt: sim_v7(th_v7, u, y0, dt, u0_v7, tau_acc_v7),
    "v8 (final robust)":   lambda u, y0, dt: sim_v8(th_v8, u, y0, dt),
}

def calc_fit(y, yh):
    if not np.all(np.isfinite(yh)):
        return -999.0
    d = np.linalg.norm(y - np.mean(y))
    if d < 1e-6:
        return 0.0
    return 100.0 * (1.0 - np.linalg.norm(y - yh) / d)

def calc_rmse(y, yh):
    if not np.all(np.isfinite(yh)):
        return np.nan
    return np.rad2deg(np.sqrt(np.mean((y - yh)**2)))

results = []
sim_trajectories = {m: [] for m in models}

for exp in exps:
    t = exp['t']
    y_rad = exp['y']
    y_deg = np.rad2deg(y_rad)
    u = exp['u']
    dt = exp['dt']
    u_max = np.max(u)
    
    # Regime permanente real (últimos 5 segundos)
    mask_ss = t >= (t[-1] - 5.0)
    ss_real = np.mean(y_deg[mask_ss])
    pk_real = np.max(y_deg)
    
    for m_name, sim_fn in models.items():
        yh_rad = sim_fn(u, y_rad[0], dt)
        sim_trajectories[m_name].append(yh_rad)
        yh_deg = np.rad2deg(yh_rad)
        
        loop = bool(not np.all(np.isfinite(yh_deg)) or np.nanmax(np.abs(yh_deg)) > 180.0)
        ss_mod = np.mean(yh_deg[mask_ss]) if not loop else np.nan
        pk_mod = np.nanmax(yh_deg) if not loop else np.nan
        fit = calc_fit(y_rad, yh_rad) if not loop else np.nan
        rmse = calc_rmse(y_rad, yh_rad) if not loop else np.nan
        
        results.append({
            "ensaio": exp['name'],
            "u_step": u_max,
            "modelo": m_name,
            "ss_real": ss_real,
            "ss_mod": ss_mod,
            "erro_ss": abs(ss_mod - ss_real) if not loop else np.nan,
            "pk_real": pk_real,
            "pk_mod": pk_mod,
            "fit": fit,
            "rmse": rmse,
            "looping": loop
        })

df_res = pd.DataFrame(results)

# ==============================================================================
# 5. Impressão dos Resultados Consolidados
# ==============================================================================
for m_name in models:
    df_m = df_res[df_res['modelo'] == m_name]
    n_loop = df_m['looping'].sum()
    df_ok = df_m[~df_m['looping']]
    fit_m = df_ok['fit'].mean() if len(df_ok) > 0 else np.nan
    rmse_m = df_ok['rmse'].mean() if len(df_ok) > 0 else np.nan
    ess_m = df_ok['erro_ss'].mean() if len(df_ok) > 0 else np.nan
    
    print("\n" + "=" * 90)
    print(f"MODELO: {m_name}")
    print("=" * 90)
    print(f"Ensaios com Looping/Divergência: {n_loop}/{len(df_m)}")
    print(f"FIT Médio (sem loop): {fit_m:.1f}% | RMSE Médio: {rmse_m:.2f}° | Erro de Regime Médio: {ess_m:.2f}°")
    print("-" * 90)
    cols = ['ensaio', 'u_step', 'ss_real', 'ss_mod', 'erro_ss', 'pk_real', 'pk_mod', 'rmse', 'fit', 'looping']
    print(df_m[cols].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

# ==============================================================================
# 6. Geração dos Gráficos Comparativos
# ==============================================================================
# Gráfico 1: Free-Run por Ensaio
n = len(exps)
ncol = 3
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.0 * nrow), squeeze=False)

colors = {
    "v1 (c3 livre)": "#ff7f0e",
    "v1 (c3 fixo)":  "#8c564b",
    "v7 (turbina fixada)": "#1f77b4",
    "v8 (final robust)":   "#2ca02c",
}

for i, exp in enumerate(exps):
    ax = axes[i // ncol, i % ncol]
    t = exp['t']
    y_deg = np.rad2deg(exp['y'])
    ax.plot(t, y_deg, 'k-', lw=1.5, label='Medição Real')
    
    for m_name in models:
        yh_deg = np.rad2deg(sim_trajectories[m_name][i])
        ax.plot(t, np.clip(yh_deg, -20, 160), '--', color=colors[m_name], lw=1.2, label=m_name)
        
    ax.set_title(f"{exp['name']} (u = {exp['u'].max():.0f}%)", fontsize=9, fontweight='bold')
    ax.set_xlabel("Tempo [s]", fontsize=8)
    ax.set_ylabel(r"Ângulo $\theta$ [$^\circ$]", fontsize=8)
    ax.set_ylim(-15, min(150, max(y_deg.max() + 25, 60)))
    ax.axhline(90, color='gray', ls=':', lw=0.7)
    ax.grid(True, ls='--', alpha=0.5)
    if i == 0:
        ax.legend(fontsize=7, loc='upper left')

for j in range(n, nrow * ncol):
    axes[j // ncol, j % ncol].axis('off')

fig.suptitle("Simulação Livre (Free-Run) dos Modelos Grey-Box nos Ensaios de MALHA ABERTA", fontsize=14, fontweight='bold')
fig.tight_layout()
fig_path1 = os.path.join(current_dir, "malha_aberta_freerun_comparativo.png")
fig.savefig(fig_path1, dpi=150)
plt.close(fig)
print(f"\n[OK] Gráfico temporal salvo em: {fig_path1}")

# Gráfico 2: Curva Quase-Estática / Regime Permanente (Regime vs u)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

df_steps = df_res[df_res['ensaio'] != 'degraus_0917_15-33']

# Subplot 1: Ângulo de Regime Permanente
ax1.scatter(df_steps[df_steps['modelo'] == 'v7 (turbina fixada)']['u_step'], 
            df_steps[df_steps['modelo'] == 'v7 (turbina fixada)']['ss_real'], 
            c='black', s=55, zorder=5, label='Regime Real Medido')

u_grid = np.linspace(10, 52, 200)

for m_name in models:
    d_m = df_steps[df_steps['modelo'] == m_name]
    ax1.plot(d_m['u_step'], d_m['ss_mod'], 'o--', color=colors[m_name], lw=1.2, ms=4, label=f"{m_name}")

ax1.set_xlabel("Degrau de Entrada $u$ [%]", fontsize=10)
ax1.set_ylabel(r"Ângulo de Regime $\theta_{ss}$ [$^\circ$]", fontsize=10)
ax1.set_title("Equilíbrio Estático: Modelo vs. Bancada Real", fontsize=11, fontweight='bold')
ax1.grid(True, ls='--', alpha=0.5)
ax1.legend(fontsize=8.5)

# Subplot 2: Sobressinal de Pico
ax2.scatter(df_steps[df_steps['modelo'] == 'v7 (turbina fixada)']['u_step'], 
            df_steps[df_steps['modelo'] == 'v7 (turbina fixada)']['pk_real'], 
            c='black', s=55, zorder=5, label='Pico Máximo Real')

for m_name in models:
    d_m = df_steps[df_steps['modelo'] == m_name]
    ax2.plot(d_m['u_step'], d_m['pk_mod'], '^--', color=colors[m_name], lw=1.2, ms=4, label=f"{m_name}")

ax2.axhline(90, color='gray', ls=':', lw=0.8, label="Equilíbrio Instável (90°)")
ax2.set_xlabel("Degrau de Entrada $u$ [%]", fontsize=10)
ax2.set_ylabel(r"Ângulo de Pico $\theta_{\max}$ [$^\circ$]", fontsize=10)
ax2.set_title("Pico Transitório do Degrau (Risco de Looping)", fontsize=11, fontweight='bold')
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend(fontsize=8.5)

fig.tight_layout()
fig_path2 = os.path.join(current_dir, "malha_aberta_curva_estatica.png")
fig.savefig(fig_path2, dpi=150)
plt.close(fig)
print(f"[OK] Gráfico estático salvo em: {fig_path2}")
