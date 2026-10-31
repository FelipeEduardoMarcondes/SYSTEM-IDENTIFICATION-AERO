"""
gerar_plots_malha_aberta_v8.py
==============================
Simulação Livre (Free-Run) dos Modelos Grey-Box nos Ensaios de Malha Aberta
Gera figuras de alta resolução para validação do modelo v8_final_robust.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Diretórios
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
artifact_dir = r'C:\Users\vicio\.gemini\antigravity-ide\brain\1108f149-d1f7-4a7b-9355-68bff8eeeb40'
ma_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
ident_dir = os.path.join(root_dir, "sysid", "notebooks", "identificação-collab", "IDENT-FELIPE")

# 1. Carrega Parâmetros
with open(os.path.join(ident_dir, "modelo_greybox_v8.json"), "r", encoding="utf-8") as f:
    mod_v8 = json.load(f)

th_v8_robust = mod_v8["models"]["v8_final_robust"]["theta"]
th_v8b_agil = mod_v8["models"]["v8b_sub85_agil"]["theta"]
th_v7 = mod_v8["models"]["baseline_v7"]["theta"]

# 2. Simulador RK4 Numérico
def sim_v8(th, u, y0, dt):
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

# 3. Carrega ensaios
step_files = [
    ("malha_aberta_20_0917_16-09.csv", "Degrau 20%"),
    ("malha_aberta_25_0917_15-57.csv", "Degrau 25%"),
    ("malha_aberta_30_0917_15-34.csv", "Degrau 30%"),
    ("malha_aberta_35_0917_15-35.csv", "Degrau 35%"),
    ("malha_aberta_40_0917_15-36.csv", "Degrau 40%"),
    ("malha_aberta_45_0917_15-37.csv", "Degrau 45%"),
    ("malha_aberta_48_0917_15-54.csv", "Degrau 48%"),
    ("malha_aberta_50_0917_15-38.csv", "Degrau 50%"),
    ("degraus_0917_15-33.csv", "Degraus Sucessivos"),
]

data_list = []
for fname, label in step_files:
    fpath = os.path.join(ma_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        t = (df['tempo_ms'].values - df['tempo_ms'].values[0]) / 1000.0
        y_rad = np.deg2rad(df['angulo_deg'].values)
        u_pct = np.clip(df['u_pct'].values.astype(float), 0, None)
        dt = float(np.median(np.diff(t)))
        data_list.append(dict(fname=fname, label=label, t=t, y=y_rad, u=u_pct, dt=dt))

# Executa simulações
results = []
for d in data_list:
    y_real = d['y']
    u = d['u']
    dt = d['dt']
    y0 = y_real[0]
    
    yh_v8 = sim_v8(th_v8_robust, u, y0, dt)
    yh_v7 = sim_v8(th_v7, u, y0, dt)
    yh_v8b = sim_v8(th_v8b_agil, u, y0, dt)
    
    fit_v8 = calc_fit(y_real, yh_v8)
    rmse_v8 = calc_rmse(y_real, yh_v8)
    
    fit_v7 = calc_fit(y_real, yh_v7)
    rmse_v7 = calc_rmse(y_real, yh_v7)
    
    # Regime permanente (últimos 4 segundos)
    mask_ss = d['t'] >= (d['t'][-1] - 4.0)
    ss_real = np.rad2deg(np.mean(y_real[mask_ss]))
    ss_v8 = np.rad2deg(np.mean(yh_v8[mask_ss])) if np.all(np.isfinite(yh_v8)) else np.nan
    ss_v7 = np.rad2deg(np.mean(yh_v7[mask_ss])) if np.all(np.isfinite(yh_v7)) else np.nan
    
    results.append(dict(
        label=d['label'], fname=d['fname'], t=d['t'], y_real=y_real, u=u,
        yh_v8=yh_v8, yh_v7=yh_v7, yh_v8b=yh_v8b,
        fit_v8=fit_v8, rmse_v8=rmse_v8, ss_real=ss_real, ss_v8=ss_v8,
        fit_v7=fit_v7, rmse_v7=rmse_v7, ss_v7=ss_v7,
        err_ss_v8=abs(ss_v8 - ss_real), err_ss_v7=abs(ss_v7 - ss_real)
    ))

print(f"Total de ensaios simulados: {len(results)}")

# ==============================================================================
# Gráfico 1: Grid 3x3 com Free-Run do v8_final_robust vs Real vs v7
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False)
axes = axes.flatten()

for i, r in enumerate(results):
    ax = axes[i]
    t = r['t']
    y_real_deg = np.rad2deg(r['y_real'])
    y_v8_deg = np.rad2deg(r['yh_v8'])
    y_v7_deg = np.rad2deg(r['yh_v7'])
    
    # Plota resposta angular
    ax.plot(t, y_real_deg, 'k-', lw=1.8, label=r'Medição Real $\theta(t)$')
    ax.plot(t, y_v8_deg, color='#1f77b4', ls='--', lw=2.0, label=r'v8\_final\_robust')
    ax.plot(t, y_v7_deg, color='#d62728', ls=':', lw=1.5, alpha=0.8, label=r'baseline\_v7')
    
    # Eixo secundário para a entrada u [%]
    ax_u = ax.twinx()
    ax_u.plot(t, r['u'], color='#7f7f7f', lw=1.0, alpha=0.45, ls='-')
    ax_u.set_ylabel(r'$u$ [%]', color='#7f7f7f', fontsize=8)
    ax_u.tick_params(axis='y', labelcolor='#7f7f7f', labelsize=8)
    ax_u.set_ylim(-5, 105)
    ax_u.grid(False)
    
    # Text box de métricas
    if r['fname'] != "degraus_0917_15-33.csv":
        info_txt = (f"v8 FIT: {r['fit_v8']:.1f}%\n"
                    f"RMSE: {r['rmse_v8']:.2f}°\n"
                    f"Erro SS: {r['err_ss_v8']:.2f}°\n"
                    f"v7 FIT: {r['fit_v7']:.1f}%")
    else:
        info_txt = (f"v8 FIT: {r['fit_v8']:.1f}%\n"
                    f"RMSE: {r['rmse_v8']:.2f}°\n"
                    f"v7 FIT: {r['fit_v7']:.1f}%")
        
    ax.text(0.03, 0.95, info_txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
    
    ax.set_title(f"{r['label']} (Max u = {r['u'].max():.0f}%)", fontsize=10, fontweight='bold', pad=4)
    ax.set_ylabel(r"$\theta$ [°]", fontsize=9)
    ax.set_xlabel("Tempo [s]", fontsize=8.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if i == 0:
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

fig.suptitle("Simulação Livre (Free-Run) do Modelo Físico v8_final_robust em Degraus de Malha Aberta (Sem Realimentação)",
             fontsize=13, fontweight='bold', y=0.995)
fig.tight_layout()

# Salva
out1 = os.path.join(artifact_dir, "malha_aberta_freerun_grid_v8.png")
fig.savefig(out1, dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(current_dir, "malha_aberta_freerun_grid_v8.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[OK] Grid de Free-run salvo em: {out1}")

# ==============================================================================
# Gráfico 2: Regime Permanente Estático e Amplitude de Pico
# ==============================================================================
# Carrega todos os 14 ensaios para curva estática
all_ma_files = sorted(glob.glob(os.path.join(ma_dir, "malha_aberta_[0-9]*.csv")))
static_data = []

for f in all_ma_files:
    fname = os.path.basename(f)
    if fname.startswith("malha_aberta_-") or "16-23" in fname:
        continue
    df = pd.read_csv(f)
    t = (df['tempo_ms'].values - df['tempo_ms'].values[0]) / 1000.0
    y_deg = df['angulo_deg'].values
    u_val = float(df['u_pct'].max())
    dt = float(np.median(np.diff(t)))
    
    # Regime real
    mask_ss = t >= (t[-1] - 4.0)
    ss_real = float(np.mean(y_deg[mask_ss]))
    pk_real = float(np.max(y_deg))
    
    # Simula v8 e v7
    yh_v8 = np.rad2deg(sim_v8(th_v8_robust, df['u_pct'].values, np.deg2rad(y_deg[0]), dt))
    yh_v7 = np.rad2deg(sim_v8(th_v7, df['u_pct'].values, np.deg2rad(y_deg[0]), dt))
    yh_v8b = np.rad2deg(sim_v8(th_v8b_agil, df['u_pct'].values, np.deg2rad(y_deg[0]), dt))
    
    ss_v8 = float(np.mean(yh_v8[mask_ss]))
    pk_v8 = float(np.max(yh_v8))
    
    ss_v7 = float(np.mean(yh_v7[mask_ss]))
    pk_v7 = float(np.max(yh_v7))
    
    ss_v8b = float(np.mean(yh_v8b[mask_ss]))
    pk_v8b = float(np.max(yh_v8b))
    
    static_data.append(dict(
        fname=fname, u=u_val,
        ss_real=ss_real, pk_real=pk_real,
        ss_v8=ss_v8, pk_v8=pk_v8,
        ss_v7=ss_v7, pk_v7=pk_v7,
        ss_v8b=ss_v8b, pk_v8b=pk_v8b,
    ))

df_stat = pd.DataFrame(static_data).sort_values('u')

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Subplot 1: Ângulo de Regime Permanente
ax1.plot(df_stat['u'], df_stat['ss_real'], 'ko', ms=7, label='Medição Real (Regime Permanente)', zorder=5)
ax1.plot(df_stat['u'], df_stat['ss_v8'], 's--', color='#1f77b4', lw=2.0, ms=6, label=r'v8\_final\_robust ($p=1{,}627$)')
ax1.plot(df_stat['u'], df_stat['ss_v8b'], '^-.', color='#ff7f0e', lw=1.5, ms=5, label=r'v8b\_sub85 ($p=1{,}627$)')
ax1.plot(df_stat['u'], df_stat['ss_v7'], 'd:', color='#d62728', lw=1.8, ms=5, label=r'baseline\_v7 ($p=2{,}000$)')

ax1.set_title("Equilíbrio de Regime Permanente: Modelo vs Bancada Real", fontsize=11, fontweight='bold')
ax1.set_xlabel(r"Comando de Degrau $u$ [%]", fontsize=10)
ax1.set_ylabel(r"Ângulo de Regime $\theta_{ss}$ [°]", fontsize=10)
ax1.grid(True, ls='--', alpha=0.5)
ax1.legend(fontsize=9, loc='upper left')

# Subplot 2: Pico Transitório do Degrau
ax2.plot(df_stat['u'], df_stat['pk_real'], 'ko', ms=7, label='Pico Transitório Real', zorder=5)
ax2.plot(df_stat['u'], df_stat['pk_v8'], 's--', color='#1f77b4', lw=2.0, ms=6, label=r'v8\_final\_robust')
ax2.plot(df_stat['u'], df_stat['pk_v8b'], '^-.', color='#ff7f0e', lw=1.5, ms=5, label=r'v8b\_sub85')
ax2.plot(df_stat['u'], df_stat['pk_v7'], 'd:', color='#d62728', lw=1.8, ms=5, label=r'baseline\_v7')
ax2.axhline(90, color='crimson', ls='--', lw=1.2, label=r'Ponto Invertido ($\theta = 90^\circ$)')

ax2.set_title("Pico Máximo Transitório (Avaliação de Risco de Giro)", fontsize=11, fontweight='bold')
ax2.set_xlabel(r"Comando de Degrau $u$ [%]", fontsize=10)
ax2.set_ylabel(r"Ângulo de Pico $\theta_{\max}$ [°]", fontsize=10)
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend(fontsize=9, loc='upper left')

fig2.suptitle("Caracterização Estática e Dinâmica nos Ensaios Puros de Malha Aberta (17/09/2026)",
              fontsize=12, fontweight='bold', y=0.99)
fig2.tight_layout()

out2 = os.path.join(artifact_dir, "malha_aberta_curva_estatica_comparativo.png")
fig2.savefig(out2, dpi=200, bbox_inches='tight')
fig2.savefig(os.path.join(current_dir, "malha_aberta_curva_estatica_comparativo.png"), dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"[OK] Curva estática salva em: {out2}")
