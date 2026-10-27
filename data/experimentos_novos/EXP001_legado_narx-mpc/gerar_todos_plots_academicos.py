"""
Script de Geração de Plots no Padrão Acadêmico / IEEE / PUCPR
Projeto: Controle de Aeropêndulo com NMPC Aproximado por Rede Neural (TinyML)
Autor: Felipe Eduardo Marcondes

Gera gráficos padronizados com:
- Fundo branco limpo (sem modo escuro)
- Eixos com unidades explícitas: Tempo [s], Ângulo theta [deg], Controle u [%]
- Subplots duplos (Ângulo em cima, Controle em baixo) com sharex=True
- Tipografia proporcional e legível (Arial / Sans-serif)
- Cores e estilos de linha acessíveis (preto tracejado para ref, azul para simulação,
  verde para SIL, vermelho para bancada real)
- Exportação simultânea em PNG (300 DPI) e PDF vetorial
"""

import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==============================================================================
# CONFIGURAÇÃO DE ESTILO PADRÃO ACADÊMICO
# ==============================================================================
def setup_academic():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.85",
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

setup_academic()

# Diretórios
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
import sys
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

exp_dir = os.path.join(root_dir, "data", "experimentos_novos", "EXP001_legado_narx-mpc")
py_dir = os.path.join(exp_dir, "2_mpc_python")
sil_dir = os.path.join(exp_dir, "3_sil_stm32")
real_dir = os.path.join(exp_dir, "4_aeropendulo")

dest_dirs = [
    os.path.join(root_dir, "tcc", "img"),
    os.path.join(exp_dir, "plots_academicos")
]
for d in dest_dirs:
    os.makedirs(d, exist_ok=True)

def save_academic_plot(fig, base_name, extra_dirs=None):
    all_dirs = list(dest_dirs)
    if extra_dirs:
        for ed in extra_dirs:
            if ed not in all_dirs:
                all_dirs.append(ed)
    for d in all_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{base_name}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"{base_name}.pdf"), format="pdf")
    print(f"-> Salvo: {base_name} (.png e .pdf)")

# ==============================================================================
# 1. TRÍADE COMPARATIVA (Python vs SIL vs Bancada Real)
# ==============================================================================
print("\n[1/7] Gerando gráficos comparativos da tríade...")
comparacoes = [
    {
        "id": 0,
        "base_name": "comparacao_plot_0",
        "py_csv": os.path.join(py_dir, "simulacao_python.csv"),
        "sil_csv": os.path.join(sil_dir, "20261006_161042_referencia_mpc.csv"),
        "real_csv": os.path.join(real_dir, "20261006_181737_referencia_mpc.csv"),
        "leg_pos": "lower right"
    },
    {
        "id": 1,
        "base_name": "comparacao_plot_1",
        "py_csv": os.path.join(py_dir, "simulacao_python_val2_degraus.csv"),
        "sil_csv": os.path.join(sil_dir, "20261006_161234_referencia_val2_degraus.csv"),
        "real_csv": os.path.join(real_dir, "20261006_182048_referencia_val2_degraus.csv"),
        "leg_pos": "upper right"
    },
    {
        "id": 2,
        "base_name": "comparacao_plot_2",
        "py_csv": os.path.join(py_dir, "simulacao_python_val3_senoidal.csv"),
        "sil_csv": os.path.join(sil_dir, "20261006_161422_referencia_val3_senoidal.csv"),
        "real_csv": os.path.join(real_dir, "20261006_182839_referencia_val3_senoidal.csv"),
        "leg_pos": "lower right"
    }
]

for item in comparacoes:
    df_py = pd.read_csv(item["py_csv"])
    df_sil = pd.read_csv(item["sil_csv"])
    df_real = pd.read_csv(item["real_csv"])
    
    t_py = df_py['tempo_ms'] / 1000.0
    t_sil = df_sil['tempo_ms'] / 1000.0
    t_real = df_real['tempo_ms'] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    # Subplot 1: Ângulo
    ax1.plot(t_py, df_py['referencia'], color='black', linestyle='--', linewidth=1.1, 
             alpha=0.85, label='Referência')
    ax1.plot(t_py, df_py['angulo_deg'], color='#1f77b4', linestyle='-', linewidth=1.4, 
             alpha=0.9, label='Python (Simulador)')
    ax1.plot(t_sil, df_sil['angulo_deg'], color='#2ca02c', linestyle='-.', linewidth=1.3, 
             alpha=0.85, label='STM32 (SIL)')
    ax1.plot(t_real, df_real['angulo_deg'], color='#d62728', linestyle=':', linewidth=1.3, 
             alpha=0.9, label='Bancada Real')
    
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc=item["leg_pos"], ncol=4, framealpha=0.95)
    
    # Subplot 2: Controle
    ax2.plot(t_py, df_py['u_pct'], color='#1f77b4', linestyle='-', linewidth=1.2, 
             alpha=0.85, label='Python')
    ax2.plot(t_sil, df_sil['u_pct'], color='#2ca02c', linestyle='-.', linewidth=1.2, 
             alpha=0.85, label='SIL')
    ax2.plot(t_real, df_real['u_pct'], color='#d62728', linestyle=':', linewidth=1.1, 
             alpha=0.85, label='Bancada Real')
    
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc='upper right', ncol=3, framealpha=0.95)
    
    max_t = max(t_py.max(), t_sil.max(), t_real.max())
    ax1.set_xlim(0, max_t)
    
    plt.tight_layout()
    save_academic_plot(fig, item["base_name"])
    plt.close(fig)

# ==============================================================================
# 2. SIMULAÇÃO LIVRE DO MODELO NARX (Free-Run)
# ==============================================================================
print("\n[2/7] Gerando gráfico de Simulação Livre (NARX Free-Run)...")
narx_json_path = os.path.join(root_dir, "sysid", "notebooks", "identificação-collab", "IDENT-HELON", "narx_model.json")
with open(narx_json_path, 'r') as f:
    narx_data = json.load(f)

terms = narx_data['terms']
theta = narx_data['theta']
ny = narx_data['ny']
nu = narx_data['nu']

# Carrega multisseno de validação
from aerodata import readData
y_raw, u_raw, t_raw, ref_raw = readData('data/coletas/RODADA-7_20260904/multi-seno-45-040Hz_0904_20-26.csv', return_ref=True)
trim_start, trim_end = 10.0, 13.0
i0 = np.searchsorted(t_raw, t_raw[0] + trim_start)
i1 = np.searchsorted(t_raw, t_raw[-1] - trim_end)
u_fr, y_fr, t_fr = u_raw[i0:i1], y_raw[i0:i1], t_raw[i0:i1]
t_fr = t_fr - t_fr[0]

ml = max(ny, nu)

def eval_narx_term(term, y_h, u_h):
    if term == 'constant': return 1.0
    val = 1.0
    for tok in re.findall(r'([yu])\(k-(\d+)\)', term):
        var, lag = tok[0], int(tok[1])
        if var == 'y': val *= y_h[-lag]
        else: val *= u_h[-lag]
    return val

y_sim = np.zeros(len(y_fr))
y_sim[:ml] = y_fr[:ml]

for k in range(ml, len(y_fr)):
    val = 0.0
    for term, th in zip(terms, theta):
        val += th * eval_narx_term(term, y_sim[:k], u_fr[:k])
    y_sim[k] = val

rmse_fr = np.sqrt(np.mean((y_fr[ml:] - y_sim[ml:])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

ax1.plot(t_fr, y_fr, color='black', linestyle='-', linewidth=1.1, label='Medido $y(k)$')
ax1.plot(t_fr, y_sim, color='#d62728', linestyle='--', linewidth=1.3, 
         label=f'NARX Free-Run (RMSE = {rmse_fr:.3f}' + r'$^\circ$)')
ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax1.grid(True)
ax1.legend(loc='upper right', framealpha=0.95)

ax2.plot(t_fr, u_fr, color='#1f77b4', linestyle='-', linewidth=1.1, label=r'Excitação Multisseno $u(k)$')
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel(r'Controle $u$ [\%]')
ax2.grid(True)
ax2.legend(loc='upper right', framealpha=0.95)
ax1.set_xlim(0, t_fr[-1])

plt.tight_layout()
save_academic_plot(fig, "narx_freerun", extra_dirs=[py_dir])
plt.close(fig)

# ==============================================================================
# 3. NMPC NOMINAL DE TREINAMENTO (CasADi)
# ==============================================================================
print("\n[3/7] Gerando gráfico de Rastreamento Nominal NMPC...")
df_mpc = pd.read_csv(os.path.join(py_dir, "simulacao_mpc_treino.csv"))
t_mpc = df_mpc['tempo_ms'] / 1000.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

ax1.plot(t_mpc, df_mpc['referencia'], color='black', linestyle='--', linewidth=1.1, 
         alpha=0.85, label='Referência $r(k)$')
ax1.plot(t_mpc, df_mpc['angulo_deg'], color='#1f77b4', linestyle='-', linewidth=1.3, 
         label='NMPC Nominal ($y$)')
ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax1.grid(True)
ax1.legend(loc='lower right', ncol=2, framealpha=0.95)

ax2.plot(t_mpc, df_mpc['u_pct'], color='#d62728', linestyle='-', linewidth=1.1, 
         label=r'Ação de Controle $u$')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel(r'Controle $u$ [\%]')
ax2.set_ylim(-10, 95)
ax2.grid(True)
ax2.legend(loc='upper right', framealpha=0.95)
ax1.set_xlim(0, t_mpc.max())

plt.tight_layout()
save_academic_plot(fig, "mpc_treino", extra_dirs=[py_dir])
plt.close(fig)

# ==============================================================================
# 4. TEMPO COMPUTACIONAL DO NMPC (Comprovando Inviabilidade Online)
# ==============================================================================
print("\n[4/7] Gerando gráfico de Tempo Computacional do NMPC...")
# Extração de alta precisão da curva de solve time original
orig_time_img = mpimg.imread(os.path.join(py_dir, "MPC_Computational_Time.png"))
r, g, b = orig_time_img[:,:,0], orig_time_img[:,:,1], orig_time_img[:,:,2]
blue_mask = (b > 0.6) & (r < 0.3) & (g < 0.6)

# Spines detectadas: x in [191, 1038] (t=0 a t=25s), y in [36, 267] (dt=0.50 a dt=0.0s)
x_coords = np.arange(191, 1039)
t_extracted = (x_coords - 191) / (1038 - 191) * 25.0
dt_extracted = []

for x in x_coords:
    y_pix = np.where(blue_mask[:, x])[0]
    if len(y_pix) > 0:
        y_mean = np.mean(y_pix)
        dt = (267.0 - y_mean) / (267.0 - 36.0) * 0.50
        dt_extracted.append(max(dt, 0.0))
    else:
        dt_extracted.append(dt_extracted[-1] if dt_extracted else 0.10)

dt_extracted = np.array(dt_extracted)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(t_extracted, dt_extracted * 1000.0, color='#1f77b4', linestyle='-', linewidth=1.3, 
        label='Tempo de Solução NMPC (IPOPT)')
ax.axhline(10.0, color='#d62728', linestyle='--', linewidth=1.4, 
           label=r'Período de Amostragem $T_s = 10$ ms (Deadline)')

# Seta para o pico inicial de cold-start (t = 0 s, 458 ms)
ax.annotate('Cold-start inicial\n(458 ms)', 
            xy=(0.0, 458), xytext=(1.8, 420),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

# Seta para o pico de transição dinâmica (t = 9.17 s, 333 ms)
ax.annotate('Pico em transição\nde degrau (333 ms)', 
            xy=(9.17, 333), xytext=(11.5, 320),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

ax.set_xlabel('Tempo [s]')
ax.set_ylabel('Tempo de Solução [ms]')
ax.set_xlim(0, 25)
ax.set_ylim(0, 500)
ax.grid(True)
ax.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()
save_academic_plot(fig, "mpc_tempo", extra_dirs=[py_dir])
plt.close(fig)

# ==============================================================================
# 5. SIMULAÇÕES INDIVIDUAIS PYTHON (py_val1, py_val2, py_val3)
# ==============================================================================
print("\n[5/7] Gerando gráficos individuais de Simulação Python...")
py_sims = [
    ("py_val1", "simulacao_python.csv", "Validação 1 (Multisseno + Degraus)"),
    ("py_val2", "simulacao_python_val2_degraus.csv", "Validação 2 (Degraus Aleatórios)"),
    ("py_val3", "simulacao_python_val3_senoidal.csv", "Validação 3 (Senoidal Suave)")
]

for base_name, csv_file, label_name in py_sims:
    df = pd.read_csv(os.path.join(py_dir, csv_file))
    t = df['tempo_ms'] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    ax1.plot(t, df['referencia'], color='black', linestyle='--', linewidth=1.1, label='Referência')
    ax1.plot(t, df['angulo_deg'], color='#1f77b4', linestyle='-', linewidth=1.4, label='Rede Neural MLP (Python)')
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc='lower right' if 'val2' not in base_name else 'upper right', framealpha=0.95)
    
    ax2.plot(t, df['u_pct'], color='#1f77b4', linestyle='-', linewidth=1.2, label='Controle $u$')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax1.set_xlim(0, t.max())
    
    plt.tight_layout()
    save_academic_plot(fig, base_name, extra_dirs=[py_dir])
    plt.close(fig)

# ==============================================================================
# 6. EXECUÇÕES INDIVIDUAIS SIL (sil_val1, sil_val2, sil_val3)
# ==============================================================================
print("\n[6/7] Gerando gráficos individuais de SIL (STM32)...")
sil_sims = [
    ("sil_val1", "20261006_161042_referencia_mpc.csv"),
    ("sil_val2", "20261006_161234_referencia_val2_degraus.csv"),
    ("sil_val3", "20261006_161422_referencia_val3_senoidal.csv")
]

for base_name, csv_file in sil_sims:
    df = pd.read_csv(os.path.join(sil_dir, csv_file))
    t = df['tempo_ms'] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    ax1.plot(t, df['referencia'], color='black', linestyle='--', linewidth=1.1, label='Referência')
    ax1.plot(t, df['angulo_deg'], color='#2ca02c', linestyle='-', linewidth=1.4, label='STM32 em C (SIL)')
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc='lower right' if 'val2' not in base_name else 'upper right', framealpha=0.95)
    
    ax2.plot(t, df['u_pct'], color='#2ca02c', linestyle='-', linewidth=1.2, label='Controle $u$')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax1.set_xlim(0, t.max())
    
    plt.tight_layout()
    save_academic_plot(fig, base_name, extra_dirs=[sil_dir])
    plt.close(fig)

# ==============================================================================
# 7. EXECUÇÕES INDIVIDUAIS BANCADA REAL (real_val1, real_val2, real_val3)
# ==============================================================================
print("\n[7/7] Gerando gráficos individuais da Bancada Real...")
real_sims = [
    ("real_val1", "20261006_181737_referencia_mpc.csv"),
    ("real_val2", "20261006_182048_referencia_val2_degraus.csv"),
    ("real_val3", "20261006_182839_referencia_val3_senoidal.csv")
]

for base_name, csv_file in real_sims:
    df = pd.read_csv(os.path.join(real_dir, csv_file))
    t = df['tempo_ms'] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    ax1.plot(t, df['referencia'], color='black', linestyle='--', linewidth=1.1, label='Referência')
    ax1.plot(t, df['angulo_deg'], color='#d62728', linestyle='-', linewidth=1.4, label='Bancada Real (Aeropêndulo)')
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc='lower right' if 'val2' not in base_name else 'upper right', framealpha=0.95)
    
    ax2.plot(t, df['u_pct'], color='#d62728', linestyle='-', linewidth=1.2, label='Controle $u$')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax1.set_xlim(0, t.max())
    
    plt.tight_layout()
    save_academic_plot(fig, base_name, extra_dirs=[real_dir])
    plt.close(fig)

# ==============================================================================
# 8. CÓPIA E PADRONIZAÇÃO DE ANN LOSS E ANN R2
# ==============================================================================
# Gera versão vetorial e 300 DPI de ann_loss e ann_r2
for name in ["ann_loss", "ann_r2"]:
    src_p = os.path.join(py_dir, "Training_and_Validation_Loss.png" if name == "ann_loss" else "Test_Set_R2__0_9990.png")
    if os.path.exists(src_p):
        img_arr = mpimg.imread(src_p)
        fig, ax = plt.subplots(figsize=(6.0, 3.5) if name == "ann_loss" else (7.2, 2.5))
        ax.imshow(img_arr)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_academic_plot(fig, name, extra_dirs=[py_dir])
        plt.close(fig)

print("\n=======================================================")
print("TODOS OS PLOTS FORAM GERADOS E SALVOS COM SUCESSO!")
print(f"Formatos gerados: .png (300 DPI) e .pdf (vetorial)")
print(f"Pastas de destino:")
for d in dest_dirs:
    print(f"  - {d}")
print("=======================================================\n")
