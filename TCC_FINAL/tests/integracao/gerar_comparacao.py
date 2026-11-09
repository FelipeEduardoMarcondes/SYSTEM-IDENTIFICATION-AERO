import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuracao de estilo (modo escuro)
plt.style.use('dark_background')
cores = {'python': '#58a6ff', 'sil': '#3fb950', 'real': '#da3633', 'ref': '#f0883e'}

exp_dir = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_legado_narx-mpc"
py_dir = os.path.join(exp_dir, "2_mpc_python")
sil_dir = os.path.join(exp_dir, "3_sil_stm32")
real_dir = os.path.join(exp_dir, "4_aeropendulo")

def get_latest_csv(folder, pattern):
    files = glob.glob(os.path.join(folder, f"*{pattern}*.csv"))
    if not files: return None
    return max(files, key=os.path.getmtime)

tests = [
    {"name": "Validação Principal (Multiseno + Degraus)", "py_csv": "simulacao_python.csv", "pattern": "referencia_mpc"},
    {"name": "Validação 2 (Degraus Aleatórios)", "py_csv": "simulacao_python_val2_degraus.csv", "pattern": "referencia_val2_degraus"},
    {"name": "Validação 3 (Senoidal Suave)", "py_csv": "simulacao_python_val3_senoidal.csv", "pattern": "referencia_val3_senoidal"},
]

report_md = "# Comparação de Experimentos (Python vs SIL vs Real)\n\n"
report_md += "Esta tabela compara o **RMSE** (Erro Quadrático Médio) do ângulo em relação à referência para os três ambientes testados.\n\n"
report_md += "| Teste | RMSE Python | RMSE SIL (STM32) | RMSE Bancada Real |\n"
report_md += "|-------|-------------|------------------|-------------------|\n"

artifact_dir = r"c:\Users\vicio\.gemini\antigravity-ide\brain\d56c58f0-a7c1-4a55-b68a-6c404f696320\scratch"
os.makedirs(artifact_dir, exist_ok=True)

for idx, test in enumerate(tests):
    py_path = os.path.join(py_dir, test["py_csv"])
    sil_path = get_latest_csv(sil_dir, test["pattern"])
    real_path = get_latest_csv(real_dir, test["pattern"])
    
    datasets = {}
    
    # Load Python
    if os.path.exists(py_path):
        df_py = pd.read_csv(py_path)
        # Python format: tempo_ms, angulo_deg, controle_u
        # For python, the reference was exported differently? Let's check.
        # It's usually "referencia" or we extract it from the SIL dataset if missing.
        # Actually in mpc_v4.py we exported it as:
        # tempo_ms, angulo_deg, controle_u
        df_py['tempo_s'] = df_py['tempo_ms'] / 1000.0
        datasets['python'] = df_py
        
    # Load SIL
    if sil_path and os.path.exists(sil_path):
        df_sil = pd.read_csv(sil_path)
        df_sil.columns = [c.strip() for c in df_sil.columns]
        df_sil['tempo_s'] = df_sil['tempo_ms'] / 1000.0
        datasets['sil'] = df_sil
        
    # Load Real
    if real_path and os.path.exists(real_path):
        df_real = pd.read_csv(real_path)
        df_real.columns = [c.strip() for c in df_real.columns]
        df_real['tempo_s'] = df_real['tempo_ms'] / 1000.0
        datasets['real'] = df_real

    # Calculate RMSE
    rmses = {'python': 'N/A', 'sil': 'N/A', 'real': 'N/A'}
    
    ref_signal = None
    t_ref = None
    
    for key, df in datasets.items():
        if 'referencia' in df.columns:
            ref = df['referencia'].values
            t = df['tempo_s'].values
            if ref_signal is None:
                ref_signal = ref
                t_ref = t
            # Only calculate RMSE where reference is valid (e.g. > 0 or whatever, but here it's 45-60)
            valid = (ref > 10.0) & (df['tempo_s'] > 5.0) # After 5s startup
            if valid.sum() > 0:
                rmse = np.sqrt(np.mean((df['angulo_deg'].values[valid] - ref[valid])**2))
                rmses[key] = f"{rmse:.2f}°"
        else:
            # If reference is missing in python, calculate against SIL's reference interpolation
            if ref_signal is not None:
                ref_interp = np.interp(df['tempo_s'], t_ref, ref_signal)
                valid = (ref_interp > 10.0) & (df['tempo_s'] > 5.0)
                if valid.sum() > 0:
                    rmse = np.sqrt(np.mean((df['angulo_deg'].values[valid] - ref_interp[valid])**2))
                    rmses[key] = f"{rmse:.2f}°"

    report_md += f"| {test['name']} | {rmses['python']} | {rmses['sil']} | {rmses['real']} |\n"

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.suptitle(test['name'], fontsize=14, color='white')
    
    if ref_signal is not None:
        ax1.plot(t_ref, ref_signal, color=cores['ref'], ls='--', lw=2, label='Referência')
        
    if 'python' in datasets:
        ax1.plot(datasets['python']['tempo_s'], datasets['python']['angulo_deg'], color=cores['python'], lw=1.5, alpha=0.8, label='Python (Simulador)')
        ax2.plot(datasets['python']['tempo_s'], datasets['python']['u_pct'], color=cores['python'], lw=1.5, alpha=0.8)
        
    if 'sil' in datasets:
        ax1.plot(datasets['sil']['tempo_s'], datasets['sil']['angulo_deg'], color=cores['sil'], lw=1.5, alpha=0.8, label='STM32 (SIL)')
        ax2.plot(datasets['sil']['tempo_s'], datasets['sil']['u_pct'], color=cores['sil'], lw=1.5, alpha=0.8)
        
    if 'real' in datasets:
        ax1.plot(datasets['real']['tempo_s'], datasets['real']['angulo_deg'], color=cores['real'], lw=1.5, alpha=0.8, label='Bancada Real')
        ax2.plot(datasets['real']['tempo_s'], datasets['real']['u_pct'], color=cores['real'], lw=1.5, alpha=0.8)

    ax1.set_ylabel('Ângulo (deg)')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    ax2.set_ylabel('Controle (%)')
    ax2.set_xlabel('Tempo (s)')
    ax2.grid(True, alpha=0.2)
    
    # Restrict xlim to max time
    max_t = 0
    for key, df in datasets.items():
        if df['tempo_s'].max() > max_t: max_t = df['tempo_s'].max()
    ax1.set_xlim(0, max_t)

    plt.tight_layout()
    plot_path = os.path.join(artifact_dir, f"comparacao_plot_{idx}.png")
    # Windows path replacement for markdown embedding
    embed_path = plot_path.replace('\\', '/')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_md += f"\n![Comparativo {test['name']}](file:///{embed_path})\n\n"

# Save the markdown report to artifact dir
report_path = os.path.join(artifact_dir, "..", "comparacao_experimentos.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)

print("Comparacao gerada com sucesso!")
