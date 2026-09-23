import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import EXP_DIR, CORES, MPL_RC
plt.rcParams.update(MPL_RC)

def carregar_dados(caminho):
    df = pd.read_csv(caminho, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    if "tempo_ms" in df.columns:
        t = df["tempo_ms"].values / 1000.0
        y = df["angulo_deg"].values
        u = df["u_pct"].values
        ref = df["referencia"].values if "referencia" in df.columns else None
        return t, y, u, ref
    elif "Tempo_s" in df.columns:
        t = df["Tempo_s"].values
        y = df["Angulo_Medido"].values
        u = df["Controle_u"].values
        ref = df["Ref"].values if "Ref" in df.columns else None
        return t, y, u, ref
    return None, None, None, None

def selecionar_multiplos():
    arquivos = sorted(glob.glob(f"{EXP_DIR}/**/*.csv", recursive=True) + glob.glob("*.csv"))
    if not arquivos:
        print("Nenhum CSV encontrado.")
        return []
    
    data_dir = os.path.dirname(EXP_DIR)
    print("\nArquivos CSV disponíveis para comparação:")
    for i, f in enumerate(arquivos):
        size_kb = os.path.getsize(f) // 1024
        rel_f = os.path.relpath(f, data_dir)
        print(f"  [{i + 1:2d}] {rel_f}  ({size_kb} KB)")
        
    print("\nDigite os números dos arquivos separados por espaço (ex: 1 4 5):")
    escolha = input("Escolha: ").strip()
    
    selecionados = []
    for val in escolha.split():
        try:
            idx = int(val) - 1
            if 0 <= idx < len(arquivos):
                selecionados.append(arquivos[idx])
        except ValueError:
            pass
            
    return selecionados

def comparar_arquivos(arquivos):
    if not arquivos:
        return
        
    fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
    fig.suptitle("COMPARAÇÃO DE DADOS", fontsize=14, fontweight="bold", color="#e6edf3", fontfamily="monospace")
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    cores_plot = ["#58a6ff", "#3fb950", "#f0883e", "#da3633", "#a371f7", "#d2a8ff"]
    
    ref_plotada = False
    
    for i, caminho in enumerate(arquivos):
        nome = os.path.basename(caminho).replace(".csv", "")
        t, y, u, ref = carregar_dados(caminho)
        
        if t is None:
            continue
            
        cor = cores_plot[i % len(cores_plot)]
        
        # Plota a referência apenas do primeiro arquivo que tiver referência
        if ref is not None and not ref_plotada:
            ax1.plot(t, ref, color="#6e7681", lw=1.5, ls="--", label="Referência Base")
            ref_plotada = True
            
        ax1.plot(t, y, color=cor, lw=1.5, label=f"Ang: {nome}", alpha=0.85)
        ax2.plot(t, u, color=cor, lw=1.5, label=f"PWM: {nome}", alpha=0.85)

    ax1.set_ylabel("Angulo (deg)", fontsize=10)
    ax1.grid(True, color="#21262d")
    ax1.legend(loc="lower right", framealpha=0.9, facecolor="#0d1117", edgecolor="#30363d", fontsize=9)
    ax1.axhline(0, color=CORES["zero"], lw=0.7, ls=":")
    
    ax2.set_ylabel("Controle (%)", fontsize=10)
    ax2.set_xlabel("Tempo (s)", fontsize=10)
    ax2.grid(True, color="#21262d")
    ax2.legend(loc="lower right", framealpha=0.9, facecolor="#0d1117", edgecolor="#30363d", fontsize=9)
    ax2.axhline(0, color=CORES["zero"], lw=0.7, ls=":")
    ax2.set_ylim(-85, 85)
    
    plt.show()

if __name__ == "__main__":
    arquivos = selecionar_multiplos()
    comparar_arquivos(arquivos)
