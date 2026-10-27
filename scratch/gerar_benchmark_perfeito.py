import os
import numpy as np
import matplotlib.pyplot as plt

def gerar_benchmark_horizontal(save_paths):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    labels = [
        'C Nativo (-O0)',
        'C Nativo (-O1)',
        'C Nativo (-Os)',
        'C Nativo (-O3)',
        'X-CUBE-AI (Sem Comp.)',
        'X-CUBE-AI (Alta Comp.)'
    ]

    tempos = [4.96, 2.46, 2.08, 1.98, 1.69, 1.82]
    flashes = [130.50, 130.50, 130.50, 130.50, 130.50, 17.75]
    speedups = [1.00, 2.02, 2.38, 2.51, 2.93, 2.73]
    ocupacoes = [49.6, 24.6, 20.8, 19.8, 16.9, 18.2]

    # Inverter ordem para que -O0 fique no topo (leitura de cima para baixo)
    y_labels = labels[::-1]
    y_tempos = tempos[::-1]
    y_flashes = flashes[::-1]
    y_speedups = speedups[::-1]
    y_ocupacoes = ocupacoes[::-1]

    # y_pos com leve separação entre GCC e X-CUBE-AI
    # 0: X-CUBE-AI Alta Comp
    # 1: X-CUBE-AI Sem Comp
    # 2.3: C Nativo -O3
    # 3.3: C Nativo -Os
    # 4.3: C Nativo -O1
    # 5.3: C Nativo -O0
    y_pos = np.array([0.5, 1.5, 2.8, 3.8, 4.8, 5.8])

    # Cores acadêmicas elegantes
    time_colors = [
        '#0284c7',  # X-CUBE Alta Comp
        '#0369a1',  # X-CUBE Sem Comp (Mais rápido - azul petróleo escuro)
        '#1d4ed8',  # C Nativo -O3
        '#2563eb',  # C Nativo -Os
        '#3b82f6',  # C Nativo -O1
        '#64748b',  # C Nativo -O0 (Cinza ardósia - base)
    ]

    flash_colors = [
        '#16a34a',  # Alta Comp - verde esmeralda destacando economia
        '#94a3b8',  # Demais em cinza neutro
        '#94a3b8',
        '#94a3b8',
        '#94a3b8',
        '#94a3b8',
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)

    # Fundos suaves para identificar visualmente os blocos
    ax1.axhspan(2.15, 6.4, facecolor='#f8fafc', alpha=0.9, zorder=0)
    ax1.axhspan(-0.1, 2.15, facecolor='#f0fdf4', alpha=0.6, zorder=0)

    ax2.axhspan(2.15, 6.4, facecolor='#f8fafc', alpha=0.9, zorder=0)
    ax2.axhspan(-0.1, 2.15, facecolor='#f0fdf4', alpha=0.6, zorder=0)

    # Linha divisória sutil
    ax1.axhline(2.15, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)
    ax2.axhline(2.15, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)

    # --- SUBPLOT 1: TEMPO DE INFERÊNCIA ---
    bars1 = ax1.barh(y_pos, y_tempos, height=0.60, color=time_colors,
                     edgecolor='#0f172a', linewidth=0.65, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, color='#0f172a')
    ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
    ax1.set_title('(a) Tempo de Inferência por Passo ($t_{\\mathrm{ann}}$)', fontweight='bold', pad=10)
    ax1.set_xlim(0, 6.6)
    ax1.set_ylim(-0.2, 6.5)
    ax1.grid(True, axis='x', zorder=2)

    for b, t, sp, oc in zip(bars1, y_tempos, y_speedups, y_ocupacoes):
        w = b.get_width()
        is_best = (t == 1.69)
        y_mid = b.get_y() + b.get_height() / 2
        if is_best:
            txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%) [Mais Rápido]"
            ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                     fontsize=8.5, fontweight='bold', color='#0369a1', zorder=4)
        elif sp == 1.0:
            txt = f" {t:.2f} ms (1.00× Ref. | {oc:.1f}%)"
            ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                     fontsize=8.5, color='#475569', zorder=4)
        else:
            txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%)"
            ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                     fontsize=8.5, color='#1e293b', zorder=4)

    # --- SUBPLOT 2: FLASH ---
    bars2 = ax2.barh(y_pos, y_flashes, height=0.60, color=flash_colors,
                     edgecolor='#0f172a', linewidth=0.65, zorder=3)
    ax2.set_xlabel('Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
    ax2.set_title('(b) Pegada em Memória Flash dos Pesos', fontweight='bold', pad=10)
    ax2.set_xlim(0, 185)
    ax2.set_ylim(-0.2, 6.5)
    ax2.grid(True, axis='x', zorder=2)

    for b, f in zip(bars2, y_flashes):
        w = b.get_width()
        is_comp = (f < 50)
        y_mid = b.get_y() + b.get_height() / 2
        if is_comp:
            txt = f" {f:.2f} KiB (-86,4%) [Mais Econômico]"
            ax2.text(w + 3.5, y_mid, txt, va='center', ha='left',
                     fontsize=8.5, fontweight='bold', color='#15803d', zorder=4)
        else:
            txt = f" {f:.1f} KiB"
            ax2.text(w + 3.5, y_mid, txt, va='center', ha='left',
                     fontsize=8.5, color='#475569', zorder=4)

    plt.tight_layout()

    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300)
        print("Salvo:", p)
    plt.close(fig)

if __name__ == "__main__":
    paths = [
        r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_horizontal_perfeito.png",
        r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_horizontal_perfeito.pdf"
    ]
    gerar_benchmark_horizontal(paths)
