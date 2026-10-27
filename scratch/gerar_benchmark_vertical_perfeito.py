import os
import numpy as np
import matplotlib.pyplot as plt

def gerar_benchmark_vertical(save_paths):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    labels = [
        'C Nativo\n(-O0)',
        'C Nativo\n(-O1)',
        'C Nativo\n(-Os)',
        'C Nativo\n(-O3)',
        'X-CUBE-AI\n(Float32)',
        'X-CUBE-AI\n(Alta Comp.)'
    ]

    tempos = [4.96, 2.46, 2.08, 1.98, 1.69, 1.82]
    flashes = [130.50, 130.50, 130.50, 130.50, 130.50, 17.75]
    speedups = [1.00, 2.02, 2.38, 2.51, 2.93, 2.73]
    ocupacoes = [49.6, 24.6, 20.8, 19.8, 16.9, 18.2]

    x_pos = np.arange(len(labels))

    time_colors = [
        '#64748b',  # -O0
        '#3b82f6',  # -O1
        '#2563eb',  # -Os
        '#1d4ed8',  # -O3
        '#0284c7',  # X-CUBE Float32 (Mais Rápido)
        '#0ea5e9',  # X-CUBE Alta Comp
    ]

    flash_colors = [
        '#94a3b8',
        '#94a3b8',
        '#94a3b8',
        '#94a3b8',
        '#94a3b8',
        '#16a34a',  # Alta Comp
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.1))

    # --- SUBPLOT 1: TEMPO ---
    bars1 = ax1.bar(x_pos, tempos, width=0.55, color=time_colors,
                    edgecolor='#0f172a', linewidth=0.65, zorder=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
    ax1.set_title('(a) Tempo de Inferência por Passo ($t_{\\mathrm{ann}}$)', fontweight='bold', pad=10)
    ax1.set_ylim(0, 6.2)
    ax1.grid(True, axis='y', zorder=1)

    # Divisória entre GCC e X-CUBE
    ax1.axvline(3.5, color='#cbd5e1', linestyle='--', linewidth=1.2, zorder=2)

    for b, t, sp in zip(bars1, tempos, speedups):
        h = b.get_height()
        is_best = (t == 1.69)
        if is_best:
            lbl = f"{t:.2f} ms\n({sp:.2f}×)\n[Mais Rápido]"
            ax1.text(b.get_x() + b.get_width()/2, h + 0.15, lbl,
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0284c7')
        elif sp == 1.0:
            lbl = f"{t:.2f} ms\n(1.00× Ref.)"
            ax1.text(b.get_x() + b.get_width()/2, h + 0.15, lbl,
                     ha='center', va='bottom', fontsize=8, color='#64748b')
        else:
            lbl = f"{t:.2f} ms\n({sp:.2f}×)"
            ax1.text(b.get_x() + b.get_width()/2, h + 0.15, lbl,
                     ha='center', va='bottom', fontsize=8, color='#1e293b')

    # --- SUBPLOT 2: FLASH ---
    bars2 = ax2.bar(x_pos, flashes, width=0.55, color=flash_colors,
                    edgecolor='#0f172a', linewidth=0.65, zorder=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel('Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
    ax2.set_title('(b) Pegada em Memória Flash dos Pesos', fontweight='bold', pad=10)
    ax2.set_ylim(0, 165)
    ax2.grid(True, axis='y', zorder=1)

    # Divisória entre GCC e X-CUBE
    ax2.axvline(3.5, color='#cbd5e1', linestyle='--', linewidth=1.2, zorder=2)

    for b, f in zip(bars2, flashes):
        h = b.get_height()
        is_comp = (f < 50)
        if is_comp:
            lbl = f"{f:.1f} KiB\n(-86,4%)\n[Econômico]"
            ax2.text(b.get_x() + b.get_width()/2, h + 3.0, lbl,
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#15803d')
        else:
            lbl = f"{f:.1f} KiB"
            ax2.text(b.get_x() + b.get_width()/2, h + 3.0, lbl,
                     ha='center', va='bottom', fontsize=8, color='#475569')

    plt.tight_layout()

    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300)
        print("Salvo:", p)
    plt.close(fig)

if __name__ == "__main__":
    paths = [
        r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_vertical_perfeito.png",
        r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_vertical_perfeito.pdf"
    ]
    gerar_benchmark_vertical(paths)
