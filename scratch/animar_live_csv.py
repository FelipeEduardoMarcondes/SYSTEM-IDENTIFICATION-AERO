"""
animar_live_csv.py — Gera vídeos MP4 (H.264) em velocidade 1.0x REAL (82.75 s, 25 fps)
apenas com o gráfico limpo (SEM legendas, sem barra de status e sem títulos),
ideal para colocar lado a lado com o vídeo real da bancada nos slides.

Arquivos gerados (todos em tempo real 1.0x):
1. *_grafico_1x_escuro_sem_legenda.mp4    : Eixo fixo (0 a 83s) no tema escuro
2. *_grafico_1x_claro_sem_legenda.mp4     : Eixo fixo (0 a 83s) no tema claro do TCC
3. *_grafico_1x_janela30s_sem_legenda.mp4 : Janela deslizante (30s) no tema escuro
"""

import os
import time
import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

CORES_DARK = {
    "angulo":   "#58a6ff",
    "ref":      "#f0883e",
    "controle": "#3fb950",
    "erro":     "#da3633",
    "zero":     "#6e7681",
}

MPL_DARK = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "text.color":       "#c9d1d9",
    "font.family":      "monospace",
    "lines.linewidth":  1.6,
}

CSV_DEFAULT = r"c:\Users\vicio\Documents\AEROPENDULO\data\coletas\RODADA_20261117\20261117_182241_referencia_legado_val1_mpc_hybrid.csv"


def carregar_dados(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    t = df["tempo_ms"].values.astype(float) / 1000.0
    ang = df["angulo_deg"].values.astype(float)
    u = df["u_pct"].values.astype(float)
    ref = df["referencia"].values.astype(float)
    erro = ang - ref
    return t, ang, u, ref, erro


def canvas_to_rgb(fig) -> np.ndarray:
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w, _ = rgba.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    return rgba[:h2, :w2, :3].copy()


def gerar_grafico_completo_1x_escuro_mp4(t, ang, u, ref, erro, out_mp4: str, step: int = 4):
    """
    Gráfico completo (0 a 82.75 s) em MP4 (25 fps = 1.0x exato), tema escuro, SEM legendas/títulos.
    """
    plt.rcParams.update(MPL_DARK)
    t_total = t[-1]
    fps = 100.0 / step  # 25.0 fps

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#0d1117", dpi=100)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[3, 2],
        hspace=0.10, left=0.08, right=0.97, top=0.96, bottom=0.10,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.grid(True, linestyle="-", color="#21262d", linewidth=0.8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.grid(True, which="minor", color="#1c2128", linewidth=0.5, alpha=0.7)

    (ln_ref,) = ax1.plot([], [], color=CORES_DARK["ref"], lw=1.6, ls="--", alpha=0.95)
    (ln_ang,) = ax1.plot([], [], color=CORES_DARK["angulo"], lw=2.2)
    (pt_ang,) = ax1.plot([], [], "o", color="#ffffff", markeredgecolor=CORES_DARK["angulo"], markeredgewidth=2.0, markersize=6)

    (ln_u,) = ax2.plot([], [], color=CORES_DARK["controle"], lw=1.9)
    (pt_u,) = ax2.plot([], [], "o", color="#ffffff", markeredgecolor=CORES_DARK["controle"], markeredgewidth=2.0, markersize=6)

    ax1.axhline(0, color=CORES_DARK["zero"], lw=0.6, ls=":")
    ax2.axhline(0, color=CORES_DARK["zero"], lw=0.6, ls=":")

    ax1.set_ylabel("Angulo (deg)", fontsize=11)
    ax1.set_xlim(0, t_total + 0.5)
    ax1.set_ylim(-5, 78)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_ylabel("Controle (%)", fontsize=11)
    ax2.set_xlabel("Tempo (s)", fontsize=11)
    ax2.set_xlim(0, t_total + 0.5)
    ax2.set_ylim(-5, 85)

    fig.canvas.draw()
    bg_full = fig.canvas.copy_from_bbox(fig.bbox)

    indices = list(range(step - 1, len(t), step))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)

    t0 = time.time()
    print(f"[1/3] Exportando MP4 Completo 1x Sem Legenda (Tema Escuro, {fps:.0f} fps): {len(indices)} frames...")

    writer = imageio.get_writer(
        out_mp4, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2
    )

    for idx in indices:
        fig.canvas.restore_region(bg_full)

        ta = t[:idx + 1]
        anga = ang[:idx + 1]
        ua = u[:idx + 1]
        refa = ref[:idx + 1]
        t_max = ta[-1]

        ln_ref.set_data(ta, refa)
        ln_ang.set_data(ta, anga)
        pt_ang.set_data([t_max], [anga[-1]])

        ln_u.set_data(ta, ua)
        pt_u.set_data([t_max], [ua[-1]])

        ax1.draw_artist(ln_ref)
        ax1.draw_artist(ln_ang)
        ax1.draw_artist(pt_ang)
        ax2.draw_artist(ln_u)
        ax2.draw_artist(pt_u)

        fig.canvas.blit(fig.bbox)
        writer.append_data(canvas_to_rgb(fig))

    writer.close()
    plt.close(fig)
    print(f"-> Salvo em {time.time() - t0:.1f}s: {out_mp4} ({os.path.getsize(out_mp4) / 1024:.1f} KB)")


def gerar_grafico_completo_1x_claro_mp4(t, ang, u, ref, erro, out_mp4: str, step: int = 4):
    """
    Gráfico completo (0 a 82.75 s) em MP4 (25 fps = 1.0x exato), tema claro do TCC, SEM legendas/títulos.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#111111",
        "axes.titlecolor":  "#111111",
        "xtick.color":      "#222222",
        "ytick.color":      "#222222",
        "grid.color":       "#cccccc",
        "grid.linewidth":   0.6,
        "text.color":       "#111111",
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "lines.linewidth":  1.6,
    })
    t_total = t[-1]
    fps = 100.0 / step

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white", dpi=100)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[2.2, 1.2],
        hspace=0.10, left=0.08, right=0.97, top=0.96, bottom=0.10,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    (ln_ref,) = ax1.plot([], [], color="black", lw=1.5, ls="--")
    (ln_ang,) = ax1.plot([], [], color="#d62728", lw=2.0)
    (pt_ang,) = ax1.plot([], [], "o", color="#d62728", markersize=6)

    ax2.axhline(0, color="gray", ls="--", lw=0.7)
    ax2.axhline(80, color="gray", ls="--", lw=0.7)
    (ln_u,) = ax2.plot([], [], color="#d62728", lw=1.8)
    (pt_u,) = ax2.plot([], [], "o", color="#d62728", markersize=6)

    ax1.set_ylabel("Ângulo θ [°]", fontsize=11)
    ax1.set_xlim(0, t_total + 0.5)
    ax1.set_ylim(-5, 78)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_ylabel("Controle u [%]", fontsize=11)
    ax2.set_xlabel("Tempo [s]", fontsize=11)
    ax2.set_xlim(0, t_total + 0.5)
    ax2.set_ylim(-5, 85)

    fig.canvas.draw()
    bg_full = fig.canvas.copy_from_bbox(fig.bbox)

    indices = list(range(step - 1, len(t), step))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)

    t0 = time.time()
    print(f"[2/3] Exportando MP4 Completo 1x Sem Legenda (Tema Claro TCC, {fps:.0f} fps): {len(indices)} frames...")

    writer = imageio.get_writer(
        out_mp4, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2
    )

    for idx in indices:
        fig.canvas.restore_region(bg_full)

        ta = t[:idx + 1]
        anga = ang[:idx + 1]
        ua = u[:idx + 1]
        refa = ref[:idx + 1]
        t_max = ta[-1]

        ln_ref.set_data(ta, refa)
        ln_ang.set_data(ta, anga)
        pt_ang.set_data([t_max], [anga[-1]])

        ln_u.set_data(ta, ua)
        pt_u.set_data([t_max], [ua[-1]])

        ax1.draw_artist(ln_ref)
        ax1.draw_artist(ln_ang)
        ax1.draw_artist(pt_ang)
        ax2.draw_artist(ln_u)
        ax2.draw_artist(pt_u)

        fig.canvas.blit(fig.bbox)
        writer.append_data(canvas_to_rgb(fig))

    writer.close()
    plt.close(fig)
    print(f"-> Salvo em {time.time() - t0:.1f}s: {out_mp4} ({os.path.getsize(out_mp4) / 1024:.1f} KB)")


def gerar_grafico_janela30s_1x_mp4(t, ang, u, ref, erro, out_mp4: str, step: int = 10):
    """
    Gráfico com janela deslizante de 30s em MP4 (10 fps = 1.0x exato), SEM legendas/títulos.
    """
    plt.rcParams.update(MPL_DARK)
    janela_s = 30.0
    fps = 100.0 / step  # 10.0 fps

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#0d1117", dpi=100)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[3, 2],
        hspace=0.10, left=0.08, right=0.97, top=0.96, bottom=0.10,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True, which="minor", color="#1c2128", linewidth=0.5, alpha=0.7)

    ax1.set_ylabel("Angulo (deg)", fontsize=11)
    ax2.set_ylabel("Controle (%)", fontsize=11)
    ax2.set_xlabel("Tempo (s)", fontsize=11)
    ax2.set_ylim(-5, 85)
    plt.setp(ax1.get_xticklabels(), visible=False)

    (ln_ang,) = ax1.plot([], [], color=CORES_DARK["angulo"], lw=2.2)
    (ln_ref,) = ax1.plot([], [], color=CORES_DARK["ref"], lw=1.6, ls="--", alpha=0.95)
    (ln_u,) = ax2.plot([], [], color=CORES_DARK["controle"], lw=1.9)

    ax1.axhline(0, color=CORES_DARK["zero"], lw=0.6, ls=":")
    ax2.axhline(0, color=CORES_DARK["zero"], lw=0.6, ls=":")

    indices = list(range(step - 1, len(t), step))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)

    t0 = time.time()
    print(f"[3/3] Exportando MP4 Janela Deslizante 30s Sem Legenda em 1x ({fps:.0f} fps): {len(indices)} frames...")

    writer = imageio.get_writer(
        out_mp4, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2
    )

    for idx in indices:
        ta = t[:idx + 1]
        anga = ang[:idx + 1]
        ua = u[:idx + 1]
        refa = ref[:idx + 1]

        t_max = ta[-1]
        t_min = max(0.0, t_max - janela_s)
        mask = ta >= t_min

        ln_ang.set_data(ta[mask], anga[mask])
        ln_ref.set_data(ta[mask], refa[mask])
        ln_u.set_data(ta[mask], ua[mask])

        vals = np.concatenate([anga[mask], refa[mask]])
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        mg = max(8.0, (vmax - vmin) * 0.2)
        ax1.set_ylim(vmin - mg, vmax + mg)
        ax1.set_xlim(t_min, t_max + 0.5)

        fig.canvas.draw()
        writer.append_data(canvas_to_rgb(fig))

    writer.close()
    plt.close(fig)
    print(f"-> Salvo em {time.time() - t0:.1f}s: {out_mp4} ({os.path.getsize(out_mp4) / 1024:.1f} KB)")


if __name__ == "__main__":
    csv_path = CSV_DEFAULT
    t, ang, u, ref, erro = carregar_dados(csv_path)
    pasta = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    mp4_escuro = os.path.join(pasta, f"{base}_grafico_1x_escuro_sem_legenda.mp4")
    mp4_claro  = os.path.join(pasta, f"{base}_grafico_1x_claro_sem_legenda.mp4")
    mp4_jan30  = os.path.join(pasta, f"{base}_grafico_1x_janela30s_sem_legenda.mp4")

    gerar_grafico_completo_1x_escuro_mp4(t, ang, u, ref, erro, mp4_escuro, step=4)
    gerar_grafico_completo_1x_claro_mp4(t, ang, u, ref, erro, mp4_claro, step=4)
    gerar_grafico_janela30s_1x_mp4(t, ang, u, ref, erro, mp4_jan30, step=10)
