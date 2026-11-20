"""
animar_pendulo_gif.py — Gera a animação 2D EXCLUSIVA do Aeropêndulo
TOTALMENTE LIMPA (sem legendas, sem título e sem caixas de texto).

Gera tanto em GIF quanto em MP4 (temas Escuro e Claro):
1. Ensaio Validação 1 (seguindo o CSV real, em versão rápida e em 1.0x)
2. Varredura do Envelope Operacional (0° a 120°)
"""

import os
import time
import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

CSV_DEFAULT = r"c:\Users\vicio\Documents\AEROPENDULO\data\coletas\RODADA_20261117\20261117_182241_referencia_legado_val1_mpc_hybrid.csv"


def carregar_dados(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    t = df["tempo_ms"].values.astype(float) / 1000.0
    ang = df["angulo_deg"].values.astype(float)
    u = df["u_pct"].values.astype(float)
    ref = df["referencia"].values.astype(float)
    return t, ang, u, ref


def canvas_to_rgb(fig) -> np.ndarray:
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w, _ = rgba.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    return rgba[:h2, :w2, :3].copy()


def rgb_to_pil_quantized(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).quantize(colors=64, method=Image.Quantize.FASTOCTREE)


def renderizar_pendulo_val1_limpo(t, ang, u, ref, pasta_out: str, base_nome: str, tema: str = "escuro", step: int = 20, salvar_1x_mp4: bool = True):
    """
    Renderiza APENAS o desenho 2D do aeropêndulo seguindo os dados da Validação 1,
    sem nenhuma legenda, título ou caixa de texto.
    """
    is_dark = (tema == "escuro")
    bg_color = "#0d1117" if is_dark else "#f8fafc"
    tower_color = "#484f58" if is_dark else "#64748b"
    arc_color = "#30363d" if is_dark else "#cbd5e1"
    txt_ang_color = "#8b949e" if is_dark else "#475569"
    arm_color = "#e6edf3" if is_dark else "#1e293b"
    edf_fill = "#58a6ff" if is_dark else "#0284c7"
    edf_edge = "#ffffff" if is_dark else "#0f172a"
    ref_color = "#f0883e" if is_dark else "#dc2626"
    thrust_color = "#3fb950" if is_dark else "#16a34a"
    cw_color = "#8b949e" if is_dark else "#475569"

    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=bg_color, dpi=100)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(-1.38, 1.12)
    ax.axis("off")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Arco graduado e marcações discretas dos ângulos (0°, 30°, 45°, 60°, 90°)
    for ang_marc in [0, 30, 45, 60, 90]:
        rad_m = np.deg2rad(ang_marc)
        xm = np.sin(rad_m)
        ym = -np.cos(rad_m)
        ax.plot([0, 1.02 * xm], [0, 1.02 * ym], color=arc_color, lw=0.9, ls=":", zorder=1)
        ax.text(1.13 * xm, 1.13 * ym, f"{ang_marc}°", color=txt_ang_color,
                fontsize=9.5, fontfamily="monospace", fontweight="bold", ha="center", va="center")

    arco = patches.Arc((0, 0), 2.0, 2.0, angle=-90, theta1=-5, theta2=105,
                       color=arc_color, lw=1.4, ls="--", zorder=1)
    ax.add_patch(arco)

    # Estrutura física da torre / suporte triangular e base
    ax.plot([-0.38, 0.0, 0.38], [-1.25, 0.0, -1.25], color=tower_color, lw=3.0, solid_capstyle="round", zorder=2)
    ax.plot([-0.62, 0.62], [-1.25, -1.25], color=tower_color, lw=5.0, solid_capstyle="round", zorder=2)

    # Elementos móveis (Referência, Jato de Empuxo, Haste, Contrapeso, Turbina EDF, Pivô)
    (ln_ref,) = ax.plot([], [], color=ref_color, lw=2.2, ls="--", alpha=0.9, zorder=3)
    (pt_ref_tip,) = ax.plot([], [], "o", color=ref_color, markersize=6, zorder=3)

    (ln_thrust1,) = ax.plot([], [], color=thrust_color, lw=4.5, solid_capstyle="round", alpha=0.9, zorder=3)
    (ln_thrust2,) = ax.plot([], [], color=thrust_color, lw=2.2, solid_capstyle="round", alpha=0.6, zorder=3)
    (ln_thrust3,) = ax.plot([], [], color=thrust_color, lw=2.2, solid_capstyle="round", alpha=0.6, zorder=3)

    (ln_haste,) = ax.plot([], [], color=arm_color, lw=5.5, solid_capstyle="round", zorder=4)
    (pt_cw,) = ax.plot([], [], "s", color=cw_color, markersize=10, markeredgecolor=arm_color, markeredgewidth=1.2, zorder=5)
    (pt_edf,) = ax.plot([], [], "o", color=edf_fill, markersize=18, markeredgecolor=edf_edge, markeredgewidth=2.2, zorder=6)
    ax.plot([0], [0], "o", color=ref_color, markersize=7, zorder=7)

    fig.canvas.draw()
    bg_static = fig.canvas.copy_from_bbox(fig.bbox)

    # 1) GIF Fluido (step=20 -> ~414 frames) e MP4 1.0x (step=4 -> 25 fps, 82.75s)
    def render_sequence(step_val, out_gif_path=None, out_mp4_path=None):
        indices = list(range(step_val - 1, len(t), step_val))
        if indices[-1] != len(t) - 1:
            indices.append(len(t) - 1)
        fps_val = 100.0 / step_val

        writer = None
        if out_mp4_path:
            writer = imageio.get_writer(
                out_mp4_path, fps=fps_val, codec="libx264", quality=8,
                pixelformat="yuv420p", macro_block_size=2
            )
        frames_gif = []

        for idx in indices:
            fig.canvas.restore_region(bg_static)

            ang_atual = ang[idx]
            ref_atual = ref[idx]
            u_atual = u[idx]

            rad_ang = np.deg2rad(ang_atual)
            rad_ref = np.deg2rad(ref_atual)

            x_edf, y_edf = np.sin(rad_ang), -np.cos(rad_ang)
            x_cw, y_cw = -0.28 * np.sin(rad_ang), 0.28 * np.cos(rad_ang)
            x_ref, y_ref = 1.0 * np.sin(rad_ref), -1.0 * np.cos(rad_ref)

            ln_ref.set_data([0, x_ref], [0, y_ref])
            pt_ref_tip.set_data([x_ref], [y_ref])

            ln_haste.set_data([x_cw, x_edf], [y_cw, y_edf])
            pt_cw.set_data([x_cw], [y_cw])
            pt_edf.set_data([x_edf], [y_edf])

            # Jato de empuxo perpendicular à haste
            esc = max(0.0, u_atual / 100.0) * 0.45
            nx, ny = -np.cos(rad_ang), -np.sin(rad_ang)
            tx, ty = x_edf + esc * nx, y_edf + esc * ny
            ln_thrust1.set_data([x_edf, tx], [y_edf, ty])
            ln_thrust2.set_data([x_edf, tx + 0.04 * np.sin(rad_ang)], [y_edf, ty - 0.04 * np.cos(rad_ang)])
            ln_thrust3.set_data([x_edf, tx - 0.04 * np.sin(rad_ang)], [y_edf, ty + 0.04 * np.cos(rad_ang)])

            for artist in (ln_ref, pt_ref_tip, ln_thrust1, ln_thrust2, ln_thrust3, ln_haste, pt_cw, pt_edf):
                ax.draw_artist(artist)

            fig.canvas.blit(fig.bbox)
            rgb = canvas_to_rgb(fig)
            if writer is not None:
                writer.append_data(rgb)
            if out_gif_path is not None:
                frames_gif.append(rgb_to_pil_quantized(rgb))

        if writer is not None:
            writer.close()
            print(f"-> MP4 Salvo: {out_mp4_path} ({os.path.getsize(out_mp4_path)/1024:.1f} KB)")

        if out_gif_path is not None and frames_gif:
            # 25 fps no GIF resumo (ou 100ms se step=10)
            dur_ms = 40 if step_val == 25 else int(step_val * 10)
            frames_gif[0].save(
                out_gif_path, save_all=True, append_images=frames_gif[1:],
                duration=dur_ms, loop=0, optimize=True
            )
            print(f"-> GIF Salvo: {out_gif_path} ({os.path.getsize(out_gif_path)/1024:.1f} KB)")

    # Gera MP4 1.0x (25 fps, 82.75 s) e GIF 1.0x (10 fps, 82.75 s) + GIF rápido (25 fps, 13 s)
    mp4_1x = os.path.join(pasta_out, f"{base_nome}_apenas_aero_1x_{tema}_sem_legenda.mp4")
    gif_1x = os.path.join(pasta_out, f"{base_nome}_apenas_aero_1x_{tema}_sem_legenda.gif")
    gif_fast = os.path.join(pasta_out, f"{base_nome}_apenas_aero_{tema}_sem_legenda.gif")

    if salvar_1x_mp4:
        render_sequence(step_val=4, out_gif_path=None, out_mp4_path=mp4_1x)
    render_sequence(step_val=10, out_gif_path=gif_1x, out_mp4_path=None)
    render_sequence(step_val=25, out_gif_path=gif_fast, out_mp4_path=None)
    plt.close(fig)


def renderizar_pendulo_120deg_limpo(pasta_out: str, tema: str = "escuro"):
    """
    Renderiza APENAS o aeropêndulo varrendo suavemente de 0° a 120° e voltando,
    TOTALMENTE LIMPO (sem legendas, sem título e sem caixa de texto).
    """
    is_dark = (tema == "escuro")
    bg_color = "#0d1117" if is_dark else "#f8fafc"
    tower_color = "#484f58" if is_dark else "#64748b"
    arc_color = "#30363d" if is_dark else "#cbd5e1"
    txt_ang_color = "#8b949e" if is_dark else "#475569"
    arm_color = "#e6edf3" if is_dark else "#1e293b"
    edf_fill = "#58a6ff" if is_dark else "#0284c7"
    edf_edge = "#ffffff" if is_dark else "#0f172a"
    thrust_color = "#3fb950" if is_dark else "#16a34a"
    grav_color = "#da3633" if is_dark else "#dc2626"
    cw_color = "#8b949e" if is_dark else "#475569"

    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=bg_color, dpi=100)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(-1.38, 1.12)
    ax.axis("off")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    for ang_marc in [0, 30, 45, 60, 90, 120]:
        rad_m = np.deg2rad(ang_marc)
        xm = np.sin(rad_m)
        ym = -np.cos(rad_m)
        cor_l = grav_color if ang_marc == 120 else ("#a855f7" if ang_marc == 90 else arc_color)
        lw_l = 1.4 if ang_marc in (90, 120) else 0.9
        ls_l = "--" if ang_marc == 120 else ":"
        ax.plot([0, 1.02 * xm], [0, 1.02 * ym], color=cor_l, lw=lw_l, ls=ls_l, zorder=1)
        ax.text(1.13 * xm, 1.13 * ym, f"{ang_marc}°", color=(cor_l if ang_marc in (90, 120) else txt_ang_color),
                fontsize=9.5, fontfamily="monospace", fontweight="bold", ha="center", va="center")

    arco = patches.Arc((0, 0), 2.0, 2.0, angle=-90, theta1=0, theta2=120,
                       color=arc_color, lw=1.4, ls="--", zorder=1)
    ax.add_patch(arco)

    ax.plot([-0.38, 0.0, 0.38], [-1.25, 0.0, -1.25], color=tower_color, lw=3.0, solid_capstyle="round", zorder=2)
    ax.plot([-0.62, 0.62], [-1.25, -1.25], color=tower_color, lw=5.0, solid_capstyle="round", zorder=2)

    (ln_grav_arm,) = ax.plot([], [], color="#a855f7", lw=2.0, ls="-.", alpha=0.85, zorder=2)
    (ln_weight,) = ax.plot([], [], color=grav_color, lw=2.0, ls="-", alpha=0.85, zorder=3)
    (ln_thrust,) = ax.plot([], [], color=thrust_color, lw=4.5, solid_capstyle="round", alpha=0.9, zorder=3)
    (ln_haste,) = ax.plot([], [], color=arm_color, lw=5.5, solid_capstyle="round", zorder=4)
    (pt_cw,) = ax.plot([], [], "s", color=cw_color, markersize=10, markeredgecolor=arm_color, markeredgewidth=1.2, zorder=5)
    (pt_edf,) = ax.plot([], [], "o", color=edf_fill, markersize=18, markeredgecolor=edf_edge, markeredgewidth=2.2, zorder=6)
    ax.plot([0], [0], "o", color=grav_color, markersize=7, zorder=7)

    fig.canvas.draw()
    bg_static = fig.canvas.copy_from_bbox(fig.bbox)

    # Trajetória suave de 0° a 120° e retorno a 0°
    ang_up = np.linspace(0, 120, 90)
    ang_hold = np.full(20, 120.0)
    ang_down = np.linspace(120, 0, 70)
    ang_pause = np.zeros(15)
    ang_seq = np.concatenate([ang_pause, ang_up, ang_hold, ang_down])

    out_gif = os.path.join(pasta_out, f"aeropendulo_120deg_isolado_{tema}_sem_legenda.gif")
    out_mp4 = os.path.join(pasta_out, f"aeropendulo_120deg_isolado_{tema}_sem_legenda.mp4")

    writer = imageio.get_writer(out_mp4, fps=25, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2)
    frames_gif = []

    for th in ang_seq:
        fig.canvas.restore_region(bg_static)
        rad = np.deg2rad(th)
        x_edf, y_edf = np.sin(rad), -np.cos(rad)
        x_cw, y_cw = -0.28 * np.sin(rad), 0.28 * np.cos(rad)

        ln_haste.set_data([x_cw, x_edf], [y_cw, y_edf])
        pt_cw.set_data([x_cw], [y_cw])
        pt_edf.set_data([x_edf], [y_edf])

        # Braço de alavanca gravitacional L*sin(theta)
        ln_grav_arm.set_data([0, x_edf], [y_edf, y_edf])
        # Vetor peso vertical para baixo
        ln_weight.set_data([x_edf, x_edf], [y_edf, y_edf - 0.28])

        # Empuxo proporcional a sqrt(sin(theta))
        u_eq = np.sqrt(max(0.0, np.sin(rad))) if th <= 90 else np.sqrt(max(0.0, np.sin(rad))) * 1.05
        esc = 0.42 * u_eq
        tx = x_edf - esc * np.cos(rad)
        ty = y_edf - esc * np.sin(rad)
        ln_thrust.set_data([x_edf, tx], [y_edf, ty])

        for artist in (ln_grav_arm, ln_weight, ln_thrust, ln_haste, pt_cw, pt_edf):
            ax.draw_artist(artist)

        fig.canvas.blit(fig.bbox)
        rgb = canvas_to_rgb(fig)
        writer.append_data(rgb)
        frames_gif.append(rgb_to_pil_quantized(rgb))

    writer.close()
    frames_gif[0].save(out_gif, save_all=True, append_images=frames_gif[1:], duration=40, loop=0, optimize=True)
    plt.close(fig)
    print(f"-> Envelope 120° Sem Legenda ({tema}) salvo: {out_gif} e {out_mp4}")


if __name__ == "__main__":
    t0 = time.time()
    csv_path = CSV_DEFAULT
    t, ang, u, ref = carregar_dados(csv_path)
    pasta = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    print("1. Gerando animação 2D apenas do Aeropêndulo (Validação 1) SEM legendas...")
    renderizar_pendulo_val1_limpo(t, ang, u, ref, pasta, base, tema="escuro")
    renderizar_pendulo_val1_limpo(t, ang, u, ref, pasta, base, tema="claro")

    print("2. Gerando animação 2D apenas do Aeropêndulo (Envelope 0° a 120°) SEM legendas...")
    renderizar_pendulo_120deg_limpo(pasta, tema="escuro")
    renderizar_pendulo_120deg_limpo(pasta, tema="claro")
    print(f"Concluído em {time.time() - t0:.1f}s!")
