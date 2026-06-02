"""
live_plot.py — Visualizador em tempo real para aquisição do aeropêndulo.

Completamente desacoplado da lógica de serial e aquisição.
Recebe listas de dados e redesenha a uma taxa controlada (LIVE_UPDATE).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from config import CORES, MPL_RC, LIVE_JANELA_S, LIVE_UPDATE

plt.rcParams.update(MPL_RC)


class LivePlot:
    """
    Janela de live plot com dois painéis (ângulo + controle) e barra de status.

    Uso
    ---
        lp = LivePlot(nome="chirp", janela_s=30)
        lp.atualizar(t_list, ang_list, u_list, ref_list)
        lp.fechar()
    """

    def __init__(
        self,
        nome: str = "",
        janela_s: float = LIVE_JANELA_S,
        transicoes: list | None = None,
    ):
        """
        Parâmetros
        ----------
        nome        : texto exibido no título (tipo de sinal ou nome do ensaio)
        janela_s    : largura da janela deslizante (s)
        transicoes  : lista de t_s onde a referência muda (marca linhas verticais)
        """
        self.janela_s       = janela_s
        self._ultimo_update = 0.0
        self._estop_visivel = False

        plt.ion()
        self.fig = plt.figure(figsize=(13, 7), facecolor="#0d1117")
        self.fig.suptitle(
            f"AEROPENDULO  |  LIVE  |  {nome}",
            fontsize=10, fontweight="bold",
            color="#e6edf3", y=0.97, fontfamily="monospace",
        )

        gs = gridspec.GridSpec(
            3, 1, height_ratios=[3, 2, 0.35],
            hspace=0.10, left=0.08, right=0.97, top=0.91, bottom=0.07,
        )

        self.ax1     = self.fig.add_subplot(gs[0])
        self.ax2     = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.ax_info = self.fig.add_subplot(gs[2])
        self.ax_info.axis("off")

        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#161b22")
            ax.grid(True)

        self.ax1.set_ylabel("Angulo  (deg)", fontsize=10)
        self.ax2.set_ylabel("Controle  (%)", fontsize=10)
        self.ax2.set_xlabel("Tempo  (s)", fontsize=10)
        self.ax2.set_ylim(-85, 85)
        self.ax1.set_xticklabels([])

        (self.ln_ang,) = self.ax1.plot([], [], color=CORES["angulo"], lw=1.8,
                                       label="angulo (deg)")
        (self.ln_ref,) = self.ax1.plot([], [], color=CORES["ref"], lw=1.2,
                                       ls="--", label="referencia (deg)", alpha=0.9)
        (self.ln_err,) = self.ax1.plot([], [], color=CORES["erro"], lw=0.9,
                                       ls=":", alpha=0.7, label="erro")
        (self.ln_u,)   = self.ax2.plot([], [], color=CORES["controle"], lw=1.6,
                                       label="controle (%)")

        self.ax1.axhline(0, color=CORES["zero"], lw=0.6, ls=":")
        self.ax2.axhline(0, color=CORES["zero"], lw=0.6, ls=":")
        self.ax1.legend(loc="upper right", framealpha=0, fontsize=9)
        self.ax2.legend(loc="upper right", framealpha=0, fontsize=9)

        # Marcadores de transição de referência
        if transicoes:
            for t_tr in transicoes:
                self.ax1.axvline(t_tr, color=CORES["ref"], lw=0.6, ls=":", alpha=0.4)
                self.ax2.axvline(t_tr, color=CORES["ref"], lw=0.6, ls=":", alpha=0.3)

        # Grade fina no eixo X
        for ax in (self.ax1, self.ax2):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.grid(True, which="minor", color="#1c2128", linewidth=0.5, alpha=0.7)

        self._txt = self.ax_info.text(
            0.5, 0.5, "", ha="center", va="center", fontsize=8, color="#8b949e",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#161b22",
                      edgecolor="#30363d", linewidth=0.8),
        )

        self._estop_txt = self.fig.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="#ff4444", alpha=0.0,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a0000",
                      edgecolor="#ff4444", linewidth=2),
        )

        self.fig.canvas.draw()
        plt.pause(0.01)

    # ── Atualização ──────────────────────────────────────────────────────────

    def atualizar(
        self,
        t:   list,
        ang: list,
        u:   list,
        ref: list,
    ):
        """Redesenha o plot se o intervalo mínimo (LIVE_UPDATE) foi respeitado."""
        agora = time.time()
        if agora - self._ultimo_update < LIVE_UPDATE:
            return
        self._ultimo_update = agora

        if len(t) < 2:
            return

        ta   = np.asarray(t,   dtype=float)
        anga = np.asarray(ang, dtype=float)
        ua   = np.asarray(u,   dtype=float)
        refa = np.asarray(ref, dtype=float)
        erro = anga - refa

        t_max = ta[-1]
        t_min = max(0.0, t_max - self.janela_s)
        mask  = ta >= t_min

        self.ln_ang.set_data(ta[mask], anga[mask])
        self.ln_ref.set_data(ta[mask], refa[mask])
        self.ln_err.set_data(ta[mask], erro[mask])
        self.ln_u.set_data(ta[mask], ua[mask])

        vals = np.concatenate([anga[mask], refa[mask]])
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        mg = max(8.0, (vmax - vmin) * 0.2)
        self.ax1.set_ylim(vmin - mg, vmax + mg)
        self.ax1.set_xlim(t_min, t_max + 0.5)

        rmse = float(np.sqrt(np.mean(erro ** 2)))
        n    = len(ta)
        freq = float(n / t_max) if t_max > 0 else 0.0

        self._txt.set_text(
            f"amostras: {n}   t: {t_max:.1f} s   freq: {freq:.1f} Hz   "
            f"angulo: {anga[-1]:.1f} deg   ref: {refa[-1]:.1f} deg   "
            f"erro: {erro[-1]:+.1f} deg   RMSE: {rmse:.2f} deg   "
            f"u: {ua[-1]:.1f}%"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── Emergência ───────────────────────────────────────────────────────────

    def mostrar_emergencia(self):
        self._estop_txt.set_text("PARADA DE EMERGENCIA")
        self._estop_txt.set_alpha(1.0)
        self._estop_visivel = True
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── Encerramento ─────────────────────────────────────────────────────────

    def fechar(self):
        plt.ioff()
        plt.close(self.fig)
