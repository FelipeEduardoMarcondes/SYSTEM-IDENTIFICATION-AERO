"""
config.py — Constantes globais do projeto aeropêndulo.

Edite aqui antes de rodar qualquer módulo.
"""

# ── Serial ────────────────────────────────────────────────────────────────────
BAUD           = 500_000
SERIAL_TIMEOUT = 1.0          # s — timeout de readline em coleta normal
WAVE_CHUNK     = 20          # amostras por bloco DATA= (não exceder 200)

# ── Amostragem ────────────────────────────────────────────────────────────────
FS             = 100.0        # Hz — deve bater com o firmware (INTERVALO_US=10000)
TS             = 1.0 / FS    # s

# ── Live plot ─────────────────────────────────────────────────────────────────
LIVE_JANELA_S  = 30.0         # s — janela de visualização
LIVE_UPDATE    = 0.25         # s — intervalo mínimo entre redesenhos

# ── Caminhos ──────────────────────────────────────────────────────────────────
DADOS_DIR      = "dados"

# ── Limites físicos ───────────────────────────────────────────────────────────
REF_MIN        = 0.0          # graus
REF_MAX        = 135.0        # graus
U_MAX          = 80.0         # % (espelho do firmware)

# ── Cores do plot (tema escuro) ───────────────────────────────────────────────
CORES = {
    "angulo":   "#58a6ff",
    "ref":      "#f0883e",
    "controle": "#3fb950",
    "erro":     "#da3633",
    "zero":     "#6e7681",
    "fill_a":   "#1f3a5f",
    "fill_r":   "#3a2010",
    "fill_c":   "#1a3a22",
    "fill_e":   "#3a1010",
}

# ── Matplotlib ────────────────────────────────────────────────────────────────
MPL_RC = {
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
