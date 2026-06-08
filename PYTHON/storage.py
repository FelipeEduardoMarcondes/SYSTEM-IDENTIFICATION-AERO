"""
storage.py — Salvar/carregar CSVs e gerar plots estáticos.

Funções públicas
----------------
salvar_csv(linhas, prefixo)       -> caminho
carregar_csv(caminho)             -> pd.DataFrame
plotar(caminho)                   -> None
estatisticas(df)                  -> dict
selecionar_csv()                  -> caminho | None
"""

import os
import glob
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from config import DADOS_DIR, CORES, MPL_RC

plt.rcParams.update(MPL_RC)


# ── Persistência ──────────────────────────────────────────────────────────────

def salvar_csv(linhas_dados: list, prefixo: str = "ensaio") -> str:
    """
    Salva linhas CSV brutas (vindas do Arduino) em arquivo com cabeçalho.

    Retorna o caminho do arquivo criado.
    """
    os.makedirs(DADOS_DIR, exist_ok=True)
    ts      = datetime.datetime.now().strftime("%m%d_%H-%M")
    caminho = os.path.join(DADOS_DIR, f"{prefixo}_{ts}.csv")
    with open(caminho, "w") as f:
        f.write("tempo_ms,angulo_deg,u_pct,referencia\n")
        for linha in linhas_dados:
            f.write(linha + "\n")
    print(f"  CSV salvo em: {caminho}")
    return caminho


def carregar_csv(caminho: str) -> pd.DataFrame:
    """
    Carrega um CSV de experimento e normaliza as colunas para:
        tempo_s, angulo_deg, controle_u, referencia
    Suporta os formatos gerados pelo firmware (tempo_ms) e pelo MATLAB (Tempo_s).
    """
    df = pd.read_csv(caminho)
    df.columns = [c.strip() for c in df.columns]

    if "tempo_ms" in df.columns:
        df["tempo_s"]    = df["tempo_ms"] / 1_000.0
        df["controle_u"] = df["u_pct"].astype(float)
        if "referencia" in df.columns:
            df["referencia"] = df["referencia"].astype(float)
        else:
            df["referencia"] = np.nan
        return df

    if "Tempo_s" in df.columns:
        df["tempo_s"]    = df["Tempo_s"].astype(float)
        df["angulo_deg"] = df["Angulo_Medido"].astype(float)
        df["referencia"] = df["Ref"].astype(float)
        df["controle_u"] = df["Controle_u"].astype(float)
        return df

    # Fallback genérico
    if "tempo_us" in df.columns:
        df["tempo_s"] = df["tempo_us"] / 1_000_000.0
    else:
        raise ValueError(f"Formato de CSV não reconhecido: {caminho}")
    df["referencia"] = np.nan
    df["controle_u"] = df.get("motor_percent", np.zeros(len(df)))
    return df


def carregar_sequencia_csv(caminho: str) -> tuple:
    """
    Lê um CSV de dois campos (tempo, referência) exportado pelo MATLAB e
    extrai os instantes onde a referência muda.

    Retorna (duracao_s, [(t_s, ref_deg), ...]).
    """
    df = pd.read_csv(caminho, header=None)
    df.columns = ["t", "u"]
    df["u_shift"] = df["u"].shift(1)
    df_mud = df[df["u"] != df["u_shift"]]

    degraus    = [(float(r["t"]), float(r["u"])) for _, r in df_mud.iterrows()]
    duracao_s  = float(df["t"].iloc[-1])
    return duracao_s, degraus


def selecionar_csv() -> str | None:
    """Menu interativo para escolher um CSV existente."""
    arquivos = sorted(glob.glob(f"{DADOS_DIR}/*.csv") + glob.glob("*.csv"))
    if not arquivos:
        return None
    if len(arquivos) == 1:
        print(f"  Usando: {arquivos[0]}")
        return arquivos[0]
    print("\n  Arquivos CSV disponíveis:")
    for i, f in enumerate(arquivos):
        size_kb = os.path.getsize(f) // 1024
        print(f"  [{i + 1:2d}] {f}  ({size_kb} KB)")
    while True:
        try:
            idx = int(input(f"\n  Escolha [1-{len(arquivos)}]: ").strip()) - 1
            if 0 <= idx < len(arquivos):
                return arquivos[idx]
        except ValueError:
            pass
        print("  Opção inválida.")


# ── Estatísticas ──────────────────────────────────────────────────────────────

def estatisticas(df: pd.DataFrame) -> dict:
    t   = df["tempo_s"].values
    ang = df["angulo_deg"].values
    ref = df["referencia"].values
    u   = df["controle_u"].values

    duracao = float(t[-1]) if len(t) > 0 else 0.0
    freq_hz = float(1.0 / np.diff(t).mean()) if len(t) > 1 else 0.0
    tem_ref = not np.all(np.isnan(ref))

    if tem_ref:
        erro    = ang - ref
        rmse    = float(np.sqrt(np.nanmean(erro ** 2)))
        idx_ss  = max(1, int(len(erro) * 0.7))
        erro_ss = float(np.nanmean(erro[idx_ss:]))
    else:
        rmse    = float("nan")
        erro_ss = float("nan")

    return {
        "amostras":  len(df),
        "duracao_s": duracao,
        "freq_hz":   freq_hz,
        "ang_min":   float(np.nanmin(ang)),
        "ang_max":   float(np.nanmax(ang)),
        "ang_medio": float(np.nanmean(ang)),
        "u_min":     float(np.nanmin(u)),
        "u_max":     float(np.nanmax(u)),
        "rmse":      rmse,
        "erro_ss":   erro_ss,
    }


# ── Plot estático ─────────────────────────────────────────────────────────────

def plotar(caminho: str):
    df   = carregar_csv(caminho)
    nome = os.path.splitext(os.path.basename(caminho))[0]
    stat = estatisticas(df)

    t   = df["tempo_s"].values
    ang = df["angulo_deg"].values
    ref = df["referencia"].values
    u   = df["controle_u"].values

    tem_ref  = not np.all(np.isnan(ref))
    tem_erro = tem_ref

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.suptitle(
        f"AEROPENDULO  |  {nome}",
        fontsize=13, fontweight="bold",
        color="#e6edf3", y=0.97, fontfamily="monospace",
    )

    n_paineis     = 4 if tem_erro else 3
    height_ratios = [3, 2, 1.5, 0.5] if tem_erro else [3, 2, 0.5]

    gs = gridspec.GridSpec(
        n_paineis, 1,
        height_ratios=height_ratios,
        hspace=0.08, left=0.07, right=0.97, top=0.91, bottom=0.07,
    )

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(t, ang, alpha=0.12, color=CORES["fill_a"])

    if tem_ref:
        ax1.fill_between(t, ref, alpha=0.10, color=CORES["fill_r"])
        ax1.plot(t, ref, color=CORES["ref"], lw=1.4,
                 ls="--", label="referencia (deg)", alpha=0.9)

        mudancas = np.where(np.diff(ref) != 0)[0] + 1
        
        # Só desenha linhas verticais se for uma sequência de degraus discretos (poucas mudanças)
        if len(mudancas) < 50:
            for idx_m in mudancas:
                ax1.axvline(t[idx_m], color=CORES["ref"], lw=0.7, ls=":", alpha=0.5)

    ax1.plot(t, ang, color=CORES["angulo"], lw=1.8, label="angulo medido (deg)")
    ax1.axhline(0, color=CORES["zero"], lw=0.7, ls=":")

    # Escala inclui tanto o ângulo medido quanto a referência
    vals_painel = np.concatenate([ang, ref[~np.isnan(ref)]]) if tem_ref else ang
    y_min = float(np.nanmin(vals_painel))
    y_max = float(np.nanmax(vals_painel))
    margem = max(10.0, (y_max - y_min) * 0.15 + 5.0)
    ax1.set_ylabel("Angulo  (deg)", fontsize=10)
    ax1.set_ylim(y_min - margem, y_max + margem)
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.legend(loc="upper right", framealpha=0, fontsize=9)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(t, u, alpha=0.2, color=CORES["fill_c"])
    ax2.plot(t, u, color=CORES["controle"], lw=1.6, label="controle u (%)")
    ax2.axhline(0, color=CORES["zero"], lw=0.7, ls=":")
    ax2.set_ylabel("Controle  (%)", fontsize=10)
    ax2.set_ylim(-85, 85)
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.legend(loc="upper right", framealpha=0, fontsize=9)

    if tem_erro:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        erro = ang - ref
        ax3.fill_between(t, erro, alpha=0.20, color=CORES["fill_e"])
        ax3.plot(t, erro, color=CORES["erro"], lw=1.4, label="erro (deg)")
        ax3.axhline(0, color=CORES["zero"], lw=0.7, ls=":")
        if not np.isnan(stat["rmse"]):
            ax3.axhline( stat["rmse"], color=CORES["erro"], lw=0.8,
                         ls=":", alpha=0.5, label=f"RMSE {stat['rmse']:.2f} deg")
            ax3.axhline(-stat["rmse"], color=CORES["erro"], lw=0.8,
                         ls=":", alpha=0.5)
        ax3.set_ylabel("Erro  (deg)", fontsize=10)
        ax3.grid(True)
        ax3.set_xticklabels([])
        ax3.legend(loc="upper right", framealpha=0, fontsize=9)
        ultimo_ax = ax3
        stats_idx = 3
    else:
        ultimo_ax = ax2
        stats_idx = 2

    ax_s = fig.add_subplot(gs[stats_idx])
    ax_s.axis("off")
    rmse_str  = f"RMSE: {stat['rmse']:.2f} deg   " if not np.isnan(stat["rmse"]) else ""
    errss_str = f"erro_ss: {stat['erro_ss']:+.2f} deg   |   " if not np.isnan(stat["erro_ss"]) else ""
    info = (
        f"amostras: {stat['amostras']}   duracao: {stat['duracao_s']:.1f} s   "
        f"freq: {stat['freq_hz']:.1f} Hz   |   "
        + rmse_str + errss_str
        + f"angulo [{stat['ang_min']:+.1f}, {stat['ang_max']:+.1f}] deg   "
        f"controle [{stat['u_min']:.1f}, {stat['u_max']:.1f}]%"
    )
    ax_s.text(
        0.5, 0.5, info, ha="center", va="center", fontsize=8, color="#8b949e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                  edgecolor="#30363d", linewidth=0.8),
    )

    ultimo_ax.set_xlabel("Tempo  (s)", fontsize=10)

    dur      = stat["duracao_s"]
    major_s  = (1 if dur <= 20 else 2 if dur <= 40 else 5 if dur <= 75 else
                10 if dur <= 150 else 20 if dur <= 300 else 30)
    minor_s  = major_s // 5 if major_s >= 5 else 1
    eixos_x  = [ax1, ax2] + ([ax3] if tem_erro else [])
    for ax in eixos_x:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_s))
        if minor_s > 0:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_s))
        ax.grid(True, which="minor", color="#1c2128", linewidth=0.5, alpha=0.7)

    saida = caminho.replace(".csv", ".png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Grafico salvo em: {saida}")
    plt.show()