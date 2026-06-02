"""
signals.py — Geradores de sinais de excitação para identificação do aeropêndulo.

Todos os geradores retornam (t, u) como np.ndarray com passo TS.
Os sinais são expressos em graus de referência angular (posição do pêndulo).

Sinais disponíveis
------------------
degraus_aleatorios   — sequência de degraus com amplitudes sorteadas
chirp                — swept sine de f_ini a f_fim
multisine            — soma de senoides com fases aleatórias (persistentemente excitante)
zeropad              — utilidade interna

Uso
---
    from signals import chirp, multisine, degraus_aleatorios
    t, u = chirp(duracao=40, amp=30, f_ini=0.025, f_fim=0.5, dc=45)
"""

import numpy as np
from config import FS, TS

# ── Utilidade ─────────────────────────────────────────────────────────────────

def _zeropad(n: int) -> np.ndarray:
    return np.zeros(n)


def _enforce_limits(u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
    """Recorta o sinal aos limites físicos sem alterar a forma."""
    return np.clip(u, u_min, u_max)


# ── Gerador: sequência de degraus aleatórios ──────────────────────────────────

def degraus_aleatorios(
    duracao: float,
    amp: float = 30.0,
    dc: float = 45.0,
    t_degrau: float = 2.0,
    pad_s: float = 5.0,
    seed: int = 0,
    u_min: float = 0.0,
    u_max: float = 135.0,
) -> tuple:
    """
    Sequência de degraus de amplitudes aleatórias em [-amp, +amp] centrados em dc.

    Parâmetros
    ----------
    duracao  : duração do sinal ativo (s), excluindo os pads
    amp      : amplitude máxima dos degraus (graus)
    dc       : valor médio / ponto de operação (graus)
    t_degrau : duração de cada degrau (s)
    pad_s    : zeros antes e depois do sinal ativo (s)
    seed     : semente do gerador aleatório
    u_min    : limite inferior de referência (graus)
    u_max    : limite superior de referência (graus)

    Retorna
    -------
    (t, u) : arrays de tempo (s) e referência (graus)
    """
    rng = np.random.default_rng(seed)

    n_pad  = int(pad_s * FS)
    n_ativ = int(duracao * FS)
    n_deg  = max(1, int(t_degrau * FS))
    n_rand = int(np.ceil(n_ativ / n_deg))

    amplitudes = amp * (2 * rng.random(n_rand) - 1)
    sinal_ativ = np.repeat(amplitudes, n_deg)[:n_ativ] + dc

    u = np.concatenate([_zeropad(n_pad), sinal_ativ, _zeropad(n_pad)])
    u = _enforce_limits(u, u_min, u_max)
    t = np.arange(len(u)) * TS
    return t, u


# ── Gerador: chirp (swept sine) ───────────────────────────────────────────────

def chirp(
    duracao: float = 40.0,
    amp: float = 30.0,
    dc: float = 45.0,
    f_ini: float = 0.025,
    f_fim: float = 0.5,
    pad_s: float = 10.0,
    u_min: float = 0.0,
    u_max: float = 135.0,
) -> tuple:
    """
    Swept sine com varredura linear de frequência de f_ini a f_fim.

    A fase instantânea é integrada analiticamente para garantir continuidade:
        phi(t) = 2*pi * [f_ini*t + (f_fim - f_ini)/(2*T) * t^2]

    Parâmetros
    ----------
    duracao : duração do sinal ativo (s)
    amp     : amplitude pico (graus)
    dc      : ponto de operação (graus)
    f_ini   : frequência inicial (Hz)
    f_fim   : frequência final (Hz)
    pad_s   : zeros antes e depois do sinal ativo (s)
    u_min   : limite inferior (graus)
    u_max   : limite superior (graus)

    Retorna
    -------
    (t, u) : arrays de tempo (s) e referência (graus)
    """
    n_pad  = int(pad_s * FS)
    n_ativ = int(duracao * FS)

    t_loc = np.arange(n_ativ) * TS
    taxa  = (f_fim - f_ini) / duracao
    fase  = 2.0 * np.pi * (f_ini * t_loc + 0.5 * taxa * t_loc ** 2)
    sinal_ativ = amp * np.sin(fase) + dc

    u = np.concatenate([_zeropad(n_pad) + dc,
                        sinal_ativ,
                        _zeropad(n_pad) + dc])
    u = _enforce_limits(u, u_min, u_max)
    t = np.arange(len(u)) * TS
    return t, u


# ── Gerador: multi-seno ───────────────────────────────────────────────────────

def multisine(
    duracao: float = 40.0,
    amp: float = 30.0,
    dc: float = 45.0,
    f_max: float = 0.5,
    n_harm: int | None = None,
    pad_s: float = 10.0,
    seed: int = 0,
    u_min: float = 0.0,
    u_max: float = 135.0,
) -> tuple:
    """
    Multi-seno com fases aleatórias — sinal persistentemente excitante de ordem 2*n_harm.

    Construído como soma de n_harm senoides igualmente espaçadas em frequência
    até f_max, com amplitudes iguais (amp / sqrt(n_harm) para normalizar RMS)
    e fases sorteadas uniformemente em [0, 2*pi).

    Parâmetros
    ----------
    duracao : duração do sinal ativo (s)
    amp     : amplitude RMS aproximada (graus)
    dc      : ponto de operação (graus)
    f_max   : frequência máxima das harmônicas (Hz)
    n_harm  : número de harmônicas; None => floor(f_max / f_res)
    pad_s   : zeros antes e depois do sinal ativo (s)
    seed    : semente do gerador aleatório
    u_min   : limite inferior (graus)
    u_max   : limite superior (graus)

    Retorna
    -------
    (t, u) : arrays de tempo (s) e referência (graus)
    """
    rng   = np.random.default_rng(seed)
    f_res = 1.0 / duracao                     # resolução espectral

    if n_harm is None:
        n_harm = max(1, int(f_max / f_res))

    freqs = np.linspace(f_res, f_max, n_harm) # Hz
    fases = rng.uniform(0.0, 2.0 * np.pi, n_harm)

    n_pad  = int(pad_s * FS)
    n_ativ = int(duracao * FS)
    t_loc  = np.arange(n_ativ) * TS

    a_k = amp / np.sqrt(n_harm)               # amplitude por harmônica
    sinal_ativ = sum(
        a_k * np.sin(2.0 * np.pi * f * t_loc + phi)
        for f, phi in zip(freqs, fases)
    ) + dc

    u = np.concatenate([_zeropad(n_pad) + dc,
                        sinal_ativ,
                        _zeropad(n_pad) + dc])
    u = _enforce_limits(u, u_min, u_max)
    t = np.arange(len(u)) * TS
    return t, u


# ── Exportação CSV (compatível com o firmware) ─────────────────────────────────

def exportar_csv(t: np.ndarray, u: np.ndarray, caminho: str) -> None:
    """
    Salva o sinal em CSV com duas colunas: tempo_s, referencia_deg.
    Compatível com a função carregar_sequencia_csv() do Python e com
    o protocolo WAVE do firmware.
    """
    dados = np.column_stack([t, u])
    np.savetxt(caminho, dados, delimiter=",",
               header="tempo_s,referencia_deg", comments="")
    print(f"  Sinal salvo em: {caminho}")


# ── Conversão para lista de degraus (protocolo SEQ=) ─────────────────────────

def sinal_para_degraus(t: np.ndarray, u: np.ndarray) -> list:
    """
    Extrai os instantes de mudança do sinal e retorna lista de (t_s, ref_deg).
    Útil para enviar sinais de degrau via protocolo SEQ= do firmware.
    Não deve ser usado para chirp/multisine — esses exigem o protocolo WAVE.
    """
    mudancas = [0]
    for i in range(1, len(u)):
        if abs(u[i] - u[i - 1]) > 0.01:
            mudancas.append(i)
    return [(float(t[i]), float(u[i])) for i in mudancas]


# ── Carregar sinal de CSV ─────────────────────────────────────────────────────

def carregar_sinal_csv(caminho: str) -> tuple:
    """
    Carrega um sinal de referencia a partir de um CSV gerado por exportar_csv()
    ou pelo MATLAB (duas colunas: tempo_s, referencia_deg).

    Aceita arquivos com ou sem linha de cabecalho.
    Se o passo de tempo diferir de TS (1/FS), reamostrea por interpolacao linear
    para garantir compatibilidade com o protocolo WAVE (100 Hz fixo).

    Parametros
    ----------
    caminho : caminho para o arquivo CSV

    Retorna
    -------
    (t, u) : arrays numpy com passo TS = 1/FS
    """
    # Tenta ler com cabecalho; se a primeira linha for numerica, sem cabecalho
    try:
        raw = np.loadtxt(caminho, delimiter=",", skiprows=0)
        if raw.shape[0] < 2:
            raise ValueError("Menos de 2 linhas numericas.")
    except ValueError:
        raw = np.loadtxt(caminho, delimiter=",", skiprows=1)

    t_orig = raw[:, 0].astype(float)
    u_orig = raw[:, 1].astype(float)

    duracao = float(t_orig[-1])
    t_novo  = np.arange(0.0, duracao + TS * 0.5, TS)
    u_novo  = np.interp(t_novo, t_orig, u_orig)

    passo_orig = float(np.median(np.diff(t_orig)))
    if abs(passo_orig - TS) > 1e-4:
        print(
            f"  Aviso: passo original {passo_orig*1000:.2f} ms reamostrado "
            f"para {TS*1000:.2f} ms ({len(u_novo)} amostras)."
        )

    return t_novo, u_novo


# ── Informações do sinal ──────────────────────────────────────────────────────

def info(t: np.ndarray, u: np.ndarray, nome: str = "sinal") -> None:
    """Imprime estatísticas básicas do sinal gerado."""
    print(f"\n  [{nome}]")
    print(f"    amostras  : {len(t)}")
    print(f"    duracao   : {t[-1]:.2f} s")
    print(f"    min / max : {u.min():.2f} / {u.max():.2f} graus")
    print(f"    media     : {u.mean():.2f} graus")
    print(f"    RMS       : {np.sqrt(np.mean(u**2)):.2f} graus")