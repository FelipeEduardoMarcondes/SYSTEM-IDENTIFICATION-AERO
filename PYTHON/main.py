"""
main.py — Ponto de entrada do sistema de aquisição do aeropêndulo.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import BAUD, FS, DADOS_DIR

try:
    from serial_comm import SerialManager, selecionar_porta, SERIAL_OK
except ImportError:
    SERIAL_OK = False

from signals     import multisine, carregar_sinal_csv, info
from storage     import salvar_csv, carregar_csv, plotar, selecionar_csv
from live_plot   import LivePlot
from acquisition import Aquisicao
from presets     import SEQUENCIAS_PRESET

def _pedir_float(prompt: str, padrao: float) -> float:
    while True:
        try:
            raw = input(f"  {prompt} [{padrao}]: ").strip() or str(padrao)
            return float(raw)
        except ValueError:
            print("  Valor inválido.")

def _pedir_int(prompt: str, padrao: int) -> int:
    while True:
        try:
            raw = input(f"  {prompt} [{padrao}]: ").strip() or str(padrao)
            return int(raw)
        except ValueError:
            print("  Valor inválido.")

def _definir_sequencia_degraus() -> tuple:
    from storage import carregar_sequencia_csv
    import glob

    n_pre     = len(SEQUENCIAS_PRESET)
    opt_manual = n_pre + 1
    opt_csv    = n_pre + 2

    print("\n  Sequencias predefinidas:")
    for i, (nome, dur, _) in enumerate(SEQUENCIAS_PRESET):
        print(f"  [{i + 1}] {nome}  ({dur} s)")
    print(f"  [{opt_manual}] Definir manualmente")
    print(f"  [{opt_csv}]   Carregar de CSV (MATLAB)\n")

    escolha = None
    while escolha is None:
        try:
            v = int(input(f"  Escolha [1-{opt_csv}]: ").strip())
            if 1 <= v <= opt_csv:
                escolha = v
        except ValueError:
            pass

    if escolha <= n_pre:
        nome, duracao_s, degraus = SEQUENCIAS_PRESET[escolha - 1]
        degraus = list(degraus)
        print(f"\n  Preset: {nome}")
    elif escolha == opt_manual:
        duracao_s = _pedir_float("Duração total (s)", 30)
        print("\n  Degrau: <t_inicio_s> <angulo_deg>  |  linha vazia = encerrar\n")
        degraus = []
        idx = 1
        while True:
            try:
                raw = input(f"  Degrau {idx}: ").strip()
            except EOFError:
                break
            if not raw:
                if not degraus:
                    degraus = [(0.0, 40.0)]
                break
            partes = raw.replace(",", " ").replace(":", " ").split()
            if len(partes) == 2:
                try:
                    degraus.append((float(partes[0]), float(partes[1])))
                    idx += 1
                    continue
                except ValueError:
                    pass
            print("  Formato: <tempo_s> <angulo_deg>  ex: 0 40")
    else:
        csvs = sorted(glob.glob("*.csv"))
        if not csvs:
            print("  Nenhum CSV encontrado. Usando padrão.")
            return 10.0, [(0.0, 45.0)]
        print("\n  CSVs disponíveis:")
        for i, f in enumerate(csvs):
            print(f"  [{i + 1}] {f}")
        idx_csv = -1
        while not (0 <= idx_csv < len(csvs)):
            try:
                idx_csv = int(input(f"  Escolha [1-{len(csvs)}]: ").strip()) - 1
            except ValueError:
                pass
        duracao_s, degraus = carregar_sequencia_csv(csvs[idx_csv])
        print(f"  CSV carregado: {csvs[idx_csv]}")

    degraus.sort(key=lambda x: x[0])
    if degraus[0][0] != 0.0:
        degraus[0] = (0.0, degraus[0][1])
    degraus = [(t, r) for t, r in degraus if t < duracao_s]
    if not degraus:
        degraus = [(0.0, 40.0)]

    print("\n  Resumo:")
    for i, (t, r) in enumerate(degraus):
        t_fim = degraus[i + 1][0] if i + 1 < len(degraus) else duracao_s
        print(f"    t={t:.1f}s  ->  t={t_fim:.1f}s  :  {r:.1f} deg  ({t_fim - t:.1f}s)")
    print(f"  Duração total: {duracao_s:.1f} s\n")
    return duracao_s, degraus

def _configurar_chirp_params() -> tuple:
    print("\n  Configuracao do CHIRP (Embarcado)")
    print("  [1] Ler parametros_chirp.csv (gerado pelo MATLAB)")
    print("  [2] Inserir manualmente\n")
    
    opcao = None
    while opcao not in ("1", "2"):
        opcao = input("  Opcao [1/2]: ").strip()

    if opcao == "1":
        try:
            df = pd.read_csv('parametros_chirp.csv', header=None)
            amp  = float(df.iloc[0, 0])
            fmax = float(df.iloc[0, 1])
            t0   = float(df.iloc[0, 2])
            dc   = float(df.iloc[0, 3])
            print(f"  [Lido do CSV] Amp={amp}, Fmax={fmax}Hz, T0={t0}s, DC={dc}")
            return t0, amp, fmax, dc
        except Exception as e:
            print(f"  [ERRO] Falha ao ler parametros_chirp.csv: {e}")
            print("  Recorrendo a entrada manual.")

    t0   = _pedir_float("Duracao do chirp (s)", 20.0)
    amp  = _pedir_float("Amplitude pico (graus)", 30.0)
    fmax = _pedir_float("Frequencia maxima (Hz)", 0.5)
    dc   = _pedir_float("Ponto de operacao (graus)", 45.0)
    return t0, amp, fmax, dc

def _configurar_multisine() -> tuple:
    import glob
    print("\n  Configuracao do MULTI-SENO")
    print("  [1] Gerar novo sinal")
    print("  [2] Carregar sinal de CSV existente\n")

    opcao = None
    while opcao not in ("1", "2"):
        opcao = input("  Opcao [1/2]: ").strip()

    if opcao == "2":
        csvs = sorted(glob.glob("*.csv") + glob.glob(f"{DADOS_DIR}/*.csv"))
        if not csvs:
            print("  Nenhum CSV encontrado. Gerando novo sinal.")
        else:
            print("\n  CSVs disponíveis:")
            for i, f in enumerate(csvs):
                print(f"  [{i + 1}] {f}")
            idx_csv = -1
            while not (0 <= idx_csv < len(csvs)):
                try:
                    idx_csv = int(input(f"  Escolha [1-{len(csvs)}]: ").strip()) - 1
                except ValueError:
                    pass
            t, u = carregar_sinal_csv(csvs[idx_csv])
            info(t, u, f"multisine (CSV: {csvs[idx_csv]})")
            return float(t[-1]), t, u

    duracao  = _pedir_float("Duracao ativa (s)", 40.0)
    amp      = _pedir_float("Amplitude RMS (graus)", 30.0)
    dc       = _pedir_float("Ponto de operacao (graus)", 45.0)
    f_max    = _pedir_float("Frequencia maxima (Hz)", 0.5)
    pad      = _pedir_float("Pad de zeros antes/depois (s)", 10.0)
    seed     = _pedir_int("Semente aleatoria", 0)

    t, u = multisine(duracao=duracao, amp=amp, dc=dc, f_max=f_max, pad_s=pad, seed=seed)
    info(t, u, "multisine")
    return duracao + 2 * pad, t, u

def _conectar_e_aguardar(porta: str) -> "SerialManager | None":
    mgr = SerialManager(porta)
    if not mgr.conectar():
        return None
    if not mgr.aguardar_token("PRONTO", timeout_s=30.0):
        mgr.fechar()
        return None
    return mgr

def _pedir_recal(mgr: "SerialManager"):
    if input("  Recalibrar giroscopio? (s/N): ").strip().lower() == "s":
        mgr.recalibrar()

def rodar_degraus(porta: str) -> str | None:
    mgr = _conectar_e_aguardar(porta)
    if not mgr: return None
    duracao_s, degraus = _definir_sequencia_degraus()
    _pedir_recal(mgr)

    if not mgr.enviar_seq(degraus):
        mgr.fechar()
        return None

    transicoes = [t for t, _ in degraus[1:]]
    nome = "SEQ  " + "  ".join(f"{t:.0f}s->{r:.0f}deg" for t, r in degraus)
    live = LivePlot(nome=nome, transicoes=transicoes)

    if not mgr.iniciar_experimento():
        live.fechar()
        mgr.fechar()
        return None

    aq = Aquisicao(mgr, live)
    linhas = aq.rodar(duracao_s=duracao_s, modo="seq", degraus=degraus)
    live.fechar()
    mgr.fechar()

    if len(linhas) < 2:
        print("  Dados insuficientes.")
        return None
    return salvar_csv(linhas, prefixo="degraus")

def rodar_chirp(porta: str) -> str | None:
    mgr = _conectar_e_aguardar(porta)
    if not mgr: return None
    
    t0, amp, fmax, dc = _configurar_chirp_params()
    duracao_total = t0 + 5.0 # Margem de segurança pós-sinal
    _pedir_recal(mgr)

    print("\n  Enviando parametros do CHIRP para o Arduino...")
    if not mgr.enviar_chirp(amp, fmax, t0, dc):
        print("  [ERRO] Falha ao enviar parametros. Abortando.")
        mgr.fechar()
        return None

    live = LivePlot(nome="CHIRP EMBARCADO", janela_s=min(30.0, duracao_total))
    if not mgr.iniciar_experimento():
        live.fechar()
        mgr.fechar()
        return None

    aq = Aquisicao(mgr, live)
    linhas = aq.rodar(duracao_s=duracao_total, modo="wave") # Python apenas coleta
    live.fechar()
    mgr.fechar()

    if len(linhas) < 2:
        return None
    return salvar_csv(linhas, prefixo="chirp")

def rodar_multisine(porta: str) -> str | None:
    mgr = _conectar_e_aguardar(porta)
    if not mgr: return None

    duracao_total, t_sig, u_sig = _configurar_multisine()
    _pedir_recal(mgr)

    print(f"\n  Pré-carregando {len(u_sig)} amostras no Arduino...")
    if not mgr.enviar_wave(u_sig):
        print("  [ERRO] Falha ao enviar WAVE. Abortando.")
        mgr.fechar()
        return None

    live = LivePlot(nome="MULTISINE", janela_s=min(30.0, duracao_total / 2))
    if not mgr.iniciar_experimento():
        live.fechar()
        mgr.fechar()
        return None

    aq = Aquisicao(mgr, live)
    linhas = aq.rodar(duracao_s=duracao_total + 2.0, modo="wave")
    live.fechar()
    mgr.fechar()

    if len(linhas) < 2:
        return None
    return salvar_csv(linhas, prefixo="multisine")

def rodar_repouso(porta: str) -> str | None:
    mgr = _conectar_e_aguardar(porta)
    if not mgr: return None
    duracao_s = _pedir_float("Duracao da coleta em repouso (s)", 20.0)
    
    mgr.ser.reset_input_buffer()
    mgr.enviar("FREE")
    if not mgr.aguardar_token("EXP_START", timeout_s=10.0):
        mgr.fechar()
        return None

    cab = mgr.readline()
    if cab: print(f"  [Arduino] {cab}")

    live = LivePlot(nome="REPOUSO (motor desligado)")
    aq = Aquisicao(mgr, live)
    linhas = aq.rodar(duracao_s=duracao_s, modo="wave")
    live.fechar()
    mgr.fechar()

    if len(linhas) < 2: return None
    return salvar_csv(linhas, prefixo="repouso")

def recalibrar(porta: str):
    mgr = _conectar_e_aguardar(porta)
    if not mgr: return
    print("  Recalibrando...")
    mgr.recalibrar()
    mgr.fechar()

def _menu() -> str:
    print()
    if SERIAL_OK:
        print("  [1] Experimento com degraus (SEQ)")
        print("  [2] Experimento com chirp (Embarcado)")
        print("  [3] Experimento com multi-seno (WAVE buffer)")
        print("  [4] Coleta em repouso (motor desligado)")
        print("  [5] Recalibrar giroscopio")
        print("  [6] Plotar CSV existente")
    else:
        print("  [6] Plotar CSV existente")
        print("\n  OBS: pyserial nao instalado.")
    print("  [0] Sair")

    validas = {"0", "6"} | ({"1", "2", "3", "4", "5"} if SERIAL_OK else set())
    while True:
        op = input("\n  Opcao: ").strip()
        if op in validas: return op
        print("  Opcao invalida.")

if __name__ == "__main__":
    print("=" * 60)
    print("  Aeropendulo  |  Aquisicao de Dados  |  v2")
    print("=" * 60)

    if len(sys.argv) > 1:
        plotar(sys.argv[1])
        sys.exit(0)

    op = _menu()

    if op == "0": sys.exit(0)
    elif op in ("1", "2", "3", "4", "5"):
        porta = selecionar_porta()
        if not porta: sys.exit(1)

        if op == "1":
            caminho = rodar_degraus(porta)
        elif op == "2":
            caminho = rodar_chirp(porta)
        elif op == "3":
            caminho = rodar_multisine(porta)
        elif op == "4":
            caminho = rodar_repouso(porta)
        elif op == "5":
            recalibrar(porta)
            caminho = None

        if caminho:
            print(f"\n  Plotando resultado: {caminho}")
            plotar(caminho)

    elif op == "6":
        caminho = selecionar_csv()
        if caminho: plotar(caminho)
        else: print("  Nenhum CSV encontrado.")