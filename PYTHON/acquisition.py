"""
acquisition.py — Loop de aquisição em malha fechada para o aeropêndulo.

Responsável por:
  - Receber linhas CSV do Arduino em tempo real
  - Atualizar o LivePlot
  - Detectar parada (Enter / ESC / timeout / duração)
  - Para o modo SEQ: streaming da referência via R= em thread paralela
  - Para o modo WAVE: nenhuma ação durante a coleta (Arduino é autônomo)

Uso
---
    from acquisition import Aquisicao
    aq = Aquisicao(serial_manager, live_plot)
    linhas = aq.rodar(duracao_s=40, modo="wave")
"""

import time
import threading
import numpy as np

try:
    import keyboard
    KEYBOARD_OK = True
except ImportError:
    KEYBOARD_OK = False


class Aquisicao:
    """
    Encapsula o loop de coleta de dados do aeropêndulo.

    Parâmetros
    ----------
    mgr  : SerialManager já conectado e pronto (token PRONTO recebido)
    live : LivePlot instanciado
    """

    MAX_TIMEOUTS = 3

    def __init__(self, mgr, live):
        self._mgr  = mgr
        self._live = live

    # ── Ponto de entrada público ──────────────────────────────────────────────

    def rodar(
        self,
        duracao_s: float,
        modo: str = "seq",
        degraus: list | None = None,
    ) -> list:
        """
        Executa o experimento e retorna a lista de strings CSV coletadas.

        Parâmetros
        ----------
        duracao_s : duração total esperada (s); 0 = sem limite
        modo      : "seq"  — streaming R= via thread (degraus discretos)
                    "wave" — Arduino autônomo; Python só coleta
        degraus   : lista (t_s, ref_deg) usada apenas no modo "seq"

        Retorna
        -------
        linhas_dados : list[str] — cada elemento é uma linha CSV
        """
        self._parar  = threading.Event()
        self._lock   = threading.Lock()

        t_list       = []
        ang_list     = []
        u_list       = []
        ref_list     = []
        linhas_dados = []
        timeouts     = 0

        prazo = time.time() + duracao_s if duracao_s > 0 else float("inf")
        self._mgr.ser.timeout = 3.0

        # Thread: parar com Enter
        threading.Thread(target=self._watch_enter, daemon=True).start()

        # Thread: parar com ESC (emergência)
        if KEYBOARD_OK:
            print("  [ESC] = PARADA DE EMERGENCIA imediata")
            threading.Thread(
                target=self._watch_esc,
                daemon=True,
            ).start()
        else:
            print("  OBS: 'keyboard' nao instalado — ESC indisponivel.")

        # Thread: streaming de referência (apenas modo SEQ)
        t_inicio_mono = time.monotonic()
        if modo == "seq" and degraus:
            threading.Thread(
                target=self._streaming_ref,
                args=(degraus, t_inicio_mono),
                daemon=True,
            ).start()

        print("  Coletando dados ao vivo...\n")

        while time.time() < prazo and not self._parar.is_set():
            raw = self._mgr.ser.readline()
            if not raw:
                timeouts += 1
                print(f"  Timeout {timeouts}/{self.MAX_TIMEOUTS}")
                if timeouts >= self.MAX_TIMEOUTS:
                    print("  Arduino sem resposta. Abortando.")
                    with self._lock:
                        self._mgr.enviar("STOP")
                    break
                continue
            timeouts = 0

            linha = raw.decode("utf-8", errors="replace").strip()
            if not linha:
                continue
            if linha.startswith("#"):
                print(f"  [Arduino] {linha}")
                if "PARADO" in linha or "FIM" in linha:
                    break
                continue

            partes = linha.split(",")
            if len(partes) < 3:
                continue

            try:
                t_ms = float(partes[0])
                ang  = float(partes[1])
                u    = float(partes[2])
                ref  = float(partes[3]) if len(partes) >= 4 else float("nan")
            except ValueError:
                continue

            t_s = t_ms / 1000.0
            t_list.append(t_s)
            ang_list.append(ang)
            u_list.append(u)
            ref_list.append(ref)
            linhas_dados.append(linha)

            print(
                f"\r  t={t_s:7.2f}s  ang={ang:6.1f} deg  ref={ref:5.1f} deg  "
                f"err={ang - ref:+6.1f} deg  u={u:5.1f}%  n={len(t_list)}",
                end="", flush=True,
            )
            self._live.atualizar(t_list, ang_list, u_list, ref_list)

        # Encerramento
        self._parar.set()
        self._mgr.parar()
        print(f"\n\n  Total de amostras: {len(linhas_dados)}")
        return linhas_dados

    # ── Threads auxiliares ────────────────────────────────────────────────────

    def _watch_enter(self):
        input("\n  [Enter = parar normalmente]\n")
        if not self._parar.is_set():
            self._parar.set()

    def _watch_esc(self):
        while not self._parar.is_set():
            if keyboard.is_pressed("esc"):
                print("\n\n  *** PARADA DE EMERGENCIA (ESC) ***")
                with self._lock:
                    try:
                        self._mgr.enviar("STOP")
                    except Exception:
                        pass
                self._live.mostrar_emergencia()
                self._parar.set()
                return
            time.sleep(0.05)

    def _streaming_ref(self, degraus: list, t_inicio_mono: float):
        """
        Envia R=<valor> no momento exato de cada transição, usando
        time.monotonic para evitar drift. Ativo apenas no modo SEQ.
        """
        idx = 0
        with self._lock:
            self._mgr.enviar(f"R={degraus[0][1]:.2f}")

        while not self._parar.is_set() and idx + 1 < len(degraus):
            t_prox  = degraus[idx + 1][0]
            t_agora = time.monotonic() - t_inicio_mono
            folga   = t_prox - t_agora

            if folga > 0.015:
                time.sleep(folga - 0.010)
                continue

            if t_agora >= t_prox:
                idx += 1
                ref = degraus[idx][1]
                with self._lock:
                    self._mgr.enviar(f"R={ref:.2f}")
                print(f"\n  [SEQ] t={t_agora:.3f}s  ref={ref:.2f} deg")
            else:
                time.sleep(0.001)
