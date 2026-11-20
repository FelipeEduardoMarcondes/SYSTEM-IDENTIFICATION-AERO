"""
acquisition.py — Loop de aquisição em malha fechada para o aeropêndulo.

Responsável por:
  - Receber linhas CSV do STM32 em tempo real
  - Atualizar o LivePlot
  - Detectar parada (Enter / ESC / timeout / duração)
  - Para o modo SEQ: streaming da referência via R= em thread paralela
  - Para o modo WAVE: nenhuma ação durante a coleta (STM32 é autônomo)

Uso
---
    from acquisition import Aquisicao
    aq = Aquisicao(serial_manager, live_plot)
    linhas = aq.rodar(duracao_s=40, modo="wave")
"""

import time
import queue
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
    ultimo_profile = []

    def __init__(self, mgr, live):
        self._mgr  = mgr
        self._live = live
        self.linhas_profile = []

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
                    "wave" — STM32 autônomo; Python só coleta
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
        self._mgr.ser.timeout = 0.5
        max_timeouts = self.MAX_TIMEOUTS * 6  # 18 * 0.5s = 9s sem resposta

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

        fila_rx: queue.Queue = queue.Queue()

        def _reader_serial():
            nonlocal timeouts
            while time.time() < prazo and not self._parar.is_set():
                try:
                    raw = self._mgr.ser.readline()
                except Exception:
                    break
                if not raw:
                    timeouts += 1
                    if timeouts % 6 == 0:
                        print(f"\n  Timeout {timeouts // 6}/{self.MAX_TIMEOUTS}")
                    if timeouts >= max_timeouts:
                        print("  STM32 sem resposta. Abortando.")
                        with self._lock:
                            try:
                                self._mgr.enviar("STOP")
                            except Exception:
                                pass
                        self._parar.set()
                        break
                    continue
                timeouts = 0
                fila_rx.put(raw)

        t_reader = threading.Thread(target=_reader_serial, daemon=True)
        t_reader.start()

        ultimo_print = 0.0
        while (time.time() < prazo and not self._parar.is_set()) or not fila_rx.empty():
            # Drena todos os pacotes disponíveis na fila sem bloquear na UI
            processei_algo = False
            ultimo_status = ""

            while not fila_rx.empty():
                try:
                    raw = fila_rx.get_nowait()
                except queue.Empty:
                    break

                linha = raw.decode("utf-8", errors="replace").strip()
                if not linha:
                    continue
                if linha.startswith("#"):
                    print(f"\n  [STM32] {linha}")
                    if linha.startswith("# PROFILE"):
                        self.linhas_profile.append(linha)
                    if "PARADO" in linha or "FIM" in linha:
                        self._parar.set()
                        break
                    continue

                partes = linha.split(",")
                if len(partes) not in (4, 5):
                    continue

                try:
                    t_ms  = float(partes[0])
                    ang   = float(partes[1])
                    u     = float(partes[2])
                    ref   = float(partes[3])
                    t_inf = float(partes[4]) if len(partes) == 5 else None
                except ValueError:
                    continue

                # Valida limites para descartar eventuais linhas truncadas
                if abs(u) > 100 or abs(ang) > 360 or abs(ref) > 360 or t_ms < 0 or t_ms > 7_200_000:
                    continue

                t_s = t_ms / 1000.0
                if t_list and (t_s < t_list[-1] or (t_s - t_list[-1]) > 15.0):
                    continue

                t_list.append(t_s)
                ang_list.append(ang)
                u_list.append(u)
                ref_list.append(ref)
                linhas_dados.append(linha)
                processei_algo = True

                inf_str = f"  inf={t_inf:.3f}ms" if t_inf is not None else ""
                ultimo_status = (
                    f"\r  t={t_s:7.2f}s  ang={ang:6.1f} deg  ref={ref:5.1f} deg  "
                    f"err={ang - ref:+6.1f} deg  u={u:5.1f}%{inf_str}  n={len(t_list)}"
                )

            if processei_algo:
                agora = time.time()
                if agora - ultimo_print >= 0.05:
                    print(ultimo_status, end="", flush=True)
                    ultimo_print = agora
                self._live.atualizar(t_list, ang_list, u_list, ref_list)
            else:
                if self._parar.is_set() or time.time() >= prazo:
                    break
                time.sleep(0.005)

        # Encerramento
        self._parar.set()
        t_reader.join(timeout=1.5)
        Aquisicao.ultimo_profile = list(self.linhas_profile)
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
