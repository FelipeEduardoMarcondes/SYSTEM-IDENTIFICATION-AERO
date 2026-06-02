"""
serial_comm.py — Camada de comunicação serial com o firmware do aeropêndulo.
"""

import time
from config import BAUD, SERIAL_TIMEOUT, WAVE_CHUNK

try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


def listar_portas() -> list:
    if not SERIAL_OK:
        return []
    return list(serial.tools.list_ports.comports())


def selecionar_porta() -> str | None:
    portas = listar_portas()
    if not portas:
        print("  Nenhuma porta serial encontrada.")
        return None
    if len(portas) == 1:
        print(f"  Porta detectada: {portas[0].device}  ({portas[0].description})")
        return portas[0].device
    print("\n  Portas seriais disponíveis:")
    for i, p in enumerate(portas):
        print(f"  [{i + 1}] {p.device}  -- {p.description}")
    while True:
        try:
            idx = int(input(f"\n  Escolha [1-{len(portas)}]: ").strip()) - 1
            if 0 <= idx < len(portas):
                return portas[idx].device
        except ValueError:
            pass
        print("  Opção inválida.")


class SerialManager:
    def __init__(self, porta: str, baud: int = BAUD, timeout: float = SERIAL_TIMEOUT):
        self._porta   = porta
        self._baud    = baud
        self._timeout = timeout
        self.ser      = None

    def conectar(self) -> bool:
        if not SERIAL_OK:
            print("  [ERRO] pyserial não instalado.")
            return False
        try:
            self.ser = serial.Serial(self._porta, self._baud, timeout=self._timeout)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            print(f"  Conectado em {self._porta} @ {self._baud} baud")
            return True
        except serial.SerialException as exc:
            print(f"  [ERRO] Não foi possível abrir {self._porta}: {exc}")
            return False

    def fechar(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        self.conectar()
        return self

    def __exit__(self, *_):
        self.fechar()

    def enviar(self, cmd: str):
        self.ser.write((cmd + "\n").encode())

    def readline(self) -> str:
        raw = self.ser.readline()
        return raw.decode("utf-8", errors="replace").strip()

    def aguardar_token(self, token: str, timeout_s: float = 20.0, verbose: bool = True) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            linha = self.readline()
            if linha and verbose:
                print(f"  [Arduino] {linha}")
            if token in linha:
                return True
        print(f"  Timeout aguardando '{token}'.")
        return False

    def parar(self):
        for _ in range(3):
            try:
                self.enviar("STOP")
                self.ser.flush()
            except Exception:
                pass
            time.sleep(0.1)
        self.aguardar_token("PARADO", timeout_s=5.0, verbose=False)

    def enviar_seq(self, degraus: list) -> bool:
        partes = ",".join(f"{int(t * 1000)}:{r:.2f}" for t, r in degraus)
        self.enviar(f"SEQ={partes}")
        return self.aguardar_token("SEQ_OK", timeout_s=10.0)

    def enviar_wave(self, u_array, chunk: int = WAVE_CHUNK) -> bool:
        import numpy as np
        u = np.asarray(u_array, dtype=float)
        n = len(u)

        self.enviar(f"WAVE={n}")
        if not self.aguardar_token("WAVE_OK", timeout_s=10.0):
            return False

        n_blocos = int(np.ceil(n / chunk))
        for bloco_idx in range(n_blocos):
            ini = bloco_idx * chunk
            fim = min(ini + chunk, n)
            valores = ",".join(f"{v:.2f}" for v in u[ini:fim])
            self.enviar(f"DATA={valores}")

            token_esperado = f"DATA_ACK {bloco_idx}"
            if not self.aguardar_token(token_esperado, timeout_s=10.0, verbose=False):
                print(f"  [ERRO] Sem ACK para bloco {bloco_idx}")
                return False

            pct = int(100 * (bloco_idx + 1) / n_blocos)
            print(f"\r  Enviando WAVE: {pct:3d}%  ({fim}/{n} amostras)", end="", flush=True)

        print()
        self.enviar("DATA_END")
        return self.aguardar_token("WAVE_READY", timeout_s=15.0)

    def enviar_chirp(self, amp: float, fmax: float, t0: float, dc: float) -> bool:
        comando = f"CHIRP={amp:.2f},{fmax:.4f},{t0:.2f},{dc:.2f}"
        self.enviar(comando)
        return self.aguardar_token("CHIRP_OK", timeout_s=5.0)

    def recalibrar(self) -> bool:
        self.enviar("RECAL")
        return self.aguardar_token("PRONTO", timeout_s=30.0)

    def iniciar_experimento(self, timeout_s: float = 10.0) -> bool:
        self.ser.reset_input_buffer()
        self.enviar("START")
        if not self.aguardar_token("EXP_START", timeout_s=timeout_s):
            return False
        cab = self.readline()
        if cab:
            print(f"  [Arduino] {cab}")
        return True