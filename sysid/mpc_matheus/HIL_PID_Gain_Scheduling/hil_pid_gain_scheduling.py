# ─────────────────────────────────────────────────────────────────────────────
# HIL — PID com GAIN SCHEDULING para o 1/4-drone (aeropêndulo)
# ─────────────────────────────────────────────────────────────────────────────
# Implementa o controlador projetado em pid_gain_scheduling_14_drone.ipynb e o
# coloca em malha fechada via SERIAL com o "Arduino" (real ou emulado).
#
# Protocolo serial (idêntico ao arduino_real_slave / arduino_emulator):
#   PC      -> Arduino:  "<u>\n"   u = ação de controle em PORCENTAGEM
#   Arduino -> PC:       "<y>\n"   y = ângulo medido em GRAUS
#
# DOIS MODOS DE EXECUÇÃO (controlados por USE_EMULATOR abaixo):
#
#   USE_EMULATOR = True   -> tudo no notebook. A porta serial é substituída por
#                            FakeSerialArduino, que roda o MESMO modelo físico do
#                            arduino_emulator.ino e emula a troca de bytes u<->y.
#                            Não precisa de hardware nem de porta serial.
#
#   USE_EMULATOR = False  -> testa em hardware. Abre SERIAL_PORT com pyserial e
#                            conversa com o Arduino real (flash do arduino_real_slave
#                            para a bancada física, ou do arduino_emulator para
#                            validar com o modelo rodando no próprio Arduino).
#
# O laço de controle é EXATAMENTE o mesmo nos dois modos — é isso que valida o
# software antes de mandar para os alunos testarem no hardware.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser(description="HIL PID Gain Scheduling")
parser.add_argument('--port', default='COM8', help="Porta Serial do Arduino")
args = parser.parse_args()

# ==========================================
# 0. CHAVES DE CONFIGURAÇÃO
# ==========================================
REALTIME     = True          # O tempo real é forçado
SERIAL_PORT  = args.port
BAUD_RATE    = 115200

# ==========================================
# 1. MODELO E PROJETO DO PID (hardcoded)
# ==========================================
# Parâmetros físicos identificados (phys_14_drone.ipynb), os mesmos usados no
# projeto do gain scheduling.
C1 = -8.29151533
C2 =  0.00418887
C3 = -1.43673669

Ts = 0.05                    # tempo de amostragem [s] (20 Hz) — igual ao projeto

# Saturação da ação de controle (%). O projeto usou +/-100 e o ESC real
# (arduino_real_slave/pctParaUs) já está casado: +/-100 % -> 1000..2000 us
# (faixa cheia do ESC).
U_MIN, U_MAX = -100.0, 100.0

# --- Ganhos do PID: carregados a partir do JSON gerado pelo notebook
try:
    with open(os.path.join(os.path.dirname(__file__), 'pid_gains.json'), 'r') as f:
        gains = json.load(f)
except FileNotFoundError:
    print("ERRO: pid_gains.json não encontrado. Rode o notebook pid_gain_scheduling_14_drone.ipynb primeiro.")
    sys.exit(1)

SCHED_NODES = np.array(gains["SCHED_NODES"], dtype=float)
KP_NODES = np.array(gains["KP_NODES"], dtype=float)
KI_NODES = np.array(gains["KI_NODES"], dtype=float)
KD_NODES = np.array(gains["KD_NODES"], dtype=float)
VEQ_NODES = np.array(gains["VEQ_NODES"], dtype=float)

# Como KI e KD são constantes em nosso projeto (por conta do pole placement),
# o script histórico os consome como escalares. Usaremos o valor do primeiro nó:
KD = KD_NODES[0]
KI = KI_NODES[0]

def kp_scheduled(theta_deg):
    """Kp agendado pela SAÍDA medida (interp. linear entre nós, como no notebook)."""
    return np.interp(np.clip(theta_deg, 0.0, 180.0), SCHED_NODES, KP_NODES)

def v_eq_feedforward(ref_deg):
    """Feedforward de equilíbrio v_eq agendado pela REFERÊNCIA, através de interpolação."""
    return np.interp(np.clip(ref_deg, 0.0, 180.0), SCHED_NODES, VEQ_NODES)


# ==========================================
# 2. SINAL DE REFERÊNCIA (0 -> 180, de 10 em 10)
# ==========================================
REF_LEVELS = np.arange(0, 181, 10)       # 0, 10, 20, ..., 180  (19 níveis)
T_STEP     = 5.0                          # segundos em cada degrau
SWEEP_DOWN = True                        # True = sobe e desce (0..180..0)

if SWEEP_DOWN:
    REF_LEVELS = np.concatenate([REF_LEVELS, REF_LEVELS[-2::-1]])

n_per_step = int(round(T_STEP / Ts))
ref_signal = np.repeat(REF_LEVELS, n_per_step).astype(float)
n_steps    = len(ref_signal)
t_vec      = np.arange(n_steps) * Ts

# ==========================================
# 3. CONEXÃO (porta serial agnóstica)
# ==========================================
try:
    import serial
except ImportError:
    print("A biblioteca 'pyserial' não está instalada. Execute 'pip install pyserial'.")
    sys.exit(1)

print(f"Conectando ao Arduino em {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Aguardando 6s para a calibração automática do IMU/Filtro no Arduino...")
    time.sleep(6)
    print("Conectado com sucesso ao Arduino!")
    input("\n[ATENÇÃO] Pressione ENTER para iniciar a comunicação serial... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    sys.exit(1)

# Resumo do projeto (deve bater com a tabela impressa no notebook)
print("\nGanhos do PID (gain scheduling):")
print(f"  Kd = {KD:.1f} (constante) | Ki = {KI:.1f} (constante)")
for ang, kp in zip(SCHED_NODES, KP_NODES):
    print(f"  Angle {ang:>3.0f} deg | Kp = {kp:7.1f}")
print(f"\nReferência: {REF_LEVELS[0]:.0f} -> {max(REF_LEVELS):.0f} graus, "
      f"de 10 em 10, {T_STEP:.0f}s por degrau | {n_steps} passos a {1/Ts:.0f} Hz")


# ==========================================
# 5. SINCRONIZAÇÃO INICIAL
# ==========================================
y_history, u_history, ref_history = [], [], []

# Estado do PID
e_int  = 0.0
e_prev = 0.0
y_atual = 0.0

print("\nSincronizando estado inicial com o Arduino...")
ser.reset_input_buffer()
ser.reset_output_buffer()

# Roda em vazio (u=0) por 0.5 s para o filtro/estado do Arduino convergir
for _ in range(50):
    ser.write(b"0.0000\n")
    reply = ser.readline().decode('utf-8', errors='ignore').strip()
    if reply:
        try:
            y_atual = float(reply)
        except ValueError:
            pass
    if REALTIME:
        time.sleep(Ts)

# Inicializa a memória do termo derivativo com o estado real lido
e_prev = np.deg2rad(ref_signal[0]) - np.deg2rad(y_atual)
print(f"Estado inicial lido: {y_atual:.2f} graus. Iniciando controle!")


# ==========================================
# 6. LAÇO DE CONTROLE HIL — PID GAIN SCHEDULING
# ==========================================
print("\nIniciando Loop de Controle HIL...\n")

for step in range(n_steps):
    loop_start = time.time()

    r_deg = ref_signal[step]

    # --- Lei de controle (em RADIANOS, como no projeto) ---
    e = np.deg2rad(r_deg) - np.deg2rad(y_atual)
    e_int += e * Ts
    e_der = (e - e_prev) / Ts
    e_prev = e

    Kp   = kp_scheduled(y_atual)          # agendado pela saída
    v_eq = v_eq_feedforward(r_deg)        # feedforward agendado pela referência

    dv_k = Kp * e + KI * e_int + KD * e_der
    v_k  = v_eq + dv_k                     # esforço de controle virtual total

    # Volta para a entrada física: u = sign(v)*sqrt(|v|)
    u_calc = np.sign(v_k) * np.sqrt(abs(v_k))
    u_opt  = float(np.clip(u_calc, U_MIN, U_MAX))

    # --- Troca com o hardware/emulador ---
    ser.write(f"{u_opt:.4f}\n".encode())
    reply = ser.readline().decode('utf-8', errors='ignore').strip()
    y_meas = y_atual
    if reply:
        try:
            y_meas = float(reply)
            y_atual = y_meas
        except ValueError:
            print(f"Resposta inválida do Arduino: '{reply}'")

    y_history.append(y_meas)
    u_history.append(u_opt)
    ref_history.append(r_deg)

    # --- Tempo real estrito (só no hardware ou se REALTIME=True) ---
    if REALTIME:
        elapsed = time.time() - loop_start
        if elapsed < Ts:
            time.sleep(Ts - elapsed)
        elif step > 0:
            print(f"AVISO: loop atrasado em {elapsed - Ts:.3f}s (step {step})")

    if step % 20 == 0:
        print(f"Step {step:04d} | Ref: {r_deg:5.1f} | u: {u_opt:7.2f} | y: {y_meas:7.2f} | Kp: {Kp:6.0f}")

# Desliga o motor por segurança
print("\nDesligando motor por segurança...")
ser.write(b"0.0000\n")
ser.close()

print("Experimento HIL (PID gain scheduling) finalizado! Salvando dados e gráficos...")

# ==========================================
# 7. SALVA RESULTADOS
# ==========================================
out_dir = os.path.dirname(os.path.abspath(__file__))
out_csv = os.path.join(out_dir, "hil_gain_scheduling_hardware.csv")

import csv as _csv
with open(out_csv, 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['Tempo_s', 'Referencia', 'Saida_y', 'Controle_u'])
    for i in range(n_steps):
        w.writerow([f"{t_vec[i]:.4f}", f"{ref_history[i]:.4f}",
                    f"{y_history[i]:.4f}", f"{u_history[i]:.4f}"])
print(f"Dados salvos em: {out_csv}")

# ==========================================
# 8. GRÁFICOS
# ==========================================
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_vec, ref_history, 'k--', lw=2, label='Referência')
plt.plot(t_vec, y_history, 'b-', lw=1.5, label='Saída y (Arduino)')
plt.ylabel('Ângulo [graus]')
plt.title('HIL PID Gain Scheduling — Teste Hardware')
plt.legend(); plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_vec, u_history, 'm-', lw=1.5, label='Controle u [%]')
plt.axhline(U_MAX, color='k', ls=':', alpha=0.5)
plt.axhline(U_MIN, color='k', ls=':', alpha=0.5)
plt.ylabel('Controle u [%]'); plt.xlabel('Tempo [s]')
plt.legend(); plt.grid(True)

plt.tight_layout()
out_png = os.path.join(out_dir, "hil_gain_scheduling_hardware.png")
plt.savefig(out_png, dpi=110)
print(f"Gráfico salvo em: {out_png}")
plt.show()
