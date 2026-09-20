import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd

try:
    import serial
except ImportError:
    print("A biblioteca 'pyserial' não está instalada. Execute 'pip install pyserial' no terminal.")
    sys.exit(1)

# ==========================================
# 1. SETUP DA COMUNICAÇÃO SERIAL
# ==========================================
SERIAL_PORT = 'COM8'
BAUD_RATE = 115200

print(f"Iniciando conexão com o Arduino na porta {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Aguardando 6s para a calibração automática do IMU no Arduino...")
    time.sleep(6) # Aguarda o Arduino fazer a calibração do IMU e iniciar o ESC
    print("Conectado com sucesso ao Arduino (IMU Calibrado)!")
    input("\n[ATENÇÃO] Pressione ENTER para LIGAR O MOTOR e iniciar o voo... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    print("Abortando...")
    sys.exit(1)

# ==========================================
# 2. DEFINIÇÃO DO SINAL DE CONTROLE (MALHA ABERTA)
# ==========================================
Ts = 0.05
# Tempo total: 40s (800 passos)
n_steps = 800
t_vec = np.arange(n_steps) * Ts
u_signal = np.zeros(n_steps)

# Sequência de degraus de 0% a 30% (salto de 5% a cada 5s)
for i, t in enumerate(t_vec):
    if t < 5.0:
        u_signal[i] = 0.0
    elif t < 35.0:
        u_signal[i] = float((int(t) // 5) * 5)
    else:
        u_signal[i] = 0.0

y_history = []
u_history = []

print("\nIniciando Loop de Controle em Malha Aberta...")
for step in range(n_steps):
    loop_start = time.time()
    
    # 1. Lê o controle programado diretamente do vetor
    u_opt = u_signal[step]
    
    # 2. Comunica com o Hardware (Arduino)
    y_meas = 0.0
    if 'ser' in locals() and ser.is_open:
        ser.write(f"{u_opt:.4f}\n".encode())
        arduino_reply = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # Opcional: Descomente para ver tudo
        # print(f"[DEBUG ARDUINO] Recebido: '{arduino_reply}'")
        
        if arduino_reply:
            try:
                y_meas = float(arduino_reply)
            except ValueError:
                print(f"Erro ao ler Arduino: '{arduino_reply}'.")
                y_meas = y_history[-1] if len(y_history) > 0 else 0.0
                
    # 3. Registra histórico
    y_history.append(y_meas)
    u_history.append(u_opt)
    
    # 4. Controle de Tempo Real Estrito (Soft Real-Time)
    elapsed_time = time.time() - loop_start
    if elapsed_time < Ts:
        time.sleep(Ts - elapsed_time)
    else:
        print(f"AVISO: Loop atrasado em {elapsed_time - Ts:.3f}s (Step {step})")
        
    if step % 10 == 0:
        print(f"Step {step:03d} | u (Sinal Aberto): {u_opt:5.2f}% | y (Leitura): {y_meas:5.2f} graus")

print("\nDesligando motor por segurança...")
if 'ser' in locals() and ser.is_open:
    ser.write(f"0.0000\n".encode())
    ser.close()

print("Experimento Malha Aberta Finalizado! Gerando gráficos e CSV...")

# Salva em CSV
df = pd.DataFrame({
    'Tempo(s)': t_vec,
    'Controle_U(%)': u_history,
    'Medicao_Y(graus)': y_history
})
df.to_csv('open_loop_results.csv', index=False)

# ==========================================
# 3. GRÁFICOS DE RESULTADO
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_vec, y_history, 'g-', label='Ângulo Medido (y)')
plt.ylabel('Ângulo (graus)')
plt.title('Experimento Malha Aberta: Resposta ao Degrau no Aeropêndulo')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_vec, u_history, 'm-', label='Controle Programado (u)')
plt.ylabel('PWM (%)')
plt.xlabel('Tempo (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('open_loop_results.png')
plt.show()
