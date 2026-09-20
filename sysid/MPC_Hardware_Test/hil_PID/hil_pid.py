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
# 1. SETUP DA COMUNICAÇÃO SERIAL (HIL)
# ==========================================
SERIAL_PORT = 'COM8'
BAUD_RATE = 115200

print(f"Iniciando conexão com o Arduino na porta {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Aguardando 6s para a calibração automática do IMU no Arduino...")
    time.sleep(6) 
    print("Conectado com sucesso ao Arduino (IMU Calibrado)!")
    input("\n[ATENÇÃO] Pressione ENTER para LIGAR O MOTOR e iniciar o voo... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    sys.exit(1)

# ==========================================
# 2. CONFIGURAÇÕES E SINAL DE REFERÊNCIA
# ==========================================
Ts = 0.01  # Alterado para 100 Hz

# ALTERE AQUI PARA O CAMINHO DO SEU ARQUIVO CSV DE REFERÊNCIA GERADO PELO MATLAB !!!!!
csv_filename = r'c:\Users\mathe\OneDrive - Grupo Marista\PUCPR\15 Semestre\CONTROLE AVANÇADO\GITs\1-4 DRONE - GIT\SYSTEM-IDENTIFICATION-AERO\MPC_Hardware_Test\generate_ref_signal\sinal2_degraus_malha_fechada.csv'
try:
    df_ref = pd.read_csv(csv_filename)
    csv_time = df_ref['Tempo_s'].values
    csv_amp = df_ref['Amplitude'].values
    
    # O CSV foi gerado com passos de 0.01s, mas nosso hardware roda a Ts=0.05s.
    # Precisamos criar nosso vetor de tempo e sincronizar (interpolar) os valores.
    tempo_total = csv_time[-1]
    n_steps = int(tempo_total / Ts) + 1
    t_vec = np.arange(n_steps) * Ts
    
    # Sincronização Perfeita: Pega o valor exato da Amplitude para o nosso tempo de loop
    ref_signal = np.interp(t_vec, csv_time, csv_amp)
    
    print(f"Sincronização de Tempo Concluída!")
    print(f"Duração Real: {tempo_total:.1f}s | Passos a 20Hz: {n_steps}")
except Exception as e:
    print("\n" + "="*50)
    print(f" ERRO: Não foi possível ler '{csv_filename}'.")
    print("="*50)
    print("Certifique-se de que o arquivo 'referencias.csv' está na mesma pasta")
    print("e contém uma coluna chamada 'referencia' (ou apenas os valores na primeira coluna).")
    print(f"Detalhe do erro: {e}")
    sys.exit(1)

t_vec = np.arange(n_steps) * Ts

y_history = []
u_history = []
ref_history = []

# Variáveis do PID
Kp = 0.5793
Ki = 0.6647
Kd = 0.2
e_1 = 0.0
u_i = 0.0
y_1 = 0.0

# ==========================================
# 3. LOOP DE CONTROLE (HIL puramente PID)
# ==========================================
print("\nSincronizando estado inicial com o Arduino...")

if 'ser' in locals() and ser.is_open:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    y_atual = 0.0
    # Roda vazio por 0.5 segundo (50 passos) para o filtro do Arduino convergir
    for _ in range(50):
        ser.write(f"0.0000\n".encode())
        arduino_reply = ser.readline().decode('utf-8', errors='ignore').strip()
        if arduino_reply:
            try:
                y_atual = float(arduino_reply)
            except ValueError:
                pass
        time.sleep(Ts)
        
    # Inicializa a memória do PID com o estado real lido para evitar "Derivative Kick"
    y_1 = y_atual
    e_1 = ref_signal[0] - y_atual
    print(f"Estado inicial lido com sucesso: {y_atual:.2f} graus. Iniciando controle!")
else:
    y_atual = 0.0
    y_1 = 0.0
    e_1 = 0.0

print("\nIniciando Loop de Controle HIL com PID...")

for step in range(n_steps):
    loop_start = time.time()
    
    # 1. Lei de Controle (PID)
    erro = ref_signal[step] - y_atual
    u_p = Kp * erro
    u_i = u_i + Ki * (Ts / 2.0) * (erro + e_1)
    u_d = -(Kd / Ts) * (y_atual - y_1)
    u_calc = u_p + u_i + u_d
    
    u_opt = np.clip(u_calc, -10.0, 80.0)
    
    # Anti-windup
    if u_calc != u_opt:
        u_i -= (u_calc - u_opt)
        
    e_1 = erro
    y_1 = y_atual
    
    # 2. Comunica com o Hardware (Arduino)
    y_meas = 0.0
    if 'ser' in locals() and ser.is_open:
        ser.write(f"{u_opt:.4f}\n".encode())
        arduino_reply = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if arduino_reply:
            try:
                y_meas = float(arduino_reply)
                y_atual = y_meas # Atualiza para o proximo ciclo
            except ValueError:
                print(f"Erro ao ler Arduino: '{arduino_reply}'.")
                y_meas = y_atual
                
    # 3. Registra histórico
    y_history.append(y_meas)
    u_history.append(u_opt)
    ref_history.append(ref_signal[step])
    
    # 4. Controle de Tempo Real Estrito (Soft Real-Time)
    elapsed_time = time.time() - loop_start
    if elapsed_time < Ts:
        time.sleep(Ts - elapsed_time)
    else:
        print(f"AVISO: Loop atrasado em {elapsed_time - Ts:.3f}s (Step {step})")
        
    if step % 10 == 0:
        print(f"Step {step:03d} | Ref: {ref_signal[step]:.1f} | u (PID): {u_opt:5.2f} | y (Leitura): {y_meas:5.2f}")

print("\nDesligando motor por segurança...")
if 'ser' in locals() and ser.is_open:
    ser.write(f"0.0000\n".encode())
    ser.close()

print("Experimento HIL (PID) Finalizado! Gerando gráficos e salvando dados...")

# ==========================================
# 4. SALVANDO RESULTADOS EM CSV
# ==========================================
basename = os.path.splitext(os.path.basename(csv_filename))[0]
out_csv = f"{basename}_malha_fechada.csv"

df_out = pd.DataFrame({
    'Tempo_s': t_vec,
    'Referencia': ref_history,
    'Saida_y': y_history,
    'Controle_u': u_history
})
df_out.to_csv(out_csv, index=False)
print(f"Dados salvos com sucesso em: {out_csv}")

# ==========================================
# 5. GRÁFICOS DE RESULTADO
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_vec, ref_history, 'k--', label='Referência')
plt.plot(t_vec, y_history, 'g-', label='Saída do Arduino (y)')
plt.ylabel('Ângulo / Estado')
plt.title('Experimento HIL: Python PID Controller + Hardware Real')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_vec, u_history, 'm-', label='Controle do PID (u)')
plt.ylabel('Controle (PWM)')
plt.xlabel('Tempo (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('hil_pid_results.png')
plt.show()
