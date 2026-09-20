import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

# Configurações da Serial (Ajuste a COM port para a sua placa!)
COM_PORT = 'COM3' # <-- MUDE ISSO PARA A SUA PORTA COM
BAUD_RATE = 500000

print("Carregando referência do Python...")
try:
    df_py = pd.read_csv('simulacao_python.csv')
    referencia = df_py['referencia'].values
    tempo_py = df_py['tempo_ms'].values
    y_py = df_py['angulo_deg'].values
    u_py = df_py['u_pct'].values
except FileNotFoundError:
    print("Erro: simulacao_python.csv não encontrado. Rode o mpc_v2.py primeiro.")
    sys.exit(1)

total_amostras = len(referencia)
print(f"Total de amostras: {total_amostras}")

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"Conectado na {COM_PORT}")
except Exception as e:
    print(f"Erro ao abrir a porta serial: {e}")
    sys.exit(1)

# Reseta o estado enviando um STOP
ser.write(b'STOP\r\n')
time.sleep(0.5)
ser.read_all()

# Envia o comando WAVE
cmd = f"WAVE={total_amostras}\r\n"
ser.write(cmd.encode())
time.sleep(0.1)

# Envia os dados em blocos
MAX_VALS_PER_LINE = 10
print("Enviando curva de referência...")
for i in range(0, total_amostras, MAX_VALS_PER_LINE):
    chunk = referencia[i : i + MAX_VALS_PER_LINE]
    chunk_str = ",".join([f"{val:.2f}" for val in chunk])
    cmd = f"DATA={chunk_str}\r\n"
    ser.write(cmd.encode())
    time.sleep(0.01) # Pequeno atraso para não engasgar o buffer

# Finaliza o envio
ser.write(b"DATA_END\r\n")
time.sleep(0.5)

print("Iniciando experimento...")
ser.write(b"START\r\n")

# Aguarda "# EXP_START"
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line:
        print(line)
    if "# EXP_START" in line:
        break

# Descarta o cabeçalho "tempo_ms,angulo_deg,u_pct,referencia"
line = ser.readline().decode(errors='ignore').strip()
print(line)

print(f"Coletando {total_amostras} amostras do STM32 SIL...")
linhas_csv = []
amostras_recebidas = 0

while amostras_recebidas < total_amostras:
    line = ser.readline().decode(errors='ignore').strip()
    if not line or line.startswith('#'):
        continue
    
    linhas_csv.append(line.split(','))
    amostras_recebidas += 1
    
    if amostras_recebidas % 100 == 0:
        print(f"{amostras_recebidas}/{total_amostras} coletadas...")

ser.write(b'STOP\r\n')
ser.close()

print("Experimento concluído. Processando dados...")

# Transforma em DataFrame
df_stm = pd.DataFrame(linhas_csv, columns=['tempo_ms', 'angulo_deg', 'u_pct', 'referencia'], dtype=float)
df_stm.to_csv('simulacao_stm32.csv', index=False)

# PLOTAGEM DE COMPARAÇÃO
plt.figure(figsize=(14, 6))
plt.plot(df_py['tempo_ms']/1000.0, df_py['angulo_deg'], 'b-', lw=2, label='Python (Angle)')
plt.plot(df_stm['tempo_ms']/1000.0, df_stm['angulo_deg'], 'r--', lw=2, label='STM32 (Angle)')
plt.plot(df_py['tempo_ms']/1000.0, df_py['referencia'], 'k:', label='Reference')
plt.title("Comparação SIL: PyTorch vs STM32 (C)")
plt.xlabel("Tempo [s]")
plt.ylabel("Ângulo [deg]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(df_py['tempo_ms']/1000.0, df_py['u_pct'], 'b-', lw=2, label='Python (Control u)')
plt.plot(df_stm['tempo_ms']/1000.0, df_stm['u_pct'], 'r--', lw=2, label='STM32 (Control u)')
plt.title("Comparação de Ação de Controle")
plt.xlabel("Tempo [s]")
plt.ylabel("Controle [%]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Tudo pronto! Verifique o gráfico.")
