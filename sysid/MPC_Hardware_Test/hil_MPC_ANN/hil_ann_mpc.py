import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

try:
    import serial
except ImportError:
    print("A biblioteca 'pyserial' não está instalada. Execute 'pip install pyserial' no terminal.")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import joblib
except ImportError:
    print("Bibliotecas 'torch' ou 'joblib' faltando. Execute: pip install torch joblib scikit-learn")
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
    time.sleep(6) # Aguarda 6s para o Arduino fazer a calibração do IMU e iniciar o ESC
    print("Conectado com sucesso ao Arduino (IMU Calibrado)!")
    input("\n[ATENÇÃO] Pressione ENTER para LIGAR O MOTOR e iniciar o voo... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    print("Continuando em modo de simulação virtual...")

# ==========================================
# 2. DEFINIÇÃO DA REDE NEURAL (ANN)
# ==========================================
Ts = 0.05
ny_model = 5
nu_model = 5
nx = ny_model + nu_model
N = 10
input_dim = nx + N
output_dim = 1

# Arquitetura idêntica à que configuramos no Notebook
class MPCApproximator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MPCApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Caminho para os pesos da rede e o padronizador (scaler)
model_path = 'ann_model.pth'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("\n" + "="*60)
    print(" ERRO: Arquivos da Rede Neural não encontrados!")
    print("="*60)
    print(f" Não encontrei '{model_path}' e/ou '{scaler_path}' na pasta atual.")
    print(" \nPara rodar o HIL com a ANN, você precisa salvar a rede no notebook.")
    print(" \nAdicione esta célula ao final do seu Jupyter Notebook e execute-a:")
    print(" -------------------------------------------------------------")
    print(" import joblib")
    print(" import torch")
    print(" ")
    print(" # Salva os pesos da rede")
    print(" torch.save(model.state_dict(), 'ann_model.pth')")
    print(" ")
    print(" # Salva o padronizador (MUITO IMPORTANTE para as escalas)")
    print(" joblib.dump(scaler, 'scaler.pkl')")
    print(" -------------------------------------------------------------")
    print("\nDepois, copie os dois arquivos gerados para esta pasta MPC_Hardware_Test.")
    sys.exit(1)

print("Carregando Rede Neural e Scaler...")
model = MPCApproximator(input_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval() # Modo de inferência
scaler = joblib.load(scaler_path)
print("Carregados com sucesso!")

def F_virtual(x_current, u_opt):
    # y_k = 1.9373*y(k-1) - 0.9545*y(k-2) + 0.0007*u(k-2)^2 + 0.0000*u(k-4)*u(k-5)^2 - 0.0004*u(k-2)*u(k-3)
    y_k = (1.9373 * x_current[0] 
         - 0.9545 * x_current[1] 
         + 0.0007 * (x_current[ny_model+1] * x_current[ny_model+1])
         + 0.0000 * (x_current[ny_model+3] * x_current[ny_model+4] * x_current[ny_model+4])
         - 0.0004 * (x_current[ny_model+1] * x_current[ny_model+2]))
    return y_k

# ==========================================
# 3. LOOP DE CONTROLE (HIL)
# ==========================================
# Tempo total: 40s (800 passos)
n_steps = 800
t_vec = np.arange(n_steps + N) * Ts
ref_signal = np.ones(n_steps + N) * 45.0  # Inicia em 45 graus

# Step de 45 para 50 graus aos 30 segundos
mask = (t_vec >= 30.0)
ref_signal[mask] = 50.0

x_current = np.zeros(nx) # [y(k-1)... y(k-5), u(k-1)... u(k-5)]
y_history = []
u_history = []
ref_history = []

# Variáveis do PID (Fase de Aquecimento)
Kp = 0.5793
Ki = 0.6647
Kd = 0.2
e_1 = 0.0
u_i = 0.0
y_1 = 0.0

print("\nIniciando Loop de Controle Híbrido (PID -> ANN)...")
for step in range(n_steps):
    loop_start = time.time()
    
    # 1. Lei de Controle (Bumpless Transfer: PID -> ANN)
    y_atual = x_current[0] # Usa o estado atual conhecido
    
    if step * Ts < 20.0:
        # -- FASE 1: PID (0 a 20s) --
        erro = ref_signal[step] - y_atual
        u_p = Kp * erro
        u_i = u_i + Ki * (Ts / 2.0) * (erro + e_1)
        u_d = -(Kd / Ts) * (y_atual - y_1)
        u_calc = u_p + u_i + u_d
        
        u_opt = np.clip(u_calc, -10.0, 80.0)
        if u_calc != u_opt:
            u_i -= (u_calc - u_opt) # Anti-windup
            
        e_1 = erro
        y_1 = y_atual
    else:
        # -- FASE 2: ANN MPC (a partir de 20s) --
        pref_k = ref_signal[step : step + N]
        P_k = np.concatenate((x_current, pref_k)).reshape(1, -1)
        P_scaled = scaler.transform(P_k)
        tensor_P = torch.tensor(P_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            u_opt = model(tensor_P).item()
        
        u_opt = np.clip(u_opt, -10.0, 80.0)
    
    # 2. Comunica com o Hardware (Arduino)
    y_meas = 0.0
    if 'ser' in locals() and ser.is_open:
        ser.write(f"{u_opt:.4f}\n".encode())
        arduino_reply = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"[DEBUG ARDUINO] Recebido: '{arduino_reply}'")
        if arduino_reply:
            try:
                y_meas = float(arduino_reply)
            except ValueError:
                print(f"Erro ao ler Arduino: '{arduino_reply}'.")
                y_meas = x_current[0]
    else:
        # Fallback de simulação virtual
        y_meas = F_virtual(x_current, u_opt)     #CUIDADO
        
    # 3. Atualiza o estado atual (shift registers)
    for i in range(ny_model-1, 0, -1):
        x_current[i] = x_current[i-1]
    x_current[0] = y_meas
    
    for i in range(nu_model-1, 0, -1):
        x_current[ny_model+i] = x_current[ny_model+i-1]
    x_current[ny_model] = u_opt
    
    # 4. Registra histórico

    # ----------------------------------------------------
    # Controle de Tempo Real Estrito (Soft Real-Time)
    # ----------------------------------------------------
    elapsed_time = time.time() - loop_start
    if elapsed_time < Ts:
        time.sleep(Ts - elapsed_time)
    else:
        print(f"AVISO: Loop atrasado em {elapsed_time - Ts:.3f}s (Step {step})")
        
    y_history.append(y_meas)
    u_history.append(u_opt)
    ref_history.append(ref_signal[step])
    
    if step % 10 == 0:
        print(f"Step {step:03d} | Ref: {ref_signal[step]:.1f} | u (ANN): {u_opt:5.2f} | y (Leitura): {y_meas:5.2f}")

print("Experimento HIL (ANN) Finalizado! Gerando gráficos...")

# ==========================================
# 4. GRÁFICOS DE RESULTADO
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(ref_history, 'k--', label='Referência')
plt.plot(y_history, 'g-', label='Saída do Arduino (y)')
plt.ylabel('Ângulo / Estado')
plt.title('Experimento HIL: Python ANN Controller + Arduino Emulator')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(u_history, 'm-', label='Controle da ANN (u)')
plt.ylabel('Controle (PWM)')
plt.xlabel('Passo (Step)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('hil_ann_results.png')
plt.show()
