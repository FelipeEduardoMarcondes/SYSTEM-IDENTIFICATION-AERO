import serial
import time
import torch
import torch.nn as nn
import numpy as np
import pickle
import threading
import sys
import os

# Adiciona o caminho para importar o live_plot do projeto antigo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PYTHON'))
from live_plot import LivePlot

# ── Configurações ─────────────────────────────────────────────────────────────
PORTA = 'COM8'  # Ajuste para a sua porta COM
BAUDRATE = 500000

ny_model = 5
nu_model = 5
N_mpc = 10
nx = ny_model + nu_model

# ── Modelo PyTorch ─────────────────────────────────────────────────────────────
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

# ── Carregamento do Modelo e Scaler ──────────────────────────────────────────
try:
    model = MPCApproximator(input_dim=nx + N_mpc, output_dim=1)
    model.load_state_dict(torch.load('mpc_model.pt'))
    model.eval()

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("[OK] Modelo e Scaler carregados com sucesso!")
except FileNotFoundError:
    print("[ERRO] Arquivos 'mpc_model.pt' e/ou 'scaler.pkl' nao encontrados.")
    print("       Execute o bloco de exportacao no seu Jupyter Notebook primeiro.")
    exit(1)

# ── Loop de Controle PC-in-the-Loop ──────────────────────────────────────────
def controle_real_time():
    try:
        ser = serial.Serial(PORTA, BAUDRATE, timeout=0.1)
        time.sleep(2)  # Aguarda reset do Arduino
        ser.reset_input_buffer()
        print(f"[SERIAL] Conectado na porta {PORTA}")
    except Exception as e:
        print(f"[ERRO] Nao foi possivel conectar na porta {PORTA}: {e}")
        return

    # Passo de Calibracao
    input("\n[ATENCAO] Pressione ENTER para calibrar o giroscopio (Nao encoste no Drone!)...")
    ser.write(b"RECAL\n")
    print("Calibrando (aguarde 3 segundos)...")
    time.sleep(3)
    
    # Aguarda usuario iniciar
    input("\n[ATENCAO] Pressione ENTER para LIGAR O MOTOR e iniciar o controle MPC...")

    # Inicia a coleta
    ser.write(b"START_PC\n")
    print("[INFO] Comando START_PC enviado.")

    # Buffers de estado (ny_model=5, nu_model=5)
    y_buffer = np.zeros(ny_model) # Inicia em 0 graus
    u_buffer = np.zeros(nu_model) # Inicia com u=0
    
    # Variáveis da Trava de Segurança (PID Fallback)
    ann_falhou = False
    Kp = 0.5793
    Ki = 0.6647
    Kd = 0.2
    e_1 = 0.0
    u_i = 0.0
    y_1 = 0.0
    Ts = 0.05
    
    def get_reference_sequence(t, N):
        Ts = 0.05
        refs = []
        for k in range(N):
            t_future = t + k * Ts
            if t_future < 5.0:
                refs.append(5.0)
            elif t_future < 15.0:
                # 1 periodo em 10s. Amplitude: 5 a 45 (A=20 -> 5 + 20*2 = 45)
                t_osc = t_future - 5.0
                omega = 2 * np.pi / 10.0
                ref_val = 5.0 + 20.0 * (1 - np.cos(omega * t_osc))
                refs.append(ref_val)
            else:
                refs.append(5.0)
        return np.array(refs)
    
    # Log no arquivo e Listas do Plot
    log_file = open("log_pc_mpc.csv", "w")
    log_file.write("tempo_pc,y_medido,u_aplicado,referencia\n")
    
    t_list = []
    ang_list = []
    u_list = []
    ref_list = []
    
    live = LivePlot(nome="MPC ANN Control", janela_s=15.0)
    
    print("[INFO] Aguardando dados do Arduino...")
    tempo_inicio = time.time()
    
    try:
        while True:
            linha = ser.readline().decode('utf-8').strip()
            
            if linha.startswith("Y:"):
                y_medido = float(linha[2:])
                
                y_buffer = np.roll(y_buffer, -1)
                y_buffer[-1] = y_medido
                
                t_atual = time.time() - tempo_inicio
                referencia_futura = get_reference_sequence(t_atual, N_mpc)
                ref_atual = referencia_futura[0]
                
                entrada_bruta = np.concatenate([y_buffer, u_buffer, referencia_futura])
                entrada_scaled = scaler.transform(entrada_bruta.reshape(1, -1))
                entrada_tensor = torch.tensor(entrada_scaled, dtype=torch.float32)
                
                erro = ref_atual - y_medido
                
                # Gatilho de Segurança
                if not ann_falhou and abs(erro) > 15.0:
                    print(f"\n[ALERTA] Erro passou de 15 graus ({erro:.1f}g). ANN cortada! Ativando PID!")
                    ann_falhou = True
                    e_1 = erro
                    y_1 = y_medido
                
                if ann_falhou:
                    u_p = Kp * erro
                    u_i = u_i + Ki * (Ts / 2.0) * (erro + e_1)
                    u_d = -(Kd / Ts) * (y_medido - y_1)
                    u_calc = u_p + u_i + u_d
                    
                    u_sat = np.clip(u_calc, -10.0, 80.0)
                    if u_calc != u_sat:
                        u_i -= (u_calc - u_sat)
                    
                    e_1 = erro
                    y_1 = y_medido
                else:
                    with torch.no_grad():
                        u_calc = model(entrada_tensor).item()
                    u_sat = np.clip(u_calc, -10.0, 80.0)
                
                u_buffer = np.roll(u_buffer, -1)
                u_buffer[-1] = u_sat
                
                # Envia
                comando = f"U={u_sat:.2f}\n"
                ser.write(comando.encode('utf-8'))
                
                # Armazena historico
                t_list.append(t_atual)
                ang_list.append(y_medido)
                u_list.append(u_sat)
                ref_list.append(ref_atual)
                
                # Log em disco
                log_file.write(f"{t_atual:.3f},{y_medido:.2f},{u_sat:.2f},{ref_atual:.2f}\n")
                
                # Plota ao vivo
                live.atualizar(t_list, ang_list, u_list, ref_list)
                
                if int(t_atual * 20) % 10 == 0:
                    print(f"t={t_atual:.2f}s | ref={ref_atual:.1f} | y={y_medido:.2f} | u={u_sat:.2f}")
                    
                # Desliga tudo ao final de 20s
                if t_atual > 20.0:
                    print("\n[INFO] Fim dos 20 segundos de simulação! Desligando tudo...")
                    raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        print("\n[INFO] Teste interrompido pelo usuario. Parando motor...")
        ser.write(b"STOP\n")
        time.sleep(0.5)
        live.fechar()
        ser.close()
        log_file.close()
        print("[INFO] Sistema finalizado.")

if __name__ == "__main__":
    controle_real_time()
