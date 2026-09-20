import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, decimate
import os

# ──────────────────────────────────────────────────────────────────────
# 1. PRÉ-PROCESSAMENTO (Idêntico ao node_v7.py)
# ──────────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
DECIMACAO = 2
START_IDX = 200
END_IDX = -150

def carregar_experimento(url, decimacao=DECIMACAO, start_idx=START_IDX, end_idx=END_IDX):
    try:
        df = pd.read_csv(url)
    except:
        local_path = url.replace(BASE, "")
        df = pd.read_csv(local_path)

    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    u_full = df['u_pct'].values.astype(np.float64) if 'u_pct' in df.columns \
             else df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_full[::decimacao]
    else:
        u_raw, y_raw = u_full, y_full

    dt    = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt
    min_l = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_l], u_raw[:min_l], y_raw[:min_l]

    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]

    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    t_raw  = t_raw - t_raw[0]
    y_rad  = y_raw * (np.pi / 180.0)
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)
    dt_m   = np.mean(np.diff(t_raw))
    v_rad  = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_m)
    x_mat  = np.vstack((y_rad, v_rad)).T
    return (
        torch.tensor(t_raw, dtype=torch.float32),
        torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(x_mat,  dtype=torch.float32),
        y_rad, v_rad, u_norm,
    )

def carregar_lista(file_list):
    datasets = []
    for f in file_list:
        print(f"Carregando {f}...")
        t, u, y = carregar_experimento(BASE + f)
        t_ten, u_ten, x_ten, *_ = processar_dataset(t, u, y)
        datasets.append({'name': os.path.basename(f), 't': t_ten, 'u': u_ten, 'x': x_ten})
    return datasets


# ──────────────────────────────────────────────────────────────────────
# 2. DEFINIÇÃO DOS DATASETS (Padrão node_v7.py)
# ──────────────────────────────────────────────────────────────────────
# Para um treino leve, vamos pegar só alguns degraus e mix. 
# Você pode expandir para a lista inteira do node_v7 depois.
TRAIN_FILES = [
    "RODADA-7/MIX_DC_60.csv",
    "RODADA-7/MIX_DC_45.csv",
    "RODADA-7/multi-seno-45-030Hz_0904_20-23.csv"
]

VAL_FILES = [
    "RODADA-3/chirp-1_0807_16-34.csv", # O mesmo chirp valid do node_v7
    "RODADA-5/seq-degraus-1_0827_17-46.csv"
]

TEST_FILES = [
    "RODADA-5/aprbs-1_0827_17-19.csv",
    "RODADA-5/multi-seno-1_0827_17-34.csv"
]


# ──────────────────────────────────────────────────────────────────────
# 3. MODELO GRU (Rede Recorrente Leve)
# ──────────────────────────────────────────────────────────────────────
class MicroGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, state_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, state_dim)

    def forward(self, u_seq, h0=None):
        gru_out, _ = self.gru(u_seq, h0)
        pred_x = self.out(gru_out)
        return pred_x

# ──────────────────────────────────────────────────────────────────────
# 4. VALIDAÇÃO (FREE RUN) E TREINAMENTO
# ──────────────────────────────────────────────────────────────────────
def avalia_free_run(model, datasets):
    """Simulação livre: roda todo o u_seq e gera o y_seq sem realimentação."""
    model.eval()
    resultados = []
    with torch.no_grad():
        for ds in datasets:
            u_seq = ds['u'].unsqueeze(0) # (1, seq_len, 1)
            x_real = ds['x'].numpy()
            y_real = x_real[:, 0] * (180 / np.pi)
            
            x_pred = model(u_seq).squeeze(0).numpy()
            y_pred = x_pred[:, 0] * (180 / np.pi)
            
            rmse = float(np.sqrt(np.mean((y_pred - y_real) ** 2)))
            resultados.append({
                'name': ds['name'], 'rmse': rmse, 
                'y_real': y_real, 'y_pred': y_pred, 't': ds['t'].numpy(),
            })
    return resultados


def train_gru():
    print(">>> Carregando datasets...")
    train_datasets = carregar_lista(TRAIN_FILES)
    val_datasets   = carregar_lista(VAL_FILES)
    test_datasets  = carregar_lista(TEST_FILES)
    
    model = MicroGRU(input_dim=1, hidden_dim=16, state_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 600
    patience = 50
    best_val = float('inf')
    best_sd = None
    no_improve = 0
    
    print("\n>>> Iniciando Treinamento da Micro GRU...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # Treina passando a sequência inteira (Truncated BPTT nativo não é necessário p/ sec pequenas)
        for ds in train_datasets:
            u_seq = ds['u'].unsqueeze(0)
            x_seq = ds['x'].unsqueeze(0)
            
            optimizer.zero_grad()
            x_pred = model(u_seq)
            loss = criterion(x_pred, x_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validação cruzada (Free run na validação)
        if epoch % 20 == 0:
            val_res = avalia_free_run(model, val_datasets)
            val_rmse = np.mean([r['rmse'] for r in val_res])
            
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss/len(train_datasets):.6f} | Val RMSE: {val_rmse:.2f} graus")
            
            if val_rmse < best_val:
                best_val = val_rmse
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 20
                
            if no_improve >= patience:
                print(f"    Early stop acionado (paciência esgotada).")
                break
                
    # Recupera o melhor modelo do Early Stopping
    model.load_state_dict(best_sd)
    print("\n>>> Treinamento finalizado. Melhor RMSE na Validação:", best_val)
    
    # ──────────────────────────────────────────────────────────────────
    # 5. TESTE FINAL (FREE RUN) E PLOT
    # ──────────────────────────────────────────────────────────────────
    print("\n>>> Avaliando Free Run nos datasets de Teste...")
    test_res = avalia_free_run(model, test_datasets)
    
    fig, axes = plt.subplots(len(test_res), 1, figsize=(12, 4*len(test_res)))
    if len(test_res) == 1: axes = [axes]
    
    for i, res in enumerate(test_res):
        axes[i].plot(res['t'], res['y_real'], label='Real', color='black', lw=1.5)
        axes[i].plot(res['t'], res['y_pred'], label='Predição GRU', color='orange', linestyle='--', lw=1.5)
        axes[i].set_title(f"Free Run: {res['name']} (RMSE = {res['rmse']:.2f} graus)")
        axes[i].set_ylabel('Ângulo [Graus]')
        axes[i].set_xlabel('Tempo [s]')
        axes[i].legend()
        axes[i].grid()
        print(f" Teste {res['name']}: RMSE = {res['rmse']:.2f} graus")
        
    plt.tight_layout()
    plt.savefig('resultado_gru_freerun.png')
    print("Gráfico final salvo como 'resultado_gru_freerun.png'.")
    
    # Salvar os pesos do modelo
    torch.save(model.state_dict(), 'modelo_gru.pth')
    print("Pesos do modelo salvos em 'modelo_gru.pth'.")

if __name__ == '__main__':
    train_gru()
