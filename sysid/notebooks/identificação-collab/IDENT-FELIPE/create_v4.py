import json

input_file = "c:/Users/vicio/Documents/AEROPENDULO/identificação-collab/ANN_NARX_AEROPENDULO_v3.ipynb"
output_file = "c:/Users/vicio/Documents/AEROPENDULO/identificação-collab/ANN_NARX_AEROPENDULO_v4.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if "Identificação NARX v3" in source:
            new_source = source.replace("Identificação NARX v3 (Free-Run Val + ResNet)", "Identificação NARX v4 (Multi-Step BPTT)")
            new_source += "\n7. Treinamento Multi-Step (BPTT) com horizonte H=10.\n"
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Replace the data prep block
        if "Phie_all, Ye_all = [], []" in source and "matReg" in source:
            new_data_prep = """ny = 10
nu = 10
H = 10 # Horizonte Multi-Step

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def make_bptt_windows(y, u, ny, nu, H):
    p_start = max(ny, nu)
    p_end = len(y) - H + 1
    Y_past_list, Y_target_list, U_seq_list = [], [], []
    for p in range(p_start, p_end):
        # Ordem cronológica
        Y_past_list.append(y[p - ny : p])
        Y_target_list.append(y[p : p + H])
        U_seq_list.append(u[p - nu : p + H - 1])
    return np.array(Y_past_list), np.array(Y_target_list), np.array(U_seq_list)

Y_past_all, Y_target_all, U_seq_all = [], [], []
for y_n, u_n in zip(y_train_list, u_train_list):
    yp, yt, us = make_bptt_windows(y_n, u_n, ny, nu, H)
    Y_past_all.append(yp)
    Y_target_all.append(yt)
    U_seq_all.append(us)

Y_past_t = torch.tensor(np.concatenate(Y_past_all, axis=0), dtype=torch.float32).to(device)
Y_target_t = torch.tensor(np.concatenate(Y_target_all, axis=0), dtype=torch.float32).to(device)
U_seq_t = torch.tensor(np.concatenate(U_seq_all, axis=0), dtype=torch.float32).to(device)

dataset = TensorDataset(Y_past_t, U_seq_t, Y_target_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
"""
            cell['source'] = [line + '\n' for line in new_data_prep.split('\n')]
            
        # Modify NARXModel
        if "class NARXModel(nn.Module):" in source:
            new_model = """class NARXModel(nn.Module):
    def __init__(self, ny, nu, hidden_dim1=128, hidden_dim2=64, dropout=0.1):
        super().__init__()
        self.ny = ny
        self.nu = nu
        self.linear_bypass = nn.Linear(ny + nu, 1)
        self.net = nn.Sequential(
            nn.Linear(ny + nu, hidden_dim1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.linear_bypass(x) + self.net(x)
        
    def multi_step_forward(self, y_past, u_seq, H):
        # y_past: (B, ny) in chronological order
        # u_seq: (B, nu + H - 1) in chronological order
        y_preds = []
        y_buffer = y_past.clone()
        for h in range(H):
            y_in = y_buffer[:, -self.ny:]
            u_in = u_seq[:, h : h + self.nu]
            x = torch.cat([y_in, u_in], dim=1)
            pred = self.forward(x) # (B, 1)
            y_preds.append(pred)
            y_buffer = torch.cat([y_buffer, pred], dim=1)
        return torch.cat(y_preds, dim=1) # (B, H)
"""
            cell['source'] = [line + '\n' for line in new_model.split('\n')]

        # Modify Training loop
        if "for epoch in range(epochs):" in source and "epoch_loss /= len(dataloader)" in source:
            new_loop = """model = NARXModel(ny, nu, 128, 64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

epochs = 1000
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 50

# Validacao Free-Run no primeiro dataset de validacao
y_val_fr = y_val_list[0]
u_val_fr = u_val_list[0]
H_val = len(y_val_fr) - max(ny, nu)
yp_val, yt_val, us_val = make_bptt_windows(y_val_fr, u_val_fr, ny, nu, H_val)

Yp_val_t = torch.tensor(yp_val[0:1], dtype=torch.float32).to(device) 
Us_val_t = torch.tensor(us_val[0:1], dtype=torch.float32).to(device)
Yt_val_t = torch.tensor(yt_val[0:1], dtype=torch.float32).to(device)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for y_past, u_seq, targets in dataloader:
        optimizer.zero_grad()
        # Aqui o modelo prevê a trajetória de H=10 passos
        outputs = model.multi_step_forward(y_past, u_seq, H)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss /= len(dataloader)
    
    # Validação via Free-Run Completo no dataset de validacao
    model.eval()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            preds_val = model.multi_step_forward(Yp_val_t, Us_val_t, H_val)
            val_loss = criterion(preds_val, Yt_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_narx.pth')
            patience_counter = 0
        else:
            patience_counter += 5
            
        print(f'Epoch {epoch+1:3d}/{epochs} | Train Loss (Multi-Step H={H}): {epoch_loss:.5f} | Val Loss (Free-Run): {val_loss:.5f} | Best Loss: {best_val_loss:.5f}')
        
        if patience_counter >= patience_limit:
            print(f"Early stopping na epoch {epoch+1}")
            break
"""
            cell['source'] = [line + '\n' for line in new_loop.split('\n')]

        # Modify testing loop (freeRun)
        if "y_test_pred0  = freeRun(model, y_n, u_n, ny, nu, device)" in source:
            new_test = """print("--- FREE-RUN NOS DATASETS DE TESTE ---")
model.load_state_dict(torch.load('best_narx.pth'))

for i, (y_n, u_n, file_name) in enumerate(zip(y_val_list, u_val_list, val_files)):
    H_test = len(y_n) - max(ny, nu)
    yp_test, yt_test, us_test = make_bptt_windows(y_n, u_n, ny, nu, H_test)
    
    if len(yp_test) == 0:
        continue
        
    Yp_t = torch.tensor(yp_test[0:1], dtype=torch.float32).to(device)
    Us_t = torch.tensor(us_test[0:1], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        y_test_pred0 = model.multi_step_forward(Yp_t, Us_t, H_test).cpu().numpy().flatten()
    
    y_real_eval = yt_test[0]
    
    # Desfaz a normalizacao
    y_real_original = scaler_y.inverse_transform(y_real_eval.reshape(-1, 1)).flatten()
    y_pred_original = scaler_y.inverse_transform(y_test_pred0.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_real_original, y_pred_original)
    rmse = np.sqrt(np.mean((y_real_original - y_pred_original)**2))
    
    print(f"Dataset: {file_name}")
    print(f"  R2: {r2:.4f} | RMSE: {rmse:.2f} graus")
    
    plt.figure(figsize=(10,4))
    plt.plot(y_real_original, label='Real')
    plt.plot(y_pred_original, label='NARX BPTT Free-Run', linestyle='dashed')
    plt.title(f'Free-Run: {file_name}')
    plt.legend()
    plt.grid()
    plt.show()
"""
            cell['source'] = [line + '\n' for line in new_test.split('\n')]
            
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("v4 criado com sucesso.")
