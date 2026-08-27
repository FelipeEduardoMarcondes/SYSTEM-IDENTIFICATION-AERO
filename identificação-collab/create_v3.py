import json

input_file = "c:/Users/vicio/Documents/AEROPENDULO/identificação-collab/ANN_NARX_AEROPENDULO_v2.ipynb"
output_file = "c:/Users/vicio/Documents/AEROPENDULO/identificação-collab/ANN_NARX_AEROPENDULO_v3.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if "Identificação NARX v2" in source:
            new_source = source.replace("Identificação NARX v2", "Identificação NARX v3 (Free-Run Val + ResNet)")
            new_source += "\n5. Validação guiada pelo erro de Free-Run.\n6. Bypass Linear na Arquitetura (ResNet).\n"
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 1. Modificar a classe NARXModel
        if "class NARXModel(nn.Module):" in source:
            new_model = """class NARXModel(nn.Module):
    def __init__(self, ny, nu, hidden_dim1=256, hidden_dim2=128, dropout=0.1):
        super().__init__()
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
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)"""
            cell['source'] = [line + '\n' for line in new_model.split('\n')]
            
        # 2. Modificar o laço de treinamento
        if "for epoch in range(epochs):" in source and "noise_std" in source:
            new_loop = """model = NARXModel(ny, nu, 128, 64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

epochs = 1000
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 50  # épocas sem melhora

for epoch in range(epochs):
    # Ruído decresce ao longo do treino: começa alto, termina baixo
    noise_std = 0.1 * (1 - epoch / epochs)  # de 0.1 a 0.0
    
    model.train()
    epoch_loss = 0.0
    for inputs, targets in dataloader:
        noisy_inputs = inputs.clone()
        noisy_inputs[:, :ny] += torch.randn_like(noisy_inputs[:, :ny]) * noise_std
        
        optimizer.zero_grad()
        outputs = model(noisy_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss /= len(dataloader)
    
    # Validação via Free-Run
    # Vamos avaliar o Free-Run no primeiro dataset de validação para guiar o Early Stopping
    model.eval()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        y_val_fr = y_val_list[0]
        u_val_fr = u_val_list[0]
        y_test_pred_fr = freeRun(model, y_val_fr, u_val_fr, ny, nu, device)
        
        p = max(ny, nu) + 1
        y_real_eval = y_val_fr[p-1:]
        
        # MSE do Free-Run
        val_loss = np.mean((y_real_eval - y_test_pred_fr)**2)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_narx.pth')
            patience_counter = 0
        else:
            patience_counter += 5 # Incrementa 5 pois checamos a cada 5 épocas
            
        print(f'Epoch {epoch+1:3d}/{epochs} | Train Loss (OSA): {epoch_loss:.5f} | Val Loss (Free-Run): {val_loss:.5f}')
        
        if patience_counter >= patience_limit:
            print(f"Early stopping na epoch {epoch+1}")
            break
"""
            cell['source'] = [line + '\n' for line in new_loop.split('\n')]
            
        # Remover a saída anterior para limpar o notebook
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("v3 criado com sucesso.")
