import os
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_comparative import PhysicsODE_Baseline, PhysicsODE_Coulomb, PhysicsODE_Tustin, PhysicsODE_Asymmetric, PhysicsODE_Hybrid as PhysicsODE_Hybrid_16
from visualize_custom import PhysicsODE_Hybrid as PhysicsODE_Hybrid_32, BlackBoxODE, processar_dataset, carregar_experimento

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_URL = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
test_files = [
    "multi-seno-1_0807_16-57.csv",
    "seq-degraus-1_0807_16-38.csv"
]
decimacao = 2
test_datasets = []
for f in test_files:
    t_raw, u_raw, y_raw = carregar_experimento(BASE_URL + f, decimacao=decimacao)
    t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
    test_datasets.append({
        'name': f, 't': t_ten, 'u': u_ten, 'x': x_ten
    })

models_dir = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\modelos_salvos'
files = os.listdir(models_dir)
pth_files = [f for f in files if f.endswith('.pth')]

constructors = [
    ("Baseline", PhysicsODE_Baseline),
    ("Coulomb", PhysicsODE_Coulomb),
    ("Tustin", PhysicsODE_Tustin),
    ("Asymmetric", PhysicsODE_Asymmetric),
    ("Hybrid_16", lambda: PhysicsODE_Hybrid_16(hidden_dim=16)),
    ("Hybrid_32", lambda: PhysicsODE_Hybrid_32(hidden_dim=32)),
    ("BlackBox", BlackBoxODE)
]

results = []

for pth in pth_files:
    filepath = os.path.join(models_dir, pth)
    # try with weights_only=True first, if fails then False
    try:
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
    except:
        state_dict = torch.load(filepath, map_location=device, weights_only=False)
        
    best_model_name = None
    best_rmse = float('inf')
    
    for name, ctor in constructors:
        try:
            model = ctor()
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            
            total_rmse = 0
            with torch.no_grad():
                for ds in test_datasets:
                    t_t, u_t, x_t = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)
                    y_real_deg = x_t[:, 0].cpu().numpy() * (180.0 / np.pi)
                    x0 = x_t[0].unsqueeze(0)
                    model.u_series = u_t
                    model.t_series = t_t
                    model.batch_start_times = torch.zeros(1, 1, device=device)
                    pred = odeint(model, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
                    pred_deg = pred[:, 0] * (180.0 / np.pi)
                    rmse = np.sqrt(mean_squared_error(y_real_deg, pred_deg))
                    total_rmse += rmse
            avg_rmse = total_rmse / len(test_datasets)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_model_name = name
                
        except Exception as e:
            pass
            
    if best_model_name:
        method = best_model_name
        if "Hybrid" in method: method = "Hybrid"
        results.append({"file": pth, "method": method, "rmse": best_rmse})
        print(f"File: {pth} | Method: {method} | RMSE: {best_rmse:.2f}")
    else:
        print(f"File: {pth} | Failed to load with any constructor")

df = pd.DataFrame(results)
if len(df) > 0:
    df.sort_values("rmse", inplace=True)
    print("\n--- Best models by method ---")
    best_by_method = df.groupby("method").first().reset_index()
    best_by_method.sort_values("rmse", inplace=True)
    print(best_by_method)
