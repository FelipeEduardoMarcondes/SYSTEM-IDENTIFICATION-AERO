import pandas as pd
import numpy as np

hw_file = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos\referencia_mpc_0921_00-53.csv"
py_file = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos\sil\simulacao_python.csv"

df_hw = pd.read_csv(hw_file)
df_py = pd.read_csv(py_file)

min_len = min(len(df_hw), len(df_py))

ang_hw = df_hw['angulo_deg'].values[:min_len]
ang_py = df_py['angulo_deg'].values[:min_len]

u_hw = df_hw['u_pct'].values[:min_len]
u_py = df_py['u_pct'].values[:min_len]

t_ms = df_hw['tempo_ms'].values[:min_len]
mask_ann = (t_ms >= 5000) & (t_ms < 65000)

ang_hw_ann = ang_hw[mask_ann]
ang_py_ann = ang_py[mask_ann]
u_hw_ann = u_hw[mask_ann]
u_py_ann = u_py[mask_ann]

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))
def max_err(a, b):
    return np.max(np.abs(a - b))

print("=== GERAL (Toda a simulacao) ===")
print(f"Angulo RMSE: {rmse(ang_hw, ang_py):.4f} graus")
print(f"Angulo Max Error: {max_err(ang_hw, ang_py):.4f} graus")
print(f"Controle RMSE: {rmse(u_hw, u_py):.4f} %")
print(f"Controle Max Error: {max_err(u_hw, u_py):.4f} %")

print("\n=== APENAS REGIAO DA ANN (5s a 65s) ===")
print(f"Angulo RMSE: {rmse(ang_hw_ann, ang_py_ann):.4f} graus")
print(f"Angulo Max Error: {max_err(ang_hw_ann, ang_py_ann):.4f} graus")
print(f"Controle RMSE: {rmse(u_hw_ann, u_py_ann):.4f} %")
print(f"Controle Max Error: {max_err(u_hw_ann, u_py_ann):.4f} %")
