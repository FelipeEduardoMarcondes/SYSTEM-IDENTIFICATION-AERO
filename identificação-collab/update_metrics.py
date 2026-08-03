import json
import re

nb_path = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\identificação-collab\NODE_AEROPENDULO.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Look for the conversion to degrees line to insert metrics
        if "y_test_deg_real = x_test[:, 0] * (180.0 / np.pi)" in source and "mse_phys" not in source:
            metrics_code = """
from sklearn.metrics import mean_squared_error

mse_phys = mean_squared_error(y_test_deg_real, phys_deg_pred)
rmse_phys = np.sqrt(mse_phys)

mse_bb = mean_squared_error(y_test_deg_real, bb_deg_pred)
rmse_bb = np.sqrt(mse_bb)

print(f"--- Métricas de Erro (Free-Run em Graus) ---")
print(f"Physics-Informed -> MSE: {mse_phys:.2f} | RMSE: {rmse_phys:.2f}º")
print(f"Black-Box        -> MSE: {mse_bb:.2f} | RMSE: {rmse_bb:.2f}º\\n")
"""
            
            source = source.replace(
                "bb_deg_pred = full_pred_bb[:, 0] * (180.0 / np.pi)\n",
                "bb_deg_pred = full_pred_bb[:, 0] * (180.0 / np.pi)\n" + metrics_code
            )
            
            cell['source'] = source.splitlines(True)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
