import json

filename = "c:\\\\Users\\\\vicio\\\\Documents\\\\SYSTEM-IDENTIFICATION-AERO-main\\\\identificação-collab\\\\ANN_NARX_AEROPENDULO.ipynb"

with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify imports to add StandardScaler
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'import pandas as pd' in source and 'StandardScaler' not in source:
            cell['source'].insert(9, "from sklearn.preprocessing import StandardScaler\n")

# Modify data prep to include normalization
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'carregar_experimento(BASE' in source:
            new_source = """def carregar_experimento(url, decimacao=1):
    df = pd.read_csv(url)
    df_sub = df.iloc[::decimacao].copy().reset_index(drop=True)
    u_raw = df_sub['motor_percent'].values
    y_raw = df_sub['angulo_deg'].values + 90.0
    u = u_raw - u_raw[0]
    y = y_raw  - y_raw[0]
    return u, y

BASE = (
    "https://raw.githubusercontent.com/FelipeEduardoMarcondes/"
    "SYSTEM-IDENTIFICATION-AERO/main/PYTHON/dados/"
)

ue, ye = carregar_experimento(BASE + "step_35_open.csv") # Estimação
ut, yt = carregar_experimento(BASE + "step_39_open.csv") # Teste (Validação)

# --- NORMALIZAÇÃO DOS DADOS ---
# Redes Neurais (especialmente com SELU) são altamente sensíveis à escala.
# O Free-Run diverge muito rápido se os dados não estiverem normalizados.
scaler_u = StandardScaler()
scaler_y = StandardScaler()

ue_norm = scaler_u.fit_transform(ue.reshape(-1, 1)).flatten()
ye_norm = scaler_y.fit_transform(ye.reshape(-1, 1)).flatten()

ut_norm = scaler_u.transform(ut.reshape(-1, 1)).flatten()
yt_norm = scaler_y.transform(yt.reshape(-1, 1)).flatten()

plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.plot(ue_norm, color='green')
plt.title('ue (Entrada Estimação - Norm)')
plt.grid()
plt.subplot(222)
plt.plot(ut_norm, color='orange')
plt.title('ut (Entrada Teste - Norm)')
plt.grid()
plt.subplot(223)
plt.plot(ye_norm, color='blue')
plt.title('ye (Saída Estimação - Norm)')
plt.grid()
plt.subplot(224)
plt.plot(yt_norm, color='red')
plt.title('yt (Saída Teste - Norm)')
plt.grid()
plt.tight_layout()
plt.show()"""
            lines = [line + '\\n' for line in new_source.split('\\n')]
            if lines: lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines

# Modify matReg calls to use normalized data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '(Ye, Phie) = matReg(ye, ue, ny, nu)' in source:
            new_source = source.replace('matReg(ye, ue, ny, nu)', 'matReg(ye_norm, ue_norm, ny, nu)')
            new_source = new_source.replace('matReg(yt, ut, ny, nu)', 'matReg(yt_norm, ut_norm, ny, nu)')
            lines = [line + '\\n' for line in new_source.split('\\n')]
            if lines: lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines

# Modify freeRun to use normalized data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'freeRun(model, yt, ut, ny, nu)' in source:
            new_source = source.replace('freeRun(model, yt, ut, ny, nu)', 'freeRun(model, yt_norm, ut_norm, ny, nu)')
            lines = [line + '\\n' for line in new_source.split('\\n')]
            if lines: lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
