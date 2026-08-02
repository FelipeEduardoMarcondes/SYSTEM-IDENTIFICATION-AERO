import json

nb_path = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\identificação-collab\ANN_NARX_AEROPENDULO.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_code = """# --- NORMALIZAÇÃO DOS DADOS ---
# Redes Neurais (especialmente com SELU) são altamente sensíveis à escala.
scaler_u = StandardScaler()
scaler_y = StandardScaler()
ue_norm = scaler_u.fit_transform(ue.reshape(-1, 1)).flatten()
ye_norm = scaler_y.fit_transform(ye.reshape(-1, 1)).flatten()"""

new_code = """# --- NORMALIZAÇÃO DOS DADOS ---
# Redes Neurais (especialmente com SELU) são altamente sensíveis à escala.
scaler_u = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

# Define os limites reais (físicos) da bancada
limites_motor = np.array([[0], [100]])
# Como y_raw recebe angulo_deg + 90.0 na leitura, aplicamos o mesmo aos limites (-90 a 180)
limites_angulo = np.array([[-90], [180]]) + 90.0 

# "Engana" o scaler para que ele use as fronteiras físicas da bancada
scaler_u.fit(limites_motor)
scaler_y.fit(limites_angulo)

# Transforma usando o scaler treinado com os limites físicos
ue_norm = scaler_u.transform(ue.reshape(-1, 1)).flatten()
ye_norm = scaler_y.transform(ye.reshape(-1, 1)).flatten()"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if old_code in source:
            source = source.replace(old_code, new_code)
            # Reconstruct cell['source'] as a list of lines keeping newline characters
            lines = source.splitlines(True)
            cell['source'] = lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
