import sys
file_path = r'c:\Users\vicio\Documents\AEROPENDULO\interface\main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'csvs = sorted(glob.glob("*.csv") + glob.glob(f"{CONTROLE_DIR}/*.csv"))',
    'csvs = sorted(glob.glob("*.csv") + glob.glob(f"{CONTROLE_DIR}/*.csv") + glob.glob(f"{SINAIS_REF_DIR}/*.csv"))'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed glob in main.py')
