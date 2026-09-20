import os
import glob
import shutil

base_dir = r"c:\Users\vicio\Documents\AEROPENDULO"
dados_dir = os.path.join(base_dir, "dados")
controle_dir = os.path.join(base_dir, "controle")
exp_dir = os.path.join(base_dir, "experimentos")

os.makedirs(controle_dir, exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)

if os.path.exists(dados_dir):
    for f in os.listdir(dados_dir):
        path = os.path.join(dados_dir, f)
        if not os.path.isfile(path): continue
        
        # Se for PNG ou contiver timestamp "_0731_" é experimento
        if path.endswith(".png") or "_0731_" in f:
            shutil.move(path, os.path.join(exp_dir, f))
        else:
            # Arquivos base
            shutil.move(path, os.path.join(controle_dir, f))
            
    # Remove a pasta dados se estiver vazia
    if not os.listdir(dados_dir):
        os.rmdir(dados_dir)
        print("Pasta 'dados' deletada e arquivos reorganizados.")
    else:
        print("Arquivos movidos, mas 'dados' não está vazia.")
else:
    print("'dados' não existe mais.")
