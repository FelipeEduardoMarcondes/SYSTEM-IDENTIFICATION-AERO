import os
import glob
import pandas as pd
import numpy as np

dados_dir = r"c:\Users\vicio\Documents\AEROPENDULO\dados"
arquivos = sorted(glob.glob(os.path.join(dados_dir, "multisine_*.csv")))

for arq in arquivos:
    nome = os.path.basename(arq)
    try:
        df = pd.read_csv(arq, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        
        if "tempo_ms" not in df.columns:
            continue
            
        t = df["tempo_ms"].values / 1000.0
        ref = df["referencia"].values
        
        # Check if time resets
        resets = np.where(np.diff(t) < 0)[0]
        if len(resets) > 0:
            print(f"{nome}: TEMPO REINICIOU {len(resets)} vezes! (Ex: linha {resets[0]})")
            
        dt = np.diff(t)
        gaps = np.where(dt > 0.05)[0]  # Gaps maiores que 50ms
        
        if len(gaps) > 0:
            print(f"{nome}: Teve {len(gaps)} buracos de tempo (pacotes perdidos). Max gap: {np.max(dt):.2f}s")
        else:
            if len(resets) == 0:
                print(f"{nome}: PERFEITO! 0 buracos, 0 resets. Média dt = {np.mean(dt)*1000:.2f} ms")
            
    except Exception as e:
        print(f"Erro em {nome}: {e}")
