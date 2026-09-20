import os
import glob
import pandas as pd
import numpy as np

pasta = "experimentos"
arquivos = glob.glob(os.path.join(pasta, "*.csv"))
arquivos = [f for f in arquivos if "_0904_" in f]

print(f"{'Arquivo':<35} | {'Dur(s)':<7} | {'RMSE':<6} | {'Sat(%)':<7} | {'Ang Max':<7} | {'Status'}")
print("-" * 85)

for arq in sorted(arquivos):
    try:
        df = pd.read_csv(arq)
        if "tempo_ms" in df.columns:
            t = df["tempo_ms"].values / 1000.0
            ang = df["angulo_deg"].values
            ref = df["referencia"].values
            u = df["u_pct"].values
            
            duracao = t[-1] if len(t) > 0 else 0
            rmse = np.sqrt(np.mean((ang - ref)**2)) if len(ang) > 0 else 0
            
            saturado = np.sum((u >= 99.0) | (u <= -84.0)) / len(u) * 100 if len(u) > 0 else 0
            ang_max = np.max(ang) if len(ang) > 0 else 0
            
            if saturado > 20:
                status = "Saturado"
            elif rmse > 20:
                status = "Alto RMSE"
            else:
                status = "OK"
                
            nome = os.path.basename(arq).replace(".csv", "")
            if len(nome) > 33:
                nome = nome[:30] + "..."
                
            print(f"{nome:<35} | {duracao:>7.1f} | {rmse:>6.2f} | {saturado:>6.1f}% | {ang_max:>7.1f} | {status}")
    except Exception as e:
        print(f"Erro ao ler {arq}: {e}")
