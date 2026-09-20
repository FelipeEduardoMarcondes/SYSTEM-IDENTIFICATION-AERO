import os
import glob
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, r"c:\Users\vicio\Documents\AEROPENDULO\python")
from signals import carregar_sinal_csv

dados_dir = r"c:\Users\vicio\Documents\AEROPENDULO\dados"

fontes = sorted(glob.glob(os.path.join(dados_dir, "multi-sine-*.csv")))
fontes.append(os.path.join(dados_dir, "dados_swept_sine.csv"))
sinais_fonte = {}

for f in fontes:
    nome = os.path.basename(f)
    if "_" in nome and "swept" not in nome:
        continue
    try:
        t, u = carregar_sinal_csv(f)
        sinais_fonte[nome] = u
    except Exception as e:
        print(f"Erro {nome}: {e}")

experimentos = sorted(glob.glob(os.path.join(dados_dir, "*_0731_*.csv")))

for exp in experimentos:
    nome_exp = os.path.basename(exp)
    try:
        df = pd.read_csv(exp, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        if "referencia" not in df.columns:
            continue
            
        ref_exp = df["referencia"].values
        melhor_fonte = None
        menor_erro = float('inf')
        
        for nome_fonte, u_fonte in sinais_fonte.items():
            tamanho = min(len(ref_exp), len(u_fonte), 5000)
            if tamanho == 0: continue
            
            erro_medio = np.mean(np.abs(ref_exp[:tamanho] - u_fonte[:tamanho]))
            if erro_medio < menor_erro:
                menor_erro = erro_medio
                melhor_fonte = nome_fonte
                
        if melhor_fonte:
            base_fonte = melhor_fonte.replace(".csv", "")
            if nome_exp.startswith(base_fonte):
                print(f"✅ {nome_exp:<30} -> CORRETO! ({melhor_fonte})")
            elif melhor_fonte == "dados_swept_sine.csv" and "17-47" in nome_exp:
                print(f"✅ {nome_exp:<30} -> CORRETO! ({melhor_fonte})")
            else:
                print(f"❌ {nome_exp:<30} -> ERRADO! (Veio de {melhor_fonte})")
        else:
            print(f"?? {nome_exp} -> Nenhuma fonte encontrada")
            
    except Exception as e:
        print(f"Erro EXP {nome_exp}: {e}")
