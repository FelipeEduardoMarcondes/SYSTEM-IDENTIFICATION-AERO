import os
import glob
import pandas as pd
import numpy as np

dados_dir = r"c:\Users\vicio\Documents\AEROPENDULO\dados"
arquivos = sorted(glob.glob(os.path.join(dados_dir, "multisine_*.csv")))

print("Análise dos arquivos CSV (multisine_*.csv):")
print("-" * 80)

referencias = {}

for arq in arquivos:
    nome = os.path.basename(arq)
    try:
        df = pd.read_csv(arq, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        
        if "tempo_ms" not in df.columns:
            continue
            
        t = df["tempo_ms"].values
        ang = df["angulo_deg"].values
        ref = df["referencia"].values
        
        n_samples = len(df)
        dt = np.diff(t)
        
        if len(dt) > 0:
            dt_mean = np.mean(dt)
            dt_std = np.std(dt)
            dt_min = np.min(dt)
            dt_max = np.max(dt)
        else:
            dt_mean = dt_std = dt_min = dt_max = 0
            
        duracao = t[-1] if n_samples > 0 else 0
        
        # Guardar a referência para comparação
        referencias[nome] = ref
        
        print(f"Arquivo: {nome}")
        print(f"  - Amostras: {n_samples}")
        print(f"  - Duração: {duracao} ms ({duracao/1000:.2f} s)")
        print(f"  - Intervalo de tempo (dt): Média = {dt_mean:.2f} ms | Desvio Padrão = {dt_std:.2f} ms")
        print(f"  - dt Mínimo = {dt_min} ms | dt Máximo = {dt_max} ms")
        print("-" * 80)
        
    except Exception as e:
        print(f"Erro ao ler {nome}: {e}")

print("\nComparação das funções de Referência (são iguais?)")
print("-" * 80)
nomes = list(referencias.keys())
if len(nomes) >= 2:
    ref_base = referencias[nomes[0]]
    len_base = len(ref_base)
    for i in range(1, len(nomes)):
        ref_atual = referencias[nomes[i]]
        len_atual = len(ref_atual)
        
        min_len = min(len_base, len_atual)
        if min_len > 0:
            diff = np.abs(ref_base[:min_len] - ref_atual[:min_len])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"Comparando '{nomes[0]}' com '{nomes[i]}':")
            print(f"  - Tamanhos: {len_base} vs {len_atual}")
            print(f"  - Diferença Máxima: {max_diff:.4f}")
            print(f"  - Diferença Média: {mean_diff:.4f}")
            if max_diff < 1e-3:
                print("  => CONCLUSÃO: A função de referência é IGUAL (idêntica).")
            else:
                print("  => CONCLUSÃO: A função de referência é DIFERENTE.")
        print("-" * 80)
