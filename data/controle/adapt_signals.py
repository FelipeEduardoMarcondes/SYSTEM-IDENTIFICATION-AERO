import pandas as pd
import glob
import os

# Arquivos validos na pasta controle
files = [
    "sinal1_semi_estatica_malha_fechada.csv",
    "sinal3_swept_sine_malha_fechada.csv",
    "sysid_multiseno_validation_ref.csv"
]

for file in files:
    path = os.path.join(r"c:\Users\vicio\Documents\AEROPENDULO\controle", file)
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        continue
        
    try:
        # Tenta ler o CSV
        df = pd.read_csv(path)
        
        # Verifica se as colunas necessárias existem
        if 'Tempo_s' in df.columns and 'Saida_y' in df.columns:
            # Cria um novo DataFrame com apenas Tempo_s e Saida_y
            df_new = df[['Tempo_s', 'Saida_y']]
            
            # Nome do novo arquivo
            new_file_name = file.replace('.csv', '_ref_y.csv')
            new_path = os.path.join(r"c:\Users\vicio\Documents\AEROPENDULO\controle", new_file_name)
            
            # Salva sem o cabecalho ou com cabecalho? O carregar_sinal_csv lida com ambos.
            # Vamos salvar sem cabecalho para seguir o padrao do aprbs-60-1.csv
            df_new.to_csv(new_path, index=False, header=False)
            
            print(f"Sucesso! Gerado sinal adaptado: {new_file_name}")
        else:
            print(f"As colunas 'Tempo_s' e 'Saida_y' não foram encontradas em {file}")
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")

print("Note: sinal2_degraus_malha_fechada.csv e sinal4_multiseno_malha_fechada.csv contêm erros HTML (503) e precisam ser baixados novamente.")
