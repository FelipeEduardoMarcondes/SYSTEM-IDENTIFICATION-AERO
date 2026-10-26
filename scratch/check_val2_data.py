import os
import pandas as pd
import numpy as np

# Let's inspect the exact data in comparacao_plot_1
# In comparacao_plot_1:
# Blue is SIL (20261023_214546_referencia_legado_val2_degraus_hybrid.csv)
# Red is Bancada Real (20261023_180711_referencia_legado_val2_degraus_hybrid.csv)
# Let's check if there are 2 curves or 3 curves
p_sil = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\3_sil_stm32\20261023_214546_referencia_legado_val2_degraus_hybrid.csv'
p_real = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo\20261023_180711_referencia_legado_val2_degraus_hybrid.csv'

df_sil = pd.read_csv(p_sil)
df_real = pd.read_csv(p_real)
print("SIL len:", len(df_sil), "Real len:", len(df_real))
print("Max time SIL:", df_sil['tempo_ms'].iloc[-1]/1000, "Real:", df_real['tempo_ms'].iloc[-1]/1000)
