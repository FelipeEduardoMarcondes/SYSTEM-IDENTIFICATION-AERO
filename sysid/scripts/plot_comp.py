import pandas as pd
import matplotlib.pyplot as plt

df_py = pd.read_csv('simulacao_python.csv')
df_stm = pd.read_csv('experimentos/referencia_mpc_0920_19-19.csv')

plt.figure(figsize=(14, 6))
plt.plot(df_py['tempo_ms']/1000.0, df_py['angulo_deg'], 'b-', lw=4, alpha=0.5, label='Inferencia Python/PyTorch')
plt.plot(df_stm['tempo_ms']/1000.0, df_stm['angulo_deg'], 'r--', lw=2, label='Inferencia STM32 (C)')
plt.plot(df_py['tempo_ms']/1000.0, df_py['referencia'], 'k:', label='Referencia')
plt.title('Comparacao Direta: Python SIL vs STM32 SIL a 100 Hz (PID -> ANN -> PID)')
plt.xlabel('Tempo [s]')
plt.ylabel('Angulo [deg]')
plt.legend()
plt.grid(True)
plt.savefig('comparacao_python_stm.png')
print('Gráfico de comparação gerado em comparacao_python_stm.png')
