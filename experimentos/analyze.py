import os
import glob
import pandas as pd

print('| Arquivo | Data/Hora | Sinal Motor Max (%) | Sinal Motor Min (%) | Ângulo Max (deg) | Ângulo Min (deg) | Ganho Estimado (deg/%) |')
print('|---|---|---|---|---|---|---|')

for f in sorted(glob.glob('*.csv')):
    try:
        df = pd.read_csv(f)
    except:
        continue
        
    u_col = 'u_pct' if 'u_pct' in df.columns else ('motor_percent' if 'motor_percent' in df.columns else None)
    if not u_col:
        continue
        
    u_max = df[u_col].max()
    u_min = df[u_col].min()
    y_max = df['angulo_deg'].max()
    y_min = df['angulo_deg'].min()
    
    # Estimate gain (amplitude of Y / amplitude of U)
    amp_u = u_max - u_min
    gain = (y_max - y_min) / amp_u if amp_u != 0 else 0
    
    time_str = f.split('_')[-1].replace('.csv', '') if '_' in f else 'N/A'
    
    print(f"| {f} | {time_str} | {u_max:.1f} | {u_min:.1f} | {y_max:.1f} | {y_min:.1f} | {gain:.2f} |")
