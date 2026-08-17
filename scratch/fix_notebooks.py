import json
import os

def fix_rls():
    file_path = 'identificação-collab/RLS_AEROPENDULO.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Create the new cell to append
    new_cell_source = """# ==========================================
# 3. VALIDAÇÃO NA BASE DE TESTE
# ==========================================

N_test = len(u_TEST)
y_est_rls_test = np.zeros(N_test)
y_est_rls_test[:p] = y_TEST[:p]

for k in range(p, N_test):
    phi_k = np.zeros((num_params, 1))
    for i in range(1, na + 1):
        phi_k[i - 1, 0] = -y_TEST[k - i]
    for i in range(1, nb + 1):
        phi_k[na + i - 1, 0] = u_TEST[k - i]
        
    y_hat = (phi_k.T @ theta_rls).item()
    y_est_rls_test[k] = y_hat

t_test = np.arange(p, N_test)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_test, y_TEST[p:], color='black', lw=2, label='Saída Medida (Real)')
ax.plot(t_test, y_est_rls_test[p:], color='blue', ls='-.', lw=2, alpha=0.8, label='Predição RLS (OSA)')

ax.set_title('Saída Real vs. Estimada (Validação / Teste)')
ax.set_xlabel('Amostras (k)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
"""
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(new_cell_source.split('\n')) - 1 else line for i, line in enumerate(new_cell_source.split('\n'))]
    }
    
    # We append it at the end of the notebook.
    nb['cells'].append(new_cell)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("RLS_AEROPENDULO.ipynb fixed.")

def fix_bls():
    file_path = 'identificação-collab/BLS_AEROPENDULO.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Fix 1: 404 URL in Cell 3
            if '"multi-seno-2_0807_17-13.csv"' in source:
                new_source = source.replace(
                    '"multi-seno-2_0807_17-13.csv"',
                    '"RODADA-2/multi-seno-2_0804_19-31.csv"'
                )
                cell['source'] = [line + '\n' if i < len(new_source.split('\n')) - 1 else line for i, line in enumerate(new_source.split('\n'))]
                
            # Fix 2: Grid Search limits in Cell 19
            if 'ordens_teste = [(' in source and '(30, 30)' in source:
                new_source = source.replace(
                    'ordens_teste = [(1,1),(1,2),(2,1),(2,2),(3,2),(2,3),(3,3),(4,3),(4,4),(8, 24), (30, 30), (24, 8)]',
                    'ordens_teste = [(1,1),(1,2),(2,1),(2,2),(3,2),(2,3),(3,3),(4,3),(4,4),(8, 24), (24, 8), (30, 30), (30, 50), (40, 40), (50, 30), (50, 50), (60, 60)]'
                )
                cell['source'] = [line + '\n' if i < len(new_source.split('\n')) - 1 else line for i, line in enumerate(new_source.split('\n'))]
                
            # Fix 3: Error shapes in Cell 23
            if 'erro_FR_best = y_tst_b.flatten()' in source:
                target_str = """# Sinais de erro (Validação) para os melhores e piores
erro_FR_best = y_tst_b.flatten() - fr_tst_b.flatten()
erro_FR_worst = y_tst_w.flatten() - fr_tst_w.flatten()"""
                replacement_str = """# Alinhamento das matrizes
p_max = max(p_w, p_b)

y_tst_b_aligned = y_tst_b[p_max - p_b:]
fr_tst_b_aligned = fr_tst_b[p_max - p_b:]
y_tst_w_aligned = y_tst_w[p_max - p_w:]
fr_tst_w_aligned = fr_tst_w[p_max - p_w:]

# Sinais de erro (Validação) para os melhores e piores
erro_FR_best = y_tst_b_aligned.flatten() - fr_tst_b_aligned.flatten()
erro_FR_worst = y_tst_w_aligned.flatten() - fr_tst_w_aligned.flatten()"""
                new_source = source.replace(target_str, replacement_str)
                cell['source'] = [line + '\n' if i < len(new_source.split('\n')) - 1 else line for i, line in enumerate(new_source.split('\n'))]
                
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("BLS_AEROPENDULO.ipynb fixed.")

if __name__ == '__main__':
    fix_rls()
    fix_bls()
