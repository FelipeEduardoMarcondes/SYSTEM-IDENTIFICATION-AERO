import json

filename = "BLS_AEROPENDULO_corrigido (2).ipynb"
with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def free_run_simulation' in source:
            new_source = """def create_regression_matrix(u, y, na, nb):
    \"\"\"
    Constrói a matriz de regressão Phi e o vetor alvo y_target
    para um modelo ARX de ordens na (saída) e nb (entrada).

    Linha de Phi em k: [-y(k-1), ..., -y(k-na), u(k-1), ..., u(k-nb)]
    \"\"\"
    u = np.asarray(u).flatten()
    y = np.asarray(y).flatten()
    N = len(y)
    if N != len(u):
        raise ValueError("Dimensões de u e y inconsistentes!")

    p = max(na, nb)
    y_target = y[p:]
    Phi = np.zeros((N - p, na + nb))

    for i in range(1, na + 1):
        Phi[:, i - 1] = -y[p - i: N - i]
    for i in range(1, nb + 1):
        Phi[:, na + i - 1] = u[p - i: N - i]

    return Phi, y_target


def free_run_simulation(th_hat, u, y_measured, na, nb):
    \"\"\"
    Simulação Free-Run: usa apenas saídas simuladas anteriores
    (sem realimentação das saídas medidas após as condições iniciais).
    \"\"\"
    u = np.asarray(u).flatten()
    y_measured = np.asarray(y_measured).flatten()
    N = len(u)
    p = max(na, nb)
    y_sim = np.zeros(N)
    y_sim[:p] = y_measured[:p]   # condições iniciais = dados medidos

    for k in range(p, N):
        # Utilizando a função do passo 1 (create_regression_matrix) para construir o regressor.
        # Passamos uma janela de p+1 amostras (do instante k-p até k)
        u_window = u[k-p : k+1]
        y_window = y_sim[k-p : k+1]
        
        # A função retorna a matriz de regressão (Phi) com 1 linha e o vetor y_target.
        Phi_k_matrix, _ = create_regression_matrix(u_window, y_window, na, nb)
        
        # Faz a predição para o instante k
        y_sim[k] = Phi_k_matrix[0] @ th_hat

    return y_sim[p:]


def calcular_metricas(y_real, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
    print(f"  {label:30s} → RMSE: {rmse:.6e}  |  R²: {r2:.6f}")
    return rmse, r2


def espectro_amplitude(sinal, Ts):
    N   = len(sinal)
    xf  = scipy.fft.fftfreq(N, Ts)[:N // 2]
    amp = 2.0 / N * np.abs(scipy.fft.fft(sinal)[:N // 2])
    return xf, amp"""
            lines = [line + '\\n' for line in new_source.split('\\n')]
            if lines:
                lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines


new_cell_source = """# Comparação do Espectro entre o Melhor e o Pior Modelo (Validação)
worst = df_res.loc[df_res['R2_FR_TST'].idxmin()]
print(f"Pior configuração (R² Free-Run na Validação): na={int(worst.na)}, nb={int(worst.nb)}")
print(f"Melhor configuração (R² Free-Run na Validação): na={int(best.na)}, nb={int(best.nb)}")

na_w, nb_w = int(worst.na), int(worst.nb)
p_w = max(na_w, nb_w)

# Estimar e simular o pior modelo
Phi_tra_w, y_tra_w = create_regression_matrix(u_TRA,  y_TRA,  na_w, nb_w)
Phi_tst_w, y_tst_w = create_regression_matrix(u_TEST, y_TEST, na_w, nb_w)
th_w, _, _, _ = np.linalg.lstsq(Phi_tra_w, y_tra_w, rcond=None)
fr_tst_w = free_run_simulation(th_w, u_TEST, y_TEST, na_w, nb_w)

# Sinais de erro (Validação) para os melhores e piores
erro_FR_best = y_tst_b.flatten() - fr_tst_b.flatten()
erro_FR_worst = y_tst_w.flatten() - fr_tst_w.flatten()

# Calcular espectros
_, amp_erro_best = espectro_amplitude(erro_FR_best, Ts)
xf, amp_erro_worst = espectro_amplitude(erro_FR_worst, Ts)

fig, axes = plt.subplots(1, 1, figsize=(8, 5))
axes.plot(xf, amp_erro_worst, label=f'Erro Free-Run (Pior: {na_w},{nb_w})', color='purple', lw=1.5)
axes.plot(xf, amp_erro_best, label=f'Erro Free-Run (Melhor: {na_b},{nb_b})', color='green', lw=1.5, ls='--')
axes.set_title('Comparação dos Espectros de Erro (Melhor vs Pior Modelo)')
axes.set_xlabel('Frequência (Hz)')
axes.set_ylabel('Amplitude')
axes.legend()
axes.grid(True)
plt.show()"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\\n' for line in new_cell_source.split('\\n')]
}
if new_cell['source']:
    new_cell['source'][-1] = new_cell['source'][-1].rstrip('\\n')

# Append only if it's not already there
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Comparação do Espectro entre o Melhor e o Pior Modelo' in ''.join(cell['source']):
        found = True
        break
        
if not found:
    nb['cells'].append(new_cell)

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
