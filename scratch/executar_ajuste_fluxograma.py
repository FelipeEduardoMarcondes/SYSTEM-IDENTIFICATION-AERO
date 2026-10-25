import os
import shutil
import glob

BASE_DIR = r"c:\Users\vicio\Documents\AEROPENDULO"
FLUXO_DIR = os.path.join(BASE_DIR, "fluxograma-tcc")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def safe_copy(src, dst):
    if not os.path.exists(src):
        print(f"[AVISO] Arquivo de origem não encontrado: {src}")
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True

def safe_copy_dir(src, dst):
    if not os.path.exists(src):
        print(f"[AVISO] Diretório de origem não encontrado: {src}")
        return False
    ensure_dir(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    return True

def main():
    print("Iniciando organização da pasta fluxograma-tcc...")

    # Definição das pastas de destino
    d01 = os.path.join(FLUXO_DIR, "01_malha_aberta", "ensaios_estaticos")
    d02_degraus = os.path.join(FLUXO_DIR, "02_pid_preliminar", "ensaios_degraus")
    d02_scripts = os.path.join(FLUXO_DIR, "02_pid_preliminar", "scripts_sintonia")
    d03_ranges = os.path.join(FLUXO_DIR, "03_coleta_malha_fechada_rodada7", "definicao_ranges")
    d03_matlab = os.path.join(FLUXO_DIR, "03_coleta_malha_fechada_rodada7", "geradores_matlab")
    d03_refs = os.path.join(FLUXO_DIR, "03_coleta_malha_fechada_rodada7", "referencias_planejadas")
    d03_dados = os.path.join(FLUXO_DIR, "03_coleta_malha_fechada_rodada7", "dados_coletados_bancada")
    d04_narx = os.path.join(FLUXO_DIR, "04_modelagem_dinamica", "1_narx_polinomial")
    d04_grey = os.path.join(FLUXO_DIR, "04_modelagem_dinamica", "2_modelo_fisico_greybox")
    d05_scripts = os.path.join(FLUXO_DIR, "05_nmpc_casadi_e_dataset", "scripts_mpc")
    d05_inviab = os.path.join(FLUXO_DIR, "05_nmpc_casadi_e_dataset", "inviabilidade_tempo_real")
    d05_dataset = os.path.join(FLUXO_DIR, "05_nmpc_casadi_e_dataset", "dataset_gerado")
    d06_treino = os.path.join(FLUXO_DIR, "06_rede_neural_mlp", "treinamento")
    d06_pesos = os.path.join(FLUXO_DIR, "06_rede_neural_mlp", "pesos_e_modelos")
    d07_py = os.path.join(FLUXO_DIR, "07_validacao_em_simulacao", "1_python_sim")
    d07_sil_sem = os.path.join(FLUXO_DIR, "07_validacao_em_simulacao", "2_stm32_sil", "sem_integrador")
    d07_sil_com = os.path.join(FLUXO_DIR, "07_validacao_em_simulacao", "2_stm32_sil", "com_integrador")
    d07_sil_graf = os.path.join(FLUXO_DIR, "07_validacao_em_simulacao", "2_stm32_sil", "graficos_consolidados")
    d08_sem = os.path.join(FLUXO_DIR, "08_ensaios_bancada_fisica", "nominal_sem_integrador")
    d08_com = os.path.join(FLUXO_DIR, "08_ensaios_bancada_fisica", "hibrido_com_integrador")
    d08_pid = os.path.join(FLUXO_DIR, "08_ensaios_bancada_fisica", "comparacao_com_pid_rodada7")
    d09_opt = os.path.join(FLUXO_DIR, "09_otimizacoes_tinyml")
    d10_triade = os.path.join(FLUXO_DIR, "10_comparativos_finais", "triade_sim_to_real")
    d10_hib = os.path.join(FLUXO_DIR, "10_comparativos_finais", "nominal_vs_hibrido")
    d10_pid = os.path.join(FLUXO_DIR, "10_comparativos_finais", "ann_vs_pid_rodada7")
    d10_scripts = os.path.join(FLUXO_DIR, "10_comparativos_finais", "scripts_geracao_comparativos")

    # 1. Malha Aberta
    src_ma = os.path.join(FLUXO_DIR, "coleta-malha-aberta", "segunda-coleta")
    if os.path.exists(src_ma):
        safe_copy_dir(src_ma, d01)

    # 2. PID Preliminar
    src_degraus = os.path.join(FLUXO_DIR, "coleta-malha-aberta", "primeira-coleta")
    if os.path.exists(src_degraus):
        safe_copy_dir(src_degraus, d02_degraus)
    # Gráficos de degrau 30, 40, 60
    for p in ["pid-30.png", "pid-40.png", "pid-60.png"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", p), os.path.join(d02_degraus, p))

    src_pid_scripts = os.path.join(FLUXO_DIR, "modelagem-pid")
    if os.path.exists(src_pid_scripts):
        for f in ["analise.m", "testenovo.m", "tf1.mat"]:
            safe_copy(os.path.join(src_pid_scripts, f), os.path.join(d02_scripts, f))

    # 3. Coleta Malha Fechada Rodada 7
    safe_copy(os.path.join(FLUXO_DIR, "coleta-malha-fechada", "definicao-ranges-angulos.png"),
              os.path.join(d03_ranges, "definicao-ranges-angulos.png"))
    safe_copy_dir(os.path.join(FLUXO_DIR, "coleta-malha-fechada", "gerador-matlab"), d03_matlab)
    safe_copy_dir(os.path.join(FLUXO_DIR, "coleta-malha-fechada", "CONTROLE-RODADA-7"), d03_refs)
    safe_copy_dir(os.path.join(FLUXO_DIR, "coleta-malha-fechada", "RODADA-7"), d03_dados)

    # 4. Modelagem Dinâmica
    src_narx = os.path.join(FLUXO_DIR, "modelagem-narx", "narx")
    if os.path.exists(src_narx):
        safe_copy_dir(src_narx, d04_narx)
    for f in ["narx_freerun.png", "narx_freerun.pdf", "casadi_narx_freerun.png", "casadi_narx_freerun.pdf", 
              "narx_train_datasets.png", "narx_train_datasets.pdf", "narx_val_consolidado.png", "narx_val_consolidado.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d04_narx, f))

    # Greybox Físico
    safe_copy(os.path.join(BASE_DIR, "sysid", "scripts", "node_v5_multiple_shooting.py"),
              os.path.join(d04_grey, "node_v5_multiple_shooting.py"))
    safe_copy(os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP005_greybox_v5_mpc", "1_identificacao", "parametros_greybox_v5.json"),
              os.path.join(d04_grey, "parametros_greybox_v5.json"))
    safe_copy(os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP005_greybox_v5_mpc", "2_mpc_python", "casadi_greybox_v5_freerun_multisine.png"),
              os.path.join(d04_grey, "casadi_greybox_v5_freerun_multisine.png"))
    for f in ["identificacao_atrito.py", "identificacao_queda_livre.py", "identificacao_estatica.py"]:
        safe_copy(os.path.join(BASE_DIR, "sysid", "scripts", f), os.path.join(d04_grey, f))

    # 5. NMPC CasADi e Dataset
    safe_copy(os.path.join(FLUXO_DIR, "modelagem-narx", "mpc-narx", "mpc_v4.py"),
              os.path.join(d05_scripts, "mpc_v4.py"))
    for f in ["run_mpc_narx.py", "run_mpc_node.py"]:
        safe_copy(os.path.join(BASE_DIR, "sysid", "scripts", "mpc-control", f), os.path.join(d05_scripts, f))
    for f in ["mpc_tempo.png", "mpc_tempo.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d05_inviab, f))
    for f in ["mpc_treino.png", "mpc_treino.pdf", "mpc_referencia_treino.png", "mpc_referencia_treino.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d05_dataset, f))
    safe_copy(os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP001_reteste_narx-mpc", "2_mpc_python", "simulacao_mpc_treino.csv"),
              os.path.join(d05_dataset, "simulacao_mpc_treino.csv"))
    safe_copy(os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP002_legado_narx-mpc", "2_mpc_python", "dataset_mpc_treino.npz"),
              os.path.join(d05_dataset, "dataset_mpc_treino.npz"))

    # 6. Rede Neural MLP
    for f in ["ann_loss.png", "ann_loss.pdf", "ann_r2.png", "ann_r2.pdf", "ann_regression_fit.png", "ann_regression_fit.pdf",
              "ann_loss_evolution.png", "ann_loss_evolution.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d06_treino, f))
    safe_copy(os.path.join(BASE_DIR, "sysid", "notebooks", "identificaçao-collab", "IDENT-FELIPE", "ANN_NARX_AEROPENDULO_v5.ipynb"),
              os.path.join(d06_treino, "ANN_NARX_AEROPENDULO_v5.ipynb"))
    # Pesos
    safe_copy(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "sil", "com-integrador", "ann_weights.h"),
              os.path.join(d06_pesos, "ann_weights.h"))
    # ONNX
    src_opt_dir = None
    for cand in os.listdir(FLUXO_DIR):
        if "testes-otimiza" in cand:
            src_opt_dir = os.path.join(FLUXO_DIR, cand)
            break
    if src_opt_dir:
        safe_copy(os.path.join(src_opt_dir, "ann_model.onnx"), os.path.join(d06_pesos, "ann_model.onnx"))
    safe_copy(os.path.join(BASE_DIR, "sysid", "pesos", "ann_mpc.pth"), os.path.join(d06_pesos, "ann_mpc.pth"))

    # 7. Validação em Simulação
    src_py = os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP001_reteste_narx-mpc", "2_mpc_python")
    for f in ["simulacao_python.csv", "simulacao_python_val2_degraus.csv", "simulacao_python_val3_senoidal.csv",
              "py_val1.png", "py_val1.pdf", "py_val2.png", "py_val2.pdf", "py_val3.png", "py_val3.pdf",
              "val2_degraus_plot.png", "val3_senoidal_plot.png"]:
        safe_copy(os.path.join(src_py, f), os.path.join(d07_py, f))
    
    # SIL
    safe_copy_dir(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "sil", "sem-integrador"), d07_sil_sem)
    safe_copy_dir(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "sil", "com-integrador"), d07_sil_com)
    for f in ["sil_validacao_completa.png", "sil_validacao_completa.pdf", "sil_xcube_tracking_comparacao.png", "sil_xcube_tracking_comparacao.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d07_sil_graf, f))

    # 8. Ensaios Bancada Física
    safe_copy_dir(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "aeropendulo", "sem-integrador"), d08_sem)
    safe_copy_dir(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "aeropendulo", "com-integrador"), d08_com)
    safe_copy_dir(os.path.join(FLUXO_DIR, "modelagem-narx", "testes", "aeopendulo-comp-pid"), d08_pid)

    # 9. Otimizações TinyML
    if src_opt_dir:
        safe_copy_dir(src_opt_dir, d09_opt)
    safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", "benchmark_inferencia_flash.png"),
              os.path.join(d09_opt, "benchmark_inferencia_flash.png"))
    safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", "benchmark_inferencia_flash.pdf"),
              os.path.join(d09_opt, "benchmark_inferencia_flash.pdf"))

    # 10. Comparativos Finais
    for f in ["comparacao_plot_0.png", "comparacao_plot_0.pdf", "comparacao_plot_1.png", "comparacao_plot_1.pdf", "comparacao_plot_2.png", "comparacao_plot_2.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d10_triade, f))
    
    for f in ["comparacao_hibrido_val1.png", "comparacao_hibrido_val1.pdf", "comparacao_hibrido_val2.png", "comparacao_hibrido_val2.pdf",
              "comparacao_hibrido_val3.png", "comparacao_hibrido_val3.pdf", "comparacao_bancada_sem_vs_com_val1.png", "comparacao_bancada_sem_vs_com_val1.pdf",
              "comparacao_bancada_sem_vs_com_val2.png", "comparacao_bancada_sem_vs_com_val2.pdf", "comparacao_bancada_sem_vs_com_val3.png", "comparacao_bancada_sem_vs_com_val3.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d10_hib, f))

    for f in ["comparacao_rodada7_multiseno45.png", "comparacao_rodada7_multiseno45.pdf",
              "comparacao_rodada7_multiseno45_completo.png", "comparacao_rodada7_multiseno45_completo.pdf",
              "comparacao_rodada7_chirp60.png", "comparacao_rodada7_chirp60.pdf"]:
        safe_copy(os.path.join(BASE_DIR, "tcc", "img", "resultados", f), os.path.join(d10_pid, f))

    src_scripts_comp = os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP001_reteste_narx-mpc")
    for f in ["gerar_comparacao.py", "gerar_plots_academicos_hibrido.py", "gerar_plots_rodada7_vs_ann.py", "gerar_todos_plots_academicos.py"]:
        safe_copy(os.path.join(src_scripts_comp, f), os.path.join(d10_scripts, f))

    # Criação do README de documentação do diretório
    readme_path = os.path.join(FLUXO_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""# Fluxograma e Resultados do TCC — Aeropêndulo

Este diretório consolida de ponta a ponta todos os dados, códigos, modelos e resultados experimentais utilizados no Trabalho de Conclusão de Curso:
**"NMPC APROXIMADO POR REDES NEURAIS EM HARDWARE EMBARCADO: VALIDAÇÃO EXPERIMENTAL"**
Autor: Felipe Eduardo Marcondes | Orientador: Prof. Dr. Helon Vicente Hultmann Ayala

---

## Estrutura dos Diretórios

### `01_malha_aberta/`
- **`ensaios_estaticos/`**: Ensaios experimentais de sustentação e resposta estática de empuxo ($PWM \times \\text{ângulo}$) na bancada para caracterização das saturações físicas e faixas de equilíbrio.

### `02_pid_preliminar/`
- **`ensaios_degraus/`**: Dados experimentais da resposta ao degrau estabilizados pelo PID preliminar no Arduino em $30^\\circ$, $40^\\circ$ e $60^\\circ$ (`pi-30`, `pi-40`, `pi-60`).
- **`scripts_sintonia/`**: Códigos MATLAB (`analise.m`, `testenovo.m`, `tf1.mat`) para identificação da função de transferência linear em torno de $40^\\circ$ e sintonia dos ganhos $K_p, K_i, K_d$.

### `03_coleta_malha_fechada_rodada7/`
- **`definicao_ranges/`**: Gráfico esquemático de definição dos patamares de operação ($45^\\circ$ e $60^\\circ$).
- **`geradores_matlab/`**: Scripts de síntese dos sinais com riqueza espectral ($f_{\\max} = 0{,}50$\\,Hz): multisseno, swept sine (chirp) e sequências de degraus (APRBS).
- **`referencias_planejadas/`**: Arquivos CSV das trajetórias teóricas de referência geradas no MATLAB.
- **`dados_coletados_bancada/`**: Telemetria serial completa a 100 Hz ($T_s = 10$\\,ms) coletada com a bancada estabilizada por PID durante a Rodada 7 (CSVs e gráficos PNG).

### `04_modelagem_dinamica/`
- **`1_narx_polinomial/`**: Algoritmo FROLS/ERR para seleção parcimoniosa de 10 regressores, equações polinomiais em LaTeX, métricas de validação e simulação em regime livre (*free-run* com $\\text{RMSE} = 4{,}685^\\circ$).
- **`2_modelo_fisico_greybox/`**: Modelo fenomenológico não linear (ODE) parametrizado experimentalmente via *multiple shooting* com CasADi/JAX (`node_v5_multiple_shooting.py` e ensaios de atrito/queda livre).

### `05_nmpc_casadi_e_dataset/`
- **`scripts_mpc/`**: Formulação simbólica do Problema de Controle Ótimo (OCP) discretizado no CasADi com resolvedor não linear IPOPT.
- **`inviabilidade_tempo_real/`**: Registro quantitativo do tempo de resolução por passo (90–120 ms no computador de mesa), demonstrando a inviabilidade de execução online direta no microcontrolador STM32 ($T_s = 10$\\,ms).
- **`dataset_gerado/`**: Rastreamento ótimo da referência multivariada de 187 s e banco de dados gerado com pares de estados e ações ótimas de controle $u^*(k)$.

### `06_rede_neural_mlp/`
- **`treinamento/`**: Notebook PyTorch e curvas de treinamento/validação com convergência em menos de 5 épocas, sem *overfitting* e coeficiente de determinação $R^2 = 0{,}999$.
- **`pesos_e_modelos/`**: Checkpoint PyTorch (`ann_mpc.pth`), modelo exportado em ONNX (`ann_model.onnx`) e pesos exportados em cabeçalho C (`ann_weights.h`).

### `07_validacao_em_simulacao/`
- **`1_python_sim/`**: Simulação completa em malha fechada do controlador neural no ambiente Python (validações 1, 2 e 3).
- **`2_stm32_sil/`**: Testes em *Software-in-the-Loop* (SIL) no microcontrolador STM32 Nucleo-F446RE sem e com integrador residual, registrando perfis temporais de inferência e fidelidade numérica em relação ao Python.

### `08_ensaios_bancada_fisica/`
- **`nominal_sem_integrador/`**: Validação experimental da política neural pura na bancada física, evidenciando o *Sim-to-Real gap* (erro estático de regime decorrente de atritos não modelados e flutuação de bateria).
- **`hibrido_com_integrador/`**: Controlador neural híbrido com compensador integrador residual, eliminando o erro de regime permanente.
- **`comparacao_com_pid_rodada7/`**: Ensaios espelhados da Rodada 7 executados na bancada com a rede neural para comparação direta de rastreamento.

### `09_otimizacoes_tinyml/`
- Benchmarks de latência e consumo de memória (Flash e SRAM) variando flags de compilação GCC (`-O0`, `-O1`, `-O2`, `-O3`, `-Ofast`) e compilação otimizada via X-CUBE-AI, demonstrando tempo de inferência de $\\approx 2{,}08$\\,ms (folga de CPU $\\approx 79\\%$).

### `10_comparativos_finais/`
- **`triade_sim_to_real/`**: Comparação direta das trajetórias Python vs. SIL vs. Bancada Física.
- **`nominal_vs_hibrido/`**: Impacto do compensador integrador residual no fechamento do *gap* experimental.
- **`ann_vs_pid_rodada7/`**: Comparação de atraso de fase, antecipação dinâmica e erro RMSE entre o controle preditivo neural aproximado e a malha clássica PID.
- **`scripts_geracao_comparativos/`**: Scripts Python para geração dos gráficos acadêmicos em formato PNG e PDF.
""")

    print("Cópia e estruturação finalizadas com sucesso!")

if __name__ == "__main__":
    main()
