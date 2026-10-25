# Fluxograma e Resultados do TCC — Aeropêndulo

Este diretório consolida de ponta a ponta todos os dados, códigos, modelos e resultados experimentais utilizados no Trabalho de Conclusão de Curso:
**"NMPC APROXIMADO POR REDES NEURAIS EM HARDWARE EMBARCADO: VALIDAÇÃO EXPERIMENTAL"**
Autor: Felipe Eduardo Marcondes | Orientador: Prof. Dr. Helon Vicente Hultmann Ayala

---

## Estrutura dos Diretórios

### `01_malha_aberta/`
- **`ensaios_estaticos/`**: Ensaios experimentais de sustentação e resposta estática de empuxo ($PWM 	imes \text{ângulo}$) na bancada para caracterização das saturações físicas e faixas de equilíbrio.

### `02_pid_preliminar/`
- **`ensaios_degraus/`**: Dados experimentais da resposta ao degrau estabilizados pelo PID preliminar no Arduino em $30^\circ$, $40^\circ$ e $60^\circ$ (`pi-30`, `pi-40`, `pi-60`).
- **`scripts_sintonia/`**: Códigos MATLAB (`analise.m`, `testenovo.m`, `tf1.mat`) para identificação da função de transferência linear em torno de $40^\circ$ e sintonia dos ganhos $K_p, K_i, K_d$.

### `03_coleta_malha_fechada_rodada7/`
- **`definicao_ranges/`**: Gráfico esquemático de definição dos patamares de operação ($45^\circ$ e $60^\circ$).
- **`geradores_matlab/`**: Scripts de síntese dos sinais com riqueza espectral ($f_{\max} = 0{,}50$\,Hz): multisseno, swept sine (chirp) e sequências de degraus (APRBS).
- **`referencias_planejadas/`**: Arquivos CSV das trajetórias teóricas de referência geradas no MATLAB.
- **`dados_coletados_bancada/`**: Telemetria serial completa a 100 Hz ($T_s = 10$\,ms) coletada com a bancada estabilizada por PID durante a Rodada 7 (CSVs e gráficos PNG).

### `04_modelagem_dinamica/`
- **`1_narx_polinomial/`**: Algoritmo FROLS/ERR para seleção parcimoniosa de 10 regressores, equações polinomiais em LaTeX, métricas de validação e simulação em regime livre (*free-run* com $\text{RMSE} = 4{,}685^\circ$).
- **`2_modelo_fisico_greybox/`**: Modelo fenomenológico não linear (ODE) parametrizado experimentalmente via *multiple shooting* com CasADi/JAX (`node_v5_multiple_shooting.py` e ensaios de atrito/queda livre).

### `05_nmpc_casadi_e_dataset/`
- **`scripts_mpc/`**: Formulação simbólica do Problema de Controle Ótimo (OCP) discretizado no CasADi com resolvedor não linear IPOPT.
- **`inviabilidade_tempo_real/`**: Registro quantitativo do tempo de resolução por passo (90–120 ms no computador de mesa), demonstrando a inviabilidade de execução online direta no microcontrolador STM32 ($T_s = 10$\,ms).
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
- Benchmarks de latência e consumo de memória (Flash e SRAM) variando flags de compilação GCC (`-O0`, `-O1`, `-O2`, `-O3`, `-Ofast`) e compilação otimizada via X-CUBE-AI, demonstrando tempo de inferência de $\approx 2{,}08$\,ms (folga de CPU $\approx 79\%$).

### `10_comparativos_finais/`
- **`triade_sim_to_real/`**: Comparação direta das trajetórias Python vs. SIL vs. Bancada Física.
- **`nominal_vs_hibrido/`**: Impacto do compensador integrador residual no fechamento do *gap* experimental.
- **`ann_vs_pid_rodada7/`**: Comparação de atraso de fase, antecipação dinâmica e erro RMSE entre o controle preditivo neural aproximado e a malha clássica PID.
- **`scripts_geracao_comparativos/`**: Scripts Python para geração dos gráficos acadêmicos em formato PNG e PDF.
