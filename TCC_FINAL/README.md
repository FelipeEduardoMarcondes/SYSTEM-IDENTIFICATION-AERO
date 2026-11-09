# NMPC Aproximado por Redes Neurais em Hardware Embarcado: Validação Experimental

**Curso:** Engenharia de Computação — Escola Politécnica (PUCPR)  
**Autor:** Felipe Eduardo Marcondes  
**Orientador:** Prof. Dr. Helon Vicente Hultmann Ayala  

---

## 📌 Visão Geral do Projeto

Este repositório contém o código-fonte, firmwares embarcados, scripts de identificação de sistemas, controle preditivo não linear (NMPC), treinamento de redes neurais artificiais (MLP/TinyML), interface de aquisição em tempo real e conjuntos de dados experimentais desenvolvidos para o controle de posição angular de um **Aeropêndulo** utilizando o microcontrolador **STM32 Nucleo-F446RE** (ARM Cortex-M4F @ 180 MHz).

O projeto resolve o gargalo de viabilidade temporal do Controle Preditivo Baseado em Modelo Não Linear (NMPC via CasADi/IPOPT, cujo tempo de otimização online é de ~90–120 ms por passo) por meio da **aproximação explícita da lei de controle via aprendizagem supervisionada (*Behavioral Cloning*)** embarcada em C puro e via X-CUBE-AI, atingindo tempo de inferência de **~2,08 ms** (taxa de amostragem determinística de 100 Hz / $T_s = 10\text{ ms}$) com compensação híbrida integradora para eliminação do *Sim-to-Real gap*.

---

## 📁 Estrutura do Repositório (Padrão Institucional PUCPR)

```text
.
├── data/
│   ├── datasets/                                  # Dados experimentais e telemetria (100 Hz)
│   │   ├── 01_malha_aberta/                       # Ensaios estáticos (PWM vs. ângulo de equilíbrio)
│   │   ├── 02_pid_preliminar/                     # Respostas ao degrau (30°, 40°, 60°) e modelo linear (tf1.mat)
│   │   ├── 03_coleta_malha_fechada_rodada7/       # Ensaios dinâmicos (APRBS, Chirp, Multisseno, MIX_DC_45/60)
│   │   ├── 05_dataset_nmpc/                       # Trajetórias ótimas geradas pelo NMPC/IPOPT (18.700 amostras)
│   │   ├── 07_validacao_simulacao/                # Telemetria de validação em Python e STM32 Software-in-the-Loop (SIL)
│   │   ├── 08_ensaios_bancada_fisica/             # Ensaios reais na bancada (Nominal, Híbrido com Integrador e PID)
│   │   └── 09_benchmarks_tinyml/                  # Perfis de latência de inferência e otimização de compilador GCC
│   └── exemplos/
│       ├── referencias_planejadas/                # Sinais de excitação e referências CSV para o protocolo WAVE
│       └── modelos_exportados/                    # Pesos PyTorch (.pth), grafo ONNX (.onnx), header C (ann_weights.h) e JSONs
├── src/
│   ├── backend/                                   # Identificação de sistemas, NMPC e Redes Neurais
│   │   ├── aerodata.py                            # Módulo utilitário de leitura e pré-processamento de CSVs
│   │   ├── 01_sintonia_pid_e_sinais/              # Scripts MATLAB de sintonia PID e síntese de sinais de excitação
│   │   ├── 02_identificacao_sistemas/             # Identificação NARX Polinomial (FROLS/ERR) e Grey-Box Física (NODE)
│   │   ├── 03_controle_nmpc_casadi/               # Formulação simbólica do OCP no CasADi + solver IPOPT (mpc_v4.py)
│   │   └── 04_treinamento_rede_neural/            # Treinamento PyTorch da MLP (R² = 0,999) e exportação ONNX/C
│   ├── firmware/                                  # Códigos-fonte de sistemas embarcados
│   │   ├── arduino/                               # Firmware preliminar de aquisição e controle PID
│   │   ├── stm32-fisico/                          # Firmware STM32F446RE da bancada física (IMU ICM-20948 + ESC BLDC + MLP)
│   │   ├── stm32-sil/                             # Firmware STM32F446RE para validação Software-in-the-Loop e X-CUBE-AI
│   │   └── stm32-standard/                        # Projeto base STM32CubeIDE de referência
│   └── frontend/                                  # Interface desktop de aquisição, controle serial e visualização
│       ├── main.py                                # Ponto de entrada interativo (CLI) e plotagem rápida
│       ├── acquisition.py                         # Loop de coleta e threads de streaming a 100 Hz
│       ├── serial_comm.py                         # Comunicação UART (500.000 bps) e protocolos determinísticos SEQ e WAVE
│       ├── signals.py                             # Geradores de sinais (chirp, multisine, degraus aleatórios)
│       ├── live_plot.py                           # Visualizador gráfico em tempo real
│       ├── storage.py                             # Persistência de telemetria CSV e gráficos estáticos
│       ├── presets.py                             # Sequências de degraus de validação predefinidas
│       └── config.py                              # Parâmetros globais (BAUD=500000, FS=100 Hz, limites de segurança)
├── tests/
│   ├── unitarios/                                 # Testes de modelos, amortecimento, grid search e benchmark TinyML
│   └── integracao/                                # Validação cruzada (Python vs. SIL vs. Bancada) e geração de métricas
├── requirements.txt                               # Lista de dependências Python
└── README.md                                      # Documentação técnica do projeto
```

---

## ⚙️ Configuração do Ambiente e Instalação

### 1. Requisitos de Software (PC / Host)
- **Python:** 3.10 ou superior
- **STM32CubeIDE:** 1.13+ (com pacote `X-CUBE-AI` opcional para inspeção do grafo ONNX)
- **MATLAB:** R2022b+ (opcional, apenas para reexecutar os scripts `.m` em `src/backend/01_sintonia_pid_e_sinais/`)

Instale as dependências Python executando na raiz do projeto:

```bash
python -m venv .venv
# Ativação no Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Ativação no Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Requisitos de Hardware (Bancada Experimental)
- **Microcontrolador:** STM32 Nucleo-F446RE (ARM Cortex-M4F com FPU, 180 MHz, 512 KB Flash, 128 KB SRAM)
- **Sensor Inercial (IMU):** ICM-20948 via barramento I2C (com filtro complementar embarcado a 100 Hz)
- **Atuador:** Motor Brushless DC (BLDC) + ESC acionado por PWM (1000–2000 µs)
- **Comunicação:** UART via ST-LINK Virtual COM Port a **500.000 bps**

---

## 🛠️ Como Compilar e Gravar o Firmware (STM32)

1. Abra o **STM32CubeIDE**.
2. Vá em `File` → `Open Projects from File System...` e selecione:
   - `src/firmware/stm32-fisico` (para operação na bancada real do aeropêndulo); ou
   - `src/firmware/stm32-sil` (para validação *Software-in-the-Loop* onde a dinâmica NARX roda embarcada no próprio STM32).
3. Os pesos da rede neural já estão incluídos em `Core/Inc/ann_weights.h`. Caso treine um novo modelo, substitua esse cabeçalho usando `src/backend/04_treinamento_rede_neural/export_onnx.py`.
4. Para atingir a latência nominal de **2,08 ms**, certifique-se de que a flag de otimização do compilador GCC em `Project Properties` → `C/C++ Build` → `Settings` → `MCU GCC Compiler` → `Optimization` esteja configurada em `-O3` ou `-Ofast` com FPU ativa (`-mfpu=fpv4-sp-d16 -mfloat-abi=hard`).
5. Compile (`Ctrl+B`) e grave na placa Nucleo-F446RE (`Run` / `F11`).

---

## 🚀 Como Executar o Sistema

### 1. Interface de Aquisição e Operação da Bancada (`src/frontend/`)
Com a placa STM32 conectada via USB:

```bash
python src/frontend/main.py
```

A interface interativa permite:
- Selecionar a porta COM e testar a conexão serial (500 kbps);
- Enviar sequências de degraus determinísticas via **Protocolo SEQ**;
- Pré-carregar sinais contínuos (Chirp, Multisseno ou trajetórias CSV de `data/exemplos/referencias_planejadas/`) na SRAM do STM32 via **Protocolo WAVE**, garantindo execução livre de *jitter* dentro da ISR de 100 Hz;
- Visualizar a resposta angular, referência e sinal de controle em tempo real e salvar automaticamente os resultados em CSV.

Para inspecionar diretamente qualquer arquivo CSV já coletado:
```bash
python src/frontend/main.py data/datasets/08_ensaios_bancada_fisica/hibrido_com_integrador/20261023_181157_referencia_legado_val1_mpc_hybrid.csv
```

### 2. Identificação de Sistemas (`src/backend/02_identificacao_sistemas/`)
- **Modelo NARX Polinomial (FROLS/ERR):**
  ```bash
  python src/backend/02_identificacao_sistemas/1_narx_polinomial/narx.py
  ```
- **Modelo Físico Caixa-Cinza (Neural ODE / Multiple Shooting):**
  ```bash
  python src/backend/02_identificacao_sistemas/2_modelo_fisico_greybox/node_v5_multiple_shooting.py
  ```

### 3. Controle Preditivo Não Linear (NMPC) e Geração de Dataset (`src/backend/03_controle_nmpc_casadi/`)
Executa o controlador ótimo simbólico (CasADi + IPOPT) sobre o modelo identificado para gerar o conjunto de treinamento da política neural:
```bash
python src/backend/03_controle_nmpc_casadi/mpc_v4.py
```

### 4. Treinamento da Rede Neural e Exportação Embarcada (`src/backend/04_treinamento_rede_neural/`)
- Abra o notebook `src/backend/04_treinamento_rede_neural/ANN_NARX_AEROPENDULO_v5.ipynb` no Jupyter/VS Code para treinar a arquitetura MLP `[23 -> 64 -> 32 -> 1]` e exportar `ann_weights.h`.
- Para converter o cabeçalho C para o formato **ONNX** (compatível com STM32 X-CUBE-AI):
  ```bash
  python src/backend/04_treinamento_rede_neural/export_onnx.py
  ```

---

## 🧪 Como Executar os Testes e Validações (`tests/`)

### Testes Unitários e Benchmarks Embarcados
Avaliam a estabilidade em predição livre (*free-run*), amortecimento físico e métricas de desempenho computacional (ciclos de clock, latência em ms e ocupação de memória Flash/SRAM para diferentes níveis de otimização GCC):

```bash
python tests/unitarios/eval_full.py
python tests/unitarios/gerar_benchmark_tinyml.py
```

### Testes de Integração (Python vs. STM32 SIL vs. Bancada Física)
Reproduzem as análises comparativas de ponta a ponta (fidelidade numérica Python vs. STM32 SIL, fechamento do *Sim-to-Real gap* pelo integrador híbrido e comparação de rastreamento NMPC Neural vs. PID clássico):

```bash
python tests/integracao/gerar_comparacao.py
python tests/integracao/gerar_plots_academicos_hibrido.py
python tests/integracao/gerar_plots_rodada7_vs_ann.py
python tests/integracao/gerar_todos_plots_academicos.py
```

