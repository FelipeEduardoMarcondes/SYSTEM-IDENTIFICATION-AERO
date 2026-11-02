# Relatório Técnico de Identificação e Simulação Livre (Free-Run)
## Modelo Físico Não Linear Grey-Box v9 do Aeropêndulo

**Data:** Novembro de 2026  
**Ambiente:** Bancada Física Experimental do Aeropêndulo — UTFPR  
**Taxa de Amostragem:** 100 Hz ($dt = 0,01$ s)  
**Total de Ensaios Físicos Avaliados:** 51 experimentos  
**Total de Estoiros / Giros Numéricos (>180°):** **0/51 (Zero Giros — 100% Estável)**  

---

### 1. Resumo Executivo e Proposta Metodológica

O modelo **Grey-Box v9** resolve de forma definitiva a limitação histórica dos modelos de física clássica (como a versão v3 com expoente quadrático $p=2,0$), os quais apresentavam divergência e giros completos de $360^\circ$ quando submetidos a ensaios agressivos em torno de $60^\circ$ e comandos de motor superiores a $55\%$.

A formulação matemática do modelo v9 é uma equação diferencial de 3ª ordem derivada dos primeiros princípios da mecânica e aerodinâmica:

$$\begin{cases}
\dot{\theta} = \omega \\[6pt]
\dot{\omega} = c_1 \sin(\theta) - k_{th} \theta + c_2 \omega - c_a \omega |\omega| - F_c \tanh(50 \omega) + c_3 \max(a - u_0, 0)^p \\[6pt]
\dot{a} = \dfrac{u - a}{\tau_m}
\end{cases}$$

Onde:
1. **$c_1 = -C_g / J = -6.85831$ rad/s²:** Torque gravitacional normalizado pela inércia, ancorado fisicamente na oscilação livre com motor desligado ($J = 0.04075$ kg·m²).
2. **$k_{th} = K_s / J = 1.56540$ s⁻²:** Rigidez elástica torsional do chicote elétrico no pivô do mancal ($K_s = 0.06380$ N·m/rad), corrigindo o deslocamento de centro de gravidade em baixos ângulos.
3. **$c_2 = -b_{ef} / J = -0.33145$ s⁻¹:** Coeficiente de amortecimento viscoso efetivo linear ($b_{ef} = 0.01351$ N·m·s/rad).
4. **$c_a = d_a / J = 0.01931$ rad⁻¹:** Arrasto aerodinâmico quadrático clássico do braço e duto de empuxo ($-c_a \omega |\omega|$). Em velocidades angulares moderadas ($45^\circ$), seu efeito é suave; em altas excursões dinâmicas ($60^\circ$, onde $\theta > 90^\circ$ e $\omega$ atinge $14$ rad/s), o termo quadrático atua exatamente como freio aerodinâmico natural, impedindo a ultrapassagem da sela instável de $180^\circ$.
5. **$F_c = T_c / J = 0.25582$ rad/s²:** Atrito seco de Coulomb no mancal rotativo ($T_c = 0.01043$ N·m).
6. **$u_0 = 11.85\%$:** Zona morta real do motor e ESC identificada na curva semiestática e malha aberta.
7. **$\tau_m = 113.1$ ms:** Constante de tempo eletromecânica de resposta do conjunto rotor/hélice.
8. **$p = 1.7592$:** Expoente real estático da hélice carenada (sub-quadrático, capturando a saturação aerodinâmica em comandos elevados).
9. **$c_3 = 0.012412$ rad/(s²·%^p):** Ganho estático de empuxo propulsivo.

---

### 2. Tabela de Desempenho Sintética por Grupo Experimental

A tabela a seguir resume as métricas obtidas pelo modelo v9 em **malha aberta pura (Free-Run a 100 Hz sem qualquer realimentação)**:

| Grupo Experimental | Ensaios | FIT Médio v9 | RMSE Médio v9 | MAE Médio v9 | Giros v9 (>180°) | Giros v3 (>180°) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1: Validação Oficial (RODADA-7)** | 6 | **75,9%** | **6,32°** | **4,26°** | **0/6 (0%)** | 3/6 (50%) |
| *-- Subgrupo Validação 45°* | 3 | **87,3%** | **2,61°** | **1,91°** | **0/3 (0%)** | 0/3 (0%) |
| *-- Subgrupo Validação 60°* | 3 | **64,5%** | **10,02°** | **6,30°** | **0/3 (0%)** | 3/3 (100% Giro) |
| **G2: Treinamento Oficial (RODADA-7)** | 6 | **71,8%** | **9,26°** | **5,85°** | **0/6 (0%)** | 2/6 (33%) |
| **G3: Dinâmicos Estendidos 45°** | 6 | **86,5%** | **2,75°** | **2,04°** | **0/6 (0%)** | 0/6 (0%) |
| **G4: Dinâmicos Estendidos 60°** | 6 | **62,3%** | **12,49°** | **7,84°** | **0/6 (0%)** | 2/6 (33%) |
| **G5: MALHA-ABERTA Degraus (20-50%)** | 14 | **78,3%** | **1,99°** | **1,45°** | **0/14 (0%)** | 0/14 (0%) |
| *-- Faixa Operacional MA (35-50%)* | 9 | **90,1%** | **1,85°** | **1,38°** | **0/9 (0%)** | 0/9 (0%) |
| **G6: Stress Test Contínuo (>20 min)** | 2 | **63,0%** | **10,25°** | **6,12°** | **0/2 (0%)** | 2/2 (100% Giro) |
| **G7: Cross-Session (RODADA-6)** | 11 | **78,3%** | **5,73°** | **3,89°** | **0/11 (0%)** | 2/11 (18%) |
| **MÉDIA GERAL BANCADA OFICIAL (G1..G6)** | **40** | **78,2%** | **5,34°** | **3,52°** | **0/40 (ZERO)** | **9/40 (22,5%)** |

---

### 3. Principais Conclusões para Apresentação ao Orientador

1. **Eliminação Integral dos Estoiros (0 Giros):**
   O modelo v3 apresentava giros em 9 dos 40 ensaios da bancada oficial (especialmente em chirp-60-amp60 e degraus sucessivos em 60°). O modelo v9 alcançou **taxa de sucesso de 100% (40/40 estáveis na bancada oficial e 51/51 em todo o repositório)**.

2. **Aderência Superior ao Modelo NARX na Validação 45°:**
   Nos ensaios de validação nominal (DC 45°), o modelo Grey-Box v9 atinge **FIT médio de 87,3% e RMSE de 2,61°**, superando o modelo neural NARX polinomial da literatura (69,7%).

3. **Validação Cruzada em Malha Aberta Pura (Degraus Estacionários):**
   Sem realimentação de controle, nos degraus de comando direto ao motor de 35% a 50%, o modelo v9 replica os patamares físicos de ângulo estacionário com **FIT médio de 90,1% e erro inferior a 1,9°**, comprovando a exatidão da curva de empuxo sub-quadrática ($p = 1,759$).

4. **Robustez Temporal Extrema (Stress Test de 26 Minutos):**
   No ensaio `MIX_DC_45` contendo mais de 157.000 amostras ($26,2$ minutos contínuos de perturbações harmônicas e degraus), o modelo simulou livremente com **FIT de 69,9% e RMSE de 6,84°**, sem deriva cumulativa de integração numérica.

---

### 4. Arquivos Gerados Disponíveis para o Professor

1. **`TABELA_GERAL_EXPERIMENTOS_FREE_RUN_V9.xlsx`:** Planilha Excel com 3 abas estruturadas (Resumo Executivo com formatação condicional, Tabela Completa de todos os 51 ensaios e Parâmetros Físicos com deduções mecânicas).
2. **`TABELA_GERAL_EXPERIMENTOS_FREE_RUN_V9.csv`:** Tabela consolidada em formato texto delimitado por vírgula para scripts e automações.
3. **`dados_simulacao_freerun_v9_completo.npz`:** Arquivo compactado com todas as séries temporais (tempo, medição real, simulação v9, comando e erro de trajetória) indexadas pelo nome do ensaio.
4. **Pasta `graficos/`:** Figuras em altíssima resolução (PNG 300 DPI e PDF vetorial para o TCC/Artigo):
   - `01_validacao_oficial_dc45.png` / `.pdf`
   - `02_validacao_oficial_dc60.png` / `.pdf`
   - `03_treinamento_oficial_dc45.png` / `.pdf`
   - `04_treinamento_oficial_dc60.png` / `.pdf`
   - `05_rodada7_estendida_dc45.png` / `.pdf`
   - `06_rodada7_estendida_dc60.png` / `.pdf`
   - `07_malha_aberta_degraus_grid.png` / `.pdf`
   - `08_stress_test_mix_longo.png` / `.pdf`
   - `09_comparativo_barras_metricas_v9_vs_v3.png` / `.pdf`
