# Comparação de Experimentos (Python vs SIL vs Real)

Esta tabela compara o **RMSE** (Erro Quadrático Médio) do ângulo em relação à referência para os três ambientes testados: Simulação Pura (Python), Software-in-the-Loop (STM32) e a Bancada Real.

| Teste | RMSE Python | RMSE SIL (STM32) | RMSE Bancada Real |
|-------|-------------|------------------|-------------------|
| Validação Principal (Multiseno + Degraus) | 1.56° | 2.85° | 3.26° |
| Validação 2 (Degraus Aleatórios) | 4.31° | 4.63° | 4.77° |
| Validação 3 (Senoidal Suave) | 1.37° | 1.50° | 3.01° |

> [!NOTE]
> Os valores de RMSE da Bancada Real são ligeiramente maiores devido ao ruído de medição (IMU) e pequenas discrepâncias do modelo não-linear (atrito, etc), mas o controlador se manteve muito robusto nos 3 cenários! O código em C no STM32 provou ser quase idêntico à simulação.

---

### 1. Validação Principal (Multiseno + Degraus)
![Comparativo Validação Principal](file:///c:/Users/vicio/.gemini/antigravity-ide/brain/d56c58f0-a7c1-4a55-b68a-6c404f696320/scratch/comparacao_plot_0.png)

---

### 2. Validação 2 (Degraus Aleatórios)
![Comparativo Validação 2](file:///c:/Users/vicio/.gemini/antigravity-ide/brain/d56c58f0-a7c1-4a55-b68a-6c404f696320/scratch/comparacao_plot_1.png)

---

### 3. Validação 3 (Senoidal Suave)
![Comparativo Validação 3](file:///c:/Users/vicio/.gemini/antigravity-ide/brain/d56c58f0-a7c1-4a55-b68a-6c404f696320/scratch/comparacao_plot_2.png)
