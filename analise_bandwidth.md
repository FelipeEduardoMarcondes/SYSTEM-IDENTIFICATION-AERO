# Análise de Largura de Banda do Aeropêndulo
## Frequência Máxima para Controlabilidade e Modelagem

---

## Resumo Executivo

> [!IMPORTANT]
> **Frequência máxima para modelagem: ~1.0 a 1.5 Hz (≈ 6 a 9 rad/s)**
>
> - **Faixa segura (conservadora):** até **~0.6 Hz (3.7 rad/s)** — coerência ref→y ≥ 0.7 em todos os ensaios
> - **Faixa limite (modelável com cuidado):** até **~1.0 Hz (6.3 rad/s)** — coerência ref→y ≥ 0.5 no swept-sine
> - **Frequência de -3dB do ETFE médio:** **~1.4 Hz (8.6 rad/s)** — onde a magnitude da resposta cai para metade
> - **Acima de ~2 Hz:** o sistema não rastreia mais a referência; ruído domina

---

## 1. Coerência ref→y (Malha Fechada)

A coerência entre a referência e a saída mede quão linearmente correlacionados esses sinais são em cada frequência. Quanto mais próximo de 1.0, melhor o modelo linear captura a dinâmica.

| Ensaio | Cxy ≥ 0.7 (Hz) | Cxy ≥ 0.7 (rad/s) | Cxy ≥ 0.5 (Hz) | Cxy ≥ 0.5 (rad/s) |
|--------|:---:|:---:|:---:|:---:|
| chirp-1 (R3, 16:32) | 0.59 | 3.7 | 28.1* | 176.7* |
| chirp-1 (R3, 16:34) | 0.59 | 3.7 | 39.1* | 245.4* |
| chirp-2 (R3, 17:07) | 0.59 | 3.7 | 0.59 | 3.7 |
| chirp-2 (R3, 17:09) | 0.59 | 3.7 | 0.59 | 3.7 |
| **swept-sine-1 (19:02)** | **0.78** | **4.9** | **0.98** | **6.1** |

> [!NOTE]
> Os valores marcados com (*) na tabela acima são artefatos — o método "última frequência acima do limiar" pega picos espúrios em frequências altas. O gráfico de coerência mostra claramente que a coerência **cai abaixo de 0.5 após ~1 Hz** e depois oscila entre 0 e 0.5 de forma aleatória (ruído).

### Gráficos de Coerência e ETFE

![Coerência e ETFE para todos os ensaios de chirp e swept-sine](C:/Users/vicio/.gemini/antigravity-ide/brain/03876d93-9323-4bde-abc1-0f7ac163df9a/coerencia_etfe_chirps.png)

**Observação-chave:** Em todos os 5 ensaios, a coerência faz um pico próximo de 0.3-0.6 Hz e depois despenca. No swept-sine (último painel), o decaimento é mais suave, atingindo ~0.5 em torno de 1 Hz. A magnitude do ETFE também cai consistentemente após ~0.5-1 Hz.

---

## 2. Diagrama de Bode Empírico (ETFE Médio)

![Bode empírico médio de todos os ensaios](C:/Users/vicio/.gemini/antigravity-ide/brain/03876d93-9323-4bde-abc1-0f7ac163df9a/bode_empirico.png)

| Métrica | Frequência (Hz) | Frequência (rad/s) |
|---------|:---:|:---:|
| **Frequência de -3dB** | **~1.37** | **~8.6** |
| Frequência de -6dB | ~1.37 | ~8.6 |
| Pico de ressonância | ~0.4-0.6 | ~2.5-3.8 |

> [!NOTE]
> O pico de ressonância em ~0.5 Hz é consistente com a frequência natural do aeropêndulo. Isso indica um sistema sub-amortecido de 2ª ordem. Após o pico, a magnitude cai com ~-40 dB/dec, comportamento típico de sistema de 2ª ordem.
>
> A fase mostra comportamento errático acima de ~1.5 Hz, confirmando que não há informação dinâmica confiável acima disso.

---

## 3. Degradação do Tracking no Domínio do Tempo

![Erro RMS por janela temporal mostrando degradação progressiva](C:/Users/vicio/.gemini/antigravity-ide/brain/03876d93-9323-4bde-abc1-0f7ac163df9a/erro_por_janela.png)

**Padrão consistente em todos os ensaios:**
- **0-20s:** RMSE baixo (~3-5°) — sistema rastreia bem (frequências baixas)
- **20-40s:** RMSE começa a crescer (~5-15°) — frequência do chirp entrando na zona de transição
- **40-80s:** RMSE sobe rapidamente (15-50°) — sistema não consegue acompanhar
- **80s+:** erro máximo ou sinal de referência constante

O swept-sine (último painel) mostra claramente que a partir de ~80s o erro explode para >60°.

---

## 4. PSD Comparativa

![PSD de referência vs saída para chirps e multi-senos](C:/Users/vicio/.gemini/antigravity-ide/brain/03876d93-9323-4bde-abc1-0f7ac163df9a/psd_comparativa.png)

**Observações:**
- Nos **chirps**: as curvas de PSD da saída (sólida) e referência (tracejada) se separam em ~1-1.5 Hz, com a saída caindo mais rápido
- Nos **multi-senos**: a separação é mais dramática — acima de ~1.5 Hz, a PSD da saída é ~100x menor que a da referência em alguns ensaios (indicando que o sinal na saída acima de 1.5 Hz é dominado por ruído/dinâmica não-modelável)

---

## 5. Nota sobre Coerência u→y (Malha Aberta)

![Coerência u→y em malha aberta](C:/Users/vicio/.gemini/antigravity-ide/brain/03876d93-9323-4bde-abc1-0f7ac163df9a/coerencia_malha_aberta.png)

> [!WARNING]
> A coerência u→y mostrou valores altos (>0.7) até frequências muito elevadas (40-50 Hz). Isso **NÃO** significa que o sistema responde nessas frequências. É um artefato de **feedback em malha fechada**: o controlador cria correlação espúria entre u e y em todas as frequências.
>
> Para identificação de sistemas, a coerência **ref→y** (ou melhor ainda, usar a referência como instrumento) é mais confiável quando os dados são coletados em malha fechada.

---

## 6. Conclusão e Recomendação

### Frequência máxima para modelagem

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  FAIXA RECOMENDADA PARA IDENTIFICAÇÃO:  0 a 1.5 Hz           ║
║                                         (0 a ~9.4 rad/s)     ║
║                                                               ║
║  FAIXA CONSERVADORA (alta confiança):   0 a 0.8 Hz           ║
║                                         (0 a ~5 rad/s)       ║
║                                                               ║
║  FREQUÊNCIA NATURAL DO SISTEMA:         ~0.4 a 0.6 Hz        ║
║                                         (~2.5 a 3.8 rad/s)   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Para o projeto de controlador

| Parâmetro | Valor |
|-----------|-------|
| Largura de banda máxima do controlador | **~0.8 a 1.0 Hz** (5-6.3 rad/s) |
| Com margem de segurança (×0.7) | **~0.6 a 0.7 Hz** (3.7-4.4 rad/s) |
| Frequência de amostragem atual | 100 Hz (amplamente suficiente) |
| Frequência de Nyquist | 50 Hz |

### Justificativas

1. **Coerência ref→y cai abaixo de 0.7** consistentemente em **~0.6 Hz** nos chirps da Rodada 3 e em **~0.8 Hz** no swept-sine
2. **ETFE mostra -3dB em ~1.4 Hz** — a resposta em magnitude cai para metade do ganho DC
3. **Fase fica errática acima de ~1.5 Hz** — sem informação de fase confiável para projetar controlador
4. **RMSE do tracking explode** progressivamente à medida que a frequência do chirp ultrapassa ~0.5 Hz
5. **PSD da saída separa da referência** a partir de ~1-1.5 Hz

> [!TIP]
> Para identificação, ao usar métodos como BLS, ARX, ou NARX:
> - **Filtro anti-aliasing do modelo:** corte em **~1.5 Hz** (filtro Butterworth de 4ª ordem)
> - **Reamostragem:** os dados podem ser decimados para ~10-20 Hz sem perda de informação útil (a dinâmica relevante está abaixo de 1.5 Hz)
> - **Excitação ideal:** sinais multi-seno ou APRBS com energia concentrada entre **0.05 e 1.5 Hz**
