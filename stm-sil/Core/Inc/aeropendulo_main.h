/*
 * aeropendulo_main.h
 * ==================
 * Interface publica do firmware do aeropendulo para STM32F446RE.
 *
 * Inclua este header em main.c gerado pelo CubeMX e chame as duas funcoes
 * abaixo conforme indicado.
 */

#ifndef AEROPENDULO_MAIN_H
#define AEROPENDULO_MAIN_H

#include "stm32f4xx_hal.h"

/*
 * Chamada UMA vez apos MX_xxx_Init() no main():
 *   - Inicializa IMU, ESC, calibra giroscopio, inicia timer e UART RX.
 */
void aeropendulo_init(void);

/*
 * Chamada dentro do while(1) do main():
 *   - Processa comandos UART pendentes.
 *   - Executa o ciclo PID a cada tick de 10 ms (flag setada pelo TIM6 ISR).
 */
void aeropendulo_loop(void);

/*
 * Callbacks HAL que devem ser expostos ao linker.
 * O HAL ja declara essas assinaturas como __weak; basta incluir este header
 * no arquivo que contem as implementacoes (aeropendulo_main.c) para que o
 * linker use as versoes fortes aqui definidas.
 *
 * Se o seu projeto ja possui um stm32f4xx_it.c chamando HAL_TIM_IRQHandler
 * e HAL_UART_IRQHandler, nao e necessario nenhuma alteracao adicional.
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim);

#endif /* AEROPENDULO_MAIN_H */
