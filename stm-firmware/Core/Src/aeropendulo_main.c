/*
 * aeropendulo_main.c
 * ==================
 * Firmware do aeropendulo portado para STM32F446RE (HAL).
 *
 * Mapeamento de perifericos (ver Tabela 3 do TCC):
 *   I2C1_SCL  -> PB8   (IMU ICM-20948, 400 kHz)
 *   I2C1_SDA  -> PB9
 *   TIM1_CH1  -> PA8   (PWM para o ESC, 50 Hz, 1000-2000 us)
 *   USART2_TX -> PA2   (telemetria 500000 baud)
 *   USART2_RX -> PA3
 *
 * Configuracao necessaria no CubeMX / CubeIDE:
 *   - TIM1: CH1 PWM, PSC=179, ARR=19999  => 50 Hz, resolucao 1 us
 *   - TIM6: Base de tempo 10 ms (100 Hz), interrupcao habilitada
 *   - I2C1: Fast Mode 400 kHz
 *   - USART2: 500000 baud, 8N1, TX DMA (opcional), RX interrupcao
 *   - FPU: habilitada (CortexM4F padrao)
 *
 * Protocolos UART (identicos ao firmware Arduino):
 *   SEQ=t1_ms:ref1,t2_ms:ref2,...
 *   WAVE=<n>
 *   DATA=v0,v1,...
 *   DATA_END
 *   CHIRP=Amp,Fmax,T0,DC,PadS
 *   R=<graus>
 *   START | STOP | RECAL | FREE
 */

#include "aeropendulo_main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Handles HAL (declarados em main.c pelo CubeMX) ─────────────────────── */
extern I2C_HandleTypeDef  hi2c1;
extern TIM_HandleTypeDef  htim1;   /* PWM ESC                                */
extern TIM_HandleTypeDef  htim6;   /* base de tempo 100 Hz                   */
extern UART_HandleTypeDef huart2;

/* I2C_ClearBus() definida em main.c — 9 pulsos de clock para destravar IMU */
extern void I2C_ClearBus(void);

/* ── Endere\u00e7o I2C do ICM-20948 (AD0=GND => 0x68 << 1 para HAL) ─────────────── */
#define ICM_ADDR        (0x68 << 1)

/* ── Registradores do ICM-20948 ─────────────────────────────────────────── */
#define REG_BANK_SEL    0x7F
#define REG_PWR_MGMT_1  0x06
#define REG_PWR_MGMT_2  0x07
#define REG_GYRO_SMPLRT 0x00   /* banco 2 */
#define REG_GYRO_CFG_1  0x01   /* banco 2 */
#define REG_ACCEL_SMPLRT2 0x11 /* banco 2 */
#define REG_ACCEL_CFG   0x14   /* banco 2 */
#define REG_INT_STATUS_1 0x1A  /* banco 0 */
#define REG_ACCEL_XOUT_H 0x2D  /* banco 0 */
#define REG_GYRO_XOUT_H  0x33  /* banco 0 */
#define ACCEL_SENS      4096.0f  /* LSB/g  — escala +-8g  */
#define GYRO_SENS       65.5f    /* LSB/(deg/s) — escala +-500 dps */

/* ── ESC / PWM ───────────────────────────────────────────────────────────── */
#define ESC_NEUTRO_US   1500
#define ESC_MIN_US      1000
#define ESC_MAX_US      2000
#define TIM1_CH1        TIM_CHANNEL_1

/* ── Amostragem ──────────────────────────────────────────────────────────── */
#define FS              100.0f
#define TS              (1.0f / FS)   /* 10 ms */

/* ── Ganhos PID ──────────────────────────────────────────────────────────── */
#define KP              0.5793f
#define KI              0.6647f
#define KD              0.2f

/* ── Limites de controle ─────────────────────────────────────────────────── */
#define U_MAX           100.0f

/* ── Buffers de dados ────────────────────────────────────────────────────── */
#define MAX_STEPS       50
#define WAVE_MAX        16500  /* STM32F446 tem 128 KB SRAM; 16500*4=66 KB OK */

/* ── UART RX ─────────────────────────────────────────────────────────────── */
#define RX_BUF_LEN      2048
#define I2C_TIMEOUT      5     /* ms — timeout (12 bytes a 100kHz leva ~1.5ms, 5ms evita falsos positivos do SysTick) */
#define I2C_TIMEOUT_INIT 100   /* ms — timeout generoso para inicialização */

/* ======================================================================== */
/* Tipos                                                                      */
/* ======================================================================== */

typedef struct {
    uint32_t t_ms;
    float    ref;
} Degrau_t;

typedef enum {
    ESTADO_IDLE,
    ESTADO_LOADING_WAVE,
    ESTADO_RUNNING,
    ESTADO_FREE_RUN
} Estado_t;

/* ======================================================================== */
/* Variaveis globais do modulo                                                */
/* ======================================================================== */

/* --- IMU / angulo -------------------------------------------------------- */
static uint16_t i2c_tmo       = I2C_TIMEOUT_INIT; /* começa generoso, reduz após init */
static float angulo_filtrado = 0.0f;
static float gyro_bias       = 0.0f;

/* --- PID ----------------------------------------------------------------- */
static float e_1 = 0.0f;
static float u_i = 0.0f;
static float y_1 = 0.0f;
static float r   = 40.0f;   /* referencia corrente (graus) */

/* --- SEQ ----------------------------------------------------------------- */
static Degrau_t degraus[MAX_STEPS];
static int      n_degraus   = 0;
static int      idx_degrau  = 0;

/* --- WAVE ---------------------------------------------------------------- */
static float    wave_buf[WAVE_MAX];
static int      wave_len        = 0;
static int      wave_total_esp  = 0;
static int      wave_idx        = 0;
static int      wave_bloco      = 0;
static uint8_t  wave_ativo      = 0;

/* --- CHIRP --------------------------------------------------------------- */
static uint8_t  chirp_ativo   = 0;
static float    chirp_amp     = 0.0f;
static float    chirp_t0      = 0.0f;
static float    chirp_dc      = 0.0f;
static float    chirp_pad_s   = 0.0f;
static float    chirp_a       = 0.0f;
static float    chirp_b       = 0.0f;

/* --- Temporização --------------------------------------------------------- */
static uint32_t tempo_inicio = 0;
static uint32_t ultimo_dado_us = 0;
static uint32_t proxima_amostra_us = 0;

/* --- Estado da maquina --------------------------------------------------- */
static volatile Estado_t estado = ESTADO_IDLE;

/* --- Ultimas leituras validas da IMU ------------------------------------- */
static float last_ax = 0.0f;
static float last_az = -1.0f;
static float last_gy = 0.0f;

/* --- Flag de tick (setada pela ISR do TIM6) ------------------------------ */
volatile uint8_t tick_flag = 0;

/* --- UART RX (modo interrupcao byte a byte) ------------------------------ */
static uint8_t  rx_byte;
static char     rx_buffer[RX_BUF_LEN];
static uint16_t rx_pos = 0;
static volatile uint8_t cmd_ready = 0;
static char     cmd_line[2][RX_BUF_LEN];
static volatile uint8_t cmd_wr_idx = 0;

/* ======================================================================== */
/* Prototipos internos                                                        */
/* ======================================================================== */

static void     icm_write_reg(uint8_t bank, uint8_t reg, uint8_t val);
static uint8_t  icm_read_reg(uint8_t bank, uint8_t reg);
static uint8_t  icm_read_burst(uint8_t bank, uint8_t reg, uint8_t *buf, uint16_t len);
static void     icm_configurar(void);
static HAL_StatusTypeDef icm_iniciar(void);
static void     calibrar_giroscopio(void);
static float    angulo_acelerometro(void);
static void     esc_set_us(uint16_t us);
static void     resetar_controlador(void);
static void     parsear_seq(const char *cmd);
static void     iniciar_wave(const char *cmd);
static void     processar_data(const char *cmd);
static void     finalizar_wave(void);
static void     parsear_chirp(const char *cmd);
static void     processar_comando(const char *cmd);
static void     ciclo_controle(void);
static void     uart_print(const char *s);
static void     uart_println(const char *s);

/* ======================================================================== */
/* Utilitarios UART                                                           */
/* ======================================================================== */

/* --- UART TX Buffer ------------------------------------------------------ */
static char tx_buffer[256];

static void uart_print(const char *s)
{
    /* Espera o envio anterior terminar antes de sobrescrever o buffer */
    while (huart2.gState != HAL_UART_STATE_READY) {}
    
    strncpy(tx_buffer, s, sizeof(tx_buffer) - 1);
    tx_buffer[sizeof(tx_buffer) - 1] = '\0';
    HAL_UART_Transmit_IT(&huart2, (uint8_t *)tx_buffer, (uint16_t)strlen(tx_buffer));
}

static void uart_println(const char *s)
{
    while (huart2.gState != HAL_UART_STATE_READY) {}
    
    snprintf(tx_buffer, sizeof(tx_buffer), "%s\r\n", s);
    HAL_UART_Transmit_IT(&huart2, (uint8_t *)tx_buffer, (uint16_t)strlen(tx_buffer));
}

/* ======================================================================== */
/* Formatacao de float sem %f (evita -u _printf_float no newlib-nano)         */
/* ======================================================================== */

/*
 * Formata float como "[-]inteiro.dd" (2 casas, arredondado).
 * Retorna o numero de caracteres escritos (como sprintf).
 * Resolve bug de perda de sinal para valores entre -1.0 e 0.0.
 */
static int fmt_float(char *dst, float val)
{
    int neg = (val < 0.0f);
    if (neg) val = -val;
    int cents = (int)(val * 100.0f + 0.5f);
    int inteiro = cents / 100;
    int frac = cents % 100;
    if (neg && (inteiro > 0 || frac > 0))
        return sprintf(dst, "-%d.%02d", inteiro, frac);
    return sprintf(dst, "%d.%02d", inteiro, frac);
}

/* ======================================================================== */
/* Temporização DWT (microssegundos) e Acesso ao ICM-20948 via I2C HAL     */
/* ======================================================================== */

static uint32_t hclk_mhz = 0;
static void dwt_init(void)
{
    hclk_mhz = HAL_RCC_GetHCLKFreq() / 1000000U;
    if (hclk_mhz == 0) hclk_mhz = 16;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint64_t get_cycles64(void)
{
    static uint32_t last_cyc = 0;
    static uint64_t accum = 0;
    uint32_t curr = DWT->CYCCNT;
    accum += (curr - last_cyc);
    last_cyc = curr;
    return accum;
}

static uint32_t micros(void)
{
    return (uint32_t)(get_cycles64() / hclk_mhz);
}

static void icm_write_reg(uint8_t bank, uint8_t reg, uint8_t val)
{
    uint8_t buf[2];
    /* seleciona banco */
    buf[0] = REG_BANK_SEL;
    buf[1] = (uint8_t)(bank << 4);
    HAL_I2C_Master_Transmit(&hi2c1, ICM_ADDR, buf, 2, i2c_tmo);
    /* escreve registrador */
    buf[0] = reg;
    buf[1] = val;
    HAL_I2C_Master_Transmit(&hi2c1, ICM_ADDR, buf, 2, i2c_tmo);
}

static uint8_t icm_read_reg(uint8_t bank, uint8_t reg)
{
    uint8_t buf[2];
    buf[0] = REG_BANK_SEL;
    buf[1] = (uint8_t)(bank << 4);
    HAL_I2C_Master_Transmit(&hi2c1, ICM_ADDR, buf, 2, i2c_tmo);
    HAL_I2C_Master_Transmit(&hi2c1, ICM_ADDR, &reg, 1, i2c_tmo);
    uint8_t val = 0;
    HAL_I2C_Master_Receive(&hi2c1, ICM_ADDR, &val, 1, i2c_tmo);
    return val;
}

static uint8_t icm_read_burst(uint8_t bank, uint8_t reg, uint8_t *buf, uint16_t len)
{
    /* O banco já está no 0. Apenas faz a leitura direta para evitar 
       duas transações e reduzir a chance de corromper o I2C pelo ruído do motor */
    return (HAL_I2C_Mem_Read(&hi2c1, ICM_ADDR, reg, I2C_MEMADD_SIZE_8BIT, buf, len, i2c_tmo) == HAL_OK) ? 1 : 0;
}

static uint8_t icm_data_ready(void)
{
    uint8_t status = 0;
    /* REG_INT_STATUS_1 = 0x1A. O bit 0 indica RAW_DATA_0_RDY_INT */
    if (icm_read_burst(0, 0x1A, &status, 1)) {
        return (status & 0x01);
    }
    return 0;
}

static void icm_configurar(void)
{
    /* Habilita RAW_DATA_0_RDY_INT no banco 0 (INT_ENABLE_1 = 0x11) */
    icm_write_reg(0, 0x11, 0x01);

    /* Banco 2: giroscopio — ODR divider=10 => ~100 Hz, DLPF cfg=4 (~51 Hz) */
    icm_write_reg(2, REG_GYRO_SMPLRT, 0x0A);   /* GYRO_SMPLRT_DIV = 10 => ~102 Hz (igual Arduino) */
    icm_write_reg(2, REG_GYRO_CFG_1,  0x23);   /* GYRO_FS_SEL=01 (+-500dps), DLPFCFG=4, FCHOICE=1 */

    /* Banco 2: acelerometro — ODR divider=10, DLPF cfg=4 */
    icm_write_reg(2, REG_ACCEL_SMPLRT2, 0x0A); /* ACCEL_SMPLRT_DIV LSB = 10 (igual Arduino) */
    icm_write_reg(2, REG_ACCEL_CFG,     0x25); /* ACCEL_FS_SEL=10 (+-8g), DLPFCFG=4, FCHOICE=1 */

    /* Volta o banco para 0 permanentemente, essencial para icm_read_burst */
    icm_read_reg(0, 0x00);
}

static HAL_StatusTypeDef icm_iniciar(void)
{
    /* Reseta o dispositivo (DEVICE_RESET = bit 7) */
    icm_write_reg(0, REG_PWR_MGMT_1, 0x80);
    HAL_Delay(100);
    
    /* Acorda o dispositivo (Auto select clock) */
    icm_write_reg(0, REG_PWR_MGMT_1, 0x01);
    HAL_Delay(50);
    icm_write_reg(0, REG_PWR_MGMT_2, 0x00);   /* liga accel e gyro   */
    HAL_Delay(50);

    /* Verifica WHO_AM_I (banco 0, reg 0x00 == 0xEA) */
    uint8_t who = icm_read_reg(0, 0x00);
    if (who != 0xEA) return HAL_ERROR;

    icm_configurar();
    return HAL_OK;
}

static float angulo_acelerometro(void)
{
    uint8_t buf[6] = {0};
    uint32_t t0 = HAL_GetTick();
    
    /* Espera o IMU sinalizar que tem dado novo com limite de tempo */
    while (!icm_data_ready()) {
        if (HAL_GetTick() - t0 > 200) break;
        HAL_Delay(2);
    }
    
    t0 = HAL_GetTick();
    /* Tenta ler com limite de tempo */
    while (!icm_read_burst(0, REG_ACCEL_XOUT_H, buf, 6)) {
        if (HAL_GetTick() - t0 > 200) break;
        HAL_Delay(2);
    }
    int16_t ax = (int16_t)((buf[0] << 8) | buf[1]);
    int16_t az = (int16_t)((buf[4] << 8) | buf[5]);
    float ax_g = (float)ax / ACCEL_SENS;
    float az_g = (float)az / ACCEL_SENS;
    
    /* Pre-carrega os globais para o filtro não zerar se o 1º burst falhar */
    last_ax = ax_g;
    last_az = az_g;
    last_gy = gyro_bias; /* gy inicial = 0 (considera em repouso) */
    
    return atan2f(-az_g, ax_g) * 180.0f / (float)M_PI;
}


/* Le giroscopio eixo Y */
static float icm_read_gyro_y(void)
{
    uint8_t buf[2] = {0};
    icm_read_burst(0, REG_GYRO_XOUT_H + 2, buf, 2); /* Y = offset +2 */
    int16_t gy_raw = (int16_t)((buf[0] << 8) | buf[1]);
    return (float)gy_raw / GYRO_SENS;
}

/* Le acelerometro e giroscopio em uma unica leitura burst */
static uint8_t icm_read_ag(float *accel_x, float *accel_z, float *gyro_y)
{
    uint8_t buf[12] = {0};
    /* ACCEL: 6 bytes a partir de REG_ACCEL_XOUT_H                         */
    /* GYRO:  6 bytes a partir de REG_GYRO_XOUT_H (= ACCEL_XOUT_H + 6)    */
    if (!icm_read_burst(0, REG_ACCEL_XOUT_H, buf, 12)) return 0;

    int16_t ax_r = (int16_t)((buf[0]  << 8) | buf[1]);
    int16_t az_r = (int16_t)((buf[4]  << 8) | buf[5]);
    int16_t gy_r = (int16_t)((buf[8]  << 8) | buf[9]);  /* Y = 2o eixo do gyro (offset +8) */

    *accel_x = (float)ax_r / ACCEL_SENS;
    *accel_z = (float)az_r / ACCEL_SENS;
    *gyro_y  = (float)gy_r / GYRO_SENS;
    return 1;
}

/* ======================================================================== */
/* Calibracao do giroscopio                                                   */
/* ======================================================================== */

static void calibrar_giroscopio(void)
{
    uart_println("# CAL_START");
    HAL_Delay(1000);
    float soma = 0.0f;
    int   n    = 0;
    uint32_t t0 = HAL_GetTick();
    while (n < 500) {
        if (HAL_GetTick() - t0 > 5000) break;   /* timeout de seguranca */
        
        soma += icm_read_gyro_y();
        n++;
        HAL_Delay(2);
        
    }
    gyro_bias = (n > 0) ? soma / (float)n : 0.0f;

    /* Valida: bias fora de +-5 deg/s indica problema no I2C ou IMU */
    if (fabsf(gyro_bias) > 5.0f) {
        uart_println("# CAL_WARN bias fora da faixa, usando 0");
        gyro_bias = 0.0f;
    }

    char buf[64];
    char s_bias[16];
    fmt_float(s_bias, gyro_bias);
    snprintf(buf, sizeof(buf), "# CAL_BIAS %s n=%d", s_bias, n);
    uart_println(buf);
    uart_println("# CAL_OK");
}

/* ======================================================================== */
/* ESC / PWM                                                                  */
/* ======================================================================== */

/* TIM1 configurado com PSC=179, ARR=19999 => periodo=20 ms (50 Hz).
   CCR1 em microsegundos corresponde diretamente ao pulse width.             */
static void esc_set_us(uint16_t us)
{
    if (us < ESC_MIN_US) us = ESC_MIN_US;
    if (us > ESC_MAX_US) us = ESC_MAX_US;
    __HAL_TIM_SET_COMPARE(&htim1, TIM1_CH1, us);
}

/* pct em [-100, 100] -> microsegundos: 1500 + 5*pct (ESC Bidirecional) */
static uint16_t pct_para_us(float pct)
{
    if (pct >  U_MAX) pct =  U_MAX;
    if (pct < -U_MAX) pct = -U_MAX;
    
    /* Usando + 5.0f para manter a direcao de giro que acabou de funcionar pra voce */
    int us = (int)(1500.0f + 5.0f * pct);
    return (uint16_t)us;
}

/* ======================================================================== */
/* Controlador                                                                */
/* ======================================================================== */

static void resetar_controlador(void)
{
    e_1 = 0.0f;
    u_i = 0.0f;
    y_1 = 0.0f;
}

/* ======================================================================== */
/* Parsers de comandos                                                        */
/* ======================================================================== */

static void parsear_seq(const char *cmd)
{
    /* cmd = "SEQ=t1_ms:ref1,t2_ms:ref2,..." */
    n_degraus = 0;
    const char *p = cmd + 4;   /* pula "SEQ=" */
    char tok[32];

    while (*p && n_degraus < MAX_STEPS) {
        /* extrai proximo token ate ',' ou fim */
        int i = 0;
        while (*p && *p != ',' && i < 31) tok[i++] = *p++;
        tok[i] = '\0';
        if (*p == ',') p++;

        /* separa t_ms:ref */
        char *col = strchr(tok, ':');
        if (col) {
            *col = '\0';
            degraus[n_degraus].t_ms = (uint32_t)atoi(tok);
            degraus[n_degraus].ref  = strtof(col + 1, NULL);
            n_degraus++;
        }
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "# SEQ_OK n=%d", n_degraus);
    uart_println(buf);
    if (n_degraus > 0) r = degraus[0].ref;
}

static void iniciar_wave(const char *cmd)
{
    wave_total_esp = atoi(cmd + 5);   /* pula "WAVE=" */
    wave_len   = 0;
    wave_bloco = 0;

    if (wave_total_esp <= 0 || wave_total_esp > WAVE_MAX) {
        char buf[64];
        snprintf(buf, sizeof(buf), "# WAVE_ERR total=%d limite=%d",
                 wave_total_esp, WAVE_MAX);
        uart_println(buf);
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "# WAVE_OK n=%d", wave_total_esp);
    uart_println(buf);
    estado = ESTADO_LOADING_WAVE;
}

static void processar_data(const char *cmd)
{
    const char *p = cmd + 5;   /* pula "DATA=" */
    char tok[32];

    while (*p && wave_len < WAVE_MAX) {
        int i = 0;
        while (*p && *p != ',' && i < 31) tok[i++] = *p++;
        tok[i] = '\0';
        if (*p == ',') p++;
        if (i > 0) wave_buf[wave_len++] = strtof(tok, NULL);
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "# DATA_ACK %d", wave_bloco++);
    uart_println(buf);
}

static void finalizar_wave(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "# WAVE_READY n=%d", wave_len);
    uart_println(buf);
    wave_ativo = 1;
    estado = ESTADO_IDLE;
}

static void parsear_chirp(const char *cmd)
{
    /* cmd = "CHIRP=amp,fmax,t0,dc,pad_s" */
    const char *p = cmd + 6;
    float params[5] = {0};
    int   n = 0;
    char  tok[32];

    while (*p && n < 5) {
        int i = 0;
        while (*p && *p != ',' && i < 31) tok[i++] = *p++;
        tok[i] = '\0';
        if (*p == ',') p++;
        if (i > 0) params[n++] = strtof(tok, NULL);
    }

    if (n < 4) { uart_println("# CHIRP_ERR"); return; }

    chirp_amp   = params[0];
    float f_max = params[1];
    chirp_t0    = params[2];
    chirp_dc    = params[3];
    chirp_pad_s = (n >= 5) ? params[4] : 0.0f;

    float f0 = 1.0f / chirp_t0;
    chirp_a  = (float)M_PI * (f_max / f0 - 1.0f) * (f0 * f0);
    chirp_b  = 2.0f * (float)M_PI * f0;

    chirp_ativo = 1;
    wave_ativo  = 0;
    n_degraus   = 0;

    uart_println("# CHIRP_OK");
}

/* ======================================================================== */
/* Processamento de um comando completo recebido pela UART                   */
/* ======================================================================== */

static void processar_comando(const char *cmd)
{
    if (strcmp(cmd, "REBOOT") == 0) {
        NVIC_SystemReset();
    }

    if (strcmp(cmd, "STOP") == 0) {
        esc_set_us(ESC_NEUTRO_US);
        resetar_controlador();
        wave_ativo  = 0;
        chirp_ativo = 0;
        wave_idx    = 0;
        n_degraus   = 0;
        estado = ESTADO_IDLE;
        uart_println("# PARADO");
        return;
    }

    if (strcmp(cmd, "RECAL") == 0 && estado == ESTADO_IDLE) {
        calibrar_giroscopio();
        uart_println("# PRONTO");
        return;
    }

    if (strncmp(cmd, "R=", 2) == 0) {
        r = strtof(cmd + 2, NULL);
        return;
    }

    if (strncmp(cmd, "SEQ=", 4) == 0 && estado == ESTADO_IDLE) {
        wave_ativo  = 0;
        chirp_ativo = 0;
        parsear_seq(cmd);
        return;
    }

    if (strncmp(cmd, "WAVE=", 5) == 0 && estado == ESTADO_IDLE) {
        n_degraus   = 0;
        chirp_ativo = 0;
        iniciar_wave(cmd);
        return;
    }

    if (strncmp(cmd, "DATA=", 5) == 0 && estado == ESTADO_LOADING_WAVE) {
        processar_data(cmd);
        return;
    }

    if (strcmp(cmd, "DATA_END") == 0 && estado == ESTADO_LOADING_WAVE) {
        finalizar_wave();
        return;
    }

    if (strncmp(cmd, "CHIRP=", 6) == 0 && estado == ESTADO_IDLE) {
        parsear_chirp(cmd);
        return;
    }

    if (strcmp(cmd, "START") == 0 && estado == ESTADO_IDLE) {
        angulo_filtrado = angulo_acelerometro();
        resetar_controlador();
        tempo_inicio = HAL_GetTick();
        ultimo_dado_us = micros();
        proxima_amostra_us = ultimo_dado_us + 10000UL;
        idx_degrau   = 0;
        wave_idx     = 0;

        if      (chirp_ativo)  r = chirp_dc;
        else if (wave_ativo)   r = wave_buf[0];
        else if (n_degraus > 0) r = degraus[0].ref;

        estado = ESTADO_RUNNING;
        uart_println("# EXP_START");
        uart_println("tempo_ms,angulo_deg,u_pct,referencia");
        return;
    }

    if (strcmp(cmd, "FREE") == 0 && estado == ESTADO_IDLE) {
        angulo_filtrado = angulo_acelerometro();
        resetar_controlador();
        tempo_inicio = HAL_GetTick();
        ultimo_dado_us = micros();
        proxima_amostra_us = ultimo_dado_us + 10000UL;

        estado = ESTADO_FREE_RUN;
        uart_println("# EXP_START");
        uart_println("tempo_ms,angulo_deg,u_pct,referencia");
        return;
    }
}

/* ======================================================================== */
/* Ciclo de controle — chamado a cada 10 ms pela ISR do TIM6                 */
/* ======================================================================== */

static void ciclo_controle(void)
{
    uint32_t t_exp = HAL_GetTick() - tempo_inicio;

    /* --- FREE_RUN: apenas coleta sem controle ----------------------------- */
    if (estado == ESTADO_FREE_RUN) {
        icm_read_ag(&last_ax, &last_az, &last_gy); // Atualiza globais se tiver sucesso
        float angulo_acel = atan2f(-last_az, last_ax) * 180.0f / (float)M_PI;
        float gy          = last_gy - gyro_bias;
        angulo_filtrado   = 0.98f * (angulo_filtrado + (-gy) * TS)
                          + 0.02f * angulo_acel;

        char buf[96];
        char s_ang[16];
        fmt_float(s_ang, angulo_filtrado + 90.0f);
        snprintf(buf, sizeof(buf), "%lu,%s,0.00,0.00\r\n",
                 (unsigned long)t_exp, s_ang);
        uart_print(buf);
        return;
    }

    if (estado != ESTADO_RUNNING) return;

    /* --- Leitura da IMU --------------------------------------------------- */
    icm_read_ag(&last_ax, &last_az, &last_gy); // Atualiza globais se tiver sucesso

    /* --- Filtro complementar (alpha=0.98) --------------------------------- */
    float angulo_acel = atan2f(-last_az, last_ax) * 180.0f / (float)M_PI;
    float gy          = last_gy - gyro_bias;
    angulo_filtrado   = 0.98f * (angulo_filtrado + (-gy) * TS)
                      + 0.02f * angulo_acel;

    /* --- Avanco de referencia --------------------------------------------- */
    if (chirp_ativo) {
        float t_sec = t_exp / 1000.0f;
        float t_fim = chirp_pad_s + chirp_t0;
        if (t_sec < chirp_pad_s) {
            r = chirp_dc;
        } else if (t_sec < t_fim) {
            float t_loc = t_sec - chirp_pad_s;
            r = chirp_amp * sinf((chirp_a * t_loc + chirp_b) * t_loc) + chirp_dc;
        } else {
            r = chirp_dc;
        }
    } else if (wave_ativo) {
        if (wave_idx < wave_len) r = wave_buf[wave_idx++];
    } else if (n_degraus > 0) {
        while (idx_degrau + 1 < n_degraus &&
               t_exp >= degraus[idx_degrau + 1].t_ms) {
            idx_degrau++;
            r = degraus[idx_degrau].ref;
            char buf[64];
            char s_ref[16];
            fmt_float(s_ref, r);
            snprintf(buf, sizeof(buf),
                     "# STEP_CHANGE idx=%d t=%lums ref=%s",
                     idx_degrau, (unsigned long)t_exp, s_ref);
            uart_println(buf);
        }
    }

    /* --- PID discreto ----------------------------------------------------- */
    float y_med = angulo_filtrado + 90.0f;
    float e     = r - y_med;

    float u_p = KP * e;
    u_i       = u_i + KI * (TS / 2.0f) * (e + e_1);          /* Tustin      */
    float u_d = -(KD / TS) * (y_med - y_1);                   /* backward    */
    float u   = u_p + u_i + u_d;

    /* Anti-windup por back-calculation */
    float u_sat = u;
    if (u_sat >  U_MAX) u_sat =  U_MAX;
    if (u_sat < -U_MAX) u_sat = -U_MAX;
    if (u != u_sat) u_i -= (u - u_sat);
    u = u_sat;

    esc_set_us(pct_para_us(u));
    e_1 = e;
    y_1 = y_med;

    /* --- Telemetria CSV --------------------------------------------------- */
    char buf[96];
    char s_y[16], s_u[16], s_r[16];
    fmt_float(s_y, y_med);
    fmt_float(s_u, u);
    fmt_float(s_r, r);
    snprintf(buf, sizeof(buf), "%lu,%s,%s,%s\r\n",
             (unsigned long)t_exp, s_y, s_u, s_r);
    uart_print(buf);
}

/* ======================================================================== */
/* Callbacks HAL                                                              */
/* ======================================================================== */

/*
 * Chamado pelo HAL a cada byte recebido na UART2 (habilitar com
 * HAL_UART_Receive_IT(&huart2, &rx_byte, 1) no inicio e reativar aqui).
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) return;

    char c = (char)rx_byte;
    if (c == '\n' || c == '\r') {
        if (rx_pos > 0) {
            rx_buffer[rx_pos] = '\0';
            memcpy(cmd_line[cmd_wr_idx], rx_buffer, rx_pos + 1);
            cmd_wr_idx ^= 1;
            cmd_ready = 1;
            rx_pos = 0;
        }
    } else if (rx_pos < RX_BUF_LEN - 1) {
        rx_buffer[rx_pos++] = c;
    }

    /* Reativa a interrupcao para o proximo byte */
    HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
}

/*
 * Chamado pelo HAL a cada estouro do TIM6 (100 Hz).
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        tick_flag = 1;
    }
}

/* ======================================================================== */
/* Ponto de entrada (chamar de main() apos inicializacao HAL/CubeMX)         */
/* ======================================================================== */

void aeropendulo_init(void)
{
    /* Inicia contador de microssegundos (DWT) */
    dwt_init();

    /* Inicia recepcao UART o mais cedo possivel */
    HAL_UART_Receive_IT(&huart2, &rx_byte, 1);

    /* INICIA O PWM AQUI! O ESC precisa do sinal o mais rapido possivel
       ao ligar a placa, senao ele entra em modo de falha (timeout de sinal). */
    HAL_TIM_PWM_Start(&htim1, TIM1_CH1);
    esc_set_us(ESC_NEUTRO_US);
    uart_println("# PWM INICIADO (1500us - Aguardando IMU...)");

    /* Inicializa IMU (com recovery completo do I2C a cada tentativa) */
    {
        int imu_tentativas = 0;
        while (icm_iniciar() != HAL_OK) {
            imu_tentativas++;
            char buf[48];
            snprintf(buf, sizeof(buf), "# ERRO_IMU tentativa %d...", imu_tentativas);
            uart_println(buf);

            /* Recovery: DeInit do periferico I2C, clear bus fisico, re-Init */
            HAL_I2C_DeInit(&hi2c1);
            I2C_ClearBus();
            HAL_I2C_Init(&hi2c1);
            HAL_Delay(500);
        }
    }

    /* Apos init OK, reduz timeout I2C para runtime */
    i2c_tmo = I2C_TIMEOUT;

    /* Aguarda o restante do tempo de armacao do ESC (3 segundos totais recomendados) */
    HAL_Delay(3000);
    uart_println("# ESC_OK");

    /* Calibra giroscopio */
    calibrar_giroscopio();

    /* Inicializa estado do filtro com leitura do acelerometro */
    angulo_filtrado = angulo_acelerometro();

    /* Inicia timer de base de tempo 100 Hz */
    HAL_TIM_Base_Start_IT(&htim6);

    uart_println("# PRONTO");
    uart_println("# Comandos: START | STOP | RECAL | FREE");
    uart_println("# SEQ=t1:r1,...  |  R=<graus>");
    uart_println("# WAVE=<n> | DATA=v0,... | DATA_END");
    uart_println("# CHIRP=Amp,Fmax,T0,DC,PadS");
}

void aeropendulo_loop(void)
{
    /* Processa comando pendente */
    if (cmd_ready) {
        cmd_ready = 0;
        processar_comando(cmd_line[cmd_wr_idx ^ 1]);
    }

    if (estado == ESTADO_LOADING_WAVE || estado == ESTADO_IDLE) {
        return;
    }

    uint32_t agora_us = micros();
    if ((int32_t)(agora_us - proxima_amostra_us) < 0) {
        return;
    }

    if (!icm_data_ready()) {
        if (agora_us - ultimo_dado_us > 50000UL) {
            uart_println("# IMU_REINIT");
            HAL_I2C_DeInit(&hi2c1);
            I2C_ClearBus();
            HAL_I2C_Init(&hi2c1);
            icm_iniciar();
            ultimo_dado_us = micros();
        }
        return;
    }

    ultimo_dado_us = micros();
    
    /* Incrementa baseado no alvo anterior para manter exatos 100 Hz,
       em vez de usar agora_us (que acumula o atraso do loop) */
    proxima_amostra_us += 10000UL;

    ciclo_controle();
}
