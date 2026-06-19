/*
  firmware_pc_in_the_loop.ino
  ========================
  Firmware otimizado para o Aeropêndulo funcionar como "Escravo" (PC-in-the-Loop).
  Neste modo, o PID interno é ignorado. O Arduino apenas lê o sensor a cada 50ms,
  envia via Serial para o Python, e recebe o comando "U=" com o ciclo de trabalho (%).
  Possui um Watchdog embutido para segurança.
*/

#include <Wire.h>
#include <Servo.h>
#include "ICM_20948.h"

#define WIRE_PORT     Wire
#define AD0_VAL       0
#define ESC_PIN       4
#define ESC_MIN       1000
#define ESC_MAX       2000
#define ESC_NEUTRO    1500

#define ODR_DIV       10
#define INTERVALO_US  50000UL        // 50 ms = 20 Hz (Exigência do MPC ANN original)

ICM_20948_I2C myICM;
Servo esc;

float anguloFiltrado = 0.0f;
float gyroBias       = 0.0f;
const float Ts       = INTERVALO_US / 1e6f;

unsigned long proximaAmostra = 0;
unsigned long ultimoDadoUs   = 0;
unsigned long tempoInicio    = 0;
unsigned long ultimo_u_recebido_ms = 0;

enum Estado { IDLE, PC_MPC_MODE };
Estado estado = IDLE;

String rxBuffer = "";

int pctParaUs(float pct) {
  pct = constrain(pct, -80.0f, 80.0f);
  return (int)(1500.0f - 5.0f * pct);
}

float anguloAcelAtual() {
  unsigned long limite = millis() + 200;
  while (millis() < limite) {
    if (myICM.dataReady()) {
      myICM.getAGMT();
      return atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
    }
  }
  return 0.0f;
}

void configurarIMU() {
  ICM_20948_smplrt_t smplrt;
  smplrt.g = ODR_DIV;
  smplrt.a = ODR_DIV;
  myICM.setSampleRate(ICM_20948_Internal_Gyr, smplrt);
  myICM.setSampleRate(ICM_20948_Internal_Acc, smplrt);

  ICM_20948_dlpcfg_t dlp;
  dlp.g = 4;
  dlp.a = 4;
  myICM.setDLPFcfg(ICM_20948_Internal_Gyr | ICM_20948_Internal_Acc, dlp);
  myICM.enableDLPF(ICM_20948_Internal_Gyr, true);
  myICM.enableDLPF(ICM_20948_Internal_Acc, true);
}

bool reiniciarIMU() {
  myICM.begin(WIRE_PORT, AD0_VAL);
  if (myICM.status != ICM_20948_Stat_Ok) return false;
  configurarIMU();
  return true;
}

void calibrarGiroscopio() {
  Serial.println("# CAL_START");
  delay(1000);
  float soma = 0.0f;
  for (int i = 0; i < 500;) {
    if (myICM.dataReady()) {
      myICM.getAGMT();
      soma += myICM.gyrY();
      i++;
      delay(2);
    }
  }
  gyroBias = soma / 500.0f;
  Serial.print("# CAL_BIAS ");
  Serial.println(gyroBias, 4);
  Serial.println("# CAL_OK");
}

void inicializarESC() {
  esc.attach(ESC_PIN, ESC_MIN, ESC_MAX);
  esc.writeMicroseconds(ESC_NEUTRO);
  delay(3000);
  Serial.println("# ESC_OK");
}

String lerComandoSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      String cmd = rxBuffer;
      rxBuffer = "";
      cmd.trim();
      return cmd;
    }
    rxBuffer += c;
  }
  return "";
}

void setup() {
  Serial.begin(500000);
  while (!Serial);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);
  WIRE_PORT.setWireTimeout(3000, true);

  myICM.begin(WIRE_PORT, AD0_VAL);
  if (myICM.status != ICM_20948_Stat_Ok) {
    Serial.println("# ERRO_IMU");
    while (1);
  }

  configurarIMU();
  inicializarESC();
  calibrarGiroscopio();

  Serial.println("# PRONTO_PC_LOOP");
  Serial.println("# Comandos: START_PC | STOP | RECAL");
  estado = IDLE;
}

void loop() {
  String cmd = lerComandoSerial();
  if (cmd.length() > 0) {
    if (cmd == "STOP") {
      esc.writeMicroseconds(ESC_NEUTRO);
      estado = IDLE;
      Serial.println("# PARADO");
      return;
    }

    if (cmd == "RECAL" && estado == IDLE) {
      calibrarGiroscopio();
      Serial.println("# PRONTO");
      return;
    }

    if (cmd == "START_PC" && estado == IDLE) {
      anguloFiltrado = anguloAcelAtual();
      tempoInicio    = millis();
      proximaAmostra = micros() + INTERVALO_US;
      ultimoDadoUs   = micros();
      ultimo_u_recebido_ms = millis();
      
      estado = PC_MPC_MODE;
      Serial.println("# EXP_START_PC");
      return;
    }

    // Recebimento da ação de controle do Python
    if (cmd.startsWith("U=") && estado == PC_MPC_MODE) {
      float u_val = cmd.substring(2).toFloat();
      esc.writeMicroseconds(pctParaUs(u_val));
      ultimo_u_recebido_ms = millis(); // Reseta o watchdog
      return;
    }
  }

  if (estado == IDLE) return;

  // -- MODO PC_MPC_MODE --
  unsigned long agoraMicros = micros();
  
  // Watchdog de Segurança (Trava Motor se Python não responder em 500ms)
  if (millis() - ultimo_u_recebido_ms > 500) {
    esc.writeMicroseconds(ESC_NEUTRO); // Força desligar
    // Continua lendo sensores, mas força o ESC a 0%
  }

  if (agoraMicros < proximaAmostra) return;
  
  if (!myICM.dataReady()) {
    if (micros() - ultimoDadoUs > 50000UL) {
      Serial.println("# IMU_REINIT");
      reiniciarIMU();
      ultimoDadoUs = micros();
    }
    return;
  }

  ultimoDadoUs   = micros();
  proximaAmostra = agoraMicros + INTERVALO_US;
  myICM.getAGMT();

  const float dt   = Ts;
  float anguloAcel = atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
  float gy         = myICM.gyrY() - gyroBias;
  anguloFiltrado   = 0.98f * (anguloFiltrado + (-gy) * dt) + 0.02f * anguloAcel;
  
  float y_med = anguloFiltrado + 90.0f; // Escala 0 a 180

  // Envia a leitura imediatamente ao Python para ele computar a rede neural
  Serial.print("Y:");
  Serial.println(y_med, 2);
}
