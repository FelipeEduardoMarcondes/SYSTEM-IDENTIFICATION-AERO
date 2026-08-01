#include <Wire.h>
#include <Servo.h>
#include "ICM_20948.h"

// ── Pinos e limites do ESC ────────────────────────────────────────────────────
#define WIRE_PORT     Wire
#define AD0_VAL       0
#define ESC_PIN       4
#define ESC_MIN       1000
#define ESC_MAX       2000
#define ESC_NEUTRO    1500

// ── Temporização ──────────────────────────────────────────────────────────────
#define ODR_DIV       10
#define INTERVALO_US  10000UL        // 10 ms = 100 Hz

ICM_20948_I2C myICM;
Servo esc;

// ── Filtro complementar ───────────────────────────────────────────────────────
float anguloFiltrado = 0.0f;
float gyroBias       = 0.0f;

// ── Temporização ──────────────────────────────────────────────────────────────
unsigned long proximaAmostra = 0;
unsigned long ultimoDadoUs   = 0;

// ── Buffer de recepção serial ─────────────────────────────────────────────────
String rxBuffer = "";

int pctParaUs(float pct) {
  pct = constrain(pct, -10.0f, 80.0f);
  return (int)(1500.0f - 5.0f * pct);
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
}

void inicializarESC() {
  esc.attach(ESC_PIN, ESC_MIN, ESC_MAX);
  esc.writeMicroseconds(ESC_NEUTRO);
  delay(3000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Leitura não-bloqueante da serial (IDÊNTICA AO aeropendulo_firmware.ino)
// ─────────────────────────────────────────────────────────────────────────────
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
  Serial.begin(115200);
  while (!Serial);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);
  WIRE_PORT.setWireTimeout(3000, true);

  myICM.begin(WIRE_PORT, AD0_VAL);
  if (myICM.status != ICM_20948_Stat_Ok) {
    while (1);
  }

  configurarIMU();
  inicializarESC();
  calibrarGiroscopio();

  proximaAmostra = micros() + INTERVALO_US;
  ultimoDadoUs   = micros();
}

void loop() {
  // 1. COMUNICAÇÃO: Ler u e responder com y_med
  String cmd = lerComandoSerial();
  if (cmd.length() > 0) {
    float u_pct = cmd.toFloat();
    esc.writeMicroseconds(pctParaUs(u_pct));
    
    // Devolve imediatamente o ângulo filtrado que foi lido na etapa de baixo
    float y_med = anguloFiltrado + 90.0f;
    Serial.println(y_med, 2);
  }

  // 2. LEITURA IMU NÃO-BLOQUEANTE EM EXATOS 100 Hz (IDÊNTICO AO aeropendulo_firmware.ino)
  unsigned long agoraMicros = micros();
  if (agoraMicros < proximaAmostra) return;
  
  if (!myICM.dataReady()) {
    if (micros() - ultimoDadoUs > 50000UL) {
      reiniciarIMU();
      ultimoDadoUs = micros();
    }
    return;
  }

  ultimoDadoUs   = micros();
  proximaAmostra = agoraMicros + INTERVALO_US;
  myICM.getAGMT();

  const float dt   = INTERVALO_US / 1e6f;
  float anguloAcel = atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
  float gy         = myICM.gyrY() - gyroBias;
  anguloFiltrado   = 0.98f * (anguloFiltrado + (-gy) * dt) + 0.02f * anguloAcel;
}
