#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- Configuração LoRa ---
#define SS      18
#define RST     14
#define DI0     26
#define BAND    915E6

// --- Configuração OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RST 16
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RST);

// --- Variáveis de Estado (Simulação do Aeropêndulo) ---
bool rodando = false;
unsigned long tempoInicio = 0;
unsigned long ultimaAmostra = 0;
// IMPORTANTE: 100ms = 10Hz. O LoRa não suporta enviar pacotes de texto a 100Hz (10ms).
int intervaloCsvMs = 100; 

int pacotesEnviados = 0;
String ultimoCmd = "Nenhum";

void setup() {
  Serial.begin(115200);

  // Inicializa OLED (Pinos 4 e 15 para Heltec V2)
  Wire.begin(4, 15);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  display.setCursor(0,0);
  display.println("LoRa Dummy Drone TX");
  display.display();

  // Inicializa LoRa
  LoRa.setPins(SS, RST, DI0);
  if (!LoRa.begin(BAND)) {
    display.println("Erro no LoRa!");
    display.display();
    while (1);
  }

  display.println("Aguardando Python...");
  display.display();
}

void loop() {
  // 1. Verificar comandos vindos do Gateway via LoRa
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String cmd = "";
    while (LoRa.available()) {
      cmd += (char)LoRa.read();
    }
    ultimoCmd = cmd;
    
    // Interpreta os comandos originais do protocolo
    if (cmd == "START" || cmd == "FREE") {
      rodando = true;
      tempoInicio = millis();
      ultimaAmostra = millis();
      pacotesEnviados = 0;
      
      // Responde ao Gateway
      LoRa.beginPacket();
      LoRa.print("# EXP_START");
      LoRa.endPacket();
    } 
    else if (cmd == "STOP") {
      rodando = false;
      LoRa.beginPacket();
      LoRa.print("# PARADO");
      LoRa.endPacket();
    }
    // Para simplificar, ignoramos SEQ, WAVE, CHIRP e apenas disparamos com START.

    // Atualiza OLED
    display.fillRect(0, 16, 128, 48, BLACK);
    display.setCursor(0, 16);
    display.print("Cmd RX: "); display.println(ultimoCmd);
    display.print("Status: "); display.println(rodando ? "Rodando" : "Parado");
    display.display();
  }

  // 2. Gerar dados falsos (dummy) e enviar via LoRa
  if (rodando && (millis() - ultimaAmostra >= intervaloCsvMs)) {
    ultimaAmostra = millis();
    unsigned long t_exp = millis() - tempoInicio;
    
    // Gera dados simulados (senóide para imitar o balanço do pêndulo)
    float angulo = 20.0 * sin(t_exp / 1000.0) + 90.0;
    float u = 10.0 * cos(t_exp / 1000.0);
    float ref = 90.0;

    // Formato exato esperado pelo Python: tempo_ms,angulo_deg,u_pct,referencia
    String csv = String(t_exp) + "," + String(angulo, 2) + "," + String(u, 2) + "," + String(ref, 2);
    
    LoRa.beginPacket();
    LoRa.print(csv);
    LoRa.endPacket();
    
    pacotesEnviados++;

    // Atualiza o display periodicamente
    if (pacotesEnviados % 5 == 0) {
      display.fillRect(0, 32, 128, 32, BLACK);
      display.setCursor(0, 32);
      display.print("Pkts TX: "); display.println(pacotesEnviados);
      display.print("Tx: "); display.println(csv.substring(0, min(16, (int)csv.length())));
      display.display();
    }
  }
}
