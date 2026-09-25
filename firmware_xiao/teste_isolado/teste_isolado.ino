// Teste isolado v3: bloco real de SD -> RTC -> IMU(Wire1), igual ao
// firmware principal, mas com Serial.flush() depois de cada print pra
// garantir que a linha realmente saiu pela USB antes de qualquer coisa
// que possa travar. A ultima linha que aparecer e' o ponto exato do
// travamento, com certeza.

#include <SPI.h>
#include <SdFat.h>
#include <Wire.h>
#include <ThreeWire.h>
#include <RtcDS1302.h>

const int chipSelectSD = 7;
ThreeWire myWire(1, 2, 0);
RtcDS1302<ThreeWire> Rtc(myWire);
SdFs SD;

const uint8_t LSM6DS3_ADDR = 0x6A;
const uint8_t CTRL1_XL = 0x10;
const uint8_t CTRL1_XL_CONFIG = 0x6C;

void log(const char* msg) {
  Serial.println(msg);
  Serial.flush();  // forca transmitir AGORA, nao deixa em buffer
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) delay(10);

  log("1: boot ok");

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(100);
  log("2: IMU alimentada");

  bool sdOk = false;
  for (int tentativa = 1; tentativa <= 5 && !sdOk; tentativa++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "3.%d: tentando SD.begin()...", tentativa);
    log(buf);
    sdOk = SD.begin(chipSelectSD, SD_SCK_MHZ(1));
    if (!sdOk) delay(300);
  }
  log(sdOk ? "4: SD ok" : "4: SD falhou (mas nao travou)");

  log("5: chamando Rtc.Begin()...");
  Rtc.Begin();
  log("6: Rtc.Begin() retornou");

  if (!Rtc.GetIsRunning()) Rtc.SetIsRunning(true);
  log("7: RTC checado/ligado");

  log("8: chamando Wire1.begin()...");
  Wire1.begin();
  Wire1.setClock(400000);
  log("9: Wire1.begin() retornou");

  log("10: iniciando transacao I2C com a IMU...");
  Wire1.beginTransmission(LSM6DS3_ADDR);
  Wire1.write(CTRL1_XL);
  Wire1.write(CTRL1_XL_CONFIG);
  uint8_t erro = Wire1.endTransmission();
  char buf2[48];
  snprintf(buf2, sizeof(buf2), "11: endTransmission retornou, codigo=%d", erro);
  log(buf2);

  log("12: CHEGOU ATE O FIM DO TESTE");
}

void loop() {
  static unsigned long t = 0;
  static bool s = false;
  if (millis() - t >= 500) {
    t = millis();
    s = !s;
    digitalWrite(LED_BUILTIN, s);
    Serial.println("loop rodando...");
    Serial.flush();
  }
}
