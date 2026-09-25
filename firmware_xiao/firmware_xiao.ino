#include <SPI.h>
#include <SdFat.h>
#include <Wire.h>
#include <ThreeWire.h>
#include <RtcDS1302.h>
#include <LSM6DS3.h>

// ==========================================
// CONFIGURACOES DE PINOS E ENDERECOS
// ==========================================
// D7 confirmado por teste real (sketch de referencia a 10Hz que gravou com
// sucesso). Se a fiacao mudar, atualizar aqui.
const int chipSelectSD = 7;

ThreeWire myWire(1, 2, 0);           // DAT, CLK, RST
RtcDS1302<ThreeWire> Rtc(myWire);

SdFs SD;          // SdFs detecta FAT16/32 e exFAT automaticamente
FsFile arquivoLog;

// Acesso a IMU via biblioteca SparkFun (nao registro cru via Wire1): testes
// isolados mostraram que uma transacao I2C real (nao so probe de endereco)
// trava usando Wire1 diretamente nesta placa. A biblioteca e' o mesmo
// caminho usado no sketch de referencia que gravou com sucesso, entao evita
// esse problema por completo em vez de tentar consertar o acesso cru.
LSM6DS3 imu(I2C_MODE, 0x6A);

const uint8_t FULL_SCALE_G    = 8;
const uint16_t SAMPLE_RATE_HZ = 416;

// ==========================================
// ESTRUTURAS BINARIAS (empacotadas, sem padding -- formato estavel pro parser)
// ==========================================
#pragma pack(push, 1)
struct Cabecalho {
  uint32_t magic;        // assinatura do formato, ajuda o parser a validar o arquivo
  uint16_t ano;
  uint8_t  mes, dia, hora, minuto, segundo;
  uint16_t sampleRateHz;
  uint8_t  fullScaleG;
};

struct PacoteDados {
  uint32_t tempoMicros;  // elapsed desde o cabecalho DESTE arquivo, nunca ultrapassa 1h
  int16_t  x, y, z;
};
#pragma pack(pop)

#define MAGIC_CABECALHO 0x31424956UL  // "VIB1"

// ==========================================
// ROTACAO DE ARQUIVO (evita overflow do micros() e resincroniza o RTC)
// ==========================================
// micros() (uint32_t) estoura em ~71,6 min. Rotacionando a cada 1h e usando
// tempo RELATIVO ao inicio do arquivo (nao o micros() bruto), o valor gravado
// nunca chega perto do limite, mesmo que o micros() global estoure no meio.
const uint32_t INTERVALO_ROTACAO_US = 3600000000UL; // 1h

// Indice do proximo arquivo de sessao (F00000.DAT, F00001.DAT, ...).
// Escaneado UMA VEZ no boot (ver setup()) -- nas rotacoes seguintes so incrementa
// em RAM, sem varrer o cartao de novo. Sem isso, o SD.exists() em loop fica mais
// lento a cada rotacao conforme o numero de arquivos existentes cresce ao longo
// do dia, aumentando o "buraco" de gravacao na virada da hora.
uint16_t proximoIndiceArquivo = 0;

// ==========================================
// DOUBLE BUFFERING (ping-pong)
// ==========================================
#define TAMANHO_BUFFER 50   // 50 amostras * 10 bytes = 500 bytes por lote

PacoteDados bufferA[TAMANHO_BUFFER];
PacoteDados bufferB[TAMANHO_BUFFER];

PacoteDados* bufferLeitura  = bufferA;
PacoteDados* bufferGravacao = nullptr;

unsigned int indiceLeitura = 0;
bool dadosProntosParaSD = false;

// flush() e caro (atualiza metadado do FAT) -- nao fazer em todo buffer.
// 8 buffers ~ 1s de dado a 416Hz. Ajuste o trade-off seguranca x desempenho aqui.
const uint8_t FLUSH_A_CADA_N_BUFFERS = 8;
uint8_t contadorBuffersDesdeFlush = 0;

// Amostras descartadas por causa da guarda do item abaixo -- fica no log serial,
// nao para no meio do dado (ver nota no passo 3 do loop()).
uint32_t amostrasDescartadas = 0;

unsigned long startMicros = 0;

// ==========================================
// ACHA O PROXIMO NOME DE ARQUIVO LIVRE (so roda no boot)
// ==========================================
void calcularIndiceInicial() {
  char nome[13];
  while (true) {
    snprintf(nome, sizeof(nome), "F%05u.DAT", proximoIndiceArquivo);
    if (!SD.exists(nome)) break;
    proximoIndiceArquivo++;
  }
}

// ==========================================
// ABRE NOVO ARQUIVO, GRAVA CABECALHO, RESETA REFERENCIA DE TEMPO
// ==========================================
void abrirNovoArquivo() {
  if (arquivoLog) {
    arquivoLog.flush();
    arquivoLog.close();
  }

  char nome[13];
  snprintf(nome, sizeof(nome), "F%05u.DAT", proximoIndiceArquivo);
  proximoIndiceArquivo++;   // so incrementa em RAM -- sem novo SD.exists()

  arquivoLog = SD.open(nome, FILE_WRITE);
  if (!arquivoLog) {
    if (Serial) Serial.println("Falha ao abrir novo arquivo!");
    while (1);
  }

  RtcDateTime agora = Rtc.GetDateTime();  // unico ponto do loop de rotacao que toca o RTC
  Cabecalho cab;
  cab.magic = MAGIC_CABECALHO;
  cab.ano = agora.Year();
  cab.mes = agora.Month();
  cab.dia = agora.Day();
  cab.hora = agora.Hour();
  cab.minuto = agora.Minute();
  cab.segundo = agora.Second();
  cab.sampleRateHz = SAMPLE_RATE_HZ;
  cab.fullScaleG = FULL_SCALE_G;

  arquivoLog.write((const uint8_t*)&cab, sizeof(cab));
  arquivoLog.flush();

  startMicros = micros();
  indiceLeitura = 0;
  contadorBuffersDesdeFlush = 0;
  dadosProntosParaSD = false;
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  // Espera ate 3s o monitor serial conectar (USB nativo do nRF52840 -- se
  // esperasse pra sempre, no campo sem PC conectado o firmware nunca sairia
  // do setup()). Passado o timeout, segue mesmo sem monitor aberto. Todo
  // print daqui pra frente fica atras de "if (Serial)": escrever num USB
  // CDC sem host do outro lado pode enfileirar e travar o loop rodando so
  // na bateria.
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 3000) delay(10);

  if (Serial) Serial.println("== boot ==");

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);  // apagado (LED onboard e ativo em LOW)

  // A IMU embarcada do XIAO Sense fica DESLIGADA por padrao, controlada por
  // este pino. Sem isso, mesmo com I2C/fiacao perfeitos, ela nao responde.
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(100);
  if (Serial) Serial.println("IMU alimentada");

  // SD antes do RTC -- nao ha dependencia logica entre os dois (o RTC so e
  // lido de fato dentro de abrirNovoArquivo(), mais adiante). Retry + clock
  // baixo (1 MHz) porque cartao/fiacao podem nao estar "assentados" logo
  // apos ligar, principalmente na bateria -- mesmo padrao do teste que
  // funcionou.
  bool sdOk = false;
  for (int tentativa = 1; tentativa <= 5 && !sdOk; tentativa++) {
    sdOk = SD.begin(chipSelectSD, SD_SCK_MHZ(1));
    if (!sdOk) delay(300);
  }
  if (!sdOk) {
    if (Serial) {
      Serial.println("Falha no SD! (checar CS, fiacao, cartao)");
      SD.initErrorPrint(&Serial);  // motivo detalhado: cartao vs filesystem
    }
    while (1);
  }
  if (Serial) Serial.println("SD ok");

  Rtc.Begin();
  if (Serial) Serial.println("RTC iniciado");
  if (!Rtc.GetIsRunning()) Rtc.SetIsRunning(true);

  // Configura ODR/faixa ANTES de imu.begin() -- a biblioteca aplica esses
  // valores durante o begin(). Valores validos de accelSampleRate: 13, 26,
  // 52, 104, 208, 416, 833, 1660, 3330, 6660 (Hz). accelRange: 2, 4, 8, 16 (g).
  imu.settings.accelSampleRate = SAMPLE_RATE_HZ;
  imu.settings.accelRange = FULL_SCALE_G;
  imu.settings.gyroEnabled = 0;  // nao usamos giroscopio, desliga pra economizar

  if (imu.begin() != 0) {
    if (Serial) Serial.println("Falha ao iniciar a IMU!");
    while (1);
  }
  if (Serial) Serial.println("IMU configurada");

  calcularIndiceInicial();  // unica varredura do cartao, feita so aqui
  if (Serial) Serial.println("Indice de arquivo calculado, abrindo primeiro arquivo...");
  abrirNovoArquivo();
  if (Serial) Serial.println("Gravando.");

  // Sinal de "pronto" visivel sem Serial Monitor: 3 piscadas de 1s.
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, LOW);
    delay(1000);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
  }
}

// ==========================================
// LOOP PRINCIPAL
// ==========================================
void loop() {
  // 0. HEARTBEAT: alterna o LED a cada 500ms, sem usar delay() (nao pode
  // atrapalhar a taxa de amostragem). So serve pra provar visualmente que
  // o loop() esta realmente rodando -- pisca = vivo, independente de qual
  // nivel (HIGH/LOW) corresponde a "aceso" nessa placa.
  static unsigned long ultimoPisca = 0;
  static bool estadoLed = false;
  if (millis() - ultimoPisca >= 500) {
    ultimoPisca = millis();
    estadoLed = !estadoLed;
    digitalWrite(LED_BUILTIN, estadoLed);
  }

  // 1. LEITURA DIRETA (a biblioteca cuida da transacao I2C internamente).
  // Sem checagem de data-ready por enquanto -- a leitura crua troca menos
  // que 1ms tipicamente, bem abaixo do periodo de 2,4ms a 416Hz, entao nao
  // deve duplicar amostra na pratica. Se precisar do data-ready de volta,
  // da pra usar imu.readRegister() no STATUS_REG (0x1E) igual antes, mas
  // testar separado primeiro pra nao reintroduzir o travamento.
  if (!dadosProntosParaSD) {
    bufferLeitura[indiceLeitura].tempoMicros = (uint32_t)(micros() - startMicros);
    bufferLeitura[indiceLeitura].x = imu.readRawAccelX();
    bufferLeitura[indiceLeitura].y = imu.readRawAccelY();
    bufferLeitura[indiceLeitura].z = imu.readRawAccelZ();
    indiceLeitura++;
  } else {
    amostrasDescartadas++;
  }

  // 2. TROCA DE BUFFER
  if (indiceLeitura >= TAMANHO_BUFFER) {
    bufferGravacao = bufferLeitura;
    bufferLeitura = (bufferLeitura == bufferA) ? bufferB : bufferA;
    indiceLeitura = 0;
    dadosProntosParaSD = true;
  }

  // 3. GRAVACAO NO SD (flush espacado, nao a cada buffer)
  if (dadosProntosParaSD) {
    arquivoLog.write((const uint8_t*)bufferGravacao, sizeof(PacoteDados) * TAMANHO_BUFFER);

    contadorBuffersDesdeFlush++;
    if (contadorBuffersDesdeFlush >= FLUSH_A_CADA_N_BUFFERS) {
      arquivoLog.flush();
      contadorBuffersDesdeFlush = 0;
    }
    dadosProntosParaSD = false;
  }

  // 4. ROTACAO DE ARQUIVO A CADA 1H (reseta referencia de tempo + resincroniza RTC)
  if ((uint32_t)(micros() - startMicros) >= INTERVALO_ROTACAO_US) {
    abrirNovoArquivo();
  }
}
