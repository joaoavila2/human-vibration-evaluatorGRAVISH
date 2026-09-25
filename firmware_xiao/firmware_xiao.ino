#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <ThreeWire.h>
#include <RtcDS1302.h>

// ==========================================
// CONFIGURACOES DE PINOS E ENDERECOS
// ==========================================
const int chipSelectSD = 3;

ThreeWire myWire(1, 2, 0);           // DAT, CLK, RST
RtcDS1302<ThreeWire> Rtc(myWire);

const uint8_t LSM6DS3_ADDR = 0x6A;
const uint8_t CTRL1_XL     = 0x10;
const uint8_t STATUS_REG   = 0x1E;   // bit0 = XLDA (novo dado do acelerometro pronto)
const uint8_t OUTX_L_XL    = 0x28;

// CTRL1_XL: ODR_XL[7:4] | FS_XL[3:2] | LPF2[1] | 0
// ODR 416Hz = 0110. CUIDADO: no LSM6DS3 a ordem de FS_XL NAO e 2/4/8/16g crescente
// (00=2g, 01=16g, 10=4g, 11=8g) -- confira no datasheet antes de trocar.
// Aqui: 0110 11 0 0 = 0x6C -> ODR 416Hz, FS +-8g (11).
const uint8_t CTRL1_XL_CONFIG = 0x6C;
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

File arquivoLog;
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
    Serial.println("Falha ao abrir novo arquivo!");
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

  Rtc.Begin();
  if (!Rtc.GetIsRunning()) Rtc.SetIsRunning(true);

  if (!SD.begin(chipSelectSD)) {
    Serial.println("Falha no SD!");
    while (1);
  }

  Wire.begin();
  Wire.setClock(400000);

  Wire.beginTransmission(LSM6DS3_ADDR);
  Wire.write(CTRL1_XL);
  Wire.write(CTRL1_XL_CONFIG);
  Wire.endTransmission();

  calcularIndiceInicial();  // unica varredura do cartao, feita so aqui
  abrirNovoArquivo();
}

// ==========================================
// LOOP PRINCIPAL
// ==========================================
void loop() {
  // 1. CHECA SE HA DADO NOVO ANTES DE LER (evita duplicar/pular amostra)
  Wire.beginTransmission(LSM6DS3_ADDR);
  Wire.write(STATUS_REG);
  Wire.endTransmission(false);
  Wire.requestFrom(LSM6DS3_ADDR, (uint8_t)1);
  if (Wire.available() != 1) return;
  uint8_t status = Wire.read();
  if (!(status & 0x01)) return;  // XLDA = 0, ainda nao tem amostra nova

  // 2. LEITURA EM BLOCO DOS 6 BYTES
  Wire.beginTransmission(LSM6DS3_ADDR);
  Wire.write(OUTX_L_XL);
  Wire.endTransmission(false);
  Wire.requestFrom(LSM6DS3_ADDR, (uint8_t)6);

  if (Wire.available() == 6) {
    // 2b. GUARDA DE COLISAO: se o buffer de leitura atual ainda esta marcado
    // como "pronto para o SD" (ou seja, o buffer PARCEIRO ainda nao foi
    // consumido), nao empilha amostra nova -- descarta e espera.
    // Na arquitetura atual (escrita 100% bloqueante dentro do proprio loop(),
    // ver passo 4) isso nunca deve disparar, porque o loop() so volta pro
    // passo 1 depois que a escrita anterior ja terminou. Mas se essa parte
    // virar assincrona/DMA no futuro (proximo passo natural pra fechar o
    // duty cycle de vez), essa guarda passa a ser o que evita ler por cima
    // de um buffer que o SD ainda esta consumindo -- mais barato descartar
    // 1 amostra do que corromper o alinhamento de 10 bytes do arquivo.
    if (dadosProntosParaSD) {
      amostrasDescartadas++;
      return;
    }

    bufferLeitura[indiceLeitura].tempoMicros = (uint32_t)(micros() - startMicros);

    uint8_t x_L = Wire.read(); uint8_t x_H = Wire.read();
    uint8_t y_L = Wire.read(); uint8_t y_H = Wire.read();
    uint8_t z_L = Wire.read(); uint8_t z_H = Wire.read();

    bufferLeitura[indiceLeitura].x = (int16_t)(x_H << 8 | x_L);
    bufferLeitura[indiceLeitura].y = (int16_t)(y_H << 8 | y_L);
    bufferLeitura[indiceLeitura].z = (int16_t)(z_H << 8 | z_L);

    indiceLeitura++;
  }

  // 3. TROCA DE BUFFER
  if (indiceLeitura >= TAMANHO_BUFFER) {
    bufferGravacao = bufferLeitura;
    bufferLeitura = (bufferLeitura == bufferA) ? bufferB : bufferA;
    indiceLeitura = 0;
    dadosProntosParaSD = true;
  }

  // 4. GRAVACAO NO SD (flush espacado, nao a cada buffer)
  if (dadosProntosParaSD) {
    arquivoLog.write((const uint8_t*)bufferGravacao, sizeof(PacoteDados) * TAMANHO_BUFFER);

    contadorBuffersDesdeFlush++;
    if (contadorBuffersDesdeFlush >= FLUSH_A_CADA_N_BUFFERS) {
      arquivoLog.flush();
      contadorBuffersDesdeFlush = 0;
    }
    dadosProntosParaSD = false;   // libera a guarda do passo 2b
  }

  // 5. ROTACAO DE ARQUIVO A CADA 1H (reseta referencia de tempo + resincroniza RTC)
  if ((uint32_t)(micros() - startMicros) >= INTERVALO_ROTACAO_US) {
    abrirNovoArquivo();
  }
}
