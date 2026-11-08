# Avaliador de Vibração Humana

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24045956.svg)](https://doi.org/10.5281/zenodo.24045956)

Software de avaliação da exposição humana à vibração de corpo inteiro,
desenvolvido no GRAVI-SH (Departamento de Engenharia Mecânica, UFMG).

Implementa, em Python e com interface gráfica, os métodos de avaliação de risco
da **ISO 2631-5:2018** — Anexos A e E (dose de compressão e fator de risco por
nível discal) e Cláusula 5 / Anexo C (exposições com impacto) — e as avaliações
de saúde e conforto da **ISO 2631-1**, da **NHO-09** (Fundacentro) e da
**Diretiva 2002/44/CE**. Reproduz os resultados de referência do software
oficial da ISO 2631-5 em todos os dígitos publicados.

> Interface em inglês; documentação em português.

---

## Sumário

- [Instalação](#instalação)
- [Visão geral das abas](#visão-geral-das-abas)
- [ISO 2631-5 — Anexos A e E](#iso-2631-5--anexos-a-e-e)
- [ISO 2631-5 — Cláusula 5 e Anexo C](#iso-2631-5--cláusula-5-e-anexo-c)
- [Explorador de sinais](#explorador-de-sinais)
- [Avaliação diária de campo (Directive)](#avaliação-diária-de-campo-directive)
- [Ponderações da ISO 2631-1](#ponderações-da-iso-2631-1)
- [Formatos de entrada](#formatos-de-entrada)
- [Validação](#validação)
- [Decisões de implementação](#decisões-de-implementação)
- [Limitações conhecidas](#limitações-conhecidas)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como citar](#como-citar)
- [Licença](#licença)

---

## Instalação

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
```

### Dados de função de transferência (obrigatório para os Anexos A/E)

As funções de transferência do modelo biomecânico da ISO 2631-5:2018 **não são
distribuídas neste repositório**, por serem material protegido por direitos
autorais da ISO. Elas acompanham a norma como um pacote MATLAB
(`ISO-DIS_2631-5_Transferfunctions_used_in_the_software_MATLAB_format.zip`).
Com o pacote em mãos, converta-o uma única vez:

```bash
python tools/build_data.py ISO-DIS_2631-5_Transferfunctions_used_in_the_software_MATLAB_format.zip data/
```

A conversão transforma ~172 MB de CSV em ~35 MB de arquivos `.npz` em `data/`
(seis grupos de postura e as forças estáticas). Depois disso o zip não é mais
necessário. Sem esses arquivos o programa abre e todas as outras abas
funcionam; só os cálculos dos Anexos A/E (e o gráfico de força do Anexo A na
aba Directive) indicam o arquivo ausente e como gerá-lo.

### Executar

```bash
python run.py          # avaliador
python run_editor.py   # construtor de planilhas de entrada
```

### Gerar executável (Windows)

```bash
pip install pyinstaller
pyinstaller iso2631-5.spec
```

Gera em `dist/` o avaliador (com os dados de `data/` embutidos, ~165 MB) e o
construtor de planilhas (~130 MB). O PyInstaller não faz compilação cruzada:
para rodar no Windows, compile no Windows. Os dados precisam ter sido gerados
antes da compilação.

---

## Visão geral das abas

| Aba | Função |
|---|---|
| **Home** | Ponto de entrada e atalhos |
| **Annex A / Annex E** | Dose de compressão $S_d^A$ e fator de risco $R^A$ (ISO 2631-5) |
| **Clause 5 / Annex C** | Exposições com impacto e perda de contato (ISO 2631-5) |
| **Signal explorer → Signal records** | Análise de registros de aceleração, sem planilha |
| **Signal explorer → Directive** | Dia de trabalho: NHO-09, Diretiva 2002/44/CE, conforto |
| **Signal explorer → Campaign** | Tabela de acompanhamento de uma campanha de medições |
| **Live FFT** | Espectro em tempo real de um acelerômetro via porta serial |
| **Visualizador** | Visualização rápida de sinais brutos (tempo e FFT) |
| **Processamento em Lote** | Cláusula 5 / Anexo C aplicada a uma pasta inteira de arquivos |

Após a primeira análise dos Anexos A/E aparecem também **Results**,
**Frequency domain**, **Time domain** e **Signal processing**.

---

## ISO 2631-5 — Anexos A e E

Calcula, para os seis níveis discais de T12/L1 a L5/S1, a dose de compressão
diária $S_d^A$ (Anexo A) e o fator de risco $R^A$ (Anexo E), a partir de
registros de aceleração triaxial em até quatro pontos de contato: assento, pés,
mãos e encosto.

### Cadeia de cálculo

1. **Espectro** — FFT do registro inteiro, janela retangular, espectro de
   amplitude unilateral; banda limitada a 0,5–100 Hz.
2. **Aceleração → deslocamento** — divisão por $-\omega^2$. As funções de
   transferência do Anexo A são deslocamento→força.
3. **Forças intervertebrais** — Fórmula (A.1), somando os quatro pontos de
   contato e as três direções. A função de transferência (0,25–30 Hz) é
   interpolada por spline em módulo e fase e zerada fora da faixa.
4. **Domínio do tempo** — IFFT; o valor médio é substituído pela força
   estática do modelo de elementos finitos.
5. **Picos de compressão** — extremos negativos entre cruzamentos do valor
   médio (Figura A.2).
6. **Dose** — Fórmulas (A.2) e (A.3), expoente 6 (Palmgren-Miner).
7. **Fator de risco** — Fórmulas (E.1) a (E.4), com resistência última
   decrescente com a idade e subtração da tensão estática.

Nenhuma ponderação em frequência entra nessa cadeia: a dependência em
frequência já está nas funções de transferência. A aba **Signal processing**
documenta, para cada excitação, todos os parâmetros efetivamente usados
(amostras, taxa, resolução, cortes, bins retidos).

### Verificações automáticas

- **Critério dos 9,81 m/s².** Os Anexos A/E só se aplicam se o pico de
  aceleração em z, após os filtros limitadores de banda, não exceder 9,81 m/s².
  O resultado da verificação aparece acima da tabela de resultados; os valores
  continuam sendo calculados, mas marcados como fora do domínio de validade.
- **Base de tempo comum.** A Fórmula (A.1) soma espectros dos pontos de
  contato, o que exige grade de frequência comum. Taxas de amostragem
  diferentes ou diferença de duração acima de 1% são recusadas; abaixo de 1%,
  os registros são truncados ao menor comprimento e um aviso informa quanto foi
  descartado (`core.TIME_BASE_TOLERANCE`).
- **Encosto marcado como `Zero`.** O encosto sem medição entra como sinal nulo,
  e o programa emite um aviso (tela, PDF e aba `Warnings` do xlsx). A
  ISO 2631-5:2018, A.3.1, admite substituir o encosto não medido pelo sinal do
  assento, e o executável oficial faz essa substituição; com sinal nulo os
  resultados podem ficar abaixo da referência.

### Saídas

Tabela de resultados por nível discal, gráficos de espectro de excitação,
funções de transferência, espectro de força e força no tempo (com seleção de
ponto de contato, eixo e nível discal), exportação para planilha `.xlsx` e
relatório PDF.

---

## ISO 2631-5 — Cláusula 5 e Anexo C

Caminho da norma para exposições **com** perda de contato com o assento —
justamente as que o critério dos 9,81 m/s² exclui dos Anexos A/E. A cadeia é
independente da anterior, porque os requisitos são incompatíveis:

| | Cláusula 5 / Anexo C | Anexos A / E |
|---|---|---|
| Passa-alta | 0,1 Hz, 2 polos | 0,5 Hz |
| Passa-baixa | 80 Hz, 4 polos | 100 Hz |
| Filtros da ISO 2631-1 | não aplicados | só na verificação de validade |
| Direções | z, no assento | x, y, z em quatro pontos |
| Excitação | aceleração | deslocamento |

O corte inferior em 0,1 Hz preserva a fase de queda livre que antecede o
impacto; um corte em 0,5 Hz a eliminaria e erraria o pico em até 1 g.

Cadeia: condicionamento → função de transferência assento-coluna (Fórmula 1:
um zero e seis polos) → dose de aceleração (Fórmula 3) → normalização diária
(Fórmulas 4 e 5) → dose de compressão (C.1) → variável R (C.3) → probabilidade
de dano pelo modelo de Weibull (C.5).

O fator $m_z$ é parametrizável: fração de massa acima da pelve e área da placa
terminal vêm com os valores de referência da norma por sexo, editáveis; massa
de equipamento no tronco ou na cabeça entra em campo próprio.

**Detecção de perda de contato.** A norma pede que os trechos sem contato
sejam identificados por instrumento (chave de contato, vídeo). Na ausência
dele, o programa oferece uma **heurística que não faz parte da norma**: marca
como queda livre os trechos com aceleração vertical abaixo de −7 m/s² por pelo
menos 0,15 s (`annexc.FREEFALL_THRESHOLD`, `annexc.FREEFALL_MIN_DURATION`). A
opção pode ser desligada, e os trechos marcados devem ser conferidos.

**Processamento em lote.** A aba *Processamento em Lote* aplica a mesma
avaliação a todos os arquivos de uma pasta (cada arquivo = uma avaliação),
com exportação da tabela em CSV e, opcionalmente, um PDF por arquivo.

**Postura de cadeira de rodas (experimental).** A aba oferece um protótipo
baseado no modelo de Kawai & Matsuoka (SAE 2000-01-0645; ASME J. Mech. Des.
125, 2003). **Não faz parte da cadeia validada e não deve ser usado em
laudo**; as correções aplicadas aos parâmetros publicados estão documentadas
em `iso26315/annexc_wheelchair.py`.

---

## Explorador de sinais

### Signal records

Análise direta de registros de aceleração, sem planilha. Não calcula dose
nem fator de risco: serve para triagem e caracterização dos sinais.

Carrega vários arquivos ao mesmo tempo, sobrepostos, com seleção individual.
Visualizações disponíveis:

- série temporal, RMS acumulado (estacionariedade), histograma/CDF;
- espectro (cinco estimadores, abaixo), fase, espectrograma;
- coerência, coerência múltipla, transmissibilidade H1 e SEAT, H1 por classe
  de amplitude e em bandas de 1/3 de oitava;
- espectro de resposta ao choque (SRS), valores por intervalo
  (a_w / MTVV / VDV), ODS e resposta dos filtros de ponderação.

Métricas por arquivo: RMS, pico, fator de crista, VDV, curtose, assimetria,
frequência dominante; métricas ponderadas da ISO 2631-1 (a_w, A(8), MTVV, VDV);
avaliação pela Diretiva 2002/44/CE; zona de saúde do Anexo C; validade dos
Anexos A/E.

Registros consecutivos podem ser concatenados em um único sinal
(**Join selected → 1 signal**), desde que tenham a mesma taxa de amostragem.
Exporta relatório PDF (página de estatísticas + um gráfico por página, à
escolha) e espectro em CSV.

**Curvas de referência.** Sobre a transmissibilidade medida podem ser
desenhadas, apenas para comparação visual, a curva assento-cabeça da
ISO 5982:2001 (Anexo C; adultos, sem encosto) e o modelo de um grau de
liberdade de Giacomin (2007), com parâmetros escalonados pela massa do
sujeito — útil para crianças, fora do domínio da ISO 5982. O programa avisa
quando a entrada sai do domínio de validade de cada referência.

### Estimação espectral

| Método | Descrição |
|---|---|
| Single block | FFT complexa do registro inteiro, janela retangular — a mesma do cálculo dos Anexos A/E |
| Welch — amplitude | Média de periodogramas, corrigida em amplitude |
| Welch — PSD | Densidade espectral de potência, (m/s²)²/Hz |
| Welch — RMS por banda | PSD integrada em bandas de largura escolhida |
| 1/3 de oitava | RMS em bandas de 1/3 de oitava (centros da ISO 266) |

Janelas: retangular, Hann, Hamming, Flat top, Blackman-Harris. Verificação:
senoide de amplitude 1,0 a 5 Hz → pico 1,0000 (Flat top) e 0,9959 (Hann); PSD
de ruído branco de variância unitária a 500 Hz → 0,00399 (teórico 0,00400).

Esses estimadores alteram **apenas a visualização**. A média de Welch descarta
a fase, sem a qual não é possível reconstruir a força no tempo; o cálculo dos
Anexos A/E usa sempre o espectro complexo de bloco único.

### Campaign

Tabela de acompanhamento de uma campanha de medições (sujeito, idade, massa,
posição, data, arquivos, critério e valores principais), preenchida a partir
da aba Directive e salva/recarregada em CSV.

### Live FFT

Espectro em tempo real de um acelerômetro ligado à porta serial. Protocolo:
amostras de 14 bytes little-endian — `uint16` de sincronismo `0x55AA` seguido
de três `float32` (x, y, z em g). O firmware do sensor não faz parte deste
repositório.

---

## Avaliação diária de campo (Directive)

Monta um dia de trabalho a partir dos arquivos de um registrador de campo
(vários arquivos da mesma unidade, selecionados de uma vez) e o avalia por
vários critérios em paralelo. As medições são tratadas como **contínuas**: os
blocos internos são fatias consecutivas de um mesmo sinal, e a ponderação, o
MTVV e a Cláusula 5 operam sobre o sinal contínuo, sem transientes de borda.

**Correção de ganho pela gravidade.** Em repouso, o módulo da aceleração
medida deve ser 9,784 m/s² (valor local de Belo Horizonte); o ganho de cada
unidade é ajustado por esse valor, e o eixo que contém a gravidade é tomado
como vertical. A gravidade é removida antes de qualquer métrica.

### Critérios

- **NHO-09** — o dia é tratado como uma única componente de exposição, com o
  valor eficaz agregado pela energia de todos os blocos. A média aritmética
  prescrita pela NHO-09 para repetições de uma mesma operação é mostrada como
  verificação. O tempo de exposição padrão é o período medido; a extrapolação
  para outra jornada é opção explícita, com as relações de raiz quadrada e
  raiz quarta do Anexo B da ISO 2631-1 lado a lado.
- **Diretiva 2002/44/CE** — pelo **maior eixo** (texto legal para corpo
  inteiro, em A(8) e VDV) ou pela **soma vetorial** (mais conservadora, e uma
  divergência que deve ser declarada). Opcionalmente, o VDV pode ser comparado
  aos limites da zona de cautela da ISO 2631-1 (8,5 / 17) em vez dos da
  diretiva (9,1 / 21).
- **Método do laboratório** — reprodução do aplicativo WBV_HSE_v16, com
  ponderação, fatores k, troca de eixos e tempo de RMS configuráveis. Validado
  contra dois casos com reprodução exata. O VDV é sempre calculado de forma
  correta; a opção *Reproduce the app VDV defects* existe só para conferência.

### Posição do sensor (ISO 2631-1)

- **Assento** — avaliação de saúde (Wd / Wd / Wk; k = 1,4 / 1,4 / 1,0).
- **Encosto** — avaliação de conforto (Wc / Wd / Wd; k = 0,8 / 0,5 / 0,4). O
  eixo x com Wc e k = 0,8 é mostrado como valor de saúde apenas
  **informativo**, conforme a nota de 7.2.3. O eixo perpendicular ao encosto
  é identificado pela inclinação da gravidade.
- **Pés** — avaliação de conforto (Wk; k = 0,25 / 0,25 / 0,4).

O conforto usa o valor total $a_v$, comparado às indicações aproximadas do
Anexo C da ISO 2631-1 (que a norma declara não serem limites).

### Saídas

Faixa de resumo com todos os critérios e suas cores de faixa (critérios não
normativos para a posição em cinza), quadro de A(8) por eixo, *dashboard* por
critério, comparação eixo a eixo com o método do laboratório, contribuição de
cada período para a energia diária, linha do tempo com detecção heurística de
ocupação do assento (apenas informativa). Relatório PDF com metadados do
sujeito e da medição (identificação, idade, operador, sensor, número de série,
notas).

---

## Ponderações da ISO 2631-1

Implementadas a partir das Tabelas A.1 e A.2 da ISO 2631-1: **Wk, Wd, Wc, We,
Wj, Wf e Wb**, cada uma como produto dos quatro fatores da Seção A.2
(passa-alta e passa-baixa Butterworth de dois polos, transição
aceleração-velocidade e degrau ascendente) no domínio de Laplace.

Erro relativo mediano frente às tabelas da norma em bandas de 1/3 de oitava:
0,05% (Wk), 0,04% (Wd), 0,02% (Wc). Os valores seguem a
ISO 2631-1:1997/Amd.1:2010, que corrigiu o fator de Wd a 10 Hz de 212 para 202.

O painel *Frequency weighting* (aba Frequency domain e explorador) permite
escolher o filtro por eixo a partir de predefinições — sem ponderação
(exigência da ISO 2631-5), saúde, conforto no assento e no encosto, enjoo de
movimento, BS 6841 — ou livremente em **Custom**, com aviso quando a
configuração não corresponde a nenhuma norma. As métricas exibidas seguem
sempre a configuração ativa. **Nenhuma dessas ponderações entra em $S_d$ ou
$R$.**

---

## Formatos de entrada

### Planilhas dos Anexos A/E

Mesmo formato do software original da ISO: arquivo `.xls`/`.xlsx` com as abas
`Exposure` (medições), `Pattern` (padrões diários) e `Year` (histórico anual).
Os arquivos `structure_orient.xls` e `structure_indiv.xls` do pacote de
exemplos da norma servem de modelo, e o **construtor de planilhas**
(`run_editor.py`) as monta sem edição manual: detecta a duração de cada
registro, totaliza as horas diárias e mostra em qual dos 30 modelos
antropométricos cai cada combinação de massa e altura.

Caminhos absolutos dentro da planilha (ex.: `D:\Risk Assessment\...`) são
relocalizados automaticamente pelo nome do arquivo na pasta da planilha, ou na
pasta indicada em **Pasta das acelerações**.

Para ponto de contato não medido, escreva `Zero` na célula correspondente.

### Registros de aceleração

| Formato | Descrição |
|---|---|
| ASCII 4 colunas (`.txt`, `.asc`, `.dat`, `.csv`) | tempo (s), $a_x$, $a_y$, $a_z$ (m/s²), sem cabeçalho |
| CSV `DateTime,AX,AY,AZ` | registrador de campo contínuo |
| CSV `Time_ms,AX,AY,AZ` | ensaios de bancada |
| Binário VIB1 (`.dat`) | registrador de campo; cabeçalho de 14 bytes + pacotes de 10 bytes (ver `iso26315/xiao_parser.py`) |
| MATLAB `.m` (legado) | formato antigo em blocos, mantido para reanálise de dados históricos |

---

## Validação

Os Anexos A/E reproduzem os resultados de referência do software oficial
(`Risk_Assessment-Examples/Results/Results_ISO.xls`).

**Análise orientadora** — pá carregadeira, grupo de postura 3, 110 kg, 1,85 m
($R^A$ por nível discal):

| | T12/L1 | L1/L2 | L2/L3 | L3/L4 | L4/L5 | L5/S1 |
|---|---|---|---|---|---|---|
| Obtido | 0,98423 | 1,08503 | 1,22263 | 1,31024 | 1,28424 | 1,16975 |
| Referência | 0,98423 | 1,085 | 1,2226 | 1,3102 | 1,2842 | 1,1698 |

$S_d^A$ no nível crítico: 0,57776 MPa (referência: 0,57776 MPa).

**Análise individual** — bulldozer, mãos e pés sem medição, nascido em 1950:

| | T12/L1 | L1/L2 | L2/L3 | L3/L4 | L4/L5 | L5/S1 |
|---|---|---|---|---|---|---|
| Obtido | 2,76566 | 2,68934 | 2,65014 | 2,66431 | 2,58059 | 2,39230 |
| Referência | 2,7656 | 2,6893 | 2,6501 | 2,6643 | 2,5806 | 2,3923 |

**Todos os casos de referência disponíveis:**

| Caso | Massa | Erro relativo máximo | Situação |
|---|---|---|---|
| Orientadora (WL2) | 110 kg | 3,9×10⁻⁵ | exato (arredondamento) |
| BD1 individual | 110 kg | 2,2×10⁻⁵ | exato |
| indiv1 a indiv4 | 110 kg | 7×10⁻⁶ a 3,3×10⁻⁵ | exato |
| indiv5, indiv6 | 70 a 110 kg | 5,6×10⁻⁴ a 6,4×10⁻⁴ | aproximado (ver Limitações) |

**Cláusula 5 / Anexo C.** O exemplo numérico do Anexo C (homem, 82 kg,
120 dias/ano dos 20 aos 40 anos, cinco impactos de 40 m/s² por dia) reproduz
$S_d$ = 1,623 MPa e R = 1,22 com o $m_z$ = 0,029 usado pela norma (com o valor
exato 0,02870: 1,606 MPa e 1,207). A Tabela C.2 é reproduzida nos dois sexos, e
o ganho DC da Fórmula (1) resulta 1,000000.

### Testes

```bash
python tests/test_reference.py <pasta-com-as-aceleracoes-de-exemplo>
python -m pytest tests/
```

`test_reference.py` precisa dos arquivos de aceleração do pacote de exemplos
da norma (`WL2s.txt`, `WL2b.txt`, `WL2h.txt`, `WL2f.txt`, `BD1sZ.asc`) e dos
dados de `data/`. Nota: dois exemplos da própria norma (`BD1sZ.asc` e
`WL3s.asc`) não atendem ao critério dos 9,81 m/s².

---

## Decisões de implementação

**Interpolação da fase.** A função de transferência do modelo tem um salto de
2π na fase perto de 20 Hz. O software oficial interpola a fase *enrolada*, e a
spline atravessa o salto pelo caminho errado (erro de até 200% no valor
complexo em ~6% dos pontos). Por fidelidade à referência, esse é o
comportamento padrão. A opção **Unwrap transfer function phase before
interpolating** (`core.UNWRAP_PHASE`) corrige a interpolação:

| | T12/L1 | L1/L2 | L2/L3 | L3/L4 | L4/L5 | L5/S1 |
|---|---|---|---|---|---|---|
| Fase enrolada (padrão) | 0,98423 | 1,08503 | 1,22263 | 1,31024 | 1,28424 | 1,16975 |
| Fase desenrolada | 0,98810 | 1,08905 | 1,22754 | 1,31568 | 1,29097 | 1,18076 |
| Diferença | 0,39% | 0,37% | 0,40% | 0,42% | 0,52% | 0,94% |

Ao publicar resultados, declare qual modo foi usado. A diferença de até ~1% é
uma estimativa do erro numérico presente no valor oficial.

**Estrutura dos dados de função de transferência.** Decodificada a partir do
pacote MATLAB, cuja documentação é incompleta. Cada arquivo
`BSKUefuM_{k}_{el}_{ed}` é uma matriz 3 × 847: três direções de força
(x, y, compressão) por sete blocos de 121 colunas (frequência 0,25–30 Hz e seis
níveis discais; em cada bloco, a primeira coluna é a força estática e as 120
restantes a função complexa). `el` = ponto de contato (assento, pés, mãos,
encosto), `ed` = direção da excitação, `k` = índice antropométrico de 1 a 30.
As forças estáticas formam 7 posturas × 10 antropometrias × 6 discos.

**Memória.** Em históricos longos, o resultado completo (séries de força e
espectros) é mantido apenas para as quatro primeiras combinações
antropometria × exposição, usadas nos gráficos; as demais guardam só o
necessário para dose e risco. O caso `indiv6` (treze mudanças de massa) roda
com pico de ~2,1 GB e resultado inalterado.

---

## Limitações conhecidas

**Mapeamento massa → modelo antropométrico.** O índice de 30 posições não está
univocamente determinado pelos dados disponíveis. O mapeamento adotado
reproduz os casos com massa variável (`indiv5`, `indiv6`) com erro ~6×10⁻⁴,
cerca de vinte vezes o dos casos exatos, e posiciona todas as massas testadas
no mesmo bloco do índice, inclusive combinações de IMC baixo — o que contraria
a leitura desse bloco como classe de IMC. Detalhes em
`core.ANTHROPOMETRY_NOTE`.

**Subclasse do índice antropométrico.** O terceiro fator do índice não é
documentado. Os casos de referência batem com subclasse 1 (padrão), inclusive
quando o RMS do sinal sugeriria outra classe; a interface permite alterá-la.

**`t_m` vem da planilha.** Como no software oficial, a duração da medição é
lida do campo *Measurement duration* da aba `Exposure`, mesmo quando difere da
duração real do arquivo. Mantido por compatibilidade; o construtor de
planilhas preenche o valor a partir do arquivo.

**Grupo de postura 6.** Não está presente no pacote de funções de
transferência e, portanto, não está disponível.

**Heurísticas fora das normas.** Detecção de perda de contato (Cláusula 5),
detecção de ocupação do assento (Directive) e o modelo de cadeira de rodas são
auxílios de engenharia, não procedimentos normativos.

---

## Estrutura do repositório

```
run.py                 avaliador (interface principal)
run_editor.py          construtor de planilhas
iso2631-5.spec         configuração do PyInstaller
iso26315/
  core.py              Anexos A/E: leitura, FFT, funções de transferência, dose, risco
  analysis.py          orquestração da análise e cache de resultados
  annexc.py            Cláusula 5 / Anexo C
  annexc_wheelchair.py protótipo experimental de cadeira de rodas
  nho09.py             ponderações ISO 2631-1 e NHO-09
  directive.py         Diretiva 2002/44/CE
  labmethod.py         reprodução do método do laboratório (WBV_HSE_v16)
  positions.py         tratamento por ponto de contato (assento, encosto, pés)
  summary.py           reunião de todos os critérios
  references.py        curvas de referência (ISO 5982, Giacomin)
  fielddata.py         leitura de registros de campo
  continuity.py        continuidade entre blocos de um registro contínuo
  xiao_parser.py       leitor do formato binário VIB1
  workbook.py          leitura e escrita de planilhas
  pdf_report.py        relatórios PDF
  gui.py, *_tab.py, *_panel.py, explorer.py, ...   interface gráfica
tools/build_data.py    conversão do pacote de funções de transferência da ISO
tests/                 validação contra a referência e testes unitários
data/                  (não versionado) funções de transferência geradas
CHANGELOG.md           histórico de mudanças
CITATION.cff           metadados de citação
LICENSE                licença MIT
```

---

## Como citar

> Ávila, J. P. R. *Avaliador de Vibração Humana* [software].
> GRAVI-SH, Universidade Federal de Minas Gerais, 2026.
> https://doi.org/10.5281/zenodo.24045956

Indique a versão usada (mostrada no título da janela). Os metadados de citação
estão em [`CITATION.cff`](CITATION.cff); o GitHub os exibe no botão
*Cite this repository*.

As normas implementadas devem ser citadas separadamente:
ISO 2631-1:1997 (e Amd.1:2010), ISO 2631-5:2018, NHO-09 (Fundacentro) e
Diretiva 2002/44/CE.

---

## Licença

Código distribuído sob a licença MIT (ver [`LICENSE`](LICENSE)). A licença
cobre apenas o código deste repositório. Os dados de função de transferência da
ISO 2631-5 não fazem parte dele e devem ser obtidos junto com a norma.
