# Avaliador de Vibração Humana

> Interface em inglês; documentação em português.

Aplicativo do GRAVI-SH (UFMG) para avaliação de exposição humana à vibração,
com interface gráfica, gráficos no domínio da frequência e do tempo, e
exportação para planilha. Nasceu como reimplementação em Python do software de
avaliação de risco dos Anexos A e E da ISO 2631-5, e cresceu para cobrir também
a Cláusula 5/Anexo C da mesma norma (exposições com perda de contato), as
ponderações e critérios de conforto da ISO 2631-1, a NHO-09 e a Diretiva
2002/44/CE, um explorador de sinais unificado (aba *Signal records*) que aceita
qualquer formato de arquivo e oferece análise completa ISO 2631-1, e a aba
*Directive* para avaliação por dia de trabalho com comparação contra o método do
laboratório — a ISO 2631-5 propriamente dita é hoje um dos vários caminhos de
avaliação disponíveis, não o escopo inteiro do programa.

## ISO 2631-5, Anexos A e E

Calcula a dose de compressão diária $S_d^A$ (Anexo A) e o fator de risco $R^A$
(Anexo E) para os seis níveis discais de T12/L1 a L5/S1.

## Estado da validação

Reproduz os resultados de referência do software original
(`Risk_Assessment-Examples/Results/Results_ISO.xls`) em todos os dígitos
publicados.

**Caso 1 — análise orientadora** (pá carregadeira, postura grupo 3, 110 kg / 1,85 m):

| | T12/L1 | L1/L2 | L2/L3 | L3/L4 | L4/L5 | L5/S1 |
|---|---|---|---|---|---|---|
| Obtido | 0,98423 | 1,08503 | 1,22263 | 1,31024 | 1,28424 | 1,16975 |
| Referência | 0,98423 | 1,085 | 1,2226 | 1,3102 | 1,2842 | 1,1698 |

$S_d^A$ no nível crítico: 0,57776 MPa (referência: 0,57776).

**Caso 2 — análise individual** (bulldozer, mãos e pés sem medição, nascido em 1950):
2,76566 / 2,68934 / 2,65014 / 2,66431 / 2,58059 / 2,39230 contra a referência
2,7656 / 2,6893 / 2,6501 / 2,6643 / 2,5806 / 2,3923.

Rode `python tests/test_reference.py` para reexecutar a validação (ajuste o
caminho `ACC` no topo do arquivo para a sua pasta de acelerações).

## Instalação

```bash
pip install -r requirements.txt
```

As funções de transferência da ISO 2631-5:2018 (Annex E) não estão incluídas no
repositório por serem material protegido por direitos autorais. Para gerá-las,
obtenha o pacote MATLAB junto com a norma e execute:

```bash
python tools/build_data.py ISO-DIS_2631-5_Transferfunctions_used_in_the_software_MATLAB_format.zip data/
```

A conversão roda uma vez só e gera ~35 MB de `.npz` em `data/`. Depois:

```bash
python run.py
```

## Executável

```bash
pip install pyinstaller
pyinstaller iso2631-5.spec
```

Saem dois arquivos em `dist/`: `AvaliadorDeVibracaoHumana.exe` (o avaliador, com
os dados embutidos, perto de 165 MB) e `AvaliadorDeVibracaoHumana-Planilhas.exe`
(o construtor, perto de
130 MB). Precisa ser gerado no Windows para rodar no Windows — o PyInstaller não
faz compilação cruzada.

O spec já foi testado: compila e o binário abre com os dados embutidos
corretamente. Dois detalhes que valem a menção, porque custaram uma tentativa
cada: `scipy.interpolate` puxa `scipy.optimize`, então nenhum submódulo do scipy
pode entrar na lista de exclusões; e a dependência de `scipy.stats` foi removida
em favor de `statistics.NormalDist`, da biblioteca padrão.

Se você não quiser instalar nada, o repositório traz
`.github/workflows/build.yml`. Dando push no GitHub, o `.exe` sai pronto em
Actions → build-windows → Artifacts, compilado em uma máquina Windows de
verdade.

## Construtor de planilhas

Aplicativo separado, para montar os arquivos de entrada sem editar planilha na
mao:

```bash
python run_editor.py
```

Tem tres secoes, na ordem em que a norma organiza os dados. **Exposicoes** sao as
medicoes: voce escolhe o arquivo do assento e ele detecta a duracao do sinal
sozinho, ja preenchendo o `t_m`. **Padroes diarios** combinam exposicoes num dia
de trabalho, e o rodape mostra o total em horas para voce conferir. **Anos**
gera a lista toda de uma vez a partir de um intervalo, e depois voce edita as
linhas que fogem da regra — ganho de peso, mudanca de rotina, afastamento (basta
deixar o ano sem padrao).

Ele tambem abre planilhas existentes, entao da para partir de um exemplo e
adaptar em vez de comecar do zero.

O rodape da secao de anos mostra em qual dos 30 modelos antropometricos a
combinacao de massa e altura cai. Util para saber se uma mudanca de massa
corporal de fato muda o modelo ou se cai na mesma faixa.

## Formato de entrada

Usa as mesmas planilhas do software original, com as abas `Exposure`, `Pattern` e
`Year`. Os arquivos `structure_orient.xls` e `structure_indiv.xls` do pacote de
exemplos servem de modelo.

Os caminhos de acelerações dentro da planilha costumam apontar para `D:\Risk
Assessment\...`. O aplicativo relocaliza os arquivos automaticamente pelo nome,
procurando na pasta da planilha; se estiverem em outro lugar, use o botão
**Pasta das acelerações**.

As séries de aceleração são ASCII de quatro colunas, sem cabeçalho: tempo em
segundos, depois $a_x$, $a_y$, $a_z$ em m/s². Para pontos de contato sem medição,
escreva `Zero` na célula correspondente.

## Como o cálculo funciona

1. **Leitura e FFT** — espectro de amplitude unilateral, banda limitada entre
   0,5 e 100 Hz conforme a Cláusula 5.
2. **Conversão para deslocamento** — divide o espectro de aceleração por
   $-\omega^2$. As funções de transferência do Anexo A são deslocamento→força,
   não aceleração→força. Sem esta etapa os resultados saem cerca de mil vezes
   maiores.
3. **Forças intervertebrais** — Fórmula (A.1), somando sobre os quatro pontos de
   contato e as três direções de excitação. A função de transferência do modelo
   (0,25 a 30 Hz) é interpolada por spline em módulo e fase na grade da
   excitação, e zerada fora dessa faixa.
4. **Volta ao tempo** — IFFT, e o valor médio é substituído pela força estática
   vinda do modelo de elementos finitos.
5. **Picos de compressão** — extremos negativos entre cruzamentos consecutivos
   do valor médio (Figura A.2).
6. **Dose** — Fórmulas (A.2) e (A.3), com expoente 6 de Palmgren-Miner.
7. **Fator de risco** — Fórmulas (E.1) a (E.4), com $S_u$ decaindo com a idade
   e $S_{stat}$ subtraída da resistência.

## Estrutura dos dados de função de transferência

Decodificada por engenharia reversa do pacote MATLAB, já que a documentação
original é incompleta.

Cada arquivo `BSKUefuM_{k}_{el}_{ed}` é um CSV de 3 linhas × 847 colunas:

- **847 colunas** = 7 blocos de 121. Bloco 1 é o vetor de frequência (0,25 a
  30 Hz). Blocos 2 a 7 são os seis níveis discais. Dentro de cada bloco, a
  primeira coluna é a força estática do modelo FE, e as 120 restantes são a
  função de transferência complexa.
- **3 linhas** = direção da força (x, y, compressão).
- `el` = 1 a 4: ponto de excitação na ordem assento, pés, mãos, encosto.
- `ed` = 1 a 3: direção da excitação (x, y, z).
- `k` = 1 a 30: `(classe_IMC−1)·15 + (percentil−1)·3 + subclasse`, com o limite
  de IMC em 26,1 kg/m².

O arquivo `static_forces` tem 420 linhas = 7 posturas × 10 antropometrias × 6
discos, indexado por `(postura−1)·60 + (antropometria−1)·6`.

## Processamento de sinal

A aba **Signal processing** do avaliador mostra, para cada excitação, todos os
parâmetros da FFT aplicada: número de amostras, taxa de amostragem, comprimento
da transformada, janela, resolução em frequência, frequência de Nyquist, tipo de
espectro, cortes de banda e quantos bins sobraram. Abaixo da tabela fica a cadeia
de processamento completa, passo a passo.

Resumo: transformada do registro inteiro de uma vez, janela retangular, sem
segmentação, sobreposição, preenchimento com zeros ou média. O espectro é
unilateral de amplitude — a FFT complexa escalada por 2/N com os bins de DC e
Nyquist divididos por dois, de forma que uma senoide pura de amplitude *a*
produza um bin de módulo *a*.

**Nenhuma ponderação em frequência é aplicada.** O Anexo A trabalha com
aceleração não ponderada; os filtros Wk e Wd pertencem à ISO 2631-1 e não fazem
parte deste método, porque a dependência em frequência da resposta já está
embutida nas funções de transferência. Se for preciso acrescentar ponderações ou
filtros no futuro, o ponto de inserção é a função `_prepare` em
`iso26315/core.py`, logo após o corte de banda e antes da conversão para
deslocamento. As constantes `HIGHPASS_HZ`, `LOWPASS_HZ`, `WINDOW` e `WEIGHTING`
ficam no topo do mesmo arquivo e alimentam a aba automaticamente.

Na aba **Frequency domain**, a caixa *Overlay unprocessed spectrum* desenha o
espectro bruto por cima do processado, com linhas verticais nos cortes de 0,5 e
100 Hz, e *Full range up to Nyquist* estende o eixo para mostrar o que foi
descartado.

## Métodos de estimação espectral

A aba **Frequency domain** traz um painel *Spectral estimation* com cinco
métodos, janela, comprimento de segmento, sobreposição e largura de banda:

| Método | Uso |
|---|---|
| Single block | FFT complexa do registro inteiro, janela retangular. É a que alimenta o cálculo |
| Welch autospectrum — amplitude | Média de periodogramas, corrigida em amplitude. Comparável ao bloco único para senoides |
| Welch autospectrum — PSD | Densidade espectral de potência, em (m/s²)²/Hz |
| Welch autospectrum — RMS per band | PSD integrada em bandas de largura escolhida |
| One-third octave RMS | Bandas de 1/3 de oitava, centros nominais da ISO 266 |

Janelas disponíveis: retangular, Hann, Hamming, Flat top e Blackman-Harris.

Calibração verificada: senoide de amplitude 1,0 a 5 Hz devolve pico de 1,0000
com Flat top e 0,9959 com Hann; a PSD de ruído branco de variância unitária a
500 Hz devolve 0,00399 contra o teórico 0,00400.

**Estes métodos afetam apenas o que é desenhado, nunca o que é calculado.** O
motivo é físico e está explicado na aba *Signal processing*: a média de Welch
reduz a variância do estimador justamente por promediar módulos ao quadrado, o
que descarta a fase. Sem fase não há como reconstruir a força no domínio do
tempo, e é lá que os picos de compressão da Fórmula (A.2) são contados. A cadeia
de cálculo continua usando o espectro complexo de bloco único com janela
retangular, que é a configuração validada contra a referência.

## Ponderações em frequência da ISO 2631-1

Sete ponderações implementadas a partir das Tabelas A.1 e A.2 daquela norma:
Wk, Wd, Wc, We, Wj, Wf e Wb. Cada uma é o produto dos quatro fatores da Seção
A.2 — passa-alta e passa-baixa Butterworth de dois polos, transição
aceleração-velocidade e degrau ascendente — montado no domínio de Laplace e
avaliado em `weighting_response`.

Validação contra a Tabela 3 da ISO 2631-1, em bandas de 1/3 de oitava: erro
relativo mediano de 0,05% no Wk e 0,04% no Wd. A implementação segue a
ISO 2631-1:1997/Amd.1:2010, que corrigiu o valor do Wd a 10 Hz de 212 para 202
na Tabela B.2.

### Controle por eixo

O painel *Frequency weighting* aparece na aba Frequency domain e no explorador
de sinais. Cada eixo tem seu próprio seletor de filtro, e os fatores k vêm das
tabelas da ISO 2631-1 e não são editáveis — o usuário escolhe o filtro, o k
acompanha a predefinição.

Predefinições disponíveis: sem ponderação (o que a ISO 2631-5 exige), saúde da
ISO 2631-1 (Wd/Wd/Wk com k = 1,4/1,4/1,0), conforto no assento, conforto no
encosto, enjoo de movimento (Wf só em z) e conforto da BS 6841. Os seletores de eixo e os fatores k ficam travados em qualquer predefinição e só
aceitam edição quando se escolhe **Custom**, onde tanto o filtro quanto o k são
livres. O rodapé do painel avisa em vermelho quando a configuração não
corresponde a nenhuma norma.

O seletor *Axis combination* escolhe entre uma curva por eixo, a soma vetorial
ponderada, ou as duas coisas. Com uma curva por eixo, x fica em linha cheia, y
tracejada e z pontilhada, mantendo a cor do arquivo.

A tabela de métricas do explorador segue o painel: o a_w, o pico, o fator de
crista e o VDV são sempre calculados com a configuração ativa, para que o número
na tabela corresponda à curva na tela. A linha "Weighting per axis" mostra o que
foi aplicado.

**Nada disso entra no cálculo.** S_d e R^A usam sempre aceleração não ponderada,
porque a dependência em frequência já está nas funções de transferência.

## Critério de validade dos 9,81 m/s²

A ISO 2631-5 exige que, depois de aplicados os filtros limitadores de banda da
ISO 2631-1, os picos de aceleração em z não excedam 9,81 m/s² para que os
Anexos A e E se apliquem. Acima disso valem a Cláusula 5 e os Anexos C e D, que
tratam de exposições com perda de contato.

O avaliador faz essa verificação automaticamente e mostra o resultado acima da
tabela de resultados, em verde ou vermelho. Os números continuam sendo
calculados quando o critério falha, mas o aviso deixa explícito que eles estão
fora da faixa de validade do método.

Convém saber que dois dos arquivos de exemplo da própria norma falham no teste:
o `BD1sZ.asc` chega a 30,8 m/s² ponderado e o `WL3s.asc` também excede o limite.

## Consistência da base de tempo

A Fórmula (A.1) soma as contribuições dos pontos de contato espectro a espectro,
o que exige uma grade de frequência comum. Sensores independentes derivam entre
si, então o software verifica antes de calcular:

- **Taxas de amostragem diferentes** entre pontos de contato: recusa, porque
  truncar não resolveria. É preciso reamostrar antes.
- **Deriva acima de 1%** do comprimento: recusa, com as durações de cada arquivo
  na mensagem. Uma diferença desse tamanho indica falha de aquisição, não deriva
  de relógio.
- **Deriva abaixo de 1%**: trunca todos ao menor comprimento comum e mostra um
  aviso vermelho na aba de resultados, dizendo o percentual e quantas amostras
  foram descartadas.

A tolerância está em `core.TIME_BASE_TOLERANCE`.

## Signal records (explorador de sinais unificado)

A aba **Signal records** analisa arquivos de aceleração diretamente, sem
planilha nenhuma. Está fora do escopo da norma: não calcula dose nem fator de
risco, e existe para triagem de sinais antes de montar um histórico de exposição.

Aceita todos os formatos de aquisição: ASCII de quatro colunas (`.txt`, `.asc`,
`.dat`, `.csv`), os formatos de campo actuais (DateTime CSV, VIB1 `.dat`) e o
formato legado `.m` de blocos. Um único ponto de entrada substitui as antigas abas
separadas *ASCII records* e *Field records*.

Carrega vários arquivos de uma vez e desenha todos sobrepostos, com caixas para
ligar e desligar cada um. Dezasseis visualizações: série temporal, espectro (com
os cinco métodos e todas as janelas), RMS acumulado — útil para verificar
estacionariedade —, fase, coerência, transmissibilidade SEAT, transmissibilidade
H1, espectrograma, SRS, H1 por classe de amplitude, valores de intervalo
(a_w/MTVV/VDV), histograma/CDF, ODS, coerência múltipla, resposta do filtro de
ponderação e H1 em bandas de 1/3 de oitava.

**Fase** mostra o ângulo (desenrolado) do espectro complexo de bloco único —
o mesmo que alimenta o cálculo do Anexo A — para um eixo por vez; a soma
vetorial não tem fase com significado físico, por isso exige eixo específico.

**Coerência** calcula γ²(f) = |Sxy|²/(Sxx·Syy) entre o primeiro arquivo marcado
e cada um dos demais, via densidade espectral cruzada de Welch. Só é informativa
com mais de um segmento médio — com um único periodograma, γ² vale 1 em todo bin
por construção — e só tem sentido físico entre registros de fato sincronizados.

A tabela lateral traz, para o arquivo selecionado: amostras, duração, taxa de
amostragem, RMS/pico/crest/VDV/curtose/assimetria, frequência dominante, métricas
ISO 2631-1 ponderadas completas (a_w, A(8), MTVV, VDV ponderado), avaliação da
Diretiva 2002/44/CE, zona de saúde do Anexo C e validade dos Anexos A/E.

### Junção de arquivos consecutivos

O botão **Join selected → 1 signal** concatena os registros marcados na lista
(na ordem em que aparecem) num único sinal e adiciona o resultado como novo item.
Útil quando uma sessão longa foi gravada em vários arquivos consecutivos: cada
arquivo fica como referência independente, o sinal conjunto recebe análise
completa de todo o período.

Requisito: todos os arquivos marcados devem ter a mesma taxa de amostragem.

### Exportação PDF

O botão **Export report PDF** abre um diálogo para escolher quais dos dezasseis
tipos de gráfico incluir no PDF. O relatório gerado contém uma página de
estatísticas por registro (A4 retrato, com tabela ISO 2631-1 completa, cores
semânticas por faixa de risco, metadados) seguida de uma página por gráfico
seleccionado (A4 paisagem, tamanho completo).

O botão **Export spectrum to CSV** grava o espectro em CSV, com uma coluna por
arquivo, para levar a outra ferramenta.

## Registrador de campo e NHO-09

A aba **Directive** monta um dia de trabalho a partir dos arquivos do
registrador de campo e avalia a exposição pelos critérios da NHO-09, da Diretiva
2002/44/CE e do método do laboratório.

Os registradores actuais gravam de forma contínua. Um dia é carregado
seleccionando vários arquivos da mesma unidade de uma vez.

O ganho de cada unidade é corrigido pela gravidade: parado, o módulo do vetor
medido tem de ser 9,784 m/s² em Belo Horizonte. A coluna que contém a gravidade
é tratada como eixo vertical.

### Metadados do sujeito e da medição

O painel de configuração da aba Directive contém campos para ID/nome do sujeito,
idade, operador, sensor/modelo e número de série, e notas livres. Todos são
opcionais e aparecem na página 1 do PDF exportado, facilitando a rastreabilidade
para publicação sem exigir re-exportação quando os dados mudam.

### Exportação PDF da aba Directive

O botão **Export PDF** abre um diálogo de seleção de gráficos idêntico ao do
Signal records. O relatório inclui: cabeçalho com nome do arquivo e data/hora,
caixa de configuração da avaliação, caixa de informações do sujeito/medição
(apenas campos preenchidos, em duas colunas), tabela de resultados com cores
semânticas (verde/âmbar/vermelho por faixa de risco), rodapé com paginação, e
uma página por gráfico selecionado em A4 paisagem.

### Avaliação NHO-09

Três escolhas, cada uma com motivo:

**O dia é tratado como uma única componente de exposição, e o valor eficaz é
agregado pela energia de todos os blocos.** A NHO-09 prescreve média
*aritmética* dos valores de a_mr das repetições, o que é adequado para
repetições da mesma operação. Num dia de rotina, com longos trechos parados e
poucos blocos intensos, a média aritmética subestimou em 70%. O valor
aritmético é mostrado na tabela como verificação, mas não é usado.

**O tempo de exposição padrão é o período efetivamente medido.** Assim os blocos
e o tempo declarado vêm sempre do mesmo intervalo, e o resultado fica invariante
a quanto tempo a cadeira esteve ocupada — os blocos parados contribuem quase
nada de energia, e excluí-los reduziria o tempo na mesma proporção.

**Extrapolar para mais horas é uma opção explícita**, marcada como suposição,
com as duas relações temporais do Anexo B da ISO 2631-1 lado a lado: raiz
quadrada e raiz quarta. Em torno do nível de ação elas podem dar vereditos
opostos.

### Avaliação pela Diretiva 2002/44/CE

Na mesma sub-aba, o seletor **Criterion** oferece três opções: NHO-09, a
diretiva pelo **maior eixo** — que é o texto legal para corpo inteiro — e a
diretiva pela **soma vetorial**, que é o método do laboratório descrito no
CONEM2026. A diretiva prescreve a soma dos quadrados apenas para vibração
mão-braço; para corpo inteiro, prescreve o maior dos três eixos, tanto no A(8)
quanto no VDV. A soma vetorial é uma escolha mais conservadora e deve ser
declarada como divergência.

O seletor **Exposure time basis** permite reproduzir o método do artigo, que usa
como tempo apenas o período com dado gravado. O padrão recomendado é o período
coberto, porque a vibração não para nos intervalos em que o cartão não grava.
Com o ciclo de 40%, usar só o tempo gravado reduz o A(8) em cerca de 37%.

A caixa **VDV against ISO 2631-1 zone** troca os limites do VDV de 9,1 / 21, da
diretiva, para 8,5 / 17, da zona de cautela da ISO 2631-1, que o método do
laboratório adota por ser mais restritiva.

**A gravidade é sempre removida de cada bloco antes de qualquer métrica.** Um
VDV calculado com o 1 g estático retido dá cerca de 130 m/s^1,75 num dia de oito
horas — valor que não é vibração, e sim a integração da gravidade no tempo.

### Posição do sensor

O seletor **Sensor position** aplica o tratamento que a ISO 2631-1 prevê para
cada ponto de contato:

- **Assento:** avaliação de saúde, pela NHO-09 ou pela diretiva.
- **Encosto:** avaliação de conforto, com Wc / Wd / Wd e k = 0,8 / 0,5 / 0,4.
  O eixo x com Wc e k = 0,8 aparece como valor **informativo** de saúde, sem
  veredito: a subcláusula 7.2.3 recomenda medi-lo mas o exclui da avaliação de
  severidade, por falta de evidência sobre o efeito na saúde.
- **Pés:** avaliação de conforto, com Wk e k = 0,25 / 0,25 / 0,4.

O conforto usa o valor total de vibração a_v, comparado com as indicações do
Anexo C, que a norma declara serem aproximadas e não limites. A ponderação Wc
foi validada contra a Tabela 4 da ISO 2631-1, com erro mediano de 0,02%.

No encosto, os dois eixos horizontais não são equivalentes: x é o
perpendicular ao encosto e recebe Wc. O software o identifica pela inclinação
da gravidade — um encosto reclinado a inclina em direção ao eixo perpendicular.

### Visão conjunta e comparação com o software do laboratório

Abaixo do veredito, uma **faixa de resumo** mostra todos os critérios ao mesmo
tempo, cada um com a cor do seu limiar: NHO-09, diretiva pelo maior eixo e pela
soma vetorial, conforto e o método do laboratório. Critérios que não são
normativos para a posição do sensor, ou que vêm de configuração não normativa,
aparecem em cinza, com a cor só no número.

Três gráficos novos:

- **Dashboard** — cada critério na sua própria escala, com as faixas verde,
  âmbar e vermelha.
- **Side by side** — este software e o método do laboratório, eixo a eixo, com
  a diferença percentual. A comparação é feita pelo **eixo físico** do sensor:
  com a troca x-z do aplicativo, o "z" do laboratório é outra coluna do arquivo,
  e o rótulo de cada barra diz qual.
- **Contribuição de cada hora** — a fração de cada arquivo na energia do A(8)
  diário, por eixo. Mostra onde está a exposição do dia.

O painel **Lab method** reproduz o aplicativo WBV_HSE_v16 com a configuração
escolhida — ponderação por eixo incluindo "None", fatores k, troca x-z, tempo do
RMS. Validado contra dois casos: a tela do aplicativo para 20/01/2025 às 8 h
(0,303) e o encosto do artigo CONEM2026 (0,685 em A(8) e 165,9 em VDV), ambos
reproduzidos exatamente. O VDV é sempre calculado da forma correta — com o passo de tempo, a
extrapolação pelo tempo gravado e o maior eixo. A caixa **Reproduce the app VDV
defects**, desligada por padrão, existe apenas para conferir os números contra a
tela do aplicativo.

### Quadro de A(8) por eixo

Acima da tabela, um quadro no formato do painel *Total A(8) Exposure* do
aplicativo mostra o A(8) de cada eixo e o diário, pelo maior eixo e pela soma
vetorial, em **qualquer posição do sensor**. Duas linhas: este software, com a
ponderação de saúde Wd / Wd / Wk e k 1,4 / 1,4 / 1,0 — a única definição de
A(8) da diretiva —, e o método do laboratório com a configuração escolhida,
que reproduz os números da tela do aplicativo.

No assento, as células deste software são coloridas contra 0,5 e 1,15 m/s².
Nas outras posições e no método do laboratório, as células ficam cinza e só o
número leva a cor: o valor está visível para referência, mas não é avaliação de
saúde.

A detecção de ocupação da cadeira é heurística — orientação do assento
congelada, com desvio abaixo de 0,05 grau numa janela de 5 minutos — e aparece
só como informação, em cinza na linha do tempo.

## Seleção de séries nos gráficos

Todos os gráficos permitem ver uma série por vez ou todas. No espectro de
excitação dá para escolher o ponto de contato (assento, pés, mãos, encosto) e o
eixo (x, y, z ou a soma). No espectro de força, nas funções de transferência e
no domínio do tempo dá para escolher um nível discal específico.

Na aba **Frequency domain**, o seletor de gráfico também traz **Excitation
phase spectrum** e **Coherence between contact points**: fase é por ponto de
contato e eixo específico (não a soma); coerência usa o ponto de contato como
referência (o seletor vira "Reference point") contra cada um dos outros três,
sempre em um eixo específico. Os dois recursos são só de visualização — não
entram em $S_d^A$ nem em $R^A$ — e a coerência só é confiável entre os quatro
pontos de uma mesma exposição, cuja base de tempo comum já é verificada antes
do cálculo (ver "Consistência da base de tempo" acima).

## Conveniências de interface

**Tela cheia nos gráficos.** F11, duplo clique no gráfico, ou o botão "Full
screen" na barra de navegação. Esc ou F11 voltam. A figura é o mesmo objeto nos
dois modos, então zoom e deslocamento aplicados sobrevivem à ida e volta.

**Barra de progresso com o arquivo em leitura.** Durante a análise a barra de
status mostra qual ponto de contato e qual arquivo estão sendo lidos, o que
importa com registros longos.

**Botões de pan e zoom** mostram o estado ligado em azul, e **o enquadramento
sobrevive ao redesenho**: mudar um controle apaga a figura, e com ela o zoom que
o usuário tinha aplicado. Como quase toda interação redesenha, no explorador de
sinais isso dava a impressão de que o zoom não funcionava. Agora o
enquadramento é guardado e reposto; o botão Home volta ao completo.

**As abas de resultado aparecem só depois da primeira análise.** Ao abrir, o
programa mostra Annex A / Annex E, Clause 5 / Annex C e Signal explorer. Results, Frequency
domain, Time domain e Signal processing entram quando há o que mostrar.

**Logo do GRAVI-SH** no painel de parâmetros da aba Annex A / Annex E, e como ícone da
janela e dos dois executáveis.

## Tema

As cores ficam em `iso26315/theme.py`, compartilhadas pelos dois aplicativos. A
paleta é clara e explícita de propósito: sem isso o Qt herda o tema do sistema
operacional, e no Windows em modo escuro o texto claro do tema cai em cima dos
fundos claros da folha de estilo e some.

## Estado da validação por caso

| Caso | Massa | Erro relativo | Situação |
|---|---|---|---|
| Orientadora (WL2) | 110 kg | 3,9×10⁻⁵ | exato |
| BD1 individual | 110 kg | 2,2×10⁻⁵ | exato |
| indiv1 a indiv4 | 110 kg | 7×10⁻⁶ a 3,3×10⁻⁵ | exato |
| indiv5, indiv6 | 70 a 110 kg | 5,6×10⁻⁴ a 6,4×10⁻⁴ | aproximado |

Os oito casos de referência disponíveis são reproduzidos. Os seis que usam massa
constante de 110 kg fecham no nível do arredondamento dos valores publicados; os
dois que variam a massa ficam uma ordem de grandeza acima, pelo motivo descrito
em Limitações.

## Cláusula 5 com Anexo C

Segundo caminho da norma, para exposições **com** perda de contato com o
assento — as que o critério dos 9,81 m/s² rejeita. Fica na aba própria
"Clause 5 / Annex C" e é **estritamente independente** da cadeia do Anexo A,
porque os requisitos conflitam:

| | Cláusula 5 / Anexo C | Anexo A / Anexo E |
|---|---|---|
| passa-alta | 0,1 Hz, 2 polos | 0,5 Hz |
| passa-baixa | 80 Hz, 4 polos | 100 Hz |
| filtros da ISO 2631-1 | proibidos | só na verificação de validade |
| direções | só z, no assento | x, y, z em 4 pontos |
| excitação | aceleração | deslocamento |

O corte inferior em 0,1 Hz existe para preservar a queda livre antes do
impacto: filtrar em 0,5 Hz empurraria o −1 g de volta a zero antes do choque e
erraria o pico em até 1 g.

A cadeia é: condicionamento, função de transferência assento-coluna da Fórmula
(1) com um zero e seis polos complexos, dose de aceleração da Fórmula (3),
normalização diária pelas Fórmulas (4) e (5), dose de compressão pela (C.1), a
variável de tensão R pela (C.3) e a probabilidade de dano pelo modelo de
Weibull da (C.5). A saída mostra os cinco resultados.

O $m_z$ é parâmetro aberto: a fração de massa acima da pelve e a área de placa
terminal são editáveis, com os valores de referência da norma como padrão por
sexo e o texto de referência ao lado. Massa de equipamento no tronco ou na
cabeça entra num campo separado, como a norma pede.

### Validação

O exemplo numérico do fim do Anexo C — homem de 82 kg, 120 dias/ano, dos 20 aos
40 anos, cinco acelerações de 40 m/s² por dia — reproduz $S_d$ = 1,623 MPa e
R = 1,22 exatamente, usando o $m_z$ arredondado de 0,029 que a própria norma
emprega no exemplo. Com o $m_z$ exato de 0,02870 sai 1,606 e 1,207; a diferença
é só o arredondamento da norma.

A Tabela C.2 também reproduz exatamente, nos dois sexos: R de 0,72 / 1,42 / 2,17
para 10%, 50% e 90% de risco em homens, e 0,52 / 0,87 / 1,20 em mulheres. E o
ganho em corrente contínua da Fórmula (1) dá 1,000000, coerente com a nota da
norma de que a transmissibilidade em DC foi ajustada para 1,0.

### Detecção de perda de contato

A norma exige detectar e excluir os trechos sem contato, por chave de contato,
vídeo ou método equivalente. Sem esse instrumento, o software oferece uma
**heurística que não está na norma**: marca como queda livre os trechos em que a
aceleração vertical fica abaixo de −7 m/s² por pelo menos 0,15 s. Os limiares
estão em `annexc.FREEFALL_THRESHOLD` e `FREEFALL_MIN_DURATION`, a opção pode ser
desligada, e a aba avisa que convém conferir os trechos marcados.

## Uso de memória

Um resultado completo guarda a série de força e o espectro de cada nível discal,
o que soma dezenas de MB por combinação antropometria × exposição. Um histórico
longo percorre dezenas dessas combinações — o exemplo `indiv6`, com treze
mudanças de massa corporal, chegava a derrubar o processo.

O cache guarda o resultado completo apenas para as primeiras quatro
combinações, que são as que alimentam os gráficos, e um resumo leve para as
demais, com S, S_stat e t_m, que é tudo que a dose e o risco precisam. O
espectro de deslocamento é liberado logo após o cálculo das forças, e o espectro
bruto é mantido só nas duas primeiras. O `indiv6` agora roda com pico de 2,1 GB
e resultado inalterado.

## Interpolação da fase das funções de transferência

A função de transferência do modelo tem um salto de 2π na fase perto de 20 Hz.
O software original interpola a **fase enrolada** com spline cúbica, e a spline
atravessa esse salto pelo caminho errado: nos cerca de 6% de pontos em torno
dele, o valor complexo interpolado chega a errar 200%.

Desenrolar a fase antes de interpolar corrige o problema, mas **afasta o
resultado da referência**, porque o software original carrega o mesmo defeito.
Medido na análise orientadora:

| | T12/L1 | L1/L2 | L2/L3 | L3/L4 | L4/L5 | L5/S1 |
|---|---|---|---|---|---|---|
| Fase enrolada (padrão) | 0,98423 | 1,08503 | 1,22263 | 1,31024 | 1,28424 | 1,16975 |
| Fase desenrolada | 0,98810 | 1,08905 | 1,22754 | 1,31568 | 1,29097 | 1,18076 |
| Diferença | 0,39% | 0,37% | 0,40% | 0,42% | 0,52% | 0,94% |

O padrão é a fase enrolada, por fidelidade à referência. A caixa **Unwrap
transfer function phase before interpolating**, no painel Parameters da aba
Annex A / Annex E, liga a correção; em código, `core.UNWRAP_PHASE`.

**Ao publicar resultados, deixe claro qual dos dois modos foi usado.** O número
que reproduz o software oficial é o da fase enrolada; o da fase desenrolada é
numericamente mais correto mas não corresponde ao que a ferramenta de referência
produz. A diferença de até 1% é uma estimativa razoável do erro numérico que o
valor oficial carrega.

## Limitações conhecidas

**A subclasse de 1 a 3.** O terceiro fator do índice `k` parece ser a classe de
intensidade da aceleração do `intcalc` original (RMS abaixo de 0,65 ou acima de
1,35 m/s²), mas os dois casos de referência batem com subclasse 1, inclusive o
bulldozer, cujo RMS é de 3,08 m/s². O padrão é 1, e a interface permite trocar.

**O mapeamento massa → percentil.** Ajustado contra os exemplos `indiv5` e
`indiv6`, que percorrem 70, 80, 90, 100 e 110 kg a 1,85 m. O ajuste reproduz os
dois com erro relativo de 6×10⁻⁴ — vinte vezes maior que o erro dos quatro casos
que fecham exatamente (3×10⁻⁵), o que indica mapeamento próximo mas não exato.

Além disso, o melhor ajuste coloca todas as massas no bloco 2 do índice,
inclusive 70 kg a 1,85 m, que dá IMC 20,5. Isso contraria a leitura do bloco
como classe de índice de massa corporal e mostra que a decomposição do índice de
30 posições não está univocamente determinada pelos dados disponíveis. O texto
completo está em `core.ANTHROPOMETRY_NOTE`.

**`t_m` vem da planilha.** A norma define $t_m$ como a duração da medição, mas o
software original usa o campo *Measurement duration* da aba `Exposure`, mesmo
quando ele não bate com o comprimento real do arquivo. O comportamento foi
mantido para compatibilidade.
