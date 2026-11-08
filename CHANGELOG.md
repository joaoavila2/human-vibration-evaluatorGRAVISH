# Changelog — Avaliador de Vibração Humana v3.1

---

## 2026-11-07 — v3.1.3 — Abre sem os dados ISO; licença e citação

- `core.TransferFunctionStore`: `static_forces.npz` passa a ser carregado na primeira vez que é usado, e não ao abrir a janela. Antes, quem clonasse o repositório sem gerar `data/` não conseguia abrir o programa; agora todas as abas funcionam e só os cálculos dos Anexos A/E mostram o erro (em inglês, como a interface), com o comando `tools/build_data.py` para gerar os dados. A mesma mensagem vale para posturas ausentes.
- Verificado: janela abre com `data/` vazia; `tests/test_reference.py` (com as acelerações de exemplo da norma) e os testes unitários passam sem mudança de resultado.
- `LICENSE` (MIT; cobre só o código, não os dados ISO) e `CITATION.cff` (DOI 10.5281/zenodo.24045956).
- README: comportamento sem dados, seção de licença e citação.
- `_version.py` e `iso2631-5.spec`: 3.1.2 → 3.1.3.

---

## 2026-11-07 — README reescrito

README revisado por completo contra o código: todas as abas atuais (inclusive Campaign, Live FFT, Visualizador, Processamento em Lote e o protótipo de cadeira de rodas), formatos de entrada, validação dos Anexos A/E e do Anexo C, testes, estrutura do repositório e citação com o DOI do Zenodo (10.5281/zenodo.24045956). Removidas referências ao workflow de CI e ao firmware, que saíram do repositório.

---

## 2026-11-07 — v3.1.2 — Remoção dos dados de função de transferência (conformidade ISO)

> Histórico do GitHub reescrito com `git filter-repo` + force-push: `.npz` removidos de todos os commits anteriores.

### Decisão

**Decisão do usuário (2026-11-07):** "Código sem dados ISO." Os arquivos `.npz` derivados da ISO 2631-5:2018 (funções de transferência e forças estáticas) foram removidos do controle de versão por serem material protegido por direitos autorais da ISO.

### Arquivos removidos do git

- `data/posture1.npz` … `data/posture5.npz`, `data/posture7.npz` — funções de transferência por postura
- `data/static_forces.npz` — forças estáticas normalizadas

Os arquivos **permanecem no disco local** e continuam funcionando normalmente. Apenas deixam de ser rastreados pelo git.

### Como regenerar

1. Obtenha o pacote de funções de transferência junto com a norma ISO 2631-5:2018 (Annex E);
2. Coloque o ZIP baixado em `tools/`;
3. Execute: `python tools/build_data.py`

### Alterações no código

- `.gitignore`: adicionado `data/*.npz`
- `core.py` (`TransferFunctionStore.__init__`): mensagem de erro clara quando `static_forces.npz` não existe, com instruções para regenerar

---

## 2026-11-07 — v3.1.1 — Aviso de encosto "Zero" (Opção B, aprovada pelo usuário)

### Decisão e justificativa

**Decisão:** manter o sinal nulo (`zero_excitation`) para encosto marcado "Zero"; emitir aviso não-bloqueante ao usuário. Aprovada em 2026-11-07.

Justificativa:
1. **"Zero" tem um único significado na norma:** sinal nulo, 0 m/s² (ISO 2631-5:2018 A.3.1, p. 11). Embora a norma prescrevesse que o *usuário* coloque o sinal do assento no campo do encosto quando não houver medição, ela não determina que o *programa* realize essa substituição automaticamente.
2. **A substituição pelo assento é ato do usuário, prescrito pelo texto** — o zero tem uso legítimo (assentos sem encosto; análises de sensibilidade) e não deve ser interceptado silenciosamente pelo software.
3. **O aviso elimina o erro silencioso** apontado pelo VIBRISKS Annex 18 (Hinz et al., 2007, p. 56): o usuário agora é informado da divergência potencial, e pode corrigir a planilha se quiser o comportamento do executável.

A pendência A/B/C registrada em 2026-11-07 está resolvida com a Opção B.

### Mudanças técnicas

**`iso26315/core.py`**
- Constante `BACKREST_ZERO_WARNING` com texto do aviso (PT/EN).
- `read_exposure_files` agora retorna uma tripla `(excs, drift, warnings)`. Quando o encosto (índice 3) é "Zero", `warnings` contém a constante acima **e** `warnings.warn()` é emitido para uso como biblioteca ou CLI.
- `ExposureResult` recebe campo `backrest_warning: str | None = None`.

**`iso26315/analysis.py`**
- `_LightResult.__slots__` inclui `'backrest_warning'`.
- `Runner._exposure` propaga o aviso para `res.backrest_warning`.

**`iso26315/gui.py`**
- `_show_validity()` exibe o aviso com ícone ⚠ em fundo âmbar quando qualquer exposição tiver `backrest_warning` preenchido.
- `export_pdf()` inclui o aviso na página 1 do PDF quando aplicável.
- Chamada de `read_exposure_files` no *sensitivity grid* atualizada para desempacotar a tripla.

**`iso26315/workbook.py`**
- Nova função `append_warnings_sheet(path, warning_rows)`: acrescenta (ou substitui) a aba **'Warnings'** num xlsx existente quando há avisos de encosto Zero. Recebe lista de `(exposure_name, warning_text)`.

**`iso26315/gui.py` — `export()`**
- Após gravar a aba 'Risk factors', chama `append_warnings_sheet` com os avisos coletados das exposições; aba 'Warnings' aparece no xlsx apenas quando necessário.

**`tests/test_backrest_warning.py`** (novo)
- 9 testes: aviso gerado quando back="Zero"; emissão de `warnings.warn`; resultado numérico inalterado; sem aviso quando encosto tem arquivo real; mãos e pés "Zero" não disparam aviso; exatamente um aviso com seat+três Zeros; aba Warnings criada/substituída/omitida no xlsx.

---

## 2026-11-07 — Correção da justificativa do backrest "Zero" (R13, sem mudança de código)

> Esta entrada corrige a justificativa registrada em 2026-11-06. O comportamento do código (sinal nulo para encosto "Zero") **não foi alterado**. A decisão sobre o que fazer na v3.1.1 está pendente — ver seção "Pendência" abaixo.

### Erros e imprecisões na justificativa anterior

**Item 3 original** ("Não há instrução normativa explícita na ISO 2631-5:2018 determinando copiar o sinal do assento") — **INCORRETO**.

A ISO 2631-5:2018, Seção A.3.1 (p. 11), diz que, quando não há medição do encosto, ela *"can be substituted with the measurements at the seat surface"*. Diz ainda que, havendo apenas dados do assento, *"this is used for the seat and backrest"*; o marcador "Zero" aplica-se apenas aos outros pontos (mãos e pés). Em outras palavras, a norma instrui o **usuário** a colocar o sinal do assento no campo do encosto quando o encosto não foi medido. O que a norma não especifica é que o *programa* realize essa substituição automaticamente ao encontrar "Zero" naquele campo.

**Descrição do executável** ("artefato do código MATLAB, possivelmente legado da ISO/DIS") — **IMPRECISA**.

A substituição `Anra{4} = Anra{1}` é uma regra **deliberada** do grupo de desenvolvimento do método. O VIBRISKS Annex 18 (Hinz et al., 2007, pp. 53 e 56) explicita: *"the user shall never set the inputs of the backrest or seat surface to zero"*, porque as respostas parciais do assento e do encosto se cancelam parcialmente — anular um deles distorce o resultado. Fleury et al. (2015, p. 99) confirma o mesmo comportamento no executável DIN-2012.

**Itens 1 e 2 originais** ("fisicamente correto"; "consistente com mãos/pés") — **enfraquecidos**.

- "Zero" na planilha ISO significa *"não medido"*, não *"veículo sem encosto"*. Um encosto presente mas não medido produz aceleração; um sinal nulo não representa esse estado.
- A norma trata o encosto de forma diferente de mãos e pés: para o encosto sem medição, ela prevê substituição pelo sinal do assento; para mãos e pés, prevê zero. Os dois casos não são análogos.

### Consequência prática

- Quem segue o texto normativo — isto é, coloca o arquivo do assento no campo do encosto antes de processar — obtém o **mesmo resultado** na v3.1.0 e no executável oficial.
- A divergência observada em R13 (T12/L1: v3.1 = 0,748 × executável = 0,946) surge **somente** quando o encosto é marcado "Zero" diretamente, uma entrada que o texto da norma não prevê para o encosto.

### Pendência — decisão do usuário para v3.1.1

O comportamento atual (sinal nulo para encosto "Zero") pode ser tratado de três formas na próxima versão:

| Opção | Comportamento | Conformidade |
|-------|--------------|--------------|
| A | Substituir automaticamente `Zero` do encosto pelo sinal do assento (como o executável) | Alinhada com VIBRISKS/executável |
| B | Manter zero, mas emitir aviso ao usuário quando encosto = "Zero" | Transparente; usuário decide |
| C | Recusar "Zero" no campo do encosto e exigir um arquivo ou marcação explícita | Mais restritiva |

**Decisão pendente com o usuário.** Nenhuma mudança de código nesta entrada.

---

## 2026-11-06 — Release v3.1.0 (sessão 14 — freeze)

### Congelamento da versão 3.1.0
**Arquivo:** `iso26315/_version.py`  
**Mudança:** versão bumped de `3.1.0-dev` → `3.1.0`; commit de release + tag anotada `v3.1.0`.

### Decisão técnica consolidada — backrest "Zero" (R13, sem mudança de código)
**Contexto:** a sessão anterior (R13) identificou divergência entre o comportamento do software v3.1 e o executável oficial ISO para locais de backrest marcados como "Zero":
- **Software v3.1:** usa `zero_excitation(base)` → sinal nulo (fisicamente correto — equipamento sem encosto não produz aceleração);
- **Executável oficial:** usa `Anra{4} = Anra{1}` → copia o sinal do assento (artefato do código MATLAB, possivelmente legado da ISO/DIS).

**Decisão (aprovada pelo usuário):** manter o comportamento atual do v3.1 (Option B — zero). Justificativa:
1. Fisicamente correto: ausência de backrest = excitação nula;
2. Consistente com o tratamento análogo de mãos/pés "Zero" no mesmo software;
3. Não há instrução normativa explícita na ISO 2631-5:2018 determinando copiar o sinal do assento.

**Impacto:** nenhuma mudança de código. Registrado aqui para rastreabilidade.

---

## 2026-11-03 (sessão 14)

### Nova funcionalidade — curva de Weibull no painel Clause 5 / Annex C
**Arquivo:** `iso26315/clause5_tab.py`  
**Mudança:** adicionado o item **"Weibull injury probability curve (Annex C, Table C.1)"** como plot kind 9 no combo de gráficos da aba Clause 5 / Annex C.  
- Método `_draw_weibull_curve()` plota Π(R) = 1 − exp[−(R/α)^β] de R=0 até R_max;
- Zonas de fundo coloridas: verde (Π < 10%), amarelo (10–50%), vermelho (> 50%);
- Linhas tracejadas verticais/horizontais nos limiares de Π = 10/50/90% (Tabela C.2);
- Marcador de losango (♦) na posição exata do resultado atual, com anotação de R e Π;
- Cor do marcador segue o semáforo de risco (verde/laranja/vermelho).

### Nova funcionalidade — células coloridas para R e Π na tabela de resultados
**Arquivo:** `iso26315/clause5_tab.py`  
**Mudança:** método `_show()` agora aplica `setBackground()` verde/amarelo/vermelho nas linhas "R    stress variable" e "Pi   probability of damage", indicando visualmente a classe de risco sem precisar rolar até o veredito.

### Nova funcionalidade — curva de Weibull embutida na ficha PDF
**Arquivos:** `iso26315/pdf_report.py` (novo), `iso26315/clause5_tab.py`  
**Mudança:** criado módulo `pdf_report.py` com:
- `weibull_figure(result)` → Figure matplotlib reutilizável (aba e PDF);
- `export_to_pdf(result, path_out, meta, daily_hours, n_years, days, extra_figs)` → gera ficha PDF 2+ páginas;
- Página 1: tabela de resultados + parâmetros, com fundo colorido nas linhas R e Π;
- **Página 2 (sempre incluída automaticamente):** curva de Weibull em A4 paisagem.  
O export_pdf da aba Clause 5 agora insere a curva de Weibull na página 2 de todos os PDFs gerados (não precisa de seleção manual).

### Nova funcionalidade — aba "Processamento em Lote"
**Arquivos:** `iso26315/batch_tab.py` (novo), `iso26315/gui.py`  
**Mudança:** nova aba **"Processamento em Lote"** adicionada após "Visualizador" na janela principal.
- Seleção de pasta: todos os arquivos `.csv`, `.dat`, `.txt`, `.m` = uma avaliação cada;
- Parâmetros de exposição compartilhados (sexo, massa, h/dia, dias/ano, idade, anos);
- Processamento síncrono com `QApplication.processEvents()` por arquivo (UI responsiva);
- Tabela de resultados com Dz, Dzd, Sd, R, Π%, classe de risco e coluna de erros;
- **Opção "Gerar ficha PDF por arquivo"**: cria `<nome>_annexc.pdf` (ficha + curva de Weibull) ao lado de cada arquivo de entrada;
- Exportação do resumo para **CSV** com um clique.

---

## 2026-11-03 (sessão 14 — parte 2)

### Nova funcionalidade — estilo profissional de gráficos (`plot_style.py`)
**Arquivos:** `iso26315/plot_style.py` (novo), `iso26315/gui.py`, `iso26315/pdf_report.py`, `requirements.txt`, `iso2631-5.spec`  
**Mudança:** criado módulo `plot_style.py` com função `apply()` chamada uma única vez em `main()` (antes da QApplication), garantindo estilo uniforme em toda a aplicação — abas interativas, exportações PDF e curva de Weibull.

Estilo aplicado:
- **Base:** `['science', 'no-latex']` do SciencePlots (fonte STIXGeneral — clone do Computer Modern/LaTeX); fallback automático para STIXTwoText se SciencePlots não estiver instalado.
- **Fonte:** 10,5 pt geral, 9,5 pt tick labels, legenda sem frame.
- **Ticks:** todos para dentro (`direction: in`), aparecendo nos 4 lados (como MATLAB).
- **Spines:** os 4 visíveis (como MATLAB) — removida a chamada `set_visible(False)` do `finish()` e do `pdf_report.py` que antes ocultava top e right.
- **Linhas:** 1,6 pt (mais visíveis na tela que o 1,0 pt do paper style).
- **Motivação:** fontes padrão do matplotlib (DejaVu Sans) pareciam informais; STIXGeneral dá aparência de publicação científica.

**Spec:** adicionado `scienceplots` em `datas` e `hiddenimports` para inclusão no bundle PyInstaller.

---

## 2026-10-23 (sessão 13)

### Nova funcionalidade — aba Visualizador (`signal_viewer.py`)
**Arquivo:** `iso26315/signal_viewer.py` (novo), `iso26315/gui.py`, `iso26315/_version.py`, `iso2631-5.spec`  
**Mudança:** versão bumped para `3.1.0-dev`; nova aba **Visualizador** adicionada após "Live FFT (Xiao)".

A aba permite abrir e inspecionar sinais brutos sem nenhuma relação com o pipeline ISO 2631-5:
- **Formatos suportados:** `.dat` (MATLAB, via `scipy.io.loadmat`), `.csv` (1ª coluna = tempo), `.txt` (ASCII colunas: t ax ay az)
- **Seleção de canais** via checkboxes (todos marcados por padrão)
- **Domínio do tempo** — gráfico t × amplitude com toolbar matplotlib (zoom, pan, home)
- **Domínio da frequência** — FFT com janela Hann + remoção da média DC, escala logarítmica (semilogy); normalização idêntica ao `block_spectrum` das outras abas
- **Tela cheia** — abre QDialog maximizado com toolbar própria
- **Exportar PDF** — gera arquivo PDF com o gráfico atual (via `matplotlib.backends.backend_pdf.PdfPages`)
- Barra de status integrada via callback `self.status`

**Spec:** nome atualizado para `AvaliadorDeVibracaoHumana-v31-DEV`; `scipy.io` e `scipy.io.matlab` adicionados a `hiddenimports`.  
**Motivação:** calibração do acelerômetro XIAO (arquivos `.dat` do MATLAB) exige visualização de ondas; aba independente do pipeline de análise ISO. Uso restrito a medições curtas (até ~2 min); medições longas continuam na aba Signal explorer.

---

# Changelog — Avaliador de Vibração Humana v3.0

Formato: data · arquivo(s) · descrição · motivo

---

## 2026-10-21 (sessão 12)

### Correção — bordas das classes de massa em `core.py` (pendência 1.1)
**Arquivo:** `iso26315/core.py`  
**Mudança:** `MASS_PERCENTILES_BY_BMI_CLASS` substituída por `MASS_EDGES_BY_BMI_CLASS` com as bordas corretas da DIN SPEC 45697 / ISO/CD 2631-5:2014 (Schust, Hinz & Seidel 2015, Ergonomics 58:576-597, Tab. 1):
- IMC ≤ 26,1: fronteiras em 65 / 72 / 80 / 88 kg (antes: 63 / 70 / 75 / 82 kg)
- IMC > 26,1: fronteiras em 80 / 87 / 96 / 109 kg (antes: 78 / 85 / 90 / 100 kg)

`np.searchsorted(..., side='right')` garante que o valor da fronteira pertence à classe superior (semântica "≥"), conforme o texto da norma (A.3.1 a), E.2, E.4 item 2: "a body mass larger than 109 kg"). `ANTHROPOMETRY_NOTE` atualizada para citar DIN SPEC 45697 e Schust 2015.  
**Faixas que mudam de classe:** IMC ≤ 26,1: 63–65, 76–80 e 82–88 kg; IMC > 26,1: 78–80, 90–96 e 100–109 kg. Os casos de referência (70/80/90/100/110 kg, h=1,85 m) não mudam de classe.  
**Novo arquivo:** `tests/test_anthropometry_unit.py` — testes unitários nas fronteiras exatas (lado abaixo, exatamente na fronteira, lado acima) para ambas as classes de IMC; todos os 3 grupos passam.

### Correção de texto — `annexc.py` (pendência 1.2)
**Arquivo:** `iso26315/annexc.py`  
**Mudança:** comentário sobre área de endplate: "r<=0,34" → "r_max=0,448, <= 20 % da variância explicada". Fonte: Seidel et al. 2008, Int J Ind Ergon, Tab. 3 / Fig. 5, p. 847–849. Sem efeito em cálculo.

### Correção de texto — `README.md` (pendência 1.3)
**Arquivo:** `README.md`  
**Mudança:** parágrafo sobre Wd a 10 Hz: "muito provavelmente é erro do OCR" → referência à errata oficial ISO 2631-1:1997/Amd.1:2010, p. 3 ("replace '212' by '202'").

### Correção de docstring — `tests/clause5_block_study.py` (pendência 1.4)
**Arquivo:** `tests/clause5_block_study.py`  
**Mudança:** "passa-alta de 0,1 Hz" → "0,01 Hz" (em 4 ocorrências: docstring, docstring de função, duas strings de `print`). Período correto: 100 s (não 10 s). Sem efeito em cálculo; `annexc.HIGHPASS_HZ = 0.01` estava correto.

---

## 2026-10-20 (sessão 11)

### Sem mudança líquida no software — tentativa de ordenar `.txt` revertida
**Arquivo:** `iso26315/fielddata.py`  
**Problema:** arquivos `.txt` (tempo relativo) continuavam sobrepostos na aba Directive.  
**Mudança:** testadas ordenação natural (`_natural_sort_key`) e por `mtime` em `read_ascii_accel`; **todas revertidas** ao estado da sessão 10 por decisão do usuário ("arrumar no sensor"). Exe recompilado (`pyinstaller iso2631-5.spec --noconfirm`).  
**Motivo:** a ordem correta deve vir do firmware (timestamp absoluto), não de heurística no software.

### Artigo *Sensors* — manuscrito reestruturado (v1)
- `Publicação Sensors/manuscrito/manuscript.tex` reescrito para o escopo do sensor novo (XIAO nRF52840 + LSM6DS3TR-C) com o sistema de primeira geração como linha de base; versão anterior em `manuscript_v0_escopo_antigo.tex`. Texto em inglês para Introdução, Requisitos, Métodos descritivos e esboço da Discussão; resultados como [TBC]. `
ocite{*}` removido (25 referências citadas). Mapa parágrafo→fonte em `wiki/projeto/artigo-sensors.md`.

### Artigo *Sensors* — wiki concluída (continuação, 2026-10-20)
- Ingeridas as fontes restantes (Larivière 2022 modal, Maeda 2003, Chadefaux 2024, Larivière 2023, Alves & Bezerra 2017, 3 CONEM 2026, TCC Goeking, Edge-computing confidencial) e a tese de Ferreira 2019 (156 p. escaneadas, lidas na imagem). Todos os 39 PDFs de `Artigos_Revista/pdf` têm nota de fonte; wiki com 59 páginas.
- Lint geral: 0 órfãs, 0 links quebrados, regras de sigilo e confidencialidade conferidas; ~30 números receberam referência de página.

### Artigo *Sensors* — infraestrutura fora do software (pasta `Publicação Sensors/`)
- `manuscrito/`: esqueleto LaTeX MDPI; `references.bib` com ~35 entradas obtidas via Crossref.
- `caracterizacao/`: tutoriais mínimo e completo (shaker + ICP de referência, sem ponderação — calibração ≠ uso) e folhas de registro em PDF separado.
- `scripts/doi2bib.py` (DOI → BibTeX; agora troca `&amp;` por `\&` e avisa quando o Crossref não traz autor/título) e `scripts/wiki_index.py` (gera `wiki/index.md`).
- `Central.md` (esquema da wiki) e `wiki/`: 48 páginas (fontes, conceitos, sistemas, debate) com leitura integral de ~30 artigos; log em `wiki/log.md`.
- **Ponto técnico registrado para o software:** o sensor grava a 833 Hz, abaixo dos 9·f₂ = 900 Hz que Rimell & Mansfield (2007) exigem para ponderação por IIR bilinear. `apply_weighting` (`core.py`) pondera no domínio da frequência com H(f) analógica exata, então não é afetada — mas `weighting_filter_sos` (bilinear) seria; verificar onde cada uma é usada e documentar no artigo.

### Docs — wiki do artigo *Sensors* · ingestão retomada (sessão agendada, 05:50–06:10)
**Arquivo:** `Publicação Sensors/wiki/` (fora do software; nenhum código da v3.0 alterado)  
**Mudança:** 10 notas de fonte novas (Larivière 2022 modal e Maeda 2003 complementadas sobre as versões da sessão principal; Chadefaux 2024; Larivière 2023; Alves & Bezerra 2017; CONEM 0596, 1516, 1595; TCC Goeking 2023; Edge-computing só com estrutura, `confidencial: true`). `sistemas/nosso-sistema-antigo.md` reescrita; 8 páginas de conceito atualizadas. `references.bib`: +Lariviere2022, Maeda2003, Chadefaux2024, Lariviere2023, Alves2017; `Goeking2023` conferido na capa. Comentários com as iniciais da orientação neutralizados em `references.bib` (l. 3) e `manuscript.tex` (l. 144). Índice: 58 páginas, 0 links quebrados.  
**Achados que afetam o reprocessamento com a v3.0:**
- Firmware de 2024 (`BACKUP's/Software Eduardo/vibracoes/vibracoes.ino`, `numdados 2048`) e `.m` de 2024: **blocos de 2 s a cada 5 s, MPU6050 ±2 g** (o TCC de 2023 descreve 1 s).
- CONEM 1516: no encosto, y e z ficaram **sem ponderação**; o VDV diário saiu de √Σ VDV² e não da 4ª potência; horas paradas contaram como exposição. A v3.0 deve ponderar por posição, combinar eixos pela norma e usar só o tempo medido.
- Pendente: mapear os cartões SD 1A/2B/3B → posição (o arquivo das 10 h de 17/10/2024 em `SD 1A/1AA` começa às 10:05:40; o CONEM lista 10:13:15 para o encosto).

**Motivo:** retomar a ingestão da lista PENDENTES do `wiki/log.md`.

### Docs — wiki do artigo *Sensors* · exportação para HTML
**Arquivo:** `Publicação Sensors/scripts/wiki_html.py` (novo) → `Publicação Sensors/wiki.html`  
**Mudança:** gera um HTML único e autocontido da wiki (navegação lateral, busca, tema claro/escuro, impressão). Omite páginas com `confidencial: true` e aborta se encontrar o nome da orientação ou o arquivo descartado.  
**Motivo:** pedido do usuário para ler e compartilhar a wiki.

---

## 2026-10-13 (sessão 10)

### Bug fix — `fielddata.py` · arquivos fora de ordem cronológica na aba Directive
**Arquivo:** `iso26315/fielddata.py`  
**Problema:** `_make_day` concatenava registros na ordem alfabética dos nomes de arquivo. Quando os nomes não refletem o horário real (ex.: `rec_003.csv` gravado antes de `rec_004.csv`), os blocos ficavam fora de ordem e os timestamps se sobrepunham nos gráficos.  
**Mudança:** Quando todos os registros têm `absolute_time=True` (DateTime CSV), agora ordena por `r.seconds[0]` antes de concatenar. Registros sem timestamp absoluto (`.txt`, CSV Time_ms) mantêm o comportamento anterior (ordem de nome de arquivo).

---

## 2026-10-13 (sessão 9)

### Bug fix — todos os exportadores PDF · QProgressDialog aparecia branco
**Arquivos:** `gui.py`, `clause5_tab.py`, `field_panel.py`, `explorer.py`, `field_explorer.py`  
**Problema:** O popup "Gerando PDF…" aparecia como janela branca sem conteúdo visível.  
**Causa:** (1) Passar `None` como cancelButtonText no construtor do `QProgressDialog` causa bug de layout no Qt/Windows — o dialog não renderiza seu conteúdo. (2) Falta de `prog.show()` explícito antes de `processEvents()` — sem isso o dialog aparece mas não pinta.  
**Mudança:** Todos os 5 exportadores corrigidos: `None` → `''` no construtor, `setCancelButton(None)` para remover o botão corretamente, e `prog.show()` adicionado antes de `processEvents()`.

---

## 2026-10-13 (sessão 8)

### Bug fix — `field_panel.py` · bloco de desenho do PDF fora do try/except
**Arquivo:** `iso26315/field_panel.py`  
**Problema:** O bloco de desenho da página 1 do PDF Directive (~160 linhas, de `BLACK = 'black'` até `QApplication.processEvents()`) estava fora do `try/except/finally`. Exceções durante o desenho propagavam sem ser capturadas, sem fechar o `QProgressDialog` nem restaurar `rcParams`.  
**Mudança:** Adicionado `prog = None` + `try:` antes do bloco de desenho (linha ~1387); todo o bloco re-indentado para dentro do try; o `try:` interno que envolvia apenas o `PdfPages` foi removido (agora é código normal dentro do try externo). Estrutura final: `try` cobre desenho + `QProgressDialog` + `PdfPages`; `except/finally` existentes cobrindo tudo.  
**Como foi feito:** Script Python programático (contorno ao problema de `—` escape no Edit tool).

### Infraestrutura — Arquivo de estado do projeto criado
**Arquivo:** `memory/notes.md`  
**Mudança:** Criado `notes.md` no diretório de memória para persistir entre sessões o estado atual, pendências, decisões fixas e contexto técnico do projeto.

---

## 2026-10-11 (sessão 7)

### Bug fix + Feature — `field_explorer.py` · PDF export sem tratamento de erro
**Arquivo:** `iso26315/field_explorer.py`  
**Problema:** `export_pdf` não tinha `try/except` nem `QProgressDialog` — falhas silenciosas, sem feedback ao usuário.  
**Mudança:** Todo o bloco de desenho + `PdfPages` movido para `try/except/finally`; adicionado `QProgressDialog` modal com guard `if prog is not None`; figura sempre restaurada no `finally` mesmo em caso de erro.

---

## 2026-10-11 (sessão 6)

### Tooltip — `clause5_tab.py` · checkbox "3D waterfall" (Annex C)
**Arquivo:** `iso26315/clause5_tab.py`  
**Mudança:** `self.waterfall.setToolTip(...)` — "Show a 3D spectrogram of the loaded record (frequency × time × amplitude)."

---

### Tooltip — `field_panel.py` · checkbox "3D waterfall" (Directive)
**Arquivo:** `iso26315/field_panel.py`  
**Mudança:** `self.freq_waterfall.setToolTip(...)` — "Show a 3D spectrogram of the field day (frequency × time × amplitude)."

---

### Bug fix — `field_panel.py` · PDF Directive não gerava (3 causas)
**Arquivo:** `iso26315/field_panel.py`  
**Causas:**  
1. `hline()` usava `transform=ax.transAxes` — `axhline` não aceita esse argumento → exceção silenciosa.  
2. `ax.axhline(FOOTER_Y, ..., transform=ax.transAxes)` — mesmo erro no rodapé.  
3. `ROW_H` e `SEC_H` usados sem nunca serem definidos → `NameError` silencioso.  
**Mudanças:** removido `transform=` de ambos os `axhline`; adicionado `ROW_H = 0.013` e `SEC_H = 0.018` após `FOOTER_Y = 0.024`; `finally:` agora guarda `if prog is not None` para evitar crash quando exceção ocorre antes de `prog` ser criado.

---

## 2026-10-11 (sessão 5)

### Visual — `theme.py` · checkboxes exibem símbolo de check (✓)
**Arquivo:** `iso26315/theme.py`  
**Mudança:** `QCheckBox::indicator:checked` agora usa `image:` com SVG embutido em data URI (base64) mostrando um símbolo de checkmark branco sobre o fundo azul. Estado `checked:disabled` usa fundo cinza com o mesmo checkmark. Funciona no exe sem arquivos externos.

---

## 2026-10-11 (sessão 4)

### Bug fix — `explorer.py` · WeightingPanel cortado em Signal Records
**Arquivo:** `iso26315/explorer.py`  
**Motivo:** `self.weight.setMaximumHeight(150)` cortava a grade interna do WeightingPanel (labels + combos + k spinboxes + status word-wrap requerem ~204 px).  
**Mudança:** Aumentado para `setMaximumHeight(220)`.

---

### Feature — `explorer.py` · campo "Nome / ID do sujeito" no export PDF
**Arquivo:** `iso26315/explorer.py`  
**Mudança:** O diálogo de exportação PDF do Signal Records ganhou campo "Nome / ID do sujeito". O valor digitado aparece como primeira linha na caixa de metadados da página de relatório. `_build_report_page` aceita parâmetro `nome=''`.

---

### Feature — `field_panel.py` · checkbox "Treat files as continuous in time"
**Arquivo:** `iso26315/field_panel.py`  
**Mudança:** Adicionado checkbox no GroupBox Day. Quando marcado: no Timeline (k==0) o eixo x usa índice sequencial de blocos × `block_spacing` (sem lacunas); no Time series (k==6) o eixo x usa `np.arange × dt` (sem NaN breaks). Útil quando vários arquivos do mesmo dia têm timestamps desalinhados.

---

## 2026-10-11 (sessão 3)

### Bug fix — `weighting_panel.py` · combo "Axis combination" cortado em Signal Records
**Arquivo:** `iso26315/weighting_panel.py`  
**Motivo:** `setMinimumWidth(120)` era insuficiente para "Vector sum (one curve per record)"; coluna 3 recebia o mesmo stretch que as colunas de eixo (1/4 do espaço total).  
**Mudança:** `combine.setMinimumWidth(200)` + `setSizePolicy(Expanding, Fixed)`; stretch da coluna 3 aumentado para 2 (colunas de eixo ficam com 1).

---

### Feature — `clause5_tab.py` · seleção de gráficos no export PDF (Annex C)
**Arquivo:** `iso26315/clause5_tab.py`  
**Motivo:** O export do Annex C exportava apenas o gráfico ativo; sem opção de incluir outros.  
**Mudança:** O diálogo de metadata passou a listar 9 checkboxes (um por tipo de gráfico do combo), com "Sinal condicionado e resposta espinal" pré-marcado. O loop de exportação alterna `plot_kind`, chama `redraw()`, e salva cada figura selecionada como página A4 paisagem; ao final restaura o índice original.

---

## 2026-10-11 (sessão 2)

### Bug fix — `gui.py` + `clause5_tab.py` · `hline()` com `transform=ax.transAxes`
**Arquivos:** `iso26315/gui.py`, `iso26315/clause5_tab.py`  
**Motivo:** `ax.axhline()` não aceita `transform=` como argumento — matplotlib lança `TypeError`. Como `ax.set_ylim(0,1)`, as coordenadas de dados já coincidem com coordenadas de eixo, tornando o parâmetro desnecessário.  
**Mudança:** Removido `transform=ax.transAxes` do `hline()` em ambos os exportadores.

---

### Bug fix — `clause5_tab.py` · "Time series of raw a_z" em branco
**Arquivo:** `iso26315/clause5_tab.py`  
**Motivo:** `kind==8` acessava `self.az` diretamente, mas `_on_field_load_done` (carregamento via dia de campo) não inicializa `self.az` — levantava `AttributeError` silencioso no handler de sinal do PySide6.  
**Mudança:** Adicionado `hasattr(self, 'az')` guard; fallback para `np.concatenate(self.segments)` quando carregado via dia de campo.

---

### Bug fix — `field_panel.py` · seções Day/Assessment sumindo (Directive)
**Arquivo:** `iso26315/field_panel.py`  
**Motivo:** Envolver o `QHBoxLayout top` em `QWidget` + `CollapsibleSection` causava problema de sizing dentro do `QScrollArea` do `settings_splitter`.  
**Mudança:** Revertida a abordagem. Agora cada GroupBox (`src`=Day, `ext`=Assessment) é envolto individualmente em `CollapsibleSection('', box, expanded=True)` antes de ser adicionado ao `top`. O `lay.addLayout(top)` permanece direto.

---

## 2026-10-11

### Bug fix — `gui.py` · `export_pdf` (Annex A)
**Arquivo:** `iso26315/gui.py`  
**Motivo:** Exceções lançadas durante o desenho das figuras matplotlib eram silenciadas pelo PySide6 porque o bloco `try/except` só envolvia o `PdfPages`, não o código de desenho.  
**Mudança:** Todo o código de geração do PDF (figuras + `PdfPages` + `QProgressDialog`) foi movido para dentro de um único `try/except/finally`. `prog = None` é inicializado antes do `try`; o `finally` só fecha o dialog se `prog is not None`.

### Feature — `gui.py` · `export_pdf` — exportar domínio do tempo
**Arquivo:** `iso26315/gui.py`  
**Mudança:** Adicionado checkbox "Time domain: compression peaks" ao diálogo de exportação de PDF do Annex A. Quando marcado, `self.update_time()` é chamado e `self.time_plot.figure` é salvo como página extra (paisagem A4) no PDF.

### Bug fix — `clause5_tab.py` · `export_pdf` (Annex C)
**Arquivo:** `iso26315/clause5_tab.py`  
**Motivo:** Mesmo problema de `try/except` que o Annex A — exceções durante o desenho das figuras eram silenciadas.  
**Mudança:** Todo o bloco de geração do PDF movido para dentro de `try/except/finally` com `prog = None` antes do `try`. `QApplication` e `QMessageBox`/`QProgressDialog` agora importados no topo do arquivo.

### Bug fix — `field_panel.py` · combos com texto cortado (Directive)
**Arquivo:** `iso26315/field_panel.py`  
**Motivo:** Combos `self.criterion`, `self.basis` e `self.kind` eram redimensionados para larguras mínimas insuficientes pelo layout, cortando o texto dos itens.  
**Mudança:** Adicionado `setMinimumWidth(240)` + `setSizePolicy(Expanding, Fixed)` a `criterion` e `basis`; `setMinimumWidth(260)` + `setSizeAdjustPolicy(AdjustToContents)` ao `kind`.

### Feature — `field_panel.py` · seção Day & Assessment colapsável
**Arquivo:** `iso26315/field_panel.py`  
**Mudança:** Os GroupBoxes "Day" e "Assessment" agora ficam dentro de um `CollapsibleSection('Day & Assessment settings', …, expanded=True)`, tornando toda a área de configurações do painel Directive dobrável com um clique no cabeçalho.

---

## 2026-10-10

### Bug fix — `core.py` · `signal_statistics`
**Arquivo:** `iso26315/core.py`  
**Função:** `signal_statistics()`  
**Mudança:** Adicionada verificação antes de `np.abs(exc.accel_spectrum)`. Se `accel_spectrum` for `None`, chama `_prepare(exc)` automaticamente antes de prosseguir.  
**Motivo:** Qualquer `Excitation` construída manualmente sem passar por `_prepare` causava `TypeError: bad operand type for abs(): 'NoneType'`. O guard torna a função auto-curativa.

---

### Bug fix — `explorer.py` · `_field_to_excitation`
**Arquivo:** `iso26315/explorer.py`  
**Função:** `_field_to_excitation(rec, name)`  
**Mudança:** Adicionada chamada `core._prepare(exc)` após construir o objeto `Excitation` a partir dos blocos do registrador de campo `.m`.  
**Motivo:** Carregamento de arquivos `.m` falhava com `bad operand type for abs(): 'NoneType'` porque o espectro nunca era calculado.

---

### Bug fix — `explorer.py` · `join_selected`
**Arquivo:** `iso26315/explorer.py`  
**Função:** `join_selected()`  
**Mudança:** Adicionada chamada `_core._prepare(exc)` após criar a `Excitation` concatenada.  
**Motivo:** "Join selected → 1 signal" falhava com o mesmo erro de `NoneType` ao tentar calcular `signal_statistics` no sinal unido.

---

### Feature — `explorer.py` · loading indicator em `_add_files`
**Arquivo:** `iso26315/explorer.py`  
**Funções:** `_add_files()`, `_end_add()`  
**Mudança:** Botão "Add files..." passa a mostrar "Reading N/M…" durante o carregamento; cursor muda para `Qt.WaitCursor` enquanto lê; restaurado ao terminar.  
**Motivo:** Janela ficava congelada sem feedback ao carregar muitos arquivos grandes.

---

## 2026-10-11

### Feature — `explorer.py` · `join_selected` desmarca fontes
**Arquivo:** `iso26315/explorer.py`  
**Função:** `join_selected()`  
**Mudança:** Após criar o sinal unido, todos os itens-fonte são desmarcados na lista (`Qt.Unchecked`) via `list.blockSignals(True/False)` para evitar redesenhos intermediários.  
**Motivo:** Após o join, o software ficava extremamente lento porque `refresh()` redesenhava todos os sinais originais (potencialmente dezenas de arquivos de 1 hora) além do sinal unido. Desmarcando as fontes, apenas o sinal unido é plotado.

---

### Fix UI — `weighting_panel.py` · margem superior do painel
**Arquivo:** `iso26315/weighting_panel.py`  
**Função:** `WeightingPanel.__init__()`  
**Mudança:** `g.setContentsMargins(8, 4, 8, 4)` → `g.setContentsMargins(8, 14, 8, 4)` (margem superior de 4 px para 14 px).  
**Motivo:** Título do `QGroupBox` "Frequency weighting (display only)" sobrepunha levemente a combo de preset.

---

### Feature — `field_panel.py` · campo "Data da medição" nos metadados
**Arquivo:** `iso26315/field_panel.py`  
**Função:** `_build()` (seção de metadados) e `export_pdf()`  
**Mudança:** Adicionado `QLineEdit` para "Data da medição" entre "Operador" e "Sensor / modelo"; campo incluído na seção "INFORMAÇÕES DO SUJEITO / MEDIÇÃO" do PDF exportado.  
**Motivo:** Usuário precisava registrar a data da coleta no relatório exportado.

---

### Feature — `field_panel.py` · diálogo de export PDF melhorado + loading indicator
**Arquivo:** `iso26315/field_panel.py`  
**Função:** `export_pdf()`  
**Mudança:**  
- Diálogo de seleção de gráficos ganhou título estilizado, separador visual e botão "Exportar PDF" (era apenas "OK").  
- `QProgressDialog` indeterminado aparece durante a geração do PDF ("Gerando PDF…").  
**Motivo:** Diálogo anterior estava sem estilo e não havia feedback visual durante a geração (que pode levar vários segundos com múltiplos gráficos).

---

### Feature — `clause5_tab.py` · multi-arquivo no Annex C
**Arquivo:** `iso26315/clause5_tab.py`  
**Função:** `open_file()` e `_on_open_done()`  
**Mudança:** `QFileDialog.getOpenFileName` → `QFileDialog.getOpenFileNames`; múltiplos arquivos são lidos e concatenados no eixo z; valida taxa de amostragem idêntica entre arquivos.  
**Motivo:** Medições longas (ex.: 12 h) são gravadas em arquivos de 1 h; antes só era possível carregar um por vez no Annex C.

---

### Feature — `clause5_tab.py` · export PDF no Annex C
**Arquivo:** `iso26315/clause5_tab.py`  
**Função:** `export_pdf()` (nova), `_busy()`, `_build()`, `_on_run_done()`  
**Mudança:** Adicionado botão "Export PDF report..." abaixo dos resultados; habilitado após o cálculo concluir; gera página A4 retrato com tabela de resultados + diálogo de metadados (sujeito, data, operador, notas) + página A4 paisagem com gráfico atual.  
**Motivo:** Annex C não tinha opção de exportar resultados em PDF, ao contrário da aba Directive.

---

### Feature — `gui.py` · export PDF no Annex A (Results)
**Arquivo:** `iso26315/gui.py`  
**Funções:** `_build_results()`, `export_pdf()` (nova), `_on_assessment_done()`  
**Mudança:** Adicionado botão "Export PDF report..." ao lado de "Export spreadsheet..." na aba Results; diálogo coleta metadados (sujeito, data, operador) e seleção de gráficos espectrais (Welch/Hann ativado por padrão, transfer functions e force spectrum opcionais); gera PDF com tabela de R^A por disco colorida por risco + páginas de gráficos A4 paisagem.  
**Motivo:** Annex A / Annex E não tinha exportação de PDF; apenas planilha Excel.

---

### Fix UI — `weighting_panel.py` · combo Preset cortada
**Arquivo:** `iso26315/weighting_panel.py`  
**Função:** `WeightingPanel.__init__()`  
**Mudança:** Substituído `QGridLayout` único por `QVBoxLayout` externo contendo um `QHBoxLayout` para a linha de Preset e um `QGridLayout` para os controles por eixo. A combo Preset agora tem `stretch=1` e pode expandir livremente.  
**Motivo:** A combo "Preset" e as combos de eixo ficavam cortadas/clipadas dentro do `QGroupBox` porque o grid não permitia expansão livre da coluna.

---

### Feature — `explorer.py` · cursor de espera em "Join selected → 1 signal"
**Arquivo:** `iso26315/explorer.py`  
**Função:** `join_selected()`  
**Mudança:** `QApplication.setOverrideCursor(Qt.WaitCursor)` antes do processamento; `restoreOverrideCursor()` tanto no caminho de erro quanto no de sucesso.  
**Motivo:** Operação podia levar vários segundos; janela parecia travada sem feedback.

---

### Feature — `explorer.py` · tooltip para "3D waterfall (spectrogram)"
**Arquivo:** `iso26315/explorer.py`  
**Widget:** `self.waterfall` (`QCheckBox`)  
**Mudança:** `setToolTip()` explicando o que é o gráfico 3D waterfall/espectrograma e quando é útil.  
**Motivo:** Usuário não sabia o que o checkbox fazia.

---

### Feature — `explorer.py` · label + tooltip para "Integration HP order check"
**Arquivo:** `iso26315/explorer.py`  
**Widget:** `_lbl_int_hp` (`QLabel`)  
**Mudança:** Convertido de `QLabel` inline para variável com `setToolTip()` detalhado explicando o problema de drift na integração numérica e a regra de ouro HP order ≥ n + 1.  
**Motivo:** Usuário não entendia para que serve a ferramenta.

---

### Fix UI — `explorer.py` · renomeada opção "Waveform: raw amplitude vs time"
**Arquivo:** `iso26315/explorer.py` (via `field_panel.py`)  
**Combo:** `self.kind`  
**Mudança:** `'Waveform: raw amplitude vs time'` → `'Time series: raw acceleration vs time'` (índice k==6).  
**Motivo:** Nome anterior não deixava claro que era o "time series" de aceleração bruta; usuário não encontrava a opção.

---

### Fix UI — `gui.py`, `clause5_tab.py`, `field_panel.py` · PDF preto-e-branco com fonte serifada
**Arquivos:** `iso26315/gui.py`, `iso26315/clause5_tab.py`, `iso26315/field_panel.py`  
**Funções:** `export_pdf()` em cada arquivo  
**Mudança:**  
- `matplotlib.rcParams['font.family'] = 'serif'` durante a geração (restaurado em `finally`).  
- Paleta redefinida: cabeçalho negro (`'black'`) com título branco; fundos coloridos substituídos por `#f2f2f2` (cinza claro) e linhas horizontais finas (`hline()`); texto principal em preto; texto secundário em `#555555`.  
- Rodapé passa a usar `hline()` em vez de retângulo colorido.  
**Motivo:** PDF exportado usava cores intensas (azul marinho, caixas coloridas) e fonte sans-serif; solicitação era visual mais sério/acadêmico, preto-e-branco, estilo LaTeX.

---

### Feature — `clause5_tab.py` · gráfico "Time series of raw a_z" no Annex C
**Arquivo:** `iso26315/clause5_tab.py`  
**Combo:** `self.plot_kind` (índice 8)  
**Mudança:** Adicionada opção `'Time series of raw a_z (informative)'`; `redraw()` trata `kind == 8` plotando `self.az` decimado contra tempo (stride automático para ≤ 500 000 pontos no eixo), com label "Raw $a_z$ [m/s²]".  
**Motivo:** Usuário queria ver aceleração bruta vs tempo no Annex C.

---

### Feature — `gui.py` · gráfico "Time series: raw acceleration vs time" no Annex A
**Arquivo:** `iso26315/gui.py`  
**Combo:** `self.spec_kind` (índice 9)  
**Função:** `_plot_timeseries_raw()` (nova), `update_spec()`  
**Mudança:** Adicionada opção 'Time series: raw acceleration vs time' (índice 9) no combo de gráficos do Annex A; `update_spec()` roteia para `_plot_timeseries_raw()` que plota `exc.accel[:, axis]` decimado (stride automático, ≤ 500 000 pontos) com labels de eixo adequados; `spec_loc` e `spec_axis` habilitados para esta view.  
**Motivo:** Usuário queria ver a aceleração bruta vs tempo no Annex A.

---

### Feature — `clause5_tab.py` · indicador de progresso no export PDF (Annex C)
**Arquivo:** `iso26315/clause5_tab.py`  
**Função:** `export_pdf()`  
**Mudança:** Adicionado `QProgressDialog` indeterminado ('Gerando PDF…') durante a geração, com `WindowModal` e `reset()` no bloco `finally`.  
**Motivo:** Geração de PDF no Annex C não tinha feedback visual; comportamento agora equivalente ao Directive e Annex A.

---

### Refactor — `gui.py` · removidas 3 abas de importação obsoletas
**Arquivo:** `iso26315/gui.py`  
**Mudança:** Removidos imports de `CsvImportPanel`, `DatetimeCsvImportPanel`, `TxtImportPanel` e as 3 abas correspondentes ("Continuous CSV import", "ASCII .txt import", "DateTime CSV import") do Signal Explorer.  
**Motivo:** Esses formatos de importação não são mais usados no fluxo atual (.txt/.csv/.m são carregados diretamente); as abas criavam confusão.

---

---

## 2026-10-11 (sessão 3 — correção de exportadores PDF)

### Bug fix — `gui.py` · `QDialogButtonBox` ausente dos imports
**Arquivo:** `iso26315/gui.py`
**Função:** `export_pdf()`, imports do módulo
**Mudança:** Adicionado `QDialogButtonBox` ao bloco de imports do topo do módulo.
**Motivo:** `export_pdf()` usava `QDialogButtonBox` sem importá-lo no topo; o clique no botão "Export PDF report..." lançava `NameError` silenciosamente, parecendo que o botão não fazia nada.

---

### Bug fix — `clause5_tab.py` · `r.daily_time` / `r.days_per_year` inexistentes
**Arquivo:** `iso26315/clause5_tab.py`
**Função:** `export_pdf()`
**Mudança:** Substituído `r.daily_time` por `self.daily.value()` e `r.days_per_year` por `self.days.value()` na tabela de parâmetros do PDF.
**Motivo:** `ClauseFiveResult` não possui os campos `daily_time` nem `days_per_year`; a referência levantava `AttributeError` antes de o bloco `with PdfPages()` ser alcançado, impedindo a geração do arquivo.

---

### Bug fix — `field_panel.py` · `data_rows` não definida
**Arquivo:** `iso26315/field_panel.py`
**Função:** `export_pdf()`
**Mudança:** Adicionada linha `data_rows = table_rows[1:]` logo após a construção de `table_rows`, separando o cabeçalho CSV das linhas renderizadas no PDF.
**Motivo:** `data_rows` era referenciada no corpo da função mas nunca atribuída; `NameError` ocorria antes de abrir o `PdfPages`, impedindo a geração do arquivo PDF no Directive.

---

### Feature — `clause5_tab.py` · limpar gráfico ao dar Clear
**Arquivo:** `iso26315/clause5_tab.py`
**Função:** `clear()`
**Mudança:** Adicionadas chamadas `self.plot.figure.clear()` e `self.plot.canvas.draw()` no final de `clear()`.
**Motivo:** Ao clicar em Clear, o gráfico anterior permanecia visível no canvas; agora é apagado junto com os dados.

---

### Feature — `explorer.py` · indicador de progresso no export PDF (Signal Records)
**Arquivo:** `iso26315/explorer.py`
**Função:** `export_pdf()`
**Mudança:** Adicionado `QProgressDialog` indeterminado ('Gerando PDF…') com `WindowModal` e `reset()` no `finally`; bloco envolto em `try/except` com `QMessageBox.critical` para erros.
**Motivo:** Signal Records era o único exportador de PDF sem indicador de progresso; adicionado para consistência com Directive, Annex A e Annex C.

---

### Feature — todos os exportadores PDF · `except` com QMessageBox
**Arquivos:** `iso26315/gui.py`, `iso26315/clause5_tab.py`, `iso26315/field_panel.py`, `iso26315/explorer.py`
**Função:** `export_pdf()` em cada arquivo
**Mudança:** Adicionado bloco `except Exception as _e: QMessageBox.critical(...)` em todos os exportadores; exceções antes silenciosas agora aparecem como diálogo de erro para o usuário.
**Motivo:** Falhas na geração do PDF eram descartadas sem feedback; o usuário via "nenhum arquivo gerado" sem saber o motivo.

---

### Fix UI — `weighting_panel.py` · combos e spinboxes de eixo cortando texto
**Arquivo:** `iso26315/weighting_panel.py`
**Função:** `WeightingPanel.__init__()`
**Mudança:** Adicionados `setMinimumWidth(90)` e `setSizePolicy(Expanding, Fixed)` às combos de eixo; `setMinimumWidth(70)` / `setMaximumWidth(90)` nos spinboxes `k`; `setMinimumWidth(120)` na combo "Axis combination".
**Motivo:** Combos e spinboxes no painel de ponderação clipavam o texto dos itens; largura mínima garante leitura completa em qualquer resolução.

---

## 2026-10-11 (sessão 2 — documentação TCC)

### Docs — `revisao_vci_v4.tex` · adição de B38, B39, B40
**Arquivo:** `TCC/revisao/revisao_vci_v4.tex`
**Mudança:** Adicionadas três entradas de nível B de transporte neonatal: B38 (Blaxter et al. 2017, ambulância terrestre com 12 neonatos reais), B39 (Solvik-Olsen e Heyerdahl 2024, helicóptero, manequins), B40 (Goswami et al. 2020, revisão narrativa de 11 estudos). Cada entrada inclui bloco `\entrada{}` completo com resumo, contexto e relevância para a lacuna normativa. Adicionados parágrafos sobre transporte neonatal em "Resultados bem apoiados" e em "Caminhos de pesquisa / Estender o protocolo". Lista de referências e quadro geral atualizados. Contagem total: 75 → 78.
**Motivo:** Os três artigos documentam empiricamente que a ISO 2631-x não foi validada para neonatos recumbentes e que os limites adultos são aplicados por ausência de alternativa, reforçando o argumento central do TCC.

### Docs — `TCC/wiki/catalogo.md`, `log.md`
**Mudança:** B38/B39/B40 movidos de "fora do catálogo" para o catálogo principal; contagem atualizada de 75 para 78; nota sobre N4 (ex-B38) esclarecida; log de 2026-10-11 adicionado. Wiki HTML regenerada (116 páginas, 1495 KB).

---

## Convenção para entradas futuras

```
### Tipo — arquivo · função/seção afetada
**Arquivo:** caminho relativo a iso26315/  
**Função:** nome(s)  
**Mudança:** o que foi alterado  
**Motivo:** por que foi feito  
```

Tipos: `Bug fix`, `Feature`, `Fix UI`, `Refactor`, `Docs`
