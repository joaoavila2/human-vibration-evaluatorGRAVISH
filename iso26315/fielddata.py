"""Leitura e analise dos registros de campo do GRAVI-SH.

MEDICOES ATUAIS SAO CONTINUAS. Os formatos em uso gravam a aceleracao sem
interrupcao: o CSV "DateTime,AX,AY,AZ" do firmware VCI_nRF52840_accel_SD
(833 Hz), o binario VIB1 (.dat) do firmware do XIAO nRF52840, e ainda o
CSV "Time_ms,AX,AY,AZ" de bancada e o .txt ASCII de 4 colunas. Todos viram
FieldRecord com continuous=True: o sinal e fatiado em blocos consecutivos so
por conveniencia de armazenamento, e todo o processamento que depende de
continuidade (ponderacao, MTVV, Clausula 5) trata a sequencia como um sinal
unico (ver continuity.py). load_day_auto escolhe o leitor pela extensao e
pelo cabecalho.

FORMATO ANTIGO (.m). O registrador anterior gravava arquivos .m que sao
scripts MATLAB: blocos de DadosValores(:,:,len)=int16([...]) precedidos de um
carimbo de tempo do RTC. Esse modo de gravacao nao e mais usado; os arquivos
continuam legiveis para reanalise de dados antigos.

Caracteristicas do formato .m, confirmadas no firmware (vibracoes.ino):

  - MPU6050 em +-2 g, portanto 16384 LSB/g; os valores sao contagens brutas,
    sem nenhuma conversao ou calibracao aplicada;
  - 2048 amostras por bloco a 1024 Hz nominais, ou seja 2 s de sinal;
  - um bloco a cada 5 s de relogio (o firmware espera Second() % 5 == 0), de
    modo que 3 s em cada 5 nao sao gravados. O ciclo de trabalho e de 40% e e
    intencional.

Nesses arquivos antigos a descontinuidade entre blocos e a razao para NAO
concatenar os blocos e transformar o conjunto: cada colagem e um degrau que a
transformada espalha por todo o espectro. Cada bloco e um trecho isolado, e a
dose de sexta potencia do Anexo A nao se aplica (60% dos picos nao foram
gravados).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import continuity

LSB_PER_G = 16384.0          # +-2 g, registrador 0x1C = 0b00000000
GRAVITY_BH = 9.784           # m/s2, gravidade local em Belo Horizonte
NOMINAL_RATE = 1024.0
BLOCK_SAMPLES = 2048
BLOCK_PERIOD = 5.0           # s entre carimbos no formato .m antigo

# Remocao da componente estatica em registros continuos: um offset constante
# por eixo, a media do registro inteiro -- a correcao de offset do quarto
# passo de 5.1.3 da ISO 2631-5. Nada mais agressivo e possivel: a Clausula 5
# passa o sinal a partir de 0,01 Hz (sexto passo), e uma media movel de
# segundos funcionaria como um passa-alta acima disso, deformando quedas
# livres longas. A mudanca lenta da gravidade com a inclinacao do assento
# fica para os filtros de cada cadeia (0,4 Hz no Wk/Wd, 0,08 Hz no Wf,
# 0,01 Hz na Clausula 5).

# Curva de tendencia do CONEM2024: diferenca entre o MPU6050 e o padrao ouro
# ICP 353B34, em m/s2, em funcao da frequencia. Levantada em uma unidade; a
# variacao entre unidades pode ser maior que a propria correcao.
CONEM_SLOPE = -0.00278
CONEM_INTERCEPT = 0.03618

_TS = re.compile(
    r'\.dia=uint8\((\d+)\);\s*\S*?\.mes=uint8\((\d+)\);\s*\S*?\.ano=uint16\((\d+)\);'
    r'\s*\S*?\.hora=uint8\((\d+)\);\s*\S*?\.minuto=uint8\((\d+)\);'
    r'\s*\S*?\.segundo=uint8\((\d+)\);', re.S)


@dataclass
class FieldRecord:
    """Um arquivo (ou segmento) de aceleracao em blocos de BLOCK_SAMPLES.

    continuous=True: os blocos sao fatias consecutivas de um sinal continuo
    (todos os formatos atuais). False: blocos isolados do .m antigo.
    absolute_time=True quando seconds e hora do dia de verdade (DateTime,
    VIB1, .m); False quando o arquivo nao tem relogio e seconds comeca em 0.
    """
    path: Path
    counts: np.ndarray                  # (n_blocos, 2048, 3), contagens brutas
    seconds: np.ndarray                 # segundos do dia de cada bloco
    rate: float = NOMINAL_RATE
    scale: np.ndarray = None            # fator por eixo, da calibracao
    axis_map: tuple = (0, 1, 2)         # coluna do arquivo -> eixo x, y, z
    axis_sign: tuple = (1, 1, 1)
    gravity_axis: int = None
    notes: list = field(default_factory=list)
    continuous: bool = False
    absolute_time: bool = True

    @property
    def n_blocks(self):
        return self.counts.shape[0]

    @property
    def dt(self):
        return 1.0 / self.rate

    @property
    def recorded_time(self):
        return self.n_blocks * BLOCK_SAMPLES / self.rate

    @property
    def span(self):
        return float(self.seconds[-1] - self.seconds[0] + BLOCK_SAMPLES / self.rate)

    @property
    def duty_cycle(self):
        return self.recorded_time / self.span

    def blocks_si(self, remove_gravity=True):
        """Blocos em m/s2, ja orientados, calibrados e sem a componente estatica.

        Registro continuo: a componente estatica e a media de cada eixo no
        registro inteiro (offset constante), e o array devolvido fica
        marcado como uma faixa continua. Formato .m antigo: media de cada
        bloco, que e isolado.
        """
        a = self.counts / LSB_PER_G * GRAVITY_BH
        if self.scale is not None:
            a = a * self.scale
        a = a[:, :, list(self.axis_map)] * np.array(self.axis_sign)
        if not self.continuous:
            if remove_gravity:
                a = a - a.mean(axis=1, keepdims=True)
            return a
        if remove_gravity:
            a = a - a.mean(axis=(0, 1))
        return continuity.mark(a, [(0, self.n_blocks)])


def read_m(path, rate=NOMINAL_RATE, max_blocks=None):
    """Le um arquivo .m do registrador."""
    path = Path(path)
    txt = path.read_text(errors='replace')
    stamps = [(int(d), int(mo), int(y), int(h), int(mi), int(s))
              for d, mo, y, h, mi, s in _TS.findall(txt)]
    chunks = txt.split('DadosValores(:,:,len)=int16([')[1:]
    if max_blocks:
        chunks, stamps = chunks[:max_blocks], stamps[:max_blocks]

    data, secs, dropped = [], [], 0
    for stamp, chunk in zip(stamps, chunks):
        body = chunk.split(']', 1)[0]
        # Conversao vetorizada: uma ordem de grandeza mais rapida que analisar
        # linha a linha, o que importa com gigabytes de texto.
        vals = np.array(body.split(), dtype=np.int32)
        if vals.size != BLOCK_SAMPLES * 3:
            dropped += 1          # bloco truncado, tipicamente o ultimo
            continue
        data.append(vals.reshape(BLOCK_SAMPLES, 3).astype(np.int16))
        secs.append(stamp[3] * 3600 + stamp[4] * 60 + stamp[5])

    counts = np.stack(data)
    # Falha de leitura I2C devolve 0xFFFF, que como int16 e -1. No eixo da
    # gravidade isso e um salto de ~16000 contagens (~1 g) numa unica amostra,
    # que infla pico e VDV. Mesmo tratamento do ConverteDados.m do laboratorio:
    # substituir pela media do bloco naquele eixo.
    bad = counts == -1
    n_bad = int(bad.sum())
    if n_bad:
        counts = counts.astype(np.float64)
        means = np.where(bad, np.nan, counts)
        block_mean = np.nanmean(means, axis=1, keepdims=True)
        counts = np.where(bad, block_mean, counts)
    rec = FieldRecord(path=path, counts=counts,
                      seconds=np.array(secs, dtype=float), rate=rate)
    if n_bad:
        rec.notes.append(f'{n_bad} leitura(s) -1 (falha I2C) substituida(s) '
                         f'pela media do bloco')
    if dropped:
        rec.notes.append(f'{dropped} bloco(s) incompleto(s) descartado(s)')
    rec.gravity_axis = int(np.argmax(np.abs(rec.counts[0].mean(0))))
    return rec


def stationary_blocks(rec, tol=0.15):
    """Indices dos blocos em que o sistema esta comprovadamente parado.

    Criterio: desvio padrao do sinal, ja sem a componente estatica, abaixo de
    tol. O piso de ruido do MPU6050 fica em torno de 0,07 m/s2, entao qualquer
    movimento real fica bem acima disso.
    """
    a = rec.counts / LSB_PER_G * GRAVITY_BH
    spread = np.linalg.norm((a - a.mean(1, keepdims=True)).std(1), axis=1)
    return np.flatnonzero(spread < tol)


def calibrate_from_gravity(rec, indices=None):
    """Fator de escala por eixo, usando a gravidade como referencia absoluta.

    Com o sistema parado, o modulo do vetor medido tem de ser a gravidade
    local. ATENCAO: o desvio pode ser erro de sensibilidade (ganho) OU erro de
    zero (offset), e com uma unica orientacao os dois sao indistinguiveis. O
    MPU6050 admite offset de ate 80 mg no eixo z, cerca de 0,78 m/s2, o que
    sozinho explica desvios de ate 8% no modulo. Offset nao afeta medicao de
    vibracao, porque a componente estatica e removida; corrigir como se fosse
    ganho introduziria erro. Por isso esta correcao vem DESLIGADA por padrao.
    A separacao correta exige calibracao em seis posicoes (cada eixo para cima
    e para baixo).
    Esta correcao e um ganho constante e nao substitui a curva de resposta em
    frequencia do CONEM2024, que corrige outra coisa.
    """
    idx = stationary_blocks(rec) if indices is None else np.asarray(indices)
    if not len(idx):
        rec.notes.append('nenhum bloco parado encontrado; escala nao corrigida')
        return None
    a = rec.counts[idx] / LSB_PER_G * GRAVITY_BH
    mean = a.reshape(-1, 3).mean(0)
    magnitude = float(np.linalg.norm(mean))
    factor = GRAVITY_BH / magnitude
    rec.scale = np.full(3, factor)
    rec.notes.append(
        f'escala corrigida por {factor:.4f} a partir de {len(idx)} bloco(s) '
        f'parado(s): |g| medido {magnitude:.3f} contra {GRAVITY_BH:.3f} m/s2')
    return factor


def set_orientation(rec, vertical_column, invert=False, position='seat'):
    """Define qual coluna do arquivo e o eixo vertical z do sistema basicentrico.

    As demais colunas ficam em x e y na ordem em que aparecem. invert troca o
    sinal do eixo vertical, para o caso de o sensor estar montado de cabeca
    para baixo (gravidade positiva no arquivo).
    """
    others = [c for c in range(3) if c != vertical_column]
    if position == 'backrest':
        # No encosto os dois eixos horizontais nao sao equivalentes: x e o
        # perpendicular ao encosto e recebe Wc, y e o lateral. Um encosto
        # reclinado inclina a gravidade em direcao ao eixo perpendicular, entao
        # o horizontal com maior componente estatica da gravidade e o x.
        g = np.abs(rec.counts.mean(axis=(0, 1)))
        others.sort(key=lambda c: -g[c])
    rec.axis_map = (others[0], others[1], vertical_column)
    rec.axis_sign = (1, 1, -1 if invert else 1)
    rec.notes.append(
        f'coluna {vertical_column + 1} do arquivo tratada como eixo vertical z')


def conem_correction(freq):
    """Correcao de resposta em frequencia do CONEM2024, em m/s2 aditivos."""
    return CONEM_SLOPE * np.asarray(freq, dtype=float) + CONEM_INTERCEPT


def block_spectrum(rec, axis=None, window='hann', apply_conem=False,
                   remove_gravity=True):
    """Autoespectro de amplitude, usando cada bloco como um segmento de Welch.

    Nenhuma colagem entre blocos, portanto nenhuma descontinuidade artificial.
    Devolve (frequencias, amplitude, numero de medias).
    """
    from scipy.signal import get_window
    blocks = rec.blocks_si(remove_gravity)
    cols = [axis] if axis is not None else [0, 1, 2]
    n = BLOCK_SAMPLES
    w = get_window(window, n, fftbins=True)
    f = np.arange(n // 2 + 1) * rec.rate / n

    total = np.zeros(len(f))
    for c in cols:
        acc = np.zeros(len(f))
        for blk in blocks[:, :, c]:
            X = np.fft.rfft((blk - blk.mean()) * w)
            acc += np.abs(X) ** 2
        amp = np.sqrt(acc / len(blocks)) * 2.0 / np.sum(w)
        amp[0] /= 2
        amp[-1] /= 2
        total += amp ** 2
    total = np.sqrt(total)

    if apply_conem:
        total = np.maximum(total + conem_correction(f), 0.0)
    return f, total, len(blocks)


def block_metrics(rec, remove_gravity=True):
    """Valor eficaz e VDV por eixo, tratando cada bloco como uma amostra."""
    blocks = rec.blocks_si(remove_gravity)
    dt = rec.dt
    rms = np.sqrt((blocks ** 2).mean(axis=1))              # (n_blocos, 3)
    vdv = (np.sum(blocks ** 4, axis=1) * dt) ** 0.25
    return {
        'rms_per_block': rms,
        'rms': np.sqrt((rms ** 2).mean(0)),
        'vdv_per_block': vdv,
        'vdv_total': (np.sum(vdv ** 4, axis=0)) ** 0.25,
        'peak': np.abs(blocks).max(axis=(0, 1)),
        'recorded_time': rec.recorded_time,
        'span': rec.span,
        'duty_cycle': rec.duty_cycle,
    }


def field_statistics(rec):
    """Estatisticas descritivas de um arquivo .m, para inspecao rapida.

    Fora do escopo da ISO 2631-5 e da NHO-09: nao monta um dia, nao assume
    posicao nem calcula nenhum criterio de saude ou conforto. So descreve o
    arquivo -- mesmo espirito do core.signal_statistics para os registros
    ASCII do Anexo A, adaptado ao formato de blocos do registrador.
    """
    m = block_metrics(rec)
    blocks = rec.blocks_si()
    flat = blocks.reshape(-1, 3)
    var = flat.var(0)
    kurt = np.divide(((flat - flat.mean(0)) ** 4).mean(0), var ** 2,
                     out=np.full(3, np.nan), where=var > 0)
    dom = np.full(3, np.nan)
    for c in range(3):
        f, amp, _ = block_spectrum(rec, axis=c)
        if len(f) > 1:
            dom[c] = f[1:][amp[1:].argmax()]      # ignora o bin DC
    peak = m['peak']
    return {
        'n_blocks': rec.n_blocks,
        'recorded_time': m['recorded_time'],
        'span': m['span'],
        'duty_cycle': m['duty_cycle'],
        'rms': m['rms'],
        'rms_total': float(np.sqrt((m['rms'] ** 2).sum())),
        'peak': peak,
        'crest': np.divide(peak, m['rms'], out=np.zeros(3), where=m['rms'] > 0),
        'vdv': m['vdv_total'],
        'kurtosis': kurt,
        'dominant': dom,
        'gravity_axis': rec.gravity_axis,
        'axis_map': rec.axis_map,
        'notes': list(rec.notes),
    }


# ----------------------------------------------------------------------
# Ocupacao da cadeira e carregamento de um dia
# ----------------------------------------------------------------------

OCCUPANCY_WINDOW_BLOCKS = 60      # 5 min de relogio
OCCUPANCY_TILT_STD = 0.05         # graus


def tilt_angles(rec):
    """Inclinacao do eixo vertical em relacao a gravidade, por bloco, em graus."""
    g = (rec.counts / LSB_PER_G * GRAVITY_BH).mean(axis=1)
    axis = rec.gravity_axis
    norm = np.linalg.norm(g, axis=1)
    ratio = np.divide(np.abs(g[:, axis]), norm, out=np.ones(len(norm)),
                      where=norm > 0)
    return np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0)))


def occupancy_mask(rec, window=OCCUPANCY_WINDOW_BLOCKS,
                   threshold=OCCUPANCY_TILT_STD):
    """Blocos em que a cadeira provavelmente estava ocupada.

    HEURISTICA. Uma pessoa sentada redistribui o peso continuamente, e a
    orientacao do assento oscila de 1 a 2 graus mesmo sem deslocamento. Com a
    cadeira vazia a orientacao fica congelada, com desvio de centesimos de
    grau. O criterio e o desvio da inclinacao numa janela movel.

    A classificacao e apenas informativa: o resultado da NHO-09 e invariante a
    ela desde que o tempo de exposicao e os blocos venham do mesmo periodo.
    """
    tilt = tilt_angles(rec)
    half = window // 2
    spread = np.array([tilt[max(0, i - half): i + half + 1].std()
                       for i in range(len(tilt))])
    return spread > threshold


@dataclass
class FieldDay:
    """Varios arquivos de uma mesma unidade, tratados como um dia continuo."""
    records: list
    blocks: np.ndarray            # (n_blocos, 2048, 3), m/s2, orientado
    seconds: np.ndarray           # instante de cada bloco, em segundos do dia
    occupied: np.ndarray          # mascara booleana
    rate: float

    def __post_init__(self):
        self.mark_runs()

    def mark_runs(self):
        """Marca em continuity os blocos de cada registro continuo. Chamar
        de novo sempre que self.blocks for substituido."""
        runs, pos = [], 0
        for r in self.records:
            if r.continuous:
                runs.append((pos, pos + r.n_blocks))
            pos += r.n_blocks
        continuity.mark(self.blocks, runs)

    def record_blocks(self, i):
        """Blocos do i-esimo registro (fatia de self.blocks), com a marcacao
        de continuidade propria."""
        a = sum(r.n_blocks for r in self.records[:i])
        rec = self.records[i]
        view = self.blocks[a:a + rec.n_blocks]
        if rec.continuous:
            continuity.mark(view, [(0, rec.n_blocks)])
        return view

    def contiguous_segments(self):
        """(ini, fim) de cada trecho do dia que e um sinal so: um por registro
        continuo, um por bloco no formato .m antigo."""
        return continuity.segments(len(self.blocks),
                                   continuity.runs_of(self.blocks))

    @property
    def continuous(self):
        return bool(self.records) and all(r.continuous for r in self.records)

    @property
    def block_spacing(self):
        """Intervalo tipico entre o inicio de blocos consecutivos, em s:
        5 s no .m antigo, a propria duracao do bloco num registro continuo.
        Serve para reconhecer uma lacuna real ao desenhar."""
        d = np.diff(self.seconds)
        d = d[d > 0]
        return float(np.median(d)) if d.size else BLOCK_SAMPLES * self.dt

    @property
    def dt(self):
        return 1.0 / self.rate

    @property
    def recorded_hours(self):
        return len(self.blocks) * BLOCK_SAMPLES * self.dt / 3600.0

    @property
    def span_hours(self):
        """Periodo de relogio coberto pelos arquivos, somando arquivo a arquivo.

        Somar por arquivo evita contar como medidas as lacunas entre
        arquivos, se houver.
        """
        return sum(r.span for r in self.records) / 3600.0

    @property
    def duty_cycle(self):
        return self.recorded_hours / self.span_hours

    @property
    def occupied_fraction(self):
        return float(np.mean(self.occupied)) if len(self.occupied) else 0.0


def _make_day(recs, rate):
    """Monta um FieldDay a partir de registros ja orientados.

    Registros continuos sem relogio (CSV Time_ms, .txt) comecam em 0 s; aqui
    cada um passa a comecar logo depois do anterior, com um bloco de folga
    para a separacao aparecer nos graficos. A continuidade nao atravessa a
    fronteira entre registros.

    Quando todos os registros tem absolute_time=True, ordena pelo instante
    inicial real antes de concatenar, para que arquivos carregados fora de
    ordem cronologica (nomes nao refletem o horario) fiquem corretos.
    """
    if not recs:
        raise ValueError('Nenhum registro valido.')
    if all(r.absolute_time for r in recs):
        recs = sorted(recs, key=lambda r: float(r.seconds[0]))
    blocks, secs, occ, end = [], [], [], None
    for r in recs:
        blocks.append(r.blocks_si())
        s = np.asarray(r.seconds, dtype=float)
        block = BLOCK_SAMPLES / r.rate
        if r.continuous and not r.absolute_time and end is not None:
            s = s - s[0] + end + block
        secs.append(s)
        end = s[-1] + block
        occ.append(occupancy_mask(r))
    return FieldDay(records=recs, blocks=np.concatenate(blocks),
                    seconds=np.concatenate(secs), occupied=np.concatenate(occ),
                    rate=rate)


def load_day(paths, calibrate=False, rate=NOMINAL_RATE, progress=None,
             position='seat'):
    """Le um conjunto de arquivos .m da mesma unidade como um dia."""
    recs, blocks, secs, occ = [], [], [], []
    for i, p in enumerate(sorted(paths)):
        if progress:
            progress(i / max(len(paths), 1), f'Reading {Path(p).name}')
        rec = read_m(p, rate=rate)
        if calibrate:
            calibrate_from_gravity(rec)
        set_orientation(rec, rec.gravity_axis, position=position)
        recs.append(rec)
    return _make_day(recs, rate)


def _record_from_accel_g(accel_g, rate, path, extra_notes=(),
                         start_seconds=None):
    """Constroi um FieldRecord a partir de um fluxo continuo de aceleracao
    em g, fatiando em blocos fixos de BLOCK_SAMPLES amostras consecutivas,
    sem lacuna (duty_cycle ~1.0). Nucleo comum reaproveitado por
    read_csv_accel (CSV Time_ms), read_ascii_accel (.txt do Anexo A/E) e
    load_day_csv_datetime (CSV DateTime do firmware VCI_nRF52840) -- so a
    forma de obter accel_g e rate muda entre os tres.
    """
    path = Path(path)
    n_total = len(accel_g)
    n_blocks = n_total // BLOCK_SAMPLES
    if n_blocks == 0:
        faltam_s = (BLOCK_SAMPLES - n_total) / rate
        total_necessario_s = BLOCK_SAMPLES / rate
        raise ValueError(
            f'{path.name}: so {n_total} amostra(s), menos que as '
            f'{BLOCK_SAMPLES} de um bloco. Na taxa informada ({rate:g} Hz), '
            f'precisa de pelo menos {total_necessario_s:.1f} s de gravacao '
            f'(faltam ~{faltam_s:.1f} s neste arquivo) -- grave mais tempo')

    counts = (accel_g[:n_blocks * BLOCK_SAMPLES] * LSB_PER_G).reshape(
        n_blocks, BLOCK_SAMPLES, 3)
    block_period = BLOCK_SAMPLES / rate
    seconds = np.arange(n_blocks, dtype=float) * block_period
    if start_seconds is not None:
        seconds = seconds + start_seconds

    rec = FieldRecord(path=path, counts=counts, seconds=seconds, rate=rate,
                      continuous=True,
                      absolute_time=start_seconds is not None)
    dropped = n_total - n_blocks * BLOCK_SAMPLES
    if dropped:
        rec.notes.append(
            f'{dropped} amostra(s) no final descartada(s) (sobra menor que '
            f'um bloco de {BLOCK_SAMPLES})')
    rec.notes.extend(extra_notes)
    rec.gravity_axis = int(np.argmax(np.abs(rec.counts[0].mean(0))))
    return rec


def read_csv_accel(path, rate, columns=('AX', 'AY', 'AZ'), time_column='Time_ms'):
    """Le um CSV de aceleracao continua (ja em g), tipo Time_ms,AX,AY,AZ --
    formato de sketches de bancada, diferente do .m do registrador de campo.

    Diferenca estrutural chave: aqui o fluxo e continuo, sem a lacuna de 3 s
    a cada 5 s do registrador. Por isso NAO reaproveita os blocos do arquivo
    (nao existem) -- fatia a serie em blocos de BLOCK_SAMPLES amostras
    consecutivas, sem pular nada entre eles. duty_cycle sai ~1.0, o que e
    correto para este tipo de dado.

    rate precisa ser informado pelo usuario. O timestamp em ms do arquivo e
    tipicamente so aproximado (jitter do loop de leitura), entao nao da pra
    inferir a taxa de amostragem real so a partir dele com confianca.
    """
    import csv as csv_module

    path = Path(path)
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv_module.reader(f)
        header = [h.strip() for h in next(reader)]
        try:
            idxs = [header.index(c) for c in columns]
        except ValueError as e:
            raise ValueError(
                f'{path.name}: coluna esperada nao encontrada no cabecalho '
                f'{header} ({e})') from e
        rows = [[float(row[i]) for i in idxs] for row in reader if row]

    accel_g = np.array(rows, dtype=np.float64)   # (n, 3), ja em g
    return _record_from_accel_g(accel_g, rate, path)


def read_ascii_accel(path):
    """Le o .txt ASCII de 4 colunas (tempo, ax, ay, az em m/s2) do Anexo
    A/E -- o mesmo arquivo que core.read_excitation le -- e fatia em blocos
    fixos de BLOCK_SAMPLES para entrar no pipeline de dia de campo
    (NHO-09/Diretiva). Ao contrario do CSV Time_ms, a taxa NAO precisa ser
    informada: vem da propria base de tempo uniforme do arquivo (ja
    validada por core.read_excitation).
    """
    from . import core
    exc = core.read_excitation(path)
    rate = exc.sample_rate
    accel_g = exc.accel / GRAVITY_BH
    return _record_from_accel_g(accel_g, rate, path)


def load_day_txt(paths, position='seat', progress=None):
    """Le um conjunto de .txt ASCII (Anexo A/E) como um dia continuo --
    mesmo formato de retorno de load_day()/load_day_csv().

    Cada arquivo tem sua propria taxa de amostragem inferida (ver
    read_ascii_accel); arquivos com taxas incompativeis entre si (>1% de
    diferenca) sao rejeitados, porque FieldDay assume uma unica rate para
    o dia inteiro (dt de blocos concatenados).
    """
    recs, blocks, secs, occ = [], [], [], []
    for i, p in enumerate(sorted(paths)):
        if progress:
            progress(i / max(len(paths), 1), f'Reading {Path(p).name}')
        rec = read_ascii_accel(p)
        if recs and abs(rec.rate - recs[0].rate) > 0.01 * recs[0].rate:
            raise ValueError(
                f'{Path(p).name}: taxa de amostragem ({rec.rate:.2f} Hz) nao '
                f'bate com a do primeiro arquivo ({recs[0].rate:.2f} Hz) -- '
                f'selecione arquivos de uma unica sessao de gravacao')
        set_orientation(rec, rec.gravity_axis, position=position)
        recs.append(rec)
    if not recs:
        raise ValueError('Nenhum arquivo .txt valido selecionado.')
    return _make_day(recs, recs[0].rate)


_DATETIME_META_RE = re.compile(r'rate_hz=([\d.]+)')


def read_datetime_csv_segments(path):
    """Le um CSV 'DateTime,AX,AY,AZ' (g) do firmware VCI_nRF52840, separado
    em segmentos a cada linha de cabecalho repetida.

    Firmwares mais antigos geravam o nome do arquivo a partir da hora de
    COMPILACAO, nao da hora real de cada boot -- resetar a placa reabria o
    mesmo nome (modo append), concatenando sessoes diferentes no mesmo
    arquivo, cada uma com sua propria contagem de tempo. Tratar isso como
    uma unica serie continua inventaria uma base de tempo que nunca
    existiu, por isso ainda se separa por seguranca, mesmo que o firmware
    atual (VCI_nRF52840_accel_SD_v2.ino) ja nao tenha mais esse bug (nome
    de arquivo por indice de sessao persistente).

    Se o arquivo tiver uma linha '# rate_hz=<valor> ...' antes do
    cabecalho (escrita pelo firmware v2), a taxa vem dali -- ver o segundo
    elemento de cada item retornado.

    Devolve uma lista de (linhas, metadata_rate_ou_None), onde linhas e uma
    lista de (datetime_str, [ax, ay, az]).
    """
    import csv as csv_module

    path = Path(path)
    segments = []
    metadata_rates = []
    pending_rate = None
    idx_t = idxs = None

    with open(path, newline='', errors='replace') as f:
        reader = csv_module.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()

            if first.startswith('#'):
                m = _DATETIME_META_RE.search(first)
                if m:
                    pending_rate = float(m.group(1))
                continue

            if first == 'DateTime':
                header = [c.strip() for c in row]
                idx_t = header.index('DateTime')
                idxs = [header.index(c) for c in ('AX', 'AY', 'AZ')]
                segments.append([])
                metadata_rates.append(pending_rate)
                pending_rate = None
                continue

            segments[-1].append((row[idx_t], [float(row[i]) for i in idxs]))

    return [(seg, rate) for seg, rate in zip(segments, metadata_rates) if seg]


def _estimate_datetime_rate(rows):
    """Taxa media pela diferenca de timestamp DateTime -- resolucao de 1 ms
    so, grosseira demais perto de altas taxas; ultimo recurso, quando nao
    ha --rate explicito nem metadado '# rate_hz=' no arquivo.
    """
    from datetime import datetime
    t0 = datetime.strptime(rows[0][0].strip(), '%Y-%m-%d %H:%M:%S.%f')
    t1 = datetime.strptime(rows[-1][0].strip(), '%Y-%m-%d %H:%M:%S.%f')
    span_s = (t1 - t0).total_seconds()
    if span_s <= 0:
        raise ValueError('base de tempo DateTime nao permite estimar a taxa')
    return (len(rows) - 1) / span_s


def load_day_csv_datetime(paths, rate=None, position='seat', progress=None):
    """Le um conjunto de CSVs 'DateTime,AX,AY,AZ' (g) do firmware
    VCI_nRF52840 como um dia -- mesmo dado que csv_datetime_to_annex_txt.py
    converte para .txt, mas direto, sem passar por um arquivo intermediario.

    Cada segmento de boot (ver read_datetime_csv_segments) vira um
    FieldRecord proprio. Prioridade da taxa: rate explicito (parametro) >
    metadado '# rate_hz=' do arquivo > estimativa pelo DateTime (menos
    precisa, registrada em rec.notes).
    """
    paths = sorted(paths)
    jobs = []
    for p in paths:
        for i, (rows, meta_rate) in enumerate(read_datetime_csv_segments(p)):
            jobs.append((p, i, rows, meta_rate))
    if not jobs:
        raise ValueError('Nenhum segmento com dado suficiente encontrado.')

    recs, blocks, secs, occ = [], [], [], []
    for j, (p, i, rows, meta_rate) in enumerate(jobs):
        if progress:
            progress(j / len(jobs), f'Reading {Path(p).name} '
                     f'(segment {i + 1})')
        seg_rate = rate if rate is not None else meta_rate
        notes = []
        if seg_rate is None:
            seg_rate = _estimate_datetime_rate(rows)
            notes.append(
                f'taxa nao informada nem no cabecalho do arquivo; estimada '
                f'pelo DateTime (resolucao de 1 ms): {seg_rate:.2f} Hz')
        accel_g = np.array([axes for _, axes in rows], dtype=np.float64)
        name = f'{Path(p).name}#{i + 1}' if len(jobs) > 1 else Path(p).name
        rec = _record_from_accel_g(accel_g, seg_rate, name, extra_notes=notes,
                                   start_seconds=_seconds_of_day(rows[0][0]))
        set_orientation(rec, rec.gravity_axis, position=position)
        recs.append(rec)
    if len({round(r.rate, 2) for r in recs}) > 1:
        raise ValueError(
            'Os segmentos tem taxas de amostragem diferentes ('
            + ', '.join(f'{r.rate:.2f}' for r in recs) + ' Hz); '
            'selecione arquivos de uma unica configuracao do firmware.')
    return _make_day(recs, recs[0].rate)


def _seconds_of_day(stamp):
    """Hora do dia, em segundos, de um carimbo 'AAAA-MM-DD hh:mm:ss.fff'.
    None se o carimbo nao puder ser lido."""
    from datetime import datetime
    try:
        t = datetime.strptime(stamp.strip(), '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            t = datetime.strptime(stamp.strip(), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def load_day_csv(paths, rate, position='seat', progress=None):
    """Le um conjunto de CSVs continuos (Time_ms,AX,AY,AZ) como um dia.

    Mesmo formato de retorno de load_day(), pra reusar sem alteracao todo o
    resto do pipeline (criterios, graficos, A(8) por eixo etc.) -- so a
    origem do dado muda.
    """
    recs, blocks, secs, occ = [], [], [], []
    for i, p in enumerate(sorted(paths)):
        if progress:
            progress(i / max(len(paths), 1), f'Reading {Path(p).name}')
        rec = read_csv_accel(p, rate=rate)
        set_orientation(rec, rec.gravity_axis, position=position)
        recs.append(rec)
    return _make_day(recs, rate)


def reorient_day(day, position):
    """Refaz a orientacao dos registros de um dia sem reler os arquivos."""
    blocks = []
    for rec in day.records:
        set_orientation(rec, rec.gravity_axis, position=position)
        blocks.append(rec.blocks_si())
    day.blocks = np.concatenate(blocks)
    day.mark_runs()
    return day


# ----------------------------------------------------------------------
# Binario VIB1 do XIAO nRF52840 e leitura automatica por formato
# ----------------------------------------------------------------------

def read_dat(path):
    """Le um .dat binario VIB1 (firmware do XIAO nRF52840 Sense, LSM6DS3)
    como registro continuo.

    A taxa e a media dos carimbos em microssegundos do proprio arquivo, que
    vem do relogio do microcontrolador; a nominal do cabecalho so entra numa
    nota quando as duas diferem mais de 1%. Atrasos de escrita no cartao
    aparecem como saltos de carimbo: sao contados numa nota, mas o sinal
    segue tratado como continuo (o firmware nao descarta amostras).
    """
    from . import xiao_parser
    xf = xiao_parser.parse_file(path)
    if len(xf.t) < 2:
        raise ValueError(f'{Path(path).name}: arquivo sem amostras')
    span = float(xf.t[-1] - xf.t[0])
    nominal = float(xf.sample_rate_hz)
    rate = (len(xf.t) - 1) / span if span > 0 else nominal
    notes = []
    if abs(rate - nominal) > 0.01 * nominal:
        notes.append(f'taxa pelos carimbos {rate:.2f} Hz, nominal do '
                     f'cabecalho {nominal:.0f} Hz; usada a dos carimbos')
    gaps = xf.gaps()
    if gaps:
        notes.append(f'{len(gaps)} salto(s) de carimbo acima de 2 periodos '
                     f'(maior: {max(d for _, d in gaps) * 1000:.1f} ms)')
    sens_g = xiao_parser._SENSITIVITY_MG_PER_LSB[xf.full_scale_g] / 1000.0
    accel_g = xf.counts.astype(np.float64) * sens_g
    start = (xf.start.hour * 3600 + xf.start.minute * 60 + xf.start.second
             + float(xf.t[0]))
    return _record_from_accel_g(accel_g, rate, Path(path), extra_notes=notes,
                                start_seconds=start)


FORMAT_NAMES = {
    'm': 'field logger .m (old 2 s / 5 s format)',
    'dat': 'XIAO binary .dat (VIB1)',
    'datetime': 'DateTime CSV (VCI_nRF52840)',
    'csv_ms': 'Time_ms CSV (bench firmware)',
    'txt': 'ASCII .txt, 4 columns (Annex A/E)',
}


def detect_format(path):
    """Formato de um arquivo, pela extensao e, num CSV, pelo cabecalho."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.m':
        return 'm'
    if ext == '.dat':
        return 'dat'
    if ext in ('.txt', '.asc'):
        return 'txt'
    if ext == '.csv':
        with open(path, errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('DateTime'):
                    return 'datetime'
                if 'Time_ms' in line:
                    return 'csv_ms'
                break
        raise ValueError(f'{path.name}: CSV sem cabecalho DateTime,AX,AY,AZ '
                         f'nem Time_ms,AX,AY,AZ')
    raise ValueError(f'{path.name}: formato nao reconhecido ({ext})')


def read_records(path, csv_rate=None):
    """Registros de um arquivo de qualquer formato aceito (um DateTime CSV
    pode ter varios segmentos de boot). Sem orientacao aplicada."""
    kind = detect_format(path)
    if kind == 'm':
        return [read_m(path)]
    if kind == 'dat':
        return [read_dat(path)]
    if kind == 'txt':
        return [read_ascii_accel(path)]
    if kind == 'csv_ms':
        if csv_rate is None:
            raise ValueError(f'{Path(path).name}: CSV Time_ms precisa da taxa '
                             f'de amostragem informada')
        return [read_csv_accel(path, rate=csv_rate)]
    recs = []
    segs = read_datetime_csv_segments(path)
    for i, (rows, meta_rate) in enumerate(segs):
        notes = []
        rate = meta_rate
        if rate is None:
            rate = _estimate_datetime_rate(rows)
            notes.append(f'taxa estimada pelo DateTime: {rate:.2f} Hz')
        accel_g = np.array([axes for _, axes in rows], dtype=np.float64)
        name = (f'{Path(path).name}#{i + 1}' if len(segs) > 1
                else Path(path).name)
        recs.append(_record_from_accel_g(
            accel_g, rate, name, extra_notes=notes,
            start_seconds=_seconds_of_day(rows[0][0])))
    return recs


def load_day_auto(paths, calibrate=False, position='seat', progress=None,
                  csv_rate=None, datetime_rate=None):
    """Le um dia em qualquer formato aceito, escolhendo o leitor pelo
    arquivo. Todos os arquivos tem de ser do mesmo formato."""
    paths = sorted(paths)
    kinds = {detect_format(p) for p in paths}
    if len(kinds) > 1:
        raise ValueError('Os arquivos selecionados misturam formatos ('
                         + ', '.join(FORMAT_NAMES[k] for k in sorted(kinds))
                         + '); carregue um formato por vez.')
    kind = kinds.pop()
    if kind == 'm':
        return load_day(paths, calibrate=calibrate, progress=progress,
                        position=position)
    if kind == 'txt':
        day = load_day_txt(paths, position=position, progress=progress)
    elif kind == 'csv_ms':
        day = load_day_csv(paths, rate=csv_rate, position=position,
                           progress=progress)
    elif kind == 'datetime':
        day = load_day_csv_datetime(paths, rate=datetime_rate,
                                    position=position, progress=progress)
    else:
        day = load_day_dat(paths, position=position, progress=progress)
    if calibrate:
        for rec in day.records:
            calibrate_from_gravity(rec)
        reorient_day(day, position)
    return day


def load_day_dat(paths, position='seat', progress=None):
    """Le um conjunto de .dat VIB1 do XIAO como um dia."""
    recs = []
    for i, p in enumerate(sorted(paths)):
        if progress:
            progress(i / max(len(paths), 1), f'Reading {Path(p).name}')
        rec = read_dat(p)
        if recs and abs(rec.rate - recs[0].rate) > 0.01 * recs[0].rate:
            raise ValueError(
                f'{Path(p).name}: taxa de amostragem ({rec.rate:.2f} Hz) nao '
                f'bate com a do primeiro arquivo ({recs[0].rate:.2f} Hz)')
        set_orientation(rec, rec.gravity_axis, position=position)
        recs.append(rec)
    return _make_day(recs, recs[0].rate)
