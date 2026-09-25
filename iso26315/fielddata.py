"""Leitura e analise dos registros do registrador de campo do GRAVI-SH.

O registrador grava arquivos .m que sao scripts MATLAB: blocos de
DadosValores(:,:,len)=int16([...]) precedidos de um carimbo de tempo do RTC.

Caracteristicas do formato, confirmadas no firmware (vibracoes.ino):

  - MPU6050 em +-2 g, portanto 16384 LSB/g; os valores sao contagens brutas,
    sem nenhuma conversao ou calibracao aplicada;
  - 2048 amostras por bloco a 1024 Hz nominais, ou seja 2 s de sinal;
  - um bloco a cada 5 s de relogio (o firmware espera Second() % 5 == 0), de
    modo que 3 s em cada 5 nao sao gravados. O ciclo de trabalho e de 40% e e
    intencional.

A descontinuidade entre blocos e a razao para NAO concatenar os blocos e
transformar o conjunto: cada colagem e um degrau que a transformada espalha por
todo o espectro. A analise correta trata cada bloco como um segmento de Welch,
o que dispensa qualquer colagem.

Pelo mesmo motivo, a cadeia do Anexo A nao se aplica a estes registros: a dose
de sexta potencia depende de contar todos os picos, e 60% deles nao existem.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LSB_PER_G = 16384.0          # +-2 g, registrador 0x1C = 0b00000000
GRAVITY_BH = 9.784           # m/s2, gravidade local em Belo Horizonte
NOMINAL_RATE = 1024.0
BLOCK_SAMPLES = 2048
BLOCK_PERIOD = 5.0           # s entre carimbos

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
    """Um arquivo do registrador: blocos de 2 s com carimbo de tempo."""
    path: Path
    counts: np.ndarray                  # (n_blocos, 2048, 3), contagens brutas
    seconds: np.ndarray                 # segundos do dia de cada bloco
    rate: float = NOMINAL_RATE
    scale: np.ndarray = None            # fator por eixo, da calibracao
    axis_map: tuple = (0, 1, 2)         # coluna do arquivo -> eixo x, y, z
    axis_sign: tuple = (1, 1, 1)
    gravity_axis: int = None
    notes: list = field(default_factory=list)

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
        """Blocos em m/s2, ja orientados, calibrados e sem a componente estatica."""
        a = self.counts / LSB_PER_G * GRAVITY_BH
        if self.scale is not None:
            a = a * self.scale
        a = a[:, :, list(self.axis_map)] * np.array(self.axis_sign)
        if remove_gravity:
            a = a - a.mean(axis=1, keepdims=True)
        return a


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
        blocks.append(rec.blocks_si())
        secs.append(rec.seconds)
        occ.append(occupancy_mask(rec))
    return FieldDay(records=recs, blocks=np.concatenate(blocks),
                    seconds=np.concatenate(secs), occupied=np.concatenate(occ),
                    rate=rate)


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

    rec = FieldRecord(path=path, counts=counts, seconds=seconds, rate=rate)
    dropped = n_total - n_blocks * BLOCK_SAMPLES
    if dropped:
        rec.notes.append(
            f'{dropped} amostra(s) no final descartada(s) (sobra menor que '
            f'um bloco de {BLOCK_SAMPLES})')
    rec.gravity_axis = int(np.argmax(np.abs(rec.counts[0].mean(0))))
    return rec


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
        blocks.append(rec.blocks_si())
        secs.append(rec.seconds)
        occ.append(occupancy_mask(rec))
    return FieldDay(records=recs, blocks=np.concatenate(blocks),
                    seconds=np.concatenate(secs), occupied=np.concatenate(occ),
                    rate=rate)


def reorient_day(day, position):
    """Refaz a orientacao dos registros de um dia sem reler os arquivos."""
    blocks = []
    for rec in day.records:
        set_orientation(rec, rec.gravity_axis, position=position)
        blocks.append(rec.blocks_si())
    day.blocks = np.concatenate(blocks)
    return day
