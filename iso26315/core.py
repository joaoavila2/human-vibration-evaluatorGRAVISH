"""Nucleo de calculo da ISO 2631-5:2018, Anexos A e E.

Cadeia completa:
  aceleracao medida -> FFT -> deslocamento -> forcas intervertebrais
  -> dominio do tempo -> picos de compressao -> dose S_d^A -> fator de risco R^A

Validado contra os exemplos de referencia do software original (ver tests/).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

# Areas medias de placa terminal, mm^2 (A.3.3)
ENDPLATE_AREA = np.array([1460., 1520., 1580., 1590., 1600., 1550.])
DISC_NAMES = ['T12/L1', 'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
POSTURE_NAMES = {
    1: 'Group 1 - fork lift',
    2: 'Group 2 - wheel loader',
    3: 'Group 3 - excavator',
    4: 'Group 4 - forwarder',
    5: 'Group 5 - harvester',
    7: 'Hands in lap',
}
LOCATION_NAMES = ['seat', 'feet', 'hands', 'back']
AXIS_NAMES = ['x', 'y', 'z']

# Parametros fixos do processamento de sinal, expostos na interface
HIGHPASS_HZ = 0.5      # corte inferior (Clausula 5)
LOWPASS_HZ = 100.0     # corte superior antes da reamostragem
WINDOW = 'rectangular (none)'
WEIGHTING = 'none (Annex A uses unweighted acceleration)'

# Limites de interpretacao de E.3
RISK_LOW, RISK_HIGH = 0.8, 1.2


# ----------------------------------------------------------------------
# FFT no mesmo padrao do software original (espectro de amplitude unilateral)
# ----------------------------------------------------------------------

def fft_amplitude(x, dt):
    n = x.shape[0]
    if n % 2:
        x = x[:-1]
        n -= 1
    half = n // 2
    y = np.fft.fft(x, axis=0) / half
    y[0] *= 0.5
    y[half] *= 0.5
    return y[:half + 1], np.arange(half + 1) / (dt * n)


def ifft_amplitude(X):
    half = X.shape[0] - 1
    full = np.zeros((2 * half,) + X.shape[1:], dtype=complex)
    full[:half + 1] = X
    full[half + 1:] = np.conj(X[1:half][::-1])
    full *= half
    full[0] *= 2
    full[half] *= 2
    return np.real(np.fft.ifft(full, axis=0))


# ----------------------------------------------------------------------
# Leitura das excitacoes
# ----------------------------------------------------------------------

@dataclass
class Excitation:
    """Serie temporal de aceleracao em um ponto de contato."""
    name: str
    time: np.ndarray
    accel: np.ndarray          # (n, 3) em m/s^2
    dt: float
    spectrum: np.ndarray = field(default=None, repr=False)   # deslocamento
    freq: np.ndarray = field(default=None, repr=False)
    accel_spectrum: np.ndarray = field(default=None, repr=False)
    raw_freq: np.ndarray = field(default=None, repr=False)
    raw_spectrum: np.ndarray = field(default=None, repr=False)
    n_raw: int = 0
    metadata: dict = field(default_factory=dict, repr=False)

    @property
    def sample_rate(self):
        return 1.0 / self.dt

    @property
    def resolution(self):
        return self.raw_freq[1] - self.raw_freq[0] if self.raw_freq is not None else None

    @property
    def duration(self):
        return self.time[-1] - self.time[0]

    @property
    def rms(self):
        return np.sqrt((self.accel ** 2).sum(0) / self.accel.shape[0])


_META_KEYS = ('subject_age', 'subject_mass_kg', 'subject_height_cm',
              'position', 'sensor_model', 'sensor_serial',
              'calibration', 'calibration_date', 'operator', 'notes')


def read_excitation(path, name=None):
    """Le arquivo ASCII de 4 colunas: tempo, ax, ay, az (A.2).

    Linhas de cabecalho com '# chave: valor' sao lidas como metadados do
    registro (sujeito, sensor, calibracao). As chaves reconhecidas sao as
    de _META_KEYS; outras sao ignoradas silenciosamente.
    """
    path = Path(path)
    meta = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                if s.startswith('#') and ':' in s:
                    rest = s[1:].strip()
                    k, _, v = rest.partition(':')
                    k = k.strip().lower().replace(' ', '_')
                    if k in _META_KEYS:
                        meta[k] = v.strip()
            else:
                break
    data = np.loadtxt(path, usecols=(0, 1, 2, 3))
    if data.shape[1] > data.shape[0]:
        data = data.T
    t = data[:, 0]
    dt = (t[-1] - t[0]) / (len(t) - 1)
    if abs((t[1] - t[0]) - dt) > 0.02 * dt:
        raise ValueError(
            f'{path.name}: a base de tempo nao parece uniformemente espacada.')
    exc = Excitation(name or path.stem, t, data[:, 1:4], dt, metadata=meta)
    _prepare(exc)
    return exc


def write_excitation_header(path, metadata):
    """Reescreve as linhas de cabecalho '# chave: valor' de um arquivo ASCII.

    Le o arquivo, substitui/acrescenta o bloco de metadados no topo (antes
    das linhas de dados) e salva. Chaves nao reconhecidas sao preservadas.
    """
    path = Path(path)
    with open(path, encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()
    # Separa cabecalho antigo de dados
    i = 0
    old_meta = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('#') and ':' in s:
            rest = s[1:].strip()
            k, _, v = rest.partition(':')
            old_meta[k.strip()] = v.strip()
        elif not s:
            continue
        else:
            break
    data_lines = lines[i:]
    # Mescla: novos valores sobrepoe antigos
    merged = {k: old_meta[k] for k in old_meta if k.lower().replace(' ', '_')
              not in {mk.lower().replace(' ', '_') for mk in metadata}}
    new_block = [f'# {k}: {v}\n' for k, v in metadata.items() if v.strip()]
    old_other = [f'# {k}: {v}\n' for k, v in merged.items()]
    with open(path, 'w', encoding='utf-8') as fh:
        fh.writelines(new_block + old_other + data_lines)


def zero_excitation(like, name='Zero'):
    """Excitacao identicamente nula, para pontos de contato sem medicao."""
    exc = Excitation(name, like.time, np.zeros_like(like.accel), like.dt)
    exc.freq = like.freq
    exc.accel_spectrum = np.zeros_like(like.accel_spectrum)
    exc.spectrum = np.zeros_like(like.spectrum)
    exc.raw_freq = like.raw_freq
    exc.raw_spectrum = np.zeros_like(like.raw_spectrum)
    return exc


def _band_taper(f):
    """Transicao de cosseno elevado em vez de corte vertical nas duas pontas."""
    g = np.ones(len(f))
    lo0 = HIGHPASS_HZ * (1 - BAND_TRANSITION)
    lo1 = HIGHPASS_HZ * (1 + BAND_TRANSITION)
    m = f <= lo0
    g[m] = 0.0
    m = (f > lo0) & (f < lo1)
    g[m] = 0.5 * (1 - np.cos(np.pi * (f[m] - lo0) / (lo1 - lo0)))
    hi0 = LOWPASS_HZ * (1 - BAND_TRANSITION)
    m = f >= hi0
    g[m] = 0.5 * (1 + np.cos(np.pi * (f[m] - hi0) / (f[-1] - hi0 + 1e-12)))
    return g


def _prepare(exc):
    """FFT, limitacao de banda (0,5 a 100 Hz) e conversao para deslocamento.

    A conversao para deslocamento e essencial: as funcoes de transferencia do
    Anexo A sao deslocamento -> forca, nao aceleracao -> forca.
    """
    signal = exc.accel
    exc.n_raw = signal.shape[0]
    if ZERO_PADDING:
        signal = np.vstack([signal, np.zeros_like(signal)])
    A, f = fft_amplitude(signal, exc.dt)
    exc.raw_freq, exc.raw_spectrum = f.copy(), A.copy()
    keep = np.searchsorted(f, LOWPASS_HZ, side='right')
    A, f = A[:keep].copy(), f[:keep].copy()
    if BAND_TRANSITION > 0:
        A *= _band_taper(f)[:, None]
    else:
        A[:np.searchsorted(f, HIGHPASS_HZ, side='left') + 1] = 0
    w = 2 * np.pi * f
    with np.errstate(divide='ignore', invalid='ignore'):
        U = A / -(w ** 2)[:, None]
    U[0] = 0
    exc.freq, exc.accel_spectrum, exc.spectrum = f, A, U


def intensity_class(excitations):
    """Classe de magnitude de aceleracao (intcalc do software original)."""
    r = max(e.rms.max() for e in excitations)
    return 1 if r < 0.65 else (3 if r > 1.35 else 2)


# ----------------------------------------------------------------------
# Antropometria
# ----------------------------------------------------------------------

BMI_THRESHOLD = 26.1        # confirmado pelo texto da norma, Anexo A.3.1 a)
# Bordas das cinco classes de massa corporal (kg) por classe de IMC, conforme
# DIN SPEC 45697 e ISO/CD 2631-5:2014; confirmadas por Schust, Hinz & Seidel
# (2015) Ergonomics 58(4):576-597 (Tab. 1) e pelo texto da norma:
# "a body mass larger than 109 kg" (A.3.1 a), E.2, E.4 item 2).
# Cada lista tem 4 valores -- as fronteiras entre as 5 classes. A classe e
# determinada por np.searchsorted(..., side='right'), o que faz o valor da
# propria fronteira cair na classe SUPERIOR (ex.: 65 kg → classe 2, 109 kg →
# classe 5).
MASS_EDGES_BY_BMI_CLASS = {
    1: [65.0, 72.0, 80.0, 88.0],   # IMC <= 26,1: <65 | 65-72 | 72-80 | 80-88 | >=88 kg
    2: [80.0, 87.0, 96.0, 109.0],  # IMC >  26,1: <80 | 80-87 | 87-96 | 96-109 | >=109 kg
}
PERCENTILE_NAMES = ['P05', 'P25', 'P50', 'P75', 'P95']

ANTHROPOMETRY_NOTE = """\
O indice de 30 posicoes decompoe-se em (bloco-1)*15 + (percentil-1)*3 + \
subclasse. O bloco e a classe de IMC (limiar 26,1 kg/m2, confirmado pelo \
texto da norma, Anexo A.3.1 a); o percentil de massa e calculado dentro da \
classe de IMC ja escolhida, pelas bordas definidas na DIN SPEC 45697 e \
ISO/CD 2631-5:2014, confirmadas por Schust, Hinz & Seidel (2015) \
Ergonomics 58(4):576-597, Tab. 1. As bordas derivam dos percentis P05/P25/ \
P50/P75 da Tabela 2 de Seidel, Hinz, Hofmann, Menzel (2008), Int J Ind \
Ergon 38:856-867 -- os 1163 motoristas europeus que deram origem as funcoes \
de transferencia do Anexo A. A classe mais alta (P95+) inicia em 88 kg \
(IMC<=26,1) ou 109 kg (IMC>26,1); a ISO 2631-5:2018 fixa explicitamente \
109 kg como limiar superior em A.3.1 a), E.2 e E.4 item 2). Confirmado \
contra os seis casos de referencia (indiv1-indiv6) da propria norma via \
tests/anthropometry_check.py."""


def anthropometry(body_mass, body_height, subclass=1):
    """(indice 1..30 da funcao de transferencia, indice 1..10 da forca estatica).

    Ver ANTHROPOMETRY_NOTE quanto ao grau de confianca deste mapeamento.
    """
    bmi = body_mass / body_height ** 2
    bmi_class = 2 if bmi > BMI_THRESHOLD else 1
    edges = MASS_EDGES_BY_BMI_CLASS[bmi_class]
    percentile = int(np.searchsorted(edges, body_mass, side='right')) + 1
    k = (bmi_class - 1) * 15 + (percentile - 1) * 3 + subclass
    bmi_per = (bmi_class - 1) * 5 + percentile
    return k, bmi_per


# ----------------------------------------------------------------------
# Banco de funcoes de transferencia
# ----------------------------------------------------------------------

class TransferFunctionStore:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self._cache = {}
        self.static_forces = np.load(self.dir / 'static_forces.npz')['static']

    def available_postures(self):
        return sorted(int(p.stem.replace('posture', ''))
                      for p in self.dir.glob('posture*.npz'))

    def posture(self, posture):
        if posture not in self._cache:
            path = self.dir / f'posture{posture}.npz'
            if not path.exists():
                raise FileNotFoundError(
                    f'Dados da postura {posture} nao encontrados em {self.dir}.')
            d = np.load(path)
            self._cache[posture] = (d['H'], d['freq'])
        return self._cache[posture]

    def static(self, posture, bmi_per):
        return self.static_forces[posture - 1, bmi_per - 1]      # (6, 3)


# A funcao de transferencia do modelo tem um salto de 2*pi na fase perto de
# 20 Hz. O software original interpola a fase enrolada, e a spline cubica
# atravessa o salto pelo caminho errado: nos cerca de 6% de pontos em volta
# dele o valor complexo interpolado chega a errar 200%. Desenrolar a fase
# antes de interpolar corrige isso, mas afasta o resultado da referencia em
# ate 1%. O padrao e a fidelidade; UNWRAP_PHASE liga a correcao.
UNWRAP_PHASE = False

# Duas opcoes numericas, desligadas por padrao pelo mesmo motivo da fase:
# ligadas, afastam o resultado da referencia. Ver o estudo na documentacao.
#
# ZERO_PADDING acolchoa o registro com zeros antes da transformada. Multiplicar
# no dominio da frequencia e convolucao circular no tempo: sem acolchoamento, a
# resposta ao impulso do fim do registro enrola para o comeco.
#
# BAND_TRANSITION substitui os cortes verticais de 0,5 e 100 Hz por transicoes
# de cosseno elevado. O corte vertical tem resposta em seno cardinal, que faz um
# choque isolado virar um trem de oscilacoes, e a contagem de picos conta cada
# lobulo.
ZERO_PADDING = False
BAND_TRANSITION = 0.0        # largura da transicao, em fracao da frequencia de corte


def _interp_tf(f_model, f_target, H, unwrap=None):
    """Interpola a FT do modelo na grade da excitacao (spline em modulo e fase).

    Fora da faixa do modelo (0,25 a 30 Hz) o resultado e zero: extrapolar
    nao faria sentido fisico.
    """
    out = np.zeros((H.shape[0], len(f_target)), dtype=complex)
    lo = np.searchsorted(f_target, 0.25, side='right') - 1
    hi = np.searchsorted(f_target, f_model[-1], side='left')
    lo, hi = max(lo, 0), min(hi, len(f_target) - 1)
    if hi <= lo:
        return out
    ft = f_target[lo:hi + 1]
    angle = np.angle(H)
    if UNWRAP_PHASE if unwrap is None else unwrap:
        angle = np.unwrap(angle, axis=-1)
    mag = CubicSpline(f_model, np.abs(H), axis=-1)(ft)
    pha = CubicSpline(f_model, angle, axis=-1)(ft)
    out[:, lo:hi + 1] = mag * np.exp(1j * pha)
    return out


# ----------------------------------------------------------------------
# Forcas intervertebrais e dose de compressao
# ----------------------------------------------------------------------

@dataclass
class ExposureResult:
    """Resultado de uma unica exposicao medida."""
    S: np.ndarray                  # dose de compressao por nivel, MPa
    S_stat: np.ndarray             # tensao estatica por nivel, MPa
    forces: np.ndarray             # (6, n) forca de compressao no tempo, N
    time: np.ndarray
    peaks: list                    # picos de compressao por nivel, N
    freq: np.ndarray
    force_spectrum: np.ndarray     # (6, nf) espectro da forca de compressao
    excitations: dict
    posture: int
    k: int
    bmi_per: int
    measurement_time: float
    # Caminhos dos 4 arquivos (seat, feet, hands, back), na mesma ordem de
    # LOCATION_NAMES. None para exposicoes montadas sem arquivo de origem
    # (ex. em testes). Excitation nao guarda o proprio caminho, e o
    # exc.spectrum e liberado da memoria apos o calculo em historicos longos
    # (analysis.Runner) -- a analise de sensibilidade releitura os arquivos
    # a partir daqui em vez de reusar excitations ja processadas.
    paths: tuple = None
    # Aviso gerado quando o encosto foi marcado 'Zero' (ver CHANGELOG 2026-11-07).
    backrest_warning: str | None = None


def compute_forces(store, excitations, posture, k, bmi_per):
    """Formula (A.1): soma sobre 4 pontos de contato e 3 direcoes."""
    H_all, f_model = store.posture(posture)
    f = excitations[0].freq
    F = np.zeros((3, 6, len(f)), dtype=complex)
    for el, exc in enumerate(excitations):
        if not np.any(exc.spectrum):
            continue
        for ed in range(3):
            H = H_all[k - 1, el, ed].astype(complex).reshape(18, -1)
            Hi = _interp_tf(f_model, f, H).reshape(3, 6, -1)
            F += Hi * exc.spectrum[:, ed]
    return F, f


def compression_peaks(x):
    """Picos de compressao entre cruzamentos consecutivos do valor medio (A.3.3).

    Retorna os extremos negativos relativos a media (valores de C_dyn,i).
    """
    mean = x.mean()
    y = x - mean
    s = np.sign(y)
    s[s == 0] = 1.0
    cuts = np.flatnonzero(np.diff(s)) + 1
    bounds = np.concatenate(([0], cuts, [len(y)]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = y[a:b]
        if seg.size and seg[0] < 0:
            out.append(seg.min())
    return np.asarray(out), mean


def analyse_exposure(store, excitations, posture, body_mass, body_height,
                     measurement_time=None, subclass=1):
    """Calcula S^A e as forcas para uma exposicao medida (Anexo A)."""
    k, bmi_per = anthropometry(body_mass, body_height, subclass)
    return analyse_exposure_at(store, excitations, posture, k, bmi_per,
                               measurement_time)


def analyse_exposure_at(store, excitations, posture, k, bmi_per,
                        measurement_time=None):
    """Como analyse_exposure, mas com k e bmi_per ja escolhidos.

    Usado pela analise de sensibilidade (sensitivity_grid), que varre as
    combinacoes de postura e modelo antropometrico diretamente, sem passar
    por massa/altura.
    """
    F, f = compute_forces(store, excitations, posture, k, bmi_per)
    cstat = store.static(posture, bmi_per)          # (6, 3), N

    S = np.zeros(6)
    S_stat = np.abs(cstat[:, 2]) / ENDPLATE_AREA
    n_full = 2 * (len(f) - 1)
    n_keep = n_full
    if ZERO_PADDING and excitations[0].n_raw:
        # descarta a cauda acolchoada, mantendo so o trecho real do registro
        n_keep = min(n_full, int(round(excitations[0].n_raw
                                       * (2 * LOWPASS_HZ) * excitations[0].dt)))
    forces = np.zeros((6, n_keep))
    peaks_all = []
    for dl in range(6):
        x = ifft_amplitude(F[2, dl])[:n_keep]       # direcao 3 = compressao
        pk, mean = compression_peaks(x)
        forces[dl] = x - mean + cstat[dl, 2]
        peaks_all.append(pk)
        S[dl] = (np.sum((np.abs(pk) / ENDPLATE_AREA[dl]) ** 6)) ** (1 / 6)

    dt = 1.0 / (2 * f[-1])
    tm = measurement_time or excitations[0].duration
    return ExposureResult(
        S=S, S_stat=S_stat, forces=forces,
        time=np.arange(forces.shape[1]) * dt,
        peaks=peaks_all, freq=f, force_spectrum=F[2],
        excitations={LOCATION_NAMES[i]: e for i, e in enumerate(excitations)},
        posture=posture, k=k, bmi_per=bmi_per, measurement_time=tm)


def daily_dose(results, daily_times):
    """Formula (A.3): dose diaria equivalente a partir de j exposicoes."""
    total = np.zeros(6)
    for res, td in zip(results, daily_times):
        total += res.S ** 6 * td / res.measurement_time
    return total ** (1 / 6)


def average_dose(daily_doses, days):
    """Formula (E.3): dose media anual para padroes de exposicao variaveis."""
    days = np.asarray(days, dtype=float)
    stacked = np.stack(daily_doses)                 # (h, 6)
    return ((stacked ** 6 * (days / days.sum())[:, None]).sum(0)) ** (1 / 6)


# ----------------------------------------------------------------------
# Fator de risco (Anexo E)
# ----------------------------------------------------------------------

def ultimate_strength(age, equation=2, protection=50):
    """Resistencia ultima da vertebra, MPa -- Formula (E.2) da norma,
    confirmada contra o texto da ISO 2631-5:2018 (6,765024 - 0,067184*idade).

    O parametro equation existe so por compatibilidade de assinatura com
    quem ja chama esta funcao (risk_factor, risk_factor_with_static);
    so ha uma formula implementada. Uma opcao "alternativa" existiu antes
    com constantes (5,106713/0,037747) que a auditoria de 2026-09 nao
    conseguiu confirmar contra o texto da norma -- removida do programa.
    """
    offset = 0.0 if protection == 50 else _protection_offset(protection)
    return 6.765024 - 0.067184 * np.asarray(age) - offset


# Proteger uma fracao maior da populacao significa avaliar uma vertebra MAIS
# FRACA: a resistencia desce. Seidel, Hinz, Hofmann e Menzel (2008, Int J Ind
# Ergon 38:856-867, citando Seidel et al. 1998): a Formula (E.2) vale para
# cerca de 50% da populacao, e para 95% deve ser reduzida em cerca de 1,5 MPa.
# A norma nao da formulacao para outros percentis; o desvio padrao abaixo e o
# que reproduz esses 1,5 MPa no quantil de 95% de uma normal.
STRENGTH_REDUCTION_95 = 1.5                                   # MPa
STRENGTH_SD = STRENGTH_REDUCTION_95 / 1.6448536269514722      # ~0,912 MPa


def _protection_offset(protection):
    """Reducao da resistencia (MPa) para proteger `protection`% da populacao.

    Positiva acima de 50% (resistencia menor, R maior, mais conservador),
    1,5 MPa em 95%. Uma versao anterior tinha o sinal invertido e usava
    desvio de 1 MPa: em 95% a resistencia SUBIA 1,64 MPa e o R caia.
    """
    from statistics import NormalDist
    return STRENGTH_SD * NormalDist().inv_cdf(protection / 100.0)


def risk_factor(doses_per_year, days_per_year, start_age,
                equation=2, protection=50, exponent=6):
    """Formulas (E.1)/(E.4): fator de risco R^A por nivel discal.

    doses_per_year: (n_anos, 6) dose diaria de cada ano, MPa
    days_per_year:  (n_anos,)   dias de exposicao em cada ano
    start_age:      idade no inicio da exposicao (b)
    """
    doses = np.atleast_2d(doses_per_year)
    days = np.asarray(days_per_year, dtype=float)
    n = doses.shape[0]
    ages = start_age + np.arange(1, n + 1)
    Su = ultimate_strength(ages, equation, protection)
    return _accumulate(doses, days, Su, exponent)


def _accumulate(doses, days, Su, exponent):
    R = np.zeros(6)
    for dl in range(6):
        denom = Su - 0.0 if np.isscalar(Su) else Su
        terms = (doses[:, dl] * days ** (1 / exponent)) / denom
        R[dl] = np.sum(terms ** exponent) ** (1 / exponent)
    return R


def risk_factor_with_static(doses_per_year, days_per_year, static_stress,
                            start_age, equation=2, protection=50, exponent=6):
    """Igual a risk_factor, subtraindo a tensao estatica S_stat da resistencia."""
    doses = np.atleast_2d(doses_per_year)
    days = np.asarray(days_per_year, dtype=float)
    static = np.atleast_2d(static_stress)
    if static.shape[0] == 1:
        static = np.repeat(static, doses.shape[0], axis=0)
    ages = start_age + np.arange(1, doses.shape[0] + 1)
    Su = ultimate_strength(ages, equation, protection)

    R = np.zeros(6)
    for dl in range(6):
        terms = (doses[:, dl] * days ** (1 / exponent)) / (Su - static[:, dl])
        R[dl] = np.sum(terms ** exponent) ** (1 / exponent)
    return R


N_ANTHROPOMETRY_MODELS = 30


def sensitivity_grid(store, excitations, measurement_time=None, start_age=20,
                     n_years=45, days_per_year=240.0, daily_time=14400.0,
                     equation=2, protection=50, progress=None):
    """Roda o mesmo sinal contra as 6 posturas e os 30 modelos antropometricos.

    Aplica os parametros fixos da analise orientadora (A.3.1: 45 anos, 240
    dias/ano, 4 h/dia a partir dos 20 anos) a cada uma das 180 combinacoes,
    para quantificar a dispersao de R^A que nenhuma postura ou modelo
    tabelado resolve sozinho -- ponto especialmente relevante para usuarios
    de cadeira de rodas, para quem nenhuma das 6 posturas da norma e
    adequada (Secao 7.2 da documentacao). Trabalho futuro, item 3.

    bmi_per e derivado de k por (k-1)//3 + 1: os dados de forca estatica tem
    so 10 antropometrias contra as 30 das funcoes de transferencia, na razao
    exata de 3 subclasses por antropometria -- a unica parte da decomposicao
    do indice que decorre da propria forma dos dados, nao de ajuste (Secao
    11.1 da documentacao).
    """
    postures = store.available_postures()
    n_k = N_ANTHROPOMETRY_MODELS
    S = np.zeros((len(postures), n_k, 6))
    R = np.zeros((len(postures), n_k, 6))
    total = len(postures) * n_k
    for pi, posture in enumerate(postures):
        for ki in range(n_k):
            k = ki + 1
            bmi_per = (k - 1) // 3 + 1
            if progress:
                progress((pi * n_k + ki) / total,
                         f'posture {posture}, model {k}/{n_k}')
            res = analyse_exposure_at(store, excitations, posture, k, bmi_per,
                                      measurement_time)
            S[pi, ki] = res.S
            Sd = daily_dose([res], [daily_time])
            R[pi, ki] = risk_factor_with_static(
                np.tile(Sd, (n_years, 1)), np.full(n_years, days_per_year),
                res.S_stat, start_age=start_age, equation=equation,
                protection=protection)
    if progress:
        progress(1.0, 'Done')
    return dict(postures=postures, S=S, R=R)


def classify(R):
    if R < RISK_LOW:
        return 'low probability of an adverse health effect'
    if R > RISK_HIGH:
        return 'high probability of an adverse health effect'
    return 'intermediate range'


def processing_report(exc):
    """Descricao dos parametros de FFT aplicados a uma excitacao."""
    n = exc.accel.shape[0]
    n_used = n - (n % 2)
    return {
        'Samples in file': f'{n:,}',
        'Samples used': f'{n_used:,} (even length required)',
        'Sample rate': f'{exc.sample_rate:.4g} Hz  (dt = {exc.dt:.6g} s)',
        'Record length': f'{exc.duration:.3f} s',
        'FFT length': f'{n_used} (no zero padding)',
        'Window': WINDOW,
        'Frequency resolution': f'{exc.resolution:.5g} Hz',
        'Nyquist frequency': f'{exc.raw_freq[-1]:.4g} Hz',
        'Spectrum type': 'one-sided amplitude (scaled 2/N; DC and Nyquist halved)',
        'High-pass cut': f'{HIGHPASS_HZ} Hz (bins below set to zero)',
        'Low-pass cut': f'{LOWPASS_HZ} Hz (bins above discarded)',
        'Frequency weighting': WEIGHTING,
        'Bins after band limiting': f'{len(exc.freq)} of {len(exc.raw_freq)}',
        'Displacement conversion': 'U(f) = A(f) / -(2*pi*f)^2',
        'Effective sample rate after IFFT': f'{2 * LOWPASS_HZ:.4g} Hz',
    }


# ----------------------------------------------------------------------
# Metodos de analise espectral (somente visualizacao)
#
# Estes metodos NAO alteram a cadeia de calculo de S_d^A e R^A. O motivo e
# fisico: o autoespectro descarta a fase, e sem fase nao ha como reconstruir
# a forca no dominio do tempo, que e onde os picos de compressao sao contados.
# A cadeia de calculo continua usando o espectro complexo de bloco unico com
# janela retangular, que e a configuracao validada contra a referencia.
# ----------------------------------------------------------------------

WINDOW_FUNCTIONS = {
    'Rectangular': 'boxcar',
    'Hann': 'hann',
    'Hamming': 'hamming',
    'Flat top': 'flattop',
    'Blackman-Harris': 'blackmanharris',
}

SPECTRUM_METHODS = [
    'Single block (computation chain)',
    'Welch autospectrum - amplitude',
    'Welch autospectrum - PSD',
    'Welch autospectrum - RMS per band',
    'One-third octave RMS',
    'Single block - energy spectral density (ESD)',
    'Peak hold (maximum over Welch segments)',
    'Exponential average (Welch with exp. weighting)',
]

# Frequencias centrais nominais de 1/3 de oitava, ISO 266
THIRD_OCTAVE_CENTERS = np.array([
    0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0])


def _weight_config(weighting):
    """Aceita None, o nome de uma ponderacao, ou um dicionario por eixo."""
    if weighting is None:
        return {a: (None, 1.0) for a in AXIS_NAMES}
    if isinstance(weighting, str):
        return {a: (weighting, 1.0) for a in AXIS_NAMES}
    return {a: tuple(weighting[a]) for a in AXIS_NAMES}


def _axis_gain(cfg, axis_index, freq):
    name, k = cfg[AXIS_NAMES[axis_index]]
    if name is None:
        return k
    return np.abs(weighting_response(name, freq)) * k


def _window(name, n):
    from scipy.signal import get_window
    return get_window(WINDOW_FUNCTIONS.get(name, 'hann'), n, fftbins=True)


def _segments(x, nperseg, overlap):
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = range(0, len(x) - nperseg + 1, step)
    return [x[s:s + nperseg] for s in starts] or [x[:nperseg]]


def welch_spectrum(x, dt, window='Hann', nperseg=8192, overlap=0.5, kind='amplitude'):
    """Autoespectro de Welch: media dos periodogramas de segmentos sobrepostos.

    kind='amplitude' devolve um espectro de amplitude corrigido pela janela,
    comparavel ao de bloco unico para componentes senoidais.
    kind='psd' devolve densidade espectral de potencia, em unidade^2/Hz.

    O numero de medias e o que reduz a variancia: o desvio do estimador cai
    aproximadamente com a raiz do numero de segmentos.
    """
    n = len(x)
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    w = _window(window, nperseg)
    segs = _segments(np.asarray(x, dtype=float), nperseg, overlap)
    fs = 1.0 / dt
    f = np.arange(nperseg // 2 + 1) / (nperseg * dt)

    acc = np.zeros(len(f))
    for seg in segs:
        X = np.fft.rfft((seg - seg.mean()) * w)
        acc += np.abs(X) ** 2
    acc /= len(segs)

    if kind == 'psd':
        y = acc * 2.0 / (fs * np.sum(w ** 2))
        y[0] /= 2.0
        y[-1] /= 2.0
    else:
        y = np.sqrt(acc) * 2.0 / np.sum(w)      # correcao de amplitude
        y[0] /= 2.0
        y[-1] /= 2.0
    return f, y, len(segs)


def peak_hold_spectrum(x, dt, window='Hann', nperseg=8192, overlap=0.5):
    """Espectro peak-hold: maximo pontual sobre todos os segmentos de Welch.

    Cada posicao de frequencia retém o maior valor de amplitude visto em
    qualquer segmento. Útil para detectar componentes tonais intermitentes
    que a media de Welch suavizaria.

    Devolve (f, spectrum, n_segments).
    """
    n = len(x)
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    w = _window(window, nperseg)
    w_sum = float(np.sum(w))
    segs = _segments(np.asarray(x, dtype=float), nperseg, overlap)
    f = np.arange(nperseg // 2 + 1) / (nperseg * dt)
    hold = np.zeros(len(f))
    for seg in segs:
        X = np.fft.rfft((seg - seg.mean()) * w)
        amp = np.abs(X) * 2.0 / w_sum
        amp[0] /= 2.0
        amp[-1] /= 2.0
        hold = np.maximum(hold, amp)
    return f, hold, len(segs)


def exponential_average_spectrum(x, dt, window='Hann', nperseg=8192,
                                  overlap=0.5, tau_segments=4.0):
    """Espectro com media exponencial sobre os segmentos de Welch.

    Cada segmento tem peso exp(-k/tau_segments) onde k e' o indice do
    segmento contado a partir do FINAL (o ultimo segmento tem peso 1).
    Equivale a uma memoria exponencial com constante tau_segments segmentos.

    Devolve (f, spectrum, n_segments).
    """
    n = len(x)
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    w = _window(window, nperseg)
    w_sum = float(np.sum(w))
    segs = _segments(np.asarray(x, dtype=float), nperseg, overlap)
    ns = len(segs)
    f = np.arange(nperseg // 2 + 1) / (nperseg * dt)
    # pesos: exp(-k/tau) para k = ns-1, ns-2, ..., 0 (ultimo = peso 1)
    k = np.arange(ns - 1, -1, -1, dtype=float)
    weights = np.exp(-k / max(tau_segments, 0.1))
    weights /= weights.sum()

    acc = np.zeros(len(f))
    for seg, wt in zip(segs, weights):
        X = np.fft.rfft((seg - seg.mean()) * w)
        amp = np.abs(X) * 2.0 / w_sum
        amp[0] /= 2.0
        amp[-1] /= 2.0
        acc += wt * amp
    return f, acc, ns


def energy_spectral_density(x, dt):
    """Densidade espectral de ENERGIA (ESD), em (unidade)^2 s/Hz.

    Para um sinal transiente de energia finita (um choque/impacto isolado,
    nao um processo estacionario), densidade espectral de POTENCIA (PSD)
    nao faz sentido fisico: a "potencia media" dependeria arbitrariamente
    de quanto silencio ha antes/depois do choque no registro. A ESD
    normaliza pela ENERGIA total do sinal (teorema de Parseval), que e
    uma propriedade do proprio pulso, nao do registro que o contem.

    Sem janela, sem segmentacao -- FFT unica do bloco inteiro, mesmo
    espirito do metodo "Single block" (SPECTRUM_METHODS[0]).

    Identidade de Parseval (verificada por teste): soma(ESD)*df ==
    dt*soma(x^2) (energia total).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, dt)
    esd = 2.0 * dt ** 2 * np.abs(X) ** 2
    esd[0] /= 2.0
    if n % 2 == 0:
        esd[-1] /= 2.0
    return f, esd


def limit_curve_status(f, y, limit_freqs, limit_values):
    """Interpola uma curva-limite definida pelo usuario (frequencia, valor)
    nas frequencias de f, e verifica se o espectro y a ultrapassa em algum
    ponto dentro da faixa coberta pela curva-limite (fora dela, sem base
    de comparacao, nao faz sentido opinar). Mesmo espirito da "Vector
    reference curve" de ferramentas comerciais de analise espectral.

    Devolve (limit_interp, exceeded_mask, worst_ratio) -- limit_interp e'
    NaN fora da faixa coberta; worst_ratio e' o maior y/limit encontrado
    dentro da faixa (>1 significa que y ultrapassa o limite em algum ponto,
    NaN se a faixa nao se sobrepoe a f).
    """
    f = np.asarray(f, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(limit_freqs) < 2:
        return (np.full_like(f, np.nan), np.zeros_like(f, dtype=bool), np.nan)
    order = np.argsort(limit_freqs)
    lf = np.asarray(limit_freqs, dtype=float)[order]
    lv = np.asarray(limit_values, dtype=float)[order]
    in_range = (f >= lf[0]) & (f <= lf[-1])
    limit_interp = np.full_like(f, np.nan)
    limit_interp[in_range] = np.interp(f[in_range], lf, lv)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(in_range, y / limit_interp, np.nan)
    exceeded = np.where(in_range, y > limit_interp, False)
    worst_ratio = np.nanmax(ratio) if in_range.any() else np.nan
    return limit_interp, exceeded, worst_ratio


def band_rms(f, psd, edges):
    """Valor eficaz por banda, integrando a densidade espectral de potencia."""
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = np.sqrt(np.trapezoid(psd[m], f[m])) if m.sum() > 1 \
                else np.sqrt(psd[m][0] * (edges[i + 1] - edges[i]))
    return out


def block_compression_force(store, blocks, dt, posture, k, bmi_per):
    """Espectro e forma de onda (por bloco) da forca de compressao
    intervertebral no assento -- visualizacao apenas, nao produz S^A/R^A.

    Assume excitacao SOMENTE no assento (pes/maos/encosto zerados): o unico
    caso fisicamente possivel com um sensor so (registrador de campo). Cada
    bloco de 2 s cobre poucos ciclos nas frequencias mais baixas da banda, o
    que torna o numero de um unico bloco pouco confiavel -- a media entre
    blocos (no dominio da frequencia) e' o que da uma estimativa razoavel;
    no dominio do tempo cada bloco e' reconstruido de forma independente
    (nunca concatenando aceleracao pelas lacunas de 3 s, que fabricaria
    descontinuidades e picos falsos), so' emendados com quebra na exibicao.

    blocks: (n_blocks, n_samples, 3) aceleracao no assento. Retorna
    (freq, F, force_time): F e' o espectro complexo da forca de compressao
    (n_blocks, 6, n_freq); force_time e' a forma de onda correspondente
    (n_blocks, 6, n_amostras), reamostrada implicitamente pela banda
    0,5-100 Hz do modelo (mesmo efeito de analyse_exposure_at).
    """
    n = blocks.shape[1] - (blocks.shape[1] % 2)
    blocks = blocks[:, :n]
    half = n // 2
    y = np.fft.fft(blocks, axis=1) / half
    y[:, 0] *= 0.5
    y[:, half] *= 0.5
    A = y[:, :half + 1]                                   # (n_blocks, nf, 3)
    f = np.arange(half + 1) / (dt * n)
    keep = np.searchsorted(f, LOWPASS_HZ, side='right')
    A, f = A[:, :keep].copy(), f[:keep].copy()
    A[:, f < HIGHPASS_HZ] = 0
    w = 2 * np.pi * f
    with np.errstate(divide='ignore', invalid='ignore'):
        U = A / -(w ** 2)[None, :, None]
    U[:, 0] = 0

    H_all, f_model = store.posture(posture)
    Hi = np.stack([_interp_tf(f_model, f, H_all[k - 1, 0, ed, 2].astype(complex))
                  for ed in range(3)])                     # (ed=3, disc=6, nf)
    F = np.einsum('edf,bfe->bdf', Hi, U)

    Xt = np.moveaxis(F, -1, 0)                             # (nf, n_blocks, 6)
    force_time = np.moveaxis(ifft_amplitude(Xt), 0, -1)    # (n_blocks, 6, n_out)
    return f, F, force_time


def spectrogram(x, dt, window='Hann', nperseg=1024, overlap=0.5, quantity='accel'):
    """Espectrograma (STFT): espectro de amplitude por segmento sobreposto,
    ao longo do tempo -- cada coluna e' o mesmo espectro de amplitude que
    o welch_spectrum ja calcula, so que em janelas curtas deslizantes em
    vez de uma unica media do registro inteiro. Revela COMO o conteudo em
    frequencia muda com o tempo (ex.: a assinatura de banda larga e breve
    de um choque de guia, contra vibracao continua de rolamento).
    """
    n = len(x)
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    w = _window(window, nperseg)
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = list(range(0, n - nperseg + 1, step)) or [0]
    f = np.arange(nperseg // 2 + 1) / (nperseg * dt)
    x = np.asarray(x, dtype=float)
    S = np.empty((len(f), len(starts)))
    for j, s in enumerate(starts):
        seg = x[s:s + nperseg]
        X = np.fft.rfft((seg - seg.mean()) * w)
        amp = np.abs(X) * 2.0 / np.sum(w)
        amp[0] /= 2.0
        amp[-1] /= 2.0
        S[:, j] = amp
    t = (np.array(starts, dtype=float) + nperseg / 2.0) * dt
    if quantity != 'accel':
        wr = 2 * np.pi * f
        n_exp = 1 if quantity == 'velocity' else 2
        with np.errstate(divide='ignore', invalid='ignore'):
            S = S / wr[:, None] ** n_exp
        S[0, :] = 0.0
    return t, f, S


def _octave_spaced_freqs(f_start, f_end, points_per_octave):
    """Frequencias em passo de oitava fracionaria (razao 2^(1/n)), a
    convencao real de ASTM E1049/MIL-STD-810 para reportar SRS -- permite
    comparar resultados entre laboratorios/ferramentas diferentes, ao
    contrario de um numero arbitrario de pontos log-espacados.
    """
    n_octaves = np.log2(f_end / f_start)
    n_points = int(np.ceil(n_octaves * points_per_octave)) + 1
    return f_start * 2.0 ** (np.arange(n_points) / points_per_octave)


SRS_OVERSAMPLE = 20         # taxa efetiva minima, em multiplos de fn
SRS_CHUNK = 2 ** 18         # amostras por trecho na reamostragem
SRS_PAD = 64                # sobreposicao (amostras originais) entre trechos


def _smallwood_coefficients(fn, zeta, dt):
    """Filtro rampa-invariante de Smallwood (1981) para a aceleracao
    absoluta de um sistema de 1 GL excitado pela base; e' o algoritmo de
    referencia da ISO 18431-4. Diferente da bilinear, nao comprime o eixo
    de frequencias: o pico fica em fn."""
    wn = 2 * np.pi * fn
    E = np.exp(-zeta * wn * dt)
    K = wn * np.sqrt(1 - zeta ** 2) * dt
    C = E * np.cos(K)
    Sp = E * np.sin(K) / K
    return (np.array([1 - Sp, 2 * (Sp - C), E * E - Sp]),
            np.array([1.0, -2 * C, E * E]))


def _upsampled_chunks(x, up):
    """resample_poly(x, up, 1) em trechos, para registros longos caberem
    na memoria. Cada trecho e' reamostrado com SRS_PAD amostras de cada
    lado e as bordas sao descartadas, entao a concatenacao reproduz a
    reamostragem do registro inteiro."""
    from scipy.signal import resample_poly
    if up == 1:
        for s in range(0, len(x), SRS_CHUNK):
            yield x[s:s + SRS_CHUNK]
        return
    for s in range(0, len(x), SRS_CHUNK):
        e = min(s + SRS_CHUNK, len(x))
        lo, hi = max(s - SRS_PAD, 0), min(e + SRS_PAD, len(x))
        y = resample_poly(x[lo:hi], up, 1)
        yield y[(s - lo) * up:(e - lo) * up]


def shock_response_spectrum(x, dt, freqs=None, q=10.0, points_per_octave=12):
    """SRS maximax de aceleracao absoluta: resposta de um sistema massa-
    mola-amortecedor de 1 grau de liberdade excitado pela base, simulada
    por frequencia natural via filtro rampa-invariante de Smallwood,
    tomando o pico absoluto da resposta.

    Discretizacao: ate a v2.9 era bilinear sem correcao de frequencia, que
    comprime o eixo de frequencias. Com fs = 1024 Hz e Q = 10 o oscilador
    de 200 Hz ressoava em 179 Hz e o de 409 Hz em 292 Hz, com SRS 70 % e
    98 % abaixo do correto. Agora usa Smallwood (ISO 18431-4), e quando
    fn passa de fs/SRS_OVERSAMPLE o sinal e' reamostrado (resample_poly)
    para pelo menos SRS_OVERSAMPLE*fn: com 20 amostras por ciclo o erro de
    ganho em fn fica em 0,8 % e o pico da saida amostrada nao se perde
    entre amostras. Um seno de 1 m/s2 em fn da' SRS(fn) = Q dentro de 1 %
    ate 0,4*fs (tests/test_reference.py).

    Padrao em teste de choque/transporte (MIL-STD-810): caracteriza
    severidade em funcao da frequencia de resposta, em vez de resumir um
    choque num RMS ou pico unico que nao diz nada sobre que estrutura
    ressoante ele mais excitaria.

    q = fator de qualidade (Q = 1/(2*zeta)); Q=10 (zeta=0,05) e' a
    convencao mais comum -- confirmada tambem como "standard value" no
    manual de SRS de uma ferramenta comercial (Dewesoft), independente do
    MIL-STD-810.

    points_per_octave controla a resolucao em frequencia quando freqs=None,
    em oitava fracionaria (2^(1/n)) -- a mesma convencao usada por
    ferramentas de choque comerciais (ali, um menu com 3/6/12/24/48 pontos
    por oitava), em vez de um numero arbitrario de pontos log-espacados.

    O offset DC (media do registro inteiro) e' removido antes do filtro:
    sem isso, qualquer deriva estatica do acelerometro vaza para a resposta
    de baixa frequencia dos sistemas de 1 GL, inflando o SRS ali.
    """
    if freqs is None:
        f_max = min(500.0, 0.4 / dt)
        freqs = _octave_spaced_freqs(2.0, f_max, points_per_octave)
    freqs = np.asarray(freqs, dtype=float)
    x = np.asarray(x, dtype=float)
    hi, lo = _srs_extremes(x - x.mean(), dt, freqs, q, [(0, len(x))])
    return freqs, np.maximum(np.abs(hi[0]), np.abs(lo[0]))


def _srs_extremes(x, dt, freqs, q, windows):
    """Maior e menor valor da resposta de cada oscilador dentro de cada
    janela [(inicio, fim)] em amostras do sinal original, podendo ser
    fracionarias (fim exclusivo arredondado para cima). A resposta e' uma
    so, do registro inteiro: as janelas so escolhem onde procurar o
    extremo, entao o oscilador entra na janela residual com o estado que
    o choque deixou. Devolve (max, min), cada um (janelas, f)."""
    import math
    from scipy.signal import lfilter
    zeta = 1.0 / (2 * q)
    nw, nf = len(windows), len(freqs)
    hi = np.full((nw, nf), -np.inf)
    lo = np.full((nw, nf), np.inf)
    ups = np.maximum(1, np.ceil(SRS_OVERSAMPLE * freqs * dt)).astype(int)
    # uma reamostragem por fator, todos os osciladores daquele fator juntos
    for up in np.unique(ups):
        idx = np.flatnonzero(ups == up)
        coefs = [_smallwood_coefficients(freqs[i], zeta, dt / up) for i in idx]
        states = [np.zeros(2) for _ in idx]
        pos = 0
        for chunk in _upsampled_chunks(x, up):
            end = pos + len(chunk)
            # fronteiras fracionarias convertidas para amostras do sinal
            # reamostrado com ceil: garante que a ultima amostra do pulso
            # fica no primario e a primeira do livre-amortecimento no residual
            cuts = [(max(math.ceil(a * up), pos) - pos,
                     min(math.ceil(b * up), end) - pos)
                    for a, b in windows]
            for k, (i, (b, a)) in enumerate(zip(idx, coefs)):
                resp, states[k] = lfilter(b, a, chunk, zi=states[k])
                for w, (c0, c1) in enumerate(cuts):
                    if c1 > c0:
                        part = resp[c0:c1]
                        hi[w, i] = max(hi[w, i], part.max())
                        lo[w, i] = min(lo[w, i], part.min())
            pos = end
    hi[~np.isfinite(hi)] = np.nan
    lo[~np.isfinite(lo)] = np.nan
    return hi, lo


SHOCK_THRESHOLD = 0.1       # fracao do pico que delimita a janela do choque
PRE_SHOCK_MIN_S = 0.2       # trecho minimo antes do choque para offset e ruido


def shock_window(x, dt, threshold=SHOCK_THRESHOLD):
    """Janela do choque em amostras (inicio, fim exclusivo): do primeiro ao
    ultimo instante em que |x - mediana| passa de threshold vezes o pico.
    A mediana (e nao a media) como referencia, para que o proprio choque
    nao desloque o nivel de repouso."""
    x = np.asarray(x, dtype=float)
    d = np.abs(x - np.median(x))
    if not len(d) or d.max() == 0:
        return 0, len(x)
    over = np.flatnonzero(d >= threshold * d.max())
    return int(over[0]), int(over[-1]) + 1


def shock_response_spectra(x, dt, window=None, freqs=None, q=10.0,
                           points_per_octave=12):
    """SRS primario, residual e maximax de um choque, como nas ferramentas
    comerciais de choque:

      primario   extremos da resposta DURANTE o choque (resposta forcada)
      residual   extremos DEPOIS do choque (vibracao livre): mostra se a
                 estrutura (cadeira, assento, pessoa) continua oscilando
      maximax    maior |resposta| no registro inteiro

    cada um com o maximo positivo e o minimo negativo separados (pos, neg)
    e o maior absoluto. window = (inicio, fim) em segundos, ou None para
    shock_window. O offset vem do trecho ANTES do choque (o repouso), nao
    da media do registro, que o proprio choque desloca; o SRS maximax desse
    trecho e' o piso de ruido ('noise'), NaN se ele for curto demais."""
    if freqs is None:
        f_max = min(500.0, 0.4 / dt)
        freqs = _octave_spaced_freqs(2.0, f_max, points_per_octave)
    freqs = np.asarray(freqs, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window is None:
        s0, s1 = shock_window(x, dt)
        t1_frac = float(s1)
    else:
        s0 = int(np.clip(round(window[0] / dt), 0, n))
        t1_frac = float(np.clip(window[1] / dt, s0 + 1, n))
        s1 = int(np.ceil(t1_frac))
    pre = s0 >= PRE_SHOCK_MIN_S / dt
    x = x - (x[:s0].mean() if pre else x.mean())
    hi, lo = _srs_extremes(x, dt, freqs, q,
                           [(float(s0), t1_frac), (t1_frac, float(n)), (0.0, float(n))])
    out = {'f': freqs, 'window_s': (s0 * dt, s1 * dt),
           'primary_pos': hi[0], 'primary_neg': lo[0],
           'residual_pos': hi[1], 'residual_neg': lo[1],
           'offset_from': 'pre-shock' if pre else 'whole record'}
    for key, w in (('primary', 0), ('residual', 1), ('maximax', 2)):
        out[key] = np.fmax(np.abs(hi[w]), np.abs(lo[w]))
    if pre:
        h, l = _srs_extremes(x[:s0], dt, freqs, q, [(0, s0)])
        out['noise'] = np.fmax(np.abs(h[0]), np.abs(l[0]))
    else:
        out['noise'] = np.full(len(freqs), np.nan)
    return out


def srs_quantity(freqs, srs_accel, quantity):
    """Converte um SRS de aceleracao para pseudo-velocidade/pseudo-
    deslocamento -- convencao padrao de teste de choque (MIL-STD-810,
    ASTM E1049), usada por ferramentas comerciais de SRS (ex. Dewesoft,
    que mostra Acceleration/Velocity/Displacement lado a lado).

    Diferenca importante em relacao a _convert_quantity(): ali a conversao
    e' por integracao no dominio da frequencia de um espectro CONTINUO
    (V(f)=A(f)/jw). Aqui cada ponto do SRS ja e' a resposta de PICO de um
    sistema de 1 grau de liberdade distinto, com sua propria frequencia
    natural -- a conversao e' ponto a ponto pela frequencia natural DAQUELE
    sistema (pseudo-velocidade = SRS/omega_n), nao uma integracao do sinal.
    """
    label = QUANTITY_LABELS[quantity]
    unit = QUANTITY_UNITS[quantity]
    if quantity == 'accel':
        return srs_accel, f'SRS {label.lower()} [{unit}]'
    w = 2 * np.pi * np.asarray(freqs, dtype=float)
    n = 1 if quantity == 'velocity' else 2
    with np.errstate(divide='ignore', invalid='ignore'):
        y = srs_accel / w ** n
    y = np.where(w > 0, y, 0.0)
    return y, f'SRS pseudo-{label.lower()} [{unit}]'


def third_octave_rms(x, dt, window='Hann', nperseg=8192, overlap=0.5):
    f, psd, navg = welch_spectrum(x, dt, window, nperseg, overlap, kind='psd')
    centers = THIRD_OCTAVE_CENTERS[THIRD_OCTAVE_CENTERS <= min(LOWPASS_HZ,
                                                               0.5 / dt)]
    edges = np.concatenate([centers / 2 ** (1 / 6), [centers[-1] * 2 ** (1 / 6)]])
    return centers, band_rms(f, psd, edges), navg


QUANTITY_LABELS = {'accel': 'Acceleration', 'velocity': 'Velocity',
                   'displacement': 'Displacement'}
QUANTITY_UNITS = {'accel': 'm/s$^2$', 'velocity': 'm/s', 'displacement': 'm'}


def _convert_quantity(f, y, label_template, quantity, power):
    """Converte um espectro de aceleracao pra velocidade/deslocamento por
    integracao no dominio da frequencia (V(f) = A(f)/jw, U(f) = A(f)/-w^2).
    So a MAGNITUDE importa aqui (visualizacao) -- por isso e' so 1/w^n, sem
    preservar fase, ao contrario do exc.spectrum usado na cadeia de forca.

    power=True quando y e' densidade espectral (unidade ao quadrado por Hz,
    ex. PSD) -- nesse caso a integracao escala com w^(2n), nao w^n.

    label_template usa {q} (nome da grandeza) e {u} (unidade, ja ao quadrado
    se power=True) -- ex. '{q} amplitude [{u}]', 'PSD [({u})$^2$/Hz]'.
    """
    unit = QUANTITY_UNITS[quantity]
    label = label_template.format(q=QUANTITY_LABELS[quantity], u=unit)
    if quantity == 'accel':
        return y, label
    n = 1 if quantity == 'velocity' else 2
    exponent = 2 * n if power else n
    w = 2 * np.pi * f
    with np.errstate(divide='ignore', invalid='ignore'):
        y2 = y / w ** exponent
    y2 = np.where(w > 0, y2, 0.0)
    return y2, label


def analysis_spectrum(exc, method, axis=None, window='Hann', nperseg=8192,
                      overlap=0.5, band_width=1.0, weighting=None,
                      quantity='accel'):
    """Espectro para visualizacao. Devolve (f, y, rotulo do eixo y, nota).

    weighting aplica uma ponderacao da ISO 2631-1 ao espectro exibido. Nao
    altera nada do calculo de dose ou de fator de risco. quantity troca
    aceleracao por velocidade/deslocamento (so na visualizacao, mesmo
    espirito do weighting).
    """
    sig = exc.accel if axis is None else exc.accel[:, [axis]]
    cols = list(range(3)) if axis is None else [axis]
    cfg = _weight_config(weighting)
    used = [cfg[AXIS_NAMES[c]] for c in cols]
    tag = ('' if all(n is None and k == 1 for n, k in used) else
           ' Weighted: ' + ', '.join(
               f'{AXIS_NAMES[c]}={n or "-"}' + (f'x{k:g}' if k != 1 else '')
               for c, (n, k) in zip(cols, used)) + '.')

    if method == SPECTRUM_METHODS[0]:
        if axis is None:
            y = np.sqrt(sum((np.abs(exc.accel_spectrum[:, c])
                             * _axis_gain(cfg, c, exc.freq)) ** 2
                            for c in range(3)))
        else:
            y = np.abs(exc.accel_spectrum[:, axis]) * _axis_gain(cfg, axis, exc.freq)
        y, ylabel = _convert_quantity(
            exc.freq, y, '{q} amplitude [{u}]', quantity, power=False)
        return (exc.freq, y, ylabel,
                'Single complex FFT of the whole record, rectangular window. '
                'This is what feeds the force computation.' + tag)

    # Potencias somam entre eixos; amplitudes somam em quadratura. Somar
    # amplitudes escalarmente superestimaria o resultado.
    power = method in SPECTRUM_METHODS[2:4]
    acc = None
    navg = 1
    for i, c in enumerate(cols):
        f, y, navg = welch_spectrum(
            sig[:, i], exc.dt, window, nperseg, overlap,
            kind='psd' if power else 'amplitude')
        g = _axis_gain(cfg, c, f)
        y = y * (g ** 2 if power else g)
        contrib = y if power else y ** 2
        acc = contrib if acc is None else acc + contrib
    if not power:
        acc = np.sqrt(acc)

    res = 1.0 / (nperseg * exc.dt)
    tail = (f'{navg} averaged segments of {nperseg} samples, {window} window, '
            f'{overlap * 100:.0f}% overlap, resolution {res:.4g} Hz.' + tag)

    if method == SPECTRUM_METHODS[1]:
        y, ylabel = _convert_quantity(f, acc, '{q} amplitude [{u}]',
                                      quantity, power=False)
        return f, y, ylabel, tail
    if method == SPECTRUM_METHODS[2]:
        y, ylabel = _convert_quantity(f, acc, 'PSD [({u})$^2$/Hz]',
                                      quantity, power=True)
        return f, y, ylabel, tail
    if method == SPECTRUM_METHODS[3]:
        # Converte a PSD (ainda em aceleracao) ANTES de integrar em bandas --
        # mais correto que converter o RMS por banda depois, porque evita
        # aproximar 1/w pela frequencia central da banda.
        acc_q, _ = _convert_quantity(f, acc, '', quantity, power=True)
        edges = np.arange(0, min(LOWPASS_HZ, f[-1]) + band_width, band_width)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return (centers, band_rms(f, acc_q, edges), 'Band RMS [{u}]'.format(
                    u=QUANTITY_UNITS[quantity]),
                tail + f' Integrated into {band_width:g} Hz bands.')

    if method == SPECTRUM_METHODS[5]:
        acc = None
        f = None
        for i, c in enumerate(cols):
            f, esd = energy_spectral_density(sig[:, i], exc.dt)
            g = _axis_gain(cfg, c, f)
            contrib = esd * g ** 2
            acc = contrib if acc is None else acc + contrib
        keep = f <= min(LOWPASS_HZ, 0.5 / exc.dt)
        f, acc = f[keep], acc[keep]
        y, ylabel = _convert_quantity(f, acc, 'ESD [({u})$^2$s/Hz]',
                                      quantity, power=True)
        return (f, y, ylabel,
                'Energy spectral density of the whole record, no averaging '
                'or window (single FFT, same chain as "Single block"). '
                'Meaningful for one finite-duration transient, where power '
                'spectral density (PSD) would depend arbitrarily on how '
                'much silence surrounds the shock in the record.' + tag)

    if method == SPECTRUM_METHODS[6]:
        # Peak hold: maximo pontual por segmento, somado em quadratura entre eixos
        hold = None
        for i, c in enumerate(cols):
            f, y, navg = peak_hold_spectrum(sig[:, i], exc.dt, window, nperseg, overlap)
            g = _axis_gain(cfg, c, f)
            contrib = (y * g) ** 2
            hold = contrib if hold is None else hold + contrib
        hold = np.sqrt(hold)
        y, ylabel = _convert_quantity(f, hold, '{q} amplitude [{u}]',
                                      quantity, power=False)
        tail_ph = (f'{navg} segments, {window} window, {overlap*100:.0f}% overlap '
                   f'— peak hold (maximum over all segments).' + tag)
        return f, y, ylabel, tail_ph

    if method == SPECTRUM_METHODS[7]:
        # Media exponencial: segmentos mais recentes tem peso maior
        exp_acc = None
        for i, c in enumerate(cols):
            f, y, navg = exponential_average_spectrum(
                sig[:, i], exc.dt, window, nperseg, overlap)
            g = _axis_gain(cfg, c, f)
            contrib = (y * g) ** 2
            exp_acc = contrib if exp_acc is None else exp_acc + contrib
        exp_acc = np.sqrt(exp_acc)
        y, ylabel = _convert_quantity(f, exp_acc, '{q} amplitude [{u}]',
                                      quantity, power=False)
        tail_exp = (f'{navg} segments, {window} window, {overlap*100:.0f}% overlap '
                    f'— exponential average (τ = 4 segments).' + tag)
        return f, y, ylabel, tail_exp

    # Um terco de oitava: soma as potencias de banda dos eixos escolhidos, nunca
    # os sinais crus (somar ax+ay+az amostra a amostra permitiria cancelamento
    # de fase entre eixos, o que nao tem sentido fisico).
    total, centers = None, None
    for i, c in enumerate(cols):
        centers, rms, navg = third_octave_rms(sig[:, i], exc.dt, window,
                                              nperseg, overlap)
        rms = rms * _axis_gain(cfg, c, centers)
        total = rms ** 2 if total is None else total + rms ** 2
    total = np.sqrt(total)
    # Aproximacao: converte pela frequencia CENTRAL de cada banda (nao ha
    # PSD continua disponivel aqui pra converter antes de integrar, como no
    # metodo de banda linear acima) -- razoavel pra bandas de terco de
    # oitava, estreitas o suficiente relativo ao centro.
    total_q, ylabel = _convert_quantity(centers, total, 'Band RMS [{u}]',
                                        quantity, power=False)
    return (centers, total_q, ylabel,
            f'{navg} averaged segments, {window} window. '
            f'One-third octave bands, ISO 266 nominal centres.' + tag)


def phase_spectrum(exc, axis, unwrap=True):
    """Fase da FFT de bloco unico, o mesmo espectro complexo que alimenta o
    calculo (Secao 3.2). So faz sentido para um eixo por vez: a fase da soma
    vetorial nao tem interpretacao fisica direta.

    Devolve (freq, fase em graus). A fase e desenrolada por padrao para expor
    tendencias (por exemplo, atraso de propagacao entre pontos de contato);
    a versao enrolada, em (-180, 180], e a que corresponde literalmente ao
    angulo complexo de cada bin.
    """
    ph = np.angle(exc.accel_spectrum[:, axis])
    if unwrap:
        ph = np.unwrap(ph)
    return exc.freq, np.degrees(ph)


def welch_cross_spectrum(x, y, dt, window='Hann', nperseg=8192, overlap=0.5):
    """Densidade espectral cruzada de Welch entre dois sinais sincronos.

    Devolve (freq, Sxy, Sxx, Syy, navg). Sxy e complexa; Sxx e Syy sao os
    autoespectros (reais, >= 0), na mesma normalizacao de potencia usada por
    welch_spectrum(kind='psd'). Os dois sinais devem ter o mesmo passo de
    tempo e o mesmo comprimento -- e responsabilidade do chamador garantir
    isso (ver core.check_time_base), porque a soma cruzada bin a bin so faz
    sentido numa grade de frequencia comum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    w = _window(window, nperseg)
    xs = _segments(x, nperseg, overlap)
    ys = _segments(y, nperseg, overlap)
    fs = 1.0 / dt
    f = np.arange(nperseg // 2 + 1) / (nperseg * dt)
    norm = 2.0 / (fs * np.sum(w ** 2))

    sxy = np.zeros(len(f), dtype=complex)
    sxx = np.zeros(len(f))
    syy = np.zeros(len(f))
    navg = min(len(xs), len(ys))
    for i in range(navg):
        X = np.fft.rfft((xs[i] - xs[i].mean()) * w)
        Y = np.fft.rfft((ys[i] - ys[i].mean()) * w)
        sxy += X * np.conj(Y)
        sxx += np.abs(X) ** 2
        syy += np.abs(Y) ** 2
    sxy *= norm / navg
    sxx *= norm / navg
    syy *= norm / navg
    return f, sxy, sxx, syy, navg


# ----------------------------------------------------------------------
# Espectros por segmento: rejeicao de segmentos e H1 por classe de amplitude
# ----------------------------------------------------------------------

SEGMENT_ROBUST_Z = 6.0      # desvios robustos (MAD) para marcar um segmento
CLIP_FRACTION = 0.995       # fracao do fundo de escala que conta como saturado
AMPLITUDE_EDGES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # m/s2 rms da entrada
MIN_CLASS_AVERAGES = 8


def segment_cross_spectra(x, y, dt, window='Hann', nperseg=8192, overlap=0.5):
    """Os segmentos de Welch um a um, na convencao de welch_cross_spectrum
    (media de cada segmento removida, janela aplicada). Devolve um dict com
    f, X e Y (k, nf) complexos e, por segmento, as grandezas que
    segment_flags usa (rms, curtose, degrau de media, saturacao). A escala
    dos espectros nao importa para H e coerencia, que sao razoes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    nperseg = int(min(nperseg, n - (n % 2)))
    if nperseg % 2:
        nperseg -= 1
    xs = np.array(_segments(x[:n], nperseg, overlap))
    ys = np.array(_segments(y[:n], nperseg, overlap))
    return _segment_dict(xs, ys, dt, window, x[:n], y[:n])


def block_segment_spectra(bx, by, dt, window='Hann'):
    """segment_cross_spectra para blocos (n_blocos, N) ja sincronos: cada
    bloco e' um segmento, como em block_cross_estimates."""
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    m = min(len(bx), len(by))
    n = min(bx.shape[1], by.shape[1])
    return _segment_dict(bx[:m, :n], by[:m, :n], dt, window,
                         bx[:m, :n].ravel(), by[:m, :n].ravel())


SEGMENT_CHUNK = 256         # segmentos processados por vez


def _segment_dict(xs, ys, dt, window, x_full, y_full):
    """Processa os segmentos em lotes e guarda so os espectros (complex64)
    e as grandezas por segmento: um dia inteiro a 1 kHz nao cabe na memoria
    com todos os segmentos no tempo em float64."""
    k, n = xs.shape
    w = _window(window, n)
    nf = n // 2 + 1
    out = {'f': np.arange(nf) / (n * dt), 'n': n,
           'X': np.empty((k, nf), np.complex64),
           'Y': np.empty((k, nf), np.complex64)}
    for key in ('rms_x', 'rms_y', 'step_x', 'step_y', 'kurt_x', 'kurt_y'):
        out[key] = np.zeros(k)
    h = n // 2
    for a in range(0, k, SEGMENT_CHUNK):
        b = min(k, a + SEGMENT_CHUNK)
        for tag, raw in (('x', xs[a:b]), ('y', ys[a:b])):
            s = raw - raw.mean(axis=1, keepdims=True)
            r = np.sqrt((s ** 2).mean(axis=1))
            out['rms_' + tag][a:b] = r
            d = np.abs(s[:, h:].mean(1) - s[:, :h].mean(1))
            out['step_' + tag][a:b] = np.divide(d, r, out=np.zeros(b - a),
                                                where=r > 0)
            out['kurt_' + tag][a:b] = np.divide(
                (s ** 4).mean(1), r ** 4, out=np.full(b - a, 3.0), where=r > 0)
            out[tag.upper()][a:b] = np.fft.rfft(s * w, axis=1)
    return out


def _robust_z(v, floor):
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = max(1.4826 * np.median(np.abs(v - med)), floor)
    return np.abs(v - med) / mad


def clipped_segments(raw, full_scale, nperseg, overlap=0.0):
    """Mascara (k,) dos segmentos em que |raw| chega a CLIP_FRACTION do
    fundo de escala do sensor (mesma unidade de raw, que deve conter a
    aceleracao total, com a gravidade, como o sensor a ve). Mesma
    segmentacao de segment_cross_spectra (blocos: overlap 0, nperseg =
    tamanho do bloco).

    Sem o fundo de escala nao ha criterio confiavel: platos de amostras
    iguais no pico sao normais em sinal lento amostrado rapido (um seno de
    5 Hz a 1 kHz varia menos que um LSB de 16 bits perto do pico), e o
    maior valor do registro pode ser um impacto em outro trecho."""
    raw = np.asarray(raw, dtype=float)
    hit = np.abs(raw) >= CLIP_FRACTION * full_scale
    segs = np.array(_segments(hit.astype(float), nperseg, overlap))
    return segs.any(axis=1) if len(segs) else np.zeros(0, dtype=bool)


def segment_flags(seg, z=SEGMENT_ROBUST_Z, clipped=None):
    """Marca segmentos a rejeitar antes da media, cada um com o motivo:

      clipped     saturacao (so com a mascara de clipped_segments, que
                  precisa do fundo de escala do sensor)
      gain        razao rms saida/entrada fora do resto (perda de contato,
                  sensor solto, batida so num canal)
      offset      degrau da media dentro do segmento (sensor que mexeu)
      kurtosis    impacto num canal so: razao de curtose saida/entrada
                  fora do resto. Um choque transmitido pelo corpo aparece
                  nos dois sensores e NAO e' rejeitado (e' exposicao real,
                  muitas vezes a mais severa); uma batida no sensor de
                  saida aparece so nele

    Os tres ultimos usam escore robusto (mediana e MAD) contra os outros
    segmentos do mesmo registro, com piso no MAD para que registros muito
    uniformes nao marquem variacoes minimas. Devolve {motivo: mascara} e
    'rejected' (qualquer motivo). Com menos de 4 segmentos nao ha base de
    comparacao e so a saturacao e' verificada."""
    k = len(seg['rms_x'])
    zero = np.zeros(k, dtype=bool)
    flags = {'clipped': zero.copy() if clipped is None
             else np.asarray(clipped, dtype=bool)[:k].copy()}
    if k >= 4:
        rx, ry = seg['rms_x'], seg['rms_y']
        ok = (rx > 0) & (ry > 0)
        ratio = np.log(np.where(ok, ry, 1.0) / np.where(ok, rx, 1.0))
        flags['gain'] = (_robust_z(ratio, 0.05) > z) | ~ok
        flags['offset'] = ((_robust_z(seg['step_x'], 0.05) > z) |
                           (_robust_z(seg['step_y'], 0.05) > z))
        flags['kurtosis'] = _robust_z(
            np.log(seg['kurt_y'] / np.maximum(seg['kurt_x'], 1e-9)), 0.1) > z
    else:
        flags.update(gain=zero.copy(), offset=zero.copy(), kurtosis=zero.copy())
    rej = zero.copy()
    for v in flags.values():
        rej |= v
    flags['rejected'] = rej
    return flags


def flag_summary(flags):
    """'3 of 120 segments rejected (clipped 1, gain 2)' ou ''."""
    rej = flags['rejected']
    if not rej.any():
        return ''
    why = ', '.join(f'{k} {int(v.sum())}' for k, v in flags.items()
                    if k != 'rejected' and v.any())
    return f'{int(rej.sum())} of {len(rej)} segments rejected ({why})'


def frf_from_segments(seg, mask=None):
    """frf_estimates com a media so dos segmentos em mask (todos se None).
    Devolve (estimativas, numero de medias)."""
    idx = (np.arange(len(seg['X'])) if mask is None
           else np.flatnonzero(mask))
    nf = seg['X'].shape[1]
    gxx, gyy = np.zeros(nf), np.zeros(nf)
    gxy = np.zeros(nf, dtype=complex)
    for a in range(0, len(idx), SEGMENT_CHUNK):     # soma em float64 por lote
        sel = idx[a:a + SEGMENT_CHUNK]
        X = seg['X'][sel].astype(complex)
        Y = seg['Y'][sel].astype(complex)
        gxx += (np.abs(X) ** 2).sum(0)
        gyy += (np.abs(Y) ** 2).sum(0)
        gxy += (np.conj(X) * Y).sum(0)
    m = len(idx)
    if m:
        gxx, gyy, gxy = gxx / m, gyy / m, gxy / m
    return frf_estimates(gxx, gyy, gxy, m), m


def amplitude_classes(seg, edges=AMPLITUDE_EDGES, mask=None,
                      min_avg=MIN_CLASS_AVERAGES):
    """H por classe de valor eficaz da entrada (rms do segmento sem a media
    e sem ponderacao): [(lo, hi, n, estimativas)], so as classes com pelo
    menos min_avg segmentos. A transmissibilidade do corpo e' nao linear
    (o pico desce em frequencia e em modulo com a amplitude); em campo a
    amplitude varia ao longo da viagem, e separar por classe mostra isso
    sem ensaio de bancada."""
    r = seg['rms_x']
    keep = np.ones(len(r), dtype=bool) if mask is None else mask
    bounds = [0.0, *edges, np.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = keep & (r >= lo) & (r < hi)
        if sel.sum() >= min_avg:
            est, n = frf_from_segments(seg, sel)
            out.append((lo, hi, n, est))
    return out


def coherence_spectrum(x, y, dt, window='Hann', nperseg=8192, overlap=0.5):
    """Coerencia magnitude-quadrado gamma^2(f) = |Sxy|^2 / (Sxx * Syy).

    Precisa de mais de um segmento para ser informativa: com um unico
    periodograma, gamma^2 vale 1 em todo bin por construcao, o que nao
    significa nada sobre a relacao real entre os sinais. O chamador deve
    checar navg e avisar o usuario quando ele for 1.
    """
    f, sxy, sxx, syy, navg = welch_cross_spectrum(
        x, y, dt, window, nperseg, overlap)
    denom = sxx * syy
    gamma2 = np.divide(np.abs(sxy) ** 2, denom, out=np.zeros_like(denom),
                       where=denom > 0)
    return f, np.clip(gamma2, 0.0, 1.0), navg


def cross_phase_spectrum(x, y, dt, window='Hann', nperseg=8192, overlap=0.5,
                         unwrap=True):
    """Fase de Sxy: defasagem de y em relacao a x, por frequencia, em graus.

    Ao contrario da fase de um unico canal (phase_spectrum), esta e a fase
    de uma media de varios segmentos, entao e informativa mesmo quando o
    sinal nao e periodico -- desde que x e y estejam de fato sincronos.
    """
    f, sxy, _, _, navg = welch_cross_spectrum(x, y, dt, window, nperseg, overlap)
    ph = np.angle(sxy)
    if unwrap:
        ph = np.unwrap(ph)
    return f, np.degrees(ph), navg


def modal_parameters(f, h1, fmin=0.5, fmax=None, g2=None):
    """Parametros modais de um pico de transmissibilidade: pico + meia potencia.

    f   : array de frequencias (Hz)
    h1  : array de FRF complexa (ou real se magnitude)
    fmin: frequencia minima para busca do pico (Hz)
    fmax: frequencia maxima (None = sem limite)
    g2  : coerencia (opcional) — restringe a busca a regioes com g2 >= 0.5

    Devolve dict com fn, Q, zeta, f_lo, f_hi, peak_mag, valid.
    valid=False se nao foi possivel encontrar as cruzamentos de meia potencia.
    """
    f = np.asarray(f, dtype=float)
    mag = np.abs(np.asarray(h1, dtype=complex))
    mask = f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    if g2 is not None:
        mask &= np.asarray(g2) >= 0.5
    if not mask.any():
        return {'valid': False}
    fi, mi = f[mask], mag[mask]
    pk_idx = int(mi.argmax())
    fn = float(fi[pk_idx])
    peak_mag = float(mi[pk_idx])
    half = peak_mag / 2 ** 0.5

    # Cruzamento inferior: maior f < fn onde mag < half
    lo_mask = (fi < fn) & (mi < half)
    hi_mask = (fi > fn) & (mi < half)
    if not lo_mask.any() or not hi_mask.any():
        return {'fn': fn, 'peak_mag': peak_mag, 'valid': False,
                'Q': None, 'zeta': None, 'f_lo': None, 'f_hi': None}
    # Interpolacao linear na cruzagem
    lo_idx = int(np.where(lo_mask)[0][-1])
    hi_idx = int(np.where(hi_mask)[0][0])

    def _interp_cross(i, arr_f, arr_m, target):
        if i == 0:
            return float(arr_f[0])
        r = (target - arr_m[i - 1]) / (arr_m[i] - arr_m[i - 1] + 1e-30)
        return float(arr_f[i - 1] + r * (arr_f[i] - arr_f[i - 1]))

    f_lo = _interp_cross(lo_idx + 1, fi, mi, half)
    f_hi = _interp_cross(hi_idx, fi, mi, half)
    bw = f_hi - f_lo
    zeta = bw / (2.0 * fn)
    Q = 1.0 / (2.0 * zeta) if zeta > 0 else None
    return {'fn': fn, 'Q': Q, 'zeta': zeta, 'f_lo': f_lo, 'f_hi': f_hi,
            'peak_mag': peak_mag, 'valid': True, 'method': 'half-power'}


def modal_sdof_fit(f, h1, fn0, zeta0, g2=None, fmin=None, fmax=None):
    """Ajuste SDOF complexo ao H1 medido, ponderado pela coerencia.

    Modelo: H(f) = fn2 / (fn2 - f2 + 2j*zeta*fn*f)
    Parametros: fn e zeta, ajustados por minimos quadrados nao-lineares.
    Devolve dict com fn, zeta, Q, r2 e o array H_model para sobreposicao.
    """
    from scipy.optimize import minimize

    f = np.asarray(f, dtype=float)
    h1 = np.asarray(h1, dtype=complex)
    mask = np.ones(len(f), dtype=bool)
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    if g2 is not None:
        mask &= np.asarray(g2) >= 0.5
    fw, hw = f[mask], h1[mask]
    wt = np.abs(g2)[mask] if g2 is not None else np.ones(len(fw))

    def _model(f_, fn_, z_):
        fn2 = fn_ ** 2
        return fn2 / (fn2 - f_ ** 2 + 2j * z_ * fn_ * f_)

    def _cost(p):
        fn_, z_ = p
        if fn_ <= 0 or z_ <= 0 or z_ >= 1:
            return 1e9
        res = hw - _model(fw, fn_, z_)
        return float(np.sum(wt * (res.real ** 2 + res.imag ** 2)))

    res = minimize(_cost, [fn0, max(zeta0, 0.01)],
                   method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000})
    fn_fit, zeta_fit = res.x
    H_model = _model(f, fn_fit, zeta_fit)
    # R² na magnitude
    mag_fit = np.abs(H_model[mask])
    mag_meas = np.abs(hw)
    ss_res = np.sum((mag_meas - mag_fit) ** 2)
    ss_tot = np.sum((mag_meas - mag_meas.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {'fn': fn_fit, 'zeta': zeta_fit, 'Q': 1 / (2 * zeta_fit),
            'r2': r2, 'H_model': H_model, 'valid': res.success or r2 > 0.8,
            'method': 'SDOF fit'}


# Valores de referencia da Formula (1) da ISO 2631-5 (rad/s, adimensional).
# Mantem-se aqui (e nao so em annexc) para uso no ajuste sem criar dependencia
# circular entre modulos.
_SPINE_TF_ZERO_DEFAULT = (34.0, 0.35)
_SPINE_TF_POLES_DEFAULT = [
    (31.0, 0.21), (230.0, 0.88), (260.0, 0.80),
    (320.0, 0.40), (380.0, 0.75), (420.0, 0.65),
]


def _spine_tf_model(f, zero, poles):
    """H(f) com a estrutura da Formula (1): 1 par de zeros e n pares de polos.

    zero  = (omega_rad_s, zeta)
    poles = [(omega_rad_s, zeta), ...]
    Ganho DC = 1 por construcao.
    """
    w = 2 * np.pi * np.asarray(f, dtype=float)
    s = 1j * w
    z_om, z_ze = zero
    num = 1 + 2 * z_ze * s / z_om + (s / z_om) ** 2
    den = np.ones_like(num)
    for p_om, p_ze in poles:
        den = den * (1 + 2 * p_ze * s / p_om + (s / p_om) ** 2)
    return num / den


def fit_spine_tf(f, h1, g2=None, n_poles=6, fmin=0.5, fmax=80.0,
                 zero_init=None, poles_init=None):
    """Ajusta a funcao de transferencia da Formula (1) (1 zero + n polos,
    ganho DC = 1) ao H1 medido por minimos quadrados no complexo.

    Pesos = coerencia g2 quando fornecida.
    Ponto de partida: valores da ISO 2631-5 (truncados a n_poles polos).

    Devolve dict com 'zero', 'poles', 'H_model', 'r2', 'valid'.
    """
    from scipy.optimize import minimize

    f = np.asarray(f, dtype=float)
    h1 = np.asarray(h1, dtype=complex)
    mask = (f >= fmin) & (f <= fmax)
    if g2 is not None:
        mask &= np.asarray(g2) >= 0.5
    if mask.sum() < 2 * (n_poles + 1) + 2:
        return {'valid': False,
                'H_model': _spine_tf_model(f, _SPINE_TF_ZERO_DEFAULT,
                                           _SPINE_TF_POLES_DEFAULT[:n_poles])}
    fw, hw = f[mask], h1[mask]
    wt = np.asarray(g2)[mask] if g2 is not None else np.ones(len(fw))

    if zero_init is None:
        zero_init = _SPINE_TF_ZERO_DEFAULT
    if poles_init is None:
        poles_init = list(_SPINE_TF_POLES_DEFAULT[:n_poles])
        while len(poles_init) < n_poles:
            last_om = poles_init[-1][0] if poles_init else 100.0
            poles_init.append((last_om * 1.5, 0.5))

    p0 = np.array([zero_init[0], zero_init[1]]
                  + [v for p in poles_init for v in p], dtype=float)

    def _cost(p):
        if p[0] <= 0 or not (0.001 < p[1] < 5.0):
            return 1e9
        for k in range(n_poles):
            if p[2 + 2 * k] <= 0 or not (0.001 < p[3 + 2 * k] < 5.0):
                return 1e9
        zero_p = (p[0], p[1])
        poles_p = [(p[2 + 2 * k], p[3 + 2 * k]) for k in range(n_poles)]
        H = _spine_tf_model(fw, zero_p, poles_p)
        res = hw - H
        return float(np.dot(wt, res.real ** 2 + res.imag ** 2))

    res = minimize(_cost, p0, method='Nelder-Mead',
                   options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 10000,
                             'adaptive': True})
    p = res.x
    zero_fit = (float(p[0]), float(p[1]))
    poles_fit = [(float(p[2 + 2 * k]), float(p[3 + 2 * k]))
                 for k in range(n_poles)]
    H_model = _spine_tf_model(f, zero_fit, poles_fit)
    mag_fit = np.abs(H_model[mask])
    mag_m = np.abs(hw)
    ss_res = float(np.sum((mag_m - mag_fit) ** 2))
    ss_tot = float(np.sum((mag_m - mag_m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'zero': zero_fit, 'poles': poles_fit,
            'H_model': H_model, 'r2': r2,
            'valid': r2 > 0.5 or res.success, 'n_poles': n_poles}


def make_spine_response_fn(zero, poles):
    """Cria uma funcao spine_response_fn(cond, dt) compativel com
    annexc.assess_from_segments(spine_response_fn=...).

    zero  = (omega_rad_s, zeta)
    poles = [(omega_rad_s, zeta), ...]
    """
    zero = tuple(zero)
    poles = [tuple(p) for p in poles]

    def _fn(cond, dt):
        n = len(cond)
        A = np.fft.rfft(cond)
        f = np.fft.rfftfreq(n, dt)
        H = _spine_tf_model(f, zero, poles)
        return np.fft.irfft(A * H, n)

    return _fn


def integration_hp_order_check(hp_order, n_integrations):
    """Verifica se a ordem do filtro passa-alta e suficiente para integracao
    no dominio do tempo sem deriva de baixa frequencia.

    Regra: ordem >= 2 para uma integracao (velocidade), >= 3 para dupla
    (deslocamento). Aplica-se ao filtro IIR passa-alta aplicado antes da
    integracao.

    Devolve dict com:
      'valid'     : True se a ordem e' suficiente
      'minimum'   : ordem minima recomendada para n_integrations
      'message'   : string descritiva
    """
    minimum = n_integrations + 1
    valid = hp_order >= minimum
    if valid:
        msg = (f'HP order {hp_order} is sufficient for {n_integrations}-fold '
               f'integration (minimum {minimum}).')
    else:
        msg = (f'HP order {hp_order} is too low for {n_integrations}-fold '
               f'integration: low-frequency drift will accumulate. '
               f'Minimum recommended order is {minimum}.')
    return {'valid': valid, 'minimum': minimum, 'message': msg}


def multiple_coherence_estimates(inputs, y, dt, window='hann',
                                  nperseg=1024, overlap=0.5):
    """Coerencia multipla e funções de transferencia MIMO.

    inputs : lista de arrays 1-D (sinais de entrada), todos com o mesmo
             comprimento que y.
    y      : array 1-D (saida).
    Devolve dict com:
      'f'          : array de frequencias (Hz)
      'gamma2_m'   : coerencia multipla (0–1)
      'gamma2_ord' : coerencias ordinarias de cada entrada × saida
      'H'          : lista de H_i (f) — TF de cada entrada para a saida
      'navg'       : numero de medias de Welch
    """
    from scipy.signal import csd as _csd, welch as _welch

    n_in = len(inputs)
    if n_in == 0 or len(y) == 0:
        return None

    step = max(1, int(nperseg * (1 - overlap)))
    kw = dict(fs=1.0 / dt, window=window, nperseg=nperseg,
              noverlap=nperseg - step, return_onesided=True,
              scaling='density', detrend='constant')

    # Espectros cruzados entre todas as entradas
    freqs = None
    Sxx = {}     # {(i,j): array} com i,j em 0..n_in-1
    Sxy = {}     # {i: array} entrada i × saida
    Syy = None

    for i in range(n_in):
        f_, Sii = _welch(inputs[i], **kw)
        if freqs is None:
            freqs = f_
        Sxx[(i, i)] = Sii
        for j in range(i + 1, n_in):
            _, Sij = _csd(inputs[i], inputs[j], **kw)
            Sxx[(i, j)] = Sij
            Sxx[(j, i)] = np.conj(Sij)
        _, Siy = _csd(inputs[i], y, **kw)
        Sxy[i] = Siy

    _, Syy = _welch(y, **kw)
    navg = max(1, len(y) // step)

    nf = len(freqs)
    # Coerencias ordinarias
    gamma2_ord = []
    for i in range(n_in):
        g2 = (np.abs(Sxy[i]) ** 2
              / np.maximum(Sxx[(i, i)] * Syy, 1e-300))
        gamma2_ord.append(np.clip(g2, 0.0, 1.0))

    # Coerencia multipla: gamma2_m = (s_xy^H S_xx^{-1} s_xy) / Syy
    gamma2_m = np.zeros(nf)
    H_list = [np.zeros(nf, dtype=complex) for _ in range(n_in)]
    for k in range(nf):
        # Monta S_xx(f_k) e s_xy(f_k)
        Sk = np.array([[Sxx[(i, j)][k] for j in range(n_in)]
                       for i in range(n_in)])
        sk = np.array([Sxy[i][k] for i in range(n_in)])
        try:
            h = np.linalg.solve(Sk, sk)
        except np.linalg.LinAlgError:
            h = np.zeros(n_in, dtype=complex)
        for i in range(n_in):
            H_list[i][k] = h[i]
        num = float(np.real(sk.conj() @ h))
        denom = float(np.real(Syy[k]))
        gamma2_m[k] = np.clip(num / denom if denom > 0 else 0.0, 0.0, 1.0)

    return {'f': freqs, 'gamma2_m': gamma2_m, 'gamma2_ord': gamma2_ord,
            'H': H_list, 'Syy': Syy, 'navg': navg}


def coherence_confidence_limit(navg, confidence=0.95):
    """Limite de confianca da coerencia: CL = 1 - (1-conf)^(1/(n-1)).

    Valores de gamma^2 abaixo de CL nao se distinguem de coerencia nula
    com a confianca dada (n = numero de segmentos medios). Devolve 1.0
    quando n < 2, onde nenhuma coerencia e informativa.
    """
    if navg < 2:
        return 1.0
    return 1.0 - (1.0 - confidence) ** (1.0 / (navg - 1))


def frf_estimates(gxx, gyy, gxy, navg):
    """Estimadores da funcao de resposta em frequencia a partir dos auto e
    cruzados espectros medios, com Gxy = E[conj(X) Y]:

      H1 = Gxy / Gxx   exato com ruido so na saida; com ruido na entrada
                       fica baixo (subestima o pico de ressonancia)
      H2 = Gyy / Gyx   exato com ruido so na entrada; com ruido na saida
                       fica alto
      Hv               minimos quadrados totais (ruido igual nos dois
                       canais): raiz de Gyx Hv^2 + (Gxx - Gyy) Hv - Gxy = 0
                       que da |Hv| entre |H1| e |H2|

    Como H1/H2 = gamma^2, os tres coincidem quando a coerencia e' 1, e o
    valor verdadeiro fica entre H1 e H2 quando ha ruido nos dois canais.
    Hv supoe ruido de mesma variancia em x e y, entao depende da escala dos
    dois sinais (aqui ambos em m/s2, o caso da transmissibilidade).

    eps e' o erro aleatorio normalizado de |H1| (Bendat e Piersol,
    Random Data, 4a ed., Tabela 9.6): sqrt(1 - gamma^2) / (|gamma|
    sqrt(2 n)), com n o numero de medias, e eps em radianos e' o desvio-
    padrao da fase. A formula e' de primeira ordem: num teste sintetico
    (1 GL, 117 a 233 medias, coerencia de 0,2 a 0,9) a banda |H1|
    (1 +- 2 eps) cobriu 87 a 91 % dos pontos, nao os 95 % de uma normal."""
    gxx = np.asarray(gxx, dtype=float)
    gyy = np.asarray(gyy, dtype=float)
    gxy = np.asarray(gxy, dtype=complex)
    z = np.zeros(len(gxx), dtype=complex)
    h1 = z.copy()
    np.divide(gxy, gxx, out=h1, where=gxx > 0)
    gyx = np.conj(gxy)
    nz = np.abs(gyx) > 0
    h2 = z.copy()
    np.divide(gyy, gyx, out=h2, where=nz)
    disc = np.sqrt((gxx - gyy) ** 2 + 4 * np.abs(gxy) ** 2)
    hv = z.copy()
    np.divide(gyy - gxx + disc, 2 * gyx, out=hv, where=nz)
    den = gxx * gyy
    g2 = np.clip(np.divide(np.abs(gxy) ** 2, den, out=np.zeros_like(den),
                           where=den > 0), 0.0, 1.0)
    eps = np.full(len(gxx), np.inf)
    if navg >= 1:
        np.divide(np.sqrt(1 - g2), np.sqrt(g2 * 2 * navg), out=eps,
                  where=g2 > 0)
    return {'h1': h1, 'h2': h2, 'hv': hv, 'g2': g2, 'eps': eps}


def transmissibility_estimates(x, y, dt, window='Hann', nperseg=8192,
                               overlap=0.5):
    """Como transmissibility_spectrum, mas devolve (f, frf_estimates,
    navg), com H1, H2, Hv, coerencia e erro aleatorio."""
    f, sxy, sxx, syy, navg = welch_cross_spectrum(
        x, y, dt, window, nperseg, overlap)
    return f, frf_estimates(sxx, syy, np.conj(sxy), navg), navg


def transmissibility_spectrum(x, y, dt, window='Hann', nperseg=8192,
                              overlap=0.5):
    """Estimador H1 da funcao de transmissibilidade de x (entrada) para y
    (saida): H1(f) = Gxy(f)/Gxx(f), com Gxy = E[conj(X) Y].

    welch_cross_spectrum devolve Sxy = E[X conj(Y)], entao Gxy = conj(Sxy)
    e H1 = conj(Sxy)/Sxx. Com essa convencao a fase de H1 e' positiva
    quando y ADIANTA x. Devolve (freq, H1 complexa, gamma2, navg). O H1 so
    e' confiavel onde gamma2 > coherence_confidence_limit(navg) e com ruido
    de medicao concentrado na saida.
    """
    f, sxy, sxx, syy, navg = welch_cross_spectrum(
        x, y, dt, window, nperseg, overlap)
    h = np.zeros(len(f), dtype=complex)
    np.divide(np.conj(sxy), sxx, out=h, where=sxx > 0)
    denom = sxx * syy
    g2 = np.divide(np.abs(sxy) ** 2, denom, out=np.zeros_like(denom),
                   where=denom > 0)
    return f, h, np.clip(g2, 0.0, 1.0), navg


ALIGN_MAX_LAG_S = 2.0        # a Secao 11.8 mediu -1,2 a +1,0 s entre unidades
ALIGN_WINDOW_S = 10.0        # janela do perfil de atraso; curta para que a
                             # deriva dentro de uma janela nao descorrelacione
ALIGN_DRIFT_TOL_S = 0.01     # variacao de atraso aceita dentro do registro
ALIGN_MIN_RHO = 0.3          # janelas com correlacao menor nao contam


class AlignmentError(ValueError):
    """Os dois registros nao podem ser alinhados por um atraso unico."""


def cross_correlation_lag(x, y, dt, max_lag=ALIGN_MAX_LAG_S):
    """Atraso de y em relacao a x pela correlacao cruzada: devolve (lag,
    rho) com y(t) ~ x(t - lag), lag em segundos (positivo: y atrasado) e rho
    o coeficiente de correlacao no pico. Correlacao linear (com zeros, sem
    o pico espelhado da correlacao circular), procurada so ate +-max_lag, e
    pico refinado por interpolacao parabolica, entao lag nao fica preso a
    multiplos de dt."""
    from scipy.signal import correlate
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if n < 3 or norm == 0:
        return 0.0, 0.0
    c = correlate(y, x, mode='full', method='fft') / norm   # indice n-1: lag 0
    m = min(int(round(max_lag / dt)), n - 2)
    seg = c[n - 1 - m:n + m]
    i = int(np.argmax(seg))
    frac = 0.0
    if 0 < i < len(seg) - 1:
        a, b, d = seg[i - 1], seg[i], seg[i + 1]
        den = a - 2 * b + d
        if den < 0:
            frac = 0.5 * (a - d) / den
    return (i - m + frac) * dt, float(seg[i])


def delay_profile(x, y, dt, max_lag=ALIGN_MAX_LAG_S, window_s=ALIGN_WINDOW_S):
    """Atraso em janelas consecutivas de window_s segundos: (centro de cada
    janela em s, lag, rho). Mostra se o atraso e' constante (relogios com a
    mesma taxa) ou deriva ao longo do registro."""
    n = min(len(x), len(y))
    w = int(round(window_s / dt))
    if w >= n or w < 3:
        lag, rho = cross_correlation_lag(x, y, dt, max_lag)
        return np.array([n * dt / 2]), np.array([lag]), np.array([rho])
    t, lags, rhos = [], [], []
    for s in range(0, n - w + 1, w):
        lag, rho = cross_correlation_lag(x[s:s + w], y[s:s + w], dt,
                                         min(max_lag, window_s / 4))
        t.append((s + w / 2) * dt)
        lags.append(lag)
        rhos.append(rho)
    return np.array(t), np.array(lags), np.array(rhos)


def align_by_correlation(x, y, dt, max_lag=ALIGN_MAX_LAG_S,
                         drift_tol=ALIGN_DRIFT_TOL_S, window_s=ALIGN_WINDOW_S):
    """Desloca y para coincidir com x pelo atraso da correlacao cruzada e
    corta os dois no trecho comum. Devolve (x_al, y_al, info) com info =
    {'lag', 'rho', 'drift', 'windows', 'shift'} (shift em amostras).

    Levanta AlignmentError, com o motivo, quando nao ha um atraso unico: a
    correlacao e' fraca demais (rho < ALIGN_MIN_RHO) ou o atraso varia mais
    que drift_tol entre janelas (relogios com taxas diferentes; um atraso so
    nao corrige isso, e coerencia e fase sairiam artefato).

    O deslocamento e' em amostras inteiras: o resto (ate meia amostra) so
    acrescenta fase linear ao H, sem mudar |H| nem a coerencia.

    O atraso usado e' a mediana das janelas do perfil (registros mais curtos
    que uma janela: a correlacao do registro inteiro). Correlacionar de uma
    vez um dia a 1 kHz pediria uma FFT de dezenas de milhoes de pontos."""
    t, lags, rhos = delay_profile(x, y, dt, max_lag, window_s)
    good = rhos >= ALIGN_MIN_RHO
    if not good.any():
        raise AlignmentError(
            f'cross-correlation peak {rhos.max():.2f} is below '
            f'{ALIGN_MIN_RHO}: the records do not share enough motion to '
            'measure a delay')
    lag = float(np.median(lags[good]))
    rho = float(np.median(rhos[good]))
    drift = float(np.ptp(lags[good])) if good.sum() > 1 else 0.0
    k = int(round(lag / dt))
    info = {'lag': lag, 'rho': rho, 'drift': drift,
            'windows': int(good.sum()), 'shift': k}
    if drift > drift_tol:
        raise AlignmentError(
            f'the delay drifts by {drift * 1000:.0f} ms along the record '
            f'({good.sum()} windows of {window_s:g} s, tolerance '
            f'{drift_tol * 1000:.0f} ms): the clocks run at different rates '
            'and one shift cannot align them. Use a shorter stretch.')
    x = np.asarray(x)
    y = np.asarray(y)
    if k > 0:                    # y atrasado: descarta o comeco de y
        y = y[k:]
    elif k < 0:
        x = x[-k:]
    n = min(len(x), len(y))
    return x[:n], y[:n], info


def highpass_butterworth(x, dt, fc=0.5, order=6):
    """Passa-alta Butterworth de fase zero (sosfiltfilt). A ordem efetiva
    e' o dobro de `order` (filtragem direta + reversa) e o ganho em fc e'
    -6 dB. Pre-processamento opcional, so para visualizacao/analise de
    transmissibilidade; nao altera a dose/risco da norma."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, dtype=float)
    nyq = 0.5 / dt
    if fc <= 0 or fc >= nyq or len(x) < 3 * (order + 1) * 2:
        return x
    sos = butter(order, fc / nyq, btype='highpass', output='sos')
    return sosfiltfilt(sos, x - x.mean())


def block_spectrogram(blocks, dt, window='Hann'):
    """Espectrograma de longo prazo: uma coluna por bloco (n_blocos, N).

    Devolve (f, S) com S[f, bloco] em amplitude, mesma normalizacao de
    core.spectrogram. O eixo do tempo e' decidido pelo chamador (hora do
    dia, indice do bloco), porque os blocos do registrador de campo sao
    intermitentes.
    """
    blocks = np.asarray(blocks, dtype=float)
    n = blocks.shape[1]
    w = _window(window, n)
    f = np.arange(n // 2 + 1) / (n * dt)
    X = np.fft.rfft((blocks - blocks.mean(axis=1, keepdims=True)) * w, axis=1)
    S = np.abs(X) * 2.0 / np.sum(w)
    S[:, 0] /= 2.0
    S[:, -1] /= 2.0
    return f, S.T


def block_cross_estimates(bx, by, dt, window='Hann'):
    """Como block_cross_spectra, mas devolve (f, frf_estimates, navg)."""
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    n = min(bx.shape[1], by.shape[1])
    m = min(bx.shape[0], by.shape[0])
    w = _window(window, n)
    f = np.arange(n // 2 + 1) / (n * dt)
    X = np.fft.rfft((bx[:m, :n] - bx[:m, :n].mean(1, keepdims=True)) * w, axis=1)
    Y = np.fft.rfft((by[:m, :n] - by[:m, :n].mean(1, keepdims=True)) * w, axis=1)
    return f, frf_estimates((np.abs(X) ** 2).mean(0), (np.abs(Y) ** 2).mean(0),
                            (np.conj(X) * Y).mean(0), m), m


def block_cross_spectra(bx, by, dt, window='Hann'):
    """H1, coerencia e numero de medias entre dois conjuntos de blocos
    sincronos (n_blocos, N), cada bloco sendo um segmento de Welch.

    Mesma convencao de transmissibility_spectrum: H1 = conj(Sxy)/Sxx e a
    fase de H1 e' positiva quando y adianta x. Devolve (f, H1, gamma2, navg).
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    n = min(bx.shape[1], by.shape[1])
    m = min(bx.shape[0], by.shape[0])
    w = _window(window, n)
    f = np.arange(n // 2 + 1) / (n * dt)
    X = np.fft.rfft((bx[:m, :n] - bx[:m, :n].mean(1, keepdims=True)) * w, axis=1)
    Y = np.fft.rfft((by[:m, :n] - by[:m, :n].mean(1, keepdims=True)) * w, axis=1)
    sxx = (np.abs(X) ** 2).mean(0)
    syy = (np.abs(Y) ** 2).mean(0)
    gxy = (np.conj(X) * Y).mean(0)
    h = np.zeros(len(f), dtype=complex)
    np.divide(gxy, sxx, out=h, where=sxx > 0)
    den = sxx * syy
    g2 = np.divide(np.abs(gxy) ** 2, den, out=np.zeros_like(den),
                   where=den > 0)
    return f, h, np.clip(g2, 0.0, 1.0), m


def signal_statistics(exc):
    """Estatisticas descritivas de um registro, para inspecao rapida.

    Fora do escopo da ISO 2631-5: sao metricas gerais de vibracao, uteis para
    triagem de sinais antes de montar um historico de exposicao.
    """
    a = exc.accel
    n = a.shape[0]
    rms = exc.rms
    peak = np.abs(a).max(0)
    crest = np.divide(peak, rms, out=np.zeros_like(peak), where=rms > 0)
    # Valor de dose de vibracao, definicao da ISO 2631-1
    vdv = (np.sum(a ** 4, axis=0) * exc.dt) ** 0.25
    # Curtose e assimetria
    var = a.var(0)
    dev = a - a.mean(0)
    kurt = np.divide((dev ** 4).mean(0), var ** 2,
                     out=np.full(3, np.nan), where=var > 0)
    skew = np.divide((dev ** 3).mean(0), var ** 1.5,
                     out=np.full(3, np.nan), where=var > 0)
    if exc.accel_spectrum is not None and exc.freq is not None:
        mag = np.abs(exc.accel_spectrum)
        dom = np.array([exc.freq[mag[:, c].argmax()] for c in range(3)])
    else:
        _prepare(exc)
        mag = np.abs(exc.accel_spectrum)
        dom = np.array([exc.freq[mag[:, c].argmax()] for c in range(3)])
    # Percentis de |a| por eixo
    abs_a = np.abs(a)
    abs_pcts = {p: np.percentile(abs_a, p, axis=0)
                for p in STAT_PERCENTILE_LEVELS}
    # Instante do maximo |a| por eixo
    t_peak = np.array([float(exc.time[abs_a[:, c].argmax()]) for c in range(3)])
    # Contagem de eventos (picos e cruzamentos) por limiar em |a_z|
    from scipy.signal import find_peaks as _find_peaks
    EVENT_THRESHOLDS_G = (0.5, 1.0, 2.0)
    _g = 9.80665
    az_abs = abs_a[:, 2]
    _pk_idx, _ = _find_peaks(az_abs)
    _pk_vals = az_abs[_pk_idx] if len(_pk_idx) else np.array([])
    event_counts = {}
    for tg in EVENT_THRESHOLDS_G:
        thr = tg * _g
        n_peaks = int((_pk_vals > thr).sum())
        above = az_abs > thr
        n_cross = int(np.sum(~above[:-1] & above[1:]))
        event_counts[tg] = {'peaks': n_peaks, 'crossings': n_cross}
    return {
        'samples': n,
        'duration': exc.duration,
        'sample_rate': exc.sample_rate,
        'rms': rms,
        'rms_total': float(np.sqrt((rms ** 2).sum())),
        'peak': peak,
        'crest': crest,
        'vdv': vdv,
        'kurtosis': kurt,
        'skewness': skew,
        'abs_pcts': abs_pcts,
        't_peak': t_peak,
        'event_counts': event_counts,
        'dominant': dom,
        'intensity_class': 1 if rms.max() < 0.65 else (3 if rms.max() > 1.35 else 2),
    }


# ----------------------------------------------------------------------
# Ponderacoes em frequencia da ISO 2631-1 (Anexo A daquela norma)
#
# IMPORTANTE: estas ponderacoes NAO entram no calculo de S_d^A nem de R^A.
# A ISO 2631-5 trabalha com aceleracao nao ponderada, porque a dependencia em
# frequencia da resposta ja esta dentro das funcoes de transferencia do modelo
# de elementos finitos. As ponderacoes servem para (a) a verificacao do
# criterio de validade de 9,81 m/s2 exigida pela propria 2631-5, e (b) analise
# comparativa com as metricas da 2631-1.
# ----------------------------------------------------------------------

INF = float('inf')

# Tabelas A.1 e A.2 da ISO 2631-1: f1, f2, f3, f4, Q4, f5, Q5, f6, Q6
WEIGHTINGS = {
    'Wk': dict(f1=0.4, f2=100.0, f3=12.5, f4=12.5, Q4=0.63,
               f5=2.37, Q5=0.91, f6=3.35, Q6=0.91,
               axis='z, seated', use='health, z direction'),
    'Wd': dict(f1=0.4, f2=100.0, f3=2.0, f4=2.0, Q4=0.63,
               f5=INF, Q5=None, f6=INF, Q6=None,
               axis='x, y', use='health, horizontal directions'),
    'Wc': dict(f1=0.4, f2=100.0, f3=8.0, f4=8.0, Q4=0.63,
               f5=INF, Q5=None, f6=INF, Q6=None,
               axis='x, backrest', use='comfort'),
    'We': dict(f1=0.4, f2=100.0, f3=1.0, f4=1.0, Q4=0.63,
               f5=INF, Q5=None, f6=INF, Q6=None,
               axis='rotational', use='comfort'),
    'Wj': dict(f1=0.4, f2=100.0, f3=INF, f4=INF, Q4=None,
               f5=3.75, Q5=0.91, f6=5.32, Q6=0.91,
               axis='z, recumbent head', use='comfort'),
    'Wf': dict(f1=0.08, f2=0.63, f3=INF, f4=0.25, Q4=0.86,
               f5=0.0625, Q5=0.80, f6=0.1, Q6=0.80,
               axis='z', use='motion sickness'),
    # Parametros da ISO 8041 adotados pela ISO 2631-4. A versao anterior usava
    # Q4 = 0,63 e Q5 = Q6 = 0,8, que nao correspondem a nenhuma das duas; o
    # Gera_Isofil.m do laboratorio estava correto e revelou o erro.
    'Wb': dict(f1=0.4, f2=100.0, f3=16.0, f4=16.0, Q4=0.55,
               f5=2.5, Q5=0.9, f6=4.0, Q6=0.95, K=1.024,
               axis='z, seated', use='comfort, rail vehicles (ISO 2631-4)'),
}

# Ponderacao recomendada por eixo para avaliacao de saude (8.2.1 da 2631-1)
HEALTH_WEIGHTING = {'x': ('Wd', 1.4), 'y': ('Wd', 1.4), 'z': ('Wk', 1.0)}

# Predefinicoes: nome -> {eixo: (ponderacao, fator k)}. Os fatores vem das
# tabelas da ISO 2631-1 e nao sao editaveis pelo usuario.
WEIGHTING_PRESETS = {
    'None - unweighted (ISO 2631-5 requirement)':
        {'x': (None, 1.0), 'y': (None, 1.0), 'z': (None, 1.0)},
    'ISO 2631-1 health, seat - Wd/Wd/Wk, k = 1.4/1.4/1.0':
        {'x': ('Wd', 1.4), 'y': ('Wd', 1.4), 'z': ('Wk', 1.0)},
    'ISO 2631-1 health, backrest (informative, 7.2.3) - Wc, k = 0.8/1.0/1.0':
        {'x': ('Wc', 0.8), 'y': (None, 1.0), 'z': (None, 1.0)},
    'ISO 2631-1 health, feet (informative, not in the norm) - k = 1.0':
        {'x': (None, 1.0), 'y': (None, 1.0), 'z': (None, 1.0)},
    'ISO 2631-1 comfort, seat pan - Wd/Wd/Wk, k = 1.0':
        {'x': ('Wd', 1.0), 'y': ('Wd', 1.0), 'z': ('Wk', 1.0)},
    'ISO 2631-1 comfort, backrest - Wc/Wd/Wd, k = 0.8/0.5/0.4':
        {'x': ('Wc', 0.8), 'y': ('Wd', 0.5), 'z': ('Wd', 0.4)},
    'ISO 2631-1 comfort, feet - Wk/Wk/Wk, k = 0.25/0.25/0.4':
        {'x': ('Wk', 0.25), 'y': ('Wk', 0.25), 'z': ('Wk', 0.4)},
    'ISO 2631-1 motion sickness - Wf on z':
        {'x': (None, 1.0), 'y': (None, 1.0), 'z': ('Wf', 1.0)},
    'ISO 2631-4 comfort, rail - Wd/Wd/Wb, k = 1.0':
        {'x': ('Wd', 1.0), 'y': ('Wd', 1.0), 'z': ('Wb', 1.0)},
}
DEFAULT_PRESET = 'None - unweighted (ISO 2631-5 requirement)'


def preset_for(config):
    """Nome da predefinicao que corresponde a config, ou None se for custom."""
    for name, cfg in WEIGHTING_PRESETS.items():
        if all(cfg[a] == tuple(config[a]) for a in AXIS_NAMES):
            return name
    return None

def sensor_warnings(config, fs, sensor_fc_low_hz=None):
    """Avisos de instrumentacao para uma configuracao de ponderacao.

    config         : dict eixo -> (ponderacao, k), como em WEIGHTING_PRESETS.
    fs             : taxa de amostragem efetiva (Hz).
    sensor_fc_low_hz: corte inferior do sensor (Hz), ou None se desconhecido.

    Devolve lista de strings de aviso (vazia = tudo OK).
    """
    warnings = []

    # Taxa minima para WBV (ISO 2631-1 recomenda >= 1000 Hz)
    if fs < 1000.0:
        warnings.append(
            f'Sample rate {fs:.0f} Hz is below the recommended minimum of '
            f'1000 Hz for whole-body vibration (ISO 2631-1 §4.3).'
        )

    # Alias: limite Nyquist vs. filtro de ponderacao
    f_nyq = fs / 2.0
    if f_nyq < LOWPASS_HZ:
        warnings.append(
            f'Nyquist frequency {f_nyq:.0f} Hz < {LOWPASS_HZ:.0f} Hz: '
            f'high-frequency content will be aliased before weighting.'
        )

    # Corte inferior da ponderacao vs capacidade do sensor
    for ax in AXIS_NAMES:
        wname, _ = config.get(ax, (None, 1.0))
        if wname is None:
            continue
        w = WEIGHTINGS.get(wname)
        if w is None:
            continue
        f1 = w['f1']   # corte inferior do filtro de ponderacao
        # Tempo de acomodacao estimado: 3 constantes de tempo = 3/(2*pi*f1)
        settle_s = 3.0 / (2 * np.pi * f1)
        if settle_s > 5.0:
            warnings.append(
                f'{wname} high-pass at {f1} Hz: settling time ~{settle_s:.0f} s '
                f'— discard the first {settle_s:.0f} s of each record.'
            )
        if sensor_fc_low_hz is not None and sensor_fc_low_hz > f1 * 0.9:
            warnings.append(
                f'{wname} high-pass at {f1} Hz but sensor low-frequency '
                f'limit is {sensor_fc_low_hz} Hz: the sensor rolls off '
                f'before the weighting filter — readings below '
                f'{sensor_fc_low_hz:.2g} Hz are unreliable.'
            )
    return warnings


def calibrate_from_reference(signal, dt, ref_accel, ref_freq,
                              ref_is_peak=True, search_bw=10.0):
    """Sensibilidade por calibrador de referencia (sinal sinusoidal conhecido).

    signal      : array 1-D (aceleracao como lida, antes de qualquer correcao)
    dt          : passo de tempo [s]
    ref_accel   : amplitude do calibrador [m/s2]
    ref_freq    : frequencia nominal do calibrador [Hz]
    ref_is_peak : True se ref_accel e' pico (0-to-peak); False se e' RMS
    search_bw   : janela de busca do pico em torno de ref_freq [Hz]

    Devolve dict:
      correction    : fator de correcao (multiplica o sinal lido para corrigir)
      freq_measured : frequencia do pico detectado [Hz]
      freq_error_pct: erro relativo a ref_freq [%]
      signal_rms    : RMS medido no pico [m/s2, sem correcao]
      ref_rms       : RMS esperado [m/s2]
      snr_db        : relacao sinal/ruido estimada [dB]
      valid         : bool
      message       : str descritivo
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n < 4:
        return {'valid': False, 'message': 'Signal too short.', 'correction': 1.0}

    # Espectro de amplitude unilateral (pico bilateral -> unilateral)
    A = np.fft.rfft(sig) * 2.0 / n
    f = np.fft.rfftfreq(n, dt)

    ref_rms = (ref_accel / 2.0 ** 0.5) if ref_is_peak else ref_accel

    # Janela de busca do pico
    mask = (f >= max(0.1, ref_freq - search_bw)) & (f <= ref_freq + search_bw)
    if not mask.any():
        return {'valid': False, 'message': f'No bins near {ref_freq} Hz.',
                'correction': 1.0}

    amp_band = np.abs(A[mask])
    pk = int(amp_band.argmax())
    freq_measured = float(f[mask][pk])
    # RMS do sinewave = amplitude de pico / sqrt(2)
    signal_rms = float(amp_band[pk]) / 2.0 ** 0.5

    # SNR: sinal vs ruido medio fora de ±2*search_bw
    noise_mask = (f < ref_freq - 2.0 * search_bw) | (f > ref_freq + 2.0 * search_bw)
    if noise_mask.any():
        noise_floor = float(np.sqrt(np.mean(np.abs(A[noise_mask]) ** 2)))
        snr_db = float(20.0 * np.log10(signal_rms / noise_floor)) \
            if noise_floor > 0 else 60.0
    else:
        snr_db = 40.0

    if signal_rms < 1e-12:
        return {'valid': False,
                'message': 'No detectable signal near the reference frequency.',
                'correction': 1.0}

    correction = ref_rms / signal_rms
    freq_error_pct = 100.0 * abs(freq_measured - ref_freq) / ref_freq

    valid = freq_error_pct < 10.0 and snr_db > 20.0
    if not valid:
        if freq_error_pct >= 10.0:
            msg = (f'Peak at {freq_measured:.1f} Hz is {freq_error_pct:.1f} % '
                   f'from reference {ref_freq} Hz — wrong signal or wrong frequency?')
        else:
            msg = (f'SNR {snr_db:.1f} dB is low — signal may not be a pure tone.')
    else:
        msg = (f'Peak at {freq_measured:.2f} Hz (err {freq_error_pct:.2f} %), '
               f'SNR {snr_db:.1f} dB. '
               f'Correction: {correction:.4f} ×.')

    return {
        'correction': correction,
        'freq_measured': freq_measured,
        'freq_error_pct': freq_error_pct,
        'signal_rms': signal_rms,
        'ref_rms': ref_rms,
        'snr_db': snr_db,
        'valid': valid,
        'message': msg,
    }


def calibrate_static_6pos(pos_readings, g=9.80665):
    """Offset e ganho por eixo a partir de medicoes estaticas em 6 posicoes.

    pos_readings : dict {'x+': v, 'x-': v, 'y+': v, 'y-': v, 'z+': v, 'z-': v}
                   com a aceleracao media [m/s2] em cada posicao (eixo +g e -g).
    g            : valor da gravidade local [m/s2].

    Devolve dict por eixo {'x': {'offset': ..., 'gain': ...}, 'y': ..., 'z': ...}

    offset = (pos + neg) / 2   [m/s2, subtrair do sinal bruto]
    gain   = (pos - neg) / (2*g) [adimensional, dividir o sinal corrigido de offset]
    """
    result = {}
    for ax in AXIS_NAMES:
        vp = float(pos_readings.get(f'{ax}+', g))
        vn = float(pos_readings.get(f'{ax}-', -g))
        offset = (vp + vn) / 2.0
        gain = (vp - vn) / (2.0 * g)
        result[ax] = {'offset': offset, 'gain': gain}
    return result


VALIDITY_PEAK_LIMIT = 9.81      # m/s2, criterio da ISO 2631-5 (Clausula 4 / A.2)


def band_limiting_response(freq, f1=0.4, f2=100.0):
    """So os dois filtros Butterworth de 2 polos (passa-alta em f1, passa-
    baixa em f2) que a ISO 2631-1 usa pra limitar banda -- SEM nenhuma
    ponderacao em cima. E' o que a Clausula 4 / A.2 da ISO 2631-5 manda
    aplicar antes de checar o limite de pico de 9,81 m/s2 (criterio de
    aplicabilidade dos Anexos A/E), bem diferente da curva Wk inteira
    (que soma mais 2 estagios de filtro em cima disso) -- usar Wk ali
    aplicaria uma atenuacao/formatacao que a norma nao pede nesse teste.
    """
    f = np.asarray(freq, dtype=float)
    s = 1j * 2 * np.pi * np.where(f == 0, 1e-12, f)
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    H = 1.0 / (1 + np.sqrt(2) * w1 / s + (w1 / s) ** 2)       # passa-alta
    H = H / (1 + np.sqrt(2) * s / w2 + (s / w2) ** 2)          # passa-baixa
    return np.where(f == 0, 0.0, H)


def weighting_response(name, freq):
    """Resposta complexa H(f) da ponderacao, incluindo os filtros de banda.

    Produto dos quatro fatores do Anexo A da ISO 2631-1: limitacao de banda
    (passa-alta e passa-baixa Butterworth de dois polos), transicao
    aceleracao-velocidade e degrau ascendente.
    """
    p = WEIGHTINGS[name]
    f = np.asarray(freq, dtype=float)
    s = 1j * 2 * np.pi * np.where(f == 0, 1e-12, f)
    w = lambda x: 2 * np.pi * x

    H = band_limiting_response(freq, p['f1'], p['f2'])

    if np.isfinite(p['f4']):                                   # transicao a-v
        w4 = w(p['f4'])
        num = 1.0 if not np.isfinite(p['f3']) else (1 + s / w(p['f3']))
        H = H * num / (1 + s / (p['Q4'] * w4) + (s / w4) ** 2)

    if np.isfinite(p['f5']) and np.isfinite(p['f6']):          # degrau
        w5, w6 = w(p['f5']), w(p['f6'])
        H = H * ((1 + s / (p['Q5'] * w5) + (s / w5) ** 2)
                 / (1 + s / (p['Q6'] * w6) + (s / w6) ** 2)) * (w5 / w6) ** 2

    H = H * p.get('K', 1.0)                                     # ganho da ISO 8041
    H = np.where(f == 0, 0.0, H)
    return H


def weighting_filter_zpk(name):
    """Zeros, polos (rad/s) e ganho em tempo continuo da ponderacao 'name'.

    Estrutura da ISO 2631-1, Anexo A:
      - passa-alta Butterworth de 2a ordem em f1  (zeros em s=0,0)
      - passa-baixa Butterworth de 2a ordem em f2
      - transicao aceleracao-velocidade em f4/Q4  (zero simples em f3, se finito)
      - degrau ascendente f5/Q5 (zeros) e f6/Q6 (polos), se presentes

    Retorna (zeros, poles, gain) todos em rad/s.
    """
    p = WEIGHTINGS[name]
    w = lambda x: 2.0 * np.pi * x
    sq2 = 2.0 ** 0.5

    w1, w2 = w(p['f1']), w(p['f2'])

    # HP 2nd-order Butterworth: dois zeros em 0; polos = -w1/√2 ± j·w1/√2
    zeros = [0.0 + 0j, 0.0 + 0j]
    poles = [-w1 / sq2 + 1j * w1 / sq2,
             -w1 / sq2 - 1j * w1 / sq2]
    gain = 1.0

    # LP 2nd-order Butterworth: sem zeros; polos = -w2/√2 ± j·w2/√2; ganho = w2^2
    poles += [-w2 / sq2 + 1j * w2 / sq2,
              -w2 / sq2 - 1j * w2 / sq2]
    gain *= w2 ** 2

    # Transicao aceleracao-velocidade
    if np.isfinite(p['f4']):
        w4 = w(p['f4'])
        ze4 = 1.0 / (2.0 * p['Q4'])
        wd4 = w4 * max(0.0, 1.0 - ze4 ** 2) ** 0.5
        poles += [-ze4 * w4 + 1j * wd4, -ze4 * w4 - 1j * wd4]
        if np.isfinite(p['f3']):
            w3 = w(p['f3'])
            zeros.append(-w3 + 0j)
            gain *= w4 ** 2 / w3
        else:
            gain *= w4 ** 2

    # Degrau (step up)
    f5 = p.get('f5', np.inf)
    f6 = p.get('f6', np.inf)
    if np.isfinite(f5) and np.isfinite(f6):
        w5, w6 = w(f5), w(f6)
        ze5 = 1.0 / (2.0 * p['Q5'])
        ze6 = 1.0 / (2.0 * p['Q6'])
        wd5 = w5 * max(0.0, 1.0 - ze5 ** 2) ** 0.5
        wd6 = w6 * max(0.0, 1.0 - ze6 ** 2) ** 0.5
        zeros += [-ze5 * w5 + 1j * wd5, -ze5 * w5 - 1j * wd5]
        poles += [-ze6 * w6 + 1j * wd6, -ze6 * w6 - 1j * wd6]
        # ganho do degrau = 1.0 (cancelado pelo fator (w5/w6)^2 do weighting_response)

    gain *= p.get('K', 1.0)
    return np.array(zeros), np.array(poles), float(gain)


def weighting_filter_sos(name, fs):
    """Coeficientes SOS digitais da ponderacao 'name' a fs Hz.

    Usa transformada bilinear padrao (scipy.signal.bilinear_zpk).
    Retorna ndarray (n_sections, 6): b0,b1,b2,a0,a1,a2 por secao (SOS).
    Adequado para sosfilt / sosfiltfilt do scipy.
    """
    from scipy.signal import bilinear_zpk, zpk2sos
    z, p, k = weighting_filter_zpk(name)
    zd, pd, kd = bilinear_zpk(z, p, k, fs=fs)
    return zpk2sos(zd, pd, kd)


def apply_weighting(signal, dt, name):
    """Filtra uma serie temporal com a ponderacao, no dominio da frequencia."""
    x = np.asarray(signal, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, dt)
    H = weighting_response(name, f)
    if x.ndim > 1:
        H = H[:, None]
    return np.fft.irfft(X * H, n, axis=0)


# ISO 2631-1:1997, 6.3.3: acima destas razoes o valor eficaz (metodo basico)
# subestima a exposicao e os metodos adicionais (MTVV, VDV) sao obrigatorios.
# O fator de crista 9 e' o limite do metodo basico (6.2.1).
MTVV_RATIO_LIMIT = 1.5
VDV_RATIO_LIMIT = 1.75
CREST_FACTOR_LIMIT = 9.0


STAT_PERCENTILE_LEVELS = (10, 50, 90, 95, 99)


def _running_rms_trace(y, dt, tau=1.0):
    """RMS corrente amostra a amostra (janela linear de tau s).
    Primeiras win-1 amostras: NaN (janela ainda incompleta)."""
    win = max(1, int(round(tau / dt)))
    if len(y) <= win:
        return np.full(len(y), float(np.sqrt((y ** 2).mean())))
    c = np.concatenate([[0.0], np.cumsum(y ** 2)])
    trace = np.empty(len(y))
    trace[:win - 1] = np.nan
    trace[win - 1:] = np.sqrt((c[win:] - c[:-win]) / win)
    return trace


def running_rms_max(y, dt, tau=1.0):
    """MTVV por eixo: maior RMS corrente com integracao linear numa janela
    de tau s (ISO 2631-1, Formula 2 e 6.3.2). y e' (n,) ou (n, eixos), ja
    ponderado. Registro mais curto que a janela: RMS do registro inteiro."""
    y = np.asarray(y, dtype=float)
    flat = y.ndim == 1
    y = y[:, None] if flat else y
    win = int(round(tau / dt))
    if win < 1 or len(y) == 0:
        out = np.zeros(y.shape[1])
    elif len(y) <= win:
        out = np.sqrt((y ** 2).mean(axis=0))
    else:
        c = np.concatenate([np.zeros((1, y.shape[1])),
                            np.cumsum(y ** 2, axis=0)])
        out = np.sqrt(np.maximum((c[win:] - c[:-win]).max(axis=0), 0) / win)
    return float(out[0]) if flat else out


def decision_ratios(aw, peak, vdv, mtvv, duration):
    """Razoes da ISO 2631-1 (6.2.1 e 6.3.3) por eixo, a partir do sinal
    ponderado: fator de crista, MTVV/a_w e VDV/(a_w T^1/4), com T a duracao
    medida em segundos. O fator k de cada eixo se cancela nas tres. Seno
    puro da' 1,41, 1,00 e 1,11; um trem de choques passa dos limites."""
    aw, peak, vdv, mtvv = (np.atleast_1d(np.asarray(v, dtype=float))
                           for v in (aw, peak, vdv, mtvv))
    ok = aw > 0
    nan = np.full(aw.shape, np.nan)
    crest = np.divide(peak, aw, out=nan.copy(), where=ok)
    r_mtvv = np.divide(mtvv, aw, out=nan.copy(), where=ok)
    r_vdv = np.divide(vdv, aw * duration ** 0.25, out=nan.copy(),
                      where=ok & (duration > 0))
    additional = (np.nan_to_num(r_mtvv) > MTVV_RATIO_LIMIT) | \
                 (np.nan_to_num(r_vdv) > VDV_RATIO_LIMIT)
    return {'crest': crest, 'mtvv_ratio': r_mtvv, 'vdv_ratio': r_vdv,
            'additional_required': additional,
            'crest_above': np.nan_to_num(crest) > CREST_FACTOR_LIMIT}


def weighted_metrics(exc, config=None):
    """Metricas ponderadas por eixo: a_w, pico, fator de crista, VDV, MSDV.

    config e um dicionario {eixo: (ponderacao ou None, fator k)}. Sem config,
    usa a recomendacao de saude da ISO 2631-1.

    MSDV (Motion Sickness Dose Value, ISO 2631-1 Anexo D) = a_w,rms * sqrt(T),
    unidade m/s^1.5 -- so' faz sentido fisico com a ponderacao Wf no eixo z
    (preset 'ISO 2631-1 motion sickness'), mas e' reportado por eixo como no
    modulo Human Vibration da Dewesoft, que o trata como canal de saida
    generico ao lado de RMS/VDV independente do filtro escolhido.
    """
    config = config or HEALTH_WEIGHTING
    duration = exc.duration
    out = {'weighting': [], 'k': [], 'aw': [], 'peak': [], 'crest': [], 'vdv': [],
           'msdv': [], 'mtvv': []}
    yw = []
    for c, ax in enumerate(AXIS_NAMES):
        name, k = config[ax]
        y = exc.accel[:, c] * k if name is None else \
            apply_weighting(exc.accel[:, c], exc.dt, name) * k
        yw.append(y)
        aw = float(np.sqrt((y ** 2).mean()))
        pk = float(np.abs(y).max())
        out['weighting'].append(name or '-')
        out['k'].append(k)
        out['aw'].append(aw)
        out['peak'].append(pk)
        out['crest'].append(pk / aw if aw > 0 else 0.0)
        out['vdv'].append(float((np.sum(y ** 4) * exc.dt) ** 0.25))
        out['msdv'].append(aw * np.sqrt(duration))
        out['mtvv'].append(running_rms_max(y, exc.dt))
    out['ratios'] = decision_ratios(out['aw'], out['peak'], out['vdv'],
                                    out['mtvv'], duration)
    out['aw_total'] = float(np.sqrt(sum(a ** 2 for a in out['aw'])))
    out['vdv_total'] = float(sum(v ** 4 for v in out['vdv']) ** 0.25)
    out['msdv_total'] = float(np.sqrt(sum(m ** 2 for m in out['msdv'])))
    # Percentis do RMS movel ponderado (janela 1 s) por eixo
    rms_pcts = {p: [] for p in STAT_PERCENTILE_LEVELS}
    for y in yw:
        trace = _running_rms_trace(y, exc.dt)
        finite = trace[np.isfinite(trace)]
        for p in STAT_PERCENTILE_LEVELS:
            rms_pcts[p].append(
                float(np.percentile(finite, p)) if len(finite) else 0.0)
    out['rms_pcts'] = rms_pcts
    out['config'] = {a: tuple(config[a]) for a in AXIS_NAMES}
    out['preset'] = preset_for(config)
    return out


def _apply_band_limiting(signal, dt):
    """Igual a apply_weighting(), mas so com os filtros de banda 0,4-100 Hz
    (Butterworth de 2 polos cada), sem nenhum estagio de ponderacao em
    cima -- o que validity_check() precisa, ver band_limiting_response()."""
    x = np.asarray(signal, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, dt)
    H = band_limiting_response(f)
    if x.ndim > 1:
        H = H[:, None]
    return np.fft.irfft(X * H, n, axis=0)


def validity_check(excitations):
    """Criterio de validade dos Anexos A e E da ISO 2631-5.

    A norma manda verificar o pico de aceleracao na direcao z, na pessoa,
    depois de aplicados os filtros limitadores de banda da ISO 2631-1 (dois
    Butterworth de 2 polos, 0,4 Hz e 100 Hz -- Clausula 4 / A.2). Acima de
    9,81 m/s2 os Anexos A e E nao se aplicam: valem a Clausula 5 e os
    Anexos C e D, que tratam de exposicoes com perda de contato.

    IMPORTANTE: isso NAO e' a curva Wk (que tem mais 2 estagios de filtro
    -- transicao aceleracao-velocidade e degrau ascendente -- em cima da
    limitacao de banda). Usar Wk aqui, como uma versao anterior deste
    codigo fazia, aplica atenuacao/formatacao que a norma nao pede nesse
    teste especifico, podendo mudar a decisao de roteamento Anexo A/E vs.
    Clausula 5/Anexo C.
    """
    peaks = {}
    peaks_xyz = {}  # per-axis band-limited peaks for all 3 axes
    for name, exc in excitations.items():
        if not np.any(exc.accel):
            continue
        z = _apply_band_limiting(exc.accel[:, 2], exc.dt)
        peaks[name] = float(np.abs(z).max())
        n_ax = min(3, exc.accel.shape[1])
        peaks_xyz[name] = [
            float(np.abs(_apply_band_limiting(exc.accel[:, ai], exc.dt)).max())
            for ai in range(n_ax)
        ]
    worst = max(peaks.values()) if peaks else 0.0
    return {
        'peaks': peaks,
        'peaks_xyz': peaks_xyz,
        'worst': worst,
        'limit': VALIDITY_PEAK_LIMIT,
        'valid': worst <= VALIDITY_PEAK_LIMIT,
    }


# ----------------------------------------------------------------------
# Consistencia da base de tempo entre pontos de contato
# ----------------------------------------------------------------------

TIME_BASE_TOLERANCE = 0.01      # 1% de diferenca de duracao
SAMPLE_RATE_TOLERANCE = 0.001   # 0,1% de diferenca de taxa


class TimeBaseError(ValueError):
    """Os registros nao podem ser combinados na Formula (A.1)."""


def check_time_base(excitations, tolerance=TIME_BASE_TOLERANCE):
    """Verifica se os registros dos pontos de contato sao combinaveis.

    A Formula (A.1) soma as contribuicoes espectro a espectro, o que exige
    mesma grade de frequencia: mesma taxa de amostragem e mesmo numero de
    amostras. Sensores independentes derivam entre si, entao o desvio e
    medido e reportado.

    Devolve um relatorio. Nao trunca nada: quem trunca e align_excitations.
    """
    live = {n: e for n, e in excitations.items() if e.accel.shape[0] > 0}
    rates = {n: e.sample_rate for n, e in live.items()}
    lengths = {n: e.accel.shape[0] for n, e in live.items()}
    durations = {n: e.duration for n, e in live.items()}

    rate_spread = 0.0
    if rates:
        lo, hi = min(rates.values()), max(rates.values())
        rate_spread = (hi - lo) / lo if lo > 0 else 0.0

    drift = 0.0
    if durations:
        lo, hi = min(durations.values()), max(durations.values())
        drift = (hi - lo) / lo if lo > 0 else 0.0

    return {
        'rates': rates, 'lengths': lengths, 'durations': durations,
        'rate_spread': rate_spread,
        'drift': drift,
        'samples_dropped': max(lengths.values()) - min(lengths.values())
        if lengths else 0,
        'aligned': drift == 0 and rate_spread == 0,
        'rate_mismatch': rate_spread > SAMPLE_RATE_TOLERANCE,
        'exceeds_tolerance': drift > tolerance,
        'tolerance': tolerance,
    }


def align_excitations(excitations, tolerance=TIME_BASE_TOLERANCE):
    """Trunca os registros ao menor comprimento comum.

    Devolve (lista alinhada, relatorio). Levanta TimeBaseError quando as taxas
    de amostragem diferem ou quando a deriva excede a tolerancia: nesses casos
    truncar mascararia um problema de aquisicao em vez de resolve-lo.
    """
    report = check_time_base(excitations, tolerance)

    if report['rate_mismatch']:
        detail = ', '.join(f'{n}: {r:.4g} Hz' for n, r in report['rates'].items())
        raise TimeBaseError(
            f'The contact points were sampled at different rates ({detail}). '
            f'Formula (A.1) sums the spectra bin by bin, which requires one '
            f'common frequency grid. Resample the records before analysing.')

    if report['exceeds_tolerance']:
        detail = ', '.join(f'{n}: {d:.3f} s' for n, d in report['durations'].items())
        raise TimeBaseError(
            f'The records differ in length by {report["drift"] * 100:.1f}%, '
            f'above the {tolerance * 100:g}% tolerance ({detail}). A drift '
            f'this large points to a data acquisition fault rather than to '
            f'clock drift between sensors, so the records were not truncated.')

    if report['aligned']:
        return list(excitations.values()), report

    n = min(report['lengths'].values())
    out = []
    for name, exc in excitations.items():
        if exc.accel.shape[0] == n:
            out.append(exc)
            continue
        trimmed = Excitation(exc.name, exc.time[:n], exc.accel[:n], exc.dt)
        _prepare(trimmed)
        out.append(trimmed)
    return out, report


BACKREST_ZERO_WARNING = (
    "Backrest entered as 'Zero': a null signal is used. "
    "ISO 2631-5:2018 A.3.1 (p. 11) states that an unmeasured backrest "
    "can be substituted with the seat-surface signal; the official "
    "executable makes this substitution automatically. "
    "Results with a null backrest may be lower than the reference "
    "(e.g. −21 % at T12/L1 in a test case)."
)


def read_exposure_files(paths, progress=None):
    """Le os 4 arquivos de uma exposicao (seat, feet, hands, back, na ordem
    de LOCATION_NAMES) e alinha as bases de tempo. 'Zero'/'Null'/'None'/''
    geram excitacao nula.

    Devolve (lista alinhada, relatorio de alinhamento, lista de avisos).
    A lista de avisos contem BACKREST_ZERO_WARNING quando o encosto for 'Zero'.

    Reune a leitura que analysis.Runner._exposure faz antes de chamar
    analyse_exposure; a analise de sensibilidade a reusa para reler os
    arquivos originais, porque o espectro de uma exposicao ja processada e
    liberado da memoria em historicos longos (Secao 9.3 da documentacao).
    """
    import warnings as _warnings
    excs, base = [], None
    _warnings_out = []
    for i, path in enumerate(paths):
        if str(path).strip().lower() in ('zero', 'null', 'none', ''):
            excs.append(None)
            # Aviso apenas para o encosto (indice 3); maos e pes Zero sao normais.
            if i == 3:
                _warnings_out.append(BACKREST_ZERO_WARNING)
                _warnings.warn(BACKREST_ZERO_WARNING, stacklevel=2)
            continue
        if progress:
            progress(None, f'reading {LOCATION_NAMES[i]}: {Path(path).name}')
        e = read_excitation(path)
        base = base or e
        excs.append(e)
    if base is None:
        raise ValueError('No acceleration file given: the seat must be measured.')
    excs = [e if e is not None else zero_excitation(base) for e in excs]
    named = {LOCATION_NAMES[i]: e for i, e in enumerate(excs)}
    aligned, drift = align_excitations(named)
    return aligned, drift, _warnings_out
