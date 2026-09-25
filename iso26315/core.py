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


def read_excitation(path, name=None):
    """Le arquivo ASCII de 4 colunas: tempo, ax, ay, az (A.2)."""
    path = Path(path)
    data = np.loadtxt(path, usecols=(0, 1, 2, 3))
    if data.shape[1] > data.shape[0]:
        data = data.T
    t = data[:, 0]
    dt = (t[-1] - t[0]) / (len(t) - 1)
    if abs((t[1] - t[0]) - dt) > 0.02 * dt:
        raise ValueError(
            f'{path.name}: a base de tempo nao parece uniformemente espacada.')
    exc = Excitation(name or path.stem, t, data[:, 1:4], dt)
    _prepare(exc)
    return exc


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
# Percentis de massa corporal (kg) por classe de IMC -- Tabela 2 de Seidel,
# Hinz, Hofmann, Menzel (2008) "Intraspinal forces and health risk caused by
# whole-body vibration", Int J Ind Ergon 38:856-867 (referencia [57] da
# ISO 2631-5:2018), N=1163 motoristas europeus. Confirma que o percentil de
# massa e calculado DENTRO de cada classe de IMC -- as duas escalas se
# sobrepoem (P05 da classe 2 e 78 kg, abaixo do P75 da classe 1, 82 kg) -- e
# nao sobre uma unica escala global como a versao anterior deste modulo
# assumia. Ver ANTHROPOMETRY_NOTE.
MASS_PERCENTILES_BY_BMI_CLASS = {
    1: [63.0, 70.0, 75.0, 82.0, 90.0],    # BMI <= 2,61224 g/cm2 (26,1 kg/m2)
    2: [78.0, 85.0, 90.0, 100.0, 116.0],  # BMI > 2,61224 g/cm2
}
PERCENTILE_NAMES = ['P05', 'P25', 'P50', 'P75', 'P95']

ANTHROPOMETRY_NOTE = """\
O indice de 30 posicoes decompoe-se em (bloco-1)*15 + (percentil-1)*3 + \
subclasse. O bloco e a classe de IMC (limiar 26,1 kg/m2, confirmado pelo \
texto da norma, Anexo A.3.1 a); o percentil de massa e calculado dentro da \
classe de IMC ja escolhida, pela Tabela 2 de Seidel, Hinz, Hofmann, Menzel \
(2008), Int J Ind Ergon 38:856-867 -- os 1163 motoristas europeus que deram \
origem as funcoes de transferencia do Anexo A. Essa fonte resolve a \
ambiguidade que uma versao anterior deste modulo so conseguia contornar com \
um ajuste empirico e um bloco de IMC forcado: com a tabela real, 70 kg a \
1,85 m (IMC 20,5) cai corretamente na classe 1 -- o ajuste anterior, sem a \
tabela, precisava forcar esse caso para a classe 2, o que nao fazia sentido \
como leitura de IMC. Confirmado contra os seis casos de referencia \
(indiv1-indiv6) da propria norma via tests/anthropometry_check.py."""


def anthropometry(body_mass, body_height, subclass=1):
    """(indice 1..30 da funcao de transferencia, indice 1..10 da forca estatica).

    Ver ANTHROPOMETRY_NOTE quanto ao grau de confianca deste mapeamento.
    """
    bmi = body_mass / body_height ** 2
    bmi_class = 2 if bmi > BMI_THRESHOLD else 1
    edges = MASS_PERCENTILES_BY_BMI_CLASS[bmi_class][:-1]
    percentile = int(np.searchsorted(edges, body_mass)) + 1
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
    """Resistencia ultima da vertebra, MPa.

    equation=2 e a Formula (E.2) da norma; equation=1 usa os coeficientes
    alternativos do Anexo C. protection=50 protege 50% da populacao.
    """
    offset = 0.0 if protection == 50 else _protection_offset(protection, equation)
    if equation == 1:
        return 5.106713 - 0.037747 * np.asarray(age) - offset
    return 6.765024 - 0.067184 * np.asarray(age) - offset


def _protection_offset(protection, equation):
    from statistics import NormalDist
    sigma = 1.0 if equation == 2 else 0.6
    return -NormalDist().inv_cdf(protection / 100.0) * sigma


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


def band_rms(f, psd, edges):
    """Valor eficaz por banda, integrando a densidade espectral de potencia."""
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = np.sqrt(np.trapezoid(psd[m], f[m])) if m.sum() > 1 \
                else np.sqrt(psd[m][0] * (edges[i + 1] - edges[i]))
    return out


def third_octave_rms(x, dt, window='Hann', nperseg=8192, overlap=0.5):
    f, psd, navg = welch_spectrum(x, dt, window, nperseg, overlap, kind='psd')
    centers = THIRD_OCTAVE_CENTERS[THIRD_OCTAVE_CENTERS <= min(LOWPASS_HZ,
                                                               0.5 / dt)]
    edges = np.concatenate([centers / 2 ** (1 / 6), [centers[-1] * 2 ** (1 / 6)]])
    return centers, band_rms(f, psd, edges), navg


def analysis_spectrum(exc, method, axis=None, window='Hann', nperseg=8192,
                      overlap=0.5, band_width=1.0, weighting=None):
    """Espectro para visualizacao. Devolve (f, y, rotulo do eixo y, nota).

    weighting aplica uma ponderacao da ISO 2631-1 ao espectro exibido. Nao
    altera nada do calculo de dose ou de fator de risco.
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
        return (exc.freq, y, 'Acceleration amplitude [m/s$^2$]',
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
        return f, acc, 'Acceleration amplitude [m/s$^2$]', tail
    if method == SPECTRUM_METHODS[2]:
        return f, acc, 'PSD [(m/s$^2$)$^2$/Hz]', tail
    if method == SPECTRUM_METHODS[3]:
        edges = np.arange(0, min(LOWPASS_HZ, f[-1]) + band_width, band_width)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return (centers, band_rms(f, acc, edges), 'Band RMS [m/s$^2$]',
                tail + f' Integrated into {band_width:g} Hz bands.')

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
    return (centers, total, 'Band RMS [m/s$^2$]',
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
    # Curtose: 3 para ruido gaussiano; acima disso indica conteudo de choque
    var = a.var(0)
    kurt = np.divide(((a - a.mean(0)) ** 4).mean(0), var ** 2,
                     out=np.full(3, np.nan), where=var > 0)
    mag = np.abs(exc.accel_spectrum)
    dom = np.array([exc.freq[mag[:, c].argmax()] for c in range(3)])
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

VALIDITY_PEAK_LIMIT = 9.81      # m/s2, criterio da ISO 2631-5 (6.2)


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

    w1, w2 = w(p['f1']), w(p['f2'])
    H = 1.0 / (1 + np.sqrt(2) * w1 / s + (w1 / s) ** 2)       # passa-alta
    H = H / (1 + np.sqrt(2) * s / w2 + (s / w2) ** 2)          # passa-baixa

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


def weighted_metrics(exc, config=None):
    """Metricas ponderadas por eixo: a_w, pico, fator de crista, VDV.

    config e um dicionario {eixo: (ponderacao ou None, fator k)}. Sem config,
    usa a recomendacao de saude da ISO 2631-1.
    """
    config = config or HEALTH_WEIGHTING
    out = {'weighting': [], 'k': [], 'aw': [], 'peak': [], 'crest': [], 'vdv': []}
    for c, ax in enumerate(AXIS_NAMES):
        name, k = config[ax]
        y = exc.accel[:, c] * k if name is None else \
            apply_weighting(exc.accel[:, c], exc.dt, name) * k
        aw = float(np.sqrt((y ** 2).mean()))
        pk = float(np.abs(y).max())
        out['weighting'].append(name or '-')
        out['k'].append(k)
        out['aw'].append(aw)
        out['peak'].append(pk)
        out['crest'].append(pk / aw if aw > 0 else 0.0)
        out['vdv'].append(float((np.sum(y ** 4) * exc.dt) ** 0.25))
    out['aw_total'] = float(np.sqrt(sum(a ** 2 for a in out['aw'])))
    out['vdv_total'] = float(sum(v ** 4 for v in out['vdv']) ** 0.25)
    out['config'] = {a: tuple(config[a]) for a in AXIS_NAMES}
    out['preset'] = preset_for(config)
    return out


def validity_check(excitations):
    """Criterio de validade dos Anexos A e E da ISO 2631-5.

    A norma manda verificar o pico de aceleracao na direcao z, na pessoa,
    depois de aplicados os filtros limitadores de banda da ISO 2631-1. Acima
    de 9,81 m/s2 os Anexos A e E nao se aplicam: valem a Clausula 5 e os
    Anexos C e D, que tratam de exposicoes com perda de contato.
    """
    peaks = {}
    for name, exc in excitations.items():
        if not np.any(exc.accel):
            continue
        z = apply_weighting(exc.accel[:, 2], exc.dt, 'Wk')
        peaks[name] = float(np.abs(z).max())
    worst = max(peaks.values()) if peaks else 0.0
    return {
        'peaks': peaks,
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


def read_exposure_files(paths, progress=None):
    """Le os 4 arquivos de uma exposicao (seat, feet, hands, back, na ordem
    de LOCATION_NAMES) e alinha as bases de tempo. 'Zero'/'Null'/'None'/''
    geram excitacao nula. Devolve (lista alinhada, relatorio de alinhamento).

    Reune a leitura que analysis.Runner._exposure faz antes de chamar
    analyse_exposure; a analise de sensibilidade a reusa para reler os
    arquivos originais, porque o espectro de uma exposicao ja processada e
    liberado da memoria em historicos longos (Secao 9.3 da documentacao).
    """
    excs, base = [], None
    for i, path in enumerate(paths):
        if str(path).strip().lower() in ('zero', 'null', 'none', ''):
            excs.append(None)
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
    return align_excitations(named)
