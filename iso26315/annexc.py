"""Clausula 5 e Anexo C da ISO 2631-5: exposicoes com perda de contato.

Este modulo e deliberadamente independente de core.py. As duas cadeias nao
compartilham condicionamento de sinal, porque os requisitos conflitam:

                        Clausula 5 / Anexo C      Anexo A / Anexo E
  passa-alta            0,1 Hz, 2 polos           0,5 Hz
  passa-baixa           80 Hz, 4 polos            100 Hz
  filtros da 2631-1     proibidos                 usados so na validacao
  direcoes              so z, no assento          x, y, z em 4 pontos
  excitacao             aceleracao                deslocamento

O corte inferior em 0,1 Hz existe para preservar a queda livre antes do
impacto: filtrar em 0,5 Hz empurraria o -1 g de volta a zero antes do choque
e erraria o pico em ate 1 g.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

# Formula (1): um zero complexo e seis polos complexos
ZERO = (34.0, 0.35)                     # omega1, zeta1
POLES = [(31.0, 0.21), (230.0, 0.88), (260.0, 0.80),
         (320.0, 0.40), (380.0, 0.75), (420.0, 0.65)]

DOSE_FACTOR = 1.07                      # Formula (3), resposta de amplitude
HIGHPASS_HZ, HIGHPASS_ORDER = 0.1, 2    # 5.1.3
LOWPASS_HZ, LOWPASS_ORDER = 80.0, 4
REFERENCE_DAILY_TIME = 8 * 3600.0       # td recomendado na Formula (4)

# Anexo C
SEX_DEFAULTS = {
    'male': dict(mass_fraction=0.49, endplate_area=14.0, s_age=0.052,
                 alpha=1.613, beta=2.799, ref_mass=82.0),
    'female': dict(mass_fraction=0.45, endplate_area=11.6, s_age=0.039,
                   alpha=0.959, beta=3.709, ref_mass=64.0),
}
ULTIMATE_BASE = 6.75                    # MPa, Formula (C.4)
GRAVITY = 9.81
RISK_LOW, RISK_HIGH = 0.10, 0.50        # limiares de Pi no Anexo C

# Deteccao de perda de contato: heuristica, NAO esta na norma
FREEFALL_THRESHOLD = -7.0               # m/s2, cerca de 0,7 g para baixo
FREEFALL_MIN_DURATION = 0.15            # s


def transfer_function(freq):
    """H(omega) da Formula (1): assento para coluna lombar, adimensional."""
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    s = 1j * w
    num = 1 + 2 * ZERO[1] * s / ZERO[0] + (s / ZERO[0]) ** 2
    den = np.ones_like(num)
    for wn, zn in POLES:
        den = den * (1 + 2 * zn * s / wn + (s / wn) ** 2)
    return num / den


def condition(az, dt, taper_seconds=0.0, remove_offset=True):
    """Condicionamento de sinal de 5.1.3, nesta ordem.

    Correcao de offset, janela de cosseno opcional nas pontas, passa-alta
    Butterworth de 2 polos em 0,1 Hz e passa-baixa de 4 polos em 80 Hz.
    Os filtros sao causais, como na implementacao digital do Anexo D.
    """
    x = np.asarray(az, dtype=float).copy()
    if remove_offset:
        x = x - x.mean()
    if taper_seconds > 0:
        n = min(int(taper_seconds / dt), len(x) // 2)
        if n > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]
    fs = 1.0 / dt
    sos_hp = butter(HIGHPASS_ORDER, HIGHPASS_HZ, 'highpass', fs=fs, output='sos')
    sos_lp = butter(LOWPASS_ORDER, LOWPASS_HZ, 'lowpass', fs=fs, output='sos')
    return sosfilt(sos_lp, sosfilt(sos_hp, x))


def spine_response(az, dt):
    """Formulas (2): resposta da coluna no tempo, a partir do sinal condicionado."""
    n = len(az)
    A = np.fft.rfft(az)
    f = np.fft.rfftfreq(n, dt)
    return np.fft.irfft(A * transfer_function(f), n)


def positive_peaks(response):
    """Maximos entre cruzamentos por zero consecutivos; so picos positivos."""
    s = np.sign(response)
    s[s == 0] = 1.0
    cuts = np.flatnonzero(np.diff(s)) + 1
    bounds = np.concatenate(([0], cuts, [len(response)]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = response[a:b]
        if seg.size and seg[0] > 0:
            out.append(seg.max())
    return np.asarray(out)


def acceleration_dose(peaks):
    """Formula (3): D_z = 1,07 (soma dos picos elevados a sexta)^(1/6)."""
    if not len(peaks):
        return 0.0
    return DOSE_FACTOR * float(np.sum(np.asarray(peaks) ** 6) ** (1 / 6))


def daily_dose(doses, daily_times, measured_times):
    """Formula (5): dose diaria a partir de j condicoes medidas."""
    d = np.asarray(doses, dtype=float)
    td = np.asarray(daily_times, dtype=float)
    tm = np.asarray(measured_times, dtype=float)
    return float(np.sum(d ** 6 * td / tm) ** (1 / 6))


def detect_loss_of_contact(az, dt, threshold=FREEFALL_THRESHOLD,
                           min_duration=FREEFALL_MIN_DURATION):
    """Intervalos suspeitos de queda livre, em indices de amostra.

    HEURISTICA, nao prevista na norma. A ISO 2631-5 manda detectar a perda de
    contato por chave de contato, video ou metodo equivalente, e excluir esses
    trechos da exposicao. Sem esse instrumento, marcam-se os trechos em que a
    aceleracao vertical permanece abaixo do limiar por tempo suficiente, que e
    a assinatura de queda livre. Convem conferir os trechos antes de aceitar.
    """
    below = np.asarray(az, dtype=float) < threshold
    if not below.any():
        return []
    edges = np.diff(below.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if below[0]:
        starts.insert(0, 0)
    if below[-1]:
        ends.append(len(below))
    least = int(min_duration / dt)
    return [(a, b) for a, b in zip(starts, ends) if b - a >= least]


def split_contact_segments(az, dt, intervals):
    """Separa o sinal nos trechos com contato, descartando os intervalos dados."""
    if not intervals:
        return [np.asarray(az, dtype=float)]
    out, prev = [], 0
    for a, b in intervals:
        if a - prev > 1:
            out.append(np.asarray(az[prev:a], dtype=float))
        prev = b
    if len(az) - prev > 1:
        out.append(np.asarray(az[prev:], dtype=float))
    return out


def mz_value(body_mass, mass_fraction, endplate_area_cm2, added_mass=0.0):
    """Formula (C.2): MPa por (m/s2).

    Massa acima da pelve dividida pela area de placa terminal. Equipamento
    carregado no tronco ou na cabeca entra em added_mass.
    """
    borne = body_mass * mass_fraction + added_mass
    return borne * GRAVITY / (endplate_area_cm2 * 1e-4) / 1e6 / GRAVITY


def ultimate_strength(age, s_age):
    """Formula (C.4). Derivada de espécimes de 19 a 93 anos."""
    return ULTIMATE_BASE - s_age * np.asarray(age, dtype=float)


def stress_variable(daily_doses, days_per_year, start_age, mz, s_age):
    """Formula (C.3): variavel de tensao R, acumulada ano a ano."""
    sd = np.atleast_1d(np.asarray(daily_doses, dtype=float))
    days = np.atleast_1d(np.asarray(days_per_year, dtype=float))
    if days.size == 1:
        days = np.repeat(days, sd.size)
    ages = start_age + np.arange(sd.size)          # i de 0 a n-1
    su = ultimate_strength(ages, s_age)
    s_stat = mz * GRAVITY
    denom = su - s_stat
    if np.any(denom <= 0):
        raise ValueError(
            'The ultimate strength minus the static stress is not positive for '
            'every year: the exposure extends past the age where Formula (C.4) '
            'still yields a usable strength, or m_z is too large.')
    return float(np.sum((sd * days ** (1 / 6) / denom) ** 6) ** (1 / 6))


def injury_probability(R, alpha, beta):
    """Formula (C.5): modelo de Weibull, probabilidade entre 0 e 1."""
    return float(1 - np.exp(-((R / alpha) ** beta)))


def classify(prob):
    if prob < RISK_LOW:
        return 'low likelihood of an adverse health effect'
    if prob > RISK_HIGH:
        return 'high likelihood of an adverse health effect'
    return 'intermediate range'


def clause5_sensitivity(Dzd, body_mass, days_per_year, start_age, n_years,
                        current_sex, current_fraction, current_area,
                        added_mass=0.0, span=0.2, n_steps=5):
    """Varre sexo x fracao de massa acima da pelve x area de placa terminal,
    com $D_{zd}$ fixo, analogo a core.sensitivity_grid para o Anexo A/E.

    $D_{zd}$ (Formulas 3-5) e puramente cinematico: nao depende de nenhum dos
    tres parametros varridos aqui. So a conversao em tensao ($m_z$, Formula
    C.2) e o modelo de fadiga ($R$, Formula C.3) dependem da antropometria
    assumida -- por isso a varredura roda em milissegundos sobre um unico
    $D_{zd}$ ja calculado, sem reprocessar o sinal.

    A Clausula 5/Anexo C nao tem parametro de postura (a funcao de
    transferencia assento-coluna, Formula 1, e fixa) -- diferente do Anexo
    A/E, que varre 7 posturas. A dispersao aqui e so antropometrica.

    Para o sexo correntemente selecionado, a varredura e centrada nos valores
    de fracao/area em uso (que podem ter sido editados na interface); para o
    outro sexo, centrada na referencia padrao da norma (SEX_DEFAULTS). Com
    n_steps impar e span simetrico, a combinacao exata em uso aparece como um
    ponto da grade do sexo corrente.
    """
    out = {}
    for sex in ('male', 'female'):
        cfg = SEX_DEFAULTS[sex]
        if sex == current_sex:
            f0, a0 = current_fraction, current_area
        else:
            f0, a0 = cfg['mass_fraction'], cfg['endplate_area']
        fracs = np.linspace(f0 * (1 - span), f0 * (1 + span), n_steps)
        areas = np.linspace(a0 * (1 - span), a0 * (1 + span), n_steps)
        rows = []
        for frac in fracs:
            for area in areas:
                mz = mz_value(body_mass, frac, area, added_mass)
                Sd = mz * Dzd
                R = float(stress_variable(np.repeat(Sd, n_years), days_per_year,
                                          start_age, mz, cfg['s_age']))
                prob = injury_probability(R, cfg['alpha'], cfg['beta'])
                rows.append((frac, area, mz, float(Sd), R, prob))
        out[sex] = np.array(rows, dtype=[
            ('mass_fraction', 'f8'), ('endplate_area', 'f8'), ('mz', 'f8'),
            ('Sd', 'f8'), ('R', 'f8'), ('probability', 'f8')])
    return out


@dataclass
class ClauseFiveResult:
    Dz: float
    Dzd: float
    Sd: float
    R: float
    probability: float
    mz: float
    s_stat: float
    peaks: np.ndarray = field(repr=False, default=None)
    response: np.ndarray = field(repr=False, default=None)
    conditioned: np.ndarray = field(repr=False, default=None)
    dt: float = 0.0
    contact_intervals: list = field(default_factory=list)
    discarded_seconds: float = 0.0
    sex: str = 'male'
    n_years: int = 0
    start_age: int = 20

    @property
    def verdict(self):
        return classify(self.probability)


def assess_from_segments(segments, dt, measured_time, body_mass, sex='male',
                         daily_time=REFERENCE_DAILY_TIME, days_per_year=240,
                         start_age=20, n_years=45, mass_fraction=None,
                         endplate_area=None, added_mass=0.0,
                         exclude_loss_of_contact=True, taper_seconds=0.0):
    """Cadeia completa da Clausula 5 mais Anexo C sobre varios segmentos
    independentes de az (m/s2), cada um com seu proprio tempo continuo.

    Cada segmento e condicionado e tem a perda de contato detectada por
    conta propria -- nada e colado entre segmentos. `measured_time` e o
    tempo total de fato medido (a soma da duracao de cada segmento), que
    pode ser menor que o intervalo de relogio coberto por eles (por exemplo,
    blocos de 2 s de um registrador de campo que grava a cada 5 s: o tempo
    medido conta so os 2 s de cada bloco, nao os 3 s de gravacao no cartao).
    Um unico registro continuo e so o caso de um segmento so -- e o que
    `assess()` faz.
    """
    cfg = SEX_DEFAULTS[sex]
    frac = cfg['mass_fraction'] if mass_fraction is None else mass_fraction
    area = cfg['endplate_area'] if endplate_area is None else endplate_area
    mz = mz_value(body_mass, frac, area, added_mass)

    peaks_all, response_full, cond_full, intervals_all = [], [], [], []
    discarded = 0.0
    for raw in segments:
        raw = np.asarray(raw, dtype=float)
        intervals = detect_loss_of_contact(raw, dt) if exclude_loss_of_contact else []
        discarded += sum(b - a for a, b in intervals) * dt
        intervals_all.extend(intervals)
        for seg in split_contact_segments(raw, dt, intervals):
            if len(seg) < 16:
                continue
            cond = condition(seg, dt, taper_seconds)
            resp = spine_response(cond, dt)
            peaks_all.append(positive_peaks(resp))
            response_full.append(resp)
            cond_full.append(cond)

    peaks = np.concatenate(peaks_all) if peaks_all else np.array([])
    Dz = acceleration_dose(peaks)
    effective = measured_time - discarded
    Dzd = daily_dose([Dz], [daily_time], [max(effective, 1e-9)])
    Sd = mz * Dzd
    R = stress_variable(np.repeat(Sd, n_years), days_per_year, start_age, mz,
                        cfg['s_age'])
    return ClauseFiveResult(
        Dz=Dz, Dzd=Dzd, Sd=Sd, R=R,
        probability=injury_probability(R, cfg['alpha'], cfg['beta']),
        mz=mz, s_stat=mz * GRAVITY, peaks=peaks,
        response=np.concatenate(response_full) if response_full else np.array([]),
        conditioned=np.concatenate(cond_full) if cond_full else np.array([]),
        dt=dt, contact_intervals=intervals_all, discarded_seconds=discarded,
        sex=sex, n_years=n_years, start_age=start_age)


def assess(az, dt, body_mass, sex='male', daily_time=REFERENCE_DAILY_TIME,
           measured_time=None, days_per_year=240, start_age=20, n_years=45,
           mass_fraction=None, endplate_area=None, added_mass=0.0,
           exclude_loss_of_contact=True, taper_seconds=0.0):
    """Cadeia completa da Clausula 5 mais Anexo C, para um unico registro
    continuo em z. Casca fina sobre assess_from_segments com um segmento so.
    """
    az = np.asarray(az, dtype=float)
    measured_time = measured_time or len(az) * dt
    return assess_from_segments(
        [az], dt, measured_time, body_mass, sex=sex, daily_time=daily_time,
        days_per_year=days_per_year, start_age=start_age, n_years=n_years,
        mass_fraction=mass_fraction, endplate_area=endplate_area,
        added_mass=added_mass, exclude_loss_of_contact=exclude_loss_of_contact,
        taper_seconds=taper_seconds)
