"""Curvas de referencia biodinamicas e avisos de dominio de validade.

Nada aqui entra em nenhum calculo normativo. As curvas sao desenhadas por
cima da transmissibilidade H1 medida, para comparacao visual; os avisos
apontam quando uma entrada sai da faixa em que a norma foi calibrada.

Curvas
------
ISO 5982:2001, Anexo C: transmissibilidade assento-cabeca media (alvo), na
forma H(s) = K P(s)/Q(s), com s = j f (f em hertz). Coeficientes da Tabela
C.1. Conferidos contra a Tabela 3 da norma: 1,45 a 4 Hz e 1,47 a 5 Hz, com
desvio de ate 0,05 entre 0,5 e 20 Hz. Vale para adulto de 49 a 93 kg (56 a
90 kg na transmissibilidade), sentado ereto SEM encosto, de 0,5 a 20 Hz e
1 a 5 m/s2 rms.

Giacomin (2007), Int. J. Ind. Ergon. 37:183-195: modelo de um grau de
liberdade com apoio de base, ajustado a massa aparente de oito criancas de
5 a 12 kg sentadas COM encosto. As regressoes alometricas sobre 80 modelos
(primatas, criancas pequenas, criancas maiores e adultos, de 6 a 108 kg)
dao, com M a massa total em kg:
    m1 = 0,6949 M^1,0485   k1 = 6701,58 M^0,47   c1 = 63,45 M^0,74
e frequencia natural ~ 15,63 M^-0,29. A curva desenhada e a
transmissibilidade da base ate a massa movel m1, (k + j w c)/(k - w^2 m1 +
j w c). Nao e transmissibilidade assento-cabeca medida: e o modo de corpo
inteiro do modelo, util para ver onde fica a ressonancia esperada para a
massa informada. Os coeficientes das regressoes so aparecem nas Figuras 6 a
10 do artigo.
"""
import numpy as np

# ISO 5982:2001, Tabela C.1, coluna H(s)
_ISO5982_H = dict(K=275.0, A=(13.16, 4.24, 3.62), B=(149.65, 38.0, 9.83),
                  C=(28.21, 10.69, 3.67, 3.78), D=(324.0, 121.0, 30.25, 12.96))
ISO5982_BAND = (0.5, 20.0)


def iso5982_seat_to_head(freq):
    """Transmissibilidade assento-cabeca media da ISO 5982 (complexa).
    NaN fora de 0,5 a 20 Hz, onde a norma nao define valores."""
    f = np.asarray(freq, dtype=float)
    s = 1j * f
    c = _ISO5982_H
    P = np.ones_like(s)
    for a, b in zip(c['A'], c['B']):
        P = P * (s ** 2 + a * s + b)
    Q = np.ones_like(s)
    for a, d in zip(c['C'], c['D']):
        Q = Q * (s ** 2 + a * s + d)
    H = c['K'] * P / Q
    inside = (f >= ISO5982_BAND[0]) & (f <= ISO5982_BAND[1])
    return np.where(inside, H, np.nan)


def giacomin_parameters(mass):
    """(m1, k1, c1, fn, zeta) do modelo de 1 GDL pelas regressoes de
    Giacomin (2007) para a massa total `mass` em kg."""
    m1 = 0.6949 * mass ** 1.0485
    k1 = 6701.58 * mass ** 0.47
    c1 = 63.45 * mass ** 0.74
    fn = np.sqrt(k1 / m1) / (2 * np.pi)
    zeta = c1 / (2 * np.sqrt(k1 * m1))
    return m1, k1, c1, fn, zeta


def giacomin_transmissibility(freq, mass):
    """Transmissibilidade base -> massa movel do modelo de Giacomin (2007)."""
    m1, k1, c1, _, _ = giacomin_parameters(mass)
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    return (k1 + 1j * w * c1) / (k1 - w ** 2 * m1 + 1j * w * c1)


GIACOMIN_MASS_RANGE = (5.98, 108.0)     # massas dos 80 modelos da regressao


def reference_curves(settings, freq):
    """[(rotulo, f, |H|, estilo)] para as referencias ligadas em settings,
    um dict com 'iso5982' (bool) e 'child_mass' (kg ou None)."""
    out = []
    f = np.asarray(freq, dtype=float)
    if settings.get('iso5982'):
        out.append(('ISO 5982 seat-to-head mean (adult, no backrest)',
                    f, np.abs(iso5982_seat_to_head(f)), '--'))
    mass = settings.get('child_mass')
    if mass:
        _, _, _, fn, zeta = giacomin_parameters(mass)
        out.append((f'Giacomin 2007 1-DOF model, {mass:g} kg '
                    f'(fn {fn:.1f} Hz, zeta {zeta:.2f})',
                    f, np.abs(giacomin_transmissibility(f, mass)), ':'))
    return out


def draw_references(ax, settings, fmax):
    """Desenha as referencias ligadas sobre o eixo de |H1|."""
    f = np.linspace(0.05, fmax, 2000)
    for label, ff, mag, style in reference_curves(settings, f):
        ax.plot(ff, mag, ls=style, lw=1.6, color='#212529', label=label)


# ----------------------------------------------------------------------
# Dominio de validade
# ----------------------------------------------------------------------

ADULT_START_AGE = 20        # idade de inicio de todos os exemplos da norma
SCOPE_PEAK = 137.3          # m/s2, Clausula 1 da ISO 2631-5:2018 (14 g)
CHILD_MASS = 40.0           # kg, abaixo disso m_z adulto e' claramente inadequado


def strength_at(age, annex, s_age=0.052):
    """Resistencia ultima (MPa) pela reta do Anexo C ou do Anexo E."""
    if annex == 'C':
        return 6.75 - s_age * age
    return 6.765024 - 0.067184 * age


def start_age_warning(start_age, annex, s_age=0.052):
    """Texto de aviso se a exposicao comeca antes dos 20 anos, ou ''."""
    if start_age >= ADULT_START_AGE:
        return ''
    su0 = strength_at(start_age, annex, s_age)
    su20 = strength_at(ADULT_START_AGE, annex, s_age)
    data = ('107 fatigue tests on lumbar segments of cadavers aged 19 to 93'
            if annex == 'C' else
            'European adult drivers; every example of the norm starts at 20')
    return (f'Exposure starts at age {start_age}. The ultimate-strength line '
            f'of Annex {annex} was calibrated on {data}, and below 20 it is '
            f'an extrapolation: it gives {su0:.2f} MPa at {start_age} against '
            f'{su20:.2f} MPa at 20, i.e. a child treated as stronger than a '
            f'young adult. Paediatric data (vertebral density plateau before '
            f'puberty, immature growth plates) do not support that, so R is '
            f'likely underestimated.')


def scope_peak_warning(peak):
    if peak <= SCOPE_PEAK:
        return ''
    return (f'Peak |a_z| of {peak:.1f} m/s&sup2; exceeds the 137.3 m/s&sup2; '
            f'(14 g) scope of ISO 2631-5 (Clause 1), close to the physical '
            f'limit a spine can tolerate. The method is not defined here.')


def child_mass_warning(body_mass):
    if body_mass >= CHILD_MASS:
        return ''
    return (f'Body mass {body_mass:g} kg. m_z uses the adult mass fraction '
            f'above the pelvis and an adult endplate area (14 cm&sup2; men, '
            f'11.6 cm&sup2; women); endplate area cannot be predicted from '
            f'external anthropometry even in adults (Seidel et al. 2008), and '
            f'no paediatric value exists. Treat m_z as uncertain.')


# ----------------------------------------------------------------------
# Painel de referencia de intensidade (item 7.3)
# ----------------------------------------------------------------------
# ISO 2631-1:1997 Annex C, Tabela C.1 — escalas de conforto aproximadas
# para a_v em m/s^2 (limite superior de cada faixa, +inf na ultima).
ISO2631_1_COMFORT = [
    (0.315, 'not uncomfortable'),
    (0.630, 'a little uncomfortable'),
    (1.000, 'fairly uncomfortable'),
    (1.600, 'uncomfortable'),
    (2.500, 'very uncomfortable'),
    (float('inf'), 'extremely uncomfortable'),
]

# ISO 2631-1:1997 Annex B — faixa de atencao de saude para A(8) em m/s^2
ISO2631_1_HEALTH_LOW = 0.45
ISO2631_1_HEALTH_HIGH = 0.90

PAEDIATRIC_CAVEAT = (
    'No validated paediatric whole-body vibration exposure limit exists in '
    'ISO 2631-1 or ISO 2631-5. The adult strength-regression lines '
    '(Annexes C/E), mass fractions and endplate areas do not apply below '
    '20 years; all adult thresholds shown above are for contextual '
    'reference only.'
)

THERAPY_NOTE = (
    'Whole-body vibration therapy protocols (typically 15–35 Hz, '
    '0.3–1 g, ≤15 min/day) are designed to elicit a '
    'physiological stimulus, not to set a safe-exposure threshold for '
    'unintended daily vibration from mobility devices.'
)


def comfort_label(av):
    """ISO 2631-1 Annex C approximate comfort descriptor for a_v (m/s^2)."""
    for threshold, label in ISO2631_1_COMFORT:
        if av < threshold:
            return label
    return 'extremely uncomfortable'


def health_guidance_label(a8):
    """ISO 2631-1 Annex B health guidance caution zone placement for A(8)."""
    if a8 < ISO2631_1_HEALTH_LOW:
        return 'below the health guidance caution zone'
    if a8 <= ISO2631_1_HEALTH_HIGH:
        return 'within the health guidance caution zone'
    return 'above the health guidance caution zone'
