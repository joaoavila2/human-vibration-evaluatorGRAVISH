"""Orquestracao das analises orientadora (A.3.1) e individual (Anexo E)."""
from dataclasses import dataclass, field

import numpy as np

from . import core

# Resultados completos guardam a serie de forca (6 x N) e o espectro (6 x nf),
# que somam dezenas de MB por combinacao antropometria x exposicao. Um
# historico longo percorre dezenas dessas combinacoes e estoura a memoria. Os
# calculos de dose e risco precisam apenas de S, S_stat e t_m, entao o cache
# guarda o resultado completo so para as primeiras combinacoes, que sao as que
# os graficos mostram, e um resumo leve para as demais.
MAX_FULL_RESULTS = 4


class _LightResult:
    """Resumo de uma exposicao, sem as series temporais."""
    __slots__ = ('S', 'S_stat', 'measurement_time', 'posture', 'k', 'bmi_per',
                 'excitations', 'paths')

    def __init__(self, res):
        self.S = res.S
        self.S_stat = res.S_stat
        self.measurement_time = res.measurement_time
        self.posture = res.posture
        self.k = res.k
        self.bmi_per = res.bmi_per
        self.excitations = {}
        self.paths = res.paths

# Valores padrao da analise orientadora (A.3.1)
ORIENTING = dict(
    posture=3, body_mass=110.0, body_height=1.85,
    start_age=20, n_years=45, days_per_year=240, daily_time=14400.0,
)


@dataclass
class Assessment:
    R: np.ndarray
    daily_dose: np.ndarray            # dose do primeiro ano, MPa
    S_stat: np.ndarray
    exposures: dict = field(default_factory=dict)   # nome -> ExposureResult
    kind: str = 'orientating'
    start_age: int = 20
    n_years: int = 45
    notes: list = field(default_factory=list)
    drift: dict = field(default_factory=dict)
    trimmed: int = 0

    @property
    def critical_level(self):
        return int(np.argmax(self.R))

    @property
    def max_R(self):
        return float(self.R.max())


class Runner:
    """Executa uma analise, com cache das exposicoes ja calculadas."""

    def __init__(self, store, progress=None):
        self.store = store
        self.progress = progress or (lambda frac, msg: None)
        self._cache = {}
        self.drift_reports = {}
        self._full_count = 0
        self.trimmed = 0

    def _exposure(self, exposure, mass, height, subclass=1):
        key = (exposure.identifier, round(mass, 3), round(height, 3), subclass)
        if key in self._cache:
            return self._cache[key]

        paths = exposure.paths()
        try:
            excs, drift = core.read_exposure_files(
                paths, progress=lambda _, m: self.progress(
                    None, f'"{exposure.identifier}" — {m}'))
        except ValueError:
            raise ValueError(
                f'Exposure "{exposure.identifier}" has no acceleration file: '
                f'the seat must be measured.')
        if not drift['aligned']:
            self.drift_reports[exposure.identifier] = drift
        self.progress(None, f'"{exposure.identifier}" — computing forces...')

        res = core.analyse_exposure(
            self.store, excs, exposure.posture, mass, height,
            measurement_time=exposure.measurement_time or excs[0].duration,
            subclass=subclass)
        res.paths = tuple(paths)
        # O espectro de deslocamento so serve dentro de compute_forces, e o
        # espectro bruto so serve para a sobreposicao "unprocessed" dos
        # primeiros graficos. Ambos sao os maiores arrays de uma excitacao.
        for exc in res.excitations.values():
            exc.spectrum = None
            if self._full_count >= 2:
                exc.raw_spectrum = None
        if self._full_count < MAX_FULL_RESULTS:
            self._cache[key] = res
            self._full_count += 1
        else:
            self._cache[key] = _LightResult(res)
            self.trimmed += 1
        return self._cache[key]

    # ------------------------------------------------------------------
    def run_orienting(self, history, subclass=1, equation=2, protection=50):
        """A.3.1: postura grupo 3, P95/IMC alto, 45 anos, 240 dias, 4 h/dia."""
        cfg = ORIENTING
        name, exposure = next(iter(history.exposures.items()))
        exposure = _with_posture(exposure, cfg['posture'])
        res = self._exposure(exposure, cfg['body_mass'], cfg['body_height'],
                             subclass)
        Sd = core.daily_dose([res], [cfg['daily_time']])
        n = cfg['n_years']
        R = core.risk_factor_with_static(
            np.tile(Sd, (n, 1)), np.full(n, float(cfg['days_per_year'])),
            res.S_stat, start_age=cfg['start_age'],
            equation=equation, protection=protection)
        self.progress(1.0, 'Done')
        return Assessment(R=R, daily_dose=Sd, S_stat=res.S_stat,
                          exposures={name: res}, kind='orientating',
                          start_age=cfg['start_age'], n_years=n,
                          drift=dict(self.drift_reports),
                          trimmed=self.trimmed)

    # ------------------------------------------------------------------
    def run_individual(self, history, birth_year, subclass=1,
                       equation=2, protection=50):
        """Anexo E: padroes de exposicao e antropometria variaveis por ano."""
        years = history.years
        if not any(y.patterns for y in years):
            raise ValueError('No year in the "Year" sheet has an exposure pattern.')
        # Anos sem padrao (afastamento) permanecem na lista: eles nao acumulam
        # dose, mas continuam avancando o indice i da Formula (E.1), que e a
        # idade. Descarta-los comprimiria a linha do tempo e subestimaria o
        # risco, porque a resistencia ultima decai com a idade.

        start_age = years[0].year - birth_year
        doses, days_list, statics = [], [], []
        used, notes = {}, []

        for i, yr in enumerate(years):
            self.progress(i / len(years), f'Year {yr.year}')
            if not yr.patterns:
                doses.append(np.zeros(6))
                days_list.append(0.0)
                statics.append(statics[-1] if statics else np.zeros(6))
                continue
            year_doses, year_days = [], []
            year_static = None
            for pattern_name, days in yr.patterns:
                pattern = history.patterns.get(pattern_name)
                if pattern is None:
                    notes.append(f'{yr.year}: pattern "{pattern_name}" is not in the '
                                 f'Pattern sheet; skipped.')
                    continue
                results, durations = [], []
                for exp_name, duration in pattern.entries:
                    exposure = history.exposures.get(exp_name)
                    if exposure is None:
                        notes.append(f'{yr.year}: exposure "{exp_name}" is not in the '
                                     f'Exposure sheet; skipped.')
                        continue
                    res = self._exposure(exposure, yr.body_mass,
                                         yr.body_height, subclass)
                    used[f'{exp_name} ({yr.body_mass:g} kg)'] = res
                    results.append(res)
                    durations.append(duration)
                if not results:
                    continue
                year_doses.append(core.daily_dose(results, durations))
                year_days.append(days)
                year_static = results[0].S_stat

            if not year_doses:
                doses.append(np.zeros(6))
                days_list.append(0.0)
                statics.append(statics[-1] if statics else np.zeros(6))
                continue
            doses.append(core.average_dose(year_doses, year_days))
            days_list.append(sum(year_days))
            statics.append(year_static)

        if not any(d.any() for d in doses):
            raise ValueError('No year produced a computable exposure.')

        R = core.risk_factor_with_static(
            np.array(doses), np.array(days_list), np.array(statics),
            start_age=start_age, equation=equation, protection=protection)
        self.progress(1.0, 'Done')
        return Assessment(R=R, daily_dose=doses[0], S_stat=statics[0],
                          exposures=used, kind='individual',
                          start_age=start_age, n_years=len(doses), notes=notes,
                          drift=dict(self.drift_reports),
                          trimmed=self.trimmed)


def _with_posture(exposure, posture):
    import copy
    e = copy.copy(exposure)
    e.posture = posture
    return e
