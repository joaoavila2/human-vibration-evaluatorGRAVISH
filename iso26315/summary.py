"""Todos os criterios de uma vez, para visualizacao conjunta.

Cada criterio vem com o seu valor, as suas faixas e se e normativo para a
posicao do sensor. Nada aqui muda um calculo: e so reuniao dos resultados.
"""
from dataclasses import dataclass

import numpy as np

from . import directive, labmethod, nho09, positions as ps


@dataclass
class Criterion:
    name: str
    metric: str
    value: float
    action: float        # nivel de acao ou inicio de faixa; nan se nao houver
    limit: float
    normative: bool      # se o criterio se aplica a esta posicao
    note: str = ''


def all_criteria(day, position, hours, lab_cfg):
    norm, d = normative_criteria(day, position, hours)
    lab_c, lab = lab_criteria(day, lab_cfg)
    return norm + lab_c, lab, d


def normative_criteria(day, position, hours):
    blocks, dt = day.blocks, day.dt
    seat = position == 'seat'
    out = []

    r = nho09.assess(blocks, dt, hours)
    out += [Criterion('NHO-09', 'aren', r.aren, 0.5, 1.1, seat),
            Criterion('NHO-09', 'VDVR', r.vdvr, 9.1, 21.0, seat)]

    d = directive.assess(blocks, dt, hours, 'highest')
    out += [Criterion('Directive, highest axis', 'A(8)', d.a8, 0.5, 1.15, seat),
            Criterion('Directive, highest axis', 'VDV', d.vdv, 9.1, 21.0, seat)]

    v = directive.assess(blocks, dt, hours, 'vector')
    out += [Criterion('Directive, vector sum', 'A(8)', v.a8, 0.5, 1.15, seat,
                      'diverges from the Directive text')]

    c = ps.comfort(blocks, dt, position)
    out.append(Criterion('ISO 2631-1 comfort', 'a_v', c['av'], 0.315, 1.0, True,
                         c['label']))

    if position == 'backrest':
        h = ps.backrest_health_informative(blocks, dt, hours)
        out.append(Criterion('Backrest x, Wc k=0.8', 'A(8)', h['a8'],
                             np.nan, np.nan, False, 'informative, 7.2.3'))

    return out, d


def lab_criteria(day, lab_cfg):
    lab = labmethod.assess(day.records, lab_cfg, day.rate)
    out = [Criterion('Lab method (MATLAB)', 'A(8)', lab.a8_daily, 0.5, 1.15,
                      False, 'non-normative configuration'),
            Criterion('Lab method (MATLAB)', 'VDV', lab.vdv_daily, 9.1, 21.0,
                      False, 'reproduces the app defects'
                      if lab_cfg.reproduce_vdv_defects else 'defects corrected')]
    return out, lab


def per_file(day, hours_total):
    """Contribuicao de cada arquivo (hora) para o A(8) diario, por eixo.

    Normativa (Wd, Wd, Wk com 1,4 / 1,4 / 1,0). A soma dos quadrados das
    parcelas de todos os arquivos e o quadrado do A(8) diario, entao a fracao
    de cada arquivo na energia do dia sai direto.
    """
    rows = []
    for rec in day.records:
        b = rec.blocks_si()
        w = nho09.weighted_blocks(b, day.dt)
        aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
        a8 = nho09.FACTORS * aw * np.sqrt(rec.span / (8 * 3600.0))
        peak = np.abs(w).max(axis=(0, 1))
        rows.append(dict(name=rec.path.stem[-11:], a8=a8, aw=aw,
                         crest=np.divide(peak, aw, out=np.zeros(3),
                                         where=aw > 0)))
    energy = np.array([r['a8'] ** 2 for r in rows])
    total = energy.sum(axis=0)
    for r, e in zip(rows, energy):
        r['share'] = np.divide(e, total, out=np.zeros(3), where=total > 0)
    return rows
