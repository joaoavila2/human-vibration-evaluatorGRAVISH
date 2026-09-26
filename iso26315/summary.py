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


def all_criteria(day, position, hours, lab_cfg, custom=None):
    norm, d = normative_criteria(day, position, hours, custom)
    lab_c, lab = lab_criteria(day, lab_cfg)
    return norm + lab_c, lab, d


def display_weights(position, custom):
    """Ponderacao e ganho a mostrar como "this software" fora do assento.

    None/None (o padrao de directive.assess, Wd/Wd/Wk 1,4/1,4/1,0) so faz
    sentido no assento -- e a definicao da propria diretiva. Fora dele, a
    grade "A(8) per axis" e as linhas de comparacao devem usar a mesma
    ponderacao que positions.HEALTH ja usa em health_informative, senao a
    coluna "this software" mostra Wd/Wd/Wk aplicado a um sensor de encosto
    ou pe, o que nao e nem a definicao da diretiva nem nada normativo --
    so ponderacao errada disfarcada de comparacao.
    """
    if position in ps.HEALTH:
        return ps.HEALTH[position]
    if position == 'custom' and custom is not None:
        return custom
    return None, None


def normative_criteria(day, position, hours, custom=None):
    blocks, dt = day.blocks, day.dt
    seat = position == 'seat'
    out = []

    r = nho09.assess(blocks, dt, hours)
    out += [Criterion('NHO-09', 'aren', r.aren, 0.5, 1.1, seat),
            Criterion('NHO-09', 'VDVR', r.vdvr, 9.1, 21.0, seat)]

    mtvv_val = nho09.mtvv(blocks, dt)
    out.append(Criterion('MTVV', 'a_running,max', mtvv_val, np.nan, np.nan,
                         False, 'informative, running RMS (1 s) peak, ISO '
                         '2631-1 6.4.2 -- no fixed action/limit pair'))

    msdv_val = nho09.msdv(blocks, dt, hours)
    out.append(Criterion('MSDV', 'a_wf,z', msdv_val, np.nan, np.nan, False,
                         'informative, motion sickness dose (Wf, vertical), '
                         'ISO 2631-1 Sec.7 -- no fixed action/limit pair'))

    # d segue sempre Wd/Wd/Wk: e a definicao literal da diretiva, e as duas
    # linhas de criterio abaixo (highest axis / vector sum) representam essa
    # definicao mesmo quando marcadas nao-normativas fora do assento.
    d = directive.assess(blocks, dt, hours, 'highest')
    out += [Criterion('Directive, highest axis', 'A(8)', d.a8, 0.5, 1.15, seat),
            Criterion('Directive, highest axis', 'VDV', d.vdv, 9.1, 21.0, seat)]

    v = directive.assess(blocks, dt, hours, 'vector')
    out += [Criterion('Directive, vector sum', 'A(8)', v.a8, 0.5, 1.15, seat,
                      'diverges from the Directive text')]

    c = ps.comfort(blocks, dt, position, custom=custom)
    out.append(Criterion('ISO 2631-1 comfort', 'a_v', c['av'], 0.315, 1.0,
                         position != 'custom',
                         c['label'] if position != 'custom'
                         else 'user-defined weighting, not an ISO indication'))

    if position in ('backrest', 'feet'):
        h = ps.health_informative(blocks, dt, hours, position)
        if position == 'backrest':
            name, value = 'Backrest x, Wc k=0.8', float(h['a8'][0])
            note = 'informative, 7.2.3'
        else:
            name = 'Feet, unweighted'
            value = float(np.sqrt((h['a8'] ** 2).sum()))
            note = 'informative, not defined by the norm'
        out.append(Criterion(name, 'A(8)', value, np.nan, np.nan, False, note))

    # d_display: o mesmo formato de resultado que d, mas com a ponderacao
    # certa para a posicao -- usado pela grade "A(8) per axis" e pelas linhas
    # de comparacao "this software", NUNCA pelos dois criterios acima.
    dw, dk = display_weights(position, custom)
    d_display = d if dw is None else directive.assess(
        blocks, dt, hours, 'highest', weights=dw, k=dk)

    return out, d_display


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
