"""Tratamento normativo de cada ponto de contato, segundo a ISO 2631-1.

A ISO 2631-1 so avalia SAUDE pela superficie do assento (7.2). Para o encosto,
a nota da subclausula 7.2.3 recomenda medir o eixo x com Wc e k = 0,8, mas
registra que, pela falta de evidencia sobre o efeito na saude, esse valor nao
entra na avaliacao de severidade do Anexo B. Os eixos y e z do encosto e os
tres eixos dos pes nao tem ponderacao de saude.

Encosto e pes sao avaliados por CONFORTO (8.2), pelo valor total de vibracao
a_v = sqrt(sum (k_i a_wi)^2), comparado com as indicacoes aproximadas do
Anexo C, que a propria norma diz nao serem limites.
"""
import numpy as np

from . import nho09

POSITIONS = {
    'seat': 'Seat — health assessment',
    'backrest': 'Backrest — comfort (health not assessed, ISO 2631-1 7.2.3)',
    'feet': 'Feet — comfort (health not assessed)',
}

COMFORT = {
    'seat': (('Wd', 'Wd', 'Wk'), np.array([1.0, 1.0, 1.0])),
    'backrest': (('Wc', 'Wd', 'Wd'), np.array([0.8, 0.5, 0.4])),
    'feet': (('Wk', 'Wk', 'Wk'), np.array([0.25, 0.25, 0.4])),
}

# Anexo C da ISO 2631-1: indicacoes aproximadas, com faixas sobrepostas
COMFORT_SCALE = [
    (0.315, 'not uncomfortable'),
    (0.63, 'a little uncomfortable'),
    (1.0, 'fairly uncomfortable'),
    (1.6, 'uncomfortable'),
    (2.5, 'very uncomfortable'),
    (np.inf, 'extremely uncomfortable'),
]


def comfort_label(av):
    for top, label in COMFORT_SCALE:
        if av < top:
            return label
    return COMFORT_SCALE[-1][1]


def comfort(blocks, dt, position):
    """Valor total de vibracao para conforto, a_v, e os valores por eixo."""
    weights, k = COMFORT[position]
    w = nho09.weighted_blocks(blocks, dt, weights=list(weights))
    aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
    av = float(np.sqrt(np.sum((k * aw) ** 2)))
    return {'weights': weights, 'k': k, 'aw': aw, 'av': av,
            'label': comfort_label(av)}


def backrest_health_informative(blocks, dt, exposure_hours):
    """Eixo x do encosto com Wc e k = 0,8, conforme a nota de 7.2.3.

    Valor informativo: a norma recomenda medi-lo mas o exclui da avaliacao de
    severidade. Nao deve ser comparado com nivel de acao ou limite.
    """
    w = nho09.weighted_blocks(blocks, dt, weights=['Wc', 'Wc', 'Wc'])[:, :, 0]
    aw = float(np.sqrt((w ** 2).mean()))
    t_meas = blocks.shape[0] * blocks.shape[1] * dt
    t_exp = exposure_hours * 3600.0
    a8 = 0.8 * aw * np.sqrt(t_exp / (8 * 3600.0))
    vdv = 0.8 * (np.sum(w ** 4) * dt) ** 0.25 * (t_exp / t_meas) ** 0.25
    return {'aw': aw, 'a8': float(a8), 'vdv': float(vdv)}
