"""Tratamento normativo de cada ponto de contato, segundo a ISO 2631-1.

A ISO 2631-1 so avalia SAUDE pela superficie do assento (7.2). Para o encosto,
a nota da subclausula 7.2.3 recomenda medir o eixo x com Wc e k = 0,8, mas
registra que, pela falta de evidencia sobre o efeito na saude, esse valor nao
entra na avaliacao de severidade do Anexo B. Os eixos y e z do encosto nao tem
ponderacao de saude definida (ficam sem filtro, k = 1, so para acompanhar o
numero); os pes nao tem nota nenhuma de saude na norma, entao os tres eixos
saem sem filtro e k = 1. health_informative() calcula os dois casos, sempre
como valor informativo -- nunca comparado com nivel de acao ou limite.

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
    'custom': 'Custom — user-defined weighting per axis',
}

COMFORT = {
    'seat': (('Wd', 'Wd', 'Wk'), np.array([1.0, 1.0, 1.0])),
    'backrest': (('Wc', 'Wd', 'Wd'), np.array([0.8, 0.5, 0.4])),
    'feet': (('Wk', 'Wk', 'Wk'), np.array([0.25, 0.25, 0.4])),
}

# Ponderacao "saude, informativa" de 7.2.3: so o eixo x do encosto tem
# ponderacao definida (Wc, k = 0,8); y e z ficam sem filtro (None, k = 1).
# Os pes nao tem nenhuma ponderacao de saude na norma; os tres eixos ficam
# sem filtro e com k = 1, o mesmo criterio "sem peso nenhum" que o proprio
# Anexo A usa para S_d e R^A. Nenhum dos dois entra em nivel de acao/limite.
HEALTH = {
    'backrest': (('Wc', None, None), np.array([0.8, 1.0, 1.0])),
    'feet': ((None, None, None), np.array([1.0, 1.0, 1.0])),
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


def comfort(blocks, dt, position, custom=None):
    """Valor total de vibracao para conforto, a_v, e os valores por eixo.

    `custom` e (pesos, k) e so e usado quando position == 'custom' -- a
    posicao sem entrada fixa em COMFORT, com ponderacao e ganho escolhidos
    pelo usuario por eixo. O calculo e o mesmo de qualquer outra posicao;
    so a fonte da ponderacao muda. Sem indicacao aproximada do Anexo C
    (a_v e mostrado, mas comfort_label nao se aplica a uma combinacao
    arbitraria de eixos).
    """
    weights, k = custom if position == 'custom' else COMFORT[position]
    w = nho09.weighted_blocks(blocks, dt, weights=list(weights))
    aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
    av = float(np.sqrt(np.sum((k * aw) ** 2)))
    return {'weights': weights, 'k': k, 'aw': aw, 'av': av,
            'label': comfort_label(av)}


def health_informative(blocks, dt, exposure_hours, position):
    """A(8) e VDV por eixo em 'backrest' ou 'feet', pela ponderacao de HEALTH.

    Valor informativo em ambos os casos: a norma nao define avaliacao de
    severidade para nenhuma das duas posicoes (7.2.3 para o encosto; os pes
    nao tem nem essa nota). Nao deve ser comparado com nivel de acao ou
    limite -- por isso os tres eixos saem sempre, em vez de reduzir a um so
    como fazia a versao anterior desta funcao, restrita ao x do encosto.
    """
    weights, k = HEALTH[position]
    w = nho09.weighted_blocks(blocks, dt, weights=list(weights))
    aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
    t_meas = blocks.shape[0] * blocks.shape[1] * dt
    t_exp = exposure_hours * 3600.0
    a8 = k * aw * np.sqrt(t_exp / (8 * 3600.0))
    vdv = k * (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25 * (t_exp / t_meas) ** 0.25
    return {'weights': weights, 'k': k, 'aw': aw, 'a8': a8, 'vdv': vdv}
