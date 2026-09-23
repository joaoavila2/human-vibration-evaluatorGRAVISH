"""Diretiva 2002/44/CE: avaliacao da exposicao a vibracao de corpo inteiro.

Anexo, Parte B, ponto 1: A(8) e a MAIOR das tres aceleracoes ponderadas
1,4 a_wx, 1,4 a_wy e a_wz, ou o MAIOR dos tres VDV. A soma dos quadrados dos
eixos e o criterio da diretiva para vibracao MAO-BRACO, nao para corpo inteiro.

Valores do Artigo 3(2): acao 0,5 m/s2 ou 9,1 m/s^1,75; limite 1,15 m/s2 ou
21 m/s^1,75.

Dois modos de combinacao dos eixos estao disponiveis:

  'highest'  o texto da diretiva para corpo inteiro;
  'vector'   soma dos quadrados, como no metodo do laboratorio (CONEM2026),
             que diverge da diretiva por escolha declarada, em nome de uma
             avaliacao mais conservadora.

A gravidade e sempre removida antes de qualquer calculo. A ponderacao Wk tem
passa-alta em 0,4 Hz e ja a elimina do valor eficaz, mas um VDV calculado sobre
aceleracao nao ponderada, ou por outro caminho, retem o 1 g estatico e produz
valores em torno de 130 m/s^1,75 num dia de oito horas, que nao sao vibracao.
"""
from dataclasses import dataclass

import numpy as np

from . import nho09

FACTORS = nho09.FACTORS
EAV_A8, ELV_A8 = 0.5, 1.15
EAV_VDV, ELV_VDV = 9.1, 21.0
# Zona de cautela da ISO 2631-1, usada no metodo do laboratorio por ser mais
# restritiva que a diretiva para o VDV.
ISO_VDV_LOW, ISO_VDV_HIGH = 8.5, 17.0


@dataclass
class DirectiveResult:
    aw: np.ndarray            # valor eficaz ponderado por eixo, sem fator
    a8_axis: np.ndarray       # A(8) parcial por eixo, com fator
    a8: float                 # combinado pelo modo escolhido
    vdv_axis: np.ndarray      # VDV por eixo, com fator e extrapolado
    vdv: float
    crest: np.ndarray
    mode: str
    t_exp_h: float
    t_meas_h: float
    time_basis: str

    def verdict_a8(self):
        return _classify(self.a8, EAV_A8, ELV_A8)

    def verdict_vdv(self, iso_limits=False):
        lo, hi = ((ISO_VDV_LOW, ISO_VDV_HIGH) if iso_limits
                  else (EAV_VDV, ELV_VDV))
        return _classify(self.vdv, lo, hi)


def _classify(v, action, limit):
    if v > limit:
        return 'above the exposure limit value'
    if v >= action:
        return 'above the exposure action value'
    return 'below the exposure action value'


def assess(blocks, dt, exposure_hours, mode='highest'):
    """Avaliacao pela diretiva.

    exposure_hours e o tempo de exposicao usado em A(8) = k a_w sqrt(t/8h) e no
    fator (t_exp/t_meas)^(1/4) do VDV. Para registros com ciclo de trabalho,
    o correto e o periodo coberto: a vibracao nao para nos intervalos em que o
    cartao nao grava.
    """
    w = nho09.weighted_blocks(blocks, dt)
    aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
    peak = np.abs(w).max(axis=(0, 1))
    crest = np.divide(peak, aw, out=np.zeros(3), where=aw > 0)

    t_meas = blocks.shape[0] * blocks.shape[1] * dt
    t_exp = exposure_hours * 3600.0

    a8_axis = FACTORS * aw * np.sqrt(t_exp / (8 * 3600.0))
    vdv_meas = (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25
    vdv_axis = FACTORS * vdv_meas * (t_exp / t_meas) ** 0.25

    if mode == 'vector':
        a8 = float(np.sqrt(np.sum(a8_axis ** 2)))
        vdv = float(np.sum(vdv_axis ** 4) ** 0.25)
    else:
        a8 = float(a8_axis.max())
        vdv = float(vdv_axis.max())
    return DirectiveResult(aw=aw, a8_axis=a8_axis, a8=a8, vdv_axis=vdv_axis,
                           vdv=vdv, crest=crest, mode=mode,
                           t_exp_h=exposure_hours, t_meas_h=t_meas / 3600.0,
                           time_basis='')
