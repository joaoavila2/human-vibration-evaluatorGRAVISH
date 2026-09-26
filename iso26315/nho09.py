"""NHO-09 (Fundacentro, 2013): avaliacao da exposicao a vibracao de corpo inteiro.

Independente das cadeias da ISO 2631-5. Reaproveita apenas as ponderacoes em
frequencia de core.py, que a NHO-09 adota da ISO 2631-1 (Wk em z, Wd em x e y).

Definicoes da norma, na ordem em que se encadeiam:

  am_j   = valor eficaz da aceleracao ponderada no eixo j
  amr    = sqrt((1,4 am_x)^2 + (1,4 am_y)^2 + (1,0 am_z)^2)   soma vetorial
  arep_i = media ARITMETICA dos amr das s amostras da componente i
  are    = sqrt( (1/T) * soma_i n_i arep_i^2 T_i )
  aren   = are * sqrt(T / 8 h)
  VDVexp_j = f_j VDV_j (T_exp / T_amos)^(1/4)
  VDVR   = [ soma_j VDVexp_j^4 ]^(1/4)

Criterio: nivel de acao aren 0,5 m/s2 ou VDVR 9,1 m/s^1,75; limite de
exposicao aren 1,1 m/s2 ou VDVR 21 m/s^1,75.

Diferenca importante em relacao a diretiva europeia: a NHO-09 combina os tres
eixos por soma vetorial, e nao pelo eixo de maior valor.
"""
from dataclasses import dataclass

import numpy as np

from . import core

FACTORS = np.array([1.4, 1.4, 1.0])        # f_x, f_y, f_z
WEIGHTS = ('Wd', 'Wd', 'Wk')
T0_HOURS = 8.0
ACTION_AREN, LIMIT_AREN = 0.5, 1.1
ACTION_VDVR, LIMIT_VDVR = 9.1, 21.0


_CACHE = []                  # [(blocks, dt, pesos, pad, resultado)], ultimos 4


def weighted_blocks(blocks, dt, pad_factor=3, weights=None):
    """Versao com cache: a mesma ponderacao dos mesmos blocos e pedida por
    varios criterios ao montar o resumo, e custa segundos num dia inteiro."""
    key = (tuple(weights or WEIGHTS), dt, pad_factor)
    for b, k, res in _CACHE:
        if b is blocks and k == key:
            return res
    res = _weighted_blocks(blocks, dt, pad_factor, weights)
    _CACHE.append((blocks, key, res))
    del _CACHE[:-4]
    return res


def _weighted_blocks(blocks, dt, pad_factor=3, weights=None):
    """Ponderacao da ISO 2631-1 aplicada bloco a bloco.

    Cada bloco e acolchoado com zeros antes da filtragem. Sem isso, a
    filtragem no dominio da frequencia e circular, e com blocos de 2 s a
    constante de tempo do passa-alta de 0,4 Hz faz a resposta enrolar de uma
    ponta do bloco para a outra.
    """
    n_blk, n, _ = blocks.shape
    m = n * (1 + pad_factor)
    f = np.fft.rfftfreq(m, dt)
    # None (usado pelas ponderacoes de saude do encosto/pe, que deixam y/z ou
    # todos os eixos sem filtro) significa passagem direta: resposta 1 em
    # toda a frequencia, o mesmo criterio que core._axis_gain ja usa.
    H = np.stack([np.ones_like(f, dtype=complex) if w is None
                  else core.weighting_response(w, f)
                  for w in (weights or WEIGHTS)], axis=1)   # (nf, 3)
    out = np.empty_like(blocks)
    step = 256                       # lotes de blocos: rapido sem estourar memoria
    for i in range(0, n_blk, step):
        X = np.fft.rfft(blocks[i:i + step], m, axis=1)
        out[i:i + step] = np.fft.irfft(X * H[None], m, axis=1)[:, :n]
    return out


@dataclass
class NHO09Result:
    am: np.ndarray            # valor eficaz ponderado por eixo, energia agregada
    amr: float                # soma vetorial com os fatores
    amr_arithmetic: float     # media aritmetica dos amr por bloco
    are: float
    aren: float               # normalizado por raiz quadrada do tempo
    aren_quarter: float       # alternativa com raiz quarta, ISO 2631-1 B.2
    vdv: np.ndarray           # VDV medido por eixo, sem fator
    vdv_exp: np.ndarray       # VDVexp por eixo
    vdvr: float
    t_exp_h: float
    t_amos_h: float
    n_blocks: int
    moving_fraction: float
    amr_per_block: np.ndarray = None

    @property
    def verdict_aren(self):
        return classify(self.aren, ACTION_AREN, LIMIT_AREN)

    @property
    def verdict_vdvr(self):
        return classify(self.vdvr, ACTION_VDVR, LIMIT_VDVR)


def classify(value, action, limit):
    if value > limit:
        return 'above the exposure limit'
    if value >= action:
        return 'above the action level'
    return 'below the action level'


def assess(blocks, dt, exposure_hours, moving_mask=None):
    """Avaliacao NHO-09 de um conjunto de blocos amostrados.

    blocks          (n_blocos, n_amostras, 3), aceleracao em m/s2, ja sem a
                    componente estatica e com os eixos no sistema basicentrico
    exposure_hours  T_exp: tempo real de exposicao no dia, em horas

    O dia inteiro e tratado como uma unica componente de exposicao, amostrada
    de forma sistematica. O am_j e agregado pela energia de todos os blocos,
    o que equivale a tratar o conjunto como uma medicao continua. Ver a nota
    sobre a media aritmetica em arithmetic_bias.
    """
    w = weighted_blocks(blocks, dt)
    n_blk, n, _ = w.shape

    ms_block = (w ** 2).mean(axis=1)                     # (n_blk, 3)
    am = np.sqrt(ms_block.mean(axis=0))                  # agregado pela energia
    amr = float(np.sqrt(np.sum((FACTORS * am) ** 2)))

    amr_block = np.sqrt(np.sum((FACTORS * np.sqrt(ms_block)) ** 2, axis=1))
    amr_arith = float(amr_block.mean())

    t_amos = n_blk * n * dt
    t_exp = exposure_hours * 3600.0
    are = amr                                            # componente unica
    aren = are * np.sqrt(t_exp / (T0_HOURS * 3600.0))
    aren_q = are * (t_exp / (T0_HOURS * 3600.0)) ** 0.25

    vdv = (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25     # VDV_j das s amostras
    vdv_exp = FACTORS * vdv * (t_exp / t_amos) ** 0.25
    vdvr = float(np.sum(vdv_exp ** 4) ** 0.25)

    moving = (float(np.mean(moving_mask)) if moving_mask is not None
              else float('nan'))
    return NHO09Result(
        am=am, amr=amr, amr_arithmetic=amr_arith, are=are,
        aren=float(aren), aren_quarter=float(aren_q),
        vdv=vdv, vdv_exp=vdv_exp, vdvr=vdvr,
        t_exp_h=exposure_hours, t_amos_h=t_amos / 3600.0,
        n_blocks=n_blk, moving_fraction=moving, amr_per_block=amr_block)


def arithmetic_bias(result):
    """Quanto a media aritmetica prescrita subestima a media pela energia."""
    return result.amr_arithmetic / result.amr - 1.0


def running_rms_trace(blocks, dt, tau=1.0):
    """Mesma janela deslizante do mtvv(), mas devolve o traco inteiro (nao
    so o maximo) -- para plotar RMS corrente x tempo. Formato (n_blk,
    n_pontos), soma vetorial ja aplicada (mesmos fatores de amr)."""
    w = weighted_blocks(blocks, dt)
    n_blk, n, _ = w.shape
    win = max(1, min(n, int(round(tau / dt))))

    csum = np.cumsum(w ** 2, axis=1)
    csum = np.concatenate([np.zeros((n_blk, 1, 3)), csum], axis=1)
    running_ms = (csum[:, win:, :] - csum[:, :-win, :]) / win
    running_rms = np.sqrt(running_ms)
    return np.sqrt(np.sum((FACTORS * running_rms) ** 2, axis=2))


def mtvv(blocks, dt, tau=1.0):
    """Maximum Transient Vibration Value (ISO 2631-1 6.4.2): RMS corrente
    numa janela de tau segundos do sinal ponderado, combinado por soma
    vetorial com os mesmos fatores de amr, tomando o maximo entre todos os
    pontos de todos os blocos.

    Cada bloco e' tratado como um segmento continuo isolado -- a janela
    deslizante nunca atravessa a lacuna entre blocos, mesma logica do resto
    do modulo (nao cola blocos discontinuos).
    """
    running_vec = running_rms_trace(blocks, dt, tau)
    return float(running_vec.max()) if running_vec.size else float('nan')


def msdv(blocks, dt, exposure_hours):
    """Motion Sickness Dose Value (ISO 2631-1 Secao 7): ponderacao Wf no
    eixo vertical (0,1-0,5 Hz, enjoo por movimento de baixa frequencia),
    energia agregada e extrapolada para o periodo de exposicao total --
    mesmo estilo de extrapolacao que aren/vdvr usam pra ir da amostra
    medida ate a exposicao completa.

    A norma nao define um par acao/limite fixo pra MSDV (so orientacoes
    qualitativas de percentual de pessoas com enjoo, que variam muito por
    populacao) -- por isso este valor entra como informativo, nao normativo.
    """
    w = weighted_blocks(blocks, dt, weights=(None, None, 'Wf'))
    z = w[:, :, 2]
    ms = float((z ** 2).mean())
    t_exp = exposure_hours * 3600.0
    return float(np.sqrt(ms * t_exp))
