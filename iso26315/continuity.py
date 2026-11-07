"""Continuidade entre blocos de um dia de campo.

O pipeline de campo (NHO-09, Diretiva, posicoes, Clausula 5) recebe a
aceleracao como um array (n_blocos, n_amostras, 3). Isso nasceu do formato
.m antigo, em que cada bloco de 2 s era um trecho isolado (o registrador
gravava 2 s a cada 5 s). As medicoes atuais sao continuas: os blocos sao so
fatias consecutivas de um mesmo sinal, e filtrar cada um isoladamente
introduz transientes de borda a cada 2 s e corta a janela movel do MTVV.

Este modulo guarda, para um array de blocos, quais faixas de blocos sao
consecutivas no tempo ("runs"). Quem processa (nho09.weighted_blocks,
running_rms_trace, Clausula 5) consulta runs_of() e trata cada faixa como um
sinal continuo; blocos fora de qualquer faixa continuam isolados, que e o
tratamento correto para os arquivos .m antigos.

O registro e por identidade do array (weakref): uma fatia ou copia nao herda
a marcacao e cai no tratamento por bloco isolado. Quem fatia um dia continuo
deve marcar a fatia (ver fielddata.FieldDay.record_blocks).
"""
import weakref

_REGISTRY = []          # [(weakref do array, runs)]
_MAX_ENTRIES = 32


def mark(blocks, runs):
    """Marca as faixas [ini, fim) de blocos consecutivos de `blocks`.

    Faixas de um bloco so nao mudam nada e sao descartadas. Devolve `blocks`.
    """
    runs = [(int(a), int(b)) for a, b in runs if int(b) - int(a) > 1]
    _REGISTRY[:] = [(r, v) for r, v in _REGISTRY
                    if r() is not None and r() is not blocks]
    if runs:
        _REGISTRY.append((weakref.ref(blocks), runs))
        del _REGISTRY[:-_MAX_ENTRIES]
    return blocks


def runs_of(blocks):
    """Faixas continuas marcadas para `blocks`, ou [] se nenhuma."""
    for ref, runs in _REGISTRY:
        if ref() is blocks:
            return runs
    return []


def segments(n_blocks, runs):
    """Particao de range(n_blocks) em faixas: as continuas e, entre elas,
    um bloco por faixa. Em ordem."""
    out, pos = [], 0
    for a, b in sorted(runs):
        out += [(i, i + 1) for i in range(pos, a)]
        out.append((a, b))
        pos = b
    out += [(i, i + 1) for i in range(pos, n_blocks)]
    return out
