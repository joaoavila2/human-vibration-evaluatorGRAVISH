"""NHO-09 (Fundacentro, 2013): avaliacao da exposicao a vibracao de corpo inteiro.

Independente das cadeias da ISO 2631-5. Reaproveita apenas as ponderacoes em
frequencia de core.py, que a NHO-09 adota da ISO 2631-1 (Wk em z, Wd em x e y).

Definicoes da norma, na ordem em que se encadeiam:

  am_j   = valor eficaz da aceleracao ponderada no eixo j
  amr    = sqrt((1,4 am_x)^2 + (1,4 am_y)^2 + (1,0 am_z)^2)   soma vetorial
  arep_i = media ARITMETICA dos amr das s amostras da componente i
  are    = sqrt( (1/T) * soma_i n_i arep_i^2 T_i )
  aren   = are * sqrt(T / 8 h), arredondado a 0,1 m/s2 (6.4.1)
  VDVexp_j = f_j VDV_j (T_exp / T_amos)^(1/4)
  VDVR   = [ soma_j VDVexp_j^4 ]^(1/4)

Criterio (item 5 e Quadro 1 da norma):

  aren [m/s2]   VDVR [m/s^1,75]   consideracao tecnica
  0 a 0,5       0 a 9,1           aceitavel
  >0,5 a <0,9   >9,1 a <16,4      acima do nivel de acao
  0,9 a 1,1     16,4 a 21         regiao de incerteza
  acima de 1,1  acima de 21       acima do limite de exposicao

O aren e comparado ja arredondado: 1,13 m/s2 vira 1,1 e fica na regiao de
incerteza, nao acima do limite (exemplos do proprio item 6.4.1).

Medicao continua: a norma admite manter a integracao de forma continuada
cobrindo a componente inteira (6.3.1.2), e entao o registro inteiro e uma
unica amostra. E o caso das medicoes atuais do laboratorio, por isso am_j e
agregado pela energia do registro todo; a media aritmetica por blocos so
aparece como conferencia.

Diferenca importante em relacao a diretiva europeia: a NHO-09 combina os tres
eixos por soma vetorial, e nao pelo eixo de maior valor.
"""
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.fft import next_fast_len

from . import continuity, core

FACTORS = np.array([1.4, 1.4, 1.0])        # f_x, f_y, f_z
WEIGHTS = ('Wd', 'Wd', 'Wk')
T0_HOURS = 8.0
ACTION_AREN, UNCERTAIN_AREN, LIMIT_AREN = 0.5, 0.9, 1.1
ACTION_VDVR, UNCERTAIN_VDVR, LIMIT_VDVR = 9.1, 16.4, 21.0

ACCEPTABLE = 'acceptable'
ABOVE_ACTION = 'above the action level'
UNCERTAIN = 'uncertainty region'
ABOVE_LIMIT = 'above the exposure limit'


def round_aren(value):
    """Arredondamento do item 6.4.1: ao 0,1 m/s2 mais proximo, metade para
    cima (0,87 -> 0,9; 1,13 -> 1,1; 1,15 -> 1,2). Decimal evita que 1,15 em
    ponto flutuante (1,1499...) caia para 1,1."""
    if not np.isfinite(value):
        return float(value)
    return float(Decimal(repr(float(value))).quantize(Decimal('0.1'),
                                                      ROUND_HALF_UP))


def classify(value, action, limit, uncertain=None):
    """Faixas do Quadro 1. `uncertain` e o inicio da regiao de incerteza;
    sem ele (criterios fora da NHO-09) ficam tres faixas.

    O nivel de acao em si ainda e aceitavel ("0 a 0,5"); o limite em si
    ainda esta na regiao de incerteza ("0,9 a 1,1")."""
    if value > limit:
        return ABOVE_LIMIT
    if uncertain is not None and value >= uncertain:
        return UNCERTAIN
    if value > action:
        return ABOVE_ACTION
    return ACCEPTABLE


_CACHE = []                  # [(blocks, chave, resultado)], ultimos 4


def weighted_blocks(blocks, dt, pad_factor=3, weights=None):
    """Versao com cache: a mesma ponderacao dos mesmos blocos e pedida por
    varios criterios ao montar o resumo, e custa segundos num dia inteiro."""
    runs = tuple(continuity.runs_of(blocks))
    key = (tuple(weights or WEIGHTS), dt, pad_factor, runs)
    for b, k, res in _CACHE:
        if b is blocks and k == key:
            return res
    res = _weighted_blocks(blocks, dt, pad_factor, weights, runs)
    _CACHE.append((blocks, key, res))
    del _CACHE[:-4]
    return res


def _responses(weights, f):
    # None (usado pelas ponderacoes de saude do encosto/pe, que deixam y/z ou
    # todos os eixos sem filtro) significa passagem direta: resposta 1 em
    # toda a frequencia, o mesmo criterio que core._axis_gain ja usa.
    return np.stack([np.ones_like(f, dtype=complex) if w is None
                     else core.weighting_response(w, f)
                     for w in (weights or WEIGHTS)], axis=1)   # (nf, 3)


def _weighted_blocks(blocks, dt, pad_factor=3, weights=None, runs=()):
    """Ponderacao da ISO 2631-1.

    Faixas continuas (continuity.runs_of) sao filtradas como um sinal so,
    em filter_stream. Blocos isolados, como os do formato .m antigo, sao
    filtrados um a um, acolchoados com zeros: sem isso a filtragem no
    dominio da frequencia e circular, e com blocos de 2 s a constante de
    tempo do passa-alta de 0,4 Hz faz a resposta enrolar de uma ponta do
    bloco para a outra.
    """
    n_blk, n, _ = blocks.shape
    out = np.empty_like(blocks)
    isolated = np.ones(n_blk, dtype=bool)
    for a, b in runs:
        x = blocks[a:b].reshape(-1, 3)
        out[a:b] = filter_stream(x, dt, weights).reshape(b - a, n, 3)
        isolated[a:b] = False

    idx = np.flatnonzero(isolated)
    if idx.size:
        m = n * (1 + pad_factor)
        H = _responses(weights, np.fft.rfftfreq(m, dt))
        step = 256                   # lotes de blocos: rapido sem estourar memoria
        for i in range(0, idx.size, step):
            sel = idx[i:i + step]
            X = np.fft.rfft(blocks[sel], m, axis=1)
            out[sel] = np.fft.irfft(X * H[None], m, axis=1)[:, :n]
    return out


STREAM_CHUNK = 1 << 18          # amostras por trecho processado
STREAM_PREROLL_S = 60.0         # historico anterior usado em cada trecho


def filter_stream(x, dt, weights=None):
    """Ponderacao de um sinal continuo (n, 3), por trechos com sobreposicao.

    Cada trecho e filtrado junto com STREAM_PREROLL_S segundos de sinal
    anterior, descartados depois. A resposta ao impulso das ponderacoes e
    causal e decai bem antes disso (o passa-alta mais lento, 0,08 Hz do Wf,
    tem constante de tempo de ~3 s), entao o resultado e o de filtrar o
    registro inteiro de uma vez, sem a memoria que isso exigiria num dia de
    oito horas. O inicio do registro e tratado como partindo do repouso,
    o mesmo que o acolchoamento com zeros dos blocos isolados.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    pre = int(round(STREAM_PREROLL_S / dt))
    xp = np.concatenate([np.zeros((pre, x.shape[1])), x])
    out = np.empty_like(x)
    cache = {}
    for start in range(0, n, STREAM_CHUNK):
        stop = min(n, start + STREAM_CHUNK)
        seg = xp[start:stop + pre]               # pre-roll + trecho
        m = next_fast_len(len(seg) + pre)        # folga contra o enrolamento
        if m not in cache:
            cache[m] = _responses(weights, np.fft.rfftfreq(m, dt))
        y = np.fft.irfft(np.fft.rfft(seg, m, axis=0) * cache[m], m, axis=0)
        out[start:stop] = y[pre:pre + stop - start]
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
    def aren_rounded(self):
        return round_aren(self.aren)

    @property
    def verdict_aren(self):
        return classify(self.aren_rounded, ACTION_AREN, LIMIT_AREN,
                        UNCERTAIN_AREN)

    @property
    def verdict_vdvr(self):
        return classify(self.vdvr, ACTION_VDVR, LIMIT_VDVR, UNCERTAIN_VDVR)


def assess(blocks, dt, exposure_hours, moving_mask=None):
    """Avaliacao NHO-09 de um conjunto de blocos.

    blocks          (n_blocos, n_amostras, 3), aceleracao em m/s2, ja sem a
                    componente estatica e com os eixos no sistema basicentrico
    exposure_hours  T_exp: tempo real de exposicao no dia, em horas

    O dia inteiro e tratado como uma unica componente de exposicao. O am_j e
    agregado pela energia de todos os blocos, o que equivale a uma medicao
    continua -- e o que as medicoes atuais sao. Ver arithmetic_bias.
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

    vdv = (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25     # VDV_j do registro
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
    """Quanto a media aritmetica por blocos difere da media pela energia."""
    return result.amr_arithmetic / result.amr - 1.0


def running_rms_trace(blocks, dt, tau=1.0):
    """RMS corrente numa janela de tau segundos do sinal ponderado, com a
    soma vetorial e os fatores de amr. Formato (n_blk, n_amostras): o valor
    na amostra j e o da janela que TERMINA em j.

    Em faixas continuas a janela atravessa as fronteiras entre blocos; so
    as primeiras amostras de cada faixa (antes de completar uma janela)
    ficam NaN. Blocos isolados (.m antigo) nunca colam com o vizinho: cada
    um tem as suas primeiras win-1 amostras em NaN.
    """
    w = weighted_blocks(blocks, dt)
    n_blk, n, _ = w.shape
    win = max(1, min(n, int(round(tau / dt))))
    out = np.full((n_blk, n), np.nan)
    for a, b in continuity.segments(n_blk, continuity.runs_of(blocks)):
        x = w[a:b].reshape(-1, 3)
        c = np.concatenate([np.zeros((1, 3)), np.cumsum(x ** 2, axis=0)])
        ms = (c[win:] - c[:-win]) / win                  # janelas completas
        vec = np.sqrt(np.sum(FACTORS ** 2 * ms, axis=1))
        flat = np.full(len(x), np.nan)
        flat[win - 1:] = vec
        out[a:b] = flat.reshape(b - a, n)
    return out


def mtvv(blocks, dt, tau=1.0):
    """Maximum Transient Vibration Value (ISO 2631-1 6.4.2): maior RMS
    corrente (janela de tau s) do sinal ponderado, combinado por soma
    vetorial com os mesmos fatores de amr. Em registros continuos a janela
    corre pelo registro inteiro; blocos isolados nao sao colados."""
    trace = running_rms_trace(blocks, dt, tau)
    return float(np.nanmax(trace)) if np.isfinite(trace).any() else float('nan')


INTERVAL_CHOICES_S = (1, 10, 30, 60, 300, 600)


def interval_metrics(blocks, dt, interval_s, seconds=None, tau=1.0):
    """Valores por intervalo ao longo do registro: a ponderacao e o MTVV sao
    os do registro inteiro (o filtro e a janela de 1 s atravessam os limites
    dos intervalos), so a integracao e' que zera a cada intervalo. Assim a
    soma de a_w^2 * duracao de todos os intervalos da' a energia global.

    Os intervalos nao atravessam faixas descontinuas (continuity.segments):
    cada faixa comeca um intervalo novo, e o ultimo de cada faixa pode ser
    mais curto. seconds e' o instante de cada bloco (FieldDay.seconds); sem
    ele o tempo conta do inicio do primeiro bloco.

    Devolve um dict de arrays, uma linha por intervalo: start, duration,
    aw (n, 3), amr (soma vetorial com FACTORS), vdv (n, 3), vdv_total
    (soma de quarta potencia com FACTORS), mtvv (RMS corrente com soma
    vetorial, maior valor dentro do intervalo; NaN se nenhuma janela de 1 s
    termina nele), peak (n, 3) e crest (n, 3)."""
    w = weighted_blocks(blocks, dt)
    trace = running_rms_trace(blocks, dt, tau)
    n_blk, n, _ = w.shape
    if seconds is None:
        seconds = np.arange(n_blk) * n * dt
    step = max(1, int(round(interval_s / dt)))
    rows = {k: [] for k in ('start', 'duration', 'aw', 'vdv', 'mtvv', 'peak')}
    for a, b in continuity.segments(n_blk, continuity.runs_of(blocks)):
        x = w[a:b].reshape(-1, 3)
        tr = trace[a:b].reshape(-1)
        full = len(x) // step * step
        # intervalos completos de uma vez, o resto (mais curto) a parte
        pieces = [(x[:full].reshape(-1, step, 3), tr[:full].reshape(-1, step))]
        if full < len(x):
            pieces.append((x[full:][None], tr[full:][None]))
        start = 0
        for seg, part in pieces:
            k, m = seg.shape[0], seg.shape[1]
            if k == 0:
                continue
            rows['start'].append(seconds[a] + (start + np.arange(k) * m) * dt)
            rows['duration'].append(np.full(k, m * dt))
            rows['aw'].append(np.sqrt((seg ** 2).mean(axis=1)))
            rows['vdv'].append((np.sum(seg ** 4, axis=1) * dt) ** 0.25)
            finite = np.isfinite(part).any(axis=1)
            mt = np.full(k, np.nan)
            mt[finite] = np.nanmax(part[finite], axis=1)
            rows['mtvv'].append(mt)
            rows['peak'].append(np.abs(seg).max(axis=1))
            start += k * m
    out = {}
    for key, vals in rows.items():
        cols = 3 if key in ('aw', 'vdv', 'peak') else None
        out[key] = (np.concatenate(vals) if vals else
                    np.zeros((0, 3)) if cols else np.zeros(0))
    out['amr'] = np.sqrt(np.sum((FACTORS * out['aw']) ** 2, axis=1))
    out['vdv_total'] = np.sum((FACTORS * out['vdv']) ** 4, axis=1) ** 0.25
    out['crest'] = np.divide(out['peak'], out['aw'],
                             out=np.full(out['aw'].shape, np.nan),
                             where=out['aw'] > 0)
    return out


def interval_csv_rows(m, clock=True):
    """Linhas (cabecalho primeiro) para exportar interval_metrics em CSV.
    clock: start em hora do dia (HH:MM:SS) alem dos segundos."""
    head = ['start_s'] + (['start_clock'] if clock else []) + [
        'duration_s', 'aw_x', 'aw_y', 'aw_z', 'amr', 'vdv_x', 'vdv_y',
        'vdv_z', 'vdv_total', 'mtvv', 'peak_x', 'peak_y', 'peak_z',
        'crest_x', 'crest_y', 'crest_z']
    out = [head]
    for i in range(len(m['start'])):
        t = m['start'][i]
        row = [f'{t:.3f}']
        if clock:
            s = int(t) % 86400
            row.append(f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}')
        row += [f'{m["duration"][i]:.3f}', *(f'{v:.5g}' for v in m['aw'][i]),
                f'{m["amr"][i]:.5g}', *(f'{v:.5g}' for v in m['vdv'][i]),
                f'{m["vdv_total"][i]:.5g}', f'{m["mtvv"][i]:.5g}',
                *(f'{v:.5g}' for v in m['peak'][i]),
                *(f'{v:.4g}' for v in m['crest'][i])]
        out.append(row)
    return out


def axis_ratios(blocks, dt, tau=1.0):
    """Razoes de decisao da ISO 2631-1 (core.decision_ratios) por eixo, com
    Wd, Wd, Wk sobre o tempo medido. O MTVV por eixo corre pelas faixas
    continuas (sem colar blocos isolados), como em running_rms_trace."""
    w = weighted_blocks(blocks, dt)
    n_blk, n, _ = w.shape
    aw = np.sqrt((w ** 2).mean(axis=(0, 1)))
    peak = np.abs(w).max(axis=(0, 1))
    vdv = (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25
    mtvv_axis = np.zeros(3)
    for a, b in continuity.segments(n_blk, continuity.runs_of(blocks)):
        mtvv_axis = np.maximum(mtvv_axis, core.running_rms_max(
            w[a:b].reshape(-1, 3), dt, tau))
    duration = n_blk * n * dt
    out = core.decision_ratios(aw, peak, vdv, mtvv_axis, duration)
    out.update(aw=aw, peak=peak, vdv=vdv, mtvv=mtvv_axis, duration=duration)
    return out


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
