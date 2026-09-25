"""Reproducao do metodo do software em MATLAB do laboratorio (WBV_HSE_v16).

NAO NORMATIVO. Existe para comparacao lado a lado: dada a mesma configuracao,
reproduz os numeros daquele software, inclusive os seus defeitos, para que as
duas ferramentas possam ser conferidas uma contra a outra.

Etapas reproduzidas:
  ConverteDados   subtrai a media de cada bloco em cada eixo, divide por 16384
  LeArquivos      multiplica por 9,81
  Gera_Isofil     ponderacao por transformacao bilinear, filtragem causal
                  bloco a bloco, com condicao inicial nula
  CalculaawRMS    RMS de subintervalos e media quadratica deles
  CalculaA8       k a_w sqrt(T/8h), T = ultimo menos primeiro carimbo do arquivo
  Total A(8)      por eixo, sqrt(soma dos A(8) parciais ao quadrado); diario pela
                  soma vetorial dos tres eixos
  VDV             (soma a_w^4)^(1/4), SEM o passo de tempo; parcial com
                  exposicao fixa de 1 h; diario pela raiz da soma dos quadrados

Os dois defeitos do VDV podem ser reproduzidos ou corrigidos.

Mapeamento de eixo: LabConfig.axis_mode 'raw' e 'swap' reproduzem os dois
estados do checkbox "Exchange x-z axis" do app literalmente (colunas na
ordem do arquivo, ou colunas 1 e 3 trocadas -- sempre fixo, nunca olha a
gravidade). 'auto' (padrao) detecta a coluna vertical pela gravidade em
cada arquivo; e mais correto fisicamente, mas SO e igual ao app quando a
gravidade calha de estar na coluna que 'raw' ou 'swap' assumem. Um valor de
eixo que bate exatamente entre este software e o app noutro eixo mas diverge
nesse e sinal de que a orientacao real do arquivo nao e a que 'auto' ou o
estado do checkbox do app estao assumindo -- confira os dois modos fixos
antes de suspeitar da ponderacao (Wc/Wd/Wk ja foram conferidos byte a byte
contra Gera_Isofil.m, ver tests/).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import bilinear, butter, lfilter

from . import fielddata as fd

# Parametros do Gera_Isofil.m: f1 f2 f3 f4 f5 f6 Q4 Q5 Q6
_PARAMS = {
    'wb': (0.4, 100, 16, 16, 2.5, 4, 0.55, 0.9, 0.95),
    'wc': (0.4, 100, 8, 8, np.inf, np.inf, 0.63, 1, 1),
    'wd': (0.4, 100, 2, 2, np.inf, np.inf, 0.63, 1, 1),
    'we': (0.4, 100, 1, 1, np.inf, np.inf, 0.63, 1, 1),
    'wf': (0.08, 0.63, np.inf, 0.25, 0.0625, 0.10, 0.86, 0.80, 0.80),
    'wj': (0.4, 100, np.inf, np.inf, 3.75, 5.32, 1, 0.91, 0.91),
    'wk': (0.4, 100, 12.5, 12.5, 2.37, 3.35, 0.63, 0.91, 0.91),
}
WEIGHTINGS = [None] + list(_PARAMS)


def _filters(name, fs):
    f1, f2, f3, f4, f5, f6, q4, q5, q6 = _PARAMS[name]
    nyq = fs / 2
    if nyq <= f2:
        raise ValueError(
            f'Sample rate too low for the "{name}" weighting filter: it needs '
            f'frequencies up to {f2:g} Hz, which requires a sample rate above '
            f'{2 * f2:g} Hz (Nyquist). Got {fs:g} Hz. Check the sample rate '
            f'entered for this recording.')
    w3, w4, w5, w6 = (2 * np.pi * f for f in (f3, f4, f5, f6))
    out = [butter(2, f2 / nyq), butter(2, f1 / nyq, 'high')]
    if np.isfinite(w4):
        num = [1 / w3, 1] if np.isfinite(w3) else [0, 1.0]
        out.append(bilinear(num, [1 / w4 ** 2, 1 / q4 / w4, 1], fs))
    if np.isfinite(w5) and np.isfinite(w6):
        out.append(bilinear(np.array([1 / w5 ** 2, 1 / q5 / w5, 1]) * w5 ** 2 / w6 ** 2,
                            [1 / w6 ** 2, 1 / q6 / w6, 1], fs))
    return out


def _weight(x, name, fs, cache):
    """Filtra cada linha de x de forma independente, com condicao inicial
    nula -- o mesmo que o app faz bloco a bloco, mas todos de uma vez."""
    if name is None:
        return x                     # "None" do app: sinal bruto, sem banda
    if name not in cache:
        cache[name] = _filters(name, fs)
    for b, a in cache[name]:
        x = lfilter(b, a, x, axis=-1)
    return x


@dataclass
class LabConfig:
    weights: tuple = ('wc', None, None)      # eixos x, y, z do app
    k: tuple = (0.8, 1.0, 1.0)
    # 'auto'  coluna vertical detectada pela gravidade em cada arquivo (o
    #         criterio que fielddata.set_orientation tambem usa) -- NAO e o
    #         que o app faz, e um aprimoramento deste software;
    # 'raw'   colunas do arquivo na ordem em que estao (x=1, y=2, z=3), sem
    #         nenhuma checagem de gravidade -- o app com "Exchange x-z axis"
    #         DESMARCADO, seu estado padrao;
    # 'swap'  troca fixa x<->z (colunas 3,2,1) -- o app com "Exchange x-z
    #         axis" MARCADO.
    # 'auto' e mais correto quando a orientacao varia entre arquivos, mas so
    # 'raw'/'swap' reproduzem o app literalmente: ele nunca olha a
    # gravidade, so o estado do checkbox.
    axis_mode: str = 'auto'
    rms_time: float = 2.0
    reproduce_vdv_defects: bool = False

    @property
    def force_fixed_axes(self):
        # Compatibilidade com o nome antigo (usado antes de existir 'raw').
        return self.axis_mode == 'swap'


@dataclass
class LabResult:
    files: list = field(default_factory=list)   # por arquivo: aw, A8, VDV, duracao
    a8_axis: np.ndarray = None
    a8_daily: float = 0.0
    vdv_axis: np.ndarray = None
    vdv_daily: float = 0.0
    vdv_axis_corrected: np.ndarray = None
    vdv_daily_corrected: float = 0.0
    display: list = None             # coluna do arquivo em cada eixo do app


def _record_display(rec, axis_mode):
    """Mapeamento eixo conceitual (x, y, z) -> coluna bruta do arquivo.

    O app original NUNCA olha a gravidade: ele so aplica a ponderacao as
    colunas na ordem em que estao no arquivo (x=coluna 1, y=coluna 2, z=
    coluna 3), a menos que "Exchange x-z axis" esteja marcado, caso em que
    troca as colunas 1 e 3 -- sempre a mesma troca fixa, para todo arquivo,
    non importa o que a gravidade diga. 'raw' e 'swap' reproduzem os dois
    estados desse checkbox literalmente.

    'auto' e um modo A MAIS, deste software: detecta a coluna vertical pela
    gravidade em cada arquivo (rec.gravity_axis), o mesmo criterio de
    fielddata.set_orientation. E mais correto quando a orientacao muda entre
    arquivos ou dentro do mesmo dia (Secoes 10.8 e 10.10 da documentacao),
    mas diverge do app sempre que a coluna com a gravidade nao for a mesma
    que 'raw'/'swap' assumem -- por isso NAO reproduz o app, so se aproxima
    dele quando a orientacao de fato bate com a suposicao fixa.
    """
    if axis_mode == 'swap':
        return [2, 1, 0]
    if axis_mode == 'raw':
        return [0, 1, 2]
    grav = getattr(rec, 'gravity_axis', None)
    if grav is None:
        return [0, 1, 2]
    others = [c for c in range(3) if c != grav]
    return [others[0], others[1], grav]


def _axis_of_column(display):
    """Inverso de `display`: coluna bruta -> eixo conceitual (x, y, z)."""
    inv = [0, 0, 0]
    for axis, col in enumerate(display):
        inv[col] = axis
    return inv


def assess(records, cfg=LabConfig(), rate=fd.NOMINAL_RATE):
    """Aplica o metodo do laboratorio a registros brutos do registrador."""
    dt = 1.0 / rate
    cache = {}
    k = np.array(cfg.k, dtype=float)

    rows = []
    for rec in records:
        # A troca x-z inverte a coluna que recebe cada ponderacao e a posicao
        # em que o resultado e exibido; determinada por arquivo, nao global.
        display = _record_display(rec, cfg.axis_mode)
        axis_of_col = _axis_of_column(display)
        col_weight = {col: cfg.weights[axis_of_col[col]] for col in range(3)}
        k_col = k[axis_of_col]

        c = rec.counts.astype(float)
        a = (c - c.mean(axis=1, keepdims=True)) / 16384.0 * 9.81
        aw = np.empty_like(a)
        for col in range(3):
            aw[:, :, col] = _weight(a[:, :, col], col_weight[col], rate, cache)
        n_sub = max(1, int(round(rec.counts.shape[1] * dt / cfg.rms_time)))
        sub = aw.reshape(aw.shape[0], n_sub, -1, 3)
        awrms = np.sqrt(np.mean(np.mean(sub ** 2, axis=2), axis=(0, 1)))
        dur = (rec.seconds[-1] - rec.seconds[0]) / 3600.0
        a8 = k_col * awrms * np.sqrt(dur / 8.0)

        flat = aw.reshape(-1, 3)
        s4 = np.sum(flat ** 4, axis=0)
        mvdv_bug = s4 ** 0.25                        # sem dt, como no app
        mvdv_ok = (s4 * dt) ** 0.25
        recorded = rec.n_blocks * rec.counts.shape[1] * dt / 3600.0
        pvdv_bug = k_col * mvdv_bug * (1.0 / max(dur, 1e-9)) ** 0.25
        pvdv_ok = k_col * mvdv_ok * (max(dur, 1e-9) / recorded) ** 0.25
        rows.append(dict(name=rec.path.name, aw=awrms[display], a8=a8[display],
                         vdv_bug=pvdv_bug[display], vdv_ok=pvdv_ok[display],
                         hours=dur, crest=(np.abs(flat).max(0) / np.maximum(
                             awrms, 1e-12))[display], display=display))

    A = np.array([r['a8'] for r in rows])
    a8_axis = np.sqrt((A ** 2).sum(axis=0))
    Vb = np.array([r['vdv_bug'] for r in rows])
    Vo = np.array([r['vdv_ok'] for r in rows])
    vdv_bug = (Vb ** 4).sum(axis=0) ** 0.25
    vdv_ok = (Vo ** 4).sum(axis=0) ** 0.25

    # coluna do primeiro arquivo mostrada em cada eixo do app, para alinhar
    # comparacoes (o painel avisa quando a orientacao varia entre arquivos)
    res = LabResult(files=rows, a8_axis=a8_axis,
                    a8_daily=float(np.sqrt((a8_axis ** 2).sum())))
    res.display = rows[0]['display'] if rows else [0, 1, 2]
    res.vdv_axis_corrected = vdv_ok
    res.vdv_daily_corrected = float(vdv_ok.max())      # maior eixo, diretiva
    if cfg.reproduce_vdv_defects:
        res.vdv_axis = vdv_bug
        res.vdv_daily = float(np.sqrt((vdv_bug ** 2).sum()))
    else:
        res.vdv_axis = vdv_ok
        res.vdv_daily = res.vdv_daily_corrected
    return res
