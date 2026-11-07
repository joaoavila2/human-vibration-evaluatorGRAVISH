"""Estilo global de gráficos — Avaliador de Vibração Humana.

Chamar ``plot_style.apply()`` uma vez antes de criar a QApplication
(em run.py / main()). Todos os gráficos da aplicação — abas interativas,
exportações PDF, curva de Weibull — herdam o estilo automaticamente.

Estratégia:
  1. Aplica ``['science', 'no-latex']`` do SciencePlots como base:
     fonte STIXGeneral (clone do Computer Modern/LaTeX), ticks para dentro,
     linhas finas mas limpas, sem frame na legenda.
  2. Sobrepõe ajustes de tamanho/espessura adequados para exibição em
     tela (o estilo 'science' é otimizado para papel A4 impresso).
  3. Fallback silencioso se SciencePlots não estiver instalado: aplica
     os mesmos ajustes manualmente com a fonte STIX Two, que já vem
     embutida no matplotlib — o resultado é visualmente idêntico.
"""
from __future__ import annotations

import matplotlib as _mpl


# Sobrescritas de GUI aplicadas em cima do estilo base
_GUI_OVERRIDES: dict = {
    # ── Fonte ────────────────────────────────────────────────────────────
    'font.size':              10.5,
    'axes.labelsize':         10.5,
    'axes.titlesize':         11,
    'xtick.labelsize':        9.5,
    'ytick.labelsize':        9.5,
    'legend.fontsize':        9,
    'legend.title_fontsize':  9.5,

    # ── Linhas ──────────────────────────────────────────────────────────
    'lines.linewidth':        1.6,
    'axes.linewidth':         0.7,
    'xtick.major.width':      0.7,
    'ytick.major.width':      0.7,
    'xtick.minor.width':      0.45,
    'ytick.minor.width':      0.45,
    'xtick.major.size':       3.5,
    'ytick.major.size':       3.5,
    'xtick.minor.size':       2.0,
    'ytick.minor.size':       2.0,

    # ── Ticks ────────────────────────────────────────────────────────────
    'xtick.direction':        'in',
    'ytick.direction':        'in',
    'xtick.top':              True,
    'ytick.right':            True,

    # ── Spines (4 visíveis, como MATLAB) ─────────────────────────────────
    'axes.spines.top':        True,
    'axes.spines.right':      True,
    'axes.spines.left':       True,
    'axes.spines.bottom':     True,

    # ── Layout ───────────────────────────────────────────────────────────
    'legend.frameon':         False,
    'legend.handlelength':    1.5,
    'figure.facecolor':       'white',
    'axes.facecolor':         'white',
    'figure.dpi':             100,
    'savefig.dpi':            150,
    'savefig.facecolor':      'white',
    'savefig.bbox':           'tight',
}


def apply() -> None:
    """Aplica o estilo global de gráficos.

    Deve ser chamada uma única vez, antes da criação da QApplication.
    """
    try:
        import scienceplots  # noqa: F401 — importa para registrar os estilos
        import matplotlib.pyplot as _plt
        _plt.style.use(['science', 'no-latex'])
    except ImportError:
        # Fallback sem scienceplots: usa STIXTwoText (embutida no matplotlib)
        _mpl.rcParams.update({
            'font.family':      'STIXTwoText',
            'mathtext.fontset': 'stix',
        })

    # Sobrescritas de GUI (sempre aplicadas, com ou sem scienceplots)
    _mpl.rcParams.update(_GUI_OVERRIDES)
