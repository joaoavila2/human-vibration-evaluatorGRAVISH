"""Geração de ficha PDF — ISO 2631-5:2018 Clause 5 / Annex C.

Funções exportadas:
  weibull_figure(result)           → matplotlib.figure.Figure
  export_to_pdf(result, path_out, meta, daily_hours, n_years, days, ...)

`meta` é um dict com chaves opcionais:
  subject_id, date, operator, notes
"""
from __future__ import annotations

from datetime import datetime as _dt
from typing import Sequence

import numpy as np
import matplotlib as _mpl
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import brentq

from . import annexc
from .theme import OK_COLOR, WARN_COLOR, BAD_COLOR

__all__ = ['weibull_figure', 'export_to_pdf', 'reference_r']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_r(sex: str) -> tuple[float, float, float]:
    """R nos limiares Π = 10 / 50 / 90 % (Tabela C.2) para `sex`."""
    cfg = annexc.SEX_DEFAULTS[sex]
    return tuple(
        brentq(lambda R: annexc.injury_probability(R, cfg['alpha'], cfg['beta']) - p,
               0.01, 10.0)
        for p in (0.10, 0.50, 0.90)
    )


def _risk_colour(result) -> str:
    return (OK_COLOR if result.probability < annexc.RISK_LOW else
            BAD_COLOR if result.probability > annexc.RISK_HIGH else WARN_COLOR)


def _risk_bg(result) -> str:
    return ('#d4ecd9' if result.probability < annexc.RISK_LOW else
            '#f6cfcc' if result.probability > annexc.RISK_HIGH else '#fbe8b8')


# ---------------------------------------------------------------------------
# Curva de Weibull
# ---------------------------------------------------------------------------

def weibull_figure(result) -> _mpl.figure.Figure:
    """Figura matplotlib com a curva Π(R) e a marcação do resultado atual.

    Chame plt.close(fig) quando terminar para liberar memória.
    """
    r = result
    cfg = annexc.SEX_DEFAULTS[r.sex]
    alpha, beta = cfg['alpha'], cfg['beta']
    r10, r50, r90 = reference_r(r.sex)

    R_max = max(r90 * 1.30, r.R * 1.20, 3.0)
    R_arr = np.linspace(0, R_max, 1200)
    Pi = (1.0 - np.exp(-((R_arr / alpha) ** beta))) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')

    # Zonas coloridas de fundo
    ax.axhspan(0,   10,  alpha=0.40, color='#d4ecd9', zorder=0)
    ax.axhspan(10,  50,  alpha=0.40, color='#fbe8b8', zorder=0)
    ax.axhspan(50,  100, alpha=0.40, color='#f6cfcc', zorder=0)

    # Curva
    ax.plot(R_arr, Pi, lw=2.2, color='#2c5f8a', zorder=3,
            label=f'Weibull — {r.sex} (α={alpha:.3f}, β={beta:.3f})')

    # Linhas de limiar
    for R_thr, pct, lbl in [(r10, 10, '10 %'), (r50, 50, '50 %'), (r90, 90, '90 %')]:
        ax.axvline(R_thr, ls='--', lw=0.9, color='#777', zorder=2)
        ax.axhline(pct,   ls='--', lw=0.9, color='#777', zorder=2)
        ax.text(R_thr, 103, f'R={R_thr:.2f}\n{lbl}',
                ha='center', fontsize=8, color='#555', linespacing=1.3)

    # Marcador da avaliação atual
    colour = _risk_colour(r)
    pi_pct = r.probability * 100.0
    ax.plot(r.R, pi_pct, marker='D', markersize=12, color=colour,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f'Esta avaliação: R={r.R:.3f}, Π={pi_pct:.1f} %')

    offset_x = R_max * 0.07
    offset_y = -14 if pi_pct > 50 else 10
    ax.annotate(
        f'R = {r.R:.3f}\nΠ = {pi_pct:.1f} %',
        xy=(r.R, pi_pct),
        xytext=(r.R + offset_x, pi_pct + offset_y),
        fontsize=9.5, fontweight='bold', color=colour, zorder=6,
        arrowprops=dict(arrowstyle='->', color=colour, lw=1.3),
    )

    ax.set_xlim(0, R_max)
    ax.set_ylim(0, 112)
    ax.set_xlabel('R  (variável de tensão, adimensional)', fontsize=10)
    ax.set_ylabel('Π  (probabilidade de lesão lumbar) [%]', fontsize=10)
    ax.set_title(
        f'Curva de Weibull — ISO 2631-5:2018 Annexo C, Tabela C.1 ({r.sex})',
        fontsize=10,
    )
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Primeira página da ficha
# ---------------------------------------------------------------------------

def _results_page(result, meta: dict, daily_hours: float, n_years: int, days: int
                  ) -> _mpl.figure.Figure:
    r = result
    r10, r50, r90 = reference_r(r.sex)
    colour   = _risk_colour(r)
    risk_bg  = _risk_bg(r)

    BLACK = 'black'; WHITE = 'white'; GREY = '#f2f2f2'
    RULE = '#333'; TXT = 'black'; MUTED = '#555'
    FOOTER_Y = 0.024

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def Box(x, y, w, h, fc, ec=None, lw=0.5):
        if y < FOOTER_Y:
            return
        ax.add_patch(mp.Rectangle(
            (x, y), w, h, facecolor=fc, edgecolor=(ec or fc),
            linewidth=lw, clip_on=False, transform=ax.transAxes, zorder=1))

    def Txt(x, y, s, **kw):
        if y < FOOTER_Y:
            return
        ax.text(x, y, s, transform=ax.transAxes, clip_on=False, zorder=2, **kw)

    def hline(y, lw=0.8, color=RULE):
        ax.axhline(y, lw=lw, color=color, clip_on=False, zorder=1)

    # ---- Cabeçalho --------------------------------------------------------
    Box(0, 0.960, 1, 0.040, BLACK)
    Txt(0.016, 0.980,
        'Avaliador de Vibração Humana — Clause 5 / Annex C (ISO 2631-5)',
        color=WHITE, fontsize=9.5, fontweight='bold', va='center')
    Txt(0.984, 0.980, _dt.now().strftime('%Y-%m-%d  %H:%M'),
        color='#ccc', fontsize=7.5, va='center', ha='right')
    hline(0.960, lw=1.2, color=BLACK)

    meta_str = '  ·  '.join(
        v for v in [meta.get('subject_id'), meta.get('date'), meta.get('operator')]
        if v)
    if meta_str:
        Txt(0.016, 0.948, meta_str, color=MUTED, fontsize=8, va='center')
        hline(0.934, lw=0.5)

    y = 0.918

    # ---- Resultado principal ----------------------------------------------
    Txt(0.016, y, 'Resultado — Clause 5 / Annex C (ISO 2631-5)',
        color=BLACK, fontsize=9, fontweight='bold', va='bottom')
    hline(y - 0.004, lw=0.8); y -= 0.024

    Txt(0.016, y, f'Π = {r.probability * 100:.1f}%   R = {r.R:.3f}',
        color=colour, fontsize=15, fontweight='bold', va='top')
    Txt(0.016, y - 0.036, r.verdict, color=TXT, fontsize=9, va='top')
    y -= 0.062

    # ---- Tabela de resultados --------------------------------------------
    Txt(0.016, y, 'Tabela de resultados',
        color=BLACK, fontsize=8.5, fontweight='bold', va='bottom')
    hline(y - 0.003, lw=0.6); y -= 0.010

    rows = [
        ('D_z   dose de aceleração [m/s²]',      f'{r.Dz:.4f}',               'Fórmula (3)'),
        ('D_zd  dose diária [m/s²]',              f'{r.Dzd:.4f}',              'Fórmulas (4), (5)'),
        ('S_d   dose de compressão diária [MPa]', f'{r.Sd:.4f}',               'Fórmula (C.1)'),
        ('R     variável de tensão',              f'{r.R:.4f}',                'Fórmula (C.3)'),
        ('Π     probabilidade de lesão',          f'{r.probability * 100:.2f} %', 'Fórmula (C.5)'),
        ('m_z   [MPa/(m/s²)]',                   f'{r.mz:.5f}',               'Fórmula (C.2)'),
        ('S_stat [MPa]',                          f'{r.s_stat:.4f}',           'm_z × 9,81'),
        ('Picos contados',                        f'{len(r.peaks):,}',         'positivos, entre cruzamentos por zero'),
    ]
    if r.discarded_seconds > 0:
        rows.append(('Descartado (perda de contato) [s]',
                     f'{r.discarded_seconds:.1f}',
                     f'{len(r.contact_intervals)} trecho(s), heurística'))
    rows += [
        ('', '', ''),
        (f'R em Π = 10% / 50% / 90% ({r.sex})',
         f'{r10:.3f} / {r50:.3f} / {r90:.3f}',
         'Tabela C.2 — baixo / moderado / alto'),
    ]

    ROW_H = 0.021
    for ri, (lbl, val, src) in enumerate(rows):
        if not lbl:
            y -= ROW_H * 0.25; continue
        is_key = lbl.startswith('R     ') or lbl.startswith('Π     ')
        bg = risk_bg if is_key else (GREY if ri % 2 == 0 else WHITE)
        Box(0, y - ROW_H, 1, ROW_H, bg, RULE, 0.3)
        for cx, txt in zip([0.016, 0.50, 0.76], (lbl, val, src)):
            Txt(cx, y - ROW_H * 0.6, txt, color=TXT, fontsize=7.5, va='center')
        y -= ROW_H

    # ---- Parâmetros de exposição ----------------------------------------
    y -= 0.012
    Txt(0.016, y, 'Parâmetros de exposição',
        color=BLACK, fontsize=8.5, fontweight='bold', va='bottom')
    hline(y - 0.003, lw=0.6); y -= 0.010
    for ri, (lbl, val) in enumerate([
        ('Sexo',                     r.sex),
        ('Exposição diária',         f'{daily_hours:.1f} h/dia'),
        ('Dias por ano',             str(days)),
        ('Anos de exposição',        str(n_years)),
        ('Idade no início',          str(r.start_age)),
    ]):
        bg = GREY if ri % 2 == 0 else WHITE
        Box(0, y - ROW_H, 0.5, ROW_H, bg, RULE, 0.3)
        Box(0.5, y - ROW_H, 0.5, ROW_H, bg, RULE, 0.3)
        Txt(0.016, y - ROW_H * 0.6, lbl, color=MUTED, fontsize=7.5, va='center')
        Txt(0.516, y - ROW_H * 0.6, val, color=TXT, fontsize=7.5, va='center')
        y -= ROW_H

    if meta.get('notes'):
        y -= 0.008
        Txt(0.016, y, f'Obs.: {meta["notes"]}', color=MUTED, fontsize=7, va='top')

    # ---- Rodapé -----------------------------------------------------------
    hline(FOOTER_Y, lw=0.6)
    Txt(0.016, FOOTER_Y / 2,
        'Avaliador de Vibração Humana · ISO 2631-5:2018 Clause 5 / Annex C',
        color=MUTED, fontsize=6.5, va='center')
    Txt(0.984, FOOTER_Y / 2, 'GRAVI-SH · UFMG',
        color=MUTED, fontsize=6.5, ha='right', va='center')

    return fig


# ---------------------------------------------------------------------------
# Exportação
# ---------------------------------------------------------------------------

def export_to_pdf(
    result,
    path_out: str,
    meta: dict,
    daily_hours: float,
    n_years: int,
    days: int,
    extra_figs: Sequence = (),
) -> None:
    """Gera a ficha PDF em `path_out`.

    Páginas:
      1. Resultados + tabela + parâmetros de exposição
      2. Curva de Weibull (sempre incluída)
      3+. Figuras extras passadas em `extra_figs` (opcionais)

    Fecha todas as figuras criadas aqui; as de `extra_figs` não são fechadas.
    """
    rc_save = {k: _mpl.rcParams[k] for k in ['font.family']}
    try:
        _mpl.rcParams['font.family'] = 'serif'
        fig1 = _results_page(result, meta, daily_hours, n_years, days)
        fig_w = weibull_figure(result)
        with PdfPages(path_out) as pdf:
            pdf.savefig(fig1, bbox_inches='tight', dpi=150)
            plt.close(fig1)
            pdf.savefig(fig_w, bbox_inches='tight', dpi=150)
            plt.close(fig_w)
            for ef in extra_figs:
                ef.set_size_inches(11.69, 8.27)
                pdf.savefig(ef, bbox_inches='tight', dpi=150)
    finally:
        _mpl.rcParams.update(rc_save)
