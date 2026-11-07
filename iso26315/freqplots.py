"""Graficos de frequencia compartilhados entre as abas.

Cada aba tem seu proprio modelo de dados (exposicao de quatro pontos,
registro ASCII continuo, dia de campo em janelas de 2 s, sinal az da
Clausula 5). As funcoes aqui recebem so arrays e um objeto Plot, para que o
espectrograma (com cascata 3D) e o SRS desenhem igual em todo lugar.
"""
import numpy as np

from . import core

SRS_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
              '#8c564b']


def message(plot, text):
    """Mensagem centralizada quando o grafico nao pode ser desenhado."""
    ax, = plot.axes()
    ax.text(0.5, 0.5, text, ha='center', va='center', color='#6c757d',
            wrap=True, transform=ax.transAxes)
    plot.finish(ax)


ESTIMATOR_DEFAULTS = {'h2': False, 'hv': False, 'envelope': False,
                      'error': False}


def draw_transmissibility(ax1, ax2, f, est, cl, colour, label, settings=None):
    """|H1| (forte onde a coerencia passa do limite cl, apagado no resto)
    e a coerencia em ax2 (pode ser None). settings liga as camadas opcionais
    de core.frf_estimates: 'h2' e 'hv' (curvas), 'envelope' (faixa entre
    H1 e H2, onde esta o valor verdadeiro com ruido nos dois canais) e
    'error' (|H1| (1 +- 2 eps), cerca de 90 %, ver core.frf_estimates). As
    camadas so aparecem onde
    a coerencia passa do limite, como a linha forte do H1."""
    s = dict(ESTIMATOR_DEFAULTS, **(settings or {}))
    good = est['g2'] >= cl
    mag = np.abs(est['h1'])
    ax1.plot(f, mag, lw=0.7, color=colour, alpha=0.25)
    ax1.plot(f, np.where(good, mag, np.nan), lw=1.3, color=colour, label=label)
    if s['envelope']:
        h2 = np.abs(est['h2'])
        ax1.fill_between(f, np.minimum(mag, h2), np.maximum(mag, h2),
                         where=good, color=colour, alpha=0.18, lw=0,
                         label=f'{label}: H1-H2 envelope')
    if s['error']:
        e = np.clip(2 * est['eps'], 0, 0.99)
        ax1.fill_between(f, mag * (1 - e), mag * (1 + e), where=good,
                         facecolor='none', edgecolor=colour, hatch='////',
                         lw=0, alpha=0.5, label=f'{label}: H1 ± 2ε')
    if s['h2']:
        ax1.plot(f, np.where(good, np.abs(est['h2']), np.nan), lw=1.0,
                 ls='--', color=colour, label=f'{label}: H2')
    if s['hv']:
        ax1.plot(f, np.where(good, np.abs(est['hv']), np.nan), lw=1.0,
                 ls='-.', color=colour, label=f'{label}: Hv')
    if ax2 is not None:
        ax2.plot(f, est['g2'], lw=1.0, color=colour)


def estimator_note(settings):
    """Frase para a nota abaixo do grafico, so com as camadas ligadas."""
    s = dict(ESTIMATOR_DEFAULTS, **(settings or {}))
    parts = []
    if s['h2'] or s['envelope']:
        parts.append('H2 = Gyy/Gyx is exact with noise only on the input, '
                     'H1 with noise only on the output; with noise on both '
                     'the true value lies between them (H1/H2 = coherence)')
    if s['hv']:
        parts.append('Hv assumes equal noise on both channels')
    if s['error']:
        parts.append('hatched band: |H1|(1 ± 2ε), '
                     'ε = √(1-γ²)/(|γ|√(2n)) '
                     '(Bendat and Piersol); it covered about 90 % of the '
                     'true values in a synthetic check, not 95 %')
    return ('. '.join(parts) + '.') if parts else ''


def draw_spectrogram(plot, t, f, S, title, tlabel='Time [s]', waterfall=False,
                     logy=False, fmax=core.LOWPASS_HZ, unit='m/s$^2$',
                     max_lines=100):
    """Espectrograma (mapa de cor) ou cascata 3D de S[f, t] (amplitude)."""
    keep = f <= fmax
    fk = f[keep]
    z = S[keep]
    label = f'Amplitude [{unit}]'
    if logy:
        with np.errstate(divide='ignore'):
            z = 20 * np.log10(np.maximum(z, 1e-12))
        label = f'Level [dB re 1 {unit}]'
    if waterfall:
        from matplotlib import colormaps
        ax, = plot.axes(projection='3d')
        step = max(1, int(np.ceil(len(t) / max_lines)))
        idx = np.arange(0, len(t), step)
        lo = float(np.nanmin(z))
        span = (float(np.nanmax(z)) - lo) or 1.0
        cmap = colormaps['viridis']
        for j in idx[::-1]:
            level = (float(np.nanmax(z[:, j])) - lo) / span
            ax.plot(fk, np.full_like(fk, t[j]), z[:, j], lw=0.8,
                    color=cmap(min(1.0, max(0.0, level))))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(tlabel)
        ax.set_zlabel(label)
        ax.set_title(title)
        plot.finish(ax)
        return
    ax, = plot.axes()
    if len(t) == 1:
        # pcolormesh precisa de pelo menos duas colunas.
        t = np.array([t[0] - 0.5, t[0] + 0.5])
        z = np.repeat(z, 2, axis=1)
    mesh = ax.pcolormesh(t, fk, z, shading='auto', cmap='viridis')
    cb = plot.figure.colorbar(mesh, ax=ax)
    cb.set_label(label)
    ax.set_xlabel(tlabel)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    plot.finish(ax)


def draw_srs(plot, series, title, logy=True, show_primary_residual=False):
    """SRS de varios sinais.

    series = [(rotulo, freqs, srs_or_dict)]

    Quando show_primary_residual=False (padrao), srs_or_dict e' o array
    maximax; quando True, deve ser o dict de shock_response_spectra com
    as chaves 'primary', 'residual', 'maximax'. Nesse caso sao desenhadas
    tres curvas por sinal (primario tracejado, residual pontilhado, maximax
    solido) na mesma cor.
    """
    ax, = plot.axes()
    for i, (label, freqs, srs_or_dict) in enumerate(series):
        col = SRS_COLORS[i % len(SRS_COLORS)]
        if show_primary_residual and isinstance(srs_or_dict, dict):
            ax.plot(freqs, srs_or_dict['primary'], lw=1.0, ls='--',
                    color=col, label=f'{label}: primary')
            ax.plot(freqs, srs_or_dict['residual'], lw=1.0, ls=':',
                    color=col, label=f'{label}: residual')
            ax.plot(freqs, srs_or_dict['maximax'], lw=1.6, ls='-',
                    color=col, label=f'{label}: maximax')
        else:
            ax.plot(freqs, srs_or_dict, lw=1.3, color=col, label=label)
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Natural frequency [Hz]')
    ax.set_ylabel('SRS acceleration [m/s$^2$]')
    ax.set_title(title)
    if series:
        ax.legend(fontsize=8)
    plot.finish(ax)


SRS_NOTE = ('Maximax absolute-acceleration SRS, Q=10 (zeta=0.05), MIL-STD-810 '
            'convention: severity as a function of the responding '
            "structure's natural frequency, not a single RMS or peak value.")


# ----------------------------------------------------------------------
# Anotacoes automaticas de pico
# ----------------------------------------------------------------------

def annotate_transmissibility_peak(ax, f, h1, g2=None, cl=0.5,
                                   color='#212529', fmin=0.5):
    """Anota o pico de |H1| com fn, |H|max, Q estimado e zeta.

    Encontra o maximo de |H1| na faixa [fmin, inf) onde gamma^2 >= cl
    (ou em toda a faixa se g2 nao for fornecido). Estima Q pela largura
    de -3 dB interpolada por biseccao. Desenha uma seta e um texto no ax.

    Retorna o dict {'fn': float, 'H_max': float, 'Q': float, 'zeta': float}
    ou None se nao houver pico identificavel.
    """
    f = np.asarray(f, dtype=float)
    mag = np.abs(np.asarray(h1, dtype=complex))

    # mascara de frequencia valida
    mask = f >= fmin
    if g2 is not None:
        mask &= np.asarray(g2, dtype=float) >= cl
    if not mask.any():
        return None

    fi = f[mask]
    mi = mag[mask]
    peak_idx = int(np.argmax(mi))
    H_max = float(mi[peak_idx])
    fn = float(fi[peak_idx])
    if H_max <= 1.0:
        return None   # pico abaixo de 1 nao e ressonancia

    # estimativa de Q por largura de -3 dB (H_max / sqrt(2))
    half = H_max / np.sqrt(2.0)
    Q = None
    try:
        # cruza o limiar nos dois lados do pico
        left = mi[:peak_idx + 1]
        right = mi[peak_idx:]
        fl_arr = fi[:peak_idx + 1]
        fr_arr = fi[peak_idx:]
        # lado esquerdo: ultimo cruzamento abaixo de half antes do pico
        lo_cross = np.flatnonzero(left < half)
        # lado direito: primeiro cruzamento abaixo de half apos o pico
        hi_cross = np.flatnonzero(right < half)
        if lo_cross.size and hi_cross.size:
            i_lo = lo_cross[-1]
            f_lo = np.interp(half, left[i_lo:i_lo + 2],
                             fl_arr[i_lo:i_lo + 2])
            i_hi = hi_cross[0]
            f_hi = np.interp(half, right[i_hi:i_hi + 2][::-1],
                             fr_arr[i_hi:i_hi + 2][::-1])
            bw = f_hi - f_lo
            if bw > 0:
                Q = fn / bw
    except Exception:
        pass

    # texto de anotacao
    lines = [f'$f_n$ = {fn:.2f} Hz', f'|H|$_{{max}}$ = {H_max:.2f}']
    if Q is not None:
        zeta = 1.0 / (2.0 * Q)
        lines.append(f'Q ≈ {Q:.1f},  ζ ≈ {zeta:.3f}')

    # escolhe o lado do texto para nao sair do grafico
    xlim = ax.get_xlim()
    text_x = fn * 1.08 if fn < (xlim[1] * 0.65) else fn * 0.92
    ha = 'left' if fn < (xlim[1] * 0.65) else 'right'

    ax.annotate(
        '\n'.join(lines),
        xy=(fn, H_max),
        xytext=(text_x, H_max * 0.82),
        fontsize=8,
        color=color,
        ha=ha,
        va='top',
        arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color,
                  alpha=0.85, lw=0.8),
    )
    ax.axvline(fn, ls=':', lw=0.8, color=color, alpha=0.5)

    return {'fn': fn, 'H_max': H_max,
            'Q': Q, 'zeta': 1.0 / (2.0 * Q) if Q else None}


def annotate_srs_peak(ax, freqs, srs, color='#212529'):
    """Anota o pico do SRS maximax com a frequencia e o valor."""
    freqs = np.asarray(freqs, dtype=float)
    srs = np.asarray(srs, dtype=float)
    if not srs.size:
        return
    idx = int(np.argmax(srs))
    fn = float(freqs[idx])
    peak = float(srs[idx])
    xlim = ax.get_xlim()
    ha = 'left' if fn < np.sqrt(xlim[0] * xlim[1]) else 'right'
    text_x = fn * 1.15 if ha == 'left' else fn / 1.15
    ax.annotate(
        f'peak: {peak:.2f} m/s²\nat {fn:.1f} Hz',
        xy=(fn, peak),
        xytext=(text_x, peak * 0.75),
        fontsize=8, color=color, ha=ha, va='top',
        arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color,
                  alpha=0.85, lw=0.8),
    )
