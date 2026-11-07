"""Estudo controlado: a Clausula 5/Anexo C e aplicavel aos blocos de 2 s do
registrador de campo?

Motivacao: o Anexo A esta descartado para os arquivos .m por dois motivos
independentes ja documentados (ciclo de trabalho de 40% quebra a soma de
sexta potencia; falta de sincronismo entre unidades quebra a soma multiponto
da Formula A.1). A Clausula 5/Anexo C e single-point (so z no assento), entao
o segundo motivo nao se aplica a ela. Falta saber se o primeiro -- ter so 2 s
de sinal a cada 5 s -- inviabiliza tambem esse caminho, e por qual mecanismo.

Dois efeitos, isolados separadamente:

  1. Amostragem: picos que caem nos 60% nao gravados sao estruturalmente
     invisiveis. Isso e uma limitacao estatistica do instrumento, nao um
     defeito de calculo -- e a mesma limitacao que a NHO-09 ja aceita hoje
     para estes dados.
  2. Transiente de filtro: o passa-alta de 0,01 Hz da Clausula 5 (Butterworth,
     2 polos) tem periodo de 100 s -- cinquenta vezes o bloco de 2 s. Processar
     cada bloco isoladamente reinicia o filtro do zero a cada 2 s, o que pode
     distorcer o sinal condicionado bem alem do que a amostragem sozinha
     explicaria. Isso E um defeito de metodo, mitigavel (blocos maiores,
     descarte da franja de acomodacao, ou filtro nao-causal por bloco).

O estudo mede os dois separadamente, comparando tres processamentos do MESMO
sinal sintetico de resposta conhecida:

  (a) continuo        - a cadeia inteira rodada de uma vez sobre os T segundos
                         completos (verdade de referencia, dado que nunca
                         existira de fato para o registrador de campo).
  (b) colado sem gaps  - so as amostras realmente "gravadas" (os blocos
                         mantidos pelo ciclo de trabalho), concatenadas de
                         ponta a ponta e filtradas em UMA passada. Isola o
                         efeito (1): mesmas amostras do (a) que sobreviveriam
                         ao ciclo de trabalho, sem o problema do filtro.
  (c) por blocos       - os mesmos blocos de (b), mas cada um filtrado
                         independentemente, como a importacao direta teria
                         de fazer sobre arquivos .m separados. (c) vs (b)
                         isola o efeito (2) sozinho.

Uso: python tests/clause5_block_study.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315 import annexc as C

FS = 1024.0                 # taxa nominal do registrador de campo
DT = 1.0 / FS
BLOCK_S = 2.0                # bloco gravado
CYCLE_S = 5.0                # periodo do ciclo de trabalho (40%)
DUTY = BLOCK_S / CYCLE_S


def synth_day(duration_s, seed=0):
    """Vibracao de fundo (2-15 Hz) mais choques (guias/desniveis), continuo."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) * DT

    bg = rng.standard_normal(n)
    F = np.fft.rfft(bg)
    f = np.fft.rfftfreq(n, DT)
    F[(f < 2.0) | (f > 15.0)] = 0
    bg = np.fft.irfft(F, n)
    bg = bg / bg.std() * 0.6            # ~0,6 m/s2 eficaz, vibracao continua

    x = bg.copy()
    # Choques tipo "descer guia": queda livre curta seguida de pico positivo,
    # espalhados ao longo do dia sintetico, amplitude comparavel aos 21,3 m/s2
    # observados nos dados reais do laboratorio.
    n_shocks = max(1, int(duration_s / 12))
    centres = rng.uniform(0.05, 0.95, n_shocks) * n
    for c in centres:
        c = int(c)
        w = int(0.12 * FS)
        k = np.arange(-w, w + 1)
        freefall = -8.5 * np.exp(-((k + 0.4 * w) / (0.35 * w)) ** 2)
        rebound = 18.0 * np.exp(-((k - 0.5 * w) / (0.25 * w)) ** 2)
        pulse = freefall + rebound
        a, b = c - w, c + w + 1
        lo, hi = max(a, 0), min(b, n)
        x[lo:hi] += pulse[lo - a: hi - a]
    return t, x


def dose_continuous(x, dt):
    cond = C.condition(x, dt)
    resp = C.spine_response(cond, dt)
    peaks = C.positive_peaks(resp)
    return C.acceleration_dose(peaks), len(peaks)


def kept_blocks(x, dt, block_s, cycle_s):
    block_n = int(round(block_s / dt))
    cycle_n = int(round(cycle_s / dt))
    return [x[i:i + block_n] for i in range(0, len(x) - block_n + 1, cycle_n)]


def dose_stitched(x, dt, block_s, cycle_s):
    """(b): amostras mantidas, concatenadas, filtradas em uma so passada."""
    blocks = kept_blocks(x, dt, block_s, cycle_s)
    stitched = np.concatenate(blocks)
    return dose_continuous(stitched, dt) + (len(blocks),)


def dose_blocked(x, dt, block_s, cycle_s):
    """(c): cada bloco filtrado independentemente, como sobre arquivos .m."""
    blocks = kept_blocks(x, dt, block_s, cycle_s)
    all_peaks = []
    for b in blocks:
        cond = C.condition(b, dt)
        resp = C.spine_response(cond, dt)
        all_peaks.append(C.positive_peaks(resp))
    peaks = np.concatenate(all_peaks) if all_peaks else np.array([])
    return C.acceleration_dose(peaks), len(peaks), len(blocks)


def highpass_settling(dt, block_s):
    """Resposta ao degrau do passa-alta de 0,01 Hz: quanto do bloco e transiente?"""
    n = int(round(block_s / dt))
    step = np.ones(n)
    from scipy.signal import butter, sosfilt
    sos = butter(C.HIGHPASS_ORDER, C.HIGHPASS_HZ, 'highpass', fs=1 / dt,
                output='sos')
    y = sosfilt(sos, step)
    # tempo ate |y| cair e ficar abaixo de 5% do salto inicial
    settled = np.flatnonzero(np.abs(y) > 0.05)
    return (settled[-1] + 1) * dt if settled.size else 0.0


def main():
    print('=' * 78)
    print('0. Acomodacao do passa-alta de 0,01 Hz dentro de um bloco')
    print('=' * 78)
    for b in (2.0, 5.0, 10.0, 20.0, 30.0):
        t_settle = highpass_settling(DT, max(b, 30.0))
        frac = min(t_settle / b, 1.0)
        print(f'  bloco de {b:5.1f} s: acomodacao em ~{t_settle:5.2f} s '
              f'-> {frac * 100:5.1f}% do bloco em transiente'
              f'{"  <-- registrador atual" if b == BLOCK_S else ""}')
    print()

    print('=' * 78)
    print('1-3. Dose D_z: continuo vs colado-sem-gaps vs por-blocos')
    print('=' * 78)
    header = (f'{"duracao":>8s} {"D_z continuo":>13s} {"D_z colado":>11s} '
             f'{"D_z blocos":>11s} {"colado/cont":>12s} {"blocos/cont":>12s} '
             f'{"blocos/colado":>14s}')
    print(header)
    for duration in (60.0, 180.0, 600.0):
        t, x = synth_day(duration, seed=1)
        d_full, n_full = dose_continuous(x, DT)
        d_stitch, n_stitch, nb = dose_stitched(x, DT, BLOCK_S, CYCLE_S)
        d_block, n_block, nb2 = dose_blocked(x, DT, BLOCK_S, CYCLE_S)
        print(f'{duration:7.0f}s {d_full:13.3f} {d_stitch:11.3f} '
              f'{d_block:11.3f} {d_stitch / d_full * 100:11.1f}% '
              f'{d_block / d_full * 100:11.1f}% '
              f'{d_block / d_stitch * 100:13.1f}%')
        print(f'{"":8s} peaks: continuo={n_full:4d}  colado={n_stitch:4d}  '
              f'blocos={n_block:4d}  ({nb} blocos de {BLOCK_S:g}s mantidos)')
    print()
    print('colado/continuo isola o efeito da amostragem (peaks perdidos nos')
    print('60% nao gravados) -- esperado, estatistico, nao corrigivel.')
    print('blocos/colado isola o efeito do filtro reiniciado a cada bloco,')
    print('sobre as MESMAS amostras -- este e o que decide se blocos de 2 s')
    print('sao longos o bastante para o passa-alta de 0,01 Hz.')
    print()

    print('=' * 78)
    print('4. Duracao de bloco necessaria para o efeito do filtro ficar pequeno')
    print('=' * 78)
    t, x = synth_day(300.0, seed=2)
    d_full, _ = dose_continuous(x, DT)
    print(f'{"bloco":>8s} {"ciclo":>8s} {"duty":>6s} {"D_z colado":>11s} '
          f'{"D_z blocos":>11s} {"blocos/colado":>14s}')
    for block_s, cycle_s in ((2.0, 5.0), (5.0, 5.0 / DUTY), (10.0, 10.0 / DUTY),
                             (20.0, 20.0 / DUTY)):
        d_stitch, _, nb = dose_stitched(x, DT, block_s, cycle_s)
        d_block, _, nb2 = dose_blocked(x, DT, block_s, cycle_s)
        print(f'{block_s:7.1f}s {cycle_s:7.1f}s {block_s / cycle_s * 100:5.1f}% '
              f'{d_stitch:11.3f} {d_block:11.3f} '
              f'{d_block / d_stitch * 100:13.1f}%')


if __name__ == '__main__':
    main()
