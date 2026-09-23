"""Estudo controlado dos artefatos numericos da cadeia do Anexo A.

Tres efeitos sao medidos em sinais sinteticos de resposta conhecida:

  1. oscilacao de Gibbs, do corte vertical de banda, que transforma um choque
     isolado num trem de oscilacoes e faz a contagem de picos contar cada
     lobulo;
  2. convolucao circular, de multiplicar no dominio da frequencia sem
     acolchoar, que enrola a resposta do fim do registro para o comeco;
  3. o efeito combinado das duas correcoes sobre os casos de referencia reais,
     que e o que decide se elas podem ser ligadas por padrao.

Uso: python tests/leakage_study.py [pasta com as aceleracoes de referencia]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315 import analysis, core, workbook

DATA = Path(__file__).resolve().parents[1] / 'data'
FS = 1000.0

CONFIGS = [
    ('atual (referencia)', dict(ZERO_PADDING=False, BAND_TRANSITION=0.0)),
    ('acolchoamento', dict(ZERO_PADDING=True, BAND_TRANSITION=0.0)),
    ('transicao suave', dict(ZERO_PADDING=False, BAND_TRANSITION=0.15)),
    ('as duas', dict(ZERO_PADDING=True, BAND_TRANSITION=0.15)),
]


def apply(cfg):
    for k, v in cfg.items():
        setattr(core, k, v)


def shock(n, centre, amp, width_s, dt):
    w = int(width_s * FS)
    k = np.arange(-w, w + 1)
    pulse = amp * np.exp(-(k / (0.3 * w)) ** 2)
    x = np.zeros(n)
    a, b = centre - w, centre + w + 1
    lo, hi = max(a, 0), min(b, n)
    x[lo:hi] = pulse[lo - a: hi - a]
    return x


def signals(dt):
    """Tres casos, do mais adverso ao mais realista."""
    n = 30000
    t = np.arange(n) * dt
    rng = np.random.default_rng(3)

    isolated = shock(n, n // 2, 5.0, 0.04, dt)

    train = np.zeros(n)
    for c in np.linspace(0.1, 0.9, 8):
        train += shock(n, int(c * n), 5.0, 0.04, dt)

    bg = rng.standard_normal(n)
    F = np.fft.rfft(bg)
    f = np.fft.rfftfreq(n, dt)
    F[(f < 0.6) | (f > 25)] = 0
    bg = np.fft.irfft(F, n)
    bg = bg / bg.std() * 0.4
    embedded = bg + train

    return t, {'choque isolado': isolated,
               'trem de choques': train,
               'choques em vibracao': embedded}


def band_limit_only(x, dt):
    """So a etapa de limitacao de banda, para isolar o efeito de Gibbs."""
    exc = core.Excitation('t', np.arange(len(x)) * dt,
                          np.column_stack([np.zeros(len(x)),
                                           np.zeros(len(x)), x]), dt)
    core._prepare(exc)
    y = core.ifft_amplitude(exc.accel_spectrum)[:, 2]
    return y


def part1(dt):
    print('=' * 74)
    print('1. Oscilacao de Gibbs: picos espurios criados pelo corte de banda')
    print('=' * 74)
    t, sigs = signals(dt)
    print(f'{"sinal":22s} {"config":20s} {"picos":>7s} {"pico max":>10s} '
          f'{"dose rel.":>10s}')
    for name, x in sigs.items():
        base = None
        for label, cfg in CONFIGS[:3]:
            if label == 'acolchoamento':
                continue
            apply(cfg)
            y = band_limit_only(x, dt)
            pk, _ = core.compression_peaks(y)
            dose = np.sum(np.abs(pk) ** 6) ** (1 / 6) if len(pk) else 0.0
            base = base or dose
            print(f'{name:22s} {label:20s} {len(pk):7d} {np.abs(y).max():10.3f} '
                  f'{dose / base:10.3f}')
        print()
    apply(CONFIGS[0][1])


def part2(dt):
    print('=' * 74)
    print('2. Convolucao circular: energia que enrola do fim para o comeco')
    print('=' * 74)
    store = core.TransferFunctionStore(DATA)
    H, fm = store.posture(3)
    h = H[27, 0, 2, 2, 4].astype(complex).reshape(1, -1)
    n = 30000
    t = np.arange(n) * dt

    for pos, tag in ((n - 300, 'choque no fim do registro'),
                     (n // 2, 'choque no meio')):
        x = shock(n, pos, 5.0, 0.04, dt)
        for label, cfg in (CONFIGS[0], CONFIGS[1]):
            apply(cfg)
            exc = core.Excitation('t', t, np.column_stack(
                [np.zeros(n), np.zeros(n), x]), dt)
            core._prepare(exc)
            Hi = core._interp_tf(fm, exc.freq, h)
            g = core.ifft_amplitude(Hi[0] * exc.spectrum[:, 2])
            head = np.abs(g[:int(0.5 * 2 * core.LOWPASS_HZ)]).max()
            print(f'{tag:28s} {label:18s} '
                  f'inicio {head:8.2f} N   maximo {np.abs(g).max():8.2f} N   '
                  f'razao {head / np.abs(g).max() * 100:6.2f}%')
        print()
    apply(CONFIGS[0][1])


def part3(acc_dir):
    print('=' * 74)
    print('3. Efeito nas referencias reais: o que decide se pode ligar')
    print('=' * 74)
    store = core.TransferFunctionStore(DATA)
    EX = Path(acc_dir)
    base_dir = EX if (EX / 'WL2s.txt').exists() else EX / 'Accelerations'
    sheets = sorted(base_dir.parent.glob('Example_*orient.xls'))
    if not sheets:
        print('  planilhas de exemplo nao encontradas; pulando')
        return

    results = {}
    for label, cfg in CONFIGS:
        apply(cfg)
        h = workbook.read_history(sheets[0])
        h.resolve_paths(base_dir)
        results[label] = analysis.Runner(store).run_orienting(h).R
    apply(CONFIGS[0][1])

    ref = results['atual (referencia)']
    print(f'{"config":22s} ' + ' '.join(f'{n:>8s}' for n in core.DISC_NAMES))
    for label, R in results.items():
        print(f'{label:22s} ' + ' '.join(f'{v:8.5f}' for v in R))
    print()
    print(f'{"desvio maximo":22s}')
    for label, R in results.items():
        if label == 'atual (referencia)':
            continue
        d = np.max(np.abs(R - ref) / ref) * 100
        verdict = ('pode ligar por padrao' if d < 0.1 else
                   'so como opcao documentada')
        print(f'  {label:20s} {d:6.3f}%   {verdict}')


def main():
    dt = 1.0 / FS
    part1(dt)
    part2(dt)
    if len(sys.argv) > 1:
        part3(sys.argv[1])
    else:
        print('(passe a pasta das aceleracoes de referencia para a parte 3)')


if __name__ == '__main__':
    main()
