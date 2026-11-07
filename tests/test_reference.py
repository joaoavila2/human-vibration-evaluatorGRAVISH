"""Valida o nucleo contra os resultados de referencia do software original
(Risk_Assessment-Examples/Results/Results_ISO.xls).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315 import core

DATA = Path(__file__).resolve().parents[1] / 'data'

# Pasta com WL2s.txt, WL2b.txt, WL2h.txt, WL2f.txt e BD1sZ.asc, vinda do pacote
# Risk_Assessment-Examples. Passe como argumento, ou deixe que seja procurada.
SEARCH_HINTS = ['Risk_Assessment-Examples', 'Accelerations', 'Aceleracoes']


def find_accelerations(explicit=None):
    if explicit:
        p = Path(explicit)
        if (p / 'WL2s.txt').exists():
            return p
        for cand in p.rglob('WL2s.txt'):
            return cand.parent
        raise SystemExit(f'Nao encontrei WL2s.txt em "{p}".')

    roots = [Path.cwd(), Path(__file__).resolve().parents[1],
             Path.home() / 'Desktop', Path.home() / 'Downloads', Path.home()]
    seen = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for cand in root.glob('*/WL2s.txt'):
            return cand.parent
        for hint in SEARCH_HINTS:
            for base in root.glob(f'*{hint}*'):
                for cand in base.rglob('WL2s.txt'):
                    return cand.parent
    raise SystemExit(
        'Nao localizei a pasta Accelerations dos exemplos.\n'
        'Extraia o Risk_Assessment-Examples.rar e rode assim:\n'
        '    python tests/test_reference.py "CAMINHO\\PARA\\Accelerations"')


ACC = None

REF_ORIENT_R = np.array([0.98423, 1.085, 1.2226, 1.3102, 1.2842, 1.1698])
REF_ORIENT_SD4 = 0.57776
REF_BD1_R = np.array([2.7656, 2.6893, 2.6501, 2.6643, 2.5806, 2.3923])


def load_case(store, files, posture, mass, height, tm, subclass=1):
    excs = []
    base = None
    for f in files:
        if f is None:
            excs.append(None)
        else:
            e = core.read_excitation(ACC / f)
            base = base or e
            excs.append(e)
    excs = [e if e is not None else core.zero_excitation(base) for e in excs]
    return core.analyse_exposure(store, excs, posture, mass, height,
                                 measurement_time=tm, subclass=subclass)


def check(name, got, ref, tol):
    err = np.max(np.abs(got - ref) / np.abs(ref))
    status = 'OK  ' if err < tol else 'FALHA'
    print(f'  [{status}] {name}: erro relativo maximo {err:.2e}')
    print(f'        obtido    {np.round(got, 5)}')
    print(f'        referencia{np.round(ref, 5)}')
    return err < tol


# ---------------------------------------------------------------------------
# Clausula 5 / Anexo C: exemplo numerico do fim do Anexo C e Tabela C.2
# ---------------------------------------------------------------------------

def test_annex_c():
    from iso26315 import annexc as C
    from scipy.optimize import brentq
    ok = True
    print('\nClause 5 / Annex C')

    Dz = C.acceleration_dose(np.full(5, 40.0))
    Sd = 0.029 * Dz                      # m_z arredondado, como a norma usa
    R = C.stress_variable(np.repeat(Sd, 20), 120, 20, 0.029, 0.052)
    for label, got, ref, tol in (('S_d', Sd, 1.623, 1e-3), ('R', R, 1.22, 1e-3)):
        good = abs(got - ref) / ref < tol
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] {label}: {got:.4f} vs {ref}')

    for sex, ref in (('male', [0.72, 1.42, 2.17]), ('female', [0.52, 0.87, 1.20])):
        cfg = C.SEX_DEFAULTS[sex]
        got = [brentq(lambda R: C.injury_probability(R, cfg['alpha'],
                                                     cfg['beta']) - p, .01, 10)
               for p in (0.1, 0.5, 0.9)]
        good = all(abs(g - r) < 0.005 for g, r in zip(got, ref))
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] Table C.2 {sex}: '
              f'{[round(x, 2) for x in got]} vs {ref}')

    dc = abs(C.transfer_function(np.array([0.0]))[0])
    good = abs(dc - 1.0) < 1e-9
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] Formula (1) DC gain: {dc:.6f} vs 1.0')
    return ok


def test_srs():
    """Seno de 1 m/s2 em fn: em regime, a aceleracao absoluta do oscilador
    de 1 GL e' sqrt(1 + (2 zeta)^2) / (2 zeta) = 10,05 com Q = 10. A
    bilinear sem prewarping da v2.9 dava 0,30 disso em 200 Hz."""
    ok = True
    print('\nSRS (Smallwood + reamostragem)')
    fs, q = 1024.0, 10.0
    zeta = 1 / (2 * q)
    ideal = np.sqrt(1 + (2 * zeta) ** 2) / (2 * zeta)
    t = np.arange(int(8 * fs)) / fs
    for fn in (10.0, 50.0, 100.0, 200.0, 300.0, 0.4 * fs):
        x = np.sin(2 * np.pi * fn * t)
        _, srs = core.shock_response_spectrum(x, 1 / fs, freqs=[fn], q=q)
        ratio = srs[0] / ideal
        good = abs(ratio - 1) < 0.01
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] fn = {fn:5.1f} Hz: '
              f'SRS/ideal = {ratio:.4f}')

    # registro mais longo que SRS_CHUNK: a divisao em trechos nao pode mudar
    # o resultado
    rng = np.random.default_rng(7)
    x = rng.standard_normal(3 * core.SRS_CHUNK + 1234)
    f1, s1 = core.shock_response_spectrum(x, 1 / fs)
    old = core.SRS_CHUNK
    core.SRS_CHUNK = len(x)
    try:
        _, s2 = core.shock_response_spectrum(x, 1 / fs)
    finally:
        core.SRS_CHUNK = old
    err = np.max(np.abs(s1 - s2) / s2)
    good = err < 1e-9
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] trechos vs registro inteiro: '
          f'erro maximo {err:.1e} em {len(f1)} frequencias')
    return ok


def test_ratios():
    """ISO 2631-1 6.3.3. Seno de 5 Hz continua seno depois de W_k/W_d:
    fator de crista sqrt(2), MTVV/a_w = 1 e VDV/(a_w T^1/4) = sqrt(2)
    (3/8)^1/4 = 1,107. Um pulso curto e forte a cada 10 s passa dos dois
    limites."""
    from iso26315 import continuity, nho09
    ok = True
    print('\nISO 2631-1 6.3.3 (razoes de decisao)')
    fs = 512.0
    t = np.arange(int(120 * fs)) / fs
    # entrada suave de 5 s: um seno que comeca do repouso de uma vez faz a
    # ponderacao passar do ponto nos primeiros ciclos e o fator de crista
    # sai 3 % alto, o que e' o filtro, nao a razao
    sine = np.sin(2 * np.pi * 5.0 * t) * np.clip(t / 5.0, 0, 1)
    blocks = np.repeat(sine[:, None], 3, axis=1)[None]
    continuity.mark(blocks, [(0, 1)])
    r = nho09.axis_ratios(blocks, 1 / fs)
    # com a rampa de 5 s em T = 120 s: a rampa vale 5/3 s de energia em vez
    # de 5 (integral de (t/5)^2) e 1 s de VDV^4 em vez de 5 (de (t/5)^4)
    T = t[-1] + 1 / fs
    e2 = np.sqrt((T - 10 / 3) / T)
    e4 = ((T - 4.0) / T) ** 0.25
    ref = {'crest': np.sqrt(2) / e2, 'mtvv_ratio': 1.0 / e2,
           'vdv_ratio': np.sqrt(2) * (3 / 8) ** 0.25 * e4 / e2}
    for key, val in ref.items():
        got = r[key][2]
        good = abs(got - val) / val < 0.01
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] seno, {key} (z): '
              f'{got:.4f} vs {val:.4f}')
    good = not r['additional_required'].any()
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] seno: metodo basico suficiente')

    shocks = 0.05 * np.random.default_rng(3).standard_normal(len(t))
    shocks[(np.arange(len(t)) % int(10 * fs)) < 20] += 30.0   # 40 ms a cada 10 s
    blocks = np.repeat(shocks[:, None], 3, axis=1)[None]
    continuity.mark(blocks, [(0, 1)])
    r = nho09.axis_ratios(blocks, 1 / fs)
    good = bool(r['mtvv_ratio'][2] > core.MTVV_RATIO_LIMIT and
                r['vdv_ratio'][2] > core.VDV_RATIO_LIMIT)
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] choques: MTVV/a_w '
          f'{r["mtvv_ratio"][2]:.2f} > 1,5 e VDV/(a_w T^1/4) '
          f'{r["vdv_ratio"][2]:.2f} > 1,75')
    return ok


def test_estimators():
    """H1 exato com ruido so na saida, H2 exato com ruido so na entrada,
    Hv entre os dois. Sistema: 1 GL excitado pela base (fn 5 Hz, zeta 0,2),
    ruido branco, ruido de medicao com 40 % do desvio-padrao do canal."""
    from scipy.signal import bilinear, freqz, lfilter
    ok = True
    print('\nEstimadores H1 / H2 / Hv')
    fs = 200.0
    wn, zeta = 2 * np.pi * 5.0, 0.2
    b, a = bilinear([2 * zeta * wn, wn ** 2], [1, 2 * zeta * wn, wn ** 2], fs)
    rng = np.random.default_rng(11)
    u = rng.standard_normal(int(900 * fs))
    y = lfilter(b, a, u)
    cases = (('ruido na saida', u, y + 0.4 * y.std() * rng.standard_normal(len(u)),
              'h1', 'h2'),
             ('ruido na entrada', u + 0.4 * rng.standard_normal(len(u)), y,
              'h2', 'h1'))
    for label, x_in, y_out, exact, biased in cases:
        f, est, navg = core.transmissibility_estimates(x_in, y_out, 1 / fs,
                                                       nperseg=1024)
        band = (f >= 1) & (f <= 30)
        _, h_true = freqz(b, a, worN=f[band], fs=fs)
        err = {k: float(np.median(np.abs(np.abs(est[k][band]) /
                                         np.abs(h_true) - 1)))
               for k in ('h1', 'h2', 'hv')}
        lo = np.minimum(np.abs(est['h1']), np.abs(est['h2']))[band]
        hi = np.maximum(np.abs(est['h1']), np.abs(est['h2']))[band]
        between = bool(np.all((np.abs(est['hv'][band]) >= lo * (1 - 1e-9)) &
                              (np.abs(est['hv'][band]) <= hi * (1 + 1e-9))))
        good = err[exact] < 0.02 and err[biased] > 0.05 and between
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] {label} ({navg} medias): '
              f'erro mediano H1 {err["h1"]:.3f}, H2 {err["h2"]:.3f}, '
              f'Hv {err["hv"]:.3f}; Hv entre H1 e H2: {between}')

    # banda de erro: com coerencia 0,5 e 100 medias, eps = 0,0707
    e = core.frf_estimates(np.array([1.0]), np.array([2.0]),
                           np.array([1.0 + 0j]), 100)['eps'][0]
    good = abs(e - np.sqrt(0.5) / (np.sqrt(0.5) * np.sqrt(200))) < 1e-12
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] eps(gamma2 = 0,5, n = 100) = '
          f'{e:.4f}')
    return ok


def test_intervals():
    """Registro por intervalo: zerar a integracao a cada intervalo nao pode
    mudar o dia. Soma de a_w^2 * duracao = energia global, soma de VDV^4 =
    VDV^4 global, maior MTVV dos intervalos = MTVV global. Dois trechos
    continuos separados por uma lacuna, para os intervalos recomecarem."""
    from iso26315 import continuity, nho09
    ok = True
    print('\nRegistro por intervalo')
    fs, n = 512.0, 1024
    rng = np.random.default_rng(4)
    blocks = rng.standard_normal((60, n, 3))
    blocks[17, 100:110, 2] += 25.0                    # um choque
    continuity.mark(blocks, [(0, 30), (30, 60)])
    seconds = np.r_[np.arange(30) * n / fs, 1000 + np.arange(30) * n / fs]
    dt = 1 / fs
    w = nho09.weighted_blocks(blocks, dt)
    T = w.shape[0] * n * dt
    aw_glob = np.sqrt((w ** 2).mean(axis=(0, 1)))
    vdv_glob = (np.sum(w ** 4, axis=(0, 1)) * dt) ** 0.25
    mtvv_glob = nho09.mtvv(blocks, dt)
    for sec in (1, 7, 60):
        m = nho09.interval_metrics(blocks, dt, sec, seconds)
        e = np.sqrt((m['aw'] ** 2 * m['duration'][:, None]).sum(0) / T)
        v = (m['vdv'] ** 4).sum(0) ** 0.25
        errs = (np.max(np.abs(e / aw_glob - 1)), np.max(np.abs(v / vdv_glob - 1)),
                abs(np.nanmax(m['mtvv']) / mtvv_glob - 1))
        restart = bool(np.isclose(m['start'], 1000.0).any())
        good = max(errs) < 1e-9 and restart and abs(m['duration'].sum() - T) < 1e-9
        ok &= good
        print(f'  [{"OK  " if good else "FALHA"}] {sec:2d} s: {len(m["start"])} '
              f'intervalos; erro energia {errs[0]:.1e}, VDV {errs[1]:.1e}, '
              f'MTVV {errs[2]:.1e}; recomeca na lacuna: {restart}')
    return ok


def test_alignment():
    """Atraso conhecido e fracionario (deslocamento de fase na FFT) tem de
    ser achado dentro de 1/10 de amostra (ruido ate 30 Hz, como movimento
    do corpo; em ruido branco ate Nyquist o pico e' tao estreito que a
    interpolacao parabolica erra ~0,1 amostra); depois de alinhar, a coerencia
    volta. Um relogio 300 ppm mais rapido (1 s de deriva em ~55 min) tem de
    ser recusado com o motivo."""
    ok = True
    print('\nAlinhamento por correlacao cruzada')
    fs = 256.0
    rng = np.random.default_rng(8)
    n = int(600 * fs)
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(4, 30 / (fs / 2), output='sos'),
                    rng.standard_normal(n))
    true_lag = -0.7372                                 # y adiantado
    f = np.fft.rfftfreq(n, 1 / fs)
    y = np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * true_lag), n)
    y = 1.5 * y + 0.3 * x.std() * rng.standard_normal(n)
    xa, ya, info = core.align_by_correlation(x, y, 1 / fs)
    err = abs(info['lag'] - true_lag) * fs
    fq, est0, _ = core.transmissibility_estimates(x, y, 1 / fs, nperseg=512)
    _, est1, _ = core.transmissibility_estimates(xa, ya, 1 / fs, nperseg=512)
    band = (fq >= 1) & (fq <= 25)                       # onde ha sinal
    g0, g1 = np.median(est0['g2'][band]), np.median(est1['g2'][band])
    good = err < 0.1 and info['drift'] < 1 / fs and g1 > 0.9 and g0 < 0.5
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] atraso {info["lag"]:+.4f} s vs '
          f'{true_lag:+.4f} (erro {err:.3f} amostra); coerencia mediana '
          f'{g0:.2f} -> {g1:.2f}')

    t = np.arange(n) / fs
    y2 = np.interp(t * (1 + 300e-6), t, x)             # relogio diferente
    try:
        core.align_by_correlation(x, y2, 1 / fs)
        good = False
        msg = 'nao recusou'
    except core.AlignmentError as e:
        good = 'drifts' in str(e)
        msg = str(e)[:70]
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] 300 ppm recusado: {msg}...')
    return ok


def test_segments():
    """Rejeicao de segmentos: cada defeito injetado tem de ser marcado, com
    poucas marcas falsas, e o H1 sem os segmentos ruins volta para perto do
    verdadeiro. Classes de amplitude: um sistema cujo pico desce de 6 para
    4 Hz quando a amplitude sobe tem de dar tres picos separados."""
    from scipy.signal import bilinear, freqz, lfilter
    ok = True
    print('\nRejeicao de segmentos e classes de amplitude')
    fs, nseg = 256.0, 1024
    rng = np.random.default_rng(21)

    def osc(fn, zeta=0.15):
        wn = 2 * np.pi * fn
        return bilinear([2 * zeta * wn, wn ** 2], [1, 2 * zeta * wn, wn ** 2], fs)

    b, a = osc(5.0)
    x = rng.standard_normal(200 * nseg)
    y = lfilter(b, a, x) + 0.05 * rng.standard_normal(len(x))
    bad = {'clipped': 20, 'gain': 50, 'offset': 90, 'kurtosis': 130}
    full_scale = 1.2 * np.abs(y).max()                   # fundo de escala do sensor
    s = slice(bad['clipped'] * nseg, (bad['clipped'] + 1) * nseg)
    y[s] = np.clip(4 * y[s], -full_scale, full_scale)     # satura
    s = slice(bad['gain'] * nseg, (bad['gain'] + 3) * nseg)
    y[s] = 0.05 * rng.standard_normal(3 * nseg)          # perdeu contato
    s = bad['offset'] * nseg
    y[s + nseg // 2:s + nseg] += 3.0                     # degrau
    s = bad['kurtosis'] * nseg + 300
    y[s:s + 4] += 40.0                                   # batida
    seg = core.segment_cross_spectra(x, y, 1 / fs, nperseg=nseg, overlap=0.0)
    clip = (core.clipped_segments(x, 1e9, nseg) |
            core.clipped_segments(y, full_scale, nseg))
    fl = core.segment_flags(seg, clipped=clip)
    found = {k: bool(fl[k][i]) for k, i in bad.items()}
    planted = set(bad.values()) | {51, 52}
    false = sum(1 for i in np.flatnonzero(fl['rejected']) if i not in planted)
    band = (seg['f'] >= 1) & (seg['f'] <= 20)
    _, h = freqz(b, a, worN=seg['f'][band], fs=fs)
    errs = []
    for mask in (None, ~fl['rejected']):
        est, _ = core.frf_from_segments(seg, mask)
        errs.append(np.median(np.abs(np.abs(est['h1'][band]) / np.abs(h) - 1)))
    good = (all(found.values()) and false <= 2 and errs[1] < errs[0]
            and errs[1] < 0.02)
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] marcados pelo motivo certo {found}; '
          f'falsos {false} de {len(fl["rejected"])}; erro H1 {errs[0]:.3f} -> '
          f'{errs[1]:.3f}')
    print(f'        {core.flag_summary(fl)}')

    # nao linear: tres trechos, amplitude 0,35, 1,4 e 3 m/s2, pico 6, 5 e 4 Hz
    xs, ys = [], []
    for amp, fn in ((0.35, 6.0), (1.4, 5.0), (3.0, 4.0)):    # longe das bordas
        u = amp * rng.standard_normal(60 * nseg)
        b, a = osc(fn)
        xs.append(u)
        ys.append(lfilter(b, a, u) + 0.02 * rng.standard_normal(len(u)))
    seg = core.segment_cross_spectra(np.concatenate(xs), np.concatenate(ys),
                                     1 / fs, nperseg=nseg, overlap=0.0)
    classes = core.amplitude_classes(seg)
    peaks = []
    f = seg['f']
    sel = (f > 2) & (f < 10)
    for lo, hi, n, est in classes:
        peaks.append(f[sel][np.argmax(np.abs(est['h1'][sel]))])
    good = (len(classes) == 3 and all(abs(p - t) < 0.3 for p, t in
                                       zip(peaks, (6.0, 5.0, 4.0))))
    ok &= good
    desc = ', '.join(f'{lo:g}-{hi:g} m/s2: {n} seg, pico {p:.2f} Hz'
                     for (lo, hi, n, _), p in zip(classes, peaks))
    print(f'  [{"OK  " if good else "FALHA"}] {desc}')
    return ok


def test_srs_windows():
    """SRS primario e residual de um meio-seno de 10 m/s2 por 11 ms contra
    uma referencia independente: pulso analitico amostrado a 64x e resolvido
    pelo lsim do scipy. t0 = 0.25 s (pre-choque > 0.2 s) para que
    shock_response_spectra use a media do pre-choque (zero) em vez da media
    global, evitando excitacao artificial na fase residual. Tambem: maximax
    = maior dos dois."""
    from scipy.signal import lsim
    ok = True
    print('\nSRS primario / residual (meio-seno)')
    fs, T, A, q = 2048.0, 0.011, 10.0, 10.0
    zeta = 1 / (2 * q)
    n = int(0.5 * fs)
    t = np.arange(n) / fs
    t0 = 0.25   # pre-choque >= 0.2 s -> pre=True -> media do pre-choque = 0
    x = np.where((t >= t0) & (t < t0 + T), A * np.sin(np.pi * (t - t0) / T), 0)
    # Frequencias para testar primario/residual: ate 200 Hz (Smallwood com
    # oversampling adequado). A 400 Hz com fs=2048 Hz o up=4 (20 amostras/ciclo)
    # e insuficiente para o residual pequeno; s testamos o maximax la.
    freqs_pr = np.array([10.0, 30.0, 45.0, 90.0, 200.0])
    freqs_all = np.array([10.0, 30.0, 45.0, 90.0, 200.0, 400.0])
    r = core.shock_response_spectra(x, 1 / fs, window=(t0, t0 + T),
                                    freqs=freqs_all, q=q)
    r_pr = core.shock_response_spectra(x, 1 / fs, window=(t0, t0 + T),
                                       freqs=freqs_pr, q=q)
    # referencia: pulso analitico 64x resolvido pelo lsim
    up_ref = 64
    tf = np.arange(n * up_ref) / (fs * up_ref)
    xf = np.where((tf >= t0) & (tf < t0 + T), A * np.sin(np.pi * (tf - t0) / T), 0)
    prim_mask = (tf >= t0) & (tf < t0 + T)
    resid_mask = tf >= t0 + T
    worst_pr = 0.0
    for i, fn in enumerate(freqs_pr):
        wn = 2 * np.pi * fn
        _, yr, _ = lsim(([2 * zeta * wn, wn ** 2], [1, 2 * zeta * wn, wn ** 2]),
                        xf, tf)
        ref_p = np.abs(yr[prim_mask]).max()
        ref_r = np.abs(yr[resid_mask]).max()
        e = max(abs(r_pr['primary'][i] / ref_p - 1),
                abs(r_pr['residual'][i] / ref_r - 1))
        worst_pr = max(worst_pr, e)
    worst_mm = 0.0
    for i, fn in enumerate(freqs_all):
        wn = 2 * np.pi * fn
        _, yr, _ = lsim(([2 * zeta * wn, wn ** 2], [1, 2 * zeta * wn, wn ** 2]),
                        xf, tf)
        ref_mm = max(np.abs(yr[prim_mask]).max(), np.abs(yr[resid_mask]).max())
        e = abs(r['maximax'][i] / ref_mm - 1)
        worst_mm = max(worst_mm, e)
    mm = np.allclose(r['maximax'], np.fmax(r['primary'], r['residual']))
    good = worst_pr < 0.05 and worst_mm < 0.02 and mm
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] prim/resid (10-200 Hz): '
          f'{worst_pr * 100:.2f}%; maximax (10-400 Hz): {worst_mm * 100:.2f}%; '
          f'maximax=max(p,r): {mm}')
    w = core.shock_window(x, 1 / fs)
    good = abs(w[0] / fs - t0) < 0.002 and abs(w[1] / fs - (t0 + T)) < 0.002
    ok &= good
    print(f'  [{"OK  " if good else "FALHA"}] janela automatica '
          f'{w[0] / fs * 1000:.1f}-{w[1] / fs * 1000:.1f} ms '
          f'(choque de {t0 * 1000:.0f} a {(t0 + T) * 1000:.0f} ms)')
    return ok


def main():
    global ACC
    ACC = find_accelerations(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f'Acelerações: {ACC}\n')
    store = core.TransferFunctionStore(DATA)
    ok = True

    # --- Caso 1: analise orientadora, pa carregadeira, postura grupo 3 -------
    res = load_case(store, ['WL2s.txt', 'WL2f.txt', 'WL2h.txt', 'WL2b.txt'],
                    posture=3, mass=110.0, height=1.85, tm=400.0)
    Sd = core.daily_dose([res], [14400.0])
    R = core.risk_factor_with_static(np.tile(Sd, (45, 1)), np.full(45, 240.0),
                                     res.S_stat, start_age=20)
    print('Caso 1 - analise orientadora (WL2, grupo 3, 110 kg / 1,85 m)')
    print(f'  antropometria k={res.k}, BMI/percentil={res.bmi_per}')
    ok &= check('R^A', R, REF_ORIENT_R, 1e-4)
    err_sd = abs(Sd[3] - REF_ORIENT_SD4) / REF_ORIENT_SD4
    print(f'  [{"OK  " if err_sd < 1e-4 else "FALHA"}] S_d^A(L3/L4): '
          f'{Sd[3]:.5f} MPa vs {REF_ORIENT_SD4} (erro {err_sd:.2e})')
    ok &= err_sd < 1e-4

    # --- Caso 2: analise individual, bulldozer, maos e pes sem medicao -------
    print('\nCaso 2 - analise individual (BD1, grupo 3, nascido em 1950)')
    res = load_case(store, ['BD1sZ.asc', None, None, 'BD1sZ.asc'],
                    posture=3, mass=110.0, height=1.85, tm=400.0)
    Sd = core.daily_dose([res, res], [3600.0, 10800.0])
    R = core.risk_factor_with_static(np.tile(Sd, (45, 1)), np.full(45, 240.0),
                                     res.S_stat, start_age=15)
    ok &= check('R^A', R, REF_BD1_R, 1e-4)

    ok &= test_annex_c()
    ok &= test_srs()
    ok &= test_srs_windows()
    ok &= test_ratios()
    ok &= test_estimators()
    ok &= test_intervals()
    ok &= test_alignment()
    ok &= test_segments()

    print('\n' + ('TODOS OS CASOS PASSARAM' if ok else 'HOUVE FALHAS'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())


