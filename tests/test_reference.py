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

    print('\n' + ('TODOS OS CASOS PASSARAM' if ok else 'HOUVE FALHAS'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())


