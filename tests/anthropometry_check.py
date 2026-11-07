"""Validacao das bordas de classe de massa (core.py, MASS_EDGES_BY_BMI_CLASS)
contra os seis casos indiv1-indiv6 do pacote
de exemplos da ISO 2631-5, usando o R^A oficial publicado em Results_ISO.xls.

Motivacao: uma versao anterior de core.anthropometry() so tinha indiv5 e
indiv6 para se ajustar (a unica dupla que varia massa), e mesmo assim so
por busca exaustiva, com erro residual e uma classe de IMC forcada sem
sentido fisico (Secao 11.1 da documentacao, ANTHROPOMETRY_NOTE). A Tabela 2
de Seidel, Hinz, Hofmann, Menzel (2008), Int J Ind Ergon 38:856-867 --
percentis de massa DENTRO de cada classe de IMC, dos 1163 motoristas que
originaram as funcoes de transferencia -- substitui o ajuste por um valor
publicado. Este teste roda os seis casos completos (nao so os dois que
variam massa) contra o R^A oficial, usando o pipeline de producao
(workbook.read_history + analysis.Runner.run_individual), sem nenhuma
logica duplicada.

Uso: python tests/anthropometry_check.py ["CAMINHO\\PARA\\Risk_Assessment-Examples"]
"""
import sys
from pathlib import Path

import numpy as np
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315 import core, workbook
from iso26315.analysis import Runner

SEARCH_HINTS = ['Risk_Assessment-Examples', 'Accelerations']


def find_package(explicit=None):
    if explicit:
        p = Path(explicit)
        if (p / 'Results' / 'Results_ISO.xls').exists():
            return p
        for cand in p.rglob('Results_ISO.xls'):
            return cand.parent.parent
        raise SystemExit(f'Nao encontrei Results_ISO.xls em "{p}".')
    roots = [Path.cwd(), Path(__file__).resolve().parents[1],
             Path.home() / 'Desktop', Path.home() / 'Downloads', Path.home()]
    for root in roots:
        for hint in SEARCH_HINTS:
            for cand in root.rglob(f'{hint}/Results/Results_ISO.xls'):
                return cand.parents[1]
    raise SystemExit(
        'Nao encontrei o pacote Risk_Assessment-Examples.\n'
        'Rode assim: python tests/anthropometry_check.py "CAMINHO\\PARA\\Risk_Assessment-Examples"')


def official_r(results_path):
    wb = xlrd.open_workbook(str(results_path))
    sheet = wb.sheet_by_name('Risk factors')
    out = {}
    for r in range(1, sheet.nrows):
        row = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        name = Path(str(row[0])).stem            # 'Example_Wheel loader_indiv1'
        out[name] = np.array(row[10:16], dtype=float)   # R T12/L1 .. R L5/S1
    return out


def main():
    pkg = find_package(sys.argv[1] if len(sys.argv) > 1 else None)
    acc_dir = pkg / 'Accelerations'
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    store = core.TransferFunctionStore(data_dir)
    official = official_r(pkg / 'Results' / 'Results_ISO.xls')

    cases = ['Example_Wheel loader_indiv1', 'Example_Wheel loader_indiv2',
             'Example_Wheel loader_indiv3', 'Example_Wheel loader_indiv4',
             'Example_Wheel loader_indiv5', 'Example_Wheel loader_indiv6',
             'Example_BD1_indiv']

    print(f'Pacote: {pkg}')
    print(f'{"caso":32s} {"k por ano (unicos)":22s} {"erro max R^A":>13s}')
    ok = True
    for name in cases:
        path = pkg / f'{name}.xls'
        history = workbook.read_history(path)
        missing = history.resolve_paths(acc_dir)
        if missing:
            print(f'  [AVISO] {name}: arquivos nao encontrados: {missing}')

        ks = set()
        orig_exposure = core.analyse_exposure

        def spy(store, excs, posture, mass, height, measurement_time=None,
               subclass=1, _orig=orig_exposure):
            k, bmi_per = core.anthropometry(mass, height, subclass)
            ks.add(k)
            return _orig(store, excs, posture, mass, height,
                        measurement_time=measurement_time, subclass=subclass)
        core.analyse_exposure = spy
        try:
            runner = Runner(store)
            result = runner.run_individual(history, birth_year=1950)
        finally:
            core.analyse_exposure = orig_exposure

        ref = official.get(name)
        if ref is None:
            print(f'  [FALHA] {name}: sem R^A oficial em Results_ISO.xls')
            ok = False
            continue
        err = np.abs(result.R - ref) / np.where(ref != 0, ref, 1.0)
        good = err.max() < 1e-3
        ok &= good
        print(f'{name:32s} {sorted(ks)!s:22s} {err.max():13.2e} '
             f'{"OK" if good else "FALHA"}')
        if not good:
            print(f'    obtido    {np.round(result.R, 4)}')
            print(f'    oficial   {np.round(ref, 4)}')

    print('\n' + ('TODOS OS CASOS PASSARAM' if ok else 'HOUVE FALHAS'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
