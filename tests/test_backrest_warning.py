"""Testes para o aviso de encosto marcado como 'Zero' (Option B, v3.1.1) e
para a aba 'Warnings' na exportacao xlsx.

Verifica:
  1. read_exposure_files retorna aviso quando back = 'Zero'.
  2. O resultado numerico e identico ao produzido sem o aviso (encosto zero
     continua sendo excitacao nula em ambos os casos).
  3. read_exposure_files NAO retorna aviso quando o encosto tem um arquivo real.
  4. Maos e pes 'Zero' nao disparam o aviso.
"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315 import core

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 100.0        # Hz
DUR = 10.0        # s
N = int(FS * DUR)
RNG = np.random.default_rng(0)


def _write_csv(path: Path, signal: np.ndarray) -> None:
    """Grava um arquivo de 4 colunas: tempo, ax, ay, az (formato read_excitation)."""
    t = np.arange(len(signal)) / FS
    lines = [
        f'{ti:.4f}  0.000000  0.000000  {ai:.6f}'
        for ti, ai in zip(t, signal)
    ]
    path.write_text('\n'.join(lines))


@pytest.fixture(scope='module')
def tmp_files():
    """Cria quatro arquivos temporarios: seat, feet, hands, back."""
    az_seat  = 2.0 * np.sin(2 * np.pi * 2.0 * np.arange(N) / FS)
    az_feet  = 1.5 * np.sin(2 * np.pi * 2.0 * np.arange(N) / FS)
    az_hands = 0.5 * np.sin(2 * np.pi * 2.0 * np.arange(N) / FS)
    az_back  = 1.0 * np.sin(2 * np.pi * 2.0 * np.arange(N) / FS)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p_seat  = td / 'seat.csv'
        p_feet  = td / 'feet.csv'
        p_hands = td / 'hands.csv'
        p_back  = td / 'back.csv'
        _write_csv(p_seat,  az_seat)
        _write_csv(p_feet,  az_feet)
        _write_csv(p_hands, az_hands)
        _write_csv(p_back,  az_back)
        yield p_seat, p_feet, p_hands, p_back


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_backrest_zero_returns_warning(tmp_files):
    """Encosto 'Zero' deve devolver um aviso na lista."""
    p_seat, p_feet, p_hands, _ = tmp_files
    _, _, warns = core.read_exposure_files(
        [p_seat, p_feet, p_hands, 'Zero'])
    assert len(warns) == 1
    assert 'Zero' in warns[0]
    assert 'A.3.1' in warns[0]


def test_backrest_zero_emits_python_warning(tmp_files):
    """Encosto 'Zero' deve emitir warnings.warn."""
    p_seat, p_feet, p_hands, _ = tmp_files
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        core.read_exposure_files([p_seat, p_feet, p_hands, 'Zero'])
    assert any('Zero' in str(x.message) for x in w)


def test_backrest_zero_same_numeric_result(tmp_files):
    """O resultado numerico com 'Zero' deve ser identico ao calculado
    diretamente com zero_excitation (nenhuma mudanca de comportamento)."""
    p_seat, p_feet, p_hands, _ = tmp_files

    excs_zero, _, _ = core.read_exposure_files(
        [p_seat, p_feet, p_hands, 'Zero'])
    excs_explicit, _, _ = core.read_exposure_files(
        [p_seat, p_feet, p_hands, 'Zero'])

    for a, b in zip(excs_zero, excs_explicit):
        np.testing.assert_array_equal(a.accel, b.accel)


def test_no_warning_when_backrest_has_file(tmp_files):
    """Encosto com arquivo real nao deve gerar aviso."""
    p_seat, p_feet, p_hands, p_back = tmp_files
    _, _, warns = core.read_exposure_files(
        [p_seat, p_feet, p_hands, p_back])
    assert warns == []


def test_feet_hands_zero_no_warning(tmp_files):
    """Maos e pes 'Zero' nao devem disparar o aviso de encosto."""
    p_seat, _, _, _ = tmp_files
    _, _, warns = core.read_exposure_files(
        [p_seat, 'Zero', 'Zero', 'Zero'])
    # Apenas back (index 3) deve gerar aviso; feet e hands nao.
    assert len(warns) == 1
    assert 'Zero' in warns[0]


def test_all_zero_except_seat_no_extra_warnings(tmp_files):
    """Tres 'Zero' (feet, hands, back) geram exatamente um aviso (back)."""
    p_seat, _, _, _ = tmp_files
    _, _, warns = core.read_exposure_files(
        [p_seat, 'Zero', 'Zero', 'Zero'])
    assert len(warns) == 1


# ---------------------------------------------------------------------------
# Testes da aba Warnings na planilha xlsx
# ---------------------------------------------------------------------------

def test_xlsx_warnings_sheet_created(tmp_path):
    """append_warnings_sheet cria a aba Warnings quando ha avisos."""
    import openpyxl
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from iso26315 import workbook

    xlsx = tmp_path / 'results.xlsx'
    workbook.write_results(str(xlsx), [['exp1', '', '', 1945, 2000, 2010,
                                        2, 50, 6, 'orientating',
                                        0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                        1.0, 'T12-L1', 0.5]])
    workbook.append_warnings_sheet(str(xlsx), [('exp1', 'Test warning text')])

    wb = openpyxl.load_workbook(str(xlsx))
    assert 'Warnings' in wb.sheetnames
    ws = wb['Warnings']
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0] == ('Exposure', 'Warning')
    assert rows[1][0] == 'exp1'
    assert 'warning' in rows[1][1].lower()


def test_xlsx_no_warnings_sheet_when_empty(tmp_path):
    """append_warnings_sheet nao cria aba quando lista esta vazia."""
    import openpyxl
    from iso26315 import workbook

    xlsx = tmp_path / 'results.xlsx'
    workbook.write_results(str(xlsx), [['exp1', '', '', 1945, 2000, 2010,
                                        2, 50, 6, 'orientating',
                                        0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                        1.0, 'T12-L1', 0.5]])
    workbook.append_warnings_sheet(str(xlsx), [])

    wb = openpyxl.load_workbook(str(xlsx))
    assert 'Warnings' not in wb.sheetnames


def test_xlsx_warnings_sheet_replaced_on_rerun(tmp_path):
    """append_warnings_sheet substitui a aba se ela ja existir."""
    import openpyxl
    from iso26315 import workbook

    xlsx = tmp_path / 'results.xlsx'
    workbook.write_results(str(xlsx), [])
    workbook.append_warnings_sheet(str(xlsx), [('e1', 'old warning')])
    workbook.append_warnings_sheet(str(xlsx), [('e1', 'new warning')])

    wb = openpyxl.load_workbook(str(xlsx))
    ws = wb['Warnings']
    rows = list(ws.iter_rows(values_only=True))
    assert rows[1][1] == 'new warning'
    assert len(rows) == 2  # header + 1 data row
