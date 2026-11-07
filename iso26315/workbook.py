"""Leitura e escrita das planilhas de historico de exposicao no formato do
software original da ISO 2631-5 (abas Exposure, Pattern, Year).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POSTURE_FROM_NAME = {
    'group1': 1, 'group2': 2, 'group3': 3, 'group4': 4, 'group5': 5,
    'forwarder': 4, 'harvester': 5, 'group7': 7, 'handsinlap': 7,
}


@dataclass
class Exposure:
    identifier: str
    posture: int
    measurement_time: float
    seat: str
    back: str
    hands: str
    feet: str

    def paths(self):
        return [self.seat, self.feet, self.hands, self.back]   # ordem G, F, H, L


@dataclass
class Pattern:
    name: str
    entries: list = field(default_factory=list)   # [(nome_exposicao, duracao_s)]

    @property
    def total_duration(self):
        return sum(d for _, d in self.entries)


@dataclass
class YearRow:
    year: int
    body_mass: float
    body_height: float
    patterns: list = field(default_factory=list)  # [(nome_padrao, dias)]

    @property
    def total_days(self):
        return sum(d for _, d in self.patterns)


@dataclass
class ExposureHistory:
    exposures: dict
    patterns: dict
    years: list
    source: Path = None

    @property
    def first_year(self):
        return self.years[0].year

    @property
    def last_year(self):
        return self.years[-1].year

    def resolve_paths(self, base_dir):
        """Reescreve os caminhos das planilhas (tipicamente D:\\...) para uma
        pasta local, casando pelo nome do arquivo."""
        base_dir = Path(base_dir)
        index = {p.name.lower(): p for p in base_dir.rglob('*') if p.is_file()}
        missing = []
        for exp in self.exposures.values():
            for attr in ('seat', 'back', 'hands', 'feet'):
                val = getattr(exp, attr)
                if not val or val.strip().lower() in ('zero', 'null', 'none'):
                    setattr(exp, attr, 'Zero')
                    continue
                name = val.replace('\\', '/').split('/')[-1].lower()
                if name in index:
                    setattr(exp, attr, str(index[name]))
                else:
                    missing.append(val)
        return missing


def _rows(sheet):
    return [[c for c in row] for row in sheet]


def _read_sheets(path):
    """Devolve {nome_da_aba: lista de linhas} para .xls ou .xlsx."""
    path = Path(path)
    if path.suffix.lower() == '.xls':
        import xlrd
        wb = xlrd.open_workbook(path)
        return {sh.name: [[c.value for c in sh.row(r)] for r in range(sh.nrows)]
                for sh in wb.sheets()}
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    return {name: [list(r) for r in wb[name].iter_rows(values_only=True)]
            for name in wb.sheetnames}


def _clean(v):
    if v is None:
        return ''
    if isinstance(v, str):
        return v.strip()
    return v


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_history(path):
    sheets = {k.lower(): v for k, v in _read_sheets(path).items()}
    for required in ('exposure', 'pattern', 'year'):
        if required not in sheets:
            raise ValueError(f'The spreadsheet has no "{required}" sheet.')

    exposures = {}
    for row in sheets['exposure'][1:]:
        ident = _clean(row[0]) if row else ''
        if not ident:
            continue
        posture_name = str(_clean(row[1])).lower()
        exposures[ident] = Exposure(
            identifier=ident,
            posture=POSTURE_FROM_NAME.get(posture_name, 3),
            measurement_time=_num(row[3]),
            seat=str(_clean(row[4])), back=str(_clean(row[5])),
            hands=str(_clean(row[6])), feet=str(_clean(row[7])),
        )

    patterns = {}
    for row in sheets['pattern'][1:]:
        name = _clean(row[0]) if row else ''
        if not name:
            continue
        entries = []
        for i in range(1, len(row) - 1, 2):
            exp_name, dur = _clean(row[i]), _num(row[i + 1])
            if exp_name and dur:
                entries.append((str(exp_name), dur))
        patterns[str(name)] = Pattern(str(name), entries)

    years = []
    for row in sheets['year'][1:]:
        year = _num(row[0]) if row else None
        if year is None:
            continue
        pats = []
        for i in range(3, len(row) - 1, 2):
            pname, days = _clean(row[i]), _num(row[i + 1])
            if pname and days:
                pats.append((str(pname), days))
        years.append(YearRow(int(year), _num(row[1]), _num(row[2]), pats))

    return ExposureHistory(exposures, patterns, years, Path(path))


def write_results(path, rows, header=None):
    """Grava a planilha de saida no mesmo layout do software original."""
    import openpyxl
    header = header or [
        'Exposure history', 'Description', 'Endplate area [cm2]',
        'Year of birth', 'First year', 'Last year', 'Equation',
        'Protection [%]', 'Exponent', 'Analysis type',
        'R T12-L1', 'R L1-L2', 'R L2-L3', 'R L3-L4', 'R L4-L5', 'R L5-S1',
        'max R', 'Critical level', 'Sd [MPa]',
    ]
    path = Path(path)
    if path.exists():
        wb = openpyxl.load_workbook(path)
        ws = wb['Risk factors'] if 'Risk factors' in wb.sheetnames \
            else wb.create_sheet('Risk factors')
        if ws.max_row == 1 and ws.cell(1, 1).value is None:
            ws.append(header)
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Risk factors'
        ws.append(header)
    for row in rows:
        ws.append(list(row))
    wb.save(path)
    return path


def append_warnings_sheet(path, warning_rows):
    """Acrescenta (ou substitui) a aba 'Warnings' num xlsx existente.

    warning_rows: lista de (exposure_name, warning_text).
    Se a lista estiver vazia, nao faz nada.
    """
    if not warning_rows:
        return
    import openpyxl
    wb = openpyxl.load_workbook(path)
    if 'Warnings' in wb.sheetnames:
        del wb['Warnings']
    ws = wb.create_sheet('Warnings')
    ws.append(['Exposure', 'Warning'])
    for exp_name, warn_text in warning_rows:
        ws.append([exp_name, warn_text])
    wb.save(path)


def write_history(path, exposures, patterns, years):
    """Grava um historico de exposicao no formato lido por read_history().

    exposures: lista de Exposure
    patterns:  lista de Pattern
    years:     lista de YearRow
    """
    import openpyxl
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = 'Exposure'
    ws.append(['Identifier', 'Posture', 'Filename', 'Measurement duration [s]',
               'Excitation at seat', 'Excitation at back',
               'Excitation at hands', 'Excitation at feet'])
    for e in exposures:
        ws.append([e.identifier, f'group{e.posture}', '',
                   e.measurement_time, e.seat, e.back, e.hands, e.feet])

    ws = wb.create_sheet('Pattern')
    width = max((len(p.entries) for p in patterns), default=1)
    head = ['Pattern-name']
    for i in range(width):
        head += [f'Exposure {i + 1}', 'Duration [s]']
    ws.append(head)
    for p in patterns:
        row = [p.name]
        for name, dur in p.entries:
            row += [name, dur]
        ws.append(row)

    ws = wb.create_sheet('Year')
    width = max((len(y.patterns) for y in years), default=1)
    head = ['Year', 'BM [kg]', 'BH [m]']
    for i in range(width):
        head += [f'Pattern {i + 1}', 'Days / Year']
    ws.append(head)
    for y in years:
        row = [y.year, y.body_mass, y.body_height]
        for name, days in y.patterns:
            row += [name, days]
        ws.append(row)

    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = bold
        for col in sheet.columns:
            longest = max(len(str(c.value)) if c.value is not None else 0
                          for c in col)
            sheet.column_dimensions[col[0].column_letter].width = \
                min(max(longest + 2, 12), 46)

    wb.save(path)
    return Path(path)
