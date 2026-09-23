"""Construtor de planilhas de historico de exposicao da ISO 2631-5.

Aplicativo separado do avaliador: serve so para montar os arquivos de entrada
sem precisar editar planilha na mao.
"""
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QStatusBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import core, workbook
from PySide6.QtGui import QIcon
from .theme import ACCENT, BAD_COLOR, TEXT, TEXT_MUTED, apply_theme

APP_NAME = 'Avaliador de Vibração Humana — Planilhas'
MAX_PATTERN_ENTRIES = 4     # exposicoes por padrao diario
MAX_YEAR_PATTERNS = 3       # padroes diferentes por ano


POSTURE_ITEMS = [(n, core.POSTURE_NAMES[n]) for n in (1, 2, 3, 4, 5, 7)]


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _cell(text='', editable=True, tip=None):
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setForeground(QColor(TEXT_MUTED))
    if tip:
        it.setToolTip(tip)
    return it


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1150, 800)
        self.last_dir = str(Path.home())

        central = QWidget()
        root = QVBoxLayout(central)

        intro = QLabel(
            'Build the exposure history and save it as .xlsx. The file opens '
            'directly in the risk assessment tool.<br>'
            'An <b>exposure</b> is one measurement (a machine, a task). A '
            '<b>daily pattern</b> combines exposures into a working day. The '
            '<b>years</b> say how many days each pattern is repeated.')
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        intro.setObjectName('muted')
        root.addWidget(intro)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self._build_exposures())
        split.addWidget(self._build_patterns())
        split.addWidget(self._build_years())
        split.setSizes([250, 190, 330])
        root.addWidget(split, 1)

        bar = QHBoxLayout()
        b_open = QPushButton('Open existing spreadsheet...')
        b_open.setObjectName('secondary')
        b_open.clicked.connect(self.load)
        bar.addWidget(b_open)
        bar.addStretch()
        b_save = QPushButton('Save spreadsheet...')
        b_save.clicked.connect(self.save)
        bar.addWidget(b_save)
        root.addLayout(bar)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Start by adding an exposure.')

    # -- exposicoes ----------------------------------------------------
    def _build_exposures(self):
        box = QGroupBox('1. Measured exposures')
        lay = QVBoxLayout(box)
        self.t_exp = _table(['Identifier', 'Posture', 't_m [s]',
                             'Seat', 'Back', 'Hands', 'Feet'])
        for i, w in enumerate((150, 190, 80, 170, 170, 170, 170)):
            self.t_exp.setColumnWidth(i, w)
        self.t_exp.itemChanged.connect(lambda *_: self._refresh_names())
        lay.addWidget(self.t_exp)

        row = QHBoxLayout()
        b = QPushButton('Add exposure...')
        b.clicked.connect(self.add_exposure)
        row.addWidget(b)
        for label, slot in (('Change seat...', 'seat'),
                            ('Change back...', 'back'),
                            ('Change hands...', 'hands'),
                            ('Change feet...', 'feet')):
            btn = QPushButton(label)
            btn.setObjectName('secondary')
            btn.clicked.connect(lambda _=False, s=slot: self.pick_file(s))
            row.addWidget(btn)
        b2 = QPushButton('Set to Zero')
        b2.setObjectName('secondary')
        b2.clicked.connect(self.set_zero)
        row.addWidget(b2)
        b3 = QPushButton('Remove')
        b3.setObjectName('secondary')
        b3.clicked.connect(lambda: self._remove(self.t_exp))
        row.addWidget(b3)
        row.addStretch()
        lay.addLayout(row)

        hint = QLabel(
            'With no measurement at the back, use the same file as the seat. '
            'With no measurement at hands and feet, set them to Zero. t_m is '
            'the measurement duration that enters Formula (A.3).')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        lay.addWidget(hint)
        return box

    def add_exposure(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Seat acceleration file', self.last_dir,
            'Time series (*.txt *.asc *.dat *.csv);;All files (*)')
        if not path:
            return
        self.last_dir = str(Path(path).parent)
        duration, name = self._probe(path)

        t = self.t_exp
        r = t.rowCount()
        t.blockSignals(True)
        t.insertRow(r)
        t.setItem(r, 0, _cell(name))
        combo = QComboBox()
        for num, label in POSTURE_ITEMS:
            combo.addItem(label, num)
        combo.setCurrentIndex(2)
        t.setCellWidget(r, 1, combo)
        t.setItem(r, 2, _cell(f'{duration:.0f}' if duration else '400'))
        t.setItem(r, 3, _cell(path, tip=path))
        t.setItem(r, 4, _cell(path, tip='Same as the seat; change it if the back was measured.'))
        t.setItem(r, 5, _cell('Zero'))
        t.setItem(r, 6, _cell('Zero'))
        t.blockSignals(False)
        self._refresh_names()
        msg = f'"{name}" added'
        if duration:
            msg += f' - {duration:.1f} s of signal detected'
        self.statusBar().showMessage(msg)

    def _probe(self, path):
        """Le so o inicio e o fim do arquivo para estimar a duracao."""
        name = Path(path).stem
        try:
            with open(path, 'rb') as fh:
                first = fh.readline().decode('latin-1')
                fh.seek(max(0, fh.seek(0, 2) - 4096))
                last = fh.read().decode('latin-1').strip().splitlines()[-1]
            t0 = float(first.split()[0].replace(',', '.'))
            t1 = float(last.split()[0].replace(',', '.'))
            return (t1 - t0) if t1 > t0 else None, name
        except Exception:
            return None, name

    def pick_file(self, which):
        r = self.t_exp.currentRow()
        if r < 0:
            self.statusBar().showMessage('Select an exposure row first.')
            return
        col = {'seat': 3, 'back': 4, 'hands': 5, 'feet': 6}[which]
        path, _ = QFileDialog.getOpenFileName(
            self, 'Acceleration file', self.last_dir,
            'Time series (*.txt *.asc *.dat *.csv);;All files (*)')
        if not path:
            return
        self.last_dir = str(Path(path).parent)
        self.t_exp.setItem(r, col, _cell(path, tip=path))

    def set_zero(self):
        r, c = self.t_exp.currentRow(), self.t_exp.currentColumn()
        if r < 0 or c < 3:
            self.statusBar().showMessage(
                'Select a file cell (seat, back, hands or feet).')
            return
        if c == 3:
            self.statusBar().showMessage(
                'The seat cannot be Zero: it is the mandatory measurement.')
            return
        self.t_exp.setItem(r, c, _cell('Zero'))

    # -- padroes -------------------------------------------------------
    def _build_patterns(self):
        box = QGroupBox('2. Daily patterns')
        lay = QVBoxLayout(box)
        heads = ['Name']
        for i in range(MAX_PATTERN_ENTRIES):
            heads += [f'Exposure {i + 1}', f'Duration {i + 1} [s]']
        self.t_pat = _table(heads)
        self.t_pat.setColumnWidth(0, 120)
        lay.addWidget(self.t_pat)

        row = QHBoxLayout()
        b = QPushButton('Add pattern')
        b.clicked.connect(self.add_pattern)
        row.addWidget(b)
        b2 = QPushButton('Remove')
        b2.setObjectName('secondary')
        b2.clicked.connect(lambda: self._remove(self.t_pat))
        row.addWidget(b2)
        self.pat_sum = QLabel('')
        self.pat_sum.setObjectName('muted')
        row.addWidget(self.pat_sum)
        row.addStretch()
        lay.addLayout(row)

        hint = QLabel('Durations are the daily time spent in each exposure. '
                      'A typical 4 h day totals 14400 s.')
        hint.setObjectName('muted')
        lay.addWidget(hint)
        return box

    def add_pattern(self):
        t = self.t_pat
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, _cell(f'P{r + 1}'))
        names = self._exposure_names()
        for i in range(MAX_PATTERN_ENTRIES):
            combo = QComboBox()
            combo.addItem('(none)')
            combo.addItems(names)
            if i == 0 and names:
                combo.setCurrentIndex(1)
            combo.currentIndexChanged.connect(self._update_pattern_sum)
            t.setCellWidget(r, 1 + 2 * i, combo)
            t.setItem(r, 2 + 2 * i, _cell('14400' if i == 0 else ''))
        self._update_pattern_sum()

    def _update_pattern_sum(self):
        totals = []
        for r in range(self.t_pat.rowCount()):
            tot = 0.0
            for i in range(MAX_PATTERN_ENTRIES):
                w = self.t_pat.cellWidget(r, 1 + 2 * i)
                if w and w.currentIndex() > 0:
                    tot += _f(self.t_pat.item(r, 2 + 2 * i))
            name = self.t_pat.item(r, 0)
            totals.append(f'{name.text() if name else "?"}: {tot / 3600:.2f} h')
        self.pat_sum.setText('   '.join(totals))

    # -- anos ----------------------------------------------------------
    def _build_years(self):
        box = QGroupBox('3. Exposure years')
        lay = QVBoxLayout(box)

        gen = QHBoxLayout()
        self.y_first = QSpinBox(); self.y_first.setRange(1900, 2100)
        self.y_first.setValue(1990)
        self.y_last = QSpinBox(); self.y_last.setRange(1900, 2100)
        self.y_last.setValue(2034)
        self.y_mass = QDoubleSpinBox(); self.y_mass.setRange(30, 200)
        self.y_mass.setValue(85.0); self.y_mass.setSuffix(' kg')
        self.y_height = QDoubleSpinBox(); self.y_height.setRange(1.2, 2.3)
        self.y_height.setValue(1.78); self.y_height.setDecimals(2)
        self.y_height.setSuffix(' m')
        self.y_days = QSpinBox(); self.y_days.setRange(1, 365)
        self.y_days.setValue(240); self.y_days.setSuffix(' days')
        for label, w in (('From', self.y_first), ('to', self.y_last),
                         ('mass', self.y_mass), ('height', self.y_height),
                         ('', self.y_days)):
            if label:
                gen.addWidget(QLabel(label))
            gen.addWidget(w)
        b = QPushButton('Fill years')
        b.clicked.connect(self.fill_years)
        gen.addWidget(b)
        b2 = QPushButton('Clear')
        b2.setObjectName('secondary')
        b2.clicked.connect(lambda: self.t_year.setRowCount(0))
        gen.addWidget(b2)
        gen.addStretch()
        lay.addLayout(gen)

        heads = ['Year', 'Body mass [kg]', 'Height [m]']
        for i in range(MAX_YEAR_PATTERNS):
            heads += [f'Pattern {i + 1}', f'Days {i + 1}']
        self.t_year = _table(heads)
        lay.addWidget(self.t_year)

        self.bmi_label = QLabel('')
        self.bmi_label.setObjectName('muted')
        lay.addWidget(self.bmi_label)

        hint = QLabel('After filling, edit individual rows to reflect changes in body '
                      'mass or routine. Leave a year without a pattern to '
                      'represent time away from work.')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        lay.addWidget(hint)
        return box

    def fill_years(self):
        first, last = self.y_first.value(), self.y_last.value()
        if last < first:
            QMessageBox.warning(self, APP_NAME,
                                'The last year is before the first.')
            return
        names = self._pattern_names()
        if not names:
            QMessageBox.warning(self, APP_NAME,
                                'Create at least one daily pattern first.')
            return
        t = self.t_year
        t.setRowCount(0)
        for year in range(first, last + 1):
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, _cell(year))
            t.setItem(r, 1, _cell(f'{self.y_mass.value():g}'))
            t.setItem(r, 2, _cell(f'{self.y_height.value():g}'))
            for i in range(MAX_YEAR_PATTERNS):
                combo = QComboBox()
                combo.addItem('(none)')
                combo.addItems(names)
                if i == 0:
                    combo.setCurrentIndex(1)
                t.setCellWidget(r, 3 + 2 * i, combo)
                t.setItem(r, 4 + 2 * i,
                          _cell(self.y_days.value() if i == 0 else ''))
        self._update_bmi()

    def _update_bmi(self):
        m, h = self.y_mass.value(), self.y_height.value()
        bmi = m / h ** 2
        k, per = core.anthropometry(m, h)
        self.bmi_label.setText(
            f'BMI {bmi:.1f} kg/m2  ->  {"high" if bmi > 26.1 else "low"} class, '
            f'{core.PERCENTILE_NAMES[(per - 1) % 5]} mass percentile '
            f'(model {k} of 30)')

    # -- utilitarios ---------------------------------------------------
    def _remove(self, table):
        r = table.currentRow()
        if r >= 0:
            table.removeRow(r)
            self._refresh_names()

    def _exposure_names(self):
        out = []
        for r in range(self.t_exp.rowCount()):
            it = self.t_exp.item(r, 0)
            if it and it.text().strip():
                out.append(it.text().strip())
        return out

    def _pattern_names(self):
        out = []
        for r in range(self.t_pat.rowCount()):
            it = self.t_pat.item(r, 0)
            if it and it.text().strip():
                out.append(it.text().strip())
        return out

    def _refresh_names(self):
        """Mantem as listas suspensas em sincronia com os nomes das tabelas."""
        exps = self._exposure_names()
        for r in range(self.t_pat.rowCount()):
            for i in range(MAX_PATTERN_ENTRIES):
                w = self.t_pat.cellWidget(r, 1 + 2 * i)
                if w:
                    _resync(w, exps)
        pats = self._pattern_names()
        for r in range(self.t_year.rowCount()):
            for i in range(MAX_YEAR_PATTERNS):
                w = self.t_year.cellWidget(r, 3 + 2 * i)
                if w:
                    _resync(w, pats)

    # -- entrada e saida -----------------------------------------------
    def collect(self):
        exposures = []
        for r in range(self.t_exp.rowCount()):
            name = _s(self.t_exp.item(r, 0))
            if not name:
                raise ValueError(f'The exposure in row {r + 1} has no name.')
            seat = _s(self.t_exp.item(r, 3))
            if not seat or seat.lower() == 'zero':
                raise ValueError(f'"{name}" needs a seat file.')
            tm = _f(self.t_exp.item(r, 2))
            if tm <= 0:
                raise ValueError(f'"{name}" has an invalid t_m.')
            exposures.append(workbook.Exposure(
                identifier=name,
                posture=self.t_exp.cellWidget(r, 1).currentData(),
                measurement_time=tm,
                seat=seat, back=_s(self.t_exp.item(r, 4)) or 'Zero',
                hands=_s(self.t_exp.item(r, 5)) or 'Zero',
                feet=_s(self.t_exp.item(r, 6)) or 'Zero'))
        if not exposures:
            raise ValueError('Add at least one exposure.')

        patterns = []
        for r in range(self.t_pat.rowCount()):
            name = _s(self.t_pat.item(r, 0))
            if not name:
                continue
            entries = []
            for i in range(MAX_PATTERN_ENTRIES):
                w = self.t_pat.cellWidget(r, 1 + 2 * i)
                dur = _f(self.t_pat.item(r, 2 + 2 * i))
                if w and w.currentIndex() > 0 and dur > 0:
                    entries.append((w.currentText(), dur))
            if entries:
                patterns.append(workbook.Pattern(name, entries))
        if not patterns:
            raise ValueError('Create at least one daily pattern with a duration.')

        years = []
        for r in range(self.t_year.rowCount()):
            year = _f(self.t_year.item(r, 0))
            if not year:
                continue
            pats = []
            for i in range(MAX_YEAR_PATTERNS):
                w = self.t_year.cellWidget(r, 3 + 2 * i)
                days = _f(self.t_year.item(r, 4 + 2 * i))
                if w and w.currentIndex() > 0 and days > 0:
                    pats.append((w.currentText(), days))
            years.append(workbook.YearRow(
                int(year), _f(self.t_year.item(r, 1)),
                _f(self.t_year.item(r, 2)), pats))
        if not years:
            raise ValueError('Fill in the exposure years.')

        known = {p.name for p in patterns}
        for y in years:
            for name, _ in y.patterns:
                if name not in known:
                    raise ValueError(
                        f'Year {y.year} uses pattern "{name}", which does not exist.')
            if y.total_days > 365:
                raise ValueError(
                    f'Year {y.year} totals {y.total_days:g} exposure days.')
            if not y.body_mass or not y.body_height:
                raise ValueError(f'Year {y.year} has no body mass or height.')
        return exposures, patterns, years

    def save(self):
        try:
            exposures, patterns, years = self.collect()
        except ValueError as exc:
            QMessageBox.warning(self, 'Missing information', str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save exposure history', str(Path(self.last_dir) / 'exposure_history.xlsx'),
            'Excel spreadsheet (*.xlsx)')
        if not path:
            return
        try:
            workbook.write_history(path, exposures, patterns, years)
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self.statusBar().showMessage(f'Saved to {path}')
        QMessageBox.information(
            self, APP_NAME,
            f'Spreadsheet saved.\n\nNow open the risk assessment tool and load '
            f'"{Path(path).name}".')

    def load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open exposure history', self.last_dir,
            'Spreadsheets (*.xlsx *.xls);;All files (*)')
        if not path:
            return
        try:
            h = workbook.read_history(path)
        except Exception as exc:
            QMessageBox.critical(self, 'Could not read the file', str(exc))
            return
        self.last_dir = str(Path(path).parent)
        self._populate(h)
        self.statusBar().showMessage(f'Loaded: {Path(path).name}')

    def _populate(self, h):
        self.t_exp.blockSignals(True)
        self.t_exp.setRowCount(0)
        for e in h.exposures.values():
            r = self.t_exp.rowCount()
            self.t_exp.insertRow(r)
            self.t_exp.setItem(r, 0, _cell(e.identifier))
            combo = QComboBox()
            for num, label in POSTURE_ITEMS:
                combo.addItem(label, num)
            idx = next((i for i, (n, _) in enumerate(POSTURE_ITEMS)
                        if n == e.posture), 2)
            combo.setCurrentIndex(idx)
            self.t_exp.setCellWidget(r, 1, combo)
            self.t_exp.setItem(r, 2, _cell(e.measurement_time or 400))
            for c, v in enumerate((e.seat, e.back, e.hands, e.feet), start=3):
                self.t_exp.setItem(r, c, _cell(v, tip=v))
        self.t_exp.blockSignals(False)

        self.t_pat.setRowCount(0)
        names = self._exposure_names()
        for p in h.patterns.values():
            r = self.t_pat.rowCount()
            self.t_pat.insertRow(r)
            self.t_pat.setItem(r, 0, _cell(p.name))
            for i in range(MAX_PATTERN_ENTRIES):
                combo = QComboBox()
                combo.addItem('(none)')
                combo.addItems(names)
                if i < len(p.entries):
                    combo.setCurrentText(p.entries[i][0])
                    self.t_pat.setItem(r, 2 + 2 * i, _cell(p.entries[i][1]))
                else:
                    self.t_pat.setItem(r, 2 + 2 * i, _cell(''))
                combo.currentIndexChanged.connect(self._update_pattern_sum)
                self.t_pat.setCellWidget(r, 1 + 2 * i, combo)

        self.t_year.setRowCount(0)
        pats = self._pattern_names()
        for y in h.years:
            r = self.t_year.rowCount()
            self.t_year.insertRow(r)
            self.t_year.setItem(r, 0, _cell(y.year))
            self.t_year.setItem(r, 1, _cell(y.body_mass or ''))
            self.t_year.setItem(r, 2, _cell(y.body_height or ''))
            for i in range(MAX_YEAR_PATTERNS):
                combo = QComboBox()
                combo.addItem('(none)')
                combo.addItems(pats)
                if i < len(y.patterns):
                    combo.setCurrentText(y.patterns[i][0])
                    self.t_year.setItem(r, 4 + 2 * i, _cell(y.patterns[i][1]))
                else:
                    self.t_year.setItem(r, 4 + 2 * i, _cell(''))
                self.t_year.setCellWidget(r, 3 + 2 * i, combo)
        self._update_pattern_sum()


def _resync(combo, names):
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem('(none)' if combo.count() == 0 else '(none)')
    combo.addItems(names)
    if current in names:
        combo.setCurrentText(current)
    combo.blockSignals(False)


def _s(item):
    return item.text().strip() if item else ''


def _f(item, default=0.0):
    try:
        return float(_s(item).replace(',', '.'))
    except ValueError:
        return default


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    from .gui import asset
    icon = QIcon(str(asset('icon.png')))
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app)
    win = EditorWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
