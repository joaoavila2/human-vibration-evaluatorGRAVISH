"""Explorador de sinais: inspeção direta de registros de aceleração.

Fora do escopo da ISO 2631-5. Não calcula dose nem fator de risco: serve para
olhar um registro antes de montar um histórico de exposição, comparar arquivos
entre si e exportar espectros.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import core
from .theme import BAD_COLOR, TEXT
from .weighting_panel import WeightingPanel

SERIES_COLORS = ['#1f6feb', '#c2255c', '#1b7a33', '#b25c00', '#6741d9',
                 '#0b7285', '#d6336c', '#495057']

STAT_ROWS = [
    ('Samples', lambda s: f"{s['samples']:,}"),
    ('Duration [s]', lambda s: f"{s['duration']:.3f}"),
    ('Sample rate [Hz]', lambda s: f"{s['sample_rate']:.4g}"),
    ('RMS x, y, z [m/s2]', lambda s: _v(s['rms'])),
    ('Vector sum RMS [m/s2]', lambda s: f"{s['rms_total']:.4f}"),
    ('Peak x, y, z [m/s2]', lambda s: _v(s['peak'], 2)),
    ('Crest factor', lambda s: _v(s['crest'], 1)),
    ('VDV x, y, z [m/s1.75]', lambda s: _v(s['vdv'], 2)),
    ('Kurtosis', lambda s: _v(s['kurtosis'], 1)),
    ('Dominant frequency [Hz]', lambda s: _v(s['dominant'], 2)),
    ('Intensity class', lambda s: str(s['intensity_class'])),
    ('--- ISO 2631-1 weighted ---', lambda s: ''),
    ('Weighting per axis', lambda s: ',  '.join(
        f"{w}(k={k:g})" for w, k in zip(s['w']['weighting'], s['w']['k']))),
    ('a_w x, y, z [m/s2]', lambda s: _v(s['w']['aw'])),
    ('Vector sum a_w [m/s2]', lambda s: f"{s['w']['aw_total']:.4f}"),
    ('Weighted peak [m/s2]', lambda s: _v(s['w']['peak'], 2)),
    ('Weighted crest factor', lambda s: _v(s['w']['crest'], 1)),
    ('Weighted VDV [m/s1.75]', lambda s: _v(s['w']['vdv'], 2)),
    ('--- ISO 2631-5 validity ---', lambda s: ''),
    ('Wk-weighted z peak [m/s2]', lambda s: f"{s['w']['peak'][2]:.2f}"),
    ('Annex A / E applicable', lambda s:
        'yes' if s['w']['peak'][2] <= core.VALIDITY_PEAK_LIMIT
        else f'NO - above {core.VALIDITY_PEAK_LIMIT} m/s2, use Clause 5 / Annexes C, D'),
]


def _v(arr, dec=4):
    return ',  '.join(f'{x:.{dec}f}' for x in arr)


class SignalExplorer(QWidget):
    """Widget autônomo: não depende de planilha nem de funções de transferência."""

    def __init__(self, plot_factory, status):
        super().__init__()
        self.signals = {}          # nome -> Excitation
        self.stats = {}
        self.status = status
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        warn = QLabel(
            'Outside the scope of ISO 2631-5. This tab inspects raw '
            'acceleration records: it computes no dose and no risk factor. '
            'Use it to screen a signal before building an exposure history.')
        warn.setWordWrap(True)
        warn.setStyleSheet(
            'background:#fdf3d2; border:1px solid #e0cf96; border-radius:5px;'
            f'padding:8px; color:{TEXT};')
        lay.addWidget(warn)

        row = QHBoxLayout()

        files = QGroupBox('Records')
        fl = QVBoxLayout(files)
        self.list = QListWidget()
        self.list.setMaximumWidth(260)
        self.list.setMaximumHeight(130)
        self.list.itemChanged.connect(lambda *_: self.refresh())
        self.list.currentRowChanged.connect(lambda *_: self.refresh_stats())
        fl.addWidget(self.list)
        b1 = QPushButton('Add files...')
        b1.clicked.connect(self.add_files)
        fl.addWidget(b1)
        b2 = QPushButton('Remove selected')
        b2.setObjectName('secondary')
        b2.clicked.connect(self.remove_current)
        fl.addWidget(b2)
        b3 = QPushButton('Clear all')
        b3.setObjectName('secondary')
        b3.clicked.connect(self.clear_all)
        fl.addWidget(b3)
        row.addWidget(files)

        ctrl = QGroupBox('View')
        g = QGridLayout(ctrl)
        g.setContentsMargins(8, 4, 8, 4)
        g.setVerticalSpacing(3)
        g.addWidget(QLabel('Plot'), 0, 0)
        self.kind = QComboBox()
        self.kind.addItems(['Time series', 'Spectrum', 'Cumulative RMS',
                            'Phase spectrum', 'Coherence (vs first checked)'])
        self.kind.setCurrentIndex(1)
        g.addWidget(self.kind, 0, 1)
        g.addWidget(QLabel('Axis'), 0, 2)
        self.axis = QComboBox()
        self.axis.addItem('Sum of x, y, z', None)
        for i, n in enumerate(core.AXIS_NAMES):
            self.axis.addItem(n, i)
        self.axis.setCurrentIndex(3)
        g.addWidget(self.axis, 0, 3)

        g.addWidget(QLabel('Method'), 1, 0)
        self.method = QComboBox()
        self.method.addItems(core.SPECTRUM_METHODS)
        self.method.setCurrentIndex(1)
        g.addWidget(self.method, 1, 1)
        g.addWidget(QLabel('Window'), 1, 2)
        self.window = QComboBox()
        self.window.addItems(list(core.WINDOW_FUNCTIONS))
        self.window.setCurrentText('Hann')
        g.addWidget(self.window, 1, 3)

        g.addWidget(QLabel('Segment'), 2, 0)
        self.nperseg = QComboBox()
        for n in (1024, 2048, 4096, 8192, 16384, 32768):
            self.nperseg.addItem(f'{n} samples', n)
        self.nperseg.setCurrentIndex(3)
        g.addWidget(self.nperseg, 2, 1)
        g.addWidget(QLabel('Overlap'), 2, 2)
        self.overlap = QComboBox()
        for o in (0.0, 0.25, 0.5, 0.75):
            self.overlap.addItem(f'{o * 100:.0f} %', o)
        self.overlap.setCurrentIndex(2)
        g.addWidget(self.overlap, 2, 3)

        g.addWidget(QLabel('Band width'), 3, 0)
        self.band = QDoubleSpinBox()
        self.band.setRange(0.1, 10.0)
        self.band.setSingleStep(0.25)
        self.band.setValue(1.0)
        self.band.setSuffix(' Hz')
        g.addWidget(self.band, 3, 1)
        self.logy = QCheckBox('Logarithmic amplitude')
        self.logy.setChecked(True)
        g.addWidget(self.logy, 3, 2, 1, 2)
        self.note = QLabel('')
        self.note.setObjectName('muted')
        self.note.setWordWrap(True)
        g.addWidget(self.note, 5, 0, 1, 4)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        row.addWidget(ctrl, 1)
        lay.addLayout(row)

        self.weight = WeightingPanel()
        self.weight.setMaximumHeight(150)
        self.weight.changed.connect(self._on_weighting)
        lay.addWidget(self.weight)

        for w in (self.kind, self.axis, self.method, self.window,
                  self.nperseg, self.overlap):
            w.currentIndexChanged.connect(self.refresh)
        self.band.valueChanged.connect(self.refresh)
        self.logy.stateChanged.connect(self.refresh)

        body = QHBoxLayout()
        self.plot = plot_factory()
        body.addWidget(self.plot, 4)
        self.table = QTableWidget(len(STAT_ROWS), 2)
        self.table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumWidth(380)
        body.addWidget(self.table, 1)
        lay.addLayout(body, 1)
        lay.setStretch(lay.count() - 1, 1)

        bar = QHBoxLayout()
        bar.addStretch()
        b3 = QPushButton('Export spectrum to CSV...')
        b3.setObjectName('secondary')
        b3.clicked.connect(self.export_csv)
        bar.addWidget(b3)
        lay.addLayout(bar)

    # ------------------------------------------------------------------
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add acceleration records', '',
            'Time series (*.txt *.asc *.dat *.csv);;All files (*)')
        failed = []
        for p in paths:
            name = Path(p).name
            if name in self.signals:
                continue
            self.status(f'Reading {name}...')
            try:
                exc = core.read_excitation(p, name)
            except Exception as err:
                failed.append(f'{name}: {err}')
                continue
            self.signals[name] = exc
            st = core.signal_statistics(exc)
            st['w'] = core.weighted_metrics(exc, self.weight.config())
            self.stats[name] = st
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setToolTip(p)
            self.list.addItem(item)
        self.status('')
        if failed:
            QMessageBox.warning(self, 'Some files could not be read',
                                '\n'.join(failed))
        if self.list.count() and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        self.refresh()
        self.refresh_stats()

    def remove_current(self):
        r = self.list.currentRow()
        if r < 0:
            return
        name = self.list.item(r).text()
        self.list.takeItem(r)
        self.signals.pop(name, None)
        self.stats.pop(name, None)
        self.refresh()
        self.refresh_stats()

    def clear_all(self):
        self.list.clear()
        self.signals.clear()
        self.stats.clear()
        self.refresh()
        self.refresh_stats()

    def _checked(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    # ------------------------------------------------------------------
    def _on_weighting(self):
        for name, exc in self.signals.items():
            self.stats[name]['w'] = core.weighted_metrics(
                exc, self.weight.config())
        self.refresh()
        self.refresh_stats()

    def refresh(self):
        kind = self.kind.currentIndex()
        # 0 time series, 1 spectrum, 2 cumulative RMS, 3 phase, 4 coherence.
        spectrum_kind = kind == 1
        phase_kind = kind == 3
        coherence_kind = kind == 4
        block = self.method.currentText() == core.SPECTRUM_METHODS[0]
        self.method.setEnabled(spectrum_kind)
        self.band.setEnabled(
            spectrum_kind and self.method.currentText() == core.SPECTRUM_METHODS[3])
        for w in (self.window, self.nperseg, self.overlap):
            w.setEnabled((spectrum_kind and not block) or coherence_kind)

        names = self._checked()
        ax, = self.plot.axes()
        if not names:
            ax.text(0.5, 0.5, 'Add a record to begin', ha='center',
                    va='center', color='#6c757d', transform=ax.transAxes)
            self.plot.finish(ax)
            self.note.setText('')
            return

        axis = self.axis.currentData()
        note = ''

        if phase_kind and axis is None:
            ax.text(0.5, 0.5, 'Pick a single axis (x, y or z): phase of '
                    'the vector sum has no direct physical meaning.',
                    ha='center', va='center', color='#6c757d', wrap=True,
                    transform=ax.transAxes)
            self.plot.finish(ax)
            self.note.setText('')
            return

        if coherence_kind:
            if axis is None or len(names) < 2:
                msg = ('Pick a single axis (x, y or z).' if axis is None else
                       'Check at least two records: the first checked one '
                       'is the reference.')
                ax.text(0.5, 0.5, msg, ha='center', va='center',
                        color='#6c757d', wrap=True, transform=ax.transAxes)
                self.plot.finish(ax)
                self.note.setText('')
                return
            ref_name = names[0]
            ref = self.signals[ref_name]
            min_navg, skipped = None, []
            for i, name in enumerate(names[1:], start=1):
                exc = self.signals[name]
                if exc.dt != ref.dt or exc.accel.shape[0] != ref.accel.shape[0]:
                    skipped.append(name)
                    continue
                f, gamma2, navg = core.coherence_spectrum(
                    ref.accel[:, axis], exc.accel[:, axis], ref.dt,
                    window=self.window.currentText(),
                    nperseg=self.nperseg.currentData(),
                    overlap=self.overlap.currentData())
                min_navg = navg if min_navg is None else min(min_navg, navg)
                ax.plot(f, gamma2, lw=1.1,
                        color=SERIES_COLORS[i % len(SERIES_COLORS)],
                        label=f'{ref_name} vs {name}')
            ax.set_xlim(0, core.LOWPASS_HZ)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(r'Coherence $\gamma^2$')
            ax.set_title(f'Coherence vs {ref_name} - {core.AXIS_NAMES[axis]}')
            if skipped:
                note = ('Skipped (different sample rate or length than the '
                        f'reference): {", ".join(skipped)}. ')
            if min_navg is None:
                note += 'No comparable record left to plot.'
            elif min_navg < 4:
                note += (f'Only {min_navg} averaged segment(s): coherence is '
                         'close to 1 everywhere by construction, not yet a '
                         'meaningful estimate. Shorten the segment length '
                         'for more averages.')
            else:
                note += (f'{min_navg} averaged Welch segments. Meaningful '
                         'only between records that are genuinely '
                         'synchronous (same run, same start instant) - '
                         'this generic explorer does not check that the way '
                         'the four-point exposure pipeline does '
                         '(core.check_time_base).')
            n = len(ax.get_legend_handles_labels()[0])
            ax.legend(fontsize=8 if n <= 6 else 7,
                      ncol=1 if n <= 6 else 2, framealpha=0.85,
                      handlelength=1.6, columnspacing=1.0)
            self.note.setText(note)
            self.plot.finish(ax)
            return

        for i, name in enumerate(names):
            exc = self.signals[name]
            colour = SERIES_COLORS[i % len(SERIES_COLORS)]
            if kind == 0:
                y = (np.linalg.norm(exc.accel, axis=1) if axis is None
                     else exc.accel[:, axis])
                ax.plot(exc.time, y, lw=0.4, color=colour, label=name)
            elif kind == 1:
                bands = self.method.currentText() in core.SPECTRUM_METHODS[3:]
                mode = self.weight.combination()
                if axis is None and mode in (1, 2):
                    series = [(c, f'{name} ({core.AXIS_NAMES[c]})',
                               ['-', '--', ':'][c]) for c in range(3)]
                    if mode == 2:
                        series.append((None, f'{name} (vector)', '-'))
                else:
                    series = [(axis, name, '-')]
                for c, lbl, style in series:
                    f, y, ylabel, note = core.analysis_spectrum(
                        exc, self.method.currentText(), axis=c,
                        window=self.window.currentText(),
                        nperseg=self.nperseg.currentData(),
                        overlap=self.overlap.currentData(),
                        band_width=self.band.value(),
                        weighting=self.weight.config())
                    lw = 1.4 if c is None else 0.9
                    if bands:
                        ax.step(f, y, where='mid', lw=1.2, ls=style,
                                color=colour, label=lbl)
                    else:
                        ax.plot(f, y, lw=lw, ls=style, color=colour, label=lbl)
            elif phase_kind:
                f, ph = core.phase_spectrum(exc, axis)
                ax.plot(f, ph, lw=1.0, color=colour, label=name)
            else:
                a = (np.linalg.norm(exc.accel, axis=1) if axis is None
                     else exc.accel[:, axis])
                n = np.arange(1, len(a) + 1)
                ax.plot(exc.time, np.sqrt(np.cumsum(a ** 2) / n), lw=1.0,
                        color=colour, label=name)

        if kind == 0:
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Acceleration [m/s$^2$]')
            ax.set_title('Time series')
        elif kind == 1:
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Spectrum - {self.method.currentText()}')
            if self.method.currentText() == core.SPECTRUM_METHODS[4]:
                ax.set_xscale('log')
            else:
                ax.set_xlim(0, core.LOWPASS_HZ)
            if self.logy.isChecked():
                ax.set_yscale('log')
        elif phase_kind:
            ax.set_xlim(0, core.LOWPASS_HZ)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Phase [deg], unwrapped')
            ax.set_title(f'Phase spectrum (single-block FFT) - '
                        f'{core.AXIS_NAMES[axis]}')
            ax.axhline(0, lw=0.6, color='#adb5bd')
            note = ('Phase of the single-block complex FFT, unwrapped. '
                    'Welch averaging (used by the Spectrum plot) discards '
                    'phase, so it has no phase view of its own.')
        else:
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Cumulative RMS [m/s$^2$]')
            ax.set_title('Running RMS - flat means the record is stationary')
        n = len(ax.get_legend_handles_labels()[0])
        ax.legend(fontsize=8 if n <= 6 else 7,
                  ncol=1 if n <= 6 else 2, framealpha=0.85,
                  handlelength=1.6, columnspacing=1.0)
        self.note.setText(note)
        self.plot.finish(ax)

    def refresh_stats(self):
        r = self.list.currentRow()
        if r < 0:
            self.table.clearContents()
            return
        s = self.stats.get(self.list.item(r).text())
        if s is None:
            return
        for i, (label, fn) in enumerate(STAT_ROWS):
            a = QTableWidgetItem(label)
            fnt = a.font()
            fnt.setBold(True)
            a.setFont(fnt)
            a.setForeground(QColor(TEXT))
            b = QTableWidgetItem(fn(s))
            b.setForeground(QColor(TEXT))
            self.table.setItem(i, 0, a)
            self.table.setItem(i, 1, b)

    # ------------------------------------------------------------------
    def export_csv(self):
        names = self._checked()
        if not names:
            QMessageBox.information(self, 'Export',
                                    'Select at least one record.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export spectrum', 'spectrum.csv', 'CSV (*.csv)')
        if not path:
            return
        axis = self.axis.currentData()
        cols, header = None, ['frequency_Hz']
        for name in names:
            f, y, ylabel, _ = core.analysis_spectrum(
                self.signals[name], self.method.currentText(), axis=axis,
                window=self.window.currentText(),
                nperseg=self.nperseg.currentData(),
                overlap=self.overlap.currentData(),
                band_width=self.band.value(),
                weighting=self.weight.config())
            if cols is None:
                cols = [f]
            elif len(f) != len(cols[0]):
                QMessageBox.warning(
                    self, 'Export',
                    f'"{name}" has a different sample rate, so its frequency '
                    f'grid does not match the others. Export one at a time.')
                return
            cols.append(y)
            header.append(name)
        np.savetxt(path, np.column_stack(cols), delimiter=',',
                   header=','.join(header), comments='', fmt='%.6g')
        self.status(f'Spectrum written to {Path(path).name}')
