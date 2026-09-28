"""Explorador de sinais do registrador de campo: inspecao direta de .m.

Fora do escopo da ISO 2631-5 e da NHO-09/Diretiva: nao monta um dia, nao
assume posicao do sensor e nao calcula nenhum criterio de saude ou conforto.
Serve para olhar um arquivo .m antes de usa-lo na aba "Field logger", ou para
investigar um arquivo especifico (orientacao, blocos com defeito, espectro).
Mesmo espirito do SignalExplorer (explorer.py) para registros ASCII do
Anexo A, adaptado ao formato de blocos de 2 s a cada 5 s do registrador.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import core, fielddata as fd
from .theme import TEXT

SERIES_COLORS = ['#1f6feb', '#c2255c', '#1b7a33', '#b25c00', '#6741d9',
                 '#0b7285', '#d6336c', '#495057']

STAT_ROWS = [
    ('Blocks', lambda s: f"{s['n_blocks']:,}"),
    ('Recorded time [h]', lambda s: f"{s['recorded_time'] / 3600:.2f}"),
    ('Period covered [h]', lambda s: f"{s['span'] / 3600:.2f}"),
    ('Duty cycle', lambda s: f"{s['duty_cycle'] * 100:.0f}%"),
    ('Gravity column (file, 1-3)', lambda s: str(s['gravity_axis'] + 1)),
    ('RMS x, y, z [m/s2]', lambda s: _v(s['rms'])),
    ('Vector sum RMS [m/s2]', lambda s: f"{s['rms_total']:.4f}"),
    ('Peak x, y, z [m/s2]', lambda s: _v(s['peak'], 2)),
    ('Crest factor', lambda s: _v(s['crest'], 1)),
    ('VDV x, y, z [m/s1.75]', lambda s: _v(s['vdv'], 2)),
    ('Kurtosis', lambda s: _v(s['kurtosis'], 1)),
    ('Dominant frequency [Hz]', lambda s: _v(s['dominant'], 2)),
    ('Notes', lambda s: '; '.join(s['notes']) if s['notes'] else '-'),
]


def _v(arr, dec=4):
    return ',  '.join('nan' if np.isnan(x) else f'{x:.{dec}f}' for x in arr)


class FieldSignalExplorer(QWidget):
    """Widget autonomo: nao depende de dia, posicao ou planilha."""

    def __init__(self, plot_factory, status):
        super().__init__()
        self.records = {}          # nome -> FieldRecord
        self.stats = {}
        self.status = status
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        warn = QLabel(
            'Outside the ISO 2631-5 / NHO-09 scope. Inspects raw .m records '
            'one file at a time: no day is assembled, no sensor position is '
            'assumed, no health or comfort criterion is computed. The '
            'vertical axis shown here is only the column that carries '
            'gravity in that file, for labelling -- use "Field logger" for '
            'an actual assessment.')
        warn.setWordWrap(True)
        warn.setStyleSheet(
            'background:#fdf3d2; border:1px solid #e0cf96; border-radius:5px;'
            f'padding:8px; color:{TEXT};')
        lay.addWidget(warn)

        row = QHBoxLayout()

        files = QGroupBox('Records (.m)')
        fl = QVBoxLayout(files)
        self.list = QListWidget()
        self.list.setMaximumWidth(260)
        self.list.setMaximumHeight(130)
        self.list.itemChanged.connect(lambda *_: self.refresh())
        self.list.currentRowChanged.connect(lambda *_: self._on_current_changed())
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
        self.kind.addItems(['Timeline: RMS per block',
                            'Spectrum per block (Welch)',
                            'Single block waveform'])
        self.kind.setCurrentIndex(1)
        g.addWidget(self.kind, 0, 1)
        g.addWidget(QLabel('Axis'), 0, 2)
        self.axis = QComboBox()
        self.axis.addItem('x, y, z', None)
        for i, n in enumerate(core.AXIS_NAMES):
            self.axis.addItem(n, i)
        g.addWidget(self.axis, 0, 3)

        g.addWidget(QLabel('Window'), 1, 0)
        self.window = QComboBox()
        self.window.addItems(list(core.WINDOW_FUNCTIONS))
        self.window.setCurrentText('Hann')
        g.addWidget(self.window, 1, 1)
        self.conem = QCheckBox('CONEM2024 curve')
        self.conem.setToolTip('Frequency-response correction of the sensor, '
                              'derived for one unit.')
        g.addWidget(self.conem, 1, 2, 1, 2)

        g.addWidget(QLabel('Block #'), 2, 0)
        self.block_index = QSpinBox()
        self.block_index.setRange(0, 0)
        g.addWidget(self.block_index, 2, 1)
        self.logy = QCheckBox('Logarithmic amplitude')
        self.logy.setChecked(True)
        g.addWidget(self.logy, 2, 2, 1, 2)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        row.addWidget(ctrl, 1)
        lay.addLayout(row)

        for w in (self.kind, self.axis, self.window):
            w.currentIndexChanged.connect(self.refresh)
        self.conem.stateChanged.connect(self.refresh)
        self.logy.stateChanged.connect(self.refresh)
        self.block_index.valueChanged.connect(self.refresh)

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

    # ------------------------------------------------------------------
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add field logger records', '',
            'Field logger (*.m);;All files (*)')
        failed = []
        for p in paths:
            name = Path(p).name
            if name in self.records:
                continue
            self.status(f'Reading {name}...')
            try:
                rec = fd.read_m(p)
                fd.set_orientation(rec, rec.gravity_axis)
            except Exception as err:
                failed.append(f'{name}: {err}')
                continue
            self.records[name] = rec
            self.stats[name] = fd.field_statistics(rec)
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
        self._on_current_changed()

    def remove_current(self):
        r = self.list.currentRow()
        if r < 0:
            return
        name = self.list.item(r).text()
        self.list.takeItem(r)
        self.records.pop(name, None)
        self.stats.pop(name, None)
        self.refresh()
        self._on_current_changed()

    def clear_all(self):
        self.list.clear()
        self.records.clear()
        self.stats.clear()
        self.refresh()
        self._on_current_changed()

    def _checked(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _on_current_changed(self):
        r = self.list.currentRow()
        name = self.list.item(r).text() if r >= 0 else None
        rec = self.records.get(name)
        self.block_index.setMaximum(max(0, rec.n_blocks - 1) if rec else 0)
        self.refresh_stats()

    # ------------------------------------------------------------------
    def refresh(self):
        kind = self.kind.currentIndex()
        self.plot.reset_if_changed(kind)
        self.window.setEnabled(kind == 1)
        self.conem.setEnabled(kind == 1)
        self.block_index.setEnabled(kind == 2)
        self.axis.setEnabled(kind != 2)

        names = self._checked()
        ax, = self.plot.axes()
        if not names:
            ax.text(0.5, 0.5, 'Add a .m file to begin', ha='center',
                    va='center', color='#6c757d', transform=ax.transAxes)
            self.plot.finish(ax)
            return

        axis = self.axis.currentData()
        window = core.WINDOW_FUNCTIONS[self.window.currentText()]
        for i, name in enumerate(names):
            rec = self.records[name]
            colour = SERIES_COLORS[i % len(SERIES_COLORS)]
            if kind == 0:
                blocks = rec.blocks_si()
                t = (rec.seconds - rec.seconds[0]) / 3600.0
                rms = np.sqrt((blocks ** 2).mean(axis=1))
                axes_to_plot = [axis] if axis is not None else [0, 1, 2]
                for c in axes_to_plot:
                    style = '-' if axis is not None else ['-', '--', ':'][c]
                    ax.plot(t, rms[:, c], lw=0.6, color=colour, ls=style,
                            label=f'{name} ({core.AXIS_NAMES[c]})')
            elif kind == 1:
                axes_to_plot = [axis] if axis is not None else [0, 1, 2]
                for c in axes_to_plot:
                    style = '-' if axis is not None else ['-', '--', ':'][c]
                    f, amp, n = fd.block_spectrum(
                        rec, axis=c, window=window,
                        apply_conem=self.conem.isChecked())
                    ax.plot(f, amp, lw=0.9, color=colour, ls=style,
                            label=f'{name} ({core.AXIS_NAMES[c]}, {n} blocks)')
            else:
                bi = min(self.block_index.value(), rec.n_blocks - 1)
                blocks = rec.blocks_si()
                t = np.arange(blocks.shape[1]) * rec.dt
                for c, n in enumerate(core.AXIS_NAMES):
                    ax.plot(t, blocks[bi, :, c], lw=0.7,
                            color=SERIES_COLORS[c], label=f'{name} ({n})')

        if kind == 0:
            ax.set_xlabel('Time since first block [h]')
            ax.set_ylabel('RMS of the 2 s block [m/s$^2$]')
            ax.set_title('Unweighted RMS per block')
        elif kind == 1:
            ax.set_xlim(0, 100)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Amplitude [m/s$^2$]')
            ax.set_title('Welch spectrum, each block a segment')
        else:
            ax.set_xlabel('Time within block [s]')
            ax.set_ylabel('Acceleration [m/s$^2$]')
            ax.set_title(f'Block {self.block_index.value()} waveform')
        if self.logy.isChecked() and kind != 2:
            ax.set_yscale('log')
        n = len(ax.get_legend_handles_labels()[0])
        if n:
            ax.legend(fontsize=8 if n <= 6 else 7,
                      ncol=1 if n <= 6 else 2, framealpha=0.85,
                      handlelength=1.6, columnspacing=1.0)
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
