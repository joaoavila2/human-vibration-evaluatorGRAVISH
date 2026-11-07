"""Explorador de sinais dos registradores de campo: inspecao direta.

Fora do escopo da ISO 2631-5 e da NHO-09/Diretiva: nao monta um dia, nao
assume posicao do sensor e nao calcula nenhum criterio de saude ou conforto.
Serve para olhar um arquivo antes de usa-lo na aba Directive, ou para
investigar um arquivo especifico (orientacao, trechos com defeito, espectro).
Aceita os formatos de fielddata.read_records: os continuos atuais (DateTime
CSV, .dat VIB1, .txt) e o .m antigo de blocos de 2 s a cada 5 s. Nos
continuos os blocos de 2 s sao so janelas de analise consecutivas.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from . import core, freqplots, fielddata as fd, references
from .loaddialog import DAY_FILTER
from .qtworker import run_in_thread
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
            'Outside the ISO 2631-5 / NHO-09 scope. Inspects raw field '
            'records (DateTime CSV, XIAO .dat, ASCII .txt, old .m) one file '
            'at a time: no day is assembled, no sensor position is assumed, '
            'no health or comfort criterion is computed. The vertical axis '
            'shown here is only the column that carries gravity in that '
            'file, for labelling -- use the Directive tab for an actual '
            'assessment.')
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
        self.list.currentRowChanged.connect(lambda *_: self._on_current_changed())
        fl.addWidget(self.list)
        b1 = QPushButton('Add files...')
        b1.clicked.connect(self.add_files)
        fl.addWidget(b1)
        self.add_btn = b1
        self.busy_label = QLabel('')
        self.busy_label.setObjectName('muted')
        self.busy_label.setWordWrap(True)
        fl.addWidget(self.busy_label)
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
                            'Single block waveform',
                            'Spectrogram across blocks (first checked)',
                            'Coherence between records (vs first checked)',
                            'Transmissibility H1 (vs first checked)',
                            'Shock Response Spectrum (one block)',
                            'Transmissibility H1 by input amplitude class '
                            '(vs first checked)'])
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
        self.waterfall = QCheckBox('3D waterfall (spectrogram)')
        g.addWidget(self.waterfall, 3, 0, 1, 2)
        self.align = QCheckBox('Align pairs by cross-correlation')
        self.align.setToolTip(
            'Coherence and H1 between continuous records from different '
            'loggers: line them up by the time of day, then by the '
            'cross-correlation peak (up to '
            f'{core.ALIGN_MAX_LAG_S:g} s), before comparing. A pair whose '
            'delay drifts along the record is refused, with the reason.')
        g.addWidget(self.align, 3, 2, 1, 2)
        self.reject = QCheckBox('Reject bad blocks')
        self.reject.setToolTip(
            'Coherence and H1: leave out blocks whose output/input RMS ratio, '
            'offset step or kurtosis is far from the other blocks (lost '
            'contact, loose sensor, impact), and blocks that reach the '
            'sensor range when it is given. The note says how many and why.')
        g.addWidget(self.reject, 4, 0, 1, 2)
        self.full_scale = QDoubleSpinBox()
        self.full_scale.setRange(0.0, 500.0)
        self.full_scale.setDecimals(1)
        self.full_scale.setSuffix(' g range')
        self.full_scale.setSpecialValueText('range unknown')
        self.full_scale.setToolTip(
            'Sensor full scale (for example 2, 4, 8 or 16 g). Blocks where the '
            'total acceleration, gravity included, reaches 99.5 % of it count '
            'as clipped. 0 = unknown: clipping is not checked.')
        g.addWidget(self.full_scale, 4, 2, 1, 2)
        self.annotate_peaks = QCheckBox(
            'Annotate H1 / SRS peak (fn, |H|, Q, ζ)')
        self.annotate_peaks.setToolTip(
            'H1 view: marks the highest coherent peak with fn, |H|max, Q '
            'and ζ (3 dB bandwidth). SRS view: marks the maximum.')
        g.addWidget(self.annotate_peaks, 5, 0, 1, 2)
        self.freq_note = QLabel('')
        self.freq_note.setObjectName('muted')
        self.freq_note.setWordWrap(True)
        g.addWidget(self.freq_note, 6, 0, 1, 4)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        row.addWidget(ctrl, 1)
        lay.addLayout(row)

        for w in (self.kind, self.axis, self.window):
            w.currentIndexChanged.connect(self.refresh)
        self.conem.stateChanged.connect(self.refresh)
        self.logy.stateChanged.connect(self.refresh)
        self.block_index.valueChanged.connect(self.refresh)
        self.waterfall.stateChanged.connect(self.refresh)
        self.align.stateChanged.connect(self.refresh)
        self.reject.stateChanged.connect(self.refresh)
        self.full_scale.valueChanged.connect(self.refresh)
        self.annotate_peaks.stateChanged.connect(self.refresh)

        body = QHBoxLayout()
        self.plot = plot_factory()
        self.plot.references_changed.connect(self.refresh)
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
        cal_btn = QPushButton('Calibrate...')
        cal_btn.setObjectName('secondary')
        cal_btn.setToolTip(
            'Reference calibrator (FFT peak detection) and static 6-position '
            'calibration for the selected record. Applies the correction '
            'in-place — use "Remove selected" and re-add the file to undo.')
        cal_btn.clicked.connect(self._open_calibration_dialog)
        bar.addWidget(cal_btn)
        export = QPushButton('Export current plot to CSV...')
        export.setObjectName('secondary')
        export.setToolTip(
            'Writes every curve drawn in the plot above (long format: '
            'series, x, y). Spectrogram and waterfall views are not '
            'exported, because they are surfaces rather than curves.')
        export.clicked.connect(self.export_csv)
        bar.addWidget(export)
        export_pdf_btn = QPushButton('Export PDF...')
        export_pdf_btn.setObjectName('secondary')
        export_pdf_btn.setToolTip('Salva o gráfico atual em PDF (A4 landscape).')
        export_pdf_btn.clicked.connect(self.export_pdf)
        bar.addWidget(export_pdf_btn)
        lay.addLayout(bar)

    # ------------------------------------------------------------------
    def add_files(self):
        if getattr(self, '_busy_flag', False):
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add field records', '', DAY_FILTER)
        paths = [p for p in paths if Path(p).name not in self.records]
        if not paths:
            return
        csv_rate = None
        try:
            if any(fd.detect_format(p) == 'csv_ms' for p in paths):
                from PySide6.QtWidgets import QInputDialog
                csv_rate, ok = QInputDialog.getDouble(
                    self, 'Sample rate', 'Time_ms CSV: sample rate the '
                    'firmware was configured for [Hz]:', 100.0, 1.0,
                    20000.0, 1)
                if not ok:
                    return
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, 'Unrecognised file', str(exc))
            return

        def job(report):
            out, failed = [], []
            for i, p in enumerate(paths):
                report(i / len(paths), f'Reading {Path(p).name}...')
                try:
                    for rec in fd.read_records(p, csv_rate=csv_rate):
                        fd.set_orientation(rec, rec.gravity_axis)
                        out.append((Path(str(rec.path)).name, p, rec,
                                    fd.field_statistics(rec)))
                except Exception as err:
                    failed.append(f'{Path(p).name}: {err}')
            return out, failed

        self._busy_flag = True
        self.add_btn.setEnabled(False)
        self.busy_label.setText(f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_added, self._on_add_fail,
                      on_progress=lambda f, m: self.busy_label.setText(m),
                      thread_attr='_add_thread', worker_attr='_add_worker')

    def _end_add(self):
        thread = getattr(self, '_add_thread', None)
        if thread is not None:
            thread.quit()
            thread.wait()
        self._busy_flag = False
        self.add_btn.setEnabled(True)
        self.busy_label.setText('')

    def _on_add_fail(self, tb):
        self._end_add()
        QMessageBox.critical(self, 'Could not read the files',
                             tb.strip().splitlines()[-1])

    def _on_added(self, payload):
        self._end_add()
        loaded, failed = payload
        for name, p, rec, stats in loaded:
            if name in self.records:
                continue
            self.records[name] = rec
            self.stats[name] = stats
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
        self.block_index.setEnabled(kind in (2, 6))
        self.axis.setEnabled(kind != 2)
        self.window.setEnabled(kind in (1, 3, 4, 5, 7))
        self.waterfall.setEnabled(kind == 3)
        self.align.setEnabled(kind in (4, 5, 7))
        self.reject.setEnabled(kind in (4, 5, 7))
        self.full_scale.setEnabled(kind in (4, 5, 7) and self.reject.isChecked())
        self.freq_note.setText('')

        names = self._checked()
        ax, = self.plot.axes()
        if not names:
            ax.text(0.5, 0.5, 'Add a .m file to begin', ha='center',
                    va='center', color='#6c757d', transform=ax.transAxes)
            self.plot.finish(ax)
            return

        axis = self.axis.currentData()
        window = core.WINDOW_FUNCTIONS[self.window.currentText()]
        if kind >= 3:
            self._draw_freq_extra(kind, names, axis)
            return
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

    def _draw_freq_extra(self, kind, names, axis):
        """Espectrograma, coerencia, H1 e SRS sobre os blocos de 2 s."""
        if axis is None:
            freqplots.message(self.plot, 'Pick a single axis (x, y or z).')
            return
        win = self.window.currentText()
        if kind == 3:
            rec = self.records[names[0]]
            blocks = rec.blocks_si()[:, :, axis]
            f, S = core.block_spectrogram(blocks, rec.dt, win)
            t = (rec.seconds - rec.seconds[0]) / 3600.0
            freqplots.draw_spectrogram(
                self.plot, t, f, S,
                f'Spectrogram across blocks - {names[0]} '
                f'({core.AXIS_NAMES[axis]})',
                tlabel='Time since first block [h]',
                waterfall=self.waterfall.isChecked(),
                logy=self.logy.isChecked())
            self.freq_note.setText(
                'One column per 2 s block (consecutive windows of a '
                'continuous record; in old .m files the blocks are '
                'intermittent and gaps are not interpolated). First checked '
                'record only.')
            return
        if kind == 6:
            series = []
            for name in names:
                rec = self.records[name]
                bi = min(self.block_index.value(), rec.n_blocks - 1)
                x = rec.blocks_si()[bi, :, axis]
                fr, sr = core.shock_response_spectrum(x, rec.dt)
                series.append((f'{name} (block {bi})', fr, sr))
            freqplots.draw_srs(
                self.plot, series,
                f'Shock Response Spectrum (Q=10) - {core.AXIS_NAMES[axis]}',
                self.logy.isChecked())
            self.freq_note.setText(freqplots.SRS_NOTE)
            return
        # 4 coerencia, 5 H1, 7 H1 por classe de amplitude
        if len(names) < 2:
            freqplots.message(
                self.plot, 'Check at least two records: the first checked '
                'one is the reference (input).')
            return
        ref = self.records[names[0]]
        classes_kind = kind == 7
        pairs, skipped, aligned, rejected = [], [], [], []
        for name in names[1:2] if classes_kind else names[1:]:
            bx, by, raw, why = self._pair_blocks(ref, self.records[name], axis)
            if bx is None:
                skipped.append(f'{name} ({why})')
                continue
            pairs.append((name, bx, by, raw))
            if why:
                aligned.append(f'{name} {why}')
        if not pairs:
            freqplots.message(
                self.plot, 'No pair can be compared. ' + '; '.join(skipped)
                + '. Coherence and H1 need records from synchronised '
                'loggers, or continuous records aligned by cross-correlation '
                '(option "Align pairs").')
            return
        two = kind in (5, 7)
        axes = self.plot.axes(2) if two else self.plot.axes()
        ax1 = axes[0]
        ax2 = axes[1] if two else None
        navgs = []
        for i, (name, bx, by, raw) in enumerate(pairs, start=1):
            colour = SERIES_COLORS[i % len(SERIES_COLORS)]
            if self.reject.isChecked() or classes_kind:
                seg = core.block_segment_spectra(bx, by, ref.dt, win)
                mask = None
                if self.reject.isChecked():
                    clip = None
                    if self.full_scale.value() > 0:
                        lim = self.full_scale.value() * 9.80665
                        n = seg['n']
                        clip = (core.clipped_segments(raw[0].ravel(), lim, n) |
                                core.clipped_segments(raw[1].ravel(), lim, n))
                    fl = core.segment_flags(seg, clipped=clip)
                    mask = ~fl['rejected']
                    if core.flag_summary(fl):
                        rejected.append(f'{name}: {core.flag_summary(fl)}')
                f = seg['f']
                if classes_kind:
                    classes = core.amplitude_classes(seg, mask=mask)
                    if not classes:
                        skipped.append(f'{name} (no input-amplitude class has '
                                       f'{core.MIN_CLASS_AVERAGES} blocks)')
                    for j, (lo, hi, navg, est) in enumerate(classes):
                        navgs.append(navg)
                        freqplots.draw_transmissibility(
                            ax1, ax2, f, est,
                            core.coherence_confidence_limit(navg),
                            SERIES_COLORS[j % len(SERIES_COLORS)],
                            f'{lo:g}-{hi:g} m/s\u00b2 rms ({navg})',
                            self.plot.estimators)
                    continue
                est, navg = core.frf_from_segments(seg, mask)
            else:
                f, est, navg = core.block_cross_estimates(bx, by, ref.dt, win)
            navgs.append(navg)
            cl = core.coherence_confidence_limit(navg)
            if two:
                freqplots.draw_transmissibility(
                    ax1, ax2, f, est, cl, colour, name, self.plot.estimators)
            else:
                ax1.plot(f, est['g2'], lw=1.1, color=colour,
                         label=f'{names[0]} vs {name}')
        cax = ax2 if two else ax1
        if not navgs:
            freqplots.message(self.plot, 'Nothing to plot. ' + '; '.join(skipped))
            return
        navg = min(navgs)
        cl = core.coherence_confidence_limit(navg)
        cax.axhline(cl, color='#c0392b', lw=1.0, ls='--',
                    label=f'95% confidence limit ({cl:.2f})')
        cax.set_ylim(0, 1.05)
        cax.set_ylabel(r'Coherence $\gamma^2$')
        cax.set_xlabel('Frequency [Hz]')
        ax1.set_xlim(0, core.LOWPASS_HZ)
        if two:
            ax1.axhline(1.0, color='#6c757d', lw=0.6, ls=':')
            references.draw_references(ax1, self.plot.references,
                                       core.LOWPASS_HZ)
            ax1.set_ylabel(f'|H1| (record / {names[0]})')
            if self.logy.isChecked():
                ax1.set_yscale('log')
            ax1.set_title(f'Transmissibility H1 from {names[0]}'
                          + (f' to {pairs[0][0]}, by input amplitude'
                             if classes_kind and pairs else '')
                          + f' - {core.AXIS_NAMES[axis]}')
        else:
            ax1.set_title(f'Coherence vs {names[0]} - {core.AXIS_NAMES[axis]}')
        for a in axes:
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        if classes_kind:
            msg = (f'{sum(navgs)} blocks in {len(navgs)} amplitude classes, '
                   f'each block one segment ({win} window).')
        else:
            msg = (f'{navg} blocks averaged, each block one segment '
                   f'({win} window).')
        if two:
            extra = freqplots.estimator_note(self.plot.estimators)
            if extra:
                msg += ' ' + extra
        if aligned:
            msg += (' Aligned by cross-correlation (positive = record lags '
                    'the reference): ' + ', '.join(aligned) + '.')
        if classes_kind:
            msg += (' One H1 per class of input RMS (block RMS of '
                    f'{names[0]}, unweighted; a class needs '
                    f'{core.MIN_CLASS_AVERAGES} blocks; only the first record '
                    'after the reference is used). The human body is not '
                    'linear: the resonance moves down in frequency and '
                    'magnitude as the amplitude rises.')
        if rejected:
            msg += ' Rejected before averaging: ' + '; '.join(rejected) + '.'
        if skipped:
            msg += ' Skipped: ' + '; '.join(skipped) + '.'
        if (self.annotate_peaks.isChecked() and two and not classes_kind
                and pairs):
            # Anota o pico mais alto de cada par
            for _i, (_name, _bx, _by, _) in enumerate(pairs):
                _f2, _est2, _n2 = core.block_cross_estimates(
                    _bx, _by, ref.dt, win)
                _cl2 = core.coherence_confidence_limit(_n2)
                freqplots.annotate_transmissibility_peak(
                    ax1, _f2, _est2['h1'], _est2['g2'], _cl2,
                    color=SERIES_COLORS[(_i + 1) % len(SERIES_COLORS)])
        self.freq_note.setText(msg)
        self.plot.finish(*axes)

    def _pair_blocks(self, ref, rec, axis):
        """Blocos (n, 2048) da referencia e do registro para os espectros
        cruzados: (bx, by, raw, texto). raw = (rx, ry) sao os mesmos blocos
        com a gravidade (o que o sensor ve, para comparar com o fundo de
        escala). Texto: '' se os blocos ja sao simultaneos (mesmos
        instantes, ate 1 ms), o atraso se "Align pairs" alinhou os dois por
        correlacao cruzada; (None, None, None, motivo) se nao da.

        Alinhamento: so registros continuos com a mesma taxa. Quando os
        dois tem hora do dia, ela da' o encaixe grosso; a correlacao
        (core.align_by_correlation, ate ALIGN_MAX_LAG_S) corrige o resto,
        que e' o erro entre os relogios das unidades."""
        if rec.rate != ref.rate:
            return None, None, None, 'different sample rate'
        same = (rec.n_blocks == ref.n_blocks
                and np.allclose(rec.seconds, ref.seconds, atol=1e-3))
        both = ref.continuous and rec.continuous
        if not (self.align.isChecked() and both):
            if same:
                return (ref.blocks_si()[:, :, axis],
                        rec.blocks_si()[:, :, axis],
                        (ref.blocks_si(False)[:, :, axis],
                         rec.blocks_si(False)[:, :, axis]), '')
            if not self.align.isChecked():
                return None, None, None, ('different block timestamps; try '
                                          '"Align pairs"')
            return None, None, None, ('alignment needs continuous records, '
                                      'not the old intermittent .m blocks')
        dt = ref.dt
        x = ref.blocks_si()[:, :, axis].reshape(-1)
        y = rec.blocks_si()[:, :, axis].reshape(-1)
        rx = ref.blocks_si(False)[:, :, axis].reshape(-1)
        ry = rec.blocks_si(False)[:, :, axis].reshape(-1)
        clock, k0 = 0.0, 0
        if ref.absolute_time and rec.absolute_time:
            clock = float(rec.seconds[0] - ref.seconds[0])
            k0 = int(round(clock / dt))   # y comeca k0 amostras depois de x
        ox, oy = max(0, k0), max(0, -k0)
        n = min(len(x) - ox, len(y) - oy)
        if n * dt < 4 * core.ALIGN_WINDOW_S:
            return None, None, None, (f'overlap of {max(n, 0) * dt:.0f} s is '
                                      'too short to align')
        try:
            xa, ya, info = core.align_by_correlation(
                x[ox:ox + n], y[oy:oy + n], dt)
        except core.AlignmentError as e:
            return None, None, None, str(e)
        k = info['shift']
        ox, oy = ox + max(0, -k), oy + max(0, k)
        m = len(xa) // fd.BLOCK_SAMPLES * fd.BLOCK_SAMPLES
        shape = (-1, fd.BLOCK_SAMPLES)
        total = clock + info['lag']
        return (xa[:m].reshape(shape), ya[:m].reshape(shape),
                (rx[ox:ox + m].reshape(shape), ry[oy:oy + m].reshape(shape)),
                f'{total * 1000:+.0f} ms (clock {clock * 1000:+.0f}, '
                f'correlation {info["lag"] * 1000:+.0f}, rho {info["rho"]:.2f})')

    def export_pdf(self):
        """Exporta o gráfico atual em PDF (A4 landscape) com cabeçalho formal."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime as _dt

        cur_name = self.list.currentItem()
        stem = Path(cur_name.text()).stem if cur_name else 'field_record'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF', f'{stem}_plot.pdf', 'PDF (*.pdf)')
        if not path:
            return

        NAVY  = '#1a2b4a'
        MUTED = '#6b7a8d'

        from PySide6.QtWidgets import QMessageBox, QProgressDialog
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QApplication

        orig_size = self.plot.figure.get_size_inches()
        prog = None
        try:
            self.plot.figure.set_size_inches(11.69, 8.27)
            self.plot.figure.subplots_adjust(top=0.88)
            self.plot.figure.text(
                0.012, 0.975,
                'AVALIADOR DE VIBRAÇÃO HUMANA  ·  v3.0',
                fontsize=11, fontweight='bold', color='white',
                transform=self.plot.figure.transFigure,
                bbox=dict(facecolor=NAVY, edgecolor='none',
                          boxstyle='square,pad=0.4', alpha=1))
            self.plot.figure.text(
                0.988, 0.975,
                f'GRAVI-SH  ·  UFMG  |  {_dt.now().strftime("%Y-%m-%d  %H:%M")}',
                fontsize=8.5, color=MUTED, ha='right',
                transform=self.plot.figure.transFigure)
            self.plot.figure.text(
                0.012, 0.945,
                f'Field Records  —  {cur_name.text() if cur_name else ""}  '
                f'—  {self.kind.currentText()}',
                fontsize=8, color=NAVY, fontweight='bold',
                transform=self.plot.figure.transFigure)

            prog = QProgressDialog('Gerando PDF…', '', 0, 0, self)
            prog.setWindowTitle('Exportar PDF')
            prog.setWindowModality(_Qt.WindowModality.WindowModal)
            prog.setCancelButton(None)
            prog.setMinimumDuration(0)
            prog.show()
            QApplication.processEvents()

            with PdfPages(path) as pdf:
                pdf.savefig(self.plot.figure, bbox_inches='tight', dpi=150)
        except Exception as _e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(_e))
            return
        finally:
            if prog is not None:
                prog.close()
            self.plot.figure.texts.clear()
            self.plot.figure.subplots_adjust(top=0.88)
            self.plot.figure.set_size_inches(*orig_size)
            self.plot.figure.canvas.draw_idle()

        self.status(f'PDF salvo em {Path(path).name}.')

    def export_csv(self):
        """CSV (formato longo) das curvas desenhadas no grafico atual."""
        import csv
        from datetime import datetime
        rows = []
        for ax in self.plot.figure.axes:
            if hasattr(ax, 'get_zlim'):
                continue
            for line in ax.get_lines():
                label = line.get_label()
                if not label or label.startswith('_'):
                    continue
                x, y = line.get_data()
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                if len(x) > 2 and np.isfinite(y).any():
                    rows.append((label, x, y))
        if not rows:
            QMessageBox.information(
                self, 'Export',
                'The current plot has no exportable curves (spectrogram and '
                'waterfall views are surfaces). Choose another plot.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export plot data', 'field_plot.csv', 'CSV (*.csv)')
        if not path:
            return
        title = self.plot.figure.axes[0].get_title() if self.plot.figure.axes else ''
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            fh.write(f'# exported {datetime.now().isoformat(timespec="seconds")}\n')
            fh.write(f'# plot={self.kind.currentText()}\n')
            fh.write(f'# title={title}\n')
            w = csv.writer(fh)
            w.writerow(['series', 'x', 'y'])
            for label, x, y in rows:
                for xv, yv in zip(x, y):
                    if np.isfinite(yv):
                        w.writerow([label, f'{xv:.9g}', f'{yv:.9g}'])
        self.status(f'Exported {len(rows)} curve(s) to {path}')

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
    def _open_calibration_dialog(self):
        """Calibrador de referencia e 6 posicoes estaticas para o registro
        selecionado. Aplica a correcao in-place nos blocos — use
        'Remove selected' e recarregue o arquivo para desfazer."""
        r = self.list.currentRow()
        if r < 0:
            QMessageBox.information(self, 'Calibration',
                                    'Select a record in the list first.')
            return
        name = self.list.item(r).text()
        record = self.records[name]

        dlg = QDialog(self)
        dlg.setWindowTitle(f'Calibration — {name}')
        dlg.setMinimumWidth(500)
        vlay = QVBoxLayout(dlg)

        tabs = QTabWidget()

        # ---- Tab 1: Reference calibrator ---------------------------------
        tab1 = QWidget()
        form1 = QFormLayout(tab1)
        form1.setContentsMargins(8, 8, 8, 8)

        note1 = QLabel(
            'Input a sinusoidal reference signal recorded on the selected '
            'axis. The FFT peak is located and a gain correction is '
            'computed and applied to that axis in all blocks.')
        note1.setWordWrap(True)
        form1.addRow(note1)

        ax1 = QComboBox()
        ax1.addItems(core.AXIS_NAMES)
        ax1.setCurrentIndex(record.gravity_axis)
        form1.addRow('Axis to calibrate', ax1)

        ref_val = QDoubleSpinBox()
        ref_val.setRange(0.01, 10000.0)
        ref_val.setDecimals(4)
        ref_val.setValue(10.0)
        ref_val.setSuffix(' m/s²')
        form1.addRow('Reference amplitude', ref_val)

        ref_type = QComboBox()
        ref_type.addItems(['Peak (0-to-peak)', 'RMS'])
        ref_type.setToolTip(
            'Check the calibrator certificate: some report peak, others RMS. '
            'Mixing them gives ~41 % error.')
        form1.addRow('Amplitude type', ref_type)

        ref_freq = QDoubleSpinBox()
        ref_freq.setRange(1.0, 20000.0)
        ref_freq.setDecimals(1)
        ref_freq.setValue(159.2)
        ref_freq.setSuffix(' Hz')
        form1.addRow('Reference frequency', ref_freq)

        res1_lbl = QLabel('Click "Compute" to calculate the correction factor.')
        res1_lbl.setWordWrap(True)
        form1.addRow('Result', res1_lbl)

        _res1 = [None]

        def _compute_ref():
            ai = ax1.currentIndex()
            sig = record.blocks[:, :, ai].reshape(-1)
            is_peak = ref_type.currentIndex() == 0
            r1 = core.calibrate_from_reference(
                sig, record.dt, ref_val.value(), ref_freq.value(), is_peak)
            _res1[0] = r1
            color = 'black' if r1['valid'] else '#c0392b'
            res1_lbl.setStyleSheet(f'color:{color}')
            res1_lbl.setText(r1['message'])

        btn_c1 = QPushButton('Compute')
        btn_c1.clicked.connect(_compute_ref)
        form1.addRow('', btn_c1)

        tabs.addTab(tab1, 'Reference calibrator')

        # ---- Tab 2: Static 6-position ------------------------------------
        tab2 = QWidget()
        form2 = QFormLayout(tab2)
        form2.setContentsMargins(8, 8, 8, 8)

        note2 = QLabel(
            'Place the sensor in each orientation and enter the mean '
            'reading. Offset and gain are applied to all three axes in '
            'all blocks.')
        note2.setWordWrap(True)
        form2.addRow(note2)

        pos_fields = {}
        for ax_name in core.AXIS_NAMES:
            for sign, direction in [('+', 'up / +g'), ('-', 'down / −g')]:
                key = f'{ax_name}{sign}'
                sp = QDoubleSpinBox()
                sp.setRange(-200.0, 200.0)
                sp.setDecimals(4)
                sp.setValue(9.80665 if sign == '+' else -9.80665)
                sp.setSuffix(' m/s²')
                form2.addRow(f'{key}  ({ax_name} axis {direction})', sp)
                pos_fields[key] = sp

        res2_lbl = QLabel('Click "Compute" to calculate offset/gain per axis.')
        res2_lbl.setWordWrap(True)
        form2.addRow('Result', res2_lbl)

        _res2 = [None]

        def _compute_static():
            readings = {k: v.value() for k, v in pos_fields.items()}
            r2 = core.calibrate_static_6pos(readings)
            _res2[0] = r2
            lines = [f"{ax_name}: offset {r2[ax_name]['offset']:+.4f} m/s², "
                     f"gain {r2[ax_name]['gain']:.4f}"
                     for ax_name in core.AXIS_NAMES]
            res2_lbl.setText('  |  '.join(lines))

        btn_c2 = QPushButton('Compute')
        btn_c2.clicked.connect(_compute_static)
        form2.addRow('', btn_c2)

        tabs.addTab(tab2, '6-position static')

        vlay.addWidget(tabs)

        # ---- Apply / Close buttons ---------------------------------------
        btns = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)

        def _apply():
            tidx = tabs.currentIndex()
            if tidx == 0:
                r1 = _res1[0]
                if not r1 or not r1['valid']:
                    QMessageBox.information(dlg, 'Nothing to apply',
                        'Click Compute first and verify the result is valid.')
                    return
                ai = ax1.currentIndex()
                record.blocks[:, :, ai] = record.blocks[:, :, ai] * r1['correction']
                msg = (f"Axis {ax1.currentText()} scaled by "
                       f"{r1['correction']:.4f}.")
            else:
                r2 = _res2[0]
                if not r2:
                    QMessageBox.information(dlg, 'Nothing to apply',
                        'Click Compute first.')
                    return
                for i, ax_name in enumerate(core.AXIS_NAMES):
                    d = r2[ax_name]
                    if abs(d['gain']) > 1e-6:
                        record.blocks[:, :, i] = (
                            (record.blocks[:, :, i] - d['offset']) / d['gain'])
                msg = 'Offset and gain correction applied to all axes.'

            # Recompute statistics so the stats panel reflects the correction.
            self.stats[name] = fd.field_statistics(record)
            self.refresh_stats()
            self.refresh()
            QMessageBox.information(dlg, 'Applied', msg)

        btns.button(QDialogButtonBox.Apply).clicked.connect(_apply)
        btns.rejected.connect(dlg.reject)
        vlay.addWidget(btns)
        dlg.exec()
