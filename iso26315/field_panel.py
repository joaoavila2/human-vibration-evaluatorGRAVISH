"""Sub-aba do explorador: dia de campo e avaliacao NHO-09 / Diretiva.

Recebe um dia inteiro (varios arquivos da mesma unidade) em qualquer formato
de fielddata.load_day_auto. As medicoes atuais sao continuas; o formato .m
antigo, com blocos de 2 s a cada 5 s, continua aceito para reanalise.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import freqplots
from .layoututil import CollapsibleSection, settings_splitter
from .loaddialog import ask_day_files
from . import (core, directive, fielddata as fd, labmethod, nho09,
               positions as ps, references as refs, summary)
from .qtworker import run_in_thread
from .theme import ACCENT, BAD_COLOR, OK_COLOR, TEXT, WARN_COLOR

UNCERTAIN_COLOR = '#d9480f'     # regiao de incerteza da NHO-09

AXIS_COLORS = ['#1f6feb', '#c2255c', '#1b7a33']
SERIES_COLORS = ['#1f6feb', '#c2255c', '#1b7a33', '#b25c00', '#6741d9', '#0b7285']


def _prime(day, position, hours, cfg, criterion_idx, extra_hours, custom=None):
    """So esquenta os caches de ponderacao em frequencia (nho09.weighted_
    blocks), fora da UI thread. Os retornos saem descartados aqui de
    proposito: quando o codigo de renderizacao roda depois, na UI thread,
    faz as MESMAS chamadas e acha tudo em cache, entao fica rapido. Nao muda
    nenhum resultado, so onde ele e calculado -- ver FieldPanel._refresh.

    Roda inteiramente com valores simples (nao widgets do Qt), porque tocar
    um QWidget fora da UI thread corrompe a interface.
    """
    summary.normative_criteria(day, position, hours, custom)
    summary.lab_criteria(day, cfg)
    summary.per_file(day, hours)
    if position != 'seat':
        ps.comfort(day.blocks, day.dt, position, custom=custom)
        if position in ('backrest', 'feet'):
            ps.health_informative(day.blocks, day.dt, hours, position)
    elif criterion_idx > 0:
        mode = 'highest' if criterion_idx == 1 else 'vector'
        directive.assess(day.blocks, day.dt, hours, mode=mode)
    else:
        nho09.assess(day.blocks, day.dt, hours, moving_mask=day.occupied)
    if extra_hours is not None:
        nho09.assess(day.blocks, day.dt, extra_hours)


class FieldPanel(QWidget):
    _LAB_APPLY_CLEAN_STYLE = 'padding: 4px 10px;'
    _LAB_APPLY_DIRTY_STYLE = (
        f'background: {WARN_COLOR}; color: #1a1d21; padding: 4px 10px; '
        f'border: 2px solid #9a5a00; border-radius: 4px;')

    def __init__(self, plot_factory, status, store=None, campaign=None):
        super().__init__()
        self.status = status
        self.store = store
        self.campaign = campaign
        self.day = None
        self.result = None
        self.extra = None
        self._busy_flag = False
        self._refresh_then = None
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        upper = QWidget()
        lay = QVBoxLayout(upper)
        lay.setContentsMargins(0, 0, 0, 0)

        note = QLabel(
            'Field day: continuous records from the current loggers '
            '(DateTime CSV, XIAO .dat, ASCII .txt). Each file is weighted '
            'and assessed as one continuous signal. Old .m files (2 s every '
            '5 s) are still accepted; for those, blocks are never glued and '
            'the Annex A dose does not apply, because 60% of the shocks '
            'were not recorded.')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        busy_row = QHBoxLayout()
        self.busy_label = QLabel('')
        self.busy_label.setObjectName('muted')
        busy_row.addWidget(self.busy_label)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        busy_row.addWidget(self.progress)
        busy_row.addStretch()
        lay.addLayout(busy_row)

        top = QHBoxLayout()

        src = QGroupBox('Day')
        sv = QVBoxLayout(src)
        self.load_btn = QPushButton('Load day (DateTime CSV, .dat, .txt, .m)...')
        self.load_btn.clicked.connect(self.load)
        sv.addWidget(self.load_btn)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.setObjectName('secondary')
        self.clear_btn.clicked.connect(self.clear_day)
        sv.addWidget(self.clear_btn)
        self.info = QLabel('No data loaded')
        self.info.setObjectName('muted')
        self.info.setWordWrap(True)
        sv.addWidget(self.info)
        self.calibrate = QCheckBox('Correct gain from gravity (see note)')
        self.calibrate.setChecked(False)
        self.calibrate.setToolTip(
            'Off by default. The deviation of the measured gravity from '
            '9.784 m/s2 can be a gain error OR a zero-g offset, and a single '
            'orientation cannot tell them apart. The MPU6050 offset alone '
            'explains deviations of up to 8%. Offsets do not affect vibration; '
            'correcting them as gain would add error. A six-position '
            'calibration separates the two.')
        sv.addWidget(self.calibrate)

        self.force_continuous = QCheckBox('Treat files as continuous in time')
        self.force_continuous.setToolTip(
            'Remove time gaps between files in timeline and time-series plots. '
            'Useful when multiple recordings from the same session are loaded '
            'together but their timestamps are not perfectly aligned.')
        self.force_continuous.stateChanged.connect(self.redraw)
        sv.addWidget(self.force_continuous)

        self.lab_open_btn = QPushButton('Lab method (MATLAB app)...')
        self.lab_open_btn.clicked.connect(self._open_lab_dialog)
        sv.addWidget(self.lab_open_btn)
        self.lab_summary = QLabel('')
        self.lab_summary.setObjectName('muted')
        self.lab_summary.setWordWrap(True)
        sv.addWidget(self.lab_summary)
        sv.addStretch()
        top.addWidget(CollapsibleSection('', src, expanded=True), 1)

        self.lab_dialog = QDialog(self)
        self.lab_dialog.setWindowTitle('Lab method (MATLAB app) — for comparison only')
        self.lab_dialog.setMinimumWidth(520)
        dv = QVBoxLayout(self.lab_dialog)
        lg = QGridLayout()
        dv.addLayout(lg)
        lg.setContentsMargins(8, 4, 8, 4)
        lg.setVerticalSpacing(3)
        self.lab_apply = QPushButton('Apply weighting / gain')
        self.lab_apply.setToolTip(
            'Weighting and gain changes above are staged, not recomputed on '
            'every click — refiltering a whole day is slow. Change as many '
            'axes as needed, then click here (or Apply below) once.')
        self.lab_apply.setMinimumHeight(34)
        f = self.lab_apply.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        self.lab_apply.setFont(f)
        self.lab_apply.setStyleSheet(self._LAB_APPLY_CLEAN_STYLE)
        self.lab_apply.clicked.connect(self._apply_lab)
        lg.addWidget(self.lab_apply, 0, 0, 1, 4)
        self.lab_w, self.lab_k = [], []
        for c, ax in enumerate('xyz'):
            lg.addWidget(QLabel(f'{ax}'), 1, c + 1, alignment=Qt.AlignCenter)
            box = QComboBox()
            for w in labmethod.WEIGHTINGS:
                box.addItem('None' if w is None else w, w)
            box.setCurrentIndex(labmethod.WEIGHTINGS.index(
                labmethod.LabConfig().weights[c]))
            box.currentIndexChanged.connect(self._mark_lab_dirty)
            lg.addWidget(box, 2, c + 1)
            kk = QDoubleSpinBox()
            kk.setRange(0.0, 5.0)
            kk.setSingleStep(0.1)
            kk.setValue(labmethod.LabConfig().k[c])
            kk.valueChanged.connect(self._mark_lab_dirty)
            lg.addWidget(kk, 3, c + 1)
            self.lab_w.append(box)
            self.lab_k.append(kk)
        lg.addWidget(QLabel('Weighting'), 2, 0)
        lg.addWidget(QLabel('Gain k'), 3, 0)
        self.lab_axis_mode = QComboBox()
        self.lab_axis_mode.addItem(
            'Auto-detect vertical axis from gravity (not what the app does)',
            'auto')
        self.lab_axis_mode.addItem(
            'Raw column order (app: "Exchange x-z axis" OFF, its default)',
            'raw')
        self.lab_axis_mode.addItem(
            'Swap x <-> z (app: "Exchange x-z axis" ON)', 'swap')
        self.lab_axis_mode.setToolTip(
            'The app NEVER looks at gravity: it always applies the '
            'weighting to the file\'s columns in a fixed order (or swaps '
            'columns 1 and 3 if you tick its checkbox), no matter which one '
            'is actually vertical. "Auto" is an improvement this software '
            'adds on top — it only matches the app when the file\'s '
            'orientation happens to agree with the fixed assumption. If '
            'this software\'s reproduction of the app diverges from the '
            'app\'s own screen, try "Raw column order" or "Swap x <-> z" '
            'here to match whichever checkbox state produced that screen, '
            'before suspecting the weighting itself.')
        self.lab_axis_mode.currentIndexChanged.connect(self._mark_lab_dirty)
        lg.addWidget(self.lab_axis_mode, 4, 0, 1, 4)
        self.lab_rms = QComboBox()
        for t in (0.5, 1.0, 2.0):
            self.lab_rms.addItem(f'RMS time {t:g} s', t)
        self.lab_rms.setCurrentIndex(2)
        self.lab_rms.currentIndexChanged.connect(self._mark_lab_dirty)
        lg.addWidget(self.lab_rms, 5, 0, 1, 2)
        self.lab_bug = QCheckBox('Reproduce the app VDV defects')
        self.lab_bug.setChecked(False)
        self.lab_bug.setToolTip(
            'The app omits the time step in the VDV (x 1024^1/4 = 5.657), '
            'extrapolates the partial VDV with the covered period instead of '
            'the recorded time, and combines axes by root-sum-of-squares. Off '
            '(default): corrected VDV, highest axis. On: only to check the '
            'numbers against the app screen.')
        self.lab_bug.stateChanged.connect(self._mark_lab_dirty)
        lg.addWidget(self.lab_bug, 6, 0, 1, 4)
        close_btn = QPushButton('Close')
        close_btn.setObjectName('secondary')
        close_btn.clicked.connect(self.lab_dialog.hide)
        dv.addWidget(close_btn)
        self._refresh_lab_summary()

        ext = QGroupBox('Assessment')
        eg = QGridLayout(ext)
        eg.addWidget(QLabel('Sensor position'), 8, 0)
        self.position = QComboBox()
        for key, label in ps.POSITIONS.items():
            self.position.addItem(label, key)
        self.position.setToolTip(
            'ISO 2631-1 assesses health only at the seat surface. The '
            'backrest x-axis (Wc, k = 0.8) is measured but excluded from the '
            'health assessment (7.2.3); backrest and feet are assessed for '
            'comfort. Custom lets you set weighting and gain per axis '
            'yourself, for a sensor position the norm does not cover.')
        self.position.currentIndexChanged.connect(self._on_position)
        eg.addWidget(self.position, 8, 1)
        self.custom_open_btn = QPushButton('Custom position weighting...')
        self.custom_open_btn.setEnabled(False)
        self.custom_open_btn.clicked.connect(self._open_custom_dialog)
        eg.addWidget(self.custom_open_btn, 9, 0, 1, 2)
        self.custom_summary = QLabel('')
        self.custom_summary.setObjectName('muted')
        self.custom_summary.setWordWrap(True)
        eg.addWidget(self.custom_summary, 10, 0, 1, 2)

        self.custom_dialog = QDialog(self)
        self.custom_dialog.setWindowTitle('Custom position weighting')
        self.custom_dialog.setMinimumWidth(360)
        cdv = QVBoxLayout(self.custom_dialog)
        cg = QGridLayout()
        cdv.addLayout(cg)
        cg.setContentsMargins(6, 4, 6, 4)
        self.custom_w, self.custom_k = [], []
        for c, ax in enumerate('xyz'):
            cg.addWidget(QLabel(ax), 0, c, alignment=Qt.AlignCenter)
            box = QComboBox()
            box.addItem('None', None)
            for w in core.WEIGHTINGS:
                box.addItem(w, w)
            box.currentIndexChanged.connect(self._on_custom_changed)
            cg.addWidget(box, 1, c)
            kk = QDoubleSpinBox()
            kk.setRange(0.0, 5.0)
            kk.setSingleStep(0.1)
            kk.setValue(1.0)
            kk.valueChanged.connect(self._on_custom_changed)
            cg.addWidget(kk, 2, c)
            self.custom_w.append(box)
            self.custom_k.append(kk)
        custom_close = QPushButton('Close')
        custom_close.setObjectName('secondary')
        custom_close.clicked.connect(self.custom_dialog.hide)
        cdv.addWidget(custom_close)
        self._refresh_custom_summary()
        eg.addWidget(QLabel('Criterion'), 4, 0)
        self.criterion = QComboBox()
        self.criterion.addItems([
            'NHO-09 (Fundacentro)',
            'Directive 2002/44/EC — highest axis',
            'Directive 2002/44/EC — vector sum (lab method)',
        ])
        self.criterion.setMinimumWidth(240)
        from PySide6.QtWidgets import QSizePolicy as _QSP
        self.criterion.setSizePolicy(_QSP.Policy.Expanding, _QSP.Policy.Fixed)
        self.criterion.currentIndexChanged.connect(self._recompute)
        eg.addWidget(self.criterion, 4, 1)
        eg.addWidget(QLabel('Exposure time basis'), 5, 0)
        self.basis = QComboBox()
        self.basis.addItems(['Period covered (recommended)',
                             'Recorded time only (as in CONEM2026)'])
        self.basis.setMinimumWidth(240)
        self.basis.setSizePolicy(_QSP.Policy.Expanding, _QSP.Policy.Fixed)
        self.basis.setToolTip(
            'Continuous records: both options give the same time. Old .m '
            'files (2 s every 5 s): vibration does not stop in the 3 s gaps, '
            'so the covered period is the exposure; using only the recorded '
            'time lowers A(8) by sqrt(0.4), about 37%.')
        self.basis.currentIndexChanged.connect(self._recompute)
        eg.addWidget(self.basis, 5, 1)
        self.iso_vdv = QCheckBox('VDV against ISO 2631-1 zone (8.5 / 17)')
        self.iso_vdv.setToolTip('The lab method uses the ISO 2631-1 health '
                                'guidance caution zone for VDV, stricter than '
                                'the Directive (9.1 / 21).')
        self.iso_vdv.stateChanged.connect(self._recompute)
        eg.addWidget(self.iso_vdv, 6, 0, 1, 2)
        self.show_variability = QCheckBox(
            'Show variability across files (off by default)')
        self.show_variability.setToolTip(
            'When the day combines more than one recording file, computes '
            'A(8) and VDV separately for each file (using the same exposure '
            'time basis as the combined value) and shows the range across '
            'them below the headline numbers -- a single combined value can '
            'hide a day where one session was much worse than the others. '
            'Off by default to keep the default view uncluttered.')
        self.show_variability.stateChanged.connect(self._recompute)
        eg.addWidget(self.show_variability, 7, 0, 1, 2)
        grav = QLabel('Gravity is always removed from every block before any '
                      'metric. A VDV computed with the static 1 g retained '
                      'is around 130 m/s^1.75 for an 8 h day and is not '
                      'vibration.')
        grav.setWordWrap(True)
        grav.setObjectName('muted')
        eg.addWidget(grav, 11, 0, 1, 2)

        # ── Sujeito / Medição ──────────────────────────────────────
        _sep = QLabel('Sujeito / Medição')
        _sep.setStyleSheet(
            'font-size:11px;font-weight:700;color:#2d4a7a;margin-top:6px;')
        eg.addWidget(_sep, 12, 0, 1, 2)

        eg.addWidget(QLabel('ID / nome'), 13, 0)
        self.meta_subject_id = QLineEdit()
        self.meta_subject_id.setPlaceholderText('— omitir se anônimo')
        eg.addWidget(self.meta_subject_id, 13, 1)

        eg.addWidget(QLabel('Idade'), 14, 0)
        self.meta_subject_age = QLineEdit()
        self.meta_subject_age.setPlaceholderText('anos')
        eg.addWidget(self.meta_subject_age, 14, 1)

        eg.addWidget(QLabel('Operador'), 15, 0)
        self.meta_operator = QLineEdit()
        eg.addWidget(self.meta_operator, 15, 1)

        eg.addWidget(QLabel('Data da medição'), 16, 0)
        self.meta_date = QLineEdit()
        self.meta_date.setPlaceholderText('ex.: 2026-10-11')
        eg.addWidget(self.meta_date, 16, 1)

        eg.addWidget(QLabel('Sensor / modelo'), 17, 0)
        self.meta_sensor = QLineEdit()
        eg.addWidget(self.meta_sensor, 17, 1)

        eg.addWidget(QLabel('Nº série'), 18, 0)
        self.meta_serial = QLineEdit()
        eg.addWidget(self.meta_serial, 18, 1)

        eg.addWidget(QLabel('Notas'), 19, 0)
        self.meta_notes = QLineEdit()
        eg.addWidget(self.meta_notes, 19, 1)

        self.span_label = QLabel('Measured period: —')
        eg.addWidget(self.span_label, 0, 0, 1, 2)
        self.extrapolate = QCheckBox('Extrapolate to a longer daily exposure')
        self.extrapolate.toggled.connect(self._on_extrapolate)
        eg.addWidget(self.extrapolate, 1, 0, 1, 2)
        eg.addWidget(QLabel('Assumed daily exposure'), 2, 0)
        self.hours = QDoubleSpinBox()
        self.hours.setRange(0.5, 24.0)
        self.hours.setValue(16.0)
        self.hours.setSuffix(' h')
        self.hours.setEnabled(False)
        self.hours.valueChanged.connect(self._on_extrapolate_hours)
        eg.addWidget(self.hours, 2, 1)
        warn = QLabel(
            'Extrapolation assumes the unmeasured hours had the same average '
            'exposure as the measured ones. Report it as an assumption.')
        warn.setWordWrap(True)
        warn.setObjectName('muted')
        eg.addWidget(warn, 3, 0, 1, 2)
        top.addWidget(CollapsibleSection('', ext, expanded=True), 1)
        lay.addLayout(top)

        self.verdict = QLabel('No results yet')
        self.verdict.setAlignment(Qt.AlignCenter)
        self.verdict.setObjectName('verdict')
        self.verdict.setTextFormat(Qt.RichText)
        lay.addWidget(self.verdict)

        self.strip = QLabel('')
        self.strip.setTextFormat(Qt.RichText)
        self.strip.setWordWrap(True)
        lay.addWidget(self.strip)
        self._summary_key = None

        self.ref_note = QLabel('')
        self.ref_note.setTextFormat(Qt.RichText)
        self.ref_note.setWordWrap(True)
        lay.addWidget(self.ref_note)

        body = QHBoxLayout()
        left = QVBoxLayout()
        a8box = QGroupBox('A(8) per axis')
        ag = QGridLayout(a8box)
        ag.setContentsMargins(6, 4, 6, 4)
        ag.setHorizontalSpacing(3)
        ag.setVerticalSpacing(3)
        heads = ['', 'x', 'y', 'z', 'daily\nhighest', 'daily\nvector sum']
        for c, h in enumerate(heads):
            lab = QLabel(h)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet('font-weight:600;font-size:11px;')
            ag.addWidget(lab, 0, c)
        self.a8_cells = {}
        for r, name in enumerate(('this software', 'lab method'), start=1):
            lab = QLabel(name)
            lab.setStyleSheet('font-size:11px;')
            ag.addWidget(lab, r, 0)
            for c in range(1, 6):
                cell = QLabel('—')
                cell.setAlignment(Qt.AlignCenter)
                cell.setMinimumWidth(62)
                cell.setStyleSheet('background:#eef1f4;border-radius:3px;'
                                   'padding:4px;font-weight:700;')
                ag.addWidget(cell, r, c)
                self.a8_cells[(r, c)] = cell
        self.a8_note = QLabel('')
        self.a8_note.setObjectName('muted')
        self.a8_note.setWordWrap(True)
        self.a8_note.setStyleSheet('font-size:11px;')
        ag.addWidget(self.a8_note, 3, 0, 1, 6)
        left.addWidget(a8box)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Quantity', 'Value', 'Note'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumWidth(620)
        self.table.setWordWrap(True)
        left.addWidget(self.table, 1)
        holder = QWidget()
        holder.setLayout(left)
        holder.setMaximumWidth(620)
        body.addWidget(holder, 1)

        right = QVBoxLayout()
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel('Plot'))
        self.kind = QComboBox()
        self.kind.addItems([
            'Timeline: RMS per block and chair occupancy',
            'Spectrum per block (Welch)',
            'Weighted spectrum (NHO-09 weightings)',
            'Dashboard: every criterion against its bands',
            'Side by side: this software vs lab method, per axis',
            'Contribution of each hour to the daily A(8)',
            'Time series: raw acceleration vs time',
            'Running RMS (1 s) vs time — MTVV trace',
            'Annex A: compression force spectrum (block-averaged, seat only)',
            'Annex A: compression force waveform (per block, seat only)',
            'Spectrogram across blocks (time of day; 3D waterfall option)',
            'Shock Response Spectrum (one block)',
            'Interval values: a_w, MTVV and VDV per interval (time of day)',
        ])
        self.kind.setMinimumWidth(260)
        self.kind.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.kind.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(self.kind, 1)
        self.window = QComboBox()
        self.window.addItems(list(core.WINDOW_FUNCTIONS))
        self.window.setCurrentText('Hann')
        self.window.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(QLabel('Window'))
        ctl.addWidget(self.window)
        self.conem = QCheckBox('CONEM2024 curve')
        self.conem.setToolTip('Frequency-response correction of the sensor, '
                              'derived for one unit.')
        self.conem.stateChanged.connect(self.redraw)
        ctl.addWidget(self.conem)
        right.addLayout(ctl)

        annex_row = QHBoxLayout()
        annex_row.addWidget(QLabel('Annex A model:'))
        self.annex_posture = QComboBox()
        for key, label in core.POSTURE_NAMES.items():
            self.annex_posture.addItem(label, key)
        self.annex_posture.setCurrentIndex(
            list(core.POSTURE_NAMES).index(3))
        self.annex_posture.currentIndexChanged.connect(self.redraw)
        annex_row.addWidget(self.annex_posture)
        annex_row.addWidget(QLabel('Mass'))
        self.annex_mass = QDoubleSpinBox()
        self.annex_mass.setRange(40.0, 150.0)
        self.annex_mass.setValue(80.0)
        self.annex_mass.setSuffix(' kg')
        self.annex_mass.valueChanged.connect(self.redraw)
        annex_row.addWidget(self.annex_mass)
        annex_row.addWidget(QLabel('Height'))
        self.annex_height = QDoubleSpinBox()
        self.annex_height.setRange(1.40, 2.10)
        self.annex_height.setSingleStep(0.01)
        self.annex_height.setValue(1.75)
        self.annex_height.setSuffix(' m')
        self.annex_height.valueChanged.connect(self.redraw)
        annex_row.addWidget(self.annex_height)
        annex_row.addWidget(QLabel(
            '— used only by the two Annex A plot kinds above; graphical, '
            'no S^A/R^A dose is computed here'))
        annex_row.addStretch()
        right.addLayout(annex_row)

        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel('Spectrogram / SRS axis:'))
        self.freq_axis = QComboBox()
        for i, n in enumerate(core.AXIS_NAMES):
            self.freq_axis.addItem(n, i)
        self.freq_axis.setCurrentIndex(2)
        self.freq_axis.currentIndexChanged.connect(self.redraw)
        freq_row.addWidget(self.freq_axis)
        freq_row.addWidget(QLabel('Block #'))
        self.freq_block = QSpinBox()
        self.freq_block.setRange(0, 0)
        self.freq_block.valueChanged.connect(self.redraw)
        freq_row.addWidget(self.freq_block)
        self.freq_waterfall = QCheckBox('3D waterfall')
        self.freq_waterfall.setToolTip(
            'Show a 3D spectrogram of the field day (frequency × time × amplitude).')
        self.freq_waterfall.stateChanged.connect(self.redraw)
        freq_row.addWidget(self.freq_waterfall)
        freq_row.addStretch()
        right.addLayout(freq_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel('Interval:'))
        self.interval = QComboBox()
        for sec in nho09.INTERVAL_CHOICES_S:
            self.interval.addItem(f'{sec} s' if sec < 60 else
                                  f'{sec // 60} min', sec)
        self.interval.setCurrentIndex(
            list(nho09.INTERVAL_CHOICES_S).index(60))
        self.interval.currentIndexChanged.connect(self.redraw)
        interval_row.addWidget(self.interval)
        self.interval_export = QPushButton('Export intervals (CSV)')
        self.interval_export.setObjectName('secondary')
        self.interval_export.clicked.connect(self.export_intervals)
        interval_row.addWidget(self.interval_export)
        interval_row.addWidget(QLabel(
            '— values restart each interval; the day totals are unchanged'))
        interval_row.addStretch()
        right.addLayout(interval_row)

        self.plot = plot_factory()
        right.addWidget(self.plot, 1)
        plot_bar = QHBoxLayout()
        plot_bar.addStretch()
        self.campaign_btn = QPushButton('Add to campaign…')
        self.campaign_btn.setObjectName('secondary')
        self.campaign_btn.setToolTip(
            'Add the current assessment to the Campaign tracker tab.')
        self.campaign_btn.setEnabled(False)
        self.campaign_btn.clicked.connect(self._add_to_campaign)
        plot_bar.addWidget(self.campaign_btn)
        self.export_pdf_btn = QPushButton('Export PDF report…')
        self.export_pdf_btn.setObjectName('secondary')
        self.export_pdf_btn.setToolTip(
            'Save a multi-page PDF: summary table + current plot.')
        self.export_pdf_btn.setEnabled(False)
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        plot_bar.addWidget(self.export_pdf_btn)
        self.export_day_btn = QPushButton('Export results + plot…')
        self.export_day_btn.setObjectName('secondary')
        self.export_day_btn.setToolTip(
            'Save the current plot as PNG and the results table as CSV.')
        self.export_day_btn.setEnabled(False)
        self.export_day_btn.clicked.connect(self.export_directive)
        plot_bar.addWidget(self.export_day_btn)
        right.addLayout(plot_bar)
        body.addLayout(right, 2)
        lower = QWidget()
        lower_lay = QVBoxLayout(lower)
        lower_lay.setContentsMargins(0, 0, 0, 0)
        lower_lay.addLayout(body, 1)
        outer.addWidget(settings_splitter(
            upper, lower, 'settings and summary', upper_share=0.34), 1)

    # ------------------------------------------------------------------
    # Carregar arquivos e recalcular os criterios sao as duas operacoes
    # lentas do painel (leitura de muitos .m, e ponderacao em frequencia
    # sobre um dia inteiro de blocos). As duas rodam em background: _busy
    # mostra a barra de progresso e trava os controles que disparariam outro
    # calculo por cima; _refresh esquenta os caches (ver _prime, no topo do
    # arquivo) fora da UI thread e so entao chama a renderizacao de sempre,
    # que passa a achar tudo pronto e fica rapida.
    def _busy(self, on, message=''):
        self._busy_flag = on
        self.progress.setVisible(on)
        self.busy_label.setText(message if on else '')
        for w in (self.load_btn, self.clear_btn, self.lab_open_btn,
                 self.position, self.basis, self.extrapolate, self.calibrate,
                 self.lab_axis_mode, self.lab_rms, self.lab_bug, self.lab_apply,
                 *self.lab_w, *self.lab_k, *self.custom_w, *self.custom_k):
            w.setEnabled(not on)
        # Estes tres tem regra propria de habilitacao (posicao/extrapolacao),
        # nao "sempre habilitado quando ocioso" como os demais controles.
        if on:
            self.criterion.setEnabled(False)
            self.iso_vdv.setEnabled(False)
            self.hours.setEnabled(False)
            self.custom_open_btn.setEnabled(False)
        else:
            seat = self.position.currentData() == 'seat'
            self.criterion.setEnabled(seat)
            self.iso_vdv.setEnabled(seat)
            self.hours.setEnabled(self.extrapolate.isChecked())
            self.custom_open_btn.setEnabled(self.position.currentData() == 'custom')
        if message:
            self.status(message)

    def _finish_thread(self, thread_attr):
        """Encerra a QThread de verdade (quit + wait) antes de seguir.

        worker.run() terminar e emitir finished/failed NAO para o loop de
        eventos da QThread por si so -- sem quit()+wait() a thread continua
        viva indefinidamente, e o proximo carregamento/recalculo (que
        substitui self._load_thread etc. por um objeto novo) pode derrubar o
        processo ao deixar a thread anterior presa sem dono.
        """
        thread = getattr(self, thread_attr, None)
        if thread is not None:
            thread.quit()
            thread.wait()

    def _refresh(self, then=None):
        """Roda _prime fora da UI thread, depois chama `then` (por padrao,
        so redesenha). Todos os gatilhos do painel (carregar um dia, trocar
        posicao/criterio/base de tempo, mudar a configuracao do metodo do
        laboratorio, ligar a extrapolacao) passam por aqui em vez de chamar
        _show()/compute() direto.

        Reentrancia: se uma rodada anterior ainda estiver rodando, este
        gatilho e ignorado em vez de substituir a QThread em andamento --
        sobrescrever self._refresh_thread enquanto o objeto antigo ainda
        executa _prime() na thread derruba o processo ao ser coletado pelo
        Python com o C++ ainda vivo. Os controles ficam desabilitados
        durante o calculo (ver _busy), entao isso so importa para gatilhos
        que nao passam por um clique do usuario.
        """
        if self.day is None or self._busy_flag:
            return
        day = self.day
        position = self.position.currentData()
        hours = self._exposure_hours()
        cfg = self.lab_config()
        criterion_idx = self.criterion.currentIndex()
        extra_hours = self.hours.value() if self.extrapolate.isChecked() else None
        custom = self.custom_config() if position == 'custom' else None

        def job(report):
            report(None, 'Computing the normative criteria...')
            _prime(day, position, hours, cfg, criterion_idx, extra_hours, custom)
            return True

        # `then` fica guardado em self, e os sinais sao ligados a metodos
        # vinculados (nunca a uma lambda): um sinal entregue entre threads so
        # sabe migrar a chamada de volta para a UI thread quando o Qt
        # reconhece a afinidade de thread do receptor, o que so acontece de
        # forma confiavel com um metodo de um QObject de verdade. Uma lambda
        # "solta" pode acabar rodando a chamada na propria worker thread, que
        # e exatamente o tipo de violacao que derruba a interface.
        self._refresh_then = then
        self._busy(True, 'Computing...')
        run_in_thread(self, job, self._on_refresh_done, self._on_refresh_fail,
                      on_progress=self._on_progress,
                      thread_attr='_refresh_thread', worker_attr='_refresh_worker')

    def _on_progress(self, frac, msg):
        self.status(msg)

    def _on_refresh_done(self, _result):
        self._finish_thread('_refresh_thread')
        self._busy(False)
        (self._refresh_then or self.redraw)()

    def _on_refresh_fail(self, tb):
        self._finish_thread('_refresh_thread')
        self._busy(False)
        self.status('Computation failed.')
        box = QMessageBox(QMessageBox.Critical, 'Computation error',
                          tb.strip().splitlines()[-1], parent=self)
        box.setDetailedText(tb)
        box.exec()

    def load(self):
        if self._busy_flag:
            return
        picked = ask_day_files(self)
        if picked is None:
            return
        paths, options = picked
        calibrate = self.calibrate.isChecked()
        position = self.position.currentData()

        def job(report):
            return fd.load_day_auto(paths, calibrate=calibrate,
                                    progress=report, position=position,
                                    **options)

        self._busy(True, f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_load_done, self._on_load_fail,
                      on_progress=self._on_progress,
                      thread_attr='_load_thread', worker_attr='_load_worker')

    def _on_load_done(self, day):
        self._finish_thread('_load_thread')
        self._busy(False)
        self.day = day
        d = self.day
        if d.continuous:
            self.info.setText(
                f'{len(d.records)} continuous record(s), '
                f'{d.recorded_hours:.2f} h at {d.rate:.4g} Hz.')
        else:
            self.info.setText(
                f'{len(d.records)} old-format .m file(s), {len(d.blocks):,} '
                f'blocks. Recorded {d.recorded_hours:.2f} h of '
                f'{d.span_hours:.2f} h ({d.duty_cycle * 100:.0f}% duty cycle).')
        self.info.setStyleSheet(f'color:{TEXT};')
        self.span_label.setText(
            f'Measured period: <b>{d.span_hours:.2f} h</b> — used by default')
        self.status(f'Loaded {len(d.blocks):,} blocks.')
        self.compute()

    def _on_load_fail(self, tb):
        self._finish_thread('_load_thread')
        self._busy(False)
        self.status('Could not read the files.')
        QMessageBox.critical(self, 'Could not read the files',
                             tb.strip().splitlines()[-1])

    def clear_day(self):
        if self._busy_flag:
            return
        self.day = None
        self.result = None
        self.extra = None
        self._norm_key = None
        self._lab_key = None
        self.export_day_btn.setEnabled(False)
        self.export_pdf_btn.setEnabled(False)
        self.campaign_btn.setEnabled(False)
        self.info.setText('No data loaded')
        self.info.setStyleSheet(f'color:{TEXT};')
        self.span_label.setText('Measured period: —')
        self.verdict.setText('No results yet')
        self.strip.setText('')
        self.ref_note.setText('')
        self.table.setRowCount(0)
        for r in (1, 2):
            for c in range(1, 6):
                self.a8_cells[(r, c)].setText('—')
                self.a8_cells[(r, c)].setStyleSheet(
                    'background:#eef1f4;border-radius:3px;padding:4px;'
                    'font-weight:700;')
        self.a8_note.setText('')
        ax, = self.plot.axes()
        ax.text(0.5, 0.5, 'Load a day to begin', ha='center', va='center',
                color='#6c757d', transform=ax.transAxes)
        self.plot.finish(ax)
        self.status('Cleared.')

    def compute(self):
        self._refresh(self._compute_sync)

    def _compute_sync(self):
        d = self.day
        self.result = nho09.assess(d.blocks, d.dt, d.span_hours,
                                   moving_mask=d.occupied)
        self._compute_extra_sync()
        self._show()
        self.export_day_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(True)
        self.campaign_btn.setEnabled(self.campaign is not None)
        self.redraw()

    def lab_config(self):
        return labmethod.LabConfig(
            weights=tuple(b.currentData() for b in self.lab_w),
            k=tuple(k.value() for k in self.lab_k),
            axis_mode=self.lab_axis_mode.currentData(),
            rms_time=self.lab_rms.currentData(),
            reproduce_vdv_defects=self.lab_bug.isChecked())

    def custom_config(self):
        return (tuple(b.currentData() for b in self.custom_w),
                np.array([k.value() for k in self.custom_k]))

    def _open_custom_dialog(self):
        self.custom_dialog.show()
        self.custom_dialog.raise_()
        self.custom_dialog.activateWindow()

    def _refresh_custom_summary(self):
        w, k = self.custom_config()
        wl = ' / '.join('None' if x is None else x for x in w)
        self.custom_summary.setText(
            f'{wl}, k {k[0]:g} / {k[1]:g} / {k[2]:g}')

    def _on_custom_changed(self):
        self._refresh_custom_summary()
        self._recompute()

    def _on_lab(self):
        self._refresh(self._show_and_redraw)

    def _mark_lab_dirty(self):
        # So muda a aparencia do botao Apply; nenhum recalculo roda aqui.
        # Refiltrar um dia inteiro para cada eixo/ganho trocado, um por um,
        # e caro -- o mesmo motivo que fez o app do laboratorio so recalcular
        # ao clicar "Perform Results", nao a cada dropdown.
        self.lab_apply.setStyleSheet(self._LAB_APPLY_DIRTY_STYLE)
        self.lab_apply.setText('▶  Apply weighting / gain  —  changed, click to recompute')
        self.lab_summary.setText(self._lab_summary_text() + ' — pending, not applied yet')
        self.lab_summary.setStyleSheet(f'color:{WARN_COLOR}; font-weight:600;')

    def _apply_lab(self):
        self.lab_apply.setStyleSheet(self._LAB_APPLY_CLEAN_STYLE)
        self.lab_apply.setText('Apply weighting / gain')
        self._refresh_lab_summary()
        self._on_lab()

    def _open_lab_dialog(self):
        self.lab_dialog.show()
        self.lab_dialog.raise_()
        self.lab_dialog.activateWindow()

    def _lab_summary_text(self):
        cfg = self.lab_config()
        wl = ' / '.join('None' if w is None else w for w in cfg.weights)
        return (f'Lab method: {wl}, k {cfg.k[0]:g} / {cfg.k[1]:g} / '
               f'{cfg.k[2]:g}, {self._axis_mode_text(cfg)}')

    def _refresh_lab_summary(self):
        self.lab_summary.setText(self._lab_summary_text())
        self.lab_summary.setStyleSheet('')
        self.lab_summary.setObjectName('muted')
        self.lab_summary.style().unpolish(self.lab_summary)
        self.lab_summary.style().polish(self.lab_summary)

    def _show_and_redraw(self):
        self._show()
        if self.day is not None:
            self.export_day_btn.setEnabled(True)
            self.export_pdf_btn.setEnabled(True)
            self.campaign_btn.setEnabled(self.campaign is not None)
        self.redraw()

    def _update_summary(self):
        cfg = self.lab_config()
        position = self.position.currentData()
        custom = self.custom_config() if position == 'custom' else None
        nkey = (id(self.day), position, round(self._exposure_hours(), 4),
                repr(custom))
        lkey = (id(self.day), repr(cfg))
        if nkey != getattr(self, '_norm_key', None):
            self.status('Computing the normative criteria...')
            self._norm, self.dres = summary.normative_criteria(
                self.day, position, self._exposure_hours(), custom)
            self.hours_rows = summary.per_file(self.day, self._exposure_hours())
            self._norm_key = nkey
        if lkey != getattr(self, '_lab_key', None):
            self.status('Reproducing the lab method...')
            self._labc, self.lab = summary.lab_criteria(self.day, cfg)
            self._lab_key = lkey
        self.crit = self._norm + self._labc
        self._fill_a8_grid()
        self.status('')
        chips = []
        for c in self.crit:
            if np.isnan(c.limit):
                col, bg = TEXT, '#eef1f4'
            elif c.value > c.limit:
                col, bg = '#ffffff', BAD_COLOR
            elif c.value >= c.action:
                col, bg = '#1a1d21', '#f5c451'
            else:
                col, bg = '#ffffff', OK_COLOR
            if not c.normative:
                # nao normativo: fundo cinza, e a cor do nivel so no numero
                num = (BAD_COLOR if not np.isnan(c.limit) and c.value > c.limit
                       else '#9a5a00' if not np.isnan(c.limit)
                       and c.value >= c.action else '#1b5e2a')
                chips.append(
                    f'<span style="background:#dfe3e8;color:#4a5157;">'
                    f'&nbsp;<i>{c.name}</i> {c.metric} <b style="color:{num}">'
                    f'{c.value:.3g}</b> <small>(non-normative)</small>&nbsp;'
                    f'</span>')
                continue
            chips.append(
                f'<span style="background:{bg};color:{col};">&nbsp;<b>{c.name}'
                f'</b> {c.metric} {c.value:.3g}&nbsp;</span>')
        self.strip.setText(
            '<div style="line-height:190%">' + ' &nbsp; '.join(chips) +
            '</div><div style="color:#4a5157;font-size:11px">Green below the '
            'action level, amber between action and limit, red above the limit. '
            'Grey: not normative for this sensor position, or non-normative '
            'method — only the number carries the colour.</div>')

    def _on_position(self):
        seat = self.position.currentData() == 'seat'
        for w in (self.criterion, self.iso_vdv):
            w.setEnabled(seat)
        self.custom_open_btn.setEnabled(self.position.currentData() == 'custom')
        if self.day is None:
            return
        fd.reorient_day(self.day, self.position.currentData())
        self._refresh(self._show_and_redraw)

    def _recompute(self):
        self._refresh(self._show)

    def _exposure_hours(self):
        d = self.day
        if self.basis.currentIndex() == 1:
            return d.recorded_hours
        return d.span_hours

    def _on_extrapolate(self, on):
        self.hours.setEnabled(on)
        self._refresh(self._compute_extra_and_show)

    def _on_extrapolate_hours(self, _value):
        self._refresh(self._compute_extra_and_show)

    def _compute_extra_and_show(self):
        self._compute_extra_sync()
        self._show()

    def _compute_extra_sync(self):
        if self.extrapolate.isChecked():
            self.extra = nho09.assess(self.day.blocks, self.day.dt,
                                      self.hours.value())
        else:
            self.extra = None

    # ------------------------------------------------------------------
    def _colour(self, value, action, limit, uncertain=None):
        if value > limit:
            return BAD_COLOR
        if uncertain is not None and value >= uncertain:
            return UNCERTAIN_COLOR
        return WARN_COLOR if value > action else OK_COLOR

    def _show(self):
        if self.day is None:
            return
        self._update_summary()
        if self.position.currentData() != 'seat':
            self._show_comfort()
            return
        if self.criterion.currentIndex() > 0:
            self._show_directive()
            return
        r = nho09.assess(self.day.blocks, self.day.dt, self._exposure_hours(),
                         moving_mask=self.day.occupied)
        self.result = r
        ca = self._colour(r.aren_rounded, nho09.ACTION_AREN,
                          nho09.LIMIT_AREN, nho09.UNCERTAIN_AREN)
        cv = self._colour(r.vdvr, nho09.ACTION_VDVR, nho09.LIMIT_VDVR,
                          nho09.UNCERTAIN_VDVR)
        worst = max((r.aren_rounded / nho09.ACTION_AREN,
                     r.vdvr / nho09.ACTION_VDVR))
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700">'
            f'<span style="color:{ca}">a<sub>ren</sub> = '
            f'{r.aren_rounded:.1f} m/s&sup2;</span>'
            f'<span style="font-size:13px;font-weight:400;color:#4a5157">'
            f' ({r.aren:.3f} before rounding)</span> &nbsp;&nbsp; '
            f'<span style="color:{cv}">VDVR = {r.vdvr:.2f} m/s<sup>1,75</sup>'
            f'</span></div>'
            f'<div style="color:{TEXT};font-size:14px">'
            f'a<sub>ren</sub>: {r.verdict_aren} &nbsp;|&nbsp; '
            f'VDVR: {r.verdict_vdvr}'
            + (' &mdash; <b>one criterion is enough</b>'
               if worst > 1 else '') + '</div>'
            f'<div style="color:#4a5157;font-size:12px">NHO-09, seat, measured '
            f'period {r.t_exp_h:.2f} h. Table 1 of the norm: acceptable up to '
            f'0.5 m/s&sup2; / 9.1 m/s<sup>1,75</sup>; above the action level '
            f'up to 0.9 / 16.4; uncertainty region 0.9&ndash;1.1 / '
            f'16.4&ndash;21; above the exposure limit beyond 1.1 / 21. '
            f'a<sub>ren</sub> is compared after rounding to 0.1 m/s&sup2; '
            f'(item 6.4.1).</div>')

        rows = [
            ('am x, y, z [m/s2]', ',  '.join(f'{v:.3f}' for v in r.am),
             'Wd, Wd, Wk'),
            ('amr [m/s2]', f'{r.amr:.3f}',
             'vector sum, factors 1.4 / 1.4 / 1.0'),
            ('aren [m/s2]', f'{r.aren:.3f} \u2192 {r.aren_rounded:.1f}',
             f'{r.verdict_aren} (rounded to 0.1, item 6.4.1)'),
            ('VDV x, y, z measured', ',  '.join(f'{v:.2f}' for v in r.vdv),
             f'over {r.t_amos_h:.2f} h recorded'),
            ('VDVexp x, y, z', ',  '.join(f'{v:.2f}' for v in r.vdv_exp),
             'with factor and (Texp/Tamos)^1/4'),
            ('VDVR [m/s1.75]', f'{r.vdvr:.2f}', r.verdict_vdvr),
            ('— check —', '', ''),
            ('amr, mean of 2 s frames', f'{r.amr_arithmetic:.3f}',
             f'{nho09.arithmetic_bias(r) * 100:+.0f}% — reference only; a '
             f'continuous record is a single NHO-09 sample'),
            ('chair probably occupied', f'{self.day.occupied_fraction * 100:.0f}%',
             'heuristic; the result does not depend on it'),
        ]
        if self.extra is not None:
            e = self.extra
            rows += [
                ('— extrapolation —', '', f'assumption: {e.t_exp_h:g} h/day'),
                ('aren, square-root relation',
                 f'{e.aren:.3f} \u2192 {e.aren_rounded:.1f}', e.verdict_aren),
                ('aren, fourth-root relation', f'{e.aren_quarter:.3f}',
                 nho09.classify(nho09.round_aren(e.aren_quarter),
                                nho09.ACTION_AREN, nho09.LIMIT_AREN,
                                nho09.UNCERTAIN_AREN)),
                ('VDVR', f'{e.vdvr:.2f}', e.verdict_vdvr),
            ]
        self._fill_table(rows + self._a8_rows())
        self._update_ref_note(a8=self.dres.a8)

    def _draw_dashboard(self):
        crit = [c for c in self.crit
                if c.kind == 'exposure' and not np.isnan(c.limit)]
        ax, = self.plot.axes()
        y = np.arange(len(crit))[::-1]
        for yi, c in zip(y, crit):
            top = max(c.limit * 1.35, c.value * 1.1)
            ax.barh(yi, c.action / top, color='#d4ecd9', height=0.62)
            ax.barh(yi, (c.limit - c.action) / top, left=c.action / top,
                    color='#fbe8b8', height=0.62)
            ax.barh(yi, 1 - c.limit / top, left=c.limit / top,
                    color='#f6cfcc', height=0.62)
            colour = (BAD_COLOR if c.value > c.limit else
                      WARN_COLOR if c.value >= c.action else OK_COLOR)
            ax.plot([c.value / top] * 2, [yi - 0.36, yi + 0.36], lw=3.5,
                    color=colour if c.normative else '#6c757d',
                    ls='-' if c.normative else '--')
            ax.text(min(c.value / top, 0.97) + 0.01, yi, f'{c.value:.3g}',
                    va='center', fontsize=9, fontweight='bold')
        ax.set_yticks(y, [f'{c.name}\n{c.metric}' +
                          ('' if c.normative else '  (non-normative)')
                          for c in crit], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title('Each criterion on its own scale: green below action, '
                     'amber to the limit, red above')
        self.plot.finish(ax)

    def _draw_side_by_side(self):
        # Alinha pelo eixo FISICO do sensor. Se o modo de auditoria forcar a
        # troca x-z, o "z" do laboratorio pode ser outra coluna do arquivo
        # que o "z" deste software, e comparar pelo rotulo poria lado a lado
        # grandezas diferentes.
        ours = self.dres.a8_axis
        amap = self.day.records[0].axis_map          # eixo deste software -> coluna
        lab = np.array([self.lab.a8_axis[self.lab.display.index(amap[i])]
                        for i in range(3)])
        lab_names = ['xyz'[self.lab.display.index(amap[i])] for i in range(3)]
        mixed = len({r.axis_map for r in self.day.records}) > 1
        ax, = self.plot.axes()
        x = np.arange(3)
        wd = 0.36
        b1 = ax.bar(x - wd / 2, ours, wd, color=ACCENT,
                    label=f'this software ({self._display_weighting_text()})')
        cfg = self.lab_config()
        lab_lbl = ' / '.join('None' if w is None else w for w in cfg.weights)
        b2 = ax.bar(x + wd / 2, lab, wd, color='#adb5bd', hatch='//',
                    edgecolor='#6c757d',
                    label=f'lab method ({lab_lbl}, k {cfg.k[0]:g} / '
                          f'{cfg.k[1]:g} / {cfg.k[2]:g})')
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f'{b.get_height():.3f}', ha='center', va='bottom',
                        fontsize=8)
        for i in range(3):
            if ours[i] > 0:
                d = (lab[i] / ours[i] - 1) * 100
                ax.text(i, max(ours[i], lab[i]) * 1.12, f'{d:+.0f}%',
                        ha='center', fontsize=9, color=BAD_COLOR
                        if abs(d) > 10 else TEXT, fontweight='bold')
        ax.axhline(0.5, ls='--', lw=1, color=WARN_COLOR, label='EAV 0.5')
        ax.set_xticks(x, [f'{a} here  =  {b} in the lab app\n(sensor column '
                          f'{amap[i] + 1})' for i, (a, b) in
                          enumerate(zip('xyz', lab_names))], fontsize=8)
        if mixed:
            ax.text(0.5, 0.97, 'Sensor orientation changes between files: '
                    'alignment uses the first file', transform=ax.transAxes,
                    ha='center', va='top', fontsize=8, color=BAD_COLOR)
        ax.set_ylabel('A(8) per axis [m/s$^2$]')
        ax.set_ylim(0, max(ours.max(), lab.max(), 0.5) * 1.3)
        ax.set_title(f'Daily: this software {self.dres.a8:.3f} (highest axis) '
                     f'vs lab method {self.lab.a8_daily:.3f} (vector sum)')
        ax.legend(fontsize=8, loc='upper left')
        self.plot.finish(ax)

    def _draw_waveform(self):
        """Amostra a amostra, nao RMS por bloco -- util pra inspecionar
        visualmente um teste de bancada (choque, pico saturando, ruido)."""
        d = self.day
        ax, = self.plot.axes()
        n_blocks, n_samples, _ = d.blocks.shape
        dt = d.dt

        amp_flat = d.blocks.reshape(-1, 3)
        if self.force_continuous.isChecked():
            t_flat = np.arange(n_blocks * n_samples) * dt
        else:
            within = np.arange(n_samples) * dt
            t = (d.seconds[:, None] - d.seconds[0]) + within[None, :]
            t_flat = t.reshape(-1)
            if n_blocks > 1:
                gap = np.diff(d.seconds) > d.block_spacing * 1.5
                if gap.any():
                    breaks = (np.flatnonzero(gap) + 1) * n_samples
                    t_flat = np.insert(t_flat, breaks, np.nan)
                    amp_flat = np.insert(amp_flat, breaks, np.nan, axis=0)

        total = n_blocks * n_samples
        MAX_POINTS = 400_000
        stride = max(1, total // MAX_POINTS)
        if stride > 1:
            t_flat = t_flat[::stride]
            amp_flat = amp_flat[::stride]

        for c, name in enumerate(core.AXIS_NAMES):
            ax.plot(t_flat, amp_flat[:, c], lw=0.5, color=AXIS_COLORS[c],
                   label=name)
        ax.set_xlabel('Time since first block [s]')
        ax.set_ylabel('Amplitude [m/s$^2$]')
        title = f'{total:,} raw samples'
        if stride > 1:
            title += f' (decimated 1:{stride} for display)'
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3)
        self.plot.finish(ax)

    def _draw_running_rms(self):
        """RMS corrente (janela de 1s, ISO 2631-1 6.4.2) x tempo -- em
        resolucao bem mais fina que o Timeline (RMS por bloco de 2s).
        Mostra exatamente onde um transiente empurrou o MTVV pra cima,
        com as faixas de acao/limite da NHO-09 desenhadas como referencia."""
        d = self.day
        ax, = self.plot.axes()
        tau = 1.0
        trace = nho09.running_rms_trace(d.blocks, d.dt, tau)
        n_blk, n_pts = trace.shape

        # valor na amostra j = janela que termina em j
        within = (np.arange(n_pts) + 1) * d.dt
        t = (d.seconds[:, None] - d.seconds[0]) + within[None, :]
        t_flat = t.reshape(-1)
        trace_flat = trace.reshape(-1)

        if n_blk > 1:
            gap = np.diff(d.seconds) > d.block_spacing * 1.5
            if gap.any():
                breaks = (np.flatnonzero(gap) + 1) * n_pts
                t_flat = np.insert(t_flat, breaks, np.nan)
                trace_flat = np.insert(trace_flat, breaks, np.nan)

        total = n_blk * n_pts
        MAX_POINTS = 400_000
        stride = max(1, total // MAX_POINTS)
        if stride > 1:
            t_flat = t_flat[::stride]
            trace_flat = trace_flat[::stride]

        ax.plot(t_flat, trace_flat, lw=0.7, color=AXIS_COLORS[2])
        ax.axhline(nho09.ACTION_AREN, color=WARN_COLOR, ls='--', lw=1,
                  label=f'NHO-09 action ({nho09.ACTION_AREN:g})')
        ax.axhline(nho09.LIMIT_AREN, color=BAD_COLOR, ls='--', lw=1,
                  label=f'NHO-09 limit ({nho09.LIMIT_AREN:g})')
        peak = float(np.nanmax(trace_flat)) if trace_flat.size else 0.0
        ax.set_xlabel('Time since first block [s]')
        ax.set_ylabel(f'Running RMS ({tau:g} s) [m/s$^2$]')
        ax.set_title(f'MTVV = {peak:.3f} m/s$^2$ (running RMS peak)')
        ax.legend(fontsize=8)
        self.plot.finish(ax)

    def _intervals(self):
        sec = self.interval.currentData()
        key = (id(self.day.blocks), sec)
        if getattr(self, '_interval_key', None) != key:
            self._interval_data = nho09.interval_metrics(
                self.day.blocks, self.day.dt, sec, self.day.seconds)
            self._interval_key = key
        return self._interval_data

    def _draw_intervals(self):
        """Registro por intervalo (lista Dewesoft, item 2.2): onde, ao longo
        do dia, esta a exposicao. Em cima a soma vetorial do a_w e o MTVV de
        cada intervalo, embaixo o VDV acumulado so dentro do intervalo."""
        d = self.day
        m = self._intervals()
        ax1, ax2 = self.plot.axes(2)
        if not len(m['start']):
            ax1.text(0.5, 0.5, 'No data.', ha='center', transform=ax1.transAxes)
            self.plot.finish(ax1, ax2)
            return
        h = m['start'] / 3600.0
        mid = h + m['duration'] / 7200.0
        width = m['duration'] / 3600.0
        # degraus preenchidos num objeto so (milhares de barras travam o
        # desenho num dia inteiro a 1 s), cortados com NaN nas lacunas
        xs = np.column_stack([h, h + width]).ravel()
        ys = np.repeat(m['amr'], 2)
        gap = np.flatnonzero(h[1:] > (h + width)[:-1] + 1e-6) + 1
        xs = np.insert(xs, 2 * gap, np.nan)
        ys = np.insert(ys, 2 * gap, np.nan)
        ax1.fill_between(xs, 0, ys, color=AXIS_COLORS[2], alpha=0.55, lw=0,
                         label='a_w vector sum (1.4/1.4/1.0)')
        ax1.plot(mid, m['mtvv'], '.', ms=3, color=BAD_COLOR,
                 label='MTVV within the interval')
        ax1.axhline(nho09.ACTION_AREN, color=WARN_COLOR, ls='--', lw=1,
                    label=f'NHO-09 action ({nho09.ACTION_AREN:g})')
        ax1.set_ylabel('[m/s$^2$]')
        ax1.legend(fontsize=8, ncol=3)
        for c, name in enumerate(core.AXIS_NAMES):
            ax2.plot(mid, m['vdv'][:, c], '.', ms=3, color=AXIS_COLORS[c],
                     label=f'VDV {name}')
        ax2.set_ylabel('VDV in the interval [m/s$^{1.75}$]')
        ax2.set_xlabel('Time of day [h]' if d.records[0].absolute_time
                       else 'Time since first block [h]')
        ax2.legend(fontsize=8, ncol=3)
        sec = self.interval.currentData()
        worst = int(np.nanargmax(m['amr']))
        t = int(m['start'][worst]) % 86400
        ax1.set_title(f'{len(m["start"]):,} intervals of {sec:g} s; highest '
                      f'a_w {m["amr"][worst]:.2f} m/s$^2$ at '
                      f'{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}')
        self.plot.finish(ax1, ax2)

    def export_intervals(self):
        if self.day is None:
            return
        import csv
        sec = self.interval.currentData()
        stem = Path(str(self.day.records[0].path)).stem
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export interval values', f'{stem}_intervals_{sec}s.csv',
            'CSV (*.csv)')
        if not path:
            return
        rows = nho09.interval_csv_rows(self._intervals(),
                                       self.day.records[0].absolute_time)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(rows)
        self.status(f'{len(rows) - 1} intervals written to {path}')

    def _add_to_campaign(self):
        """Adiciona a avaliacao corrente ao painel de campanha."""
        if self.day is None or self.campaign is None:
            return
        from datetime import datetime as _dt
        pos = self.position.currentData()
        files_str = '; '.join(
            Path(str(r.path)).name for r in self.day.records)
        date_str = _dt.now().strftime('%Y-%m-%d')
        entry = {
            'Date': date_str,
            'Position': pos,
            'File(s)': files_str,
        }
        seat = pos == 'seat'
        if seat and self.criterion.currentIndex() == 0 and self.result:
            r = self.result
            entry['Criterion'] = 'NHO-09 aren'
            entry['Main value [m/s²]'] = f'{r.aren_rounded:.1f}'
            entry['VDVR [m/s^1.75]'] = f'{r.vdvr:.2f}'
            if hasattr(self, 'dres'):
                entry['A(8) highest [m/s²]'] = (
                    f'{self.dres.a8_axis.max():.3f}')
        elif seat and self.criterion.currentIndex() > 0:
            from . import directive as _dir
            r = _dir.assess(self.day.blocks, self.day.dt,
                            self._exposure_hours(),
                            'highest' if self.criterion.currentIndex() == 1
                            else 'vector')
            entry['Criterion'] = 'Directive A(8)'
            entry['Main value [m/s²]'] = f'{r.a8:.3f}'
            entry['VDVR [m/s^1.75]'] = f'{r.vdv:.2f}'
            entry['A(8) highest [m/s²]'] = f'{r.a8:.3f}'
        else:
            entry['Criterion'] = f'ISO 2631-1 comfort ({pos})'
        self.campaign.add_entry(entry)
        self.status('Entry added to Campaign tracker.')

    def export_pdf(self):
        """Exporta relatorio PDF: tabela de resultados + graficos selecionados."""
        if self.day is None:
            return
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime as _dt

        # ── Seleção de gráficos ─────────────────────────────────────
        from PySide6.QtWidgets import QApplication, QProgressDialog
        from PySide6.QtCore import Qt as _Qt
        plot_names = [self.kind.itemText(i) for i in range(self.kind.count())]
        cur_idx    = self.kind.currentIndex()

        dlg = QDialog(self)
        dlg.setWindowTitle('Exportar PDF — Diretiva 2002/44/CE')
        dlg.setMinimumWidth(520)
        vl = QVBoxLayout(dlg)
        vl.setSpacing(10)
        vl.setContentsMargins(16, 16, 16, 12)

        title_lbl = QLabel('Selecionar gráficos')
        title_lbl.setStyleSheet('font-weight: bold; font-size: 13px;')
        vl.addWidget(title_lbl)
        vl.addWidget(QLabel('Gráficos a incluir no PDF (além da tabela de resultados):'))
        checks = []
        for i, name in enumerate(plot_names):
            cb = QCheckBox(name)
            cb.setChecked(i == cur_idx)
            vl.addWidget(cb)
            checks.append(cb)

        from PySide6.QtWidgets import QFrame
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #c4d0e4;')
        vl.addWidget(sep)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText('Exportar PDF')
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_indices = [i for i, cb in enumerate(checks) if cb.isChecked()]
        if not selected_indices:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, 'Export PDF', 'Nenhum gráfico selecionado.')
            return

        stem = Path(str(self.day.records[0].path)).stem
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF report', f'{stem}_report.pdf', 'PDF (*.pdf)')
        if not path:
            return

        # ── Coletar tabela de resultados ────────────────────────────
        table_rows = [['Quantity', 'Value', 'Note']]
        for row in range(self.table.rowCount()):
            cells = [self.table.item(row, col).text()
                     if self.table.item(row, col) else ''
                     for col in range(self.table.columnCount())]
            if any(c.strip() for c in cells):
                table_rows.append(cells)
        data_rows = table_rows[1:]  # skip CSV header row for PDF rendering

        n_pages = 1 + len(selected_indices)

        # ── Página 1: relatório formatado (A4 retrato) ──────────────
        import matplotlib as _mpl
        _rc_save = {k: _mpl.rcParams[k] for k in ['font.family']}
        _mpl.rcParams['font.family'] = 'serif'

        prog = None
        try:
            BLACK   = 'black'
            SECT_BG = '#f2f2f2'
            SECT_FG = 'black'
            ALT_BG  = '#f2f2f2'
            BORDER  = '#333333'
            TXT     = 'black'
            MUTED   = '#555555'
            WHITE   = 'white'
            GREEN_FG = '#1a5c35'; GREEN_BG = '#e6f4ed'
            WARN_FG  = '#7b5800'; WARN_BG  = '#fff8e1'
            RED_FG   = '#8b1a1a'; RED_BG   = '#fde8e8'
            FOOTER_Y = 0.024
            ROW_H    = 0.013
            SEC_H    = 0.018

            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(WHITE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

            def R(x, y, w, h, fc, ec=None, lw=0.5):
                if y < FOOTER_Y:
                    return
                ax.add_patch(mp.Rectangle(
                    (x, y), w, h, facecolor=fc,
                    edgecolor=(ec or fc), linewidth=lw,
                    transform=ax.transAxes, clip_on=True))

            def T(x, y, s_, **kw):
                kw.setdefault('transform', ax.transAxes)
                kw.setdefault('va', 'center')
                kw.setdefault('fontsize', 7.5)
                kw.setdefault('color', TXT)
                kw['clip_on'] = True
                ax.text(x, y, s_, **kw)

            def hline(y, lw=0.8, color=BORDER):
                ax.axhline(y, lw=lw, color=color, clip_on=False, zorder=1)

            # Cabeçalho
            R(0, 0.960, 1, 0.040, BLACK)
            T(0.015, 0.980, 'Avaliador de Vibração Humana — Diretiva 2002/44/CE',
              color='white', fontsize=9.5, fontweight='bold', va='center', clip_on=False)
            T(0.985, 0.980, 'GRAVI-SH  ·  UFMG',
              color='#cccccc', fontsize=8, ha='right', va='center', clip_on=False)
            hline(0.960, lw=1.2, color=BLACK)

            # Arquivo e configuração
            T(0.015, 0.947,
              f'Arquivo:  {Path(str(self.day.records[0].path)).name}',
              color=BLACK, fontsize=8, fontweight='bold', clip_on=False)
            T(0.985, 0.947, _dt.now().strftime('%Y-%m-%d  %H:%M'),
              color=MUTED, fontsize=7.5, ha='right', clip_on=False)
            hline(0.934, lw=0.5)
            T(0.015, 0.929,
              f'Posição: {self.position.currentText()}   ·   '
              f'Exposição: {self._exposure_hours():.2f} h   ·   '
              f'Registros: {len(self.day.records)}',
              color=MUTED, fontsize=7.5, clip_on=False)
            hline(0.916, lw=0.5)

            # Informações do sujeito / medição
            _meta_items = [
                ('ID / nome',       self.meta_subject_id.text().strip()),
                ('Idade',           self.meta_subject_age.text().strip()),
                ('Operador',        self.meta_operator.text().strip()),
                ('Data medição',    self.meta_date.text().strip()),
                ('Sensor / modelo', self.meta_sensor.text().strip()),
                ('Nº série',        self.meta_serial.text().strip()),
                ('Notas',           self.meta_notes.text().strip()),
            ]
            _meta_items = [(k, v) for k, v in _meta_items if v]
            MROW_H  = 0.013
            _n_mrow = max(1, (len(_meta_items) + 1) // 2)
            META_H  = _n_mrow * MROW_H

            if _meta_items:
                T(0.015, 0.912, 'Informações do sujeito / medição',
                  color=BLACK, fontsize=7.5, fontweight='bold', clip_on=False)
                meta_top = 0.908
                for _i, (_k, _v) in enumerate(_meta_items):
                    _col = _i % 2
                    _row = _i // 2
                    _x   = 0.015 + _col * 0.50
                    _y   = meta_top - (_row + 0.5) * MROW_H
                    T(_x,        _y, f'{_k}:', color=MUTED, fontsize=7, clip_on=False)
                    T(_x + 0.15, _y, _v,       color=TXT,   fontsize=7, clip_on=False)
                TABLE_TOP = meta_top - META_H - 0.006
                hline(TABLE_TOP + 0.002, lw=0.5)
            else:
                TABLE_TOP = 0.908
            TABLE_TOP  = TABLE_TOP  # keep name for reference below
            CLIP_FLOOR = FOOTER_Y + 0.002

            n_data = sum(1 for r in data_rows
                         if not (r[0].startswith('\u2014') or
                                 (not r[1].strip() and not r[2].strip())))
            n_sect = len(data_rows) - n_data
            total_h = n_data * ROW_H + n_sect * SEC_H
            available = TABLE_TOP - CLIP_FLOOR
            if total_h > available and n_data > 0:
                ROW_H = max(0.008, (available - n_sect * SEC_H) / max(n_data, 1))

            y = TABLE_TOP
            alt = 0
            for cells in data_rows:
                label = cells[0] if cells else ''
                value = cells[1] if len(cells) > 1 else ''
                note  = cells[2] if len(cells) > 2 else ''

                is_section = (label.startswith('\u2014') or
                              (not value.strip() and not note.strip()))

                if is_section:
                    if y - SEC_H < CLIP_FLOOR:
                        break
                    T(0.015, y - SEC_H / 2,
                      label.strip('— ').capitalize(),
                      color=BLACK, fontsize=7, fontweight='bold')
                    hline(y - SEC_H, lw=0.5)
                    y -= SEC_H
                    alt = 0
                else:
                    if y - ROW_H < CLIP_FLOOR:
                        break
                    row_bg  = ALT_BG if alt % 2 == 0 else WHITE
                    val_clr = TXT
                    n_lower = note.lower()
                    if 'below' in n_lower:
                        row_bg = GREEN_BG; val_clr = GREEN_FG
                    elif 'above' in n_lower or 'exceeded' in n_lower:
                        row_bg = RED_BG;   val_clr = RED_FG
                    elif 'caution' in n_lower or 'action' in n_lower:
                        row_bg = WARN_BG;  val_clr = WARN_FG

                    R(0, y - ROW_H, 1, ROW_H, row_bg, BORDER, 0.2)
                    T(0.015, y - ROW_H / 2, label,  color=TXT,     fontsize=6.5)
                    T(0.54,  y - ROW_H / 2, value,  color=val_clr, fontsize=6.5)
                    T(0.75,  y - ROW_H / 2, note,   color=MUTED,   fontsize=6)
                    y -= ROW_H
                    alt += 1

            if y > CLIP_FLOOR:
                hline(y, lw=0.5)

            # Rodapé
            ax.axhline(FOOTER_Y, lw=0.6, color=BORDER, clip_on=False, zorder=1)
            T(0.015, FOOTER_Y / 2,
              'Avaliador de Vibração Humana v3.0  ·  GRAVI-SH / UFMG',
              color=MUTED, fontsize=6.5, clip_on=False)
            T(0.985, FOOTER_Y / 2, f'Página 1 / {n_pages}',
              color=MUTED, fontsize=6.5, ha='right', clip_on=False)

            # ── Gerar PDF com indicador de progresso ────────────────────
            prog = QProgressDialog('Gerando PDF…', '', 0, 0, self)
            prog.setWindowTitle('Exportar PDF')
            prog.setWindowModality(_Qt.WindowModality.WindowModal)
            prog.setCancelButton(None)
            prog.setMinimumDuration(0)
            prog.show()
            QApplication.processEvents()

            with PdfPages(path) as pdf:
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)

                orig_idx = self.kind.currentIndex()
                orig_size = self.plot.figure.get_size_inches()
                for k in selected_indices:
                    self.kind.setCurrentIndex(k)
                    self.redraw()
                    self.plot.figure.set_size_inches(11.69, 8.27)
                    pdf.savefig(self.plot.figure, bbox_inches='tight', dpi=150)
                    self.plot.figure.set_size_inches(*orig_size)

                self.kind.setCurrentIndex(orig_idx)
                self.redraw()
                self.plot.figure.canvas.draw_idle()
        except Exception as _e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(_e))
            return
        finally:
            if prog is not None:
                prog.close()
            if _rc_save:
                _mpl.rcParams.update(_rc_save)

        self.status(f'PDF saved to {Path(path).name}.')

    def export_directive(self):
        """Exporta o grafico atual (PNG) e a tabela de resultados (CSV)."""
        if self.day is None:
            return
        import csv as csv_mod
        stem = Path(str(self.day.records[0].path)).stem
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export results + plot', f'{stem}_results.csv',
            'CSV (*.csv)')
        if not path:
            return
        path = Path(path)
        # CSV: all non-empty rows from the results table
        table_rows = [['Quantity', 'Value', 'Note']]
        for row in range(self.table.rowCount()):
            cells = []
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                cells.append(item.text() if item else '')
            if any(c.strip() for c in cells):
                table_rows.append(cells)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv_mod.writer(fh).writerows(table_rows)
        # PNG: current plot figure
        png_path = path.with_suffix('.png')
        self.plot.figure.savefig(str(png_path), dpi=150, bbox_inches='tight')
        self.status(f'Saved {path.name} and {png_path.name}.')

    def _draw_hours(self):
        rows = self.hours_rows
        ax, = self.plot.axes()
        # .m antigo: Dados_AAAAMMDD_HH -> 'HHh'; outros nomes ficam inteiros
        names = [r['name'][-2:] + 'h'
                 if len(r['name']) >= 3 and r['name'][-3] == '_'
                 else r['name'] for r in rows]
        share = np.array([r['share'] for r in rows]) * 100
        x = np.arange(len(rows))
        wd = 0.27
        for c, lab_ax in enumerate('xyz'):
            ax.bar(x + (c - 1) * wd, share[:, c], wd, color=AXIS_COLORS[c],
                   label=lab_ax)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel('Share of the daily A(8) energy [%]')
        ax.set_title('Share of each file in the daily A(8)$^2$, per axis')
        top = share.max() if share.size else 1
        for i, r in enumerate(rows):
            cf = r['crest'].max()
            # Em arquivos quase parados o a_w esta no piso de ruido e um pico
            # isolado infla a razao; o fator de crista so tem sentido onde ha
            # exposicao de fato.
            if cf > 9 and r['share'].max() > 0.02:
                ax.text(i, top * 1.02, f'CF {cf:.0f}', ha='center', fontsize=7,
                        color=BAD_COLOR)
        ax.set_ylim(0, top * 1.12)
        ax.legend(fontsize=8)
        self.plot.finish(ax)

    def _draw_annex_a(self, time_domain):
        """Espectro/forma de onda da forca de compressao intervertebral
        (Anexo A), so' grafico -- nenhum S^A/R^A e' calculado aqui. Ver
        core.block_compression_force quanto as limitacoes fisicas: excitacao
        assumida so' no assento (sensor unico), e cada bloco de 2 s cobre
        poucos ciclos nas frequencias mais baixas da banda (nos registros
        continuos os blocos sao so as janelas de analise deste grafico)."""
        ax, = self.plot.axes()
        if self.store is None:
            ax.text(0.5, 0.5, 'Annex A transfer-function data not available.',
                    ha='center', va='center', color='#6c757d',
                    transform=ax.transAxes)
            self.plot.finish(ax)
            return
        if self.position.currentData() != 'seat':
            ax.text(0.5, 0.5, 'The Annex A model applies to the seat only. '
                    'Switch the sensor position to "Seat" to see this plot.',
                    ha='center', va='center', wrap=True,
                    transform=ax.transAxes)
            self.plot.finish(ax)
            return
        d = self.day
        posture = self.annex_posture.currentData()
        k, bmi_per = core.anthropometry(self.annex_mass.value(),
                                        self.annex_height.value())
        try:
            f, F, force_time = core.block_compression_force(
                self.store, d.blocks, d.dt, posture, k, bmi_per)
        except FileNotFoundError as exc:
            ax.text(0.5, 0.5, str(exc), ha='center', va='center', wrap=True,
                    color=BAD_COLOR, transform=ax.transAxes)
            self.plot.finish(ax)
            return

        caption = (f'Graphical only, no S^A/R^A here. Seat excitation only '
                  f'(feet/hands/back zero); '
                  f'{core.POSTURE_NAMES.get(posture, posture)}, model '
                  f'{k}/30.')

        if not time_domain:
            mag = np.sqrt((np.abs(F) ** 2).mean(axis=0))       # (6, nf)
            for dl in range(6):
                ax.plot(f, mag[dl], lw=0.9, color=SERIES_COLORS[dl],
                       label=core.DISC_NAMES[dl])
            ax.set_xlim(0, 32)
            ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Force amplitude [N] (RMS over blocks)')
            ax.set_title(f'Intervertebral compressive force spectrum, seat '
                        f'only -- {d.blocks.shape[0]:,} blocks averaged')
            ax.legend(fontsize=8, ncol=2)
        else:
            n_blk, n_disc, n_pts = force_time.shape
            dt_out = 1.0 / (2 * f[-1])
            within = np.arange(n_pts) * dt_out
            t = (d.seconds[:, None] - d.seconds[0]) + within[None, :]
            t_flat = t.reshape(-1)
            force_flat = force_time.transpose(0, 2, 1).reshape(-1, n_disc)
            if n_blk > 1:
                # Cada bloco e' reconstruido de forma independente (ver
                # docstring de core.block_compression_force) -- toda borda
                # de bloco e' uma quebra real nesta forma de onda, nao so'
                # onde ha lacuna de tempo de verdade.
                breaks = np.arange(1, n_blk) * n_pts
                t_flat = np.insert(t_flat, breaks, np.nan)
                force_flat = np.insert(force_flat, breaks, np.nan, axis=0)
            total = n_blk * n_pts
            MAX_POINTS = 400_000
            stride = max(1, total // MAX_POINTS)
            if stride > 1:
                t_flat = t_flat[::stride]
                force_flat = force_flat[::stride]
            for dl in range(6):
                ax.plot(t_flat, force_flat[:, dl], lw=0.6,
                       color=SERIES_COLORS[dl], label=core.DISC_NAMES[dl])
            ax.set_xlabel('Time since first block [s]')
            ax.set_ylabel('Dynamic compressive force [N]')
            ax.set_title(f'Intervertebral compressive force waveform, seat '
                        f'only -- {n_blk:,} blocks')
            ax.legend(fontsize=8, ncol=3)
        ax.text(0.5, 0.97, caption, transform=ax.transAxes, ha='center',
                va='top', fontsize=8, color='#6c757d')
        self.plot.finish(ax)

    def _fill_a8_grid(self):
        seat = self.position.currentData() == 'seat'
        d = self.dres.a8_axis
        rows = {1: (list(d) + [d.max(), float(np.sqrt((d ** 2).sum()))], True),
                2: (list(self.lab.a8_axis) + [float(self.lab.a8_axis.max()),
                                              self.lab.a8_daily], False)}
        for r, (vals, ours) in rows.items():
            for c, v in enumerate(vals, start=1):
                cell = self.a8_cells[(r, c)]
                cell.setText(f'{v:.3f}')
                normative = seat and ours
                if v > 1.15:
                    bg, fg = BAD_COLOR, '#ffffff'
                elif v >= 0.5:
                    bg, fg = '#f5c451', '#1a1d21'
                else:
                    bg, fg = OK_COLOR, '#ffffff'
                if not normative:
                    fg = {BAD_COLOR: BAD_COLOR, OK_COLOR: '#1b5e2a'}.get(bg, '#9a5a00')
                    bg = '#dfe3e8'
                cell.setStyleSheet(f'background:{bg};color:{fg};border-radius:3px;'
                                   f'padding:4px;font-weight:700;')
        cfg = self.lab_config()
        wl = ' / '.join('None' if w is None else w for w in cfg.weights)
        self.a8_note.setText(
            f'This software: {self._display_weighting_text()}. Lab method: '
            f'{wl}, k {cfg.k[0]:g} / {cfg.k[1]:g} / {cfg.k[2]:g}, '
            f'{self._axis_mode_text(cfg)}'
            '. Coloured against EAV 0.5 / ELV 1.15'
            + ('.' if seat else ' for reference only: at this position the '
               'A(8) is not a health assessment, so every cell is grey.'))

    def _display_weighting_text(self):
        """Ponderacao/ganho que self.dres de fato usa nesta posicao -- ver
        summary.display_weights. Wd/Wd/Wk 1,4/1,4/1,0 so no assento."""
        pos = self.position.currentData()
        custom = self.custom_config() if pos == 'custom' else None
        dw, dk = summary.display_weights(pos, custom)
        if dw is None:
            return 'Wd / Wd / Wk, k 1.4 / 1.4 / 1.0'
        wl = ' / '.join('None' if w is None else w for w in dw)
        return f'{wl}, k {dk[0]:g} / {dk[1]:g} / {dk[2]:g}'

    def _axis_mode_text(self, cfg):
        return {'raw': 'raw column order (app default)',
                'swap': 'x <-> z swap forced (app checkbox on)',
                'auto': 'vertical axis auto-detected per file (not the app)',
                }[cfg.axis_mode]

    def _a8_rows(self):
        """A(8) por eixo e diario, sempre visivel, em qualquer posicao.

        Duas reguas: a deste software, com a ponderacao de saude do assento
        (Wd/Wd/Wk, a definicao da propria diretiva) quando a posicao e o
        assento, ou a ponderacao de positions.HEALTH quando nao e -- ver
        _display_weighting_text/summary.display_weights, para nao mostrar
        Wd/Wd/Wk aplicado a um sensor de encosto ou pe. E o metodo do
        laboratorio com a configuracao escolhida, que reproduz o painel
        'Total A(8) Exposure' do aplicativo. Fora do assento os dois valem
        como informacao, nao como avaliacao de saude.
        """
        seat = self.position.currentData() == 'seat'
        tag = '' if seat else ' — reference only, not a health verdict here'
        d, lab = self.dres, self.lab
        cfg = self.lab_config()
        wl = ' / '.join('None' if w is None else w for w in cfg.weights)
        return [
            ('— A(8) per axis —', '', self._display_weighting_text() + tag),
            ('A(8) x, y, z', ',  '.join(f'{v:.3f}' for v in d.a8_axis),
             'this software'),
            ('A(8) daily, highest axis', f'{d.a8_axis.max():.3f}',
             'Directive text'),
            ('A(8) daily, vector sum', f'{np.sqrt((d.a8_axis ** 2).sum()):.3f}',
             'lab convention'),
            ('— lab method (app panel) —', '',
             f'{wl}, k {cfg.k[0]:g} / {cfg.k[1]:g} / {cfg.k[2]:g}, '
             f'{self._axis_mode_text(cfg)}'),
            ('A(8) x, y, z (app axes)', ',  '.join(f'{v:.3f}' for v in lab.a8_axis),
             'as shown by the app'),
            ('A(8) daily (app)', f'{lab.a8_daily:.3f}',
             'vector sum, as the app'),
        ]

    def _fill_table(self, rows):
        self.table.setRowCount(len(rows))
        for i, (a, b, c) in enumerate(rows):
            for j, v in enumerate((a, b, c)):
                it = QTableWidgetItem(v)
                it.setForeground(QColor(TEXT))
                if j == 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    f = it.font(); f.setBold(True); it.setFont(f)
                if a.startswith('—'):
                    f = it.font(); f.setItalic(True); it.setFont(f)
                self.table.setItem(i, j, it)

    def _update_ref_note(self, a8=None, av=None):
        """ISO 2631-1 reference context panel with paediatric caveat.

        a8: daily A(8) in m/s² (seat / directive) or None
        av: vibration total value in m/s² (comfort positions) or None
        Either may be supplied; both may be supplied.
        """
        lines = ['<div style="font-size:11px;border-left:3px solid #adb5bd;'
                 'padding-left:7px;margin:3px 0;color:#4a5157">',
                 '<b>Reference context (ISO 2631-1, informative only — '
                 'not normative for this assessment)</b>']
        if a8 is not None:
            hlabel = refs.health_guidance_label(a8)
            if a8 < refs.ISO2631_1_HEALTH_LOW:
                hcol = '#1b7a33'
            elif a8 <= refs.ISO2631_1_HEALTH_HIGH:
                hcol = '#9a5a00'
            else:
                hcol = '#c0392b'
            lines.append(
                f'A(8) = {a8:.3f} m/s²: '
                f'<b style="color:{hcol}">{hlabel}</b> '
                f'(ISO 2631-1 Annex B: '
                f'{refs.ISO2631_1_HEALTH_LOW}–'
                f'{refs.ISO2631_1_HEALTH_HIGH} m/s²).')
        if av is not None:
            clabel = refs.comfort_label(av)
            lines.append(
                f'a<sub>v</sub> = {av:.3f} m/s²: '
                f'<b>{clabel}</b> (ISO 2631-1 Annex C comfort '
                f'scale: &lt;0.315 not uncomfortable → '
                f'&gt;2.0 extremely uncomfortable).')
        lines.append(
            f'<i>{refs.THERAPY_NOTE}</i>')
        lines.append(
            f'<span style="color:#c0392b"><b>Paediatric:</b> '
            f'{refs.PAEDIATRIC_CAVEAT}</span>')
        lines.append('</div>')
        self.ref_note.setText(''.join(f'<br>' + ln if i else ln
                                      for i, ln in enumerate(lines)))

    def _show_comfort(self):
        d = self.day
        pos = self.position.currentData()
        custom = self.custom_config() if pos == 'custom' else None
        c = ps.comfort(d.blocks, d.dt, pos, custom=custom)
        hours = self._exposure_hours()
        h = None
        rows = [
            ('a_w x, y, z [m/s2]', ',  '.join(f'{v:.4f}' for v in c['aw']),
             ' / '.join('None' if w is None else w for w in c['weights'])),
            ('k x, y, z', ' / '.join(f'{v:g}' for v in c['k']),
             'ISO 2631-1, 8.2' if pos != 'custom' else 'user-defined'),
            ('a_v, vibration total value', f"{c['av']:.3f}",
             c['label'] if pos != 'custom' else 'no ISO indication applies'),
        ]
        if pos in ('backrest', 'feet'):
            h = ps.health_informative(d.blocks, d.dt, hours, pos)
            wl = ' / '.join('None' if w is None else w for w in h['weights'])
            kl = ' / '.join(f'{v:g}' for v in h['k'])
            note = ('ISO 2631-1, 7.2.3' if pos == 'backrest'
                    else 'not defined by ISO 2631-1')
            rows += [
                ('— health, informative only —', '', note),
                ('x, y, z: A(8)', ',  '.join(f'{v:.3f}' for v in h['a8']),
                 f'{wl}, k {kl} — not compared with any action level'),
                ('x, y, z: VDV', ',  '.join(f'{v:.2f}' for v in h['vdv']),
                 'not compared with any action level'),
            ]
        if pos == 'backrest':
            extra = (' The backrest x-axis is shown for information: ISO '
                     '2631-1 recommends measuring it but excludes it from '
                     'the health assessment.')
        elif pos == 'feet':
            extra = (' ISO 2631-1 defines no health criterion for the feet; '
                     'the unweighted values above are shown for information '
                     'only.')
        else:
            extra = (' Custom position: weighting and gain per axis are '
                     'user-defined, above. Neither the comfort scale nor '
                     'any action/limit level applies to this combination.')
        rows += self._a8_rows()
        colour = (OK_COLOR if c['av'] < 0.315 else
                  WARN_COLOR if c['av'] < 1.0 else BAD_COLOR)
        health_line = ('health: not assessed at this position' if pos != 'custom'
                       else 'health: not assessed — custom weighting')
        comfort_label_shown = c['label'] if pos != 'custom' else 'not an ISO scale'
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700;color:{colour}">'
            f'a<sub>v</sub> = {c["av"]:.3f} m/s&sup2;</div>'
            f'<div style="color:{TEXT};font-size:14px">comfort: '
            f'<b>{comfort_label_shown}</b> &nbsp;|&nbsp; {health_line}</div>'
            f'<div style="color:#4a5157;font-size:12px">ISO 2631-1 comfort, '
            f'{self.position.currentText().split(" —")[0].lower()}. Annex C '
            f'values are approximate indications, not limits.{extra}</div>')
        self._fill_table(rows)
        a8_ref = float(h['a8'].max()) if pos in ('backrest', 'feet') else None
        self._update_ref_note(a8=a8_ref, av=c['av'])

    def _show_directive(self):
        d = self.day
        mode = 'highest' if self.criterion.currentIndex() == 1 else 'vector'
        hours = self._exposure_hours()
        r = directive.assess(d.blocks, d.dt, hours, mode=mode)
        iso = self.iso_vdv.isChecked()
        lo, hi = ((directive.ISO_VDV_LOW, directive.ISO_VDV_HIGH) if iso
                  else (directive.EAV_VDV, directive.ELV_VDV))
        ca = self._colour(r.a8, directive.EAV_A8, directive.ELV_A8)
        cv = self._colour(r.vdv, lo, hi)
        label = ('highest axis — text of the Directive' if mode == 'highest'
                 else 'vector sum — lab method, diverges from the Directive')
        basis = self.basis.currentText().split(' (')[0].lower()
        variability_html = ''
        if self.show_variability.isChecked() and len(d.records) > 1:
            a8_list, vdv_list = [], []
            for i in range(len(d.records)):
                rb = d.record_blocks(i)
                if len(rb) == 0:
                    continue
                rr = directive.assess(rb, d.dt, hours, mode=mode)
                a8_list.append(rr.a8)
                vdv_list.append(rr.vdv)
            if len(a8_list) > 1:
                variability_html = (
                    f'<div style="color:#4a5157;font-size:12px">Variability '
                    f'across {len(a8_list)} files, same {hours:.2f} h basis: '
                    f'A(8) {min(a8_list):.3f}–{max(a8_list):.3f} m/s&sup2;'
                    f', VDV {min(vdv_list):.2f}–{max(vdv_list):.2f} '
                    f'm/s<sup>1,75</sup> -- a single combined value can hide a '
                    f'session much worse than the others.</div>')
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700">'
            f'<span style="color:{ca}">A(8) = {r.a8:.3f} m/s&sup2;</span>'
            f' &nbsp;&nbsp; <span style="color:{cv}">VDV = {r.vdv:.2f} '
            f'm/s<sup>1,75</sup></span></div>'
            f'<div style="color:{TEXT};font-size:14px">A(8): '
            f'{r.verdict_a8()} &nbsp;|&nbsp; VDV: {r.verdict_vdv(iso)}</div>'
            f'<div style="color:#4a5157;font-size:12px">Directive 2002/44/EC, '
            f'{label}. Exposure time {hours:.2f} h ({basis}). '
            f'A(8) 0.5 / 1.15 m/s&sup2;, VDV {lo:g} / {hi:g}.</div>'
            f'{variability_html}')

        rows = [
            ('a_w x, y, z [m/s2]', ',  '.join(f'{v:.4f}' for v in r.aw),
             'weighted Wd, Wd, Wk; no factor'),
            ('A(8) per axis', ',  '.join(f'{v:.3f}' for v in r.a8_axis),
             'k a_w sqrt(t / 8 h)'),
            ('A(8)', f'{r.a8:.3f}', r.verdict_a8()),
            ('Exposure points per axis',
             ',  '.join(f'{v:.0f}' for v in directive.exposure_points(r.a8_axis)),
             '100 pts = action value (0.5 m/s2)'),
            ('Exposure points (combined)',
             f'{directive.exposure_points(r.a8):.0f}',
             f'{directive.exposure_points(directive.ELV_A8):.0f} pts = limit value (1.15 m/s2)'),
            ('VDV per axis', ',  '.join(f'{v:.2f}' for v in r.vdv_axis),
             'k VDV (t_exp / t_meas)^1/4'),
            ('VDV', f'{r.vdv:.2f}', r.verdict_vdv(iso)),
            ('crest factor x, y, z', ',  '.join(f'{v:.1f}' for v in r.crest),
             'above 9: VDV is the appropriate metric'),
            ('— time —', '', ''),
            ('exposure time used', f'{hours:.2f} h', basis),
            ('time actually recorded', f'{r.t_meas_h:.2f} h',
             'continuous record' if d.continuous else
             f'{d.duty_cycle * 100:.0f}% of the period covered (old .m)'),
        ]
        self._fill_table(rows + self._a8_rows())
        self._update_ref_note(a8=r.a8)

    # ------------------------------------------------------------------
    def _draw_freq_extra(self, k):
        """Espectrograma de longo prazo (hora do dia) e SRS de um bloco."""
        d = self.day
        axis = self.freq_axis.currentData()
        name = core.AXIS_NAMES[axis]
        if k == 10:
            f, S = core.block_spectrogram(
                d.blocks[:, :, axis], d.dt, self.window.currentText())
            t = (d.seconds - d.seconds[0]) / 3600.0
            freqplots.draw_spectrogram(
                self.plot, t, f, S,
                f'Spectrogram across {len(t):,} blocks - {name}',
                tlabel='Time since first block [h]',
                waterfall=self.freq_waterfall.isChecked(), logy=True)
            return
        self.freq_block.blockSignals(True)
        self.freq_block.setRange(0, max(len(d.blocks) - 1, 0))
        self.freq_block.blockSignals(False)
        bi = min(self.freq_block.value(), len(d.blocks) - 1)
        fr, sr = core.shock_response_spectrum(d.blocks[bi, :, axis], d.dt)
        freqplots.draw_srs(
            self.plot, [(f'block {bi} ({name})', fr, sr)],
            f'Shock Response Spectrum (Q=10) - block {bi}, {name}')

    def redraw(self):
        if self.day is None:
            return
        k = self.kind.currentIndex()
        self.plot.reset_if_changed(k)
        d = self.day

        if k == 0:
            ax, = self.plot.axes()
            rms = np.sqrt((d.blocks ** 2).mean(axis=1))
            if self.force_continuous.isChecked():
                t = np.arange(len(d.seconds)) * d.block_spacing / 3600.0
                t_plot = t
                rms_plot = rms
            else:
                t = (d.seconds - d.seconds[0]) / 3600.0
                gap = np.diff(d.seconds) > d.block_spacing * 1.5
                breaks = np.flatnonzero(gap) + 1
                t_plot = np.insert(t, breaks, np.nan)
                rms_plot = np.insert(rms, breaks, np.nan, axis=0)
            for c, name in enumerate(core.AXIS_NAMES):
                ax.plot(t_plot, rms_plot[:, c], lw=0.6, color=AXIS_COLORS[c],
                       label=name)
            top = rms.max() * 1.05
            empty = ~d.occupied
            edges = np.flatnonzero(np.diff(np.r_[0, empty.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[a], t[min(b, len(t) - 1)], color='#d5dbe1',
                           alpha=0.6, lw=0,
                           label='chair probably empty' if a == edges[0] else None)
            ax.set_ylim(0, top)
            ax.set_xlabel('Time since first block [h]')
            ax.set_ylabel('RMS of the 2 s block [m/s$^2$]')
            ax.set_title('Unweighted RMS per block; grey = orientation frozen')
            ax.legend(fontsize=8, ncol=4)
            self.plot.finish(ax)
            return

        if k == 3:
            self._draw_dashboard()
            return
        if k == 4:
            self._draw_side_by_side()
            return
        if k == 5:
            self._draw_hours()
            return
        if k == 6:
            self._draw_waveform()
            return
        if k == 7:
            self._draw_running_rms()
            return
        if k in (8, 9):
            self._draw_annex_a(k == 9)
            return
        if k in (10, 11):
            self._draw_freq_extra(k)
            return
        if k == 12:
            self._draw_intervals()
            return

        ax, = self.plot.axes()
        from scipy.signal import get_window
        n = fd.BLOCK_SAMPLES
        w = get_window(core.WINDOW_FUNCTIONS[self.window.currentText()], n,
                       fftbins=True)
        f = np.arange(n // 2 + 1) * d.rate / n
        blocks = d.blocks
        for c, name in enumerate(core.AXIS_NAMES):
            X = np.fft.rfft((blocks[:, :, c]
                             - blocks[:, :, c].mean(1, keepdims=True)) * w,
                            axis=1)
            amp = np.sqrt((np.abs(X) ** 2).mean(0)) * 2.0 / w.sum()
            if k == 2:
                amp = amp * np.abs(core.weighting_response(nho09.WEIGHTS[c], f)) \
                    * nho09.FACTORS[c]
            if self.conem.isChecked():
                amp = np.maximum(amp + fd.conem_correction(f), 1e-9)
            ax.plot(f, amp, lw=0.9, color=AXIS_COLORS[c],
                    label=f'{name}' + (f' ({nho09.WEIGHTS[c]} x '
                                       f'{nho09.FACTORS[c]:g})' if k == 2 else ''))
        ax.set_xlim(0, 100)
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [m/s$^2$]')
        ax.set_title(f'{len(blocks):,} blocks averaged, '
                     f'{self.window.currentText()} window, '
                     f'resolution {f[1]:.2f} Hz')
        ax.legend(fontsize=8)
        self.plot.finish(ax)
