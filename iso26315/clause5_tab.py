"""Aba da Clausula 5 / Anexo C: exposicoes com perda de contato.

Independente da aba do Anexo A. Le o arquivo, condiciona pela Clausula 5 e
calcula ate a probabilidade de dano pelo modelo de Weibull.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import annexc, annexc_wheelchair as acw, core, freqplots, references
from . import fielddata as fd
from .loaddialog import ask_day_files
from .qtworker import run_in_thread
from .theme import ACCENT, BAD_COLOR, OK_COLOR, TEXT, WARN_COLOR
from . import pdf_report

RESULT_ROWS = [
    ('D_z  acceleration dose [m/s2]', 'Dz', 'Formula (3)'),
    ('D_zd daily dose [m/s2]', 'Dzd', 'Formulae (4), (5)'),
    ('S_d  daily compression dose [MPa]', 'Sd', 'Formula (C.1)'),
    ('R    stress variable', 'R', 'Formula (C.3)'),
    ('Pi   probability of damage', 'probability', 'Formula (C.5)'),
]


class ClauseFiveTab(QWidget):
    def __init__(self, plot_factory, status):
        super().__init__()
        self.status = status
        self.path = None
        self.result = None
        self.segments = None
        self.day = None
        self._busy_flag = False
        self._spine_zero = core._SPINE_TF_ZERO_DEFAULT
        self._spine_poles = list(core._SPINE_TF_POLES_DEFAULT)
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        intro = QLabel(
            'Clause 5 with Annex C — the path for exposures <b>with</b> loss of '
            'contact with the seat. Independent of the Annex A chain: the '
            'signal is conditioned with a 2-pole high pass at 0.01 Hz and a '
            '4-pole low pass at 80 Hz, and the ISO 2631-1 band-limiting and '
            'weighting filters are not applied. Only the vertical acceleration '
            'at the seat is used.')
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        intro.setObjectName('warning')
        lay.addWidget(intro)

        top = QHBoxLayout()

        src = QGroupBox('Vertical acceleration at the seat')
        sv = QVBoxLayout(src)
        btn = QPushButton('Open record...')
        btn.clicked.connect(self.open_file)
        sv.addWidget(btn)
        self.open_btn = btn
        field_btn = QPushButton('Load field day (DateTime CSV, .dat, .txt, .m)...')
        field_btn.setToolTip(
            'Reads the files of one seat, one day. Continuous records (the '
            'current loggers) go through the Clause 5 chain as one signal per '
            'file. Old .m files (2 s blocks every 5 s) are still accepted, '
            'each block conditioned on its own.')
        field_btn.clicked.connect(self.load_field)
        sv.addWidget(field_btn)
        self.field_btn = field_btn
        self.field_progress = QProgressBar()
        self.field_progress.setRange(0, 0)
        self.field_progress.setVisible(False)
        sv.addWidget(self.field_progress)
        self.file_label = QLabel('No record loaded')
        self.file_label.setObjectName('muted')
        self.file_label.setWordWrap(True)
        sv.addWidget(self.file_label)
        self.field_note = QLabel('')
        self.field_note.setWordWrap(True)
        self.field_note.setObjectName('warning')
        self.field_note.setVisible(False)
        sv.addWidget(self.field_note)
        self.contact = QCheckBox('Detect and exclude loss of contact')
        self.contact.setChecked(True)
        sv.addWidget(self.contact)
        note = QLabel(
            'Detection is a heuristic, not part of the standard: stretches '
            'where the vertical acceleration stays below -7 m/s2 for at least '
            '0.15 s are treated as free fall. ISO 2631-5 asks for a contact '
            'switch or video. Check the marked stretches before trusting them.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        sv.addWidget(note)

        self.wc_enable = QCheckBox(
            'v2.9 EXPERIMENTAL: wheelchair posture prototype (not validated)')
        self.wc_enable.setToolTip(
            'Replaces Formula (1) -- the norm\'s single, posture-less '
            'seat-to-spine transfer function -- with a wheelchair-specific '
            '6-DOF articulated model (Kawai & Matsuoka, SAE 2000-01-0645; '
            'Matsuoka, Kawai & Sato, ASME J. Mech. Design 125:779-785, '
            '2003), corrected against an independent replication (Hatano '
            '& Takahashi 2016) and cross-checked against the 2-8 Hz '
            '(dominant 4-6 Hz) resonance band reported by three independent '
            'sources -- NOT an exact reproduction of those papers\' own '
            'validation numbers (see annexc_wheelchair.py docstring for the '
            'full derivation and its limits). NOT validated against any '
            'ISO 2631-5 reference case, and NOT for reporting -- research '
            'exploration only.')
        self.wc_enable.stateChanged.connect(self._wc_toggled)
        sv.addWidget(self.wc_enable)
        wc_row = QHBoxLayout()
        wc_row.addWidget(QLabel('Output segment'))
        self.wc_output = QComboBox()
        for key, label in acw.OUTPUT_SEGMENTS.items():
            self.wc_output.addItem(label, key)
        self.wc_output.setCurrentIndex(1)   # tronco inferior, o mais proximo da lombar
        self.wc_output.currentIndexChanged.connect(self._wc_changed)
        wc_row.addWidget(self.wc_output)
        sv.addLayout(wc_row)
        wc_row2 = QHBoxLayout()
        wc_row2.addWidget(QLabel('Posture softening'))
        self.wc_softening = QDoubleSpinBox()
        self.wc_softening.setRange(0.1, 1.0)
        self.wc_softening.setSingleStep(0.05)
        self.wc_softening.setValue(1.0)
        self.wc_softening.setToolTip(
            '1.0 = generic seated posture (no correction). Lower values '
            'loosen the seat-pelvis coupling, the qualitative mechanism '
            'Matsuoka et al. (2003) reported for a hip-slid-forward '
            'posture -- their paper gives no calibrated factor, so this is '
            'a free knob for exploration, not a validated correction.')
        self.wc_softening.valueChanged.connect(self._wc_changed)
        wc_row2.addWidget(self.wc_softening)
        sv.addLayout(wc_row2)
        self.wc_freqs = QLabel('')
        self.wc_freqs.setObjectName('muted')
        self.wc_freqs.setWordWrap(True)
        sv.addWidget(self.wc_freqs)
        self._wc_toggled(False)

        self.spine_tf_enable = QCheckBox(
            'v3.0 EXPERIMENTAL: use lab-fitted spine TF (Formula 1 structure)')
        self.spine_tf_enable.setToolTip(
            'Replaces the ISO 2631-5 Formula (1) seat-to-spine transfer '
            'function with a custom one of the same mathematical structure '
            '(1 zero pair + 6 pole pairs, DC gain = 1). Fit parameters '
            'come from your own measurement — use the Explorer tab (plot '
            'kind "Seat-to-spine TF fit") to obtain them and enter them '
            'with the Parameters button. NOT normative and NOT validated '
            'against any reference case — for research exploration only.')
        self.spine_tf_enable.stateChanged.connect(self._spine_tf_toggled)
        sv.addWidget(self.spine_tf_enable)
        spine_btn_row = QHBoxLayout()
        self.spine_tf_btn = QPushButton('TF parameters...')
        self.spine_tf_btn.setEnabled(False)
        self.spine_tf_btn.clicked.connect(self._open_spine_tf_dialog)
        spine_btn_row.addWidget(self.spine_tf_btn)
        spine_btn_row.addStretch()
        sv.addLayout(spine_btn_row)
        self.spine_tf_label = QLabel('')
        self.spine_tf_label.setObjectName('muted')
        self.spine_tf_label.setWordWrap(True)
        sv.addWidget(self.spine_tf_label)

        sv.addStretch()
        cal_row = QHBoxLayout()
        cal_btn = QPushButton('Calibrate...')
        cal_btn.setObjectName('secondary')
        cal_btn.setToolTip(
            'Reference calibrator (FFT peak detection) and static 6-position '
            'calibration. Applies the correction in-place to the loaded signal '
            '— reload the file to undo.')
        cal_btn.clicked.connect(self._open_calibration_dialog)
        cal_row.addWidget(cal_btn)
        clear_btn = QPushButton('Clear')
        clear_btn.setObjectName('secondary')
        clear_btn.setToolTip('Discard the loaded signal/day and all results.')
        clear_btn.clicked.connect(self.clear)
        cal_row.addWidget(clear_btn)
        cal_row.addStretch()
        sv.addLayout(cal_row)
        top.addWidget(src, 1)

        person = QGroupBox('Individual')
        f = QFormLayout(person)
        self.sex = QComboBox()
        self.sex.addItems(['male', 'female'])
        self.sex.currentIndexChanged.connect(self._sex_changed)
        f.addRow('Sex', self.sex)
        self.mass = QDoubleSpinBox()
        self.mass.setRange(10.0, 250.0)
        self.mass.setValue(82.0)
        self.mass.setSuffix(' kg')
        f.addRow('Body mass', self.mass)
        self.added = QDoubleSpinBox()
        self.added.setRange(0.0, 100.0)
        self.added.setSuffix(' kg')
        f.addRow('Gear on torso/head', self.added)
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(10.0, 90.0)
        self.fraction.setSuffix(' %')
        self.fraction.setDecimals(1)
        f.addRow('Mass above pelvis', self.fraction)
        self.area = QDoubleSpinBox()
        self.area.setRange(2.0, 40.0)
        self.area.setDecimals(2)
        self.area.setSuffix(' cm2')
        f.addRow('Endplate area', self.area)
        self.mz_label = QLabel('')
        self.mz_label.setObjectName('muted')
        self.mz_label.setWordWrap(True)
        f.addRow('', self.mz_label)
        top.addWidget(person, 1)

        expo = QGroupBox('Exposure')
        g = QFormLayout(expo)
        self.daily = QDoubleSpinBox()
        self.daily.setRange(0.1, 24.0)
        self.daily.setValue(8.0)
        self.daily.setSuffix(' h/day')
        g.addRow('Daily exposure', self.daily)
        self.measured = QDoubleSpinBox()
        self.measured.setRange(1.0, 100000.0)
        self.measured.setValue(400.0)
        self.measured.setSuffix(' s')
        g.addRow('Measurement duration', self.measured)
        self.days = QSpinBox()
        self.days.setRange(1, 365)
        self.days.setValue(240)
        self.days.setSuffix(' days/year')
        self.days.setToolTip(
            "Default of 240 days/year, 45 years is Annex A/E's orientating "
            "schedule (A.3.1), not a Clause 5/Annex C requirement -- Annex "
            "C's own worked example (C.2) uses 120 days/year over 20 "
            "years instead. Edit freely to match your actual exposure.")
        g.addRow('Days per year', self.days)
        self.start_age = QSpinBox()
        self.start_age.setRange(1, 90)
        self.start_age.setValue(20)
        g.addRow('Age at first exposure', self.start_age)
        self.years = QSpinBox()
        self.years.setRange(1, 70)
        self.years.setValue(45)
        self.years.setToolTip(self.days.toolTip())
        g.addRow('Years of exposure', self.years)
        self.run_btn = QPushButton('Run assessment')
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run)
        g.addRow('', self.run_btn)
        top.addWidget(expo, 1)
        lay.addLayout(top)

        self.verdict = QLabel('No results yet')
        self.verdict.setAlignment(Qt.AlignCenter)
        self.verdict.setObjectName('verdict')
        self.verdict.setTextFormat(Qt.RichText)
        lay.addWidget(self.verdict)

        body = QHBoxLayout()
        self.table = QTableWidget(len(RESULT_ROWS) + 4, 3)
        self.table.setHorizontalHeaderLabels(['Quantity', 'Value', 'Source'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumWidth(470)
        body.addWidget(self.table, 1)

        right = QVBoxLayout()
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel('Plot'))
        self.plot_kind = QComboBox()
        self.plot_kind.addItems([
            'Conditioned signal and spine response',
            'Counted peaks',
            'Seat-to-spine transfer function',
            'Frequency spectrum (informative — Clause 5 does not weight by frequency)',
            'Risk reference bands (R and Pi vs. Table C.2)',
            'Anthropometric sensitivity (sex, mass fraction, endplate area)',
            'Spectrogram of raw a_z (informative)',
            'Shock Response Spectrum of raw a_z (informative)',
            'Time series of raw a_z (informative)',
            'Weibull injury probability curve (Annex C, Table C.1)',
        ])
        self.plot_kind.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(self.plot_kind, 1)
        self.waterfall = QCheckBox('3D waterfall')
        self.waterfall.setToolTip(
            'Show a 3D spectrogram of the loaded record (frequency × time × amplitude).')
        self.waterfall.stateChanged.connect(self.redraw)
        ctl.addWidget(self.waterfall)
        right.addLayout(ctl)
        self.plot = plot_factory()
        right.addWidget(self.plot, 1)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)

        bar = QHBoxLayout()
        bar.addStretch()
        self.export_btn = QPushButton('Export PDF report...')
        self.export_btn.setObjectName('secondary')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_pdf)
        bar.addWidget(self.export_btn)
        lay.addLayout(bar)

        self._sex_changed()

    # ------------------------------------------------------------------
    def _sex_changed(self):
        cfg = annexc.SEX_DEFAULTS[self.sex.currentText()]
        self.fraction.setValue(cfg['mass_fraction'] * 100)
        self.area.setValue(cfg['endplate_area'])
        self.mz_label.setText(
            f'Standard reference: {cfg["mass_fraction"] * 100:.0f}% of body '
            f'mass above the pelvis and {cfg["endplate_area"]:g} cm2 endplate '
            f'area for a {cfg["ref_mass"]:g} kg young '
            f'{self.sex.currentText()}. Both fields are editable — the standard '
            f'asks for values representative of the individual or population '
            f'being assessed.')

    def _wc_toggled(self, on):
        for w in (self.wc_output, self.wc_softening):
            w.setEnabled(bool(on))
        self._wc_changed()

    def _wc_changed(self):
        if not self.wc_enable.isChecked():
            self.wc_freqs.setText('')
            return
        fn = acw.natural_frequencies(self.wc_softening.value())
        self.wc_freqs.setText(
            'Undamped natural frequencies of this chain: ' +
            ', '.join(f'{f:.2f} Hz' for f in fn))

    def _spine_tf_toggled(self, on):
        self.spine_tf_btn.setEnabled(bool(on))
        if on:
            z_hz = self._spine_zero[0] / (2 * np.pi)
            self.spine_tf_label.setText(
                f'Zero: {z_hz:.1f} Hz, ζ={self._spine_zero[1]:.2f}  '
                f'— {len(self._spine_poles)} pole pair(s). '
                f'Results are NOT normative.')
        else:
            self.spine_tf_label.setText('')

    def _spine_tf_response_fn(self):
        if not self.spine_tf_enable.isChecked():
            return None
        return core.make_spine_response_fn(self._spine_zero, self._spine_poles)

    def _open_spine_tf_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Spine TF parameters (Formula 1 structure)')
        dlg.setMinimumWidth(420)
        vlay = QVBoxLayout(dlg)

        note = QLabel(
            '<b>Formula (1) structure:</b> 1 zero pair + 6 pole pairs, DC gain=1. '
            'Enter ω (rad/s) and ζ for each pair. Obtain fitted values from the '
            'Explorer "Seat-to-spine TF fit" plot. '
            '<span style="color:#c0392b">Results using these values are NOT '
            'normative — research exploration only.</span>')
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        vlay.addWidget(note)

        form = QFormLayout()
        spins = []

        def add_pair(label, default_om, default_ze):
            om_sp = QDoubleSpinBox()
            om_sp.setRange(0.1, 10000.0)
            om_sp.setDecimals(3)
            om_sp.setSingleStep(1.0)
            om_sp.setSuffix(' rad/s')
            om_sp.setValue(default_om)
            ze_sp = QDoubleSpinBox()
            ze_sp.setRange(0.001, 5.0)
            ze_sp.setDecimals(4)
            ze_sp.setSingleStep(0.01)
            ze_sp.setValue(default_ze)
            row = QHBoxLayout()
            row.addWidget(QLabel('ω ='))
            row.addWidget(om_sp)
            row.addWidget(QLabel('  ζ ='))
            row.addWidget(ze_sp)
            form.addRow(label, row)
            spins.append((om_sp, ze_sp))

        add_pair('Zero pair', *self._spine_zero)
        for i, (om, ze) in enumerate(self._spine_poles):
            add_pair(f'Pole pair {i + 1}', om, ze)

        vlay.addLayout(form)

        def reset_defaults():
            spins[0][0].setValue(core._SPINE_TF_ZERO_DEFAULT[0])
            spins[0][1].setValue(core._SPINE_TF_ZERO_DEFAULT[1])
            for i, (om, ze) in enumerate(core._SPINE_TF_POLES_DEFAULT):
                spins[i + 1][0].setValue(om)
                spins[i + 1][1].setValue(ze)

        reset_btn = QPushButton('Reset to ISO defaults')
        reset_btn.clicked.connect(reset_defaults)
        vlay.addWidget(reset_btn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vlay.addWidget(bb)

        if dlg.exec() != QDialog.Accepted:
            return

        self._spine_zero = (spins[0][0].value(), spins[0][1].value())
        self._spine_poles = [(spins[i + 1][0].value(), spins[i + 1][1].value())
                             for i in range(6)]
        self._spine_tf_toggled(True)

    def open_file(self):
        if self._busy_flag:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open vertical acceleration record(s)', '',
            'Time series (*.txt *.asc *.dat *.csv);;All files (*)')
        if not paths:
            return

        def job(report):
            az_parts = []
            dt_ref = None
            for i, p in enumerate(paths):
                report(i / len(paths), f'Reading {Path(p).name}...')
                data = np.loadtxt(p, usecols=(0, 1, 2, 3))
                t = data[:, 0]
                dt = (t[-1] - t[0]) / (len(t) - 1)
                if dt_ref is None:
                    dt_ref = dt
                elif abs(dt - dt_ref) > dt_ref * 0.01:
                    raise ValueError(
                        f'{Path(p).name}: sample rate differs from first file '
                        f'({1/dt:.4g} Hz vs {1/dt_ref:.4g} Hz). '
                        'All files must have the same sample rate.')
                az_parts.append(data[:, 3])
            return paths, np.concatenate(az_parts), dt_ref

        self._busy(True, f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_open_done, self._on_open_fail,
                      on_progress=self._on_progress,
                      thread_attr='_open_thread', worker_attr='_open_worker')

    def _on_open_fail(self, tb):
        self._finish_thread('_open_thread')
        self._busy(False)
        QMessageBox.critical(self, 'Could not read the record',
                             tb.strip().splitlines()[-1])

    def _on_open_done(self, payload):
        self._finish_thread('_open_thread')
        paths, az, dt = payload
        self.dt = dt
        self.az = az
        self.path = Path(paths[0])
        self.day = None
        self.segments = None
        self.field_note.setVisible(False)
        duration = len(az) * dt
        self.measured.setValue(duration)
        if len(paths) == 1:
            label = (f'{self.path.name} — {len(az):,} samples, '
                     f'{duration:.1f} s at {1/dt:.4g} Hz. '
                     f'Peak |a_z| = {np.abs(az).max():.2f} m/s2.')
        else:
            label = (f'{len(paths)} files concatenated — {len(az):,} samples, '
                     f'{duration:.1f} s at {1/dt:.4g} Hz. '
                     f'Peak |a_z| = {np.abs(az).max():.2f} m/s2.')
        self.file_label.setText(label)
        self.file_label.setStyleSheet(f'color:{TEXT};')
        self._busy(False)
        self.status(f'Loaded {len(paths)} file(s).')

    def load_field(self):
        if self._busy_flag:
            return
        picked = ask_day_files(self, 'Select the files of one seat, one day')
        if picked is None:
            return
        paths, options = picked

        def job(report):
            return fd.load_day_auto(paths, calibrate=False, progress=report,
                                    position='seat', **options)

        self._busy(True, f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_field_load_done,
                      self._on_field_load_fail, on_progress=self._on_progress,
                      thread_attr='_field_thread', worker_attr='_field_worker')

    def _busy(self, on, message=''):
        self._busy_flag = on
        self.field_progress.setVisible(on)
        if message:
            self.status(message)
        for b in (self.open_btn, self.field_btn):
            b.setEnabled(not on)
        self.run_btn.setEnabled(not on and (self.path is not None or
                                            self.day is not None))
        self.export_btn.setEnabled(not on and self.result is not None)

    def _on_progress(self, frac, msg):
        self.status(msg)

    def _finish_thread(self, thread_attr):
        thread = getattr(self, thread_attr, None)
        if thread is not None:
            thread.quit()
            thread.wait()

    def _on_field_load_done(self, day):
        self._finish_thread('_field_thread')
        self.day = day
        self.path = None
        self.dt = day.dt
        # Um segmento por trecho continuo: um por arquivo nos formatos
        # atuais, um por bloco de 2 s no .m antigo.
        self.segments = [day.blocks[a:b, :, 2].reshape(-1)
                         for a, b in day.contiguous_segments()]
        measured_time = len(day.blocks) * fd.BLOCK_SAMPLES * day.dt
        self.measured.setValue(measured_time)
        self.file_label.setStyleSheet(f'color:{TEXT};')
        if day.continuous:
            self.file_label.setText(
                f'{len(day.records)} continuous record(s), '
                f'{measured_time / 3600:.2f} h at {day.rate:.4g} Hz.')
            self.field_note.setText(
                'Each record is conditioned and has loss of contact detected '
                'as one continuous signal. Records are not glued to each '
                'other. "Measurement duration" below is the recorded time.')
        else:
            self.file_label.setText(
                f'{len(day.records)} old-format .m file(s), {len(day.blocks):,} '
                f'blocks of 2 s. Measured {measured_time:.0f} s of '
                f'{day.span_hours * 3600:.0f} s clock time covered '
                f'({day.duty_cycle * 100:.0f}% duty cycle).')
            self.field_note.setText(
                'Old .m format: each 2 s block is conditioned and has loss of '
                'contact detected on its own, never glued to the next. '
                '"Measurement duration" counts only the seconds actually '
                'recorded (blocks x 2 s).')
        self.field_note.setVisible(True)
        self._busy(False)

    def _on_field_load_fail(self, tb):
        self._finish_thread('_field_thread')
        self._busy(False)
        QMessageBox.critical(self, 'Could not read the files',
                             tb.strip().splitlines()[-1])

    # ------------------------------------------------------------------
    def _wc_response_fn(self):
        if not self.wc_enable.isChecked():
            return None
        output = self.wc_output.currentData()
        softening = self.wc_softening.value()
        return lambda cond, dt: acw.spine_response_wheelchair(
            cond, dt, output=output, posture_softening=softening)

    def run(self):
        if self._busy_flag or (self.path is None and self.segments is None):
            return
        # Os valores dos widgets sao lidos aqui, na thread da interface; o
        # calculo (FFT do registro inteiro, picos, dose) vai para segundo
        # plano, para a janela nao congelar num dia de varias horas.
        params = dict(
            body_mass=self.mass.value(),
            sex=self.sex.currentText(),
            daily_time=self.daily.value() * 3600.0,
            measured_time=self.measured.value(),
            days_per_year=self.days.value(),
            start_age=self.start_age.value(),
            n_years=self.years.value(),
            mass_fraction=self.fraction.value() / 100.0,
            endplate_area=self.area.value(),
            added_mass=self.added.value(),
            exclude_loss_of_contact=self.contact.isChecked(),
            spine_response_fn=self._wc_response_fn() or self._spine_tf_response_fn())
        segments, az, dt = self.segments, getattr(self, 'az', None), self.dt

        def job(report):
            report(None, 'Conditioning the signal and computing the response...')
            if segments is not None:
                return annexc.assess_from_segments(segments, dt, **params)
            return annexc.assess(az, dt, **params)

        self._busy(True, 'Conditioning the signal and computing the response...')
        run_in_thread(self, job, self._on_run_done, self._on_run_fail,
                      on_progress=self._on_progress,
                      thread_attr='_run_thread', worker_attr='_run_worker')

    def _on_run_done(self, result):
        self._finish_thread('_run_thread')
        self.result = result
        self._busy(False)
        self._show()
        self.redraw()
        self.status('Clause 5 / Annex C assessment complete.')

    def _on_run_fail(self, tb):
        self._finish_thread('_run_thread')
        self._busy(False)
        self.status('Assessment failed.')
        QMessageBox.critical(self, 'Assessment failed',
                             tb.strip().splitlines()[-1])

    def _reference_r(self, sex):
        """R nas fronteiras de Pi = 10/50/90% (Tabela C.2), por sexo.

        Resolvido numericamente porque a norma da alpha/beta do modelo de
        Weibull, nao os R correspondentes -- e o mesmo calculo que
        tests/test_reference.py usa para validar a Tabela C.2 contra a norma.
        """
        cfg = annexc.SEX_DEFAULTS[sex]
        return [brentq(lambda R: annexc.injury_probability(
            R, cfg['alpha'], cfg['beta']) - p, 0.01, 10.0)
            for p in (0.10, 0.50, 0.90)]

    def _domain_warnings(self, r):
        """Entradas fora da faixa em que a Clausula 5 e o Anexo C foram
        calibrados (adultos; ver references.py). O calculo segue normal."""
        cfg = annexc.SEX_DEFAULTS[r.sex]
        peak = (float(np.abs(r.conditioned).max())
                if len(r.conditioned) else 0.0)
        msgs = [references.start_age_warning(r.start_age, 'C', cfg['s_age']),
                references.child_mass_warning(self.mass.value()),
                references.scope_peak_warning(peak)]
        msgs = [m for m in msgs if m]
        if not msgs:
            return ''
        return (f'<div style="color:{BAD_COLOR};font-size:12px;'
                f'text-align:left;margin-top:4px"><b>Outside the calibration '
                f'domain of the norm.</b> ' + ' '.join(msgs) + '</div>')

    def _show(self):
        r = self.result
        colour = (OK_COLOR if r.probability < annexc.RISK_LOW else
                  BAD_COLOR if r.probability > annexc.RISK_HIGH else WARN_COLOR)
        r10, r50, r90 = self._reference_r(r.sex)
        self.verdict.setText(
            f'<div style="font-size:26px;font-weight:700;color:{colour}">'
            f'&Pi; = {r.probability * 100:.1f}% &nbsp;&nbsp; R = {r.R:.3f}</div>'
            f'<div style="color:{TEXT};font-size:14px">{r.verdict}</div>'
            f'<div style="color:#4a5157;font-size:12px">Clause 5 with Annex C, '
            f'{r.n_years} year(s) from age {r.start_age}, {r.sex}. '
            f'Thresholds: below 10% low, above 50% high — for a {r.sex} '
            f'individual that is R below {r10:.2f} (low) or above {r50:.2f} '
            f'(high); R = {r90:.2f} marks 90% probability, the top of Table '
            f'C.2.</div>' +
            (f'<div style="color:{BAD_COLOR};font-size:12px;font-weight:700">'
             f'v2.9 EXPERIMENTAL: seat-to-spine response from the wheelchair '
             f'posture prototype ({self.wc_output.currentText()}, softening '
             f'{self.wc_softening.value():.2f}), NOT Formula (1). Not '
             f'validated -- research exploration only, do not report.</div>'
             if self.wc_enable.isChecked() else '') +
            (f'<div style="color:{BAD_COLOR};font-size:12px;font-weight:700">'
             f'v3.0 EXPERIMENTAL: seat-to-spine response from a lab-fitted '
             f'Formula (1) structure, NOT the ISO 2631-5 Formula (1) values. '
             f'Results are NOT normative -- research exploration only, do not '
             f'report.</div>'
             if self.spine_tf_enable.isChecked() else '') +
            self._domain_warnings(r))

        rows = [(lbl, getattr(r, attr), src) for lbl, attr, src in RESULT_ROWS]
        rows += [
            ('m_z [MPa/(m/s2)]', r.mz, 'Formula (C.2)'),
            ('S_stat [MPa]', r.s_stat, 'm_z x 9.81'),
            ('Counted peaks', len(r.peaks), 'positive, between zero crossings'),
            ('Discarded to loss of contact [s]', r.discarded_seconds,
             f'{len(r.contact_intervals)} stretch(es), heuristic'),
            (f'— risk reference, Table C.2 ({r.sex}) —', '', ''),
            ('R at Pi = 10% / 50% / 90%', f'{r10:.3f} / {r50:.3f} / {r90:.3f}',
             'low / high / top of table'),
            ('Risk bands (Π)', '<10% / 10-50% / >50%',
             'low / moderate / high, Annex C 6.2'),
        ]
        risk_bg = ('#d4ecd9' if r.probability < annexc.RISK_LOW else
                   '#f6cfcc' if r.probability > annexc.RISK_HIGH else '#fbe8b8')
        self.table.setRowCount(len(rows))
        for i, (lbl, val, src) in enumerate(rows):
            txt = (val if isinstance(val, str) else
                   f'{val * 100:.2f} %' if lbl.startswith('Pi') else
                   f'{val:,}' if isinstance(val, int) else f'{val:.4f}')
            is_key = lbl.startswith('R    ') or lbl.startswith('Pi   ')
            for c, v in enumerate((lbl, txt, src)):
                it = QTableWidgetItem(v)
                it.setForeground(QColor(TEXT))
                if is_key:
                    it.setBackground(QColor(risk_bg))
                if c == 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    fnt = it.font()
                    fnt.setBold(True)
                    it.setFont(fnt)
                self.table.setItem(i, c, it)

    def redraw(self):
        if self.result is None:
            return
        r = self.result
        kind = self.plot_kind.currentIndex()
        self.plot.reset_if_changed(kind)

        if kind == 2:
            f = np.logspace(-1, np.log10(100), 600)
            H = annexc.transfer_function(f)
            ax1, ax2 = self.plot.axes(2)
            spine_on = self.spine_tf_enable.isChecked()
            # When the lab-fitted TF is active the ISO Formula (1) becomes the
            # dashed gray reference and the fitted curve is the primary (solid).
            iso_color = '#8a949d' if spine_on else ACCENT
            iso_ls = '--' if spine_on else '-'
            iso_lw = 1.0 if spine_on else 1.4
            iso_label = ('ISO 2631-5 Formula (1) — reference'
                         if spine_on else 'Formula (1), vehicle')
            ax1.semilogx(f, np.abs(H), lw=iso_lw, color=iso_color, ls=iso_ls,
                        label=iso_label)
            ax1.set_ylabel('|H|')
            title = 'Seat-to-spine transfer function'
            if self.wc_enable.isChecked():
                Hwc = acw.transfer_function(
                    f, output=self.wc_output.currentData(),
                    posture_softening=self.wc_softening.value())
                ax1.semilogx(f, np.abs(Hwc), lw=1.4, color=BAD_COLOR, ls='--',
                            label=f'v2.9 prototype, {self.wc_output.currentText()}')
                ax2.semilogx(f, np.unwrap(np.angle(Hwc)), lw=1.4,
                            color=BAD_COLOR, ls='--')
                title += ' — dashed: EXPERIMENTAL, not Formula (1)'
            if spine_on:
                H_fit = core._spine_tf_model(f, self._spine_zero, self._spine_poles)
                ax1.semilogx(f, np.abs(H_fit), lw=1.6, color=ACCENT,
                            label='Lab-fitted TF (Formula 1 structure) — NOT normative')
                ax2.semilogx(f, np.unwrap(np.angle(H_fit)), lw=1.6, color=ACCENT)
                title += ' — solid: lab-fitted (NOT normative)'
            ax1.set_title(title)
            ax1.legend(fontsize=8)
            ax2.semilogx(f, np.unwrap(np.angle(H)), lw=iso_lw, color=iso_color,
                        ls=iso_ls)
            ax2.set_ylabel('Phase [rad]')
            ax2.set_xlabel('Frequency [Hz]')
            self.plot.finish(ax1, ax2)
            return

        if kind == 1:
            ax, = self.plot.axes()
            if len(r.peaks):
                ax.hist(r.peaks, bins=60, color=ACCENT, alpha=0.85)
                ax.set_yscale('log')
                ax.axvline(r.peaks.max() / 3, ls='--', lw=1, color=WARN_COLOR,
                           label='one third of the highest peak')
                ax.legend(fontsize=8)
            ax.set_xlabel('Response acceleration peak [m/s$^2$]')
            ax.set_ylabel('Count')
            ax.set_title(f'{len(r.peaks)} positive peaks, highest '
                         f'{r.peaks.max():.1f} m/s$^2$' if len(r.peaks)
                         else 'No peaks counted')
            self.plot.finish(ax)
            return

        if kind == 3:
            self._draw_spectrum()
            return

        if kind in (6, 7):
            self._draw_freq_extra(kind)
            return

        if kind == 8:
            ax, = self.plot.axes()
            if hasattr(self, 'az') and self.az is not None:
                az = self.az
            elif self.segments is not None:
                az = np.concatenate(self.segments)
            else:
                return
            dt = self.dt
            stride = max(1, len(az) // 500_000)
            t = np.arange(0, len(az), stride) * dt
            ax.plot(t, az[::stride], lw=0.5, color='#2a2a2a')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Raw $a_z$ [m/s$^2$]')
            ax.set_title('Raw vertical acceleration a_z (input to Clause 5, not conditioned)')
            self.plot.finish(ax)
            return

        if kind == 9:
            self._draw_weibull_curve()
            return

        if kind == 4:
            self._draw_risk_bands()
            return

        if kind == 5:
            self._draw_sensitivity()
            return

        ax1, ax2 = self.plot.axes(2)
        cond, resp = r.conditioned, r.response
        stride = max(1, len(cond) // 500_000)
        t = np.arange(0, len(cond), stride) * r.dt
        ax1.plot(t, cond[::stride], lw=0.5, color='#8a949d')
        ax1.set_ylabel('Conditioned $a_z$ [m/s$^2$]')
        split = self.segments is not None and len(self.segments) > 1
        ax1.set_title('Clause 5 conditioning: 0.01 Hz high pass, 80 Hz low pass'
                      + (f' — {len(self.segments)} segments shown back to '
                         f'back, each conditioned on its own' if split
                         else ''))
        ax2.plot(t, resp[::stride], lw=0.5, color=ACCENT)
        ax2.set_ylabel('Spine response $A_z$ [m/s$^2$]')
        ax2.set_xlabel('Recorded time, segments back to back [s]' if split
                       else 'Time [s]')
        self.plot.finish(ax1, ax2)

    # ------------------------------------------------------------------
    def _draw_freq_extra(self, kind):
        """Espectrograma e SRS do az bruto: so diagnostico, a Clausula 5
        continua sendo dominio do tempo e nao pondera por frequencia."""
        if kind == 6:
            if self.segments is not None and not self.day.continuous:
                blocks = np.asarray(self.segments)
                f, S = core.block_spectrogram(blocks, self.dt)
                t = np.arange(len(blocks)) * fd.BLOCK_PERIOD
                freqplots.draw_spectrogram(
                    self.plot, t, f, S,
                    'Spectrogram of raw $a_z$ across blocks (informative)',
                    tlabel='Block start [s]',
                    waterfall=self.waterfall.isChecked(), logy=True)
            else:
                az = (np.concatenate(self.segments)
                      if self.segments is not None else self.az)
                t, f, S = core.spectrogram(az, self.dt, nperseg=1024)
                freqplots.draw_spectrogram(
                    self.plot, t, f, S,
                    'Spectrogram of raw $a_z$ (informative)'
                    + (', records back to back' if self.segments is not None
                       and len(self.segments) > 1 else ''),
                    waterfall=self.waterfall.isChecked(), logy=True)
            return
        if self.segments is not None and self.day.continuous:
            windows = _peak_windows(self.segments, self.dt, half_s=5.0,
                                    count=200)
            srs_all = None
            for x in windows:
                fr, sr = core.shock_response_spectrum(x, self.dt)
                srs_all = sr if srs_all is None else np.maximum(srs_all, sr)
            series = [(f'envelope over 10 s windows around the '
                       f'{len(windows)} highest peaks', fr, srs_all)]
        elif self.segments is not None:
            peaks = np.array([np.abs(b).max() for b in self.segments])
            keep = np.argsort(peaks)[::-1][:200]
            srs_all = None
            for i in keep:
                fr, sr = core.shock_response_spectrum(self.segments[i], self.dt)
                srs_all = sr if srs_all is None else np.maximum(srs_all, sr)
            series = [(f'envelope of the {len(keep)} highest-peak blocks',
                       fr, srs_all)]
        else:
            fr, sr = core.shock_response_spectrum(self.az, self.dt)
            series = [('raw $a_z$', fr, sr)]
        freqplots.draw_srs(self.plot, series,
                           'Shock Response Spectrum of raw $a_z$ (Q=10, '
                           'informative)')

    def _draw_spectrum(self):
        """Espectro do az bruto (nao condicionado) -- so diagnostico.

        Clause 5/Anexo C nao pondera por frequencia (docstring do modulo:
        os filtros da 2631-1 sao proibidos aqui). Este grafico existe so
        para inspecionar o conteudo em frequencia do que foi medido --
        ressonancias, ruido, sensor solto -- antes/depois de rodar a cadeia
        de tempo que a norma de fato usa. Nao entra em nenhum resultado.
        """
        overlap = 0.5
        if self.segments is not None:
            raw = np.concatenate(self.segments)
            nperseg = fd.BLOCK_SAMPLES
            if not self.day.continuous:
                overlap = 0.0        # blocos isolados: segmento = bloco
        else:
            raw = self.az
            nperseg = min(8192, len(raw))
        f, y, navg = core.welch_spectrum(raw, self.dt, nperseg=nperseg,
                                         overlap=overlap)
        ax, = self.plot.axes()
        ax.semilogy(f, y, lw=0.9, color=ACCENT)
        ax.set_xlim(0, min(100.0, 0.5 / self.dt))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [m/s$^2$]')
        ax.set_title(f'Raw $a_z$ spectrum, {navg} block(s) averaged, informative '
                     f'only — Clause 5 itself is time-domain, no frequency '
                     f'weighting is applied')
        self.plot.finish(ax)

    def _draw_risk_bands(self):
        """Onde R e Pi caem frente as faixas de referencia da Tabela C.2."""
        r = self.result
        r10, r50, r90 = self._reference_r(r.sex)
        top = max(r90 * 1.15, r.R * 1.1)
        ax, = self.plot.axes()
        ax.barh(0, r10 / top, color='#d4ecd9', height=0.5)
        ax.barh(0, (r50 - r10) / top, left=r10 / top, color='#fbe8b8', height=0.5)
        ax.barh(0, 1 - r50 / top, left=r50 / top, color='#f6cfcc', height=0.5)
        colour = (OK_COLOR if r.probability < annexc.RISK_LOW else
                  BAD_COLOR if r.probability > annexc.RISK_HIGH else WARN_COLOR)
        ax.plot([r.R / top] * 2, [-0.32, 0.32], lw=3.5, color=colour)
        ax.text(min(r.R / top, 0.97), 0.4, f'R = {r.R:.3f}\nΠ = '
                f'{r.probability * 100:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        for val, lbl in ((r10, '10%'), (r50, '50%'), (r90, '90%')):
            ax.axvline(val / top, color='#4a5157', lw=0.8, ls='--')
            ax.text(val / top, -0.42, f'{lbl}\nR={val:.2f}', ha='center',
                   va='top', fontsize=8, color='#4a5157')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'R and Π against Table C.2 ({r.sex}): green below the '
                     f'10% probability boundary, amber to 50%, red above — '
                     f'dashed lines mark the R value at each named probability')
        self.plot.finish(ax)

    def _draw_weibull_curve(self):
        """Curva Π(R) de Weibull com zonas de risco coloridas (Tabela C.1)."""
        r = self.result
        cfg = annexc.SEX_DEFAULTS[r.sex]
        alpha, beta = cfg['alpha'], cfg['beta']
        r10, r50, r90 = self._reference_r(r.sex)

        R_max = max(r90 * 1.30, r.R * 1.20, 3.0)
        R_arr = np.linspace(0, R_max, 1000)
        Pi = (1.0 - np.exp(-((R_arr / alpha) ** beta))) * 100.0

        ax, = self.plot.axes()

        ax.axhspan(0,   10,  alpha=0.40, color='#d4ecd9', zorder=0, label='Low (<10%)')
        ax.axhspan(10,  50,  alpha=0.40, color='#fbe8b8', zorder=0, label='Moderate (10–50%)')
        ax.axhspan(50, 100,  alpha=0.40, color='#f6cfcc', zorder=0, label='High (>50%)')

        ax.plot(R_arr, Pi, lw=2.2, color='#2c5f8a', zorder=3,
                label=f'Weibull — {r.sex} (α={alpha:.3f}, β={beta:.3f})')

        for R_thr, pct, lbl in [(r10, 10, '10 %'), (r50, 50, '50 %'), (r90, 90, '90 %')]:
            ax.axvline(R_thr, ls='--', lw=0.9, color='#777', zorder=2)
            ax.axhline(pct,   ls='--', lw=0.9, color='#777', zorder=2)
            ax.text(R_thr, 103, f'R={R_thr:.2f}\n{lbl}',
                    ha='center', fontsize=8, color='#555', linespacing=1.3)

        colour = (OK_COLOR if r.probability < annexc.RISK_LOW else
                  BAD_COLOR if r.probability > annexc.RISK_HIGH else WARN_COLOR)
        pi_pct = r.probability * 100.0
        ax.plot(r.R, pi_pct, marker='D', markersize=12, color=colour,
                markeredgecolor='black', markeredgewidth=1.2, zorder=5,
                label=f'This assessment: R={r.R:.3f}, Π={pi_pct:.1f} %')

        offset_x = R_max * 0.07
        offset_y = -14 if pi_pct > 50 else 10
        ax.annotate(
            f'R = {r.R:.3f}\nΠ = {pi_pct:.1f} %',
            xy=(r.R, pi_pct),
            xytext=(r.R + offset_x, pi_pct + offset_y),
            fontsize=9.5, fontweight='bold', color=colour, zorder=6,
            arrowprops=dict(arrowstyle='->', color=colour, lw=1.3),
        )

        ax.set_xlim(0, R_max)
        ax.set_ylim(0, 112)
        ax.set_xlabel('R  (stress variable, dimensionless)')
        ax.set_ylabel('Π  (probability of lumbar injury) [%]')
        ax.set_title(
            f'Weibull injury probability — ISO 2631-5:2018 Annex C, Table C.1 ({r.sex})')
        ax.legend(fontsize=9, loc='upper left')
        self.plot.finish(ax)

    def _draw_sensitivity(self):
        """Dispersao de R por sexo/fracao de massa/area de placa terminal.

        Analogo ao boxplot de sensibilidade do Anexo A/E (postura x 30
        modelos), mas so na dimensao antropometrica -- a Clausula 5 nao tem
        parametro de postura. D_zd fica fixo (Seção 'Lições de estudos
        semelhantes' da documentação): a varredura recalcula so m_z/S_d/R/Pi,
        entao roda instantaneamente sobre o resultado ja calculado.
        """
        r = self.result
        grid = annexc.clause5_sensitivity(
            r.Dzd, body_mass=self.mass.value(), days_per_year=self.days.value(),
            start_age=self.start_age.value(), n_years=self.years.value(),
            current_sex=r.sex, current_fraction=self.fraction.value() / 100.0,
            current_area=self.area.value(), added_mass=self.added.value())

        ax, = self.plot.axes()
        data = [grid['male']['R'], grid['female']['R']]
        box = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False,
                         patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('#dfe3e8')
        colour = (OK_COLOR if r.probability < annexc.RISK_LOW else
                  BAD_COLOR if r.probability > annexc.RISK_HIGH else WARN_COLOR)
        xi = 0 if r.sex == 'male' else 1
        ax.plot(xi, r.R, marker='D', markersize=9, color=colour,
               markeredgecolor='#1a1d21', zorder=5,
               label=f'this assessment: R = {r.R:.3f}')
        ax.axhline(annexc.RISK_LOW, ls='--', lw=0.8, color='#4a5157')
        ax.axhline(annexc.RISK_HIGH, ls='--', lw=0.8, color='#4a5157')
        ax.set_xticks([0, 1], ['male', 'female'])
        ax.set_ylabel('R (stress variable)')
        n = grid['male'].size
        ax.set_title(f'{2 * n} R values, mass fraction and endplate area each '
                     f'±20% around the value in use — dispersion from not '
                     f'knowing the exact individual anthropometry, not from '
                     f'measurement noise')
        ax.legend(fontsize=8, loc='best')
        self.plot.finish(ax)

    def clear(self):
        """Descarta o sinal/dia carregado e todos os resultados."""
        if self._busy_flag:
            return
        self.path = None
        self.day = None
        self.segments = None
        self.result = None
        for attr in ('az', 'dt'):
            if hasattr(self, attr):
                delattr(self, attr)
        self.file_label.setText('No record loaded')
        self.file_label.setStyleSheet('')
        self.file_label.setObjectName('muted')
        self.field_note.setVisible(False)
        self.verdict.setText('No results yet')
        self.table.setRowCount(0)
        self._busy(False)
        self.plot.figure.clear()
        self.plot.canvas.draw()
        self.status('Clause 5 cleared.')

    def _open_calibration_dialog(self):
        """Calibrador de referencia e 6 posicoes estaticas para o sinal
        carregado. Aplica a correcao in-place — recarregue o arquivo para desfazer."""
        if self._busy_flag:
            return
        has_az = hasattr(self, 'az')
        has_seg = self.segments is not None
        if not has_az and not has_seg:
            QMessageBox.information(self, 'Calibration',
                                    'Load a record or field day first.')
            return

        dlg = QDialog(self)
        dlg.setWindowTitle('Calibration — Clause 5 signal')
        dlg.setMinimumWidth(480)
        vlay = QVBoxLayout(dlg)

        tabs = QTabWidget()

        # ---- Tab 1: Reference calibrator ---------------------------------
        tab1 = QWidget()
        form1 = QFormLayout(tab1)
        form1.setContentsMargins(8, 8, 8, 8)

        ref_note = QLabel(
            'Input a sinusoidal reference signal at a known amplitude and '
            'frequency. The FFT peak is located and a correction factor '
            'is computed. Applied to the vertical (z) column of the loaded '
            'signal.')
        ref_note.setWordWrap(True)
        form1.addRow(ref_note)

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
            sig = (self.az if has_az else
                   np.concatenate([s for s in self.segments]))
            is_peak = ref_type.currentIndex() == 0
            r1 = core.calibrate_from_reference(
                sig, self.dt, ref_val.value(), ref_freq.value(), is_peak)
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

        static_note = QLabel(
            'Place the sensor in each of 6 orientations and enter the mean '
            'reading. Applied to all three axes of the loaded signal.')
        static_note.setWordWrap(True)
        form2.addRow(static_note)

        pos_fields = {}
        for ax in core.AXIS_NAMES:
            for sign, direction in [('+', 'up / +g'), ('-', 'down / −g')]:
                key = f'{ax}{sign}'
                sp = QDoubleSpinBox()
                sp.setRange(-200.0, 200.0)
                sp.setDecimals(4)
                sp.setValue(9.80665 if sign == '+' else -9.80665)
                sp.setSuffix(' m/s²')
                form2.addRow(f'{key}  ({ax} axis {direction})', sp)
                pos_fields[key] = sp

        res2_lbl = QLabel('Click "Compute" to calculate offset/gain per axis.')
        res2_lbl.setWordWrap(True)
        form2.addRow('Result', res2_lbl)

        _res2 = [None]

        def _compute_static():
            readings = {k: v.value() for k, v in pos_fields.items()}
            r2 = core.calibrate_static_6pos(readings)
            _res2[0] = r2
            lines = [f"{ax}: offset {r2[ax]['offset']:+.4f} m/s², "
                     f"gain {r2[ax]['gain']:.4f}"
                     for ax in core.AXIS_NAMES]
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
                if has_az:
                    self.az *= r1['correction']
                else:
                    for seg in self.segments:
                        seg *= r1['correction']
                msg = f"Vertical axis scaled by {r1['correction']:.4f}."
            else:
                r2 = _res2[0]
                if not r2:
                    QMessageBox.information(dlg, 'Nothing to apply',
                        'Click Compute first.')
                    return
                if has_az:
                    # self.az is vertical only: apply the z-axis correction
                    dz = r2['z']
                    if abs(dz['gain']) > 1e-6:
                        self.az = (self.az - dz['offset']) / dz['gain']
                else:
                    for axis_name, axis_idx in zip(core.AXIS_NAMES, range(3)):
                        d = r2[axis_name]
                        if abs(d['gain']) > 1e-6:
                            for seg in self.segments:
                                seg[:] = (seg - d['offset']) / d['gain']
                msg = 'Offset and gain correction applied.'
            QMessageBox.information(dlg, 'Applied', msg +
                                    ' Run the assessment again to see updated results.')

        btns.button(QDialogButtonBox.Apply).clicked.connect(_apply)
        btns.rejected.connect(dlg.reject)
        vlay.addWidget(btns)
        dlg.exec()


    # ------------------------------------------------------------------
    def export_pdf(self):
        """Exporta relatorio PDF: tabela de resultados + grafico atual (A4)."""
        if self.result is None:
            return
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime as _dt

        r = self.result

        # ── Dialogo de metadata + seleção de gráficos ─────────────────
        from PySide6.QtWidgets import QLineEdit, QFrame
        dlg_meta = QDialog(self)
        dlg_meta.setWindowTitle('Informações e gráficos para o relatório')
        dlg_meta.setMinimumWidth(440)
        vl_meta = QVBoxLayout(dlg_meta)
        vl_meta.setContentsMargins(12, 12, 12, 8)
        fm = QFormLayout()
        meta_fields = {}
        for key, label in [
            ('subject_id', 'Identificação do sujeito'),
            ('date', 'Data da medição (ex.: 2026-10-11)'),
            ('operator', 'Operador / avaliador'),
            ('notes', 'Observações'),
        ]:
            le = QLineEdit()
            if key == 'date':
                le.setText(_dt.today().strftime('%Y-%m-%d'))
            fm.addRow(label, le)
            meta_fields[key] = le
        vl_meta.addLayout(fm)

        sep_lbl = QLabel('Gráficos a incluir (além da tabela de resultados)')
        sep_lbl.setStyleSheet('font-weight: bold; margin-top: 6px;')
        vl_meta.addWidget(sep_lbl)

        _plot_labels = [
            (0, 'Sinal condicionado e resposta espinal', True),
            (1, 'Picos contados (histograma)', False),
            (2, 'Função de transferência assento→espinha', False),
            (3, 'Espectro de frequência (informativo)', False),
            (4, 'Bandas de referência de risco (R e Π)', False),
            (5, 'Sensibilidade antropométrica', False),
            (6, 'Espectrograma de a_z bruto (informativo)', False),
            (7, 'Shock Response Spectrum de a_z bruto', False),
            (8, 'Série temporal de a_z bruto', False),
            (9, 'Curva de Weibull (já incluída na pág. 2 — adicionar cópia extra)', False),
        ]
        plot_checks = []
        for idx, lbl, checked in _plot_labels:
            cb = QCheckBox(lbl)
            cb.setChecked(checked)
            vl_meta.addWidget(cb)
            plot_checks.append((cb, idx))

        btns_meta = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns_meta.button(QDialogButtonBox.StandardButton.Ok).setText('Exportar PDF')
        btns_meta.accepted.connect(dlg_meta.accept)
        btns_meta.rejected.connect(dlg_meta.reject)
        vl_meta.addWidget(btns_meta)
        if dlg_meta.exec() != QDialog.DialogCode.Accepted:
            return
        meta = {k: v.text().strip() for k, v in meta_fields.items()}

        stem = self.path.stem if self.path else 'annex_c'
        path_out, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF report', f'{stem}_annexc_report.pdf', 'PDF (*.pdf)')
        if not path_out:
            return

        import matplotlib as _mpl
        _rc_save = {k: _mpl.rcParams[k] for k in ['font.family']}
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        from PySide6.QtCore import Qt as _Qt
        prog = None
        try:
            _mpl.rcParams['font.family'] = 'serif'

            BLACK = 'black'; WHITE = 'white'; GREY = '#f2f2f2'
            RULE = '#333333'; TXT = 'black'; MUTED = '#555555'
            from .theme import OK_COLOR, WARN_COLOR, BAD_COLOR
            colour = (OK_COLOR if r.probability < annexc.RISK_LOW else
                      BAD_COLOR if r.probability > annexc.RISK_HIGH else WARN_COLOR)
            FOOTER_Y = 0.024

            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(WHITE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

            def R(x, y, w, h, fc, ec=None, lw=0.5):
                if y < FOOTER_Y:
                    return
                ax.add_patch(mp.Rectangle(
                    (x, y), w, h, facecolor=fc,
                    edgecolor=(ec or fc), linewidth=lw, clip_on=False,
                    transform=ax.transAxes, zorder=1))

            def T(x, y, s, **kw):
                if y < FOOTER_Y:
                    return
                ax.text(x, y, s, transform=ax.transAxes, clip_on=False,
                        zorder=2, **kw)

            def hline(y, lw=0.8, color=RULE):
                ax.axhline(y, lw=lw, color=color, clip_on=False, zorder=1)

            # Cabeçalho
            R(0, 0.960, 1, 0.040, BLACK)
            T(0.016, 0.980, 'Avaliador de Vibração Humana — Clause 5 / Annex C (ISO 2631-5)',
              color=WHITE, fontsize=9.5, fontweight='bold', va='center')
            T(0.984, 0.980, _dt.now().strftime('%Y-%m-%d  %H:%M'),
              color='#cccccc', fontsize=7.5, va='center', ha='right')
            hline(0.960, lw=1.2, color=BLACK)

            meta_str = '  ·  '.join(
                v for v in [meta.get('subject_id'), meta.get('date'),
                            meta.get('operator')] if v)
            if meta_str:
                T(0.016, 0.948, meta_str, color=MUTED, fontsize=8, va='center')
                hline(0.934, lw=0.5)

            y = 0.918

            # Seção de resultado
            T(0.016, y, 'Resultado — Clause 5 / Annex C (ISO 2631-5)',
              color=BLACK, fontsize=9, fontweight='bold', va='bottom')
            hline(y - 0.004, lw=0.8)
            y -= 0.024

            # Veredito
            pi_str = f'Π = {r.probability * 100:.1f}%   R = {r.R:.3f}'
            T(0.016, y, pi_str, color=colour, fontsize=15, fontweight='bold', va='top')
            T(0.016, y - 0.036, r.verdict, color=TXT, fontsize=9, va='top')
            y -= 0.062

            # Tabela de resultados
            T(0.016, y, 'Tabela de resultados',
              color=BLACK, fontsize=8.5, fontweight='bold', va='bottom')
            hline(y - 0.003, lw=0.6)
            y -= 0.010

            rows = []
            for row in range(self.table.rowCount()):
                cells = [self.table.item(row, c).text()
                         if self.table.item(row, c) else ''
                         for c in range(self.table.columnCount())]
                if any(c.strip() for c in cells):
                    rows.append(cells)

            ROW_H = 0.022
            for ri, cells in enumerate(rows):
                bg = GREY if ri % 2 == 0 else WHITE
                R(0, y - ROW_H, 1, ROW_H, bg, RULE, 0.3)
                col_x = [0.016, 0.50, 0.76]
                for cx, txt in zip(col_x, cells):
                    T(cx, y - ROW_H * 0.6, txt, color=TXT, fontsize=7.5,
                      va='center', ha='left')
                y -= ROW_H

            # Parâmetros de exposição
            y -= 0.010
            T(0.016, y, 'Parâmetros',
              color=BLACK, fontsize=8.5, fontweight='bold', va='bottom')
            hline(y - 0.003, lw=0.6)
            y -= 0.010
            params = [
                ('Sexo', r.sex),
                ('Exposição diária', f'{self.daily.value():.1f} h/dia'),
                ('Dias/ano', str(self.days.value())),
                ('Anos de exposição', str(r.n_years)),
            ]
            for ri, (lbl, val) in enumerate(params):
                bg = GREY if ri % 2 == 0 else WHITE
                R(0, y - ROW_H, 0.5, ROW_H, bg, RULE, 0.3)
                R(0.5, y - ROW_H, 0.5, ROW_H, bg, RULE, 0.3)
                T(0.016, y - ROW_H * 0.6, lbl, color=MUTED, fontsize=7.5, va='center')
                T(0.516, y - ROW_H * 0.6, val, color=TXT, fontsize=7.5, va='center')
                y -= ROW_H

            if meta.get('notes'):
                y -= 0.008
                T(0.016, y, f'Obs.: {meta["notes"]}', color=MUTED, fontsize=7, va='top')

            # Rodapé
            hline(FOOTER_Y, lw=0.6)
            T(0.016, FOOTER_Y / 2, 'Avaliador de Vibração Humana · ISO 2631-5 Clause 5 / Annex C',
              color=MUTED, fontsize=6.5, va='center')
            T(0.984, FOOTER_Y / 2, 'GRAVI-SH · UFMG',
              color=MUTED, fontsize=6.5, ha='right', va='center')

            prog = QProgressDialog('Gerando PDF…', '', 0, 0, self)
            prog.setWindowTitle('Exportar PDF')
            prog.setWindowModality(_Qt.WindowModality.WindowModal)
            prog.setCancelButton(None)
            prog.setMinimumDuration(0)
            prog.show()
            QApplication.processEvents()

            with PdfPages(path_out) as pdf:
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
                # Página 2: curva de Weibull (sempre incluída)
                fig_w = pdf_report.weibull_figure(r)
                fig_w.set_size_inches(11.69, 8.27)
                pdf.savefig(fig_w, bbox_inches='tight', dpi=150)
                plt.close(fig_w)
                # Páginas de gráficos selecionados (A4 paisagem)
                orig_kind = self.plot_kind.currentIndex()
                orig_size = self.plot.figure.get_size_inches()
                for cb, idx in plot_checks:
                    if not cb.isChecked():
                        continue
                    self.plot_kind.setCurrentIndex(idx)
                    self.redraw()
                    QApplication.processEvents()
                    self.plot.figure.set_size_inches(11.69, 8.27)
                    pdf.savefig(self.plot.figure, bbox_inches='tight', dpi=150)
                    self.plot.figure.set_size_inches(*orig_size)
                self.plot_kind.setCurrentIndex(orig_kind)
                self.redraw()
        except Exception as _e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(_e))
            return
        finally:
            if prog is not None:
                prog.close()
            _mpl.rcParams.update(_rc_save)

        self.status(f'PDF saved to {Path(path_out).name}.')


def _peak_windows(segments, dt, half_s=5.0, count=200):
    """Janelas de 2*half_s segundos centradas nos `count` maiores picos de
    |a_z|, sem sobreposicao, para o envelope do SRS de um registro longo:
    calcular o SRS de horas de sinal inteiro levaria minutos, e os maiores
    picos sao o que define o envelope maximax."""
    half = max(1, int(round(half_s / dt)))
    out = []
    for x in segments:
        mag = np.abs(x)
        order = np.argsort(mag)[::-1]
        taken = np.zeros(len(x), dtype=bool)
        found = 0
        for i in order:
            if found >= count:
                break
            if taken[i]:
                continue
            a, b = max(0, i - half), min(len(x), i + half)
            taken[a:b] = True
            out.append((mag[i], x[a:b]))
            found += 1
    out.sort(key=lambda p: -p[0])
    return [w for _, w in out[:count]]
