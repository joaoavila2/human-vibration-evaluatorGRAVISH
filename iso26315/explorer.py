"""Explorador de sinais: inspeção direta de registros de aceleração.

Fora do escopo da ISO 2631-5. Não calcula dose nem fator de risco: serve para
olhar um registro antes de montar um histórico de exposição, comparar arquivos
entre si e exportar espectros.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import continuity, core, fielddata as fd, freqplots, nho09, references
from .qtworker import run_in_thread
from .theme import BAD_COLOR, TEXT
from .loaddialog import DAY_FILTER
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
    ('Skewness', lambda s: _v(s['skewness'], 2)),
    ('P50 |a| x, y, z [m/s2]', lambda s: _v(s['abs_pcts'][50])),
    ('P90 |a| x, y, z [m/s2]', lambda s: _v(s['abs_pcts'][90])),
    ('P95 |a| x, y, z [m/s2]', lambda s: _v(s['abs_pcts'][95])),
    ('P99 |a| x, y, z [m/s2]', lambda s: _v(s['abs_pcts'][99])),
    ('Time of peak [s] x, y, z', lambda s: _v(s['t_peak'], 2)),
    ('Dominant frequency [Hz]', lambda s: _v(s['dominant'], 2)),
    ('Intensity class', lambda s: str(s['intensity_class'])),
    ('--- |a_z| events ---', lambda s: ''),
    ('Peaks > 0.5 g (z)',
     lambda s: f"{s['event_counts'][0.5]['peaks']}  "
               f"(crossings: {s['event_counts'][0.5]['crossings']})"),
    ('Peaks > 1.0 g (z)',
     lambda s: f"{s['event_counts'][1.0]['peaks']}  "
               f"(crossings: {s['event_counts'][1.0]['crossings']})"),
    ('Peaks > 2.0 g (z)',
     lambda s: f"{s['event_counts'][2.0]['peaks']}  "
               f"(crossings: {s['event_counts'][2.0]['crossings']})"),
    ('--- ISO 2631-1 weighted ---', lambda s: ''),
    ('Weighting per axis', lambda s: ',  '.join(
        f"{w}(k={k:g})" for w, k in zip(s['w']['weighting'], s['w']['k']))),
    ('a_w x, y, z [m/s2]', lambda s: _v(s['w']['aw'])),
    ('Vector sum a_w [m/s2]', lambda s: f"{s['w']['aw_total']:.4f}"),
    ('A(8) x, y, z [m/s2]',
     lambda s: _v([m / 28800 ** 0.5 for m in s['w']['msdv']])),
    ('A(8) vector sum [m/s2]',
     lambda s: f"{s['w']['msdv_total'] / 28800 ** 0.5:.4f}"),
    ('Weighted peak [m/s2]', lambda s: _v(s['w']['peak'], 2)),
    ('Weighted crest factor', lambda s: _v(s['w']['crest'], 1)),
    ('Weighted VDV [m/s1.75]', lambda s: _v(s['w']['vdv'], 2)),
    ('Weighted MSDV [m/s1.5]', lambda s: _v(s['w']['msdv'], 2)),
    ('MTVV x, y, z [m/s2]', lambda s: _v(s['w']['mtvv'], 3)),
    ('P50 RMS(1s) x, y, z [m/s2]',
     lambda s: _v(s['w']['rms_pcts'][50])),
    ('P90 RMS(1s) x, y, z [m/s2]',
     lambda s: _v(s['w']['rms_pcts'][90])),
    ('P95 RMS(1s) x, y, z [m/s2]',
     lambda s: _v(s['w']['rms_pcts'][95])),
    ('P99 RMS(1s) x, y, z [m/s2]',
     lambda s: _v(s['w']['rms_pcts'][99])),
    ('--- ISO 2631-1 6.3.3 decision ---', lambda s: ''),
    (f'MTVV / a_w (limit {core.MTVV_RATIO_LIMIT:g})',
     lambda s: _v(s['w']['ratios']['mtvv_ratio'], 2)),
    (f'VDV / (a_w T^1/4) (limit {core.VDV_RATIO_LIMIT:g})',
     lambda s: _v(s['w']['ratios']['vdv_ratio'], 2)),
    ('Basic method (RMS) sufficient', lambda s: _basic_method(s['w']['ratios'])),
    ('--- ISO 2631-5 validity ---', lambda s: ''),
    ('Band-limited z peak (0.4-100 Hz) [m/s2]',
     lambda s: f"{s['band_peak_z']:.2f}"),
    ('Annex A / E applicable', lambda s:
        'yes' if s['band_peak_z'] <= core.VALIDITY_PEAK_LIMIT
        else f'NO - above {core.VALIDITY_PEAK_LIMIT} m/s2, use Clause 5 / Annexes C, D'),
    ('--- Reference levels (ISO 2631-1, adults) ---', lambda s: ''),
    ('Comfort (Table 3, a_w total)',
     lambda s: _comfort_label(s['w']['aw_total'])),
    ('EU Dir. 2002/44 A(8) action (0.5 m/s2)',
     lambda s: _eu_directive_status(s['w']['msdv_total'] / 28800 ** 0.5)),
    ('ISO 2631-1 Ann. C health zone (A(8))',
     lambda s: _annex_c_label(s['w']['msdv_total'] / 28800 ** 0.5)),
    ('Weighting validity note', lambda s: _small_mass_note(s)),
    ('Sensor / instrument warnings',
     lambda s: '; '.join(core.sensor_warnings(
         s.get('weighting_config', {'x': (None, 1.0), 'y': (None, 1.0),
                                    'z': (None, 1.0)}),
         s.get('fs', 0),
         sensor_fc_low_hz=float(s.get('meta', {}).get('sensor_fc_low_hz', 0) or 0)
             if s.get('meta', {}).get('sensor_fc_low_hz') else None
     )) or '—'),
    ('--- Metadata ---', lambda s: ''),
    ('Subject age', lambda s: s.get('meta', {}).get('subject_age', '—')),
    ('Subject mass [kg]', lambda s: s.get('meta', {}).get('subject_mass_kg', '—')),
    ('Subject height [cm]', lambda s: s.get('meta', {}).get('subject_height_cm', '—')),
    ('Position', lambda s: s.get('meta', {}).get('position', '—')),
    ('Sensor model', lambda s: s.get('meta', {}).get('sensor_model', '—')),
    ('Sensor serial', lambda s: s.get('meta', {}).get('sensor_serial', '—')),
    ('Calibration', lambda s: s.get('meta', {}).get('calibration', '—')),
    ('Operator / notes', lambda s: s.get('meta', {}).get('notes',
        s.get('meta', {}).get('operator', '—'))),
]


def _band_peak_z(exc):
    """Pico de z depois dos filtros de banda 0,4-100 Hz, o criterio dos
    Anexos A e E. Ate a v2.9 esta tabela usava o pico W_k, que nao e' o que
    a norma pede (ver core.validity_check)."""
    return core.validity_check({'x': exc})['worst']


def _basic_method(r):
    axes = [a for a, flag in zip('xyz', r['additional_required']) if flag]
    crest = [a for a, flag in zip('xyz', r['crest_above']) if flag]
    if not axes and not crest:
        return 'yes'
    parts = []
    if axes:
        parts.append(f"no on {', '.join(axes)}: report MTVV or VDV")
    if crest:
        parts.append(f"crest factor above {core.CREST_FACTOR_LIMIT:g} on "
                     f"{', '.join(crest)}")
    return '; '.join(parts)


def _v(arr, dec=4):
    return ',  '.join(f'{x:.{dec}f}' for x in arr)


def _comfort_label(aw_total):
    """ISO 2631-1:1997 Table 3 — conforto em transporte (vetorial a_w)."""
    if aw_total < 0.315:
        return f'{aw_total:.3f} m/s² — not uncomfortable'
    elif aw_total < 0.5:
        return f'{aw_total:.3f} m/s² — a little uncomfortable'
    elif aw_total < 0.8:
        return f'{aw_total:.3f} m/s² — fairly uncomfortable'
    elif aw_total < 1.25:
        return f'{aw_total:.3f} m/s² — uncomfortable'
    elif aw_total < 2.0:
        return f'{aw_total:.3f} m/s² — very uncomfortable'
    return f'{aw_total:.3f} m/s² — extremely uncomfortable'


def _eu_directive_status(a8):
    """EU Directive 2002/44/EC: action 0.5 m/s², limit 1.15 m/s² A(8)."""
    if a8 < 0.5:
        return f'{a8:.3f} m/s² — below action value'
    elif a8 < 1.15:
        return f'{a8:.3f} m/s² — ABOVE action value (0.5 m/s²)'
    return f'{a8:.3f} m/s² — ABOVE LIMIT VALUE (1.15 m/s²)'


def _annex_c_label(a8):
    """ISO 2631-1:1997 Annex C health guidance caution zones."""
    if a8 < 0.5:
        return f'{a8:.3f} m/s² — health effects not known'
    elif a8 < 1.7:
        return f'{a8:.3f} m/s² — caution zone (0.5–1.7 m/s²)'
    return f'{a8:.3f} m/s² — health effects indicated (> 1.7 m/s²)'


def _small_mass_note(s):
    """Item 7.4: aviso de validez das curvas de ponderacao para massas pequenas."""
    preset = s['w'].get('preset') or ''
    if 'health' in preset.lower() or 'comfort' in preset.lower():
        return ('Adult ISO 2631-1 curves. Not validated for body mass '
                '< ~2 kg (Solvik-Olsen et al. 2024).')
    if 'unweighted' in preset.lower() or 'none' in preset.lower():
        return 'Unweighted — no frequency correction applied.'
    return 'Custom weighting — verify applicability.'


def _field_to_excitation(rec, name):
    """Converte FieldRecord para Excitation concatenando todos os blocos."""
    blocks = rec.blocks_si(remove_gravity=True)  # (n_blocos, 2048, 3)
    accel = blocks.reshape(-1, 3)
    dt = 1.0 / rec.rate
    t = np.arange(accel.shape[0]) * dt
    exc = core.Excitation(name=name, time=t, accel=accel, dt=dt)
    core._prepare(exc)
    return exc


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
        self.add_btn = b1
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
                            'Phase spectrum', 'Coherence (vs first checked)',
                            'Transmissibility (SEAT: vs first checked)',
                            'Transmissibility H1 (vs first checked)',
                            'Spectrogram (first checked; 3D waterfall option)',
                            'Shock Response Spectrum (SRS)',
                            'Transmissibility H1 by input amplitude class '
                            '(vs first checked)',
                            'Interval values (a_w, MTVV, VDV per interval)',
                            'Acceleration histogram / CDF',
                            'ODS: H1 magnitude + phase (field, vs first)',
                            'Multiple coherence (multiple inputs vs first)',
                            'Weighting filter response (magnitude + phase)',  # 14
                            'Transmissibility H1 in 1/3-octave bands'])
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
        self.hp = QCheckBox('High-pass Butterworth pre-filter')
        self.hp.setToolTip(
            'Zero-phase Butterworth high-pass (order 6) applied before '
            'coherence / H1. View only.')
        g.addWidget(self.hp, 4, 0, 1, 2)
        self.hp_fc = QDoubleSpinBox()
        self.hp_fc.setRange(0.05, 20.0)
        self.hp_fc.setSingleStep(0.1)
        self.hp_fc.setValue(0.5)
        self.hp_fc.setSuffix(' Hz cutoff')
        g.addWidget(self.hp_fc, 4, 2)
        self.waterfall = QCheckBox('3D waterfall (spectrogram)')
        self.waterfall.setToolTip(
            'Replaces the 2D spectrum plot with a 3D waterfall / spectrogram: '
            'frequency on the x axis, record index (or time of day for Directive) '
            'on the y axis, and spectral amplitude on the z axis. Useful for '
            'spotting how the vibration content evolves over a long recording.')
        g.addWidget(self.waterfall, 4, 3)
        self.align = QCheckBox('Align pairs by cross-correlation')
        self.align.setToolTip(
            'Coherence and H1: shift each record onto the reference by the '
            'delay of the cross-correlation peak (up to '
            f'{core.ALIGN_MAX_LAG_S:g} s) before comparing. Records from '
            'loggers without a shared clock start at different instants. A '
            'pair whose delay drifts along the record is refused, with the '
            'reason.')
        g.addWidget(self.align, 5, 0, 1, 2)
        self.reject = QCheckBox('Reject bad segments')
        self.reject.setToolTip(
            'Coherence and H1: leave out segments whose output/input RMS '
            'ratio, offset step or kurtosis is far from the other segments '
            '(lost contact, loose sensor, impact), and segments that reach '
            'the sensor range when it is given. The note says how many and '
            'why.')
        g.addWidget(self.reject, 5, 2)
        self.full_scale = QDoubleSpinBox()
        self.full_scale.setRange(0.0, 500.0)
        self.full_scale.setDecimals(1)
        self.full_scale.setSuffix(' g range')
        self.full_scale.setSpecialValueText('range unknown')
        self.full_scale.setToolTip(
            'Sensor full scale (for example 2, 4, 8 or 16 g). Segments where '
            'the record, as loaded, reaches 99.5 % of it count as clipped. '
            '0 = unknown: clipping is not checked, because no rule without '
            'the range is reliable.')
        g.addWidget(self.full_scale, 5, 3)
        self.iv_label = QLabel('Interval')
        g.addWidget(self.iv_label, 6, 0)
        self.interval_sec = QComboBox()
        for s in nho09.INTERVAL_CHOICES_S:
            self.interval_sec.addItem(f'{s} s' if s < 60 else
                                      f'{s // 60} min', s)
        self.interval_sec.setCurrentText('10 s')
        g.addWidget(self.interval_sec, 6, 1)
        self.modal_fit = QCheckBox('Modal fit (half-power + SDOF)')
        self.modal_fit.setToolTip(
            'H1 / transmissibility view only. Finds the highest peak in '
            '0.5–80 Hz, estimates fn and zeta by the half-power method, '
            'then refines with a least-squares SDOF fit weighted by '
            'coherence. Overlays the model curve and shows fn / zeta.')
        g.addWidget(self.modal_fit, 6, 2, 1, 2)
        self.spine_tf_fit = QCheckBox('Spine TF fit (Formula 1 structure)')
        self.spine_tf_fit.setToolTip(
            'H1 view only. Fits a function with one zero pair and six pole '
            'pairs (Formula 1 structure of ISO 2631-5, DC gain = 1) to the '
            'measured H1 by complex least-squares. The fitted model can be '
            'used as spine_response_fn in Clause 5 assessment. '
            'Starting point: ISO 2631-5 Table (31/34 rad/s etc.).')
        g.addWidget(self.spine_tf_fit, 7, 0, 1, 2)
        self.annotate_peaks = QCheckBox('Annotate H1 / SRS peak (fn, |H|, Q, ζ)')
        self.annotate_peaks.setToolTip(
            'H1 / transmissibility view: marks the highest peak in the '
            'coherent band with fn, |H|max, Q and ζ estimated by the '
            '3 dB half-power bandwidth.\n'
            'SRS view: marks the frequency and value of the maximum.')
        g.addWidget(self.annotate_peaks, 7, 2, 1, 2)

        self.note = QLabel('')
        self.note.setObjectName('muted')
        self.note.setWordWrap(True)
        g.addWidget(self.note, 8, 0, 1, 4)

        # 5.2 — HP filter order check for time-domain integration
        _lbl_int_hp = QLabel('Integration HP order check:')
        _lbl_int_hp.setToolTip(
            'Diagnostic tool for processing pipelines that need velocity or '
            'displacement from an accelerometer by time-domain integration.\n\n'
            'Problem: a low-frequency offset in the raw signal grows without '
            'bound when you integrate numerically. A high-pass Butterworth '
            'filter removes it, but if the filter order is too low relative '
            'to the number of integrations, the residual drift still dominates '
            'the result. Rule of thumb: HP order ≥ n + 1 where n = '
            'number of integrations (1 for velocity, 2 for displacement).\n\n'
            'This widget checks that rule and shows the minimum order needed.')
        g.addWidget(_lbl_int_hp, 9, 0)
        self.int_hp_order = QSpinBox()
        self.int_hp_order.setRange(1, 12)
        self.int_hp_order.setValue(2)
        self.int_hp_order.setToolTip(
            'Order of the high-pass Butterworth pre-filter you plan to use '
            'before time-domain integration (cumulative trapz or similar).')
        g.addWidget(self.int_hp_order, 9, 1)
        self.int_hp_n = QComboBox()
        self.int_hp_n.addItem('— select —', 0)
        self.int_hp_n.addItem('Single integration (accel → velocity)', 1)
        self.int_hp_n.addItem('Double integration (accel → displacement)', 2)
        self.int_hp_n.setToolTip(
            'Number of time-domain integrations planned after HP filtering. '
            'Rule: order ≥ n + 1 (ISO 2631-1 / signal-processing '
            'best practice).')
        g.addWidget(self.int_hp_n, 9, 2)
        self.int_hp_result = QLabel('')
        g.addWidget(self.int_hp_result, 9, 3)
        self.int_hp_order.valueChanged.connect(self._check_int_hp_order)
        self.int_hp_n.currentIndexChanged.connect(self._check_int_hp_order)

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        row.addWidget(ctrl, 1)
        lay.addLayout(row)

        self.weight = WeightingPanel()
        self.weight.setMaximumHeight(220)
        self.weight.changed.connect(self._on_weighting)
        lay.addWidget(self.weight)

        for w in (self.kind, self.axis, self.method, self.window,
                  self.nperseg, self.overlap, self.interval_sec):
            w.currentIndexChanged.connect(self.refresh)
        self.band.valueChanged.connect(self.refresh)
        self.logy.stateChanged.connect(self.refresh)
        self.hp.stateChanged.connect(self.refresh)
        self.waterfall.stateChanged.connect(self.refresh)
        self.align.stateChanged.connect(self.refresh)
        self.reject.stateChanged.connect(self.refresh)
        self.full_scale.valueChanged.connect(self.refresh)
        self.hp_fc.valueChanged.connect(self.refresh)
        self.modal_fit.stateChanged.connect(self.refresh)
        self.spine_tf_fit.stateChanged.connect(self.refresh)
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
        b_meta = QPushButton('Edit metadata...')
        b_meta.setObjectName('secondary')
        b_meta.clicked.connect(self.edit_metadata)
        bar.addWidget(b_meta)
        b_cal = QPushButton('Calibrate...')
        b_cal.setObjectName('secondary')
        b_cal.clicked.connect(self._open_calibration_dialog)
        bar.addWidget(b_cal)
        self.export_iv_btn = QPushButton('Export intervals to CSV...')
        self.export_iv_btn.setObjectName('secondary')
        self.export_iv_btn.clicked.connect(self.export_iv_csv)
        bar.addWidget(self.export_iv_btn)
        b_join = QPushButton('Join selected → 1 signal')
        b_join.setObjectName('secondary')
        b_join.setToolTip(
            'Concatenates all checked records in list order into a single '
            'Excitation and adds it to the list. Use this when you recorded '
            'a long session as several consecutive files.')
        b_join.clicked.connect(self.join_selected)
        bar.addWidget(b_join)
        b3 = QPushButton('Export spectrum to CSV...')
        b3.setObjectName('secondary')
        b3.clicked.connect(self.export_csv)
        bar.addWidget(b3)
        b_pdf = QPushButton('Export report PDF...')
        b_pdf.setObjectName('secondary')
        b_pdf.setToolTip(
            'Save a PDF with the current plot and a statistics/metadata table.')
        b_pdf.clicked.connect(self.export_report_pdf)
        bar.addWidget(b_pdf)
        lay.addLayout(bar)

    # ------------------------------------------------------------------
    def add_files(self):
        if getattr(self, '_busy_flag', False):
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add signal records', '',
            'Signal records (*.txt *.asc *.dat *.csv *.m);;All files (*)')
        paths = [p for p in paths if Path(p).name not in self.signals]
        if not paths:
            return
        csv_rate = None
        try:
            if any(fd.detect_format(str(p)) == 'csv_ms' for p in paths):
                from PySide6.QtWidgets import QInputDialog
                csv_rate, ok = QInputDialog.getDouble(
                    self, 'Sample rate',
                    'Time_ms CSV: sample rate configured in firmware [Hz]:',
                    100.0, 1.0, 20000.0, 1)
                if not ok:
                    return
        except (OSError, ValueError):
            pass
        config = self.weight.config()

        # Leitura e metricas em segundo plano: registros longos levam
        # segundos cada, e a janela congelava ate o ultimo terminar.
        def job(report):
            out, failed = [], []
            for i, p in enumerate(paths):
                name = Path(p).name
                report(i / len(paths), f'Reading {name}...')
                exc = None
                try:
                    exc = core.read_excitation(p, name)
                except Exception:
                    pass
                if exc is None:
                    try:
                        recs = list(fd.read_records(p, csv_rate=csv_rate))
                        if not recs:
                            raise ValueError('no records found')
                        rec = recs[0]
                        fd.set_orientation(rec, rec.gravity_axis)
                        exc = _field_to_excitation(rec, name)
                    except Exception as err:
                        failed.append(f'{name}: {err}')
                        continue
                try:
                    st = core.signal_statistics(exc)
                    st['w'] = core.weighted_metrics(exc, config)
                    st['band_peak_z'] = _band_peak_z(exc)
                    st['meta'] = dict(exc.metadata)
                    st['fs'] = 1.0 / exc.dt
                    st['weighting_config'] = dict(config)
                except Exception as err:
                    failed.append(f'{name}: {err}')
                    continue
                out.append((name, p, exc, st))
            return out, failed

        self._busy_flag = True
        self.add_btn.setEnabled(False)
        self.add_btn.setText('Reading…')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        n_total = len(paths)

        def _progress(f, m):
            done = max(1, round(f * n_total))
            self.add_btn.setText(f'Reading {done}/{n_total}…')
            self.status(m)

        run_in_thread(self, job, self._on_added, self._on_add_fail,
                      on_progress=_progress,
                      thread_attr='_add_thread', worker_attr='_add_worker')

    def _end_add(self):
        thread = getattr(self, '_add_thread', None)
        if thread is not None:
            thread.quit()
            thread.wait()
        self._busy_flag = False
        self.add_btn.setEnabled(True)
        self.add_btn.setText('Add files...')
        QApplication.restoreOverrideCursor()

    def _on_add_fail(self, tb):
        self._end_add()
        self.status('')
        QMessageBox.critical(self, 'Could not read the files',
                             tb.strip().splitlines()[-1])

    def _on_added(self, payload):
        self._end_add()
        loaded, failed = payload
        for name, p, exc, st in loaded:
            if name in self.signals:
                continue
            self.signals[name] = exc
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
    def _check_int_hp_order(self):
        n = self.int_hp_n.currentData()
        if not n:
            self.int_hp_result.setText('')
            self.int_hp_result.setStyleSheet('')
            return
        r = core.integration_hp_order_check(self.int_hp_order.value(), n)
        if r['valid']:
            self.int_hp_result.setText(f'✓ OK (min {r["minimum"]})')
            self.int_hp_result.setStyleSheet('color: #2d7a4a; font-weight: bold;')
        else:
            self.int_hp_result.setText(f'✗ Need ≥ {r["minimum"]}')
            self.int_hp_result.setStyleSheet('color: #c0392b; font-weight: bold;')

    # ------------------------------------------------------------------
    def _on_weighting(self):
        cfg = self.weight.config()
        for name, exc in self.signals.items():
            self.stats[name]['w'] = core.weighted_metrics(exc, cfg)
            self.stats[name]['weighting_config'] = dict(cfg)
        self.refresh()
        self.refresh_stats()

    def _sig(self, exc, axis):
        x = exc.accel[:, axis]
        if self.hp.isChecked():
            x = core.highpass_butterworth(x, exc.dt, self.hp_fc.value())
        return x

    def _pair(self, ref, exc, axis):
        """(x, y, texto) do par referencia/registro, ou (None, None,
        motivo) quando nao da para comparar. Com "Align" ligado, alinha
        pela correlacao cruzada (core.align_by_correlation) e o texto e' o
        atraso medido."""
        if exc.dt != ref.dt:
            return None, None, 'different sample rate', None
        x, y = self._sig(ref, axis), self._sig(exc, axis)
        if not self.align.isChecked():
            if len(x) != len(y):
                return None, None, 'different length; try aligning', None
            return x, y, '', (0, 0)
        try:
            x, y, info = core.align_by_correlation(x, y, ref.dt)
        except core.AlignmentError as e:
            return None, None, str(e), None
        k = info['shift']
        return x, y, (f'{info["lag"] * 1000:+.0f} ms '
                      f'(rho {info["rho"]:.2f})'), (max(0, -k), max(0, k))

    def _segments(self, ref, exc, axis, xs, ys, offs):
        """Espectros por segmento do par e a mascara de segmentos aceitos
        (None = todos). Devolve (seg, flags ou None)."""
        nperseg = self.nperseg.currentData()
        overlap = self.overlap.currentData()
        seg = core.segment_cross_spectra(xs, ys, ref.dt,
                                         self.window.currentText(), nperseg,
                                         overlap)
        if not self.reject.isChecked():
            return seg, None
        clip = None
        fs_g = self.full_scale.value()
        if fs_g > 0:
            n = len(xs)
            nps = seg['n']
            ox, oy = offs
            lim = fs_g * 9.80665
            clip = (core.clipped_segments(ref.accel[ox:ox + n, axis], lim, nps,
                                          overlap) |
                    core.clipped_segments(exc.accel[oy:oy + n, axis], lim, nps,
                                          overlap))
        return seg, core.segment_flags(seg, clipped=clip)

    def _pair_note(self, skipped, aligned, rejected=()):
        note = ''
        if rejected:
            note += 'Rejected before averaging: ' + '; '.join(rejected) + '. '
        if skipped:
            note += 'Skipped: ' + '; '.join(skipped) + '. '
        if aligned:
            note += ('Aligned by cross-correlation (positive = record '
                     'lags the reference): ' + ', '.join(aligned) + '. ')
        return note

    def refresh(self):
        kind = self.kind.currentIndex()
        # 0 time series, 1 spectrum, 2 cumulative RMS, 3 phase, 4 coherence,
        # 5 SEAT transmissibility.
        self.plot.reset_if_changed(kind)
        spectrum_kind = kind == 1
        phase_kind = kind == 3
        coherence_kind = kind == 4
        seat_kind = kind == 5
        class_kind = kind == 9
        h1_kind = kind in (6, 9)
        spg_kind = kind == 7
        srs_kind = kind == 8
        iv_kind = kind == 10
        hist_kind = kind == 11
        ods_kind = kind == 12
        mcoh_kind = kind == 13
        filt_kind = kind == 14
        oct3_kind = kind == 15
        frf_kind = h1_kind or ods_kind or mcoh_kind
        no_welch = self.method.currentText() in (
            core.SPECTRUM_METHODS[0], core.SPECTRUM_METHODS[5])
        self.method.setEnabled(spectrum_kind)
        self.band.setEnabled(
            spectrum_kind and self.method.currentText() == core.SPECTRUM_METHODS[3])
        for w in (self.window, self.nperseg, self.overlap):
            w.setEnabled((spectrum_kind and not no_welch) or coherence_kind
                         or frf_kind or spg_kind)
        self.hp.setEnabled(coherence_kind or frf_kind or spg_kind)
        self.align.setEnabled(coherence_kind or frf_kind)
        self.reject.setEnabled(coherence_kind or h1_kind)
        self.full_scale.setEnabled((coherence_kind or h1_kind)
                                   and self.reject.isChecked())
        self.hp_fc.setEnabled((coherence_kind or frf_kind or spg_kind)
                              and self.hp.isChecked())
        self.waterfall.setEnabled(spg_kind)
        self.modal_fit.setEnabled(h1_kind and not class_kind)
        self.spine_tf_fit.setEnabled(h1_kind and not class_kind)
        for w in (self.iv_label, self.interval_sec):
            w.setVisible(iv_kind)
        self.export_iv_btn.setVisible(iv_kind)

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
            min_navg, skipped, aligned, rejected = None, [], [], []
            for i, name in enumerate(names[1:], start=1):
                exc = self.signals[name]
                xs, ys, why, offs = self._pair(ref, exc, axis)
                if xs is None:
                    skipped.append(f'{name} ({why})')
                    continue
                if why:
                    aligned.append(f'{name} {why}')
                if self.reject.isChecked():
                    seg, fl = self._segments(ref, exc, axis, xs, ys, offs)
                    est, navg = core.frf_from_segments(seg, ~fl['rejected'])
                    f, gamma2 = seg['f'], est['g2']
                    if core.flag_summary(fl):
                        rejected.append(f'{name}: {core.flag_summary(fl)}')
                else:
                    f, gamma2, navg = core.coherence_spectrum(
                        xs, ys, ref.dt,
                        window=self.window.currentText(),
                        nperseg=self.nperseg.currentData(),
                        overlap=self.overlap.currentData())
                min_navg = navg if min_navg is None else min(min_navg, navg)
                ax.plot(f, gamma2, lw=1.1,
                        color=SERIES_COLORS[i % len(SERIES_COLORS)],
                        label=f'{ref_name} vs {name}')
            if min_navg is not None and min_navg >= 2:
                cl = core.coherence_confidence_limit(min_navg)
                ax.axhline(cl, color='#c0392b', lw=1.0, ls='--',
                           label=f'95% confidence limit ({cl:.2f})')
            ax.set_xlim(0, core.LOWPASS_HZ)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(r'Coherence $\gamma^2$')
            ax.set_title(f'Coherence vs {ref_name} - {core.AXIS_NAMES[axis]}')
            note = self._pair_note(skipped, aligned, rejected)
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

        if iv_kind:
            self._draw_interval_values(names)
            return

        if hist_kind:
            self._draw_histogram(names, axis)
            return

        if ods_kind:
            self._draw_ods(names, axis)
            return

        if mcoh_kind:
            self._draw_multiple_coherence(names, axis)
            return

        if filt_kind:
            self._draw_filter_response()
            return

        if oct3_kind:
            self._draw_h1_third_octave(names, axis)
            return

        if spg_kind or srs_kind:
            if axis is None:
                freqplots.message(self.plot, 'Pick a single axis (x, y or z).')
                self.note.setText('')
                return
            if srs_kind:
                series = []
                for name in names:
                    exc = self.signals[name]
                    r = core.shock_response_spectra(
                        exc.accel[:, axis], exc.dt)
                    series.append((name, r['f'], r))
                freqplots.draw_srs(
                    self.plot, series,
                    f'Shock Response Spectrum (Q=10) - {core.AXIS_NAMES[axis]}',
                    self.logy.isChecked(),
                    show_primary_residual=True)
                if self.annotate_peaks.isChecked() and series:
                    ax_srs, = self.plot.axes()
                    for _sn, _fr, _r in series:
                        _srs = (_r['maximax'] if isinstance(_r, dict)
                                else _r)
                        freqplots.annotate_srs_peak(ax_srs, _fr, _srs)
                self.note.setText(
                    freqplots.SRS_NOTE +
                    ' Dashed: primary (during shock). '
                    'Dotted: residual (free vibration after). '
                    'Solid: maximax.')
                return
            name = names[0]
            exc = self.signals[name]
            t, f, S = core.spectrogram(
                self._sig(exc, axis), exc.dt,
                window=self.window.currentText(),
                nperseg=self.nperseg.currentData(),
                overlap=self.overlap.currentData())
            freqplots.draw_spectrogram(
                self.plot, t, f, S,
                f'Spectrogram - {name} ({core.AXIS_NAMES[axis]})',
                waterfall=self.waterfall.isChecked(),
                logy=self.logy.isChecked())
            self.note.setText(
                f'Shows the first checked record only ({name}). '
                f'{self.nperseg.currentData()}-sample segments, '
                f'{self.window.currentText()} window.')
            return

        if h1_kind:
            if axis is None or len(names) < 2:
                msg = ('Pick a single axis (x, y or z).' if axis is None else
                       'Check at least two records: the first checked one '
                       'is the reference (input).')
                ax.text(0.5, 0.5, msg, ha='center', va='center',
                        color='#6c757d', wrap=True, transform=ax.transAxes)
                self.plot.finish(ax)
                self.note.setText('')
                return
            _modal_on = self.modal_fit.isChecked() and not class_kind
            if _modal_on:
                ax1, ax2, ax3 = self.plot.axes(3)
            else:
                ax1, ax2 = self.plot.axes(2)
                ax3 = None
            ref_name = names[0]
            ref = self.signals[ref_name]
            min_navg, skipped, aligned, rejected = None, [], [], []
            others = names[1:2] if class_kind else names[1:]
            for i, name in enumerate(others, start=1):
                exc = self.signals[name]
                xs, ys, why, offs = self._pair(ref, exc, axis)
                if xs is None:
                    skipped.append(f'{name} ({why})')
                    continue
                if why:
                    aligned.append(f'{name} {why}')
                if self.reject.isChecked() or class_kind:
                    seg, fl = self._segments(ref, exc, axis, xs, ys, offs)
                    mask = None if fl is None else ~fl['rejected']
                    if fl is not None and core.flag_summary(fl):
                        rejected.append(f'{name}: {core.flag_summary(fl)}')
                    f = seg['f']
                    if class_kind:
                        classes = core.amplitude_classes(seg, mask=mask)
                        for j, (lo, hi, navg, est) in enumerate(classes):
                            min_navg = (navg if min_navg is None
                                        else min(min_navg, navg))
                            freqplots.draw_transmissibility(
                                ax1, ax2, f, est,
                                core.coherence_confidence_limit(navg),
                                SERIES_COLORS[j % len(SERIES_COLORS)],
                                f'{lo:g}-{hi:g} m/s\u00b2 rms ({navg})',
                                self.plot.estimators)
                        if not classes:
                            skipped.append(
                                f'{name} (no input-amplitude class has '
                                f'{core.MIN_CLASS_AVERAGES} segments)')
                        continue
                    est, navg = core.frf_from_segments(seg, mask)
                else:
                    f, est, navg = core.transmissibility_estimates(
                        xs, ys, ref.dt,
                        window=self.window.currentText(),
                        nperseg=self.nperseg.currentData(),
                        overlap=self.overlap.currentData())
                min_navg = navg if min_navg is None else min(min_navg, navg)
                freqplots.draw_transmissibility(
                    ax1, ax2, f, est, core.coherence_confidence_limit(navg),
                    SERIES_COLORS[i % len(SERIES_COLORS)], name,
                    self.plot.estimators)
            if min_navg is not None and min_navg >= 2:
                ax2.axhline(core.coherence_confidence_limit(min_navg),
                            color='#c0392b', lw=1.0, ls='--',
                            label='95% confidence limit')
            ax1.axhline(1.0, color='#6c757d', lw=0.6, ls=':')
            ax1.set_xlim(0, core.LOWPASS_HZ)
            ax2.set_xlim(0, core.LOWPASS_HZ)
            ax2.set_ylim(0, 1.05)
            if ax3 is not None:
                ax3.set_xlim(0, core.LOWPASS_HZ)
                ax3.axhline(1.0, color='#6c757d', lw=0.6, ls=':')
                ax3.set_ylabel('|H1| / |H_model|')
                ax3.set_xlabel('Frequency [Hz]')
            if self.logy.isChecked():
                ax1.set_yscale('log')
            references.draw_references(ax1, self.plot.references,
                                       core.LOWPASS_HZ)
            ax1.set_ylabel(f'|H1| (record / {ref_name})')
            ax1.set_title(f'Transmissibility H1 from {ref_name}'
                          + (f' to {others[0]}, by input amplitude'
                             if class_kind and others else '')
                          + f' - {core.AXIS_NAMES[axis]}')
            _coh_xlabel = '' if ax3 is not None else 'Frequency [Hz]'
            ax2.set_xlabel(_coh_xlabel)
            ax2.set_ylabel(r'Coherence $\gamma^2$')
            for a in (ax1, ax2) + ((ax3,) if ax3 is not None else ()):
                if a.get_legend_handles_labels()[0]:
                    a.legend(fontsize=8)
            note = self._pair_note(skipped, aligned, rejected)
            if class_kind:
                note += (f'One H1 per class of input RMS (segment RMS of '
                         f'{ref_name}, unweighted; classes need '
                         f'{core.MIN_CLASS_AVERAGES} segments; only the first '
                         'record after the reference is used). The human '
                         'body is not linear: the resonance moves down in '
                         'frequency and magnitude as the amplitude rises. ')
            if min_navg is None:
                note += 'No comparable record left to plot.'
            elif min_navg < 4:
                note += (f'Only {min_navg} averaged segment(s): H1 and '
                         'coherence are not yet meaningful. Shorten the '
                         'segment length.')
            else:
                note += (f'H1 = Gxy/Gxx from {min_navg} averaged Welch '
                         'segments. Solid line: coherence above the 95% '
                         'confidence limit; faint: below it. The records '
                         'must be synchronous (same run, same start '
                         'instant); this explorer does not verify that.')
                extra = freqplots.estimator_note(self.plot.estimators)
                if extra:
                    note += ' ' + extra
            if (_modal_on and min_navg is not None):
                modal_note = self._draw_modal_overlay(
                    ax1, ax3, names, ref, axis)
                if modal_note:
                    note += ' ' + modal_note
            if (self.spine_tf_fit.isChecked() and not class_kind
                    and min_navg is not None):
                spine_note = self._draw_spine_tf_overlay(ax1, names, ref, axis)
                if spine_note:
                    note += ' ' + spine_note
            if (self.annotate_peaks.isChecked() and not class_kind
                    and min_navg is not None):
                # Coleta o H1 de todos os pares desenhados e anota o pico
                # mais alto no eixo principal.  Recalcula apenas as
                # estimativas; nao refaz o grafico.
                for _i, _name in enumerate(names[1:], start=1):
                    _exc = self.signals.get(_name)
                    if _exc is None:
                        continue
                    _xs, _ys, _why, _offs = self._pair(ref, _exc, axis)
                    if _xs is None:
                        continue
                    _f, _est, _navg = core.transmissibility_estimates(
                        _xs, _ys, ref.dt,
                        window=self.window.currentText(),
                        nperseg=self.nperseg.currentData(),
                        overlap=self.overlap.currentData())
                    _cl = core.coherence_confidence_limit(_navg)
                    freqplots.annotate_transmissibility_peak(
                        ax1, _f, _est['h1'], _est['g2'], _cl,
                        color=SERIES_COLORS[_i % len(SERIES_COLORS)])
            self.note.setText(note)
            self.plot.finish(ax1, ax2)
            return

        if seat_kind:
            if len(names) < 2:
                ax.text(0.5, 0.5, 'Check at least two records: the first '
                        'checked one is the floor (reference), the others '
                        'are the seat pan.', ha='center', va='center',
                        color='#6c757d', wrap=True, transform=ax.transAxes)
                self.plot.finish(ax)
                self.note.setText('')
                return
            ref_name = names[0]
            ref_w = self.stats[ref_name]['w']
            labels, rms_vals, vdv_vals = [], [], []
            for name in names[1:]:
                w = self.stats[name]['w']
                if axis is None:
                    ref_rms, rms = ref_w['aw_total'], w['aw_total']
                    ref_vdv, vdv = ref_w['vdv_total'], w['vdv_total']
                else:
                    ref_rms, rms = ref_w['aw'][axis], w['aw'][axis]
                    ref_vdv, vdv = ref_w['vdv'][axis], w['vdv'][axis]
                labels.append(name)
                rms_vals.append(rms / ref_rms if ref_rms > 0 else np.nan)
                vdv_vals.append(vdv / ref_vdv if ref_vdv > 0 else np.nan)
            x_pos = np.arange(len(labels))
            width = 0.35
            ax.bar(x_pos - width / 2, rms_vals, width, label='SEAT (RMS)',
                  color=SERIES_COLORS[0])
            ax.bar(x_pos + width / 2, vdv_vals, width, label='SEAT (VDV)',
                  color=SERIES_COLORS[1])
            ax.axhline(1.0, ls='--', lw=1, color='#6c757d')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('SEAT (ratio, dimensionless)')
            axis_tag = f' ({core.AXIS_NAMES[axis]})' if axis is not None else \
                ' (vector sum)'
            ax.set_title(f'Seat Effective Amplitude Transmissibility vs '
                        f'{ref_name}{axis_tag}')
            self.note.setText(
                'SEAT = a_w,seat / a_w,floor (or the same ratio using VDV). '
                'Below 1 (100%): the seat attenuates vibration and improves '
                'ride comfort. Above 1: the seat amplifies it. The first '
                'checked record is treated as the floor (reference); use '
                'RMS when the vibration is smooth/continuous and VDV when '
                'it has shocks or transients (ISO 2631-1 guidance).')
            ax.legend(fontsize=8)
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
                bands = self.method.currentText() in core.SPECTRUM_METHODS[3:5]
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
    def _draw_interval_values(self, names):
        """Grafico dual: a_w/MTVV (cima) e VDV por eixo (baixo) por intervalo."""
        sec = self.interval_sec.currentData()
        ax1, ax2 = self.plot.axes(2)

        for i, name in enumerate(names):
            exc = self.signals[name]
            blocks = exc.accel[np.newaxis]
            m = nho09.interval_metrics(blocks, exc.dt, sec)
            if len(m['start']) == 0:
                continue
            t = m['start']
            dur = m['duration']
            col = SERIES_COLORS[i % len(SERIES_COLORS)]

            # Constroi arrays passo para step plots: cada intervalo como
            # barra horizontal de largura duration
            t_step = np.repeat(t, 2)
            t_step[1::2] = t + dur
            amr_step = np.repeat(m['amr'], 2)
            mtvv_step = np.repeat(m['mtvv'], 2)
            vdv_step = np.repeat(m['vdv'], 2, axis=0)

            ax1.plot(t_step, amr_step, lw=1.3, color=col,
                     label=f'{name}: a_w')
            ax1.plot(t_step, mtvv_step, lw=1.0, ls='--', color=col,
                     alpha=0.7, label=f'{name}: MTVV')
            ax2.plot(t_step, vdv_step[:, 0], lw=1.0, ls='-', color=col,
                     label=f'{name}: VDV x')
            ax2.plot(t_step, vdv_step[:, 1], lw=1.0, ls='--', color=col,
                     label=f'{name}: VDV y')
            ax2.plot(t_step, vdv_step[:, 2], lw=1.0, ls=':', color=col,
                     label=f'{name}: VDV z')

        ax1.set_ylabel('a_w / MTVV [m/s²]')
        ax1.set_title(f'Interval values — {sec} s intervals '
                      f'(Wd/Wd/Wk, k=1.4/1.4/1.0)')
        ax1.legend(fontsize=8, framealpha=0.85)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('VDV [m/s¹·⁷⁵]')
        ax2.legend(fontsize=8, framealpha=0.85)
        self.note.setText(
            'a_w: RMS of frequency-weighted acceleration per interval '
            '(ISO 2631-1 Wd/Wd/Wk with k=1.4/1.4/1.0). '
            'MTVV: peak of 1 s running RMS within the interval. '
            'VDV: vibration dose per interval per axis.')
        self.plot.finish(ax1, ax2)

    def _draw_histogram(self, names, axis):
        """Histograma de densidade + CDF da aceleracao por eixo."""
        ax1, ax2 = self.plot.axes(2)
        _g = 9.80665
        thresholds_g = (0.5, 1.0, 2.0)

        for i, name in enumerate(names):
            exc = self.signals[name]
            col = SERIES_COLORS[i % len(SERIES_COLORS)]
            a = (np.linalg.norm(exc.accel, axis=1) if axis is None
                 else np.abs(exc.accel[:, axis]))
            bins = np.linspace(0, min(a.max(), 5 * _g), 120)
            counts, edges = np.histogram(a, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax1.plot(centers / _g, counts * _g, lw=1.2, color=col, label=name)
            cdf = np.cumsum(counts) * (edges[1] - edges[0])
            ax2.plot(centers / _g, cdf, lw=1.2, color=col, label=name)

        ax1.set_xlabel('|a| [g]')
        ax1.set_ylabel('Density [1/g]')
        axis_tag = 'vector' if axis is None else core.AXIS_NAMES[axis]
        ax1.set_title(f'Acceleration distribution — {axis_tag}')
        for tg in thresholds_g:
            ax1.axvline(tg, lw=0.7, ls=':', color='#c0392b', alpha=0.6)
            ax2.axvline(tg, lw=0.7, ls=':', color='#c0392b', alpha=0.6)
        ax2.set_xlabel('|a| [g]')
        ax2.set_ylabel('Cumulative probability')
        ax2.set_ylim(0, 1.05)
        if names:
            ax1.legend(fontsize=8)
        self.note.setText(
            'Histogram of |a| (100-bin density) and its CDF. '
            'Dotted lines at 0.5 g, 1 g, 2 g. '
            'Clipped at 5 g for readability.')
        self.plot.finish(ax1, ax2)

    def _draw_modal_overlay(self, ax1, ax3, names, ref, axis):
        """Sobreposicao de ajuste modal no grafico de transmissibilidade H1.

        Para cada sinal (exceto a referencia) calcula H1, extrai parametros
        pelo metodo de meia potencia e refina com ajuste SDOF.  Sobrepoe a
        curva do modelo em ax1 e o residuo |H1|/|H_model| em ax3 (pode ser
        None).  Devolve string de resumo para a nota.
        """
        ref_name = names[0]
        FMAX_MODAL = core.LOWPASS_HZ
        summaries = []
        for i, name in enumerate(names[1:], start=1):
            exc = self.signals[name]
            xs, ys, why, _ = self._pair(ref, exc, axis)
            if xs is None:
                continue
            try:
                f, est, navg = core.transmissibility_estimates(
                    xs, ys, ref.dt,
                    window=self.window.currentText(),
                    nperseg=self.nperseg.currentData(),
                    overlap=self.overlap.currentData())
            except Exception:
                continue
            col = SERIES_COLORS[i % len(SERIES_COLORS)]
            h1 = est['h1']
            g2 = est['g2']
            # Meia potencia
            mp = core.modal_parameters(f, h1, fmin=0.5, fmax=FMAX_MODAL,
                                        g2=g2)
            if not mp.get('valid'):
                continue
            fn0, z0 = mp['fn'], mp.get('zeta') or 0.05
            # Ajuste SDOF refinado
            flo_fit = max(0.5, fn0 * 0.5)
            fhi_fit = min(FMAX_MODAL, fn0 * 2.0)
            sd = core.modal_sdof_fit(f, h1, fn0, z0, g2=g2,
                                     fmin=flo_fit, fmax=fhi_fit)
            fn_use = sd['fn'] if sd.get('valid') else fn0
            z_use = sd['zeta'] if sd.get('valid') else z0
            Q_use = sd.get('Q') or (1 / (2 * z_use))
            method = sd.get('method', 'half-power') if sd.get('valid') else 'half-power'
            r2_str = (f', R²={sd["r2"]:.3f}'
                      if sd.get('valid') and sd.get('r2') is not None else '')
            lbl = (f'{name}: fn={fn_use:.2f} Hz, '
                   f'ζ={z_use:.4f}, Q={Q_use:.1f}{r2_str} ({method})')
            # Curva do modelo sobre toda a faixa
            H_model = sd['H_model'] if sd.get('valid') else None
            if H_model is not None:
                ax1.plot(f, np.abs(H_model), lw=1.8, ls='--',
                         color=col, alpha=0.85, label=lbl, zorder=5)
                # Residuo: razao |H1_medido| / |H_modelo|
                if ax3 is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.abs(h1) / np.where(
                            np.abs(H_model) > 1e-6, np.abs(H_model), np.nan)
                    ax3.plot(f, ratio, lw=1.0, color=col,
                             label=f'{name} residual', alpha=0.8)
                summaries.append(
                    f'{name}: fn={fn_use:.2f} Hz, ζ={z_use:.4f}, '
                    f'Q={Q_use:.1f}{r2_str} (fit {flo_fit:.1f}–{fhi_fit:.1f} Hz, '
                    f'{method})'
                )
            else:
                # So marca o pico
                ax1.axvline(fn_use, lw=1.2, ls=':', color=col,
                            alpha=0.7, label=lbl)
                summaries.append(
                    f'{name}: fn={fn_use:.2f} Hz (half-power only; SDOF fit failed)'
                )
            # Banda de meia potencia
            if mp.get('f_lo') and mp.get('f_hi'):
                ax1.axvspan(mp['f_lo'], mp['f_hi'], alpha=0.07,
                            color=col, lw=0)
        if ax1.get_legend_handles_labels()[0]:
            ax1.legend(fontsize=8, framealpha=0.9)
        if ax3 is not None and ax3.get_legend_handles_labels()[0]:
            ax3.legend(fontsize=8)
        if not summaries:
            return ''
        return ('Modal fit — ' + '; '.join(summaries)
                + '. Dashed line: SDOF model. '
                  'Bottom panel: |H1|/|H_model| (1 = perfect fit).')

    def _draw_spine_tf_overlay(self, ax1, names, ref, axis):
        """Ajusta a funcao de transferencia Formula (1) ao H1 medido e sobrepoe.

        Executa fit_spine_tf para cada par (referencia → sinal), sobrepoe a
        curva do modelo em ax1 com linha tracejada mais grossa e devolve
        string de resumo com R², zero e 2 polos mais proximos da ressonancia.
        """
        ref_name = names[0]
        summaries = []
        for i, name in enumerate(names[1:], start=1):
            exc = self.signals[name]
            xs, ys, why, _ = self._pair(ref, exc, axis)
            if xs is None:
                continue
            try:
                f, est, navg = core.transmissibility_estimates(
                    xs, ys, ref.dt,
                    window=self.window.currentText(),
                    nperseg=self.nperseg.currentData(),
                    overlap=self.overlap.currentData())
            except Exception:
                continue
            col = SERIES_COLORS[i % len(SERIES_COLORS)]
            h1 = est['h1']
            g2 = est['g2']
            sd = core.fit_spine_tf(f, h1, g2=g2,
                                   fmin=0.5, fmax=core.LOWPASS_HZ)
            if not sd.get('valid'):
                summaries.append(f'{name}: Spine TF fit did not converge.')
                continue
            H_model = sd['H_model']
            ax1.plot(f, np.abs(H_model), lw=2.2, ls=':',
                     color=col, alpha=0.9,
                     label=f'{name}: Spine TF fit (R²={sd["r2"]:.3f})',
                     zorder=6)
            # Resumo: zero e polo mais baixo (mais relevante para WBV)
            z_hz = sd['zero'][0] / (2 * np.pi)
            p_hz = sorted(p[0] / (2 * np.pi) for p in sd['poles'])
            summaries.append(
                f'{name}: zero={z_hz:.1f} Hz (ζ={sd["zero"][1]:.2f}), '
                f'poles={p_hz[0]:.1f}/{p_hz[1]:.1f} Hz..., '
                f'R²={sd["r2"]:.3f}'
            )
        if ax1.get_legend_handles_labels()[0]:
            ax1.legend(fontsize=8, framealpha=0.9)
        if not summaries:
            return ''
        return ('Spine TF fit (Formula 1 structure, 1 zero + 6 poles, '
                'DC=1) — ' + '; '.join(summaries)
                + '. Dotted line: fitted model. '
                  'Labeled as lab estimate — not a normative result.')

    def _draw_ods(self, names, axis):
        """ODS (Operational Deflection Shape): |H1| e fase de H1 em Bode.

        Sem entrada controlada — a excitacao e o proprio sinal de campo.
        O primeiro sinal marcado e a referencia; os demais sao as respostas.
        """
        if axis is None or len(names) < 2:
            ax, = self.plot.axes()
            msg = ('Pick a single axis (x, y or z).' if axis is None else
                   'Check at least two records: the first is the reference.')
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    color='#6c757d', wrap=True, transform=ax.transAxes)
            self.plot.finish(ax)
            self.note.setText('')
            return
        ref_name = names[0]
        ref = self.signals[ref_name]
        ax1, ax2 = self.plot.axes(2)
        min_navg, skipped, aligned = None, [], []
        for i, name in enumerate(names[1:], start=1):
            exc = self.signals[name]
            xs, ys, why, _ = self._pair(ref, exc, axis)
            if xs is None:
                skipped.append(f'{name} ({why})')
                continue
            if why:
                aligned.append(f'{name} {why}')
            try:
                f, est, navg = core.transmissibility_estimates(
                    xs, ys, ref.dt,
                    window=self.window.currentText(),
                    nperseg=self.nperseg.currentData(),
                    overlap=self.overlap.currentData())
            except Exception as e:
                skipped.append(f'{name} ({e})')
                continue
            min_navg = navg if min_navg is None else min(min_navg, navg)
            col = SERIES_COLORS[i % len(SERIES_COLORS)]
            cl = core.coherence_confidence_limit(navg)
            good = est['g2'] >= cl
            mag = np.abs(est['h1'])
            phase_deg = np.angle(est['h1'], deg=True)
            ax1.plot(f, mag, lw=0.7, color=col, alpha=0.25)
            ax1.plot(f, np.where(good, mag, np.nan), lw=1.3,
                     color=col, label=name)
            ax2.plot(f, np.where(good, phase_deg, np.nan), lw=1.0,
                     color=col, label=name)
        ax1.axhline(1.0, color='#6c757d', lw=0.6, ls=':')
        ax1.set_xlim(0, core.LOWPASS_HZ)
        ax2.set_xlim(0, core.LOWPASS_HZ)
        if self.logy.isChecked():
            ax1.set_yscale('log')
        ax1.set_ylabel(f'|H1| (response / {ref_name})')
        ax1.set_title(f'ODS — H1 Bode from {ref_name}'
                      f' - {core.AXIS_NAMES[axis]}')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Phase [°]')
        ax2.set_ylim(-185, 185)
        ax2.axhline(0, color='#6c757d', lw=0.6, ls=':')
        for a in (ax1, ax2):
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        note = self._pair_note(skipped, aligned, [])
        if min_navg is not None:
            note += (f'ODS: field data, no controlled input. Phase is '
                     f'angle(H1) relative to {ref_name}. Solid: coherence '
                     f'≥ 95% confidence limit ({min_navg} segments). '
                     'Phase is meaningless below the coherence limit.')
        self.note.setText(note)
        self.plot.finish(ax1, ax2)

    # ------------------------------------------------------------------
    # Kind 14: resposta em frequencia dos filtros de ponderacao (5.1)
    # ------------------------------------------------------------------
    def _draw_filter_response(self):
        """Magnitude (dB) e fase (graus) de todos os filtros de ponderacao
        disponiveis. Indica a ponderacao ativa no painel de ponderacao."""
        cfg = self.weight.config()
        active = {wname for wname, _ in cfg.values() if wname}

        f = np.logspace(-2, 2, 2000)   # 0.01 a 100 Hz
        ax1, ax2 = self.plot.axes(2)

        colors = SERIES_COLORS
        for i, name in enumerate(sorted(core.WEIGHTINGS)):
            col = colors[i % len(colors)]
            lw = 2.0 if name in active else 0.9
            alpha = 1.0 if name in active else 0.45
            H = core.weighting_response(name, f)
            mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
            phase_deg = np.degrees(np.angle(H))
            p = core.WEIGHTINGS[name]
            label = f"{name}  [{p['axis']}]"
            ax1.semilogx(f, mag_db, color=col, lw=lw, alpha=alpha, label=label)
            ax2.semilogx(f, phase_deg, color=col, lw=lw, alpha=alpha)

        ax1.set_ylabel('Magnitude [dB]')
        ax1.axhline(-3.0, ls=':', lw=0.7, color='#888')
        ax1.set_xlim(f[0], f[-1])
        ax1.legend(fontsize=7, ncol=2)
        ax1.set_title('Weighting filter responses (ISO 2631-1 Annex A)')

        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_xlim(f[0], f[-1])
        ax2.set_ylim(-185, 185)
        ax2.axhline(0, ls=':', lw=0.7, color='#888')

        # Informacao de polos/zeros para as ponderacoes ativas
        pz_lines = []
        for name in sorted(active):
            z, p, k = core.weighting_filter_zpk(name)
            fn_poles = sorted(set(
                abs(pi) / (2.0 * np.pi) for pi in p if abs(pi) > 1e-6
            ))
            fn_zeros = sorted(set(
                abs(zi) / (2.0 * np.pi) for zi in z if abs(zi) > 1e-6
            ))
            pz_lines.append(
                f"{name}: poles {[f'{f:.2f}' for f in fn_poles]} Hz  |  "
                f"zeros {[f'{f:.2f}' for f in fn_zeros]} Hz  |  "
                f"gain {k:.4g}"
            )
        self.note.setText('\n'.join(pz_lines) if pz_lines
                          else 'No weighting active — select one in the Weighting panel.')
        self.plot.finish(ax1, ax2)

    # ------------------------------------------------------------------
    # Kind 15: transmissibilidade H1 em bandas de 1/3 de oitava (3.10)
    # ------------------------------------------------------------------
    def _draw_h1_third_octave(self, names, axis):
        """H1 em bandas de 1/3 de oitava: media de energia (|H1|^2 por banda)
        e media complexa (preserva fase). Cada par vs primeiro registro."""
        if axis is None or len(names) < 2:
            ax, = self.plot.axes()
            freqplots.message(
                self.plot,
                'Pick a single axis and at least two records '
                '(first = reference).')
            self.note.setText('')
            return

        ref_name = names[0]
        ref_exc = self.signals[ref_name]
        nperseg = self.nperseg.currentData()
        overlap = self.overlap.currentData()
        window = self.window.currentText()
        cl = core.coherence_confidence_limit(
            max(1, int(len(ref_exc.accel[:, axis]) / nperseg / (1 - overlap))))

        ax1, ax2 = self.plot.axes(2)

        for i, name in enumerate(names[1:]):
            exc = self.signals[name]
            x, y, lag_info, _ = self._pair(ref_exc, exc, axis)
            if x is None:
                continue
            col = SERIES_COLORS[(i + 1) % len(SERIES_COLORS)]
            try:
                est = core.frf_estimates(x, y, exc.dt, window=window,
                                         nperseg=nperseg, overlap=overlap)
            except Exception:
                continue
            f = est['f']
            h1 = est['h1']
            g2 = est['g2']

            # Bandas de 1/3 de oitava ate LOWPASS_HZ
            centers = core.THIRD_OCTAVE_CENTERS[
                core.THIRD_OCTAVE_CENTERS <= core.LOWPASS_HZ]
            fc2 = 2.0 ** (1.0 / 6.0)
            mag_band, coh_band, cplx_band = [], [], []
            for fc in centers:
                mask = (f >= fc / fc2) & (f < fc * fc2)
                if not mask.any():
                    mag_band.append(np.nan)
                    coh_band.append(np.nan)
                    cplx_band.append(np.nan + 0j)
                    continue
                mag2 = np.abs(h1[mask]) ** 2
                coh = g2[mask]
                good = coh >= cl
                mag_band.append(np.sqrt(np.mean(mag2[good])) if good.any()
                                 else np.sqrt(np.nanmean(mag2)))
                coh_band.append(float(np.nanmean(coh)))
                cplx_band.append(np.mean(h1[mask]))

            mag_band = np.array(mag_band, dtype=float)
            coh_band = np.array(coh_band, dtype=float)
            label = name if len(name) <= 30 else name[:27] + '...'

            # Barras de |H1| medio por banda
            ax1.bar(centers, mag_band, width=centers * (fc2 - 1.0 / fc2),
                    align='center', alpha=0.65, color=col, label=label)
            # Coerencia media por banda
            ax2.semilogx(centers, coh_band, 'o-', color=col, ms=4, lw=1.0)

        ax1.set_xscale('log')
        ax1.set_xlim(centers[0] / fc2, centers[-1] * fc2)
        ax1.set_xlabel('')
        ax1.set_ylabel('|H1| (energy average per band)')
        ax1.set_title(
            f'H1 in 1/3-octave bands — axis {core.AXIS_NAMES[axis]}')
        ax1.legend(fontsize=8)

        ax2.axhline(cl, ls='--', lw=1.0, color='#888',
                    label=f'95 % CL ({cl:.2f})')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Centre frequency [Hz]')
        ax2.set_ylabel('Mean coherence per band')
        ax2.legend(fontsize=8)

        self.note.setText(
            '1/3-octave H1: energy average (√mean|H1|²) of bins within each '
            'band where coherence ≥ 95 % confidence limit. '
            'Faint bars: all bins below limit (energy average of all bins used).')
        self.plot.finish(ax1, ax2)

    def _draw_multiple_coherence(self, names, axis):
        """Coerencia multipla: primeiro sinal = saida, demais = entradas."""
        if axis is None or len(names) < 3:
            ax, = self.plot.axes()
            msg = ('Pick a single axis.' if axis is None else
                   'Check at least three records: the first is the OUTPUT '
                   '(response), the others are the INPUTS.')
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    color='#6c757d', wrap=True, transform=ax.transAxes)
            self.plot.finish(ax)
            self.note.setText('')
            return
        out_name = names[0]
        out_exc = self.signals[out_name]
        inputs_x, input_names, skipped = [], [], []
        for name in names[1:]:
            exc = self.signals[name]
            xs, ys, why, _ = self._pair(out_exc, exc, axis)
            if xs is None:
                skipped.append(f'{name} ({why})')
                continue
            # xs = reference (output), ys = input signal
            inputs_x.append(ys)
            input_names.append(name)
        if len(inputs_x) < 2:
            ax, = self.plot.axes()
            ax.text(0.5, 0.5,
                    'Need at least 2 input signals to compute multiple '
                    'coherence (check 3+ records).',
                    ha='center', va='center', color='#6c757d', wrap=True,
                    transform=ax.transAxes)
            self.plot.finish(ax)
            self.note.setText('Not enough records. ' + (
                'Skipped: ' + '; '.join(skipped) if skipped else ''))
            return
        y_sig = self._sig(out_exc, axis)
        inp_sigs = [self._sig(self.signals[n], axis) for n in input_names]
        # Align lengths
        min_len = min(len(y_sig), min(len(x) for x in inp_sigs))
        y_sig = y_sig[:min_len]
        inp_sigs = [x[:min_len] for x in inp_sigs]
        result = core.multiple_coherence_estimates(
            inp_sigs, y_sig, out_exc.dt,
            window=self.window.currentText(),
            nperseg=self.nperseg.currentData(),
            overlap=self.overlap.currentData())
        if result is None:
            self.note.setText('Could not compute multiple coherence.')
            return
        f = result['f']
        fmask = f <= core.LOWPASS_HZ
        ax1, ax2 = self.plot.axes(2)
        ax1.plot(f[fmask], result['gamma2_m'][fmask], lw=2.0,
                 color='#2c3e50', label='Multiple γ²')
        for i, name in enumerate(input_names):
            col = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax2.plot(f[fmask], result['gamma2_ord'][i][fmask], lw=1.0,
                     color=col, label=f'{name} → {out_name}')
        cl = core.coherence_confidence_limit(result['navg'])
        for a in (ax1, ax2):
            a.axhline(cl, color='#c0392b', lw=1.0, ls='--',
                      label='95% confidence limit')
            a.set_xlim(0, core.LOWPASS_HZ)
            a.set_ylim(0, 1.05)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        ax1.set_ylabel(r'Multiple coherence $\gamma^2_m$')
        ax1.set_title(f'Multiple coherence — output: {out_name}'
                      f' ({core.AXIS_NAMES[axis]})')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel(r'Ordinary coherence $\gamma^2$')
        note = (f'Multiple coherence (γ²_m): fraction of output power '
                f'explained by all {len(input_names)} inputs simultaneously. '
                f'Inputs: {", ".join(input_names)}. '
                f'Ordinary coherences per input in lower panel. '
                f'{result["navg"]} Welch segments.')
        if skipped:
            note += ' Skipped: ' + '; '.join(skipped) + '.'
        self.note.setText(note)
        self.plot.finish(ax1, ax2)

    def _open_calibration_dialog(self):
        """Assistente de calibracao: calibrador de referencia (4.1) e
        6 posicoes estaticas (4.2)."""
        r = self.list.currentRow()
        if r < 0:
            QMessageBox.information(self, 'Calibration',
                                    'Select a record in the list first.')
            return
        name = self.list.item(r).text()
        exc = self.signals[name]

        dlg = QDialog(self)
        dlg.setWindowTitle(f'Calibration — {name}')
        dlg.setMinimumWidth(500)
        vlayout = QVBoxLayout(dlg)

        tabs = QTabWidget()

        # ------------------------------------------------------------------
        # Tab 1: calibrador de referencia (4.1)
        # ------------------------------------------------------------------
        tab1 = QWidget()
        form1 = QFormLayout(tab1)
        form1.setContentsMargins(8, 8, 8, 8)

        ax1 = QComboBox()
        ax1.addItems(core.AXIS_NAMES)
        ax1.setCurrentIndex(2)  # z por padrao
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
            'Check the calibrator certificate: some report 10 m/s² peak, '
            'others 10 m/s² RMS (= 1 g RMS). Mixing them gives 41 % error.')
        form1.addRow('Amplitude type', ref_type)

        ref_freq = QDoubleSpinBox()
        ref_freq.setRange(1.0, 20000.0)
        ref_freq.setDecimals(1)
        ref_freq.setValue(159.2)
        ref_freq.setSuffix(' Hz')
        form1.addRow('Reference frequency', ref_freq)

        res1_lbl = QLabel('—')
        res1_lbl.setWordWrap(True)
        form1.addRow('Result', res1_lbl)

        _res1 = [None]

        def _compute_ref():
            ai = ax1.currentIndex()
            sig = exc.accel[:, ai]
            is_peak = ref_type.currentIndex() == 0
            r1 = core.calibrate_from_reference(
                sig, exc.dt, ref_val.value(), ref_freq.value(), is_peak)
            _res1[0] = r1
            color = 'black' if r1['valid'] else '#c0392b'
            res1_lbl.setStyleSheet(f'color:{color}')
            res1_lbl.setText(r1['message'])

        btn_c1 = QPushButton('Compute')
        btn_c1.clicked.connect(_compute_ref)
        form1.addRow('', btn_c1)

        tabs.addTab(tab1, 'Reference calibrator')

        # ------------------------------------------------------------------
        # Tab 2: 6 posicoes estaticas (4.2)
        # ------------------------------------------------------------------
        tab2 = QWidget()
        form2 = QFormLayout(tab2)
        form2.setContentsMargins(8, 8, 8, 8)
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

        res2_lbl = QLabel('—')
        res2_lbl.setWordWrap(True)
        form2.addRow('Result', res2_lbl)

        _res2 = [None]

        def _compute_static():
            readings = {k: v.value() for k, v in pos_fields.items()}
            r2 = core.calibrate_static_6pos(readings)
            _res2[0] = r2
            lines = [
                f"{ax}: offset {r2[ax]['offset']:+.4f} m/s², "
                f"gain {r2[ax]['gain']:.4f}"
                for ax in core.AXIS_NAMES
            ]
            res2_lbl.setText('  |  '.join(lines))

        btn_c2 = QPushButton('Compute')
        btn_c2.clicked.connect(_compute_static)
        form2.addRow('', btn_c2)

        tabs.addTab(tab2, '6-position static')

        vlayout.addWidget(tabs)

        # ------------------------------------------------------------------
        # Botoes Apply / Close
        # ------------------------------------------------------------------
        btns = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)

        def _apply():
            tidx = tabs.currentIndex()
            meta = dict(self.stats[name].get('meta', {}))
            if tidx == 0:
                r1 = _res1[0]
                if not r1 or not r1['valid']:
                    QMessageBox.information(dlg, 'Nothing to apply',
                        'Click Compute first and verify the result is valid.')
                    return
                ai = ax1.currentIndex()
                exc.accel[:, ai] *= r1['correction']
                meta['calibration'] = (
                    f"ref-cal: axis={ax1.currentText()}, "
                    f"correction={r1['correction']:.4f}, "
                    f"freq={r1['freq_measured']:.2f}Hz, "
                    f"SNR={r1['snr_db']:.1f}dB"
                )
                msg = (f"Axis {ax1.currentText()} scaled by "
                       f"{r1['correction']:.4f}. Stats recomputed.")
            else:
                r2 = _res2[0]
                if not r2:
                    QMessageBox.information(dlg, 'Nothing to apply',
                        'Click Compute first.')
                    return
                for i, ax in enumerate(core.AXIS_NAMES):
                    d = r2[ax]
                    if abs(d['gain']) > 1e-6:
                        exc.accel[:, i] = (exc.accel[:, i] - d['offset']) / d['gain']
                parts = [f"{ax}: off={r2[ax]['offset']:+.4f}, "
                         f"gain={r2[ax]['gain']:.4f}"
                         for ax in core.AXIS_NAMES]
                meta['calibration'] = '6-pos: ' + '; '.join(parts)
                msg = 'Offset and gain correction applied. Stats recomputed.'

            cfg = self.weight.config()
            new_st = core.signal_statistics(exc)
            new_st['w'] = core.weighted_metrics(exc, cfg)
            new_st['band_peak_z'] = _band_peak_z(exc)
            new_st['meta'] = meta
            new_st['fs'] = 1.0 / exc.dt
            new_st['weighting_config'] = dict(cfg)
            self.stats[name] = new_st
            self.signals[name].metadata = meta
            self.refresh_stats()
            self.refresh()
            QMessageBox.information(dlg, 'Applied', msg)

        btns.button(QDialogButtonBox.Apply).clicked.connect(_apply)
        btns.button(QDialogButtonBox.Close).clicked.connect(dlg.reject)
        vlayout.addWidget(btns)

        dlg.exec()

    def edit_metadata(self):
        """Dialog para editar e salvar metadados do registro selecionado."""
        r = self.list.currentRow()
        if r < 0:
            QMessageBox.information(self, 'Metadata',
                                    'Select a record in the list first.')
            return
        name = self.list.item(r).text()
        path = self.list.item(r).toolTip()
        meta = self.stats[name].get('meta', {})

        dlg = QDialog(self)
        dlg.setWindowTitle(f'Metadata — {name}')
        form = QFormLayout(dlg)
        form.setContentsMargins(12, 12, 12, 8)
        fields = {}
        labels = [
            ('subject_age', 'Subject age (years or months)'),
            ('subject_mass_kg', 'Body mass [kg]'),
            ('subject_height_cm', 'Body height [cm]'),
            ('position', 'Measurement position'),
            ('sensor_model', 'Sensor model'),
            ('sensor_serial', 'Sensor serial number'),
            ('calibration', 'Calibration applied'),
            ('calibration_date', 'Calibration date'),
            ('operator', 'Operator'),
            ('notes', 'Notes'),
        ]
        for key, lbl in labels:
            ed = QLineEdit(meta.get(key, ''))
            form.addRow(lbl, ed)
            fields[key] = ed

        btns = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.Accepted:
            return
        new_meta = {k: fields[k].text().strip() for k in fields
                    if fields[k].text().strip()}
        self.stats[name]['meta'] = new_meta
        self.signals[name].metadata = new_meta
        self.refresh_stats()
        # Salva no arquivo se o caminho ainda existir
        if path and Path(path).exists():
            try:
                core.write_excitation_header(path, new_meta)
            except Exception as err:
                QMessageBox.warning(self, 'Metadata',
                                    f'Saved in session but could not write '
                                    f'to file:\n{err}')

    def export_iv_csv(self):
        names = self._checked()
        if not names:
            QMessageBox.information(self, 'Export',
                                    'Select at least one record.')
            return
        sec = self.interval_sec.currentData()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export interval values', 'intervals.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='') as fh:
            fh.write(f'# exported {datetime.now().isoformat(timespec="seconds")}\n')
            fh.write(f'# interval_s={sec}\n')
            fh.write(f'# weighting=Wd/Wd/Wk k=1.4/1.4/1.0 (ISO 2631-1 health)\n')
            first = True
            for name in names:
                exc = self.signals[name]
                blocks = exc.accel[np.newaxis]
                m = nho09.interval_metrics(blocks, exc.dt, sec)
                fh.write(f'# signal={name}\n')
                rows = nho09.interval_csv_rows(m, clock=False)
                for j, row in enumerate(rows):
                    if j == 0 and first:
                        fh.write(','.join(row) + '\n')
                        first = False
                    elif j > 0:
                        fh.write(','.join(row) + '\n')

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
        weight_cfg = self.weight.config()
        weight_desc = ', '.join(
            f'{ax}={name or "none"}(k={k:g})' for ax, (name, k) in weight_cfg.items())
        meta = [
            f'# exported {datetime.now().isoformat(timespec="seconds")}',
            f'# method={self.method.currentText()}',
            f'# window={self.window.currentText()}',
            f'# segment_length={self.nperseg.currentData()}',
            f'# overlap={self.overlap.currentData()}',
            f'# band_width_Hz={self.band.value():g}',
            f'# axis={"sum" if axis is None else core.AXIS_NAMES[axis]}',
            f'# weighting: {weight_desc}',
        ]
        # Inclui metadados de cada sinal exportado como comentários de cabeçalho.
        for sig_name in names:
            sig_meta = self.stats.get(sig_name, {}).get('meta', {})
            if sig_meta:
                meta.append(f'# --- metadata: {sig_name} ---')
                for k, v in sig_meta.items():
                    meta.append(f'# {k}={v}')
        with open(path, 'w', newline='') as fh:
            fh.write('\n'.join(meta) + '\n')
            np.savetxt(fh, np.column_stack(cols), delimiter=',',
                       header=','.join(header), comments='', fmt='%.6g')
        self.status(f'Spectrum written to {Path(path).name}')

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_row(label, value):
        """Retorna (bg_color, fg_color) para colorização semântica de linhas."""
        GREEN_FG = '#1a5c35'; GREEN_BG = '#e6f4ed'
        WARN_FG  = '#7b5800'; WARN_BG  = '#fff8e1'
        RED_FG   = '#8b1a1a'; RED_BG   = '#fde8e8'
        TXT      = '#2d3748'

        v = str(value).lower()
        l = label.lower()

        if 'dir. 2002' in l or 'health zone' in l:
            if 'below' in v:
                return GREEN_BG, GREEN_FG
            elif 'above' in v or 'exceeded' in v:
                return RED_BG, RED_FG
            else:
                return WARN_BG, WARN_FG
        if 'comfort' in l:
            if 'not uncomfortable' in v or 'a little' in v:
                return GREEN_BG, GREEN_FG
            elif 'very uncomfortable' in v or 'extremely' in v:
                return RED_BG, RED_FG
            elif 'uncomfortable' in v:
                return WARN_BG, WARN_FG
        if 'applicable' in l:
            if v.startswith('yes'):
                return GREEN_BG, GREEN_FG
            elif 'no' in v:
                return RED_BG, RED_FG
        if 'sufficient' in l:
            if v.startswith('yes'):
                return GREEN_BG, GREEN_FG
            elif 'no' in v:
                return WARN_BG, WARN_FG
        return None, TXT

    # ------------------------------------------------------------------
    def _build_report_page(self, sig_name, s, nome=''):
        """Cria figura A4 de relatório profissional (layout manual, sem ax.table)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp

        NAVY    = '#1a2b4a'
        SECT_BG = '#e8eef6'
        SECT_FG = '#2d4a7a'
        ALT_BG  = '#f5f7fc'
        BORDER  = '#c4d0e4'
        TXT     = '#2d3748'
        MUTED   = '#6b7a8d'
        WHITE   = 'white'

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(WHITE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        def R(x, y, w, h, fc, ec=None, lw=0.4):
            ax.add_patch(mp.Rectangle(
                (x, y), w, h, facecolor=fc,
                edgecolor=(ec or fc), linewidth=lw,
                transform=ax.transAxes, clip_on=False))

        def T(x, y, s_, **kw):
            kw.setdefault('transform', ax.transAxes)
            kw.setdefault('va', 'center')
            kw.setdefault('fontsize', 7.5)
            kw.setdefault('color', TXT)
            ax.text(x, y, s_, **kw)

        # ── Cabeçalho ────────────────────────────────────────────────
        R(0, 0.956, 1, 0.044, NAVY)
        T(0.015, 0.978, 'AVALIADOR DE VIBRAÇÃO HUMANA  ·  v3.0',
          color='white', fontsize=10.5, fontweight='bold')
        T(0.985, 0.978, 'GRAVI-SH  ·  UFMG',
          color='#8daac4', fontsize=8.5, ha='right')

        # ── Barra de identificação do sinal ───────────────────────────
        R(0, 0.926, 1, 0.030, '#f0f4f8', ec=BORDER)
        T(0.015, 0.941, f'Sinal:  {sig_name}',
          color=NAVY, fontsize=8.5, fontweight='bold')
        T(0.985, 0.941, datetime.now().strftime('%Y-%m-%d  %H:%M'),
          color=MUTED, fontsize=8, ha='right')

        # ── Caixa de metadados ────────────────────────────────────────
        meta  = s.get('meta', {})
        MHDR_H = 0.020
        MROW_H = 0.013

        meta_items = [
            ('Nome',     nome if nome else None),
            ('Idade',    meta.get('subject_age')),
            ('Massa',    (str(meta['subject_mass_kg']) + ' kg')
                          if meta.get('subject_mass_kg') else None),
            ('Estatura', (str(meta['subject_height_cm']) + ' cm')
                          if meta.get('subject_height_cm') else None),
            ('Posição',  meta.get('position')),
            ('Sensor',   meta.get('sensor_model')),
            ('Nº série', meta.get('sensor_serial')),
            ('Operador', meta.get('operator')),
            ('Notas',    meta.get('notes')),
        ]
        meta_items = [(k, v) for k, v in meta_items if v]

        n_meta_rows = max(1, (len(meta_items) + 1) // 2)
        META_BODY_H = n_meta_rows * MROW_H

        R(0, 0.905, 1, MHDR_H, SECT_BG)
        T(0.015, 0.905 + MHDR_H / 2,
          'INFORMAÇÕES DO SUJEITO / MEDIÇÃO',
          color=SECT_FG, fontsize=7.5, fontweight='bold')

        R(0, 0.905 - META_BODY_H, 1, META_BODY_H, WHITE, ec=BORDER)
        for i, (k, v) in enumerate(meta_items):
            col = i % 2
            row = i // 2
            x   = 0.015 + col * 0.50
            y   = 0.905 - (row + 0.5) * MROW_H
            T(x,        y, f'{k}:', color=MUTED, fontsize=7)
            T(x + 0.14, y, str(v),  color=TXT,   fontsize=7)

        # ── Tabela de estatísticas ────────────────────────────────────
        sections = []
        curr = {'title': 'CARACTERÍSTICAS DO SINAL', 'rows': []}
        _SKIP = {
            'P50 RMS(1s) x, y, z [m/s2]', 'P90 RMS(1s) x, y, z [m/s2]',
            'P95 RMS(1s) x, y, z [m/s2]', 'P99 RMS(1s) x, y, z [m/s2]',
        }
        for label, fn in STAT_ROWS:
            if label.startswith('---'):
                title = label.strip('- ').strip().upper()
                if curr['rows']:
                    sections.append(curr)
                curr = None if 'METADATA' in title else {'title': title, 'rows': []}
            elif curr is not None and label not in _SKIP:
                try:
                    value = fn(s)
                except Exception:
                    value = '—'
                curr['rows'].append((label, str(value)))
        if curr and curr.get('rows'):
            sections.append(curr)

        ROW_H = 0.0125
        SEC_H = 0.019

        total_data = sum(len(sec['rows']) for sec in sections)
        total_h    = total_data * ROW_H + len(sections) * SEC_H
        TABLE_TOP  = 0.905 - META_BODY_H - 0.006
        available  = TABLE_TOP - 0.024

        if total_h > available and total_data > 0:
            ROW_H = max(0.009,
                        (available - len(sections) * SEC_H) / total_data)

        y = TABLE_TOP
        for sec in sections:
            R(0, y - SEC_H, 1, SEC_H, SECT_BG)
            T(0.015, y - SEC_H / 2, sec['title'],
              color=SECT_FG, fontsize=7, fontweight='bold')
            y -= SEC_H

            for r_idx, (label, value) in enumerate(sec['rows']):
                row_bg, val_color = self._classify_row(label, value)
                if row_bg is None:
                    row_bg = ALT_BG if r_idx % 2 == 0 else WHITE
                R(0, y - ROW_H, 1, ROW_H, row_bg)
                ax.plot([0, 1], [y, y], color=BORDER, lw=0.25,
                        transform=ax.transAxes)
                T(0.015, y - ROW_H / 2, label,
                  color=TXT, fontsize=6.5)
                T(0.54,  y - ROW_H / 2, value,
                  color=val_color, fontsize=6.5)
                y -= ROW_H

            ax.plot([0, 1], [y, y], color=BORDER, lw=0.5,
                    transform=ax.transAxes)

        # ── Rodapé ───────────────────────────────────────────────────
        R(0, 0, 1, 0.022, '#f0f4f8', ec=BORDER)
        T(0.015, 0.011,
          'Gerado por Avaliador de Vibração Humana v3.0  ·  GRAVI-SH / UFMG',
          color=MUTED, fontsize=6.5)
        T(0.985, 0.011, 'Página 1 / 2',
          color=MUTED, fontsize=6.5, ha='right')

        return fig

    # ------------------------------------------------------------------
    def export_report_pdf(self):
        """Exporta PDF: página(s) de relatório + gráficos selecionados (A4)."""
        names = self._checked()
        if not names:
            QMessageBox.information(self, 'Export',
                                    'Select at least one record.')
            return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # ── Diálogo de seleção de gráficos ──────────────────────────────
        plot_names = [self.kind.itemText(i) for i in range(self.kind.count())]
        cur_idx = self.kind.currentIndex()

        dlg = QDialog(self)
        dlg.setWindowTitle('Selecionar gráficos para o PDF')
        dlg.setMinimumWidth(520)
        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(12, 12, 12, 8)

        from PySide6.QtWidgets import QLineEdit, QFormLayout as _QFL
        _lbl_meta = QLabel('Identificação do sujeito (opcional)')
        _lbl_meta.setStyleSheet('font-weight: bold;')
        vl.addWidget(_lbl_meta)
        _fm = _QFL()
        _le_nome = QLineEdit()
        _fm.addRow('Nome / ID do sujeito', _le_nome)
        vl.addLayout(_fm)

        _sep = QFrame(); _sep.setFrameShape(QFrame.Shape.HLine)
        _sep.setStyleSheet('color: #c4d0e4; margin: 4px 0;')
        vl.addWidget(_sep)

        _lbl_plots = QLabel('Gráficos a incluir (além das páginas de estatísticas)')
        _lbl_plots.setStyleSheet('font-weight: bold;')
        vl.addWidget(_lbl_plots)
        checks = []
        for i, pname in enumerate(plot_names):
            cb = QCheckBox(pname)
            cb.setChecked(i == cur_idx)
            vl.addWidget(cb)
            checks.append(cb)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText('Exportar PDF')
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        _nome = _le_nome.text().strip()
        selected_indices = [i for i, cb in enumerate(checks) if cb.isChecked()]

        stem = names[0]
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item and item.text() in names:
                tp = item.toolTip()
                if tp:
                    stem = Path(tp).stem
                break

        path, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF report', f'{stem}_report.pdf', 'PDF (*.pdf)')
        if not path:
            return

        orig_idx = self.kind.currentIndex()
        n_pages = len(names) + len(selected_indices)

        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt as _Qt
        prog = QProgressDialog('Gerando PDF…', '', 0, 0, self)
        prog.setWindowTitle('Exportar PDF')
        prog.setWindowModality(_Qt.WindowModality.WindowModal)
        prog.setCancelButton(None)
        prog.setMinimumDuration(0)
        prog.show()
        QApplication.processEvents()

        try:
            with PdfPages(path) as pdf:
                # Páginas de estatísticas (A4 retrato, uma por registro)
                for name in names:
                    s = self.stats.get(name)
                    if s is None:
                        continue
                    fig = self._build_report_page(name, s, nome=_nome)
                    pdf.savefig(fig, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                # Páginas de gráfico (A4 paisagem)
                orig_size = self.plot.figure.get_size_inches()
                for k in selected_indices:
                    self.kind.setCurrentIndex(k)
                    self.refresh()
                    self.plot.figure.set_size_inches(11.69, 8.27)
                    pdf.savefig(self.plot.figure, bbox_inches='tight', dpi=150)
                    self.plot.figure.set_size_inches(*orig_size)
                # Restaura
                self.kind.setCurrentIndex(orig_idx)
                self.refresh()
                if hasattr(self.plot, 'canvas'):
                    self.plot.canvas.draw_idle()
        except Exception as _e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(_e))
            return
        finally:
            prog.reset()

        self.status(f'PDF report saved to {Path(path).name}.')

    # ------------------------------------------------------------------
    def join_selected(self):
        """Concatena os registros marcados (em ordem de lista) num unico sinal."""
        names = self._checked()
        if len(names) < 2:
            QMessageBox.information(
                self, 'Join selected',
                'Mark at least 2 records in the list to join.')
            return
        excs = [self.signals[n] for n in names if n in self.signals]
        if len(excs) < 2:
            return

        # Verificar taxa de amostragem identica
        dts = [e.dt for e in excs]
        if max(dts) - min(dts) > 1e-9:
            QMessageBox.warning(
                self, 'Join selected',
                'Records have different sample rates and cannot be joined.')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        dt = excs[0].dt
        accels = [e.accel for e in excs]
        joined_accel = np.concatenate(accels, axis=0)
        N = joined_accel.shape[0]
        joined_time = np.arange(N) * dt
        joined_name = '+'.join(n.split('.')[0] for n in names[:3])
        if len(names) > 3:
            joined_name += f'+({len(names)-3}more)'
        joined_name += '_joined'

        # Evitar duplicata
        suffix = ''
        base = joined_name
        k = 1
        while joined_name + suffix in self.signals:
            k += 1
            suffix = f'_{k}'
        joined_name = base + suffix

        from . import core as _core
        exc = _core.Excitation(
            name=joined_name,
            time=joined_time,
            accel=joined_accel,
            dt=dt)
        _core._prepare(exc)

        config = self.weight.config()
        try:
            st = _core.signal_statistics(exc)
            st['w'] = _core.weighted_metrics(exc, config)
            st['band_peak_z'] = _band_peak_z(exc)
            st['meta'] = dict(exc.metadata)
            st['fs'] = 1.0 / exc.dt
            st['weighting_config'] = dict(config)
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Join error', str(err))
            return

        self.signals[joined_name] = exc
        self.stats[joined_name] = st

        # Desmarca as fontes para evitar que o plot redesenhe todos os sinais
        # (melhora performance com sinais grandes) e deixa só o resultado
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it and it.text() in names:
                it.setCheckState(Qt.Unchecked)
        self.list.blockSignals(False)

        item = QListWidgetItem(joined_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setToolTip(f'Joined: {", ".join(names)}')
        self.list.addItem(item)
        self.list.setCurrentRow(self.list.count() - 1)
        self.refresh()
        self.refresh_stats()
        QApplication.restoreOverrideCursor()
        self.status(f'Joined {len(names)} records into {joined_name} '
                    f'({N:,} samples, {N*dt/3600:.2f} h).')
