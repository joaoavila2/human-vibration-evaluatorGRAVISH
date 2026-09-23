"""Sub-aba do explorador: dados do registrador de campo e avaliacao NHO-09.

Separada do explorador de registros ASCII porque o dado e estruturalmente
diferente: blocos de 2 s com lacunas de 3 s, em vez de um registro continuo.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import (core, directive, fielddata as fd, labmethod, nho09,
               positions as ps, summary)
from .qtworker import run_in_thread
from .theme import ACCENT, BAD_COLOR, OK_COLOR, TEXT, WARN_COLOR

AXIS_COLORS = ['#1f6feb', '#c2255c', '#1b7a33']


def _prime(day, position, hours, cfg, criterion_idx, extra_hours):
    """So esquenta os caches de ponderacao em frequencia (nho09.weighted_
    blocks), fora da UI thread. Os retornos saem descartados aqui de
    proposito: quando o codigo de renderizacao roda depois, na UI thread,
    faz as MESMAS chamadas e acha tudo em cache, entao fica rapido. Nao muda
    nenhum resultado, so onde ele e calculado -- ver FieldPanel._refresh.

    Roda inteiramente com valores simples (nao widgets do Qt), porque tocar
    um QWidget fora da UI thread corrompe a interface.
    """
    summary.normative_criteria(day, position, hours)
    summary.lab_criteria(day, cfg)
    summary.per_file(day, hours)
    if position != 'seat':
        ps.comfort(day.blocks, day.dt, position)
        if position == 'backrest':
            ps.backrest_health_informative(day.blocks, day.dt, hours)
    elif criterion_idx > 0:
        mode = 'highest' if criterion_idx == 1 else 'vector'
        directive.assess(day.blocks, day.dt, hours, mode=mode)
    else:
        nho09.assess(day.blocks, day.dt, hours, moving_mask=day.occupied)
    if extra_hours is not None:
        nho09.assess(day.blocks, day.dt, extra_hours)


class FieldPanel(QWidget):
    def __init__(self, plot_factory, status):
        super().__init__()
        self.status = status
        self.day = None
        self.result = None
        self.extra = None
        self._busy_flag = False
        self._refresh_then = None
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        note = QLabel(
            'Field logger records: 2 s blocks every 5 s, a 40% duty cycle. '
            'The blocks are never glued together: spectra use each block as a '
            'Welch segment. The ISO 2631-5 Annex A dose does not apply here, '
            'because 60% of the shocks were not recorded.')
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
        self.load_btn = QPushButton('Load day (.m files)...')
        self.load_btn.clicked.connect(self.load)
        sv.addWidget(self.load_btn)
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

        lab = QGroupBox('Lab method (MATLAB app) — for comparison only')
        lg = QGridLayout(lab)
        lg.setContentsMargins(8, 4, 8, 4)
        lg.setVerticalSpacing(3)
        self.lab_w, self.lab_k = [], []
        for c, ax in enumerate('xyz'):
            lg.addWidget(QLabel(f'{ax}'), 0, c + 1, alignment=Qt.AlignCenter)
            box = QComboBox()
            for w in labmethod.WEIGHTINGS:
                box.addItem('None' if w is None else w, w)
            box.setCurrentIndex(labmethod.WEIGHTINGS.index(
                labmethod.LabConfig().weights[c]))
            box.currentIndexChanged.connect(self._on_lab)
            lg.addWidget(box, 1, c + 1)
            kk = QDoubleSpinBox()
            kk.setRange(0.0, 5.0)
            kk.setSingleStep(0.1)
            kk.setValue(labmethod.LabConfig().k[c])
            kk.valueChanged.connect(self._on_lab)
            lg.addWidget(kk, 2, c + 1)
            self.lab_w.append(box)
            self.lab_k.append(kk)
        lg.addWidget(QLabel('Weighting'), 1, 0)
        lg.addWidget(QLabel('Gain k'), 2, 0)
        self.lab_force_axes = QCheckBox('Force fixed x-z swap (audit mode)')
        self.lab_force_axes.setChecked(False)
        self.lab_force_axes.setToolTip(
            'Off (default): the vertical axis is detected from gravity in '
            'each file, the same criterion this software\'s own pipeline '
            'uses. Orientation is not fixed — it changes between campaigns '
            'and sometimes within the same day (documentation Sec. 10.8, '
            '10.10). On: forces the original lab app\'s fixed x <-> z '
            'exchange regardless of gravity, to reproduce one specific '
            'published result for audit (Sec. 10.9).')
        self.lab_force_axes.stateChanged.connect(self._on_lab)
        lg.addWidget(self.lab_force_axes, 3, 0, 1, 2)
        self.lab_rms = QComboBox()
        for t in (0.5, 1.0, 2.0):
            self.lab_rms.addItem(f'RMS time {t:g} s', t)
        self.lab_rms.setCurrentIndex(2)
        self.lab_rms.currentIndexChanged.connect(self._on_lab)
        lg.addWidget(self.lab_rms, 3, 2, 1, 2)
        self.lab_bug = QCheckBox('Reproduce the app VDV defects')
        self.lab_bug.setChecked(False)
        self.lab_bug.setToolTip(
            'The app omits the time step in the VDV (x 1024^1/4 = 5.657), '
            'extrapolates the partial VDV with the covered period instead of '
            'the recorded time, and combines axes by root-sum-of-squares. Off '
            '(default): corrected VDV, highest axis. On: only to check the '
            'numbers against the app screen.')
        self.lab_bug.stateChanged.connect(self._on_lab)
        lg.addWidget(self.lab_bug, 4, 0, 1, 4)
        sv.addWidget(lab)
        sv.addStretch()
        top.addWidget(src, 1)

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
            'comfort.')
        self.position.currentIndexChanged.connect(self._on_position)
        eg.addWidget(self.position, 8, 1)
        eg.addWidget(QLabel('Criterion'), 4, 0)
        self.criterion = QComboBox()
        self.criterion.addItems([
            'NHO-09 (Fundacentro)',
            'Directive 2002/44/EC — highest axis',
            'Directive 2002/44/EC — vector sum (lab method)',
        ])
        self.criterion.currentIndexChanged.connect(self._recompute)
        eg.addWidget(self.criterion, 4, 1)
        eg.addWidget(QLabel('Exposure time basis'), 5, 0)
        self.basis = QComboBox()
        self.basis.addItems(['Period covered (recommended)',
                             'Recorded time only (as in CONEM2026)'])
        self.basis.setToolTip(
            'The logger records 2 s every 5 s. Vibration does not stop in the '
            '3 s gaps, so the covered period is the exposure. Using only the '
            'recorded time lowers A(8) by sqrt(0.4), about 37%.')
        self.basis.currentIndexChanged.connect(self._recompute)
        eg.addWidget(self.basis, 5, 1)
        self.iso_vdv = QCheckBox('VDV against ISO 2631-1 zone (8.5 / 17)')
        self.iso_vdv.setToolTip('The lab method uses the ISO 2631-1 health '
                                'guidance caution zone for VDV, stricter than '
                                'the Directive (9.1 / 21).')
        self.iso_vdv.stateChanged.connect(self._recompute)
        eg.addWidget(self.iso_vdv, 6, 0, 1, 2)
        grav = QLabel('Gravity is always removed from every block before any '
                      'metric. A VDV computed with the static 1 g retained '
                      'is around 130 m/s^1.75 for an 8 h day and is not '
                      'vibration.')
        grav.setWordWrap(True)
        grav.setObjectName('muted')
        eg.addWidget(grav, 7, 0, 1, 2)
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
        top.addWidget(ext, 1)
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
        ])
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
        self.plot = plot_factory()
        right.addWidget(self.plot, 1)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)

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
        for w in (self.load_btn, self.position, self.basis, self.extrapolate,
                 self.calibrate, self.lab_force_axes, self.lab_rms,
                 self.lab_bug, *self.lab_w, *self.lab_k):
            w.setEnabled(not on)
        # Estes tres tem regra propria de habilitacao (posicao/extrapolacao),
        # nao "sempre habilitado quando ocioso" como os demais controles.
        if on:
            self.criterion.setEnabled(False)
            self.iso_vdv.setEnabled(False)
            self.hours.setEnabled(False)
        else:
            seat = self.position.currentData() == 'seat'
            self.criterion.setEnabled(seat)
            self.iso_vdv.setEnabled(seat)
            self.hours.setEnabled(self.extrapolate.isChecked())
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

        def job(report):
            report(None, 'Computing the normative criteria...')
            _prime(day, position, hours, cfg, criterion_idx, extra_hours)
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
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select the .m files of one unit (one day)', '',
            'Field logger (*.m);;All files (*)')
        if not paths:
            return
        calibrate = self.calibrate.isChecked()
        position = self.position.currentData()

        def job(report):
            return fd.load_day(paths, calibrate=calibrate, progress=report,
                               position=position)

        self._busy(True, f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_load_done, self._on_load_fail,
                      on_progress=self._on_progress,
                      thread_attr='_load_thread', worker_attr='_load_worker')

    def _on_load_done(self, day):
        self._finish_thread('_load_thread')
        self._busy(False)
        self.day = day
        d = self.day
        self.info.setText(
            f'{len(d.records)} file(s), {len(d.blocks):,} blocks. '
            f'Recorded {d.recorded_hours:.2f} h of {d.span_hours:.2f} h '
            f'({d.duty_cycle * 100:.0f}% duty cycle).')
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

    def compute(self):
        self._refresh(self._compute_sync)

    def _compute_sync(self):
        d = self.day
        self.result = nho09.assess(d.blocks, d.dt, d.span_hours,
                                   moving_mask=d.occupied)
        self._compute_extra_sync()
        self._show()
        self.redraw()

    def lab_config(self):
        return labmethod.LabConfig(
            weights=tuple(b.currentData() for b in self.lab_w),
            k=tuple(k.value() for k in self.lab_k),
            force_fixed_axes=self.lab_force_axes.isChecked(),
            rms_time=self.lab_rms.currentData(),
            reproduce_vdv_defects=self.lab_bug.isChecked())

    def _on_lab(self):
        self._refresh(self._show_and_redraw)

    def _show_and_redraw(self):
        self._show()
        self.redraw()

    def _update_summary(self):
        cfg = self.lab_config()
        nkey = (id(self.day), self.position.currentData(),
                round(self._exposure_hours(), 4))
        lkey = (id(self.day), repr(cfg))
        if nkey != getattr(self, '_norm_key', None):
            self.status('Computing the normative criteria...')
            self._norm, self.dres = summary.normative_criteria(
                self.day, self.position.currentData(), self._exposure_hours())
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
    def _colour(self, value, action, limit):
        return (BAD_COLOR if value > limit else
                WARN_COLOR if value >= action else OK_COLOR)

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
        ca = self._colour(r.aren, nho09.ACTION_AREN, nho09.LIMIT_AREN)
        cv = self._colour(r.vdvr, nho09.ACTION_VDVR, nho09.LIMIT_VDVR)
        worst = max((r.aren / nho09.ACTION_AREN, r.vdvr / nho09.ACTION_VDVR))
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700">'
            f'<span style="color:{ca}">a<sub>ren</sub> = {r.aren:.3f} m/s&sup2;'
            f'</span> &nbsp;&nbsp; '
            f'<span style="color:{cv}">VDVR = {r.vdvr:.2f} m/s<sup>1,75</sup>'
            f'</span></div>'
            f'<div style="color:{TEXT};font-size:14px">'
            f'a<sub>ren</sub>: {r.verdict_aren} &nbsp;|&nbsp; '
            f'VDVR: {r.verdict_vdvr}'
            + (' &mdash; <b>one criterion exceeded is enough</b>'
               if worst >= 1 else '') + '</div>'
            f'<div style="color:#4a5157;font-size:12px">NHO-09, seat, measured '
            f'period {r.t_exp_h:.2f} h. Action level 0.5 m/s&sup2; / '
            f'9.1 m/s<sup>1,75</sup>, exposure limit 1.1 / 21.</div>')

        rows = [
            ('am x, y, z [m/s2]', ',  '.join(f'{v:.3f}' for v in r.am),
             'Wd, Wd, Wk'),
            ('amr [m/s2]', f'{r.amr:.3f}',
             'vector sum, factors 1.4 / 1.4 / 1.0'),
            ('aren [m/s2]', f'{r.aren:.3f}', r.verdict_aren),
            ('VDV x, y, z measured', ',  '.join(f'{v:.2f}' for v in r.vdv),
             f'over {r.t_amos_h:.2f} h recorded'),
            ('VDVexp x, y, z', ',  '.join(f'{v:.2f}' for v in r.vdv_exp),
             'with factor and (Texp/Tamos)^1/4'),
            ('VDVR [m/s1.75]', f'{r.vdvr:.2f}', r.verdict_vdvr),
            ('— check —', '', ''),
            ('amr, arithmetic mean', f'{r.amr_arithmetic:.3f}',
             f'what the literal formula gives: '
             f'{nho09.arithmetic_bias(r) * 100:+.0f}% — not used'),
            ('chair probably occupied', f'{self.day.occupied_fraction * 100:.0f}%',
             'heuristic; the result does not depend on it'),
        ]
        if self.extra is not None:
            e = self.extra
            rows += [
                ('— extrapolation —', '', f'assumption: {e.t_exp_h:g} h/day'),
                ('aren, square-root relation', f'{e.aren:.3f}', e.verdict_aren),
                ('aren, fourth-root relation', f'{e.aren_quarter:.3f}',
                 nho09.classify(e.aren_quarter, nho09.ACTION_AREN,
                                nho09.LIMIT_AREN)),
                ('VDVR', f'{e.vdvr:.2f}', e.verdict_vdvr),
            ]
        self._fill_table(rows + self._a8_rows())

    def _draw_dashboard(self):
        crit = [c for c in self.crit if not np.isnan(c.limit)]
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
                    label='this software (Wd / Wd / Wk, 1.4 / 1.4 / 1.0)')
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

    def _draw_hours(self):
        rows = self.hours_rows
        ax, = self.plot.axes()
        names = [r['name'][-2:] + 'h' if r['name'][-3] == '_' else r['name']
                 for r in rows]
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
            f'This software: Wd / Wd / Wk, k 1.4 / 1.4 / 1.0. Lab method: {wl}, '
            f'k {cfg.k[0]:g} / {cfg.k[1]:g} / {cfg.k[2]:g}'
            + (', fixed x-z swap forced (audit mode)' if cfg.force_fixed_axes
               else ', vertical axis auto-detected per file') +
            '. Coloured against EAV 0.5 / ELV 1.15'
            + ('.' if seat else ' for reference only: at this position the '
               'A(8) is not a health assessment, so every cell is grey.'))

    def _a8_rows(self):
        """A(8) por eixo e diario, sempre visivel, em qualquer posicao.

        Duas reguas: a deste software, com a ponderacao de saude do assento
        (a unica definicao de A(8) que a diretiva tem), e a do metodo do
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
            ('— A(8) per axis —', '', 'Wd / Wd / Wk, k 1.4 / 1.4 / 1.0' + tag),
            ('A(8) x, y, z', ',  '.join(f'{v:.3f}' for v in d.a8_axis),
             'this software'),
            ('A(8) daily, highest axis', f'{d.a8_axis.max():.3f}',
             'Directive text'),
            ('A(8) daily, vector sum', f'{np.sqrt((d.a8_axis ** 2).sum()):.3f}',
             'lab convention'),
            ('— lab method (app panel) —', '',
             f'{wl}, k {cfg.k[0]:g} / {cfg.k[1]:g} / {cfg.k[2]:g}'
             + (', fixed x-z swap forced' if cfg.force_fixed_axes
                else ', axis auto-detected')),
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

    def _show_comfort(self):
        d = self.day
        pos = self.position.currentData()
        c = ps.comfort(d.blocks, d.dt, pos)
        hours = self._exposure_hours()
        rows = [
            ('a_w x, y, z [m/s2]', ',  '.join(f'{v:.4f}' for v in c['aw']),
             ' / '.join(c['weights'])),
            ('k x, y, z', ' / '.join(f'{v:g}' for v in c['k']),
             'ISO 2631-1, 8.2'),
            ('a_v, vibration total value', f"{c['av']:.3f}", c['label']),
        ]
        extra = ''
        if pos == 'backrest':
            h = ps.backrest_health_informative(d.blocks, d.dt, hours)
            rows += [
                ('— health, informative only —', '', 'ISO 2631-1, 7.2.3'),
                ('x-axis, Wc, k = 0.8: A(8)', f"{h['a8']:.3f}",
                 'not compared with any action level'),
                ('x-axis, Wc, k = 0.8: VDV', f"{h['vdv']:.2f}",
                 'not compared with any action level'),
            ]
            extra = (' The backrest x-axis is shown for information: ISO 2631-1 '
                     'recommends measuring it but excludes it from the health '
                     'assessment.')
        rows += self._a8_rows()
        colour = (OK_COLOR if c['av'] < 0.315 else
                  WARN_COLOR if c['av'] < 1.0 else BAD_COLOR)
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700;color:{colour}">'
            f'a<sub>v</sub> = {c["av"]:.3f} m/s&sup2;</div>'
            f'<div style="color:{TEXT};font-size:14px">comfort: '
            f'<b>{c["label"]}</b> &nbsp;|&nbsp; health: not assessed at this '
            f'position</div>'
            f'<div style="color:#4a5157;font-size:12px">ISO 2631-1 comfort, '
            f'{self.position.currentText().split(" —")[0].lower()}. Annex C '
            f'values are approximate indications, not limits.{extra}</div>')
        if pos == 'feet':
            rows += self._a8_rows()
        self._fill_table(rows)

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
        self.verdict.setText(
            f'<div style="font-size:24px;font-weight:700">'
            f'<span style="color:{ca}">A(8) = {r.a8:.3f} m/s&sup2;</span>'
            f' &nbsp;&nbsp; <span style="color:{cv}">VDV = {r.vdv:.2f} '
            f'm/s<sup>1,75</sup></span></div>'
            f'<div style="color:{TEXT};font-size:14px">A(8): '
            f'{r.verdict_a8()} &nbsp;|&nbsp; VDV: {r.verdict_vdv(iso)}</div>'
            f'<div style="color:#4a5157;font-size:12px">Directive 2002/44/EC, '
            f'{label}. Exposure time {hours:.2f} h ({basis}). '
            f'A(8) 0.5 / 1.15 m/s&sup2;, VDV {lo:g} / {hi:g}.</div>')

        rows = [
            ('a_w x, y, z [m/s2]', ',  '.join(f'{v:.4f}' for v in r.aw),
             'weighted Wd, Wd, Wk; no factor'),
            ('A(8) per axis', ',  '.join(f'{v:.3f}' for v in r.a8_axis),
             'k a_w sqrt(t / 8 h)'),
            ('A(8)', f'{r.a8:.3f}', r.verdict_a8()),
            ('VDV per axis', ',  '.join(f'{v:.2f}' for v in r.vdv_axis),
             'k VDV (t_exp / t_meas)^1/4'),
            ('VDV', f'{r.vdv:.2f}', r.verdict_vdv(iso)),
            ('crest factor x, y, z', ',  '.join(f'{v:.1f}' for v in r.crest),
             'above 9: VDV is the appropriate metric'),
            ('— time —', '', ''),
            ('exposure time used', f'{hours:.2f} h', basis),
            ('time actually recorded', f'{r.t_meas_h:.2f} h',
             f'{d.duty_cycle * 100:.0f}% of the period covered'),
        ]
        self._fill_table(rows + self._a8_rows())

    # ------------------------------------------------------------------
    def redraw(self):
        if self.day is None:
            return
        k = self.kind.currentIndex()
        d = self.day

        if k == 0:
            ax, = self.plot.axes()
            t = (d.seconds - d.seconds[0]) / 3600.0
            rms = np.sqrt((d.blocks ** 2).mean(axis=1))
            for c, name in enumerate(core.AXIS_NAMES):
                ax.plot(t, rms[:, c], lw=0.6, color=AXIS_COLORS[c], label=name)
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
