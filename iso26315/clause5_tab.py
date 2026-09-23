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
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import annexc, core
from . import fielddata as fd
from .qtworker import run_in_thread
from .theme import ACCENT, BAD_COLOR, OK_COLOR, TEXT, WARN_COLOR

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
        self._build(plot_factory)

    # ------------------------------------------------------------------
    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        intro = QLabel(
            'Clause 5 with Annex C — the path for exposures <b>with</b> loss of '
            'contact with the seat. Independent of the Annex A chain: the '
            'signal is conditioned with a 2-pole high pass at 0.1 Hz and a '
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
        field_btn = QPushButton('Load field logger day (.m files)...')
        field_btn.setToolTip(
            'Reads a set of .m files from the field logger (2 s blocks every '
            '5 s) and feeds each block to the Clause 5 chain independently — '
            'no gluing. The measured time counts only the 2 s actually '
            'recorded per block, not the 3 s the logger spends writing to '
            'the SD card.')
        field_btn.clicked.connect(self.load_field)
        sv.addWidget(field_btn)
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
        sv.addStretch()
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
        g.addRow('Days per year', self.days)
        self.start_age = QSpinBox()
        self.start_age.setRange(1, 90)
        self.start_age.setValue(20)
        g.addRow('Age at first exposure', self.start_age)
        self.years = QSpinBox()
        self.years.setRange(1, 70)
        self.years.setValue(45)
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
        ])
        self.plot_kind.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(self.plot_kind, 1)
        right.addLayout(ctl)
        self.plot = plot_factory()
        right.addWidget(self.plot, 1)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)

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

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open vertical acceleration record', '',
            'Time series (*.txt *.asc *.dat *.csv);;All files (*)')
        if not path:
            return
        self.status(f'Reading {Path(path).name}...')
        try:
            data = np.loadtxt(path, usecols=(0, 1, 2, 3))
        except Exception as exc:
            QMessageBox.critical(self, 'Could not read the record', str(exc))
            return
        t = data[:, 0]
        self.dt = (t[-1] - t[0]) / (len(t) - 1)
        self.az = data[:, 3]
        self.path = Path(path)
        self.day = None
        self.segments = None
        self.field_note.setVisible(False)
        self.measured.setValue(float(t[-1] - t[0]))
        self.file_label.setText(
            f'{self.path.name} — {len(t):,} samples, '
            f'{t[-1] - t[0]:.1f} s at {1 / self.dt:.4g} Hz. '
            f'Peak |a_z| = {np.abs(self.az).max():.2f} m/s2.')
        self.file_label.setStyleSheet(f'color:{TEXT};')
        self.run_btn.setEnabled(True)

    def load_field(self):
        if self._busy_flag:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select the .m files of one seat, one day', '',
            'Field logger (*.m);;All files (*)')
        if not paths:
            return

        def job(report):
            return fd.load_day(paths, calibrate=False, progress=report,
                               position='seat')

        self._busy(True, f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_field_load_done,
                      self._on_field_load_fail, on_progress=self._on_progress,
                      thread_attr='_field_thread', worker_attr='_field_worker')

    def _busy(self, on, message=''):
        self._busy_flag = on
        self.field_progress.setVisible(on)
        if message:
            self.status(message)
        self.run_btn.setEnabled(not on and (self.path is not None or
                                            self.day is not None))

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
        self.segments = [day.blocks[i, :, 2] for i in range(len(day.blocks))]
        measured_time = len(day.blocks) * fd.BLOCK_SAMPLES * day.dt
        self.measured.setValue(measured_time)
        self.file_label.setText(
            f'{len(day.records)} field logger file(s), {len(day.blocks):,} '
            f'blocks of 2 s. Measured {measured_time:.0f} s of {day.span_hours * 3600:.0f} s '
            f'clock time covered ({day.duty_cycle * 100:.0f}% duty cycle).')
        self.file_label.setStyleSheet(f'color:{TEXT};')
        self.field_note.setText(
            'Each 2 s block is conditioned and has loss of contact detected '
            'independently — blocks are never glued together. "Measurement '
            'duration" below counts only the seconds actually recorded '
            '(blocks x 2 s), not the SD-card write gaps; edit it if the '
            'measured condition should be scaled differently before Formula '
            '(5) extrapolates it to the daily exposure.')
        self.field_note.setVisible(True)
        self._busy(False)

    def _on_field_load_fail(self, tb):
        self._finish_thread('_field_thread')
        self._busy(False)
        QMessageBox.critical(self, 'Could not read the files',
                             tb.strip().splitlines()[-1])

    # ------------------------------------------------------------------
    def run(self):
        if self.path is None and self.segments is None:
            return
        self.status('Conditioning the signal and computing the response...')
        try:
            if self.segments is not None:
                self.result = annexc.assess_from_segments(
                    self.segments, self.dt,
                    measured_time=self.measured.value(),
                    body_mass=self.mass.value(),
                    sex=self.sex.currentText(),
                    daily_time=self.daily.value() * 3600.0,
                    days_per_year=self.days.value(),
                    start_age=self.start_age.value(),
                    n_years=self.years.value(),
                    mass_fraction=self.fraction.value() / 100.0,
                    endplate_area=self.area.value(),
                    added_mass=self.added.value(),
                    exclude_loss_of_contact=self.contact.isChecked())
                self._show()
                self.redraw()
                self.status('Clause 5 / Annex C assessment complete.')
                return
            self.result = annexc.assess(
                self.az, self.dt,
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
                exclude_loss_of_contact=self.contact.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, 'Assessment failed', str(exc))
            return
        self._show()
        self.redraw()
        self.status('Clause 5 / Annex C assessment complete.')

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
            f'C.2.</div>')

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
        self.table.setRowCount(len(rows))
        for i, (lbl, val, src) in enumerate(rows):
            txt = (val if isinstance(val, str) else
                   f'{val * 100:.2f} %' if lbl.startswith('Pi') else
                   f'{val:,}' if isinstance(val, int) else f'{val:.4f}')
            for c, v in enumerate((lbl, txt, src)):
                it = QTableWidgetItem(v)
                it.setForeground(QColor(TEXT))
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

        if kind == 2:
            f = np.logspace(-1, np.log10(100), 600)
            H = annexc.transfer_function(f)
            ax1, ax2 = self.plot.axes(2)
            ax1.semilogx(f, np.abs(H), lw=1.4, color=ACCENT)
            ax1.set_ylabel('|H|')
            ax1.set_title('Seat-to-spine transfer function, Formula (1)')
            ax2.semilogx(f, np.unwrap(np.angle(H)), lw=1.4, color=ACCENT)
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
        ax1.set_title('Clause 5 conditioning: 0.1 Hz high pass, 80 Hz low pass'
                      + (f' — {len(self.segments)} blocks shown back to back, '
                         f'each conditioned on its own; real blocks are '
                         f'5 s apart, not 2 s' if self.segments is not None
                         else ''))
        ax2.plot(t, resp[::stride], lw=0.5, color=ACCENT)
        ax2.set_ylabel('Spine response $A_z$ [m/s$^2$]')
        ax2.set_xlabel('Block-relative time [s]' if self.segments is not None
                       else 'Time [s]')
        self.plot.finish(ax1, ax2)

    # ------------------------------------------------------------------
    def _draw_spectrum(self):
        """Espectro do az bruto (nao condicionado) -- so diagnostico.

        Clause 5/Anexo C nao pondera por frequencia (docstring do modulo:
        os filtros da 2631-1 sao proibidos aqui). Este grafico existe so
        para inspecionar o conteudo em frequencia do que foi medido --
        ressonancias, ruido, sensor solto -- antes/depois de rodar a cadeia
        de tempo que a norma de fato usa. Nao entra em nenhum resultado.
        """
        if self.segments is not None:
            raw = np.concatenate(self.segments)
            nperseg = fd.BLOCK_SAMPLES
        else:
            raw = self.az
            nperseg = min(8192, len(raw))
        f, y, navg = core.welch_spectrum(raw, self.dt, nperseg=nperseg,
                                         overlap=0.0)
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
