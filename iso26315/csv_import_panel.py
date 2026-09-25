"""Aba de importacao de CSV continuo (Time_ms,AX,AY,AZ em g) -- sketches de
bancada como o teste de 10Hz do Xiao, diferente do .m do registrador de
campo (que vem em blocos de 5 s com lacuna).

Nao duplica a UI de analise: so alimenta o mesmo FieldPanel usado pela aba
Directive e delega pra la (mesma logica de calculo/exibicao de sempre).
"""
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from . import fielddata as fd
from .qtworker import run_in_thread
from .theme import TEXT


class CsvImportPanel(QWidget):
    def __init__(self, field_panel, status, on_loaded=None):
        super().__init__()
        self.field_panel = field_panel
        self.status = status
        self.on_loaded = on_loaded
        self._busy_flag = False
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        note = QLabel(
            'For continuous CSV loggers (columns Time_ms,AX,AY,AZ, already '
            'in g) -- e.g. bench-test firmware, not the .m field register. '
            'This data has no 5 s gap, so it is sliced into contiguous '
            'blocks and the duty cycle comes out at ~100%. The sample rate '
            'is NOT read from the file (the millisecond timestamp is only '
            'approximate, subject to loop jitter) -- enter below whatever '
            'the firmware was actually configured for.')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        box = QGroupBox('CSV file(s)')
        form = QFormLayout(box)

        self.rate = QDoubleSpinBox()
        self.rate.setRange(1.0, 10000.0)
        self.rate.setDecimals(1)
        self.rate.setSuffix(' Hz')
        self.rate.setValue(100.0)
        self.rate.setToolTip(
            'Sample rate the firmware was configured for. Used to slice the '
            'continuous stream into analysis blocks and to compute VDV/dose '
            '-- get this wrong and every derived value is wrong.')
        form.addRow('Sample rate:', self.rate)

        row = QHBoxLayout()
        self.load_btn = QPushButton('Load CSV(s)...')
        self.load_btn.clicked.connect(self.load)
        row.addWidget(self.load_btn)
        row.addStretch()
        form.addRow(row)

        self.info = QLabel('No data loaded')
        self.info.setWordWrap(True)
        form.addRow(self.info)

        lay.addWidget(box)
        lay.addStretch()

    def load(self):
        if self._busy_flag:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select CSV file(s) of one unit (one day)', '',
            'CSV (*.csv *.txt);;All files (*)')
        if not paths:
            return
        rate = self.rate.value()
        position = self.field_panel.position.currentData()

        def job(report):
            return fd.load_day_csv(paths, rate=rate, progress=report,
                                   position=position)

        self._busy_flag = True
        self.load_btn.setEnabled(False)
        self.info.setText('Reading...')
        self.status(f'Reading {len(paths)} CSV file(s)...')
        run_in_thread(self, job, self._on_done, self._on_fail,
                      thread_attr='_load_thread', worker_attr='_load_worker')

    def _on_done(self, day):
        self._busy_flag = False
        self.load_btn.setEnabled(True)
        self.info.setText(
            f'{len(day.records)} file(s), {len(day.blocks):,} block(s), '
            f'{day.recorded_hours:.3f} h. Handed off to the Directive tab.')
        self.info.setStyleSheet(f'color:{TEXT};')
        self.status(f'Loaded {len(day.blocks):,} block(s) from CSV.')
        # Alimenta o mesmo FieldPanel usado pela aba Directive -- reusa toda
        # a logica de calculo/exibicao de la, sem duplicar UI aqui.
        self.field_panel._on_load_done(day)
        if self.on_loaded:
            self.on_loaded()

    def _on_fail(self, tb):
        self._busy_flag = False
        self.load_btn.setEnabled(True)
        self.info.setText('Failed to read the file(s).')
        self.status('Could not read the CSV file(s).')
        QMessageBox.critical(self, 'Could not read the CSV file(s)',
                             tb.strip().splitlines()[-1])
