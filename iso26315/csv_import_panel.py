"""Abas de importacao de sinal continuo para o dia de campo (NHO-09 /
Diretiva) a partir de formatos que nao sao o .m do registrador:

  - CsvImportPanel: CSV "Time_ms,AX,AY,AZ" (ja em g) -- sketches de bancada.
  - TxtImportPanel: .txt ASCII de 4 colunas (tempo, ax, ay, az em m/s2) --
    o mesmo formato do Anexo A/E e do "ASCII records" do Signal Explorer.
  - DatetimeCsvImportPanel: CSV "DateTime,AX,AY,AZ" (g) do firmware
    VCI_nRF52840_accel_SD -- o mesmo que csv_datetime_to_annex_txt.py
    converte por linha de comando, aqui direto na interface.

Nenhuma duplica a UI de analise: todas so alimentam o mesmo FieldPanel
usado pela aba Directive e delegam pra la (mesma logica de calculo/exibicao
de sempre).
"""
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from . import fielddata as fd
from .qtworker import run_in_thread
from .theme import TEXT


class _ImportPanelBase(QWidget):
    """Esqueleto comum: estado de "ocupado", entrega do dia lido ao
    FieldPanel da aba Directive, e mensagens de erro. Subclasses so
    definem _build() (widgets proprios) e load() (dialogo de arquivo +
    chamada ao carregador de fielddata correspondente).
    """

    def __init__(self, field_panel, status, on_loaded=None):
        super().__init__()
        self.field_panel = field_panel
        self.status = status
        self.on_loaded = on_loaded
        self._busy_flag = False
        self._build()

    def _run(self, paths, job):
        self._busy_flag = True
        self.load_btn.setEnabled(False)
        self.info.setText('Reading...')
        self.status(f'Reading {len(paths)} file(s)...')
        run_in_thread(self, job, self._on_done, self._on_fail,
                      thread_attr='_load_thread', worker_attr='_load_worker')

    def _on_done(self, day):
        self._busy_flag = False
        self.load_btn.setEnabled(True)
        self.info.setText(
            f'{len(day.records)} file(s), {len(day.blocks):,} block(s), '
            f'{day.recorded_hours:.3f} h. Handed off to the Directive tab.')
        self.info.setStyleSheet(f'color:{TEXT};')
        self.status(f'Loaded {len(day.blocks):,} block(s).')
        # Alimenta o mesmo FieldPanel usado pela aba Directive -- reusa toda
        # a logica de calculo/exibicao de la, sem duplicar UI aqui.
        self.field_panel._on_load_done(day)
        if self.on_loaded:
            self.on_loaded()

    def _on_fail(self, tb):
        self._busy_flag = False
        self.load_btn.setEnabled(True)
        self.info.setText('Failed to read the file(s).')
        self.status('Could not read the file(s).')
        QMessageBox.critical(self, 'Could not read the file(s)',
                             tb.strip().splitlines()[-1])


class CsvImportPanel(_ImportPanelBase):
    """CSV continuo (Time_ms,AX,AY,AZ em g) -- sketches de bancada como o
    teste de 10Hz do Xiao, diferente do .m do registrador de campo (que
    vem em blocos de 5 s com lacuna).
    """

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

        self._run(paths, job)


class TxtImportPanel(_ImportPanelBase):
    """.txt ASCII de 4 colunas (tempo, ax, ay, az em m/s2) -- o mesmo
    formato do Anexo A/E e do sub-aba "ASCII records", montado aqui como um
    dia continuo para a Diretiva/NHO-09 em vez de uma Excitation isolada.
    """

    def _build(self):
        lay = QVBoxLayout(self)

        note = QLabel(
            'For the four-column ASCII record used by Annex A/E and read '
            'by the "ASCII records" sub-tab (time, ax, ay, az in m/s2). No '
            '5 s gap, so it is sliced into contiguous blocks (duty cycle '
            '~100%). The sample rate is read from the file\'s own uniform '
            'time base -- nothing to enter here. All selected files must '
            'come from the same recording session (same sample rate).')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        box = QGroupBox('ASCII .txt file(s)')
        form = QFormLayout(box)

        row = QHBoxLayout()
        self.load_btn = QPushButton('Load .txt file(s)...')
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
            self, 'Select ASCII .txt file(s) of one unit (one day)', '',
            'ASCII record (*.txt);;All files (*)')
        if not paths:
            return
        position = self.field_panel.position.currentData()

        def job(report):
            return fd.load_day_txt(paths, progress=report, position=position)

        self._run(paths, job)


class DatetimeCsvImportPanel(_ImportPanelBase):
    """CSV bruto 'DateTime,AX,AY,AZ' (g) do firmware VCI_nRF52840_accel_SD
    -- o mesmo formato que csv_datetime_to_annex_txt.py converte por linha
    de comando; aqui entra direto, sem passar por um arquivo .txt
    intermediario. Cada reboot do firmware (segmento de cabecalho repetido
    no CSV) vira um registro proprio dentro do dia.
    """

    def _build(self):
        lay = QVBoxLayout(self)

        note = QLabel(
            'For the raw "DateTime,AX,AY,AZ" CSV (acceleration in g) written '
            'by the VCI_nRF52840_accel_SD firmware -- the same file '
            'csv_datetime_to_annex_txt.py converts from the command line. '
            'Each firmware reboot (a repeated header line) is split into its '
            'own record automatically. Sample rate priority: the override '
            'below, if checked; otherwise the "# rate_hz=" line the firmware '
            'v2 writes into the file; otherwise an estimate from the '
            'DateTime column (millisecond resolution, coarse).')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        box = QGroupBox('CSV file(s)')
        form = QFormLayout(box)

        self.override_rate = QCheckBox('Override sample rate')
        form.addRow(self.override_rate)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(1.0, 10000.0)
        self.rate.setDecimals(1)
        self.rate.setSuffix(' Hz')
        self.rate.setValue(833.0)
        self.rate.setEnabled(False)
        self.rate.setToolTip(
            'Use only if the file has no "# rate_hz=" metadata line and the '
            'DateTime-based estimate is not trusted -- the value actually '
            'configured in the firmware (--rate) is exact; the timestamp is '
            'only approximate.')
        self.override_rate.toggled.connect(self.rate.setEnabled)
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
            self, 'Select DateTime CSV file(s) of one unit (one day)', '',
            'CSV (*.csv);;All files (*)')
        if not paths:
            return
        rate = self.rate.value() if self.override_rate.isChecked() else None
        position = self.field_panel.position.currentData()

        def job(report):
            return fd.load_day_csv_datetime(paths, rate=rate, progress=report,
                                            position=position)

        self._run(paths, job)
