"""Graphical interface of the Avaliador de Vibração Humana application."""
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSpinBox, QStatusBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from . import core, workbook
from .analysis import Runner
from .clause5_tab import ClauseFiveTab
from .explorer import SignalExplorer
from .field_explorer import FieldSignalExplorer
from .field_panel import FieldPanel
from .csv_import_panel import (
    CsvImportPanel, DatetimeCsvImportPanel, TxtImportPanel)
from .live_fft_panel import LiveFFTPanel
from .qtworker import Worker
from .theme import (ACCENT, BAD_COLOR, BG_ALT, BORDER, HIGHLIGHT, OK_COLOR,
                    TEXT, WARN_COLOR, apply_theme)
from .weighting_panel import WeightingPanel
from ._version import __version__

APP_NAME = 'Avaliador de Vibração Humana'
SERIES_COLORS = ['#1f6feb', '#c2255c', '#1b7a33', '#b25c00', '#6741d9', '#0b7285']


def _redraw_twice(canvas):
    """Forca o motor de constrained_layout a assentar (ver _FullscreenDialog)."""
    canvas.draw()
    canvas.draw()


class _FullscreenDialog(QDialog):
    """QDialog que redesenha o canvas a cada resize de verdade.

    resizeEvent so dispara depois que o widget ja tem o tamanho novo, entao
    e um sinal mais confiavel que um timer com atraso fixo para saber quando
    a transicao para tela cheia terminou.

    O redesenho duplo e necessario, nao cosmetico: o motor de constrained_
    layout reserva o espaco do eixo X com base no tamanho dos rotulos do
    eixo Y medido no desenho ANTERIOR. Com escala logaritmica (as abas de
    frequencia) os rotulos mudam de forma (ex. 10^-8) e de tamanho de uma
    hora para outra quando o canvas salta de tamanho pequeno para tela
    cheia; um so desenho usa a medida antiga e reserva zero espaco para o
    eixo X. O segundo desenho ja usa a medida certa. Series lineares
    (dominio do tempo) nao mudam de forma o bastante para expor o problema,
    o que bate com o que foi observado.
    """

    def __init__(self, parent, canvas):
        super().__init__(parent)
        self._canvas = canvas

    def resizeEvent(self, event):
        super().resizeEvent(event)
        _redraw_twice(self._canvas)


class Plot(QWidget):
    """Canvas com barra de navegacao e modo de tela cheia.

    Tela cheia por F11, Esc para sair, duplo clique no grafico alterna. A
    figura e o mesmo objeto nos dois modos, entao zoom e deslocamento
    aplicados sobrevivem a ida e volta.
    """

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(7, 4.4), dpi=100, layout='constrained')
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = NavigationToolbar2QT(self.canvas, self)
        btn = QPushButton('Full screen')
        btn.setObjectName('secondary')
        btn.setToolTip('F11, or double-click the plot')
        btn.clicked.connect(self.toggle_fullscreen)
        bar.addWidget(btn)
        lay.addWidget(bar)
        lay.addWidget(self.canvas)

        self._window = None
        self._saved = None
        self._force_reset = False
        self._last_kind = object()      # sentinel: primeira chamada sempre "muda"
        self.canvas.mpl_connect('button_press_event', self._on_click)
        QShortcut(QKeySequence('F11'), self, self.toggle_fullscreen)

    def _on_click(self, event):
        if event.dblclick:
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        if self._window is not None:
            self._window.close()
            return
        win = _FullscreenDialog(self.window(), self.canvas)
        win.setWindowTitle('Plot — press Esc or F11 to return')
        lay = QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = NavigationToolbar2QT(self.canvas, win)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        close_btn = QPushButton('✕')
        close_btn.setObjectName('secondary')
        close_btn.setToolTip('Close (Esc)')
        close_btn.setFixedWidth(28)
        close_btn.clicked.connect(win.close)
        bar.addWidget(close_btn)
        lay.addWidget(bar)
        lay.addWidget(self.canvas)
        QShortcut(QKeySequence('F11'), win, win.close)
        win.finished.connect(self._restore)
        self._window = win

        # Ajusta o tamanho da figura para o da tela ANTES de mostrar, em vez
        # de deixar o Qt redimensionar o widget pequeno e so depois corrigir.
        screen = win.screen() or QApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            dpi = self.figure.dpi
            self.figure.set_size_inches(geo.width() / dpi,
                                        max(1.0, (geo.height() - 40) / dpi),
                                        forward=True)
        # O motor de constrained_layout se adapta bem ao widget pequeno, mas
        # nao estava reservando espaco confiavel para o eixo X no salto de
        # tamanho para tela cheia (some especificamente nos graficos com
        # escala logaritmica). Em tela cheia a margem passa a ser FIXA:
        # sobra espaco de sobra em qualquer tamanho de tela, sem depender de
        # nenhum calculo automatico. Custo: nao se adapta ao conteudo, mas
        # so nesse modo -- o widget pequeno continua usando constrained.
        self.figure.set_layout_engine(None)
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.93,
                                    bottom=0.09)
        win.showFullScreen()
        QTimer.singleShot(0, lambda: _redraw_twice(self.canvas))

    def _restore(self):
        self._window = None
        self.figure.set_layout_engine('constrained')
        lay = self.layout()
        lay.addWidget(self.canvas)
        self.canvas.setParent(self)
        self.canvas.show()
        QTimer.singleShot(0, lambda: _redraw_twice(self.canvas))

    def _toolbar(self):
        return self.findChild(NavigationToolbar2QT)

    def _user_navigated(self):
        tb = self._toolbar()
        return tb is not None and len(getattr(tb, '_nav_stack', [])) > 1

    def reset_if_changed(self, kind):
        """Descarta o enquadramento salvo se `kind` mudou desde a ultima
        chamada -- equivalente a apertar Home no grafico.

        `kind` e qualquer valor que identifique o TIPO de grafico atual
        (tipicamente o indice do combo "Plot"). Trocar de eixo, janela ou
        escala dentro do MESMO tipo continua preservando o zoom
        normalmente (ver axes()); so a mudanca de dominio (ex. serie
        temporal -> FFT, onde os eixos antigos nao significam nada no
        grafico novo) forca o proximo axes()/finish() a comecar do zero.
        """
        if kind != self._last_kind:
            self._force_reset = True
            self._last_kind = kind

    def axes(self, n=1):
        """Limpa a figura e devolve os eixos, preservando o enquadramento.

        Redesenhar apaga a figura, e com ela o zoom que o usuario tinha
        aplicado. Como quase toda mudanca de controle redesenha, o zoom
        parecia nao funcionar. Aqui o enquadramento e guardado e reposto,
        desde que o usuario tenha de fato navegado; o botao Home volta ao
        enquadramento completo. reset_if_changed() pode forcar o descarte
        mesmo com navegacao, quando o TIPO de grafico muda.
        """
        old = self.figure.axes
        if self._force_reset:
            self._saved = None
            self._force_reset = False
        else:
            self._saved = ([(a.get_xlim(), a.get_ylim()) for a in old]
                           if old and len(old) == n and self._user_navigated()
                           else None)
        self.figure.clear()
        if n == 1:
            return [self.figure.add_subplot(111)]
        first = self.figure.add_subplot(n, 1, 1)
        return [first] + [self.figure.add_subplot(n, 1, i + 1, sharex=first)
                          for i in range(1, n)]

    def finish(self, *axes):
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
        tb = self._toolbar()
        if tb is not None:
            tb.update()                     # descarta o historico dos eixos mortos
            tb.push_current()               # enquadramento completo vira o Home
        if self._saved:
            for ax, (xl, yl) in zip(axes, self._saved):
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            if tb is not None:
                tb.push_current()
            self._saved = None
        self.canvas.draw_idle()


def _legend(ax, base=8):
    """Legenda compacta; sai do eixo quando ha muitas series."""
    n = len(ax.get_legend_handles_labels()[0])
    if n == 0:
        return
    if n <= 6:
        ax.legend(fontsize=base)
    else:
        ax.legend(fontsize=max(6, base - 1), ncol=2, loc='upper right',
                  framealpha=0.85, handlelength=1.6, columnspacing=1.0)


class MainWindow(QMainWindow):
    def __init__(self, data_dir):
        super().__init__()
        self.setWindowTitle(f'{APP_NAME} — v{__version__}')
        self.resize(1220, 830)
        self.store = core.TransferFunctionStore(data_dir)
        self.history = None
        self.assessment = None
        self.thread = None
        self.sens_thread = None
        self.sens_grid = None

        tabs = QTabWidget()
        tabs.addTab(self._build_home(), 'Home')
        self._setup_page = self._build_setup()
        tabs.addTab(self._setup_page, 'Annex A / Annex E')
        self.clause5 = ClauseFiveTab(Plot, self.status)
        self.explorer = SignalExplorer(Plot, self.status)
        # As abas que so fazem sentido depois de uma analise ficam escondidas
        # ate a primeira execucao, para nao oferecer paineis vazios.
        self._result_tabs = [
            (self._build_results(), 'Results'),
            (self._build_spectra(), 'Frequency domain'),
            (self._build_time(), 'Time domain'),
            (self._build_signal(), 'Signal processing'),
            (self._build_sensitivity(), 'Sensitivity'),
        ]
        tabs.addTab(self.clause5, 'Clause 5 / Annex C')
        # O explorador ganha uma sub-aba para os dados do registrador de campo.
        # O explorador de registros ASCII fica intocado na primeira sub-aba.
        self.field = FieldPanel(Plot, self.status)
        self.field_explorer = FieldSignalExplorer(Plot, self.status)
        explorer_tabs = QTabWidget()
        explorer_tabs.addTab(self.explorer, 'ASCII records')
        explorer_tabs.addTab(self.field_explorer, 'Field logger records (.m)')
        explorer_tabs.addTab(self.field, 'Directive')
        _to_directive = lambda: explorer_tabs.setCurrentWidget(self.field)
        self.csv_import = CsvImportPanel(
            self.field, self.status, on_loaded=_to_directive)
        explorer_tabs.addTab(self.csv_import, 'Continuous CSV import')
        self.txt_import = TxtImportPanel(
            self.field, self.status, on_loaded=_to_directive)
        explorer_tabs.addTab(self.txt_import, 'ASCII .txt import')
        self.datetime_csv_import = DatetimeCsvImportPanel(
            self.field, self.status, on_loaded=_to_directive)
        explorer_tabs.addTab(self.datetime_csv_import, 'DateTime CSV import')
        self.explorer_tabs = explorer_tabs
        tabs.addTab(explorer_tabs, 'Signal explorer')
        self.live_fft = LiveFFTPanel(Plot, self.status)
        tabs.addTab(self.live_fft, 'Live FFT (Xiao)')
        self.tabs = tabs
        self._tabs_shown = False
        self.setCentralWidget(tabs)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        sb = QStatusBar()
        sb.addPermanentWidget(self.progress)
        self.setStatusBar(sb)
        self.status('Load an exposure history to begin.')

    def status(self, msg):
        self.statusBar().showMessage(msg)

    # -- Home ------------------------------------------------------------
    def _build_home(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addStretch(1)

        col = QVBoxLayout()
        col.setSpacing(6)

        logo = QLabel()
        pix = QPixmap(str(asset('icon.png')))
        if not pix.isNull():
            logo.setPixmap(pix.scaledToWidth(
                200, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        col.addWidget(logo)

        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignCenter)
        f = title.font()
        f.setPointSize(f.pointSize() + 7)
        f.setBold(True)
        title.setFont(f)
        col.addWidget(title)

        subtitle = QLabel(
            f'v{__version__} — Grupo de Acústica e Vibrações em Seres '
            f'Humanos (GRAVI-SH), Universidade Federal de Minas Gerais')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName('muted')
        subtitle.setWordWrap(True)
        col.addWidget(subtitle)
        col.addSpacing(14)

        desc = QLabel(
            'Assessment of health risk from whole-body vibration and '
            'mechanical shock, implementing ISO 2631-5:2018 (Annex A/E '
            'dynamic compressive force and Clause 5/Annex C simplified '
            'dose), ISO 2631-1:1997 (frequency weighting), Fundacentro '
            'NHO-09 and EU Directive 2002/44/EC.')
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setMaximumWidth(620)
        col.addWidget(desc)
        col.addSpacing(18)

        guide = QGroupBox('Where to start')
        gl = QVBoxLayout(guide)
        for name, hint in [
            ('Annex A / Annex E', 'Shock/impact exposure history and '
             'spinal compressive-force risk (R^A).'),
            ('Clause 5 / Annex C', 'Simplified acceleration-dose risk '
             'model, for exposures outside the Annex A/E validity range.'),
            ('Signal explorer', 'Inspect raw records (ASCII, field logger '
             '.m, CSV) before building an exposure history, or run the '
             'NHO-09/Directive assessment on a field-logged day.'),
            ('Live FFT (Xiao)', 'Real-time spectrum from the Xiao IMU '
             'over a serial connection.'),
        ]:
            row = QLabel(f'<b>{name}</b> — {hint}')
            row.setWordWrap(True)
            gl.addWidget(row)
        guide.setMaximumWidth(620)
        col.addWidget(guide)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addLayout(col)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
        return page

    # -- Setup ---------------------------------------------------------
    def _build_setup(self):
        page = QWidget()
        outer = QVBoxLayout(page)

        src = QGroupBox('Exposure history')
        g = QGridLayout(src)
        self.file_label = QLabel('No spreadsheet loaded')
        self.file_label.setObjectName('muted')
        btn = QPushButton('Open spreadsheet...')
        btn.clicked.connect(self.open_history)
        g.addWidget(btn, 0, 0)
        g.addWidget(self.file_label, 0, 1)
        self.acc_label = QLabel('Acceleration folder: same as the spreadsheet')
        self.acc_label.setObjectName('muted')
        btn2 = QPushButton('Acceleration folder...')
        btn2.setObjectName('secondary')
        btn2.clicked.connect(self.choose_acc_dir)
        g.addWidget(btn2, 1, 0)
        g.addWidget(self.acc_label, 1, 1)
        g.setColumnStretch(1, 1)
        outer.addWidget(src)

        row = QHBoxLayout()
        kind = QGroupBox('Analysis type')
        kl = QVBoxLayout(kind)
        self.rb_orient = QRadioButton('Orientating (A.3.1)')
        self.rb_indiv = QRadioButton('Individual (Annex E)')
        self.rb_orient.setChecked(True)
        self.rb_orient.toggled.connect(self._sync_enabled)
        kl.addWidget(self.rb_orient)
        kl.addWidget(self.rb_indiv)
        note = QLabel('The orientating analysis fixes posture group 3, '
                      '110 kg / 1.85 m,\n45 years from age 20, 240 days/year, '
                      '4 h/day.')
        note.setObjectName('muted')
        kl.addWidget(note)
        kl.addStretch()
        row.addWidget(kind, 1)

        params = QGroupBox('Parameters')
        f = QFormLayout(params)
        self.birth = QSpinBox()
        self.birth.setRange(1900, 2100)
        self.birth.setValue(1950)
        f.addRow('Year of birth', self.birth)
        self.protection = QSpinBox()
        self.protection.setRange(1, 99)
        self.protection.setValue(50)
        self.protection.setSuffix(' %')
        f.addRow('Population protection', self.protection)
        self.subclass = QComboBox()
        self.subclass.addItems(['1 (default)', '2', '3'])
        f.addRow('Model subclass', self.subclass)
        self.unwrap = QCheckBox('Unwrap transfer function phase before '
                                'interpolating')
        self.unwrap.setToolTip(
            'Numerical correction. The model phase has a 2-pi jump near 20 Hz '
            'and the original software interpolates across it, which is wrong '
            'by up to 200% for the points around the jump. Correcting it '
            'shifts R by up to 1% away from the published reference values, so '
            'it is off by default.')
        self.unwrap.toggled.connect(
            lambda on: setattr(core, 'UNWRAP_PHASE', on))
        f.addRow('', self.unwrap)

        logo_row = QHBoxLayout()
        logo_row.addStretch()
        logo = QLabel()
        pix = QPixmap(str(asset('gravi-sh.png')))
        if not pix.isNull():
            logo.setPixmap(pix.scaledToWidth(230, Qt.SmoothTransformation))
        logo_row.addWidget(logo)
        f.addRow('', logo_row)
        row.addWidget(params, 1)
        outer.addLayout(row)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(185)
        outer.addWidget(self.summary)

        bar = QHBoxLayout()
        bar.addStretch()
        self.run_btn = QPushButton('Run analysis')
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_analysis)
        bar.addWidget(self.run_btn)
        outer.addLayout(bar)
        self._sync_enabled()
        return page

    def _sync_enabled(self):
        self.birth.setEnabled(self.rb_indiv.isChecked())

    # -- Results -------------------------------------------------------
    def _build_results(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        self.verdict = QLabel('No results yet')
        self.verdict.setAlignment(Qt.AlignCenter)
        self.verdict.setObjectName('verdict')
        self.verdict.setTextFormat(Qt.RichText)
        lay.addWidget(self.verdict)

        self.validity = QLabel('')
        self.validity.setWordWrap(True)
        self.validity.setTextFormat(Qt.RichText)
        self.validity.setVisible(False)
        lay.addWidget(self.validity)

        self.table = QTableWidget(6, 4)
        self.table.setHorizontalHeaderLabels(
            ['Disc level', 'S_d^A [MPa]', 'S_stat [MPa]', 'R^A'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(232)
        lay.addWidget(self.table)

        self.risk_plot = Plot()
        lay.addWidget(self.risk_plot, 1)

        bar = QHBoxLayout()
        bar.addStretch()
        self.export_btn = QPushButton('Export spreadsheet...')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        bar.addWidget(self.export_btn)
        lay.addLayout(bar)
        return page

    # -- Frequency domain ---------------------------------------------
    def _build_spectra(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        ctrl = QGroupBox('Plot controls')
        g = QGridLayout(ctrl)
        g.setContentsMargins(8, 4, 8, 4)
        g.setVerticalSpacing(3)
        g.addWidget(QLabel('Exposure'), 0, 0)
        self.spec_exp = QComboBox()
        g.addWidget(self.spec_exp, 0, 1)
        g.addWidget(QLabel('Plot'), 0, 2)
        self.spec_kind = QComboBox()
        self.spec_kind.addItems([
            'Excitation acceleration spectrum',
            'Excitation phase spectrum',
            'Coherence between contact points',
            'Compressive force spectrum',
            'Transfer functions in use',
            'Compression dose vs static stress',
            'Spectrogram (time-frequency)',
            'Shock Response Spectrum (SRS)',
        ])
        g.addWidget(self.spec_kind, 0, 3)

        self.spec_loc_label = QLabel('Contact point')
        g.addWidget(self.spec_loc_label, 1, 0)
        self.spec_loc = QComboBox()
        self.spec_loc.addItem('All', None)
        for i, n in enumerate(core.LOCATION_NAMES):
            self.spec_loc.addItem(n.capitalize(), i)
        g.addWidget(self.spec_loc, 1, 1)
        g.addWidget(QLabel('Axis'), 1, 2)
        self.spec_axis = QComboBox()
        self.spec_axis.addItem('Sum of x, y, z', None)
        for i, n in enumerate(core.AXIS_NAMES):
            self.spec_axis.addItem(n, i)
        g.addWidget(self.spec_axis, 1, 3)

        g.addWidget(QLabel('Disc level'), 2, 0)
        self.spec_disc = QComboBox()
        self.spec_disc.addItem('All', None)
        for i, n in enumerate(core.DISC_NAMES):
            self.spec_disc.addItem(n, i)
        g.addWidget(self.spec_disc, 2, 1)
        self.spec_raw = QCheckBox('Overlay unprocessed spectrum')
        g.addWidget(self.spec_raw, 2, 2, 1, 2)

        self.spec_full = QCheckBox('Full range up to Nyquist')
        g.addWidget(self.spec_full, 3, 0, 1, 2)
        self.log_scale = QCheckBox('Logarithmic amplitude')
        self.log_scale.setChecked(True)
        g.addWidget(self.log_scale, 3, 2, 1, 2)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        lay.addWidget(ctrl)

        spec = QGroupBox('Spectral estimation (display only)')
        sg = QGridLayout(spec)
        sg.setContentsMargins(8, 4, 8, 4)
        sg.setVerticalSpacing(3)
        sg.addWidget(QLabel('Method'), 0, 0)
        self.spec_method = QComboBox()
        self.spec_method.addItems(core.SPECTRUM_METHODS)
        sg.addWidget(self.spec_method, 0, 1)
        sg.addWidget(QLabel('Window'), 0, 2)
        self.spec_window = QComboBox()
        self.spec_window.addItems(list(core.WINDOW_FUNCTIONS))
        self.spec_window.setCurrentText('Hann')
        sg.addWidget(self.spec_window, 0, 3)

        sg.addWidget(QLabel('Segment length'), 1, 0)
        self.spec_nperseg = QComboBox()
        for n in (1024, 2048, 4096, 8192, 16384, 32768):
            self.spec_nperseg.addItem(f'{n} samples', n)
        self.spec_nperseg.setCurrentIndex(3)
        sg.addWidget(self.spec_nperseg, 1, 1)
        sg.addWidget(QLabel('Overlap'), 1, 2)
        self.spec_overlap = QComboBox()
        for o in (0.0, 0.25, 0.5, 0.75):
            self.spec_overlap.addItem(f'{o * 100:.0f} %', o)
        self.spec_overlap.setCurrentIndex(2)
        sg.addWidget(self.spec_overlap, 1, 3)

        sg.addWidget(QLabel('Band width'), 2, 0)
        self.spec_band = QDoubleSpinBox()
        self.spec_band.setRange(0.1, 10.0)
        self.spec_band.setSingleStep(0.25)
        self.spec_band.setValue(1.0)
        self.spec_band.setSuffix(' Hz')
        sg.addWidget(self.spec_band, 2, 1)

        sg.addWidget(QLabel('Quantity'), 2, 2)
        self.spec_quantity = QComboBox()
        self.spec_quantity.addItem('Acceleration', 'accel')
        self.spec_quantity.addItem('Velocity (1x integration)', 'velocity')
        self.spec_quantity.addItem('Displacement (2x integration)', 'displacement')
        self.spec_quantity.setToolTip(
            'Converts the displayed spectrum by frequency-domain '
            'integration (V(f)=A(f)/jw, U(f)=A(f)/-w^2). Visualization '
            'only -- does not change the dose/risk computation, which '
            'always uses acceleration/displacement internally as the norm '
            'defines. For the SRS plot, converts to pseudo-velocity/'
            'pseudo-displacement instead (SRS/omega_n per natural '
            'frequency) -- the standard shock-testing convention, not the '
            'same integration used for the other plots.')
        sg.addWidget(self.spec_quantity, 2, 3)

        self.spec_note = QLabel('')
        self.spec_note.setObjectName('muted')
        self.spec_note.setWordWrap(True)
        sg.addWidget(self.spec_note, 4, 0, 1, 4)
        sg.setColumnStretch(1, 1)
        sg.setColumnStretch(3, 1)
        lay.addWidget(spec)

        for w in (self.spec_method, self.spec_window, self.spec_nperseg,
                  self.spec_overlap, self.spec_quantity):
            w.currentIndexChanged.connect(self.update_spectra)
        self.spec_band.valueChanged.connect(self.update_spectra)

        self.spec_weight = WeightingPanel()
        self.spec_weight.changed.connect(self.update_spectra)
        lay.addWidget(self.spec_weight)

        for w in (self.spec_exp, self.spec_kind, self.spec_loc, self.spec_axis,
                  self.spec_disc):
            w.currentIndexChanged.connect(self.update_spectra)
        for w in (self.spec_raw, self.log_scale, self.spec_full):
            w.stateChanged.connect(self.update_spectra)

        for panel in (ctrl, spec, self.spec_weight):
            panel.setMaximumHeight(150)
        self.spec_plot = Plot()
        lay.addWidget(self.spec_plot, 1)
        lay.setStretch(lay.indexOf(self.spec_plot), 1)
        return page

    # -- Time domain ---------------------------------------------------
    def _build_time(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel('Exposure'))
        self.time_exp = QComboBox()
        top.addWidget(self.time_exp, 1)
        top.addWidget(QLabel('Disc level'))
        self.time_disc = QComboBox()
        self.time_disc.addItem('All', None)
        for i, n in enumerate(core.DISC_NAMES):
            self.time_disc.addItem(n, i)
        self.time_disc.setCurrentIndex(5)
        top.addWidget(self.time_disc)
        self.time_hist = QCheckBox('Peak histogram')
        self.time_hist.setChecked(True)
        top.addWidget(self.time_hist)
        top.addStretch()
        lay.addLayout(top)
        for w in (self.time_exp, self.time_disc):
            w.currentIndexChanged.connect(self.update_time)
        self.time_hist.stateChanged.connect(self.update_time)
        self.time_plot = Plot()
        lay.addWidget(self.time_plot, 1)
        return page

    # -- Signal processing --------------------------------------------
    def _build_signal(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel('Exposure'))
        self.proc_exp = QComboBox()
        self.proc_exp.currentIndexChanged.connect(self.update_processing)
        top.addWidget(self.proc_exp, 1)
        top.addWidget(QLabel('Contact point'))
        self.proc_loc = QComboBox()
        for i, n in enumerate(core.LOCATION_NAMES):
            self.proc_loc.addItem(n.capitalize(), i)
        self.proc_loc.currentIndexChanged.connect(self.update_processing)
        top.addWidget(self.proc_loc)
        top.addStretch()
        lay.addLayout(top)

        self.proc_table = QTableWidget(0, 2)
        self.proc_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.proc_table.horizontalHeader().setStretchLastSection(True)
        self.proc_table.setAlternatingRowColors(True)
        lay.addWidget(self.proc_table, 1)

        chain = QTextEdit()
        chain.setReadOnly(True)
        chain.setMaximumHeight(210)
        chain.setHtml(
            '<b>Processing chain applied to every excitation</b>'
            '<ol style="margin-left:-18px">'
            '<li>Read the four-column ASCII record; the last sample is dropped '
            'if the record length is odd.</li>'
            '<li>Discrete Fourier transform of the whole record at once: '
            'rectangular window, no segmenting, no overlap, no zero padding, '
            'no averaging. The result is a one-sided amplitude spectrum, the '
            'complex FFT scaled by 2/N with the DC and Nyquist bins halved, so '
            'a pure sine of amplitude a gives a bin of magnitude a.</li>'
            '<li>Band limiting: bins below 0.5 Hz are set to zero (Clause 5); '
            'bins above 100 Hz are discarded, which makes the effective sample '
            'rate of the reconstructed force signal 200 Hz.</li>'
            '<li><b>No frequency weighting is applied.</b> Annex A works with '
            'unweighted acceleration. The W<sub>k</sub> and W<sub>d</sub> '
            'filters belong to ISO 2631-1 and are not part of this method; the '
            'frequency dependence of the response is already carried by the '
            'transfer functions.</li>'
            '<li>Conversion to displacement, U(f) = A(f) / -(2&pi;f)&sup2;, '
            'because the Annex A transfer functions map displacement to force, '
            'not acceleration to force.</li>'
            '<li>Multiplication by the interpolated transfer function and '
            'summation over four contact points and three excitation '
            'directions, Formula (A.1).</li>'
            '<li>Inverse transform back to the time domain; the mean is then '
            'replaced by the static force from the finite element model.</li>'
            '</ol>'
            '<b>Why the averaged autospectrum is a display option only.</b> '
            'Welch averaging reduces the variance of the spectral estimate by '
            'averaging squared magnitudes, which discards phase. Without phase '
            'the force cannot be transformed back into a time history, and the '
            'compression peaks of Formula (A.2) are counted in the time '
            'domain. The autospectrum, the windows and the band analyses in '
            'the Frequency domain tab therefore change what is plotted, never '
            'what is computed. The computation always uses the single-block '
            'complex spectrum with a rectangular window, which is the '
            'configuration validated against the reference results.')
        lay.addWidget(chain)
        return page

    # -- Sensitivity -----------------------------------------------------
    def _build_sensitivity(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        note = QLabel(
            'Runs the same measured signal against all 6 posture groups and '
            'all 30 anthropometric models (180 combinations), applying the '
            'fixed orientating-analysis schedule (45 years from age 20, 240 '
            'days/year, 4 h/day) to each. Neither the posture nor the exact '
            'anthropometric model is well matched to a wheelchair user (see '
            'the documentation, Sec. 7.2 and 11.1) — this spread is the '
            'defensible result in that case, not a single number.')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        top = QHBoxLayout()
        top.addWidget(QLabel('Exposure'))
        self.sens_exp = QComboBox()
        top.addWidget(self.sens_exp, 1)
        self.sens_run_btn = QPushButton('Run sensitivity sweep')
        self.sens_run_btn.clicked.connect(self.run_sensitivity)
        top.addWidget(self.sens_run_btn)
        lay.addLayout(top)

        self.sens_summary = QLabel(
            'Select an exposure and run the sweep. Takes about 15-30 s.')
        self.sens_summary.setObjectName('muted')
        self.sens_summary.setWordWrap(True)
        lay.addWidget(self.sens_summary)

        body = QHBoxLayout()
        self.sens_table = QTableWidget(6, 5)
        self.sens_table.setHorizontalHeaderLabels(
            ['Disc level', 'min R^A', 'median R^A', 'max R^A', 'this exposure'])
        self.sens_table.verticalHeader().setVisible(False)
        self.sens_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sens_table.setAlternatingRowColors(True)
        self.sens_table.setMaximumWidth(560)
        body.addWidget(self.sens_table, 1)
        self.sens_plot = Plot()
        body.addWidget(self.sens_plot, 2)
        lay.addLayout(body, 1)
        return page

    # ------------------------------------------------------------------
    def open_history(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open exposure history', '',
            'Spreadsheets (*.xls *.xlsx);;All files (*)')
        if not path:
            return
        try:
            self.history = workbook.read_history(path)
            missing = self.history.resolve_paths(Path(path).parent)
        except Exception as exc:
            QMessageBox.critical(self, 'Could not read the spreadsheet', str(exc))
            return
        self.file_label.setText(Path(path).name)
        self.file_label.setObjectName('')
        self.file_label.setStyleSheet(f'color:{TEXT}; font-weight:600;')
        if self.history.years:
            self.birth.setValue(self.history.first_year - 20)
        self._describe(missing)
        self.run_btn.setEnabled(True)

    def choose_acc_dir(self):
        if self.history is None:
            QMessageBox.information(self, APP_NAME,
                                    'Load the spreadsheet first.')
            return
        d = QFileDialog.getExistingDirectory(self, 'Acceleration folder')
        if not d:
            return
        missing = self.history.resolve_paths(d)
        self.acc_label.setText(f'Acceleration folder: {d}')
        self.acc_label.setObjectName('')
        self.acc_label.setStyleSheet(f'color:{TEXT};')
        self._describe(missing)

    def _describe(self, missing):
        h = self.history
        lines = [f'<b>{len(h.exposures)}</b> exposure(s), '
                 f'<b>{len(h.patterns)}</b> pattern(s), '
                 f'<b>{len(h.years)}</b> year(s) '
                 f'({h.first_year} to {h.last_year}).']
        for name, e in h.exposures.items():
            lines.append(
                f'&nbsp;&nbsp;<b>{name}</b> &mdash; '
                f'{core.POSTURE_NAMES.get(e.posture, e.posture)}, '
                f't<sub>m</sub> = {e.measurement_time:g} s')
        masses = sorted({y.body_mass for y in h.years if y.body_mass})
        if masses:
            lines.append(f'Body mass: {min(masses):g} to {max(masses):g} kg')
        if missing:
            lines.append(f'<span style="color:{BAD_COLOR}">Files not found: '
                         + ', '.join(Path(m).name for m in missing[:6])
                         + ('...' if len(missing) > 6 else '')
                         + ' &mdash; use "Acceleration folder".</span>')
        self.summary.setHtml('<br>'.join(lines))

    # ------------------------------------------------------------------
    def run_analysis(self):
        if self.history is None:
            return
        indiv = self.rb_indiv.isChecked()
        birth = self.birth.value()
        sub = self.subclass.currentIndex() + 1
        prot = self.protection.value()
        runner = Runner(self.store)

        def job(report):
            runner.progress = report
            if indiv:
                return runner.run_individual(self.history, birth, sub, protection=prot)
            return runner.run_orienting(self.history, sub, protection=prot)

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self.thread = QThread()
        self.worker = Worker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.thread.start()

    def _on_progress(self, frac, msg):
        if frac < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(frac * 100))
        self.status(msg)

    def _finish_thread(self):
        self.thread.quit()
        self.thread.wait()
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

    def _on_fail(self, tb):
        self._finish_thread()
        self.status('The analysis failed.')
        box = QMessageBox(QMessageBox.Critical, 'Analysis error',
                          tb.strip().splitlines()[-1], parent=self)
        box.setDetailedText(tb)
        box.exec()

    def _reveal_result_tabs(self):
        if self._tabs_shown:
            return
        start = self.tabs.indexOf(self._setup_page) + 1
        for i, (widget, title) in enumerate(self._result_tabs, start=start):
            self.tabs.insertTab(i, widget, title)
        self._tabs_shown = True

    def _on_done(self, assessment):
        self._finish_thread()
        self._reveal_result_tabs()
        self.assessment = assessment
        self._show_results()
        self.export_btn.setEnabled(True)
        plottable = [n for n, r in assessment.exposures.items()
                     if getattr(r, 'forces', None) is not None]
        for combo in (self.spec_exp, self.time_exp, self.proc_exp):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(plottable)
            combo.blockSignals(False)
        self.update_spectra()
        self.update_time()
        self.update_processing()
        self.sens_exp.blockSignals(True)
        self.sens_exp.clear()
        self.sens_exp.addItems(list(assessment.exposures))
        self.sens_exp.blockSignals(False)
        self.sens_grid = None
        self.sens_summary.setText(
            'Select an exposure and run the sweep. Takes about 15-30 s.')
        self.sens_summary.setObjectName('muted')
        self.sens_table.clearContents()
        self.tabs.setCurrentIndex(self.tabs.indexOf(self._result_tabs[0][0]))
        self.status(f'{assessment.kind.capitalize()} analysis complete.')

    # ------------------------------------------------------------------
    def _show_results(self):
        a = self.assessment
        maxR, lvl = a.max_R, core.DISC_NAMES[a.critical_level]
        colour = OK_COLOR if maxR < core.RISK_LOW else (
            BAD_COLOR if maxR > core.RISK_HIGH else WARN_COLOR)
        self.verdict.setText(
            f'<div style="font-size:27px;font-weight:700;color:{colour}">'
            f'R<sup>A</sup> = {maxR:.3f}</div>'
            f'<div style="color:{TEXT};font-size:14px">critical level '
            f'<b>{lvl}</b> &mdash; {core.classify(maxR)}</div>'
            f'<div style="color:#4a5157;font-size:12px">{a.kind} analysis, '
            f'{a.n_years} year(s) from age {a.start_age}</div>')

        for i, name in enumerate(core.DISC_NAMES):
            vals = [name, f'{a.daily_dose[i]:.4f}',
                    f'{a.S_stat[i]:.4f}', f'{a.R[i]:.4f}']
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setForeground(QColor(TEXT))
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 3:
                    item.setForeground(QColor(
                        OK_COLOR if a.R[i] < core.RISK_LOW else
                        BAD_COLOR if a.R[i] > core.RISK_HIGH else WARN_COLOR))
                    fnt = item.font()
                    fnt.setBold(True)
                    item.setFont(fnt)
                if i == a.critical_level:
                    item.setBackground(QColor(HIGHLIGHT))
                self.table.setItem(i, c, item)

        self._show_validity()

        ax, = self.risk_plot.axes()
        x = np.arange(6)
        colours = [OK_COLOR if v < core.RISK_LOW else
                   BAD_COLOR if v > core.RISK_HIGH else WARN_COLOR for v in a.R]
        ax.bar(x, a.R, color=colours, width=0.6)
        ax.axhline(core.RISK_LOW, ls='--', lw=1, color=OK_COLOR)
        ax.axhline(core.RISK_HIGH, ls='--', lw=1, color=BAD_COLOR)
        ax.set_xticks(x, core.DISC_NAMES)
        ax.set_ylabel('$R^A$')
        ax.set_title('Risk factor per disc level')
        self.risk_plot.finish(ax)

    def _show_validity(self):
        """Criterio de 9,81 m/s2 da ISO 2631-5: acima dele os Anexos A e E
        nao se aplicam e valem a Clausula 5 e os Anexos C e D."""
        excs = {}
        for res in self.assessment.exposures.values():
            excs.update(res.excitations)
        chk = core.validity_check(excs)
        drift_msg = ''
        for name, rep in (self.assessment.drift or {}).items():
            drift_msg += (
                f'<br><b style="color:{BAD_COLOR}">Time base drift in '
                f'"{name}":</b> the records differ by '
                f'{rep["drift"] * 100:.2f}% in length. They were truncated to '
                f'the shortest one, discarding {rep["samples_dropped"]} '
                f'samples. Check the synchronisation between sensors.')
        self.validity.setVisible(True)
        if chk['valid'] and not drift_msg:
            self.validity.setStyleSheet(
                f'background:#e8f4ea; border:1px solid #9fc9a8; border-radius:5px;'
                f'padding:8px; color:{TEXT};')
            self.validity.setText(
                f'Validity check passed: highest W<sub>k</sub>-weighted z peak is '
                f'{chk["worst"]:.2f} m/s&sup2;, within the '
                f'{chk["limit"]:.2f} m/s&sup2; limit for Annexes A and E.'
                + drift_msg)
        else:
            worst = max(chk['peaks'], key=chk['peaks'].get)
            self.validity.setStyleSheet(
                f'background:#fdeaea; border:1px solid #e0a6a2; border-radius:5px;'
                f'padding:8px; color:{TEXT};')
            self.validity.setText(
                f'<b style="color:{BAD_COLOR}">Validity check failed.</b> '
                f'The W<sub>k</sub>-weighted z peak reaches '
                f'<b>{chk["worst"]:.2f} m/s&sup2;</b> in "{worst}", above the '
                f'{chk["limit"]:.2f} m/s&sup2; limit. ISO 2631-5 states that '
                f'Annexes A and E do not apply to such exposures; Clause 5 '
                f'and Annexes C and D apply instead &mdash; see the '
                f'"Clause 5 / Annex C" tab. The numbers below are still '
                f'computed, but they fall outside the validity range of the '
                f'method that produced them.' + drift_msg)

    # ------------------------------------------------------------------
    def _current(self, combo):
        if self.assessment is None or not combo.currentText():
            return None
        return self.assessment.exposures.get(combo.currentText())

    def _selected(self, combo, total):
        v = combo.currentData()
        return list(range(total)) if v is None else [v]

    def update_spectra(self):
        res = self._current(self.spec_exp)
        if res is None:
            return
        kind = self.spec_kind.currentIndex()
        # 0 accel spectrum, 1 phase, 2 coherence, 3 force spectrum,
        # 4 transfer functions, 5 dose vs static stress, 6 spectrogram,
        # 7 shock response spectrum.
        self.spec_plot.reset_if_changed(kind)
        self.spec_loc.setEnabled(kind in (0, 1, 2, 6, 7))
        self.spec_loc_label.setText(
            'Reference point' if kind == 2 else 'Contact point')
        self.spec_axis.setEnabled(kind in (0, 1, 2, 6, 7))
        self.spec_raw.setEnabled(kind == 0)
        self.spec_full.setEnabled(kind == 0)
        self.spec_disc.setEnabled(kind in (3, 4))
        method = self.spec_method.currentText()
        block = method == core.SPECTRUM_METHODS[0]
        for w in (self.spec_window, self.spec_nperseg, self.spec_overlap):
            w.setEnabled(kind in (0, 2, 6))
        self.spec_method.setEnabled(kind == 0)
        self.spec_band.setEnabled(kind == 0)
        self.spec_weight.setEnabled(kind == 0)
        self.spec_quantity.setEnabled(kind in (0, 6, 7))
        if kind == 0:
            self.spec_window.setEnabled(not block)
            self.spec_nperseg.setEnabled(not block)
            self.spec_overlap.setEnabled(not block)
            self.spec_band.setEnabled(method == core.SPECTRUM_METHODS[3])
            self.spec_raw.setEnabled(block)
            self.spec_full.setEnabled(block)
        logy = self.log_scale.isChecked()

        # _plot_force_spectrum/_plot_transfer/_plot_dose have no note of
        # their own; clear here so a note from a previous view (e.g. the
        # coherence averaging caveat) doesn't linger under an unrelated plot.
        self.spec_note.setText('')
        if kind == 0:
            self._plot_excitation(res, logy)
        elif kind == 1:
            self._plot_phase(res)
        elif kind == 2:
            self._plot_coherence(res)
        elif kind == 3:
            self._plot_force_spectrum(res, logy)
        elif kind == 4:
            self._plot_transfer(res, logy)
        elif kind == 5:
            self._plot_dose(res)
        elif kind == 6:
            self._plot_spectrogram(res, logy)
        else:
            self._plot_srs(res, logy)

    def _plot_excitation(self, res, logy):
        ax, = self.spec_plot.axes()
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        method = self.spec_method.currentText()
        block = method == core.SPECTRUM_METHODS[0]
        bands = method in core.SPECTRUM_METHODS[3:]
        raw = self.spec_raw.isChecked() and block
        note = ''

        for li in self._selected(self.spec_loc, 4):
            exc = res.excitations[names[li]]
            if exc.accel_spectrum is None or not np.any(exc.accel_spectrum):
                continue
            colour = SERIES_COLORS[li % len(SERIES_COLORS)]
            label = core.LOCATION_NAMES[li]
            if axis is not None:
                label += f' ({core.AXIS_NAMES[axis]})'
            mode = self.spec_weight.combination()
            if axis is None and mode in (1, 2):
                series = [(c, f'{label} ({core.AXIS_NAMES[c]})',
                           ['-', '--', ':'][c]) for c in range(3)]
                if mode == 2:
                    series.append((None, f'{label} (vector)', '-'))
            else:
                series = [(axis, label, '-')]
            for c, lbl, style in series:
                f, y, ylabel, note = core.analysis_spectrum(
                    exc, method, axis=c,
                    window=self.spec_window.currentText(),
                    nperseg=self.spec_nperseg.currentData(),
                    overlap=self.spec_overlap.currentData(),
                    band_width=self.spec_band.value(),
                    weighting=self.spec_weight.config(),
                    quantity=self.spec_quantity.currentData())
                if bands:
                    ax.step(f, y, where='mid', lw=1.2, ls=style, color=colour,
                            label=lbl)
                else:
                    ax.plot(f, y, lw=1.4 if c is None else 0.9, ls=style,
                            color=colour, label=lbl)
            if raw and exc.raw_spectrum is not None:
                full = (np.abs(exc.raw_spectrum).sum(1) if axis is None
                        else np.abs(exc.raw_spectrum[:, axis]))
                ax.plot(exc.raw_freq, full, lw=0.7, color=colour, alpha=0.28,
                        label=f'{label} - unprocessed')

        first = res.excitations[names[0]]
        if method == core.SPECTRUM_METHODS[4]:
            ax.set_xscale('log')
        else:
            ax.set_xlim(0, first.raw_freq[-1] if self.spec_full.isChecked()
                        and block else core.LOWPASS_HZ)
        if raw:
            for fc in (core.HIGHPASS_HZ, core.LOWPASS_HZ):
                ax.axvline(fc, ls=':', lw=1, color='#6c757d')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Excitation spectrum - {method}')
        if logy:
            ax.set_yscale('log')
        ax.legend(fontsize=8)
        self.spec_note.setText(note)
        self.spec_plot.finish(ax)

    def _plot_phase(self, res):
        ax, = self.spec_plot.axes()
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        if axis is None:
            ax.text(0.5, 0.5, 'Pick a single axis (x, y or z): the phase of '
                    'the vector sum has no direct physical meaning.',
                    ha='center', va='center', color='#6c757d', wrap=True,
                    transform=ax.transAxes)
            self.spec_note.setText('')
            self.spec_plot.finish(ax)
            return
        for li in self._selected(self.spec_loc, 4):
            exc = res.excitations[names[li]]
            if exc.accel_spectrum is None or not np.any(exc.accel_spectrum):
                continue
            f, ph = core.phase_spectrum(exc, axis)
            ax.plot(f, ph, lw=1.1, color=SERIES_COLORS[li % len(SERIES_COLORS)],
                    label=core.LOCATION_NAMES[li])
        ax.set_xlim(0, core.LOWPASS_HZ)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Phase [deg], unwrapped')
        ax.set_title(f'Excitation phase - {core.AXIS_NAMES[axis]}')
        ax.axhline(0, lw=0.6, color='#adb5bd')
        ax.legend(fontsize=8)
        self.spec_note.setText(
            'Phase of the single-block complex FFT (Section 3.2), unwrapped. '
            'This is the exact spectrum that feeds Formula (A.1); it has no '
            'display option because Welch averaging discards phase '
            '(see the Signal processing tab).')
        self.spec_plot.finish(ax)

    def _plot_coherence(self, res):
        ax, = self.spec_plot.axes()
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        ref_i = self.spec_loc.currentData()
        if axis is None or ref_i is None:
            ax.text(0.5, 0.5, 'Pick a reference contact point and a single '
                    'axis (x, y or z) to compute coherence against.',
                    ha='center', va='center', color='#6c757d', wrap=True,
                    transform=ax.transAxes)
            self.spec_note.setText('')
            self.spec_plot.finish(ax)
            return
        ref = res.excitations[names[ref_i]]
        if ref.accel.shape[0] == 0:
            ax.text(0.5, 0.5, f'"{core.LOCATION_NAMES[ref_i]}" has no '
                    'measurement (Zero) in this exposure.', ha='center',
                    va='center', color='#6c757d', transform=ax.transAxes)
            self.spec_note.setText('')
            self.spec_plot.finish(ax)
            return
        nperseg = self.spec_nperseg.currentData()
        overlap = self.spec_overlap.currentData()
        window = self.spec_window.currentText()
        min_navg = None
        for li in range(4):
            if li == ref_i:
                continue
            exc = res.excitations[names[li]]
            if exc.accel.shape[0] == 0:
                continue
            if exc.dt != ref.dt or exc.accel.shape[0] != ref.accel.shape[0]:
                continue
            f, gamma2, navg = core.coherence_spectrum(
                ref.accel[:, axis], exc.accel[:, axis], ref.dt,
                window=window, nperseg=nperseg, overlap=overlap)
            min_navg = navg if min_navg is None else min(min_navg, navg)
            ax.plot(f, gamma2, lw=1.1,
                    color=SERIES_COLORS[li % len(SERIES_COLORS)],
                    label=f'{core.LOCATION_NAMES[ref_i]} vs '
                          f'{core.LOCATION_NAMES[li]}')
        ax.set_xlim(0, core.LOWPASS_HZ)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Coherence $\gamma^2$')
        ax.set_title(f'Coherence vs {core.LOCATION_NAMES[ref_i]} - '
                     f'{core.AXIS_NAMES[axis]}')
        ax.legend(fontsize=8)
        if min_navg is None:
            note = ('No other contact point shares this one\'s sample rate '
                    'and length, so no pair could be compared.')
        elif min_navg < 4:
            note = (f'Only {min_navg} averaged segment(s): coherence is '
                    'close to 1 everywhere by construction and is not yet a '
                    'meaningful estimate. Shorten the segment length '
                    '(Segment length control) to get more averages.')
        else:
            note = (f'Magnitude-squared coherence from {min_navg} averaged '
                    f'Welch segments ({window} window). Meaningful only '
                    'because the four contact points share a verified '
                    'common time base (Section 9.5 / core.check_time_base) '
                    '- do not compare files that were not loaded together '
                    'as one exposure.')
        self.spec_note.setText(note)
        self.spec_plot.finish(ax)

    def _plot_force_spectrum(self, res, logy):
        ax, = self.spec_plot.axes()
        for dl in self._selected(self.spec_disc, 6):
            ax.plot(res.freq, np.abs(res.force_spectrum[dl]), lw=0.9,
                    color=SERIES_COLORS[dl % len(SERIES_COLORS)],
                    label=core.DISC_NAMES[dl])
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Force amplitude [N]')
        ax.set_title('Intervertebral compressive force spectrum')
        ax.set_xlim(0, 32)
        if logy:
            ax.set_yscale('log')
        ax.legend(fontsize=8, ncol=2)
        self.spec_plot.finish(ax)

    def _plot_transfer(self, res, logy):
        H, fm = self.store.posture(res.posture)
        ax1, ax2 = self.spec_plot.axes(2)
        for dl in self._selected(self.spec_disc, 6):
            h = H[res.k - 1, 0, 2, 2, dl]      # seat, z excitation, compression
            colour = SERIES_COLORS[dl % len(SERIES_COLORS)]
            ax1.plot(fm, np.abs(h), lw=1.1, color=colour,
                     label=core.DISC_NAMES[dl])
            ax2.plot(fm, np.unwrap(np.angle(h)), lw=1.1, color=colour)
        ax1.set_ylabel('|H| [N/m]')
        ax1.set_title(f'Transfer functions: seat displacement (z) to '
                      f'compressive force - model {res.k} of 30')
        if logy:
            ax1.set_yscale('log')
        ax1.legend(fontsize=8, ncol=3)
        ax2.set_ylabel('Phase [rad]')
        ax2.set_xlabel('Frequency [Hz]')
        self.spec_plot.finish(ax1, ax2)

    def _plot_dose(self, res):
        ax, = self.spec_plot.axes()
        x = np.arange(6)
        w = 0.35
        ax.bar(x - w / 2, res.S, w, label='$S^A$ measured', color=ACCENT)
        ax.bar(x + w / 2, res.S_stat, w, label='$S_{stat}$', color='#8a949d')
        ax.set_xticks(x, core.DISC_NAMES)
        ax.set_ylabel('MPa')
        ax.set_title('Compression dose and static stress per level')
        ax.legend(fontsize=9)
        self.spec_plot.finish(ax)

    def _spec_single_signal(self, res, ax, need_msg):
        """Resolve ponto de contato + eixo unicos para um plot que so faz
        sentido de um sinal por vez (espectrograma, SRS). Devolve exc/axis
        ou None (ja tendo desenhado a mensagem de aviso, se for o caso)."""
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        loc_i = self.spec_loc.currentData()
        if axis is None or loc_i is None:
            ax.text(0.5, 0.5, need_msg, ha='center', va='center',
                    color='#6c757d', wrap=True, transform=ax.transAxes)
            self.spec_note.setText('')
            self.spec_plot.finish(ax)
            return None
        exc = res.excitations[names[loc_i]]
        if exc.accel.shape[0] == 0:
            ax.text(0.5, 0.5, f'"{core.LOCATION_NAMES[loc_i]}" has no '
                    'measurement (Zero) in this exposure.', ha='center',
                    va='center', color='#6c757d', transform=ax.transAxes)
            self.spec_note.setText('')
            self.spec_plot.finish(ax)
            return None
        return exc, axis, loc_i

    def _plot_spectrogram(self, res, logy):
        ax, = self.spec_plot.axes()
        picked = self._spec_single_signal(
            res, ax, 'Pick a single contact point and a single axis (x, y '
            'or z): a spectrogram is one signal at a time.')
        if picked is None:
            return
        exc, axis, loc_i = picked
        qty = self.spec_quantity.currentData()
        t, f, S = core.spectrogram(
            exc.accel[:, axis], exc.dt,
            window=self.spec_window.currentText(),
            nperseg=self.spec_nperseg.currentData(),
            overlap=self.spec_overlap.currentData(),
            quantity=qty)
        keep = f <= core.LOWPASS_HZ
        z = S[keep]
        label = f'{core.QUANTITY_LABELS[qty]} amplitude [{core.QUANTITY_UNITS[qty]}]'
        if logy:
            with np.errstate(divide='ignore'):
                z = 20 * np.log10(np.maximum(z, 1e-12))
            label = f'{core.QUANTITY_LABELS[qty]} level [dB re 1 {core.QUANTITY_UNITS[qty]}]'
        mesh = ax.pcolormesh(t, f[keep], z, shading='auto', cmap='viridis')
        cb = self.spec_plot.figure.colorbar(mesh, ax=ax)
        cb.set_label(label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Spectrogram - {core.LOCATION_NAMES[loc_i]} '
                    f'({core.AXIS_NAMES[axis]})')
        self.spec_note.setText(
            f'{self.spec_nperseg.currentData()}-sample segments, '
            f'{self.spec_window.currentText()} window, '
            f'{self.spec_overlap.currentData() * 100:.0f}% overlap. Shows '
            'how the frequency content changes over time -- unlike Welch/'
            'octave views, which only give the whole-record average.')
        self.spec_plot.finish(ax)

    def _plot_srs(self, res, logy):
        ax, = self.spec_plot.axes()
        picked = self._spec_single_signal(
            res, ax, 'Pick a single contact point and a single axis (x, y '
            'or z): SRS is one signal at a time.')
        if picked is None:
            return
        exc, axis, loc_i = picked
        freqs, srs = core.shock_response_spectrum(exc.accel[:, axis], exc.dt)
        quantity = self.spec_quantity.currentData()
        y, ylabel = core.srs_quantity(freqs, srs, quantity)
        ax.plot(freqs, y, lw=1.3, color=SERIES_COLORS[loc_i % len(SERIES_COLORS)],
               label=core.LOCATION_NAMES[loc_i])
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('Natural frequency [Hz]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Shock Response Spectrum (Q=10) - '
                    f'{core.LOCATION_NAMES[loc_i]} ({core.AXIS_NAMES[axis]})')
        ax.legend(fontsize=8)
        note = ('Maximax absolute-acceleration SRS: single-DOF base-excited '
                'system, Q=10 (zeta=0.05), MIL-STD-810 convention. Severity as '
                'a function of the responding structure\'s natural frequency, '
                'not a single RMS/peak number. Frequency division: 12 points '
                'per octave (fractional-octave spacing, ASTM E1049 '
                'convention).')
        if quantity != 'accel':
            note += (' Pseudo-' + quantity + ' = SRS acceleration divided '
                    'by the natural frequency of each point\'s own '
                    'single-DOF system, not a frequency-domain integration '
                    'of the signal.')
        self.spec_note.setText(note)
        self.spec_plot.finish(ax)

    def update_time(self):
        res = self._current(self.time_exp)
        if res is None:
            return
        discs = self._selected(self.time_disc, 6)
        show_hist = self.time_hist.isChecked()
        axes = self.time_plot.axes(2 if show_hist else 1)
        ax1 = axes[0]

        for dl in discs:
            ax1.plot(res.time, res.forces[dl], lw=0.5,
                     color=SERIES_COLORS[dl % len(SERIES_COLORS)],
                     label=core.DISC_NAMES[dl])
        if len(discs) == 1:
            ax1.axhline(res.forces[discs[0]].mean(), ls='--', lw=1,
                        color=BAD_COLOR, label='static compressive force')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Force [N]')
        ax1.set_title('Compressive force')
        ax1.legend(fontsize=8, ncol=4)

        if show_hist:
            ax2 = axes[1]
            total = 0
            for dl in discs:
                pk = np.abs(res.peaks[dl])
                if not pk.size:
                    continue
                total += pk.size
                ax2.hist(pk, bins=50, histtype='stepfilled', alpha=0.5,
                         color=SERIES_COLORS[dl % len(SERIES_COLORS)],
                         label=core.DISC_NAMES[dl])
            ax2.set_yscale('log')
            ax2.set_xlabel('$|C_{dyn,i}|$ [N]')
            ax2.set_ylabel('Count')
            ax2.set_title(f'{total} compression peaks')
            ax2.legend(fontsize=8, ncol=4)
            self.time_plot.finish(ax1, ax2)
        else:
            self.time_plot.finish(ax1)

    def update_processing(self):
        res = self._current(self.proc_exp)
        if res is None:
            return
        name = list(res.excitations)[self.proc_loc.currentIndex()]
        report = core.processing_report(res.excitations[name])
        self.proc_table.setRowCount(len(report))
        for r, (k, v) in enumerate(report.items()):
            a = QTableWidgetItem(k)
            fnt = a.font()
            fnt.setBold(True)
            a.setFont(fnt)
            a.setForeground(QColor(TEXT))
            b = QTableWidgetItem(str(v))
            b.setForeground(QColor(TEXT))
            self.proc_table.setItem(r, 0, a)
            self.proc_table.setItem(r, 1, b)

    # -- Sensitivity -----------------------------------------------------
    def run_sensitivity(self):
        res = self._current(self.sens_exp)
        if res is None:
            return
        if not res.paths:
            QMessageBox.information(
                self, APP_NAME, 'This exposure has no source file recorded '
                '(it was not loaded from a spreadsheet), so the sweep '
                'cannot reread the acceleration data.')
            return

        def job(report):
            # Rele os arquivos: o espectro de uma exposicao ja processada e
            # liberado da memoria em historicos longos (analysis.py).
            excs, _ = core.read_exposure_files(res.paths, progress=report)
            return core.sensitivity_grid(
                self.store, excs, measurement_time=res.measurement_time,
                progress=report)

        self.sens_run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self._sens_res = res

        self.sens_thread = QThread()
        self.sens_worker = Worker(job)
        self.sens_worker.moveToThread(self.sens_thread)
        self.sens_thread.started.connect(self.sens_worker.run)
        self.sens_worker.progress.connect(self._on_progress)
        self.sens_worker.finished.connect(self._on_sensitivity_done)
        self.sens_worker.failed.connect(self._on_sensitivity_fail)
        self.sens_thread.start()

    def _finish_sens_thread(self):
        self.sens_thread.quit()
        self.sens_thread.wait()
        self.progress.setVisible(False)
        self.sens_run_btn.setEnabled(True)

    def _on_sensitivity_fail(self, tb):
        self._finish_sens_thread()
        self.status('The sensitivity sweep failed.')
        box = QMessageBox(QMessageBox.Critical, 'Sensitivity sweep error',
                          tb.strip().splitlines()[-1], parent=self)
        box.setDetailedText(tb)
        box.exec()

    def _on_sensitivity_done(self, grid):
        self._finish_sens_thread()
        self.sens_grid = grid
        self.status('Sensitivity sweep complete.')
        self._show_sensitivity()

    def _show_sensitivity(self):
        grid, res = self.sens_grid, self._sens_res
        R = grid['R']                                    # (postures, 30, 6)
        pi = grid['postures'].index(res.posture)
        official = R[pi, res.k - 1]

        by_posture = R.mean(axis=(1, 2))
        worst_p = grid['postures'][int(np.argmax(by_posture))]
        best_p = grid['postures'][int(np.argmin(by_posture))]
        self.sens_summary.setObjectName('')
        self.sens_summary.setStyleSheet(f'color:{TEXT};')
        self.sens_summary.setText(
            f'{R.size} R^A values from {len(grid["postures"])} postures x 30 '
            f'anthropometric models, same signal, same 45-year orientating '
            f'schedule. This exposure was run with '
            f'<b>{core.POSTURE_NAMES.get(res.posture, res.posture)}</b>, '
            f'model {res.k}/30. Averaged over models, posture '
            f'<b>{core.POSTURE_NAMES.get(worst_p, worst_p)}</b> gives the '
            f'highest R^A and <b>{core.POSTURE_NAMES.get(best_p, best_p)}</b> '
            f'the lowest — the spread below is not measurement noise, it is '
            f'model choice.')

        for i, name in enumerate(core.DISC_NAMES):
            col = R[:, :, i].ravel()
            vals = [name, f'{col.min():.3f}', f'{np.median(col):.3f}',
                    f'{col.max():.3f}', f'{official[i]:.3f}']
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setForeground(QColor(TEXT))
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 4:
                    item.setForeground(QColor(
                        OK_COLOR if official[i] < core.RISK_LOW else
                        BAD_COLOR if official[i] > core.RISK_HIGH else WARN_COLOR))
                    fnt = item.font()
                    fnt.setBold(True)
                    item.setFont(fnt)
                self.sens_table.setItem(i, c, item)

        ax, = self.sens_plot.axes()
        x = np.arange(6)
        box = ax.boxplot([R[:, :, i].ravel() for i in range(6)], positions=x,
                         widths=0.5, showfliers=False, patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('#dfe3e8')
            patch.set_edgecolor('#6c757d')
        for element in ('whiskers', 'caps', 'medians'):
            for line in box[element]:
                line.set_color('#6c757d')
        colours = [OK_COLOR if v < core.RISK_LOW else
                  BAD_COLOR if v > core.RISK_HIGH else WARN_COLOR
                  for v in official]
        ax.scatter(x, official, color=colours, zorder=5, s=70,
                  edgecolor='black', linewidth=0.8,
                  label='this exposure (as run)')
        ax.axhline(core.RISK_LOW, ls='--', lw=1, color=OK_COLOR)
        ax.axhline(core.RISK_HIGH, ls='--', lw=1, color=BAD_COLOR)
        ax.set_xticks(x, core.DISC_NAMES)
        ax.set_ylabel('$R^A$ (fixed 45-year orientating schedule)')
        ax.set_title('Dispersion across 6 postures x 30 anthropometric models')
        ax.legend(fontsize=8, loc='upper left')
        self.sens_plot.finish(ax)

    # ------------------------------------------------------------------
    def export(self):
        if self.assessment is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export results', 'iso2631-5_results.xlsx',
            'Excel spreadsheet (*.xlsx)')
        if not path:
            return
        a = self.assessment
        src = self.history.source
        row = [
            str(src) if src else '', '',
            ' '.join(f'{v / 100:g}' for v in core.ENDPLATE_AREA),
            self.birth.value() if a.kind == 'individual' else 1945,
            self.history.first_year, self.history.last_year,
            2,  # Formula (E.2) da norma -- unica opcao, ver ultimate_strength()
            self.protection.value(), 6, a.kind,
            *[round(float(v), 5) for v in a.R],
            round(a.max_R, 5), core.DISC_NAMES[a.critical_level],
            round(float(a.daily_dose[a.critical_level]), 5),
        ]
        try:
            workbook.write_results(path, [row])
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))
            return
        self.status(f'Results written to {Path(path).name}')


def asset(name):
    base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) \
        else Path(__file__).resolve().parent
    return base / 'assets' / name


def data_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'data'
    return Path(__file__).resolve().parent.parent / 'data'


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    icon = QIcon(str(asset('icon.png')))
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app)
    win = MainWindow(data_dir())
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
