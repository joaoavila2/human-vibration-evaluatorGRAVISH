"""Graphical interface of the Avaliador de Vibração Humana application."""
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QInputDialog,
    QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSpinBox, QStatusBar,
    QMenu, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from . import core, freqplots, workbook
from . import references
from .layoututil import CollapsibleSection, settings_splitter
from .analysis import Runner
from .campaign import CampaignPanel
from .clause5_tab import ClauseFiveTab
from .batch_tab import BatchTab
from .explorer import SignalExplorer
from .field_panel import FieldPanel
from .live_fft_panel import LiveFFTPanel
from .signal_viewer import SignalViewer
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


# Vistas prontas da cascata 3D (elevacao, azimute). Nos graficos de cascata
# x e a frequencia, y o tempo e z a amplitude.
VIEWS_3D = [
    ('3D (default)', (35, -60)),
    ('Top: spectrogram (time x frequency)', (90, -90)),
    ('Front: spectrum (amplitude x frequency)', (0, -90)),
    ('Side: history (amplitude x time)', (0, 0)),
]


class Plot(QWidget):
    """Canvas com barra de navegacao e modo de tela cheia.

    Tela cheia por F11, Esc para sair, duplo clique no grafico alterna. A
    figura e o mesmo objeto nos dois modos, entao zoom e deslocamento
    aplicados sobrevivem a ida e volta.

    Graficos 3D giram arrastando com o botao esquerdo (com Pan e Zoom da
    barra desligados). O angulo escolhido sobrevive aos redesenhos, e o
    botao "3D view" da barra oferece vistas prontas.

    O botao "References" liga curvas de referencia (ISO 5982 e o modelo
    infantil de Giacomin 2007) nos graficos de transmissibilidade H1. A
    configuracao fica em self.references; quem desenha o H1 chama
    references.draw_references, e references_changed pede o redesenho.
    O botao "Estimators" faz o mesmo com self.estimators (H2, Hv, envelope
    H1-H2 e banda de erro), lido por freqplots.draw_transmissibility.
    """

    references_changed = Signal()

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
        self.view_btn = QToolButton()
        self.view_btn.setText('3D view')
        self.view_btn.setPopupMode(QToolButton.InstantPopup)
        self.view_btn.setToolTip(
            'Drag the 3D plot with the left mouse button to rotate it (Pan '
            'and Zoom must be off). The angle is kept when the plot is '
            'redrawn. This menu sets a ready view.')
        menu = QMenu(self.view_btn)
        for name, view in VIEWS_3D:
            menu.addAction(name, lambda v=view: self.set_view3d(*v))
        self.view_btn.setMenu(menu)
        self.view_btn.setEnabled(False)
        bar.addWidget(self.view_btn)
        self.references = {'iso5982': False, 'child_mass': None}
        self.ref_btn = QToolButton()
        self.ref_btn.setText('References')
        self.ref_btn.setPopupMode(QToolButton.InstantPopup)
        self.ref_btn.setToolTip(
            'Reference curves drawn over transmissibility (H1) plots, for '
            'comparison only. ISO 5982: adult, 49-93 kg, no backrest. '
            'Giacomin 2007: one-degree-of-freedom child model scaled by '
            'body mass.')
        ref_menu = QMenu(self.ref_btn)
        self.ref_iso = ref_menu.addAction(
            'ISO 5982 seat-to-head mean (adult, no backrest)')
        self.ref_iso.setCheckable(True)
        self.ref_iso.toggled.connect(self._toggle_iso5982)
        self.ref_child = ref_menu.addAction(
            'Child 1-DOF model (Giacomin 2007)...')
        self.ref_child.setCheckable(True)
        self.ref_child.triggered.connect(self._ask_child_mass)
        self.ref_btn.setMenu(ref_menu)
        bar.addWidget(self.ref_btn)
        self.estimators = dict(freqplots.ESTIMATOR_DEFAULTS)
        self.est_btn = QToolButton()
        self.est_btn.setText('Estimators')
        self.est_btn.setPopupMode(QToolButton.InstantPopup)
        self.est_btn.setToolTip(
            'Extra layers on transmissibility (H1) plots. H1 is exact with '
            'noise only on the output channel, H2 with noise only on the '
            'input; on a seated person both channels are noisy and the true '
            'value lies between them.')
        est_menu = QMenu(self.est_btn)
        for key, text in (('envelope', 'H1-H2 envelope (shaded)'),
                          ('h2', 'H2 = Gyy/Gyx (dashed)'),
                          ('hv', 'Hv, total least squares (dash-dot)'),
                          ('error', 'H1 random error band, ±2ε '
                                    '(hatched)')):
            act = est_menu.addAction(text)
            act.setCheckable(True)
            act.toggled.connect(lambda on, k=key: self._toggle_estimator(k, on))
        self.est_btn.setMenu(est_menu)
        bar.addWidget(self.est_btn)
        lay.addWidget(bar)
        lay.addWidget(self.canvas)

        self._view3d = VIEWS_3D[0][1]
        self._window = None
        self._fs_view_btn = None
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
        # Mirror the 3D view / References / Estimators buttons: share QMenus
        # so the same state is reflected in both the main and fullscreen bars.
        self._fs_view_btn = QToolButton()
        self._fs_view_btn.setText('3D view')
        self._fs_view_btn.setPopupMode(QToolButton.InstantPopup)
        self._fs_view_btn.setMenu(self.view_btn.menu())
        self._fs_view_btn.setEnabled(self.view_btn.isEnabled())
        bar.addWidget(self._fs_view_btn)
        fs_ref_btn = QToolButton()
        fs_ref_btn.setText('References')
        fs_ref_btn.setPopupMode(QToolButton.InstantPopup)
        fs_ref_btn.setMenu(self.ref_btn.menu())
        bar.addWidget(fs_ref_btn)
        fs_est_btn = QToolButton()
        fs_est_btn.setText('Estimators')
        fs_est_btn.setPopupMode(QToolButton.InstantPopup)
        fs_est_btn.setMenu(self.est_btn.menu())
        bar.addWidget(fs_est_btn)
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
        self._fs_view_btn = None
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

    def axes(self, n=1, projection=None):
        """Limpa a figura e devolve os eixos, preservando o enquadramento.

        Redesenhar apaga a figura, e com ela o zoom que o usuario tinha
        aplicado. Como quase toda mudanca de controle redesenha, o zoom
        parecia nao funcionar. Aqui o enquadramento e guardado e reposto,
        desde que o usuario tenha de fato navegado; o botao Home volta ao
        enquadramento completo. reset_if_changed() pode forcar o descarte
        mesmo com navegacao, quando o TIPO de grafico muda.
        """
        if projection is not None:
            # Eixos 3D: o enquadramento salvo e' 2D e nao se aplica, mas o
            # angulo de visao (girado com o mouse) e guardado e reposto.
            self._force_reset = False
            self._saved = None
            self._remember_view3d()
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection=projection)
            ax.view_init(*self._view3d)
            self.view_btn.setEnabled(True)
            if self._fs_view_btn is not None:
                self._fs_view_btn.setEnabled(True)
            return [ax]
        self._remember_view3d()
        self.view_btn.setEnabled(False)
        if self._fs_view_btn is not None:
            self._fs_view_btn.setEnabled(False)
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

    def _toggle_estimator(self, key, on):
        self.estimators[key] = on
        self.references_changed.emit()

    def _toggle_iso5982(self, on):
        self.references['iso5982'] = on
        self.references_changed.emit()

    def _ask_child_mass(self, checked):
        if not checked:
            self.references['child_mass'] = None
            self.references_changed.emit()
            return
        mass, ok = QInputDialog.getDouble(
            self, 'Child model', 'Body mass of the child [kg]\n'
            '(Giacomin 2007 regressions cover 6 to 108 kg; the measured '
            'children were 5 to 12 kg):',
            self.references['child_mass'] or 15.0, 3.0, 120.0, 1)
        if not ok:
            self.ref_child.setChecked(self.references['child_mass'] is not None)
            return
        self.references['child_mass'] = mass
        self.ref_child.setText(f'Child 1-DOF model (Giacomin 2007), {mass:g} kg...')
        self.references_changed.emit()

    def _axes3d(self):
        return [a for a in self.figure.axes if a.name == '3d']

    def _remember_view3d(self):
        for a in self._axes3d():
            self._view3d = (a.elev, a.azim)
            break

    def set_view3d(self, elev, azim):
        self._view3d = (elev, azim)
        for a in self._axes3d():
            a.view_init(elev, azim)
        self.canvas.draw_idle()

    def finish(self, *axes):
        for ax in axes:
            pass  # spines controlled by plot_style (all 4 visible, like MATLAB)
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
        self.campaign = CampaignPanel()
        self.field = FieldPanel(Plot, self.status, store=self.store,
                                campaign=self.campaign)
        explorer_tabs = QTabWidget()
        explorer_tabs.addTab(self.explorer, 'Signal records')
        explorer_tabs.addTab(self.field, 'Directive')
        explorer_tabs.addTab(self.campaign, 'Campaign')
        self.explorer_tabs = explorer_tabs
        tabs.addTab(explorer_tabs, 'Signal explorer')
        self.live_fft = LiveFFTPanel(Plot, self.status)
        tabs.addTab(self.live_fft, 'Live FFT (Xiao)')
        self.signal_viewer = SignalViewer(self.status)
        tabs.addTab(self.signal_viewer, 'Visualizador')
        self.batch = BatchTab(self.status)
        tabs.addTab(self.batch, 'Processamento em Lote')
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
        self.protection.setToolTip(
            'Formula (E.2) represents about 50% of the population (the norm '
            'value). Other values are NOT in ISO 2631-5: the ultimate strength '
            'is lowered following Seidel et al. (2008), about 1.5 MPa less for '
            '95%, so a higher protection gives a higher R.')
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
        self.regime_override = QComboBox()
        self.regime_override.addItems([
            'Auto (ISO 2631-5, 9.81 m/s²)',
            'Force: Annex A/E — vehicle vibration',
            'Force: Clause 5 / Annex C — shock / loss of contact',
        ])
        self.regime_override.setToolTip(
            'The ISO 2631-5 validity criterion (Clause 4 / A.2) routes exposures: '
            'band-limited z peak ≤ 9.81 m/s² → Annex A/E; '
            'above → Clause 5 / Annex C.\n'
            'Forcing overrides the routing label for documentation purposes — '
            'the Annex A/E computation still runs. NOT normative.')
        self.regime_override.currentIndexChanged.connect(
            lambda _: self._show_validity() if self.assessment is not None else None)
        f.addRow('Regime', self.regime_override)

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
        clear_setup_btn = QPushButton('Clear')
        clear_setup_btn.setObjectName('secondary')
        clear_setup_btn.setToolTip('Discard the loaded spreadsheet and all results.')
        clear_setup_btn.clicked.connect(self.clear_setup)
        bar.addWidget(clear_setup_btn)
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
        self.export_pdf_btn = QPushButton('Export PDF report...')
        self.export_pdf_btn.setObjectName('secondary')
        self.export_pdf_btn.setEnabled(False)
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        bar.addWidget(self.export_pdf_btn)
        lay.addLayout(bar)
        return page

    # -- Frequency domain ---------------------------------------------
    def _build_spectra(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        upper = QWidget()
        lay = QVBoxLayout(upper)
        lay.setContentsMargins(0, 0, 0, 0)

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
            'Transmissibility H1 (reference -> other points)',
            'Time series: raw acceleration vs time',
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
        self.spec_weight_curve = QCheckBox(
            'Overlay weighting curve (from the panel below)')
        self.spec_weight_curve.setToolTip(
            'Draws the frequency-weighting filter\'s own gain, normalised to '
            'its peak, on a secondary axis -- shows where the excitation '
            'energy sits relative to what the norm boosts/attenuates. Only '
            'drawn on the excitation acceleration spectrum, and only for '
            'axes with a weighting selected in the panel below.')
        g.addWidget(self.spec_weight_curve, 4, 0, 1, 4)
        self.spec_hp = QCheckBox('High-pass Butterworth pre-filter')
        self.spec_hp.setToolTip(
            'Zero-phase Butterworth high-pass (order 6, applied forward and '
            'backward) on the signals before computing coherence, '
            'transmissibility or the spectrogram. Removes slow drift and '
            'voluntary movement below the cutoff (e.g. 0.5 Hz for infant '
            'transport). View only: does not change dose or risk.')
        g.addWidget(self.spec_hp, 5, 0, 1, 2)
        self.spec_hp_fc = QDoubleSpinBox()
        self.spec_hp_fc.setRange(0.05, 20.0)
        self.spec_hp_fc.setSingleStep(0.1)
        self.spec_hp_fc.setValue(0.5)
        self.spec_hp_fc.setSuffix(' Hz cutoff')
        g.addWidget(self.spec_hp_fc, 5, 2)
        self.spec_waterfall = QCheckBox('3D waterfall (spectrogram)')
        self.spec_waterfall.setToolTip(
            'Draws the spectrogram as a waterfall: one spectrum line per '
            'time step, stacked along the time axis.')
        g.addWidget(self.spec_waterfall, 5, 3)
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
        self.spec_hp.toggled.connect(self.update_spectra)
        self.spec_hp_fc.valueChanged.connect(self.update_spectra)
        self.spec_waterfall.toggled.connect(self.update_spectra)

        self.spec_weight = WeightingPanel()
        self.spec_weight.changed.connect(self.update_spectra)
        lay.addWidget(CollapsibleSection(
            'Frequency weighting (display only)', self.spec_weight, False))

        limit_box = QGroupBox('Reference limit curve (optional)')
        self.spec_limit_box = limit_box
        limit_v = QVBoxLayout(limit_box)
        self.spec_limit_enable = QCheckBox(
            'Show a limit curve on the excitation acceleration spectrum')
        self.spec_limit_enable.setToolTip(
            'Overlays a user-defined upper-bound curve (frequency, limit) '
            'on the plot and flags whether the measured spectrum crosses '
            'it anywhere within the frequency range the curve covers. '
            'Only evaluated when a single contact point AND a single axis '
            'are picked above -- a limit check against several overlaid '
            'series at once would be ambiguous.')
        limit_v.addWidget(self.spec_limit_enable)

        table_row = QHBoxLayout()
        self.spec_limit_table = QTableWidget(0, 2)
        self.spec_limit_table.setHorizontalHeaderLabels(
            ['Frequency [Hz]', 'Limit (same units as the plot)'])
        self.spec_limit_table.verticalHeader().setVisible(False)
        self.spec_limit_table.setMaximumHeight(90)
        self.spec_limit_table.horizontalHeader().setStretchLastSection(True)
        table_row.addWidget(self.spec_limit_table, 1)
        btn_col = QVBoxLayout()
        add_row_btn = QPushButton('+ row')
        add_row_btn.setObjectName('secondary')
        add_row_btn.clicked.connect(self._add_limit_row)
        btn_col.addWidget(add_row_btn)
        del_row_btn = QPushButton('- row')
        del_row_btn.setObjectName('secondary')
        del_row_btn.clicked.connect(self._remove_limit_row)
        btn_col.addWidget(del_row_btn)
        btn_col.addStretch()
        table_row.addLayout(btn_col)
        limit_v.addLayout(table_row)

        self.spec_limit_status = QLabel('')
        self.spec_limit_status.setWordWrap(True)
        limit_v.addWidget(self.spec_limit_status)
        lay.addWidget(CollapsibleSection(
            'Reference limit curve (optional)', limit_box, False))

        self.spec_limit_enable.toggled.connect(self.update_spectra)
        self.spec_limit_table.itemChanged.connect(self._on_limit_table_changed)

        for w in (self.spec_exp, self.spec_kind, self.spec_loc, self.spec_axis,
                  self.spec_disc):
            w.currentIndexChanged.connect(self.update_spectra)
        for w in (self.spec_raw, self.log_scale, self.spec_full,
                  self.spec_weight_curve):
            w.stateChanged.connect(self.update_spectra)

        self.spec_plot = Plot()
        self.spec_plot.references_changed.connect(self.update_spectra)
        outer.addWidget(settings_splitter(
            upper, self.spec_plot, 'plot settings', upper_share=0.30), 1)
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
    def clear_setup(self):
        """Descarta a planilha carregada e todos os resultados do Anexo A/E."""
        self.history = None
        self.assessment = None
        self.file_label.setText('No spreadsheet loaded')
        self.file_label.setObjectName('muted')
        self.file_label.setStyleSheet('')
        self.acc_label.setText('Acceleration folder: same as the spreadsheet')
        self.acc_label.setObjectName('muted')
        self.acc_label.setStyleSheet('')
        self.summary.clear()
        self.run_btn.setEnabled(False)
        if self._tabs_shown:
            # Remove as sub-abas de resultado do Anexo A da barra de abas.
            # Iterar ao contrário para que remoções não deslocam índices das
            # seguintes.
            for widget, _ in reversed(self._result_tabs):
                idx = self.tabs.indexOf(widget)
                if idx >= 0:
                    self.tabs.removeTab(idx)
            self._tabs_shown = False
        self.status('Annex A / Annex E cleared.')

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
        self.export_pdf_btn.setEnabled(True)
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
        override = self.regime_override.currentIndex()  # 0=auto, 1=force A/E, 2=force C5

        # Per-axis peaks summary (blind spot of Section 16.1 — norm only checks z,
        # but x/y peaks help the user understand the full exposure character)
        axis_lines = []
        for name, xyz in (chk.get('peaks_xyz') or {}).items():
            labels = ['x', 'y', 'z']
            vals = ', '.join(
                f'{labels[i]}: {xyz[i]:.2f}' for i in range(len(xyz)))
            axis_lines.append(f'&nbsp;&nbsp;<b>{name}</b>: {vals} m/s&sup2; (band-limited)')
        per_axis_msg = ('<br><i>Per-axis peaks:</i><br>' + '<br>'.join(axis_lines)
                        if axis_lines else '')

        drift_msg = ''
        # Inicio de exposicao antes dos 20 anos: a reta da Formula (E.2) e
        # extrapolada (ver references.start_age_warning).
        age_msg = references.start_age_warning(
            getattr(self.assessment, 'start_age', 20), 'E')
        if age_msg:
            drift_msg += (f'<br><b style="color:{BAD_COLOR}">Outside the '
                          f'calibration domain:</b> {age_msg}')
        for name, rep in (self.assessment.drift or {}).items():
            drift_msg += (
                f'<br><b style="color:{BAD_COLOR}">Time base drift in '
                f'"{name}":</b> the records differ by '
                f'{rep["drift"] * 100:.2f}% in length. They were truncated to '
                f'the shortest one, discarding {rep["samples_dropped"]} '
                f'samples. Check the synchronisation between sensors.')
        self.validity.setVisible(True)

        forced_msg = ''
        effective_valid = chk['valid']
        if override == 1 and not chk['valid']:
            effective_valid = True
            forced_msg = (f'<br><b style="color:#9a6200">⚠ Regime forced to '
                          f'Annex A/E (vehicle vibration) &mdash; NOT normative. '
                          f'The ISO 2631-5 validity check failed for this exposure '
                          f'({chk["worst"]:.2f} m/s² &gt; '
                          f'{chk["limit"]:.2f} m/s²).</b>')
        elif override == 2 and chk['valid']:
            effective_valid = False
            forced_msg = (f'<br><b style="color:#9a6200">⚠ Regime forced to '
                          f'Clause 5 / Annex C (shock) &mdash; NOT normative. '
                          f'The ISO 2631-5 validity check passed for this exposure '
                          f'({chk["worst"]:.2f} m/s² &le; '
                          f'{chk["limit"]:.2f} m/s²).</b>')

        if effective_valid:
            has_warnings = bool(drift_msg) or bool(forced_msg)
            bg, border = (('#fff4e0', '#e6c38a') if has_warnings
                          else ('#e8f4ea', '#9fc9a8'))
            self.validity.setStyleSheet(
                f'background:{bg}; border:1px solid {border}; border-radius:5px;'
                f'padding:8px; color:{TEXT};')
            self.validity.setText(
                f'Validity check passed: highest band-limited (0.4–100 Hz) '
                f'z peak is {chk["worst"]:.2f} m/s&sup2;, within the '
                f'{chk["limit"]:.2f} m/s&sup2; limit for Annexes A and E.'
                + per_axis_msg + forced_msg + drift_msg)
        else:
            worst_name = max(chk['peaks'], key=chk['peaks'].get) if chk['peaks'] else ''
            self.validity.setStyleSheet(
                f'background:#fdeaea; border:1px solid #e0a6a2; border-radius:5px;'
                f'padding:8px; color:{TEXT};')
            self.validity.setText(
                f'<b style="color:{BAD_COLOR}">Validity check failed.</b> '
                f'The band-limited (0.4–100 Hz) z peak reaches '
                f'<b>{chk["worst"]:.2f} m/s&sup2;</b>'
                + (f' in &ldquo;{worst_name}&rdquo;' if worst_name else '') +
                f', above the {chk["limit"]:.2f} m/s&sup2; limit. '
                f'ISO 2631-5 states that Annexes A and E do not apply to such '
                f'exposures; Clause 5 and Annexes C and D apply instead '
                f'&mdash; see the &ldquo;Clause 5 / Annex C&rdquo; tab. '
                f'The numbers below are still computed, but they fall outside '
                f'the validity range of the method that produced them.'
                + per_axis_msg + forced_msg + drift_msg)

    # ------------------------------------------------------------------
    def _current(self, combo):
        if self.assessment is None or not combo.currentText():
            return None
        return self.assessment.exposures.get(combo.currentText())

    def _selected(self, combo, total):
        v = combo.currentData()
        return list(range(total)) if v is None else [v]

    def _add_limit_row(self):
        r = self.spec_limit_table.rowCount()
        self.spec_limit_table.insertRow(r)
        self.spec_limit_table.setItem(r, 0, QTableWidgetItem(''))
        self.spec_limit_table.setItem(r, 1, QTableWidgetItem(''))

    def _remove_limit_row(self):
        r = self.spec_limit_table.currentRow()
        if r >= 0:
            self.spec_limit_table.removeRow(r)
        if self.spec_limit_enable.isChecked():
            self.update_spectra()

    def _on_limit_table_changed(self, *_):
        if self.spec_limit_enable.isChecked():
            self.update_spectra()

    def _limit_curve_data(self):
        freqs, vals = [], []
        for r in range(self.spec_limit_table.rowCount()):
            fi = self.spec_limit_table.item(r, 0)
            vi = self.spec_limit_table.item(r, 1)
            if fi is None or vi is None:
                continue
            ft, vt = fi.text().strip(), vi.text().strip()
            if not ft or not vt:
                continue
            try:
                freqs.append(float(ft.replace(',', '.')))
                vals.append(float(vt.replace(',', '.')))
            except ValueError:
                continue
        return np.array(freqs), np.array(vals)

    def _update_limit_status(self, res, names, method):
        """Chamada de dentro de _plot_excitation (kind==0): sobrepoe a
        curva-limite (se habilitada) e atualiza o rotulo de passa/falha.
        Recalcula o espectro do ponto/eixo escolhido de forma independente
        do loop principal de series, para nao precisar generalizar aquele
        loop (que trata soma/individual/vetor) so' por causa disto.
        """
        ax = self.spec_plot.figure.axes[0]
        if not self.spec_limit_enable.isChecked():
            self.spec_limit_status.setText('')
            self.spec_limit_status.setStyleSheet('')
            return
        lf, lv = self._limit_curve_data()
        axis_i = self.spec_axis.currentData()
        loc_i = self.spec_loc.currentData()
        self.spec_limit_status.setStyleSheet('')
        if len(lf) < 2:
            self.spec_limit_status.setText(
                'Enter at least two (frequency, limit) rows above.')
            return
        if axis_i is None or loc_i is None:
            self.spec_limit_status.setText(
                'Pick a single contact point and a single axis above to '
                'check against the limit curve.')
            return
        exc = res.excitations[names[loc_i]]
        f, y, _, _ = core.analysis_spectrum(
            exc, method, axis=axis_i,
            window=self.spec_window.currentText(),
            nperseg=self.spec_nperseg.currentData(),
            overlap=self.spec_overlap.currentData(),
            band_width=self.spec_band.value(),
            weighting=self.spec_weight.config(),
            quantity=self.spec_quantity.currentData())
        _, exceeded, worst_ratio = core.limit_curve_status(f, y, lf, lv)
        order = np.argsort(lf)
        ax.plot(lf[order], lv[order], ls='--', lw=1.3, color='#d6336c',
               label='Limit')
        ax.legend(fontsize=8)
        if np.isnan(worst_ratio):
            self.spec_limit_status.setText(
                'The limit curve does not overlap the plotted frequency '
                'range.')
        elif worst_ratio > 1.0:
            over = f[exceeded]
            where = f' near {over[0]:.1f} Hz' if len(over) else ''
            self.spec_limit_status.setText(
                f'⚠ Exceeds the limit curve (worst point '
                f'{worst_ratio:.2f}× the limit{where}).')
            self.spec_limit_status.setStyleSheet(
                f'background:#fdeaea; border:1px solid #e0a6a2; '
                f'border-radius:4px; padding:4px; color:{TEXT};')
        else:
            self.spec_limit_status.setText(
                f'✓ Within the limit curve (worst point '
                f'{worst_ratio:.2f}× the limit).')
            self.spec_limit_status.setStyleSheet(
                f'background:#e8f4ea; border:1px solid #9fc9a8; '
                f'border-radius:4px; padding:4px; color:{TEXT};')

    def update_spectra(self):
        res = self._current(self.spec_exp)
        if res is None:
            return
        kind = self.spec_kind.currentIndex()
        # 0 accel spectrum, 1 phase, 2 coherence, 3 force spectrum,
        # 4 transfer functions, 5 dose vs static stress, 6 spectrogram,
        # 7 shock response spectrum.
        self.spec_plot.reset_if_changed(kind)
        self.spec_loc.setEnabled(kind in (0, 1, 2, 6, 7, 8, 9))
        self.spec_loc_label.setText(
            'Reference point' if kind in (2, 8) else 'Contact point')
        self.spec_axis.setEnabled(kind in (0, 1, 2, 6, 7, 8, 9))
        self.spec_hp.setEnabled(kind in (2, 6, 8))
        self.spec_hp_fc.setEnabled(
            kind in (2, 6, 8) and self.spec_hp.isChecked())
        self.spec_waterfall.setEnabled(kind == 6)
        self.spec_raw.setEnabled(kind == 0)
        self.spec_full.setEnabled(kind == 0)
        self.spec_disc.setEnabled(kind in (3, 4))
        method = self.spec_method.currentText()
        block = method == core.SPECTRUM_METHODS[0]
        no_welch = method in (core.SPECTRUM_METHODS[0], core.SPECTRUM_METHODS[5])
        for w in (self.spec_window, self.spec_nperseg, self.spec_overlap):
            w.setEnabled(kind in (0, 2, 6, 8))
        self.spec_method.setEnabled(kind == 0)
        self.spec_band.setEnabled(kind == 0)
        self.spec_weight.setEnabled(kind == 0)
        self.spec_quantity.setEnabled(kind in (0, 6, 7))
        self.spec_limit_box.setEnabled(kind == 0)
        if kind == 0:
            self.spec_window.setEnabled(not no_welch)
            self.spec_nperseg.setEnabled(not no_welch)
            self.spec_overlap.setEnabled(not no_welch)
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
        elif kind == 7:
            self._plot_srs(res, logy)
        elif kind == 8:
            self._plot_transmissibility(res, logy)
        else:
            self._plot_timeseries_raw(res)

    def _spec_signal(self, exc, axis):
        """Sinal de um canal, com o passa-alta opcional da aba."""
        x = exc.accel[:, axis]
        if self.spec_hp.isChecked():
            x = core.highpass_butterworth(x, exc.dt, self.spec_hp_fc.value())
        return x

    def _plot_excitation(self, res, logy):
        ax, = self.spec_plot.axes()
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        method = self.spec_method.currentText()
        block = method == core.SPECTRUM_METHODS[0]
        bands = method in core.SPECTRUM_METHODS[3:5]
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
        ax.legend(fontsize=8, loc='upper left')
        band_note = self._band_diagnostic(res.excitations[names[self._selected(self.spec_loc, 4)[0]]], axis) \
            if self._selected(self.spec_loc, 4) else ''
        self.spec_note.setText((note + '  ' if note else '') + band_note)
        self._update_limit_status(res, names, method)
        self._overlay_weighting_curve(ax)
        self.spec_plot.finish(ax)

    BAND_EDGES = (0.5, 2.0, 8.0, 20.0, 80.0)
    BAND_LABELS = ('0.5-2 Hz (rigid-body/low-freq)', '2-8 Hz (whole-body resonance)',
                   '8-20 Hz (seat structure/impulse)', '20-80 Hz (high-freq/shake)')

    def _band_diagnostic(self, exc, axis):
        """Diagnostico informativo: RMS em 4 faixas de frequencia padrao,
        pra ver se o problema e' ressonancia baixa, impulso, ou vibracao de
        alta frequencia -- nao muda nenhum calculo de dose/risco."""
        c = 2 if axis is None else axis
        rms = core.band_rms(*core.welch_spectrum(exc.accel[:, c], exc.dt, kind='psd')[:2],
                             self.BAND_EDGES)
        parts = [f'{lbl.split(" ", 1)[0]}: {v:.3f}' for lbl, v in zip(self.BAND_LABELS, rms)]
        return 'Band RMS [m/s2] -- ' + ', '.join(parts)

    def _overlay_weighting_curve(self, ax):
        """Desenha, num eixo Y secundario, o ganho |H(f)| de cada ponderacao
        selecionada no painel abaixo (normalizado pelo pico) -- so' pra
        contexto visual de onde a energia da excitacao cai em relacao ao
        que a norma reforca/atenua. Nunca muda o calculo de dose/risco."""
        if not self.spec_weight_curve.isChecked():
            return
        cfg = self.spec_weight.config()
        seen = {}
        for ax_name, (name, _k) in cfg.items():
            if name is None or name in seen:
                continue
            seen[name] = ax_name
        if not seen:
            return
        xlim = ax.get_xlim()
        freq = np.linspace(max(xlim[0], 0.1), max(xlim[1], 1.0), 400)
        ax2 = ax.twinx()
        ax2.set_ylabel('Weighting gain (normalised to peak)')
        ax2.set_ylim(0, 1.15)
        styles = ['--', '-.', ':']
        for i, (name, ax_name) in enumerate(seen.items()):
            gain = np.abs(core.weighting_response(name, freq))
            peak = gain.max()
            if peak > 0:
                gain = gain / peak
            ax2.plot(freq, gain, ls=styles[i % len(styles)], lw=1.1,
                     color='#6c757d', alpha=0.8, label=f'{name} ({ax_name})')
        ax2.legend(fontsize=8, loc='upper right')

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
                self._spec_signal(ref, axis), self._spec_signal(exc, axis),
                ref.dt, window=window, nperseg=nperseg, overlap=overlap)
            min_navg = navg if min_navg is None else min(min_navg, navg)
            ax.plot(f, gamma2, lw=1.1,
                    color=SERIES_COLORS[li % len(SERIES_COLORS)],
                    label=f'{core.LOCATION_NAMES[ref_i]} vs '
                          f'{core.LOCATION_NAMES[li]}')
        if min_navg is not None and min_navg >= 2:
            cl = core.coherence_confidence_limit(min_navg)
            ax.axhline(cl, color='#c0392b', lw=1.0, ls='--',
                       label=f'95% confidence limit ({cl:.2f})')
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
            self._spec_signal(exc, axis), exc.dt,
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
        title = (f'Spectrogram - {core.LOCATION_NAMES[loc_i]} '
                 f'({core.AXIS_NAMES[axis]})')
        if self.spec_waterfall.isChecked():
            self._draw_waterfall(t, f[keep], z, label, title)
        else:
            mesh = ax.pcolormesh(t, f[keep], z, shading='auto', cmap='viridis')
            cb = self.spec_plot.figure.colorbar(mesh, ax=ax)
            cb.set_label(label)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)
        self.spec_note.setText(
            f'{self.spec_nperseg.currentData()}-sample segments, '
            f'{self.spec_window.currentText()} window, '
            f'{self.spec_overlap.currentData() * 100:.0f}% overlap. Shows '
            'how the frequency content changes over time -- unlike Welch/'
            'octave views, which only give the whole-record average.')
        if self.spec_waterfall.isChecked():
            self.spec_plot.finish(*self.spec_plot.figure.axes)
        else:
            self.spec_plot.finish(ax)

    def _draw_waterfall(self, t, f, z, zlabel, title, max_lines=100):
        """Cascata 3D: uma linha de espectro por instante, empilhadas no
        tempo. Limita a ~max_lines linhas (decimando no tempo) para manter
        a figura legivel e rapida."""
        from matplotlib import colormaps
        ax, = self.spec_plot.axes(projection='3d')
        step = max(1, int(np.ceil(len(t) / max_lines)))
        idx = np.arange(0, len(t), step)
        zmax = float(np.nanmax(z)) or 1.0
        cmap = colormaps['viridis']
        for j in idx[::-1]:
            level = float(np.nanmax(z[:, j])) / zmax
            ax.plot(f, np.full_like(f, t[j]), z[:, j], lw=0.8,
                    color=cmap(min(1.0, max(0.0, level))))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Time [s]')
        ax.set_zlabel(zlabel)
        ax.set_title(title)

    def _plot_transmissibility(self, res, logy):
        ax1, ax2 = self.spec_plot.axes(2)
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        ref_i = self.spec_loc.currentData()
        if axis is None or ref_i is None:
            ax1.text(0.5, 0.5, 'Pick a reference (input) contact point and '
                     'a single axis (x, y or z). H1 is computed from the '
                     'reference to every other point.', ha='center',
                     va='center', color='#6c757d', wrap=True,
                     transform=ax1.transAxes)
            self.spec_plot.finish(ax1, ax2)
            return
        ref = res.excitations[names[ref_i]]
        if ref.accel.shape[0] == 0:
            ax1.text(0.5, 0.5, f'"{core.LOCATION_NAMES[ref_i]}" has no '
                     'measurement (Zero) in this exposure.', ha='center',
                     va='center', color='#6c757d', transform=ax1.transAxes)
            self.spec_plot.finish(ax1, ax2)
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
            f, est, navg = core.transmissibility_estimates(
                self._spec_signal(ref, axis), self._spec_signal(exc, axis),
                ref.dt, window=window, nperseg=nperseg, overlap=overlap)
            min_navg = navg if min_navg is None else min(min_navg, navg)
            freqplots.draw_transmissibility(
                ax1, ax2, f, est, core.coherence_confidence_limit(navg),
                SERIES_COLORS[li % len(SERIES_COLORS)],
                core.LOCATION_NAMES[li], self.spec_plot.estimators)
        if min_navg is not None and min_navg >= 2:
            ax2.axhline(core.coherence_confidence_limit(min_navg),
                        color='#c0392b', lw=1.0, ls='--',
                        label='95% confidence limit')
        ax1.axhline(1.0, color='#6c757d', lw=0.6, ls=':')
        ax1.set_xlim(0, core.LOWPASS_HZ)
        ax2.set_xlim(0, core.LOWPASS_HZ)
        if logy:
            ax1.set_yscale('log')
        references.draw_references(ax1, self.spec_plot.references,
                                   core.LOWPASS_HZ)
        ax1.set_ylabel('|H1| (output / reference)')
        ax1.set_title(f'Transmissibility from {core.LOCATION_NAMES[ref_i]} - '
                      f'{core.AXIS_NAMES[axis]}')
        if ax1.get_legend_handles_labels()[0]:
            ax1.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel(r'Coherence $\gamma^2$')
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=8)
        if min_navg is None:
            note = ("No other contact point shares the reference's sample "
                    'rate and length, so no pair could be compared.')
        elif min_navg < 4:
            note = (f'Only {min_navg} averaged segment(s): H1 and coherence '
                    'are not yet meaningful. Shorten the segment length.')
        else:
            note = (f'H1 = Gxy/Gxx from {min_navg} averaged Welch segments. '
                    'Solid line: frequencies where coherence exceeds the '
                    '95% confidence limit (the only ones to trust); faint '
                    'line: below the limit. Values above 1 mean the output '
                    'point amplifies the reference.')
            extra = freqplots.estimator_note(self.spec_plot.estimators)
            if extra:
                note += ' ' + extra
        self.spec_note.setText(note)
        self.spec_plot.finish(ax1, ax2)

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

    def _plot_timeseries_raw(self, res):
        ax, = self.spec_plot.axes()
        names = list(res.excitations)
        axis = self.spec_axis.currentData()
        loc_i = self._selected(self.spec_loc, 4)
        if not loc_i:
            loc_i = [0]
        for li in loc_i:
            exc = res.excitations[names[li]]
            sig = exc.accel[:, axis]
            dt = exc.dt
            stride = max(1, len(sig) // 500_000)
            t = np.arange(0, len(sig), stride) * dt
            ax.plot(t, sig[::stride], lw=0.5,
                    color=SERIES_COLORS[li % len(SERIES_COLORS)],
                    label=core.LOCATION_NAMES[li])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Raw acceleration — {core.AXIS_NAMES[axis]} [m/s²]')
        ax.set_title('Raw input acceleration vs time (not filtered, not conditioned)')
        ax.legend(fontsize=8)
        self.spec_note.setText(
            'Informative plot. Shows the raw excitation signal as loaded, '
            'before any weighting or spine TF processing. Downsampled to '
            '≤ 500 000 points for display.')
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

    # ------------------------------------------------------------------
    def export_pdf(self):
        """Exporta PDF: página de resultados (tabela R^A) + gráficos selecionados."""
        if self.assessment is None:
            return
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime as _dt
        from PySide6.QtCore import Qt as _Qt

        # ── Diálogo: metadados + seleção de gráficos ─────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Exportar PDF — Annex A / Annex E')
        dlg.setMinimumWidth(520)
        vl = QVBoxLayout(dlg)
        vl.setSpacing(10)
        vl.setContentsMargins(16, 16, 16, 12)

        _lbl_info = QLabel('Informações para o relatório')
        _lbl_info.setStyleSheet('font-weight: bold; font-size: 13px;')
        vl.addWidget(_lbl_info)
        from PySide6.QtWidgets import QLineEdit, QFrame, QProgressDialog
        fm = QFormLayout()
        fm.setContentsMargins(0, 0, 0, 0)
        meta_fields = {}
        for key, label, placeholder in [
            ('subject_id', 'Identificação do sujeito', ''),
            ('date', 'Data da medição', _dt.today().strftime('%Y-%m-%d')),
            ('operator', 'Operador', ''),
        ]:
            le = QLineEdit()
            le.setText(placeholder)
            fm.addRow(label, le)
            meta_fields[key] = le
        vl.addLayout(fm)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #c4d0e4;')
        vl.addWidget(sep)

        _lbl_plots = QLabel('Gráficos a incluir (além da tabela de resultados)')
        _lbl_plots.setStyleSheet('font-weight: bold;')
        vl.addWidget(_lbl_plots)
        plot_options = [
            ('Frequency domain: acceleration spectrum (Welch/Hann)', 0),
            ('Frequency domain: transfer functions', 3),
            ('Frequency domain: compressive force spectrum', 2),
            ('Risk vs daily dose', None),
        ]
        checks = []
        cb_welch = QCheckBox(plot_options[0][0]); cb_welch.setChecked(True)
        checks.append((cb_welch, 'spec', 0))
        vl.addWidget(cb_welch)
        cb_tf = QCheckBox(plot_options[1][0]); cb_tf.setChecked(False)
        checks.append((cb_tf, 'spec', 3))
        vl.addWidget(cb_tf)
        cb_force = QCheckBox(plot_options[2][0]); cb_force.setChecked(False)
        checks.append((cb_force, 'spec', 2))
        vl.addWidget(cb_force)
        cb_time = QCheckBox('Time domain: compression peaks'); cb_time.setChecked(False)
        vl.addWidget(cb_time)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet('color: #c4d0e4;')
        vl.addWidget(sep2)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText('Exportar PDF')
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        meta = {k: v.text().strip() for k, v in meta_fields.items()}

        path, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF report', 'iso2631-5_report.pdf', 'PDF (*.pdf)')
        if not path:
            return

        import matplotlib as _mpl
        _rc_save = {k: _mpl.rcParams[k] for k in ['font.family']}
        prog = None
        try:
            _mpl.rcParams['font.family'] = 'serif'

            # ── Página 1: tabela de resultados (A4 retrato) ───────────────
            a = self.assessment
            BLACK = 'black'; WHITE = 'white'; GREY = '#f2f2f2'
            RULE  = '#333333'; TXT = 'black'; MUTED = '#555555'
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
                    edgecolor=(ec or fc), linewidth=lw,
                    transform=ax.transAxes, clip_on=False, zorder=1))

            def T(x, y, s, **kw):
                if y < FOOTER_Y:
                    return
                ax.text(x, y, s, transform=ax.transAxes, clip_on=False,
                        zorder=2, **kw)

            def hline(y, lw=0.8, color=RULE):
                ax.axhline(y, lw=lw, color=color, clip_on=False, zorder=1)

            # Cabeçalho
            R(0, 0.960, 1, 0.040, BLACK)
            T(0.015, 0.980, 'Avaliador de Vibração Humana — Annex A / Annex E (ISO 2631-5)',
              color=WHITE, fontsize=9.5, fontweight='bold', va='center')
            T(0.985, 0.980, _dt.now().strftime('%Y-%m-%d  %H:%M'),
              color='#cccccc', fontsize=7.5, va='center', ha='right')
            hline(0.960, lw=1.2, color=BLACK)

            meta_str = '  ·  '.join(v for v in [
                meta.get('subject_id'), meta.get('date'), meta.get('operator')] if v)
            if meta_str:
                T(0.015, 0.948, meta_str, color=MUTED, fontsize=8, va='center')
                hline(0.934, lw=0.5)

            y = 0.918
            T(0.015, y, 'Resultado — Risco Espinal (R^A)',
              color=BLACK, fontsize=9, fontweight='bold', va='bottom')
            hline(y - 0.004, lw=0.8)
            y -= 0.024

            from .theme import OK_COLOR, WARN_COLOR, BAD_COLOR
            ra_colour = (OK_COLOR if a.max_R < core.RISK_LOW else
                         BAD_COLOR if a.max_R > core.RISK_HIGH else WARN_COLOR)
            T(0.015, y, f'Rᴀ máximo = {a.max_R:.3f}   ({core.DISC_NAMES[a.critical_level]})',
              color=ra_colour, fontsize=14, fontweight='bold', va='top')
            y -= 0.042

            # Tabela por nível de disco
            T(0.015, y, 'Rᴀ por nível de disco',
              color=BLACK, fontsize=8.5, fontweight='bold', va='bottom')
            hline(y - 0.003, lw=0.6)
            y -= 0.010

            headers = ['Disco', 'S_d^A [MPa]', 'S_stat [MPa]', 'R^A']
            col_x = [0.015, 0.30, 0.55, 0.76]
            ROW_H = 0.024
            R(0, y - ROW_H, 1, ROW_H, GREY, RULE, 0.4)
            for cx, h in zip(col_x, headers):
                T(cx, y - ROW_H * 0.55, h, color=BLACK, fontsize=7.5,
                  va='center', fontweight='bold')
            y -= ROW_H

            for ri in range(self.table.rowCount()):
                items = [self.table.item(ri, c) for c in range(self.table.columnCount())]
                if not any(it and it.text().strip() for it in items):
                    continue
                bg = GREY if ri % 2 == 0 else WHITE
                R(0, y - ROW_H, 1, ROW_H, bg, RULE, 0.3)
                ra_item = self.table.item(ri, 3)
                try:
                    r_val = float(ra_item.text()) if ra_item else 0.0
                except ValueError:
                    r_val = 0.0
                clr = (OK_COLOR if r_val < core.RISK_LOW else
                       BAD_COLOR if r_val > core.RISK_HIGH else WARN_COLOR)
                for ci, (cx, it) in enumerate(zip(col_x, items)):
                    txt = it.text() if it else ''
                    T(cx, y - ROW_H * 0.55, txt,
                      color=clr if ci == 3 else TXT, fontsize=7.5, va='center',
                      fontweight='bold' if ci == 3 else 'normal')
                y -= ROW_H

            # Rodapé
            hline(FOOTER_Y, lw=0.6)
            T(0.015, FOOTER_Y / 2, 'Avaliador de Vibração Humana · ISO 2631-5 Annex A / Annex E',
              color=MUTED, fontsize=6.5, va='center')
            T(0.985, FOOTER_Y / 2, 'GRAVI-SH · UFMG',
              color=MUTED, fontsize=6.5, ha='right', va='center')

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

                # Páginas de gráficos espectrais (A4 paisagem)
                orig_spec_kind = self.spec_kind.currentIndex()
                orig_spec_size = self.spec_plot.figure.get_size_inches()
                for cb, kind, idx in checks:
                    if not cb.isChecked():
                        continue
                    if kind == 'spec':
                        self.spec_kind.setCurrentIndex(idx)
                        self.update_spectra()
                        self.spec_plot.figure.set_size_inches(11.69, 8.27)
                        pdf.savefig(self.spec_plot.figure, bbox_inches='tight', dpi=150)
                        self.spec_plot.figure.set_size_inches(*orig_spec_size)
                self.spec_kind.setCurrentIndex(orig_spec_kind)
                self.update_spectra()

                # Página de domínio do tempo
                if cb_time.isChecked():
                    self.update_time()
                    orig_time_size = self.time_plot.figure.get_size_inches()
                    self.time_plot.figure.set_size_inches(11.69, 8.27)
                    pdf.savefig(self.time_plot.figure, bbox_inches='tight', dpi=150)
                    self.time_plot.figure.set_size_inches(*orig_time_size)

        except Exception as _e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(_e))
            return
        finally:
            if prog is not None:
                prog.close()
            _mpl.rcParams.update(_rc_save)

        self.status(f'PDF report saved to {Path(path).name}.')


def asset(name):
    base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) \
        else Path(__file__).resolve().parent
    return base / 'assets' / name


def data_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'data'
    return Path(__file__).resolve().parent.parent / 'data'


def main():
    from . import plot_style
    plot_style.apply()
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
