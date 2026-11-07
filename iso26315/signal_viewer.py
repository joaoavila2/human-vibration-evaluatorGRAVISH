"""Aba Visualizador — leitura e visualizacao de sinais brutos.

Suporta:
  .dat  — arquivos MATLAB (scipy.io.loadmat)
  .csv  — primeira coluna tempo ou DateTime; demais colunas = canais
  .txt  — ASCII 4 colunas: tempo ax ay az (formato registrador de campo)

Controles:
  - Abrir arquivo (botao ou arrastar)
  - Selecionar canais a exibir (checkboxes)
  - Dominio do tempo / dominio da frequencia (FFT magnitude)
  - Zoom com roda do mouse e arrastar
  - Tela cheia
  - Exportar PDF
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from .theme import ACCENT, BG_ALT, BORDER, TEXT

# cores dos canais (cicladas)
_CHAN_COLORS = ['#1f6feb', '#c2255c', '#1b7a33', '#b25c00', '#6741d9', '#0b7285']


# ---------------------------------------------------------------------------
# Leitores de arquivo
# ---------------------------------------------------------------------------

def _load_dat(path: Path):
    """Carrega arquivo MATLAB .dat; retorna (t, {nome: array})."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError('scipy nao instalado — instale com: pip install scipy')

    raw = sio.loadmat(str(path), squeeze_me=False)
    dado = raw.get('Dado')
    if dado is None:
        raise ValueError('Variavel "Dado" nao encontrada no arquivo.')

    t = dado['Time'][0, 0].ravel().astype(float)
    channels = dado['Channel'][0, 0]
    canais = {}
    for i in range(channels.shape[1]):
        ampl = channels[0, i]['Ampl']
        if ampl.size == 0:
            continue
        canais[f'Canal {i + 1}'] = ampl.ravel().astype(float)

    if not canais:
        raise ValueError('Nenhum canal com dados encontrado.')
    return t, canais


def _load_csv(path: Path):
    """Carrega CSV; primeira coluna = tempo (ou DateTime descartado)."""
    import csv
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = [r for r in reader if r]

    if not rows:
        raise ValueError('Arquivo CSV vazio.')

    arr = []
    for row in rows:
        try:
            arr.append([float(v) for v in row])
        except ValueError:
            continue  # pula linhas com texto (ex: cabecalho extra)

    arr = np.array(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError('CSV precisa de pelo menos 2 colunas numericas.')

    t = arr[:, 0]
    # converte ms -> s se a coluna de tempo parecer ser milissegundos
    if t[-1] > 1e4:
        t = t / 1000.0

    nomes = []
    if header:
        nomes = header[1:]
    nomes += [f'Canal {i + 1}' for i in range(len(nomes), arr.shape[1] - 1)]

    canais = {nomes[i]: arr[:, i + 1] for i in range(arr.shape[1] - 1)}
    return t, canais


def _load_txt(path: Path):
    """Carrega .txt ASCII 4 colunas: tempo ax ay az."""
    arr = np.loadtxt(str(path))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError('Arquivo .txt precisa de pelo menos 2 colunas.')
    t = arr[:, 0]
    if t[-1] > 1e4:
        t = t / 1000.0
    nomes = ['AX', 'AY', 'AZ'] if arr.shape[1] >= 4 else [f'Canal {i+1}' for i in range(arr.shape[1]-1)]
    canais = {nomes[i]: arr[:, i + 1] for i in range(min(arr.shape[1] - 1, len(nomes)))}
    return t, canais


def carregar_arquivo(path: Path):
    ext = path.suffix.lower()
    if ext == '.dat':
        return _load_dat(path)
    elif ext == '.csv':
        return _load_csv(path)
    elif ext == '.txt':
        return _load_txt(path)
    else:
        raise ValueError(f'Formato nao suportado: {ext}')


# ---------------------------------------------------------------------------
# Widget de plot reutilizavel (tempo ou frequencia)
# ---------------------------------------------------------------------------

class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(constrained_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_time(self, t, canais, selecao):
        self.ax.cla()
        for i, (nome, dados) in enumerate(canais.items()):
            if nome in selecao:
                cor = _CHAN_COLORS[i % len(_CHAN_COLORS)]
                self.ax.plot(t, dados, label=nome, color=cor, linewidth=0.9)
        self.ax.set_xlabel('Tempo [s]')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.35)
        self.draw()

    def plot_freq(self, t, canais, selecao):
        self.ax.cla()
        n = len(t)
        if n < 4:
            return
        dt = float(np.mean(np.diff(t)))
        freqs = np.fft.rfftfreq(n, dt)
        w = np.hanning(n)
        w_norm = np.sum(w)
        for i, (nome, dados) in enumerate(canais.items()):
            if nome in selecao:
                sinal = dados - dados.mean()
                espectro = np.abs(np.fft.rfft(sinal * w)) / w_norm * 2
                espectro[0] /= 2
                espectro[-1] /= 2
                cor = _CHAN_COLORS[i % len(_CHAN_COLORS)]
                self.ax.semilogy(freqs[1:], espectro[1:], label=nome,
                                 color=cor, linewidth=0.9)
        self.ax.set_xlabel('Frequência [Hz]')
        self.ax.set_ylabel('Amplitude (janela Hann)')
        self.ax.legend(fontsize=8)
        self.ax.grid(True, which='both', alpha=0.35)
        self.draw()


# ---------------------------------------------------------------------------
# Aba principal
# ---------------------------------------------------------------------------

class SignalViewer(QWidget):
    """Aba Visualizador — leitura e inspecao de sinais brutos."""

    def __init__(self, status_cb=None):
        super().__init__()
        self._status = status_cb or (lambda msg: None)
        self._t = None
        self._canais: dict[str, np.ndarray] = {}
        self._checkboxes: list[QCheckBox] = []
        self._arquivo_atual: Path | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Construção da UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Barra superior ---
        top = QHBoxLayout()

        btn_open = QPushButton('Abrir arquivo…')
        btn_open.clicked.connect(self._abrir)
        top.addWidget(btn_open)

        self._lbl_arquivo = QLabel('Nenhum arquivo carregado')
        self._lbl_arquivo.setStyleSheet(f'color: {TEXT}; font-style: italic;')
        top.addWidget(self._lbl_arquivo, 1)

        self._combo_dominio = QComboBox()
        self._combo_dominio.addItems(['Tempo', 'Frequência (FFT)'])
        self._combo_dominio.currentIndexChanged.connect(self._redesenhar)
        top.addWidget(QLabel('Domínio:'))
        top.addWidget(self._combo_dominio)

        btn_fullscreen = QPushButton('Tela cheia')
        btn_fullscreen.clicked.connect(self._tela_cheia)
        top.addWidget(btn_fullscreen)

        btn_pdf = QPushButton('Exportar PDF')
        btn_pdf.clicked.connect(self._exportar_pdf)
        top.addWidget(btn_pdf)

        root.addLayout(top)

        # --- Área central: canais (esquerda) + plot (direita) ---
        central = QHBoxLayout()

        # painel de canais
        self._canais_box = QGroupBox('Canais')
        self._canais_box.setFixedWidth(160)
        self._canais_layout = QVBoxLayout(self._canais_box)
        self._canais_layout.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._canais_box)
        scroll.setFixedWidth(175)
        central.addWidget(scroll)

        # plot
        plot_col = QVBoxLayout()
        self._canvas = _PlotCanvas()
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        plot_col.addWidget(self._toolbar)
        plot_col.addWidget(self._canvas)
        central.addLayout(plot_col, 1)

        root.addLayout(central, 1)

    # ------------------------------------------------------------------
    # Lógica
    # ------------------------------------------------------------------

    def _abrir(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Abrir sinal',
            str(Path.home()),
            'Arquivos de sinal (*.dat *.csv *.txt);;Todos (*.*)',
        )
        if not path:
            return
        self._carregar(Path(path))

    def _carregar(self, path: Path):
        try:
            t, canais = carregar_arquivo(path)
        except Exception as e:
            QMessageBox.critical(self, 'Erro ao abrir arquivo',
                                 f'{path.name}:\n{e}\n\n{traceback.format_exc()}')
            return

        self._t = t
        self._canais = canais
        self._arquivo_atual = path
        self._lbl_arquivo.setText(path.name)
        self._lbl_arquivo.setStyleSheet(f'color: {TEXT};')
        self._status(f'Visualizador: {path.name} — {len(t)} amostras, {len(canais)} canal(is)')

        # reconstrói checkboxes
        for cb in self._checkboxes:
            self._canais_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        for nome in canais:
            cb = QCheckBox(nome)
            cb.setChecked(True)
            cb.stateChanged.connect(self._redesenhar)
            self._canais_layout.addWidget(cb)
            self._checkboxes.append(cb)

        self._redesenhar()

    def _selecao(self) -> set[str]:
        return {cb.text() for cb in self._checkboxes if cb.isChecked()}

    def _redesenhar(self):
        if self._t is None or not self._canais:
            return
        sel = self._selecao()
        if not sel:
            return
        if self._combo_dominio.currentIndex() == 0:
            self._canvas.plot_time(self._t, self._canais, sel)
        else:
            self._canvas.plot_freq(self._t, self._canais, sel)

    def _tela_cheia(self):
        from PySide6.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle(
            self._arquivo_atual.name if self._arquivo_atual else 'Visualizador'
        )
        dlg.setWindowState(Qt.WindowMaximized)
        lay = QVBoxLayout(dlg)

        canvas2 = _PlotCanvas()
        toolbar2 = NavigationToolbar2QT(canvas2, dlg)
        lay.addWidget(toolbar2)
        lay.addWidget(canvas2)

        sel = self._selecao()
        if self._combo_dominio.currentIndex() == 0:
            canvas2.plot_time(self._t, self._canais, sel)
        else:
            canvas2.plot_freq(self._t, self._canais, sel)

        dlg.exec()

    def _exportar_pdf(self):
        if self._t is None:
            QMessageBox.information(self, 'Visualizador', 'Abra um arquivo primeiro.')
            return

        stem = self._arquivo_atual.stem if self._arquivo_atual else 'sinal'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Exportar PDF', str(Path.home() / f'{stem}_visualizador.pdf'),
            'PDF (*.pdf)',
        )
        if not path:
            return

        sel = self._selecao()
        prog = None
        try:
            from PySide6.QtWidgets import QProgressDialog
            prog = QProgressDialog('Gerando PDF…', None, 0, 0, self)
            prog.setWindowModality(Qt.WindowModal)
            prog.show()

            with PdfPages(path) as pdf:
                fig = Figure(figsize=(11, 5), constrained_layout=True)
                ax = fig.add_subplot(111)

                if self._combo_dominio.currentIndex() == 0:
                    for i, (nome, dados) in enumerate(self._canais.items()):
                        if nome in sel:
                            ax.plot(self._t, dados, label=nome,
                                    color=_CHAN_COLORS[i % len(_CHAN_COLORS)],
                                    linewidth=0.8)
                    ax.set_xlabel('Tempo [s]')
                    ax.set_ylabel('Amplitude')
                else:
                    n = len(self._t)
                    dt = float(np.mean(np.diff(self._t)))
                    freqs = np.fft.rfftfreq(n, dt)
                    w = np.hanning(n)
                    w_norm = np.sum(w)
                    for i, (nome, dados) in enumerate(self._canais.items()):
                        if nome in sel:
                            sinal = dados - dados.mean()
                            esp = np.abs(np.fft.rfft(sinal * w)) / w_norm * 2
                            esp[0] /= 2
                            esp[-1] /= 2
                            ax.semilogy(freqs[1:], esp[1:], label=nome,
                                        color=_CHAN_COLORS[i % len(_CHAN_COLORS)],
                                        linewidth=0.8)
                    ax.set_xlabel('Frequência [Hz]')
                    ax.set_ylabel('Amplitude (janela Hann)')

                titulo = self._arquivo_atual.name if self._arquivo_atual else 'Sinal'
                ax.set_title(titulo)
                ax.legend(fontsize=8)
                ax.grid(True, which='both', alpha=0.35)
                pdf.savefig(fig)

            self._status(f'PDF salvo: {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Erro ao exportar PDF', str(e))
        finally:
            if prog:
                prog.close()
