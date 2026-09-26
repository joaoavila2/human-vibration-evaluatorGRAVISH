"""Aba de FFT ao vivo: le o XIAO nRF52840 (firmware_xiao/live_stream) via
Serial e desenha o espectro em tempo real, sem passar pelo cartao SD.

Formato de cada amostra (ver o .ino): 14 bytes little-endian, uint16 de
sincronismo (0x55AA) + 3 floats (x, y, z em g). O sincronismo deixa
resincronizar se a leitura comecar no meio do stream.
"""
import struct
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from .theme import BAD_COLOR, OK_COLOR, TEXT

AXIS_COLORS = ['#1f6feb', '#c2255c', '#1b7a33']
AXIS_NAMES = ['x', 'y', 'z']

_SAMPLE_STRUCT = struct.Struct('<Hfff')   # sync, x, y, z
_SYNC_MARKER = 0x55AA
_BAUD = 921600


class _SerialReader(QObject):
    """Roda numa QThread separada: le a porta serial, resincroniza pelo
    marcador e empilha amostras num deque compartilhado. Tocar widget
    daqui de dentro corromperia a UI -- por isso so mexe no buffer e emite
    sinais, igual ao padrao do resto do app (ver qtworker.py)."""
    error = Signal(str)
    stopped = Signal()

    def __init__(self, port, buf, stats):
        super().__init__()
        self.port = port
        self.buf = buf
        self.stats = stats   # dict compartilhado: bytes_total, bytes_descartados
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            ser = serial.Serial(self.port, _BAUD, timeout=0.5)
        except Exception as e:
            self.error.emit(str(e))
            self.stopped.emit()
            return

        raw = bytearray()
        size = _SAMPLE_STRUCT.size
        try:
            while self._running:
                chunk = ser.read(4096)
                if not chunk:
                    continue
                self.stats['bytes_total'] += len(chunk)
                raw.extend(chunk)
                while len(raw) >= size:
                    sync = raw[0] | (raw[1] << 8)
                    if sync != _SYNC_MARKER:
                        del raw[0]
                        self.stats['bytes_descartados'] += 1
                        continue
                    _, x, y, z = _SAMPLE_STRUCT.unpack_from(raw, 0)
                    del raw[:size]
                    self.buf.append((x, y, z))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            ser.close()
            self.stopped.emit()


class LiveFFTPanel(QWidget):
    def __init__(self, plot_factory, status):
        super().__init__()
        self.status = status
        self.buf = deque(maxlen=8192)
        self.stats = {'bytes_total': 0, 'bytes_descartados': 0}
        self.thread = None
        self.reader = None
        self._build(plot_factory)

    def _build(self, plot_factory):
        lay = QVBoxLayout(self)

        note = QLabel(
            'Le o XIAO em tempo real pela Serial (firmware_xiao/live_stream, '
            'separado do firmware de gravacao no SD) e mostra o espectro da '
            'ultima janela de amostras. Nao grava nada -- e so pra ver o '
            'espectro ao vivo, por exemplo pra achar ressonancia ou '
            'confirmar que um choque aparece onde deveria antes de sair '
            'gravando um teste inteiro.')
        note.setWordWrap(True)
        note.setObjectName('warning')
        lay.addWidget(note)

        box = QGroupBox('Conexao')
        form = QFormLayout(box)

        port_row = QHBoxLayout()
        self.port = QComboBox()
        self._refresh_ports()
        port_row.addWidget(self.port, 1)
        refresh_btn = QPushButton('Refresh')
        refresh_btn.setObjectName('secondary')
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(refresh_btn)
        form.addRow('Porta:', port_row)

        self.rate = QDoubleSpinBox()
        self.rate.setRange(1.0, 10000.0)
        self.rate.setDecimals(1)
        self.rate.setSuffix(' Hz')
        self.rate.setValue(833.0)
        self.rate.setToolTip('Taxa configurada no firmware live_stream.ino '
                             '-- so afeta o eixo de frequencia do grafico.')
        form.addRow('Taxa da IMU:', self.rate)

        self.window_s = QDoubleSpinBox()
        self.window_s.setRange(0.5, 9.0)
        self.window_s.setDecimals(1)
        self.window_s.setSuffix(' s')
        self.window_s.setValue(2.0)
        self.window_s.setToolTip('Janela deslizante usada pro FFT -- mais '
                                 'longa da mais resolucao em frequencia, '
                                 'mais curta reage mais rapido.')
        form.addRow('Janela:', self.window_s)

        row = QHBoxLayout()
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self.start)
        row.addWidget(self.start_btn)
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setObjectName('secondary')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.stop_btn)
        row.addStretch()
        form.addRow(row)

        self.info = QLabel('Desconectado.')
        self.info.setWordWrap(True)
        form.addRow(self.info)

        lay.addWidget(box)

        self.plot = plot_factory()
        lay.addWidget(self.plot, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._redraw)

    def _refresh_ports(self):
        self.port.clear()
        if list_ports is None:
            self.port.addItem('pyserial nao instalado', None)
            return
        ports = list(list_ports.comports())
        if not ports:
            self.port.addItem('Nenhuma porta encontrada', None)
            return
        for p in ports:
            self.port.addItem(f'{p.device} — {p.description}', p.device)

    def start(self):
        if serial is None:
            QMessageBox.critical(self, 'pyserial ausente',
                                 'Instale a biblioteca pyserial (pip install '
                                 'pyserial) para usar esta aba.')
            return
        port = self.port.currentData()
        if not port:
            QMessageBox.warning(self, 'Sem porta',
                                'Selecione uma porta serial valida.')
            return

        self.buf.clear()
        self.stats['bytes_total'] = 0
        self.stats['bytes_descartados'] = 0
        self.thread = QThread()
        self.reader = _SerialReader(port, self.buf, self.stats)
        self.reader.moveToThread(self.thread)
        self.thread.started.connect(self.reader.run)
        self.reader.error.connect(self._on_error)
        self.reader.stopped.connect(self._on_stopped)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.port.setEnabled(False)
        self.info.setText(f'Conectado em {port}, aguardando amostras...')
        self.info.setStyleSheet(f'color:{TEXT};')
        self.status(f'FFT ao vivo: conectado em {port}')
        self.timer.start()

    def stop(self):
        if self.reader:
            self.reader.stop()
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.port.setEnabled(True)
        self.status('FFT ao vivo: parado')

    def _on_error(self, msg):
        self.info.setText(f'Erro: {msg}')
        self.info.setStyleSheet(f'color:{BAD_COLOR};')
        self.stop()

    def _on_stopped(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()

    def _redraw(self):
        n = len(self.buf)
        window_n = int(self.window_s.value() * self.rate.value())
        if n < 64:
            bt = self.stats['bytes_total']
            bd = self.stats['bytes_descartados']
            if bt == 0:
                diag = ('nenhum byte chegou da porta ainda -- confira se '
                        'gravou o live_stream.ino (nao o firmware de '
                        'gravacao) e se e a porta certa')
            elif bd > 200 and n == 0:
                diag = (f'{bt} byte(s) recebido(s), mas {bd} descartado(s) '
                        f'sem achar o marcador de sincronismo -- provavelmente '
                        f'esta chegando outra coisa na porta (texto de debug, '
                        f'firmware errado, ou baud/porta errados), nao o '
                        f'stream binario esperado')
            else:
                diag = f'{bt} byte(s) recebido(s) ate agora'
            self.info.setText(f'{n} amostra(s) decodificada(s). {diag}.')
            self.info.setStyleSheet(f'color:{TEXT};')
            return

        data = np.array(list(self.buf)[-window_n:], dtype=np.float64)
        n_used = len(data)
        rate = self.rate.value()

        self.info.setText(
            f'{n:,} amostra(s) no buffer, mostrando as ultimas '
            f'{n_used:,} ({n_used / rate:.1f}s).')
        self.info.setStyleSheet(f'color:{OK_COLOR};')

        ax, = self.plot.axes()
        from scipy.signal import get_window
        w = get_window('hann', n_used, fftbins=True)
        f = np.arange(n_used // 2 + 1) * rate / n_used
        for c, name in enumerate(AXIS_NAMES):
            col = data[:, c] - data[:, c].mean()
            X = np.fft.rfft(col * w)
            amp = np.abs(X) * 2.0 / w.sum()
            ax.plot(f, amp, lw=0.9, color=AXIS_COLORS[c], label=name)
        ax.set_xlim(0, min(rate / 2, 200))
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [g]')
        ax.set_title(f'Live FFT — janela de {n_used / rate:.1f}s')
        ax.legend(fontsize=8)
        self.plot.finish(ax)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
