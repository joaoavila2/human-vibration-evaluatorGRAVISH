"""Utilitarios de layout: secoes recolhiveis e pagina com divisor.

Os paineis de controle ficam numa area rolavel acima do grafico, separada
dele por um divisor arrastavel, e podem ser escondidos por inteiro. Assim
nenhum controle e' comprimido ou cortado quando a janela e' pequena, e o
grafico ocupa todo o espaco que sobra.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QPushButton, QScrollArea, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)


class CollapsibleSection(QWidget):
    """Cabecalho clicavel que abre e fecha um widget de conteudo.

    Se o conteudo for um QGroupBox, o titulo dele passa para o cabecalho.
    """

    def __init__(self, title, content, expanded=False, parent=None):
        super().__init__(parent)
        if isinstance(content, QGroupBox):
            title = title or content.title()
            content.setTitle('')
            content.setFlat(True)
        self._title = title
        self.content = content
        self.header = QToolButton()
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setStyleSheet(
            'QToolButton { border: none; font-weight: 600; text-align: left;'
            ' padding: 3px 2px; }')
        self.header.toggled.connect(self._toggle)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.header)
        lay.addWidget(content)
        self._toggle(expanded)

    def _toggle(self, on):
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.header.setText(' ' + self._title)
        self.content.setVisible(on)


def settings_splitter(upper, lower, label='settings', upper_share=0.4):
    """Devolve um widget com `upper` (rolavel, escondivel) sobre `lower`."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(upper)

    split = QSplitter(Qt.Vertical)
    split.addWidget(scroll)
    split.addWidget(lower)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setChildrenCollapsible(False)
    split.setSizes([int(1000 * upper_share), int(1000 * (1 - upper_share))])

    toggle = QPushButton(f'Hide {label}')
    toggle.setObjectName('secondary')
    toggle.setCheckable(True)

    def _flip(hidden):
        scroll.setVisible(not hidden)
        toggle.setText(f'Show {label}' if hidden else f'Hide {label}')

    toggle.toggled.connect(_flip)

    box = QWidget()
    v = QVBoxLayout(box)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    v.addWidget(toggle, 0, Qt.AlignLeft)
    v.addWidget(split, 1)
    box.toggle_button = toggle
    box.splitter = split
    return box
