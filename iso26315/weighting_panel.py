"""Painel de ponderacao por eixo, compartilhado pelas abas que desenham espectros."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QWidget)

from . import core
from .theme import BAD_COLOR, TEXT_MUTED


class WeightingPanel(QGroupBox):
    """Predefinicao, ponderacao por eixo e modo de combinacao dos eixos.

    Os fatores k vem das tabelas da ISO 2631-1 e nao sao editaveis: o usuario
    escolhe a ponderacao de cada eixo, e o k acompanha a predefinicao.
    """
    changed = Signal()

    COMBINE = ['Vector sum (one curve per record)', 'Separate curve per axis',
               'Both']
    VECTOR, PER_AXIS, BOTH = 0, 1, 2

    def __init__(self, title='Frequency weighting (display only)'):
        super().__init__(title)
        self._loading = False
        g = QGridLayout(self)
        g.setContentsMargins(8, 4, 8, 4)
        g.setVerticalSpacing(3)

        g.addWidget(QLabel('Preset'), 0, 0)
        self.preset = QComboBox()
        for name in core.WEIGHTING_PRESETS:
            self.preset.addItem(name)
        self.preset.addItem('Custom')
        g.addWidget(self.preset, 0, 1, 1, 3)

        self.filters, self.gains = {}, {}
        for col, ax in enumerate(core.AXIS_NAMES):
            g.addWidget(QLabel(f'Axis {ax}'), 1, col)
            box = QComboBox()
            box.addItem('None', None)
            for nm, cfg in core.WEIGHTINGS.items():
                box.addItem(nm, nm)
            box.currentIndexChanged.connect(self._on_filter)
            g.addWidget(box, 2, col)
            klay = QHBoxLayout()
            klay.setContentsMargins(0, 0, 0, 0)
            tag = QLabel('k =')
            tag.setObjectName('muted')
            klay.addWidget(tag)
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            spin.setMaximumWidth(80)
            spin.valueChanged.connect(self._on_filter)
            klay.addWidget(spin)
            klay.addStretch()
            holder = QWidget()
            holder.setLayout(klay)
            g.addWidget(holder, 3, col)
            self.filters[ax] = box
            self.gains[ax] = spin

        g.addWidget(QLabel('Axis combination'), 1, 3)
        self.combine = QComboBox()
        self.combine.addItems(self.COMBINE)
        self.combine.setCurrentIndex(self.VECTOR)
        self.combine.currentIndexChanged.connect(lambda *_: self.changed.emit())
        g.addWidget(self.combine, 2, 3)

        self.status = QLabel('')
        self.status.setWordWrap(True)
        g.addWidget(self.status, 4, 0, 1, 4)
        for c in range(4):
            g.setColumnStretch(c, 1)

        self.preset.currentIndexChanged.connect(self._on_preset)
        self.preset.setCurrentText(core.DEFAULT_PRESET)
        for ax in core.AXIS_NAMES:
            self.filters[ax].setEnabled(False)
            self.gains[ax].setEnabled(False)
        self._apply_preset()

    # ------------------------------------------------------------------
    def config(self):
        return {ax: (self.filters[ax].currentData(), self._k(ax))
                for ax in core.AXIS_NAMES}

    def combination(self):
        return self.combine.currentIndex()      # 0 separado, 1 soma, 2 ambos

    def _k(self, ax):
        return float(self.gains[ax].value())

    # ------------------------------------------------------------------
    def _on_preset(self):
        if self._loading:
            return
        # Os seletores por eixo so aceitam edicao em "Custom". Nas predefinicoes
        # eles mostram o que a norma manda e ficam travados, para evitar que
        # alguem altere um eixo e continue lendo a tabela como se fosse ISO.
        custom = self.preset.currentText() == 'Custom'
        for ax in core.AXIS_NAMES:
            self.filters[ax].setEnabled(custom)
            self.gains[ax].setEnabled(custom)
        if not custom:
            self._apply_preset()
        self._refresh_status()
        self.changed.emit()

    def _apply_preset(self):
        cfg = core.WEIGHTING_PRESETS.get(self.preset.currentText())
        if cfg is None:
            return
        self._loading = True
        for ax, (name, k) in cfg.items():
            self.filters[ax].setCurrentIndex(
                max(0, self.filters[ax].findData(name)))
            self.gains[ax].setValue(k)
        self._loading = False
        self._refresh_status()

    def _on_filter(self):
        if self._loading:
            return
        # So chega aqui em modo Custom, onde filtro e fator k sao livres. Se a
        # combinacao coincidir com uma predefinicao, o rodape passa a nomea-la,
        # mas os controles seguem editaveis: quem entrou em Custom fica em
        # Custom.
        self._refresh_status()
        self.changed.emit()

    def _refresh_status(self):
        cfg = self.config()
        if all(n is None for n, _ in cfg.values()):
            self.status.setStyleSheet(f'color:{TEXT_MUTED};')
            self.status.setText(
                'Unweighted. This is what ISO 2631-5 uses for S_d and R^A.')
            return
        name = core.preset_for(cfg)
        if name and self.preset.currentText() != 'Custom':
            self.status.setStyleSheet(f'color:{TEXT_MUTED};')
            self.status.setText(
                f'{name}. Affects the plotted curves and the metrics table; '
                f'S_d and R^A remain unweighted.')
        else:
            self.status.setStyleSheet(f'color:{BAD_COLOR}; font-weight:600;')
            match = core.preset_for(cfg)
            if match:
                self.status.setStyleSheet(f'color:{TEXT_MUTED};')
                self.status.setText(
                    f'Custom, currently identical to "{match}".')
            else:
                self.status.setText(
                    'Custom combination, not one of the standard presets. The '
                    'metrics below follow it, so they are not ISO 2631-1 '
                    'values.')
