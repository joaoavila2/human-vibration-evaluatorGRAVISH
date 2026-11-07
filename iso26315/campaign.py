"""Tabela de progresso de campanha de medicoes.

Registra cada sujeito / posicao / amplitude medida, com preenchimento
automatico a partir do painel Field Logger. Os dados podem ser salvos
e recarregados como CSV.
"""
import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .theme import TEXT

COLUMNS = [
    'Subject', 'Age (y)', 'Mass (kg)', 'Position',
    'Date', 'File(s)', 'Criterion',
    'Main value [m/s²]', 'VDVR [m/s^1.75]',
    'A(8) highest [m/s²]', 'Notes',
]
_COL = {name: i for i, name in enumerate(COLUMNS)}

# Posicao das colunas numericas (recebem alinhamento a direita)
_NUMERIC_COLS = {_COL['Age (y)'], _COL['Mass (kg)'],
                 _COL['Main value [m/s²]'], _COL['VDVR [m/s^1.75]'],
                 _COL['A(8) highest [m/s²]']}


class CampaignPanel(QWidget):
    """Painel de acompanhamento de campanha.

    O botao "Add from Field Logger" e' ligado de fora (ver gui.py); este
    painel so armazena e exibe os dados.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        note = QLabel(
            'Campaign tracker: log each subject and position measured. '
            '"Add from Field Logger" auto-fills from the current assessment; '
            'click any cell to edit, select rows and click "Delete" to remove. '
            'Save / Load as CSV to persist across sessions.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        outer.addWidget(note)

        bar = QHBoxLayout()
        self.add_btn = QPushButton('Add empty row')
        self.add_btn.clicked.connect(self._add_empty)
        bar.addWidget(self.add_btn)
        self.del_btn = QPushButton('Delete selected')
        self.del_btn.setObjectName('secondary')
        self.del_btn.clicked.connect(self._delete_selected)
        bar.addWidget(self.del_btn)
        bar.addStretch()
        self.save_btn = QPushButton('Save CSV…')
        self.save_btn.clicked.connect(self.save_csv)
        bar.addWidget(self.save_btn)
        self.load_btn = QPushButton('Load CSV…')
        self.load_btn.setObjectName('secondary')
        self.load_btn.clicked.connect(self.load_csv)
        bar.addWidget(self.load_btn)
        outer.addLayout(bar)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        for c in range(len(COLUMNS) - 1):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setWordWrap(True)
        self.table.itemChanged.connect(self._update_summary)
        outer.addWidget(self.table, 1)

        self.summary_label = QLabel('')
        self.summary_label.setObjectName('muted')
        outer.addWidget(self.summary_label)

    # ------------------------------------------------------------------
    def _cell_text(self, r, c):
        it = self.table.item(r, c)
        return it.text().strip() if it else ''

    def _set_cell(self, r, c, value, editable=True):
        it = QTableWidgetItem(str(value) if value is not None else '')
        it.setForeground(QColor(TEXT))
        if c in _NUMERIC_COLS:
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if not editable:
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, c, it)

    def _add_empty(self):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._set_cell(r, _COL['Date'], datetime.now().strftime('%Y-%m-%d'))
        self.table.blockSignals(False)
        self._update_summary()

    def _delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        reply = QMessageBox.question(
            self, 'Delete rows',
            f'Delete {len(rows)} selected row(s)?',
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.table.blockSignals(True)
        for r in rows:
            self.table.removeRow(r)
        self.table.blockSignals(False)
        self._update_summary()

    # ------------------------------------------------------------------
    def add_entry(self, entry: dict):
        """Insert one row from a dict keyed by COLUMNS labels.

        Unknown keys are silently ignored; missing keys leave cells blank.
        """
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, col in enumerate(COLUMNS):
            v = entry.get(col, '')
            self._set_cell(r, c, v if v is not None else '')
        self.table.blockSignals(False)
        self.table.scrollToItem(self.table.item(r, 0))
        self._update_summary()

    # ------------------------------------------------------------------
    def _update_summary(self, *_):
        n = self.table.rowCount()
        subjects = set()
        positions = set()
        for r in range(n):
            s = self._cell_text(r, _COL['Subject'])
            p = self._cell_text(r, _COL['Position'])
            if s:
                subjects.add(s)
            if p:
                positions.add(p)
        pos_str = ', '.join(sorted(positions)) if positions else '—'
        self.summary_label.setText(
            f'{n} entr{"y" if n == 1 else "ies"} — '
            f'{len(subjects)} unique subject(s) — '
            f'positions: {pos_str}')

    # ------------------------------------------------------------------
    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save campaign CSV', 'campaign.csv', 'CSV (*.csv)')
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(COLUMNS)
                for r in range(self.table.rowCount()):
                    w.writerow([self._cell_text(r, c)
                                 for c in range(len(COLUMNS))])
        except OSError as exc:
            QMessageBox.critical(self, 'Could not save', str(exc))

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load campaign CSV', '', 'CSV (*.csv)')
        if not path:
            return
        try:
            with open(path, newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                header = next(reader)
                rows = list(reader)
        except Exception as exc:
            QMessageBox.critical(self, 'Could not read CSV', str(exc))
            return
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        src_col_map = {name: i for i, name in enumerate(header)}
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col_name, dc in _COL.items():
                sc = src_col_map.get(col_name)
                if sc is not None and sc < len(row):
                    self._set_cell(r, dc, row[sc])
        self.table.blockSignals(False)
        self._update_summary()
