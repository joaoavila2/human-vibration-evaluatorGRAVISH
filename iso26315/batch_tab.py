"""Aba de processamento em lote — Clause 5 / Annex C.

Cada arquivo CSV / DAT / TXT / .m de uma pasta representa uma avaliação
independente (um sujeito / uma sessão). Todos os arquivos são processados
com os mesmos parâmetros de exposição. O resultado fica numa tabela
exportável como CSV, com a opção de gerar uma ficha PDF por arquivo.
"""
from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QComboBox, QAbstractItemView,
)

from . import annexc
from . import fielddata as fd
from . import pdf_report
from .theme import ACCENT, BAD_COLOR, OK_COLOR, TEXT, WARN_COLOR

_EXTS = {'.csv', '.dat', '.txt', '.m'}


def _risk_colour(probability: float) -> str:
    return (OK_COLOR if probability < annexc.RISK_LOW else
            BAD_COLOR if probability > annexc.RISK_HIGH else WARN_COLOR)


def _risk_label(probability: float) -> str:
    return ('Low' if probability < annexc.RISK_LOW else
            'High' if probability > annexc.RISK_HIGH else 'Moderate')


class BatchTab(QWidget):
    """Processamento em lote: uma pasta, N arquivos, N avaliações."""

    def __init__(self, status=None):
        super().__init__()
        self._status = status or (lambda msg: None)
        self._folder: Path | None = None
        self._results: list[dict] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        intro = QLabel(
            '<b>Processamento em lote — Clause 5 / Annex C</b><br>'
            'Selecione uma pasta. Cada arquivo (.csv, .dat, .txt, .m) '
            'corresponde a uma avaliação independente e é processado com '
            'os parâmetros de exposição abaixo.')
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        top = QHBoxLayout()
        self._lbl_folder = QLabel('Nenhuma pasta selecionada')
        self._lbl_folder.setStyleSheet(f'color: {TEXT}; font-style: italic;')
        top.addWidget(self._lbl_folder, 1)
        btn_folder = QPushButton('Selecionar pasta…')
        btn_folder.clicked.connect(self._choose_folder)
        top.addWidget(btn_folder)
        lay.addLayout(top)

        # Parâmetros de exposição
        params_box = QGroupBox('Parâmetros de exposição (aplicados a todos os arquivos)')
        pf = QFormLayout(params_box)
        self.sex = QComboBox()
        self.sex.addItems(['male', 'female'])
        pf.addRow('Sexo', self.sex)
        self.mass = QDoubleSpinBox()
        self.mass.setRange(5.0, 200.0)
        self.mass.setValue(60.0)
        self.mass.setSuffix(' kg')
        pf.addRow('Massa corporal', self.mass)
        self.daily = QDoubleSpinBox()
        self.daily.setRange(0.1, 24.0)
        self.daily.setValue(8.0)
        self.daily.setSuffix(' h/dia')
        pf.addRow('Exposição diária', self.daily)
        self.days = QSpinBox()
        self.days.setRange(1, 365)
        self.days.setValue(220)
        pf.addRow('Dias por ano', self.days)
        self.start_age = QSpinBox()
        self.start_age.setRange(0, 80)
        self.start_age.setValue(10)
        pf.addRow('Idade no início', self.start_age)
        self.years = QSpinBox()
        self.years.setRange(1, 50)
        self.years.setValue(10)
        pf.addRow('Anos de exposição', self.years)
        lay.addWidget(params_box)

        # Opções de saída
        out_box = QGroupBox('Saída')
        of = QHBoxLayout(out_box)
        self.chk_pdf = QCheckBox('Gerar ficha PDF por arquivo')
        self.chk_pdf.setChecked(True)
        of.addWidget(self.chk_pdf)
        of.addStretch()
        lay.addWidget(out_box)

        # Botões de ação
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton('Processar lote')
        self.run_btn.setObjectName('primary')
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)
        self.csv_btn = QPushButton('Exportar resumo CSV…')
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self.csv_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Barra de progresso
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        # Tabela de resultados
        cols = ['Arquivo', 'Dz (m/s²)', 'Dzd (m/s²)', 'Sd (MPa)',
                'R', 'Π (%)', 'Risco', 'Erro']
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(cols)):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 1)

    # ------------------------------------------------------------------
    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Selecionar pasta com arquivos')
        if not folder:
            return
        self._folder = Path(folder)
        self._lbl_folder.setText(str(self._folder))
        self._lbl_folder.setStyleSheet(f'color: {TEXT};')
        # Mostra quantos arquivos elegíveis existem
        files = [f for f in self._folder.iterdir()
                 if f.is_file() and f.suffix.lower() in _EXTS]
        self._status(f'Pasta: {self._folder.name} — {len(files)} arquivo(s) elegível(s)')

    def _collect_files(self) -> list[Path]:
        if self._folder is None:
            return []
        return sorted(
            f for f in self._folder.iterdir()
            if f.is_file() and f.suffix.lower() in _EXTS
        )

    # ------------------------------------------------------------------
    def _run(self):
        files = self._collect_files()
        if not files:
            QMessageBox.information(self, 'Processamento em lote',
                                    'Nenhum arquivo elegível (.csv, .dat, .txt, .m) '
                                    'encontrado na pasta selecionada.')
            return

        self.run_btn.setEnabled(False)
        self.csv_btn.setEnabled(False)
        self._results.clear()
        self.table.setRowCount(0)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        QApplication.processEvents()

        sex       = self.sex.currentText()
        mass      = self.mass.value()
        daily_h   = self.daily.value()
        days      = self.days.value()
        start_age = self.start_age.value()
        n_years   = self.years.value()
        gen_pdf   = self.chk_pdf.isChecked()

        try:
            for i, path in enumerate(files):
                rec: dict = {'file': path.name, 'path': path,
                             'Dz': '', 'Dzd': '', 'Sd': '', 'R': '', 'Pi': '',
                             'risk': '', 'error': ''}
                try:
                    day = fd.load_day_auto([path], calibrate=False, position='seat')
                    segs = day.contiguous_segments()
                    dt = day.dt
                    measured = sum(len(s) for s in segs) * dt

                    result = annexc.assess_from_segments(
                        segs, dt,
                        measured_time=measured,
                        body_mass=mass,
                        sex=sex,
                        daily_time=daily_h * 3600.0,
                        days_per_year=days,
                        start_age=start_age,
                        n_years=n_years,
                    )
                    rec['Dz']   = result.Dz
                    rec['Dzd']  = result.Dzd
                    rec['Sd']   = result.Sd
                    rec['R']    = result.R
                    rec['Pi']   = result.probability * 100.0
                    rec['risk'] = _risk_label(result.probability)

                    if gen_pdf:
                        pdf_path = path.parent / (path.stem + '_annexc.pdf')
                        meta = {'subject_id': path.stem, 'date': '', 'operator': ''}
                        pdf_report.export_to_pdf(
                            result, str(pdf_path), meta,
                            daily_hours=daily_h, n_years=n_years, days=days)
                        rec['pdf'] = str(pdf_path)

                except Exception as exc:
                    rec['error'] = str(exc)

                self._results.append(rec)
                self._add_table_row(rec)
                self.progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            self.progress.setVisible(False)
            self.run_btn.setEnabled(True)

        ok  = sum(1 for r in self._results if not r['error'])
        err = len(self._results) - ok
        self._status(
            f'Lote concluído: {ok} avaliação(ões) com êxito'
            + (f', {err} erro(s)' if err else '') + '.')
        if ok:
            self.csv_btn.setEnabled(True)

    # ------------------------------------------------------------------
    def _add_table_row(self, rec: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        has_err = bool(rec.get('error'))
        risk_colour = ''
        if not has_err and rec.get('Pi') != '':
            risk_colour = _risk_colour(rec['Pi'] / 100.0)

        vals = [
            rec['file'],
            f"{rec['Dz']:.4f}"   if not has_err else '',
            f"{rec['Dzd']:.4f}"  if not has_err else '',
            f"{rec['Sd']:.4f}"   if not has_err else '',
            f"{rec['R']:.4f}"    if not has_err else '',
            f"{rec['Pi']:.2f}"   if not has_err else '',
            rec['risk']          if not has_err else '',
            rec.get('error', ''),
        ]
        for c, val in enumerate(vals):
            it = QTableWidgetItem(val)
            it.setForeground(QColor(TEXT))
            if c == 6 and risk_colour:
                it.setForeground(QColor(risk_colour))
                fnt = it.font()
                fnt.setBold(True)
                it.setFont(fnt)
            if c in (1, 2, 3, 4, 5):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if c == 7 and has_err:
                it.setForeground(QColor(BAD_COLOR))
            self.table.setItem(row, c, it)

    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self._results:
            return
        default = str(self._folder / 'lote_annexc.csv') if self._folder else 'lote_annexc.csv'
        path_out, _ = QFileDialog.getSaveFileName(
            self, 'Exportar resumo CSV', default, 'CSV (*.csv)')
        if not path_out:
            return
        try:
            with open(path_out, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Arquivo', 'Dz (m/s²)', 'Dzd (m/s²)',
                             'Sd (MPa)', 'R', 'Pi (%)', 'Risco', 'Erro'])
                for rec in self._results:
                    has_err = bool(rec.get('error'))
                    w.writerow([
                        rec['file'],
                        f"{rec['Dz']:.4f}"  if not has_err else '',
                        f"{rec['Dzd']:.4f}" if not has_err else '',
                        f"{rec['Sd']:.4f}"  if not has_err else '',
                        f"{rec['R']:.4f}"   if not has_err else '',
                        f"{rec['Pi']:.2f}"  if not has_err else '',
                        rec['risk']         if not has_err else '',
                        rec.get('error', ''),
                    ])
            self._status(f'CSV salvo: {Path(path_out).name}')
        except Exception as e:
            QMessageBox.critical(self, 'Erro ao exportar CSV', str(e))
