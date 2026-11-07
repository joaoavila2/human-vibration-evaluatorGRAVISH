"""Dialogo comum para escolher os arquivos de um dia de campo.

Aceita todos os formatos que fielddata.load_day_auto le. O unico que precisa
de informacao extra e o CSV Time_ms, cuja taxa nao e confiavel a partir do
arquivo: nesse caso a taxa e perguntada aqui, na thread da interface, antes
de a leitura ir para segundo plano.
"""
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from . import fielddata as fd

DAY_FILTER = ('All supported (*.csv *.dat *.txt *.asc *.m);;'
              'DateTime CSV, VCI_nRF52840 (*.csv);;'
              'XIAO binary VIB1 (*.dat);;'
              'ASCII 4 columns (*.txt *.asc);;'
              'Old field logger, 2 s every 5 s (*.m);;'
              'All files (*)')


def ask_day_files(parent, title='Select the files of one unit (one day)'):
    """Devolve (paths, opcoes) para load_day_auto, ou None se cancelado.

    opcoes e um dict com csv_rate quando os arquivos sao CSV Time_ms.
    """
    paths, _ = QFileDialog.getOpenFileNames(parent, title, '', DAY_FILTER)
    if not paths:
        return None
    try:
        kinds = {fd.detect_format(p) for p in paths}
    except (OSError, ValueError) as exc:
        QMessageBox.critical(parent, 'Unrecognised file', str(exc))
        return None
    if len(kinds) > 1:
        QMessageBox.critical(
            parent, 'Mixed formats',
            'The selected files mix formats:\n'
            + '\n'.join(fd.FORMAT_NAMES[k] for k in sorted(kinds))
            + '\n\nLoad one format at a time.')
        return None
    options = {}
    if kinds == {'csv_ms'}:
        rate, ok = QInputDialog.getDouble(
            parent, 'Sample rate',
            'Time_ms CSV: the millisecond timestamp is too coarse to give '
            'the sample rate.\nEnter the rate the firmware was configured '
            'for [Hz]:', 100.0, 1.0, 20000.0, 1)
        if not ok:
            return None
        options['csv_rate'] = rate
    return paths, options
