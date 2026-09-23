"""Worker generico para rodar uma funcao demorada fora da UI thread.

Usado por qualquer painel que precise ler muitos arquivos ou rodar um
calculo pesado (ponderacao em frequencia sobre um dia inteiro de blocos,
por exemplo) sem travar a janela.
"""
import traceback

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(float, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.finished.emit(self.fn(self._report))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _report(self, frac, msg):
        self.progress.emit(-1.0 if frac is None else frac, msg)


def run_in_thread(owner, job, on_done, on_fail=None, on_progress=None,
                  thread_attr='_thread', worker_attr='_worker'):
    """Inicia job(report) numa QThread e liga finished/failed/progress.

    owner precisa manter uma referencia as duas QThread/Worker vivas
    (thread_attr, worker_attr) para o Python nao coletar o objeto enquanto a
    thread roda. job(report) roda fora da UI thread: report(frac, msg) e a
    unica forma segura de avisar a UI do progresso de la de dentro -- tocar
    um widget diretamente do worker corrompe a interface. Devolve a QThread
    ja iniciada.
    """
    thread = QThread()
    worker = Worker(job)
    worker.moveToThread(thread)
    setattr(owner, thread_attr, thread)
    setattr(owner, worker_attr, worker)
    thread.started.connect(worker.run)
    worker.finished.connect(on_done)
    worker.failed.connect(on_fail or (lambda tb: None))
    if on_progress:
        worker.progress.connect(on_progress)
    thread.start()
    return thread
