# flopa/widgets/utils/threading.py

from qtpy.QtCore import QObject, Signal, QRunnable, Slot
import traceback
import sys

class WorkerSignals(QObject):
    '''Defines the signals available from a running worker thread.'''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(str) 

class Worker(QRunnable):
    '''Worker thread for running a function in the background.'''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
            
class ThreadSafeLogger(QObject):
    """
    A logger that can be passed to a worker thread and will safely emit
    Qt signals back to the main GUI thread.
    """
    log_updated = Signal(str)

    def log(self, message: str):
        """Emits the message via a Qt signal."""
        self.log_updated.emit(message)