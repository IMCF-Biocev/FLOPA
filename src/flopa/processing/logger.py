# flopa/processing/logger.py
from qtpy.QtCore import QObject, Signal

class ProgressLogger(QObject):
    """
    A universal progress logger that can either print to the console
    or emit Qt signals for a GUI.
    """
    log_updated = Signal(str)

    def __init__(self, mode='print'):
        super().__init__()
        if mode not in ['print', 'qt']:
            raise ValueError("mode must be 'print' or 'qt'")
        self.mode = mode

    def log(self, message: str):
        if self.mode == 'print':
            print(message)
        elif self.mode == 'qt':
            self.log_updated.emit(message)

    def connect(self, slot):
        if self.mode == 'qt':
            self.log_updated.connect(slot)