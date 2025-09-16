# flopa/widgets/calibration_dialog.py

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, 
    QDoubleSpinBox, QLabel
)

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Phasor Calibration Factor")
        
        self.factor = None # This will store the result
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.tau_spinbox = QDoubleSpinBox()
        self.tau_spinbox.setDecimals(4)
        self.tau_spinbox.setRange(0.0001, 100.0)
        self.tau_spinbox.setValue(3.8) # A reasonable default
        
        self.g_spinbox = QDoubleSpinBox()
        self.g_spinbox.setDecimals(4)
        self.g_spinbox.setRange(-2.0, 2.0)
        
        self.s_spinbox = QDoubleSpinBox()
        self.s_spinbox.setDecimals(4)
        self.s_spinbox.setRange(-2.0, 2.0)

        form_layout.addRow("Theoretical Lifetime (ns):", self.tau_spinbox)
        form_layout.addRow("Measured Phasor (s):", self.s_spinbox)
        form_layout.addRow("Measured Phasor (g):", self.g_spinbox)

        
        layout.addLayout(form_layout)
        
        # Standard OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_values(self):
        """Returns the user-entered values."""
        return {
            "tau_ns": self.tau_spinbox.value(),
            "g": self.g_spinbox.value(),
            "s": self.s_spinbox.value()
        }

    @staticmethod
    def calculate_from_user(parent=None):
        """Static method to show the dialog and return the user input."""
        dialog = CalibrationDialog(parent)
        # .exec_() shows the dialog modally and waits for the user to close it
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            return dialog.get_values()
        return None