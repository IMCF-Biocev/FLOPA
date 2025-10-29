# flopa/widgets/utils/legend_checkbox.py

from qtpy.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QFrame
from qtpy.QtGui import QPalette, QColor
import numpy as np

class LegendCheckBox(QWidget):
    """
    A custom widget that combines a QCheckBox with a colored square
    to act as an interactive legend item.
    """
    def __init__(self, text="", color="black", parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.color_swatch = QFrame()
        self.color_swatch.setFixedSize(12, 12)
        # self.color_swatch.setFrameShape(QFrame.StyledPanel)
        self.color_swatch.setStyleSheet("border: 1px solid gray;")

        self.set_color(color)
        
        self.checkbox = QCheckBox(text)

        layout.addWidget(self.color_swatch)
        layout.addWidget(self.checkbox)
        layout.addStretch()
        
    def set_color(self, color):
        """Sets the background color of the swatch."""
        palette = self.color_swatch.palette()


        if isinstance(color, (tuple, list, np.ndarray)):
            r, g, b, *a = color
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        else:
            q_color = QColor(color)
            hex_color = q_color.name()

        self.color_swatch.setStyleSheet(
            f"background-color: {hex_color};"
            "border: 1px solid gray;" 
        )
        # ----------------------------------------------



        
    def __getattr__(self, name):
        """
        Pass-through method to make this widget behave like a QCheckBox.
        This allows us to call .setChecked(), .isChecked(), and connect to
        its .toggled signal directly on the LegendCheckBox instance.
        """
        return getattr(self.checkbox, name)