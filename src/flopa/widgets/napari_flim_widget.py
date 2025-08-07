# src/my_tool/widgets/flim_widget.py

# from qtpy.QtWidgets import (
#     QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
#     QTabWidget, QGridLayout, QComboBox, QRadioButton, QGroupBox,
#     QSpinBox, QTextEdit
# )
# from qtpy.QtCore import Qt

# # For embedding matplotlib plots in the Qt widget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D # For custom plot elements
# from matplotlib.widgets import RectangleSelector
# from matplotlib.colors import LogNorm


# from magicgui.widgets import Container, SpinBox, FloatRangeSlider
# import numpy as np
# from pathlib import Path
# import warnings

# from flopa.widgets.phasor_panel import PhasorPanel

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget
#from flopa.widgets.load_view_panel import LoadViewPanel
from flopa.widgets.ptu_processing_panel import PtuProcessingPanel # <-- USE THIS
from flopa.widgets.phasor_panel import PhasorPanel
from flopa.io.loader import read_ptu_file


class FlimWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        self.data = None             # Holds loaded data + metadata, corrected or not
        self.flim_image = None       # Holds current generated FLIM image

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QTabWidget()
        self.layout().addWidget(self.tabs)

        # Load Data tab is always enabled and first
        #self.load_view_tab = LoadViewPanel(viewer, self.on_data_loaded, self.on_flim_image_generated)
        #self.tabs.addTab(self.load_view_tab, "Load and View")

        self.processing_tab = PtuProcessingPanel(
            viewer,
            self.on_data_loaded,
            self.on_flim_image_generated
        )
        self.tabs.addTab(self.processing_tab, "Process and View")


        # Phasor and Decay tabs start disabled
        self.phasor_tab = PhasorPanel(viewer)
        self.tabs.addTab(self.phasor_tab, "Phasor")
        self.tabs.setTabEnabled(1, False)

    
        # from flopa.widgets.decay_panel import DecayTab
        # self.decay_tab = DecayPanel(viewer)
        # self.tabs.addTab(self.decay_tab, "Decay")
        # self.tabs.setTabEnabled(2, False)

    def on_data_loaded(self, loaded_data):
        """Callback when data is loaded via LoadViewPanel."""
        self.data = loaded_data
        # No need to enable generate button manually anymore
        # Any other logic you want to run after data loading can go here
        print("Data loaded:", self.data["intensity"].shape)

    def on_flim_image_generated(self, flim_image, intensity_slice, lifetime_slice):
        # Called when user clicks "Generate FLIM Image" button after load/correction
        self.flim_image = flim_image
        # Now that FLIM image exists, enable downstream tabs
        self.tabs.setTabEnabled(1, True)  # Phasor tab
        #self.tabs.setTabEnabled(2, True)  # Decay tab
        # Inform phasor/decay tabs about new image
        #self.phasor_tab.set_flim_image(flim_image)
        self.phasor_tab.update_data(intensity_slice, lifetime_slice)
        #self.decay_tab.set_flim_image(flim_image)
    
 