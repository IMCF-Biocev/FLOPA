# flopa/widgets/decay_panel.py
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QSpinBox, QFormLayout, QProgressBar,
    QApplication, QPlainTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from flopa.widgets.utils.style import dark_plot, light_plot
from flopa.processing.decay import calculate_decays_for_mask

import numpy as np
import napari
import itertools
import matplotlib.pyplot as plt

# flopa/widgets/decay_panel.py
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QSpinBox, QFormLayout
)

class DecayPanel(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        
        # Data sources needed for on-demand calculation
        self.ptu_filepath = None
        self.scan_config = None
        
        # This will store the result: {label_id: decay_curve_array}
        self.last_calculated_decays = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(); self.setLayout(layout)
        
        calc_group = QGroupBox("1. Calculate Integrated Decays"); calc_layout = QFormLayout()
        self.mask_combobox = QComboBox()
        self.viewer.layers.events.connect(self._update_mask_combobox)
        calc_layout.addRow("Select Mask Layer:", self.mask_combobox)
        self.binning_spinbox = QSpinBox(); self.binning_spinbox.setRange(1, 256); self.binning_spinbox.setValue(4)
        calc_layout.addRow("TCSPC Binning:", self.binning_spinbox)
        self.btn_calculate_decay = QPushButton("Calculate Decays from PTU"); self.btn_calculate_decay.clicked.connect(self._on_calculate_decay_clicked)
        calc_layout.addRow(self.btn_calculate_decay)
        calc_group.setLayout(calc_layout)
        layout.addWidget(calc_group)

        plot_group = QGroupBox("2. Plotting Options")
        plot_layout = QHBoxLayout(); plot_group.setLayout(plot_layout)
        self.log_scale_checkbox = QCheckBox("Logarithmic Scale"); self.log_scale_checkbox.setChecked(True)
        self.dark_mode_checkbox = QCheckBox("Use Dark Theme"); self.dark_mode_checkbox.setChecked(True)
        self.log_scale_checkbox.stateChanged.connect(self._redraw_plot)
        self.dark_mode_checkbox.stateChanged.connect(self._redraw_plot)
        plot_layout.addWidget(self.log_scale_checkbox); plot_layout.addWidget(self.dark_mode_checkbox)
        layout.addWidget(plot_group)

        self.decay_figure = Figure(figsize=(5, 4)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111); self.decay_toolbar = NavigationToolbar(self.decay_canvas, self)
        layout.addWidget(self.decay_toolbar); layout.addWidget(self.decay_canvas)
        self._update_mask_combobox()

    def update_source_info(self, ptu_filepath, scan_config):
        self.ptu_filepath = ptu_filepath; self.scan_config = scan_config
        self.btn_calculate_decay.setEnabled(ptu_filepath is not None and scan_config is not None)

    def _update_mask_combobox(self, event=None):
        self.mask_combobox.clear()
        self.mask_combobox.addItems([layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])

    def _on_calculate_decay_clicked(self):
        if not self.ptu_filepath or not self.scan_config: return
        mask_name = self.mask_combobox.currentText()
        if not mask_name: QMessageBox.warning(self, "No Mask", "Please select a mask layer."); return
        
        mask_data = self.viewer.layers[mask_name].data
        binning = self.binning_spinbox.value()
        
        self.btn_calculate_decay.setText("Calculating...")
        QApplication.processEvents()

        try:
            # This is where the magic happens!
            self.last_calculated_decays = calculate_decays_for_mask(
                self.ptu_filepath, self.scan_config, mask_data, binning
            )
            self._redraw_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate decays:\n{e}")
        finally:
            self.btn_calculate_decay.setText("Calculate Decays from PTU")

    def _redraw_plot(self):
        self._plot_decay(self.last_calculated_decays)

    def _plot_decay(self, decay_dict):
        self.decay_ax.clear()
        if self.dark_mode_checkbox.isChecked(): dark_plot(self.decay_ax, self.decay_figure)
        else: light_plot(self.decay_ax, self.decay_figure)

        if not decay_dict:
            msg = "Calculate decays for a masked region to begin."
            self.decay_ax.text(0.5, 0.5, msg, ha='center', va='center', transform=self.decay_ax.transAxes)
        else:
            color_cycle = itertools.cycle(plt.cm.tab10.colors)
            for label_id, decay_curve in decay_dict.items():
                self.decay_ax.plot(decay_curve, label=f"Label {label_id}", color=next(color_cycle))
            self.decay_ax.legend()
        
        if self.log_scale_checkbox.isChecked(): self.decay_ax.set_yscale('log')
        self.decay_ax.set_xlabel(f"TCSPC Bin (Binned x{self.binning_spinbox.value()})")
        self.decay_ax.set_ylabel("Photon Count"); self.decay_ax.set_title("Integrated Decay Curves")
        self.decay_ax.grid(True, which='both', linestyle='--', alpha=0.5)
        self.decay_canvas.draw_idle()