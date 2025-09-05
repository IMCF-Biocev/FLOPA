# flopa/widgets/decay_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QCheckBox, QHBoxLayout, QMessageBox,
    QScrollArea, QFrame
)
from qtpy.QtCore import Slot, Qt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from functools import partial

from .utils.style import dark_plot, light_plot

class DecayPanel(QWidget):
    """
    An independent panel for exploring and plotting TCSPC decay histograms.
    """
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.last_plot_data = None
        self.plotted_lines = {} # Maps checkbox object to matplotlib line object
        
        self._init_ui()
        self._plot_decay(message="No data loaded. Please run a reconstruction.")

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Group 1: Decay Summation Controls ---
        controls_group = QGroupBox("Decay Summation Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.sum_frames_check = QCheckBox("Sum Frames"); self.sum_sequences_check = QCheckBox("Sum Sequences")
        self.sum_detectors_check = QCheckBox("Sum Detectors")
        controls_layout.addWidget(self.sum_frames_check); controls_layout.addWidget(self.sum_sequences_check)
        controls_layout.addWidget(self.sum_detectors_check); controls_layout.addStretch()
        self.sum_frames_check.toggled.connect(self.plot_current_decay)
        self.sum_sequences_check.toggled.connect(self.plot_current_decay)
        self.sum_detectors_check.toggled.connect(self.plot_current_decay)
        layout.addWidget(controls_group)

        # --- Group 2: Plot Options ---
        options_group = QGroupBox("Plot Options"); options_layout = QHBoxLayout(options_group)
        self.dark_mode_check = QCheckBox("Use Dark Theme"); self.dark_mode_check.setChecked(True)
        self.dark_mode_check.toggled.connect(self._on_theme_changed)
        options_layout.addWidget(self.dark_mode_check); options_layout.addStretch()
        layout.addWidget(options_group)

        # --- Group 3: Decay Plot Display ---
        plot_group = QGroupBox("Decay Curve"); plot_layout = QVBoxLayout(plot_group)
        self.decay_figure = Figure(figsize=(5, 4)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111)
        plot_layout.addWidget(self.decay_canvas)
        layout.addWidget(plot_group)
        
        # --- NEW: Group 4: Trace Visibility ---
        self.visibility_group = QGroupBox("Trace Visibility")
        visibility_scroll = QScrollArea(); visibility_scroll.setWidgetResizable(True)
        visibility_scroll.setFrameShape(QFrame.NoFrame) # Cleaner look
        self.visibility_container = QWidget()
        self.visibility_layout = QVBoxLayout(self.visibility_container)
        visibility_scroll.setWidget(self.visibility_container)
        visibility_main_layout = QVBoxLayout(self.visibility_group)
        visibility_main_layout.addWidget(visibility_scroll)
        layout.addWidget(self.visibility_group)
        self.visibility_group.setVisible(False) # Hide until there are traces

        layout.addStretch()

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        """Public slot to receive the full dataset after reconstruction."""
        self.dataset = dataset
        if "tcspc_histogram" in self.dataset.data_vars:
            hist = self.dataset.tcspc_histogram
            self.sum_frames_check.setEnabled(hist.sizes.get('frame', 1) > 1)
            self.sum_sequences_check.setEnabled(hist.sizes.get('sequence', 1) > 1)
            self.sum_detectors_check.setEnabled(hist.sizes.get('detector', 1) > 1)
            self.plot_current_decay()
        else:
            self.clear_plot()

    def plot_current_decay(self):
        """Calculates and plots the decay based on the state of its own controls."""
        if self.dataset is None or "tcspc_histogram" not in self.dataset.data_vars:
            self.clear_plot(); return

        histogram = self.dataset.tcspc_histogram
        dims_to_sum = []
        if self.sum_frames_check.isChecked(): dims_to_sum.append('frame')
        if self.sum_sequences_check.isChecked(): dims_to_sum.append('sequence')
        if self.sum_detectors_check.isChecked(): dims_to_sum.append('detector')
        
        summed_hist = histogram.sum(dim=dims_to_sum) if dims_to_sum else histogram
        
        instrument_params = self.dataset.attrs.get('instrument_params', {})
        tcspc_res_s = instrument_params.get('MeasDesc_Resolution', 5e-12)
        time_axis_ns = np.arange(histogram.sizes['tcspc_channel']) * tcspc_res_s * 1e9

        self._plot_decay(time_axis_ns, summed_hist)

    def _update_visibility_controls(self):
        """Clears and recreates the checkboxes for toggling plot lines."""
        # Clear old checkboxes
        while self.visibility_layout.count():
            child = self.visibility_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not self.plotted_lines:
            self.visibility_group.setVisible(False)
            return
            
        # Create new checkboxes for current lines
        for checkbox, line in self.plotted_lines.items():
            checkbox.setChecked(line.get_visible())
            self.visibility_layout.addWidget(checkbox)
        
        self.visibility_group.setVisible(True)

    def _toggle_line_visibility(self, checkbox, line, state):
        """Slot to be called when a visibility checkbox is toggled."""
        line.set_visible(state == Qt.Checked)
        self.decay_canvas.draw_idle()
        
    def clear_plot(self):
        self.last_plot_data = None; self.plotted_lines.clear()
        self._plot_decay(message="TCSPC Histogram not available.")
        self._update_visibility_controls()
        
    def _on_theme_changed(self):
        if self.last_plot_data: self._plot_decay(**self.last_plot_data)

    def _plot_decay(self, time_axis=None, decay_data=None, message=None):
        """Clears and redraws the decay plot canvas."""
        self.decay_ax.clear(); self.plotted_lines.clear()
        self.last_plot_data = {"time_axis": time_axis, "decay_data": decay_data}

        if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure); text_color = 'white'
        else: light_plot(self.decay_ax, self.decay_figure); text_color = 'black'
            
        if message:
            self.decay_ax.text(0.5, 0.5, message, color=text_color, ha='center', va='center', transform=self.decay_ax.transAxes)
        
        if time_axis is not None and decay_data is not None:
            remaining_dims = [dim for dim in decay_data.dims if dim != 'tcspc_channel']
            
            if not remaining_dims:
                line, = self.decay_ax.semilogy(time_axis, decay_data.values, label='Summed Decay')
                checkbox = QCheckBox("Summed Decay"); checkbox.setChecked(True)
                checkbox.stateChanged.connect(partial(self._toggle_line_visibility, checkbox, line))
                self.plotted_lines[checkbox] = line
            else:
                color_cycle = plt.get_cmap("viridis")(np.linspace(0, 1, len(list(decay_data.groupby(remaining_dims)))))
                for i, (coord_tuple, single_curve) in enumerate(decay_data.groupby(remaining_dims)):
                    label = ", ".join([f"{dim[0].upper()}:{val}" for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))])
                    line, = self.decay_ax.semilogy(time_axis, single_curve.values, label=label, color=color_cycle[i])
                    checkbox = QCheckBox(label); checkbox.setChecked(True)
                    checkbox.stateChanged.connect(partial(self._toggle_line_visibility, checkbox, line))
                    self.plotted_lines[checkbox] = line

            if self.decay_ax.get_legend_handles_labels()[0]: self.decay_ax.legend(fontsize='small')
            if decay_data.values.max() > 0: self.decay_ax.set_ylim(bottom=0.5)

        self.decay_ax.set_title("Fluorescence Decay"); self.decay_ax.set_xlabel("Time (ns)"); self.decay_ax.set_ylabel("Photon Count (log scale)")
        
        if self.decay_ax.get_legend():
            if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure)
            else: light_plot(self.decay_ax, self.decay_figure)
        
        self.decay_canvas.draw_idle()
        self._update_visibility_controls() # Update the checkboxes after plotting