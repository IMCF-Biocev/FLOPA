# flopa/widgets/decay_panel.py

import traceback
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QCheckBox, QHBoxLayout, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QGridLayout, QSpinBox, QFormLayout, QLabel, QComboBox
)
from qtpy.QtCore import Slot, Qt, Signal
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from functools import partial
import itertools
from pathlib import Path

from .utils.style import dark_plot, light_plot
from flopa.io.ptuio.utils import aggregate_dataset
from .utils.legend_checkbox import LegendCheckBox
from flopa.io.ptuio.utils import shift_decay
from .utils.exporter import export_decay_data


class DecayPanel(QWidget):
    """
    An independent panel for exploring and plotting TCSPC decay histograms.
    """
    decay_shift_changed = Signal(int)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.last_plot_data = None
        self.plotted_lines = {} # Maps checkbox object to matplotlib line object
        
        self._init_ui()
        self._plot_decay(message="No data loaded.")

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Group 1: Decay Summation Controls ---
        controls_group = QGroupBox("")
        # controls_layout = QHBoxLayout(controls_group)
        # self.sum_frames_check = QCheckBox("Sum Frames"); self.sum_sequences_check = QCheckBox("Sum Sequences")
        # self.sum_detectors_check = QCheckBox("Sum Detectors")
        # controls_layout.addWidget(self.sum_frames_check); controls_layout.addWidget(self.sum_sequences_check)
        # controls_layout.addWidget(self.sum_detectors_check); controls_layout.addStretch()
        # self.sum_frames_check.toggled.connect(self.plot_current_decay)
        # self.sum_sequences_check.toggled.connect(self.plot_current_decay)
        # self.sum_detectors_check.toggled.connect(self.plot_current_decay)
        controls_layout = QFormLayout(controls_group)
        sum_widget = QWidget()
        sum_hbox = QHBoxLayout(sum_widget)
        sum_hbox.setContentsMargins(0, 0, 0, 0)
        self.sum_frames_check = QCheckBox("Frames")
        self.sum_sequences_check = QCheckBox("Sequences")
        self.sum_detectors_check = QCheckBox("Detectors")
        sum_hbox.addWidget(self.sum_frames_check)
        sum_hbox.addWidget(self.sum_sequences_check)
        sum_hbox.addWidget(self.sum_detectors_check)
        sum_hbox.addStretch()
        
        # Now, add the compound widget to the form layout
        controls_layout.addRow("Aggregate:", sum_widget)

        # The Decay Shift control is a simple label-field pair
        self.shift_spinbox = QSpinBox()
        self.shift_spinbox.setRange(-500, 500)
        self.shift_spinbox.setValue(0)
        self.shift_spinbox.setToolTip("Apply a circular shift to the decay curves (in channels).")
        controls_layout.addRow("Decay Shift (channels):", self.shift_spinbox)

        layout.addWidget(controls_group)

        self.sum_frames_check.toggled.connect(self.plot_current_decay)
        self.sum_sequences_check.toggled.connect(self.plot_current_decay)
        self.sum_detectors_check.toggled.connect(self.plot_current_decay)
        
        # The shift spinbox also triggers a replot
        self.shift_spinbox.valueChanged.connect(self.plot_current_decay)
        self.shift_spinbox.valueChanged.connect(self.decay_shift_changed.emit)

        # --- Group 3: Decay Plot Display ---
        plot_group = QGroupBox("Decay Curve"); 
        plot_group.setStyleSheet("""
            QGroupBox {
                /* Style the box itself */
                margin-top: 12px;       /* Space for the title */
            }
            
            QGroupBox::title {
                /* Style the title */
                subcontrol-origin: margin;
                
                padding-left: 20px;
                padding-right: 20px;

                font-size: 12pt; /* Use points for better scaling */
                font-weight: bold;
                color: #f5ea1d; /* Light gray for text */
            }
        """)
        plot_layout = QVBoxLayout(plot_group)
        self.decay_figure = Figure(figsize=(5, 3)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111)
        plot_layout.addWidget(self.decay_canvas)
        layout.addWidget(plot_group)
        
        # --- Group 2: Plot Options ---
        options_group = QGroupBox(""); options_layout = QHBoxLayout(options_group)
        self.dark_mode_check = QCheckBox("Use Dark Theme"); self.dark_mode_check.setChecked(True)
        self.dark_mode_check.toggled.connect(self._on_theme_changed)
        options_layout.addWidget(self.dark_mode_check); options_layout.addStretch()

        options_layout.addWidget(QLabel(""))
        
        self.export_combo = QComboBox()
        self.export_combo.addItem("Plot (.png)", "png")
        self.export_combo.addItem("Data (.csv)", "csv")
        options_layout.addWidget(self.export_combo)
        
        self.btn_export = QPushButton("Save...")
        self.btn_export.clicked.connect(self._on_export)
        options_layout.addWidget(self.btn_export)

        layout.addWidget(options_group)


        # --- NEW: Group 4: Trace Visibility ---
        self.visibility_group = QGroupBox("")

        visibility_main_layout = QVBoxLayout(self.visibility_group)

        actions_layout = QHBoxLayout()
        btn_check_all = QPushButton("All"); btn_check_all.setToolTip("Show all traces")
        btn_check_all.clicked.connect(self._check_all_visibility)
        btn_uncheck_all = QPushButton("None"); btn_uncheck_all.setToolTip("Hide all traces")
        btn_uncheck_all.clicked.connect(self._uncheck_all_visibility)
        
        actions_layout.addWidget(btn_check_all)
        actions_layout.addWidget(btn_uncheck_all)
        actions_layout.addStretch()
        
        # Add the button layout to the top of the group
        visibility_main_layout.addLayout(actions_layout)

        # Create the Scroll Area that will contain the checkboxes
        visibility_scroll = QScrollArea()
        visibility_scroll.setWidgetResizable(True)
        visibility_scroll.setFrameShape(QFrame.NoFrame)
        
        # Create the container widget that will be placed INSIDE the scroll area
        self.visibility_container = QWidget()
        # The layout for this container is the grid where checkboxes will be placed
        self.visibility_layout = QGridLayout(self.visibility_container)
        self.visibility_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        # Set the container as the scroll area's widget
        visibility_scroll.setWidget(self.visibility_container)

        # Add the scroll area to the main layout of the group box
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

        #histogram_ds = self.dataset.tcspc_histogram
        histogram_ds = self.dataset[["tcspc_histogram"]] 

        dims_to_sum = []
        if self.sum_frames_check.isChecked(): dims_to_sum.append('frame')
        if self.sum_sequences_check.isChecked(): dims_to_sum.append('sequence')
        if self.sum_detectors_check.isChecked(): dims_to_sum.append('detector')
        
        summed_ds = aggregate_dataset(histogram_ds, dims_to_sum) if dims_to_sum else histogram_ds
        summed_hist = summed_ds.tcspc_histogram    

        instrument_params = self.dataset.attrs.get('instrument_params', {})
        # tcspc_res_s = instrument_params.get('MeasDesc_Resolution', 5e-12)
        tcspc_res_ns = instrument_params.get('tcspc_resolution_ns', 1)
        # time_axis_ns = np.arange(histogram_ds.sizes['tcspc_channel']) * tcspc_res_s * 1e9
        time_axis_ns = np.arange(histogram_ds.sizes['tcspc_channel']) * tcspc_res_ns

        current_shift = self.shift_spinbox.value()
        self.decay_shift_changed.emit(current_shift)
        self._plot_decay(time_axis_ns, summed_hist, shift=current_shift)

    def _update_visibility_controls(self):
        """Clears and recreates the LegendCheckBoxes in a grid."""
        while self.visibility_layout.count():
            child = self.visibility_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not self.plotted_lines:
            self.visibility_group.setVisible(False)
            return
            
        num_columns = 4
        for i, legend_checkbox in enumerate(self.plotted_lines.keys()):
            row = i // num_columns
            col = i % num_columns
            
            line = self.plotted_lines[legend_checkbox]
            legend_checkbox.setChecked(line.get_visible())
            # Add the entire custom widget to the layout
            self.visibility_layout.addWidget(legend_checkbox, row, col)
        
        self.visibility_group.setVisible(True)

    def _toggle_line_visibility(self, line, checked):
        """Slot to be called when a visibility checkbox is toggled."""
        line.set_visible(checked)
        self.decay_canvas.draw_idle()
        
    def clear_plot(self):
        self.last_plot_data = None; self.plotted_lines.clear()
        self._plot_decay(message="TCSPC Histogram not available.")
        self._update_visibility_controls()
        
    def _on_theme_changed(self):
        if self.last_plot_data: self._plot_decay(**self.last_plot_data)

    def _plot_decay(self, time_axis=None, decay_data=None, message=None, shift=0):
        """Clears and redraws the decay plot canvas."""
        self.decay_ax.clear(); self.plotted_lines.clear()
        self.last_plot_data = {"time_axis": time_axis, "decay_data": decay_data, "shift": shift}

        if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure); text_color = 'white'
        else: light_plot(self.decay_ax, self.decay_figure); text_color = 'black'
            
        if message:
            self.decay_ax.text(0.5, 0.5, message, color=text_color, ha='center', va='center', transform=self.decay_ax.transAxes)
        
        if time_axis is not None and decay_data is not None:
            remaining_dims = [dim for dim in decay_data.dims if dim != 'tcspc_channel']
            num_curves = len(list(itertools.product(*(decay_data[dim].values for dim in remaining_dims)))) if remaining_dims else 1
            color_cycle = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, num_curves)))

            if not remaining_dims:
                line, = self.decay_ax.semilogy(time_axis, decay_data.values, color=color_cycle[0])
                # --- CREATE THE LEGEND CHECKBOX ---
                checkbox = LegendCheckBox(text="Summed Decay", color=color_cycle[0])
                checkbox.toggled.connect(partial(self._toggle_line_visibility, line))
                self.plotted_lines[checkbox] = line
            else:
                color_iterator = iter(color_cycle)
                for coord_tuple, single_curve in decay_data.groupby(remaining_dims, squeeze=False):
                    decay_to_plot = shift_decay(single_curve.values.squeeze(), shift)
                    label = ", ".join([f"{dim[0].upper()}:{val}" for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))])
                    color = next(color_iterator)
                    line, = self.decay_ax.semilogy(time_axis, decay_to_plot, label=label, color=color)
                    # --- CREATE THE LEGEND CHECKBOX ---
                    checkbox = LegendCheckBox(text=label, color=color)
                    checkbox.toggled.connect(partial(self._toggle_line_visibility, line))
                    self.plotted_lines[checkbox] = line

            
            if decay_data.values.max() > 0: self.decay_ax.set_ylim(bottom=0.5)

        instrument_params = self.dataset.attrs.get('instrument_params', {}) if self.dataset else {}
        time_units = instrument_params.get('resolution_unit', 'ns')

        self.decay_ax.set_xlabel(f"Time ({time_units})", fontsize=9)
        self.decay_ax.set_ylabel("Photon Count", fontsize=9)
        self.decay_ax.tick_params(axis='both', which='major', labelsize=9)
        self.decay_figure.tight_layout(pad=0.5) # Use standard tight_layout now
        self.decay_canvas.draw_idle()
        self._update_visibility_controls()


    def _check_all_visibility(self):
        """Checks all visibility checkboxes."""
        for checkbox in self.plotted_lines.keys():
            checkbox.setChecked(True)

    def _uncheck_all_visibility(self):
        """Unchecks all visibility checkboxes."""
        for checkbox in self.plotted_lines.keys():
            checkbox.setChecked(False)

    def _on_export(self):
        """Handles exporting the plot or the currently displayed decay data."""
        if not hasattr(self, 'last_plot_data') or self.last_plot_data is None:
            QMessageBox.warning(self, "No Data", "Please plot a decay curve first."); return

        export_format = self.export_combo.currentData()
        
        if export_format == "png":
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", "", "PNG Files (*.png)")
            if not save_path: return
            try:
                self.decay_figure.savefig(save_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")

        elif export_format == "csv":
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Decay Data as CSV", "", "CSV Files (*.csv)")
            if not save_path: return

            try:
                # --- Retrieve the cached data from the last plot ---
                time_axis = self.last_plot_data.get("time_axis")
                decay_data_da = self.last_plot_data.get("decay_data")
                
                if time_axis is None or decay_data_da is None:
                    raise ValueError("Last plot data is incomplete.")

                # --- Re-generate the curves and labels ---
                # This ensures we export exactly what's plotted
                remaining_dims = [dim for dim in decay_data_da.dims if dim != 'tcspc_channel']
                decay_curves = []
                curve_labels = []

                if not remaining_dims:
                    decay_curves.append(decay_data_da.values)
                    curve_labels.append("Summed_Decay")
                else:
                    for coord_tuple, single_curve in decay_data_da.groupby(remaining_dims, squeeze=False):
                        label = ", ".join([f"{dim[0].upper()}:{val}" for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))])
                        decay_curves.append(single_curve.values.squeeze())
                        curve_labels.append(label)
                
                dataset_name = self.dataset.attrs.get("source_filename", "N/A")

                # --- Call the backend exporter ---
                export_decay_data(
                    output_path=Path(save_path),
                    time_axis=time_axis,
                    decay_curves=decay_curves,
                    curve_labels=curve_labels,
                    dataset_name=dataset_name.replace(".ptu", "") # Remove extension
                )
                QMessageBox.information(self, "Success", f"Data saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")