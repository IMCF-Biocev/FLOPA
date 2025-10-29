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

from .utils.style import dark_plot, light_plot, GROUP_BOX_STYLE_A, apply_style
from flopa.io.ptuio.utils import aggregate_dataset
from .utils.legend_checkbox import LegendCheckBox
from flopa.io.ptuio.utils import shift_decay
from .utils.exporter import export_decay_data


class DecayPanel(QWidget):
    decay_shift_changed = Signal(int)

    def __init__(self, viewer, flim_view_panel=None):
        super().__init__()
        self.viewer = viewer
        self.flim_view_panel = flim_view_panel
        self.dataset = None
        self.last_plot_data = None
        self.plotted_lines = {}
        self.detector_toggles = []
        
        self._init_ui()
        self._plot_decay(message="No data loaded.")


    def _init_ui(self):
        layout = QVBoxLayout(self)
        controls_group = QGroupBox(""); controls_layout = QFormLayout(controls_group)
        sum_widget = QWidget(); sum_hbox = QHBoxLayout(sum_widget); sum_hbox.setContentsMargins(0, 0, 0, 0)
        self.sum_frames_check = QCheckBox("Frames")
        self.sum_sequences_check = QCheckBox("Sequences")
        self.sum_detectors_check = QCheckBox("Detectors")
        sum_hbox.addWidget(self.sum_frames_check); sum_hbox.addWidget(self.sum_sequences_check)
        sum_hbox.addWidget(self.sum_detectors_check); sum_hbox.addStretch()
        controls_layout.addRow("Aggregate:", sum_widget)
        self.shift_spinbox = QSpinBox(); self.shift_spinbox.setRange(-500, 500); self.shift_spinbox.setValue(0)
        self.shift_spinbox.setToolTip("Apply a circular shift to the decay curves (in channels).")
        controls_layout.addRow("Decay Shift:", self.shift_spinbox)
        layout.addWidget(controls_group)
        self.sum_frames_check.toggled.connect(self.plot_current_decay)
        self.sum_sequences_check.toggled.connect(self.plot_current_decay)
        self.sum_detectors_check.toggled.connect(self.plot_current_decay)
        self.shift_spinbox.valueChanged.connect(self.plot_current_decay)
        self.shift_spinbox.valueChanged.connect(self.decay_shift_changed.emit)
        plot_group = QGroupBox("Decay Plot"); apply_style(plot_group, GROUP_BOX_STYLE_A)
        plot_layout = QVBoxLayout(plot_group)
        self.decay_figure = Figure(figsize=(5, 3)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111); plot_layout.addWidget(self.decay_canvas)
        layout.addWidget(plot_group)
        self.legend_group = QGroupBox("Legend"); apply_style(self.legend_group, GROUP_BOX_STYLE_A)
        self.legend_main_layout = QVBoxLayout(self.legend_group)
        layout.addWidget(self.legend_group)
        self.legend_group.setVisible(False)
        options_group = QGroupBox(""); options_layout = QHBoxLayout(options_group)
        self.dark_mode_check = QCheckBox("Use Dark Theme"); self.dark_mode_check.setChecked(True)
        self.dark_mode_check.toggled.connect(self._on_theme_changed)
        options_layout.addWidget(self.dark_mode_check); options_layout.addStretch()
        options_layout.addWidget(QLabel(""))
        self.export_combo = QComboBox(); self.export_combo.addItem("Plot (.png)", "png"); self.export_combo.addItem("Data (.csv)", "csv")
        options_layout.addWidget(self.export_combo)
        self.btn_export = QPushButton("Save..."); self.btn_export.clicked.connect(self._on_export)
        options_layout.addWidget(self.btn_export)
        layout.addWidget(options_group)
        layout.addStretch()

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
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
        if self.dataset is None or "tcspc_histogram" not in self.dataset.data_vars:
            self.clear_plot(); return
        histogram_ds = self.dataset[["tcspc_histogram"]] 
        dims_to_sum = []
        if self.sum_frames_check.isChecked(): dims_to_sum.append('frame')
        if self.sum_sequences_check.isChecked(): dims_to_sum.append('sequence')
        if self.sum_detectors_check.isChecked(): dims_to_sum.append('detector')
        summed_ds = aggregate_dataset(histogram_ds, dims_to_sum) if dims_to_sum else histogram_ds
        instrument_params = self.dataset.attrs.get('instrument_params', {})
        tcspc_res_ns = instrument_params.get('tcspc_resolution_ns', 1)
        time_axis_ns = np.arange(histogram_ds.sizes['tcspc_channel']) * tcspc_res_ns
        current_shift = self.shift_spinbox.value()
        self.decay_shift_changed.emit(current_shift)
        self._plot_decay(time_axis_ns, summed_ds.tcspc_histogram, shift=current_shift)

    def _create_legend_controls(self):
        while self.legend_main_layout.count():
            child = self.legend_main_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub_child = child.layout().takeAt(0)
                    if sub_child.widget(): sub_child.widget().deleteLater()
        self.detector_toggles.clear()
        if not self.plotted_lines:
            self.legend_group.setVisible(False); return
        actions_layout = QHBoxLayout()
        btn_check_all = QPushButton("All"); btn_check_all.clicked.connect(self._check_all_visibility)
        actions_layout.addWidget(btn_check_all)
        btn_uncheck_all = QPushButton("None"); btn_uncheck_all.clicked.connect(self._uncheck_all_visibility)
        actions_layout.addWidget(btn_uncheck_all)
        btn_from_view = QPushButton("From View"); btn_from_view.setToolTip("Set aggregation and selection from the FLIM View panel")
        if self.flim_view_panel is None:
            btn_from_view.setEnabled(False)
        else:
            btn_from_view.clicked.connect(self._on_from_view_clicked)
        actions_layout.addWidget(btn_from_view)


        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        actions_layout.addWidget(separator)

        if 'detector' in self.dataset.dims and self.dataset.sizes['detector'] > 1:
            detector_label = QLabel("<b>Detector:</b>")
            actions_layout.addWidget(detector_label)

            for i in self.dataset.coords['detector'].values:
                btn = QPushButton(f"D{i}")
                btn.setCheckable(True)
                btn.setChecked(False)
                btn.setToolTip(f"Toggle all decays for Detector {i}")
                btn.setFixedWidth(40)
                btn.toggled.connect(lambda checked, det_idx=i: self._on_detector_toggle(checked, det_idx))
                actions_layout.addWidget(btn)
                self.detector_toggles.append(btn)
        
        actions_layout.addStretch()
        self.legend_main_layout.addLayout(actions_layout)

        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_container = QWidget()
        grid_layout = QGridLayout(scroll_container); grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        num_columns = 4
        for i, (checkbox, data) in enumerate(self.plotted_lines.items()):
            row, col = i // num_columns, i % num_columns
            checkbox.setChecked(data['line'].get_visible())
            grid_layout.addWidget(checkbox, row, col)
        scroll_area.setWidget(scroll_container)
        self.legend_main_layout.addWidget(scroll_area)
        self.legend_group.setVisible(True)
        self._sync_detector_toggles() 

    def _toggle_line_visibility(self, line, checked):
        line.set_visible(checked)
        self.decay_canvas.draw_idle()
        self._sync_detector_toggles() 
        
    def clear_plot(self):
        self.last_plot_data = None; self.plotted_lines.clear()
        self._plot_decay(message="TCSPC Histogram not available.")
        self._create_legend_controls()
        
    def _on_theme_changed(self):
        if self.last_plot_data: self._plot_decay(**self.last_plot_data)

    def _plot_decay(self, time_axis=None, decay_data=None, message=None, shift=0):
        self.decay_ax.clear(); self.plotted_lines.clear()
        self.last_plot_data = {"time_axis": time_axis, "decay_data": decay_data, "shift": shift}
        if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure); text_color = 'white'
        else: light_plot(self.decay_ax, self.decay_figure); text_color = 'black'
        if message: self.decay_ax.text(0.5, 0.5, message, color=text_color, ha='center', va='center', transform=self.decay_ax.transAxes)
        if time_axis is not None and decay_data is not None:
            remaining_dims = [dim for dim in decay_data.dims if dim != 'tcspc_channel']
            num_curves = len(list(itertools.product(*(decay_data[dim].values for dim in remaining_dims)))) if remaining_dims else 1
            color_cycle = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, num_curves)))
            if not remaining_dims:
                line, = self.decay_ax.semilogy(time_axis, decay_data.values, color=color_cycle[0])
                checkbox = LegendCheckBox(text="Summed Decay", color=color_cycle[0])
                checkbox.toggled.connect(partial(self._toggle_line_visibility, line))
                self.plotted_lines[checkbox] = {'line': line, 'coords': {}}
            else:
                color_iterator = iter(color_cycle)
                for coord_tuple, single_curve in decay_data.groupby(remaining_dims, squeeze=False):
                    decay_to_plot = shift_decay(single_curve.values.squeeze(), shift)
                    label = ", ".join([f"{dim[0].upper()}:{val}" for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))])
                    coords = {dim: val for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))}
                    color = next(color_iterator)
                    line, = self.decay_ax.semilogy(time_axis, decay_to_plot, label=label, color=color)
                    checkbox = LegendCheckBox(text=label, color=color)
                    checkbox.toggled.connect(partial(self._toggle_line_visibility, line))
                    self.plotted_lines[checkbox] = {'line': line, 'coords': coords}
            if decay_data.values.max() > 0: self.decay_ax.set_ylim(bottom=0.5)
        instrument_params = self.dataset.attrs.get('instrument_params', {}) if self.dataset else {}
        time_units = instrument_params.get('resolution_unit', 'ns')
        self.decay_ax.set_xlabel(f"Time ({time_units})", fontsize=9); self.decay_ax.set_ylabel("Photon Count", fontsize=9)
        self.decay_ax.tick_params(axis='both', which='major', labelsize=9)
        self.decay_figure.tight_layout(pad=0.5); self.decay_canvas.draw_idle()
        self._create_legend_controls()

    def _check_all_visibility(self):
        for checkbox in self.plotted_lines.keys(): checkbox.setChecked(True)
        self._sync_detector_toggles()

    def _uncheck_all_visibility(self):
        for checkbox in self.plotted_lines.keys(): checkbox.setChecked(False)
        self._sync_detector_toggles()

    def _on_detector_toggle(self, is_checked, detector_index):
        for checkbox, data in self.plotted_lines.items():
            if data['coords'].get('detector') == detector_index:
                checkbox.setChecked(is_checked)

    def _on_from_view_clicked(self):
        if self.flim_view_panel is None: return
        view_selectors = self.flim_view_panel.selectors
        
        # --- Part 1: Set aggregation state and replot ---
        self.sum_frames_check.blockSignals(True); self.sum_sequences_check.blockSignals(True); self.sum_detectors_check.blockSignals(True)
        self.sum_frames_check.setChecked(view_selectors['sum_frames'].isChecked())
        self.sum_sequences_check.setChecked(view_selectors['sum_sequences'].isChecked())
        self.sum_detectors_check.setChecked(view_selectors['sum_detectors'].isChecked())
        self.sum_frames_check.blockSignals(False); self.sum_sequences_check.blockSignals(False); self.sum_detectors_check.blockSignals(False)
        self.plot_current_decay()

        # --- Part 2: Set selection state based on slicers ---
        frame_val = view_selectors['frame'].value()
        sequence_val = view_selectors['sequence'].value()
        detector_val = view_selectors['detector'].value()

        is_sum_frames = self.sum_frames_check.isChecked()
        is_sum_sequences = self.sum_sequences_check.isChecked()
        is_sum_detectors = self.sum_detectors_check.isChecked()

        for checkbox, data in self.plotted_lines.items():
            coords = data['coords']
            frame_match = is_sum_frames or coords.get('frame') == frame_val
            sequence_match = is_sum_sequences or coords.get('sequence') == sequence_val
            detector_match = is_sum_detectors or coords.get('detector') == detector_val
            
            is_match = frame_match and sequence_match and detector_match
            checkbox.setChecked(is_match)
        
        self._sync_detector_toggles() 

    def _sync_detector_toggles(self):
        """Updates detector toggle buttons based on the current selection state."""
        if not self.detector_toggles: return

        is_sum_detectors = self.sum_detectors_check.isChecked()

        for i, btn in enumerate(self.detector_toggles):
            btn.blockSignals(True) 
            if is_sum_detectors:
                btn.setChecked(False)
                btn.setEnabled(False)
                btn.setToolTip("Detectors are aggregated")
            else:
                btn.setEnabled(True)
                btn.setToolTip(f"Toggle all decays for Detector {i}")
                
                traces_for_this_detector = [
                    cb for cb, data in self.plotted_lines.items() 
                    if data['coords'].get('detector') == i
                ]
                
                if not traces_for_this_detector:
                    btn.setChecked(False) 
                else:
                    all_checked = all(cb.isChecked() for cb in traces_for_this_detector)
                    btn.setChecked(all_checked)
            btn.blockSignals(False)
        
    def _on_export(self):
        if not hasattr(self, 'last_plot_data') or self.last_plot_data is None: QMessageBox.warning(self, "No Data", "Please plot a decay curve first."); return
        export_format = self.export_combo.currentData()
        if export_format == "png":
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", "", "PNG Files (*.png)");
            if not save_path: return
            try: self.decay_figure.savefig(save_path, dpi=300, bbox_inches='tight'); QMessageBox.information(self, "Success", f"Plot saved to:\n{save_path}")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
        elif export_format == "csv":
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Decay Data as CSV", "", "CSV Files (*.csv)");
            if not save_path: return
            try:
                time_axis = self.last_plot_data.get("time_axis"); decay_data_da = self.last_plot_data.get("decay_data")
                if time_axis is None or decay_data_da is None: raise ValueError("Last plot data is incomplete.")
                remaining_dims = [dim for dim in decay_data_da.dims if dim != 'tcspc_channel']; decay_curves = []; curve_labels = []
                if not remaining_dims: decay_curves.append(decay_data_da.values); curve_labels.append("Summed_Decay")
                else:
                    for coord_tuple, single_curve in decay_data_da.groupby(remaining_dims, squeeze=False):
                        label = ", ".join([f"{dim[0].upper()}:{val}" for dim, val in zip(remaining_dims, np.atleast_1d(coord_tuple))])
                        decay_curves.append(single_curve.values.squeeze()); curve_labels.append(label)
                dataset_name = self.dataset.attrs.get("source_filename", "N/A")
                export_decay_data(output_path=Path(save_path), time_axis=time_axis, decay_curves=decay_curves, curve_labels=curve_labels, dataset_name=dataset_name.replace(".ptu", ""))
                QMessageBox.information(self, "Success", f"Data saved to:\n{save_path}")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")