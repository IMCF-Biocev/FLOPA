# flopa/widgets/phasor_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, QSlider,
    QComboBox, QRadioButton, QMessageBox, QApplication, QHBoxLayout, QCheckBox, QGridLayout, QLineEdit, QSpinBox
)
from qtpy.QtCore import Slot, Qt
import napari
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .utils.style import dark_plot, light_plot
from flopa.io.ptuio.utils import draw_unitary_circle, average_phasor, sum_dataset, smooth_phasor
from .calibration_dialog import CalibrationDialog
from flopa.processing.phasor import calculate_phasor_calibration_factor, apply_phasor_calibration

class PhasorPanel(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.active_selectors = {}
        self.last_plot_data = None
        self._init_ui()
        self._plot_phasor(message="No data available.")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Group 1: Intensity Thresholding ---
        threshold_group = QGroupBox("1. Intensity Thresholding"); threshold_layout = QFormLayout(threshold_group)
        self.thresh_enable_check = QCheckBox("Enable Live Threshold Preview"); self.thresh_enable_check.toggled.connect(self._on_threshold_enable_toggled)
        threshold_layout.addRow(self.thresh_enable_check)
        self.slider_container = QWidget(); slider_layout = QGridLayout(self.slider_container); slider_layout.setContentsMargins(0, 0, 0, 0)
        self.thresh_min_slider = QSlider(Qt.Horizontal); self.thresh_max_slider = QSlider(Qt.Horizontal)
        self.thresh_min_label = QLabel("Min: 0"); self.thresh_max_label = QLabel("Max: 1")
        slider_layout.addWidget(self.thresh_min_label, 0, 0); slider_layout.addWidget(self.thresh_max_label, 0, 1)
        slider_layout.addWidget(self.thresh_min_slider, 1, 0); slider_layout.addWidget(self.thresh_max_slider, 1, 1)
        self.thresh_min_slider.valueChanged.connect(self._update_threshold_preview); self.thresh_max_slider.valueChanged.connect(self._update_threshold_preview)
        threshold_layout.addRow(self.slider_container)
        self.slider_container.setVisible(False)
        layout.addWidget(threshold_group)

        # --- Group 2: Computation ---
        compute_group = QGroupBox("2. Compute & Plot Phasor"); compute_layout = QFormLayout(compute_group)
        self.mask_combobox = QComboBox(); self.viewer.layers.events.inserted.connect(self._update_mask_combobox); self.viewer.layers.events.removed.connect(self._update_mask_combobox)
        compute_layout.addRow("Select Mask Layer:", self.mask_combobox)
        self.per_object_radio = QRadioButton("Per Object"); self.per_object_radio.setChecked(True)
        self.per_pixel_radio = QRadioButton("Per Pixel")
        mode_hbox = QHBoxLayout(); mode_hbox.addWidget(self.per_object_radio); mode_hbox.addWidget(self.per_pixel_radio)
        compute_layout.addRow("Mode:", mode_hbox)
        self.btn_compute = QPushButton("Plot Phasor from Current Slice"); self.btn_compute.clicked.connect(self._on_compute_clicked)
        compute_layout.addRow(self.btn_compute)
        layout.addWidget(compute_group)

        # --- NEW Group: Calibration & Smoothing ---
        cal_group = QGroupBox("3. Calibration & Smoothing")
        cal_layout = QFormLayout(cal_group)

        self.calibrate_check = QCheckBox("Apply Calibration")
        self.calibrate_check.toggled.connect(self._on_compute_clicked) # Re-plot when toggled
        cal_layout.addRow(self.calibrate_check)
        
        self.cal_factor_display = QLineEdit("1.0 + 0.0j")
        self.cal_factor_display.setToolTip("Current calibration factor (g + sj)")
        self.btn_calculate_cal = QPushButton("Calculate...")
        self.btn_calculate_cal.clicked.connect(self._on_calculate_cal_factor)
        
        cal_hbox = QHBoxLayout()
        cal_hbox.addWidget(self.cal_factor_display)
        cal_hbox.addWidget(self.btn_calculate_cal)
        cal_layout.addRow("Factor:", cal_hbox)

        self.smooth_check = QCheckBox("Apply Spatial Smoothing")
        self.smooth_check.toggled.connect(self._on_compute_clicked) # Re-plot when toggled
        self.smooth_size_spin = QSpinBox()
        self.smooth_size_spin.setRange(3, 15)
        self.smooth_size_spin.setSingleStep(2) # Odd numbers are typical for kernels
        self.smooth_size_spin.setValue(3)
        self.smooth_size_spin.valueChanged.connect(self._on_compute_clicked)

        smooth_hbox = QHBoxLayout()
        smooth_hbox.addWidget(self.smooth_check)
        smooth_hbox.addWidget(QLabel("Kernel Size:"))
        smooth_hbox.addWidget(self.smooth_size_spin)
        smooth_hbox.addStretch()
        cal_layout.addRow(smooth_hbox)
        
        layout.addWidget(cal_group)

        # --- Group 3: Plot Display ---
        plot_group = QGroupBox("Phasor Plot"); plot_layout = QVBoxLayout(plot_group)
        options_layout = QHBoxLayout(); self.dark_mode_check = QCheckBox("Use Dark Theme"); self.dark_mode_check.setChecked(True); self.dark_mode_check.toggled.connect(self._on_theme_changed); options_layout.addWidget(self.dark_mode_check); options_layout.addStretch()
        plot_layout.addLayout(options_layout)
        self.phasor_figure = Figure(figsize=(5, 5)); self.phasor_canvas = FigureCanvas(self.phasor_figure); self.phasor_ax = self.phasor_figure.add_subplot(111)
        plot_layout.addWidget(self.phasor_canvas)
        layout.addWidget(plot_group)
        layout.addStretch()

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        """Receives the full dataset once."""
        self.dataset = dataset
        self._update_mask_combobox()
        self.btn_compute.setEnabled("phasor_g" in self.dataset.data_vars)

    @Slot(dict)
    def on_view_changed(self, selectors: dict):
        """Receives the selector state from the FlimViewPanel."""
        self.active_selectors = selectors
        # The thresholding UI needs to know about the current intensity slice
        self._update_thresholding_data()

    def _update_thresholding_data(self):
        """Updates the thresholding UI based on the current active slice."""
        if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars:
            self.thresh_enable_check.setEnabled(False)
            return

        # Calculate the current intensity slice using the received selectors
        intensity_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
        
        if intensity_slice is not None:
            self.thresh_enable_check.setEnabled(True)
            max_photons = int(np.nanmax(intensity_slice))
            if max_photons > 0:
                self.thresh_min_slider.setRange(0, max_photons)
                self.thresh_max_slider.setRange(0, max_photons)
                self.thresh_max_slider.setValue(max_photons)
        else:
            self.thresh_enable_check.setEnabled(False)

    def _get_active_slice(self, data_array: xr.DataArray) -> xr.DataArray:
        """Helper to slice/sum a DataArray based on the active selectors."""
        if data_array is None:
            return None
        if not self.active_selectors:
            # Return the original array if selectors haven't been set yet
            return data_array

        selection_dict, dims_to_sum = {}, []
        for dim in ['frame', 'sequence', 'detector']:
            if dim in data_array.dims:
                if self.active_selectors[f'sum_{dim}s'].isChecked():
                    dims_to_sum.append(dim)
                else:
                    selection_dict[dim] = self.active_selectors[dim].value()
        
        sliced_array = data_array.isel(**selection_dict)

        if dims_to_sum:
            # Here we apply the correct aggregation. This is a simplified version
            # of your sum_dataset logic, applied to a single DataArray.
            if data_array.name in ['mean_arrival_time', 'phasor_g', 'phasor_s']:
                # For these, a simple mean is a reasonable approximation
                final_array = sliced_array.mean(dim=dims_to_sum)
            else: # For photon_count
                final_array = sliced_array.sum(dim=dims_to_sum)
        else:
            final_array = sliced_array
            
        return final_array
        # return sum_dataset(sliced_array.to_dataset(), dims_to_sum) if dims_to_sum else sliced_array.to_dataset()

    def _on_compute_clicked(self):

        if not self.active_selectors or self.dataset is None or "phasor_g" not in self.dataset.data_vars:
            QMessageBox.warning(self, "No Data", "Phasor data not available or no slice selected."); return

        # Calculate the final 2D slice of the relevant data
        # final_ds = self._get_active_slice(self.dataset)
        # photon_count_slice = final_ds.photon_count.values.squeeze()
        # phasor_g_slice = final_ds.phasor_g.values.squeeze()
        # phasor_s_slice = final_ds.phasor_s.values.squeeze()
        final_ds_photon_count = self._get_active_slice(self.dataset.photon_count)
        final_ds_phasor_g = self._get_active_slice(self.dataset.phasor_g)
        final_ds_phasor_s = self._get_active_slice(self.dataset.phasor_s)
        
        # Now get the numpy arrays from the final DataArrays
        photon_count_slice = final_ds_photon_count.values.squeeze()
        phasor_g_slice = final_ds_phasor_g.values.squeeze()
        phasor_s_slice = final_ds_phasor_s.values.squeeze()
        
        mask_name = self.mask_combobox.currentData(); mask_array, mask_layer = None, None
        if mask_name != "None":
            if mask_name not in self.viewer.layers: QMessageBox.critical(self, "Error", f"Mask layer '{mask_name}' no longer exists."); return
            mask_layer = self.viewer.layers[mask_name]; mask_array = mask_layer.data
            if mask_array.shape != photon_count_slice.shape: QMessageBox.warning(self, "Shape Mismatch", f"Mask shape {mask_array.shape} does not match image shape {photon_count_slice.shape}."); return
        if self.per_pixel_radio.isChecked() and mask_name == "None":
            if QMessageBox.question(self, 'Performance Warning', "Computing per-pixel phasor for the entire image may be slow. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            phasor_array = phasor_g_slice + 1j * phasor_s_slice

            final_phasor_array = phasor_array
            
            if self.calibrate_check.isChecked():
                try:
                    cal_factor = complex(self.cal_factor_display.text().replace("j", "j").replace(" ", ""))
                    final_phasor_array = apply_phasor_calibration(final_phasor_array, cal_factor)
                except (ValueError, SyntaxError):
                    QMessageBox.warning(self, "Invalid Factor", "Could not parse calibration factor. Using 1+0j.")

            if self.smooth_check.isChecked():
                # smooth_phasor needs the photon count for weighting
                kernel_size = self.smooth_size_spin.value()
                final_phasor_array = smooth_phasor(final_phasor_array, photon_count_slice, size=kernel_size)

            g_coords, s_coords, colors = [], [], []
            final_g = np.real(final_phasor_array)
            final_s = np.imag(final_phasor_array)
            if self.per_pixel_radio.isChecked():
                mask_to_use = mask_array if mask_array is not None else np.ones_like(photon_count_slice, dtype=np.uint8)
                valid_pixels = np.isfinite(final_phasor_array) & (photon_count_slice > 0) & (mask_to_use > 0)
                g_coords, s_coords = final_g[valid_pixels], final_s[valid_pixels]
                if mask_layer:
                    pixel_labels = mask_array[valid_pixels]; colormap = mask_layer.colormap.colors
                    colors = colormap[pixel_labels % len(colormap)]
                else: colors = 'white'
            elif self.per_object_radio.isChecked():
                if mask_array is None:
                    avg_phasor = average_phasor(final_phasor_array, photon_count_slice)
                    if np.isfinite(avg_phasor): g_coords, s_coords, colors = [avg_phasor.real], [avg_phasor.imag], ['white']
                else:
                    label_ids = np.unique(mask_array); label_ids = label_ids[label_ids > 0]; colormap = mask_layer.colormap.colors
                    for label_id in label_ids:
                        avg_phasor = average_phasor(final_phasor_array, photon_count_slice, mask=(mask_array == label_id))
                        if np.isfinite(avg_phasor):
                            g_coords.append(avg_phasor.real); s_coords.append(avg_phasor.imag); colors.append(colormap[label_id % len(colormap)])
            sync_rate = self.dataset.attrs.get('instrument_params', {}).get('TTResult_SyncRate', 40e6)
            self._plot_phasor(np.array(g_coords), np.array(s_coords), colors, sync_rate, mask_layer=mask_layer)
        finally: QApplication.restoreOverrideCursor()

    def _plot_phasor(self, g=None, s=None, colors=None, sync_rate=40e6, message=None, mask_layer=None):
        self.last_plot_data = {"g": g, "s": s, "colors": colors, "sync_rate": sync_rate, "message": message, "mask_layer": mask_layer}
        self.phasor_ax.clear()
        if self.dark_mode_check.isChecked(): dark_plot(self.phasor_ax, self.phasor_figure); circle_color, edge_color = 'white', 'w'
        else: light_plot(self.phasor_ax, self.phasor_figure); circle_color, edge_color = 'black', 'k'
        draw_unitary_circle(self.phasor_ax, sync_rate, color=circle_color, label_color=circle_color)
        if message: self.phasor_ax.text(0.5, 0.5, message, color=circle_color, ha='center', va='center', transform=self.phasor_ax.transAxes)
        if g is not None and g.size > 0: self.phasor_ax.scatter(g, s, c=colors, s=(3 if g.size > 1000 else 40), alpha=0.8, edgecolor=edge_color, linewidth=0.2)
        self.phasor_ax.set_title('Phasor Plot'); self.phasor_ax.set_xlabel('g'); self.phasor_ax.set_ylabel('s')
        if self.per_object_radio.isChecked() and mask_layer is not None:
            from matplotlib.lines import Line2D
            label_ids = [lbl for lbl in np.unique(mask_layer.data) if lbl > 0]
            if label_ids: self.phasor_ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label=f'Object {lid}', markerfacecolor=mask_layer.colormap.colors[lid % len(mask_layer.colormap.colors)], markersize=8) for i, lid in enumerate(label_ids) if i < len(g)])
        elif self.per_object_radio.isChecked() and mask_layer is None and g is not None and g.size > 0: self.phasor_ax.legend(["Entire Image"])
        if self.phasor_ax.get_legend():
            if self.dark_mode_check.isChecked(): dark_plot(self.phasor_ax, self.phasor_figure)
            else: light_plot(self.phasor_ax, self.phasor_figure)
        self.phasor_canvas.draw_idle()

    def _on_theme_changed(self):
        if self.last_plot_data: self._plot_phasor(**self.last_plot_data)

    def _update_mask_combobox(self, event=None):
        current_text = self.mask_combobox.currentText()
        self.mask_combobox.clear(); self.mask_combobox.addItem("Entire Image", "None")
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels): self.mask_combobox.addItem(layer.name, layer.name)
        index = self.mask_combobox.findText(current_text)
        if index != -1: self.mask_combobox.setCurrentIndex(index)

    def _on_threshold_enable_toggled(self, checked):
        self.slider_container.setVisible(checked)
        preview_layer_name = "intensity_threshold_preview"
        if checked:
            if not self.active_selectors: QMessageBox.warning(self, "No Slice", "No active slice selected."); self.thresh_enable_check.setChecked(False); return
            intensity_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
            if intensity_slice is None: QMessageBox.warning(self, "No Data", "Intensity data not available."); self.thresh_enable_check.setChecked(False); return
            if preview_layer_name not in self.viewer.layers: self.viewer.add_labels(np.zeros_like(intensity_slice, dtype=np.uint8), name=preview_layer_name, opacity=0.5)
            self.viewer.layers[preview_layer_name].visible = True; self._update_threshold_preview()
        else:
            if preview_layer_name in self.viewer.layers: self.viewer.layers[preview_layer_name].visible = False

    def _update_threshold_preview(self):
        preview_layer_name = "intensity_threshold_preview"
        if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars or preview_layer_name not in self.viewer.layers: return
        photon_count_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
        min_val, max_val = self.thresh_min_slider.value(), self.thresh_max_slider.value()
        self.thresh_min_label.setText(f"Min: {min_val}"); self.thresh_max_label.setText(f"Max: {max_val}")
        if min_val > max_val: self.thresh_min_slider.setValue(max_val); return
        self.viewer.layers[preview_layer_name].data = ((photon_count_slice >= min_val) & (photon_count_slice <= max_val)).astype(np.uint8)

    def _on_calculate_cal_factor(self):
        """Launches the dialog to calculate a new calibration factor."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load data first to get the instrument parameters."); return
        
        # Launch the dialog and get the user's input
        user_input = CalibrationDialog.calculate_from_user(self)
        
        if user_input is not None:
            try:
                sync_rate = self.dataset.attrs.get('instrument_params', {}).get('TTResult_SyncRate', 40e6)
                measured_phasor = user_input['g'] + 1j * user_input['s']
                
                factor = calculate_phasor_calibration_factor(
                    theoretical_lifetime_ns=user_input['tau_ns'],
                    measured_phasor=measured_phasor,
                    sync_rate_hz=sync_rate
                )
                
                # Display the new factor in the UI
                self.cal_factor_display.setText(f"{factor.real:.4f} + {factor.imag:.4f}j")
                
                # If calibration is already enabled, trigger a replot
                if self.calibrate_check.isChecked():
                    self._on_compute_clicked()
                    
            except (ValueError, ZeroDivisionError) as e:
                QMessageBox.critical(self, "Calculation Error", f"Could not calculate factor:\n{e}")