# flopa/widgets/phasor_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, QSlider, QButtonGroup, QFileDialog, 
    QComboBox, QRadioButton, QMessageBox, QApplication, QHBoxLayout, QCheckBox, QGridLayout, QLineEdit, QSpinBox
)
from superqt import QDoubleRangeSlider

from qtpy.QtCore import Slot, Qt
import napari
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path

from .utils.style import dark_plot, light_plot
from flopa.io.ptuio.utils import draw_unitary_circle, average_phasor, aggregate_dataset, smooth_weighted
from .calibration_dialog import CalibrationDialog
from flopa.processing.phasor import calculate_phasor_calibration_factor, apply_phasor_calibration
from .utils.exporter import export_phasor_data

class PhasorPanel(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.active_selectors = {}
        self.last_plot_data = None

        self._final_plot_data = None
        self._init_ui()
        self._plot_phasor(message="No data.")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Group 1: Intensity Thresholding ---
        threshold_group = QGroupBox("1. Intensity Thresholding")
        threshold_layout = QFormLayout(threshold_group)
        
        self.thresh_enable_check = QCheckBox("Enable Live Threshold Preview")
        self.thresh_enable_check.toggled.connect(self._on_threshold_enable_toggled)
        threshold_layout.addRow(self.thresh_enable_check)
        
        # This container will hold the new range slider
        self.slider_container = QWidget()
        slider_vbox = QVBoxLayout(self.slider_container)
        slider_vbox.setContentsMargins(0, 0, 0, 0)

        # Use QDoubleRangeSlider for a min/max slider
        self.thresh_range_slider = QDoubleRangeSlider(Qt.Horizontal)
        self.thresh_range_slider.setRange(0, 1) # Placeholder
        self.thresh_range_slider.valueChanged.connect(self._update_threshold_preview)
        
        # Labels to show the current range
        self.thresh_label = QLabel("Range: 0 - 1")
        self.thresh_label.setAlignment(Qt.AlignCenter)
        
        slider_vbox.addWidget(self.thresh_range_slider)
        slider_vbox.addWidget(self.thresh_label)
        
        threshold_layout.addRow(self.slider_container)
        self.slider_container.setVisible(False)
        layout.addWidget(threshold_group)

        # --- Group 2: Computation ---
        compute_group = QGroupBox("2. Analysis Configuration"); compute_layout = QFormLayout(compute_group)
        self.mask_combobox = QComboBox(); self.viewer.layers.events.inserted.connect(self._update_mask_combobox); self.viewer.layers.events.removed.connect(self._update_mask_combobox)
        compute_layout.addRow("Select Mask Layer:", self.mask_combobox)
        self.per_object_radio = QRadioButton("Per Object"); self.per_object_radio.setChecked(True)
        self.per_pixel_radio = QRadioButton("Per Pixel")
        mode_hbox = QHBoxLayout(); mode_hbox.addWidget(self.per_object_radio); mode_hbox.addWidget(self.per_pixel_radio)
        compute_layout.addRow("Mode:", mode_hbox)
        layout.addWidget(compute_group)


        # --- NEW Group: Calibration & Smoothing ---
        # cal_group = QGroupBox("3. Processing")
        # cal_layout = QFormLayout(cal_group)
        # self.calibrate_check = QCheckBox("Apply Calibration")
        # cal_layout.addRow(self.calibrate_check)
        # self.cal_factor_display = QLineEdit("1.0 - 0.0j"); self.cal_factor_display.setToolTip("Calibration factor (g ± sj)")
        # self.btn_calculate_cal = QPushButton("Calculate..."); self.btn_calculate_cal.clicked.connect(self._on_calculate_cal_factor)
        # cal_hbox = QHBoxLayout(); cal_hbox.addWidget(self.cal_factor_display); cal_hbox.addWidget(self.btn_calculate_cal)
        # cal_layout.addRow("Factor:", cal_hbox)
        # self.smooth_check = QCheckBox("Apply Smoothing")
        # self.smooth_size_spin = QSpinBox(); self.smooth_size_spin.setRange(3, 15); self.smooth_size_spin.setSingleStep(2); self.smooth_size_spin.setValue(3)
        # smooth_hbox = QHBoxLayout(); smooth_hbox.addWidget(self.smooth_check); smooth_hbox.addWidget(QLabel("Kernel Size:")); smooth_hbox.addWidget(self.smooth_size_spin); smooth_hbox.addStretch()
        # cal_layout.addRow(smooth_hbox)
        # layout.addWidget(cal_group)


# In PhasorPanel._init_ui method

        # --- Group 3: Calibration & Smoothing ---
        cal_group = QGroupBox("3. Processing")
        cal_layout = QFormLayout(cal_group)

        # --- Calibration Mode Selection ---
        self.calibrate_check = QCheckBox("Apply Calibration")
        # When the main checkbox is toggled, it's good practice to re-plot
        self.calibrate_check.toggled.connect(self._on_compute_clicked)
        cal_layout.addRow(self.calibrate_check)

        # A container for the two radio button modes
        cal_mode_widget = QWidget()
        cal_mode_hbox = QHBoxLayout(cal_mode_widget)
        cal_mode_hbox.setContentsMargins(0, 0, 0, 0)

        self.cal_by_factor_radio = QRadioButton("By Factor")
        self.cal_by_shift_radio = QRadioButton("By Decay Shift")
        
        # Add to a button group to make them mutually exclusive
        self.cal_mode_buttons = QButtonGroup(self)
        self.cal_mode_buttons.addButton(self.cal_by_factor_radio)
        self.cal_mode_buttons.addButton(self.cal_by_shift_radio)
        self.cal_by_factor_radio.setChecked(True) # Default mode

        # Initially disable the shift option until a shift is set
        self.cal_by_shift_radio.setEnabled(False)
        self.cal_by_shift_radio.setToolTip("Enable by setting a non-zero shift in the Decay panel.")
        
        # Also trigger a re-plot when the mode is changed
        self.cal_by_factor_radio.toggled.connect(self._on_compute_clicked)
        self.cal_by_shift_radio.toggled.connect(self._on_compute_clicked)

        cal_mode_hbox.addWidget(self.cal_by_factor_radio)
        cal_mode_hbox.addWidget(self.cal_by_shift_radio)
        cal_mode_hbox.addStretch()
        cal_layout.addRow("Mode:", cal_mode_widget)

        # --- Calibration Factor Input ---
        self.cal_factor_display = QLineEdit(self._format_complex_to_str(1.0 + 0.0j))
        self.cal_factor_display.setToolTip("Calibration factor (g ± sj)")
        self.btn_calculate_cal = QPushButton("Calculate...")
        self.btn_calculate_cal.clicked.connect(self._on_calculate_cal_factor)
        
        cal_factor_hbox = QHBoxLayout()
        cal_factor_hbox.addWidget(self.cal_factor_display)
        cal_factor_hbox.addWidget(self.btn_calculate_cal)
        cal_layout.addRow("Factor:", cal_factor_hbox)

        # --- Spatial Smoothing Controls ---
        smooth_widget = QWidget()
        smooth_hbox = QHBoxLayout(smooth_widget)
        smooth_hbox.setContentsMargins(0, 0, 0, 0)
        
        self.smooth_check = QCheckBox("Apply Spatial Smoothing")
        self.smooth_check.toggled.connect(self._on_compute_clicked) # Re-plot when toggled
        
        self.smooth_size_spin = QSpinBox()
        self.smooth_size_spin.setRange(3, 15)
        self.smooth_size_spin.setSingleStep(2) # Odd numbers are typical for kernels
        self.smooth_size_spin.setValue(3)
        self.smooth_size_spin.valueChanged.connect(self._on_compute_clicked)
        
        smooth_hbox.addWidget(self.smooth_check)
        smooth_hbox.addWidget(QLabel("Kernel Size:"))
        smooth_hbox.addWidget(self.smooth_size_spin)
        smooth_hbox.addStretch()
        
        cal_layout.addRow(smooth_widget)
        
        layout.addWidget(cal_group)

        # --- Group 4: Action ---
        self.btn_compute = QPushButton("Plot Phasor from Current Slice"); self.btn_compute.clicked.connect(self._on_compute_clicked)
        layout.addWidget(self.btn_compute)

        # --- Group Plot Display ---
        plot_group = QGroupBox("Phasor Plot")
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
        self.phasor_figure = Figure(figsize=(5, 4)); self.phasor_canvas = FigureCanvas(self.phasor_figure); self.phasor_ax = self.phasor_figure.add_subplot(111)
        plot_layout.addWidget(self.phasor_canvas)
        layout.addWidget(plot_group)

        # --- Group Plot Options ---
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

        layout.addStretch()


    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        """Receives the full dataset once."""
        self.dataset = dataset
        self._update_mask_combobox()
        has_intensity = "photon_count" in self.dataset.data_vars
        has_phasor = "phasor_g" in self.dataset.data_vars
        
        # Enable the controls based on the available data.
        self.thresh_enable_check.setEnabled(has_intensity)
        self.btn_compute.setEnabled(has_phasor)

    @Slot(dict)
    def on_view_changed(self, selectors: dict):
        """Receives the selector state from the FlimViewPanel."""
        self.active_selectors = selectors
        # The thresholding UI needs to know about the current intensity slice
        self._update_thresholding_data()
        self.btn_compute.setEnabled(self.dataset is not None and "phasor_g" in self.dataset.data_vars)


    def _update_thresholding_data(self):
        """
        Updates the thresholding UI's data and range based on the
        current active slice of the intensity image.
        """
        if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars:
            self.thresh_enable_check.setEnabled(False)
            return

        # Calculate the current intensity slice
        intensity_da = self.dataset.photon_count
        final_ds = self._get_active_slice(intensity_da)
        intensity_slice = final_ds[intensity_da.name].values.squeeze()
        
        if intensity_slice is not None and intensity_slice.size > 0:
            self.thresh_enable_check.setEnabled(True)
            max_photons = int(np.nanmax(intensity_slice))
            if max_photons > 0:
                # Update the range slider's limits
                self.thresh_range_slider.setRange(0, max_photons)
                # Set a reasonable default value (e.g., 1 to max)
                self.thresh_range_slider.setValue((1, max_photons))
                self.thresh_label.setText(f"Range: 1 - {max_photons}")
        else:
            self.thresh_enable_check.setEnabled(False)

    def _on_threshold_enable_toggled(self, checked):
        """Shows/hides the sliders and the preview layer."""
        self.slider_container.setVisible(checked)
        preview_layer_name = "intensity_threshold_preview"
        
        if checked:
            # Check if we have data to threshold
            if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars:
                QMessageBox.warning(self, "No Slice", "No active slice selected in the FLIM View panel.")
                self.thresh_enable_check.setChecked(False); return
            
            intensity_slice = self._get_active_slice(self.dataset.photon_count).photon_count.values.squeeze()
            max_photons = int(np.nanmax(intensity_slice))
            
            # --- THIS IS THE FIX ---
            if max_photons > 0:
                # Use the correct widget name: self.thresh_range_slider
                # .setRange() takes two arguments: min and max
                self.thresh_range_slider.setRange(0, max_photons)
                # .setValue() takes a tuple: (min_value, max_value)
                self.thresh_range_slider.setValue((1, max_photons))
            # --- END FIX ---

            if preview_layer_name not in self.viewer.layers:
                self.viewer.add_labels(np.zeros_like(intensity_slice, dtype=np.uint8), name=preview_layer_name, opacity=1.0)
            
            self.viewer.layers[preview_layer_name].visible = True
            self._update_threshold_preview()
        else:
            if preview_layer_name in self.viewer.layers:
                self.viewer.layers[preview_layer_name].visible = False

    def _update_threshold_preview(self):
        """Calculates and updates the preview mask layer in real-time."""
        preview_layer_name = "intensity_threshold_preview"
        if not self.thresh_enable_check.isChecked() or preview_layer_name not in self.viewer.layers:
            return
            
        # Get the most up-to-date intensity slice
        intensity_slice = self._get_active_slice(self.dataset.photon_count).photon_count.values.squeeze()
        
        min_val, max_val = self.thresh_range_slider.value()

        # Update the label
        self.thresh_label.setText(f"Range: {int(min_val)} - {int(max_val)}")

        # Create the boolean mask and update the layer data
        mask_data = (intensity_slice >= min_val) & (intensity_slice <= max_val)
        self.viewer.layers[preview_layer_name].data = mask_data.astype(np.uint8)


    def _get_active_slice(self, data_array: xr.DataArray) -> xr.Dataset:
        """
        Helper to slice and/or sum a single DataArray based on the active selectors.
        Returns a Dataset containing the final, processed DataArray.
        """
        if data_array is None:
            return None
        if not self.active_selectors:
            # If selectors haven't been received, return the data as-is in a dataset
            return data_array.to_dataset()

        # --- THIS IS THE FIX ---
        # Build selection and summing lists by getting VALUES from the widgets
        selection_dict, dims_to_sum = {}, []
        for dim in ['frame', 'sequence', 'detector']:
            if dim in data_array.dims:
                # Use .isChecked() for checkboxes and .value() for spinboxes
                if self.active_selectors[f'sum_{dim}s'].isChecked():
                    dims_to_sum.append(dim)
                else:
                    selection_dict[dim] = self.active_selectors[dim].value()
        # --- END FIX ---
        
        sliced_ds = data_array.to_dataset().isel(**selection_dict)
        return aggregate_dataset(sliced_ds, dims_to_sum) if dims_to_sum else sliced_ds
            
        return final_array
        # return sum_dataset(sliced_array.to_dataset(), dims_to_sum) if dims_to_sum else sliced_array.to_dataset()


    def _on_compute_clicked(self):
        """
        Calculates and plots the phasor based on the active slice, selected mask,
        and processing options. This is the main analysis function for this panel.
        """
        if not self.active_selectors or self.dataset is None or "phasor_g" not in self.dataset.data_vars:
            QMessageBox.warning(self, "No Data", "Phasor data not available or no slice selected."); return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # --- 1. SLICE AND SUM THE DATA ---
            # Start with the full dataset
            ds_to_process = self.dataset

            # Build selection and summing lists from the active selectors
            selection_dict, dims_to_sum = {}, []
            for dim in ['frame', 'sequence', 'detector']:
                if dim in ds_to_process.dims:
                    if self.active_selectors[f'sum_{dim}s'].isChecked():
                        dims_to_sum.append(dim)
                    else:
                        selection_dict[dim] = self.active_selectors[dim].value()

            # Perform the slicing and aggregation
            sliced_ds = ds_to_process.isel(**selection_dict)
            final_ds = aggregate_dataset(sliced_ds, dims_to_sum) if dims_to_sum else sliced_ds
            
            # Extract the final 2D numpy arrays
            photon_count = final_ds.photon_count.values.squeeze()
            phasor_g = final_ds.phasor_g.values.squeeze()
            phasor_s = final_ds.phasor_s.values.squeeze()

            # --- 2. APPLY PROCESSING (SMOOTHING & CALIBRATION) ---
            if self.smooth_check.isChecked():
                kernel_size = self.smooth_size_spin.value()
                phasor_g, _ = smooth_weighted(phasor_g, photon_count, size=kernel_size)
                phasor_s, _ = smooth_weighted(phasor_s, photon_count, size=kernel_size)

            phasor_array = phasor_g + 1j * phasor_s

            if self.calibrate_check.isChecked():
                # try:
                #     cal_factor = self._parse_str_to_complex(self.cal_factor_display.text())
                #     phasor_array = apply_phasor_calibration(phasor_array, cal_factor)
                # except (ValueError, SyntaxError):
                #     QMessageBox.warning(self, "Invalid Factor", "Could not parse calibration factor. Skipping.")
                cal_factor = 1.0 + 0.0j # Default
                
                if self.cal_by_factor_radio.isChecked():
                    try:
                        cal_factor = self._parse_str_to_complex(self.cal_factor_display.text())
                    except ValueError as e:
                        QMessageBox.warning(self, "Invalid Factor", str(e))

                elif self.cal_by_shift_radio.isChecked():
                    # Get instrument params to calculate the factor from the shift
                    instrument_params = self.dataset.attrs.get('instrument_params', {})
                    shift = instrument_params.get('current_decay_shift', 0)
                    tcspc_res = instrument_params.get('tcspc_resolution_s', 5e-12)
                    rep_rate = instrument_params.get('repetition_rate', 40e6)
                    
                    cal_factor = np.exp(-1j * shift * tcspc_res * 2 * np.pi * rep_rate)

                phasor_array = apply_phasor_calibration(phasor_array, cal_factor)

            # --- 3. GET MASK AND PERFORM ANALYSIS ---
            mask_name = self.mask_combobox.currentData(); mask_array, mask_layer = None, None
            if mask_name != "None":
                if mask_name not in self.viewer.layers: QMessageBox.critical(self, "Error", f"Mask layer '{mask_name}' no longer exists."); return
                mask_layer = self.viewer.layers[mask_name]; mask_array = mask_layer.data
                if mask_array.shape != photon_count.shape: QMessageBox.warning(self, "Shape Mismatch", f"Mask shape {mask_array.shape} does not match image shape {photon_count.shape}."); return

            # --- 4. PREPARE COORDS FOR PLOTTING BASED ON MODE ---
            g_coords, s_coords, photon_counts, labels, colors, areas = [], [], [], [], [], []

            if self.per_pixel_radio.isChecked():
                # if mask_name == "None" and QMessageBox.question(self, 'Performance Warning', "Plotting all pixels may be slow. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
                #     QApplication.restoreOverrideCursor(); return
                
                mask_to_use = mask_array if mask_array is not None else np.ones_like(photon_count, dtype=np.uint8)
                valid_pixels = np.isfinite(phasor_array) & (photon_count > 0) & (mask_to_use > 0)
                
                g_coords = np.real(phasor_array[valid_pixels])
                s_coords = np.imag(phasor_array[valid_pixels])
                photon_counts = photon_count[valid_pixels]
                areas = np.full(g_coords.shape, np.nan)

                if mask_layer is not None:
                    pixel_labels = mask_array[valid_pixels]
                    labels = pixel_labels
                    colormap = mask_layer.colormap.colors
                    colors = colormap[pixel_labels % len(colormap)]
                else:
                    labels = np.full(g_coords.shape, np.nan)
                    colors = 'grey'

            elif self.per_object_radio.isChecked():
                if mask_array is None:
                    avg_phasor = average_phasor(phasor_array, photon_count)
                    if np.isfinite(avg_phasor): 
                        g_coords, s_coords, colors = [avg_phasor.real], [avg_phasor.imag], ['grey']
                        photon_counts = [np.sum(photon_count[np.isfinite(phasor_array)])]
                        labels = [np.nan]
                        areas = [np.sum(valid_pixels)]
                else:
                    label_ids = np.unique(mask_array)
                    label_ids = label_ids[label_ids > 0]
                    colormap = mask_layer.colormap.colors
                    for label_id in label_ids:
                        object_mask = (mask_array == label_id)
                        avg_phasor = average_phasor(phasor_array, photon_count, mask=object_mask)                        
                        if np.isfinite(avg_phasor):
                            g_coords.append(avg_phasor.real); s_coords.append(avg_phasor.imag)
                            total_photons = np.sum(photon_count[object_mask])
                            photon_counts.append(total_photons)
                            labels.append(label_id)
                            areas.append(np.sum(object_mask))
                            colors.append(colormap[label_id % len(colormap)])
            
            g_coords, s_coords = np.array(g_coords), np.array(s_coords)
            photon_counts, labels, areas = np.array(photon_counts), np.array(labels), np.array(areas)
            
            # Cache all the final, processed data
            self._final_plot_data = {
                "g_coords": g_coords, "s_coords": s_coords, 
                "photon_counts": photon_counts, "labels": labels,
                "areas": areas, "colors": colors, "mask_layer": mask_layer
            }
            # --- 5. PLOT ---
            sync_rate = self.dataset.attrs.get('instrument_params', {}).get('TTResult_SyncRate', 40e6)
            self._plot_phasor(g_coords, s_coords, colors, sync_rate, mask_layer=mask_layer)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during phasor computation:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _plot_phasor(self, g=None, s=None, colors=None, sync_rate=40e6, message=None, mask_layer=None):

        self.last_plot_data = {"g": g, "s": s, "colors": colors, "sync_rate": sync_rate, "message": message, "mask_layer": mask_layer}

        self.phasor_ax.clear()
        if self.dark_mode_check.isChecked(): 
            dark_plot(self.phasor_ax, self.phasor_figure)
            circle_color, edge_color = 'white', 'w'
        else: 
            light_plot(self.phasor_ax, self.phasor_figure)
            circle_color, edge_color = 'black', 'k'

        draw_unitary_circle(self.phasor_ax, sync_rate, color=circle_color, label_color=circle_color)

        if message: self.phasor_ax.text(0.5, 0.5, message, color=circle_color, ha='center', va='center', transform=self.phasor_ax.transAxes)
        if g is not None and g.size > 0: 
            self.phasor_ax.scatter(g, s, c=colors, s=(3 if g.size > 1000 else 40), alpha=0.8, edgecolor=edge_color, linewidth=0.2)
    
        self.phasor_ax.set_xlabel('g', fontsize=9)
        self.phasor_ax.set_ylabel('s', fontsize=9)
        self.phasor_ax.tick_params(axis='both', which='major', labelsize=9)
        #self.phasor_figure.tight_layout(pad=0.5) # Use standard tight_layout now
        self.phasor_figure.subplots_adjust(
            left=0.1, 
            right=0.95, 
            bottom=0.15, 
            top=0.9
        )
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

    # def _update_thresholding_data(self):
    #     """Updates the thresholding UI based on the current active slice."""
    #     if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars:
    #         self.thresh_enable_check.setEnabled(False)
    #         return

    #     # Calculate the current intensity slice using the received selectors
    #     intensity_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
        
    #     if intensity_slice is not None:
    #         self.thresh_enable_check.setEnabled(True)
    #         max_photons = int(np.nanmax(intensity_slice))
    #         if max_photons > 0:
    #             self.thresh_min_slider.setRange(0, max_photons)
    #             self.thresh_max_slider.setRange(0, max_photons)
    #             self.thresh_max_slider.setValue(max_photons)
    #     else:
    #         self.thresh_enable_check.setEnabled(False)

    # def _on_threshold_enable_toggled(self, checked):
    #     self.slider_container.setVisible(checked)
    #     preview_layer_name = "intensity_threshold_preview"
    #     if checked:
    #         if not self.active_selectors: QMessageBox.warning(self, "No Slice", "No active slice selected."); self.thresh_enable_check.setChecked(False); return
    #         intensity_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
    #         if intensity_slice is None: QMessageBox.warning(self, "No Data", "Intensity data not available."); self.thresh_enable_check.setChecked(False); return
    #         if preview_layer_name not in self.viewer.layers: self.viewer.add_labels(np.zeros_like(intensity_slice, dtype=np.uint8), name=preview_layer_name, opacity=0.5)
    #         self.viewer.layers[preview_layer_name].visible = True; self._update_threshold_preview()
    #     else:
    #         if preview_layer_name in self.viewer.layers: self.viewer.layers[preview_layer_name].visible = False

    # def _update_threshold_preview(self):
    #     preview_layer_name = "intensity_threshold_preview"
    #     if not self.active_selectors or self.dataset is None or "photon_count" not in self.dataset.data_vars or preview_layer_name not in self.viewer.layers: return
    #     photon_count_slice = self._get_active_slice(self.dataset.photon_count).values.squeeze()
    #     min_val, max_val = self.thresh_min_slider.value(), self.thresh_max_slider.value()
    #     self.thresh_min_label.setText(f"Min: {min_val}"); self.thresh_max_label.setText(f"Max: {max_val}")
    #     if min_val > max_val: self.thresh_min_slider.setValue(max_val); return
    #     self.viewer.layers[preview_layer_name].data = ((photon_count_slice >= min_val) & (photon_count_slice <= max_val)).astype(np.uint8)

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
                # self.cal_factor_display.setText(f"{factor.real:.4f} + {factor.imag:.4f}j")
                self.cal_factor_display.setText(self._format_complex_to_str(factor))
                
                # If calibration is already enabled, trigger a replot
                if self.calibrate_check.isChecked():
                    self._on_compute_clicked()
                    
            except (ValueError, ZeroDivisionError) as e:
                QMessageBox.critical(self, "Calculation Error", f"Could not calculate factor:\n{e}")

    def _format_complex_to_str(self, c: complex) -> str:
        """Formats a complex number into a clean 'g ± sj' string."""
        # Use a '+' sign for positive imaginary parts and rely on the
        # default '-' sign for negative parts.
        sign = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sign} {abs(c.imag):.4f}j"

    def _parse_str_to_complex(self, s: str) -> complex:
        """Reliably parses a string like '1.2 + 3.4j' into a complex number."""
        # Replace 'j' with 'j' to ensure compatibility, remove spaces
        s = s.replace("j", "j").replace(" ", "")
        try:
            return complex(s)
        except (ValueError, SyntaxError) as e:
            # Re-raise with a more user-friendly message
            raise ValueError(f"Invalid complex number format: '{s}'") from e
        

    def _on_export(self):
        """Exports the currently plotted data."""
        # Use hasattr to check if a plot has been computed yet
        if not hasattr(self, '_final_plot_data') or self._final_plot_data is None:
            QMessageBox.warning(self, "No Data", "Please compute a phasor plot first."); return

        export_format = self.export_combo.currentData()
        
        if export_format == "png":
            # --- Export Plot as PNG ---
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", "", "PNG Files (*.png)")
            if not save_path: return
            
            try:
                self.phasor_figure.savefig(save_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")

        elif export_format == "csv":
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Phasor Data as CSV", "", "CSV Files (*.csv)")
                if not save_path: return

                try:
                    # --- THIS IS THE SIMPLIFIED LOGIC ---
                    # 1. Get the cached final data from the last plot
                    plot_data = self._final_plot_data
                    
                    # 2. Get the dataset name
                    dataset_name = self.dataset.attrs.get("source_filename", "N/A")

                    # 3. Call the new, simpler backend exporter
                    export_phasor_data(
                        output_path=Path(save_path),
                        g_coords=plot_data['g_coords'],
                        s_coords=plot_data['s_coords'],
                        photon_counts=plot_data['photon_counts'],
                        labels=plot_data['labels'],
                        areas=plot_data['areas'], 
                        dataset_name=dataset_name
                    )
                    QMessageBox.information(self, "Success", f"Data saved to:\n{save_path}")
                    # ------------------------------------
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")