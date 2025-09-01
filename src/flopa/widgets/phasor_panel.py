# flopa/widgets/phasor_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, QSlider,
    QComboBox, QRadioButton, QButtonGroup, QMessageBox, QApplication, QHBoxLayout, QGridLayout, QCheckBox
)
from qtpy.QtCore import Slot, Qt
import napari
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import your backend utility functions
from flopa.io.ptuio.utils import draw_unitary_circle, average_phasor
from flopa.widgets.utils.style import dark_plot, light_plot


class PhasorPanel(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None  
        self.phasor_g = None
        self.phasor_s = None
        self.photon_count = None
        
        self._init_ui()
        self._plot_phasor(message="No data loaded or computed.")

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Group 1: Interactive Intensity Thresholding ---
        threshold_group = QGroupBox("Interactive Intensity Thresholding")
        threshold_layout = QFormLayout(threshold_group)

        self.thresh_enable_check = QCheckBox("Enable Live Threshold Preview")
        self.thresh_enable_check.toggled.connect(self._on_threshold_enable_toggled)
        threshold_layout.addRow(self.thresh_enable_check)

        self.slider_container = QWidget()
        slider_layout = QGridLayout(self.slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        self.thresh_min_slider = QSlider(Qt.Horizontal)
        self.thresh_max_slider = QSlider(Qt.Horizontal)
        
        self.thresh_min_label = QLabel("Min: 0")
        self.thresh_max_label = QLabel("Max: 1")
        
        slider_layout.addWidget(self.thresh_min_label, 0, 0)
        slider_layout.addWidget(self.thresh_max_label, 0, 1)
        slider_layout.addWidget(self.thresh_min_slider, 1, 0)
        slider_layout.addWidget(self.thresh_max_slider, 1, 1)

        threshold_layout.addRow(self.slider_container)
        
        self.thresh_min_slider.valueChanged.connect(self._update_threshold_preview)
        self.thresh_max_slider.valueChanged.connect(self._update_threshold_preview)

        layout.addWidget(threshold_group)
        self.slider_container.setVisible(False) 

        # --- Group 2: Computation ---
        compute_group = QGroupBox("Compute & Plot Phasor")
        compute_layout = QFormLayout(compute_group)
        
        self.mask_combobox = QComboBox()
        self.viewer.layers.events.inserted.connect(self._update_mask_combobox)
        self.viewer.layers.events.removed.connect(self._update_mask_combobox)
        compute_layout.addRow("Select Mask Layer:", self.mask_combobox)

        self.per_object_radio = QRadioButton("Per Object (average per label)"); self.per_object_radio.setChecked(True)
        self.per_pixel_radio = QRadioButton("Per Pixel (within mask)")
        mode_hbox = QHBoxLayout()
        mode_hbox.addWidget(self.per_object_radio); mode_hbox.addWidget(self.per_pixel_radio)
        compute_layout.addRow("Mode:", mode_hbox)
        
        self.btn_compute = QPushButton("Plot Phasor")
        self.btn_compute.clicked.connect(self._on_compute_clicked)
        compute_layout.addRow(self.btn_compute)
        layout.addWidget(compute_group)
        
        # theme selection 
        options_group = QGroupBox("Plot Options")
        options_layout = QHBoxLayout(options_group)
        self.dark_mode_check = QCheckBox("Use Dark Theme")
        self.dark_mode_check.setChecked(True) # Default to dark
        self.dark_mode_check.toggled.connect(self._on_theme_changed)
        options_layout.addWidget(self.dark_mode_check)
        options_layout.addStretch()
        layout.addWidget(options_group)

        # --- Group 3: Phasor Plot Display ---
        plot_group = QGroupBox("Phasor Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.phasor_figure = Figure(figsize=(5, 5))
        self.phasor_canvas = FigureCanvas(self.phasor_figure)
        self.phasor_ax = self.phasor_figure.add_subplot(111)
        plot_layout.addWidget(self.phasor_canvas)
        layout.addWidget(plot_group)
        
        layout.addStretch()


    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        """Public slot to receive a new dataset."""
        self.dataset = dataset
        self._update_mask_combobox()


    @Slot(dict)
    def on_slice_changed(self, data_package: dict):
        """
        Public slot to receive the currently active data slice from FlimViewPanel.
        This is called every time the user changes a slider or sum checkbox.
        """
        # Update the internal state with the new slice
        self.photon_count = data_package.get('intensity')
        self.phasor_g = data_package.get('phasor_g')
        self.phasor_s = data_package.get('phasor_s')

        if self.photon_count is not None:
            max_photons = int(np.nanmax(self.photon_count))
            if max_photons > 0:
                self.thresh_min_slider.setRange(0, max_photons)
                self.thresh_max_slider.setRange(0, max_photons)
        
        self.btn_compute.setEnabled(self.phasor_g is not None)


    def _update_mask_combobox(self, event=None):
        current_text = self.mask_combobox.currentText()
        self.mask_combobox.clear()
        self.mask_combobox.addItem("none", "None")
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.mask_combobox.addItem(layer.name, layer.name)
        index = self.mask_combobox.findText(current_text)
        if index != -1: self.mask_combobox.setCurrentIndex(index)


    def _update_threshold_label(self, min_val, max_val):
        self.thresh_min_label.setText(f"Min: {min_val}")
        self.thresh_max_label.setText(f"Max: {max_val}")


    def _create_threshold_mask(self):
        if self.photon_count is None:
            QMessageBox.warning(self, "No Data", "Photon count data is not available to create a mask."); return
        
        min_photons = self.thresh_slider.value()
        mask_data = (self.photon_count >= min_photons).astype(np.uint8)
        
        layer_name = f"threshold_mask"
        self.viewer.add_labels(mask_data, name=layer_name)


    def _on_compute_clicked(self):
        if self.phasor_g is None:
            QMessageBox.warning(self, "No Data", "Phasor data is not available."); return

        mask_name = self.mask_combobox.currentData()
        mask_array, mask_layer = None, None
        
        if mask_name != "None":
            if mask_name not in self.viewer.layers:
                 QMessageBox.critical(self, "Error", f"Mask layer '{mask_name}' no longer exists."); return
            mask_layer = self.viewer.layers[mask_name]
            mask_array = mask_layer.data
            
            if mask_array.shape != self.photon_count.shape:
                 QMessageBox.warning(self, "Shape Mismatch", f"Mask shape {mask_array.shape} does not match image shape {self.photon_count.shape}."); return

        if self.per_pixel_radio.isChecked() and mask_name == "None":
            reply = QMessageBox.question(self, 'Performance Warning',
                "Computing per-pixel phasor for the entire image may be slow. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            phasor_array = self.phasor_g + 1j * self.phasor_s
            g_coords, s_coords, colors = [], [], []

            if self.per_pixel_radio.isChecked():
                mask_to_use = mask_array if mask_array is not None else np.ones_like(self.photon_count, dtype=np.uint8)
                valid_pixels = np.isfinite(phasor_array) & (self.photon_count > 0) & (mask_to_use > 0)
                
                g_coords = self.phasor_g[valid_pixels]
                s_coords = self.phasor_s[valid_pixels]
                
                if mask_layer is not None:
                    pixel_labels = mask_array[valid_pixels]
                    colormap = mask_layer.colormap.colors
                    colors = colormap[pixel_labels % len(colormap)]
                else:
                    colors = 'white'
            
            elif self.per_object_radio.isChecked():
                if mask_array is None:
                    avg_phasor = average_phasor(phasor_array, self.photon_count)
                    if np.isfinite(avg_phasor):
                        g_coords, s_coords, colors = [avg_phasor.real], [avg_phasor.imag], ['white']
                else:
                    label_ids = np.unique(mask_array); label_ids = label_ids[label_ids > 0]
                    colormap = mask_layer.colormap.colors
                    
                    for label_id in label_ids:
                        avg_phasor = average_phasor(phasor_array, self.photon_count, mask=(mask_array == label_id))
                        if np.isfinite(avg_phasor):
                            g_coords.append(avg_phasor.real); s_coords.append(avg_phasor.imag)
                            colors.append(colormap[label_id % len(colormap)])

            instrument_params = self.dataset.attrs.get('instrument_params', {})
            sync_rate = instrument_params.get('TTResult_SyncRate', 40e6)
            self.last_plot_data = {
                "g": np.array(g_coords),
                "s": np.array(s_coords),
                "colors": colors,
                "sync_rate": sync_rate,
                "mask_layer": mask_layer
            }            
            self._plot_phasor(**self.last_plot_data)

        finally:
            QApplication.restoreOverrideCursor()


    def _plot_phasor(self, g=None, s=None, colors=None, sync_rate=40e6, message=None, mask_layer=None):
        """Clears and redraws the phasor plot with the selected theme."""
        self.phasor_ax.clear()

        if self.dark_mode_check.isChecked():
            dark_plot(self.phasor_ax, self.phasor_figure)
            circle_color, edge_color = 'white', 'w'
        else:
            light_plot(self.phasor_ax, self.phasor_figure)
            circle_color, edge_color = 'black', 'k'
        
        draw_unitary_circle(self.phasor_ax, sync_rate, color=circle_color, label_color=circle_color)
        
        if message:
            self.phasor_ax.text(0.5, 0.5, message, color=circle_color, ha='center', va='center')
        
        if g is not None and g.size > 0:
            point_size = 3 if g.size > 1000 else 40
            self.phasor_ax.scatter(g, s, c=colors, s=point_size, alpha=0.8, edgecolor=edge_color, linewidth=0.2)
        
        self.phasor_ax.set_xlabel('g'); self.phasor_ax.set_ylabel('s')
        
        self.phasor_canvas.draw_idle()


    def _on_threshold_enable_toggled(self, checked):
        """Shows/hides the sliders and the preview layer."""
        self.slider_container.setVisible(checked)
        
        preview_layer_name = "threshold_preview"
        if checked:
            if self.photon_count is None:
                QMessageBox.warning(self, "No Data", "Photon count data is not available.")
                self.thresh_enable_check.setChecked(False) 
                return
            
            if preview_layer_name not in self.viewer.layers:
                mask_data = np.zeros_like(self.photon_count, dtype=np.uint8)
                self.viewer.add_labels(mask_data, name=preview_layer_name, opacity=0.5)
            
            self.viewer.layers[preview_layer_name].visible = True
            self._update_threshold_preview() 
        else:
            if preview_layer_name in self.viewer.layers:
                self.viewer.layers[preview_layer_name].visible = False

    def _update_threshold_preview(self):
        """Calculates and updates the preview mask layer in real-time."""
        preview_layer_name = "threshold_preview"
        if self.photon_count is None or preview_layer_name not in self.viewer.layers:
            return

        min_val = self.thresh_min_slider.value()
        max_val = self.thresh_max_slider.value()

        self.thresh_min_label.setText(f"Min: {min_val}")
        self.thresh_max_label.setText(f"Max: {max_val}")
        
        if min_val > max_val:
            self.thresh_min_slider.setValue(max_val)
            return 

        mask_data = (self.photon_count >= min_val) & (self.photon_count <= max_val)
        self.viewer.layers[preview_layer_name].data = mask_data.astype(np.uint8)


    def _on_theme_changed(self):
        """Triggers a replot of the last data with the new theme."""
        if hasattr(self, 'last_plot_data'):
             self._plot_phasor(**self.last_plot_data)
        else:
             self._plot_phasor(message="No data loaded or computed.")