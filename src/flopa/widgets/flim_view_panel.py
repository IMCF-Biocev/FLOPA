# flopa/widgets/flim_view_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QMessageBox, QCheckBox, QHBoxLayout, 
    QGridLayout, QLabel, QSpinBox, QFormLayout, QComboBox, QPushButton, QFileDialog, QButtonGroup, QRadioButton
)
from qtpy.QtCore import Signal, Slot, QSize, Qt
from qtpy.QtGui import QFont, QIcon
from pathlib import Path
from magicgui.widgets import ComboBox as MagicComboBox
import numpy as np
import xarray as xr
from matplotlib import cm
import traceback

from flopa.io.ptuio.utils import aggregate_dataset, smooth_weighted, smooth_count
from .utils.style import apply_style, GROUP_BOX_STYLE_A, GROUP_BOX_STYLE_B
from .histogram_slider import HistogramSlider
from flopa.processing.flim_image import create_FLIM_image
from .utils.exporter import export_dataset_as_hdf5, export_view_as_tiff


class FlimViewPanel(QWidget):
    view_changed = Signal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.selectors = {}
        self._cached_intensity = None
        self._cached_lifetime = None
        self.export_status_label = QLabel("")

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.view_controls_container = QGroupBox("FLIM View")
        apply_style(self.view_controls_container, GROUP_BOX_STYLE_B)
        self.view_controls_container.setFixedHeight(160)
        self.view_layout = QVBoxLayout(self.view_controls_container) 
        main_layout.addWidget(self.view_controls_container)
        main_layout.addStretch()
        self.setVisible(False)

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset, is_from_h5: False):
        """Receives the full dataset and rebuilds the control UI."""
        self.dataset = dataset
        has_intensity = "photon_count" in self.dataset.data_vars
        has_lifetime = "mean_arrival_time" in self.dataset.data_vars
        if not has_intensity and not has_lifetime: self.setVisible(False); return
        self.setVisible(True)
        self._create_view_controls(has_intensity, has_lifetime, is_from_h5)

    def _create_view_controls(self, has_intensity, has_lifetime, is_from_h5):
        while self.view_layout.count():
            child = self.view_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        n_frames = self.dataset.sizes.get('frame', 1)
        n_sequences = self.dataset.sizes.get('sequence', 1)
        n_detectors = self.dataset.sizes.get('detector', 1)

        grid_layout = QGridLayout(); self.view_layout.addLayout(grid_layout)

        controls_group = QGroupBox("")
        controls_layout = QFormLayout(controls_group)
        apply_style(controls_group, GROUP_BOX_STYLE_A)

        # --- Slicing Rows ---
        frame_selector = QSpinBox(); frame_selector.setRange(0, max(0, n_frames - 1))
        sum_frames_check = QCheckBox("Aggregate"); sum_frames_check.setEnabled(n_frames > 1)
        # Create a compound widget for the right-hand side of the form row
        hbox_f = QHBoxLayout(); hbox_f.addWidget(frame_selector); hbox_f.addWidget(sum_frames_check)
        controls_layout.addRow("Frame:", hbox_f)

        sequence_selector = QSpinBox(); sequence_selector.setRange(0, max(0, n_sequences - 1))
        sum_sequences_check = QCheckBox("Aggregate"); sum_sequences_check.setEnabled(n_sequences > 1)
        hbox_s = QHBoxLayout(); hbox_s.addWidget(sequence_selector); hbox_s.addWidget(sum_sequences_check)
        controls_layout.addRow("Sequence:", hbox_s)

        detector_selector = QSpinBox(); detector_selector.setRange(0, max(0, n_detectors - 1))
        sum_detectors_check = QCheckBox("Aggregate"); sum_detectors_check.setEnabled(n_detectors > 1)
        hbox_c = QHBoxLayout(); hbox_c.addWidget(detector_selector); hbox_c.addWidget(sum_detectors_check)
        controls_layout.addRow("Detector:", hbox_c)

        grid_layout.addWidget(controls_group, 0, 0, 2, 1) # Span 2 rows to align with other columns


        # --- Column 2: Intensity Controls ---
        instrument_params = self.dataset.attrs.get('instrument_params', {})
        lifetime_units = instrument_params.get('resolution_unit', 'ch')

        intensity_group = QGroupBox("Intensity")
        intensity_layout = QHBoxLayout(intensity_group)
        apply_style(intensity_group, GROUP_BOX_STYLE_A)
        self.intensity_slider = HistogramSlider(integer_mode=True)
        int_controls_hbox = QHBoxLayout()
        int_controls_hbox.addWidget(QLabel("ON:"))
        self.show_intensity_check = QCheckBox()
        self.show_intensity_check.setChecked(True)
        self.int_colormap_combo = QComboBox()
        self.int_colormap_combo.addItems(["gray", "viridis", "magma"])
        # --- SET DEFAULT VALUE ---
        self.int_colormap_combo.setCurrentText("gray")
        int_controls_hbox.addWidget(self.show_intensity_check)
        int_controls_hbox.addWidget(self.int_colormap_combo)
        int_controls_hbox.addStretch()

        
        int_smooth_hbox = QHBoxLayout()
        self.smooth_intensity_check = QCheckBox()
        self.smooth_intensity_spin = QSpinBox(); self.smooth_intensity_spin.setRange(2, 30); self.smooth_intensity_spin.setValue(3)
        self.smooth_intensity_spin.setEnabled(False)
        self.smooth_intensity_check.toggled.connect(self.smooth_intensity_spin.setEnabled)
        int_smooth_hbox.addWidget(QLabel("Smooth"))
        int_smooth_hbox.addWidget(self.smooth_intensity_check)
        # int_smooth_hbox.addStretch()

        # int_kernel_hbox = QHBoxLayout()
        # int_kernel_hbox.addWidget(QLabel("Kernel"))
        int_smooth_hbox.addWidget(self.smooth_intensity_spin)
        # int_kernel_hbox.addStretch()

        int_controls_vbox = QVBoxLayout()
        int_controls_vbox.addLayout(int_controls_hbox)
        int_controls_vbox.addLayout(int_smooth_hbox)
        # int_controls_vbox.addLayout(int_kernel_hbox)

        intensity_layout.addWidget(self.intensity_slider)
        intensity_layout.addLayout(int_controls_vbox)
        intensity_layout.addStretch()
        grid_layout.addWidget(intensity_group, 0, 1, 2, 1)

        # --- Column 3: Lifetime Controls ---
        lifetime_group = QGroupBox(f"Lifetime ({lifetime_units})")
        lifetime_layout = QHBoxLayout(lifetime_group)
        apply_style(lifetime_group, GROUP_BOX_STYLE_A)
        self.lifetime_slider = HistogramSlider(integer_mode=False)
        
        lt_controls_hbox = QHBoxLayout()
        lt_controls_hbox.addWidget(QLabel("ON:"))
        self.show_lifetime_check = QCheckBox()
        self.show_lifetime_check.setChecked(True)
        self.lt_colormap_combo = QComboBox()
        self.lt_colormap_combo.addItems(["rainbow", "hsv", "viridis"])
        # --- SET DEFAULT VALUE ---
        self.lt_colormap_combo.setCurrentText("rainbow")
        lt_controls_hbox.addWidget(self.show_lifetime_check)
        lt_controls_hbox.addWidget(self.lt_colormap_combo)
        lt_controls_hbox.addStretch()

        lt_smooth_hbox = QHBoxLayout()
        self.smooth_lifetime_check = QCheckBox()
        self.smooth_lifetime_spin = QSpinBox(); self.smooth_lifetime_spin.setRange(2, 30); self.smooth_lifetime_spin.setValue(3)
        self.smooth_lifetime_spin.setEnabled(False)
        self.smooth_lifetime_check.toggled.connect(self.smooth_lifetime_spin.setEnabled)
        lt_smooth_hbox.addWidget(QLabel("Smooth"))
        lt_smooth_hbox.addWidget(self.smooth_lifetime_check)
        lt_smooth_hbox.addWidget(self.smooth_lifetime_spin)

        lt_controls_vbox = QVBoxLayout()
        lt_controls_vbox.addLayout(lt_controls_hbox)
        lt_controls_vbox.addLayout(lt_smooth_hbox)

        lifetime_layout.addWidget(self.lifetime_slider)
        lifetime_layout.addLayout(lt_controls_vbox)
        intensity_layout.addStretch()
        grid_layout.addWidget(lifetime_group, 0, 2, 2, 1)

        # --- Column 4: Export Controls ---
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        apply_style(export_group, GROUP_BOX_STYLE_A)

        # Row 1: Format Selector and Save Button
        format_hbox = QHBoxLayout()
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["TIFF", "HDF5 (Full Dataset)"])
        format_hbox.addWidget(self.export_format_combo)
        self.btn_export = QPushButton(); icon_path = "./assets/icons/save_icon.png" 
        if Path(icon_path).is_file(): self.btn_export.setIcon(QIcon(icon_path)); self.btn_export.setIconSize(QSize(16, 16))
        else: self.btn_export.setText("Plot")
        # self.btn_export = QPushButton("Save...")
        self.btn_export.clicked.connect(self._on_export)
        format_hbox.addWidget(self.btn_export)
        export_layout.addLayout(format_hbox)

        # Container for TIFF-specific options
        self.tiff_options_container = QWidget()
        tiff_options_layout = QVBoxLayout(self.tiff_options_container)
        tiff_options_layout.setContentsMargins(0, 5, 0, 0)
        export_layout.addWidget(self.tiff_options_container)

        # Row 2: Data Selection (Checkboxes, simplified)
        data_hbox = QHBoxLayout()
        self.export_intensity_check = QCheckBox("Intensity")
        self.export_lifetime_check = QCheckBox("Lifetime")
        self.export_rgb_check = QCheckBox("RGB FLIM")
        
        data_hbox.addWidget(self.export_intensity_check)
        data_hbox.addWidget(self.export_lifetime_check)
        data_hbox.addWidget(self.export_rgb_check)
        data_hbox.addStretch()
        tiff_options_layout.addLayout(data_hbox)

        self.export_status_label = QLabel("") # Create the label
        self.export_status_label.setStyleSheet("color: gray; font-style: italic;")
        self.export_status_label.setAlignment(Qt.AlignCenter)
        export_layout.addWidget(self.export_status_label) # Add it to the layout

        grid_layout.addWidget(export_group, 0, 3, 2, 1)

        intensity_group.setEnabled(has_intensity)
        lifetime_group.setEnabled(has_lifetime)
        export_group.setEnabled(has_intensity or has_lifetime)

        def on_format_changed(text):
            is_hdf5 = "HDF5" in text
            self.tiff_options_container.setEnabled(not is_hdf5)
        
        self.export_format_combo.currentTextChanged.connect(on_format_changed)
        if is_from_h5:
            # If data is from H5, force format to TIFF and disable the combo box
            self.export_format_combo.setCurrentText("TIFF")
            self.export_format_combo.setEnabled(False)
        else:
            self.export_format_combo.setEnabled(True)
        on_format_changed(self.export_format_combo.currentText())

        # Enable/disable based on available data
        self.export_intensity_check.setEnabled(has_intensity)
        self.export_lifetime_check.setEnabled(has_lifetime)
        self.export_rgb_check.setEnabled(has_intensity and has_lifetime)

        self.selectors = {'frame': frame_selector, 'sequence': sequence_selector, 'detector': detector_selector, 'sum_frames': sum_frames_check, 'sum_sequences': sum_sequences_check, 'sum_detectors': sum_detectors_check}

        def update_data_slice():
            """SLOWER function: Re-slices/sums data and updates histograms."""
            try:
                selection_dict, dims_to_sum = {}, []
                for dim in ['frame', 'sequence', 'detector']:
                    if dim in self.dataset.dims and self.selectors[f'sum_{dim}s'].isChecked(): dims_to_sum.append(dim)
                    elif dim in self.dataset.dims: selection_dict[dim] = self.selectors[dim].value()
                
                sliced_ds = self.dataset.isel(**selection_dict)
                final_ds = aggregate_dataset(sliced_ds, dims_to_sum) if dims_to_sum else sliced_ds
                
                # instrument_params = self.dataset.attrs.get('instrument_params', {})
                tcspc_res_ns = instrument_params.get("tcspc_resolution_ns", 1.0)

                raw_intensity = final_ds.photon_count.values.squeeze() if "photon_count" in final_ds else None
                raw_lifetime = final_ds.mean_arrival_time.values.squeeze() if "mean_arrival_time" in final_ds else None

                smoothed_intensity = raw_intensity
                smoothed_lifetime = raw_lifetime
                
                if has_intensity and self.smooth_intensity_check.isChecked():
                    kernel_size = self.smooth_intensity_spin.value()
                    smoothed_intensity = smooth_count(count=raw_intensity, size=kernel_size)

                if has_lifetime and self.smooth_lifetime_check.isChecked():
                    kernel_size = self.smooth_lifetime_spin.value()
                    # Lifetime is weighted by the raw (unsmoothed) intensity
                    smoothed_lifetime, _ = smooth_weighted(array=raw_lifetime, count=raw_intensity, size=kernel_size)
                # --- END MODIFIED ---

                # 3. Cache the processed data for fast updates
                tcspc_res_ns = instrument_params.get("tcspc_resolution_ns", 1.0)
                self._cached_intensity = np.atleast_2d(smoothed_intensity) if smoothed_intensity is not None else None
                self._cached_lifetime = np.atleast_2d(smoothed_lifetime) * tcspc_res_ns if smoothed_lifetime is not None else None
                
                # 4. Update histogram sliders with the new data
                if self._cached_intensity is not None: self.intensity_slider.update_data(self._cached_intensity)
                if self._cached_lifetime is not None: self.lifetime_slider.update_data(self._cached_lifetime)
            
                # 5. Trigger a full display update, creating new layers
                update_display(recalculate_rgb=True, create_new_layers=True)
                self.view_changed.emit(self.selectors)

                # self.viewer.reset_view()

            except Exception: traceback.print_exc()

        def update_display(recalculate_rgb=False, create_new_layers=False):
            """
            FAST function: Applies smoothing (if active) and updates napari
            layer properties without re-adding them.
            """
            try:
                final_intensity, final_lifetime = self._cached_intensity, self._cached_lifetime
                show_intensity = self.show_intensity_check.isChecked(); show_lifetime = self.show_lifetime_check.isChecked()
                is_flim_mode = has_intensity and has_lifetime and show_intensity and show_lifetime
                is_intensity_mode = has_intensity and show_intensity and not is_flim_mode
                is_lifetime_mode = has_lifetime and show_lifetime and not is_flim_mode

                if create_new_layers:
                    for layer_name in ['FLIM', 'Intensity', 'Lifetime']:
                        if layer_name in self.viewer.layers: self.viewer.layers.remove(layer_name)
                    
                    # Create placeholder layers for all possible views
                    if has_intensity and has_lifetime:
                        self.viewer.add_image(np.zeros((*final_intensity.shape, 3), dtype=np.float32), name='FLIM', rgb=True)
                    if has_intensity:
                        self.viewer.add_image(final_intensity, name='Intensity', colormap='gray')
                    if has_lifetime:
                        self.viewer.add_image(final_lifetime, name='Lifetime', colormap='rainbow')
                    
                    # self.viewer.reset_view()


                self.int_colormap_combo.setEnabled(is_intensity_mode)
                if is_flim_mode and self.int_colormap_combo.currentText() != 'gray': self.int_colormap_combo.setCurrentText('gray')
         

                if 'FLIM' in self.viewer.layers:
                    self.viewer.layers['FLIM'].visible = is_flim_mode
                    if is_flim_mode and recalculate_rgb: # Only update data when needed
                        lt_min, lt_max = self.lifetime_slider.value(); int_min, int_max = self.intensity_slider.value()
                        self.viewer.layers['FLIM'].data = create_FLIM_image(mean_photon_arrival_time=final_lifetime, intensity=final_intensity, colormap=cm.get_cmap(self.lt_colormap_combo.currentText()), lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)

                if 'Intensity' in self.viewer.layers:
                    self.viewer.layers['Intensity'].visible = is_intensity_mode
                    if is_intensity_mode:
                        int_min, int_max = self.intensity_slider.value()
                        self.viewer.layers['Intensity'].contrast_limits = (int_min, int_max)
                        self.viewer.layers['Intensity'].colormap = self.int_colormap_combo.currentText()

                if 'Lifetime' in self.viewer.layers:
                    self.viewer.layers['Lifetime'].visible = is_lifetime_mode
                    if is_lifetime_mode:
                        lt_min, lt_max = self.lifetime_slider.value()
                        self.viewer.layers['Lifetime'].contrast_limits = (lt_min, lt_max)
                        self.viewer.layers['Lifetime'].colormap = self.lt_colormap_combo.currentText()
                

            except Exception: traceback.print_exc()


        def update_colormaps():
            if has_intensity: self.intensity_slider.set_colormap(cm.get_cmap(self.int_colormap_combo.currentText()))
            if has_lifetime: self.lifetime_slider.set_colormap(cm.get_cmap(self.lt_colormap_combo.currentText()))
            # Changing a colormap might require an RGB update
            update_display(recalculate_rgb=True)


        # --- Connect signals ---
        for selector in self.selectors.values():
            if isinstance(selector, QSpinBox): selector.valueChanged.connect(update_data_slice)
            if isinstance(selector, QCheckBox): selector.toggled.connect(update_data_slice)
            
        self.smooth_intensity_check.toggled.connect(update_data_slice)
        self.smooth_intensity_spin.valueChanged.connect(update_data_slice)
        
        # --- Connections for fast updates ---
        self.show_intensity_check.toggled.connect(lambda: update_display(recalculate_rgb=True))
        self.int_colormap_combo.currentTextChanged.connect(update_colormaps)
        self.intensity_slider.valueChanged.connect(lambda: update_display(recalculate_rgb=False))
        self.intensity_slider.sliderReleased.connect(lambda: update_display(recalculate_rgb=True))
        
        if has_lifetime:
            # --- NEW: Connect new smoothing widgets to the SLOW update function ---
            self.smooth_lifetime_check.toggled.connect(update_data_slice)
            self.smooth_lifetime_spin.valueChanged.connect(update_data_slice)

            self.show_lifetime_check.toggled.connect(lambda: update_display(recalculate_rgb=True))
            self.lt_colormap_combo.currentTextChanged.connect(update_colormaps)
            self.lifetime_slider.valueChanged.connect(lambda: update_display(recalculate_rgb=False))
            self.lifetime_slider.sliderReleased.connect(lambda: update_display(recalculate_rgb=True))

        update_data_slice()
        update_colormaps()

    # def _on_export(self):
    #     export_format = self.export_format_combo.currentText()
    #     is_hdf5_export = "HDF5" in export_format

    #     if is_hdf5_export:
    #         # HDF5 logic is unchanged
    #         if self.dataset is None:
    #             QMessageBox.warning(self, "Export Error", "No dataset available.")
    #             return
    #         save_path, _ = QFileDialog.getSaveFileName(self, "Save Full Dataset as HDF5", "", "HDF5 Files (*.h5)")
    #         if not save_path: return
    #         success, error = export_dataset_as_hdf5(self.dataset, Path(save_path))
    #         if success: QMessageBox.information(self, "Success", f"Full dataset saved to:\n{save_path}")
    #         else: QMessageBox.critical(self, "Error", f"Failed to export HDF5 file:\n{error}")
        
    #     else: # TIFF Export
    #         items_to_save = {
    #             "intensity": self.export_intensity_check.isChecked(),
    #             "lifetime": self.export_lifetime_check.isChecked(),
    #             "flim_rgb": self.export_rgb_check.isChecked(),
    #         }
    #         if not any(items_to_save.values()):
    #             QMessageBox.warning(self, "Export", "Please select at least one data type to export.")
    #             return
            
    #         save_path, _ = QFileDialog.getSaveFileName(self, "Save Current View as TIFF", "", "TIFF files (*.tif *.tiff)")
    #         if not save_path: return

    #         success, error = export_view_as_tiff( # Note: new simpler function name
    #             output_path=Path(save_path),
    #             items_to_save=items_to_save,
    #             base_dataset=self.dataset,
    #             selectors=self.selectors,
    #             processed_data={
    #                 "intensity": self._cached_intensity,
    #                 "lifetime": self._cached_lifetime,
    #             },
    #             lifetime_colormap=self.lt_colormap_combo.currentText(),
    #             intensity_clims=self.intensity_slider.value(),
    #             lifetime_clims=self.lifetime_slider.value()
    #         )
    #         if success: QMessageBox.information(self, "Success", f"TIFF file(s) saved successfully to:\n{save_path}")
    #         else: QMessageBox.critical(self, "Error", f"Failed to export TIFF file(s):\n{error}")

    def _on_export(self):
        self.export_status_label.setText("")
        export_format = self.export_format_combo.currentText()
        is_hdf5_export = "HDF5" in export_format

        try:
            if is_hdf5_export:
                if self.dataset is None: raise ValueError("No dataset available.")
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Full Dataset as HDF5", "", "HDF5 Files (*.h5)")
                if not save_path: 
                    # self.export_status_label.setText("")
                    return
                success, error = export_dataset_as_hdf5(self.dataset, Path(save_path))
                if not success: raise error
            
            else: # TIFF Export
                items_to_save = {
                    "intensity": self.export_intensity_check.isChecked(),
                    "lifetime": self.export_lifetime_check.isChecked(),
                    "flim_rgb": self.export_rgb_check.isChecked(),
                }
                if not any(items_to_save.values()): raise ValueError("Please select at least one data type to export.")
                
                folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save TIFF Files")
                if not folder_path: return

                # --- THE FIX: The function call now includes all required arguments ---
                success, error = export_view_as_tiff( # Using your latest function name
                    output_folder=Path(folder_path),
                    items_to_save=items_to_save,
                    base_dataset=self.dataset,
                    processed_data={
                        "intensity": self._cached_intensity,
                        "lifetime": self._cached_lifetime,
                    },
                    lifetime_colormap=self.lt_colormap_combo.currentText(),
                    intensity_clims=self.intensity_slider.value(),
                    lifetime_clims=self.lifetime_slider.value()
                )
                if not success: raise error

            self.export_status_label.setStyleSheet("color: #ffbc2b; font-style: italic;") # Green for success
            self.export_status_label.setText("Saved successfully!")

        except Exception as e:
            # --- On failure, update the status label and show a detailed popup ---
            self.export_status_label.setStyleSheet("color: #F44336; font-style: italic;") # Red for failure
            self.export_status_label.setText("Export failed.")
            QMessageBox.warning(self, "Export Error", str(e))