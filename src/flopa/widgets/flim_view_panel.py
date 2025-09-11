# flopa/widgets/flim_view_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QMessageBox, QCheckBox, QHBoxLayout, 
    QGridLayout, QLabel, QSpinBox, QFormLayout, QComboBox, QPushButton
)
from qtpy.QtCore import Signal, Slot, QSize
from qtpy.QtGui import QFont, QIcon
from pathlib import Path
from magicgui.widgets import ComboBox as MagicComboBox
import numpy as np
import xarray as xr
from matplotlib import cm
import traceback

from flopa.io.ptuio.utils import create_FLIM_image, aggregate_dataset, smooth_weighted
from .utils.style import apply_style, GROUP_BOX_STYLE_A, GROUP_BOX_STYLE_B
from .histogram_slider import HistogramSlider

class FlimViewPanel(QWidget):
    view_changed = Signal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None; self.selectors = {}
        self._cached_intensity = None; self._cached_lifetime = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self); self.view_controls_container = QGroupBox("FLIM View")
        apply_style(self.view_controls_container, GROUP_BOX_STYLE_B)
        self.view_controls_container.setFixedHeight(160)
        self.view_layout = QVBoxLayout(self.view_controls_container); main_layout.addWidget(self.view_controls_container)
        main_layout.addStretch(); self.setVisible(False)

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        """Receives the full dataset and rebuilds the control UI."""
        self.dataset = dataset
        has_intensity = "photon_count" in self.dataset.data_vars
        has_lifetime = "mean_arrival_time" in self.dataset.data_vars
        if not has_intensity and not has_lifetime: self.setVisible(False); return
        self.setVisible(True)
        self._create_view_controls(has_intensity, has_lifetime)

    def _create_view_controls(self, has_intensity, has_lifetime):
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

        # --- Processing Row ---
        self.smooth_check = QCheckBox("Smooth Image")
        self.smooth_size_spin = QSpinBox()
        self.smooth_size_spin.setRange(2, 30); self.smooth_size_spin.setSingleStep(1); self.smooth_size_spin.setValue(1)
        self.smooth_size_spin.setEnabled(False)
        self.smooth_check.toggled.connect(self.smooth_size_spin.setEnabled)

        # Create another compound widget for the smoothing controls
        smooth_hbox = QHBoxLayout()
        smooth_hbox.addWidget(self.smooth_check)
        smooth_hbox.addWidget(QLabel("Kernel Size:"))
        smooth_hbox.addWidget(self.smooth_size_spin)
        smooth_hbox.addStretch() # This works inside a QHBoxLayout

        # Add the smoothing controls as a new row in the form.
        # It doesn't need a label on the left, so we can pass the widget as the first argument.
        controls_layout.addRow(smooth_hbox)
        grid_layout.addWidget(controls_group, 0, 0, 2, 1) # Span 2 rows to align with other columns

        # # grid_layout.setVerticalSpacing(5)

        # grid_layout.setRowStretch(0, 4) 
        # grid_layout.setRowStretch(1, 1)
        # slice_group = QGroupBox(""); slice_layout = QFormLayout(slice_group)
        # frame_selector = QSpinBox(); frame_selector.setRange(0, max(0, n_frames - 1))
        # sum_frames_check = QCheckBox("Aggregate"); sum_frames_check.setEnabled(n_frames > 1)
        # hbox_f = QHBoxLayout(); hbox_f.addWidget(frame_selector); hbox_f.addWidget(sum_frames_check); slice_layout.addRow("Frame:", hbox_f)
        # sequence_selector = QSpinBox(); sequence_selector.setRange(0, max(0, n_sequences - 1))
        # sum_sequences_check = QCheckBox("Aggregate"); sum_sequences_check.setEnabled(n_sequences > 1)
        # hbox_s = QHBoxLayout(); hbox_s.addWidget(sequence_selector); hbox_s.addWidget(sum_sequences_check); slice_layout.addRow("Sequence:", hbox_s)
        # detector_selector = QSpinBox(); detector_selector.setRange(0, max(0, n_detectors - 1))
        # sum_detectors_check = QCheckBox("Aggregate"); sum_detectors_check.setEnabled(n_detectors > 1)
        # hbox_c = QHBoxLayout(); hbox_c.addWidget(detector_selector); hbox_c.addWidget(sum_detectors_check); slice_layout.addRow("Detector:", hbox_c)

        # grid_layout.addWidget(slice_group, 0, 0)

        # processing_group = QGroupBox("")
        # processing_layout = QHBoxLayout(processing_group)
        
        # self.smooth_check = QCheckBox("Smooth Image")
        # self.smooth_size_spin = QSpinBox()
        # self.smooth_size_spin.setRange(2, 15); self.smooth_size_spin.setSingleStep(1); self.smooth_size_spin.setValue(2)
        # self.smooth_size_spin.setEnabled(False)
        # self.smooth_check.toggled.connect(self.smooth_size_spin.setEnabled)

        # processing_layout.addWidget(self.smooth_check)
        # processing_layout.addWidget(QLabel("Kernel Size:"))
        # processing_layout.addWidget(self.smooth_size_spin)
        # processing_layout.addStretch()
        
        # grid_layout.addWidget(processing_group, 1, 0)

        # --- Column 2: Intensity Controls ---
        instrument_params = self.dataset.attrs.get('instrument_params', {})
        lifetime_units = instrument_params.get('resolution_unit', 'ch')

        intensity_group = QGroupBox("Intensity")
        intensity_layout = QHBoxLayout(intensity_group)
        apply_style(intensity_group, GROUP_BOX_STYLE_A)
        self.intensity_slider = HistogramSlider(integer_mode=True)
        int_controls_hbox = QVBoxLayout()
        self.show_intensity_check = QCheckBox("Visible")
        self.show_intensity_check.setChecked(True)
        self.int_colormap_combo = QComboBox()
        self.int_colormap_combo.addItems(["gray", "viridis", "magma"])
        # --- SET DEFAULT VALUE ---
        self.int_colormap_combo.setCurrentText("gray")
        int_controls_hbox.addWidget(self.show_intensity_check)
        int_controls_hbox.addWidget(self.int_colormap_combo)
        
        intensity_layout.addWidget(self.intensity_slider)
        intensity_layout.addLayout(int_controls_hbox)
        grid_layout.addWidget(intensity_group, 0, 1, 2, 1)

        # --- Column 3: Lifetime Controls ---
        lifetime_group = QGroupBox(f"Lifetime ({lifetime_units})")
        lifetime_layout = QHBoxLayout(lifetime_group)
        apply_style(lifetime_group, GROUP_BOX_STYLE_A)
        self.lifetime_slider = HistogramSlider(integer_mode=False)
        lt_controls_hbox = QVBoxLayout()
        self.show_lifetime_check = QCheckBox("Visible")
        self.show_lifetime_check.setChecked(True)
        self.lt_colormap_combo = QComboBox()
        self.lt_colormap_combo.addItems(["rainbow", "hsv", "viridis"])
        # --- SET DEFAULT VALUE ---
        self.lt_colormap_combo.setCurrentText("rainbow")
        lt_controls_hbox.addWidget(self.show_lifetime_check)
        lt_controls_hbox.addWidget(self.lt_colormap_combo)

        lifetime_layout.addWidget(self.lifetime_slider)
        lifetime_layout.addLayout(lt_controls_hbox)
        grid_layout.addWidget(lifetime_group, 0, 2, 2, 1)

        # --- NEW: Column 4 for Export ---
        export_group = QGroupBox("Export"); export_layout = QVBoxLayout(export_group)
        apply_style(export_group, GROUP_BOX_STYLE_A)

        self.export_raw_intensity_check = QCheckBox("Raw Intensity")
        self.export_raw_lifetime_check = QCheckBox("Raw Lifetime")
        self.export_rgb_flim_check = QCheckBox("RGB FLIM Image")
        export_layout.addWidget(self.export_raw_intensity_check)
        export_layout.addWidget(self.export_raw_lifetime_check)
        export_layout.addWidget(self.export_rgb_flim_check)
        # self.export_format_combo = QComboBox(); self.export_format_combo.addItems(["HDF5 (.h5)", "TIFF Stack (.tif)"])
        # export_layout.addWidget(self.export_format_combo)
        # self.btn_export = QPushButton("Export..."); self.btn_export.clicked.connect(self._on_export)
        # export_layout.addStretch()
        # export_layout.addWidget(self.btn_export)
        export_action_hbox = QHBoxLayout()
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["HDF5 (.h5)", "TIFF Stack (.tif)"])
        self.btn_export = QPushButton(); icon_path = "./assets/icons/save_icon.png" 
        if Path(icon_path).exists(): self.btn_export.setIcon(QIcon(icon_path)); self.btn_export.setIconSize(QSize(16, 16))
        else: self.btn_export.setText("Save")
        self.btn_export.clicked.connect(self._on_export)
        
        # Add the combo box and button to the horizontal layout
        export_action_hbox.addWidget(self.export_format_combo)
        export_action_hbox.addWidget(self.btn_export)
        export_action_hbox.addStretch()

        # Add this horizontal row of controls to the main vertical layout
        export_layout.addLayout(export_action_hbox)
        grid_layout.addWidget(export_group, 0, 3, 2, 1)

        intensity_group.setEnabled(has_intensity)
        lifetime_group.setEnabled(has_lifetime)
        export_group.setEnabled(has_intensity or has_lifetime)
        self.export_raw_intensity_check.setEnabled(has_intensity); self.export_raw_lifetime_check.setEnabled(has_lifetime)
        self.export_rgb_flim_check.setEnabled(has_intensity and has_lifetime)

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

                # --- 2. NEW: Apply smoothing if requested ---
                if self.smooth_check.isChecked() and raw_intensity is not None:
                    kernel_size = self.smooth_size_spin.value()
                    smoothed_intensity, _ = smooth_weighted(raw_intensity, raw_intensity, size=kernel_size)
                    smoothed_lifetime = None
                    if has_lifetime and raw_lifetime is not None:
                        smoothed_lifetime, _ = smooth_weighted(raw_lifetime, raw_intensity, size=kernel_size)
                    self._cached_intensity = np.atleast_2d(smoothed_intensity); 
                    self._cached_lifetime = np.atleast_2d(smoothed_lifetime) * tcspc_res_ns if smoothed_lifetime is not None else None
                else:
                    self._cached_intensity = np.atleast_2d(raw_intensity) if raw_intensity is not None else None
                    self._cached_lifetime = np.atleast_2d(raw_lifetime) * tcspc_res_ns if raw_lifetime is not None else None
                # for layer_name in ['FLIM', 'Intensity', 'Lifetime']:
                #     if layer_name in self.viewer.layers: self.viewer.layers.remove(layer_name)

                # Add the initial layers
                if self._cached_intensity is not None: self.intensity_slider.update_data(self._cached_intensity)
                if self._cached_lifetime is not None: self.lifetime_slider.update_data(self._cached_lifetime)
            
                # for layer_name in ['FLIM', 'Intensity', 'Lifetime']:
                #     if layer_name in self.viewer.layers: self.viewer.layers.remove(layer_name)

                # # Add the new persistent layers, initially invisible
                # if has_intensity and has_lifetime: self.viewer.add_image(np.zeros((*self._cached_intensity.shape, 3), dtype=np.float32), name='FLIM', rgb=True, visible=False)
                # if has_intensity: self.viewer.add_image(self._cached_intensity, name='Intensity', colormap='gray', visible=False)
                # if has_lifetime:
                #     self.viewer.add_image(self._cached_lifetime, name='Lifetime', colormap='rainbow', visible=False)
                
                # update_display() # Call fast update to set initial visibility and properties
                update_display(recalculate_rgb=True, create_new_layers=True)

                self.viewer.reset_view()
                self.view_changed.emit(self.selectors)

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
                    
                    self.viewer.reset_view()


                self.int_colormap_combo.setEnabled(is_intensity_mode)
                if is_flim_mode and self.int_colormap_combo.currentText() != 'gray': self.int_colormap_combo.setCurrentText('gray')
                
                # if is_flim_mode:
                #     # --- The key performance logic ---
                #     # Only recalculate the expensive RGB image when told to
                #     if recalculate_rgb and 'FLIM' in self.viewer.layers:
                #         lt_min, lt_max = self.lifetime_slider.value()
                #         int_min, int_max = self.intensity_slider.value()
                #         self.viewer.layers['FLIM'].data = create_FLIM_image(mean_photon_arrival_time=final_lifetime, intensity=final_intensity, colormap=cm.get_cmap(self.lt_colormap_combo.currentText()), lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)
                #         self.viewer.layers['FLIM'].visible = True
                #     if 'Intensity' in self.viewer.layers: self.viewer.layers['Intensity'].visible = False
                #     if 'Lifetime' in self.viewer.layers: self.viewer.layers['Lifetime'].visible = False
                   
                #     # Note: We don't need to do anything on a simple drag, as the
                #     # histogram lines and contrast limits are updated separately.
                    
                # elif is_intensity_mode:
                #     if 'Intensity' in self.viewer.layers:
                #         int_min, int_max = self.intensity_slider.value()
                #         self.viewer.layers['Intensity'].contrast_limits = (int_min, int_max)
                #         self.viewer.layers['Intensity'].colormap = self.int_colormap_combo.currentText()
                #         self.viewer.layers['Intensity'].visible = True
                #     if 'FLIM' in self.viewer.layers: self.viewer.layers['FLIM'].visible = False
                #     if 'Lifetime' in self.viewer.layers: self.viewer.layers['Lifetime'].visible = False

                # elif is_lifetime_mode:
                #     if 'Lifetime' in self.viewer.layers:
                #         lt_min, lt_max = self.lifetime_slider.value()
                #         self.viewer.layers['Lifetime'].contrast_limits = (lt_min, lt_max)
                #         self.viewer.layers['Lifetime'].colormap = self.lt_colormap_combo.currentText()
                #         self.viewer.layers['Lifetime'].visible = True
                #     if 'FLIM' in self.viewer.layers: self.viewer.layers['FLIM'].visible = False
                #     if 'Intensity' in self.viewer.layers: self.viewer.layers['Intensity'].visible = False

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
                


                # # Update layer properties (very fast)
                # if is_intensity_mode:
                #     int_min, int_max = self.intensity_slider.value()
                #     self.viewer.layers['Intensity'].contrast_limits = (int_min, int_max)
                #     self.viewer.layers['Intensity'].colormap = self.int_colormap_combo.currentText()
                
                # # The FLIM image update is slightly slower, so it has its own function
                # if is_flim_mode:
                #     recalculate_flim_rgb()

                # self._ensure_preview_on_top()
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
            
        self.smooth_check.toggled.connect(update_data_slice)
        self.smooth_size_spin.valueChanged.connect(update_data_slice)

        self.intensity_slider.valueChanged.connect(lambda: update_display(recalculate_rgb=False))
        
        # SHOW/HIDE CHECKBOXES -> calls update_display WITH recalculating RGB
        self.show_intensity_check.toggled.connect(lambda: update_display(recalculate_rgb=True))
        
        # COLORMAPS -> call update_colormaps (which then calls update_display with recalc)
        self.int_colormap_combo.currentTextChanged.connect(update_colormaps)
        
        if has_lifetime:
            self.lifetime_slider.valueChanged.connect(lambda: update_display(recalculate_rgb=False))
            # SLIDER RELEASE -> calls update_display WITH recalculating RGB
            self.intensity_slider.sliderReleased.connect(lambda: update_display(recalculate_rgb=True))
            self.lifetime_slider.sliderReleased.connect(lambda: update_display(recalculate_rgb=True))
            
            self.show_lifetime_check.toggled.connect(lambda: update_display(recalculate_rgb=True))
            self.lt_colormap_combo.currentTextChanged.connect(update_colormaps)

        update_data_slice()
        update_colormaps()


    def _on_export(self):
        """Placeholder function for the export button."""
        export_format = self.export_format_combo.currentText()
        
        data_to_save = []
        if self.export_raw_intensity_check.isChecked():
            data_to_save.append("Raw Intensity")
        if self.export_raw_lifetime_check.isChecked():
            data_to_save.append("Raw Lifetime")
        if self.export_rgb_flim_check.isChecked():
            data_to_save.append("RGB FLIM")

        if not data_to_save:
            QMessageBox.warning(self, "Export", "Please select at least one data type to export.")
            return

        message = (
            f"Exporting to format: {export_format}\n\n"
            f"Data selected:\n- " + "\n- ".join(data_to_save) +
            "\n\n(Export logic not yet implemented)"
        )
        QMessageBox.information(self, "Export Action Triggered", message)