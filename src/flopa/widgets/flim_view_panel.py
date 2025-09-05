# flopa/widgets/flim_view_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QMessageBox, QCheckBox, QHBoxLayout, 
    QGridLayout, QLabel, QSpinBox, QFormLayout
)
from qtpy.QtCore import Signal, Slot
from magicgui.widgets import ComboBox as MagicComboBox
import numpy as np
import xarray as xr
from matplotlib import cm
import traceback

from flopa.io.ptuio.utils import create_FLIM_image, sum_dataset
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
        main_layout = QVBoxLayout(self); self.view_controls_container = QGroupBox("FLIM View Controls")
        self.view_controls_container.setFixedHeight(175)
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

        n_frames = self.dataset.sizes.get('frame', 1); n_sequences = self.dataset.sizes.get('sequence', 1); n_detectors = self.dataset.sizes.get('detector', 1)
        grid_layout = QGridLayout(); self.view_layout.addLayout(grid_layout)
        slice_group = QGroupBox(""); slice_layout = QFormLayout(slice_group)
        frame_selector = QSpinBox(); frame_selector.setRange(0, max(0, n_frames - 1))
        sum_frames_check = QCheckBox("Sum All"); sum_frames_check.setEnabled(n_frames > 1)
        hbox_f = QHBoxLayout(); hbox_f.addWidget(frame_selector); hbox_f.addWidget(sum_frames_check); slice_layout.addRow("Frame:", hbox_f)
        sequence_selector = QSpinBox(); sequence_selector.setRange(0, max(0, n_sequences - 1))
        sum_sequences_check = QCheckBox("Sum All"); sum_sequences_check.setEnabled(n_sequences > 1)
        hbox_s = QHBoxLayout(); hbox_s.addWidget(sequence_selector); hbox_s.addWidget(sum_sequences_check); slice_layout.addRow("Sequence:", hbox_s)
        detector_selector = QSpinBox(); detector_selector.setRange(0, max(0, n_detectors - 1))
        sum_detectors_check = QCheckBox("Sum All"); sum_detectors_check.setEnabled(n_detectors > 1)
        hbox_c = QHBoxLayout(); hbox_c.addWidget(detector_selector); hbox_c.addWidget(sum_detectors_check); slice_layout.addRow("Detector:", hbox_c)
        # grid_layout.addWidget(slice_group, 0, 0)

        # intensity_group = QGroupBox("Intensity"); intensity_layout = QVBoxLayout(intensity_group)
        # self.intensity_slider = HistogramSlider(name="Intensity")
        # self.int_colormap_combo = MagicComboBox(name="Colormap", choices=["gray", "viridis", "magma"])
        # intensity_layout.addWidget(self.intensity_slider); intensity_layout.addWidget(self.int_colormap_combo.native)
        # grid_layout.addWidget(intensity_group, 0, 1)

        # lifetime_group = QGroupBox("Lifetime"); lifetime_layout = QVBoxLayout(lifetime_group)
        # self.lifetime_slider = HistogramSlider(name="Lifetime (ns)")
        # self.lt_colormap_combo = MagicComboBox(name="Colormap", choices=["rainbow", "hsv", "viridis"])
        # lifetime_layout.addWidget(self.lifetime_slider); lifetime_layout.addWidget(self.lt_colormap_combo.native)
        # grid_layout.addWidget(lifetime_group, 0, 2)
        
        # --- Column 1: Slicing Controls (Unchanged) ---
        # ... (Your code to create slice_group is the same) ...
        grid_layout.addWidget(slice_group, 0, 0)

        # --- Column 2: Intensity Controls (New Nested Layout) ---
        intensity_group = QGroupBox("Intensity"); intensity_layout = QVBoxLayout(intensity_group)
        self.intensity_slider = HistogramSlider() # Name is now from the GroupBox title
        int_colormap_choices = ["Hide", "gray", "viridis", "magma"]
        self.int_colormap_combo = MagicComboBox(label="Colormap", choices=int_colormap_choices, value="gray")

        # Create a horizontal layout to hold the slider and its colormap
        int_hbox = QHBoxLayout()
        int_hbox.addWidget(self.intensity_slider)
        int_hbox.addWidget(self.int_colormap_combo.native)
        intensity_layout.addLayout(int_hbox)
        grid_layout.addWidget(intensity_group, 0, 1)

        # --- Column 3: Lifetime Controls (New Nested Layout) ---
        lifetime_group = QGroupBox("Lifetime")
        lifetime_layout = QVBoxLayout(lifetime_group)
        self.lifetime_slider = HistogramSlider() # Name is now from the GroupBox title
        
        # Add a "Hide" option to the lifetime colormap
        lt_colormap_choices = ["Hide", "rainbow", "hsv", "viridis"]
        self.lt_colormap_combo = MagicComboBox(label="Colormap", choices=lt_colormap_choices, value="rainbow")
        
        lt_hbox = QHBoxLayout()
        lt_hbox.addWidget(self.lifetime_slider)
        lt_hbox.addWidget(self.lt_colormap_combo.native)
        lifetime_layout.addLayout(lt_hbox)
        grid_layout.addWidget(lifetime_group, 0, 2)
        
        intensity_group.setEnabled(has_intensity)
        lifetime_group.setEnabled(has_lifetime)

        self.selectors = {'frame': frame_selector, 'sequence': sequence_selector, 'detector': detector_selector, 'sum_frames': sum_frames_check, 'sum_sequences': sum_sequences_check, 'sum_detectors': sum_detectors_check}

        def update_data_slice():
            """SLOWER function: Re-slices/sums data and updates histograms."""
            try:
                selection_dict, dims_to_sum = {}, []
                for dim in ['frame', 'sequence', 'detector']:
                    if dim in self.dataset.dims and self.selectors[f'sum_{dim}s'].isChecked(): dims_to_sum.append(dim)
                    elif dim in self.dataset.dims: selection_dict[dim] = self.selectors[dim].value()
                
                sliced_ds = self.dataset.isel(**selection_dict)
                final_ds = sum_dataset(sliced_ds, dims_to_sum) if dims_to_sum else sliced_ds
                
                self._cached_intensity = np.atleast_2d(final_ds.photon_count.values.squeeze()) if "photon_count" in final_ds else None
                self._cached_lifetime = np.atleast_2d(final_ds.mean_arrival_time.values.squeeze()) if "mean_arrival_time" in final_ds else None

                # Remove all old layers before adding new ones
                for layer_name in ['FLIM', 'Intensity', 'Lifetime (ns)']:
                    if layer_name in self.viewer.layers: self.viewer.layers.remove(layer_name)

                # Add the initial layers
                if self._cached_intensity is not None: self.intensity_slider.update_data(self._cached_intensity)
                if self._cached_lifetime is not None: self.lifetime_slider.update_data(self._cached_lifetime)
            
                update_display(create_new_layers=True) # Force creation of new layers
                self.view_changed.emit(self.selectors)
            except Exception: traceback.print_exc()

        def update_display(create_new_layers=False):
            """FAST function: Updates napari layer properties without re-adding."""
            try:
                final_intensity = self._cached_intensity 
                final_lifetime = self._cached_lifetime

                show_intensity = self.int_colormap_combo.value != "Hide"
                show_lifetime = self.lt_colormap_combo.value != "Hide"

                is_flim_mode = has_intensity and has_lifetime and show_intensity and show_lifetime
                is_intensity_mode = has_intensity and show_intensity and not is_flim_mode
                is_lifetime_mode = has_lifetime and show_lifetime and not is_flim_mode

                if 'FLIM' in self.viewer.layers:
                    self.viewer.layers['FLIM'].visible = is_flim_mode
                if 'Intensity' in self.viewer.layers:
                    self.viewer.layers['Intensity'].visible = is_intensity_mode
                if 'Lifetime (ns)' in self.viewer.layers:
                    self.viewer.layers['Lifetime (ns)'].visible = is_lifetime_mode

                # Logic for full FLIM RGB image
                if is_flim_mode:
                    lt_min, lt_max = self.lifetime_slider.slider.value
                    int_min, int_max = self.intensity_slider.slider.value
                    lt_cmap_name = self.lt_colormap_combo.value if self.lt_colormap_combo.value != "Hide" else "rainbow"
                    flim_rgb = create_FLIM_image(mean_photon_arrival_time=final_lifetime, intensity=final_intensity, colormap=cm.get_cmap(self.lt_colormap_combo.value), lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)
                    
                    if 'FLIM' in self.viewer.layers:
                        self.viewer.layers['FLIM'].data = flim_rgb # Fast data update
                    elif create_new_layers:
                        self.viewer.add_image(flim_rgb, name='FLIM', rgb=True)

                # Logic for Intensity-only image
                elif is_intensity_mode:
                    int_min, int_max = self.intensity_slider.slider.value
                    cmap_name = self.int_colormap_combo.value if self.int_colormap_combo.value != "Hide" else "gray"
                    if 'Intensity' in self.viewer.layers:
                        # VERY FAST: Only update properties, not data
                        self.viewer.layers['Intensity'].contrast_limits = (int_min, int_max)
                        self.viewer.layers['Intensity'].colormap = cmap_name
                    elif create_new_layers:
                        self.viewer.add_image(final_intensity, name='Intensity', contrast_limits=(int_min, int_max), colormap=cmap_name)
                elif is_lifetime_mode:
                    lt_min, lt_max = self.lifetime_slider.slider.value
                    cmap_name = self.lt_colormap_combo.value if self.lt_colormap_combo.value != "Hide" else "rainbow"
                    if 'Lifetime (ns)' in self.viewer.layers:
                        # Fast property updates
                        self.viewer.layers['Lifetime (ns)'].contrast_limits = (lt_min, lt_max)
                        self.viewer.layers['Lifetime (ns)'].colormap = cmap_name
                    elif create_new_layers:
                        self.viewer.add_image(final_lifetime, name='Lifetime (ns)', contrast_limits=(lt_min, lt_max), colormap=self.lt_colormap_combo.value)
            except Exception: traceback.print_exc()
        
        def update_colormaps():
            """Updates the histogram backgrounds and napari view."""
            if has_intensity:
                cmap_name = self.int_colormap_combo.value
                # Only try to get a colormap if it's not "Hide"
                if cmap_name != "Hide":
                    self.intensity_slider.set_colormap(cm.get_cmap(cmap_name))
            
            if has_lifetime:
                cmap_name = self.lt_colormap_combo.value
                if cmap_name != "Hide":
                    self.lifetime_slider.set_colormap(cm.get_cmap(cmap_name))
            
            # This call is needed to update the napari layer if a colormap changes
            update_display()

        # --- Corrected Wiring ---
        for selector in [frame_selector, sequence_selector, detector_selector]:
            selector.valueChanged.connect(update_data_slice)
        for checkbox in [sum_frames_check, sum_sequences_check, sum_detectors_check]:
            checkbox.toggled.connect(update_data_slice)
            
        self.intensity_slider.slider.changed.connect(update_display)
        if has_lifetime:
            self.lifetime_slider.slider.changed.connect(update_display)
        
        self.int_colormap_combo.changed.connect(update_colormaps)
        if has_lifetime:
            self.lt_colormap_combo.changed.connect(update_colormaps)
        
        update_data_slice() # Initial call to create layers and set data
        update_colormaps()
