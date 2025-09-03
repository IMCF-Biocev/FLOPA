# flopa/widgets/flim_view_panel.py

from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QMessageBox, QCheckBox, QGridLayout, QLabel, QSpinBox
from qtpy.QtCore import Qt, Signal, Slot
from magicgui.widgets import Container, FloatRangeSlider, ComboBox as MagicComboBox
import numpy as np
import xarray as xr
from matplotlib import cm

from flopa.io.ptuio.utils import create_FLIM_image, sum_dataset, sum_hyperstack_dict


class FlimViewPanel(QWidget):
    view_updated = Signal(np.ndarray, np.ndarray, np.ndarray, dict)
    slice_changed = Signal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.full_data_package = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.view_controls_container = QGroupBox("FLIM View Controls")
        view_layout = QVBoxLayout(self.view_controls_container)
        main_layout.addWidget(self.view_controls_container)
        main_layout.addStretch()
        self.setVisible(False)

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        instrument_params = dataset.attrs.get('instrument_params', {})
        tcspc_res_ns = float(instrument_params.get("MeasDesc_Resolution", 1/1e9)) *1e9
        tcspc_res_units = 'ch' if tcspc_res_ns == 1 else 'ns'

        self.full_data_package = {}
        standard_order = ("frame", "sequence", "detector", "line", "pixel")

        if "photon_count" in dataset.data_vars:
            self.full_data_package['intensity'] = dataset.photon_count.transpose(*standard_order, missing_dims='ignore')
        if "mean_arrival_time" in dataset.data_vars:
            self.full_data_package['lifetime'] = dataset.mean_arrival_time.transpose(*standard_order, missing_dims='ignore') * tcspc_res_ns
        if "phasor_g" in dataset.data_vars:
            self.full_data_package['phasor_g'] = dataset.phasor_g.transpose(*standard_order, missing_dims='ignore')
            self.full_data_package['phasor_s'] = dataset.phasor_s.transpose(*standard_order, missing_dims='ignore')

        if not self.full_data_package:
            self.setVisible(False)
            QMessageBox.warning(self, "No Displayable Data", "The dataset contains no viewable data.")
            return

        self.setVisible(True)
        for layer_name in ['FLIM', 'Intensity', 'Lifetime']:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        has_intensity = 'intensity' in self.full_data_package
        has_lifetime = 'lifetime' in self.full_data_package

        if has_intensity or has_lifetime:
            self._create_view_controls(has_intensity, has_lifetime, tcspc_res_units)
        else:
            self.setVisible(False)

    def _create_view_controls(self, has_intensity, has_lifetime, lifetime_units):
        source_array = self.full_data_package.get('intensity') if has_intensity else self.full_data_package.get('lifetime')
        n_frames, n_sequences, n_channels, _, _ = source_array.shape
        
        # --- Build UI ---
        final_container_widget = QWidget()
        final_layout = QVBoxLayout(final_container_widget)
        final_layout.setContentsMargins(0,0,0,0)
        
        selector_grid = QGridLayout()
        selector_grid.setColumnStretch(1, 1) 

        frame_selector = QSpinBox(); frame_selector.setRange(0, max(0, n_frames - 1))
        sequence_selector = QSpinBox(); sequence_selector.setRange(0, max(0, n_sequences - 1))
        channel_selector = QSpinBox(); channel_selector.setRange(0, max(0, n_channels - 1))

        sum_frames_check = QCheckBox("Sum"); sum_frames_check.setEnabled(n_frames > 1)
        sum_sequences_check = QCheckBox("Sum"); sum_sequences_check.setEnabled(n_sequences > 1)
        sum_channels_check = QCheckBox("Sum"); sum_channels_check.setEnabled(n_channels > 1)
        
        selector_grid.addWidget(QLabel("Frame:"), 0, 0)
        selector_grid.addWidget(frame_selector, 0, 1)
        selector_grid.addWidget(sum_frames_check, 0, 2)

        selector_grid.addWidget(QLabel("Sequence:"), 1, 0)
        selector_grid.addWidget(sequence_selector, 1, 1)
        selector_grid.addWidget(sum_sequences_check, 1, 2)

        selector_grid.addWidget(QLabel("Channel:"), 2, 0)
        selector_grid.addWidget(channel_selector, 2, 1)
        selector_grid.addWidget(sum_channels_check, 2, 2)
        
        final_layout.addLayout(selector_grid)
        
        slider_widgets = []
        if has_intensity:
            int_max = np.nanmax(self.full_data_package['intensity']); int_max = 1.0 if (np.isnan(int_max) or int_max == 0) else int_max
            int_range_slider = FloatRangeSlider(name="Intensity", min=0.0, value=(0.0, int_max / 2), max=int_max)
            slider_widgets.append(int_range_slider)

        if has_lifetime:
            lt_range_slider = FloatRangeSlider(name=f"Lifetime ({lifetime_units})", min=0.0, max=10.0, value=(0.5, 4.0))
            colormap_selector = MagicComboBox(name="Colormap", choices=["rainbow", "viridis", "plasma", "gray"], value="rainbow")
            slider_widgets.extend([lt_range_slider, colormap_selector])
        
        slider_container = Container(widgets=slider_widgets)
        final_layout.addWidget(slider_container.native)

        selectors = {
            'frame': frame_selector, 'sequence': sequence_selector, 'detector': channel_selector,
            'sum_frames': sum_frames_check, 'sum_sequences': sum_sequences_check, 'sum_channels': sum_channels_check
        }


        def update_view():
            frame_selector.setEnabled(not sum_frames_check.isChecked())
            sequence_selector.setEnabled(not sum_sequences_check.isChecked())
            channel_selector.setEnabled(not sum_channels_check.isChecked())

            dims_to_sum = []
            if selectors['sum_frames'].isChecked(): dims_to_sum.append('frame')
            if selectors['sum_sequences'].isChecked(): dims_to_sum.append('sequence')
            if selectors['sum_channels'].isChecked(): dims_to_sum.append('detector')

            full_data_package_sum = sum_hyperstack_dict(self.full_data_package,dims_to_sum)

            # final_intensity = self._get_active_slice(full_data_package_sum.get('intensity'), selectors)
            # final_lifetime = self._get_active_slice(full_data_package_sum.get('lifetime'), selectors)
            final_slice_package = {
                'intensity': self._get_active_slice(full_data_package_sum.get('intensity'), selectors),
                'lifetime': self._get_active_slice(full_data_package_sum.get('lifetime'), selectors),
                'phasor_g': self._get_active_slice(full_data_package_sum.get('phasor_g'), selectors),
                'phasor_s': self._get_active_slice(full_data_package_sum.get('phasor_s'), selectors),
            }                    

            final_intensity = final_slice_package['intensity']
            final_lifetime = final_slice_package['lifetime']

            # final_intensity = self._get_active_slice(self.full_data_package.get('intensity'), selectors)
            # final_lifetime = self._get_active_slice(self.full_data_package.get('lifetime'), selectors)
            
            # final_slice_package = {
            #     'intensity': final_intensity, 'lifetime': final_lifetime,
            #     'phasor_g': self._get_active_slice(self.full_data_package.get('phasor_g'), selectors),
            #     'phasor_s': self._get_active_slice(self.full_data_package.get('phasor_s'), selectors)
            # }


            slice_params = {'frame': frame_selector.value(), 'sequence': sequence_selector.value(), 'detector': channel_selector.value()}

            if final_intensity is not None and final_lifetime is not None:
                lt_min, lt_max = lt_range_slider.value
                int_min, int_max = int_range_slider.value
                cmap_obj = cm.get_cmap(colormap_selector.value)

                flim_rgb = create_FLIM_image(mean_photon_arrival_time=final_lifetime, intensity=final_intensity, colormap=cmap_obj, lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)

                if 'FLIM' in self.viewer.layers: self.viewer.layers['FLIM'].data = flim_rgb
                else: self.viewer.add_image(flim_rgb, name='FLIM', rgb=True)
                self.view_updated.emit(flim_rgb, final_intensity, final_lifetime, slice_params)

            elif final_intensity is not None:
                int_min, int_max = int_range_slider.value

                if 'Intensity' in self.viewer.layers: self.viewer.layers['Intensity'].data = final_intensity; self.viewer.layers['Intensity'].contrast_limits = (int_min, int_max)
                else: self.viewer.add_image(final_intensity, name='Intensity', contrast_limits=(int_min, int_max))

            elif final_lifetime is not None:
                lt_min, lt_max = lt_range_slider.value
                cmap_name = colormap_selector.value
                layer_name = 'Lifetime'
                
                if layer_name in self.viewer.layers: self.viewer.layers[layer_name].data = final_lifetime; self.viewer.layers[layer_name].contrast_limits = (lt_min, lt_max); self.viewer.layers[layer_name].colormap = cmap_name
                else: self.viewer.add_image(final_lifetime, name=layer_name, contrast_limits=(lt_min, lt_max), colormap=cmap_name)

            self.slice_changed.emit(final_slice_package)

        # --- Connect all controls to the update function ---
        for widget in slider_widgets:
            widget.changed.connect(update_view)

        frame_selector.valueChanged.connect(update_view)
        sequence_selector.valueChanged.connect(update_view)
        channel_selector.valueChanged.connect(update_view)
        sum_frames_check.toggled.connect(update_view)
        sum_sequences_check.toggled.connect(update_view)
        sum_channels_check.toggled.connect(update_view)
        
        # --- Finalize the layout ---
        layout = self.view_controls_container.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        layout.addWidget(final_container_widget)
        update_view()


    def _get_active_slice(self, full_xarray_dataarray, selectors):
        if full_xarray_dataarray is None: return None

        selection_dict = {}
        if not selectors['sum_frames'].isChecked():
            selection_dict['frame'] = selectors['frame'].value()
        if not selectors['sum_sequences'].isChecked():
            selection_dict['sequence'] = selectors['sequence'].value()
        if not selectors['sum_channels'].isChecked():
            selection_dict['detector'] = selectors['detector'].value()
            
        dims_to_sum = []
        if selectors['sum_frames'].isChecked(): dims_to_sum.append('frame')
        if selectors['sum_sequences'].isChecked(): dims_to_sum.append('sequence')
        if selectors['sum_channels'].isChecked(): dims_to_sum.append('detector')
            
        active_slice = full_xarray_dataarray.isel(**selection_dict)
        
        # if dims_to_sum: # later use fun from Dalibor
        #     if full_xarray_dataarray.name in ['mean_arrival_time', 'phasor_g', 'phasor_s']:
        #         final_slice = active_slice.mean(dim=dims_to_sum) # to be weighted mean
        #     else: # For photon_count
        #         final_slice = active_slice.sum(dim=dims_to_sum)
        # else:
        #     final_slice = active_slice

        # if dims_to_sum:
        #     final_slice = sum_dataset(active_slice,dims_to_sum)
        # else:
        #     final_slice = active_slice
        final_slice = active_slice

        return final_slice.values

