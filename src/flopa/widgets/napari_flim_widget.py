# flopa/widgets/napari_flim_widget.py

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from qtpy.QtCore import Slot
import xarray as xr
import numpy as np

from flopa.widgets.ptu_processing_panel import PtuProcessingPanel
from flopa.widgets.phasor_panel import PhasorPanel
from flopa.widgets.decay_panel import DecayPanel


class FlimWidget(QWidget):
    def __init__(self, viewer, flim_view_panel):
        super().__init__()
        self.viewer = viewer
        self.reconstructed_dataset = None
        self.view_tab = flim_view_panel
        
        # --- UI Setup ---
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Add All Panels to the Tab Widget ---
        self.processing_tab = PtuProcessingPanel(viewer)
        self.tabs.addTab(self.processing_tab, "Process PTU")

        self.phasor_tab = PhasorPanel(viewer)
        self.tabs.addTab(self.phasor_tab, "Phasor")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.phasor_tab), False)

        # self.decay_tab = DecayPanel(viewer)
        self.decay_tab = DecayPanel(viewer, flim_view_panel=self.view_tab)
        self.tabs.addTab(self.decay_tab, "Decay")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.decay_tab), False)

        # --- Connect signals ---
        self.processing_tab.reconstruction_finished.connect(self._on_reconstruction_finished)
        self.view_tab.view_changed.connect(self.phasor_tab.on_view_changed)
        self.decay_tab.decay_shift_changed.connect(self._on_decay_shift_changed)

    @Slot(xr.Dataset)
    def _on_reconstruction_finished(self, dataset: xr.Dataset):
        """
        This is the central handler slot for data.
        It receives the complete reconstructed dataset and distributes it
        to all interested consumer panels.
        """        
        source_name = dataset.attrs.get('source_filename', '')
        is_from_h5 = not source_name or source_name.endswith('.h5')
        self.reconstructed_dataset = dataset

        self.view_tab.update_data(dataset, is_from_h5=is_from_h5)
        self.phasor_tab.update_data(dataset)
        self.decay_tab.update_data(dataset)

        data_vars = dataset.data_vars
        
        has_phasor_data = "phasor_g" in data_vars  
        self.tabs.setTabEnabled(self.tabs.indexOf(self.phasor_tab), has_phasor_data)

        has_decay_data = "tcspc_histogram" in data_vars
        self.tabs.setTabEnabled(self.tabs.indexOf(self.decay_tab), has_decay_data)


    @Slot(int)
    def _on_decay_shift_changed(self, shift_value: int):
        """
        Called when the user changes the shift value in the DecayPanel.
        This method updates the central state and enables/disables the
        phasor panel's calibration option.
        """        
        # Enable the phasor's shift calibration option only if the shift is non-zero
        is_shift_active = (shift_value != 0)
        self.phasor_tab.cal_by_shift_radio.setEnabled(is_shift_active)
        
        if not is_shift_active and self.phasor_tab.cal_by_shift_radio.isChecked():
            self.phasor_tab.cal_by_factor_radio.setChecked(True)

        if self.reconstructed_dataset is not None:
            if 'instrument_params' not in self.reconstructed_dataset.attrs:
                self.reconstructed_dataset.attrs['instrument_params'] = {}
            self.reconstructed_dataset.attrs['instrument_params']['current_decay_shift'] = shift_value