# flopa/widgets/napari_flim_widget.py

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from qtpy.QtCore import Slot
import xarray as xr
import numpy as np

from flopa.widgets.ptu_processing_panel import PtuProcessingPanel
from flopa.widgets.phasor_panel import PhasorPanel
# from flopa.widgets.decay_panel import DecayPanel


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
        # self.tabs.addTab(self.decay_tab, "Decay")
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.decay_tab), False)

        # --- Connect signals ---
        self.processing_tab.reconstruction_finished.connect(self._on_reconstruction_finished)
        self.view_tab.slice_changed.connect(self.phasor_tab.on_slice_changed)


    @Slot(xr.Dataset)
    def _on_reconstruction_finished(self, dataset: xr.Dataset):
        """
        This is the central handler slot for data.
        It receives the complete reconstructed dataset and distributes it
        to all interested consumer panels.
        """        
        # 1. Store the new dataset as the central state
        self.reconstructed_dataset = dataset

        # 2. Distribute the complete dataset to all consumer panels.
        self.view_tab.update_data(dataset)
        self.phasor_tab.update_data(dataset)
        # self.decay_tab.update_data(dataset)

        # 3. Enable/disable tabs based on the dataset's content
        data_vars = dataset.data_vars
        
        has_phasor_data = "phasor_g" in data_vars  
        self.tabs.setTabEnabled(self.tabs.indexOf(self.phasor_tab), has_phasor_data)

        # has_decay_data = "tcspc_histogram" in data_vars
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.decay_tab), has_decay_data)


    @Slot(np.ndarray, np.ndarray, np.ndarray, dict)
    def _on_view_updated(self, flim_image, intensity_slice, lifetime_slice, slice_params):
        """
        Slot that is called whenever the user interacts with the FLIM view controls.
        """
        pass