from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSpinBox,
    QGroupBox, QFormLayout, QLineEdit, QMessageBox, QComboBox
)

from qtpy.QtCore import Signal, Slot

from magicgui.widgets import Container, SpinBox, FloatRangeSlider, ComboBox

# Let's assume flim_rgb.py exposes generate_flim_image(data, frame, channel) -> image
from flopa.processing.flim_image import create_flim_rgb_image
from flopa.io.loader import read_ptu_file

import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib import cm

class LoadViewPanel(QWidget):
    def __init__(self, viewer, on_data_loaded, on_flim_image_generated):
        super().__init__()
        self.viewer = viewer
        self.on_data_loaded = on_data_loaded
        self.on_flim_image_generated = on_flim_image_generated

        self.data = None
        self.data_intensity = None
        self.data_lifetime = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        self.file_label = QLabel("No file selected.")
        layout.addWidget(self.file_label)

        self.select_file_btn = QPushButton("Select file")
        self.select_file_btn.clicked.connect(self._select_file)
        layout.addWidget(self.select_file_btn)

        self.metadata_box = QGroupBox("Metadata Correction")
        self.metadata_layout = QFormLayout()
        self.mock_metadata_field = QLineEdit("mock value")
        self.metadata_layout.addRow("Mock editable field:", self.mock_metadata_field)
        self.save_metadata_btn = QPushButton("Save corrected metadata")
        self.metadata_layout.addWidget(self.save_metadata_btn)
        self.metadata_box.setLayout(self.metadata_layout)
        layout.addWidget(self.metadata_box)

        self.generate_btn = QPushButton("Generate FLIM image")
        self.generate_btn.setEnabled(False)
        #self.generate_btn.clicked.connect(self._load_flim_image)
        self.generate_btn.clicked.connect(self._create_flim_view_controls)

        layout.addWidget(self.generate_btn)

        self.setLayout(layout)

    def _select_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(self, "Select HDF5 Result File", "", "HDF5 Files (*.h5 *.hdf5)")
        if not filepath_str:
            return

        filepath = Path(filepath_str)
        self.file_label.setText(f"Selected: {filepath.name}")

        try:
            result = xr.open_dataset(filepath)
            # Store the full dataset
            self.data_intensity = np.array(result.photon_count.transpose("frame", "sequence", "channel", "line", "pixel"), dtype=int)
            self.data_lifetime = np.array(result.mean_photon_arrival_time.transpose("frame", "sequence", "channel", "line", "pixel"), dtype=float) * 5e-3

            self.generate_btn.setEnabled(True)
            # This callback is fine, it signals the main widget that data is available
            self.on_data_loaded({"intensity": self.data_intensity, "lifetime": self.data_lifetime})
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load HDF5 file:\n{e}")
            self.file_label.setText("Failed to load file.")


    def _create_flim_view_controls(self):
        if self.data_intensity is None or self.data_lifetime is None:
            return
        
        n_frames, n_sequences, n_channels, _, _ = self.data_intensity.shape
        int_max = np.nanmax(self.data_intensity)


        frame_selector = SpinBox(name="Frame", min=0, max=n_frames - 1, step=1)
        sequence_selector = SpinBox(name="Sequence", min=0, max=n_sequences - 1, step=1)
        channel_selector = SpinBox(name="Channel", min=0, max=n_channels - 1, step=1)
        lt_range_slider = FloatRangeSlider(name="Lifetime", min=0.0, max=25.0, value=(0.5, 10.0), step=0.1)
        int_range_slider = FloatRangeSlider(name="Intensity", min=0.0, value=(0.0, int_max/2), max=int_max, step=1)

        COLORMAP_OPTIONS = ["viridis", "plasma", "inferno", "magma", "cividis", "rainbow", "jet", "gray"]
        colormap_selector = ComboBox(name="Lifetime Colormap", choices=COLORMAP_OPTIONS, value="rainbow")

        def update_flimgui():
            frame = frame_selector.value
            sequence = sequence_selector.value
            channel = channel_selector.value
            lt_min, lt_max = lt_range_slider.value
            int_min, int_max = int_range_slider.value

            cmap_name = colormap_selector.value
            try:
                cmap_obj = cm.get_cmap(cmap_name)
            except ValueError:
                print(f"Warning: Colormap '{cmap_name}' not found, defaulting to 'jet'.")
                cmap_obj = cm.rainbow

            lt_img = self.data_lifetime[frame, sequence, channel, :, :]
            int_img = self.data_intensity[frame, sequence, channel, :, :]

            flim_rgb = create_flim_rgb_image(
                mean_photon_arrival_time=lt_img,
                intensity=int_img,
                colormap=cmap_obj,
                lt_min=lt_min,
                lt_max=lt_max,
                int_min=int_min,
                int_max=int_max,
            )

            if 'FLIM' in self.viewer.layers:
                self.viewer.layers['FLIM'].data = flim_rgb
            else:
                self.viewer.add_image(flim_rgb, name='FLIM', rgb=True)

            self.on_flim_image_generated(flim_rgb, int_img, lt_img)

        # Connect signals
        for widget in [frame_selector, sequence_selector, channel_selector, lt_range_slider, int_range_slider, colormap_selector]:
            widget.changed.connect(update_flimgui)



        control_panel = Container(
            widgets=[
                frame_selector,
                sequence_selector,
                channel_selector,
                lt_range_slider,
                int_range_slider,
                colormap_selector
            ]
        )

        self.viewer.window.add_dock_widget(control_panel, area='right')
        update_flimgui()
