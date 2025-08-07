from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QComboBox, QTextEdit, QCheckBox,
    QProgressBar, QHBoxLayout, QApplication
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from magicgui.widgets import Container, SpinBox, FloatRangeSlider, ComboBox as MagicComboBox
import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib import cm

# Import our new backend functions
from flopa.io.ptuio.reader import TTTRReader
from flopa.io.ptuio.reconstructor import ScanConfig
from flopa.processing.reconstruction import reconstruct_ptu_to_dataset
from flopa.processing.flim_image import create_flim_rgb_image


class PtuProcessingPanel(QWidget):
    def __init__(self, viewer, on_data_loaded, on_flim_image_generated):
        super().__init__()
        self.viewer = viewer
        self.on_data_loaded = on_data_loaded
        self.on_flim_image_generated = on_flim_image_generated

        # Internal state
        self.ptu_filepath = None
        self.header_tags = {}
        self.reconstructed_dataset = None
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- 1. File Selection ---
        file_group = QGroupBox("1. Load PTU File")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file selected.")
        file_layout.addWidget(self.file_label)

        # Create a horizontal layout for the two buttons
        button_layout = QHBoxLayout()

        self.select_file_btn = QPushButton("Select PTU File...")
        self.select_file_btn.clicked.connect(self._select_ptu_file)
        button_layout.addWidget(self.select_file_btn)

        # Option B: Load an existing H5 file
        self.select_h5_btn = QPushButton("Load H5...")
        self.select_h5_btn.setToolTip("Load a previously reconstructed .h5 file.")
        self.select_h5_btn.clicked.connect(self._select_h5_file)
        button_layout.addWidget(self.select_h5_btn)
        
        file_layout.addLayout(button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- 2. Header Info & Configuration ---
        self.config_group = QGroupBox("2. Scan Configuration")
        config_form_layout = QFormLayout()

        # Configurable fields
        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 10000)
        self.pixels_spin = QSpinBox()
        self.pixels_spin.setRange(1, 10000)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 1000)
        self.frames_spin.setValue(1)
        self.accu_spin = QSpinBox()
        self.accu_spin.setRange(1, 100)
        self.accu_spin.setValue(1)
        self.bidir_check = QCheckBox("Bidirectional Scan")

        config_form_layout.addRow("Lines:", self.lines_spin)
        config_form_layout.addRow("Pixels per Line:", self.pixels_spin)
        config_form_layout.addRow("Frames:", self.frames_spin)
        config_form_layout.addRow("Line Accumulations:", self.accu_spin)
        config_form_layout.addRow(self.bidir_check)
        
        self.header_info = QTextEdit()
        self.header_info.setReadOnly(True)
        self.header_info.setFont(QFont("Courier", 8))
        self.header_info.setFixedHeight(100)
        config_form_layout.addRow("Header Info:", self.header_info)
        
        self.config_group.setLayout(config_form_layout)
        layout.addWidget(self.config_group)
        self.config_group.setVisible(False) # Hide until file is loaded

        # --- 3. Reconstruction ---
        self.recon_group = QGroupBox("3. Reconstruct and Load")
        recon_layout = QVBoxLayout()
        self.reconstruct_btn = QPushButton("Reconstruct Image")
        self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        recon_layout.addWidget(self.reconstruct_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        recon_layout.addWidget(self.progress_bar)
        self.recon_group.setLayout(recon_layout)
        layout.addWidget(self.recon_group)
        self.recon_group.setVisible(False)

        # recon_group = QGroupBox("3. Reconstruct and Load")
        # recon_layout = QVBoxLayout()

        # self.reconstruct_btn = QPushButton("Reconstruct Image")
        # self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        # recon_layout.addWidget(self.reconstruct_btn)
        
        # self.progress_bar = QProgressBar()
        # self.progress_bar.setVisible(False)
        # recon_layout.addWidget(self.progress_bar)

        # recon_group.setLayout(recon_layout)
        # layout.addWidget(recon_group)
        # self.recon_group.setVisible(False)

        # self.reconstruct_btn.setEnabled(False)

        # --- 4. FLIM View Controls ---
        self.view_controls_container = QWidget()
        self.view_controls_container.setVisible(False) # Hide until data is ready
        layout.addWidget(self.view_controls_container)

        layout.addStretch()

    def _select_ptu_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(self, "Select PTU File", "", "PicoQuant Files (*.ptu)")
        if not filepath_str:
            return

        self.ptu_filepath = Path(filepath_str)
        self.file_label.setText(f"Selected: {self.ptu_filepath.name}")

        try:
            reader = TTTRReader(str(self.ptu_filepath))
            self.header_tags = reader.header.tags
            
            # Display some key header info
            header_text = "\n".join(f"{k}: {v}" for k, v in self.header_tags.items() if "ImgHdr" in k or "SyncRate" in k or "Resolution" in k)
            self.header_info.setText(header_text)
            
            # Pre-populate config fields from header if possible
            self.lines_spin.setValue(self.header_tags.get("ImgHdr_PixY", 512))
            self.pixels_spin.setValue(self.header_tags.get("ImgHdr_PixX", 512))
            # Other fields might not be in standard headers, so we use defaults.

            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self.reconstruct_btn.setEnabled(True)
            self.view_controls_container.setVisible(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read PTU header:\n{e}")
            self.file_label.setText("Failed to load file.")

    def _select_h5_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(self, "Select Reconstructed H5 File", "", "HDF5 Files (*.h5 *.hdf5)")
        if not filepath_str: return
        
        h5_filepath = Path(filepath_str)
        self.file_label.setText(f"H5 Loaded: {h5_filepath.name}")

        try:
            loaded_dataset = xr.open_dataset(h5_filepath)
            
            # Hide the PTU-specific widgets
            self.config_group.setVisible(False)
            self.recon_group.setVisible(False)

            # Directly load the data for viewing
            self._load_data_from_dataset(loaded_dataset)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load HDF5 file:\n{e}")

    def _update_progress(self, current, total):
        self.progress_bar.setValue(int(100 * current / total))
        QApplication.processEvents() # Force UI update

    def _run_reconstruction(self):
        if not self.ptu_filepath:
            return

        self.reconstruct_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 1. Create ScanConfig from UI
        scan_config = ScanConfig(
            lines=self.lines_spin.value(),
            pixels=self.pixels_spin.value(),
            frames=self.frames_spin.value(),
            bidirectional=self.bidir_check.isChecked(),
            line_accumulations=(self.accu_spin.value(),)
        )

        try:
            # 2. Run the backend processing function
            self.reconstructed_data = reconstruct_ptu_to_dataset(
                self.ptu_filepath,
                scan_config,
                progress_callback=self._update_progress
            )
            
            # 3. Ask user where to save the H5 file
            default_h5_path = self.ptu_filepath.with_suffix('.h5')
            save_path_str, _ = QFileDialog.getSaveFileName(
                self, "Save Reconstructed Data", str(default_h5_path), "HDF5 Files (*.h5)"
            )
            
            if save_path_str:
                self.reconstructed_data.to_netcdf(save_path_str)
                QMessageBox.information(self, "Success", f"Reconstruction complete.\nData saved to {save_path_str}")
            
            # 4. Load data into the widget for viewing
            self._load_data_from_dataset(self.reconstructed_data)



        except Exception as e:
            QMessageBox.critical(self, "Reconstruction Error", f"An error occurred:\n{e}")
        finally:
            self.reconstruct_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    # def _load_data_from_dataset(self):
    #     """Called after reconstruction to load data and set up viewing controls."""
    #     if self.reconstructed_dataset is None:
    #         return

    #     # Get the resolution from the dataset attributes for unit conversion
    #     tcspc_resolution_str = self.reconstructed_dataset.attrs.get("MeasDesc_Resolution", "5e-12")
    #     tcspc_resolution_s = float(tcspc_resolution_str)

    #     # Prepare data for the rest of the application
    #     self.data_intensity = np.array(self.reconstructed_dataset.photon_count)
    #     self.data_lifetime = np.array(self.reconstructed_dataset.mean_photon_arrival_time) * tcspc_resolution_s * 1e9 # to ns

    #     # Prepare data for the rest of the application
    #     # intensity = np.array(self.reconstructed_dataset.photon_count.transpose("frame", "sequence", "channel", "line", "pixel"), dtype=int)
        
    #     # # Convert lifetime to ns
    #     # tcspc_resolution_s = self.header_tags.get("MeasDesc_Resolution", 5e-12)
    #     # lifetime = np.array(self.reconstructed_dataset.mean_photon_arrival_time.transpose("frame", "sequence", "channel", "line", "pixel"), dtype=float) * tcspc_resolution_s * 1e9

    #     # self.data_intensity = intensity
    #     # self.data_lifetime = lifetime

    #     # Signal to main widget that data is ready
    #     self.on_data_loaded({"intensity": self.data_intensity, "lifetime": self.data_lifetime})
        
    #     # Create the viewer controls
    #     self._create_flim_view_controls()
    #     self.view_controls_container.setVisible(True)
    def _load_data_from_dataset(self, dataset: xr.Dataset):
        """Prepares data from a dataset and sets up viewing controls."""
        if dataset is None: return

        self.reconstructed_dataset = dataset

        try:
            tcspc_resolution_s = float(dataset.attrs.get("MeasDesc_Resolution", "5e-12"))
        except (ValueError, TypeError):
            tcspc_resolution_s = 5e-12
            print("Warning: Could not parse 'MeasDesc_Resolution' from H5 attrs, using default 5ps.")

        # --- ENFORCE DIMENSION ORDER HERE ---
        # Define our standard internal order
        standard_order = ("frame", "sequence", "channel", "line", "pixel")

        # Transpose the DataArrays to match our standard order
        intensity_da = dataset.photon_count.transpose(*standard_order, missing_dims='ignore')
        lifetime_da = dataset.mean_photon_arrival_time.transpose(*standard_order, missing_dims='ignore')
        
        # Now convert to numpy arrays
        self.data_intensity = np.array(intensity_da)
        self.data_lifetime = np.array(lifetime_da) * tcspc_resolution_s * 1e9 # to ns
        
        print(f"Data loaded with shape (frame, sequence, channel, line, pixel): {self.data_intensity.shape}")
        
        # The rest of the function remains the same
        self.on_data_loaded({"intensity": self.data_intensity, "lifetime": self.data_lifetime})
        self._create_flim_view_controls()
        self.view_controls_container.setVisible(True)



    def _create_flim_view_controls(self):
        if self.data_intensity is None or self.data_lifetime is None:
            return
        
        n_frames, n_sequences, n_channels, _, _ = self.data_intensity.shape
        int_max = np.nanmax(self.data_intensity)

        frame_selector = SpinBox(name="Frame", min=0, max=n_frames - 1, step=1)
        sequence_selector = SpinBox(name="Sequence", min=0, max=n_sequences - 1, step=1)
        channel_selector = SpinBox(name="Channel", min=0, max=n_channels - 1, step=1)
        lt_range_slider = FloatRangeSlider(name="Lifetime (ns)", min=0.0, max=10.0, value=(0.5, 4.0), step=0.1)
        int_range_slider = FloatRangeSlider(name="Intensity", min=0.0, value=(0.0, int_max / 2 if int_max > 0 else 1.0), max=int_max if int_max > 0 else 1.0, step=1)

        COLORMAP_OPTIONS = ["rainbow", "viridis", "plasma", "inferno", "magma", "cividis", "jet", "gray"]
        colormap_selector = MagicComboBox(name="Lifetime Colormap", choices=COLORMAP_OPTIONS, value="rainbow")

        def update_flim_gui():
            frame = frame_selector.value
            sequence = sequence_selector.value
            channel = channel_selector.value
            lt_min, lt_max = lt_range_slider.value
            int_min, int_max = int_range_slider.value
            cmap_obj = cm.get_cmap(colormap_selector.value)

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

        # Connect signals to update function
        for widget in [frame_selector, sequence_selector, channel_selector, lt_range_slider, int_range_slider, colormap_selector]:
            widget.changed.connect(update_flim_gui)

        control_panel = Container(widgets=[
            frame_selector, sequence_selector, channel_selector,
            lt_range_slider, int_range_slider, colormap_selector
        ])
        
        # Embed the magicgui container into our Qt layout
        # Clear previous controls if any
        if self.view_controls_container.layout() is not None:
             # Simple way to clear existing widgets
             while self.view_controls_container.layout().count():
                 child = self.view_controls_container.layout().takeAt(0)
                 if child.widget():
                     child.widget().deleteLater()
        
        view_layout = QVBoxLayout()
        view_layout.addWidget(control_panel.native)
        self.view_controls_container.setLayout(view_layout)
        
        # Trigger initial update
        update_flim_gui()


