# flopa/widgets/ptu_processing_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QPlainTextEdit, QHBoxLayout, QApplication,
    QCheckBox, QDoubleSpinBox, QTextEdit, QDialog, QGridLayout, QScrollArea,
    QRadioButton, QButtonGroup, QSizePolicy, QComboBox
)
from qtpy.QtCore import Qt, QSize, Signal, Slot, QThreadPool
from qtpy.QtGui import QFont, QIcon
from pathlib import Path
import traceback
import xarray as xr
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json

from flopa.processing.reconstruction import reconstruct_ptu_to_dataset
from flopa.processing.logger import ProgressLogger
from flopa.io.loader import read_ptu_file, get_markers, format_ptu_header, load_h5_dataset
from flopa.io.ptuio.reconstructor import ScanConfig
from flopa.io.ptuio.utils import estimate_bidirectional_shift
from flopa.widgets.utils.bidir_shift_plot import plot_bidirectional_shift
from .utils.threading import Worker, ThreadSafeLogger
from .utils.style import apply_style, GROUP_BOX_STYLE_A



class PtuProcessingPanel(QWidget):
    """
    A widget for loading a PTU file, configuring scan parameters,
    and launching the reconstruction process.
    """

    reconstruction_finished = Signal(xr.Dataset)
    # h5_file_selected = Signal(xr.Dataset) 

    def __init__(self, viewer):
        super().__init__()
        self.threadpool = QThreadPool()

        self.viewer = viewer
        self.ptu_data = None  
        self.ptu_filepath = None
        self.shift_plot_data = None 
        self._current_ptu_header = None

        self._init_ui()
        self.setStyleSheet(GROUP_BOX_STYLE_A)


    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        file_group = QGroupBox("Load Data")
        apply_style(file_group, GROUP_BOX_STYLE_A)
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected.")
        button_layout = QHBoxLayout()
        self.select_ptu_btn = QPushButton("Read PTU File...")
        self.select_ptu_btn.clicked.connect(self._select_ptu_file)
        button_layout.addWidget(self.select_ptu_btn)
        self.select_h5_btn = QPushButton("Load H5...")
        self.select_h5_btn.setEnabled(True) 
        self.select_h5_btn.setToolTip("Load a previously exported FLOPA HDF5 dataset.")
        self.select_h5_btn.clicked.connect(self._on_load_h5) 
        button_layout.addWidget(self.select_h5_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addLayout(button_layout)
        main_layout.addWidget(file_group)


        self.ptu_controls_container = QWidget()
        ptu_layout = QVBoxLayout(self.ptu_controls_container)
        ptu_layout.setContentsMargins(2, 2, 2, 2) 


        self.header_group = QGroupBox("Header Info")
        apply_style(self.header_group, GROUP_BOX_STYLE_A)
        header_layout = QVBoxLayout(self.header_group)
        self.header_info = QTextEdit()
        self.header_info.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E; /* A dark grey, close to black */
            }
        """)
        self.header_info.setReadOnly(True)
        self.header_info.setFont(QFont("Courier", 8))
        header_layout.addWidget(self.header_info)
        marker_layout = QHBoxLayout()
        self.markers_button = QPushButton("Markers")
        self.markers_button.clicked.connect(self._show_markers)
        marker_layout.addWidget(self.markers_button)
        self.markers_output = QLabel("")
        self.markers_output.setFont(QFont("Courier", 8))
        marker_layout.addWidget(self.markers_output)
        header_layout.addLayout(marker_layout)
        ptu_layout.addWidget(self.header_group)
        self.header_group.setVisible(False)


        config_group = self._create_config_group()
        ptu_layout.addWidget(config_group)
        self.config_group = config_group
        self.config_group.setVisible(False)


        self.recon_group = QGroupBox("Reconstruction")
        apply_style(self.recon_group, GROUP_BOX_STYLE_A)
        recon_layout = QFormLayout(self.recon_group)
        self.output_combo = QComboBox()
        self.output_combo.addItem("Intensity", "photon_count")
        self.output_combo.addItem("Mean Lifetime", "mean_arrival_time")
        self.output_combo.addItem("Phasor & Decay", "all")
        recon_layout.addRow("Select Output Type:", self.output_combo)
        self.log_text_edit = QPlainTextEdit(); self.log_text_edit.setReadOnly(True); self.log_text_edit.setMinimumHeight(100)
        self.log_text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1E1E1E; /* A dark grey, close to black */
            }
        """)
        recon_layout.addRow("Reconstruction Log:", self.log_text_edit)
        self.reconstruct_btn = QPushButton("Reconstruct Image"); self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        recon_layout.addRow(self.reconstruct_btn)
        ptu_layout.addWidget(self.recon_group)
        self.recon_group.setVisible(False)
        
        ptu_layout.addStretch()
        main_layout.addWidget(self.ptu_controls_container)

        self.h5_metadata_group = QGroupBox("H5 Dataset Metadata")
        h5_layout = QVBoxLayout(self.h5_metadata_group)
        
        self.h5_metadata_display = QTextEdit()
        self.h5_metadata_display.setReadOnly(True)
        self.h5_metadata_display.setStyleSheet("font-family: Consolas, Courier New;")
        h5_layout.addWidget(self.h5_metadata_display)
        
        main_layout.addWidget(self.h5_metadata_group)


        main_layout.addStretch()

        self._show_ptu_view()

    def _show_ptu_view(self):
        """Shows the PTU controls and hides the H5 metadata display."""
        self.ptu_controls_container.setVisible(True) 
        self.h5_metadata_group.setVisible(False)
        if self._current_ptu_header:
            self._display_ptu_header(self._current_ptu_header)

    def _format_dict_for_display(self, data_dict: dict) -> str:
        """Formats a dictionary into a clean, indented, key-value string."""
        if not isinstance(data_dict, dict):
            return str(data_dict) 
        
        lines = []
        for key, value in data_dict.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _show_h5_view(self, dataset):
        """Shows the H5 metadata display and hides the PTU controls."""
        self.ptu_controls_container.setVisible(False)
        self.h5_metadata_group.setVisible(True)
        
        # Populate the metadata display with the new, cleaner format
        source_name = dataset.attrs.get('source_filename', 'N/A')
        metadata_text = f"--- Metadata from {source_name} ---\n\n"

        if 'scan_config' in dataset.attrs:
            metadata_text += "scan_config:\n"
            metadata_text += self._format_dict_for_display(dataset.attrs['scan_config']) + "\n"
        
        if 'instrument_params' in dataset.attrs:
            metadata_text += "\ninstrument_params:\n"
            metadata_text += self._format_dict_for_display(dataset.attrs['instrument_params']) + "\n"
        
        self.h5_metadata_display.setText(metadata_text)

    def _create_config_group(self) -> QGroupBox:
        group  = QGroupBox("Scan Configuration")
        # apply_style(group , GROUP_BOX_STYLE_A) # Your style is applied here
        main_layout = QVBoxLayout(group)

        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(0, 4)
        grid_layout.setColumnStretch(1, 5)

        # Left Column
        left_group = QGroupBox() 
        left_layout = QFormLayout(left_group)
        self.lines_spin = QSpinBox(); self.lines_spin.setRange(1, 10000); left_layout.addRow("Lines:", self.lines_spin)
        self.pixels_spin = QSpinBox(); self.pixels_spin.setRange(1, 10000); left_layout.addRow("Pixels per Line:", self.pixels_spin)
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(1, 1000); self.frames_spin.setValue(1); left_layout.addRow("Frames:", self.frames_spin)
        self.tcspc_bins_spin = QSpinBox(); self.tcspc_bins_spin.setRange(1, 65536); left_layout.addRow("TCSPC Bins:", self.tcspc_bins_spin)
        self.max_detector_spin = QSpinBox(); self.max_detector_spin.setRange(1, 128); self.max_detector_spin.setValue(2); left_layout.addRow("Max Detector:", self.max_detector_spin)
        grid_layout.addWidget(left_group, 0, 0)

        # Right Column
        right_group = QGroupBox()
        # right_group.setFlat(True)
        right_layout = QFormLayout(right_group)
        self.sequences_spin = QSpinBox(); self.sequences_spin.setRange(1, 16); self.sequences_spin.setValue(1); right_layout.addRow("n Sequences:", self.sequences_spin)
        self.sequences_spin.setValue(1)
        self.accu_scroll_area = QScrollArea(); self.accu_scroll_area.setWidgetResizable(True); self.accu_scroll_area.setFixedHeight(80)
        self.accu_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.accu_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.accu_container = QWidget(); self.accu_container_layout = QVBoxLayout(self.accu_container); self.accu_container_layout.setContentsMargins(5, 5, 5, 5); self.accu_container_layout.setSpacing(5)
        self.accu_scroll_area.setWidget(self.accu_container); self.accu_spinboxes = []
        accu_hbox = QHBoxLayout()
        accu_hbox.setContentsMargins(0, 0, 0, 0) 
        
        accu_hbox.addWidget(self.accu_scroll_area)
        
        accu_hbox.addStretch(1) 
        right_layout.addRow("Accumulations:", accu_hbox)
        grid_layout.addWidget(right_group, 0, 1)
        
        main_layout.addLayout(grid_layout)

        self.bidir_group = QGroupBox("Bidirectional Scan")
        self.bidir_group.setObjectName("plain") 

        self.bidir_group.setCheckable(True) 
        self.bidir_group.setChecked(False) 
        
        bidir_layout = QFormLayout(self.bidir_group)

        # Phase shift controls
        self.bidir_phase_spinbox = QDoubleSpinBox()
        self.bidir_phase_spinbox.setRange(-0.2, 0.2); self.bidir_phase_spinbox.setSingleStep(0.0001); self.bidir_phase_spinbox.setDecimals(5)
    
        # Estimate/Plot buttons
        self.btn_estimate_shift = QPushButton("Estimate"); self.btn_estimate_shift.clicked.connect(self._on_estimate_shift_clicked)
        self.btn_plot_shift = QPushButton(); icon_path = "./assets/icons/plot_icon.png" 
        if Path(icon_path).is_file(): self.btn_plot_shift.setIcon(QIcon(icon_path)); self.btn_plot_shift.setIconSize(QSize(16, 16))
        else: self.btn_plot_shift.setText("Plot")
        self.btn_plot_shift.setToolTip("Plot shift correlation curve"); self.btn_plot_shift.clicked.connect(self._on_plot_shift_clicked); self.btn_plot_shift.setEnabled(False)
        

        button_hbox = QHBoxLayout()
        self.shift_text = QLabel("Phase Shift: ")
        button_hbox.addWidget(self.shift_text)
        button_hbox.addWidget(self.bidir_phase_spinbox)
        button_hbox.addSpacing(25)
        button_hbox.addWidget(self.btn_estimate_shift)
        button_hbox.addWidget(self.btn_plot_shift)
        button_hbox.addStretch()
        
        bidir_layout.addRow(button_hbox)
        
        main_layout.addWidget(self.bidir_group)
        
        self.sequences_spin.valueChanged.connect(self._update_accumulation_widgets)
        self._update_accumulation_widgets()
        return group 

    def _display_ptu_header(self, header_dict):
        header_text = "--- PTU Header ---\n"
        for key, value in header_dict.items():
            header_text += f"{key}: {value}\n"
        self.metadata_display.setText(header_text)

    @Slot()
    def _on_load_h5(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load FLOPA HDF5 Dataset", "", "HDF5 Files (*.h5)")
        if not filepath: return

        try:
            dataset = load_h5_dataset(Path(filepath))
            dataset.attrs['source_filename'] = Path(filepath).name
            
            self._show_h5_view(dataset)
            
            self.reconstruction_finished.emit(dataset)
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "H5 Load Error", f"Failed to load HDF5 file:\n{e}")
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "H5 Load Error", f"Failed to load HDF5 file:\n{e}")

    def _select_ptu_file(self, *args, **kwargs):
        filepath_str, _ = QFileDialog.getOpenFileName(self, "Select PTU File", "", "PicoQuant Files (*.ptu)")
        if not filepath_str: return
        self.ptu_filepath = Path(filepath_str)
        self.file_label.setText(f"PTU Selected: {self.ptu_filepath.name}")
        try:
            self.ptu_data = read_ptu_file(str(self.ptu_filepath))
            header_tags = self.ptu_data["header"]
            constants = self.ptu_data["constants"]
            header_text = format_ptu_header(header_tags, constants, full_header=True)            
            self.header_info.setText(header_text)
            self.tcspc_bins_spin.setValue(constants.get('tcspc_bins', 4096))
            self.lines_spin.setValue(header_tags.get("ImgHdr_PixY", 512))
            self.pixels_spin.setValue(header_tags.get("ImgHdr_PixX", 512))
            self.header_group.setVisible(True)
            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self._show_ptu_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read PTU header:\n{e}")


    def _show_markers(self):
        if not self.ptu_data:
            self.markers_output.setText("No file loaded.")
            return
        try:
            markers_dict = get_markers(self.ptu_data['reader'], chunk_limit=20) # Limit for speed
            if "error" in markers_dict:
                self.markers_output.setText(markers_dict["error"])
            else:
                text = ", ".join(f"{k}: {v}" for k, v in markers_dict.items())
                self.markers_output.setText(text)
        except Exception as e:
            self.markers_output.setText(f"Error: {e}")


    def _update_accumulation_widgets(self):
        while self.accu_container_layout.count():
            item = self.accu_container_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.accu_spinboxes.clear()

        num_sequences = self.sequences_spin.value()
        for i in range(num_sequences):
            row_widget = QWidget(); row_layout = QHBoxLayout(row_widget) 
            row_layout.setContentsMargins(0, 0, 0, 0)            
            label = QLabel(f"S{i+1}:")
            accu_spin = QSpinBox()
            accu_spin.setRange(1, 1000)
            accu_spin.setValue(1)
            row_layout.addWidget(label)
            row_layout.addWidget(accu_spin)
            self.accu_container_layout.addWidget(row_widget)
            self.accu_spinboxes.append(accu_spin)

        self.accu_container_layout.addStretch()


    def _on_estimate_shift_clicked(self):
        if not self.ptu_data:
            QMessageBox.warning(self, "No File", "Please load a PTU file first."); return
        
        self.shift_plot_data = None; self.btn_plot_shift.setEnabled(False)
        self.btn_estimate_shift.setText("Estimating...")
        self.log_text_edit.appendPlainText("Estimating bidirectional shift...")
        QApplication.processEvents()

        try:
            accumulations = tuple([spin.value() for spin in self.accu_spinboxes])
            if not accumulations:
                QMessageBox.warning(self, "Config Error", "Please set at least one accumulation value.")
                return
            
            temp_config = ScanConfig(
                lines=self.lines_spin.value(), 
                pixels=self.pixels_spin.value(), 
                bidirectional=True, 
                bidirectional_phase_shift=self.bidir_phase_spinbox.value(),
                line_accumulations=accumulations, 
                max_detector=self.max_detector_spin.value()
            )

            best_shift, plot_data = estimate_bidirectional_shift(reader=self.ptu_data['reader'], config=temp_config, wrap=self.ptu_data['constants']['wrap'], verbose=False)
            if best_shift is not None:
                self.bidir_phase_spinbox.setValue(best_shift)
                self.log_text_edit.appendPlainText(f"Estimation complete. Best shift: {best_shift:.5f}")
                self.shift_plot_data = plot_data
                self.btn_plot_shift.setEnabled(True)
            else:
                self.log_text_edit.appendPlainText("Estimation failed.")

        except Exception as e:
            error_msg = f"An error occurred during shift estimation:\n{e}"
            self.log_text_edit.appendPlainText(f"--- ERROR ---\n{error_msg}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Estimation Error", error_msg)

        finally:
            self.btn_estimate_shift.setText("Estimate")


    def _on_plot_shift_clicked(self):
        if self.shift_plot_data is None:
            QMessageBox.warning(self, "No Data", "Please run estimation first."); return
        
        dialog = QDialog(self); dialog.setWindowTitle("Bidirectional Shift Estimation")
        dialog_layout = QVBoxLayout(dialog); canvas = FigureCanvas(Figure(figsize=(6, 4)))
        dialog_layout.addWidget(canvas)
        plot_bidirectional_shift(self.shift_plot_data, ax=canvas.figure.subplots())
        dialog.exec_()


    def _run_reconstruction(self):
        if not self.ptu_data:
            QMessageBox.warning(self, "No Data", "Please load a PTU file first."); return
        self.log_text_edit.clear(); self.reconstruct_btn.setEnabled(False)
        self.log_text_edit.appendPlainText("Starting reconstruction..."); QApplication.processEvents()
        
        accumulations = tuple([spin.value() for spin in self.accu_spinboxes])

        scan_config = ScanConfig(
                lines=self.lines_spin.value(), 
                pixels=self.pixels_spin.value(), 
                frames=self.frames_spin.value(), 
                bidirectional=self.bidir_group.isChecked(),
                line_accumulations=accumulations, 
                bidirectional_phase_shift=self.bidir_phase_spinbox.value(), 
                max_detector=self.max_detector_spin.value()
            )

        selected_output = self.output_combo.currentData()
        outputs_to_generate = [selected_output]
        tcspc_override = self.tcspc_bins_spin.value()

        self.logger = ProgressLogger(mode='qt')
        self.logger.connect(self.log_text_edit.appendPlainText)

        worker = Worker(
            reconstruct_ptu_to_dataset,
            ptu_data=self.ptu_data,
            scan_config=scan_config,
            outputs=outputs_to_generate,
            tcspc_channels_override=tcspc_override,
            logger=self.logger 
        )

        self.logger = ThreadSafeLogger()
        self.logger.log_updated.connect(self.log_text_edit.appendPlainText)

        worker.signals.result.connect(lambda ds: self._on_reconstruction_result(ds, scan_config))
        worker.signals.finished.connect(self._on_reconstruction_finished)
        worker.signals.error.connect(self._on_reconstruction_error)
        
        self.threadpool.start(worker)



    def _on_reconstruction_result(self, dataset, scan_config):
        """This slot is called when the worker thread successfully finishes."""
        self.log_text_edit.appendPlainText("Reconstruction successful. Broadcasting result...")
        
        instrument_params = self.ptu_data['constants'].copy()
        instrument_params['tcspc_bins'] = self.tcspc_bins_spin.value()
        dataset.attrs['instrument_params'] = instrument_params
        dataset.attrs['scan_config'] = scan_config.to_dict()
        dataset.attrs['source_filename'] = self.ptu_filepath.name
        self.reconstruction_finished.emit(dataset)
        
    def _on_reconstruction_finished(self):
        """This slot is called when the worker thread is done (success or fail)."""
        self.reconstruct_btn.setEnabled(True)
        self.reconstruct_btn.setText("Reconstruct Image")
        print("Reconstruction thread finished.")

    def _on_reconstruction_error(self, error_tuple):
        """This slot is called if the worker thread raises an exception."""
        exctype, value, tb = error_tuple
        error_msg = f"--- THREAD ERROR ---\n{tb}"
        self.log_text_edit.appendPlainText(error_msg)
        QMessageBox.critical(self, "Reconstruction Error", f"An error occurred in the background thread:\n{value}")