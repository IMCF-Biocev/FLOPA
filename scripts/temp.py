import napari
from napari.types import LabelsData
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QTabWidget, QGridLayout, QComboBox, QRadioButton, QGroupBox,
    QTextEdit,
)

# --- Imports for interactive plot selection ---
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from magicgui.widgets import Container, SpinBox, FloatRangeSlider
import numpy as np
from pathlib import Path
import warnings

# --- Placeholder Functions (unchanged, but necessary) ---
def read_ptu_file(filepath: Path):
    print(f"--- MOCK: Reading and decoding {filepath.name} ---")
    mock_data = {
        'intensity': np.random.randint(0, 1000, size=(2, 1, 2, 128, 128)),
        'lifetime': np.random.uniform(0.5, 3.5, size=(2, 1, 2, 128, 128)),
        'decay_raw': np.random.poisson(5, size=(2, 1, 2, 128, 128, 256)),
        'time_resolution': 0.05, 'laser_frequency': 80.0,
    }
    return mock_data

def calculate_phasor(decay_data: np.ndarray, laser_frequency_mhz: float, harmonic: int = 1):
    """Placeholder function to calculate phasor coordinates (G and S)."""
    print(f"--- MOCK: Calculating phasor for {decay_data.shape[0]} pixels ---")
    n_pixels = decay_data.shape[0]
    g = np.random.normal(0.5, 0.2, n_pixels)
    s = np.random.normal(0.2, 0.1, n_pixels)
    return g, s

def fit_decay_curve(decay_curve, time_axis, model):
    print(f"--- MOCK: Fitting decay with '{model}' model ---")
    if model == "1-Component Exponential":
        return {'tau1 (ns)': 2.5, 'amplitude1': 1.0, 'chi-squared': 1.2}
    elif model == "2-Component Exponential":
        return {'tau1 (ns)': 1.0, 'amp1': 0.4, 'tau2 (ns)': 3.5, 'amp2': 0.6, 'chi-squared': 1.05}
    return {}

def create_flim_rgb_image(mean_photon_arrival_time, intensity, **kwargs):
    lt_min, lt_max = kwargs.get('lt_min'), kwargs.get('lt_max')
    int_min, int_max = kwargs.get('int_min'), kwargs.get('int_max')
    if lt_min is None: lt_min = np.nanmin(mean_photon_arrival_time)
    if lt_max is None: lt_max = np.nanmax(mean_photon_arrival_time)
    if int_min is None: int_min = np.nanmin(intensity)
    if int_max is None: int_max = np.nanmax(intensity)
    if lt_max == lt_min: lt_max = lt_min + 1
    if int_max == int_min: int_max = int_min + 1
    lt_norm = np.clip((mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1)
    from matplotlib import colormaps
    cmap = colormaps.get_cmap(kwargs.get('colormap', 'viridis'))
    lt_rgb = cmap(lt_norm)[..., :3]
    intensity_norm = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)
    return lt_rgb * intensity_norm[..., np.newaxis]

# --- The Main Widget Class ---

class FlimAnalyzerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Data & State Storage
        self.flim_data = {}
        self.phasor_results = {}
        self.phasor_pixel_map = None
        self.current_phasor_selection_indices = None
        self.segmentation_layer = None
        self.next_selection_label = 2
        self.rect_selector = None
        self.phasor_computation_mode = "Per Object"
        self.fit_results = {}
        self.current_decay_data = None
        self.current_time_axis = None

        self.setLayout(QVBoxLayout())
        self.tabs = QTabWidget()
        self.layout().addWidget(self.tabs)
        
        self._create_load_view_tab()
        self._create_phasor_tab()
        self._create_decay_tab()
        self._create_export_tab()
        self._set_tabs_enabled(False)

        self.phasor_canvas.mpl_connect('pick_event', self.on_phasor_pick)

    def _set_tabs_enabled(self, enabled: bool):
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, enabled)
    
    def _style_dark_plot(self, ax, fig):
        dark_grey, light_grey = '#262930', '#E0E0E0'
        fig.set_facecolor(dark_grey); ax.set_facecolor(dark_grey)
        for spine in ax.spines.values(): spine.set_color(light_grey)
        ax.xaxis.label.set_color(light_grey); ax.yaxis.label.set_color(light_grey)
        ax.title.set_color(light_grey)
        ax.tick_params(axis='x', colors=light_grey); ax.tick_params(axis='y', colors=light_grey)
        if ax.get_legend():
            legend = ax.get_legend()
            legend.get_frame().set_facecolor(dark_grey)
            legend.get_frame().set_edgecolor(light_grey)
            for text in legend.get_texts(): text.set_color(light_grey)

    # --- TAB 1: Load & View (Unchanged) ---
    def _create_load_view_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(); tab_widget.setLayout(layout)
        btn_load = QPushButton("Load FLIM Data (.ptu, .h5, etc.)"); btn_load.clicked.connect(self._on_load_data_clicked); layout.addWidget(btn_load)
        self.frame_selector = SpinBox(label="Frame", value=0, max=0)
        self.sequence_selector = SpinBox(label="Sequence", value=0, max=0)
        self.channel_selector = SpinBox(label="Channel", value=0, max=0)
        self.lt_range_slider = FloatRangeSlider(label="Lifetime Range (ns)", min=0, max=5, value=(0.5, 4.0))
        self.int_range_slider = FloatRangeSlider(label="Intensity Range", min=0, max=1000, value=(10, 500))
        self.controls_container = Container(widgets=[self.frame_selector, self.sequence_selector, self.channel_selector, self.lt_range_slider, self.int_range_slider])
        for widget in self.controls_container: widget.changed.connect(self._update_flim_view)
        layout.addWidget(self.controls_container.native); layout.addStretch()
        self.tabs.addTab(tab_widget, "Load & View")

    def _on_load_data_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select FLIM Data File")
        if not filepath: return
        self.flim_data = read_ptu_file(Path(filepath))
        n_frames, n_seq, n_chan, _, _ = self.flim_data['intensity'].shape
        self.frame_selector.max = n_frames - 1; self.sequence_selector.max = n_seq - 1; self.channel_selector.max = n_chan - 1
        self._update_flim_view(); self._update_mask_combobox(); self._set_tabs_enabled(True)

    def _update_flim_view(self):
        if not self.flim_data: return
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        lt_min, lt_max = self.lt_range_slider.value; int_min, int_max = self.int_range_slider.value
        lt_img = self.flim_data['lifetime'][frame, seq, chan, :, :]; int_img = self.flim_data['intensity'][frame, seq, chan, :, :]
        flim_rgb = create_flim_rgb_image(lt_img, int_img, lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)
        try: self.viewer.layers['FLIM Image'].data = flim_rgb
        except KeyError: self.viewer.add_image(flim_rgb, name='FLIM Image', rgb=True)

    # --- TAB 2: Phasor Analysis ---
    def _create_phasor_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(); tab_widget.setLayout(layout)
        mode_group = QGroupBox("Phasor Computation Mode")
        mode_layout = QVBoxLayout(); mode_group.setLayout(mode_layout)
        self.radio_per_object = QRadioButton("Per Object (Requires Mask)"); self.radio_per_object.setChecked(True)
        self.radio_whole_image = QRadioButton("Whole Image (Average)")
        self.radio_pixel_wise = QRadioButton("Pixel-wise (for Interactive Segmentation)")
        self.radio_per_object.toggled.connect(lambda: self._on_computation_mode_changed("Per Object"))
        self.radio_whole_image.toggled.connect(lambda: self._on_computation_mode_changed("Whole Image"))
        self.radio_pixel_wise.toggled.connect(lambda: self._on_computation_mode_changed("Pixel-wise"))
        mode_layout.addWidget(self.radio_per_object); mode_layout.addWidget(self.radio_whole_image); mode_layout.addWidget(self.radio_pixel_wise); layout.addWidget(mode_group)
        self.mask_group = QGroupBox("Mask Selection")
        mask_layout = QVBoxLayout(); self.mask_group.setLayout(mask_layout)
        self.mask_combobox = QComboBox(); self._update_mask_combobox()
        self.viewer.layers.events.inserted.connect(self._update_mask_combobox); self.viewer.layers.events.removed.connect(self._update_mask_combobox)
        mask_layout.addWidget(QLabel("Select from viewer:")); mask_layout.addWidget(self.mask_combobox); layout.addWidget(self.mask_group)
        btn_compute_phasor = QPushButton("Compute Phasor Plot"); btn_compute_phasor.clicked.connect(self._on_compute_phasor_clicked); layout.addWidget(btn_compute_phasor)
        self.phasor_figure = Figure(figsize=(5, 5)); self.phasor_canvas = FigureCanvas(self.phasor_figure)
        self.phasor_ax = self.phasor_figure.add_subplot(111); self.phasor_toolbar = NavigationToolbar(self.phasor_canvas, self)
        layout.addWidget(self.phasor_toolbar); layout.addWidget(self.phasor_canvas)
        seg_group = QGroupBox("Interactive Segmentation (Pixel-wise Mode)")
        seg_layout = QGridLayout(); seg_group.setLayout(seg_layout)
        btn_rect = QPushButton("Activate Rectangle Select"); btn_rect.clicked.connect(self._activate_rect_selector)
        self.next_label_id_label = QLabel(f"Next Label ID: {self.next_selection_label}")
        btn_apply = QPushButton("Apply Selection to Layer"); btn_apply.clicked.connect(self._on_apply_selection_clicked)
        btn_reset = QPushButton("Reset Segmentation Layer"); btn_reset.clicked.connect(self._on_reset_segmentation_clicked)
        seg_layout.addWidget(btn_rect, 0, 0, 1, 2)
        seg_layout.addWidget(self.next_label_id_label, 1, 0)
        seg_layout.addWidget(btn_apply, 2, 0, 1, 2); seg_layout.addWidget(btn_reset, 3, 0, 1, 2); layout.addWidget(seg_group)
        self.seg_group = seg_group
        self.tabs.addTab(tab_widget, "Phasor Analysis"); self._on_computation_mode_changed("Per Object"); self._plot_phasor()

    def _on_computation_mode_changed(self, mode):
        self.phasor_computation_mode = mode; self.mask_group.setEnabled(mode == "Per Object"); self.seg_group.setEnabled(mode == "Pixel-wise")
        if mode != "Pixel-wise" and self.rect_selector: self.rect_selector.set_active(False)

    def _update_mask_combobox(self):
        self.mask_combobox.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels): self.mask_combobox.addItem(layer.name)

    def _get_current_mask_layer(self):
        ### --- MODIFIED --- ###
        """Gets the selected mask layer object from the combobox."""
        if self.mask_combobox.count() == 0:
            print("No mask layer selected or available.")
            return None
        mask_name = self.mask_combobox.currentText()
        try:
            mask_layer = self.viewer.layers[mask_name]
            return mask_layer
        except KeyError:
            print(f"Mask layer '{mask_name}' not found.")
            return None
        
    def _on_compute_phasor_clicked(self):
        if not self.flim_data: print("No FLIM data loaded."); return
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        raw_decays = self.flim_data['decay_raw'][frame, seq, chan]
        self.phasor_results = {}; self.phasor_pixel_map = None
        if self.phasor_computation_mode == "Per Object":
            mask_layer = self._get_current_mask_layer()
            if mask_layer is None: print("Please select a valid mask layer."); return
            mask_data = mask_layer.data; label_ids = np.unique(mask_data[mask_data > 0]); g_coords, s_coords, obj_ids = [], [], []
            for label_id in label_ids:
                mean_decay = np.mean(raw_decays[mask_data == label_id], axis=0, keepdims=True)
                g, s = calculate_phasor(mean_decay, self.flim_data['laser_frequency'])
                g_coords.append(g[0]); s_coords.append(s[0]); obj_ids.append(label_id)
            self.phasor_results = {'g': np.array(g_coords), 's': np.array(s_coords), 'label_id': np.array(obj_ids)}
        elif self.phasor_computation_mode == "Whole Image":
            mean_decay = np.mean(raw_decays.reshape(-1, raw_decays.shape[-1]), axis=0, keepdims=True)
            g, s = calculate_phasor(mean_decay, self.flim_data['laser_frequency'])
            self.phasor_results = {'g': g, 's': s, 'label_id': np.array([1])}
        elif self.phasor_computation_mode == "Pixel-wise":
            y, x, _ = raw_decays.shape
            decays_flat = raw_decays.reshape(-1, raw_decays.shape[-1]); g, s = calculate_phasor(decays_flat, self.flim_data['laser_frequency'])
            yy, xx = np.mgrid[0:y, 0:x]; self.phasor_pixel_map = np.stack([g, s, yy.flatten(), xx.flatten()], axis=1)
        self._plot_phasor()

    def _plot_phasor(self):
        self.phasor_ax.clear(); uc_angles = np.linspace(0, np.pi, 180); uc_g = 0.5 + 0.5 * np.cos(uc_angles); uc_s = 0.5 * np.sin(uc_angles)
        self.phasor_ax.plot(uc_g, uc_s, 'w--', lw=1, alpha=0.7)
        if self.phasor_computation_mode == "Pixel-wise" and self.phasor_pixel_map is not None:
            self.phasor_ax.hist2d(self.phasor_pixel_map[:, 0], self.phasor_pixel_map[:, 1], bins=256, norm=LogNorm(), cmap='viridis')
            self.phasor_ax.set_title("Phasor Plot (Pixel-wise Density)")
        elif self.phasor_results:
            self.phasor_ax.scatter(self.phasor_results['g'], self.phasor_results['s'], c=self.phasor_results.get('label_id'), cmap='viridis', alpha=0.9, picker=True, pickradius=5)
            self.phasor_ax.set_title("Phasor Plot (Per Object/Image)")
        self.phasor_ax.set_xlabel("G"); self.phasor_ax.set_ylabel("S"); self.phasor_ax.set_aspect('equal', adjustable='box'); self.phasor_ax.grid(True, color='gray', linestyle='--', alpha=0.5)
        self._style_dark_plot(self.phasor_ax, self.phasor_figure); self.phasor_canvas.draw()
    
    def _activate_rect_selector(self):
        if self.phasor_computation_mode != "Pixel-wise": print("Selector only works in 'Pixel-wise' mode."); return
        if self.rect_selector: self.rect_selector.set_active(False)
        self.rect_selector = RectangleSelector(self.phasor_ax, self._on_rect_select, useblit=False, props=dict(facecolor='white', edgecolor='white', alpha=0.2, fill=True))
        print("Rectangle selector activated. Draw on the phasor plot.")
        
    def _on_rect_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # Ensure x1 < x2 and y1 < y2 for consistent selection
        g_min, g_max = min(x1, x2), max(x1, x2)
        s_min, s_max = min(y1, y2), max(y1, y2)
        
        points_to_check = self.phasor_pixel_map[:, 0:2]
        mask = (points_to_check[:, 0] >= g_min) & (points_to_check[:, 0] <= g_max) & \
               (points_to_check[:, 1] >= s_min) & (points_to_check[:, 1] <= s_max)
        self.current_phasor_selection_indices = np.nonzero(mask)[0]
        print(f"Selected {len(self.current_phasor_selection_indices)} pixels. Click 'Apply Selection to Layer'.")
        self.rect_selector.set_active(False)

    def _on_apply_selection_clicked(self):
        if self.current_phasor_selection_indices is None: print("No selection made."); return
        selected_coords = self.phasor_pixel_map[self.current_phasor_selection_indices, 2:4].astype(int)
        if self.segmentation_layer is None or 'Phasor Segmentation' not in self.viewer.layers:
            h, w = self.flim_data['intensity'].shape[-2:]
            self.segmentation_layer = self.viewer.add_labels(np.ones((h, w), dtype=np.uint16), name="Phasor Segmentation")
        self.segmentation_layer.data[selected_coords[:, 0], selected_coords[:, 1]] = self.next_selection_label
        self.segmentation_layer.refresh()
        print(f"Applied selection as Label {self.next_selection_label}.")
        self.next_selection_label += 1; self.next_label_id_label.setText(f"Next Label ID: {self.next_selection_label}")
        self.current_phasor_selection_indices = None

    def _on_reset_segmentation_clicked(self):
        if self.segmentation_layer and 'Phasor Segmentation' in self.viewer.layers: self.viewer.layers.remove('Phasor Segmentation')
        self.segmentation_layer = None; self.next_selection_label = 2; self.next_label_id_label.setText(f"Next Label ID: {self.next_selection_label}")
        print("Segmentation layer cleared.")

    def on_phasor_pick(self, event):
        if self.phasor_computation_mode != "Per Object": return
        ind = event.ind[0]; picked_label_id = self.phasor_results['label_id'][ind]
        mask_layer = self._get_current_mask_layer()
        if mask_layer: mask_layer.selected_label = picked_label_id; print(f"Selected object ID: {picked_label_id}")

    # --- TAB 3: Decay Analysis (Re-integrated) ---
    def _create_decay_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(); tab_widget.setLayout(layout)
        selection_group = QGroupBox("Plot Decay from Selected Label")
        selection_layout = QVBoxLayout(); selection_group.setLayout(selection_layout)
        selection_layout.addWidget(QLabel("In the napari viewer, select a Labels layer\nand click on an object to select its label."))
        btn_plot_decay = QPushButton("Plot Decay for Current Selection"); btn_plot_decay.clicked.connect(self._on_plot_decay_clicked)
        selection_layout.addWidget(btn_plot_decay); layout.addWidget(selection_group)
        self.decay_figure = Figure(figsize=(5, 4)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111); self.decay_toolbar = NavigationToolbar(self.decay_canvas, self)
        layout.addWidget(self.decay_toolbar); layout.addWidget(self.decay_canvas)
        fit_group = QGroupBox("Decay Fitting")
        fit_layout = QGridLayout(); fit_group.setLayout(fit_layout)
        self.fit_model_combo = QComboBox(); self.fit_model_combo.addItems(["1-Component Exponential", "2-Component Exponential"])
        btn_fit_decay = QPushButton("Fit Plotted Decay"); btn_fit_decay.clicked.connect(self._on_fit_decay_clicked)
        self.fit_results_display = QTextEdit(); self.fit_results_display.setReadOnly(True)
        fit_layout.addWidget(QLabel("Fit Model:"), 0, 0); fit_layout.addWidget(self.fit_model_combo, 0, 1)
        fit_layout.addWidget(btn_fit_decay, 1, 0, 1, 2)
        fit_layout.addWidget(QLabel("Fit Results:"), 2, 0, 1, 2); fit_layout.addWidget(self.fit_results_display, 3, 0, 1, 2)
        layout.addWidget(fit_group); self.tabs.addTab(tab_widget, "Decay Analysis"); self._plot_decay_curve(None, None, "No data selected")

    def _on_plot_decay_clicked(self):
        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers or not isinstance(selected_layers[0], napari.layers.Labels):
            print("Please select a Labels layer in the layer list."); return
        
        labels_layer = selected_layers[0]
        active_label = labels_layer.selected_label
        if active_label == 0: print(f"Please select a specific object (label > 0) in the '{labels_layer.name}' layer."); return
        
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        raw_decays = self.flim_data['decay_raw'][frame, seq, chan]
        pixel_mask = (labels_layer.data == active_label)
        if not np.any(pixel_mask): print(f"Label {active_label} not found in the mask data."); return
            
        aggregated_decay = np.sum(raw_decays[pixel_mask], axis=0)
        time_axis = np.arange(len(aggregated_decay)) * self.flim_data['time_resolution']
        self.current_decay_data, self.current_time_axis = aggregated_decay, time_axis
        self._plot_decay_curve(time_axis, aggregated_decay, f'Decay for Label {active_label} in "{labels_layer.name}"')

    def _plot_decay_curve(self, time_axis, decay_data, title):
        self.decay_ax.clear()
        if time_axis is not None and decay_data is not None:
            self.decay_ax.semilogy(time_axis, decay_data, label='Data', color='#00A0FF')
        self.decay_ax.set_title(title); self.decay_ax.set_xlabel("Time (ns)"); self.decay_ax.set_ylabel("Photon Counts")
        self.decay_ax.legend(); self.decay_ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.5)
        self._style_dark_plot(self.decay_ax, self.decay_figure); self.decay_canvas.draw()

    def _on_fit_decay_clicked(self):
        if self.current_decay_data is None: print("No decay curve plotted."); return
        model = self.fit_model_combo.currentText()
        results = fit_decay_curve(self.current_decay_data, self.current_time_axis, model)
        self.fit_results = results; display_text = f"Fit for model: {model}\n" + "-"*20 + "\n"
        for key, value in results.items(): display_text += f"{key}: {value:.4f}\n"
        self.fit_results_display.setText(display_text)

    # --- TAB 4: Export (Re-integrated) ---
    def _create_export_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(); tab_widget.setLayout(layout)
        btn_export_phasor = QPushButton("Export Phasor Data to CSV"); btn_export_phasor.clicked.connect(self._on_export_phasor)
        btn_export_fits = QPushButton("Export Fit Results to Text"); btn_export_fits.clicked.connect(self._on_export_fits)
        layout.addWidget(btn_export_phasor); layout.addWidget(btn_export_fits); layout.addStretch()
        self.tabs.addTab(tab_widget, "Export")

    def _on_export_phasor(self):
        data_to_export = None
        if self.phasor_computation_mode in ["Per Object", "Whole Image"] and self.phasor_results:
            data_to_export = self.phasor_results
        elif self.phasor_computation_mode == "Pixel-wise" and self.phasor_pixel_map is not None:
            data_to_export = {'g': self.phasor_pixel_map[:, 0], 's': self.phasor_pixel_map[:, 1], 'y': self.phasor_pixel_map[:, 2], 'x': self.phasor_pixel_map[:, 3]}
        else:
            print("No phasor data to export."); return
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Phasor Data", filter="CSV Files (*.csv)")
        if not filepath: return
        try:
            import pandas as pd
            pd.DataFrame(data_to_export).to_csv(filepath, index=False); print(f"Phasor data successfully exported to {filepath}")
        except Exception as e: print(f"Error exporting phasor data: {e}")

    def _on_export_fits(self):
        if not self.fit_results: print("No fit results to export."); return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Fit Results", filter="Text Files (*.txt)")
        if not filepath: return
        try:
            with open(filepath, 'w') as f:
                f.write("FLIM Decay Fit Results\n" + "="*25 + "\n")
                for key, value in self.fit_results.items(): f.write(f"{key}: {value}\n")
            print(f"Fit results successfully exported to {filepath}")
        except Exception as e: print(f"Error exporting fit results: {e}")


if __name__ == '__main__':
    viewer = napari.Viewer()
    my_widget = FlimAnalyzerWidget(viewer)
    viewer.window.add_dock_widget(my_widget, area='right', name='FLIM Analyzer')
    dummy_labels = np.zeros((128, 128), dtype=int)
    dummy_labels[20:50, 20:50] = 1; dummy_labels[60:100, 70:110] = 2; dummy_labels[10:30, 80:120] = 3
    viewer.add_labels(dummy_labels, name='Sample Mask')
    napari.run()