import napari
from napari.types import ImageData, LabelsData
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QTabWidget, QGridLayout, QComboBox, QRadioButton, QGroupBox,
    QSpinBox, QTextEdit
)
from qtpy.QtCore import Qt

# For embedding matplotlib plots in the Qt widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D # For custom plot elements
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm


from magicgui.widgets import Container, SpinBox, FloatRangeSlider
import numpy as np
from pathlib import Path
import warnings

# --- Placeholder for Data Reading and Analysis Functions (Unchanged) ---
# You will replace these with your actual implementations.

def read_ptu_file(filepath: Path):
    """
    Placeholder function to read a .ptu file and decode it.
    
    TODO: Implement the actual PTU reading and decoding logic here.
    """
    print(f"--- MOCK: Reading and decoding {filepath.name} ---")
    mock_data = {
        'intensity': np.random.randint(0, 1000, size=(2, 1, 2, 128, 128)),
        'lifetime': np.random.uniform(0.5, 3.5, size=(2, 1, 2, 128, 128)),
        'decay_raw': np.random.poisson(5, size=(2, 1, 2, 128, 128, 256)),
        'time_resolution': 0.05,
        'laser_frequency': 80.0,
    }
    print("--- MOCK: Data loaded successfully ---")
    return mock_data

def calculate_phasor(decay_data: np.ndarray, laser_frequency_mhz: float, harmonic: int = 1):
    """Placeholder function to calculate phasor coordinates (G and S)."""
    print(f"--- MOCK: Calculating phasor for {decay_data.shape[0]} pixels ---")
    n_pixels = decay_data.shape[0]
    g = np.random.normal(0.5, 0.2, n_pixels)
    s = np.random.normal(0.2, 0.1, n_pixels)
    return g, s

def fit_decay_curve(decay_curve: np.ndarray, time_axis: np.ndarray, model: str):
    """Placeholder function to fit a fluorescence decay curve."""
    print(f"--- MOCK: Fitting decay with '{model}' model ---")
    if model == "1-Component Exponential":
        return {'tau1 (ns)': 2.5, 'amplitude1': 1.0, 'chi-squared': 1.2}
    elif model == "2-Component Exponential":
        return {'tau1 (ns)': 1.0, 'amp1': 0.4, 'tau2 (ns)': 3.5, 'amp2': 0.6, 'chi-squared': 1.05}
    return {}

def create_flim_rgb_image(mean_photon_arrival_time, intensity, **kwargs):
    # This function is fine as is
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
    cmap = colormaps.get_cmap(kwargs.get('colormap', 'plasma'))
    lt_rgb = cmap(lt_norm)[..., :3]
    intensity_norm = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)
    return lt_rgb * intensity_norm[..., np.newaxis]

# --- The Main Widget Class ---

class FlimAnalyzerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        self.data_path = None
        self.flim_data = {}
        self.phasor_results = {}
        self.fit_results = {}
        
        self.setLayout(QVBoxLayout())
        self.tabs = QTabWidget()
        self.layout().addWidget(self.tabs)
        
        self._create_load_view_tab()
        self._create_phasor_tab()
        self._create_decay_tab()
        self._create_export_tab()

        self._set_tabs_enabled(False)
        
        ### --- NEW --- ###
        # Connect the event handler for picking points on the phasor plot
        self.phasor_canvas.mpl_connect('pick_event', self.on_phasor_pick)


    def _set_tabs_enabled(self, enabled: bool):
        self.tabs.setTabEnabled(1, enabled)
        self.tabs.setTabEnabled(2, enabled)
        self.tabs.setTabEnabled(3, enabled)

    ### --- NEW --- ###
    def _style_dark_plot(self, ax, fig):
        """Applies a dark theme to a matplotlib axis and figure."""
        dark_grey = '#262930'  # Napari's dark grey
        light_grey = '#E0E0E0' # A light grey for text

        fig.set_facecolor(dark_grey)
        ax.set_facecolor(dark_grey)

        for spine in ax.spines.values(): spine.set_color(light_grey)

        ax.xaxis.label.set_color(light_grey)
        ax.yaxis.label.set_color(light_grey)
        ax.title.set_color(light_grey)

        ax.tick_params(axis='x', colors=light_grey)
        ax.tick_params(axis='y', colors=light_grey)
        
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.get_frame().set_facecolor(dark_grey)
            legend.get_frame().set_edgecolor(light_grey)
            for text in legend.get_texts():
                text.set_color(light_grey)

    # --- TAB 1: Load & View (Unchanged from previous version) ---
    def _create_load_view_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        tab_widget.setLayout(layout)
        btn_load = QPushButton("Load FLIM Data (.ptu, .h5, etc.)")
        btn_load.clicked.connect(self._on_load_data_clicked)
        layout.addWidget(btn_load)
        self.frame_selector = SpinBox(label="Frame", value=0, max=0)
        self.sequence_selector = SpinBox(label="Sequence", value=0, max=0)
        self.channel_selector = SpinBox(label="Channel", value=0, max=0)
        self.lt_range_slider = FloatRangeSlider(label="Lifetime Range (ns)", min=0, max=5, value=(0.5, 4.0))
        self.int_range_slider = FloatRangeSlider(label="Intensity Range", min=0, max=1000, value=(10, 500))
        self.controls_container = Container(widgets=[self.frame_selector, self.sequence_selector, self.channel_selector, self.lt_range_slider, self.int_range_slider])
        for widget in self.controls_container: widget.changed.connect(self._update_flim_view)
        layout.addWidget(self.controls_container.native)
        layout.addStretch()
        self.tabs.addTab(tab_widget, "Load & View")

    def _on_load_data_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select FLIM Data File")
        if not filepath: return
        self.data_path = Path(filepath)
        try:
            self.flim_data = read_ptu_file(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}"); return
        n_frames, n_seq, n_chan, _, _ = self.flim_data['intensity'].shape
        self.frame_selector.max = n_frames - 1
        self.sequence_selector.max = n_seq - 1
        self.channel_selector.max = n_chan - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.lt_range_slider.max = np.nanmax(self.flim_data['lifetime']) * 1.1
            self.int_range_slider.max = np.nanmax(self.flim_data['intensity']) * 1.1
            self.int_range_slider.value = (np.nanpercentile(self.flim_data['intensity'], 5), np.nanpercentile(self.flim_data['intensity'], 99.8))
        self._update_flim_view()
        self._update_mask_combobox()
        self._set_tabs_enabled(True)
        print("Data loaded and viewer updated.")

    def _update_flim_view(self):
        if not self.flim_data: return
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        lt_min, lt_max = self.lt_range_slider.value
        int_min, int_max = self.int_range_slider.value
        lt_img = self.flim_data['lifetime'][frame, seq, chan, :, :]
        int_img = self.flim_data['intensity'][frame, seq, chan, :, :]
        flim_rgb = create_flim_rgb_image(lt_img, int_img, lt_min=lt_min, lt_max=lt_max, int_min=int_min, int_max=int_max)
        try:
            self.viewer.layers['FLIM Image'].data = flim_rgb
        except KeyError:
            self.viewer.add_image(flim_rgb, name='FLIM Image', rgb=True)

    # --- TAB 2: Phasor Analysis ---
    def _create_phasor_tab(self):
        # This function structure is mostly the same, but the plot creation is what matters
        tab_widget = QWidget()
        layout = QVBoxLayout()
        tab_widget.setLayout(layout)
        mask_group = QGroupBox("Mask Selection")
        mask_layout = QVBoxLayout()
        mask_group.setLayout(mask_layout)
        self.mask_combobox = QComboBox()
        self.mask_combobox.setToolTip("Select an active Labels layer from the viewer.")
        self._update_mask_combobox()
        self.viewer.layers.events.inserted.connect(self._update_mask_combobox)
        self.viewer.layers.events.removed.connect(self._update_mask_combobox)
        btn_refresh_masks = QPushButton("Refresh Mask List")
        btn_refresh_masks.clicked.connect(self._update_mask_combobox)
        btn_load_mask_file = QPushButton("Load Mask from File")
        btn_load_mask_file.clicked.connect(self._on_load_mask_file_clicked)
        mask_layout.addWidget(QLabel("Select from viewer:"))
        mask_layout.addWidget(self.mask_combobox)
        mask_layout.addWidget(btn_refresh_masks)
        mask_layout.addWidget(QLabel("Or load from file:"))
        mask_layout.addWidget(btn_load_mask_file)
        layout.addWidget(mask_group)
        phasor_group = QGroupBox("Phasor Calculation")
        phasor_layout = QGridLayout()
        phasor_group.setLayout(phasor_layout)
        self.phasor_harmonic = QSpinBox(); self.phasor_harmonic.setMinimum(1); self.phasor_harmonic.setValue(1)
        phasor_layout.addWidget(QLabel("Harmonic:"), 0, 0)
        phasor_layout.addWidget(self.phasor_harmonic, 0, 1)
        btn_compute_phasor = QPushButton("Compute Phasor Plot per Object")
        btn_compute_phasor.clicked.connect(self._on_compute_phasor_clicked)
        phasor_layout.addWidget(btn_compute_phasor, 1, 0, 1, 2)
        layout.addWidget(phasor_group)

        # Phasor Plot Display
        self.phasor_figure = Figure(figsize=(5, 5))
        self.phasor_canvas = FigureCanvas(self.phasor_figure)
        self.phasor_ax = self.phasor_figure.add_subplot(111)
        self.phasor_toolbar = NavigationToolbar(self.phasor_canvas, self)
        
        layout.addWidget(self.phasor_toolbar)
        layout.addWidget(self.phasor_canvas)
        
        self.tabs.addTab(tab_widget, "Phasor Analysis")
        # Initialize with an empty styled plot
        self._plot_phasor()

    def _update_mask_combobox(self):
        self.mask_combobox.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.mask_combobox.addItem(layer.name)

    def _on_load_mask_file_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Mask Image File")
        if filepath:
            try:
                self.viewer.open(filepath, plugin='napari-builtins', layer_type='labels')
            except Exception as e:
                print(f"Could not load mask file: {e}")
    
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
        mask_layer = self._get_current_mask_layer()
        if mask_layer is None: return
        mask_data = mask_layer.data
        if not self.flim_data: print("No FLIM data loaded."); return
        
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        raw_decays = self.flim_data['decay_raw'][frame, seq, chan]

        label_ids = np.unique(mask_data[mask_data > 0])
        g_coords, s_coords, obj_ids = [], [], []

        for label_id in label_ids:
            pixel_mask = (mask_data == label_id)
            mean_object_decay = np.mean(raw_decays[pixel_mask], axis=0)
            g, s = calculate_phasor(mean_object_decay.reshape(1, -1), self.flim_data['laser_frequency'], self.phasor_harmonic.value)
            g_coords.append(g[0]); s_coords.append(s[0]); obj_ids.append(label_id)

        self.phasor_results = {'g': np.array(g_coords), 's': np.array(s_coords), 'label_id': np.array(obj_ids)}
        print(f"Phasor calculated for {len(obj_ids)} objects.")
        self._plot_phasor()


    def _plot_phasor(self):
        ### --- MODIFIED --- ###
        """Draws the styled, interactive phasor plot on the canvas."""
        self.phasor_ax.clear()
        
        # Plot the universal circle
        uc_angles = np.linspace(0, np.pi, 180)
        uc_g = 0.5 + 0.5 * np.cos(uc_angles)
        uc_s = 0.5 * np.sin(uc_angles)
        self.phasor_ax.plot(uc_g, uc_s, 'w--', lw=1, alpha=0.7) # White dashed line

        if self.phasor_results:
            # Add picker=True to make scatter points selectable
            self.phasor_ax.scatter(
                self.phasor_results['g'], self.phasor_results['s'],
                c=self.phasor_results['label_id'], cmap='viridis', alpha=0.9,
                picker=True, pickradius=5  # a 5-pixel radius for picking
            )
            
        self.phasor_ax.set_title("Phasor Plot (Click points to select object)")
        self.phasor_ax.set_xlabel("G")
        self.phasor_ax.set_ylabel("S")
        self.phasor_ax.set_aspect('equal', adjustable='box')
        self.phasor_ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Apply the dark theme styling
        self._style_dark_plot(self.phasor_ax, self.phasor_figure)
        
        self.phasor_canvas.draw()

    ### --- NEW --- ###
    def on_phasor_pick(self, event):
        """Event handler for clicking on a point in the phasor plot."""
        if not isinstance(event.artist, Line2D): # Ignore clicks on the universal circle
            # Get the index of the picked point
            ind = event.ind[0]
            
            # Find the corresponding label ID
            picked_label_id = self.phasor_results['label_id'][ind]
            
            # Find the mask layer in napari
            mask_layer = self._get_current_mask_layer()
            
            if mask_layer:
                # Set the selected_label attribute to highlight the object
                mask_layer.selected_label = picked_label_id
                print(f"Selected object ID: {picked_label_id}")
            else:
                print("Could not find the associated mask layer to highlight the selection.")


    # --- TAB 3: Decay Analysis ---
    def _create_decay_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        tab_widget.setLayout(layout)
        selection_group = QGroupBox("Select Data for Decay Curve")
        selection_layout = QVBoxLayout()
        selection_group.setLayout(selection_layout)
        self.radio_select_label = QRadioButton("From Selected Label in Mask"); self.radio_select_label.setChecked(True)
        selection_layout.addWidget(self.radio_select_label)
        btn_plot_decay = QPushButton("Plot Decay Curve for Selected Label")
        btn_plot_decay.clicked.connect(self._on_plot_decay_clicked)
        selection_layout.addWidget(btn_plot_decay)
        layout.addWidget(selection_group)

        self.decay_figure = Figure(figsize=(5, 4))
        self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111)
        self.decay_toolbar = NavigationToolbar(self.decay_canvas, self)
        layout.addWidget(self.decay_toolbar); layout.addWidget(self.decay_canvas)
        
        fit_group = QGroupBox("Decay Fitting")
        fit_layout = QGridLayout()
        fit_group.setLayout(fit_layout)
        self.fit_model_combo = QComboBox(); self.fit_model_combo.addItems(["1-Component Exponential", "2-Component Exponential"])
        btn_fit_decay = QPushButton("Fit Plotted Decay")
        btn_fit_decay.clicked.connect(self._on_fit_decay_clicked)
        self.fit_results_display = QTextEdit(); self.fit_results_display.setReadOnly(True)
        fit_layout.addWidget(QLabel("Fit Model:"), 0, 0); fit_layout.addWidget(self.fit_model_combo, 0, 1)
        fit_layout.addWidget(btn_fit_decay, 1, 0, 1, 2)
        fit_layout.addWidget(QLabel("Fit Results:"), 2, 0, 1, 2)
        fit_layout.addWidget(self.fit_results_display, 3, 0, 1, 2)
        layout.addWidget(fit_group)
        self.tabs.addTab(tab_widget, "Decay Analysis")
        self.current_decay_data = None; self.current_time_axis = None
        # Initialize with an empty styled plot
        self._plot_decay_curve(None, None, "No data selected")

    def _on_plot_decay_clicked(self):
        mask_layer = self._get_current_mask_layer()
        if mask_layer is None: print("Please select a mask layer."); return
        
        active_label = mask_layer.selected_label
        if active_label == 0:
            print("Please select a specific label (object) in the mask layer in the viewer."); return
        
        frame, seq, chan = self.frame_selector.value, self.sequence_selector.value, self.channel_selector.value
        raw_decays = self.flim_data['decay_raw'][frame, seq, chan]

        pixel_mask = (mask_layer.data == active_label)
        if not np.any(pixel_mask):
            print(f"Label {active_label} not found in the mask data."); return
            
        aggregated_decay = np.sum(raw_decays[pixel_mask], axis=0)
        time_axis = np.arange(len(aggregated_decay)) * self.flim_data['time_resolution']
        
        self.current_decay_data = aggregated_decay
        self.current_time_axis = time_axis
        
        self._plot_decay_curve(time_axis, aggregated_decay, f'Decay Curve for Label {active_label}')

    def _plot_decay_curve(self, time_axis, decay_data, title):
        ### --- MODIFIED --- ###
        """Helper function to draw the styled decay curve plot."""
        self.decay_ax.clear()
        if time_axis is not None and decay_data is not None:
            self.decay_ax.semilogy(time_axis, decay_data, label='Data', color='#00A0FF') # Bright blue color
            
        self.decay_ax.set_title(title)
        self.decay_ax.set_xlabel("Time (ns)")
        self.decay_ax.set_ylabel("Photon Counts")
        self.decay_ax.legend()
        self.decay_ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Apply dark theme styling
        self._style_dark_plot(self.decay_ax, self.decay_figure)
        
        self.decay_canvas.draw()

    def _on_fit_decay_clicked(self):
        if self.current_decay_data is None:
            print("No decay curve plotted. Please plot a decay first."); return
        
        model = self.fit_model_combo.currentText()
        results = fit_decay_curve(self.current_decay_data, self.current_time_axis, model)
        self.fit_results = results
        
        display_text = f"Fit for model: {model}\n" + "-"*20 + "\n"
        for key, value in results.items():
            display_text += f"{key}: {value:.4f}\n"
        self.fit_results_display.setText(display_text)
        
        print("TODO: Plot the fitted curve on the decay plot.")

    # --- TAB 4: Export (Unchanged) ---
    def _create_export_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(); tab_widget.setLayout(layout)
        btn_export_phasor = QPushButton("Export Phasor Data to CSV"); btn_export_phasor.clicked.connect(self._on_export_phasor)
        btn_export_fits = QPushButton("Export Fit Results to Text"); btn_export_fits.clicked.connect(self._on_export_fits)
        layout.addWidget(btn_export_phasor); layout.addWidget(btn_export_fits); layout.addStretch()
        self.tabs.addTab(tab_widget, "Export")

    def _on_export_phasor(self):
        if not self.phasor_results: print("No phasor data to export."); return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Phasor Data", filter="CSV Files (*.csv)")
        if not filepath: return
        try:
            import pandas as pd
            pd.DataFrame(self.phasor_results).to_csv(filepath, index=False)
            print(f"Phasor data successfully exported to {filepath}")
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

    # Example of adding a dummy labels layer for testing
    dummy_labels = np.zeros((128, 128), dtype=int)
    dummy_labels[20:50, 20:50] = 1
    dummy_labels[60:100, 70:110] = 2
    dummy_labels[10:30, 80:120] = 3
    viewer.add_labels(dummy_labels, name='Sample Mask')

    napari.run()