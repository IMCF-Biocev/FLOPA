from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QFileDialog, QRadioButton, QButtonGroup, QSlider, QMessageBox, QCheckBox, QHBoxLayout,
    QApplication
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from flopa.processing.phasor import calculate_phasor
from flopa.widgets.utils.style import dark_plot, light_plot
import numpy as np
import napari
from skimage.io import imread
from pathlib import Path

from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import itertools # For cycling through colors

class PhasorPanel(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # --- Standard data attributes ---
        self.intensity_image = None
        self.lifetime_image = None
        self.loaded_mask_data = None
        self.last_computation_mask = None

        # --- Caching for plot redrawing ---
        self.last_g_list = None
        self.last_s_list = None
        self.last_color_list = None
        
        # ### NEW/MODIFIED ###: Attributes for interactive ROI selection
        self.pixel_phasor_data = None   # Caches {g, s, y, x} for per-pixel plots
        self.object_phasor_data = None  # Caches {g, s, label_id} for per-object plots
        self.phasor_selectors = []      # MUST keep a reference to selectors
        self.phasor_rois = []           # Stores ROI data: {'extents':(l,r,b,t), 'color':'', 'id':1}
        self.roi_color_cycle = itertools.cycle([
            '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1'
        ])
        self.is_drawing_mode_active = False
        self.roi_selector = None # Will hold our single, persistent selector

        self._init_ui()

        # Connect canvas events AFTER UI is built
        self.phasor_canvas.mpl_connect('key_press_event', self._on_key_press)
        self.phasor_canvas.mpl_connect('button_press_event', self._on_plot_click)
        
        # Initialize the persistent but inactive selector
        self._initialize_roi_selector()

        self._plot_phasor(message="Load data and generate FLIM image first.")

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

# --- Thresholding Group ---
        threshold_group = QGroupBox("Intensity Thresholding")
        threshold_layout = QGridLayout()
        threshold_group.setLayout(threshold_layout)

        self.btn_threshold = QPushButton("Enable Intensity Thresholding")
        # FIX 1: Connection now works as the method is defined before this call.
        self.btn_threshold.clicked.connect(self._on_threshold_button_clicked)
        threshold_layout.addWidget(self.btn_threshold, 0, 0, 1, 2)

        self.thresh_min = QSlider()
        self.thresh_min.setMinimum(0)
        self.thresh_min.setMaximum(1) # Placeholder
        self.thresh_min.setValue(0)
        self.thresh_min.setOrientation(1) # Qt.Horizontal
        self.thresh_min.setVisible(False)
        self.thresh_min.valueChanged.connect(self._update_threshold_mask)

        self.thresh_max = QSlider()
        self.thresh_max.setMinimum(0)
        self.thresh_max.setMaximum(1) # Placeholder
        self.thresh_max.setValue(1)
        self.thresh_max.setOrientation(1) # Qt.Horizontal
        self.thresh_max.setVisible(False)
        self.thresh_max.valueChanged.connect(self._update_threshold_mask)

        self.lbl_min = QLabel("Min Intensity: 0")
        self.lbl_min.setVisible(False)
        self.lbl_max = QLabel("Max Intensity: 1")
        self.lbl_max.setVisible(False)

        threshold_layout.addWidget(self.lbl_min, 1, 0)
        threshold_layout.addWidget(self.thresh_min, 2, 0)
        threshold_layout.addWidget(self.lbl_max, 1, 1)
        threshold_layout.addWidget(self.thresh_max, 2, 1)

        layout.addWidget(threshold_group)

        # --- Mask Selection Group ---
        mask_group = QGroupBox("Mask Selection")
        mask_layout = QVBoxLayout()
        mask_group.setLayout(mask_layout)

        self.mask_combobox = QComboBox()
        self._update_mask_combobox()
        self.viewer.layers.events.inserted.connect(self._update_mask_combobox)
        self.viewer.layers.events.removed.connect(self._update_mask_combobox)

        btn_refresh_masks = QPushButton("Refresh Mask List")
        btn_refresh_masks.clicked.connect(self._update_mask_combobox)

        #btn_load_mask_file = QPushButton("Load Mask from File")
        #btn_load_mask_file.clicked.connect(self._on_load_mask_file_clicked)

        mask_layout.addWidget(QLabel("You can also drag and drop mask files into napari"))
        mask_layout.addWidget(QLabel("Select from viewer layers:"))
        mask_layout.addWidget(self.mask_combobox)
        mask_layout.addWidget(btn_refresh_masks)
        #mask_layout.addWidget(QLabel("Or load mask from file:"))
        #mask_layout.addWidget(btn_load_mask_file)
        layout.addWidget(mask_group)

        # --- Phasor Mode Selection ---
        mode_group = QGroupBox("Phasor Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.per_pixel_radio = QRadioButton("Per Pixel (within mask)")
        self.per_object_radio = QRadioButton("Per Object (in labels layer)")
        self.per_object_radio.setChecked(True)

        self.mode_buttons = QButtonGroup()
        self.mode_buttons.addButton(self.per_pixel_radio, 0)
        self.mode_buttons.addButton(self.per_object_radio, 1)

        mode_layout.addWidget(self.per_object_radio)
        mode_layout.addWidget(self.per_pixel_radio)
        layout.addWidget(mode_group)

        # --- Phasor Calculation Group ---
        phasor_group = QGroupBox("Phasor Calculation")
        phasor_layout = QGridLayout()
        phasor_group.setLayout(phasor_layout)

        self.phasor_harmonic = QSpinBox()
        #self.phasor_harmonic.setMinimum(1)
        self.phasor_harmonic.setValue(1)
        # phasor_layout.addWidget(QLabel("Harmonic:"), 0, 0)
        # phasor_layout.addWidget(self.phasor_harmonic, 0, 1)

        btn_compute_phasor = QPushButton("Compute Phasor Plot")
        btn_compute_phasor.clicked.connect(self._on_compute_phasor_clicked)
        phasor_layout.addWidget(btn_compute_phasor, 1, 0, 1, 2)
        layout.addWidget(phasor_group)


        # ### NEW/MODIFIED ###: ROI Selection and Remapping Group
        roi_group = QGroupBox("ROI Selection & Remapping")
        roi_layout = QVBoxLayout()
        roi_group.setLayout(roi_layout)

        self.btn_add_roi = QPushButton("Add New ROI")
        self.btn_add_roi.setToolTip("Enable interactive selection on the phasor plot.")
        self.btn_add_roi.clicked.connect(self._on_add_roi_clicked)
        roi_layout.addWidget(self.btn_add_roi)

        self.btn_generate_mask = QPushButton("Generate Mask from ROIs")
        self.btn_generate_mask.setToolTip("Create a new napari labels layer from the defined ROIs.")
        self.btn_generate_mask.clicked.connect(self._on_generate_mask_clicked)
        roi_layout.addWidget(self.btn_generate_mask)

        self.btn_clear_rois = QPushButton("Clear All ROIs")
        self.btn_clear_rois.clicked.connect(self._on_clear_rois_clicked)
        roi_layout.addWidget(self.btn_clear_rois)
        
        # Initially disable buttons that require a per-pixel plot
        self.btn_add_roi.setEnabled(False)
        self.btn_generate_mask.setEnabled(False)
        layout.addWidget(roi_group)

        # Theme Selection Checkbox  
        plot_options_layout = QHBoxLayout()
        self.dark_mode_checkbox = QCheckBox("Use Dark Theme")
        self.dark_mode_checkbox.setChecked(True) # Default to dark theme
        self.dark_mode_checkbox.stateChanged.connect(self._on_theme_changed)
        plot_options_layout.addWidget(self.dark_mode_checkbox)
        plot_options_layout.addStretch() # Pushes checkbox to the left
        layout.addLayout(plot_options_layout)

        # --- Phasor Plot Display ---
        self.phasor_figure = Figure(figsize=(5, 5))
        self.phasor_canvas = FigureCanvas(self.phasor_figure)
        self.phasor_ax = self.phasor_figure.add_subplot(111)
        self.phasor_toolbar = NavigationToolbar(self.phasor_canvas, self)

        layout.addWidget(self.phasor_toolbar)
        layout.addWidget(self.phasor_canvas)


    ### ====================================================================
    ### NEW METHODS FOR ROI SELECTION
    ### ====================================================================
    def _on_theme_changed(self):
        """Called when the dark mode checkbox is toggled. Redraws with last data."""
        # We simply call _plot_phasor again with the cached data.
        # It will re-apply the theme and redraw the points.
        self._plot_phasor(
            self.last_g_list, 
            self.last_s_list, 
            self.last_color_list,
            message="No data computed yet." if self.last_g_list is None else None
        )


    # def _on_add_roi_clicked(self):
    #     """Activates a new RectangleSelector on the plot."""
        
    #     # The selector MUST be stored in a list to prevent garbage collection
    #     selector = RectangleSelector(
    #         self.phasor_ax, 
    #         self._on_roi_select,
    #         useblit=True,
    #         button=[1],  # Left mouse button
    #         minspanx=0.01, minspany=0.01,
    #         spancoords='data',
    #         interactive=True
    #     )
    #     self.phasor_selectors.append(selector)
    #     QMessageBox.information(self, "Add ROI", "ROI selection is active. Please draw a rectangle on the phasor plot.")

    def _on_roi_select(self, eclick, erelease):
        """Callback function when a rectangle is drawn."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        extents = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        
        # Store the new ROI's data
        roi_data = {
            'extents': extents,
            'color': next(self.roi_color_cycle),
            'id': len(self.phasor_rois) + 1 # Assign label ID (1, 2, 3...)
        }
        self.phasor_rois.append(roi_data)
        
        # Deactivate the selector so we can add another one later
        self.phasor_selectors[-1].set_active(False)
        
        # Redraw the plot to show the new, permanent ROI box
        self._plot_phasor(self.last_g_list, self.last_s_list, self.last_color_list)
        print(f"ROI {roi_data['id']} created.")


    def _initialize_roi_selector(self):
        """Creates our persistent RectangleSelector and keeps it inactive."""
        self.roi_selector = RectangleSelector(
            self.phasor_ax, 
            self._on_roi_select_continuous,
            useblit=True,
            button=[1], # Left mouse button
            minspanx=0.01, minspany=0.01,
            spancoords='data',
            interactive=False, # We will manage interactivity ourselves
            props={'edgecolor': 'white', 'linestyle': '--', 'linewidth': 2}
        )
        self.roi_selector.set_active(False)

    def _set_drawing_mode(self, active: bool):
        """Central controller to turn drawing mode on or off."""
        self.is_drawing_mode_active = active
        if active:
            print("ROI drawing mode ENABLED.")
            self.btn_add_roi.setText("Stop Adding ROIs (Esc)")
            self.phasor_canvas.setCursor(Qt.CrossCursor)
            self.roi_selector.set_active(True)
        else:
            print("ROI drawing mode DISABLED.")
            self.btn_add_roi.setText("Add New ROI")
            self.phasor_canvas.setCursor(Qt.ArrowCursor)
            self.roi_selector.set_active(False)
        self.phasor_canvas.draw_idle()

    def _on_key_press(self, event):
        """Handles key presses on the plot canvas."""
        if event.key == 'escape' and self.is_drawing_mode_active:
            self._set_drawing_mode(False)

    def _on_plot_click(self, event):
        """Handles right-click to delete an ROI."""
        # Only act if it's a right-click and we are NOT in drawing mode
        if event.button == 3 and not self.is_drawing_mode_active:
            # Find which ROI was clicked, if any
            for i, roi in reversed(list(enumerate(self.phasor_rois))):
                xmin, xmax, ymin, ymax = roi['extents']
                if event.xdata >= xmin and event.xdata <= xmax and \
                   event.ydata >= ymin and event.ydata <= ymax:
                    
                    print(f"Deleting ROI {roi['id']} with right-click.")
                    self.phasor_rois.pop(i)
                    # Re-label remaining ROIs to be contiguous (1, 2, 3...)
                    for new_id, r in enumerate(self.phasor_rois, 1):
                        r['id'] = new_id
                    
                    self._plot_phasor(self.last_g_list, self.last_s_list, self.last_color_list)
                    return # Stop after deleting the first one found

    def _on_add_roi_clicked(self):
        """Toggles the drawing mode state."""
        # This button is now just a switch
        self._set_drawing_mode(not self.is_drawing_mode_active)

    def _on_roi_select_continuous(self, eclick, erelease):
        """Callback for CONTINUOUS drawing. Adds a new ROI without deactivating."""
        extents = (min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata), 
                   min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata))
        
        roi_data = {
            'extents': extents,
            'color': next(self.roi_color_cycle),
            'id': len(self.phasor_rois) + 1
        }
        self.phasor_rois.append(roi_data)
        
        # Redraw the plot to show the new, permanent ROI box
        # The selector remains active for the next drawing!
        self._plot_phasor(self.last_g_list, self.last_s_list, self.last_color_list)
        print(f"ROI {roi_data['id']} created. Ready to draw another.")


    def _on_generate_mask_clicked(self):
        """
        DISPATCHER: Checks which mode is active (pixel or object) and calls the
        appropriate mask generation logic.
        """
        if self.is_drawing_mode_active:
            self._set_drawing_mode(False)

        if not self.phasor_rois:
            QMessageBox.warning(self, "No ROIs", "Please define at least one ROI."); return

        if self.pixel_phasor_data is not None:
            self._generate_mask_from_pixels()
        elif self.object_phasor_data is not None:
            self._generate_mask_from_objects()
        else:
            QMessageBox.critical(self, "Error", "No cached data available. Please re-run 'Compute Phasor Plot'.")

    def _generate_mask_from_pixels(self):
        """The original 'generate mask' logic for per-pixel data."""
        print("Generating mask from PIXEL ROIs...")
        if self.last_computation_mask is None: return # Safety check

        unmatched_label = len(self.phasor_rois) + 1
        output_mask = np.zeros(self.intensity_image.shape, dtype=np.uint16)
        output_mask[self.last_computation_mask > 0] = unmatched_label
        
        data = self.pixel_phasor_data
        for roi in self.phasor_rois:
            xmin, xmax, ymin, ymax = roi['extents']
            label_id = roi['id']
            in_roi = (data['g'] >= xmin) & (data['g'] <= xmax) & (data['s'] >= ymin) & (data['s'] <= ymax)
            output_mask[data['y'][in_roi], data['x'][in_roi]] = label_id
            print(f"Assigned {np.sum(in_roi)} pixels to ROI {label_id}.")

        self.viewer.add_labels(output_mask, name="Phasor_ROIs_Pixel")
        print("Pixel mask generation complete.")
        
    def _generate_mask_from_objects(self):
        """The NEW 'generate mask' logic for per-object data."""
        print("Generating mask from OBJECT ROIs...")
        if self.last_computation_mask is None: return # Safety check

        unmatched_label = len(self.phasor_rois) + 1
        output_mask = np.zeros(self.intensity_image.shape, dtype=np.uint16)
        output_mask[self.last_computation_mask > 0] = unmatched_label

        data = self.object_phasor_data
        for roi in self.phasor_rois:
            xmin, xmax, ymin, ymax = roi['extents']
            roi_label = roi['id']

            # Find which OBJECTS (not pixels) are in the ROI
            in_roi_mask = (data['g'] >= xmin) & (data['g'] <= xmax) & \
                          (data['s'] >= ymin) & (data['s'] <= ymax)
            
            # Get the original label IDs of those objects
            selected_original_labels = data['label_id'][in_roi_mask]

            if len(selected_original_labels) == 0:
                continue

            print(f"ROI {roi_label} contains original objects: {selected_original_labels}")

            # Now, for each selected original label, find all its pixels in the
            # original computation mask and assign them the new ROI label.
            for original_label_id in selected_original_labels:
                output_mask[self.last_computation_mask == original_label_id] = roi_label
        
        self.viewer.add_labels(output_mask, name="Phasor_ROIs_Object")
        print("Object mask generation complete.")


    # def _on_generate_mask_clicked(self):
    #     """Generates a new labels layer from the defined ROIs."""
    #     if not self.phasor_rois:
    #         QMessageBox.warning(self, "No ROIs", "Please define at least one ROI before generating a mask.")
    #         return
    #     if self.pixel_phasor_data is None:
    #         QMessageBox.critical(self, "Error", "No per-pixel phasor data available. Please re-run 'Compute Phasor Plot' in 'Per Pixel' mode.")
    #         return

    #     print("Generating mask from ROIs...")
        
    #     # The value for pixels that don't fall into any ROI
    #     background_label = len(self.phasor_rois) + 1
        
    #     # Create a new mask array, initialized with the background label
    #     output_mask = np.full(self.intensity_image.shape, background_label, dtype=np.uint16)
        
    #     # Get the cached pixel data
    #     g_coords = self.pixel_phasor_data['g']
    #     s_coords = self.pixel_phasor_data['s']
    #     y_coords = self.pixel_phasor_data['y']
    #     x_coords = self.pixel_phasor_data['x']

    #     for roi in self.phasor_rois:
    #         xmin, xmax, ymin, ymax = roi['extents']
    #         label_id = roi['id']
            
    #         # Find all phasor points inside this ROI's rectangle
    #         in_roi_mask = (g_coords >= xmin) & (g_coords <= xmax) & \
    #                       (s_coords >= ymin) & (s_coords <= ymax)
            
    #         # Get the (y,x) image coordinates of these points
    #         pixels_y_in_roi = y_coords[in_roi_mask]
    #         pixels_x_in_roi = x_coords[in_roi_mask]
            
    #         # Assign the corresponding label ID to these pixels in the output mask
    #         output_mask[pixels_y_in_roi, pixels_x_in_roi] = label_id
    #         print(f"Assigned {len(pixels_y_in_roi)} pixels to ROI {label_id}.")

        # Add the completed mask as a new layer to napari
        # self.viewer.add_labels(output_mask, name="Phasor_ROIs")
        # print("Mask generation complete.")
        
    def _on_clear_rois_clicked(self):
        """Clears all ROIs and ensures drawing mode is off."""
        if self.is_drawing_mode_active:
            self._set_drawing_mode(False) # Turn off drawing mode

        """Clears all defined ROIs and their selectors."""
        self.phasor_selectors.clear()
        self.phasor_rois.clear()
        # Redraw the plot without the ROI boxes
        self._plot_phasor(self.last_g_list, self.last_s_list, self.last_color_list)


    def update_data(self, intensity_image: np.ndarray, lifetime_image: np.ndarray):
        """Receives the raw data for the current slice from the main widget."""
        self.intensity_image = intensity_image
        self.lifetime_image = lifetime_image
        #print("PhasorPanel received new data.")

        # Update UI elements that depend on the data
        if self.intensity_image is not None:
            max_intensity = int(np.nanmax(self.intensity_image))
            self.thresh_min.setMaximum(max_intensity)
            self.thresh_max.setMaximum(max_intensity)
            self.thresh_max.setValue(max_intensity) # Set to max by default
            self.lbl_max.setText(f"Max Intensity: {max_intensity}")
            self.btn_threshold.setEnabled(True)
        else:
            self.btn_threshold.setEnabled(False)

    def _on_threshold_button_clicked(self):
        if self.intensity_image is None:
            QMessageBox.warning(self, "No Data", "Please generate a FLIM image first.")
            return
            
        is_visible = self.thresh_min.isVisible()
        self.thresh_min.setVisible(not is_visible)
        self.thresh_max.setVisible(not is_visible)
        self.lbl_min.setVisible(not is_visible)
        self.lbl_max.setVisible(not is_visible)
        
        if not is_visible: # If we just made them visible
            if "intensity_threshold_mask" not in self.viewer.layers:
                shape = self.intensity_image.shape[:2]
                mask = np.ones(shape, dtype=np.uint8)
                self.viewer.add_labels(mask, name="intensity_threshold_mask", opacity=0.4)
            self.viewer.layers["intensity_threshold_mask"].visible = True
            self._update_threshold_mask()
        else: # If we just hid them
            if "intensity_threshold_mask" in self.viewer.layers:
                self.viewer.layers["intensity_threshold_mask"].visible = False


    def _update_threshold_mask(self):
        if self.intensity_image is None or "intensity_threshold_mask" not in self.viewer.layers:
            return
            
        min_val = self.thresh_min.value()
        max_val = self.thresh_max.value()

        # Update labels
        self.lbl_min.setText(f"Min: {min_val}")
        self.lbl_max.setText(f"Max: {max_val}")

        # Ensure min is not greater than max
        if min_val > max_val:
            min_val = max_val

        mask = (self.intensity_image >= min_val) & (self.intensity_image <= max_val)
        self.viewer.layers["intensity_threshold_mask"].data = mask.astype(np.uint8)
    
    def _update_mask_combobox(self, event=None):
        self.mask_combobox.clear()
        label_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]
        if label_layers:
            self.mask_combobox.addItems(label_layers)
        if self.loaded_mask_data is not None:
            self.mask_combobox.addItem("Loaded from file")

    # def _on_load_mask_file_clicked(self):
    #     fname, _ = QFileDialog.getOpenFileName(self, "Open Mask File", "", "NumPy files (*.npy);;All Files (*)")
    #     if fname:
    #         try:
    #             self.loaded_mask_data = np.load(fname)
    #             if self.intensity_image is not None and self.loaded_mask_data.shape != self.intensity_image.shape:
    #                 QMessageBox.warning(self, "Shape Mismatch", "Mask shape does not match image shape.")
    #                 self.loaded_mask_data = None
    #                 return
    #             # Add a temporary item to the combobox
    #             if self.mask_combobox.findText("Loaded from file") == -1:
    #                 self.mask_combobox.addItem("Loaded from file")
    #             self.mask_combobox.setCurrentText("Loaded from file")
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"Failed to load mask: {e}")
    #             self.loaded_mask_data = None

    def _on_load_mask_file_clicked(self):
        """
        Loads a mask from a file, supporting .npy, .tif, and .png formats.
        """
        # 1. Define a more descriptive file filter for the dialog
        file_filter = (
            "All Supported Mask Files (*.npy *.tif *.tiff *.png);;"
            "NumPy Array (*.npy);;"
            "TIFF Image (*.tif *.tiff);;"
            "PNG Image (*.png);;"
            "All Files (*)"
        )
        fname, _ = QFileDialog.getOpenFileName(self, "Open Mask File", "", file_filter)

        if not fname:
            return  # User cancelled the dialog

        filepath = Path(fname)
        loaded_data = None

        try:
            # 2. Use the correct loader based on the file's extension
            extension = filepath.suffix.lower()

            if extension == '.npy':
                print(f"Loading NumPy file: {filepath.name}")
                loaded_data = np.load(filepath)
            elif extension in ['.tif', '.tiff', '.png']:
                print(f"Loading image file: {filepath.name}")
                # Use as_gray=True to ensure the mask is 2D, not 3D (RGB)
                loaded_data = imread(filepath, as_gray=True)
                # Image values might be float, but masks are usually integers
                if np.issubdtype(loaded_data.dtype, np.floating):
                    # Scale if it's a 0-1 float image, then cast to int
                    if loaded_data.max() <= 1.0:
                        loaded_data = (loaded_data * 255).astype(np.uint8)
                    else:
                        loaded_data = loaded_data.astype(np.uint8)

            else:
                QMessageBox.warning(
                    self, 
                    "Unsupported File Type",
                    f"File type '{extension}' is not supported. Please select a .npy, .tif, or .png file."
                )
                return

            # 3. Perform post-loading checks (shape mismatch)
            if self.intensity_image is not None and loaded_data.shape != self.intensity_image.shape:
                QMessageBox.warning(
                    self, 
                    "Shape Mismatch",
                    f"The loaded mask shape ({loaded_data.shape}) does not match the image shape ({self.intensity_image.shape})."
                )
                # Do not assign the data or update the UI
                return

            # 4. If all checks pass, commit the data and update the UI
            self.loaded_mask_data = loaded_data
            
            # Add or select the "Loaded from file" item in the combobox
            if self.mask_combobox.findText("Loaded from file") == -1:
                self.mask_combobox.addItem("Loaded from file")
            self.mask_combobox.setCurrentText("Loaded from file")
            
            print(f"Successfully loaded mask from {filepath.name}")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"An error occurred while loading the mask:\n\n{e}")
            self.loaded_mask_data = None  # Ensure data is cleared on error

    def _get_current_mask_and_layer(self):
        """ ### MODIFIED ###
        Helper to get the currently selected mask data AND its layer object.
        Returns a tuple (mask_data, mask_layer_object).
        """
        if self.intensity_image is None:
            return None, None

        mask_name = self.mask_combobox.currentText()
        if not mask_name:
            return None, None
        
        mask_data = None
        mask_layer = None

        if mask_name == "Loaded from file" and self.loaded_mask_data is not None:
            mask_data = self.loaded_mask_data
            # No layer object if loaded from file
        elif mask_name in self.viewer.layers:
            mask_layer = self.viewer.layers[mask_name]
            mask_data = mask_layer.data
        
        if mask_data is not None and mask_data.shape != self.intensity_image.shape[:2]:
            QMessageBox.warning(self, "Shape Mismatch", f"Mask '{mask_name}' shape does not match image.")
            return None, None
            
        return mask_data, mask_layer

    ### ====================================================================
    ### MODIFIED CORE METHODS
    ### ====================================================================
    
    def _on_compute_phasor_clicked(self):
        if self.lifetime_image is None:
            self._plot_phasor(message="No lifetime data available.")
            return

        mask, mask_layer = self._get_current_mask_and_layer()

        if mask is None:
            self._plot_phasor(message="No valid mask selected.")
            return

        self.pixel_phasor_data = None
        self.object_phasor_data = None
        self.last_computation_mask = mask.copy() # Store a copy of the mask we're using
        self.btn_add_roi.setEnabled(False)
        self.btn_generate_mask.setEnabled(False)
        self._on_clear_rois_clicked() # Clear old ROIs on re-compute

        harmonic = self.phasor_harmonic.value()
        freq = 80.0  # Placeholder frequency in MHz

        g_list, s_list, color_list = [], [], []

        # Get the color map from the layer if it exists, otherwise it's None
        label_to_color = {}
        default_color = (0.2, 0.8, 0.8, 1.0) # Default cyan for fallback

        colormap_obj = getattr(mask_layer, 'colormap', None)
        if colormap_obj:
            color_array = colormap_obj.colors
            unique_labels_in_mask = np.unique(mask)
            
            # Create the dictionary based on your working snippet
            label_to_color = {
                label: tuple(color_array[label])
                for label in unique_labels_in_mask
                if label != 0 and label < len(color_array) # Check against colormap length
            }
            
                   
        if self.per_pixel_radio.isChecked():
            pixels_y, pixels_x = np.where(mask > 0)
            
            # Reset and prepare the cache for back-projection
            self.pixel_phasor_data = {
                'g': np.zeros(len(pixels_y), dtype=float),
                's': np.zeros(len(pixels_y), dtype=float),
                'y': pixels_y,
                'x': pixels_x
            }

            for i, (y, x) in enumerate(zip(pixels_y, pixels_x)):
                tau = self.lifetime_image[y, x]
                g, s = calculate_phasor(freq, harmonic=harmonic)
                # Populate lists for plotting
                g_list.append(g)
                s_list.append(s)
                # Populate cache for remapping
                self.pixel_phasor_data['g'][i] = g
                self.pixel_phasor_data['s'][i] = s
                # ... (color list logic is the same) ...
                label_id = mask[y, x]
                # 2. Look up the color in the pre-built dictionary
                color = label_to_color.get(label_id, default_color)
                color_list.append(color)

            # Enable ROI tools ONLY for per-pixel plots
            self.btn_add_roi.setEnabled(True)
            self.btn_generate_mask.setEnabled(True)

        else: # Per Object mode

            labels = np.unique(mask)
            labels = labels[labels != 0] # Exclude background
            if len(labels) == 0:
                self._plot_phasor(message="Mask contains no objects.")
                return

            self.object_phasor_data = {'g': [], 's': [], 'label_id': []}

            for label_id in labels:
                # Calculate the mean lifetime for the entire region
                mean_tau_in_region = np.mean(self.lifetime_image[mask == label_id])
                g, s = calculate_phasor(freq, harmonic=harmonic)
                g_list.append(g)
                s_list.append(s)
                
                # Find the color for this object
                color = label_to_color.get(label_id, default_color)
                color_list.append(color)

                # Populate the object cache
                self.object_phasor_data['g'].append(g)
                self.object_phasor_data['s'].append(s)
                self.object_phasor_data['label_id'].append(label_id)

            # Convert lists to numpy arrays for efficiency
            for key in self.object_phasor_data:
                self.object_phasor_data[key] = np.array(self.object_phasor_data[key])

        # ... (rest of the method calls _plot_phasor) ...
        color_list = [tuple(round(float(x), 1) for x in rgba) for rgba in color_list]

        if g_list:
            self.btn_add_roi.setEnabled(True)
            self.btn_generate_mask.setEnabled(True)
            self._plot_phasor(g_list, s_list, color_list)
        else:
            self._plot_phasor(message="No data points found in mask.")

    def _plot_phasor(self, g_list=None, s_list=None, color_list=None, message=None):
        self.phasor_ax.clear()

        #: Apply theme based on checkbox state
        if self.dark_mode_checkbox.isChecked():
            dark_plot(self.phasor_ax, self.phasor_figure)
        else:
            light_plot(self.phasor_ax, self.phasor_figure)

        foreground_color = self.phasor_ax.xaxis.label.get_color()

        # 3. Draw the universal circle and text using this dynamic color.
        self.phasor_ax.plot(
            0.5 + 0.5 * np.cos(np.linspace(0, np.pi, 180)), # g-coordinates
            0.5 * np.sin(np.linspace(0, np.pi, 180)),       # s-coordinates
            color=foreground_color, 
            linestyle='--', 
            lw=1, 
            alpha=0.7
        )
        
        self.phasor_ax.set_xlim(0, 1)
        self.phasor_ax.set_ylim(0, 0.6)
        self.phasor_ax.set_aspect('equal', adjustable='box')

        if message:
            self.phasor_ax.text(0.5, 0.25, message, ha='center', va='center', color='w')
        # ... (most of this method is the same) ...

        # ### NEW/MODIFIED ###: Draw the ROI boxes after plotting data
        if g_list is not None and s_list is not None:
            point_size = 12  # Marker size
            if len(g_list) < 31:
                point_size = 30
            # Use scatter plot instead of hist2d
            # If no color list is provided, use a default color
            plot_colors = 'cyan' # Start with a default
            print(np.array(color_list))
            if color_list and len(color_list) == len(g_list):
                # Convert the list of tuples into a 2D NumPy array.
                # This is the robust format Matplotlib expects.
                plot_colors = np.array(color_list)
            print(plot_colors)

            self.phasor_ax.scatter(
                g_list, s_list, 
                c=plot_colors, 
                s=point_size,          # Marker size
                alpha=0.9,     # Transparency
                edgecolor='none' # Cleaner look for many points
            )
            self.phasor_ax.set_xlabel('G')
            self.phasor_ax.set_ylabel('S')
            self.phasor_ax.set_title('Phasor Plot')

            # ### NEW/MODIFIED ###: Cache the data that was just successfully plotted
            self.last_g_list = g_list
            self.last_s_list = s_list
            self.last_color_list = color_list

            # Now, draw the saved ROIs on top
            for roi in self.phasor_rois:
                xmin, xmax, ymin, ymax = roi['extents']
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=2,
                    edgecolor=roi['color'],
                    facecolor='none',
                    alpha=0.8
                )
                self.phasor_ax.add_patch(rect)
                self.phasor_ax.text(xmax, ymax, str(roi['id']), color=roi['color'],
                                    ha='left', va='bottom', weight='bold')
                
        else:
            # If we're just plotting an empty graph, clear the cache
            self.last_g_list = None
            self.last_s_list = None
            self.last_color_list = None
            self.phasor_ax.set_title('Phasor Plot')

        self.phasor_canvas.draw_idle()
