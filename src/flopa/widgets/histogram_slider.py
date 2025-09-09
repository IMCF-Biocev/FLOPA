# flopa/widgets/histogram_slider.py

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from qtpy.QtCore import Qt, Signal
from superqt import QDoubleRangeSlider
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

class HistogramSlider(QWidget):
    """A compound widget with a histogram, a QDoubleRangeSlider, and value readouts."""
    
    valueChanged = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.data = None
        self.colormap = cm.gray
        self._is_updating = False

        layout = QVBoxLayout(self); layout.setContentsMargins(0, 5, 0, 5); layout.setSpacing(2)

        self.figure = Figure(dpi=75); self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedHeight(130)
        #self.canvas.setFixedWidth(500)
        
        self.ax = self.figure.add_subplot(111)
        
        self.slider = QDoubleRangeSlider(Qt.Horizontal)
        self.slider.setRange(0.0, 1.0); self.slider.setValue((0.0, 1.0)); self.slider.setFixedHeight(20)
        #self.slider.setFixedWidth(400)
        
        self.min_label = QLabel("0.00"); self.max_label = QLabel("1.00")
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.min_label); slider_layout.addWidget(self.slider); slider_layout.addWidget(self.max_label)

        layout.addWidget(self.canvas)
        layout.addLayout(slider_layout)
        
        self._initialize_plot()
        
        self.slider.valueChanged.connect(self._on_slider_moved)

    def _initialize_plot(self):
        self.ax.clear(); self.ax.set_xlim(0, 1)
        self.ax.text(0.5, 0.5, "No Data", color='gray', ha='center', va='center', transform=self.ax.transAxes)
        self._format_ax(); self.canvas.draw()

    def _format_ax(self):
        self.figure.patch.set_alpha(0); self.ax.patch.set_alpha(0)
        self.ax.spines['top'].set_visible(False); self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False); self.ax.get_yaxis().set_visible(False)
        self.ax.tick_params(axis='x', colors='gray', labelsize=9)
        # self.figure.tight_layout(pad=0.82, h_pad=0.1)
        self.figure.subplots_adjust(
            left=0.08,    # Increase left padding (e.g., from 0.05 to 0.1)
            right=0.95,   # Decrease right padding (e.g., from 0.95 to 0.9)
            bottom=0.6,  # Increase bottom padding to make space for x-ticks
            top=0.9     # Decrease top padding
        )

    def _on_slider_moved(self):
        if self._is_updating: return
        self._update_visuals()
        self.valueChanged.emit(self.slider.value())

    def set_colormap(self, cmap):
        self.colormap = cmap
        self._update_visuals()

    def update_data(self, data_slice: np.ndarray):
        if data_slice is None or data_slice.size == 0: return
        valid_data = data_slice[np.isfinite(data_slice)]
        if valid_data.size == 0: return
        self.data = valid_data
        min_val, max_val = np.min(self.data), np.max(self.data)

        self._is_updating = True
        self.slider.setRange(min_val, max_val)
        low, high = np.percentile(self.data, [5, 95])
        if not np.isfinite([low, high]).all():
            low, high = 0, 1

        if low >= high:
            # collapse case: enforce a minimal width
            low, high = min_val, min_val + 1

        self.slider.setRange(min_val, max_val if max_val > min_val else min_val + 1)
        self.slider.setValue((low, high))

        self._is_updating = False
        
        # When data changes, trigger a full redraw including the histogram bars
        self._update_visuals(redraw_hist=True)

    def _update_visuals(self, redraw_hist=False):
        """
        Updates all visuals on the canvas. If redraw_hist is True, it also
        re-calculates the histogram bars.
        """
        if self.data is None and not redraw_hist: return
        
        # --- THE DEFINITIVE FIX ---
        # Instead of removing artists, we clear the axes and redraw everything.
        # This is the most robust and error-free method.
        self.ax.clear()
        # ------------------------

        # If we have data, redraw the histogram
        if self.data is not None:
            # Get the full range from the slider for the histogram bins
            full_min, full_max = self.slider.minimum(), self.slider.maximum()
            self.ax.hist(self.data, bins=50, color="#FFFFFF44", log=True, range=(full_min, full_max))
            self.ax.set_xlim(full_min, full_max)
            
            # Now draw the lines and gradient
            min_val, max_val = self.slider.value()
            self.ax.axvline(min_val, color='cyan', linestyle='--', linewidth=1)
            self.ax.axvline(max_val, color='cyan', linestyle='--', linewidth=1)
            
            grad = np.linspace(0, 1, 256).reshape(1, 256)
            self.ax.imshow(
                grad, aspect='auto', extent=[min_val, max_val, *self.ax.get_ylim()],
                cmap=self.colormap, origin='lower', zorder=-1
            )
            
            # Also update the labels here
            self.min_label.setText(f"{min_val:.2f}")
            self.max_label.setText(f"{max_val:.2f}")

        self._format_ax()
        self.canvas.draw_idle()

    def value(self):
        """
        Public method to get the current slider value tuple (min, max).
        This provides a clean public API for the parent widget.
        """
        return self.slider.value()