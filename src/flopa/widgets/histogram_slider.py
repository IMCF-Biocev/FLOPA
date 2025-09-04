# flopa/widgets/histogram_slider.py

from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from magicgui.widgets import FloatRangeSlider
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

class HistogramSlider(QWidget):
    """A compound widget with an interactive histogram plot and a FloatRangeSlider."""
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.data = None
        self.colormap = cm.gray
        
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 5, 0, 5); layout.setSpacing(2)

        self.figure = Figure(dpi=75)
        self.canvas = FigureCanvas(self.figure)
        #self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFixedHeight(50)
        self.ax = self.figure.add_subplot(111)
        
        self.slider = FloatRangeSlider(name=name, min=0.0, max=1.0)
        self.slider.changed.connect(self._update_visuals)
        #self.slider.label = "" 
        #self.slider.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        #self.slider.native.setFixedHeight(40)

        layout.addWidget(self.canvas)
        layout.addWidget(self.slider.native)
        
        self._format_ax() # Initial formatting

    def _format_ax(self):
        """Helper to apply standard formatting to the axes."""
        self.figure.patch.set_alpha(0); self.ax.patch.set_alpha(0)
        self.ax.spines['top'].set_visible(False); self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False); self.ax.get_yaxis().set_visible(False)
        self.ax.tick_params(axis='x', colors='gray', labelsize=7)
        self.figure.tight_layout(pad=0.05)

    def set_colormap(self, cmap):
        """Sets the colormap and triggers a redraw."""
        self.colormap = cmap
        self._update_visuals()

    def update_data(self, data_slice: np.ndarray):
        """Heavy update: Redraws histogram and updates slider range."""
        if data_slice is None or data_slice.size == 0: return
        valid_data = data_slice[np.isfinite(data_slice)]
        if valid_data.size == 0: return
        self.data = valid_data

        min_val, max_val = np.min(self.data), np.max(self.data)
        if min_val >= max_val: max_val = min_val + 1

        with self.slider.changed.blocked():
            self.slider.min = min_val; self.slider.max = max_val
            p5, p95 = np.percentile(self.data, [5, 95])
            self.slider.value = (max(min_val, p5), min(max_val, p95))
        
        self._update_visuals(redraw_hist=True) # Force a full redraw

    def _update_visuals(self, redraw_hist=False):
        """
        Lightweight update for visuals. If redraw_hist is True, it also
        re-calculates the histogram bars.
        """
        if self.data is None: return

        # Store current limits to reapply them
        current_xlim = self.ax.get_xlim()
        
        # --- ROBUST REDRAW LOGIC ---
        # Instead of removing artists, we clear and redraw what's needed.
        if redraw_hist:
            self.ax.clear()
            self.ax.hist(self.data, bins=50, color='#FFFFFF80', log=True, range=self.slider.range)
        else:
            # If not redrawing the whole thing, just clear lines and images
            [line.remove() for line in self.ax.lines]
            [img.remove() for img in self.ax.images]
        
        min_val, max_val = self.slider.value
        
        self.ax.axvline(min_val, color='cyan', linestyle='--', linewidth=1)
        self.ax.axvline(max_val, color='cyan', linestyle='--', linewidth=1)
        
        grad = np.linspace(0, 1, 256).reshape(1, 256)
        self.ax.imshow(
            grad, aspect='auto', extent=[min_val, max_val, *self.ax.get_ylim()],
            cmap=self.colormap, origin='lower', zorder=-1
        )
        
        if redraw_hist:
            self.ax.set_xlim(self.slider.range)
        else:
            self.ax.set_xlim(current_xlim)

        self._format_ax()
        self.canvas.draw_idle()