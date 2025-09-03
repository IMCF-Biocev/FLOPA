# flopa/widgets/decay_panel.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QCheckBox, QHBoxLayout, QMessageBox
)
from qtpy.QtCore import Slot
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .utils.style import dark_plot, light_plot

class DecayPanel(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dataset = None
        self.last_plot_data = None
        self._init_ui()
        self._plot_decay(message="No data loaded.")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        controls_group = QGroupBox("Plot Options")
        options_layout = QHBoxLayout(controls_group)
        self.dark_mode_check = QCheckBox("Use Dark Theme"); self.dark_mode_check.setChecked(True)
        self.dark_mode_check.toggled.connect(self._on_theme_changed)
        options_layout.addWidget(self.dark_mode_check); options_layout.addStretch()
        layout.addWidget(controls_group)
        plot_group = QGroupBox("Decay Curve")
        plot_layout = QVBoxLayout(plot_group)
        self.decay_figure = Figure(figsize=(5, 4)); self.decay_canvas = FigureCanvas(self.decay_figure)
        self.decay_ax = self.decay_figure.add_subplot(111)
        plot_layout.addWidget(self.decay_canvas)
        layout.addWidget(plot_group); layout.addStretch()

    @Slot(xr.Dataset)
    def update_data(self, dataset: xr.Dataset):
        self.dataset = dataset
        if "tcspc_histogram" not in self.dataset.data_vars: self.clear_plot()

    @Slot(dict)
    def on_slice_changed(self, data_package: dict):
        if not self.dataset or "tcspc_histogram" not in self.dataset.data_vars: return
        
        histogram = data_package.get('tcspc_histogram')
        selectors = data_package.get('selectors')
        if histogram is None or selectors is None: return

        selection_dict, dims_to_sum = {}, []
        if 'frame' in histogram.dims and not selectors['sum_frames'].isChecked(): selection_dict['frame'] = selectors['frame'].value()
        if 'channel' in histogram.dims and not selectors['sum_channels'].isChecked(): selection_dict['channel'] = selectors['channel'].value()
        if 'frame' in histogram.dims and selectors['sum_frames'].isChecked(): dims_to_sum.append('frame')
        if 'channel' in histogram.dims and selectors['sum_channels'].isChecked(): dims_to_sum.append('channel')
            
        sliced_hist = histogram.isel(**selection_dict)
        final_decay_data = sliced_hist.sum(dim=dims_to_sum) if dims_to_sum else sliced_hist

        instrument_params = self.dataset.attrs.get('instrument_params', {})
        tcspc_res_s = instrument_params.get('MeasDesc_Resolution', 5e-12)
        time_axis_ns = np.arange(histogram.sizes['tcspc_channel']) * tcspc_res_s * 1e9

        if 'channel' in final_decay_data.dims:
            decay_curves, channel_labels = final_decay_data.values, final_decay_data.channel.values
        else:
            decay_curves = [final_decay_data.values]
            label = "Summed" if 'channel' in dims_to_sum else final_decay_data.channel.values.item()
            channel_labels = [label]
        
        self._plot_decay(time_axis_ns, decay_curves, channel_labels)

    def clear_plot(self):
        self.last_plot_data = None
        self._plot_decay(message="TCSPC Histogram not available.")
        
    def _on_theme_changed(self):
        if self.last_plot_data: self._plot_decay(**self.last_plot_data)


    def _plot_decay(self, time_axis=None, decay_curves=None, channel_labels=None, message=None):
        
        self.last_plot_data = {
            "time_axis": time_axis,
            "decay_curves": decay_curves,
            "channel_labels": channel_labels,
            "message": message
        }
        # ----------------------

        self.decay_ax.clear()
        
        if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure); text_color = 'white'
        else: light_plot(self.decay_ax, self.decay_figure); text_color = 'black'
        if message: self.decay_ax.text(0.5, 0.5, message, color=text_color, ha='center', va='center', transform=self.decay_ax.transAxes)
        if time_axis is not None and decay_curves is not None:
            for i, curve in enumerate(decay_curves):
                self.decay_ax.semilogy(time_axis, curve, label=f'Channel {channel_labels[i]}')
            if self.decay_ax.get_legend_handles_labels()[0]: self.decay_ax.legend()
            valid_points = np.concatenate([c for c in decay_curves if c.ndim == 1]);
            if valid_points.any(): self.decay_ax.set_ylim(bottom=max(0.5, valid_points[valid_points > 0].min() * 0.5))
        self.decay_ax.set_title("Fluorescence Decay"); self.decay_ax.set_xlabel("Time (ns)"); self.decay_ax.set_ylabel("Photon Count (log scale)")
        if self.decay_ax.get_legend():
            if self.dark_mode_check.isChecked(): dark_plot(self.decay_ax, self.decay_figure)
            else: light_plot(self.decay_ax, self.decay_figure)
        self.decay_canvas.draw_idle()