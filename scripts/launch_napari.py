import napari 
import numpy as np
from flopa.widgets.napari_flim_widget import FlimWidget

def main():
    viewer = napari.Viewer()
    flim_widget = FlimWidget(viewer)
    viewer.window.add_dock_widget(flim_widget, name="FLIM Analyzer")

    # Example of adding a dummy labels layer for testing
    # dummy_labels = np.zeros((512, 512), dtype=int)
    # dummy_labels[20:50, 20:50] = 1
    # dummy_labels[60:100, 70:310] = 2
    # dummy_labels[10:30, 80:120] = 4
    # viewer.add_labels(dummy_labels, name='Sample Mask')

    napari.run()

if __name__ == "__main__":
    main()