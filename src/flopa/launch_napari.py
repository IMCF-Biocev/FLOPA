# flopa/launch_napari.py

import napari

from flopa.widgets.napari_flim_widget import FlimWidget
from flopa.widgets.flim_view_panel import FlimViewPanel


def main():
    """
    Initializes and launches the napari viewer with the FLOPA widgets.
    """
    viewer = napari.Viewer()

    flim_view_panel = FlimViewPanel(viewer)
    main_flim_widget = FlimWidget(viewer, flim_view_panel)


    viewer.window.add_dock_widget(
        main_flim_widget, 
        name="FLIM Analysis", 
        area="right"
    )
    
    viewer.window.add_dock_widget(
        flim_view_panel, 
        name="FLIM View Controls", 
        area="bottom",
    )


    napari.run()


if __name__ == "__main__":
    main()