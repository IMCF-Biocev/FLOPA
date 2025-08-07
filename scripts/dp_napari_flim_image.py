import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
import napari
from magicgui import magicgui
import numpy as np
from magicgui.widgets import Container, SpinBox, FloatRangeSlider




def create_FLIM_image(mean_photon_arrival_time, intensity, colormap=cm.rainbow, 
                      lt_min=None, lt_max=None,
                      int_min=None, int_max=None):
    """
    Create an RGB FLIM image from lifetime and intensity data.

    Parameters:
    - mean_photon_arrival_time: 2D numpy array of lifetimes
    - intensity: 2D numpy array of photon counts
    - colormap: Matplotlib colormap (default: cm.rainbow)
    - lt_min: optional float, min lifetime for normalization
    - lt_max: optional float, max lifetime for normalization

    Returns:
    - FLIM_image: 3D numpy array (H, W, 3) with RGB values
    """

    # Validate shape
    if mean_photon_arrival_time.shape != intensity.shape:
        raise ValueError("Lifetime and intensity arrays must have the same shape")

    # Lifetime normalization
    if lt_min is None or lt_max is None:
        lt_min = np.nanmin(mean_photon_arrival_time)
        lt_max = np.nanmax(mean_photon_arrival_time)
    if lt_max == lt_min:
        raise ValueError(f"lt_max and lt_min must differ — got {lt_min}")

    # Intensity normalization with adjustable contrast
    if int_min is None or int_max is None:
        int_min = np.nanmin(intensity)
        int_max = np.nanmax(intensity)
    if int_max == int_min:
        raise ValueError("int_max and int_min must differ")

    LT_normalized = np.clip((mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1)
    LT_rgb = colormap(LT_normalized)[..., :3]  # Drop alpha
    intensity_normalized = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)

    return LT_rgb * intensity_normalized[..., np.newaxis]


if __name__ == "__main__":

    result = xr.open_dataset(r"./test_data/result.h5")
    intensity = np.array(result.photon_count.transpose("frame","sequence","channel","line","pixel"), dtype=int)
    lifetime = np.array(result.mean_photon_arrival_time.transpose("frame","sequence","channel","line","pixel"), dtype=float) * 5e-3

    n_frames = intensity.shape[0]
    n_sequences = intensity.shape[1]
    n_channels = intensity.shape[2]
    int_max = np.nanmax(intensity)
    

    viewer = napari.Viewer()
    frame_selector = SpinBox(name="Frame", min=0, max=n_frames - 1, step=1)
    sequence_selector = SpinBox(name="Sequence", min=0, max=n_sequences - 1, step=1)
    channel_selector = SpinBox(name="Channel", min=0, max=n_channels - 1, step=1)
    lt_range_slider = FloatRangeSlider(name="Lifetime", min=0.0, max=25.0, step=0.1)
    int_range_slider = FloatRangeSlider(name="Intensity", min=0.0, max=int_max, step=1)




    def update_flimgui():
        frame = frame_selector.value
        sequence = sequence_selector.value
        channel = channel_selector.value
        lt_min, lt_max = lt_range_slider.value
        int_min, int_max = int_range_slider.value
        

        lt_img = lifetime[frame, sequence, channel, :, :]
        int_img = intensity[frame, sequence, channel, :, :]

        flim_rgb = create_FLIM_image(
            mean_photon_arrival_time=lt_img, intensity=int_img,
            lt_min=lt_min, lt_max=lt_max,
            int_min=int_min, int_max=int_max
        )
        
        if 'FLIM' in viewer.layers:
            viewer.layers['FLIM'].data = flim_rgb
        else:
            viewer.add_image(flim_rgb, name='FLIM', rgb=True)


    # Connect all widgets to trigger update
    frame_selector.changed.connect(update_flimgui)
    sequence_selector.changed.connect(update_flimgui)
    channel_selector.changed.connect(update_flimgui)
    lt_range_slider.changed.connect(update_flimgui)
    int_range_slider.changed.connect(update_flimgui)

    
    # Make a container and add to napari
    control_panel = Container(
        widgets=[
            frame_selector, 
            sequence_selector,
            channel_selector, 
            lt_range_slider, 
            int_range_slider
                 ])
    
    viewer.window.add_dock_widget(control_panel, area='right')

    # Call once to initialize
    frame_selector.value = 0
    sequence_selector.value = 0
    channel_selector.value = 0
    lt_range_slider.value = (0,25)
    int_range_slider.value = (0,int_max)
    update_flimgui()




 
    # @magicgui(
    #     frame={"label": "Frame", "widget_type": "SpinBox", "min": 0, "max": n_frames - 1},
    #     sequence={"label": "Frame", "widget_type": "SpinBox", "min": 0, "max": n_sequences - 1},
    #     channel={"label": "Channel", "widget_type": "SpinBox", "min": 0, "max": n_channels - 1},
    #     lt_range={"label": "Lifetime Range", "widget_type": "FloatRangeSlider"},
    #     int_range={"label": "Intensity Range", "widget_type": "FloatRangeSlider"},
    #     auto_call=True,
    # )
    # def update_flimgui(
    #     frame: int = 0,
    #     sequence: int = 0,
    #     channel: int = 0,
    #     lt_range=(0.0, 1.0),
    #     int_range=(0.0, 1.0),
    # ):
    #     lt_min, lt_max = lt_range
    #     int_min, int_max = int_range
        

    #     lt_img = lifetime[frame, sequence, channel, :, :]
    #     int_img = intensity[frame, sequence, channel, :, :]

    #     flim_rgb = create_FLIM_image(
    #         lt_img, int_img,
    #         lt_min=lt_min, lt_max=lt_max,
    #         int_min=int_min, int_max=int_max
    #     )
        
    #     if 'FLIM' in viewer.layers:
    #         viewer.layers['FLIM'].data = flim_rgb
    #     else:
    #         viewer.add_image(flim_rgb, name='FLIM', rgb=True)

    # update_flimgui.frame.value = 0
    # update_flimgui.frame.native.setFixedWidth(80)
    # update_flimgui.sequence.value = 0
    # update_flimgui.sequence.native.setFixedWidth(80)
    # update_flimgui.channel.value = 0
    # update_flimgui.channel.native.setFixedWidth(80)

    # viewer.window.add_dock_widget(update_flimgui, area='bottom')
    # lt_max = float(2 * np.nanmedian(lifetime[0,0,0,:,:]))
    # int_max = float(np.nanmax(intensity))

    # update_flimgui.lt_range.min = 0
    # update_flimgui.lt_range.max = 25
    # update_flimgui.lt_range.value = (0, lt_max)

    # update_flimgui.int_range.min = 0
    # update_flimgui.int_range.max = int_max
    # update_flimgui.int_range.value = (0, int_max)



    napari.run()
