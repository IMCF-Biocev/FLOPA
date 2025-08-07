import numpy as np
import xarray as xr
from matplotlib import cm

# def create_flim_rgb_image(mean_photon_arrival_time, intensity, **kwargs):
#     lt_min, lt_max = kwargs.get('lt_min'), kwargs.get('lt_max')
#     int_min, int_max = kwargs.get('int_min'), kwargs.get('int_max')
#     if lt_min is None: lt_min = np.nanmin(mean_photon_arrival_time)
#     if lt_max is None: lt_max = np.nanmax(mean_photon_arrival_time)
#     if int_min is None: int_min = np.nanmin(intensity)
#     if int_max is None: int_max = np.nanmax(intensity)
#     if lt_max == lt_min: lt_max = lt_min + 1
#     if int_max == int_min: int_max = int_min + 1
#     lt_norm = np.clip((mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1)
#     from matplotlib import colormaps
#     cmap = colormaps.get_cmap(kwargs.get('colormap', 'viridis'))
#     lt_rgb = cmap(lt_norm)[..., :3]
#     intensity_norm = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)
#     return lt_rgb * intensity_norm[..., np.newaxis]

def create_flim_rgb_image(mean_photon_arrival_time, intensity, colormap=cm.rainbow, 
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


