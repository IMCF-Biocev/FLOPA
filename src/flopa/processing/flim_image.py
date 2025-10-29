import numpy as np
from matplotlib import cm
import warnings

try:
    from numba import njit
    _NUMBA_AVAILABLE = True

    @njit(cache=True)
    def _create_flim_image_numba_core(mean_photon_arrival_time, intensity, cmap_lut, lt_min, lt_max, int_min, int_max):
        """ Numba-accelerated core logic for creating the FLIM image. """
        lt_range = lt_max - lt_min
        int_range = int_max - int_min
        height, width = mean_photon_arrival_time.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                lifetime_val = mean_photon_arrival_time[i, j]
                # Handle NaN pixels by leaving them black
                if np.isnan(lifetime_val):
                    continue

                norm_lt = (lifetime_val - lt_min) / lt_range
                
                if norm_lt > 1.0:
                    norm_lt = 1.0
                elif norm_lt < 0.0:
                    norm_lt = 0.0

                cmap_index = int(norm_lt * 255)
                
                color_r = cmap_lut[cmap_index, 0]
                color_g = cmap_lut[cmap_index, 1]
                color_b = cmap_lut[cmap_index, 2]

                intensity_val = intensity[i, j]
                brightness = (intensity_val - int_min) / int_range

                if brightness > 1.0:
                    brightness = 1.0
                elif brightness < 0.0:
                    brightness = 0.0
                
                rgb_image[i, j, 0] = color_r * brightness
                rgb_image[i, j, 1] = color_g * brightness
                rgb_image[i, j, 2] = color_b * brightness
        return rgb_image

except ImportError:
    _NUMBA_AVAILABLE = False
    print(
        "Numba not found. Using slower NumPy fallback for FLIM image creation. "
        "For a significant performance improvement, please `pip install numba`."
    )


def _create_flim_image_numpy_core(mean_photon_arrival_time, intensity, colormap, lt_min, lt_max, int_min, int_max):
    """ Pure NumPy version of the core FLIM image creation logic. """
    LT_normalized = np.clip((mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1)
    LT_rgb = colormap(LT_normalized)[..., :3]
    intensity_normalized = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)
    return LT_rgb * intensity_normalized[..., np.newaxis]


def create_FLIM_image(mean_photon_arrival_time, intensity, colormap=cm.rainbow, 
                      lt_min=None, lt_max=None,
                      int_min=None, int_max=None):
    """
    Create an RGB FLIM image from lifetime and intensity data.
    This function automatically uses a Numba-accelerated backend if available.
    """
    if mean_photon_arrival_time.shape != intensity.shape:
        raise ValueError("Lifetime and intensity arrays must have the same shape")

    _lt_min, _lt_max = lt_min, lt_max
    _int_min, _int_max = int_min, int_max

    if _lt_min is None or _lt_max is None:
        _lt_min = np.nanmin(mean_photon_arrival_time)
        _lt_max = np.nanmax(mean_photon_arrival_time)
    
    if _lt_max == _lt_min:
        warnings.warn(f"Lifetime min and max are equal ({_lt_min}). Adding epsilon to prevent division by zero.")
        _lt_max += 1e-9

    if _int_min is None or _int_max is None:
        _int_min = np.nanmin(intensity)
        _int_max = np.nanmax(intensity)
    
    if _int_max == _int_min:
        warnings.warn(f"Intensity min and max are equal ({_int_min}). Adding epsilon to prevent division by zero.")
        _int_max += 1e-9
    
    if _NUMBA_AVAILABLE:
        cmap_lut = colormap(np.linspace(0, 1, 256))
        
        return _create_flim_image_numba_core(
            mean_photon_arrival_time.astype(np.float32),
            intensity.astype(np.float32),
            cmap_lut,
            float(_lt_min), float(_lt_max),
            float(_int_min), float(_int_max)
        )
    else:
        return _create_flim_image_numpy_core(
            mean_photon_arrival_time,
            intensity,
            colormap,
            _lt_min, _lt_max,
            _int_min, _int_max
        ).astype(np.float32)
    