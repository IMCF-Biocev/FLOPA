import numpy as np
import xarray as xr

from .reader import TTTRReader
from .decoder import T3OverflowCorrector
from .reconstructor import ScanConfig
from .reconstructor import ImageReconstructor

from matplotlib import cm
from matplotlib.axes import Axes

import copy

from typing import Optional, Dict

from scipy.optimize import curve_fit
from scipy.signal import convolve2d


# --- Reconstruction helpers ---

def _gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

def _fit_gaussian_peak(shifts: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        p0 = [scores.max() - scores.min(), shifts[np.argmax(scores)], 0.1* (shifts.max() - shifts.min()), scores.min()]
        popt, _ = curve_fit(_gaussian, shifts, scores, p0=p0)
        fit = _gaussian(shifts,*popt)
        return float(popt[1]), fit  # mu = estimated phase shift
        
    except RuntimeError:
        return None


def estimate_tcspc_bins(header_tags: dict, buffer: int = 10) -> int:
    rep_rate = header_tags.get("TTResult_SyncRate", 40e6)  # Hz
    resolution = header_tags.get("MeasDesc_Resolution", 5e-12)  # s
    bins = int(np.ceil(1 / resolution / rep_rate)) + buffer
    return bins




def estimate_bidirectional_shift(reader: TTTRReader, 
                                 config: ScanConfig,
                                 wrap: int = 1024,
                                 max_shift: float = .01, 
                                 steps: int = 11,
                                 chunk_length: int = 500_000, 
                                 verbose: bool = True) -> tuple[float, np.ndarray]:
    """
    Estimate the optimal phase shift (as fraction of line duration) for backward lines 
    in bidirectional scanning.
    
    Args:
        reader: TTTRReader instance
        config: A ScanConfig instance.
        max_shift: Maximum shift to try (±max_shift).
        steps: Number of shift steps to test.
        chunk_length: Number of events to read. Try increasing it when reconstruction fails, perhaps the reconstruction is feature-less
        verbose: Whether to print progress.

    Returns:
        Tuple of best phase shift (float) in units of line duration (e.g., -0.015) and numpy array of shifts, correlation scores, and fit for inspection.
    """
    
    

    if not config.bidirectional:
        raise ValueError("ScanConfig must have bidirectional=True to estimate phase shift.")

    if verbose:
        print("Estimating bidirectional phase shift...")

    base_config = copy.deepcopy(config)
    base_config.frames = 1
    base_config.line_accumulations = (1,)
    base_config.lines = config.lines * config.line_accumulations[0]
    base_config._total_accumulations = 1

    line_bin = config.line_accumulations[0] * 2

    shifts = np.linspace(config.bidirectional_phase_shift-max_shift,
                         config.bidirectional_phase_shift+max_shift, steps)
    scores = np.zeros_like(shifts)
    corrector = T3OverflowCorrector(wraparound=wrap)
    
    for i, shift in enumerate(shifts):
        
        # Clone config and apply shift
        test_config = copy.deepcopy(base_config)
        test_config.bidirectional_phase_shift = shift
        recon = ImageReconstructor(config=test_config,outputs=["photon_count"])
        chunk = reader.read(count=chunk_length)
        corrected_chunk = corrector.correct(chunk)
        recon.update(corrected_chunk)
        pc = xr.DataArray(
            data=recon.photon_count.astype(np.float32),
            coords={
                "frame" : 1,
                "sequence": 1,
                "line": np.arange(test_config.lines),
                "pixel": np.arange(test_config.pixels),
                "detector": np.arange(test_config.max_detector)
            }
        )


        # pc = xr.DataArray(recon.photon_count.astype(np.float32))
        # pc = pc.rename({"dim_0" : "frame",
        #     "dim_1" : "line",
        #     "dim_2" : "pixel",
        #     "dim_3" : "channel"})
        pc = pc.sum(dim = 'detector')
        pc = pc.isel(frame = 0,sequence=0)
        forward = pc[::2, :]
        backward = pc[1::2, :]
        forward = forward.coarsen(line = line_bin).sum()
        backward = backward.coarsen(line = line_bin).sum()
        
        # Ensure same number of lines
        num_pairs = min(forward.sizes['line'], backward.sizes['line'])
        fwd = forward.isel(line=slice(0, num_pairs))
        bwd = backward.isel(line=slice(0, num_pairs))

        # Mask out zero rows (xarray preserves dims, so we need numpy for row-wise masking)
        fwd_vals = fwd.values
        bwd_vals = bwd.values

        mask = ~((fwd_vals == 0).all(axis=1) | (bwd_vals == 0).all(axis=1))
        fwd_vals = fwd_vals[mask]
        bwd_vals = bwd_vals[mask]

        # Subtract mean along each line (axis=1)
        fwd_vals -= fwd_vals.mean(axis=1, keepdims=True)
        bwd_vals -= bwd_vals.mean(axis=1, keepdims=True)

        # Compute dot products (correlation at lag zero)
        score = np.sum(fwd_vals * bwd_vals)

        scores[i] = score

        if verbose:
            print(f"Shift {shift:.4f} → score {score:.2f}")

    # best_shift = shifts[np.argmax(scores)]

    best_shift, fit = _fit_gaussian_peak(shifts, scores)

    if verbose:
        print(f"Best estimated shift: {best_shift:.5f}")

    return best_shift, np.stack((shifts,scores,fit))

# --- Image functions ---

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


# def smooth_weighted(array,count,size: int = 3):
#       # array - 2D array to be smoothed, phasor coordinates or lifetime
#       # count - used for weighting, must be of same size as array


#       kernel = np.ones((size, size), dtype=np.float32)

#       # Set invalid phasors to 0
#       valid = np.isfinite(array) & (count > 0)
#       array_weighted = np.zeros_like(array, dtype=np.float32)
#       array_weighted[valid] = array[valid] * count[valid]
#       count_weighted = np.zeros_like(count, dtype=np.float32)
#       count_weighted[valid] = count[valid]

#       # Convolve
#       num = convolve2d(array_weighted, kernel, mode='same')
#       den = convolve2d(count_weighted, kernel, mode='same')

#       # Normalize
#       array_smoothed = np.full_like(array, np.nan)
#       mask = den > 0
#       array_smoothed[mask] = num[mask] / den[mask]

#       return array_smoothed, den

def smooth_weighted(array, count, size: int = 3):
    """
    Apply weighted 2D smoothing to an array using a square convolution kernel.

    The function smooths a 2D array (e.g., phasor coordinates or lifetime data) 
    with a uniform kernel while weighting values by a corresponding count matrix. 
    Invalid entries (NaN, Inf, or locations with nonpositive count) are excluded 
    from the smoothing.

    Parameters
    ----------
    array : np.ndarray
        2D array of values to be smoothed. Must have the same shape as `count`.
    count : np.ndarray
        2D array of weights (e.g., photon counts). Must be the same shape as `array`.
    size : int, optional
        Size of the square convolution kernel. Must be a positive integer. 
        Default is 3 (3×3 kernel).

    Returns
    -------
    array_smoothed : np.ndarray
        2D array of the same shape as `array`, containing smoothed values. 
        Entries where the kernel had no valid contributions are set to NaN.
    count_smoothed : np.ndarray
        2D array of the same shape, representing the weighted denominator 
        (effective counts) after convolution.

    Raises
    ------
    AssertionError
        If `array` and `count` are not the same shape.
    ValueError
        If `size` is not a positive integer or if `array`/`count` are not 2D.

    Notes
    -----
    - This function uses a uniform kernel (`np.ones`) for convolution. 
    - Normalization ensures the result is a weighted average rather than 
      an unscaled sum.

    
    """

    # Sanity checks
    if array.ndim != 2 or count.ndim != 2:
        raise ValueError("array and count must both be 2D arrays")
    assert array.shape == count.shape, "array and count must have the same shape"
    if not (isinstance(size, int) and size > 0):
        raise ValueError("size must be a positive integer")

    kernel = np.ones((size, size), dtype=np.float32)

    # Mask invalid entries
    valid = np.isfinite(array) & (count > 0)

    # Weighted arrays using np.where (avoids full zero arrays)
    array_weighted = np.where(valid, array * count, 0).astype(np.float32)
    count_valid = np.where(valid, count, 0).astype(np.float32)

    # Convolve
    num = convolve2d(array_weighted, kernel, mode='same')
    count_smoothed = convolve2d(count_valid, kernel, mode='same')

    # Normalize
    array_smoothed = np.full_like(array, np.nan, dtype=np.float32)
    mask = count_smoothed > 0
    array_smoothed[mask] = num[mask] / count_smoothed[mask]
    count_smoothed = np.array(count_smoothed)
    count_smoothed = count_smoothed.astype(np.uint32)

    return array_smoothed, count_smoothed



# --- Marker Helpers ---

def marker_events(events: np.ndarray) -> np.ndarray:
    """Return only events where channel == 63 and special != 15 (non-overflow markers)."""
    return events[(events['channel'] < 63) & (events['special'] != 0)]

def get_marker_distribution(events: np.ndarray) -> Dict[int, int]:
    """Returns a count of each special marker code."""
    mask = (events['channel'] < 63) & (events['special'] != 0)
    markers = events['channel'][mask]
    unique, counts = np.unique(markers, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


# --- Phasor functions ---


# def smooth_phasor(phasor,count,size: int = 3):
#       kernel = np.ones((size, size), dtype=np.float32)

#       # Set invalid phasors to 0
#       valid = np.isfinite(phasor) & (count > 0)
#       phasor_weighted = np.zeros_like(phasor, dtype=np.complex64)
#       phasor_weighted[valid] = phasor[valid] * count[valid]
#       count_weighted = np.zeros_like(count, dtype=np.float32)
#       count_weighted[valid] = count[valid]

#       # Convolve
#       num = convolve2d(phasor_weighted.real, kernel, mode='same') + \
#             1j * convolve2d(phasor_weighted.imag, kernel, mode='same')
#       den = convolve2d(count_weighted, kernel, mode='same')

#       # Normalize
#       phasor_smoothed = np.full_like(phasor, np.nan + 1j * np.nan)
#       mask = den > 0
#       phasor_smoothed[mask] = num[mask] / den[mask]

#       return phasor_smoothed


def get_phasor_from_decay(
    decay: np.ndarray,
    tcspc_resolution: float,
    sync_rate: float,
) -> complex:
    """
    Photon‑weighted complex phasor from a TCSPC decay.

    Parameters
    ----------
    decay : 1‑D np.ndarray
        Photon counts per TCSPC channel.
    tcspc_resolution_ns : float
        Width of one TCSPC channel in seconds.
    sync_rate : float
        Laser repetition rate in Hz.

    Returns
    -------
    complex
        Phasor Φ = g + i·s.
        Returns nan+1j*nan if the decay is empty.
    """
    if decay.ndim != 1:
        raise ValueError("`decay` must be 1‑D")

    total = decay.sum()
    if total == 0:
        return np.nan + 1j * np.nan

    # Time axis (ns)
    t = np.arange(decay.size) * tcspc_resolution

    # Angular modulation frequency (rad/ns)
    omega = 2 * np.pi * sync_rate  # Hz → ns⁻¹

    # Complex numerator and normalization
    phasor = np.dot(decay, np.exp(1j * omega * t)) / total
    return phasor


def draw_unitary_circle(ax: Axes, sync_rate, tau_max: int = None, tick_length=0.02, color='white', label_color='white'):
    if not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes object, got {type(ax).__name__}")
    
    omega = 2 * np.pi * sync_rate

    if tau_max is None:
        period_ns = 1e9 / sync_rate
        tau_max = int(np.ceil(period_ns / 2))
        
    taus_ns = np.arange(1, tau_max + 1)
    

    center = np.array([0.5, 0])
    radius = 0.5

    # Generate circle points
    theta = np.linspace(0, np.pi, 300)
    g_circle = center[0] + radius * np.cos(theta)
    s_circle = center[1] + radius * np.sin(theta)
    ax.plot(g_circle, s_circle, '-', color=color, label='Universal Circle', lw=1)

    # Phasor function
    def phasor(tau):
        g = 1 / (1 + (omega * tau)**2)
        s = (omega * tau) / (1 + (omega * tau)**2)
        return np.array([g, s])

    # Draw ticks
    for tau_ns in taus_ns:
        tau_s = tau_ns * 1e-9
        p = phasor(tau_s)
        v = p - center
        v_unit = v / np.linalg.norm(v)
        p1 = p - (tick_length / 2) * v_unit
        p2 = p + (tick_length / 2) * v_unit
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, lw=1)
        label_pos = p + (tick_length * 1.2) * v_unit
        ax.text(label_pos[0], label_pos[1], f'{tau_ns} ns', fontsize=8,
                ha='center', va='center', color=label_color)

    # Clean formatting
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('g')
    ax.set_ylabel('s')


def average_phasor(
    phasor: np.ndarray,
    photon_count: np.ndarray,
    mask: np.ndarray | None = None,
) -> complex:
    """
    Photon‑weighted mean phasor over a region of interest.

    Parameters
    ----------
    phasor : np.ndarray
        Complex array (g + i·s) for each pixel.
    photon_count : np.ndarray
        Photon counts per pixel (same shape as `phasor`).
    mask : np.ndarray or None, optional
        Boolean / int array with the same shape.
        *True* (or non‑zero) selects pixels to include.
        If None, all pixels are eligible.

    Returns
    -------
    complex
        Weighted average phasor.
        Returns nan+1j*nan if the ROI has zero total photons.
    """
    if phasor.shape != photon_count.shape:
        raise ValueError("phasor and photon_count must have identical shapes")
    if mask is not None and mask.shape != phasor.shape:
        raise ValueError("mask must have the same shape as phasor")

    # Build a validity mask:  count>0  &  phasor finite  &  (ROI mask if given)
    valid = (photon_count > 0) & np.isfinite(phasor)
    if mask is not None:
        valid &= mask.astype(bool)

    if not np.any(valid):
        return np.nan + 1j * np.nan

    # Photon‑weighted sum and total photon count
    weighted_sum = np.sum(phasor[valid] * photon_count[valid])
    total_photons = np.sum(photon_count[valid])

    if total_photons == 0:
        return np.nan + 1j * np.nan

    return weighted_sum / total_photons

def shift_decay(arr, n):
    wrapped = np.roll(arr, -n) 
    return wrapped

import xarray as xr

def sum_hyperstack_dict(data: dict[str, xr.DataArray], dims):
    """
    Collapse a dict of DataArrays along given dimensions, preserving structure and names.
    
    Keys expected:
      - 'intensity' : summed directly
      - 'lifetime'  : weighted avg by intensity; zeros where denom==0
      - 'phasor_g'  : weighted avg by intensity; NaN where denom==0; ignore input NaNs
      - 'phasor_s'  : same as phasor_g
    
    Parameters
    ----------
    data : dict of str -> xr.DataArray
        Input data package (may contain a subset of keys).
    dims : str or list of str
        Dimension(s) to sum along.
    
    Returns
    -------
    dict of str -> xr.DataArray
        Reduced data package, same keys as input, with names preserved.
    """
    if isinstance(dims, str):
        dims = [dims]

    out = {}

    # intensity (photon_count)
    if "intensity" in data:
        intensity = data["intensity"]
        photon_sum = intensity.sum(dim=[d for d in dims if d in intensity.dims], keepdims=True)
        out["intensity"] = (
            photon_sum.astype("uint64")
            .assign_attrs(intensity.attrs)
            .rename(intensity.name)  # preserve name
        )
    else:
        photon_sum = None

    # lifetime (mean_arrival_time)
    if "lifetime" in data and photon_sum is not None:
        lt = data["lifetime"]
        denom = photon_sum if set(dims) & set(lt.dims) else intensity
        numer = (lt * intensity).sum(dim=[d for d in dims if d in lt.dims], keepdims=True)
        avg = numer / xr.where(denom > 0, denom, 0)
        out["lifetime"] = (
            avg.astype("float32")
            .assign_attrs(lt.attrs)
            .rename(lt.name)  # preserve name
        )

    # phasor_g / phasor_s
    for var in ("phasor_g", "phasor_s"):
        if var in data and photon_sum is not None:
            pa = data[var]
            denom = photon_sum if set(dims) & set(pa.dims) else intensity
            numer = (pa.fillna(0) * intensity).sum(dim=[d for d in dims if d in pa.dims], keepdims=True)
            avg = numer / xr.where(denom > 0, denom, np.nan)
            out[var] = (
                avg.astype("float32")
                .assign_attrs(pa.attrs)
                .rename(pa.name)  # preserve name
            )

    return out


# def sum_dataset(ds: xr.Dataset, dims):
#     """
#     Collapse a Dataset along given dimensions, preserving structure and dtypes.
    
#     - photon_count, tcspc_histogram: summed directly (uint64 if present)
#     - mean_arrival_time: photon-count-weighted average, zeros if photon_sum == 0 (float32 if present)
#     - phasor_g, phasor_s: photon-count-weighted average, NaN if photon_sum == 0 (float32 if present)
    
#     Parameters
#     ----------
#     ds : xr.Dataset
#         Input dataset (may contain a subset of expected variables).
#     dims : str or list of str
#         Dimension(s) to sum along.
    
#     Returns
#     -------
#     xr.Dataset
#         Reduced dataset with summed/weighted variables.
#         Reduced dims remain with length = 1.
#     """
#     if isinstance(dims, str):
#         dims = [dims]

#     out = {}

#     # Photon count is mandatory if we're doing weighted averages
#     if "photon_count" in ds:
#         photon_sum = ds["photon_count"].sum(dim=dims, keepdims=True)
#         out["photon_count"] = photon_sum.astype("uint64")
#     else:
#         photon_sum = None

#     # Mean arrival time: needs photon_count
#     # if "mean_arrival_time" in ds and photon_sum is not None:
#     #     # weighted_sum = (
#     #     #     ds["mean_arrival_time"].fillna(0) * ds["photon_count"]
#     #     # ).sum(dim=dims, keepdims=True)
#     #     # avg = xr.where(photon_sum > 0, weighted_sum / photon_sum, np.nan)
#     #     valid = ds["mean_arrival_time"].notnull()
#     #     weighted_sum = (ds["mean_arrival_time"].where(valid, 0) * ds["photon_count"]).sum(dim=dims, keepdims=True)
#     #     denom   = ds["photon_count"].where(valid, 0).sum(dim=dims, keepdims=True)
#     #     avg = weighted_sum / denom
#     #     out["mean_arrival_time"] = avg.astype("float32")

#     for var in ["mean_arrival_time", "phasor_g", "phasor_s"]:
#         if var in ds and photon_sum is not None:
#             # weighted_sum = (ds[var].fillna(0) * ds["photon_count"]).sum(dim=dims, keepdims=True)
#             # avg = weighted_sum / xr.where(photon_sum > 0, photon_sum, np.nan)
#             valid = ds[var].notnull()
#             avg = (
#                 (ds[var].where(valid, 0) * ds["photon_count"]).sum(dim=dims, keepdims=True)
#                 / ds["photon_count"].where(valid, 0).sum(dim=dims, keepdims=True)
#             )
#             out[var] = xr.where(photon_sum > 0, avg, np.nan).astype("float32")

#     # Histogram: direct sum
#     if "tcspc_histogram" in ds:
#         out["tcspc_histogram"] = ds["tcspc_histogram"].sum(dim=dims, keepdims=True).astype("uint64")

#     return xr.Dataset(out)

import numpy as np
import xarray as xr

def aggregate_dataset(ds: xr.Dataset, dims):
    """
    Aggregates a Dataset along given dimensions, preserving structure, dtypes,
    and coordinates (aggregated dimensions are kept with length=1). Doesn't collapse along given dimension.

    - photon_count, tcspc_histogram: summed directly (uint64 if present)
    - mean_arrival_time: photon-count-weighted average, NaN if photon_sum == 0 (float32 if present)
    - phasor_g, phasor_s: photon-count-weighted average, NaN if photon_sum == 0 (float32 if present)
    """
    if isinstance(dims, str):
        dims = [dims]

    # Optional sanity check
    missing = [d for d in dims if d not in ds.dims]
    if missing:
        raise ValueError(f"Requested dims not in dataset: {missing}")

    out = {}

    # Photon count (needed for weighted averages)
    photon_sum = None
    if "photon_count" in ds:
        photon_sum = ds["photon_count"].sum(dim=dims, keepdims=True)
        out["photon_count"] = photon_sum.astype("uint64")

    # Weighted averages
    for var in ["mean_arrival_time", "phasor_g", "phasor_s"]:
        if var in ds and photon_sum is not None:
            valid = ds[var].notnull()
            num = (ds[var].where(valid, 0) * ds["photon_count"]).sum(dim=dims, keepdims=True)
            den = ds["photon_count"].where(valid, 0).sum(dim=dims, keepdims=True)
            avg = num / den
            out[var] = xr.where(photon_sum > 0, avg, np.nan).astype("float32")

    # Direct sum variables
    if "tcspc_histogram" in ds:
        out["tcspc_histogram"] = ds["tcspc_histogram"].sum(dim=dims, keepdims=True).astype("uint64")

    out_ds = xr.Dataset(out)

    # Preserve coords: reduce along summed dims with a length-1 slice (NOT index 0)
    coord_ds = ds.coords.to_dataset()
    indexers = {d: slice(0, 1) for d in dims if d in coord_ds.dims}
    coord_ds = coord_ds.isel(indexers)

    # Ensure each reduced dimension still has a dimension coordinate
    for d in dims:
        if d in ds.dims and d not in coord_ds:
            if d in ds.coords:
                coord_ds[d] = ds[d].isel({d: slice(0, 1)})
            else:
                coord_ds[d] = xr.DataArray(np.arange(1), dims=(d,))

    # Merge reduced coords back; keep variable dims (which already include length-1 dims)
    out_ds = xr.merge([out_ds, coord_ds], compat="override")

    return out_ds
