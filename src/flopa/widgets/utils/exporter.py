# flopa/widgets/utils/exporter.py

import pandas as pd
import json
from pathlib import Path
import numpy as np
from typing import List
import traceback
import xarray as xr
import tifffile
from matplotlib import cm
import itertools


from flopa.processing.flim_image import create_FLIM_image 



def export_phasor_data(
    output_path: Path,
    g_coords: np.ndarray,
    s_coords: np.ndarray,
    photon_counts: np.ndarray,
    labels: np.ndarray,
    areas: np.ndarray,
    dataset_name: str = "N/A"
):
    """
    Exports final, plotted phasor coordinates to a CSV file.

    """
    if not (len(g_coords) == len(s_coords) == len(photon_counts) == len(labels) == len(areas)):
        raise ValueError("All input arrays must have the same length.")

    df = pd.DataFrame({
        "dataset_name": dataset_name,
        "label_id": labels,
        "g": g_coords,
        "s": s_coords,
        "photon_count_sum": photon_counts,
        "area_pixels": areas

    })
    
    df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(df)} points to {output_path}")


def export_decay_data(
    output_path: Path,
    time_axis: np.ndarray,
    decay_curves: List[np.ndarray],
    curve_labels: List[str],
    dataset_name: str = "N/A"
):
    """
    Exports one or more decay curves to a CSV file with descriptive headers.

    """
    if len(decay_curves) != len(curve_labels):
        raise ValueError("The number of decay curves must match the number of labels.")

    data_dict = {"time": time_axis}
    
    for label, curve in zip(curve_labels, decay_curves):
        clean_label = label.replace(":", "").replace(" ", "").replace(",", "_")
        column_name = f"{clean_label}__{dataset_name}"
        data_dict[column_name] = curve
        
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(decay_curves)} decay curves to {output_path}")


def export_dataset_as_hdf5(dataset: xr.Dataset, save_path: Path):
    """
    Saves the entire reconstructed xarray Dataset to a single HDF5 file.
    
    """
    try:
        ds_to_save = dataset.copy(deep=True)
        
        for key, value in ds_to_save.attrs.items():
            if isinstance(value, dict):
                ds_to_save.attrs[key] = json.dumps(value)
        
        ds_to_save.to_netcdf(save_path, engine='h5netcdf')
        
        print(f"Successfully exported full dataset to {save_path}")
        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, e

def _sanitize_counts_for_tiff(data: np.ndarray) -> np.ndarray:
    """Sanitizes integer count data (like intensity) for TIFF saving."""
    if data is None: return None
    if data.dtype.kind == 'u' and data.dtype.itemsize > 2: 
        if data.max() > 65535:
            return data.astype(np.float32)
        else:
            return data.astype(np.uint16)
    return data

def _sanitize_float_for_tiff(data: np.ndarray) -> np.ndarray:
    """Sanitizes floating-point data (like lifetime) for TIFF saving."""
    if data is None: return None

    return data.astype(np.float32)

def export_view_as_tiff(
    output_folder: Path,
    items_to_save: dict,
    base_dataset: xr.Dataset,
    processed_data: dict,
    lifetime_colormap: str,
    intensity_clims: tuple,
    lifetime_clims: tuple
):
    """
    Saves the pre-processed data from the cache as separate, named TIFF files.
    """
    try:
        source_filename = base_dataset.attrs.get('source_filename', 'exported_data')
        base_name = Path(source_filename).stem 
        
        saved_files_count = 0

        # Save Intensity
        if items_to_save.get("intensity") and processed_data.get("intensity") is not None:
            fname = output_folder / f"{base_name}_intensity.tif"
            tifffile.imwrite(fname, _sanitize_counts_for_tiff(processed_data["intensity"]), imagej=True)
            saved_files_count += 1
            
        # Save Lifetime
        if items_to_save.get("lifetime") and processed_data.get("lifetime") is not None:
            fname = output_folder / f"{base_name}_lifetime_ns.tif"
            tifffile.imwrite(fname, _sanitize_float_for_tiff(processed_data["lifetime"]), imagej=True)
            saved_files_count += 1

        # Save RGB FLIM
        if items_to_save.get("flim_rgb"):
            intensity_data = processed_data.get("intensity")
            lifetime_data = processed_data.get("lifetime")
            if intensity_data is not None and lifetime_data is not None:
                fname = output_folder / f"{base_name}_rgb.tif"
                rgb_image = create_FLIM_image(
                    mean_photon_arrival_time=lifetime_data, intensity=intensity_data,
                    colormap=cm.get_cmap(lifetime_colormap),
                    lt_min=lifetime_clims[0], lt_max=lifetime_clims[1],
                    int_min=intensity_clims[0], int_max=intensity_clims[1]
                )
                final_rgb_data = (rgb_image * 255).astype(np.uint8)
                tifffile.imwrite(fname, final_rgb_data, imagej=True)
                saved_files_count += 1

        if saved_files_count == 0:
            return False, ValueError("No data was available for the selected options.")
        
        return True, None

    except Exception as e:
        traceback.print_exc()
        return False, e

def sexport_view_as_tiff(
    output_path: Path,
    items_to_save: dict,
    base_dataset: xr.Dataset,
    selectors: dict,
    processed_data: dict,
    lifetime_colormap: str,
    intensity_clims: tuple,
    lifetime_clims: tuple
):
    """
    Saves the pre-processed data from the cache as a single, multi-page TIFF file.
    """
    try:
        images_to_stack = []
        
        if items_to_save.get("intensity") and processed_data.get("intensity") is not None:
            images_to_stack.append(_sanitize_counts_for_tiff(processed_data["intensity"]))

        if items_to_save.get("lifetime") and processed_data.get("lifetime") is not None:
            images_to_stack.append(_sanitize_float_for_tiff(processed_data["lifetime"]))

        if images_to_stack:
            if not output_path.suffix.lower() in ['.tif', '.tiff']:
                output_path = output_path.with_suffix('.tif')
            
            stacked_data = np.stack(images_to_stack, axis=0)
            tifffile.imwrite(output_path, stacked_data, imagej=True)

        if items_to_save.get("flim_rgb"):
            intensity_data = processed_data.get("intensity")
            lifetime_data = processed_data.get("lifetime")

            if intensity_data is not None and lifetime_data is not None:
                rgb_output_path = output_path.with_name(f"{output_path.stem}_rgb.tif")
                
                rgb_image = create_FLIM_image(
                    mean_photon_arrival_time=lifetime_data, intensity=intensity_data,
                    colormap=cm.get_cmap(lifetime_colormap),
                    lt_min=lifetime_clims[0], lt_max=lifetime_clims[1],
                    int_min=intensity_clims[0], int_max=intensity_clims[1]
                )
                final_rgb_data = (rgb_image * 255).astype(np.uint8)
                tifffile.imwrite(rgb_output_path, final_rgb_data, imagej=True)
                
                if not images_to_stack:
                    return True, None
        
        if not images_to_stack and not items_to_save.get("flim_rgb"):
            return False, ValueError("No data was available for the selected options.")
        
        return True, None

    except Exception as e:
        traceback.print_exc()
        return False, e