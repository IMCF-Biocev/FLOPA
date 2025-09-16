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

    Args:
        output_path (Path): The path to save the CSV file.
        g_coords (np.ndarray): 1D array of the final g-coordinates plotted.
        s_coords (np.ndarray): 1D array of the final s-coordinates plotted.
        photon_counts (np.ndarray): 1D array of the summed photon counts for each point.
        labels (np.ndarray): 1D array of the label ID for each point.
                                 (Can be np.nan for per-pixel mode).
        dataset_name (str, optional): Name of the source dataset.
    """
    if not (len(g_coords) == len(s_coords) == len(photon_counts) == len(labels) == len(areas)):
        raise ValueError("All input arrays must have the same length.")

    # Create the DataFrame directly from the final data
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

    Args:
        output_path (Path): The path to save the CSV file.
        time_axis (np.ndarray): The 1D array for the time axis (e.g., in ns).
        decay_curves (List[np.ndarray]): A list of the 1D decay curve arrays.
        curve_labels (List[str]): A list of labels for each curve, corresponding
                                  to the decay_curves list.
        dataset_name (str, optional): The name of the source dataset file.
    """
    if len(decay_curves) != len(curve_labels):
        raise ValueError("The number of decay curves must match the number of labels.")

    # Start with the time axis column
    data_dict = {"time": time_axis}
    
    # Add each decay curve as a new column with its formatted name
    for label, curve in zip(curve_labels, decay_curves):
        # Sanitize the label for use as a column header
        clean_label = label.replace(":", "").replace(" ", "").replace(",", "_")
        column_name = f"{clean_label}__{dataset_name}"
        data_dict[column_name] = curve
        
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(decay_curves)} decay curves to {output_path}")


def export_dataset_as_hdf5(dataset: xr.Dataset, save_path: Path):
    """
    Saves the entire reconstructed xarray Dataset to a single HDF5 file.
    
    This format is ideal for archiving and reloading data within FLOPA, as it
    preserves all dimensions, coordinates, data variables, and metadata.
    
    Args:
        dataset (xr.Dataset): The complete dataset to save.
        save_path (Path): The full path to the output .h5 file.
    """
    try:
        # --- FIX 1: Sanitize attributes before saving ---
        # Create a deep copy of the dataset to avoid modifying the original in memory
        ds_to_save = dataset.copy(deep=True)
        
        # Check for and convert dictionary attributes to JSON strings
        for key, value in ds_to_save.attrs.items():
            if isinstance(value, dict):
                # Convert the dictionary to a JSON formatted string
                ds_to_save.attrs[key] = json.dumps(value)
        
        # Now, the dataset has only valid attribute types (strings, numbers, etc.)
        ds_to_save.to_netcdf(save_path, engine='h5netcdf')
        
        print(f"Successfully exported full dataset to {save_path}")
        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, e

def _sanitize_counts_for_tiff(data: np.ndarray) -> np.ndarray:
    """Sanitizes integer count data (like intensity) for TIFF saving."""
    if data is None: return None
    if data.dtype.kind == 'u' and data.dtype.itemsize > 2: # Catches uint32, uint64
        if data.max() > 65535:
            return data.astype(np.float32)
        else:
            return data.astype(np.uint16)
    return data

def _sanitize_float_for_tiff(data: np.ndarray) -> np.ndarray:
    """Sanitizes floating-point data (like lifetime) for TIFF saving."""
    if data is None: return None
    # All float types are simply converted to float32, which is standard for TIFF.
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
        # --- Step 1: Generate base filename ---
        source_filename = base_dataset.attrs.get('source_filename', 'exported_data')
        base_name = Path(source_filename).stem # Removes .ptu, .h5, etc.
        
        saved_files_count = 0

        # --- Step 2: Save each requested file ---
        
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

        # --- Save the multi-page TIFF for Intensity/Lifetime ---
        if images_to_stack:
            # Ensure the output path has a .tif extension
            if not output_path.suffix.lower() in ['.tif', '.tiff']:
                output_path = output_path.with_suffix('.tif')
            
            stacked_data = np.stack(images_to_stack, axis=0)
            tifffile.imwrite(output_path, stacked_data, imagej=True)

        # --- Save the RGB FLIM image separately if requested ---
        if items_to_save.get("flim_rgb"):
            intensity_data = processed_data.get("intensity")
            lifetime_data = processed_data.get("lifetime")

            if intensity_data is not None and lifetime_data is not None:
                # Generate a separate filename for the RGB image
                rgb_output_path = output_path.with_name(f"{output_path.stem}_rgb.tif")
                
                rgb_image = create_FLIM_image(
                    mean_photon_arrival_time=lifetime_data, intensity=intensity_data,
                    colormap=cm.get_cmap(lifetime_colormap),
                    lt_min=lifetime_clims[0], lt_max=lifetime_clims[1],
                    int_min=intensity_clims[0], int_max=intensity_clims[1]
                )
                final_rgb_data = (rgb_image * 255).astype(np.uint8)
                tifffile.imwrite(rgb_output_path, final_rgb_data, imagej=True)
                
                # If only RGB was saved, the main success message won't show, so we return here.
                if not images_to_stack:
                    return True, None
        
        if not images_to_stack and not items_to_save.get("flim_rgb"):
            return False, ValueError("No data was available for the selected options.")
        
        return True, None

    except Exception as e:
        traceback.print_exc()
        return False, e