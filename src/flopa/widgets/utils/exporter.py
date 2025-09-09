# flopa/widgets/utils/exporter.py

import pandas as pd
from pathlib import Path
import numpy as np
from typing import List


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