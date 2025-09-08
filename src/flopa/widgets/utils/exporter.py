# flopa/widgets/utils/exporter.py

import pandas as pd
from pathlib import Path
import numpy as np

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