from .ptuio.reader import TTTRReader
from .ptuio.decoder import T3OverflowCorrector
from .ptuio.utils import estimate_tcspc_bins, marker_events, get_marker_distribution
from flopa.processing.logger import ProgressLogger

from pathlib import Path
import numpy as np
import xarray as xr
import json

def format_ptu_header(header_tags, constants, full_header=False):
    """
    Generates a formatted string summary of PTU header and constants.

    Args:
        header_tags (dict): The dictionary of header tags from the PTU file.
        constants (dict): The dictionary of calculated constants.
        full_header (bool): If True, appends the entire raw header dump.

    Returns:
        str: A formatted, multi-line string with the summary.
    """
    lines = []

    measurement_sub_mode = header_tags.get("Measurement_SubMode")
    if measurement_sub_mode is not None and measurement_sub_mode < 1:
        lines.append("* WARNING: Not an image. Configure scanning settings.")

    lines += [

        "--- Key Parameters ---",
        f"Repetition Rate:   {constants['repetition_rate']:.2e} Hz",
        f"TCSPC Resolution:  {constants['tcspc_resolution_ns']:.2e}",
        f"Resolution Unit:   {constants['resolution_unit']}",
        f"TCSPC Bins:        {constants['tcspc_bins']}",
        f"Wrap Around:       {constants['wrap']}",
        f"Omega:             {constants['omega']:.4e} rad/s",
        "",  # Blank line for spacing
        "--- Image Header ---",
        f"Pixels X:          {header_tags.get('ImgHdr_PixX', 'N/A')}",
        f"Pixels Y:          {header_tags.get('ImgHdr_PixY', 'N/A')}",
        f"Frame Count:       {header_tags.get('ImgHdr_NumberOfFrames', 'N/A')}"
    ]


    if full_header:
        lines.extend([
            "",
            "--- Full Header ---"
        ])
        for key, value in header_tags.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)



def read_ptu_file(path, header=True, logger: ProgressLogger = None) -> dict:
    """
    Reads a PTU file and creates a standardized dictionary of instrument constants.

    This function reads the header, applies sensible defaults for missing
    critical tags, and calculates useful derived values.
    """
    if logger is None:
        logger = ProgressLogger(mode='print')

    logger.log(f"Reading PTU file: {path}")
    reader = TTTRReader(path)
    header_tags = reader.header.tags

    # --- 1. Read primary values from header with robust defaults ---
    repetition_rate = header_tags.get("TTResult_SyncRate", 40e6)
    
    # Default TCSPC resolution: 1 ns (equivalent to 1/1e9 s)
    tcspc_resolution = header_tags.get("MeasDesc_Resolution", 1 / 1e9)
    tcspc_resolution_ns = tcspc_resolution * 1e9
    resolution_unit = 'ch' if tcspc_resolution_ns == 1.0 else 'ns'
    tcspc_bins = estimate_tcspc_bins(header_tags, buffer=0)
    
    wrap = header_tags.get("TTResultFormat_WrapAround", 1024)
    omega = 2 * np.pi * repetition_rate * tcspc_resolution

    # --- 3. Assemble the final, standardized constants dictionary ---
    constants = {
        "repetition_rate": repetition_rate,
        "tcspc_resolution": tcspc_resolution,
        "tcspc_resolution_ns": tcspc_resolution_ns,
        "resolution_unit": resolution_unit,
        "tcspc_bins": tcspc_bins,
        "wrap": wrap,
        "omega": omega
    }

    summary_text = format_ptu_header(header_tags, constants, full_header=header)
    logger.log(summary_text)

    return {
        "reader": reader,
        "header": header_tags,
        "constants": constants
    }



def get_markers(reader: 'TTTRReader', chunk_limit: int = 0) -> dict:
    """
    Reads a PTU file, corrects overflows, and extracts the distribution of markers.
    
    Args:
        reader: An initialized TTTRReader for the PTU file.
        chunk_limit (int): The number of 1-million-record chunks to read for a quick analysis.

    Returns:
        A dictionary mapping marker channel numbers to their counts.
    """
    all_markers = []

    wrap = reader.header.tags.get("TTResultFormat_WrapAround", 1024)
    corrector = T3OverflowCorrector(wraparound=wrap)

    for i, chunk in enumerate(reader.iter_chunks(chunk_size=1_000_000)):
        if chunk_limit > 0 and i >= chunk_limit:
            break
        corrected_chunk = corrector.correct(chunk)
        all_markers.append(marker_events(corrected_chunk))

        
    all_markers_flat = np.concatenate(all_markers)
    if all_markers_flat.size == 0:
        return {"error": "No markers found."}

    return get_marker_distribution(all_markers_flat)


def analyze_marker_distribution(
    distribution: dict, 
    verbose: bool = False,
    line_start_marker: int = 1,
    frame_start_marker: int = 4,
    max_accumulations: int = 64
) -> dict:
    """
    Analyzes a marker distribution to suggest scan parameters.

    If verbose is True, it prints a formatted summary. It always returns the
    structured analysis dictionary.

    Args:
        distribution (dict): The output from get_markers.
        header (dict): The header tags from the TTTRReader.
        verbose (bool): If True, prints a formatted summary to the console.
        ... (other parameters) ...

    Returns:
        A dictionary containing the structured analysis results.
    """

    num_line_starts = distribution.get(line_start_marker, 0)
    num_frame_starts = distribution.get(frame_start_marker, 0)
        
    frames_guess = max(1, num_frame_starts)
    total_lines_per_frame = num_line_starts // frames_guess
    
    suggestion_pairs = []
    for i in range(1, max_accumulations + 1):
        if total_lines_per_frame % i == 0:
            lines = total_lines_per_frame // i
            if 64 <= lines <= 4096:
                suggestion_pairs.append((lines, i))
    
    analysis_results = {
        "num_line_starts": num_line_starts,
        "num_frame_starts": num_frame_starts,
        "frames_guess": frames_guess,
        "total_lines_per_frame": total_lines_per_frame,
        "suggestions": suggestion_pairs
    }

    if verbose:
        suggestion_text = _format_marker_suggestions(analysis_results)
        print("--- Marker Analysis Suggestions ---")
        print(suggestion_text)

    return analysis_results


def _format_marker_suggestions(analysis_results: dict) -> str:
    """
    Formats the results from analyze_scan_markers into a human-readable string.
    """

    lines = [
        f"Frame Starts: {analysis_results['num_frame_starts']} | Line Starts: {analysis_results['num_line_starts']}",
        f"For {analysis_results['frames_guess']} frame(s) ~ {analysis_results['total_lines_per_frame']} line scans per frame.",
        "",
        "Possible combinations: Lines x Accumulations"
    ]

    suggestions = analysis_results.get("suggestions", [])
    if not suggestions:
        lines.append("  - Could not find common factors. Please check header or lab notes.")
    else:
        for lines_val, acc_val in suggestions:
            lines.append(f"  - {lines_val} x {acc_val}")
            
    return "\n".join(lines)


def load_h5_dataset(filepath: Path) -> xr.Dataset:
    """
    Loads a dataset from an HDF5 file previously saved by FLOPA.
    
    This function correctly re-hydrates attributes that were saved as JSON
    strings back into Python dictionaries.
    
    Args:
        filepath (Path): The path to the .h5 file.
        
    Returns:
        xr.Dataset: The fully reconstructed xarray Dataset.
    """
    ds = xr.open_dataset(filepath, engine='h5netcdf')
    
    # Create a copy of attributes to avoid modifying during iteration
    attrs_copy = ds.attrs.copy()
    
    for key, value in attrs_copy.items():
        if isinstance(value, str) and value.strip().startswith('{'):
            try:
                ds.attrs[key] = json.loads(value)
            except json.JSONDecodeError:
                pass # Not a valid JSON string, leave as is
                
    return ds