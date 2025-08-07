import numpy as np
import xarray as xr
from pathlib import Path
from qtpy.QtWidgets import QApplication # For progress bar updates

# Import YOUR project's modules
from flopa.io.ptuio.reader import TTTRReader
from flopa.io.ptuio.decoder import T3OverflowCorrector
from flopa.io.ptuio.reconstructor import ScanConfig, ImageReconstructor

def estimate_tcspc_bins(header_tags: dict, buffer: int = 100) -> int:
    """Estimates the number of TCSPC bins from header data."""
    rep_rate = header_tags.get("TTResult_SyncRate", 40e6)
    resolution = header_tags.get("MeasDesc_Resolution", 5e-12)
    if rep_rate == 0: return 4096
    bins = int(np.ceil(1 / (resolution * rep_rate))) + buffer
    return bins

def reconstruct_ptu_to_dataset(
    ptu_filepath: Path,
    scan_config: ScanConfig,
    progress_callback=None
) -> xr.Dataset:
    """
    Reads a PTU file, reconstructs the image stack based on ScanConfig,
    and returns an xarray.Dataset.
    """
    reader = TTTRReader(str(ptu_filepath))
    header_tags = reader.header.tags

    # Get constants from the header for reconstruction
    wrap = header_tags.get("TTResultFormat_WrapAround", 1024)
    repetition_rate = header_tags.get("TTResult_SyncRate", 40e6)
    tcspc_resolution = header_tags.get("MeasDesc_Resolution", 5e-12)
    tcspc_bins = estimate_tcspc_bins(header_tags)
    
    # Calculate omega for phasor plot calculations
    omega = 2 * np.pi * repetition_rate * tcspc_resolution

    # Initialize the core components using YOUR classes
    corrector = T3OverflowCorrector(wraparound=wrap)
    reconstructor = ImageReconstructor(
        config=scan_config,
        omega=omega,
        tcspc_channels=tcspc_bins
    )

    # Process the file in chunks
    # This part needs a way to get total chunks for the progress bar.
    # Let's add a simple method to your TTTRReader for this.
    # (See note below to add `num_chunks` to your TTTRReader)
    total_records = reader.header.get("TTResult_NumberOfRecords")
    chunk_size = 1000000 # must match iter_chunks
    total_chunks = np.ceil(total_records / chunk_size)

    for i, chunk in enumerate(reader.iter_chunks(chunk_size=chunk_size)):
        if progress_callback:
            progress_callback(i + 1, total_chunks)
            QApplication.processEvents() # Force UI to update

        corrected_chunk = corrector.correct(chunk)
        reconstructor.update(corrected_chunk)

    # Finalize and return the data
    result_dataset = reconstructor.finalize(return_xarray=True)
    
    # Attach header metadata to the dataset for future reference
    # Convert all values to strings for safety, as some types aren't HDF5-compatible
    serializable_tags = {k: str(v) for k, v in header_tags.items()}
    result_dataset.attrs.update(serializable_tags)
    
    return result_dataset