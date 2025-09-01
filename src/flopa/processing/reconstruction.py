# flopa/processing/reconstruction.py

from flopa.io.ptuio.decoder import T3OverflowCorrector
from flopa.io.ptuio.reconstructor import ImageReconstructor, ScanConfig 
from flopa.processing.logger import ProgressLogger

import xarray as xr
import numpy as np

def reconstruct_ptu_to_dataset(
    ptu_data: dict,
    scan_config: ScanConfig,
    outputs: list,
    tcspc_channels_override: int = None,
    logger: ProgressLogger = None
) -> xr.Dataset:


    reader = ptu_data["reader"]
    constants = ptu_data["constants"]
    header = ptu_data["header"]

    if tcspc_channels_override is not None:
        tcspc_channels = tcspc_channels_override
    else:
        tcspc_channels = constants["tcspc_bins"]

    repetition_rate = constants["repetition_rate"]
    tcspc_resolution = constants["tcspc_resolution"]
    omega = 2 * np.pi * repetition_rate * tcspc_resolution
    total_records = header.get("TTResult_NumberOfRecords")
    if not isinstance(total_records, (int, float)) or total_records <= 0:
        total_records = None

    # widget dependencies mapping
    primary_output = outputs[0]
    dependency_map = {
        "photon_count": ["photon_count"],
        "mean_arrival_time": ["photon_count", "mean_arrival_time"],
        "all": ["photon_count", 'mean_arrival_time', "phasor", "tcspc_histogram"],
    }
    full_outputs_list = dependency_map.get(primary_output, [primary_output])
    
    
    if logger is None:
        logger = ProgressLogger(mode='print')
    logger.log("Initializing reconstructor...")

    reconstructor = ImageReconstructor(
        config=scan_config,
        omega=omega,
        tcspc_channels=tcspc_channels,
        outputs=full_outputs_list
    )
    corrector = T3OverflowCorrector(wraparound=constants["wrap"])

    processed_records = 0
    chunk_number = 0
    last_logged_percent = -1

    logger.log("Starting chunk processing...")
    for chunk in reader.iter_chunks(chunk_size=1_000_000):
        chunk_number += 1

        corrected_chunk = corrector.correct(chunk)
        reconstructor.update(corrected_chunk)

        if total_records:
            processed_records += len(chunk)
            progress_percent = int((processed_records / total_records) * 100)
            if progress_percent > last_logged_percent and progress_percent % 5 == 0:
                logger.log(f"Progress: {progress_percent}%")
                last_logged_percent = progress_percent
        else:
            logger.log(f"Processing chunk #{chunk_number}...")
    
    logger.log("Finalizing results...")
    result = reconstructor.finalize()
    logger.log("Reconstruction complete.")

    return result