#!/usr/bin/env python3

# packages
import napari

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr

from flopa.io.ptuio.file import Header
from flopa.io.ptuio.decoder import decode_t3, T3OverflowCorrector, TTTRType
from flopa.io.ptuio.reader import TTTRReader
from flopa.io.ptuio.marker import MarkerInterpreter, get_marker_distribution, marker_events, overflow_events
from flopa.io.ptuio.reconstructor import ScanConfig, ImageReconstructor


def test_header_parsing():
    hdr = Header(r"../test_data/MultiHarp150_2025-03-13_11-37-40.ptu")
    assert hdr.file_type.name == "PTU"
    assert hdr.get("CreatorSW_Name") is not None
    assert hdr.record_type is not None

def test_t3_decoding():
    sample = np.array([0x8407A1B3], dtype=np.uint32)
    rec = decode_t3(sample, 0x00010304)
    assert rec['channel'][0] >= 0
    assert isinstance(rec['dtime'][0], np.integer)
    
def test_decode_t3_basic():
    # Construct fake T3 records manually
    # One normal event, one overflow event
    # Format (32-bit): [special|channel|dtime|nsync]

    # Normal photon: nsync=10, dtime=123, channel=2, special=0
    record1 = (0 << 28) | (2 << 25) | (123 << 12) | 10

    # Overflow event: special=1, channel=63 (0x3F), rest doesn't matter
    record2 = (1 << 28) | (0x3F << 25)

    raw = np.array([record1, record2], dtype=np.uint32)

    decoded = decode_t3(raw, TTTRType.HydraHarp2T3)

    # Check normal record
    assert decoded['nsync'][0] == 10
    assert decoded['dtime'][0] == 123
    assert decoded['channel'][0] == 2
    assert decoded['special'][0] == 0

    # Check overflow record
    assert decoded['channel'][1] == 0x3F
    assert decoded['special'][1] == 1


def make_t3_record(special, channel, dtime, nsync):
    """Pack a 32-bit T3 record with given field values."""
    return np.uint32(
        ((special & 0xF) << 28) |
        ((channel & 0x7) << 25) |
        ((dtime & 0x1FFF) << 12) |
        (nsync & 0xFFF)
    )

def create_FLIM_image(mean_photon_arrival_time, intensity, colormap=cm.rainbow, 
                      lt_min=None, lt_max=None):
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

    # Set lt_min/lt_max if not given
    if lt_min is None:
        lt_min = np.nanmin(mean_photon_arrival_time)
    if lt_max is None:
        lt_max = np.nanmax(mean_photon_arrival_time)

    # Prevent divide-by-zero
    if lt_max == lt_min:
        raise ValueError("lt_max and lt_min must differ")

    # Normalize LT to [0, 1]
    LT_normalized = (mean_photon_arrival_time - lt_min) / (lt_max - lt_min)
    LT_normalized = np.clip(LT_normalized, 0, 1)

    # Map normalized LT to RGB using colormap
    LT_rgb = colormap(LT_normalized)[..., :3]  # Drop alpha

    # Normalize intensity to [0, 1]
    intensity_max = intensity.max()
    if intensity_max == 0:
        intensity_normalized = np.zeros_like(intensity)
    else:
        intensity_normalized = intensity / intensity_max

    # Apply intensity scaling to RGB
    FLIM_image = LT_rgb * intensity_normalized[..., np.newaxis]

    return FLIM_image




if __name__ == "__main__":

    # hdr = Header("test/data/MultiHarp150_2025-03-13_11-37-40.ptu")
    # for key, value in hdr.tags.items():
    #     print(f'{key}: {value}')

    # raw = hdr.read_records(10000)

    reader = TTTRReader("./test_data/MultiHarp150_2025-06-18_16-33-24.274_f1_ch1_accu1_px500.ptu")

    for key, value in reader.header.tags.items():
         print(f'{key}: {value}')
    
    wrap = reader.header.tags.get("TTResultFormat_WrapAround", 1024)  # default fallback
    corrector = T3OverflowCorrector(wraparound=wrap)
    
    cfg = ScanConfig(
        bidirectional=False, 
        frames= 1,
        lines=500, 
        pixels=500, 
        # line_start_marker=1, 
        # line_stop_marker=2, 
        # frame_start_marker=4, 
        line_accumulations= (1,)
        )
    # interpreter = MarkerInterpreter(scan_config=cfg)
    # cfg = ScanConfig()
    reconstructor = ImageReconstructor(config=cfg)
    
    # corrected = []
    # markers_corrected = []
    for chunk in reader.iter_chunks():
        corrected_chunk = corrector.correct(chunk)
        reconstructor.update(corrected_chunk)

        # time_tag.append(time.time())
        
        # line_start_markers = reconstructor._extract_markers(corrected_chunk, cfg.line_start_marker)
        # line_stop_markers = reconstructor._extract_markers(corrected_chunk, cfg.line_stop_marker)
        # current_segment = reconstructor._build_line_segments(start_markers=line_start_markers,stop_markers=line_stop_markers)
        # corrected.append(corrected_chunk)
        # markers = reconstructor._extract_markers(corrected_chunk,(1,2,3,4))
        # markers_corrected.append(markers)
        # # segment.append(current_segment)
        

    # corrected = np.concatenate(corrected)
    # print("Marker stats:", get_marker_distribution(corrected))
    # markers_corrected = np.concatenate(markers_corrected)
    
    
    result = reconstructor.finalize(return_xarray=True)
    result.to_netcdf("./test_data/result.h5")

    image = result.photon_count
    
    image = image.transpose("frame","sequence","channel","line","pixel")
    print("Image shape: ", image.shape)



    viewer = napari.Viewer()
    viewer.add_image(image.data, name="hyperstack")  # .data strips xarray metadata
    napari.run()





    