import numpy as np
from enum import IntEnum

class TTTRType(IntEnum):
    HydraHarp2T3 = 0x00010304
    MultiHarpT3 = 0x00010305
    TimeHarp260NT3 = 0x00010306
    GenericT3 = 0x00010307  # fallback

event_dtype = np.dtype([
    ("nsync",   np.uint64),
    ("dtime",   np.uint16),
    ("channel", np.uint8),
    ("special", np.uint8),
])

def get_photons(events: np.ndarray):
    return events[(events["channel"] < 63) & (events["special"] == 0)]

def get_markers(events: np.ndarray, codes):
    return events[(events["special"] != 0) & np.isin(events["channel"], codes)]

def decode_t3(records):
    out = np.empty(records.shape[0], dtype=[
        ('nsync', np.uint64),
        ('dtime', np.uint16),
        ('channel', np.uint8),
        ('special', np.uint8),
    ])

    out['nsync'] = (records & 0x3ff).astype(np.uint64) # first 10 bits
    out['dtime'] = ((records >> 10) & 0x7fff).astype(np.uint16) # next 15 bits
    out['channel'] = ((records >> 25) & 0x3f).astype(np.uint8) # next 6 bits
    out['special'] = ((records >> 31) & 0x1).astype(np.uint8) # last one bit

    return out

class T3OverflowCorrector:
    def __init__(self, wraparound=1024):
        self.wraparound = wraparound
        self.overflow_carry = 0  # Total number of wraparound events so far

    def correct(self, records: np.ndarray) -> np.ndarray:
        decoded = decode_t3(records)

        is_overflow = (decoded['special'] == 1) & (decoded['channel'] == 0x3F)
        overflow_correction = np.cumsum(is_overflow * decoded['nsync']) * self.wraparound
        overflow_total = overflow_correction + self.overflow_carry

        # Apply correction to *all* events
        decoded['nsync'] += overflow_total

        # Update carry for next chunk
        if len(overflow_correction) > 0:
            self.overflow_carry = overflow_total[-1]

        return decoded
