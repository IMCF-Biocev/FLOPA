# flopa/processing/phasor.py

import numpy as np

def calculate_phasor_calibration_factor(
    theoretical_lifetime_ns: float,
    measured_phasor: complex,
    sync_rate_hz: float
) -> complex:
    """
    Calculates the complex calibration factor for phasor plots.

    Args:
        theoretical_lifetime_ns (float): The known lifetime of the reference dye in nanoseconds.
        measured_phasor (complex): The experimentally measured g+is phasor of the reference dye.
        sync_rate_hz (float): The laser repetition rate in Hz.

    Returns:
        complex: The complex calibration factor.
    """
    if np.abs(measured_phasor) == 0:
        raise ValueError("Measured phasor cannot be zero.")
        
    tau_s = theoretical_lifetime_ns * 1e-9  # Convert to seconds
    omega = 2 * np.pi * sync_rate_hz
    
    # Calculate the theoretical phasor position for the given lifetime
    g_theory = 1 / (1 + (omega * tau_s)**2)
    s_theory = (omega * tau_s) / (1 + (omega * tau_s)**2)
    phasor_theory = g_theory + 1j * s_theory
    
    # The calibration factor is the complex number that rotates/scales the measured
    # phasor to the theoretical position.
    calibration_factor = phasor_theory / measured_phasor
    return calibration_factor


def apply_phasor_calibration(
    phasor_array: np.ndarray,
    calibration_factor: complex
) -> np.ndarray:
    """
    Applies a complex calibration factor to an array of phasors.

    Args:
        phasor_array (np.ndarray): A 2D complex array of g+is values.
        calibration_factor (complex): The calibration factor to apply.

    Returns:
        np.ndarray: The calibrated 2D complex array of phasors.
    """
    return phasor_array * calibration_factor