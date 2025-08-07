import numpy as np

def calculate_phasor(laser_frequency_mhz: float, harmonic: int = 1):
    """Placeholder function to calculate phasor coordinates (G and S)."""
    #print(f"--- MOCK: Calculating phasor ---")
    g = np.random.normal(0.5, 0.2, )
    s = np.random.normal(0.2, 0.1, )
    return g, s


def get_phasor_from_decay(
    decay: np.ndarray,
    tcspc_resolution: float,
    sync_rate: float,
) -> complex:
    """
    Photon‑weighted complex phasor from a TCSPC decay.

    Parameters
    ----------
    decay : 1‑D np.ndarray
        Photon counts per TCSPC channel.
    tcspc_resolution_ns : float
        Width of one TCSPC channel in seconds.
    sync_rate : float
        Laser repetition rate in Hz.

    Returns
    -------
    complex
        Phasor Φ = g + i·s.
        Returns nan+1j*nan if the decay is empty.
    """
    if decay.ndim != 1:
        raise ValueError("`decay` must be 1‑D")

    total = decay.sum()
    if total == 0:
        return np.nan + 1j * np.nan

    # Time axis (ns)
    t = np.arange(decay.size) * tcspc_resolution

    # Angular modulation frequency (rad/ns)
    omega = 2 * np.pi * sync_rate  # Hz → ns⁻¹

    # Complex numerator and normalization
    phasor = np.dot(decay, np.exp(1j * omega * t)) / total
    return phasor