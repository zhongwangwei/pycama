"""
Optimized Forcing Interpolation Module

This module provides Numba JIT-compiled version of forcing interpolation
for significant performance improvements.

Original: ~17 seconds for 24-hour simulation
Optimized: Expected ~1-2 seconds (10x speedup)

Note: Numba JIT does not support numpy.ma.MaskedArray, so we convert to regular arrays
"""
import numpy as np
from numba import jit


def _ensure_regular_array(arr):
    """Convert masked array to regular array, replacing masked values with fill_value"""
    if isinstance(arr, np.ma.MaskedArray):
        # Get the filled array (masked values replaced with fill_value)
        return arr.filled()
    return arr


@jit(nopython=True, cache=True)
def roff_interp_jit(pbuffin, nseqall, inpn, inpx, inpy, inpa, nxin, nyin, drofunit, rmis):
    """
    Vectorized runoff interpolation with Numba JIT compilation

    Parameters:
    -----------
    pbuffin : ndarray (nxin, nyin)
        Input runoff data [mm/dt]
    nseqall : int
        Number of river cells
    inpn : int
        Maximum number of contributing cells per river cell
    inpx : ndarray (nseqall, inpn)
        X-indices of contributing cells (1-based Fortran indexing)
    inpy : ndarray (nseqall, inpn)
        Y-indices of contributing cells (1-based Fortran indexing)
    inpa : ndarray (nseqall, inpn)
        Area fractions of contributing cells
    nxin : int
        Input grid X dimension
    nyin : int
        Input grid Y dimension
    drofunit : float
        Unit conversion factor (mm/dt to m/s)
    rmis : float
        Missing value flag

    Returns:
    --------
    pbuffout : ndarray (nseqall,)
        Interpolated runoff [m3/s]
    """
    # Initialize output
    pbuffout = np.zeros(nseqall, dtype=np.float64)

    # Loop over all river cells
    for iseq in range(nseqall):
        total_runoff = 0.0

        # Loop over all contributing input cells
        for inpi in range(inpn):
            ixin = inpx[iseq, inpi]

            # Check if valid contributing cell (Fortran uses 1-based, Python 0-based)
            if ixin > 0:
                # Convert from Fortran 1-based to Python 0-based
                ixin = ixin - 1
                iyin = inpy[iseq, inpi] - 1

                # Check bounds
                if ixin < nxin and iyin < nyin:
                    # Check if not missing value
                    if abs(pbuffin[ixin, iyin] - rmis) > 1e-6:
                        # Accumulate weighted runoff
                        # INPA is the area fraction
                        # DROFUNIT converts mm/day to m/s
                        total_runoff += pbuffin[ixin, iyin] * inpa[iseq, inpi] / drofunit

        pbuffout[iseq] = total_runoff

    return pbuffout


@jit(nopython=True, cache=True)
def conv_resol_jit(pbuffin_vec, nseqall, drofunit, rmis):
    """
    Direct resolution conversion without interpolation (JIT-compiled)

    Parameters:
    -----------
    pbuffin_vec : ndarray (n,)
        Flattened input runoff data [mm/dt]
    nseqall : int
        Number of river cells
    drofunit : float
        Unit conversion factor
    rmis : float
        Missing value flag

    Returns:
    --------
    pbuffout : ndarray (nseqall,)
        Converted runoff [m3/s]
    """
    pbuffout = np.zeros(nseqall, dtype=np.float64)

    for iseq in range(min(nseqall, len(pbuffin_vec))):
        if abs(pbuffin_vec[iseq] - rmis) > 1e-6:
            # Convert from mm/dt to m/s
            pbuffout[iseq] = max(pbuffin_vec[iseq] / drofunit, 0.0)
        else:
            pbuffout[iseq] = 0.0

    return pbuffout


def roff_interp_optimized(pbuffin, nseqall, inpn, inpx, inpy, inpa, nxin, nyin, drofunit, rmis):
    """
    Wrapper for roff_interp_jit that handles masked arrays

    This function converts masked arrays to regular arrays before calling the JIT version
    """
    # Convert masked array to regular array if needed
    pbuffin = _ensure_regular_array(pbuffin)
    inpx = _ensure_regular_array(inpx)
    inpy = _ensure_regular_array(inpy)
    inpa = _ensure_regular_array(inpa)

    # Call JIT-compiled version
    return roff_interp_jit(pbuffin, nseqall, inpn, inpx, inpy, inpa, nxin, nyin, drofunit, rmis)


def conv_resol_optimized(pbuffin_vec, nseqall, drofunit, rmis):
    """
    Wrapper for conv_resol_jit that handles masked arrays

    This function converts masked arrays to regular arrays before calling the JIT version
    """
    # Convert masked array to regular array if needed
    pbuffin_vec = _ensure_regular_array(pbuffin_vec)

    # Call JIT-compiled version
    return conv_resol_jit(pbuffin_vec, nseqall, drofunit, rmis)
