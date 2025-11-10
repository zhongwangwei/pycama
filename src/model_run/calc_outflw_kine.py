"""
Kinematic wave outflow calculation for CaMa-Flood

Based on cmf_opt_outflw_mod.F90: CMF_CALC_OUTFLW_KINE

Implements kinematic wave approximation using Manning's equation:
Q = (1/n) × A × R^(2/3) × S^(1/2)

where:
- Q = discharge [m³/s]
- n = Manning's roughness coefficient
- A = cross-sectional area [m²]
- R = hydraulic radius [m]
- S = water surface slope [m/m]

Advantages:
- More stable for steep slopes
- Simpler formulation
- No need for implicit scheme

Disadvantages:
- Less accurate for backwater effects
- Cannot simulate flow reversals
"""
import numpy as np


def calc_outflw_kine(
    nseqall, nseqriv, pgrv, pminslp,
    d2rivlen, d2rivwth, d2rivhgt, d2rivman, d2nxtdst, d2rivelv,
    d2rivdph, d2sfcelv, d2dwnelv,
    d2rivout, d2rivvel,
    i1next
):
    """
    Calculate river outflow using kinematic wave approximation

    Parameters:
    -----------
    nseqall : int
        Total number of river cells
    nseqriv : int
        Number of river cells (excluding mouth cells)
    pgrv : float
        Gravity acceleration [m/s²]
    pminslp : float
        Minimum slope for stability [m/m]
    d2rivlen : ndarray (nseqmax,)
        River channel length [m]
    d2rivwth : ndarray (nseqmax,)
        River channel width [m]
    d2rivhgt : ndarray (nseqmax,)
        River channel depth [m]
    d2rivman : ndarray (nseqmax,)
        Manning's roughness coefficient [-]
    d2nxtdst : ndarray (nseqmax,)
        Distance to downstream cell [m]
    d2rivelv : ndarray (nseqmax,)
        River bed elevation [m]
    d2rivdph : ndarray (nseqmax,)
        River water depth [m]
    d2sfcelv : ndarray (nseqmax,)
        Water surface elevation [m]
    d2dwnelv : ndarray (nseqmax,)
        Downstream water surface elevation [m]
    d2rivout : ndarray (nseqmax,) [output]
        River outflow [m³/s]
    d2rivvel : ndarray (nseqmax,) [output]
        River velocity [m/s]
    i1next : ndarray (nseqmax,)
        Index of downstream cell (0-based)

    Returns:
    --------
    None (modifies d2rivout and d2rivvel in place)
    """

    # Process each river cell
    for iseq in range(nseqall):

        # Get water depth
        dph = d2rivdph[iseq]

        # Skip if depth is too small
        if dph <= 1.0e-5:
            d2rivout[iseq] = 0.0
            d2rivvel[iseq] = 0.0
            continue

        # Get channel geometry
        wth = d2rivwth[iseq]
        hgt = d2rivhgt[iseq]
        man = d2rivman[iseq]

        # Calculate cross-sectional area [m²]
        area = wth * dph

        # Calculate wetted perimeter [m]
        # Simple rectangular channel: P = width + 2*depth
        peri = wth + 2.0 * dph

        # Calculate hydraulic radius [m]
        # R = A / P
        if peri > 0:
            radius = area / peri
        else:
            radius = dph  # Fallback for wide channels

        # Calculate water surface slope [m/m]
        # S = (upstream elevation - downstream elevation) / distance
        if iseq < nseqriv:
            # Regular river cell with downstream neighbor
            inext = i1next[iseq]
            if inext >= 0:
                # Water surface slope
                slope = (d2sfcelv[iseq] - d2sfcelv[inext]) / d2nxtdst[iseq]
            else:
                # Use bed slope if no downstream cell
                slope = (d2rivelv[iseq] - d2rivelv[iseq]) / d2nxtdst[iseq]
                slope = max(slope, pminslp)
        else:
            # Mouth cell - use bed slope or minimum slope
            slope = (d2sfcelv[iseq] - d2dwnelv[iseq]) / d2nxtdst[iseq]

        # Apply minimum slope for numerical stability
        slope = max(abs(slope), pminslp)

        # Manning's equation: Q = (1/n) × A × R^(2/3) × S^(1/2)
        # Note: Manning coefficient has units [s/m^(1/3)]
        if man > 0:
            discharge = (1.0 / man) * area * (radius ** (2.0/3.0)) * (slope ** 0.5)
        else:
            discharge = 0.0

        # Ensure non-negative discharge
        discharge = max(discharge, 0.0)

        # Calculate velocity [m/s]
        if area > 0:
            velocity = discharge / area
        else:
            velocity = 0.0

        # Store results
        d2rivout[iseq] = discharge
        d2rivvel[iseq] = velocity


def calc_fldout_kine(
    nseqall, pgrv, pmanfld, pminslp,
    d2rivlen, d2rivwth, d2rivhgt, d2rivelv, d2nxtdst,
    d2flddph, d2fldsto, d2fldgrd,
    d2fldout,
    i1next
):
    """
    Calculate floodplain outflow using kinematic wave approximation

    For floodplain flow, use simplified approach:
    - Floodplain treated as wide, shallow flow
    - Use floodplain gradient for slope
    - Manning coefficient for floodplain

    Parameters:
    -----------
    nseqall : int
        Total number of river cells
    pgrv : float
        Gravity acceleration [m/s²]
    pmanfld : float
        Manning's roughness coefficient for floodplain [-]
    pminslp : float
        Minimum slope [m/m]
    d2rivlen : ndarray (nseqmax,)
        River channel length [m]
    d2rivwth : ndarray (nseqmax,)
        River channel width [m]
    d2rivhgt : ndarray (nseqmax,)
        River channel depth [m]
    d2rivelv : ndarray (nseqmax,)
        River bed elevation [m]
    d2nxtdst : ndarray (nseqmax,)
        Distance to downstream cell [m]
    d2flddph : ndarray (nseqmax,)
        Floodplain water depth [m]
    d2fldsto : ndarray (nseqmax,)
        Floodplain storage [m³]
    d2fldgrd : ndarray (nseqmax, nlfp)
        Floodplain gradient [m/m]
    d2fldout : ndarray (nseqmax,) [output]
        Floodplain outflow [m³/s]
    i1next : ndarray (nseqmax,)
        Index of downstream cell

    Returns:
    --------
    None (modifies d2fldout in place)
    """

    # Process each river cell
    for iseq in range(nseqall):

        # Get floodplain depth and storage
        dph_fld = d2flddph[iseq]
        sto_fld = d2fldsto[iseq]

        # Skip if no floodplain water
        if dph_fld <= 1.0e-5 or sto_fld <= 1.0e-5:
            d2fldout[iseq] = 0.0
            continue

        # Get channel geometry
        rivlen = d2rivlen[iseq]

        # Estimate floodplain width from storage and depth
        # Assume storage = width * length * depth
        if dph_fld > 0:
            fld_wth = sto_fld / (rivlen * dph_fld)
        else:
            fld_wth = 0.0

        # Calculate floodplain area [m²]
        area_fld = fld_wth * dph_fld

        # For wide shallow flow, hydraulic radius ≈ depth
        radius_fld = dph_fld

        # Use floodplain gradient as slope
        # Simple approach: use first level gradient
        if hasattr(d2fldgrd, 'shape') and len(d2fldgrd.shape) > 1:
            slope_fld = d2fldgrd[iseq, 0]
        else:
            slope_fld = pminslp

        # Apply minimum slope
        slope_fld = max(abs(slope_fld), pminslp)

        # Manning's equation for floodplain
        if pmanfld > 0 and area_fld > 0:
            discharge_fld = (1.0 / pmanfld) * area_fld * (radius_fld ** (2.0/3.0)) * (slope_fld ** 0.5)
        else:
            discharge_fld = 0.0

        # Ensure non-negative
        discharge_fld = max(discharge_fld, 0.0)

        # Store result
        d2fldout[iseq] = discharge_fld


def calc_outflw_kine_mixed(
    nseqall, nseqriv, pslope, pgrv, pminslp,
    d2rivlen, d2rivwth, d2rivhgt, d2rivman, d2nxtdst, d2rivelv,
    d2rivdph, d2sfcelv, d2dwnelv,
    d2rivout, d2rivvel,
    i1next,
    calc_outflw_inertial=None
):
    """
    Mixed outflow scheme: use kinematic wave for steep slopes,
    local inertial for mild slopes

    This is the LSLPMIX option in Fortran code.

    Parameters:
    -----------
    nseqall : int
        Total number of river cells
    nseqriv : int
        Number of river cells (excluding mouth)
    pslope : float
        Threshold slope [m/m] for switching schemes
        Typically 0.005 (0.5%)
    pgrv : float
        Gravity [m/s²]
    pminslp : float
        Minimum slope [m/m]
    ... : (same as calc_outflw_kine)
    calc_outflw_inertial : function (optional)
        Function to calculate local inertial outflow
        If None, use kinematic wave for all cells

    Returns:
    --------
    None (modifies d2rivout and d2rivvel in place)
    """

    # If no inertial function provided, just use kinematic wave
    if calc_outflw_inertial is None:
        calc_outflw_kine(
            nseqall, nseqriv, pgrv, pminslp,
            d2rivlen, d2rivwth, d2rivhgt, d2rivman, d2nxtdst, d2rivelv,
            d2rivdph, d2sfcelv, d2dwnelv,
            d2rivout, d2rivvel,
            i1next
        )
        return

    # Process each cell and choose scheme based on slope
    for iseq in range(nseqall):

        # Calculate local slope
        if iseq < nseqriv:
            inext = i1next[iseq]
            if inext >= 0:
                slope = abs((d2sfcelv[iseq] - d2sfcelv[inext]) / d2nxtdst[iseq])
            else:
                slope = pminslp
        else:
            slope = abs((d2sfcelv[iseq] - d2dwnelv[iseq]) / d2nxtdst[iseq])

        # Choose scheme based on slope
        if slope > pslope:
            # Steep slope: use kinematic wave for this cell
            # Call kinematic wave for single cell
            # (This is simplified - in practice might need to vectorize better)
            pass  # TODO: Implement mixed scheme properly
        else:
            # Mild slope: use local inertial for this cell
            pass  # TODO: Implement mixed scheme properly

    # For now, fall back to pure kinematic wave
    # Full mixed scheme implementation would require refactoring calc_outflw
    print("  WARNING: Mixed scheme not fully implemented, using kinematic wave")
    calc_outflw_kine(
        nseqall, nseqriv, pgrv, pminslp,
        d2rivlen, d2rivwth, d2rivhgt, d2rivman, d2nxtdst, d2rivelv,
        d2rivdph, d2sfcelv, d2dwnelv,
        d2rivout, d2rivvel,
        i1next
    )
