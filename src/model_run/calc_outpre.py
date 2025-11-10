"""
Reconstruct previous time-step outflow from storage

Implements CMF_CALC_OUTPRE from cmf_opt_outflw_mod.F90

This module reconstructs the previous time step outflow using diffusive wave
approximation. This is necessary when restarting from storage-only restart files
(LSTOONLY).

Based on CMF_OPT_OUTFLW_MOD.F90 (C) D.Yamazaki & E. Dutra
"""
import numpy as np


def calc_outpre(nseqall, nseqriv,
                d2rivlen, d2rivwth, d2rivman, d2pmanfld,
                d2nxtdst, d2rivelv, d2elevtn, d2dwnelv,
                d2rivdph, d2flddph, d2sfcelv, d2fldsto,
                d2rivout_pre, d2fldout_pre, d2rivdph_pre,
                i1next, pdstmth=10000.0):
    """
    Reconstruct previous time-step outflow from current storage

    Uses diffusive wave approximation (Manning equation with local slope)
    to estimate outflow at t-1 from storage at t=0.

    Parameters:
    -----------
    nseqall : int
        Total number of grid cells
    nseqriv : int
        Number of river cells
    d2rivlen : ndarray
        River length [m]
    d2rivwth : ndarray
        River width [m]
    d2rivman : ndarray
        River Manning roughness coefficient
    d2pmanfld : float
        Floodplain Manning roughness coefficient
    d2nxtdst : ndarray
        Distance to next cell [m]
    d2rivelv : ndarray
        River bed elevation [m]
    d2elevtn : ndarray
        Ground elevation [m]
    d2dwnelv : ndarray
        Downstream elevation [m] (for river mouths)
    d2rivdph : ndarray
        River depth [m]
    d2flddph : ndarray
        Floodplain depth [m]
    d2sfcelv : ndarray
        Surface water elevation [m]
    d2fldsto : ndarray
        Floodplain storage [m3]
    d2rivout_pre : ndarray (output)
        Reconstructed river outflow at t-1 [m3/s]
    d2fldout_pre : ndarray (output)
        Reconstructed floodplain outflow at t-1 [m3/s]
    d2rivdph_pre : ndarray (output)
        River depth at t-1 [m] (set equal to current depth)
    i1next : ndarray
        Next cell index
    pdstmth : float, optional
        Distance to river mouth [m] (default: 10000.0)

    Notes:
    ------
    This function uses the diffusive wave approximation:
        Q = A * (1/n) * D^(2/3) * |S|^(1/2) * sign(S)

    where:
        A = cross-section area
        n = Manning roughness
        D = flow depth
        S = water surface slope
    """

    # Initialize previous depth (set equal to current)
    d2rivdph_pre[:nseqall] = d2rivdph[:nseqall]

    # Calculate surface elevation
    d2sfcelv[:nseqall] = d2rivelv[:nseqall] + d2rivdph[:nseqall]

    # ========== Process normal river cells ==========
    for iseq in range(nseqriv):
        jseq = i1next[iseq]

        # Maximum surface elevation between current and next cell
        dsfcmax = max(d2sfcelv[iseq], d2sfcelv[jseq])

        # Water surface slope
        dslope = (d2sfcelv[iseq] - d2sfcelv[jseq]) / d2nxtdst[iseq]
        dslope_f = max(-0.005, min(0.005, dslope))  # Limit slope for stability

        # === River flow ===
        dflw = dsfcmax - d2rivelv[iseq]  # Flow depth
        darea = d2rivwth[iseq] * dflw     # Cross-section area

        if darea > 1.0e-5:
            # Manning equation with sign of slope
            d2rivout_pre[iseq] = darea * (1.0 / d2rivman[iseq]) * (dflw**(2.0/3.0)) * (abs(dslope)**0.5)

            # Apply sign of slope
            if dslope < 0.0:
                d2rivout_pre[iseq] = -d2rivout_pre[iseq]
        else:
            d2rivout_pre[iseq] = 0.0

        # === Floodplain flow ===
        dflw_f = max(dsfcmax - d2elevtn[iseq], 0.0)

        # Floodplain area (excluding river channel)
        dare_f = d2fldsto[iseq] / d2rivlen[iseq]
        dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

        if dare_f > 1.0e-5 and dflw_f > 1.0e-5:
            # Manning equation for floodplain
            d2fldout_pre[iseq] = dare_f * (1.0 / d2pmanfld) * (dflw_f**(2.0/3.0)) * (abs(dslope_f)**0.5)

            # Apply sign of slope
            if dslope_f < 0.0:
                d2fldout_pre[iseq] = -d2fldout_pre[iseq]
        else:
            d2fldout_pre[iseq] = 0.0

    # ========== Process river mouth cells ==========
    for iseq in range(nseqriv, nseqall):

        # Slope to downstream elevation
        dslope = (d2sfcelv[iseq] - d2dwnelv[iseq]) / pdstmth
        dslope_f = max(-0.005, min(0.005, dslope))

        # === River mouth flow ===
        dflw = d2rivdph[iseq]
        darea = d2rivwth[iseq] * dflw

        if darea > 1.0e-5:
            d2rivout_pre[iseq] = darea * (1.0 / d2rivman[iseq]) * (dflw**(2.0/3.0)) * (abs(dslope)**0.5)

            if dslope < 0.0:
                d2rivout_pre[iseq] = -d2rivout_pre[iseq]
        else:
            d2rivout_pre[iseq] = 0.0

        # === Floodplain mouth flow ===
        dflw_f = d2sfcelv[iseq] - d2elevtn[iseq]

        dare_f = d2fldsto[iseq] / d2rivlen[iseq]
        dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

        if dare_f > 1.0e-5 and dflw_f > 1.0e-5:
            d2fldout_pre[iseq] = dare_f * (1.0 / d2pmanfld) * (dflw_f**(2.0/3.0)) * (abs(dslope_f)**0.5)

            if dslope_f < 0.0:
                d2fldout_pre[iseq] = -d2fldout_pre[iseq]
        else:
            d2fldout_pre[iseq] = 0.0


def calc_pthout_pre(npthout, npthlev, pth_upst, pth_down, pth_dst,
                    pth_elv, pth_wth, pth_man,
                    d2rivdph, d2sfcelv, d1pthflw_pre):
    """
    Reconstruct previous bifurcation channel outflow

    Parameters:
    -----------
    npthout : int
        Number of bifurcation channels
    npthlev : int
        Number of elevation levels
    pth_upst : ndarray
        Upstream cell index for each channel
    pth_down : ndarray
        Downstream cell index for each channel
    pth_dst : ndarray
        Channel distance [m]
    pth_elv : ndarray
        Channel elevation at each level [m]
    pth_wth : ndarray
        Channel width at each level [m]
    pth_man : ndarray
        Channel Manning roughness
    d2rivdph : ndarray
        River depth [m]
    d2sfcelv : ndarray
        Surface water elevation [m]
    d1pthflw_pre : ndarray (output)
        Reconstructed bifurcation outflow at t-1 [m3/s]
    """

    # Process each bifurcation channel
    for ipth in range(npthout):
        # NOTE: pth_upst and pth_down are 1-based Fortran indices
        iseqp = pth_upst[ipth]   # 1-based Fortran index
        jseqp = pth_down[ipth]   # 1-based Fortran index

        # CRITICAL FIX: Convert to 0-based Python indices for array access
        iseqp_py = iseqp - 1
        jseqp_py = jseqp - 1

        # Maximum surface elevation
        dsfcmax = max(d2sfcelv[iseqp_py], d2sfcelv[jseqp_py])

        # Water surface slope (calculated ONCE for the channel)
        dslope = (d2sfcelv[iseqp_py] - d2sfcelv[jseqp_py]) / pth_dst[ipth]
        dslope = max(-0.005, min(0.005, dslope))  # Limit slope

        # CRITICAL FIX: Calculate flow for EACH level separately
        # This matches Fortran cmf_opt_outflw_mod.F90 lines 401-411
        for ilev in range(npthlev):
            # Flow depth from this level's elevation
            dflw = max(dsfcmax - pth_elv[ipth, ilev], 0.0)

            if dflw > 1.0e-5:
                # Manning equation for THIS level
                # Q = A * (1/n) * D^(2/3) * |S|^(1/2)
                # where A = width * depth, D = depth, n = Manning coefficient
                d1pthflw_pre[ipth, ilev] = (
                    pth_wth[ipth, ilev] * dflw *
                    (1.0 / pth_man[ilev]) *
                    (dflw**(2.0/3.0)) *
                    (abs(dslope)**0.5)
                )

                # Apply sign based on slope direction
                if dslope < 0.0:
                    d1pthflw_pre[ipth, ilev] = -d1pthflw_pre[ipth, ilev]
            else:
                d1pthflw_pre[ipth, ilev] = 0.0
