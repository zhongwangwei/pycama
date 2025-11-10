"""
Mixed outflow calculation scheme (kinematic + local inertial)

Implements CMF_CALC_OUTFLW_KINEMIX from cmf_opt_outflw_mod.F90

This module provides a mixed scheme that automatically selects between
kinematic wave and local inertial approaches based on grid cell properties
(typically slope/mask).

Based on CMF_OPT_OUTFLW_MOD.F90 (C) D.Yamazaki & E. Dutra
"""
import numpy as np


def calc_outflw_kinemix(nseqall, nseqriv, pgrv, pminslp, dt,
                        d2rivlen, d2rivwth, d2rivhgt, d2rivman,
                        d2nxtdst, d2rivelv, d2elevtn, d2dwnelv,
                        d2rivdph, d2rivdph_pre, d2sfcelv,
                        d2rivout_pre, d2rivout, d2rivvel,
                        i1next, i2mask, pdstmth=10000.0):
    """
    Calculate river outflow using mixed kinematic + local inertial scheme

    Uses local inertial for flat/gentle slopes (I2MASK==0) and kinematic wave
    for steep slopes (I2MASK!=0).

    Parameters:
    -----------
    nseqall : int
        Total number of grid cells
    nseqriv : int
        Number of river cells (non-mouth)
    pgrv : float
        Gravity acceleration [m/s2]
    pminslp : float
        Minimum slope
    dt : float
        Time step [s]
    d2rivlen : ndarray
        River length [m]
    d2rivwth : ndarray
        River width [m]
    d2rivhgt : ndarray
        River bank height [m]
    d2rivman : ndarray
        Manning roughness coefficient
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
    d2rivdph_pre : ndarray
        River depth at previous time step [m]
    d2sfcelv : ndarray
        Surface water elevation [m]
    d2rivout_pre : ndarray
        River outflow at previous time step [m3/s]
    d2rivout : ndarray (output)
        River outflow [m3/s]
    d2rivvel : ndarray (output)
        River velocity [m/s]
    i1next : ndarray
        Next cell index
    i2mask : ndarray
        Mask for scheme selection (0=local inertial, 1=kinematic)
    pdstmth : float, optional
        Distance to river mouth [m] (default: 10000.0)
    """

    # Calculate surface water elevation (already done externally, but ensure consistency)
    d2sfcelv[:nseqall] = d2rivelv[:nseqall] + d2rivdph[:nseqall]

    # Previous surface elevation
    d2sfcelv_pre = d2rivelv[:nseqall] + d2rivdph_pre[:nseqall]

    # Process normal river cells
    for iseq in range(nseqriv):
        jseq = i1next[iseq]

        if i2mask[iseq] == 0:
            # === Local inertial scheme (for gentle slopes) ===

            # Maximum surface elevation between current and next cell
            dsfcmax = max(d2sfcelv[iseq], d2sfcelv[jseq])
            dsfcmax_pre = max(d2sfcelv_pre[iseq], d2sfcelv_pre[jseq])

            # Water surface slope
            dslope = (d2sfcelv[iseq] - d2sfcelv[jseq]) / d2nxtdst[iseq]

            # River flow calculation
            dflw = dsfcmax - d2rivelv[iseq]  # Flow depth
            darea = d2rivwth[iseq] * dflw     # Cross-section area

            dflw_pre = dsfcmax_pre - d2rivelv[iseq]
            dflw_imp = max(np.sqrt(dflw * dflw_pre), 1.0e-6)  # Semi-implicit depth

            if dflw_imp > 1.0e-5 and darea > 1.0e-5:
                # Unit width outflow at previous time step
                dout_pre = d2rivout_pre[iseq] / d2rivwth[iseq]

                # Local inertial equation
                numerator = dout_pre + pgrv * dt * dflw_imp * dslope
                denominator = 1.0 + pgrv * dt * d2rivman[iseq]**2 * abs(dout_pre) * dflw_imp**(-7.0/3.0)

                d2rivout[iseq] = d2rivwth[iseq] * numerator / denominator
                d2rivvel[iseq] = d2rivout[iseq] / darea
            else:
                d2rivout[iseq] = 0.0
                d2rivvel[iseq] = 0.0

        else:
            # === Kinematic wave scheme (for steep slopes) ===

            # Bed slope
            dslope = (d2elevtn[iseq] - d2elevtn[jseq]) / d2nxtdst[iseq]
            dslope = max(dslope, pminslp)

            # Manning equation
            dvel = (1.0 / d2rivman[iseq]) * (dslope**0.5) * (d2rivdph[iseq]**(2.0/3.0))
            darea = d2rivwth[iseq] * d2rivdph[iseq]

            d2rivvel[iseq] = dvel
            d2rivout[iseq] = darea * dvel

    # Process river mouth cells
    for iseq in range(nseqriv, nseqall):

        if i2mask[iseq] == 0:
            # === Local inertial for river mouth ===

            # Slope to downstream elevation
            dslope = (d2sfcelv[iseq] - d2dwnelv[iseq]) / pdstmth

            # River mouth flow
            dflw = d2rivdph[iseq]
            darea = d2rivwth[iseq] * dflw

            dflw_pre = d2rivdph_pre[iseq]
            dflw_imp = max(np.sqrt(dflw * dflw_pre), 1.0e-6)

            if dflw_imp > 1.0e-5 and darea > 1.0e-5:
                dout_pre = d2rivout_pre[iseq] / d2rivwth[iseq]

                numerator = dout_pre + pgrv * dt * dflw_imp * dslope
                denominator = 1.0 + pgrv * dt * d2rivman[iseq]**2 * abs(dout_pre) * dflw_imp**(-7.0/3.0)

                d2rivout[iseq] = d2rivwth[iseq] * numerator / denominator
                d2rivvel[iseq] = d2rivout[iseq] / darea
            else:
                d2rivout[iseq] = 0.0
                d2rivvel[iseq] = 0.0

        else:
            # === Kinematic wave for river mouth ===

            dslope = pminslp
            dvel = (1.0 / d2rivman[iseq]) * (dslope**0.5) * (d2rivdph[iseq]**(2.0/3.0))
            darea = d2rivwth[iseq] * d2rivdph[iseq]

            d2rivvel[iseq] = dvel
            d2rivout[iseq] = darea * dvel


def calc_fldout_kinemix(nseqall, nseqriv, pgrv, pmanfld, dt,
                        d2rivlen, d2rivwth, d2rivhgt,
                        d2nxtdst, d2rivelv, d2elevtn,
                        d2rivdph, d2rivdph_pre, d2sfcelv,
                        d2flddph, d2fldsto, d2fldsto_pre,
                        d2fldout_pre, d2fldout, d2rivout,
                        i1next, i2mask):
    """
    Calculate floodplain outflow using mixed scheme

    Parameters:
    -----------
    nseqall : int
        Total number of grid cells
    nseqriv : int
        Number of river cells
    pgrv : float
        Gravity acceleration [m/s2]
    pmanfld : float
        Floodplain Manning roughness
    dt : float
        Time step [s]
    d2rivlen : ndarray
        River length [m]
    d2rivwth : ndarray
        River width [m]
    d2rivhgt : ndarray
        River bank height [m]
    d2nxtdst : ndarray
        Distance to next cell [m]
    d2rivelv : ndarray
        River bed elevation [m]
    d2elevtn : ndarray
        Ground elevation [m]
    d2rivdph : ndarray
        River depth [m]
    d2rivdph_pre : ndarray
        River depth at previous time step [m]
    d2sfcelv : ndarray
        Surface water elevation [m]
    d2flddph : ndarray
        Floodplain depth [m]
    d2fldsto : ndarray
        Floodplain storage [m3]
    d2fldsto_pre : ndarray
        Floodplain storage at previous time step [m3]
    d2fldout_pre : ndarray
        Floodplain outflow at previous time step [m3/s]
    d2fldout : ndarray (output)
        Floodplain outflow [m3/s]
    d2rivout : ndarray
        River outflow [m3/s] (for stabilization check)
    i1next : ndarray
        Next cell index
    i2mask : ndarray
        Mask for scheme selection
    """

    # Previous surface elevation and floodplain depth
    d2sfcelv_pre = d2rivelv[:nseqall] + d2rivdph_pre[:nseqall]
    d2flddph_pre = np.maximum(d2rivdph_pre[:nseqall] - d2rivhgt[:nseqall], 0.0)

    # Process normal river cells
    for iseq in range(nseqriv):
        jseq = i1next[iseq]

        if i2mask[iseq] == 0:
            # === Local inertial scheme ===

            dsfcmax = max(d2sfcelv[iseq], d2sfcelv[jseq])
            dsfcmax_pre = max(d2sfcelv_pre[iseq], d2sfcelv_pre[jseq])

            dslope = (d2sfcelv[iseq] - d2sfcelv[jseq]) / d2nxtdst[iseq]
            dslope_f = max(-0.005, min(0.005, dslope))  # Limit slope for stability

            # Floodplain flow depth
            dflw_f = max(dsfcmax - d2elevtn[iseq], 0.0)

            # Floodplain area (excluding river channel)
            dare_f = d2fldsto[iseq] / d2rivlen[iseq]
            dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

            # Previous time step values
            dflw_pre_f = dsfcmax_pre - d2elevtn[iseq]
            dflw_imp_f = max(np.sqrt(max(dflw_f * dflw_pre_f, 0.0)), 1.0e-6)

            dare_pre_f = d2fldsto_pre[iseq] / d2rivlen[iseq]
            dare_pre_f = max(dare_pre_f - d2flddph_pre[iseq] * d2rivwth[iseq], 1.0e-6)
            dare_imp_f = max(np.sqrt(dare_f * dare_pre_f), 1.0e-6)

            if dflw_imp_f > 1.0e-5 and dare_imp_f > 1.0e-5:
                dout_pre_f = d2fldout_pre[iseq]

                numerator = dout_pre_f + pgrv * dt * dare_imp_f * dslope_f
                denominator = 1.0 + pgrv * dt * pmanfld**2 * abs(dout_pre_f) * dflw_imp_f**(-4.0/3.0) / dare_imp_f

                d2fldout[iseq] = numerator / denominator
            else:
                d2fldout[iseq] = 0.0

            # Stabilization: prevent opposite flow directions
            if d2fldout[iseq] * d2rivout[iseq] < 0.0:
                d2fldout[iseq] = 0.0

        else:
            # === Kinematic wave scheme ===

            dslope = (d2elevtn[iseq] - d2elevtn[jseq]) / d2nxtdst[iseq]
            dslope_f = min(0.005, max(dslope, 0.0))  # Limit max slope

            dvel_f = (1.0 / pmanfld) * (dslope_f**0.5) * (d2flddph[iseq]**(2.0/3.0))

            dare_f = d2fldsto[iseq] / d2rivlen[iseq]
            dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

            d2fldout[iseq] = dare_f * dvel_f

    # Process river mouth cells
    for iseq in range(nseqriv, nseqall):

        if i2mask[iseq] == 0:
            # === Local inertial for floodplain mouth ===

            dslope = (d2sfcelv[iseq] - d2elevtn[iseq]) / d2rivlen[iseq]
            dslope_f = max(-0.005, min(0.005, dslope))

            dflw_f = d2sfcelv[iseq] - d2elevtn[iseq]

            dare_f = d2fldsto[iseq] / d2rivlen[iseq]
            dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

            dflw_pre_f = d2sfcelv_pre[iseq] - d2elevtn[iseq]
            dflw_imp_f = max(np.sqrt(max(dflw_f * dflw_pre_f, 0.0)), 1.0e-6)

            dare_pre_f = d2fldsto_pre[iseq] / d2rivlen[iseq]
            dare_pre_f = max(dare_pre_f - d2flddph_pre[iseq] * d2rivwth[iseq], 1.0e-6)
            dare_imp_f = max(np.sqrt(dare_f * dare_pre_f), 1.0e-6)

            if dflw_imp_f > 1.0e-5 and dare_imp_f > 1.0e-5:
                dout_pre_f = d2fldout_pre[iseq]

                numerator = dout_pre_f + pgrv * dt * dare_imp_f * dslope_f
                denominator = 1.0 + pgrv * dt * pmanfld**2 * abs(dout_pre_f) * dflw_imp_f**(-4.0/3.0) / dare_imp_f

                d2fldout[iseq] = numerator / denominator
            else:
                d2fldout[iseq] = 0.0

            # Stabilization
            if d2fldout[iseq] * d2rivout[iseq] < 0.0:
                d2fldout[iseq] = 0.0

        else:
            # === Kinematic wave for floodplain mouth ===

            dslope_f = min(0.005, 0.001)  # Use minimum slope
            dvel_f = (1.0 / pmanfld) * (dslope_f**0.5) * (d2flddph[iseq]**(2.0/3.0))

            dare_f = d2fldsto[iseq] / d2rivlen[iseq]
            dare_f = max(dare_f - d2flddph[iseq] * d2rivwth[iseq], 0.0)

            d2fldout[iseq] = dare_f * dvel_f
