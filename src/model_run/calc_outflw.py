"""
Outflow Calculation Module
Calculates river and floodplain discharge using local inertial equation

Based on CMF_CALC_OUTFLW_MOD.F90
Uses local inertial approximation [Bates et al., 2010, J.Hydrol.]
"""
import numpy as np
from numba import jit, prange
from .trace_debug import trace_outflw_input, trace_outflw_river, trace_outflw_flood


@jit(nopython=True, cache=True)
def _calc_surface_elevations(nseqall, d2rivelv, d2rivdph, d2rivdph_pre, d2rivhgt,
                              d2sfcelv, d2sfcelv_pre, d2flddph_pre):
    """Calculate water surface elevations (JIT-compiled)"""
    for iseq in range(nseqall):
        d2sfcelv[iseq] = d2rivelv[iseq] + d2rivdph[iseq]
        d2sfcelv_pre[iseq] = d2rivelv[iseq] + d2rivdph_pre[iseq]
        d2flddph_pre[iseq] = max(d2rivdph_pre[iseq] - d2rivhgt[iseq], 0.0)


@jit(nopython=True, cache=True)
def _update_downstream_elevations(nseqriv, nseqall, i1next, d2sfcelv, d2sfcelv_pre,
                                   d2dwnelv, d2dwnelv_pre):
    """Update downstream elevations (JIT-compiled)"""
    for iseq in range(nseqriv):
        jseq = i1next[iseq]
        if jseq >= 0 and jseq < nseqall:
            d2dwnelv[iseq] = d2sfcelv[jseq]
            d2dwnelv_pre[iseq] = d2sfcelv_pre[jseq]


@jit(nopython=True, cache=True, parallel=True)
def _calc_river_discharge(nseqriv, dt, pgrv,
                          d2rivwth, d2rivman, d2nxtdst, d2rivelv,
                          d2sfcelv, d2sfcelv_pre, d2dwnelv, d2dwnelv_pre,
                          d2rivout_pre, d2rivout, d2rivvel):
    """Calculate river discharge using local inertial equation (JIT-compiled, parallelized)"""
    for iseq in prange(nseqriv):
        # Surface elevation (max of current and downstream)
        dsfc = max(d2sfcelv[iseq], d2dwnelv[iseq])

        # Slope (water surface gradient)
        dslp = (d2sfcelv[iseq] - d2dwnelv[iseq]) / d2nxtdst[iseq]
        # NOTE: Fortran does NOT limit slope for river flow, only for floodplain flow

        # Flow cross-section depth
        dflw = dsfc - d2rivelv[iseq]
        dare = max(d2rivwth[iseq] * dflw, 1.0e-10)

        # Previous time step values
        dsfc_pr = max(d2sfcelv_pre[iseq], d2dwnelv_pre[iseq])
        dflw_pr = dsfc_pr - d2rivelv[iseq]
        # CRITICAL FIX: Protect against negative value in sqrt (match Fortran floodplain behavior)
        dflw_im = max(np.sqrt(max(dflw * dflw_pr, 0.0)), 1.0e-6)  # Semi-implicit flow depth

        # Previous outflow (unit width)
        dout_pr = d2rivout_pre[iseq] / d2rivwth[iseq]

        # Local inertial equation
        # Q(t+1) = (Q(t) + g*dt*h*S) / (1 + g*dt*n^2*|Q|*h^(-7/3))
        numerator = dout_pr + pgrv * dt * dflw_im * dslp
        denominator = 1.0 + pgrv * dt * d2rivman[iseq]**2 * abs(dout_pr) * dflw_im**(-7.0/3.0)

        dout = d2rivwth[iseq] * numerator / denominator

        # CRITICAL FIX: Match Fortran behavior - use OLD d2rivout for velocity calculation!
        # Fortran line 75: DVEL= D2RIVOUT(ISEQ,1) * DARE**(-1._JPRB)
        # This uses the OLD D2RIVOUT value (from previous timestep), NOT the new DOUT!
        dvel = d2rivout[iseq] / dare

        # Replace small depth locations with zero
        if dflw_im > 1.0e-5 and dare > 1.0e-5:
            d2rivout[iseq] = dout
            d2rivvel[iseq] = dvel
        else:
            d2rivout[iseq] = 0.0
            d2rivvel[iseq] = 0.0


@jit(nopython=True, cache=True, parallel=True)
def _calc_floodplain_discharge(nseqriv, dt, pgrv, pmanfld,
                                d2rivlen, d2rivwth, d2elevtn, d2nxtdst,
                                d2sfcelv, d2sfcelv_pre, d2dwnelv, d2dwnelv_pre,
                                d2flddph, d2flddph_pre, p2fldsto, d2fldsto_pre,
                                d2fldout_pre, d2fldout, d2rivout):
    """Calculate floodplain discharge (JIT-compiled, parallelized)"""
    for iseq in prange(nseqriv):
        dfsto = p2fldsto[iseq]
        dsfc = max(d2sfcelv[iseq], d2dwnelv[iseq])
        dslp = (d2sfcelv[iseq] - d2dwnelv[iseq]) / d2nxtdst[iseq]

        # Limit slope to avoid instability
        dslp = max(-0.005, min(0.005, dslp))

        # Flow depth and area
        dflw = max(dsfc - d2elevtn[iseq], 0.0)
        dare = dfsto / d2rivlen[iseq]
        # FIX #20: Fortran uses 0.0 (not 1.0e-6) for current timestep dare
        # Line 94: DARE = MAX( DARE - D2FLDDPH(ISEQ,1)*D2RIVWTH(ISEQ,1), 0._JPRB )
        dare = max(dare - d2flddph[iseq] * d2rivwth[iseq], 0.0)  # Remove river channel area

        # Previous time step
        dsfc_pr = max(d2sfcelv_pre[iseq], d2dwnelv_pre[iseq])
        dflw_pr = dsfc_pr - d2elevtn[iseq]
        dflw_im = max(np.sqrt(max(dflw * dflw_pr, 0.0)), 1.0e-6)

        dare_pr = d2fldsto_pre[iseq] / d2rivlen[iseq]
        dare_pr = max(dare_pr - d2flddph_pre[iseq] * d2rivwth[iseq], 1.0e-6)
        dare_im = max(np.sqrt(dare * dare_pr), 1.0e-6)

        # FIX #16 & #17: Check conditions BEFORE calculation (match Fortran order)
        # FIX #19: Use dare (current timestep) not dare_im (time-averaged) for threshold check
        # This matches Fortran: Mask=(DFLW_im>1.E-5 .and. DARE>1.E-5)
        if dflw_im > 1.0e-5 and dare > 1.0e-5:
            # Local inertial equation for floodplain
            dout_pr = d2fldout_pre[iseq]
            numerator = dout_pr + pgrv * dt * dare_im * dslp
            denominator = 1.0 + pgrv * dt * pmanfld**2 * abs(dout_pr) * dflw_im**(-4.0/3.0) / dare_im
            d2fldout[iseq] = numerator / denominator
        else:
            d2fldout[iseq] = 0.0

        # FIX #18: Match Fortran stabilization check (< not <=)
        # Check if river and floodplain flow in opposite directions
        if d2fldout[iseq] * d2rivout[iseq] < 0.0:
            d2fldout[iseq] = 0.0


@jit(nopython=True, cache=True, parallel=True)
def _calc_river_mouth_floodplain_discharge(nseqriv, nseqall, dt, pgrv, pmanfld, pdstmth,
                                             d2rivlen, d2rivwth, d2elevtn, d2sfcelv, d2sfcelv_pre,
                                             d2dwnelv, p2fldsto, d2fldsto_pre, d2flddph, d2flddph_pre,
                                             d2fldout_pre, d2fldout, d2rivout):
    """Calculate floodplain discharge at river mouth cells (JIT-compiled, parallelized)

    This corresponds to Fortran "floodplain mouth flow" loop (lines 142-172)
    """
    for iseq in prange(nseqriv, nseqall):
        dfsto = p2fldsto[iseq]
        dslp = (d2sfcelv[iseq] - d2dwnelv[iseq]) / pdstmth

        # Limit slope to avoid instability
        dslp = max(-0.005, min(0.005, dslp))

        # Flow depth and area
        dflw = d2sfcelv[iseq] - d2elevtn[iseq]
        dare = dfsto / d2rivlen[iseq]
        # FIX #21: Match Fortran for river mouth: DARE uses 0.0 as minimum (line 152)
        dare = max(dare - d2flddph[iseq] * d2rivwth[iseq], 0.0)

        # Previous time step
        dflw_pr = d2sfcelv_pre[iseq] - d2elevtn[iseq]
        dflw_im = max(np.sqrt(max(dflw * dflw_pr, 0.0)), 1.0e-6)

        dare_pr = d2fldsto_pre[iseq] / d2rivlen[iseq]
        dare_pr = max(dare_pr - d2flddph_pre[iseq] * d2rivwth[iseq], 1.0e-6)
        dare_im = max(np.sqrt(dare * dare_pr), 1.0e-6)

        # Match Fortran threshold check: DFLW_im>1.E-5 .and. DARE>1.E-5 (line 164)
        if dflw_im > 1.0e-5 and dare > 1.0e-5:
            # Local inertial equation for floodplain
            dout_pr = d2fldout_pre[iseq]
            numerator = dout_pr + pgrv * dt * dare_im * dslp
            denominator = 1.0 + pgrv * dt * pmanfld**2 * abs(dout_pr) * dflw_im**(-4.0/3.0) / dare_im
            d2fldout[iseq] = numerator / denominator
        else:
            d2fldout[iseq] = 0.0

        # Check if river and floodplain flow in opposite directions (line 168)
        if d2fldout[iseq] * d2rivout[iseq] < 0.0:
            d2fldout[iseq] = 0.0


@jit(nopython=True, cache=True)
def _limit_negative_flow(nseqriv, dt, d2storge, d2rivout, d2fldout):
    """Limit negative flow to prevent excessive drainage (JIT-compiled)

    Returns (neg_flow_count, neg_flow_limited)
    """
    neg_flow_count = 0
    neg_flow_limited = 0

    for iseq in range(nseqriv):
        total_outflow = d2rivout[iseq] + d2fldout[iseq]
        if total_outflow < 0:  # Negative flow (backwater)
            neg_flow_count += 1
            dout = max((-total_outflow) * dt, 1.0e-10)
            rate = min(0.05 * d2storge[iseq] / dout, 1.0)
            if rate < 1.0:
                neg_flow_limited += 1
            d2rivout[iseq] = d2rivout[iseq] * rate
            d2fldout[iseq] = d2fldout[iseq] * rate

    return neg_flow_count, neg_flow_limited


@jit(nopython=True, cache=True, parallel=True)
def _calc_river_mouth_discharge(nseqriv, nseqall, dt, pgrv, pdstmth,
                                 d2rivwth, d2rivman, d2rivelv,
                                 d2sfcelv, d2dwnelv, d2rivdph, d2rivdph_pre,
                                 d2rivout_pre, d2rivout, d2rivvel):
    """Calculate discharge at river mouth (JIT-compiled, parallelized)"""
    for iseq in prange(nseqriv, nseqall):
        dslp = (d2sfcelv[iseq] - d2dwnelv[iseq]) / pdstmth

        dflw = d2rivdph[iseq]
        dare = d2rivwth[iseq] * dflw
        dare = max(dare, 1.0e-10)

        dflw_pr = d2rivdph_pre[iseq]
        # CRITICAL FIX: Protect against negative value in sqrt
        dflw_im = max(np.sqrt(max(dflw * dflw_pr, 0.0)), 1.0e-6)

        dout_pr = d2rivout_pre[iseq] / d2rivwth[iseq]
        numerator = dout_pr + pgrv * dt * dflw_im * dslp
        denominator = 1.0 + pgrv * dt * d2rivman[iseq]**2 * abs(dout_pr) * dflw_im**(-7.0/3.0)

        dout = d2rivwth[iseq] * numerator / denominator
        dvel = dout / dare

        if dflw_im > 1.0e-5 and dare > 1.0e-5:
            d2rivout[iseq] = dout
            d2rivvel[iseq] = dvel
        else:
            d2rivout[iseq] = 0.0
            d2rivvel[iseq] = 0.0


def calc_outflw(
    nseqall, nseqriv, dt, pgrv, pmanfld, lfldout,
    i1next, d2rivlen, d2rivwth, d2rivhgt, d2rivman, d2nxtdst,
    d2rivelv, d2elevtn, d2rivdph, d2rivdph_pre,
    d2sfcelv, d2sfcelv_pre, d2dwnelv, d2dwnelv_pre,
    d2rivout, d2rivout_pre, d2rivvel,
    p2fldsto, d2fldsto_pre, d2flddph, d2flddph_pre,
    d2fldout, d2fldout_pre, d2storge
):
    """
    Calculate river and floodplain discharge using local inertial equation

    The local inertial equation is a simplified form of the Saint-Venant equations
    that balances pressure gradient, friction, and inertia terms.

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    nseqriv : int
        Number of river cells (excluding river mouths)
    dt : float
        Time step [s]
    pgrv : float
        Gravity acceleration [m/s2]
    pmanfld : float
        Manning coefficient for floodplain
    lfldout : bool
        Flag for floodplain discharge
    i1next : ndarray (nseqall,)
        Downstream cell index
    d2rivlen : ndarray (nseqall,)
        River length [m]
    d2rivwth : ndarray (nseqall,)
        River width [m]
    d2rivhgt : ndarray (nseqall,)
        Channel depth [m]
    d2rivman : ndarray (nseqall,)
        Manning coefficient for river
    d2nxtdst : ndarray (nseqall,)
        Distance to next cell [m]
    d2rivelv : ndarray (nseqall,)
        River bed elevation [m]
    d2elevtn : ndarray (nseqall,)
        Bank top elevation [m]
    d2rivdph : ndarray (nseqall,)
        River depth current [m]
    d2rivdph_pre : ndarray (nseqall,)
        River depth previous [m]
    d2sfcelv : ndarray (nseqall,)
        Surface elevation current [m]
    d2sfcelv_pre : ndarray (nseqall,)
        Surface elevation previous [m]
    d2dwnelv : ndarray (nseqall,)
        Downstream elevation current [m]
    d2dwnelv_pre : ndarray (nseqall,)
        Downstream elevation previous [m]
    d2rivout : ndarray (nseqall,)
        River outflow [m3/s]
    d2rivout_pre : ndarray (nseqall,)
        River outflow previous [m3/s]
    d2rivvel : ndarray (nseqall,)
        River velocity [m/s]
    p2fldsto : ndarray (nseqall,)
        Floodplain storage [m3]
    d2fldsto_pre : ndarray (nseqall,)
        Floodplain storage previous [m3]
    d2flddph : ndarray (nseqall,)
        Floodplain depth [m]
    d2flddph_pre : ndarray (nseqall,)
        Floodplain depth previous [m]
    d2fldout : ndarray (nseqall,)
        Floodplain outflow [m3/s]
    d2fldout_pre : ndarray (nseqall,)
        Floodplain outflow previous [m3/s]
    d2storge : ndarray (nseqall,)
        Total storage (river + floodplain) [m3]

    Updates (in-place):
    -------------------
    d2sfcelv, d2sfcelv_pre : Water surface elevation
    d2dwnelv, d2dwnelv_pre : Downstream elevation
    d2flddph_pre : Previous floodplain depth
    d2rivout : River discharge
    d2rivvel : River velocity
    d2fldout : Floodplain discharge (if lfldout)
    """

    # Calculate water surface elevation (JIT-optimized)
    _calc_surface_elevations(nseqall, d2rivelv, d2rivdph, d2rivdph_pre, d2rivhgt,
                              d2sfcelv, d2sfcelv_pre, d2flddph_pre)

    # Update downstream elevation (JIT-optimized)
    _update_downstream_elevations(nseqriv, nseqall, i1next, d2sfcelv, d2sfcelv_pre,
                                   d2dwnelv, d2dwnelv_pre)

    # === Calculate river discharge for normal cells (JIT-optimized) ===
    # Check if tracing is enabled
    from .trace_debug import get_tracer
    _tracer = get_tracer()

    if _tracer.enabled:
        # Use slow version with detailed tracing
        for iseq in range(nseqriv):
            # Trace input state
            trace_outflw_input(iseq,
                rivdph=d2rivdph[iseq],
                rivdph_pre=d2rivdph_pre[iseq],
                sfcelv=d2sfcelv[iseq],
                sfcelv_pre=d2sfcelv_pre[iseq],
                dwnelv=d2dwnelv[iseq],
                dwnelv_pre=d2dwnelv_pre[iseq],
                rivout_pre=d2rivout_pre[iseq],
                rivwth=d2rivwth[iseq],
                rivman=d2rivman[iseq],
                nxtdst=d2nxtdst[iseq],
                rivelv=d2rivelv[iseq],
                storge=d2storge[iseq]
            )

            # Surface elevation (max of current and downstream)
            dsfc = max(d2sfcelv[iseq], d2dwnelv[iseq])
            dslp = (d2sfcelv[iseq] - d2dwnelv[iseq]) / d2nxtdst[iseq]
            # NOTE: Fortran does NOT limit slope for river flow
            dflw = dsfc - d2rivelv[iseq]
            dare = max(d2rivwth[iseq] * dflw, 1.0e-10)
            dsfc_pr = max(d2sfcelv_pre[iseq], d2dwnelv_pre[iseq])
            dflw_pr = dsfc_pr - d2rivelv[iseq]
            dflw_im = max(np.sqrt(dflw * dflw_pr), 1.0e-6)
            dout_pr = d2rivout_pre[iseq] / d2rivwth[iseq]
            numerator = dout_pr + pgrv * dt * dflw_im * dslp
            denominator = 1.0 + pgrv * dt * d2rivman[iseq]**2 * abs(dout_pr) * dflw_im**(-7.0/3.0)
            dout = d2rivwth[iseq] * numerator / denominator
            dvel = dout / dare

            # Trace intermediate calculation
            small_depth_check = (dflw_im > 1.0e-5 and dare > 1.0e-5)
            trace_outflw_river(iseq,
                dsfc=dsfc, dslp=dslp, dflw=dflw, dare=dare,
                dsfc_pr=dsfc_pr, dflw_pr=dflw_pr, dflw_im=dflw_im,
                dout_pr=dout_pr, numerator=numerator, denominator=denominator,
                dout_calc=dout, dvel_calc=dvel,
                small_depth_check=small_depth_check,
                threshold_depth=1.0e-5, threshold_area=1.0e-5
            )

            if small_depth_check:
                d2rivout[iseq] = dout
                d2rivvel[iseq] = dvel
            else:
                d2rivout[iseq] = 0.0
                d2rivvel[iseq] = 0.0
    else:
        # Use fast JIT-compiled version (production mode)
        _calc_river_discharge(nseqriv, dt, pgrv,
                              d2rivwth, d2rivman, d2nxtdst, d2rivelv,
                              d2sfcelv, d2sfcelv_pre, d2dwnelv, d2dwnelv_pre,
                              d2rivout_pre, d2rivout, d2rivvel)

    # === Calculate floodplain discharge (optional, JIT-optimized) ===
    if lfldout:
        _calc_floodplain_discharge(nseqriv, dt, pgrv, pmanfld,
                                    d2rivlen, d2rivwth, d2elevtn, d2nxtdst,
                                    d2sfcelv, d2sfcelv_pre, d2dwnelv, d2dwnelv_pre,
                                    d2flddph, d2flddph_pre, p2fldsto, d2fldsto_pre,
                                    d2fldout_pre, d2fldout, d2rivout)

    # === Calculate discharge at river mouth (JIT-optimized) ===
    pdstmth = 10000.0  # Default downstream distance at mouth
    _calc_river_mouth_discharge(nseqriv, nseqall, dt, pgrv, pdstmth,
                                 d2rivwth, d2rivman, d2rivelv,
                                 d2sfcelv, d2dwnelv, d2rivdph, d2rivdph_pre,
                                 d2rivout_pre, d2rivout, d2rivvel)

    # === Calculate floodplain discharge at river mouth (JIT-optimized) ===
    # FIX #22: Add missing river mouth floodplain calculation (Fortran lines 142-172)
    if lfldout:
        _calc_river_mouth_floodplain_discharge(nseqriv, nseqall, dt, pgrv, pmanfld, pdstmth,
                                                 d2rivlen, d2rivwth, d2elevtn, d2sfcelv, d2sfcelv_pre,
                                                 d2dwnelv, p2fldsto, d2fldsto_pre, d2flddph, d2flddph_pre,
                                                 d2fldout_pre, d2fldout, d2rivout)
