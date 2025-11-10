"""
Bifurcation Channel Flow Module
Calculates flow in bifurcation channels (river splits)

Based on CMF_CALC_PTHOUT_MOD.F90
Handles river bifurcation where flow can split into multiple channels
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _calc_pthout_flow_core(
    npthout, npthlev, dt, pgrv,
    pth_upst, pth_down, pth_dst, pth_elv, pth_wth, pth_man,
    i2mask, d2sfcelv, d2sfcelv_pre,
    d1pthflw, d1pthflw_pre, d1pthflwsum
):
    """
    JIT-compiled core function for bifurcation flow calculation
    """
    # Reset bifurcation flow
    d1pthflw[:, :] = 0.0

    # Calculate flow for each bifurcation channel
    for ipth in prange(npthout):
        # Get upstream and downstream indices (1-based Fortran, convert to 0-based)
        iseqp = pth_upst[ipth]
        jseqp = pth_down[ipth]

        # Check if bifurcation is within domain and active
        if iseqp <= 0 or jseqp <= 0:
            continue

        # Convert to 0-based Python indices
        iseqp_py = iseqp - 1
        jseqp_py = jseqp - 1

        if i2mask[iseqp_py] > 0 or i2mask[jseqp_py] > 0:
            continue  # Skip if kinematic, dam, or no-bifurcation

        # Slope between upstream and downstream
        dslp = (d2sfcelv[iseqp_py] - d2sfcelv[jseqp_py]) / pth_dst[ipth]
        dslp = max(-0.005, min(0.005, dslp))  # Limit slope for stability

        # Calculate flow for each elevation level
        for ilev in range(npthlev):
            # Flow depth
            dflw = max(d2sfcelv[iseqp_py], d2sfcelv[jseqp_py]) - pth_elv[ipth, ilev]
            dflw = max(dflw, 0.0)

            # Previous flow depth
            dflw_pr = max(d2sfcelv_pre[iseqp_py], d2sfcelv_pre[jseqp_py]) - pth_elv[ipth, ilev]
            dflw_pr = max(dflw_pr, 0.0)

            # Semi-implicit flow depth (protect against negative value in sqrt)
            dflw_im = np.sqrt(max(dflw * dflw_pr, 0.0))
            dflw_im = max(dflw_im, np.sqrt(max(dflw * 0.01, 0.0)))

            # Calculate flow using local inertial equation
            if dflw_im > 1.0e-5:
                # Previous outflow (unit width)
                dout_pr = d1pthflw_pre[ipth, ilev] / pth_wth[ipth, ilev]

                # Local inertial equation
                numerator = dout_pr + pgrv * dt * dflw_im * dslp
                denominator = 1.0 + pgrv * dt * pth_man[ilev]**2 * abs(dout_pr) * dflw_im**(-7.0/3.0)

                d1pthflw[ipth, ilev] = pth_wth[ipth, ilev] * numerator / denominator
            else:
                d1pthflw[ipth, ilev] = 0.0

        # Sum flows across all levels
        pthflw_sum = 0.0
        for ilev in range(npthlev):
            pthflw_sum += d1pthflw[ipth, ilev]
        d1pthflwsum[ipth] = pthflw_sum


@njit(parallel=True, cache=True)
def _calc_pthout_limiter_core(
    nseqall, npthout, npthlev, dt,
    pth_upst, pth_down,
    d1pthflw, d1pthflwsum, d2storge
):
    """
    JIT-compiled core function for bifurcation storage limiter
    """
    for ipth in prange(npthout):
        if d1pthflwsum[ipth] != 0.0:
            iseqp = pth_upst[ipth]
            jseqp = pth_down[ipth]

            # Convert to 0-based indices
            iseqp_py = iseqp - 1
            jseqp_py = jseqp - 1

            # Check indices are valid
            if 0 <= iseqp_py < nseqall and 0 <= jseqp_py < nseqall:
                # Calculate limiter rate: 5% of minimum storage divided by flow volume
                min_storage = min(d2storge[iseqp_py], d2storge[jseqp_py])
                flow_volume = abs(d1pthflwsum[ipth] * dt)

                if flow_volume > 0.0:
                    rate = 0.05 * min_storage / flow_volume
                    rate = min(rate, 1.0)  # Cap at 1.0

                    # Apply limiter to all levels
                    for ilev in range(npthlev):
                        d1pthflw[ipth, ilev] = d1pthflw[ipth, ilev] * rate
                    d1pthflwsum[ipth] = d1pthflwsum[ipth] * rate


def calc_pthout(
    nseqall, npthout, npthlev, dt, pgrv,
    pth_upst, pth_down, pth_dst, pth_elv, pth_wth, pth_man,
    i2mask, d2rivelv, d2rivdph, d2sfcelv, d2sfcelv_pre,
    d1pthflw, d1pthflw_pre, d1pthflwsum, d2storge
):
    """
    Calculate flow in bifurcation channels

    Bifurcation channels are used to represent river splits, typically in deltas
    or areas where a river divides into multiple distributaries.

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    npthout : int
        Number of bifurcation channels
    npthlev : int
        Number of elevation levels for bifurcation
    dt : float
        Time step [s]
    pgrv : float
        Gravity acceleration [m/s2]
    pth_upst : ndarray (npthout,)
        Upstream cell index for each bifurcation
    pth_down : ndarray (npthout,)
        Downstream cell index for each bifurcation
    pth_dst : ndarray (npthout,)
        Distance for each bifurcation [m]
    pth_elv : ndarray (npthout, npthlev)
        Elevation for each level [m]
    pth_wth : ndarray (npthout, npthlev)
        Width for each level [m]
    pth_man : ndarray (npthlev,)
        Manning coefficient for bifurcation channels
    i2mask : ndarray (nseqall,)
        Mask for kinematic/dam/no-bifurcation cells
    d2rivelv : ndarray (nseqall,)
        River bed elevation [m]
    d2rivdph : ndarray (nseqall,)
        River depth [m]
    d2sfcelv : ndarray (nseqall,)
        Water surface elevation [m]
    d2sfcelv_pre : ndarray (nseqall,)
        Previous water surface elevation [m]
    d1pthflw : ndarray (npthout, npthlev)
        Bifurcation outflow [m3/s] (output)
    d1pthflw_pre : ndarray (npthout, npthlev)
        Previous bifurcation outflow [m3/s]
    d1pthflwsum : ndarray (npthout,)
        Sum of outflows across all levels [m3/s] (output)
    d2storge : ndarray (nseqall,)
        Total storage [m3] (river + floodplain)

    Returns:
    --------
    d1pthflwsum : ndarray (npthout,)
        Sum of outflows (updated in-place)
    """
    # NOTE: d2sfcelv_pre should already be set by calc_outflw
    # DO NOT recalculate here, as it would use wrong depth (d2rivdph instead of d2rivdph_pre)

    # Call JIT-compiled core function for flow calculation
    _calc_pthout_flow_core(
        npthout, npthlev, dt, pgrv,
        pth_upst, pth_down, pth_dst, pth_elv, pth_wth, pth_man,
        i2mask, d2sfcelv, d2sfcelv_pre,
        d1pthflw, d1pthflw_pre, d1pthflwsum
    )

    # Call JIT-compiled core function for storage limiter
    _calc_pthout_limiter_core(
        nseqall, npthout, npthlev, dt,
        pth_upst, pth_down,
        d1pthflw, d1pthflwsum, d2storge
    )

    return d1pthflwsum
