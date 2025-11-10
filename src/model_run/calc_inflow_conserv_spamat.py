"""
Inflow calculation with water conservation using sparse matrix optimization

This module implements the LSPAMAT (sparse matrix) version of inflow calculation.
Instead of iterating over all cells to find upstream neighbors, it uses pre-built
sparse matrices (I1UPST, I1UPN, etc.) to directly access upstream cells.

This approach:
1. Avoids OMP_ATOMIC operations in parallel code
2. Reduces memory access patterns for better cache performance
3. Enables efficient vectorization for large-scale simulations

Based on CMF_CALC_INFLOW_LSPAMAT from cmf_calc_outflw_mod.F90 (C) D.Yamazaki & E. Dutra
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def calc_inflow_spamat_core(
    nseqall, nseqriv, dt,
    i1next, i1upst, i1upn,
    d2rivsto, d2fldsto,
    d2rivout, d2fldout,
    d2rivinf, d2fldinf,
    p2stoout, d2rate
):
    """
    Core sparse matrix inflow calculation (JIT-compiled for speed, parallelized)

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    nseqriv : int
        Number of river cells
    dt : float
        Time step [s]
    i1next : ndarray (nseqall,)
        Downstream cell index
    i1upst : ndarray (nseqall, upnmax)
        Upstream cell indices (sparse matrix)
    i1upn : ndarray (nseqall,)
        Number of upstream cells for each cell
    d2rivsto, d2fldsto : ndarray (nseqall,)
        River and floodplain storage [m3]
    d2rivout, d2fldout : ndarray (nseqall,)
        River and floodplain outflow [m3/s] (modified in-place)
    d2rivinf, d2fldinf : ndarray (nseqall,)
        River and floodplain inflow [m3/s] (output)
    p2stoout : ndarray (nseqall,)
        Total outflow volume [m3] (workspace)
    d2rate : ndarray (nseqall,)
        Conservation correction factor (output)
    """

    # ================================================
    # Step 1: Initialize arrays
    # ================================================
    for iseq in prange(nseqall):
        d2rivinf[iseq] = 0.0
        d2fldinf[iseq] = 0.0
        p2stoout[iseq] = 0.0
        d2rate[iseq] = 1.0

    # ================================================
    # Step 2: Calculate total outflow volume (P2STOOUT)
    # Using sparse matrix: only consider upstream neighbors
    # ================================================

    # For each cell, calculate outflow from this cell and backflow from downstream
    for iseq in prange(nseqall):
        # Positive outflow from current cell
        p2stoout[iseq] = max(d2rivout[iseq], 0.0) + max(d2fldout[iseq], 0.0)

        # Negative outflow (backflow) from upstream cells
        if i1upn[iseq] > 0:
            for inum in range(i1upn[iseq]):
                jseq = i1upst[iseq, inum]
                if 0 <= jseq < nseqall:
                    p2stoout[iseq] += max(-d2rivout[jseq], 0.0) + max(-d2fldout[jseq], 0.0)

        # Convert to volume
        p2stoout[iseq] *= dt

    # ================================================
    # Step 3: Calculate conservation correction factor
    # ================================================
    for iseq in prange(nseqall):
        if p2stoout[iseq] > 1.0e-10:
            total_sto = d2rivsto[iseq] + d2fldsto[iseq]
            d2rate[iseq] = min(total_sto / p2stoout[iseq], 1.0)

    # ================================================
    # Step 4: Apply correction to outflows
    # ================================================

    # Normal river cells
    for iseq in range(nseqriv):
        jseq = i1next[iseq]

        if d2rivout[iseq] > 0.0:
            # Positive flow: use current cell's rate
            d2rivout[iseq] *= d2rate[iseq]
            d2fldout[iseq] *= d2rate[iseq]
        else:
            # Negative flow: use downstream cell's rate
            if 0 <= jseq < nseqall:
                d2rivout[iseq] *= d2rate[jseq]
                d2fldout[iseq] *= d2rate[jseq]

    # River mouth cells
    for iseq in prange(nseqriv, nseqall):
        d2rivout[iseq] *= d2rate[iseq]
        d2fldout[iseq] *= d2rate[iseq]

    # ================================================
    # Step 5: Calculate inflows using sparse matrix
    # ================================================

    # For each cell, accumulate inflow from all upstream cells
    for iseq in prange(nseqall):
        if i1upn[iseq] > 0:
            for inum in range(i1upn[iseq]):
                jseq = i1upst[iseq, inum]
                if 0 <= jseq < nseqall:
                    d2rivinf[iseq] += d2rivout[jseq]
                    d2fldinf[iseq] += d2fldout[jseq]


@njit
def calc_inflow_spamat_bifurcation(
    npthout, nseqall, dt,
    pth_upst, pth_down, i2mask,
    i1p_out, i1p_outn, i1p_inf, i1p_infn,
    d1pthflwsum, d2rate, p2stoout, p2pthout
):
    """
    Apply sparse matrix optimization to bifurcation channels

    Parameters:
    -----------
    npthout : int
        Number of bifurcation channels
    nseqall : int
        Total number of cells
    dt : float
        Time step [s]
    pth_upst, pth_down : ndarray (npthout,)
        Upstream and downstream cell indices (1-based Fortran indices)
    i2mask : ndarray (nseqall,)
        Cell mask
    i1p_out : ndarray (nseqall, onmax)
        Sparse matrix: outgoing bifurcation channel indices
    i1p_outn : ndarray (nseqall,)
        Number of outgoing bifurcation channels
    i1p_inf : ndarray (nseqall, inmax)
        Sparse matrix: incoming bifurcation channel indices
    i1p_infn : ndarray (nseqall,)
        Number of incoming bifurcation channels
    d1pthflwsum : ndarray (npthout,)
        Bifurcation channel flow [m3/s] (modified in-place)
    d2rate : ndarray (nseqall,)
        Conservation correction factor
    p2stoout : ndarray (nseqall,)
        Total outflow volume [m3] (modified in-place)
    p2pthout : ndarray (nseqall,)
        Bifurcation outflow per cell [m3/s] (output)
    """

    # Initialize
    for iseq in range(nseqall):
        p2pthout[iseq] = 0.0

    # ================================================
    # Step 1: Add bifurcation outflow to P2STOOUT using sparse matrix
    # ================================================
    for iseq in range(nseqall):
        # Outgoing bifurcation channels
        if i1p_outn[iseq] > 0:
            for inum in range(i1p_outn[iseq]):
                jpth = i1p_out[iseq, inum]
                if 0 <= jpth < npthout:
                    p2stoout[iseq] += max(d1pthflwsum[jpth], 0.0) * dt

        # Incoming bifurcation channels (backflow)
        if i1p_infn[iseq] > 0:
            for inum in range(i1p_infn[iseq]):
                jpth = i1p_inf[iseq, inum]
                if 0 <= jpth < npthout:
                    p2stoout[iseq] += max(-d1pthflwsum[jpth], 0.0) * dt

    # ================================================
    # Step 2: Apply correction to bifurcation flows
    # ================================================
    for ipth in range(npthout):
        # NOTE: pth_upst and pth_down are 1-based Fortran indices
        iseqp = pth_upst[ipth]
        jseqp = pth_down[ipth]

        # Skip invalid channels
        if iseqp <= 0 or jseqp <= 0:
            continue
        if iseqp > nseqall or jseqp > nseqall:
            continue

        # Convert to 0-based Python indices
        iseqp_py = iseqp - 1
        jseqp_py = jseqp - 1

        # Skip masked channels
        if i2mask is not None:
            if i2mask[iseqp_py] > 0 or i2mask[jseqp_py] > 0:
                continue

        # Apply correction
        if d1pthflwsum[ipth] >= 0.0:
            # Positive flow: use upstream rate
            d1pthflwsum[ipth] *= d2rate[iseqp_py]
        else:
            # Negative flow: use downstream rate
            d1pthflwsum[ipth] *= d2rate[jseqp_py]

    # ================================================
    # Step 3: Calculate bifurcation outflow per cell using sparse matrix
    # ================================================
    for iseq in range(nseqall):
        # Outgoing bifurcation channels
        if i1p_outn[iseq] > 0:
            for inum in range(i1p_outn[iseq]):
                jpth = i1p_out[iseq, inum]
                if 0 <= jpth < npthout:
                    p2pthout[iseq] += d1pthflwsum[jpth]

        # Incoming bifurcation channels (subtract)
        if i1p_infn[iseq] > 0:
            for inum in range(i1p_infn[iseq]):
                jpth = i1p_inf[iseq, inum]
                if 0 <= jpth < npthout:
                    p2pthout[iseq] -= d1pthflwsum[jpth]


def calc_inflow_with_conservation_spamat(
    nseqall, nseqriv, dt,
    i1next, i1upst, i1upn,
    d2rivsto, d2fldsto,
    d2rivout, d2fldout, d2pthout,
    d2rivinf, d2fldinf,
    lpthout=False, npthout=0,
    pth_upst=None, pth_down=None,
    d1pthflwsum=None, i2mask=None,
    i1p_out=None, i1p_outn=None,
    i1p_inf=None, i1p_infn=None,
    d2runoff=None
):
    """
    Calculate inflow with water conservation using sparse matrix optimization

    This is the LSPAMAT version that uses pre-built sparse matrices to avoid
    iterating over all cells when looking for upstream neighbors.

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    nseqriv : int
        Number of river cells
    dt : float
        Time step [s]
    i1next : ndarray (nseqall,)
        Downstream cell index
    i1upst : ndarray (nseqall, upnmax)
        Sparse matrix: upstream cell indices
    i1upn : ndarray (nseqall,)
        Number of upstream cells
    d2rivsto, d2fldsto : ndarray (nseqall,)
        River and floodplain storage [m3]
    d2rivout, d2fldout : ndarray (nseqall,)
        River and floodplain outflow [m3/s] (modified in-place)
    d2pthout : ndarray (nseqall,)
        Bifurcation outflow [m3/s] (output)
    d2rivinf, d2fldinf : ndarray (nseqall,)
        River and floodplain inflow [m3/s] (output)
    lpthout : bool
        Enable bifurcation channels
    npthout : int
        Number of bifurcation channels
    pth_upst, pth_down : ndarray
        Bifurcation channel endpoints (1-based)
    d1pthflwsum : ndarray
        Bifurcation flow [m3/s] (modified in-place)
    i2mask : ndarray
        Cell mask
    i1p_out, i1p_outn : ndarray
        Sparse matrix: outgoing bifurcation channels
    i1p_inf, i1p_infn : ndarray
        Sparse matrix: incoming bifurcation channels
    d2runoff : ndarray, optional
        Runoff input [m/s] (included in conservation)

    Returns:
    --------
    d2rate : ndarray
        Conservation correction factor
    """

    # Workspace arrays
    p2stoout = np.zeros(nseqall, dtype=np.float64)
    d2rate = np.ones(nseqall, dtype=np.float64)

    # Core inflow calculation with sparse matrix
    calc_inflow_spamat_core(
        nseqall, nseqriv, dt,
        i1next, i1upst, i1upn,
        d2rivsto, d2fldsto,
        d2rivout, d2fldout,
        d2rivinf, d2fldinf,
        p2stoout, d2rate
    )

    # Handle bifurcation channels if enabled
    if lpthout and npthout > 0 and pth_upst is not None:
        p2pthout = np.zeros(nseqall, dtype=np.float64)

        calc_inflow_spamat_bifurcation(
            npthout, nseqall, dt,
            pth_upst, pth_down, i2mask,
            i1p_out, i1p_outn, i1p_inf, i1p_infn,
            d1pthflwsum, d2rate, p2stoout, p2pthout
        )

        # Copy to output
        d2pthout[:] = p2pthout

    return d2rate


def build_sparse_upstream_matrix(nseqall, nseqriv, i1next):
    """
    Build sparse upstream connectivity matrix

    This creates the I1UPST and I1UPN arrays that store which cells
    flow into each cell, avoiding the need to search all cells.

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    nseqriv : int
        Number of river cells
    i1next : ndarray (nseqall,)
        Downstream cell index

    Returns:
    --------
    i1upst : ndarray (nseqall, upnmax)
        Upstream cell indices (sparse matrix)
    i1upn : ndarray (nseqall,)
        Number of upstream cells
    upnmax : int
        Maximum number of upstream cells
    """

    # First pass: count upstream cells
    upcount = np.zeros(nseqall, dtype=np.int32)

    for iseq in range(nseqriv):
        jseq = i1next[iseq]
        if 0 <= jseq < nseqall:
            upcount[jseq] += 1

    upnmax = np.max(upcount)
    print(f"  Sparse matrix: maximum upstream cells = {upnmax}")

    # Allocate sparse matrix
    i1upst = np.full((nseqall, upnmax), -1, dtype=np.int32)
    i1upn = np.zeros(nseqall, dtype=np.int32)

    # Second pass: fill sparse matrix
    for iseq in range(nseqriv):
        jseq = i1next[iseq]
        if 0 <= jseq < nseqall:
            idx = i1upn[jseq]
            i1upst[jseq, idx] = iseq
            i1upn[jseq] += 1

    return i1upst, i1upn, upnmax


def build_sparse_bifurcation_matrix(nseqall, npthout, pth_upst, pth_down):
    """
    Build sparse bifurcation connectivity matrix

    Parameters:
    -----------
    nseqall : int
        Total number of cells
    npthout : int
        Number of bifurcation channels
    pth_upst, pth_down : ndarray (npthout,)
        Bifurcation endpoints (1-based Fortran indices)

    Returns:
    --------
    i1p_out, i1p_outn : ndarray
        Outgoing bifurcation channel sparse matrix
    i1p_inf, i1p_infn : ndarray
        Incoming bifurcation channel sparse matrix
    """

    # Count outgoing and incoming bifurcation channels per cell
    ocount = np.zeros(nseqall, dtype=np.int32)
    icount = np.zeros(nseqall, dtype=np.int32)

    for ipth in range(npthout):
        iseqp = pth_upst[ipth]  # 1-based
        jseqp = pth_down[ipth]  # 1-based

        if iseqp > 0 and jseqp > 0 and iseqp <= nseqall and jseqp <= nseqall:
            iseqp_py = iseqp - 1  # Convert to 0-based
            jseqp_py = jseqp - 1
            ocount[iseqp_py] += 1
            icount[jseqp_py] += 1

    onmax = np.max(ocount) if len(ocount) > 0 else 0
    inmax = np.max(icount) if len(icount) > 0 else 0

    print(f"  Bifurcation sparse matrix: max outgoing={onmax}, max incoming={inmax}")

    # Allocate sparse matrices
    i1p_out = np.zeros((nseqall, max(1, onmax)), dtype=np.int32)
    i1p_inf = np.zeros((nseqall, max(1, inmax)), dtype=np.int32)
    i1p_outn = np.zeros(nseqall, dtype=np.int32)
    i1p_infn = np.zeros(nseqall, dtype=np.int32)

    # Fill sparse matrices
    for ipth in range(npthout):
        iseqp = pth_upst[ipth]
        jseqp = pth_down[ipth]

        if iseqp > 0 and jseqp > 0 and iseqp <= nseqall and jseqp <= nseqall:
            iseqp_py = iseqp - 1
            jseqp_py = jseqp - 1

            # Outgoing from iseqp
            idx = i1p_outn[iseqp_py]
            i1p_out[iseqp_py, idx] = ipth
            i1p_outn[iseqp_py] += 1

            # Incoming to jseqp
            idx = i1p_infn[jseqp_py]
            i1p_inf[jseqp_py, idx] = ipth
            i1p_infn[jseqp_py] += 1

    return i1p_out, i1p_outn, i1p_inf, i1p_infn
