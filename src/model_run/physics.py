"""
Physics module for CaMa-Flood model run
Handles hydraulic calculations and river routing

Based on CMF_CTRL_PHYSICS_MOD.F90 and CMF_CALC_*.F90

Implements the CaMa-Flood hydraulic routing using:
- calc_fldstg: Floodplain stage calculation
- calc_outflw: River and floodplain discharge calculation
- calc_stonxt: Storage update calculation
- calc_pthout: Bifurcation flow calculation (optional)
"""
import numpy as np
import os
from numba import njit
from .calc_fldstg_optimized import calc_fldstg_optimized as calc_fldstg
from .calc_outflw import calc_outflw
from .calc_outflw_kine import calc_outflw_kine, calc_fldout_kine
from .calc_outflw_kinemix import calc_outflw_kinemix, calc_fldout_kinemix
from .calc_outpre import calc_outpre, calc_pthout_pre
from .calc_stonxt import calc_inflow, calc_stonxt, save_vars_pre
from .calc_pthout import calc_pthout
from .calc_inflow_conserv_optimized import calc_inflow_with_conservation_vectorized as calc_inflow_with_conservation
from .calc_inflow_conserv_spamat import (
    calc_inflow_with_conservation_spamat,
    build_sparse_upstream_matrix,
    build_sparse_bifurcation_matrix
)
from .trace_debug import init_tracer, get_tracer


@njit(cache=True)
def _calculate_adaptive_timestep_core(nseqriv, nseqall, pcadp, pgrv, pdstmth,
                                       d2rivdph, d2nxtdst, dt_default):
    """
    JIT-compiled core function for adaptive timestep calculation

    Parameters:
    -----------
    nseqriv : int
        Number of river cells (excluding mouth cells)
    nseqall : int
        Total number of cells
    pcadp : float
        Courant coefficient
    pgrv : float
        Gravity acceleration [m/s2]
    pdstmth : float
        Distance at river mouth [m]
    d2rivdph : ndarray
        River depth [m]
    d2nxtdst : ndarray
        Distance to downstream cell [m]
    dt_default : float
        Default time step [s]

    Returns:
    --------
    dt_min : float
        Minimum time step based on CFL condition [s]
    """
    dt_min = dt_default

    # Calculate for river cells (excluding mouth cells)
    for iseq in range(nseqriv):
        # Get water depth (minimum 0.01 m to prevent instability)
        ddph = max(d2rivdph[iseq], 0.01)

        # Get distance to downstream cell
        ddst = d2nxtdst[iseq]

        # Calculate Courant-limited time step
        # DT_COURANT = PCADP * DX / sqrt(g * h)
        dt_courant = pcadp * ddst / np.sqrt(pgrv * ddph)

        # Update minimum time step
        dt_min = min(dt_min, dt_courant)

    # Calculate for river mouth cells (use pdstmth for distance)
    for iseq in range(nseqriv, nseqall):
        # Get water depth
        ddph = max(d2rivdph[iseq], 0.01)

        # Use fixed distance for mouth cells
        ddst = pdstmth

        # Calculate Courant-limited time step
        dt_courant = pcadp * ddst / np.sqrt(pgrv * ddph)

        # Update minimum time step
        dt_min = min(dt_min, dt_courant)

    return dt_min


class CaMaPhysics:
    """CaMa-Flood physics and hydraulic calculations"""

    def __init__(self, nml, nx, ny, nseqall, nseqmax):
        """
        Initialize physics module

        Parameters:
        -----------
        nml : Namelist object
            Namelist configuration
        nx, ny : int
            Grid dimensions
        nseqall : int
            Total number of active river cells
        nseqmax : int
            Maximum number of river cells
        """
        self.nml = nml
        self.nx = nx
        self.ny = ny
        self.nseqall = nseqall
        self.nseqmax = nseqmax

        # Read physics options from namelist
        self._read_physics_config()

        # Initialize state variables
        self._initialize_state_variables()

        # Initialize parameters
        self._initialize_parameters()

    def _read_physics_config(self):
        """Read physics configuration from namelist"""
        # Physics switches - read from MODEL_RUN section
        self.ladpstp = self.nml.get('MODEL_RUN', 'ladpstp', True)   # Adaptive time step
        self.lfplain = self.nml.get('MODEL_RUN', 'lfplain', True)   # Floodplain
        self.lkine = self.nml.get('MODEL_RUN', 'lkine', False)      # Kinematic wave
        self.lslpmix = self.nml.get('MODEL_RUN', 'lslpmix', False)  # Mixed scheme (slope-dependent)
        self.lfldout = self.nml.get('MODEL_RUN', 'lfldout', True)   # Floodplain discharge
        self.lpthout = self.nml.get('MODEL_RUN', 'lpthout', False)  # Bifurcation

        # Dam operation flag: check NRUNVER first (Fortran-compatible), then MODEL_RUN (backward compatibility)
        self.ldamout = self.nml.get('NRUNVER', 'LDAMOUT', self.nml.get('MODEL_RUN', 'ldamout', False))

        self.lgdwdly = self.nml.get('MODEL_RUN', 'lgdwdly', False)  # Groundwater delay
        self.lrosplit = self.nml.get('MODEL_RUN', 'lrosplit', False)  # Runoff separation
        self.lstoonly = self.nml.get('MODEL_RUN', 'lstoonly', False)  # Storage-only restart
        self.ldebug = self.nml.get('MODEL_RUN', 'ldebug', False)    # Debug mode (NaN checks)
        self.lconserv = self.nml.get('MODEL_RUN', 'lconserv', True)  # Water conservation check
        self.lspamat = self.nml.get('MODEL_RUN', 'lspamat', True)   # Sparse matrix optimization
        self.lwevap = self.nml.get('MODEL_RUN', 'lwevap', False)    # Water evaporation extraction
        self.lwevapfix = self.nml.get('MODEL_RUN', 'lwevapfix', False)  # Water balance closure
        self.lwextractriv = self.nml.get('MODEL_RUN', 'lwextractriv', False)  # Extract from rivers

        # Tracing and debugging
        self.ltrace = self.nml.get('MODEL_RUN', 'ltrace', False)    # Enable cell tracing
        self.trace_cells = self.nml.get('MODEL_RUN', 'trace_cells', [])  # List of cells to trace

        # Physical parameters - read from MODEL_RUN section
        self.pmanriv = self.nml.get('MODEL_RUN', 'pmanriv', 0.03)    # Manning coeff river
        self.pmanfld = self.nml.get('MODEL_RUN', 'pmanfld', 0.10)    # Manning coeff floodplain
        self.pgrv = self.nml.get('MODEL_RUN', 'pgrv', 9.8)           # Gravity
        self.pdstmth = self.nml.get('MODEL_RUN', 'pdstmth', 10000.0) # Downstream dist at mouth
        self.pcadp = self.nml.get('MODEL_RUN', 'pcadp', 0.7)         # CFL coefficient
        self.pminslp = self.nml.get('MODEL_RUN', 'pminslp', 1.0e-5)  # Minimum slope
        self.pgdwsto = self.nml.get('MODEL_RUN', 'pgdwsto', 100.0)   # Groundwater storage time constant [days]

    def _initialize_state_variables(self):
        """Initialize state variables"""
        # River storage and flow variables
        self.d2rivsto = np.zeros(self.nseqmax, dtype=np.float64)  # River storage [m3]
        self.d2rivdph = np.zeros(self.nseqmax, dtype=np.float64)  # River depth [m]
        self.d2rivout = np.zeros(self.nseqmax, dtype=np.float64)  # River outflow [m3/s]
        self.d2rivvel = np.zeros(self.nseqmax, dtype=np.float64)  # River velocity [m/s]
        self.d2rivinf = np.zeros(self.nseqmax, dtype=np.float64)  # River inflow [m3/s]

        # Previous time step variables (for implicit scheme)
        self.d2rivsto_pre = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivdph_pre = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivout_pre = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2sfcelv = np.zeros(self.nseqmax, dtype=np.float64)  # Surface elevation [m]
        self.d2sfcelv_pre = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2dwnelv = np.zeros(self.nseqmax, dtype=np.float64)  # Downstream elevation [m]
        self.d2dwnelv_pre = np.zeros(self.nseqmax, dtype=np.float64)

        # Floodplain variables (if LFPLAIN)
        if self.lfplain:
            self.d2fldsto = np.zeros(self.nseqmax, dtype=np.float64)  # Floodplain storage [m3]
            self.d2flddph = np.zeros(self.nseqmax, dtype=np.float64)  # Floodplain depth [m]
            self.d2fldfrc = np.zeros(self.nseqmax, dtype=np.float64)  # Flooded fraction [-]
            self.d2fldare = np.zeros(self.nseqmax, dtype=np.float64)  # Flooded area [m2]
            self.d2fldout = np.zeros(self.nseqmax, dtype=np.float64)  # Floodplain outflow [m3/s]
            self.d2fldinf = np.zeros(self.nseqmax, dtype=np.float64)  # Floodplain inflow [m3/s]

            # Previous time step floodplain variables
            self.d2fldsto_pre = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2flddph_pre = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldout_pre = np.zeros(self.nseqmax, dtype=np.float64)
        else:
            # Even if no floodplain, initialize arrays for compatibility
            self.d2fldsto = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2flddph = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldfrc = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldare = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldout = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldinf = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldsto_pre = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2flddph_pre = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2fldout_pre = np.zeros(self.nseqmax, dtype=np.float64)

        # Total variables
        self.d2outflw = np.zeros(self.nseqmax, dtype=np.float64)  # Total outflow [m3/s]
        self.d2storge = np.zeros(self.nseqmax, dtype=np.float64)  # Total storage [m3]

        # Runoff input (will be updated each forcing time step)
        self.d1runoff = np.zeros(self.nseqmax, dtype=np.float64)  # Runoff input [m3/s]
        self.d2rofsub = np.zeros(self.nseqmax, dtype=np.float64)  # Subsurface runoff [m3/s]

        # Groundwater delay variables (if LGDWDLY)
        if self.lgdwdly:
            self.p2gdwsto = np.zeros(self.nseqmax, dtype=np.float64)  # Groundwater storage [m3]
            self.d2gdwrtn = np.zeros(self.nseqmax, dtype=np.float64)  # Groundwater return flow [m3/s]
        else:
            self.p2gdwsto = np.zeros(self.nseqmax, dtype=np.float64)
            self.d2gdwrtn = np.zeros(self.nseqmax, dtype=np.float64)

        # Evaporation variables (if LWEVAP)
        if self.lwevap:
            self.d2wevapex = np.zeros(self.nseqmax, dtype=np.float64)  # Evaporation extraction [m3/s]
        else:
            self.d2wevapex = np.zeros(self.nseqmax, dtype=np.float64)

        # Bifurcation channel flow (if LPTHOUT)
        self.d2pthout = np.zeros(self.nseqmax, dtype=np.float64)  # Bifurcation outflow [m3/s]

        # Bifurcation parameters (initialized if LPTHOUT = True)
        self.npthout = 0  # Number of bifurcation channels
        self.npthlev = 1  # Number of elevation levels for bifurcation
        self.bifurcation_loaded_from_nc = False  # Flag for whether bifurcation was loaded from NetCDF
        self.pth_upst = None  # Upstream cell index for each bifurcation
        self.pth_down = None  # Downstream cell index
        self.pth_dst = None  # Distance
        self.pth_elv = None  # Elevation for each level
        self.pth_wth = None  # Width for each level
        self.pth_man = None  # Manning coefficient
        self.d1pthflw = None  # Bifurcation flow [m3/s] (npthout, npthlev)
        self.d1pthflw_pre = None  # Previous bifurcation flow
        self.d1pthflwsum = None  # Sum of flows across all levels

        # Sparse matrix variables (initialized as None, built later if lspamat=True)
        self.i1upst = None  # Sparse upstream matrix
        self.i1upn = None   # Number of upstream cells (sparse)
        self.upnmax = 0     # Maximum upstream cells
        self.i1p_out = None  # Sparse bifurcation out matrix
        self.i1p_outn = None
        self.i1p_inf = None  # Sparse bifurcation inf matrix
        self.i1p_infn = None

    def _initialize_parameters(self):
        """Initialize parameter arrays (called from load_parameters)"""
        # River network topology
        self.i1next = np.full(self.nseqmax, -1, dtype=np.int32)  # Downstream cell index

        # Upstream topology (for inflow calculation - used by simple calc_inflow)
        max_upstream = 10  # Initial guess, will be expanded if needed
        self.i2upst = np.zeros((self.nseqmax, max_upstream), dtype=np.int32)  # Upstream cell indices
        self.i2upn = np.zeros(self.nseqmax, dtype=np.int32)  # Number of upstream cells

    def _initialize_storage_sea_surface(self):
        """
        Set initial storage assuming water surface not lower than downstream boundary.
        This replicates Fortran's STORAGE_SEA_SURFACE function from CMF_PROG_INIT.

        Fortran logic:
        ```fortran
        !! For River Mouth Grid
        DO ISEQ=NSEQRIV+1,NSEQALL
          DSEAELV=D2DWNELV(ISEQ,1)
          DDPH=MAX( DSEAELV-D2RIVELV(ISEQ,1),0._JPRB )
          DDPH=MIN( DDPH,D2RIVHGT(ISEQ,1) )
          P2RIVSTO(ISEQ,1)=DDPH*D2RIVLEN(ISEQ,1)*D2RIVWTH(ISEQ,1)
          P2RIVSTO(ISEQ,1)=MIN( P2RIVSTO(ISEQ,1),D2RIVSTOMAX(ISEQ,1) )
        END DO

        !! For River Grid (downstream to upstream)
        DO ISEQ=NSEQRIV,1,-1
          JSEQ=I1NEXT(ISEQ)
          DSEAELV=D2RIVELV(JSEQ,1)+D2RIVDPH_PRE(JSEQ,1)
          DDPH=MAX( DSEAELV-D2RIVELV(ISEQ,1),0._JPRB )
          DDPH=MIN( DDPH,D2RIVHGT(ISEQ,1) )
          P2RIVSTO(ISEQ,1)=DDPH*D2RIVLEN(ISEQ,1)*D2RIVWTH(ISEQ,1)
          P2RIVSTO(ISEQ,1)=MIN( P2RIVSTO(ISEQ,1),D2RIVSTOMAX(ISEQ,1) )
        END DO
        ```
        """
        print("  PROG_INIT: fill channels below downstream boundary")
        print("  (Fortran equivalent: STORAGE_SEA_SURFACE)")

        # Initialize all storage to zero
        self.d2rivsto[:] = 0.0
        self.d2fldsto[:] = 0.0
        self.d2rivdph[:] = 0.0
        self.d2rivdph_pre[:] = 0.0

        # 1. Process river mouth cells (NSEQRIV+1 to NSEQALL)
        print(f"    Processing {self.nseqall - self.nseqriv} river mouth cells...")
        for iseq in range(self.nseqriv, self.nseqall):
            # Downstream boundary elevation (sea level)
            dseaelv = self.d2dwnelv[iseq]

            # Initial water depth: max(sea_level - bed_elevation, 0)
            ddph = max(dseaelv - self.d2rivelv[iseq], 0.0)

            # Limit to channel depth
            ddph = min(ddph, self.d2rivhgt[iseq])

            # Calculate initial storage
            sto = ddph * self.d2rivlen[iseq] * self.d2rivwth[iseq]

            # Cap at maximum river storage if available
            if hasattr(self, 'd2rivstomax') and self.lfplain:
                self.d2rivsto[iseq] = min(sto, self.d2rivstomax[iseq])
            else:
                self.d2rivsto[iseq] = sto

            self.d2rivdph_pre[iseq] = ddph

        # 2. Process river cells from DOWNSTREAM to UPSTREAM
        # Use reverse iteration: Fortran does "DO ISEQ=NSEQRIV,1,-1"
        # The river network is numbered such that downstream cells have higher indices
        print(f"    Processing {self.nseqriv} river cells (downstream to upstream)...")
        print(f"    Using reverse iteration (downstream to upstream)...")

        # Process in REVERSE order (from high index to low index)
        # This ensures each cell's downstream cell is already processed
        for iseq in range(self.nseqriv - 1, -1, -1):
            jseq = self.i1next[iseq]  # Downstream cell

            # Downstream water surface elevation (bed + depth)
            if 0 <= jseq < self.nseqall:
                dseaelv = self.d2rivelv[jseq] + self.d2rivdph_pre[jseq]
            else:
                # If no valid downstream cell (shouldn't happen), use bed elevation
                dseaelv = self.d2rivelv[iseq]

            # Current cell water depth: not lower than downstream water level
            ddph = max(dseaelv - self.d2rivelv[iseq], 0.0)

            # Limit to channel depth
            ddph = min(ddph, self.d2rivhgt[iseq])

            # Calculate initial storage
            sto = ddph * self.d2rivlen[iseq] * self.d2rivwth[iseq]

            # Cap at maximum river storage if available
            if hasattr(self, 'd2rivstomax') and self.lfplain:
                self.d2rivsto[iseq] = min(sto, self.d2rivstomax[iseq])
            else:
                self.d2rivsto[iseq] = sto

            self.d2rivdph_pre[iseq] = ddph

        # 3. Report statistics
        # NOTE: D2RIVDPH is NOT calculated here (kept at 0)
        # It will be calculated in the first timestep from storage
        # This matches Fortran behavior exactly
        total_storage = np.sum(self.d2rivsto[:self.nseqall])
        cells_with_water = np.sum(self.d2rivsto[:self.nseqall] > 0)

        print(f"    Initial storage summary:")
        print(f"      Total river storage: {total_storage:.3e} m³")
        print(f"      Cells with water: {cells_with_water}/{self.nseqall}")
        if cells_with_water > 0:
            # Use d2rivdph_pre for reporting (d2rivdph is still 0 at this point)
            nonzero_depths = self.d2rivdph_pre[:self.nseqall][self.d2rivsto[:self.nseqall] > 0]
            avg_depth = np.mean(nonzero_depths)
            print(f"      Average depth (wet cells): {avg_depth:.4f} m")

    def _initialize_parameters(self):
        """Initialize channel parameters - will be loaded from initialization file"""
        # River network topology
        self.i1next = np.zeros(self.nseqmax, dtype=np.int32)  # Next downstream cell index
        self.i2upst = np.zeros((self.nseqmax, 10), dtype=np.int32)  # Upstream cell indices (max 10)
        self.i2upn = np.zeros(self.nseqmax, dtype=np.int32)  # Number of upstream cells

        # River channel geometry
        self.d2rivlen = np.ones(self.nseqmax, dtype=np.float64) * 1000.0   # River length [m]
        self.d2rivwth = np.ones(self.nseqmax, dtype=np.float64) * 100.0    # River width [m]
        self.d2rivhgt = np.ones(self.nseqmax, dtype=np.float64) * 5.0      # Channel depth [m]
        self.d2rivman = np.ones(self.nseqmax, dtype=np.float64) * self.pmanriv  # Manning coeff
        self.d2nxtdst = np.ones(self.nseqmax, dtype=np.float64) * 5000.0   # Distance to next cell [m]
        self.d2rivelv = np.zeros(self.nseqmax, dtype=np.float64)  # River bed elevation [m]
        self.d2elevtn = np.zeros(self.nseqmax, dtype=np.float64)  # Bank top elevation [m]

        # Catchment area
        self.d2grarea = np.ones(self.nseqmax, dtype=np.float64) * 1.0e6  # Unit catchment area [m2]

        # Mixed scheme mask (for LSLPMIX)
        # i2mask: 0 = use local inertial, 1 = use kinematic wave
        self.i2mask = np.zeros(self.nseqmax, dtype=np.int32)

        # Downstream elevation (for river mouths)
        self.d2dwnelv = np.zeros(self.nseqmax, dtype=np.float64)
        # pdstmth is already read in _read_physics_config()

        # Floodplain parameters (if LFPLAIN)
        if self.lfplain:
            # Maximum river storage
            self.d2rivstomax = self.d2rivlen * self.d2rivwth * self.d2rivhgt

            # Floodplain levels (for floodplain stage calculation)
            self.nlfp = 10  # Number of floodplain levels
            self.dfrcinc = 0.1  # Floodplain fraction increment (10% each level)
            self.d2fldstomax = np.zeros((self.nseqmax, self.nlfp), dtype=np.float64)
            self.d2fldgrd = np.ones((self.nseqmax, self.nlfp), dtype=np.float64) * 0.005  # Floodplain gradient [m/m]

            # Initialize floodplain storage levels
            for ilev in range(self.nlfp):
                # Simple linear increase in storage capacity with level
                self.d2fldstomax[:, ilev] = self.d2rivstomax * (1.0 + (ilev + 1) * 0.5)

        # Number of river mouth cells (cells with no downstream)
        # CRITICAL: This will be loaded from grid_routing_data.nc in load_parameters()
        # Do NOT set to nseqall here - that was the bug causing +6% bias!
        self.nseqriv = None  # Will be loaded from NetCDF file

        # Initialize bifurcation parameters if enabled
        if self.lpthout:
            self._initialize_bifurcation()

        print(f"  Initialized parameters for {self.nseqall} river cells")
        print(f"  Floodplain model: {'enabled' if self.lfplain else 'disabled'}")
        print(f"  Bifurcation: {'enabled' if self.lpthout else 'disabled'}")
        print(f"  Groundwater delay: {'enabled' if self.lgdwdly else 'disabled'}")
        if self.lgdwdly:
            print(f"  Groundwater time constant: {self.pgdwsto:.1f} days")

    def _initialize_bifurcation(self):
        """
        Initialize bifurcation channel parameters

        This is called during __init__ but defers actual loading to load_parameters().
        The actual bifurcation data will be loaded from:
        1. NetCDF file (in load_parameters()) if available, or
        2. Text file (cpthout) if NetCDF doesn't have it

        This method only loads from text file if explicitly specified.
        """
        # Check if bifurcation was already loaded from NetCDF
        if hasattr(self, 'bifurcation_loaded_from_nc') and self.bifurcation_loaded_from_nc:
            print(f"  Bifurcation already loaded from NetCDF ({self.npthout} channels)")
            return

        # Check if text file is explicitly specified
        cpthout = self.nml.get('MODEL_RUN', 'cpthout', '')

        if cpthout and os.path.exists(cpthout):
            # Text file explicitly specified and exists - load from it
            try:
                self._load_bifurcation_file(cpthout)
                print(f"  Loaded {self.npthout} bifurcation channels from text file")
                self.bifurcation_loaded_from_nc = True  # Mark as loaded to skip NetCDF loading
            except Exception as e:
                print(f"  WARNING: Failed to load bifurcation file: {e}")
                print("  Will try to load from NetCDF during load_parameters()")
        else:
            # No text file specified - will try to load from NetCDF in load_parameters()
            print(f"  Bifurcation enabled - will load from NetCDF or text file")

    def _load_bifurcation_file(self, cpthout):
        """
        Load bifurcation parameters from file

        File format (text):
        npthout npthlev
        For each bifurcation:
          upst down dst elv1 elv2 ... wth1 wth2 ... man1 man2 ...
        """
        import os

        with open(cpthout, 'r') as f:
            # Read header
            line = f.readline().strip()
            parts = line.split()
            self.npthout = int(parts[0])
            self.npthlev = int(parts[1]) if len(parts) > 1 else 1

            # Initialize arrays
            self.pth_upst = np.zeros(self.npthout, dtype=np.int32)
            self.pth_down = np.zeros(self.npthout, dtype=np.int32)
            self.pth_dst = np.zeros(self.npthout, dtype=np.float64)
            self.pth_elv = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
            self.pth_wth = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
            self.pth_man = np.ones(self.npthlev, dtype=np.float64) * self.pmanriv

            self.d1pthflw = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
            self.d1pthflw_pre = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
            self.d1pthflwsum = np.zeros(self.npthout, dtype=np.float64)

            # Read bifurcation data
            for ipth in range(self.npthout):
                line = f.readline().strip()
                if not line:
                    break

                parts = line.split()
                idx = 0

                self.pth_upst[ipth] = int(parts[idx])
                idx += 1
                self.pth_down[ipth] = int(parts[idx])
                idx += 1
                self.pth_dst[ipth] = float(parts[idx])
                idx += 1

                # Read elevations
                for ilev in range(self.npthlev):
                    if idx < len(parts):
                        self.pth_elv[ipth, ilev] = float(parts[idx])
                        idx += 1

                # Read widths
                for ilev in range(self.npthlev):
                    if idx < len(parts):
                        self.pth_wth[ipth, ilev] = float(parts[idx])
                        idx += 1

                # Read Manning coefficients (if provided)
                for ilev in range(self.npthlev):
                    if idx < len(parts):
                        self.pth_man[ilev] = float(parts[idx])
                        idx += 1

    def load_parameters(self, init_nc_file):
        """
        Load river network parameters from initialization NetCDF file

        This is CRITICAL - loads the actual river network topology and geometry
        from grid_routing_data.nc into the physics arrays.

        Parameters:
        -----------
        init_nc_file : str
            Path to initialization NetCDF file (grid_routing_data.nc)
        """
        import netCDF4 as nc

        print(f"  Loading river network parameters from: {init_nc_file}")

        with nc.Dataset(init_nc_file, 'r') as f:
            # Get dimensions
            if 'seq' in f.dimensions:
                nseq = len(f.dimensions['seq'])
            elif 'nseqmax' in f.dimensions:
                nseq = len(f.dimensions['nseqmax'])
            else:
                raise ValueError("Cannot find sequence dimension in NetCDF file")

            if nseq != self.nseqmax:
                print(f"  WARNING: NetCDF nseqmax ({nseq}) != physics nseqmax ({self.nseqmax})")
                nseq = min(nseq, self.nseqmax)

            # =================================================================
            # Load nseqriv (CRITICAL - fixes +6% bias!)
            # =================================================================
            # nseqriv = number of regular river cells (not including river mouths)
            # nseqall = total cells (river + mouth)
            # Cells [0, nseqriv) use full local inertial equation
            # Cells [nseqriv, nseqall) use simplified mouth dynamics
            if hasattr(f, 'nseqriv'):
                self.nseqriv = int(f.nseqriv)
                print(f"    Loaded nseqriv = {self.nseqriv} (regular river cells)")
                print(f"    River mouth cells = {self.nseqall - self.nseqriv} (use simplified dynamics)")

                if self.nseqriv > self.nseqall:
                    raise ValueError(f"nseqriv ({self.nseqriv}) > nseqall ({self.nseqall})!")
            else:
                # Fallback: assume all cells are river cells (old behavior)
                print(f"    WARNING: nseqriv not found in NetCDF file, using nseqall")
                self.nseqriv = self.nseqall

            # =================================================================
            # Load river network topology
            # =================================================================
            # Downstream cell index (1-based in file, convert to 0-based)
            if 'seq_next' in f.variables:
                seq_next = f.variables['seq_next'][:]
                self.i1next[:nseq] = seq_next - 1  # Convert to 0-based
                # Mark river mouth cells (no downstream) as -1
                temp = self.i1next[:nseq].copy()
                temp[temp >= nseq] = -1
                temp[temp < 0] = -1  # Also mark originally negative values
                self.i1next[:nseq] = temp
                print(f"    Loaded i1next (downstream indices)")

            # Upstream cell topology (if available)
            # Note: This is typically not stored in NetCDF, will be reconstructed if needed

            # =================================================================
            # Load channel geometry
            # =================================================================
            if 'topo_rivlen' in f.variables:
                self.d2rivlen[:nseq] = f.variables['topo_rivlen'][:]
                print(f"    Loaded d2rivlen (river length): mean={np.mean(self.d2rivlen[:nseq]):.1f} m")

            if 'topo_rivwth' in f.variables:
                self.d2rivwth[:nseq] = f.variables['topo_rivwth'][:]
                print(f"    Loaded d2rivwth (river width): mean={np.mean(self.d2rivwth[:nseq]):.1f} m")

            if 'topo_rivhgt' in f.variables:
                self.d2rivhgt[:nseq] = f.variables['topo_rivhgt'][:]
                print(f"    Loaded d2rivhgt (river height): mean={np.mean(self.d2rivhgt[:nseq]):.2f} m")

            if 'topo_rivman' in f.variables:
                self.d2rivman[:nseq] = f.variables['topo_rivman'][:]
                print(f"    Loaded d2rivman (Manning coefficient)")

            if 'topo_distance' in f.variables:
                self.d2nxtdst[:nseq] = f.variables['topo_distance'][:]
                print(f"    Loaded d2nxtdst (next distance): mean={np.mean(self.d2nxtdst[:nseq]):.1f} m")

            # =================================================================
            # Load elevation data
            # =================================================================
            if 'topo_rivelv' in f.variables:
                self.d2rivelv[:nseq] = f.variables['topo_rivelv'][:]
                print(f"    Loaded d2rivelv (river bed elevation)")

            if 'topo_elevation' in f.variables:
                self.d2elevtn[:nseq] = f.variables['topo_elevation'][:]
                print(f"    Loaded d2elevtn (bank top elevation)")

            # =================================================================
            # Load catchment area
            # =================================================================
            if 'topo_catcharea' in f.variables:
                self.d2grarea[:nseq] = f.variables['topo_catcharea'][:]
                print(f"    Loaded d2grarea (catchment area): mean={np.mean(self.d2grarea[:nseq])/1e6:.2f} km²")
            elif 'topo_grarea' in f.variables:
                self.d2grarea[:nseq] = f.variables['topo_grarea'][:]
                print(f"    Loaded d2grarea (catchment area)")

            # =================================================================
            # Load floodplain parameters (if LFPLAIN)
            # =================================================================
            if self.lfplain:
                # Maximum river storage
                self.d2rivstomax[:nseq] = self.d2rivlen[:nseq] * self.d2rivwth[:nseq] * self.d2rivhgt[:nseq]

                # Floodplain storage maximum (by level)
                if 'topo_fldstomax' in f.variables:
                    fldstomax = f.variables['topo_fldstomax'][:]  # (nseq, nlfp)
                    if len(fldstomax.shape) == 2:
                        nlfp_file = fldstomax.shape[1]
                        nlfp_use = min(nlfp_file, self.nlfp)
                        self.d2fldstomax[:nseq, :nlfp_use] = fldstomax[:nseq, :nlfp_use]
                        print(f"    Loaded d2fldstomax (floodplain storage max)")

                # Floodplain gradient
                if 'topo_fldgrd' in f.variables:
                    fldgrd = f.variables['topo_fldgrd'][:]
                    if len(fldgrd.shape) == 2:
                        nlfp_file = fldgrd.shape[1]
                        nlfp_use = min(nlfp_file, self.nlfp)
                        self.d2fldgrd[:nseq, :nlfp_use] = fldgrd[:nseq, :nlfp_use]
                        print(f"    Loaded d2fldgrd (floodplain gradient)")

                # Floodplain height profile
                if 'topo_fldhgt' in f.variables:
                    # This is a 1D profile, not used directly but stored for reference
                    pass

            # =================================================================
            # Load downstream boundary elevation
            # =================================================================
            # CRITICAL FIX: Load topo_dwnelv from file instead of computing it!
            # This parameter is precomputed by Fortran and stored in grid file.
            # Computing it from d2elevtn causes 43% lower initial storage!
            if 'topo_dwnelv' in f.variables:
                self.d2dwnelv[:nseq] = f.variables['topo_dwnelv'][:]
                print(f"    Loaded d2dwnelv (downstream boundary elevation)")
            else:
                # Fallback: compute if not in file (for backward compatibility)
                print(f"    WARNING: topo_dwnelv not in file, computing from d2elevtn")
                for iseq in range(self.nseqall):
                    jseq = self.i1next[iseq]
                    if jseq >= 0 and jseq < self.nseqall:
                        # Has downstream cell
                        self.d2dwnelv[iseq] = self.d2elevtn[jseq]
                    else:
                        # River mouth - use bank elevation
                        self.d2dwnelv[iseq] = self.d2elevtn[iseq]

            # =================================================================
            # Load bifurcation parameters (if enabled and available in NetCDF)
            # =================================================================
            if self.lpthout:
                if 'npthout' in f.dimensions and 'npthlev' in f.dimensions:
                    npthout_nc = len(f.dimensions['npthout'])
                    npthlev_nc = len(f.dimensions['npthlev'])

                    # Check if bifurcation variables exist in NetCDF
                    bifurcation_vars = ['bifurcation_upst', 'bifurcation_down', 'bifurcation_distance',
                                       'bifurcation_elevation', 'bifurcation_width', 'bifurcation_manning']
                    has_bifurcation = all(var in f.variables for var in bifurcation_vars)

                    if has_bifurcation and npthout_nc > 0:
                        # Load bifurcation dimensions
                        self.npthout = npthout_nc
                        self.npthlev = npthlev_nc

                        # Initialize bifurcation arrays
                        self.pth_upst = np.zeros(self.npthout, dtype=np.int32)
                        self.pth_down = np.zeros(self.npthout, dtype=np.int32)
                        self.pth_dst = np.zeros(self.npthout, dtype=np.float64)
                        self.pth_elv = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
                        self.pth_wth = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
                        self.pth_man = np.ones(self.npthlev, dtype=np.float64) * self.pmanriv

                        # Load bifurcation topology
                        self.pth_upst[:] = f.variables['bifurcation_upst'][:]
                        self.pth_down[:] = f.variables['bifurcation_down'][:]
                        self.pth_dst[:] = f.variables['bifurcation_distance'][:]

                        # Load bifurcation profiles (note: stored as (npthout, npthlev) in NetCDF)
                        self.pth_elv[:, :] = f.variables['bifurcation_elevation'][:]
                        self.pth_wth[:, :] = f.variables['bifurcation_width'][:]
                        self.pth_man[:] = f.variables['bifurcation_manning'][:]

                        # Initialize bifurcation flow arrays
                        self.d1pthflw = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
                        self.d1pthflw_pre = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
                        self.d1pthflwsum = np.zeros(self.npthout, dtype=np.float64)

                        print(f"    Loaded bifurcation parameters from NetCDF")
                        print(f"      Bifurcation channels: {self.npthout}")
                        print(f"      Bifurcation levels: {self.npthlev}")

                        # Set flag that bifurcation was loaded from NetCDF
                        self.bifurcation_loaded_from_nc = True
                    else:
                        print(f"    WARNING: Bifurcation enabled but data not found in NetCDF")
                        # Check if text file is available as fallback
                        cpthout = self.nml.get('MODEL_RUN', 'cpthout', '')
                        if cpthout and os.path.exists(cpthout):
                            print(f"    Will load from text file: {cpthout}")
                            try:
                                self._load_bifurcation_file(cpthout)
                                print(f"    Loaded {self.npthout} bifurcation channels from text file")
                                self.bifurcation_loaded_from_nc = True  # Mark as loaded
                            except Exception as e:
                                print(f"    ERROR: Failed to load from text file: {e}")
                                print(f"    Bifurcation disabled")
                                self.lpthout = False
                                self.bifurcation_loaded_from_nc = False
                        else:
                            print(f"    ERROR: No text file fallback available")
                            print(f"    Bifurcation disabled")
                            self.lpthout = False
                            self.bifurcation_loaded_from_nc = False
                else:
                    print(f"    WARNING: Bifurcation dimensions not found in NetCDF")
                    # Check if text file is available as fallback
                    cpthout = self.nml.get('MODEL_RUN', 'cpthout', '')
                    if cpthout and os.path.exists(cpthout):
                        print(f"    Will load from text file: {cpthout}")
                        try:
                            self._load_bifurcation_file(cpthout)
                            print(f"    Loaded {self.npthout} bifurcation channels from text file")
                            self.bifurcation_loaded_from_nc = True  # Mark as loaded
                        except Exception as e:
                            print(f"    ERROR: Failed to load from text file: {e}")
                            print(f"    Bifurcation disabled")
                            self.lpthout = False
                            self.bifurcation_loaded_from_nc = False
                    else:
                        print(f"    ERROR: No text file fallback available")
                        print(f"    Bifurcation disabled")
                        self.lpthout = False
                        self.bifurcation_loaded_from_nc = False
            else:
                self.bifurcation_loaded_from_nc = False

            # =================================================================
            # Reconstruct upstream topology from i1next
            # =================================================================
            self._reconstruct_upstream_topology()

            # =================================================================
            # Build sparse matrix for optimized inflow calculation (LSPAMAT)
            # =================================================================
            if self.lspamat:
                print(f"  Building sparse matrix for optimized inflow calculation")
                self._build_sparse_matrices()

        print(f"  River network parameters loaded successfully")
        print(f"  Active river cells: {self.nseqall}")
        print(f"  River mouth cells: {np.sum(self.i1next[:self.nseqall] < 0)}")

        # Initialize storage to sea surface level (Fortran-style initialization)
        # NOTE: Fortran calls STORAGE_SEA_SURFACE in PROG_INIT even when LRESTART=FALSE
        # This fills channels below downstream boundary elevation
        self._initialize_storage_sea_surface()

    def _reconstruct_upstream_topology(self):
        """
        Reconstruct upstream cell topology from i1next (downstream indices)

        For each cell, find all upstream cells that drain into it.
        This is needed for inflow calculation.
        """
        # Reset upstream topology
        self.i2upn[:] = 0
        self.i2upst[:] = 0

        # Build upstream list from downstream indices
        for iseq in range(self.nseqall):
            jseq = self.i1next[iseq]
            if jseq >= 0 and jseq < self.nseqall:
                # Add iseq as upstream cell of jseq
                n = self.i2upn[jseq]
                if n < self.i2upst.shape[1]:
                    self.i2upst[jseq, n] = iseq
                    self.i2upn[jseq] += 1

        max_upstream = np.max(self.i2upn)
        print(f"    Reconstructed upstream topology: max upstream cells = {max_upstream}")

    def _build_sparse_matrices(self):
        """
        Build sparse matrices for optimized inflow calculation (LSPAMAT mode)

        This creates:
        - i1upst, i1upn: Sparse upstream connectivity matrix
        - i1p_out, i1p_outn, i1p_inf, i1p_infn: Sparse bifurcation connectivity matrices

        These matrices allow direct access to upstream neighbors without iterating
        over all cells, enabling better parallelization and cache performance.
        """
        print(f"    Building sparse upstream matrix...")

        # Build sparse upstream matrix from i1next
        self.i1upst, self.i1upn, self.upnmax = build_sparse_upstream_matrix(
            self.nseqall, self.nseqriv, self.i1next
        )

        # Build sparse bifurcation matrices if bifurcation is enabled
        if self.lpthout and self.npthout > 0:
            print(f"    Building sparse bifurcation matrices...")
            self.i1p_out, self.i1p_outn, self.i1p_inf, self.i1p_infn = \
                build_sparse_bifurcation_matrix(
                    self.nseqall, self.npthout, self.pth_upst, self.pth_down
                )
        else:
            # Create dummy arrays for consistency
            self.i1p_out = np.zeros((self.nseqall, 1), dtype=np.int32)
            self.i1p_outn = np.zeros(self.nseqall, dtype=np.int32)
            self.i1p_inf = np.zeros((self.nseqall, 1), dtype=np.int32)
            self.i1p_infn = np.zeros(self.nseqall, dtype=np.int32)

        print(f"    Sparse matrices built successfully")
        print(f"      Upstream matrix: ({self.nseqall}, {self.upnmax})")
        if self.lpthout and self.npthout > 0:
            print(f"      Bifurcation out matrix: ({self.nseqall}, {self.i1p_out.shape[1]})")
            print(f"      Bifurcation inf matrix: ({self.nseqall}, {self.i1p_inf.shape[1]})")

    def load_restart(self, restart_manager):
        """
        Load initial conditions from restart file using RestartManager

        Parameters:
        -----------
        restart_manager : RestartManager object
            Restart manager instance
        """
        restart_manager.read_restart()

    def set_runoff_input(self, p0_rivsto):
        """
        Set runoff input for current time step

        Parameters:
        -----------
        p0_rivsto : ndarray
            Runoff input for each cell [m/s or m3/s]
            If LROSPLIT: shape (nseqall, 2) with [:, 0]=surface, [:, 1]=subsurface
            Otherwise: shape (nseqall,) total runoff
        """
        if self.lrosplit and len(p0_rivsto.shape) == 2:
            # Surface runoff goes directly to river
            self.d1runoff[:len(p0_rivsto)] = p0_rivsto[:, 0]
            # Subsurface runoff
            self.d2rofsub[:len(p0_rivsto)] = p0_rivsto[:, 1]
        else:
            # Total runoff goes directly to river
            if len(p0_rivsto.shape) == 1:
                self.d1runoff[:len(p0_rivsto)] = p0_rivsto
                self.d2rofsub[:len(p0_rivsto)] = 0.0
            else:
                # Legacy support: 2D array but LROSPLIT=False, use first column
                self.d1runoff[:len(p0_rivsto)] = p0_rivsto[:, 0] if p0_rivsto.shape[1] > 0 else p0_rivsto.flatten()
                self.d2rofsub[:len(p0_rivsto)] = 0.0

    def set_evaporation_input(self, p0_evap):
        """
        Set evaporation input for current time step

        Parameters:
        -----------
        p0_evap : ndarray or None
            Evaporation rate for each cell [m3/s]
            Shape: (nseqall,)
        """
        if p0_evap is None:
            self.d2wevapex[:] = 0.0
        else:
            self.d2wevapex[:len(p0_evap)] = p0_evap

    def reconstruct_outflow_pre(self):
        """
        Reconstruct previous time-step outflow from current storage

        This is used when restarting from storage-only restart files (LSTOONLY).
        Uses diffusive wave approximation (Manning equation) to estimate
        outflow at t-1 from current storage at t=0.

        Called after loading restart and before first physics_advance.
        """
        if not self.lstoonly:
            # Not needed for full restart files
            return

        print("  Reconstructing previous outflow from storage...")

        # First calculate depths and surface elevations
        if self.lfplain:
            diag_fldstg = calc_fldstg(
                self.d2rivsto, self.d2fldsto, self.nseqall, self.nlfp,
                self.d2grarea, self.d2rivlen, self.d2rivwth, self.d2rivelv,
                self.d2rivstomax, self.d2fldstomax, self.d2fldgrd, self.dfrcinc,
                self.d2rivdph, self.d2flddph, self.d2fldfrc, self.d2fldare,
                self.d2sfcelv, self.d2storge
            )
        else:
            for iseq in range(self.nseqall):
                area = self.d2rivlen[iseq] * self.d2rivwth[iseq]
                if area > 0:
                    self.d2rivdph[iseq] = self.d2rivsto[iseq] / area
                else:
                    self.d2rivdph[iseq] = 0.0
                self.d2sfcelv[iseq] = self.d2rivelv[iseq] + self.d2rivdph[iseq]

        # Reconstruct river and floodplain outflow
        calc_outpre(
            self.nseqall, self.nseqriv,
            self.d2rivlen, self.d2rivwth, self.d2rivman, self.pmanfld,
            self.d2nxtdst, self.d2rivelv, self.d2elevtn, self.d2dwnelv,
            self.d2rivdph, self.d2flddph, self.d2sfcelv, self.d2fldsto,
            self.d2rivout_pre, self.d2fldout_pre, self.d2rivdph_pre,
            self.i1next, self.pdstmth
        )

        # Reconstruct bifurcation flow if enabled
        if self.lpthout and self.npthout > 0:
            calc_pthout_pre(
                self.npthout, self.npthlev,
                self.pth_upst, self.pth_down, self.pth_dst,
                self.pth_elv, self.pth_wth, self.pth_man,
                self.d2rivdph, self.d2sfcelv, self.d1pthflw_pre
            )

        print("  Previous outflow reconstructed")

    def calculate_groundwater(self, dt):
        """
        Calculate groundwater return flow and update groundwater storage

        Based on simple linear reservoir model:
        dS/dt = Qsub - Qgw
        Qgw = S / tau

        where:
        S = groundwater storage [m3]
        tau = time constant [s]
        Qsub = subsurface runoff input [m3/s]
        Qgw = groundwater return flow [m3/s]

        Parameters:
        -----------
        dt : float
            Time step [seconds]
        """
        if not self.lgdwdly:
            # If groundwater delay is disabled, subsurface runoff goes directly to river
            self.d2gdwrtn[:self.nseqall] = self.d2rofsub[:self.nseqall]
            return

        # Time constant in seconds
        tau = self.pgdwsto * 86400.0  # Convert days to seconds

        # Calculate groundwater return flow (before updating storage)
        # Qgw = S / tau
        self.d2gdwrtn[:self.nseqall] = self.p2gdwsto[:self.nseqall] / tau

        # Update groundwater storage
        # dS/dt = Qsub - Qgw
        self.p2gdwsto[:self.nseqall] += (self.d2rofsub[:self.nseqall] - self.d2gdwrtn[:self.nseqall]) * dt

        # Ensure non-negative storage
        self.p2gdwsto[:self.nseqall] = np.maximum(self.p2gdwsto[:self.nseqall], 0.0)

    def apply_evaporation_extraction(self, p0_evap, dt):
        """
        Extract evaporation from floodplain and river storage

        Based on cmf_ctrl_physics_mod.F90

        Evaporation is extracted in the following order:
        1. First from floodplain storage (if available)
        2. Then from river storage (if LWEXTRACTRIV = True)

        Parameters:
        -----------
        p0_evap : ndarray or None
            Evaporation rate for each cell [m3/s]
        dt : float
            Time step [s]
        """
        if not self.lwevap or p0_evap is None:
            return

        for iseq in range(self.nseqall):
            # Calculate evaporation volume for this timestep [m³]
            evap_volume = p0_evap[iseq] * dt

            if evap_volume <= 0.0:
                continue

            # First, extract from floodplain storage
            extract_fld = min(evap_volume, self.d2fldsto[iseq])
            self.d2fldsto[iseq] -= extract_fld
            remaining = evap_volume - extract_fld

            # If enabled and there's remaining evaporation, extract from river storage
            if self.lwextractriv and remaining > 1.0e-6:
                extract_riv = min(remaining, self.d2rivsto[iseq])
                self.d2rivsto[iseq] -= extract_riv
                remaining -= extract_riv

            # If LWEVAPFIX is enabled and there's still remaining evaporation,
            # we could log a warning or adjust the evaporation rate
            # For now, we just accept that not all evaporation could be extracted
            if self.lwevapfix and remaining > 1.0e-6:
                # Water balance closure: could track this for diagnostics
                pass

    def calculate_adaptive_timestep(self, dt_default):
        """
        Calculate adaptive time step based on Courant condition

        Based on CALC_ADPSTP in cmf_ctrl_physics_mod.F90

        The Courant condition ensures numerical stability:
        DT <= C * DX / sqrt(g * h)

        where:
        - C is the Courant coefficient (PCADP, typically 0.7)
        - DX is the distance to downstream cell
        - g is gravity
        - h is water depth

        Parameters:
        -----------
        dt_default : float
            Default time step [seconds]

        Returns:
        --------
        nt : int
            Number of sub time steps required
        dt_sub : float
            Sub time step size [seconds]
        """
        if not self.ladpstp:
            return 1, dt_default

        # Call JIT-compiled core function for CFL calculation
        dt_min = _calculate_adaptive_timestep_core(
            self.nseqriv, self.nseqall, self.pcadp, self.pgrv, self.pdstmth,
            self.d2rivdph, self.d2nxtdst, dt_default
        )

        # Calculate number of sub time steps
        # NT = ceiling(DT_DEFAULT / DT_MIN)
        nt = int(dt_default / dt_min - 0.01) + 1

        # Adjust actual time step
        dt_sub = dt_default / nt

        # Print diagnostic info if multiple sub steps are needed
        if nt >= 2:
            print(f"  ADAPTIVE TIMESTEP: NT={nt:4d}, DT_DEFAULT={dt_default:10.2f}s, "
                  f"DT_MIN={dt_min:10.2f}s, DT_SUB={dt_sub:10.2f}s")

        return nt, dt_sub

    def physics_advance(self, dt):
        """
        Advance physics for one time step
        Implements the full CaMa-Flood physics based on CMF_PHYSICS_ADVANCE

        Parameters:
        -----------
        dt : float
            Time step [seconds]
        """
        # ======================================
        # 1. Calculate floodplain stage from storage
        # ======================================
        if self.lfplain:
            diag_fldstg = calc_fldstg(
                self.d2rivsto, self.d2fldsto, self.nseqall, self.nlfp,
                self.d2grarea, self.d2rivlen, self.d2rivwth, self.d2rivelv,
                self.d2rivstomax, self.d2fldstomax, self.d2fldgrd, self.dfrcinc,
                self.d2rivdph, self.d2flddph, self.d2fldfrc, self.d2fldare,
                self.d2sfcelv, self.d2storge
            )
        else:
            # Simple river-only depth calculation
            for iseq in range(self.nseqall):
                area = self.d2rivlen[iseq] * self.d2rivwth[iseq]
                if area > 0:
                    self.d2rivdph[iseq] = self.d2rivsto[iseq] / area
                else:
                    self.d2rivdph[iseq] = 0.0
                self.d2sfcelv[iseq] = self.d2rivelv[iseq] + self.d2rivdph[iseq]
                self.d2storge[iseq] = self.d2rivsto[iseq]

        # ======================================
        # 2. Calculate river and floodplain discharge
        # ======================================
        if self.lslpmix:
            # Mixed scheme (kinematic + local inertial, slope-dependent)
            calc_outflw_kinemix(
                self.nseqall, self.nseqriv, self.pgrv, self.pminslp, dt,
                self.d2rivlen, self.d2rivwth, self.d2rivhgt, self.d2rivman,
                self.d2nxtdst, self.d2rivelv, self.d2elevtn, self.d2dwnelv,
                self.d2rivdph, self.d2rivdph_pre, self.d2sfcelv,
                self.d2rivout_pre, self.d2rivout, self.d2rivvel,
                self.i1next, self.i2mask, self.pdstmth
            )

            # Calculate floodplain outflow if enabled
            if self.lfldout and self.lfplain:
                calc_fldout_kinemix(
                    self.nseqall, self.nseqriv, self.pgrv, self.pmanfld, dt,
                    self.d2rivlen, self.d2rivwth, self.d2rivhgt,
                    self.d2nxtdst, self.d2rivelv, self.d2elevtn,
                    self.d2rivdph, self.d2rivdph_pre, self.d2sfcelv,
                    self.d2flddph, self.d2fldsto, self.d2fldsto_pre,
                    self.d2fldout_pre, self.d2fldout, self.d2rivout,
                    self.i1next, self.i2mask
                )
            else:
                self.d2fldout[:] = 0.0

        elif self.lkine:
            # Kinematic wave scheme (more stable for steep slopes)
            calc_outflw_kine(
                self.nseqall, self.nseqriv, self.pgrv, self.pminslp,
                self.d2rivlen, self.d2rivwth, self.d2rivhgt, self.d2rivman,
                self.d2nxtdst, self.d2rivelv,
                self.d2rivdph, self.d2sfcelv, self.d2dwnelv,
                self.d2rivout, self.d2rivvel,
                self.i1next
            )

            # Calculate floodplain outflow if enabled
            if self.lfldout and self.lfplain:
                calc_fldout_kine(
                    self.nseqall, self.pgrv, self.pmanfld, self.pminslp,
                    self.d2rivlen, self.d2rivwth, self.d2rivhgt, self.d2rivelv, self.d2nxtdst,
                    self.d2flddph, self.d2fldsto, self.d2fldgrd,
                    self.d2fldout,
                    self.i1next
                )
            else:
                self.d2fldout[:] = 0.0
        else:
            # Local inertial scheme (default)
            calc_outflw(
                self.nseqall, self.nseqriv, dt, self.pgrv, self.pmanfld, self.lfldout,
                self.i1next, self.d2rivlen, self.d2rivwth, self.d2rivhgt, self.d2rivman,
                self.d2nxtdst, self.d2rivelv, self.d2elevtn, self.d2rivdph, self.d2rivdph_pre,
                self.d2sfcelv, self.d2sfcelv_pre, self.d2dwnelv, self.d2dwnelv_pre,
                self.d2rivout, self.d2rivout_pre, self.d2rivvel,
                self.d2fldsto, self.d2fldsto_pre, self.d2flddph, self.d2flddph_pre,
                self.d2fldout, self.d2fldout_pre, self.d2storge
            )

        # ======================================
        # 3. Calculate bifurcation flow (optional)
        # ======================================
        if self.lpthout and self.npthout > 0:
            # Create mask for cells with no bifurcation
            i2mask = np.zeros(self.nseqall, dtype=np.int32)

            # Calculate bifurcation flow
            calc_pthout(
                self.nseqall, self.npthout, self.npthlev, dt, self.pgrv,
                self.pth_upst, self.pth_down, self.pth_dst, self.pth_elv, self.pth_wth, self.pth_man,
                i2mask, self.d2rivelv, self.d2rivdph, self.d2sfcelv, self.d2sfcelv_pre,
                self.d1pthflw, self.d1pthflw_pre, self.d1pthflwsum, self.d2storge
            )

            # Map bifurcation flow to grid cells
            # CRITICAL FIX #14: Update BOTH upstream and downstream cells to match Fortran
            # Fortran cmf_calc_outflw_mod.F90 lines 360, 364:
            #   P2PTHOUT(ISEQP,1) = P2PTHOUT(ISEQP,1) + D1PTHFLWSUM(IPTH)  (upstream: positive)
            #   P2PTHOUT(JSEQP,1) = P2PTHOUT(JSEQP,1) - D1PTHFLWSUM(IPTH)  (downstream: negative)
            # In calc_stonxt, floodplain storage is updated as: - D2PTHOUT*DT
            # So: upstream floodplain loses water (-flow*dt), downstream floodplain gains water (-(-flow)*dt = +flow*dt)
            self.d2pthout[:] = 0.0
            for ipth in range(self.npthout):
                iseq_up = self.pth_upst[ipth] - 1  # Convert from Fortran 1-based to Python 0-based
                iseq_dn = self.pth_down[ipth] - 1  # Convert from Fortran 1-based to Python 0-based

                # Update upstream cell (positive flow = water leaving)
                if 0 <= iseq_up < self.nseqall:
                    self.d2pthout[iseq_up] += self.d1pthflwsum[ipth]

                # Update downstream cell (negative flow = water arriving)
                if 0 <= iseq_dn < self.nseqall:
                    self.d2pthout[iseq_dn] -= self.d1pthflwsum[ipth]

            # Save current bifurcation flow for next iteration
            np.copyto(self.d1pthflw_pre, self.d1pthflw)

        # ======================================
        # 4. Calculate inflow from upstream
        # ======================================

        # CRITICAL FIX: Prepare runoff BEFORE quality conservation check
        # Otherwise conservation check doesn't know about incoming water!
        self.calculate_groundwater(dt)
        d1runoff_total = self.d1runoff.copy()
        d1runoff_total[:self.nseqall] += self.d2gdwrtn[:self.nseqall]

        if self.lconserv:
            # Use water conservation version (recommended)
            # This prevents cells from outputting more water than they contain
            # NOW includes runoff input in the conservation check

            if self.lspamat:
                # Use sparse matrix version (LSPAMAT) - faster for large domains
                # This avoids iterating over all cells to find upstream neighbors
                calc_inflow_with_conservation_spamat(
                    self.nseqall, self.nseqriv, dt,
                    self.i1next, self.i1upst, self.i1upn,
                    self.d2rivsto, self.d2fldsto,
                    self.d2rivout, self.d2fldout, self.d2pthout,
                    self.d2rivinf, self.d2fldinf,
                    self.lpthout, self.npthout,
                    self.pth_upst, self.pth_down,
                    self.d1pthflwsum, self.i2mask,
                    self.i1p_out, self.i1p_outn,
                    self.i1p_inf, self.i1p_infn,
                    d1runoff_total  # Pass runoff to conservation check
                )
            else:
                # Use vectorized version (standard)
                calc_inflow_with_conservation(
                    self.nseqall, self.nseqriv, dt,
                    self.i1next, self.d2rivsto, self.d2fldsto,
                    self.d2rivout, self.d2fldout, self.d2pthout,
                    self.d2rivinf, self.d2fldinf,
                    self.lpthout, self.npthout, self.pth_upst, self.pth_down,
                    self.d1pthflwsum, self.i2mask,
                    d1runoff_total  # Pass runoff to conservation check
                )
        else:
            # Use simple version (for backward compatibility)
            calc_inflow(
                self.nseqall, self.i1next, self.i2upst, self.i2upn,
                self.d2rivout, self.d2fldout, self.d2rivinf, self.d2fldinf
            )

        # ======================================
        # 4.6. Apply evaporation extraction (before storage update)
        # ======================================
        # Note: d2wevapex should be set by the forcing module via set_evaporation_input()
        self.apply_evaporation_extraction(self.d2wevapex, dt)

        # ======================================
        # 4.7. Save current state for next iteration (BEFORE storage update!)
        # ======================================
        # CRITICAL FIX: Match Fortran behavior by saving vars BEFORE calc_stonxt
        # Fortran calls CALC_VARS_PRE before CMF_CALC_STONXT (line 100 vs 103)
        # This ensures _pre variables contain the PRE-UPDATE values for next sub-timestep
        # Using POST-UPDATE values (as we did before) causes +5-6% systematic bias!
        # See: cmf_ctrl_physics_mod.F90, lines 100-103
        save_vars_pre(
            self.d2rivout, self.d2rivout_pre,
            self.d2fldout, self.d2fldout_pre,
            self.d2rivdph, self.d2rivdph_pre,
            self.d2fldsto, self.d2fldsto_pre,
            self.d2flddph, self.d2flddph_pre,
            self.d2sfcelv, self.d2sfcelv_pre
        )

        # ======================================
        # 5. Update storage for next time step
        # ======================================
        diag_stonxt = calc_stonxt(
            self.nseqall, dt,
            self.d2rivsto, self.d2fldsto, self.d2rivout, self.d2fldout,
            self.d2rivinf, self.d2fldinf, self.d2pthout, d1runoff_total,
            self.d2fldfrc, self.d2storge, self.d2outflw
        )

        # ======================================
        # 6.5. RE-CALCULATE flood stage based on UPDATED storage
        # ======================================
        # CRITICAL FIX: Match Fortran behavior
        # Fortran calls CMF_PHYSICS_FLDSTG a SECOND time after STONXT
        # This ensures d2rivdph and d2sfcelv are based on the NEW storage
        # for the next timestep, not the old storage
        # See: cmf_ctrl_physics_mod.F90, line 113
        if self.lfplain:
            diag_fldstg = calc_fldstg(
                self.d2rivsto, self.d2fldsto, self.nseqall, self.nlfp,
                self.d2grarea, self.d2rivlen, self.d2rivwth, self.d2rivelv,
                self.d2rivstomax, self.d2fldstomax, self.d2fldgrd, self.dfrcinc,
                self.d2rivdph, self.d2flddph, self.d2fldfrc, self.d2fldare,
                self.d2sfcelv, self.d2storge
            )
        else:
            # Simple river-only depth calculation (recalculate based on NEW storage)
            for iseq in range(self.nseqall):
                area = self.d2rivlen[iseq] * self.d2rivwth[iseq]
                if area > 0:
                    self.d2rivdph[iseq] = self.d2rivsto[iseq] / area
                else:
                    self.d2rivdph[iseq] = 0.0
                self.d2sfcelv[iseq] = self.d2rivelv[iseq] + self.d2rivdph[iseq]
                self.d2storge[iseq] = self.d2rivsto[iseq]

        # ======================================
        # 7. Debug validation (if enabled)
        # ======================================
        if self.ldebug:
            from .utils import validate_physics_state
            validate_physics_state(self, raise_error=True, verbose=False)

    def get_state(self):
        """
        Get current model state including diagnostic variables

        Returns:
        --------
        state : dict
            Dictionary containing state variables and diagnostic variables
        """
        state = {
            # Core state variables
            'rivsto': self.d2rivsto[:self.nseqall].copy(),
            'rivdph': self.d2rivdph[:self.nseqall].copy(),
            'rivout': self.d2rivout[:self.nseqall].copy(),
            'rivvel': self.d2rivvel[:self.nseqall].copy(),
            'storge': self.d2storge[:self.nseqall].copy(),
            'outflw': self.d2outflw[:self.nseqall].copy(),
            'sfcelv': self.d2sfcelv[:self.nseqall].copy(),

            # Diagnostic variables for averaging
            'runoff': self.d1runoff[:self.nseqall].copy(),
            'pthout': self.d2pthout[:self.nseqall].copy(),
            'gdwrtn': self.d2gdwrtn[:self.nseqall].copy(),
            'rofsub': self.d2rofsub[:self.nseqall].copy(),
        }

        if self.lfplain:
            state.update({
                'fldsto': self.d2fldsto[:self.nseqall].copy(),
                'flddph': self.d2flddph[:self.nseqall].copy(),
                'fldfrc': self.d2fldfrc[:self.nseqall].copy(),
                'fldare': self.d2fldare[:self.nseqall].copy(),
                'fldout': self.d2fldout[:self.nseqall].copy(),
            })

        # Add bifurcation flow if enabled
        if self.lpthout and self.npthout > 0:
            state['pthflw'] = self.d1pthflw.copy() if hasattr(self, 'd1pthflw') else None

        # Add dam variables if enabled (matching Fortran CaMa-Flood)
        if self.ldamout:
            if hasattr(self, 'd2damsto'):
                state['damsto'] = self.d2damsto[:self.nseqall].copy()
            if hasattr(self, 'd2daminf'):
                state['daminf'] = self.d2daminf[:self.nseqall].copy()
            # Note: Dam outflow is represented by rivout at dam grids

        return state

    def set_state(self, state):
        """
        Set model state (for restart)

        Parameters:
        -----------
        state : dict
            Dictionary containing state variables
        """
        if 'rivsto' in state:
            self.d2rivsto[:self.nseqall] = state['rivsto']
        if 'rivdph' in state:
            self.d2rivdph[:self.nseqall] = state['rivdph']

        if self.lfplain:
            if 'fldsto' in state:
                self.d2fldsto[:self.nseqall] = state['fldsto']
