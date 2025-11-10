"""
Main model runner for CaMa-Flood
Orchestrates the complete model simulation

Based on MAIN_cmf.F90 and CMF_DRV_*.F90
"""
import numpy as np
import os
import time as pytime
from datetime import datetime

from .time_control import TimeControl
from .forcing import ForcingData
from .physics import CaMaPhysics
from .output import OutputManager
from .restart import RestartManager
from .dam_operation import DamOperationManager
from .diagnostic import DiagnosticManager


class CaMaFloodRunner:
    """
    Main CaMa-Flood model runner

    This class orchestrates the complete model simulation following
    the structure of MAIN_cmf.F90
    """

    def __init__(self, nml, init_nc_file=None):
        """
        Initialize CaMa-Flood model runner

        Parameters:
        -----------
        nml : Namelist object
            Namelist configuration
        init_nc_file : str, optional
            Initialization netCDF file path (from initialization step)
        """
        self.nml = nml
        self.init_nc_file = init_nc_file

        # Model dimensions (to be read from initialization file)
        self.nx = None
        self.ny = None
        self.nseqall = None
        self.nseqmax = None
        self.nxin = None
        self.nyin = None

        # Model components
        self.time_control = None
        self.forcing = None
        self.physics = None
        self.output = None
        self.diagnostic = None

        # River network data
        self.i2nextx = None
        self.i2nexty = None

        # Grid coordinate data (for CF-compliant output)
        self.lon = None  # 1D longitude array (nx,)
        self.lat = None  # 1D latitude array (ny,)

        print("=" * 70)
        print("CaMa-Flood Model Runner Initialization")
        print("=" * 70)

    def initialize(self):
        """
        Initialize model
        Equivalent to CMF_DRV_INIT in Fortran
        """
        print("\n1. Reading model configuration...")
        self._read_configuration()

        print("\n2. Loading river network data...")
        self._load_river_network()

        print("\n3. Initializing time control...")
        self._initialize_time_control()

        print("\n4. Initializing forcing data...")
        self._initialize_forcing()

        print("\n5. Initializing physics...")
        self._initialize_physics()

        print("\n6. Initializing output...")
        self._initialize_output()

        print("\n7. Initializing restart manager...")
        self._initialize_restart()

        print("\n8. Initializing dam operation...")
        self._initialize_dam_operation()

        print("\n9. Initializing diagnostics...")
        self._initialize_diagnostics()

        print("\n10. Loading initial conditions...")
        self._load_initial_conditions()

        print("\n" + "=" * 70)
        print("Initialization Complete")
        print("=" * 70)

    def _read_configuration(self):
        """Read model configuration from namelist"""
        # Simulation time - read from MODEL_RUN section
        self.syear = self.nml.get('MODEL_RUN', 'syear', 2000)
        self.smon = self.nml.get('MODEL_RUN', 'smon', 1)
        self.sday = self.nml.get('MODEL_RUN', 'sday', 1)
        self.shour = self.nml.get('MODEL_RUN', 'shour', 0)

        self.eyear = self.nml.get('MODEL_RUN', 'eyear', 2001)
        self.emon = self.nml.get('MODEL_RUN', 'emon', 1)
        self.eday = self.nml.get('MODEL_RUN', 'eday', 1)
        self.ehour = self.nml.get('MODEL_RUN', 'ehour', 0)

        # Time step - always get from namelist first (as defaults)
        self.dt = self.nml.get('MODEL_RUN', 'dt', 86400)  # 1 day default
        self.ifrq_inp = self.nml.get('MODEL_RUN', 'ifrq_inp', 24)  # 24 hours

        # Override with diminfo file if it exists
        cdiminfo = self.nml.get('MODEL_RUN', 'cdiminfo', '')
        if cdiminfo and os.path.exists(cdiminfo):
            self._read_diminfo(cdiminfo)

        self.dtin = self.ifrq_inp * 3600  # Convert hours to seconds

        # Other options
        self.lleapyr = self.nml.get('MODEL_RUN', 'lleapyr', True)
        self.lrestart = self.nml.get('MODEL_RUN', 'lrestart', False)

        print(f"  Simulation period: {self.syear}/{self.smon:02d}/{self.sday:02d} {self.shour:02d}:00 "
              f"to {self.eyear}/{self.emon:02d}/{self.eday:02d} {self.ehour:02d}:00")
        print(f"  Time step: {self.dt} seconds")
        print(f"  Input frequency: {self.ifrq_inp} hours")

    def _read_diminfo(self, diminfo_file):
        """Read dimension info file"""
        with open(diminfo_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]

                if key == 'DT':
                    self.dt = float(value)
                elif key == 'IFRQ_INP':
                    self.ifrq_inp = int(value)

    def _load_river_network(self):
        """Load river network data from initialization netCDF file"""
        if self.init_nc_file and os.path.exists(self.init_nc_file):
            print(f"  Loading from: {self.init_nc_file}")
            self._load_from_netcdf(self.init_nc_file)
        else:
            # Try to construct path from namelist
            init_dir = self.nml.get('INIT', 'output_dir', '')
            if init_dir and os.path.exists(init_dir):
                # Look for netCDF file in init directory
                nc_files = [f for f in os.listdir(init_dir) if f.endswith('.nc')]
                if nc_files:
                    self.init_nc_file = os.path.join(init_dir, nc_files[0])
                    print(f"  Found initialization file: {self.init_nc_file}")
                    self._load_from_netcdf(self.init_nc_file)
                    return

            print("  WARNING: No initialization file found, using dummy data")
            self._create_dummy_network()

    def _load_from_netcdf(self, nc_file):
        """Load river network from netCDF initialization file"""
        import netCDF4 as nc

        with nc.Dataset(nc_file, 'r') as f:
            # Read dimensions - handle both formats
            if 'nx' in f.dimensions:
                self.nx = len(f.dimensions['nx'])
                self.ny = len(f.dimensions['ny'])
                self.nseqall = len(f.dimensions['nseqmax']) if 'nseqmax' in f.dimensions else self.nx * self.ny
                self.nseqmax = self.nseqall
            elif 'x' in f.dimensions:
                self.nx = len(f.dimensions['x'])
                self.ny = len(f.dimensions['y'])
                self.nseqall = len(f.dimensions['seq']) if 'seq' in f.dimensions else self.nx * self.ny
                self.nseqmax = self.nseqall
            else:
                raise ValueError("Cannot determine grid dimensions from NetCDF file")

            # Read input dimensions (for forcing)
            # First check if stored in global attributes
            if hasattr(f, 'inpn'):
                inpn = f.inpn
                # Input dimensions not stored separately, use inpn * 0.25 degree logic
                # For 15min global: nx=1440, ny=720, inpn=24 corresponds to 0.25deg input
                # This is a rough estimate - should ideally be stored explicitly
                self.nxin = self.nx  # Simplified assumption
                self.nyin = self.ny
            elif 'xin' in f.dimensions and 'yin' in f.dimensions:
                self.nxin = len(f.dimensions['xin'])
                self.nyin = len(f.dimensions['yin'])
            else:
                # Default: same as river network grid
                self.nxin = self.nx
                self.nyin = self.ny

            # Read longitude and latitude coordinates (for CF-compliant output)
            if 'lon' in f.variables and 'lat' in f.variables:
                self.lon = f.variables['lon'][:]  # (nx,)
                self.lat = f.variables['lat'][:]  # (ny,)
            else:
                # Create dummy lat/lon if not available
                self.lon = np.arange(self.nx, dtype=np.float64)
                self.lat = np.arange(self.ny, dtype=np.float64)
                print("  WARNING: lon/lat not found in initialization file, using dummy coordinates")

            # Read sequence mapping (seq_x, seq_y)
            if 'seq_x' in f.variables and 'seq_y' in f.variables:
                seq_x = f.variables['seq_x'][:]  # 1-based
                seq_y = f.variables['seq_y'][:]  # 1-based

                # Store sequence coordinate arrays for output
                # These map sequence index to grid coordinates
                # Convert to 0-based indexing for Python
                self.i2nextx = seq_x - 1  # Now 1D array (nseqall,)
                self.i2nexty = seq_y - 1  # Now 1D array (nseqall,)

                # Also create 2D mapping for reference
                self.i2vector = np.full((self.nx, self.ny), -9999, dtype=np.int32)
                for iseq in range(self.nseqall):
                    ix = seq_x[iseq] - 1  # Convert to 0-based
                    iy = seq_y[iseq] - 1
                    if 0 <= ix < self.nx and 0 <= iy < self.ny:
                        self.i2vector[ix, iy] = iseq
            elif 'nextx' in f.variables and 'nexty' in f.variables:
                self.i2nextx = f.variables['nextx'][:]
                self.i2nexty = f.variables['nexty'][:]
            else:
                # Create dummy mapping
                self.i2nextx = np.zeros((self.nx, self.ny), dtype=np.int32)
                self.i2nexty = np.zeros((self.nx, self.ny), dtype=np.int32)
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        self.i2nextx[ix, iy] = ix
                        self.i2nexty[ix, iy] = iy

        print(f"  Grid dimensions: {self.nx} x {self.ny}")
        print(f"  Active cells: {self.nseqall}")
        print(f"  Input dimensions: {self.nxin} x {self.nyin}")

    def _create_dummy_network(self):
        """Create dummy river network for testing"""
        self.nx = 100
        self.ny = 100
        self.nseqall = 100
        self.nseqmax = 100
        self.nxin = 100
        self.nyin = 100

        self.i2nextx = np.arange(self.nseqall) % self.nx
        self.i2nexty = np.arange(self.nseqall) // self.nx

        print(f"  Using dummy network: {self.nx} x {self.ny}, {self.nseqall} cells")

    def _initialize_time_control(self):
        """Initialize time control"""
        self.time_control = TimeControl(
            self.syear, self.smon, self.sday, self.shour,
            self.eyear, self.emon, self.eday, self.ehour,
            self.dt, self.lleapyr
        )

        print(f"  Start: {self.time_control.start_time}")
        print(f"  End:   {self.time_control.end_time}")
        print(f"  Total steps: {self.time_control.nsteps}")

    def _initialize_forcing(self):
        """Initialize forcing data handler"""
        self.forcing = ForcingData(self.nml, self.nxin, self.nyin, self.dt, self.dtin)

        # Load input matrix if needed
        if self.forcing.linterp:
            # Try to load from init NetCDF file first
            if self.init_nc_file and os.path.exists(self.init_nc_file):
                try:
                    self._load_inpmat_from_netcdf()
                    print("  Input matrix loaded from initialization file")
                except Exception as e:
                    print(f"  WARNING: Failed to load input matrix from NetCDF: {e}")
                    # Fallback to separate file if specified
                    if self.forcing.cinpmat:
                        try:
                            self.forcing.load_input_matrix()
                        except Exception as e2:
                            print(f"  WARNING: Failed to load input matrix from binary: {e2}")
            elif self.forcing.cinpmat:
                try:
                    self.forcing.load_input_matrix()
                except Exception as e:
                    print(f"  WARNING: Failed to load input matrix: {e}")

    def _load_inpmat_from_netcdf(self):
        """Load input matrix from initialization NetCDF file"""
        import netCDF4 as nc

        with nc.Dataset(self.init_nc_file, 'r') as f:
            # Check if inpmat variables exist
            if 'inpmat_x' not in f.variables or 'inpmat_y' not in f.variables or 'inpmat_area' not in f.variables:
                raise ValueError("Input matrix variables not found in NetCDF file")

            # Read dimensions
            if 'inpn' in f.dimensions:
                inpn = len(f.dimensions['inpn'])
            else:
                raise ValueError("INPN dimension not found in NetCDF file")

            # Read input matrix
            inpmat_x = f.variables['inpmat_x'][:]  # (nseqmax, inpn)
            inpmat_y = f.variables['inpmat_y'][:]  # (nseqmax, inpn)
            inpmat_area = f.variables['inpmat_area'][:]  # (nseqmax, inpn) - absolute areas in m²

            # Store in forcing object
            # NOTE: inpa stores ABSOLUTE AREAS (m²), not normalized fractions!
            # The runoff interpolation (forcing.py:657) multiplies runoff[mm/day] * area[m²] / drofunit
            # This gives proper area-weighted runoff in m³/s
            self.forcing.inpn = inpn
            self.forcing.nseqmax = self.nseqmax
            self.forcing.nx = self.nx
            self.forcing.ny = self.ny
            self.forcing.inpx = inpmat_x
            self.forcing.inpy = inpmat_y
            self.forcing.inpa = inpmat_area  # Use absolute areas for proper runoff calculation
            self.forcing.inpmat_loaded = True

    def _initialize_physics(self):
        """Initialize physics module"""
        self.physics = CaMaPhysics(self.nml, self.nx, self.ny, self.nseqall, self.nseqmax)

        # CRITICAL: Load river network parameters from initialization file
        if self.init_nc_file and os.path.exists(self.init_nc_file):
            print("\n  Loading river network parameters from initialization file...")
            self.physics.load_parameters(self.init_nc_file)
        else:
            print("\n  WARNING: No initialization file found - using dummy parameters")
            print("  This will cause incorrect simulation results!")

    def _initialize_output(self):
        """Initialize output manager"""
        self.output = OutputManager(self.nml, self.nx, self.ny, self.nseqall, self.time_control,
                                     lon=self.lon, lat=self.lat)
        self.output.initialize_output()

    def _initialize_restart(self):
        """Initialize restart manager"""
        self.restart_manager = RestartManager(self.nml, self.physics)

    def _initialize_dam_operation(self):
        """Initialize dam operation manager"""
        self.dam_manager = DamOperationManager(self.nml, self.physics)
        self.dam_manager.initialize()

    def _initialize_diagnostics(self):
        """Initialize diagnostic manager"""
        # Get bifurcation parameters from physics if enabled
        npthout = self.physics.npthout if self.physics.lpthout else 0
        npthlev = self.physics.npthlev if self.physics.lpthout else 1

        # Get dam and evaporation flags
        ldamout = self.physics.ldamout
        lwevap = self.nml.get('MODEL_RUN', 'lwevap', False)

        self.diagnostic = DiagnosticManager(
            self.nseqmax,
            npthout=npthout,
            npthlev=npthlev,
            ldamout=ldamout,
            lwevap=lwevap
        )

        print(f"  Diagnostic manager initialized")
        print(f"  Dam diagnostics: {'enabled' if ldamout else 'disabled'}")
        print(f"  Bifurcation diagnostics: {'enabled' if npthout > 0 else 'disabled'}")

    def _load_initial_conditions(self):
        """Load initial conditions from restart file or start from zero"""
        if self.lrestart:
            self.physics.load_restart(self.restart_manager)
        else:
            print("  Starting from zero storage")

    def run(self):
        """
        Main model run loop
        Equivalent to the main time loop in MAIN_cmf.F90
        """
        print("\n" + "=" * 70)
        print("Starting Model Simulation")
        print("=" * 70)

        start_wall_time = pytime.time()

        # Calculate steps per input forcing update
        istepadv = int(self.dtin / self.dt)

        # Main temporal loop
        istep = 0
        while not self.time_control.is_finished():
            step_start_time = pytime.time()

            # Read forcing data
            try:
                zbuff = self.forcing.forcing_get(self.time_control.current_time)
            except FileNotFoundError as e:
                print(f"\nERROR: Forcing file not found: {e}")
                print("Stopping simulation")
                break
            except Exception as e:
                print(f"\nWARNING: Error reading forcing: {e}")
                zbuff = np.zeros((self.nxin, self.nyin, 2))

            # Interpolate and distribute forcing
            p0_rivsto, p0_evap = self.forcing.forcing_put(zbuff, self.forcing.inpmat, self.nseqall, self.time_control.current_time)
            self.physics.set_runoff_input(p0_rivsto)

            # Set evaporation if enabled
            if p0_evap is not None:
                self.physics.set_evaporation_input(p0_evap)

            # Advance model for ISTEPADV time steps
            for i in range(istepadv):
                # Update time
                self.time_control.time_next()

                # Reset adaptive timestep diagnostics at start of main timestep
                self.diagnostic.reset_adp()

                # Calculate adaptive time step if enabled
                # This determines how many sub-steps are needed for numerical stability
                nt_sub, dt_sub = self.physics.calculate_adaptive_timestep(self.dt)

                # Loop over sub time steps (nt_sub = 1 if adaptive timestep is disabled)
                for it_sub in range(nt_sub):
                    # Advance physics with sub time step
                    self.physics.physics_advance(dt_sub)

                    # Calculate dam releases (if enabled)
                    # Note: This is done at each sub-step for accuracy
                    self.dam_manager.calculate_dam_release(dt_sub)

                    # Get current state after sub-step
                    state = self.physics.get_state()

                    # Accumulate diagnostics at adaptive timestep level
                    self.diagnostic.accumulate_adp(state, dt_sub)

                # Finalize adaptive timestep diagnostics (calculate time-averaged values)
                self.diagnostic.finalize_adp()

                # Accumulate diagnostics for output interval
                self.diagnostic.accumulate_out(self.dt)

                # Get current state (after all sub-steps completed)
                state = self.physics.get_state()

                # Finalize output diagnostics if it's output time
                diagnostics = None
                if self.time_control.is_output_time(self.output.ifrq_out):
                    # Finalize diagnostic averages and maximums
                    self.diagnostic.finalize_out()
                    # Get diagnostic output data
                    diagnostics = self.diagnostic.get_output_diagnostics(self.nseqall)
                    # Reset output diagnostics for next interval
                    self.diagnostic.reset_out()

                # Write output
                self.output.write_output(state, self.i2nextx, self.i2nexty, diagnostics)

                # Write restart if needed
                self.restart_manager.write_restart(self.time_control)

                # Print progress
                if istep % max(1, self.time_control.nsteps // 20) == 0:
                    progress = self.time_control.get_progress()
                    elapsed = pytime.time() - start_wall_time
                    print(f"  {self.time_control} | Progress: {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s")

                istep += 1

        # End of simulation
        elapsed_time = pytime.time() - start_wall_time

        print("\n" + "=" * 70)
        print("Simulation Complete")
        print("=" * 70)
        print(f"Total steps: {istep}")
        print(f"Wall clock time: {elapsed_time:.2f} seconds")
        print(f"Time per step: {elapsed_time/istep*1000:.2f} ms" if istep > 0 else "")

    def finalize(self):
        """
        Finalize model run
        Equivalent to CMF_DRV_END in Fortran
        """
        print("\n" + "=" * 70)
        print("Finalizing Model")
        print("=" * 70)

        # Close output files
        if self.output:
            self.output.finalize_output()

        print("Model finalization complete")


def run_camaflood_model(nml, init_nc_file=None):
    """
    Convenience function to run CaMa-Flood model

    Parameters:
    -----------
    nml : Namelist object
        Namelist configuration
    init_nc_file : str, optional
        Initialization netCDF file path

    Returns:
    --------
    success : bool
        True if simulation completed successfully
    """
    try:
        # Create runner
        runner = CaMaFloodRunner(nml, init_nc_file)

        # Initialize
        runner.initialize()

        # Run
        runner.run()

        # Finalize
        runner.finalize()

        return True

    except Exception as e:
        print(f"\nERROR in model run: {e}")
        import traceback
        traceback.print_exc()
        return False
