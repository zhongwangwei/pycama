"""
Forcing module for CaMa-Flood model run
Handles reading and interpolating runoff forcing data

Based on CMF_CTRL_FORCING_MOD.F90

OPTIMIZED: Using Numba JIT-compiled interpolation for 10x speedup
"""
import numpy as np
import os
from datetime import datetime, timedelta
from .forcing_optimized import roff_interp_optimized, conv_resol_optimized


class ForcingData:
    """Manage CaMa-Flood forcing data (runoff)"""

    def __init__(self, nml, nxin, nyin, dt, dtin):
        """
        Initialize forcing data handler

        Parameters:
        -----------
        nml : Namelist object
            Namelist configuration
        nxin : int
            Input X dimension
        nyin : int
            Input Y dimension
        dt : float
            Model time step (seconds)
        dtin : float
            Input forcing time step (seconds)
        """
        self.nml = nml
        self.nxin = nxin
        self.nyin = nyin
        self.dt = dt
        self.dtin = dtin

        # Read forcing configuration from namelist
        self._read_forcing_config()

        # Data buffer for storing forcing data
        # ZBUFF dimensions depend on whether evaporation is enabled:
        # - Layer 0: Total runoff (or surface runoff if LROSPLIT)
        # - Layer 1: Subsurface runoff (if LROSPLIT) or duplicate
        # - Layer 2: Evaporation (if LWEVAP)
        # We always allocate 3 layers for consistency, even if not all are used
        self.zbuff = np.zeros((nxin, nyin, 3), dtype=np.float64)

        # Current forcing file date
        self.current_forcing_date = None

        # Time tracking for temporal interpolation
        self.forcing_time_prev = None  # Previous forcing time
        self.forcing_time_next = None  # Next forcing time
        self.zbuff_loaded_time = None  # Time when zbuff was last updated
        self.sim_start_time = None     # Simulation start time (set by runner)

        # Interpolation matrix (if LINTERP is True)
        self.inpmat = None
        self.inpmat_loaded = False

        # Input matrix arrays (for binary format)
        self.inpx = None  # X coordinates of contributing cells
        self.inpy = None  # Y coordinates of contributing cells
        self.inpa = None  # Weight factors (area fractions)
        self.inpn = None  # Number of contributing cells per river cell
        self.nx = None    # Grid dimensions
        self.ny = None
        self.nseqmax = None

    def _read_forcing_config(self):
        """Read forcing configuration from namelist"""
        # Forcing type - read from MODEL_RUN section
        self.linpcdf = self.nml.get('MODEL_RUN', 'linpcdf', False)  # netCDF or binary
        self.linterp = self.nml.get('MODEL_RUN', 'linterp', True)   # Use interpolation matrix
        self.linterp_time = self.nml.get('MODEL_RUN', 'linterp_time', True)  # Temporal interpolation
        self.linterpcdf = self.nml.get('MODEL_RUN', 'litrpcdf', False)  # netCDF input matrix
        self.linpend = self.nml.get('MODEL_RUN', 'linpend', False)  # Endian conversion

        # Unit conversion
        self.drofunit = self.nml.get('MODEL_RUN', 'drofunit', 86400000.0)  # mm/day to m/s

        # ------------------------------------------------------------------------
        # Forcing File Organization
        # ------------------------------------------------------------------------
        # How forcing files are organized (single/yearly/monthly/daily)
        self.forcing_file_freq = self.nml.get('MODEL_RUN', 'forcing_file_freq', 'single').lower()
        if self.forcing_file_freq not in ['single', 'yearly', 'monthly', 'daily']:
            print(f"WARNING: Invalid forcing_file_freq '{self.forcing_file_freq}', using 'single'")
            self.forcing_file_freq = 'single'

        # NetCDF forcing files
        # For 'single' mode: use crofcdf
        self.crofcdf = self.nml.get('MODEL_RUN', 'crofcdf', '')

        # For multi-file mode (yearly/monthly/daily): use directory + prefix + suffix
        self.crofdir_nc = self.nml.get('MODEL_RUN', 'crofdir_nc', '')
        self.crofpre_nc = self.nml.get('MODEL_RUN', 'crofpre_nc', 'RUNOFF_')
        self.crofsuf_nc = self.nml.get('MODEL_RUN', 'crofsuf_nc', '.nc')

        # Binary forcing files (for backwards compatibility)
        self.crofdir = self.nml.get('MODEL_RUN', 'crofdir', '')
        self.crofpre = self.nml.get('MODEL_RUN', 'crofpre', 'Roff____')
        self.crofsuf = self.nml.get('MODEL_RUN', 'crofsuf', '.one')

        # Print forcing file organization info
        if self.linpcdf:
            if self.forcing_file_freq == 'single':
                print(f"INFO: Using single NetCDF file: {self.crofcdf}")
            else:
                print(f"INFO: Using {self.forcing_file_freq} NetCDF files from: {self.crofdir_nc}")
                print(f"      File pattern: {self.crofpre_nc}[date]{self.crofsuf_nc}")
            # Calculate ifrq_inp from dtin for display
            ifrq_inp_hours = int(self.dtin / 3600)
            print(f"      Forcing temporal resolution: {ifrq_inp_hours} hours")

        # Sub-surface runoff (optional)
        self.lrosplit = self.nml.get('MODEL_RUN', 'lrosplit', False)
        self.csubdir = self.nml.get('MODEL_RUN', 'csubdir', '')
        self.csubpre = self.nml.get('MODEL_RUN', 'csubpre', '')
        self.csubsuf = self.nml.get('MODEL_RUN', 'csubsuf', '')

        # Evaporation extraction (optional)
        self.lwevap = self.nml.get('MODEL_RUN', 'lwevap', False)
        self.lwevapfix = self.nml.get('MODEL_RUN', 'lwevapfix', False)
        self.lwextractriv = self.nml.get('MODEL_RUN', 'lwextractriv', False)
        self.cevpdir = self.nml.get('MODEL_RUN', 'cevpdir', '')
        self.cevppre = self.nml.get('MODEL_RUN', 'cevppre', 'Evap____')
        self.cevpsuf = self.nml.get('MODEL_RUN', 'cevpsuf', '.one')
        self.cevpcdf = self.nml.get('MODEL_RUN', 'cevpcdf', '')
        self.cvarevp = self.nml.get('MODEL_RUN', 'cvarevp', 'evap')

        # netCDF forcing variable names
        self.cvnrof = self.nml.get('MODEL_RUN', 'cvnrof', 'runoff')
        self.cvnsub = self.nml.get('MODEL_RUN', 'cvnsub', '')

        # Input matrix file
        self.cinpmat = self.nml.get('MODEL_RUN', 'cinpmat', '')

    def load_input_matrix(self):
        """
        Load input matrix for runoff interpolation

        The input matrix defines how to map coarse-resolution runoff to
        fine-resolution river network cells. For each river cell, it stores:
        - INPX: X coordinates of contributing input cells
        - INPY: Y coordinates of contributing input cells
        - INPA: Weight factors (area fractions)

        Binary format (Fortran unformatted, direct access):
        - Records 1 to INPN: INPX data (integer, NX*NY each)
        - Records INPN+1 to 2*INPN: INPY data (integer, NX*NY each)
        - Records 2*INPN+1 to 3*INPN: INPA data (real, NX*NY each)

        After loading, arrays are in vector format (NSEQMAX, INPN)
        """
        if not self.linterp:
            print("INFO: Interpolation is disabled, skipping input matrix loading")
            return

        if not os.path.exists(self.cinpmat):
            raise FileNotFoundError(f"Input matrix file not found: {self.cinpmat}")

        print(f"Loading input matrix: {self.cinpmat}")

        if self.linterpcdf:
            # netCDF input matrix
            self._load_input_matrix_netcdf()
        else:
            # Binary input matrix
            self._load_input_matrix_binary()

        self.inpmat_loaded = True
        print(f"Input matrix loaded successfully")
        print(f"  - INPN (max contributing cells): {self.inpn}")
        print(f"  - Matrix shape: INPX/INPY/INPA ({self.nseqmax}, {self.inpn})")

    def _load_input_matrix_binary(self):
        """Load binary input matrix file (Fortran format)"""
        # First, we need to determine the grid size and INPN
        # This should come from the diminfo file or initialization
        # For now, we'll try to infer from file size

        file_size = os.path.getsize(self.cinpmat)

        # Try to find diminfo file to get NX, NY, INPN
        # Look in the same directory as the input matrix
        inpmat_dir = os.path.dirname(self.cinpmat)
        diminfo_candidates = [
            os.path.join(inpmat_dir, 'diminfo.txt'),
            self.nml.get('MODEL_RUN', 'cdiminfo', '')
        ]

        nx, ny, inpn = None, None, None
        for diminfo_path in diminfo_candidates:
            if diminfo_path and os.path.exists(diminfo_path):
                try:
                    nx, ny, inpn = self._read_diminfo_for_inpmat(diminfo_path)
                    if nx and ny and inpn:
                        break
                except:
                    continue

        if not (nx and ny and inpn):
            # Try to infer from file size
            # File contains 3*INPN records, each record is NX*NY*4 bytes
            # file_size = 3 * INPN * NX * NY * 4
            # Try common values
            for test_inpn in [10, 20, 30]:
                test_nx_ny = file_size // (3 * test_inpn * 4)
                test_nx = int(np.sqrt(test_nx_ny))
                if test_nx * test_nx == test_nx_ny:
                    nx, ny, inpn = test_nx, test_nx, test_inpn
                    print(f"  WARNING: Inferred grid size from file: NX={nx}, NY={ny}, INPN={inpn}")
                    break

            if not (nx and ny and inpn):
                raise ValueError(f"Cannot determine input matrix dimensions. File size: {file_size} bytes. "
                               "Please ensure CDIMINFO is set correctly in namelist.")

        self.nx = nx
        self.ny = ny
        self.inpn = inpn
        self.nseqmax = nx * ny  # For simple cases, assume all cells are active

        # Read binary file (Fortran unformatted, direct access)
        # Each record is 4 bytes per element (integer or real*4)
        record_size = nx * ny * 4  # bytes

        # Allocate arrays
        self.inpx = np.zeros((self.nseqmax, inpn), dtype=np.int32)
        self.inpy = np.zeros((self.nseqmax, inpn), dtype=np.int32)
        self.inpa = np.zeros((self.nseqmax, inpn), dtype=np.float32)

        with open(self.cinpmat, 'rb') as f:
            # Read INPX (records 1 to INPN)
            for inpi in range(inpn):
                f.seek(inpi * record_size)
                data = np.fromfile(f, dtype=np.int32, count=nx*ny)
                self.inpx[:, inpi] = data.flatten()

            # Read INPY (records INPN+1 to 2*INPN)
            for inpi in range(inpn):
                f.seek((inpn + inpi) * record_size)
                data = np.fromfile(f, dtype=np.int32, count=nx*ny)
                self.inpy[:, inpi] = data.flatten()

            # Read INPA (records 2*INPN+1 to 3*INPN)
            for inpi in range(inpn):
                f.seek((2*inpn + inpi) * record_size)
                data = np.fromfile(f, dtype=np.float32, count=nx*ny)
                self.inpa[:, inpi] = data.flatten()

    def _load_input_matrix_netcdf(self):
        """Load netCDF input matrix file"""
        import netCDF4 as nc

        with nc.Dataset(self.cinpmat, 'r') as f:
            # Read dimensions
            if 'nseqmax' in f.dimensions:
                self.nseqmax = len(f.dimensions['nseqmax'])
            elif 'seq' in f.dimensions:
                self.nseqmax = len(f.dimensions['seq'])
            else:
                raise ValueError("Cannot find sequence dimension in netCDF input matrix")

            if 'inpn' in f.dimensions:
                self.inpn = len(f.dimensions['inpn'])
            else:
                raise ValueError("Cannot find inpn dimension in netCDF input matrix")

            # Read variables
            self.inpx = f.variables['inpx'][:]
            self.inpy = f.variables['inpy'][:]
            self.inpa = f.variables['inpa'][:]

    def _read_diminfo_for_inpmat(self, diminfo_file):
        """Read NX, NY, INPN from diminfo file"""
        nx, ny, inpn = None, None, None

        with open(diminfo_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].upper()
                    value = parts[1]

                    if key == 'NX':
                        nx = int(value)
                    elif key == 'NY':
                        ny = int(value)
                    elif key == 'INPN':
                        inpn = int(value)

        return nx, ny, inpn

    def get_forcing_filename(self, current_time):
        """
        Get forcing filename for given time based on file organization

        Parameters:
        -----------
        current_time : datetime
            Current simulation time

        Returns:
        --------
        filename : str
            Forcing filename
        """
        if self.linpcdf:
            # NetCDF forcing
            if self.forcing_file_freq == 'single':
                # Single file containing all time steps
                return self.crofcdf
            elif self.forcing_file_freq == 'yearly':
                # One file per year: RUNOFF_1980.nc, RUNOFF_1981.nc, ...
                date_str = current_time.strftime('%Y')
                filename = os.path.join(self.crofdir_nc,
                                       f"{self.crofpre_nc}{date_str}{self.crofsuf_nc}")
                return filename
            elif self.forcing_file_freq == 'monthly':
                # One file per month: RUNOFF_198001.nc, RUNOFF_198002.nc, ...
                date_str = current_time.strftime('%Y%m')
                filename = os.path.join(self.crofdir_nc,
                                       f"{self.crofpre_nc}{date_str}{self.crofsuf_nc}")
                return filename
            elif self.forcing_file_freq == 'daily':
                # One file per day: RUNOFF_19800101.nc, RUNOFF_19800102.nc, ...
                date_str = current_time.strftime('%Y%m%d')
                filename = os.path.join(self.crofdir_nc,
                                       f"{self.crofpre_nc}{date_str}{self.crofsuf_nc}")
                return filename
            else:
                raise ValueError(f"Unknown forcing_file_freq: {self.forcing_file_freq}")
        else:
            # Binary forcing: daily files (backwards compatibility)
            date_str = current_time.strftime('%Y%m%d')
            filename = os.path.join(self.crofdir,
                                   f"{self.crofpre}{date_str}{self.crofsuf}")
            return filename

    def read_forcing_binary(self, filename):
        """
        Read binary forcing file

        Parameters:
        -----------
        filename : str
            Binary forcing filename

        Returns:
        --------
        data : ndarray
            Forcing data (nxin, nyin)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Forcing file not found: {filename}")

        # Read binary data
        dtype = np.float32
        if self.linpend:
            # Endian conversion
            dtype = dtype.newbyteorder()

        data = np.fromfile(filename, dtype=dtype)

        # Reshape to (nxin, nyin)
        if data.size != self.nxin * self.nyin:
            raise ValueError(f"Forcing file size mismatch: expected {self.nxin * self.nyin}, got {data.size}")

        data = data.reshape((self.nyin, self.nxin)).T  # Transpose for Fortran order

        return data

    def read_forcing_netcdf(self, filename, time_index):
        """
        Read netCDF forcing file

        Parameters:
        -----------
        filename : str
            netCDF forcing filename
        time_index : int
            Time index in netCDF file

        Returns:
        --------
        data : ndarray
            Forcing data (nxin, nyin)
        """
        import netCDF4 as nc

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Forcing file not found: {filename}")

        with nc.Dataset(filename, 'r') as f:
            # Read runoff data
            # NetCDF typically stores as (time, lat, lon) = (time, ny, nx)
            # We need to transpose to (nx, ny) = (lon, lat) format
            data = f.variables[self.cvnrof][time_index, :, :]

            # Read sub-surface runoff if needed
            if self.lrosplit and self.cvnsub:
                data_sub = f.variables[self.cvnsub][time_index, :, :]
                data = data + data_sub

            # Check if we need to transpose
            # If data shape is (ny, nx) but we expect (nx, ny), transpose
            if data.shape == (self.nyin, self.nxin):
                data = data.T  # Transpose from (ny, nx) to (nx, ny)
            elif data.shape != (self.nxin, self.nyin):
                # Unexpected shape
                print(f"WARNING: Forcing data shape {data.shape} doesn't match expected ({self.nxin}, {self.nyin})")
                # Try transpose anyway
                data = data.T

            # Handle missing values - convert masked array or replace fill values with 0
            if np.ma.is_masked(data):
                data = np.ma.filled(data, 0.0)
            else:
                # Replace any remaining fill values with 0
                fill_value = getattr(f.variables[self.cvnrof], '_FillValue', -999000000.0)
                data = np.where(np.abs(data - fill_value) < 1e-3, 0.0, data)

        # Ensure non-negative values
        data = np.maximum(data, 0.0)

        return data

    def forcing_get(self, current_time):
        """
        Read forcing data from file with temporal interpolation support
        Similar to CMF_FORCING_GET in Fortran

        Parameters:
        -----------
        current_time : datetime
            Current simulation time

        Returns:
        --------
        zbuff : ndarray
            Data buffer (nxin, nyin, 2)
            Layer 0: forcing at previous timestep
            Layer 1: forcing at next timestep
        """
        # Set simulation start time if not already set
        # Use the first current_time as the simulation start time
        if self.sim_start_time is None:
            self.sim_start_time = current_time

        # Calculate which forcing timestep we're in
        # Forcing is read at the START of each interval, then used for that interval
        # For example, at time 1980-01-01 00:00, use forcing index 0 for Day 1
        # At time 1980-01-02 00:00, use forcing index 1 for Day 2
        time_seconds = (current_time - self.sim_start_time).total_seconds()
        forcing_index = int(time_seconds / self.dtin)

        # Calculate the bracketing forcing times
        forcing_time_prev = self.sim_start_time + timedelta(seconds=forcing_index * self.dtin)
        forcing_time_next = forcing_time_prev + timedelta(seconds=self.dtin)

        # Check if we need to update zbuff
        # Only reload if we've moved to a new forcing interval
        if self.zbuff_loaded_time != forcing_time_prev:
            if self.linterp_time:
                # Load both previous and next timesteps for interpolation
                # Runoff (always layer 0)
                self.zbuff[:, :, 0] = self._read_forcing_at_time(forcing_time_prev, 'runoff')
                self.zbuff[:, :, 1] = self._read_forcing_at_time(forcing_time_next, 'runoff')

                # Evaporation (layer 2, if enabled)
                if self.lwevap:
                    self.zbuff[:, :, 2] = self._read_evaporation_at_time(forcing_time_prev)
                    # For temporal interpolation of evaporation, we'd need another buffer
                    # For now, use the same evap value (no temporal interpolation for evap)
                    # This is a simplification - full implementation would interpolate evap too
            else:
                # No temporal interpolation, load only current timestep
                data = self._read_forcing_at_time(forcing_time_prev, 'runoff')
                self.zbuff[:, :, 0] = data
                self.zbuff[:, :, 1] = data

                # Evaporation (if enabled)
                if self.lwevap:
                    self.zbuff[:, :, 2] = self._read_evaporation_at_time(forcing_time_prev)

            # Update tracking variables
            self.zbuff_loaded_time = forcing_time_prev
            self.forcing_time_prev = forcing_time_prev
            self.forcing_time_next = forcing_time_next

        return self.zbuff

    def _read_forcing_at_time(self, target_time, forcing_type='runoff'):
        """
        Read forcing data at a specific time

        Parameters:
        -----------
        target_time : datetime
            Target time to read forcing
        forcing_type : str
            Type of forcing to read ('runoff' or 'evap')

        Returns:
        --------
        data : ndarray (nxin, nyin)
            Forcing data at target time [m/s]
        """
        # Get filename
        filename = self.get_forcing_filename(target_time)

        # Read data
        if self.linpcdf:
            # netCDF forcing - calculate time index based on groupby mode
            time_index = self._calculate_time_index(target_time)
            data = self.read_forcing_netcdf(filename, time_index)
        else:
            # Binary forcing
            data = self.read_forcing_binary(filename)

        # NOTE: Unit conversion is done in forcing_put (roff_interp or conv_resol)
        # Do NOT convert here to avoid double conversion
        return data

    def _calculate_time_index(self, target_time):
        """
        Calculate time index WITHIN a forcing file

        This method calculates the index into the time dimension of the NetCDF file,
        considering how files are organized (forcing_file_freq).
        All forcing files contain continuous time series at the resolution defined by ifrq_inp.

        Parameters:
        -----------
        target_time : datetime
            Target time to read forcing

        Returns:
        --------
        time_index : int
            Index into netCDF time dimension for the current file
        """
        # All forcing files contain continuous time series
        # Calculate index based on position within the file's time range

        if self.forcing_file_freq == 'single':
            # Single file: calculate index from simulation start time
            # sim_start_time is set on first call to forcing_get
            if self.sim_start_time is None:
                # If not set yet, use target_time (should not happen in normal operation)
                self.sim_start_time = target_time
            time_diff = (target_time - self.sim_start_time).total_seconds()
            time_index = int(time_diff / self.dtin)

        elif self.forcing_file_freq == 'yearly':
            # Yearly files: calculate index from start of current year
            # File contains data for one year at ifrq_inp resolution
            year_start = datetime(target_time.year, 1, 1, 0, 0)
            time_diff = (target_time - year_start).total_seconds()
            time_index = int(time_diff / self.dtin)

        elif self.forcing_file_freq == 'monthly':
            # Monthly files: calculate index from start of current month
            # File contains data for one month at ifrq_inp resolution
            month_start = datetime(target_time.year, target_time.month, 1, 0, 0)
            time_diff = (target_time - month_start).total_seconds()
            time_index = int(time_diff / self.dtin)

        elif self.forcing_file_freq == 'daily':
            # Daily files: calculate index from start of current day
            # File contains data for one day at ifrq_inp resolution (e.g., 24 hourly values)
            day_start = datetime(target_time.year, target_time.month, target_time.day, 0, 0)
            time_diff = (target_time - day_start).total_seconds()
            time_index = int(time_diff / self.dtin)

        else:
            raise ValueError(f"Unknown forcing_file_freq: {self.forcing_file_freq}")

        return time_index

    def _read_evaporation_at_time(self, target_time):
        """
        Read evaporation data at a specific time

        Parameters:
        -----------
        target_time : datetime
            Target time to read evaporation

        Returns:
        --------
        data : ndarray (nxin, nyin)
            Evaporation data at target time [m/s]
        """
        # Get evaporation filename
        if self.linpcdf:
            # netCDF evaporation
            filename = self.cevpcdf
            if not filename or not os.path.exists(filename):
                # No evaporation file, return zeros
                return np.zeros((self.nxin, self.nyin), dtype=np.float64)

            import netCDF4 as nc
            # Use the same groupby logic as forcing
            time_index = self._calculate_time_index(target_time)

            with nc.Dataset(filename, 'r') as f:
                data = f.variables[self.cvarevp][time_index, :, :]
        else:
            # Binary evaporation
            date_str = target_time.strftime('%Y%m%d')
            filename = os.path.join(self.cevpdir, f"{self.cevppre}{date_str}{self.cevpsuf}")

            if not os.path.exists(filename):
                # No evaporation file, return zeros
                return np.zeros((self.nxin, self.nyin), dtype=np.float64)

            # Read binary data (same format as runoff)
            dtype = np.float32
            if self.linpend:
                dtype = dtype.newbyteorder()

            data = np.fromfile(filename, dtype=dtype)
            if data.size != self.nxin * self.nyin:
                raise ValueError(f"Evaporation file size mismatch: expected {self.nxin * self.nyin}, got {data.size}")

            data = data.reshape((self.nyin, self.nxin)).T  # Transpose for Fortran order

        # NOTE: Unit conversion is done in forcing_put (roff_interp or conv_resol)
        # Do NOT convert here to avoid double conversion
        return data

    def forcing_put(self, zbuff, inpmat, nseqall, current_time=None):
        """
        Interpolate forcing data and distribute to river grid
        Similar to CMF_FORCING_PUT in Fortran

        Parameters:
        -----------
        zbuff : ndarray
            Data buffer (nxin, nyin, 3)
            Layer 0: runoff at previous timestep
            Layer 1: runoff at next timestep
            Layer 2: evaporation (if LWEVAP)
        inpmat : ndarray or None
            Input matrix for interpolation (not used - we use self.inpx/y/a instead)
        nseqall : int
            Total number of river grid cells
        current_time : datetime, optional
            Current simulation time (required for temporal interpolation)

        Returns:
        --------
        p0_rivsto : ndarray
            Distributed runoff for each river cell [m3/s]
        p0_evap : ndarray or None
            Distributed evaporation for each river cell [m3/s], None if LWEVAP=False
        """
        # Apply temporal interpolation to runoff if enabled
        if self.linterp_time and current_time is not None and self.forcing_time_prev is not None:
            # Calculate interpolation weight
            time_seconds = (current_time - self.forcing_time_prev).total_seconds()
            dt_forcing = (self.forcing_time_next - self.forcing_time_prev).total_seconds()

            # Weight ranges from 0.0 (at forcing_time_prev) to 1.0 (at forcing_time_next)
            weight = time_seconds / dt_forcing if dt_forcing > 0 else 0.0

            # Linear interpolation between two timesteps for runoff
            pbuffin = zbuff[:, :, 0] * (1.0 - weight) + zbuff[:, :, 1] * weight
        else:
            # No temporal interpolation, use first layer
            pbuffin = zbuff[:, :, 0]

        # Apply spatial interpolation to runoff
        if self.linterp and self.inpmat_loaded:
            # Use input matrix interpolation (mass conservative)
            p0_rivsto = self._roff_interp(pbuffin, nseqall)
        else:
            # Direct mapping (no interpolation)
            # Assumes input grid matches river network grid
            p0_rivsto = self._conv_resol(pbuffin, nseqall)

        # Process evaporation if enabled
        if self.lwevap and zbuff.shape[2] >= 3:
            # Extract evaporation data (layer 2)
            pbuffin_evap = zbuff[:, :, 2]

            # Apply spatial interpolation to evaporation
            if self.linterp and self.inpmat_loaded:
                p0_evap = self._roff_interp(pbuffin_evap, nseqall)
            else:
                p0_evap = self._conv_resol(pbuffin_evap, nseqall)
        else:
            p0_evap = None

        return p0_rivsto, p0_evap

    def _roff_interp(self, pbuffin, nseqall):
        """
        Runoff interpolation with mass conservation using input matrix

        OPTIMIZED: Uses Numba JIT compilation for ~10x speedup

        Similar to ROFF_INTERP subroutine in Fortran

        Parameters:
        -----------
        pbuffin : ndarray (nxin, nyin)
            Input runoff data [mm/dt]
        nseqall : int
            Number of river cells

        Returns:
        --------
        pbuffout : ndarray (nseqall,)
            Interpolated runoff [m3/s]
        """
        # Missing value
        rmis = self.nml.get('NPARAM', 'RMIS', 1.0e20)

        # Use JIT-compiled version for speed (handles masked arrays)
        pbuffout = roff_interp_optimized(
            pbuffin, nseqall, self.inpn,
            self.inpx, self.inpy, self.inpa,
            self.nxin, self.nyin, self.drofunit, rmis
        )

        return pbuffout

    def _conv_resol(self, pbuffin, nseqall):
        """
        Direct resolution conversion without interpolation

        OPTIMIZED: Uses Numba JIT compilation for speedup

        Similar to CONV_RESOL subroutine in Fortran
        Assumes map resolution and runoff resolution are the same

        Parameters:
        -----------
        pbuffin : ndarray (nxin, nyin)
            Input runoff data [mm/dt]
        nseqall : int
            Number of river cells

        Returns:
        --------
        pbuffout : ndarray (nseqall,)
            Converted runoff [m3/s]
        """
        # Missing value
        rmis = self.nml.get('NPARAM', 'RMIS', 1.0e20)

        # Flatten input to 1D vector
        pbuffin_vec = pbuffin.flatten()

        # Use JIT-compiled version for speed (handles masked arrays)
        pbuffout = conv_resol_optimized(pbuffin_vec, nseqall, self.drofunit, rmis)

        return pbuffout
