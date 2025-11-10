"""
Output module for CaMa-Flood model run
Handles writing model outputs to netCDF or binary files

Based on CMF_CTRL_OUTPUT_MOD.F90
"""
import numpy as np
import os
from datetime import datetime
import netCDF4 as nc
from scipy.io import FortranFile
import struct


class OutputManager:
    """Manage CaMa-Flood output writing"""

    def __init__(self, nml, nx, ny, nseqall, time_control, lon=None, lat=None):
        """
        Initialize output manager

        Parameters:
        -----------
        nml : Namelist object
            Namelist configuration
        nx, ny : int
            Grid dimensions
        nseqall : int
            Total number of active river cells
        time_control : TimeControl object
            Time control instance
        lon : ndarray, optional
            Longitude array (nx,)
        lat : ndarray, optional
            Latitude array (ny,)
        """
        self.nml = nml
        self.nx = nx
        self.ny = ny
        self.nseqall = nseqall
        self.time_control = time_control
        self.lon = lon
        self.lat = lat

        # Read output configuration
        self._read_output_config()

        # Initialize output files
        self.nc_file = None
        self.nc_vars = {}
        self.time_dim = None
        self.time_var = None
        self.time_index = 0

        # Binary output files (dict: varname -> file handle)
        self.bin_files = {}
        self.bin_record_num = 0

    def _read_output_config(self):
        """Read output configuration from namelist"""
        # Read from OUTPUT section for base configuration
        output_base_dir = self.nml.get('OUTPUT', 'output_base_dir', 'output/')
        case_name = self.nml.get('OUTPUT', 'case_name', 'test')

        # Read from MODEL_RUN section
        self.loutput = self.nml.get('MODEL_RUN', 'loutput', True)
        self.loutcdf = self.nml.get('MODEL_RUN', 'loutcdf', False)  # netCDF output
        self.loutvec = self.nml.get('MODEL_RUN', 'loutvec', False)  # Vector output

        # Output directory: output_base_dir/case_name/model_output/
        self.coutdir = os.path.join(output_base_dir, case_name, 'model_output')

        self.case_name = case_name
        self.couttag = self.nml.get('MODEL_RUN', 'couttag', '')
        self.ifrq_out = self.nml.get('MODEL_RUN', 'ifrq_out', 24)  # Output frequency (hours)

        # Output file organization frequency
        # 'single'  - Single file containing all time steps
        # 'yearly'  - One file per year
        # 'monthly' - One file per month
        # 'daily'   - One file per day
        self.output_file_freq = self.nml.get('MODEL_RUN', 'output_file_freq', 'single')

        # Output variables (comma-separated list)
        cvarsout = self.nml.get('MODEL_RUN', 'cvarsout',
                               'outflw,storge,rivout,rivsto,rivdph,rivvel')
        self.cvarsout = [v.strip() for v in cvarsout.split(',')]

        # NetCDF compression level
        self.ndlevel = self.nml.get('MODEL_RUN', 'ndlevel', 0)

        # Track current output file (for multi-file mode)
        self.current_file_period = None

    def initialize_output(self):
        """Initialize output files"""
        if not self.loutput:
            print("INFO: Output is disabled")
            return

        # Create output directory
        os.makedirs(self.coutdir, exist_ok=True)

        if self.loutcdf:
            self._initialize_netcdf_output()
        else:
            self._initialize_binary_output()

    def _get_output_filename(self, current_time=None):
        """
        Get output filename based on output_file_freq and time

        Parameters:
        -----------
        current_time : datetime, optional
            Current time (for multi-file mode). If None, use start_time.

        Returns:
        --------
        filename : str
            Full path to output file
        file_period : str
            Period identifier (for tracking when to create new files)
        """
        if current_time is None:
            current_time = self.time_control.start_time

        # Base filename
        if self.output_file_freq == 'single':
            # Single file for entire simulation
            if self.couttag:
                filename = f'{self.case_name}_{self.couttag}.nc'
            else:
                filename = f'{self.case_name}.nc'
            file_period = 'single'

        elif self.output_file_freq == 'yearly':
            # One file per year: casename_YYYY.nc
            filename = f'{self.case_name}_{current_time.year:04d}.nc'
            file_period = f'{current_time.year:04d}'

        elif self.output_file_freq == 'monthly':
            # One file per month: casename_YYYYMM.nc
            filename = f'{self.case_name}_{current_time.year:04d}{current_time.month:02d}.nc'
            file_period = f'{current_time.year:04d}{current_time.month:02d}'

        elif self.output_file_freq == 'daily':
            # One file per day: casename_YYYYMMDD.nc
            filename = f'{self.case_name}_{current_time.year:04d}{current_time.month:02d}{current_time.day:02d}.nc'
            file_period = f'{current_time.year:04d}{current_time.month:02d}{current_time.day:02d}'

        else:
            raise ValueError(f"Unknown output_file_freq: {self.output_file_freq}")

        return os.path.join(self.coutdir, filename), file_period

    def _initialize_netcdf_output(self):
        """Initialize netCDF output file (CF-compliant)"""
        # For single-file mode, create file at initialization
        if self.output_file_freq == 'single':
            filename, file_period = self._get_output_filename()
            self.current_file_period = file_period
            print(f"Creating CF-compliant netCDF output file: {filename}")
            self._create_netcdf_file(filename)
        else:
            # For multi-file mode, file will be created on first write
            print(f"Output mode: {self.output_file_freq}")
            print(f"Output files will be created in: {self.coutdir}")
            print(f"File naming: {self.case_name}_[date].nc")

    def _create_netcdf_file(self, filename):
        """Create a new netCDF output file"""
        # Create netCDF file
        self.nc_file = nc.Dataset(filename, 'w', format='NETCDF4')

        # Create dimensions
        self.nc_file.createDimension('lon', self.nx)
        self.nc_file.createDimension('lat', self.ny)
        self.nc_file.createDimension('time', None)  # Unlimited dimension

        # Create coordinate variables - Longitude
        lon_var = self.nc_file.createVariable('lon', 'f8', ('lon',))
        if self.lon is not None:
            lon_var[:] = self.lon
        else:
            lon_var[:] = np.arange(self.nx)
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.axis = 'X'

        # Create coordinate variables - Latitude
        lat_var = self.nc_file.createVariable('lat', 'f8', ('lat',))
        if self.lat is not None:
            lat_var[:] = self.lat
        else:
            lat_var[:] = np.arange(self.ny)
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.axis = 'Y'

        # Create time variable (CF-compliant)
        self.time_var = self.nc_file.createVariable('time', 'f8', ('time',))
        self.time_var.standard_name = 'time'
        self.time_var.long_name = 'time'
        self.time_var.units = f'hours since {self.time_control.start_time.strftime("%Y-%m-%d %H:%M:%S")}'
        self.time_var.calendar = 'standard'
        self.time_var.axis = 'T'

        # Create output variables with CF-compliant attributes
        var_attrs = {
            # Instantaneous variables (will be time-averaged if diagnostics are enabled)
            'outflw': {
                'long_name': 'discharge (river+floodplain)',
                'units': 'm3/s',
                'cell_methods': 'time: mean',  # Following CaMa-Flood Fortran convention
            },
            'storge': {
                'long_name': 'Total water storage in grid cell',
                'units': 'm3',
            },
            'rivout': {
                'long_name': 'River channel outflow',
                'units': 'm3/s',
                'cell_methods': 'time: mean',  # Following CaMa-Flood Fortran convention
            },
            'rivsto': {
                'long_name': 'River channel water storage',
                'units': 'm3',
            },
            'rivdph': {
                'long_name': 'River channel water depth',
                'units': 'm',
                'standard_name': 'water_surface_height_above_reference_datum',
            },
            'rivvel': {
                'long_name': 'River channel flow velocity',
                'units': 'm s-1',
            },
            'fldout': {
                'long_name': 'Floodplain outflow',
                'units': 'm3 s-1',
            },
            'fldsto': {
                'long_name': 'Floodplain water storage',
                'units': 'm3',
            },
            'flddph': {
                'long_name': 'Floodplain water depth',
                'units': 'm',
            },
            'fldfrc': {
                'long_name': 'Flooded area fraction',
                'units': '1',
                'standard_name': 'flood_area_fraction',
            },
            'fldare': {
                'long_name': 'Flooded area',
                'units': 'm2',
                'standard_name': 'flood_water_area',
            },
            'sfcelv': {
                'long_name': 'Water surface elevation',
                'units': 'm',
                'standard_name': 'water_surface_height_above_reference_datum',
            },

            # Time-averaged diagnostic variables
            'rivout_avg': {
                'long_name': 'Time-averaged river channel outflow',
                'units': 'm3 s-1',
                'standard_name': 'water_volume_transport_in_river_channel',
                'cell_methods': 'time: mean',
            },
            'fldout_avg': {
                'long_name': 'Time-averaged floodplain outflow',
                'units': 'm3 s-1',
                'cell_methods': 'time: mean',
            },
            'outflw_avg': {
                'long_name': 'Time-averaged total outflow',
                'units': 'm3 s-1',
                'standard_name': 'water_volume_transport_in_river_channel',
                'cell_methods': 'time: mean',
            },
            'rivvel_avg': {
                'long_name': 'Time-averaged river channel flow velocity',
                'units': 'm s-1',
                'cell_methods': 'time: mean',
            },
            'pthout_avg': {
                'long_name': 'Time-averaged bifurcation channel outflow',
                'units': 'm3 s-1',
                'cell_methods': 'time: mean',
            },
            'runoff_avg': {
                'long_name': 'Time-averaged runoff input',
                'units': 'm3 s-1',
                'standard_name': 'runoff_flux',
                'cell_methods': 'time: mean',
            },
            'daminf_avg': {
                'long_name': 'Time-averaged dam inflow',
                'units': 'm3 s-1',
                'cell_methods': 'time: mean',
            },

            # Time-maximum diagnostic variables
            'outflw_max': {
                'long_name': 'Maximum total outflow',
                'units': 'm3 s-1',
                'standard_name': 'water_volume_transport_in_river_channel',
                'cell_methods': 'time: maximum',
            },
            'rivdph_max': {
                'long_name': 'Maximum river channel water depth',
                'units': 'm',
                'standard_name': 'water_surface_height_above_reference_datum',
                'cell_methods': 'time: maximum',
            },
            'storge_max': {
                'long_name': 'Maximum total water storage',
                'units': 'm3',
                'cell_methods': 'time: maximum',
            },

            # Legacy maximum variable names (for compatibility)
            'maxsto': {
                'long_name': 'Maximum water storage',
                'units': 'm3',
                'cell_methods': 'time: maximum',
            },
            'maxflw': {
                'long_name': 'Maximum outflow',
                'units': 'm3 s-1',
                'standard_name': 'water_volume_transport_in_river_channel',
                'cell_methods': 'time: maximum',
            },
            'maxdph': {
                'long_name': 'Maximum water depth',
                'units': 'm',
                'standard_name': 'water_surface_height_above_reference_datum',
                'cell_methods': 'time: maximum',
            },
        }

        compression = {'zlib': True, 'complevel': self.ndlevel} if self.ndlevel > 0 else {}

        for varname in self.cvarsout:
            if varname not in var_attrs:
                print(f"WARNING: Unknown output variable: {varname}, skipping")
                continue

            # Create variable with CF-compliant dimensions (time, lat, lon)
            var = self.nc_file.createVariable(
                varname, 'f4', ('time', 'lat', 'lon'),
                fill_value=-9999.0,
                **compression
            )

            # Set attributes from var_attrs dictionary
            attrs = var_attrs[varname]
            var.long_name = attrs['long_name']
            var.units = attrs['units']

            # Add standard_name if defined
            if 'standard_name' in attrs:
                var.standard_name = attrs['standard_name']

            # Add cell_methods if defined
            if 'cell_methods' in attrs:
                var.cell_methods = attrs['cell_methods']

            # Add coordinates attribute
            var.coordinates = 'lon lat'

            self.nc_vars[varname] = var

        # Global attributes (CF-compliant)
        self.nc_file.Conventions = 'CF-1.8'
        self.nc_file.title = 'CaMa-Flood Model Output'
        self.nc_file.institution = 'Generated by pycamaflood'
        self.nc_file.source = 'CaMa-Flood (Catchment-based Macro-scale Floodplain model)'
        self.nc_file.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        self.nc_file.references = 'Yamazaki et al. (2011), doi:10.1029/2010WR009726'
        self.nc_file.comment = 'Output from CaMa-Flood hydrodynamic model'

        # Reset time index for new file
        self.time_index = 0

        print(f"Created netCDF output file with variables: {', '.join(self.cvarsout)}")

    def write_output(self, state, i2nextx, i2nexty, diagnostics=None):
        """
        Write output at current time step

        Parameters:
        -----------
        state : dict
            Model state dictionary
        i2nextx : ndarray
            Next cell x coordinates (for mapping vector to 2D)
        i2nexty : ndarray
            Next cell y coordinates
        diagnostics : dict, optional
            Diagnostic variables (time-averaged and maximum values)
        """
        if not self.loutput:
            return

        current_time = self.time_control.current_time

        # Check if it's output time
        if not self.time_control.is_output_time(self.ifrq_out):
            return

        print(f"Writing output at {current_time}")

        if self.loutcdf:
            # Check if we need to create a new file (multi-file mode)
            if self.output_file_freq != 'single':
                filename, file_period = self._get_output_filename(current_time)

                # Create new file if period changed or file doesn't exist
                if file_period != self.current_file_period:
                    # Close current file if open
                    if self.nc_file is not None:
                        print(f"Closing previous output file")
                        self.nc_file.close()
                        self.nc_file = None

                    # Create new file
                    print(f"Creating new output file: {filename}")
                    self._create_netcdf_file(filename)
                    self.current_file_period = file_period

            self._write_netcdf_output(state, i2nextx, i2nexty, diagnostics)
        else:
            self._write_binary_output(state, i2nextx, i2nexty, diagnostics)

    def _write_netcdf_output(self, state, i2nextx, i2nexty, diagnostics=None):
        """Write netCDF output"""
        # Calculate time in hours since start
        time_hours = (self.time_control.current_time - self.time_control.start_time).total_seconds() / 3600.0

        # Write time
        self.time_var[self.time_index] = time_hours

        # Combine state and diagnostics into single data dictionary
        data_dict = state.copy()
        if diagnostics is not None:
            data_dict.update(diagnostics)

            # Map averaged diagnostics to non-averaged names for CaMa-Flood compatibility
            # In CaMa-Flood Fortran, the output variables are time-averaged by default
            for avg_var in ['rivout_avg', 'fldout_avg', 'outflw_avg', 'rivvel_avg', 'pthout_avg']:
                base_var = avg_var.replace('_avg', '')
                if avg_var in diagnostics and base_var in self.cvarsout:
                    # Use time-averaged value instead of instantaneous for output
                    data_dict[base_var] = diagnostics[avg_var]

        # Write each variable
        for varname in self.cvarsout:
            if varname not in self.nc_vars:
                continue

            if varname not in data_dict:
                print(f"WARNING: Variable {varname} not in data, skipping")
                continue

            # Map vector data to 2D grid
            data_2d = self._map_vector_to_2d(data_dict[varname], i2nextx, i2nexty)

            # Write data
            self.nc_vars[varname][self.time_index, :, :] = data_2d

        # Sync to disk
        self.nc_file.sync()

        self.time_index += 1

    def _initialize_binary_output(self):
        """
        Initialize binary output files (Fortran direct access format)

        Each variable is written to a separate binary file
        Format: Fortran direct access, unformatted
        Each record contains one 2D slice (nx, ny) at one time step
        """
        print("Initializing binary output...")

        # Create one file per variable
        for varname in self.cvarsout:
            filename = os.path.join(self.coutdir, f'{varname}{self.couttag}.bin')

            # Open file in binary write mode
            # We'll write Fortran unformatted records using FortranFile
            self.bin_files[varname] = {'filename': filename, 'written_count': 0}

        print(f"Binary output initialized for {len(self.cvarsout)} variables")
        self.bin_record_num = 0

    def _write_binary_output(self, state, i2nextx, i2nexty, diagnostics=None):
        """
        Write binary output in Fortran direct access format

        Parameters:
        -----------
        state : dict
            Model state dictionary
        i2nextx : ndarray
            Next cell x coordinates
        i2nexty : ndarray
            Next cell y coordinates
        diagnostics : dict, optional
            Diagnostic variables
        """
        # Combine state and diagnostics
        data_dict = state.copy()
        if diagnostics is not None:
            data_dict.update(diagnostics)

        # Write each variable to its separate file
        for varname in self.cvarsout:
            if varname not in data_dict:
                print(f"WARNING: Variable {varname} not in data, skipping")
                continue

            # Map vector data to 2D grid
            data_2d = self._map_vector_to_2d(data_dict[varname], i2nextx, i2nexty)

            # Get filename
            filename = self.bin_files[varname]['filename']

            # Write Fortran unformatted record
            self._write_fortran_record(filename, data_2d)

            self.bin_files[varname]['written_count'] += 1

        self.bin_record_num += 1

    def _write_fortran_record(self, filename, data_2d):
        """
        Write a single Fortran unformatted record

        Fortran unformatted format:
        - Record header: 4-byte integer (record length in bytes)
        - Data: record_length bytes
        - Record trailer: 4-byte integer (same as header)

        Parameters:
        -----------
        filename : str
            Output file name
        data_2d : ndarray
            2D array to write (ny, nx)
        """
        # Convert to Fortran order (column-major) and float32
        data_fortran = np.asfortranarray(data_2d, dtype=np.float32)

        # Append to file using scipy FortranFile
        try:
            with FortranFile(filename, 'a') as f:
                f.write_record(data_fortran)
        except Exception as e:
            print(f"ERROR writing Fortran record to {filename}: {e}")
            # Fallback: manual write
            self._write_fortran_record_manual(filename, data_fortran)

    def _write_fortran_record_manual(self, filename, data):
        """
        Manually write Fortran unformatted record
        (Fallback if FortranFile fails)

        Parameters:
        -----------
        filename : str
            Output file name
        data : ndarray
            Data array in Fortran order
        """
        # Calculate record length
        record_length = data.nbytes

        # Open file in append binary mode
        with open(filename, 'ab') as f:
            # Write record header (4-byte integer, little-endian)
            f.write(struct.pack('<i', record_length))

            # Write data
            f.write(data.tobytes(order='F'))

            # Write record trailer
            f.write(struct.pack('<i', record_length))

    def _map_vector_to_2d(self, vector_data, i2nextx, i2nexty):
        """
        Map vector data to 2D grid

        Parameters:
        -----------
        vector_data : ndarray
            Vector data (nseqall,)
        i2nextx : ndarray
            X coordinates for each cell
        i2nexty : ndarray
            Y coordinates for each cell

        Returns:
        --------
        data_2d : np.ma.MaskedArray
            2D grid data (ny, nx) with non-river cells masked
        """
        # Create data array filled with fill value
        data_2d = np.full((self.ny, self.nx), -9999.0, dtype=np.float32)
        # Create mask array: True=masked (invalid), False=unmasked (valid)
        mask_2d = np.ones((self.ny, self.nx), dtype=bool)  # Start with all masked

        # Check if i2nextx/i2nexty are 2D arrays or 1D sequence arrays
        if i2nextx.ndim == 2:
            # i2nextx and i2nexty are 2D arrays (nx, ny)
            # We need to iterate through the 2D grid and map sequence data
            # This assumes vector_data has length nseqall and we need to determine
            # which grid cells are active

            # For now, use a simple mapping where we assume the first nseqall
            # elements correspond to active cells in row-major order
            # This is a simplified approach - ideally we should have seq_x, seq_y arrays
            iseq = 0
            for iy in range(self.ny):
                for ix in range(self.nx):
                    if iseq < len(vector_data):
                        # Check if this is an active river cell
                        # (i2nextx >= 0 indicates active cell in some conventions)
                        if i2nextx[ix, iy] >= -9998:  # Not missing value
                            data_2d[iy, ix] = vector_data[iseq]
                            mask_2d[iy, ix] = False  # Unmask valid cell
                            iseq += 1
        else:
            # i2nextx and i2nexty are 1D sequence coordinate arrays
            for iseq in range(min(len(vector_data), len(i2nextx))):
                ix = int(i2nextx[iseq])
                iy = int(i2nexty[iseq])
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    data_2d[iy, ix] = vector_data[iseq]
                    mask_2d[iy, ix] = False  # Unmask valid cell

        # Return masked array
        return np.ma.masked_array(data_2d, mask=mask_2d, fill_value=-9999.0)

    def finalize_output(self):
        """Finalize and close output files"""
        if self.nc_file is not None:
            print("Closing netCDF output file")
            self.nc_file.close()
            self.nc_file = None

    def write_restart(self, state, restart_dir='./'):
        """
        Write restart file

        Parameters:
        -----------
        state : dict
            Model state dictionary
        restart_dir : str
            Restart output directory
        """
        current_time = self.time_control.current_time
        date_str = current_time.strftime('%Y%m%d%H%M')

        # Restart file name
        filename = os.path.join(restart_dir, f'restart{date_str}.nc')

        print(f"Writing restart file: {filename}")

        # Create netCDF restart file
        with nc.Dataset(filename, 'w', format='NETCDF4') as f:
            # Create dimensions
            f.createDimension('seq', self.nseqall)

            # Write state variables
            for varname, data in state.items():
                var = f.createVariable(varname, 'f8', ('seq',))
                var[:] = data

            # Global attributes
            f.title = 'CaMa-Flood restart file'
            f.restart_time = date_str
            f.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        print(f"Restart file written successfully")
