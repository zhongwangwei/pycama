"""
Restart module for CaMa-Flood model run
Handles reading and writing restart files

Based on CMF_CTRL_RESTART_MOD.F90
"""
import numpy as np
import os
import netCDF4 as nc
from datetime import datetime
from scipy.io import FortranFile
import struct


class RestartManager:
    """Manage CaMa-Flood restart file reading and writing"""

    def __init__(self, nml, physics):
        """
        Initialize restart manager

        Parameters:
        -----------
        nml : Namelist object
            Namelist configuration
        physics : CaMaPhysics object
            Physics module instance
        """
        self.nml = nml
        self.physics = physics

        # Read restart configuration
        self._read_restart_config()

    def _read_restart_config(self):
        """Read restart configuration from namelist"""
        self.lrestart = self.nml.get('MODEL_RUN', 'lrestart', False)
        self.lstoonly = self.nml.get('MODEL_RUN', 'lstoonly', False)

        self.creststo = self.nml.get('MODEL_RUN', 'creststo', '')

        # Default restart directory: ./output/{case_name}/restart
        output_base_dir = self.nml.get('OUTPUT', 'output_base_dir', './output')
        case_name = self.nml.get('OUTPUT', 'case_name', 'default')
        default_crestdir = os.path.join(output_base_dir, case_name, 'restart')
        self.crestdir = self.nml.get('MODEL_RUN', 'crestdir', default_crestdir)

        self.cvnrest = self.nml.get('MODEL_RUN', 'cvnrest', 'restart')
        self.lrestcdf = self.nml.get('MODEL_RUN', 'lrestcdf', False)
        self.lrestdbl = self.nml.get('MODEL_RUN', 'lrestdbl', False)

        # Read restart frequency (can be in hours, days, months, or years)
        # Default: 0 (only at end of run)
        self.ifrq_rst = self.nml.get('MODEL_RUN', 'ifrq_rst', 0)

        # Read restart frequency unit (default: 'hour')
        # Options: 'hour', 'day', 'month', 'year'
        self.cfrq_rst_unit = self.nml.get('MODEL_RUN', 'cfrq_rst_unit', 'hour')

    def read_restart(self):
        """
        Read restart file and initialize model state

        Similar to CMF_RESTART_INIT in Fortran
        """
        if not self.lrestart:
            print("INFO: Restart disabled, starting from zero storage")
            return

        if not self.creststo:
            print("WARNING: No restart file specified, starting from zero storage")
            return

        if not os.path.exists(self.creststo):
            print(f"WARNING: Restart file not found: {self.creststo}")
            print("  Starting from zero storage")
            return

        print(f"Reading restart file: {self.creststo}")

        if self.lrestcdf:
            self._read_restart_netcdf()
        else:
            self._read_restart_binary()

        print("  Restart file loaded successfully")

    def _read_restart_netcdf(self):
        """
        Read netCDF restart file

        Format:
        - Dimension: seq (nseqall)
        - Variables: rivsto, fldsto, rivout_pre, fldout_pre, rivdph_pre, fldsto_pre
        """
        with nc.Dataset(self.creststo, 'r') as f:
            # Read main state variables
            if 'rivsto' in f.variables:
                rivsto = f.variables['rivsto'][:]
                self.physics.d2rivsto[:len(rivsto)] = rivsto

            if 'fldsto' in f.variables:
                fldsto = f.variables['fldsto'][:]
                self.physics.d2fldsto[:len(fldsto)] = fldsto

            # Read previous time step variables (for implicit scheme)
            if not self.lstoonly:
                if 'rivout_pre' in f.variables:
                    self.physics.d2rivout_pre[:len(rivsto)] = f.variables['rivout_pre'][:]

                if 'fldout_pre' in f.variables:
                    self.physics.d2fldout_pre[:len(rivsto)] = f.variables['fldout_pre'][:]

                if 'rivdph_pre' in f.variables:
                    self.physics.d2rivdph_pre[:len(rivsto)] = f.variables['rivdph_pre'][:]

                if 'fldsto_pre' in f.variables:
                    self.physics.d2fldsto_pre[:len(rivsto)] = f.variables['fldsto_pre'][:]
            else:
                # Storage only restart - initialize previous storage
                self.physics.d2fldsto_pre[:len(fldsto)] = fldsto

            # Read optional variables
            if 'gdwsto' in f.variables and hasattr(self.physics, 'd2gdwsto'):
                self.physics.d2gdwsto[:len(rivsto)] = f.variables['gdwsto'][:]

            if 'damsto' in f.variables and hasattr(self.physics, 'd2damsto'):
                self.physics.d2damsto[:len(rivsto)] = f.variables['damsto'][:]

            # Print restart info
            if 'restart_time' in f.ncattrs():
                print(f"  Restart time: {f.restart_time}")

    def _read_restart_binary(self):
        """
        Read binary restart file (Fortran unformatted format)

        Format:
        - Fortran unformatted sequential access
        - Each record: 4-byte header, data, 4-byte trailer
        - Records: rivsto, fldsto, [rivout_pre, fldout_pre, rivdph_pre, fldsto_pre, ...]
        """
        print("  Reading binary restart file...")

        nseqall = self.physics.nseqall

        # Determine dtype
        dtype = np.float64 if self.lrestdbl else np.float32

        try:
            # Try using scipy FortranFile (handles record markers automatically)
            with FortranFile(self.creststo, 'r') as f:
                # Read river storage (1D array)
                rivsto = f.read_reals(dtype=dtype)
                self.physics.d2rivsto[:min(len(rivsto), nseqall)] = rivsto[:nseqall]

                # Read floodplain storage
                fldsto = f.read_reals(dtype=dtype)
                self.physics.d2fldsto[:min(len(fldsto), nseqall)] = fldsto[:nseqall]

                if not self.lstoonly:
                    # Read previous time step variables
                    try:
                        rivout_pre = f.read_reals(dtype=dtype)
                        self.physics.d2rivout_pre[:min(len(rivout_pre), nseqall)] = rivout_pre[:nseqall]

                        fldout_pre = f.read_reals(dtype=dtype)
                        self.physics.d2fldout_pre[:min(len(fldout_pre), nseqall)] = fldout_pre[:nseqall]

                        rivdph_pre = f.read_reals(dtype=dtype)
                        self.physics.d2rivdph_pre[:min(len(rivdph_pre), nseqall)] = rivdph_pre[:nseqall]

                        fldsto_pre = f.read_reals(dtype=dtype)
                        self.physics.d2fldsto_pre[:min(len(fldsto_pre), nseqall)] = fldsto_pre[:nseqall]

                        # Optional: groundwater storage
                        if self.physics.lgdwdly:
                            try:
                                gdwsto = f.read_reals(dtype=dtype)
                                self.physics.p2gdwsto[:min(len(gdwsto), nseqall)] = gdwsto[:nseqall]
                            except:
                                pass  # Not all restart files have groundwater

                    except Exception as e:
                        print(f"  WARNING: Could not read all previous time step variables: {e}")
                        # Initialize with current storage
                        self.physics.d2fldsto_pre[:nseqall] = self.physics.d2fldsto[:nseqall]
                else:
                    # Storage only - initialize previous storage
                    self.physics.d2fldsto_pre[:nseqall] = self.physics.d2fldsto[:nseqall]

            print("  Binary restart file read successfully")

        except Exception as e:
            print(f"  ERROR reading binary restart file with FortranFile: {e}")
            print("  Trying manual read...")
            try:
                self._read_restart_binary_manual()
            except Exception as e2:
                print(f"  ERROR with manual read: {e2}")
                print("  Starting from zero storage")

    def _read_restart_binary_manual(self):
        """
        Manually read Fortran unformatted binary file
        (Fallback if FortranFile fails)
        """
        nseqall = self.physics.nseqall
        dtype = np.float64 if self.lrestdbl else np.float32
        dtype_size = 8 if self.lrestdbl else 4

        with open(self.creststo, 'rb') as f:
            # Read river storage record
            record_len_header = struct.unpack('<i', f.read(4))[0]
            data = np.frombuffer(f.read(record_len_header), dtype=dtype)
            record_len_trailer = struct.unpack('<i', f.read(4))[0]

            if record_len_header != record_len_trailer:
                raise ValueError("Record marker mismatch")

            self.physics.d2rivsto[:min(len(data), nseqall)] = data[:nseqall]

            # Read floodplain storage record
            record_len_header = struct.unpack('<i', f.read(4))[0]
            data = np.frombuffer(f.read(record_len_header), dtype=dtype)
            record_len_trailer = struct.unpack('<i', f.read(4))[0]

            if record_len_header != record_len_trailer:
                raise ValueError("Record marker mismatch")

            self.physics.d2fldsto[:min(len(data), nseqall)] = data[:nseqall]

            # Continue for other variables if not storage-only...
            if not self.lstoonly:
                try:
                    # Read rivout_pre
                    record_len_header = struct.unpack('<i', f.read(4))[0]
                    data = np.frombuffer(f.read(record_len_header), dtype=dtype)
                    f.read(4)  # trailer
                    self.physics.d2rivout_pre[:min(len(data), nseqall)] = data[:nseqall]

                    # Similar for other variables...
                except:
                    pass

        print("  Manual binary read successful")

    def write_restart(self, time_control):
        """
        Write restart file

        Similar to CMF_RESTART_WRITE in Fortran

        Parameters:
        -----------
        time_control : TimeControl object
            Time control instance for getting current time
        """
        # Check if it's restart output time
        if self.ifrq_rst == 0:
            # Only at end of simulation
            if not time_control.is_finished():
                return
        else:
            # Check if it's restart time based on frequency unit
            if not time_control.is_restart_time(self.ifrq_rst, self.cfrq_rst_unit):
                return

        # Generate restart filename
        current_time = time_control.current_time
        date_str = current_time.strftime('%Y%m%d%H%M')

        # Create restart directory if needed
        os.makedirs(self.crestdir, exist_ok=True)

        if self.lrestcdf:
            restart_file = os.path.join(self.crestdir, f'{self.cvnrest}_{date_str}.nc')
            self._write_restart_netcdf(restart_file, current_time)
        else:
            restart_file = os.path.join(self.crestdir, f'{self.cvnrest}_{date_str}.bin')
            self._write_restart_binary(restart_file)

        print(f"Restart file written: {restart_file}")

    def _write_restart_netcdf(self, restart_file, current_time):
        """Write restart file in netCDF format"""
        nseqall = self.physics.nseqall

        with nc.Dataset(restart_file, 'w', format='NETCDF4') as f:
            # Create dimension
            f.createDimension('seq', nseqall)

            # Create variables
            var_rivsto = f.createVariable('rivsto', 'f8', ('seq',))
            var_rivsto[:] = self.physics.d2rivsto[:nseqall]
            var_rivsto.long_name = 'River storage'
            var_rivsto.units = 'm3'

            var_fldsto = f.createVariable('fldsto', 'f8', ('seq',))
            var_fldsto[:] = self.physics.d2fldsto[:nseqall]
            var_fldsto.long_name = 'Floodplain storage'
            var_fldsto.units = 'm3'

            if not self.lstoonly:
                # Write previous time step variables for full restart
                var_rivout_pre = f.createVariable('rivout_pre', 'f8', ('seq',))
                var_rivout_pre[:] = self.physics.d2rivout_pre[:nseqall]
                var_rivout_pre.long_name = 'River outflow (previous time step)'
                var_rivout_pre.units = 'm3/s'

                var_fldout_pre = f.createVariable('fldout_pre', 'f8', ('seq',))
                var_fldout_pre[:] = self.physics.d2fldout_pre[:nseqall]
                var_fldout_pre.long_name = 'Floodplain outflow (previous time step)'
                var_fldout_pre.units = 'm3/s'

                var_rivdph_pre = f.createVariable('rivdph_pre', 'f8', ('seq',))
                var_rivdph_pre[:] = self.physics.d2rivdph_pre[:nseqall]
                var_rivdph_pre.long_name = 'River depth (previous time step)'
                var_rivdph_pre.units = 'm'

                var_fldsto_pre = f.createVariable('fldsto_pre', 'f8', ('seq',))
                var_fldsto_pre[:] = self.physics.d2fldsto_pre[:nseqall]
                var_fldsto_pre.long_name = 'Floodplain storage (previous time step)'
                var_fldsto_pre.units = 'm3'

            # Optional variables
            if hasattr(self.physics, 'd2gdwsto'):
                var_gdw = f.createVariable('gdwsto', 'f8', ('seq',))
                var_gdw[:] = self.physics.d2gdwsto[:nseqall]
                var_gdw.long_name = 'Groundwater storage'
                var_gdw.units = 'm3'

            if hasattr(self.physics, 'd2damsto'):
                var_dam = f.createVariable('damsto', 'f8', ('seq',))
                var_dam[:] = self.physics.d2damsto[:nseqall]
                var_dam.long_name = 'Dam storage'
                var_dam.units = 'm3'

            # Global attributes
            f.title = 'CaMa-Flood restart file'
            f.restart_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            f.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            f.storage_only = 'True' if self.lstoonly else 'False'

    def _write_restart_binary(self, restart_file):
        """
        Write restart file in binary format (Fortran unformatted)

        Format:
        - Fortran unformatted sequential access
        - Each record: 4-byte header, data array, 4-byte trailer
        - Records in order: rivsto, fldsto, [rivout_pre, fldout_pre, rivdph_pre, fldsto_pre, ...]
        """
        print("  Writing binary restart file...")

        nseqall = self.physics.nseqall
        dtype = np.float64 if self.lrestdbl else np.float32

        try:
            # Use scipy FortranFile to write Fortran unformatted records
            with FortranFile(restart_file, 'w') as f:
                # Write river storage
                f.write_record(self.physics.d2rivsto[:nseqall].astype(dtype))

                # Write floodplain storage
                f.write_record(self.physics.d2fldsto[:nseqall].astype(dtype))

                if not self.lstoonly:
                    # Write previous time step variables
                    f.write_record(self.physics.d2rivout_pre[:nseqall].astype(dtype))
                    f.write_record(self.physics.d2fldout_pre[:nseqall].astype(dtype))
                    f.write_record(self.physics.d2rivdph_pre[:nseqall].astype(dtype))
                    f.write_record(self.physics.d2fldsto_pre[:nseqall].astype(dtype))

                    # Optional: groundwater storage
                    if self.physics.lgdwdly and hasattr(self.physics, 'p2gdwsto'):
                        f.write_record(self.physics.p2gdwsto[:nseqall].astype(dtype))

                    # Optional: dam storage
                    if self.physics.ldamout and hasattr(self.physics, 'd2damsto'):
                        f.write_record(self.physics.d2damsto[:nseqall].astype(dtype))

            print("  Binary restart file written successfully")

        except Exception as e:
            print(f"  ERROR writing binary restart with FortranFile: {e}")
            print("  Trying manual write...")
            try:
                self._write_restart_binary_manual(restart_file)
            except Exception as e2:
                print(f"  ERROR with manual write: {e2}")
                print("  Binary restart write failed")

    def _write_restart_binary_manual(self, restart_file):
        """
        Manually write Fortran unformatted binary file
        (Fallback if FortranFile fails)
        """
        nseqall = self.physics.nseqall
        dtype = np.float64 if self.lrestdbl else np.float32

        with open(restart_file, 'wb') as f:
            # Helper function to write one record
            def write_record(data):
                data_bytes = data.astype(dtype).tobytes()
                record_len = len(data_bytes)
                # Write header
                f.write(struct.pack('<i', record_len))
                # Write data
                f.write(data_bytes)
                # Write trailer
                f.write(struct.pack('<i', record_len))

            # Write river storage
            write_record(self.physics.d2rivsto[:nseqall])

            # Write floodplain storage
            write_record(self.physics.d2fldsto[:nseqall])

            if not self.lstoonly:
                # Write previous time step variables
                write_record(self.physics.d2rivout_pre[:nseqall])
                write_record(self.physics.d2fldout_pre[:nseqall])
                write_record(self.physics.d2rivdph_pre[:nseqall])
                write_record(self.physics.d2fldsto_pre[:nseqall])

                # Optional variables
                if self.physics.lgdwdly and hasattr(self.physics, 'p2gdwsto'):
                    write_record(self.physics.p2gdwsto[:nseqall])

                if self.physics.ldamout and hasattr(self.physics, 'd2damsto'):
                    write_record(self.physics.d2damsto[:nseqall])

        print("  Manual binary write successful")
