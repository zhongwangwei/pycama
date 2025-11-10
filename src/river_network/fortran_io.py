"""
Fortran binary file I/O utilities
Handles direct access binary files with Fortran indexing (1-based)
"""
import numpy as np
import struct


class FortranBinary:
    """Handler for Fortran direct access binary files"""

    @staticmethod
    def read_direct(filename, shape, dtype, rec=1):
        """
        Read Fortran direct access binary file

        Args:
            filename: Path to binary file
            shape: Shape of array (ny, nx) - NumPy standard, will be accessed as [iy, ix]
            dtype: Data type ('int4', 'real', 'int2', 'int1')
            rec: Record number (1-based, Fortran style)

        Returns:
            numpy array with shape (ny, nx), access as [iy, ix] (0-based)
            Fortran array(ix+1, iy+1) maps to Python array[iy, ix]
        """
        dtype_map = {
            'int4': ('i4', 4),
            'int': ('i4', 4),
            'real': ('f4', 4),
            'float': ('f4', 4),
            'int2': ('i2', 2),
            'int1': ('i1', 1)
        }

        np_dtype, bytes_per_elem = dtype_map[dtype.lower()]
        total_elements = np.prod(shape)
        recl = total_elements * bytes_per_elem

        offset = (rec - 1) * recl  # rec is 1-based

        with open(filename, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=np_dtype, count=total_elements)

        # Reshape to Fortran order
        data = data.reshape(shape, order='F')

        return data

    @staticmethod
    def write_direct(filename, data, rec=1):
        """
        Write Fortran direct access binary file

        Args:
            filename: Path to binary file
            data: numpy array to write
            rec: Record number (1-based, Fortran style)
        """
        # Flatten in Fortran order (column-major)
        flat_data = data.flatten('F')

        recl = flat_data.nbytes
        offset = (rec - 1) * recl

        # Create or open file in binary mode
        mode = 'r+b' if rec > 1 else 'wb'
        try:
            with open(filename, mode) as f:
                f.seek(offset)
                flat_data.tofile(f)
        except FileNotFoundError:
            with open(filename, 'wb') as f:
                f.seek(offset)
                flat_data.tofile(f)

    @staticmethod
    def read_multiple_records(filename, shape, dtype, num_records):
        """Read multiple records from file"""
        records = []
        for rec in range(1, num_records + 1):
            records.append(FortranBinary.read_direct(filename, shape, dtype, rec))
        return records


def read_params_txt(filename):
    """
    Read params.txt file

    Returns:
        dict with keys: nx, ny, nflp, gsize, west, east, south, north
    """
    values = []
    with open(filename, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].split('!')[0].strip()
            if line:
                # Extract the first value (before any additional text)
                parts = line.split()
                if parts:
                    values.append(parts[0])

    params = {
        'nx': int(values[0]),
        'ny': int(values[1]),
        'nflp': int(values[2]),
        'gsize': float(values[3]),
        'west': float(values[4]),
        'east': float(values[5]),
        'south': float(values[6]),
        'north': float(values[7])
    }

    return params


def write_params_txt(filename, params):
    """Write params.txt file in standard CaMa-Flood format"""
    with open(filename, 'w') as f:
        f.write(f"{params['nx']:12d}      !! grid number (east-west)\n")
        f.write(f"{params['ny']:12d}      !! grid number (north-south)\n")
        f.write(f"{params['nflp']:12d}     !! floodplain layer\n")
        f.write(f"{params['gsize']:12.8f}     !! grid size\n")
        f.write(f"{params['west']:12.3f}     !! west  edge (deg)\n")
        f.write(f"{params['east']:12.3f}     !! east  edge (deg)\n")
        f.write(f"{params['south']:12.3f}     !! south edge (deg)\n")
        f.write(f"{params['north']:12.3f}     !! north edge (deg)\n")
