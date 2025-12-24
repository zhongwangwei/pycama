#!/usr/bin/env python3
"""
CaMa-Flood Data Export to NetCDF (Python version)
Converts Fortran code to Python while maintaining 1-based indexing in output
"""

import sys
import os
import numpy as np
from netCDF4 import Dataset
import struct

# Constants
IMIS = -9999
RMIS = 1.E20
DMIS = 1.E20
PMANRIV = 0.03  # Default Manning coeff for river
PMANFLD = 0.10  # Default Manning coeff for floodplain


class CoLMGridRoutingInit:
    """Main class for CaMa-Flood data export"""

    def __init__(self):
        # Domain configuration
        self.NX = 0
        self.NY = 0
        self.NLFP = 0
        self.INPN = 0
        self.WEST = 0.0
        self.EAST = 0.0
        self.NORTH = 0.0
        self.SOUTH = 0.0

        # River network arrays
        self.I2NEXTX = None
        self.I2NEXTY = None
        self.I2REGION = None
        self.I1SEQX = None
        self.I1SEQY = None
        self.I1NEXT = None
        self.I2VECTOR = None
        self.I1UPST = None
        self.I1UPN = None
        self.NSEQRIV = 0
        self.NSEQALL = 0
        self.NSEQMAX = 0
        self.REGIONALL = 1
        self.REGIONTHIS = 1
        self.D1LON = None
        self.D1LAT = None

        # Topography arrays
        self.D2GRAREA = None
        self.D2ELEVTN = None
        self.D2NXTDST = None
        self.D2RIVLEN = None
        self.D2RIVWTH = None
        self.D2RIVHGT = None
        self.D2RIVMAN = None
        self.D2RIVELV = None
        self.D2RIVSTOMAX = None
        self.D2FLDHGT = None
        self.D2FLDSTOMAX = None
        self.D2FLDGRD = None
        self.D2DWNELV = None
        self.D2UPAREA = None
        self.I2BASIN = None
        self.I2OUTCLM = None

        # Bifurcation arrays
        self.NPTHOUT = 0
        self.NPTHLEV = 0
        self.PTH_UPST = None
        self.PTH_DOWN = None
        self.PTH_DST = None
        self.PTH_ELV = None
        self.PTH_WTH = None
        self.PTH_MAN = None

        # Input matrix arrays
        self.INPX = None
        self.INPY = None
        self.INPA = None

        # Dam parameter arrays
        self.dam_NDAMS = 0
        self.dam_GRAND_ID = None
        self.dam_DamName = None
        self.dam_DamLat = None
        self.dam_DamLon = None
        self.dam_area_CaMa = None
        self.dam_DamIX = None
        self.dam_DamIY = None
        self.dam_FldVol_mcm = None
        self.dam_ConVol_mcm = None
        self.dam_TotalVol_mcm = None
        self.dam_Qn = None
        self.dam_Qf = None
        self.dam_year = None
        self.dam_seq = None

        # File paths
        self.INPUT_DIR = ""
        self.PARAM_FILE = ""
        self.OUTPUT_DIR = ""
        self.INPMAT_FILE = ""
        self.OUTFILE = ""

        # Sediment files
        self.SED_DIR = ""
        self.SED_FRC = ""
        self.SED_SLOPE = ""
        
        # Sediment data arrays
        self.sed_frc = None
        self.sed_slope = None

    def parse_arguments(self):
        """Parse command line arguments"""
        print("\n*** Step 0: Parsing Command Line Arguments ***")

        if len(sys.argv) != 4:
            print("\nERROR: Incorrect number of arguments!")
            print("\nUsage: python grid_routing_init.py <input_dir> <param_file> <output_dir>")
            print("\nExample:")
            print("  python grid_routing_init.py /Users/zhongwangwei/Desktop/glb_15min_natural diminfo_15min.txt ./")
            sys.exit(1)

        self.INPUT_DIR = sys.argv[1]
        self.PARAM_FILE = sys.argv[2]
        self.OUTPUT_DIR = sys.argv[3]

        print(f"Input directory: {self.INPUT_DIR}")
        print(f"Parameter file:  {self.PARAM_FILE}")
        print(f"Output directory: {self.OUTPUT_DIR}")

        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def read_param_file(self):
        """Read parameter file"""
        print("\n*** Step 0.5: Reading Parameter File ***")

        param_path = os.path.join(self.INPUT_DIR, self.PARAM_FILE)
        print(f"Reading parameter file: {param_path}")

        def parse_line(line):
            """Parse a line, removing comments after !!"""
            if '!!' in line:
                line = line.split('!!')[0]
            return line.strip()

        with open(param_path, 'r') as f:
            # Line 1: NX
            self.NX = int(parse_line(f.readline()))
            print(f"NX = {self.NX}")

            # Line 2: NY
            self.NY = int(parse_line(f.readline()))
            print(f"NY = {self.NY}")

            # Line 3: NLFP
            self.NLFP = int(parse_line(f.readline()))
            print(f"NLFP = {self.NLFP}")

            # Line 4-5: skip (input nXX, nYY)
            f.readline()
            f.readline()

            # Line 6: INPN
            self.INPN = int(parse_line(f.readline()))
            print(f"INPN = {self.INPN}")

            # Line 7: Input matrix file name
            self.INPMAT_FILE = parse_line(f.readline())
            print(f"Input matrix file: {self.INPMAT_FILE}")

            # Line 8-11: WEST, EAST, NORTH, SOUTH
            self.WEST = float(parse_line(f.readline()))
            print(f"WEST = {self.WEST}")

            self.EAST = float(parse_line(f.readline()))
            print(f"EAST = {self.EAST}")

            self.NORTH = float(parse_line(f.readline()))
            print(f"NORTH = {self.NORTH}")

            self.SOUTH = float(parse_line(f.readline()))
            print(f"SOUTH = {self.SOUTH}")

        # Construct output file path
        self.OUTFILE = os.path.join(self.OUTPUT_DIR, 'grid_routing_data.nc')

        print("File paths configured successfully")

    def read_binary_2d(self, filepath, nx, ny, dtype='f4'):
        """Read 2D binary data from Fortran unformatted file"""
        # Fortran unformatted files have record markers
        # For direct access with RECL=4*NX*NY, no record markers
        with open(filepath, 'rb') as f:
            # Read the entire file
            data = np.fromfile(f, dtype=np.float32)

        # Reshape: Fortran is column-major, so we need to reshape and transpose
        n_records = len(data) // (nx * ny)
        if n_records > 0:
            data = data.reshape((n_records, ny, nx), order='F')
            # Transpose to get (record, nx, ny) in Python convention
            data = np.transpose(data, (0, 2, 1))

        return data

    def read_binary_single_record(self, filepath, nx, ny, rec=1, dtype='f4'):
        """Read a single record from binary file

        Args:
            filepath: path to binary file
            nx, ny: dimensions (Fortran array dimensions)
            rec: record number (1-based)
            dtype: 'f4' for float32, 'i4' for int32

        Note: Fortran arrays declared as (NX, NY) are stored in column-major order.
        In memory, elements are ordered as: (1,1), (2,1), ..., (NX,1), (1,2), ...
        """
        # Calculate byte offset for the record (1-based indexing in Fortran)
        recl = 4 * nx * ny  # 4 bytes per value
        offset = (rec - 1) * recl

        with open(filepath, 'rb') as f:
            f.seek(offset)
            # Read data with specified type
            if dtype == 'i4':
                data = np.fromfile(f, dtype=np.int32, count=nx*ny)
            else:
                data = np.fromfile(f, dtype=np.float32, count=nx*ny)

            # Fortran array (NX, NY) in column-major order
            # Reshape to (NX, NY) with Fortran order
            data = data.reshape((nx, ny), order='F')

        return data

    def cmf_rivmap_init(self):
        """Initialize river network"""
        print("\n*** Step 1: Initializing River Network ***")
        print("CMF::RIVMAP_INIT: river network initialization")

        # Allocate arrays (using 1-based indexing convention where needed)
        # Note: Python uses 0-based indexing, but we store NX+1, NY+1 to use 1-based
        self.I2NEXTX = np.zeros((self.NX, self.NY), dtype=np.int32)
        self.I2NEXTY = np.zeros((self.NX, self.NY), dtype=np.int32)
        self.I2REGION = np.zeros((self.NX, self.NY), dtype=np.int32)
        self.D1LON = np.zeros(self.NX, dtype=np.float64)
        self.D1LAT = np.zeros(self.NY, dtype=np.float64)

        # Read river network map
        cnextxy = os.path.join(self.INPUT_DIR, 'nextxy.bin')
        print(f"Reading nextxy binary: {cnextxy}")

        self.I2NEXTX = self.read_binary_single_record(cnextxy, self.NX, self.NY, rec=1, dtype='i4')
        self.I2NEXTY = self.read_binary_single_record(cnextxy, self.NX, self.NY, rec=2, dtype='i4')

        # Calculate lat, lon
        for ix in range(self.NX):
            # Fortran: D1LON(IX) = WEST + (IX - 0.5) * (EAST - WEST) / NX
            # Python: ix is 0-based, so ix+1 corresponds to Fortran IX
            self.D1LON[ix] = self.WEST + (ix + 0.5) * (self.EAST - self.WEST) / self.NX

        for iy in range(self.NY):
            # Fortran: D1LAT(IY) = NORTH - (IY - 0.5) * (NORTH - SOUTH) / NY
            self.D1LAT[iy] = self.NORTH - (iy + 0.5) * (self.NORTH - self.SOUTH) / self.NY

        # Calculate region
        print("Calculating regions...")
        self.I2REGION[:, :] = IMIS
        for iy in range(self.NY):
            for ix in range(self.NX):
                if self.I2NEXTX[ix, iy] != IMIS:
                    self.I2REGION[ix, iy] = 1

        self.NSEQMAX = np.sum(self.I2REGION > 0)
        print(f"NSEQMAX = {self.NSEQMAX}")

        # Convert 2D map to 1D sequence
        print("Converting 2D map to 1D sequence...")

        # Using 1-based indexing for sequence arrays (store as 1-based values)
        self.I1SEQX = np.zeros(self.NSEQMAX, dtype=np.int32)
        self.I1SEQY = np.zeros(self.NSEQMAX, dtype=np.int32)
        self.I1NEXT = np.zeros(self.NSEQMAX, dtype=np.int32)
        self.I2VECTOR = np.zeros((self.NX, self.NY), dtype=np.int32)

        NUPST = np.zeros((self.NX, self.NY), dtype=np.int32)
        UPNOW = np.zeros((self.NX, self.NY), dtype=np.int32)
        UPNMAX = 0

        # Count upstream
        for iy in range(self.NY):
            for ix in range(self.NX):
                if self.I2NEXTX[ix, iy] > 0 and self.I2REGION[ix, iy] == self.REGIONTHIS:
                    jx = self.I2NEXTX[ix, iy] - 1  # Convert to 0-based for array access
                    jy = self.I2NEXTY[ix, iy] - 1
                    if 0 <= jx < self.NX and 0 <= jy < self.NY:
                        NUPST[jx, jy] += 1
                        UPNMAX = max(UPNMAX, NUPST[jx, jy])

        # Register sequence
        iseq = 0  # Python 0-based counter
        for iy in range(self.NY):
            for ix in range(self.NX):
                if self.I2NEXTX[ix, iy] > 0 and self.I2REGION[ix, iy] == self.REGIONTHIS:
                    if NUPST[ix, iy] == UPNOW[ix, iy]:
                        # Store 1-based indices (ix+1, iy+1)
                        self.I1SEQX[iseq] = ix + 1
                        self.I1SEQY[iseq] = iy + 1
                        self.I2VECTOR[ix, iy] = iseq + 1  # Store 1-based sequence index
                        iseq += 1

        iseq1 = 0  # Python 0-based
        iseq2 = iseq - 1

        # Iterative sequence registration
        again = True
        while again:
            again = False
            jseq = iseq2
            for iseq in range(iseq1, iseq2 + 1):
                ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
                iy = self.I1SEQY[iseq] - 1
                jx = self.I2NEXTX[ix, iy] - 1  # Convert to 0-based
                jy = self.I2NEXTY[ix, iy] - 1
                if 0 <= jx < self.NX and 0 <= jy < self.NY:
                    UPNOW[jx, jy] += 1
                    if UPNOW[jx, jy] == NUPST[jx, jy] and self.I2NEXTX[jx, jy] > 0:
                        jseq += 1
                        self.I1SEQX[jseq] = jx + 1  # Store 1-based
                        self.I1SEQY[jseq] = jy + 1
                        self.I2VECTOR[jx, jy] = jseq + 1  # Store 1-based
                        again = True
            iseq1 = iseq2 + 1
            iseq2 = jseq

        self.NSEQRIV = jseq + 1  # Convert to count (1-based)

        # River mouth
        iseq = self.NSEQRIV
        for iy in range(self.NY):
            for ix in range(self.NX):
                if (self.I2NEXTX[ix, iy] < 0 and
                    self.I2NEXTX[ix, iy] != IMIS and
                    self.I2REGION[ix, iy] == self.REGIONTHIS):
                    self.I1SEQX[iseq] = ix + 1  # Store 1-based
                    self.I1SEQY[iseq] = iy + 1
                    self.I2VECTOR[ix, iy] = iseq + 1  # Store 1-based
                    iseq += 1

        self.NSEQALL = iseq

        # Next array
        for iseq in range(self.NSEQALL):
            ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
            iy = self.I1SEQY[iseq] - 1
            if self.I2NEXTX[ix, iy] > 0:
                jx = self.I2NEXTX[ix, iy] - 1
                jy = self.I2NEXTY[ix, iy] - 1
                if 0 <= jx < self.NX and 0 <= jy < self.NY:
                    self.I1NEXT[iseq] = self.I2VECTOR[jx, jy]  # Already 1-based
            else:
                self.I1NEXT[iseq] = self.I2NEXTX[ix, iy]  # Negative value

        # Upstream matrix
        self.I1UPST = np.full((self.NSEQMAX, UPNMAX), -9999, dtype=np.int32)
        self.I1UPN = np.zeros(self.NSEQMAX, dtype=np.int32)

        for iseq in range(self.NSEQRIV):
            jseq = self.I1NEXT[iseq]
            if jseq > 0:
                jseq_idx = jseq - 1  # Convert to 0-based for array access
                if jseq_idx < self.NSEQMAX:
                    upn = self.I1UPN[jseq_idx]
                    if upn < UPNMAX:
                        self.I1UPST[jseq_idx, upn] = iseq + 1  # Store 1-based
                        self.I1UPN[jseq_idx] += 1

        print(f"NSEQRIV = {self.NSEQRIV}")
        print(f"NSEQALL = {self.NSEQALL}")
        print("CMF::RIVMAP_INIT: end")

    def read_bifurcation(self):
        """Read bifurcation data"""
        print("\n*** Step 2: Reading Bifurcation Channels ***")
        print("READ_BIFURCATION: Reading bifurcation channel data")

        cpthout = os.path.join(self.INPUT_DIR, 'bifprm.txt')
        print(f"Bifurcation file: {cpthout}")

        with open(cpthout, 'r') as f:
            # Read first line: NPTHOUT, NPTHLEV
            line = f.readline().strip().split()
            self.NPTHOUT = int(line[0])
            self.NPTHLEV = int(line[1])

            print(f"Number of bifurcation channels: {self.NPTHOUT}")
            print(f"Number of bifurcation levels: {self.NPTHLEV}")

            # Allocate bifurcation arrays
            self.PTH_UPST = np.zeros(self.NPTHOUT, dtype=np.int32)
            self.PTH_DOWN = np.zeros(self.NPTHOUT, dtype=np.int32)
            self.PTH_DST = np.zeros(self.NPTHOUT, dtype=np.float64)
            self.PTH_ELV = np.zeros((self.NPTHLEV, self.NPTHOUT), dtype=np.float64)
            self.PTH_WTH = np.zeros((self.NPTHLEV, self.NPTHOUT), dtype=np.float64)
            self.PTH_MAN = np.zeros(self.NPTHLEV, dtype=np.float64)

            # Read bifurcation channel data
            npthout1 = 0
            for ipth in range(self.NPTHOUT):
                parts = f.readline().strip().split()
                ix = int(parts[0])
                iy = int(parts[1])
                jx = int(parts[2])
                jy = int(parts[3])
                self.PTH_DST[ipth] = float(parts[4])
                pelv = float(parts[5])
                pdph = float(parts[6])
                temp_wth = [float(parts[7 + i]) for i in range(self.NPTHLEV)]

                # Convert (ix, iy) to 0-based for array access
                ix_idx = ix - 1
                iy_idx = iy - 1
                jx_idx = jx - 1
                jy_idx = jy - 1

                # Get sequence indices (already 1-based in I2VECTOR)
                if 0 <= ix_idx < self.NX and 0 <= iy_idx < self.NY:
                    self.PTH_UPST[ipth] = self.I2VECTOR[ix_idx, iy_idx]
                else:
                    self.PTH_UPST[ipth] = 0

                if 0 <= jx_idx < self.NX and 0 <= jy_idx < self.NY:
                    self.PTH_DOWN[ipth] = self.I2VECTOR[jx_idx, jy_idx]
                else:
                    self.PTH_DOWN[ipth] = 0

                # Count valid bifurcations
                if self.PTH_UPST[ipth] > 0 and self.PTH_DOWN[ipth] > 0:
                    npthout1 += 1

                # Calculate elevation and store width for each level
                for ilev in range(self.NPTHLEV):
                    self.PTH_WTH[ilev, ipth] = temp_wth[ilev]

                    if ilev == 0:
                        # ILEV=1: water channel bifurcation
                        if self.PTH_WTH[ilev, ipth] > 0:
                            self.PTH_ELV[ilev, ipth] = pelv - pdph
                        else:
                            self.PTH_ELV[ilev, ipth] = 1.E20
                    else:
                        # ILEV>1: bank top levels
                        if self.PTH_WTH[ilev, ipth] > 0:
                            self.PTH_ELV[ilev, ipth] = pelv + ilev - 1.0
                        else:
                            self.PTH_ELV[ilev, ipth] = 1.E20

        # Set Manning coefficients
        for ilev in range(self.NPTHLEV):
            if ilev == 0:
                self.PTH_MAN[ilev] = PMANRIV
            else:
                self.PTH_MAN[ilev] = PMANFLD

        print(f"Bifurcation channels within domain: {npthout1}")
        if self.NPTHOUT != npthout1:
            print("Warning: Some bifurcation channels outside domain")

        print("READ_BIFURCATION: end")

    def read_inpmat(self):
        """Read input matrix"""
        print("\n*** Step 3: Reading Input Matrix ***")
        print("READ_INPMAT: Reading input matrix")

        cinpmat = os.path.join(self.INPUT_DIR, self.INPMAT_FILE)
        print(f"Input matrix file: {cinpmat}")
        print(f"NX, NY, INPN = {self.NX}, {self.NY}, {self.INPN}")

        # Allocate input matrix arrays
        self.INPX = np.zeros((self.INPN, self.NSEQMAX), dtype=np.int32)
        self.INPY = np.zeros((self.INPN, self.NSEQMAX), dtype=np.int32)
        self.INPA = np.zeros((self.INPN, self.NSEQMAX), dtype=np.float64)

        # Read input matrix file
        recl = 4 * self.NX * self.NY

        for inpi in range(self.INPN):
            # Read INPX (integer data)
            i2tmp = self.read_binary_single_record(cinpmat, self.NX, self.NY, rec=inpi+1, dtype='i4')
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
                iy = self.I1SEQY[iseq] - 1
                self.INPX[inpi, iseq] = i2tmp[ix, iy]

            # Read INPY (integer data)
            i2tmp = self.read_binary_single_record(cinpmat, self.NX, self.NY, rec=self.INPN+inpi+1, dtype='i4')
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1
                iy = self.I1SEQY[iseq] - 1
                self.INPY[inpi, iseq] = i2tmp[ix, iy]

            # Read INPA (float data)
            r2tmp = self.read_binary_single_record(cinpmat, self.NX, self.NY, rec=2*self.INPN+inpi+1, dtype='f4')
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1
                iy = self.I1SEQY[iseq] - 1
                self.INPA[inpi, iseq] = float(r2tmp[ix, iy])

        print("READ_INPMAT: end")

    def read_dam_param(self):
        """Read dam parameters"""
        print("\n*** Step 3.5: Reading Dam Parameters ***")
        print("READ_DAM_PARAM: Reading dam parameters")

        cdamparam = os.path.join(self.INPUT_DIR, 'dam_param.csv')
        print(f"Dam parameter file: {cdamparam}")

        if not os.path.exists(cdamparam):
            print(f"Warning: Dam parameter file not found: {cdamparam}")
            self.dam_NDAMS = 0
            return

        with open(cdamparam, 'r') as f:
            # Read first line: number of dams
            line = f.readline().strip()
            self.dam_NDAMS = int(line.split(',')[0])
            print(f"Number of dams: {self.dam_NDAMS}")

            # Read header line
            header = f.readline().strip()

            # Allocate dam arrays
            self.dam_GRAND_ID = np.zeros(self.dam_NDAMS, dtype=np.int32)
            self.dam_DamName = []
            self.dam_DamLat = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_DamLon = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_area_CaMa = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_DamIX = np.zeros(self.dam_NDAMS, dtype=np.int32)
            self.dam_DamIY = np.zeros(self.dam_NDAMS, dtype=np.int32)
            self.dam_FldVol_mcm = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_ConVol_mcm = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_TotalVol_mcm = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_Qn = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_Qf = np.zeros(self.dam_NDAMS, dtype=np.float64)
            self.dam_year = np.zeros(self.dam_NDAMS, dtype=np.int32)
            self.dam_seq = np.zeros(self.dam_NDAMS, dtype=np.int32)

            # Read dam data
            for idam in range(self.dam_NDAMS):
                line = f.readline().strip()
                parts = line.split(',')

                self.dam_GRAND_ID[idam] = int(parts[0])
                self.dam_DamName.append(parts[1])
                self.dam_DamLat[idam] = float(parts[2])
                self.dam_DamLon[idam] = float(parts[3])
                self.dam_area_CaMa[idam] = float(parts[4])
                self.dam_DamIX[idam] = int(parts[5])
                self.dam_DamIY[idam] = int(parts[6])
                self.dam_FldVol_mcm[idam] = float(parts[7])
                self.dam_ConVol_mcm[idam] = float(parts[8])
                self.dam_TotalVol_mcm[idam] = float(parts[9])
                self.dam_Qn[idam] = float(parts[10])
                self.dam_Qf[idam] = float(parts[11])
                self.dam_year[idam] = int(parts[12])

                # Calculate dam_seq using I2VECTOR
                ix = self.dam_DamIX[idam] - 1  # Convert to 0-based
                iy = self.dam_DamIY[idam] - 1
                if 0 <= ix < self.NX and 0 <= iy < self.NY:
                    self.dam_seq[idam] = self.I2VECTOR[ix, iy]  # Already 1-based
                else:
                    self.dam_seq[idam] = -9999

        print(f"Successfully read {self.dam_NDAMS} dams")
        print("READ_DAM_PARAM: end")

    def cmf_topo_init(self):
        """Initialize topography"""
        print("\n*** Step 4: Initializing Topography ***")
        print("CMF::TOPO_INIT: topography initialization")

        # Allocate arrays
        self.D2GRAREA = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2ELEVTN = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2NXTDST = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2RIVLEN = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2RIVWTH = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2RIVHGT = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2FLDHGT = np.zeros((self.NLFP, self.NSEQMAX), dtype=np.float64)
        self.D2RIVMAN = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2RIVELV = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2RIVSTOMAX = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2DWNELV = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.D2FLDSTOMAX = np.zeros((self.NLFP, self.NSEQMAX), dtype=np.float64)
        self.D2FLDGRD = np.zeros((self.NLFP, self.NSEQMAX), dtype=np.float64)
        self.D2UPAREA = np.zeros(self.NSEQMAX, dtype=np.float64)
        self.I2BASIN = np.zeros(self.NSEQMAX, dtype=np.int32)
        self.I2OUTCLM = np.zeros(self.NSEQMAX, dtype=np.int32)

        # Helper function to convert 2D map to 1D vector
        def map2vec(r2map):
            d1vec = np.zeros(self.NSEQMAX, dtype=np.float64)
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
                iy = self.I1SEQY[iseq] - 1
                d1vec[iseq] = float(r2map[ix, iy])
            return d1vec

        def map2vec_int(r2map):
            i1vec = np.zeros(self.NSEQMAX, dtype=np.int32)
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1
                iy = self.I1SEQY[iseq] - 1
                val = r2map[ix, iy]
                # Handle NaN and special values
                if np.isnan(val) or not np.isfinite(val):
                    i1vec[iseq] = IMIS  # -9999
                else:
                    i1vec[iseq] = int(val)
            return i1vec

        # Read topography data
        cgrarea = os.path.join(self.INPUT_DIR, 'ctmare.bin')
        print(f"Reading: {cgrarea}")
        r2temp = self.read_binary_single_record(cgrarea, self.NX, self.NY, rec=1)
        self.D2GRAREA = map2vec(r2temp)

        celevtn = os.path.join(self.INPUT_DIR, 'elevtn.bin')
        print(f"Reading: {celevtn}")
        r2temp = self.read_binary_single_record(celevtn, self.NX, self.NY, rec=1)
        self.D2ELEVTN = map2vec(r2temp)

        cnxtdst = os.path.join(self.INPUT_DIR, 'nxtdst.bin')
        print(f"Reading: {cnxtdst}")
        r2temp = self.read_binary_single_record(cnxtdst, self.NX, self.NY, rec=1)
        self.D2NXTDST = map2vec(r2temp)

        crivlen = os.path.join(self.INPUT_DIR, 'rivlen.bin')
        print(f"Reading: {crivlen}")
        r2temp = self.read_binary_single_record(crivlen, self.NX, self.NY, rec=1)
        self.D2RIVLEN = map2vec(r2temp)

        cfldhgt = os.path.join(self.INPUT_DIR, 'fldhgt.bin')
        print(f"Reading: {cfldhgt}")
        for ilfp in range(self.NLFP):
            r2temp = self.read_binary_single_record(cfldhgt, self.NX, self.NY, rec=ilfp+1)
            for iseq in range(self.NSEQMAX):
                ix = self.I1SEQX[iseq] - 1
                iy = self.I1SEQY[iseq] - 1
                self.D2FLDHGT[ilfp, iseq] = float(r2temp[ix, iy])

        crivhgt = os.path.join(self.INPUT_DIR, 'rivhgt.bin')
        print(f"Reading: {crivhgt}")
        r2temp = self.read_binary_single_record(crivhgt, self.NX, self.NY, rec=1)
        self.D2RIVHGT = map2vec(r2temp)

        # CRITICAL FIX: Use rivwth_gwdlr.bin to match Fortran simulation
        # The Fortran namelist specifies CRIVWTH = "rivwth_gwdlr.bin"
        crivwth = os.path.join(self.INPUT_DIR, 'rivwth_gwdlr.bin')
        print(f"Reading: {crivwth}")
        r2temp = self.read_binary_single_record(crivwth, self.NX, self.NY, rec=1)
        self.D2RIVWTH = map2vec(r2temp)

        crivman = os.path.join(self.INPUT_DIR, 'rivman.bin')
        print(f"Reading: {crivman}")
        r2temp = self.read_binary_single_record(crivman, self.NX, self.NY, rec=1)
        self.D2RIVMAN = map2vec(r2temp)

        cuparea = os.path.join(self.INPUT_DIR, 'uparea.bin')
        print(f"Reading: {cuparea}")
        r2temp = self.read_binary_single_record(cuparea, self.NX, self.NY, rec=1)
        self.D2UPAREA = map2vec(r2temp)

        cbasin = os.path.join(self.INPUT_DIR, 'basin.bin')
        print(f"Reading: {cbasin}")
        r2temp = self.read_binary_single_record(cbasin, self.NX, self.NY, rec=1, dtype='i4')
        self.I2BASIN = map2vec_int(r2temp)

        # Basin outlet column (outlet/mouth location code)
        # NOTE: Despite the long_name, reference file actually contains discharge values (outclm.bin)
        # Read outclm.bin as float then convert to int to match reference file behavior
        coutclm = os.path.join(self.INPUT_DIR, 'outclm.bin')
        print(f"Reading: {coutclm}")
        r2temp_float = self.read_binary_single_record(coutclm, self.NX, self.NY, rec=1, dtype='f4')
        # Convert float discharge to int (truncate) to match reference behavior
        r2temp = r2temp_float.astype('i4')
        self.I2OUTCLM = map2vec_int(r2temp)

        # Calculate channel parameters
        print("Calculating channel parameters...")
        self.D2RIVSTOMAX = self.D2RIVLEN * self.D2RIVWTH * self.D2RIVHGT
        self.D2RIVELV = self.D2ELEVTN - self.D2RIVHGT

        # Calculate floodplain parameters
        print("Calculating floodplain parameters...")
        dfrcinc = 1.0 / self.NLFP

        for iseq in range(self.NSEQMAX):
            dstopre = self.D2RIVSTOMAX[iseq]
            dhgtpre = 0.0
            dwthinc = self.D2GRAREA[iseq] / self.D2RIVLEN[iseq] * dfrcinc

            for ilfp in range(self.NLFP):
                dstonow = (self.D2RIVLEN[iseq] *
                          (self.D2RIVWTH[iseq] + dwthinc * (ilfp + 0.5)) *
                          (self.D2FLDHGT[ilfp, iseq] - dhgtpre))
                self.D2FLDSTOMAX[ilfp, iseq] = dstopre + dstonow
                self.D2FLDGRD[ilfp, iseq] = (self.D2FLDHGT[ilfp, iseq] - dhgtpre) / dwthinc
                dstopre = self.D2FLDSTOMAX[ilfp, iseq]
                dhgtpre = self.D2FLDHGT[ilfp, iseq]

        self.D2DWNELV = self.D2ELEVTN.copy()

        print("CMF::TOPO_INIT: end")

    def map2vec_general(self, r2map):
        """Convert 2D map to 1D vector.
        
        Args:
            r2map: 2D array with shape (NX, NY), i.e., (lon, lat).
                   Caller must transpose (lat, lon) data before calling.
        
        Returns:
            1D vector of length NSEQMAX with values extracted at river sequence points.
        """
        assert r2map.shape == (self.NX, self.NY), \
            f"Expected shape ({self.NX}, {self.NY}), got {r2map.shape}. Did you forget to transpose?"
            
        d1vec = np.zeros(self.NSEQMAX, dtype=np.float64)
        for iseq in range(self.NSEQMAX):
            ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
            iy = self.I1SEQY[iseq] - 1
            d1vec[iseq] = float(r2map[ix, iy])
        return d1vec


    def read_sediment_data(self):
        """Read sediment data from NC files (sedfrc and slope)"""
        print("\n*** Step 4.5: Reading Sediment Data ***")
        if not self.SED_DIR or (not self.SED_FRC and not self.SED_SLOPE):
            print("No sediment data configured, skipping.")
            return

        print(f"Sediment dir: {self.SED_DIR}")
        
        # Helper to get full path
        def get_path(fname):
            if os.path.isabs(fname): return fname
            return os.path.join(self.SED_DIR, fname)

        # 1. Read sedfrc
        if self.SED_FRC:
            fpath = get_path(self.SED_FRC)
            print(f"Reading {fpath}...")
            try:
                with Dataset(fpath, 'r') as nc:
                    # sedfrc: (sed, lat, lon) -> (3, 720, 1440)
                    v = nc.variables['sedfrc'][:]
                    # shape (3, NY, NX)
                    n_sed = v.shape[0]
                    self.sed_frc = np.zeros((n_sed, self.NSEQMAX), dtype=np.float64)
                    
                    for i in range(n_sed):
                        # v[i] is (NY, NX)
                        # Transpose to (NX, NY)
                        layer = v[i].T
                        self.sed_frc[i, :] = self.map2vec_general(layer)
                        
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

        # 2. Read slope
        if self.SED_SLOPE:
            fpath = get_path(self.SED_SLOPE)
            print(f"Reading {fpath}...")
            try:
                with Dataset(fpath, 'r') as nc:
                    # slope: (layer, lat, lon) -> (10, 720, 1440)
                    v = nc.variables['slope'][:]
                    # shape (10, NY, NX)
                    n_layer = v.shape[0]
                    self.sed_slope = np.zeros((n_layer, self.NSEQMAX), dtype=np.float64)
                    
                    for i in range(n_layer):
                        layer = v[i].T
                        self.sed_slope[i, :] = self.map2vec_general(layer)
                        
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
            
        print("READ_SEDIMENT: end")

    def export_to_netcdf(self):
        """Export to NetCDF"""
        print("\n*** Step 5: Exporting to NetCDF ***")
        print(f"Creating NetCDF file: {self.OUTFILE}")

        # Prepare sequence coordinate arrays
        i1seq = np.arange(1, self.NSEQMAX + 1, dtype=np.int32)  # 1-based
        d1seqlon = np.zeros(self.NSEQMAX, dtype=np.float64)
        d1seqlat = np.zeros(self.NSEQMAX, dtype=np.float64)

        for iseq in range(self.NSEQMAX):
            ix = self.I1SEQX[iseq] - 1  # Convert to 0-based
            iy = self.I1SEQY[iseq] - 1
            d1seqlon[iseq] = self.D1LON[ix]
            d1seqlat[iseq] = self.D1LAT[iy]

        # Create NetCDF file
        with Dataset(self.OUTFILE, 'w', format='NETCDF4') as ncfile:
            # Define dimensions
            ncfile.createDimension('nseqmax', self.NSEQMAX)
            ncfile.createDimension('nx', self.NX)
            ncfile.createDimension('ny', self.NY)
            ncfile.createDimension('nlfp', self.NLFP)
            ncfile.createDimension('upnmax', self.I1UPST.shape[1])
            ncfile.createDimension('npthout', self.NPTHOUT)
            ncfile.createDimension('npthlev', self.NPTHLEV)
            ncfile.createDimension('inpn', self.INPN)
            ncfile.createDimension('dam_ndams', self.dam_NDAMS)
            ncfile.createDimension('dam_namelen', 256)

            # Global attributes
            ncfile.title = 'CaMa-Flood River Network and Topography Data'
            ncfile.nx = self.NX
            ncfile.ny = self.NY
            ncfile.nlfp = self.NLFP
            ncfile.nseqriv = self.NSEQRIV
            ncfile.nseqall = self.NSEQALL
            ncfile.nseqmax = self.NSEQMAX
            ncfile.npthout = self.NPTHOUT
            ncfile.npthlev = self.NPTHLEV
            ncfile.inpn = self.INPN
            ncfile.dam_ndams = self.dam_NDAMS
            ncfile.west = self.WEST
            ncfile.east = self.EAST
            ncfile.north = self.NORTH
            ncfile.south = self.SOUTH

            print("Writing data to NetCDF...")

            # Define and write coordinate variables
            lon = ncfile.createVariable('lon', 'f8', ('nx',))
            lon.long_name = 'longitude'
            lon.units = 'degrees_east'
            lon[:] = self.D1LON

            lat = ncfile.createVariable('lat', 'f8', ('ny',))
            lat.long_name = 'latitude'
            lat.units = 'degrees_north'
            lat[:] = self.D1LAT

            # Define and write sequence variables
            seq = ncfile.createVariable('seq', 'i4', ('nseqmax',))
            seq.long_name = 'sequence index'
            seq.units = '1'
            seq[:] = i1seq

            seq_lon = ncfile.createVariable('seq_lon', 'f8', ('nseqmax',))
            seq_lon.long_name = 'sequence longitude'
            seq_lon.units = 'degrees_east'
            seq_lon[:] = d1seqlon

            seq_lat = ncfile.createVariable('seq_lat', 'f8', ('nseqmax',))
            seq_lat.long_name = 'sequence latitude'
            seq_lat.units = 'degrees_north'
            seq_lat[:] = d1seqlat

            seq_x = ncfile.createVariable('seq_x', 'i4', ('nseqmax',))
            seq_x.long_name = 'sequence x-index'
            seq_x.units = '1'
            seq_x[:] = self.I1SEQX

            seq_y = ncfile.createVariable('seq_y', 'i4', ('nseqmax',))
            seq_y.long_name = 'sequence y-index'
            seq_y.units = '1'
            seq_y[:] = self.I1SEQY

            seq_next = ncfile.createVariable('seq_next', 'i4', ('nseqmax',))
            seq_next.long_name = 'downstream sequence'
            seq_next.units = '1'
            seq_next[:] = self.I1NEXT

            # seq_upst: Fortran has ('upnmax', 'nseqmax'), NetCDF should match
            # I1UPST is (NSEQMAX, UPNMAX), need to transpose for NetCDF
            seq_upst = ncfile.createVariable('seq_upst', 'i4', ('upnmax', 'nseqmax'))
            seq_upst.long_name = 'upstream sequences'
            seq_upst.units = '1'
            seq_upst[:] = self.I1UPST.T

            seq_upn = ncfile.createVariable('seq_upn', 'i4', ('nseqmax',))
            seq_upn.long_name = 'number of upstream'
            seq_upn.units = '1'
            seq_upn[:] = self.I1UPN

            # Define and write topography variables
            topo_area = ncfile.createVariable('topo_area', 'f8', ('nseqmax',))
            topo_area.long_name = 'unit-catchment area'
            topo_area.units = 'm2'
            topo_area[:] = self.D2GRAREA

            topo_elevation = ncfile.createVariable('topo_elevation', 'f8', ('nseqmax',))
            topo_elevation.long_name = 'bank top elevation'
            topo_elevation.units = 'm'
            topo_elevation[:] = self.D2ELEVTN

            topo_distance = ncfile.createVariable('topo_distance', 'f8', ('nseqmax',))
            topo_distance.long_name = 'distance to downstream'
            topo_distance.units = 'm'
            topo_distance[:] = self.D2NXTDST

            topo_rivlen = ncfile.createVariable('topo_rivlen', 'f8', ('nseqmax',))
            topo_rivlen.long_name = 'river channel length'
            topo_rivlen.units = 'm'
            topo_rivlen[:] = self.D2RIVLEN

            topo_rivwth = ncfile.createVariable('topo_rivwth', 'f8', ('nseqmax',))
            topo_rivwth.long_name = 'river channel width'
            topo_rivwth.units = 'm'
            topo_rivwth[:] = self.D2RIVWTH

            topo_rivhgt = ncfile.createVariable('topo_rivhgt', 'f8', ('nseqmax',))
            topo_rivhgt.long_name = 'river channel depth'
            topo_rivhgt.units = 'm'
            topo_rivhgt[:] = self.D2RIVHGT

            topo_rivman = ncfile.createVariable('topo_rivman', 'f8', ('nseqmax',))
            topo_rivman.long_name = 'river manning coefficient'
            topo_rivman.units = '1'
            topo_rivman[:] = self.D2RIVMAN

            topo_rivelv = ncfile.createVariable('topo_rivelv', 'f8', ('nseqmax',))
            topo_rivelv.long_name = 'river bed elevation'
            topo_rivelv.units = 'm'
            topo_rivelv[:] = self.D2RIVELV

            topo_rivstomax = ncfile.createVariable('topo_rivstomax', 'f8', ('nseqmax',))
            topo_rivstomax.long_name = 'max river storage'
            topo_rivstomax.units = 'm3'
            topo_rivstomax[:] = self.D2RIVSTOMAX

            # Floodplain variables - dimension order: ('nseqmax', 'nlfp') for NetCDF
            # But we need to transpose since Fortran is (NLFP, NSEQMAX)
            topo_fldhgt = ncfile.createVariable('topo_fldhgt', 'f8', ('nseqmax', 'nlfp'))
            topo_fldhgt.long_name = 'floodplain height profile'
            topo_fldhgt.units = 'm'
            topo_fldhgt[:] = self.D2FLDHGT.T

            topo_fldstomax = ncfile.createVariable('topo_fldstomax', 'f8', ('nseqmax', 'nlfp'))
            topo_fldstomax.long_name = 'max floodplain storage'
            topo_fldstomax.units = 'm3'
            topo_fldstomax[:] = self.D2FLDSTOMAX.T

            topo_fldgrd = ncfile.createVariable('topo_fldgrd', 'f8', ('nseqmax', 'nlfp'))
            topo_fldgrd.long_name = 'floodplain gradient'
            topo_fldgrd.units = '1'
            topo_fldgrd[:] = self.D2FLDGRD.T

            topo_dwnelv = ncfile.createVariable('topo_dwnelv', 'f8', ('nseqmax',))
            topo_dwnelv.long_name = 'downstream boundary elevation'
            topo_dwnelv.units = 'm'
            topo_dwnelv[:] = self.D2DWNELV

            topo_uparea = ncfile.createVariable('topo_uparea', 'f8', ('nseqmax',))
            topo_uparea.long_name = 'upstream accumulated area'
            topo_uparea.units = 'm2'
            topo_uparea[:] = self.D2UPAREA

            topo_basin = ncfile.createVariable('topo_basin', 'i4', ('nseqmax',))
            topo_basin.long_name = 'basin ID'
            topo_basin.units = '1'
            topo_basin[:] = self.I2BASIN

            topo_outclm = ncfile.createVariable('topo_outclm', 'i4', ('nseqmax',))
            topo_outclm.long_name = 'outlet/mouth location code'
            topo_outclm.units = '1'
            topo_outclm[:] = self.I2OUTCLM

            # Define and write bifurcation variables
            bifurcation_upst = ncfile.createVariable('bifurcation_upst', 'i4', ('npthout',))
            bifurcation_upst.long_name = 'bifurcation upstream sequence'
            bifurcation_upst.units = '1'
            bifurcation_upst[:] = self.PTH_UPST

            bifurcation_down = ncfile.createVariable('bifurcation_down', 'i4', ('npthout',))
            bifurcation_down.long_name = 'bifurcation downstream sequence'
            bifurcation_down.units = '1'
            bifurcation_down[:] = self.PTH_DOWN

            bifurcation_distance = ncfile.createVariable('bifurcation_distance', 'f8', ('npthout',))
            bifurcation_distance.long_name = 'bifurcation channel distance'
            bifurcation_distance.units = 'm'
            bifurcation_distance[:] = self.PTH_DST

            # Bifurcation elevation and width - dimension order: ('npthout', 'npthlev')
            bifurcation_elevation = ncfile.createVariable('bifurcation_elevation', 'f8', ('npthout', 'npthlev'))
            bifurcation_elevation.long_name = 'bifurcation elevation profile'
            bifurcation_elevation.units = 'm'
            bifurcation_elevation[:] = self.PTH_ELV.T

            bifurcation_width = ncfile.createVariable('bifurcation_width', 'f8', ('npthout', 'npthlev'))
            bifurcation_width.long_name = 'bifurcation width profile'
            bifurcation_width.units = 'm'
            bifurcation_width[:] = self.PTH_WTH.T

            bifurcation_manning = ncfile.createVariable('bifurcation_manning', 'f8', ('npthlev',))
            bifurcation_manning.long_name = 'bifurcation Manning coefficients'
            bifurcation_manning.units = '1'
            bifurcation_manning[:] = self.PTH_MAN

            # Define and write input matrix variables - dimension order: ('nseqmax', 'inpn')
            inpmat_x = ncfile.createVariable('inpmat_x', 'i4', ('nseqmax', 'inpn'))
            inpmat_x.long_name = 'input matrix X index'
            inpmat_x.units = '1'
            inpmat_x[:] = self.INPX.T

            inpmat_y = ncfile.createVariable('inpmat_y', 'i4', ('nseqmax', 'inpn'))
            inpmat_y.long_name = 'input matrix Y index'
            inpmat_y.units = '1'
            inpmat_y[:] = self.INPY.T

            inpmat_area = ncfile.createVariable('inpmat_area', 'f8', ('nseqmax', 'inpn'))
            inpmat_area.long_name = 'input matrix area weight'
            inpmat_area.units = '1'
            inpmat_area[:] = self.INPA.T

            # Define and write dam variables
            if self.dam_NDAMS > 0:
                dam_GRAND_ID = ncfile.createVariable('dam_GRAND_ID', 'i4', ('dam_ndams',))
                dam_GRAND_ID.long_name = 'GRAND dam ID'
                dam_GRAND_ID.units = '1'
                dam_GRAND_ID[:] = self.dam_GRAND_ID

                dam_DamName = ncfile.createVariable('dam_DamName', 'S1', ('dam_ndams', 'dam_namelen'))
                dam_DamName.long_name = 'Dam name'
                dam_DamName.units = '1'
                for i, name in enumerate(self.dam_DamName):
                    # Convert string to character array
                    name_array = np.zeros(256, dtype='S1')
                    name_bytes = name.encode('utf-8')[:256]
                    for j, char in enumerate(name_bytes):
                        name_array[j] = char.to_bytes(1, 'little')
                    dam_DamName[i, :] = name_array

                dam_DamLat = ncfile.createVariable('dam_DamLat', 'f8', ('dam_ndams',))
                dam_DamLat.long_name = 'Dam latitude'
                dam_DamLat.units = 'degrees_north'
                dam_DamLat[:] = self.dam_DamLat

                dam_DamLon = ncfile.createVariable('dam_DamLon', 'f8', ('dam_ndams',))
                dam_DamLon.long_name = 'Dam longitude'
                dam_DamLon.units = 'degrees_east'
                dam_DamLon[:] = self.dam_DamLon

                dam_area_CaMa = ncfile.createVariable('dam_area_CaMa', 'f8', ('dam_ndams',))
                dam_area_CaMa.long_name = 'Dam catchment area in CaMa'
                dam_area_CaMa.units = 'km2'
                dam_area_CaMa[:] = self.dam_area_CaMa

                dam_DamIX = ncfile.createVariable('dam_DamIX', 'i4', ('dam_ndams',))
                dam_DamIX.long_name = 'Dam X index'
                dam_DamIX.units = '1'
                dam_DamIX[:] = self.dam_DamIX

                dam_DamIY = ncfile.createVariable('dam_DamIY', 'i4', ('dam_ndams',))
                dam_DamIY.long_name = 'Dam Y index'
                dam_DamIY.units = '1'
                dam_DamIY[:] = self.dam_DamIY

                dam_FldVol_mcm = ncfile.createVariable('dam_FldVol_mcm', 'f8', ('dam_ndams',))
                dam_FldVol_mcm.long_name = 'Dam flood volume'
                dam_FldVol_mcm.units = 'million cubic meters'
                dam_FldVol_mcm[:] = self.dam_FldVol_mcm

                dam_ConVol_mcm = ncfile.createVariable('dam_ConVol_mcm', 'f8', ('dam_ndams',))
                dam_ConVol_mcm.long_name = 'Dam conservation volume'
                dam_ConVol_mcm.units = 'million cubic meters'
                dam_ConVol_mcm[:] = self.dam_ConVol_mcm

                dam_TotalVol_mcm = ncfile.createVariable('dam_TotalVol_mcm', 'f8', ('dam_ndams',))
                dam_TotalVol_mcm.long_name = 'Dam total volume'
                dam_TotalVol_mcm.units = 'million cubic meters'
                dam_TotalVol_mcm[:] = self.dam_TotalVol_mcm

                dam_Qn = ncfile.createVariable('dam_Qn', 'f8', ('dam_ndams',))
                dam_Qn.long_name = 'Dam normal discharge'
                dam_Qn.units = 'm3/s'
                dam_Qn[:] = self.dam_Qn

                dam_Qf = ncfile.createVariable('dam_Qf', 'f8', ('dam_ndams',))
                dam_Qf.long_name = 'Dam flood discharge'
                dam_Qf.units = 'm3/s'
                dam_Qf[:] = self.dam_Qf

                dam_year = ncfile.createVariable('dam_year', 'i4', ('dam_ndams',))
                dam_year.long_name = 'Dam construction year'
                dam_year.units = 'year'
                dam_year[:] = self.dam_year

                dam_seq = ncfile.createVariable('dam_seq', 'i4', ('dam_ndams',))
                dam_seq.long_name = 'Dam sequence index'
                dam_seq.units = '1'
                dam_seq[:] = self.dam_seq

            # Define and write sediment variables
            if self.sed_frc is not None or self.sed_slope is not None:
                print("Writing sediment variables...")
                # Dimensions
                if self.sed_frc is not None:
                    ncfile.createDimension('sed_n', self.sed_frc.shape[0])
                if self.sed_slope is not None:
                    ncfile.createDimension('slope_layers', self.sed_slope.shape[0])

                if self.sed_frc is not None:
                    sed_frc = ncfile.createVariable('sed_frc', 'f8', ('nseqmax', 'sed_n'))
                    sed_frc.long_name = 'sediment fraction'
                    sed_frc[:] = self.sed_frc.T
                
                if self.sed_slope is not None:
                    sed_slope = ncfile.createVariable('sed_slope', 'f8', ('nseqmax', 'slope_layers'))
                    sed_slope.long_name = 'sediment slope'
                    sed_slope[:] = self.sed_slope.T

        print("NetCDF export completed!")

    def run(self):
        """Main execution flow"""
        print()
        print("=" * 46)
        print("  CaMa-Flood Data Export to NetCDF (Python)")
        print("=" * 46)
        print()

        # Parse command line arguments
        self.parse_arguments()

        # Read parameter file
        self.read_param_file()

        # Initialize river network
        self.cmf_rivmap_init()

        # Read bifurcation data
        self.read_bifurcation()

        # Read input matrix
        self.read_inpmat()

        # Read dam parameters
        self.read_dam_param()

        # Initialize topography
        self.cmf_topo_init()

        # Read sediment data
        self.read_sediment_data()

        # Export to NetCDF
        self.export_to_netcdf()

        print()
        print("=" * 46)
        print("  Export completed successfully!")
        print(f"  Output file: {self.OUTFILE}")
        print("=" * 46)


if __name__ == '__main__':
    app = CoLMGridRoutingInit()
    app.run()
