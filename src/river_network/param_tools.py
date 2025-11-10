"""
Parameter calculation tools - equivalent to src_param
"""
import numpy as np
import os
from .fortran_io import FortranBinary, read_params_txt
from .geo_utils import rgetarea


class ParamProcessor:
    """Handles parameter calculations for river routing"""

    def __init__(self, config):
        self.config = config
        self.imis = -9999
        self.rmis = 1.e20

    def generate_inpmat(self, map_dir, hires_tag, gsizein, westin, eastin,
                       northin, southin, olat, diminfo_file, inpmat_file):
        """
        Generate input matrix for runoff interpolation
        Equivalent to src_param/generate_inpmat.F90
        """
        print("=" * 60)
        print("GENERATE_INPMAT: Creating runoff interpolation matrix")
        print("=" * 60)

        # Read map parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']
        nflp = params['nflp']
        gsize = params['gsize']
        west, east, north, south = params['west'], params['east'], params['north'], params['south']

        nxin = int(round((eastin - westin) / gsizein))
        nyin = int(round((northin - southin) / gsizein))

        print(f"Input grid: {nxin} x {nyin}, resolution: {gsizein} deg")
        print(f"Map grid: {nx} x {ny}, resolution: {gsize} deg")
        print(f"North-south order: {olat}")

        # Read high-resolution catchment data
        hires_dir = os.path.join(map_dir, hires_tag)
        location_file = os.path.join(hires_dir, 'location.txt')

        if not os.path.exists(location_file):
            raise FileNotFoundError(f"High-resolution data not found: {location_file}")

        # Initialize input matrix
        nmax = 100  # Maximum number of input grids per output grid
        inpn = np.zeros((nx, ny), dtype=np.int32, order='F')
        inpx = np.zeros((nx, ny, nmax), dtype=np.int32, order='F')
        inpy = np.zeros((nx, ny, nmax), dtype=np.int32, order='F')
        inpa = np.zeros((nx, ny, nmax), dtype=np.float32, order='F')

        # Read location info and process catchments
        with open(location_file, 'r') as f:
            narea = int(f.readline().split()[0])  # Read narea from "1     narea" format
            f.readline()  # Skip column header

            for i in range(narea):
                parts = f.readline().split()
                area = parts[1]
                lon_ori = float(parts[2])
                lat_ori = float(parts[5])
                nx_area = int(parts[6])
                ny_area = int(parts[7])
                csize = float(parts[8])

                print(f"Processing area: {area} ({nx_area}x{ny_area})")

                # Read catchment XY
                catmxy_file = os.path.join(hires_dir, f"{area}.catmxy.bin")
                if not os.path.exists(catmxy_file):
                    print(f"  Skipping {area} - file not found")
                    continue

                catmXX = FortranBinary.read_direct(catmxy_file, (nx_area, ny_area), 'int2', rec=1)
                catmYY = FortranBinary.read_direct(catmxy_file, (nx_area, ny_area), 'int2', rec=2)

                # Calculate lon/lat arrays for the high-res grid
                lon = lon_ori + (np.arange(nx_area) + 0.5) * csize
                lat = lat_ori - (np.arange(ny_area) + 0.5) * csize

                # Read or calculate grdare
                grdare_file = os.path.join(hires_dir, f"{area}.grdare.bin")
                if os.path.exists(grdare_file):
                    print(f"  Using existing {area}.grdare.bin")
                    carea = FortranBinary.read_direct(grdare_file, (nx_area, ny_area), 'real', rec=1)
                    # Convert from km² to m² (match Fortran: carea*1.e6)
                    carea = carea * 1.e6
                else:
                    print(f"  Calculating grid areas...")
                    # Vectorized area calculation
                    carea = np.zeros((nx_area, ny_area), dtype=np.float32)
                    for iy in range(ny_area):
                        area_val = rgetarea(0., csize * 10., lat[iy] - 5.0 * csize,
                                           lat[iy] + 5.0 * csize) * 0.01 * 1.e6  # km² -> m²
                        carea[:, iy] = area_val  # Broadcast to all x

                # Process pixels in exact Fortran order: do iy=1, ny; do ix=1, nx
                # This ensures identical inpmat generation
                for iy in range(ny_area):
                    for ix in range(nx_area):
                        # Get catchment indices (1-based in Fortran)
                        iXX_1based = int(catmXX[ix, iy])
                        iYY_1based = int(catmYY[ix, iy])

                        # Check valid catchment (Fortran: if( catmXX(ix,iy)>0 .and. catmXX(ix,iy)<=nXX ... ))
                        if not (iXX_1based > 0 and iXX_1based <= nx and
                               iYY_1based > 0 and iYY_1based <= ny):
                            continue

                        # Convert to 0-based for Python arrays
                        iXX = iXX_1based - 1
                        iYY = iYY_1based - 1

                        # Get coordinates
                        lon0 = np.float32(lon[ix])
                        lat0 = np.float32(lat[iy])

                        # Adjust longitude for wrap-around (Fortran: if( lon0<westin ) lon0=lon0+360.)
                        if lon0 < westin:
                            lon0 = lon0 + 360.0
                        if lon0 > eastin:
                            lon0 = lon0 - 360.0

                        # Check if within input domain
                        if not (lon0 >= westin and lon0 <= eastin and
                               lat0 >= southin and lat0 <= northin):
                            continue

                        # Calculate input grid indices (Fortran: ixin=int( (lon0-westin )/gsizein )+1)
                        ixin = int((lon0 - westin) / gsizein) + 1  # 1-based
                        iyin = int((northin - lat0) / gsizein) + 1  # 1-based

                        # Clip to valid range (Fortran: ixin=max(1,min(nxin,ixin)))
                        ixin = max(1, min(nxin, ixin))
                        iyin = max(1, min(nyin, iyin))

                        # Handle South-to-North ordering
                        if olat == 'StoN':
                            iyin = nyin - iyin + 1

                        # Get pixel area
                        pixel_area = np.float32(carea[ix, iy])

                        # Check if this input grid already exists for this river grid
                        # Fortran: do inum=1, inpn(iXX,iYY); if( inpx(iXX,iYY,inum)==ixin .and. inpy(iXX,iYY,inum)==iyin )
                        new = 1
                        if inpn[iXX, iYY] >= 1:
                            for inum in range(inpn[iXX, iYY]):
                                if inpx[iXX, iYY, inum] == ixin and inpy[iXX, iYY, inum] == iyin:
                                    new = 0
                                    inpa[iXX, iYY, inum] = inpa[iXX, iYY, inum] + pixel_area
                                    break

                        # Add new entry if not found
                        if new == 1:
                            inum = inpn[iXX, iYY]
                            if inum >= nmax:
                                print(f"*** error: nmax overflow at ({iXX}, {iYY}) **********")
                                continue
                            inpx[iXX, iYY, inum] = ixin
                            inpy[iXX, iYY, inum] = iyin
                            inpa[iXX, iYY, inum] = pixel_area
                            inpn[iXX, iYY] += 1

        # Find maximum number of inputs
        mmax = int(np.max(inpn))
        print(f"Maximum number of input grids per output grid: {mmax}")

        # Write dimension info
        # Use only the basename (relative path) to match Fortran output
        inpmat_basename = os.path.basename(inpmat_file)
        with open(diminfo_file, 'w') as f:
            f.write(f"{nx:10d}     !! nXX\n")
            f.write(f"{ny:10d}     !! nYY\n")
            f.write(f"{nflp:10d}     !! floodplain layer\n")
            f.write(f"{nxin:10d}     !! input nXX\n")
            f.write(f"{nyin:10d}     !! input nYY\n")
            f.write(f"{mmax:10d}     !! input num\n")
            f.write(f"{inpmat_basename}\n")
            f.write(f"{west:12.3f}     !! west  edge\n")
            f.write(f"{east:12.3f}     !! east  edge\n")
            f.write(f"{north:12.3f}     !! north edge\n")
            f.write(f"{south:12.3f}     !! south edge\n")

        # Write input matrix
        print(f"Writing input matrix to {inpmat_file}")
        for inum in range(mmax):
            FortranBinary.write_direct(inpmat_file, inpx[:, :, inum], rec=inum + 1)
            FortranBinary.write_direct(inpmat_file, inpy[:, :, inum], rec=mmax + inum + 1)
            FortranBinary.write_direct(inpmat_file, inpa[:, :, inum], rec=2 * mmax + inum + 1)

        print("GENERATE_INPMAT completed successfully!")

    def calc_outclm(self, map_dir, runoff_file, diminfo_file, data_type='cdf', runoff_var='ro'):
        """
        Calculate annual mean discharge from runoff climatology
        Equivalent to src_param/calc_outclm.F90

        Args:
            map_dir: Output directory
            runoff_file: Runoff climatology file (NetCDF or binary)
            diminfo_file: Dimension info file
            data_type: 'cdf' for NetCDF, 'bin' for binary
            runoff_var: NetCDF variable name for runoff
        """
        print("=" * 60)
        print("CALC_OUTCLM: Calculating annual mean discharge from runoff climatology")
        print("=" * 60)
        print(f"TYPE={data_type}")
        print(f"DIMINFO={diminfo_file}")
        print(f"Runoff={runoff_file}")

        # Read dimension info
        with open(diminfo_file, 'r') as f:
            nx = int(f.readline().split()[0])
            ny = int(f.readline().split()[0])
            nflp = int(f.readline().split()[0])
            nxin = int(f.readline().split()[0])
            nyin = int(f.readline().split()[0])
            inpn = int(f.readline().split()[0])
            cinpmat = f.readline().strip()
            west = float(f.readline().split()[0])
            east = float(f.readline().split()[0])
            north = float(f.readline().split()[0])
            south = float(f.readline().split()[0])

        print(f"River network: {nx} x {ny}")
        print(f"Input grid: {nxin} x {nyin}")
        print(f"Input matrix max: {inpn}")
        print(f"Input matrix file: {cinpmat}")

        # Read input matrix
        print("calc_outclm: read input matrix")
        inpmat_path = os.path.join(map_dir, cinpmat)
        inpx = np.zeros((nx, ny, inpn), dtype=np.int32, order='F')
        inpy = np.zeros((nx, ny, inpn), dtype=np.int32, order='F')
        inpa = np.zeros((nx, ny, inpn), dtype=np.float32, order='F')

        for i in range(inpn):
            inpx[:, :, i] = FortranBinary.read_direct(inpmat_path, (nx, ny), 'int4', rec=i + 1)
            inpy[:, :, i] = FortranBinary.read_direct(inpmat_path, (nx, ny), 'int4', rec=inpn + i + 1)
            inpa[:, :, i] = FortranBinary.read_direct(inpmat_path, (nx, ny), 'real', rec=2 * inpn + i + 1)

        # Read nextxy and ctmare
        print("calc_outclm: read nextxy.bin")
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        nexty = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=2)

        ctmare = FortranBinary.read_direct(os.path.join(map_dir, 'ctmare.bin'), (nx, ny), 'real', rec=1)

        # Calculate river sequence
        print("calc_outclm: calculate river sequence")
        upst = np.zeros((nx, ny), dtype=np.int32, order='F')
        upnow = np.zeros((nx, ny), dtype=np.int32, order='F')
        xseq = []
        yseq = []

        # Count number of upstreams
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] > 0:
                    jx = nextx[ix, iy] - 1  # Convert to 0-based
                    jy = nexty[ix, iy] - 1
                    upst[jx, jy] += 1
                elif nextx[ix, iy] == self.imis:
                    upst[ix, iy] = self.imis

        # Find topmost grids (no upstream)
        for iy in range(ny):
            for ix in range(nx):
                if upst[ix, iy] == 0:
                    xseq.append(ix)
                    yseq.append(iy)

        nseqpre = 0
        nseqnow = len(xseq)

        # Find downstream grids when all upstreams are registered
        again = True
        while again:
            again = False
            jseq = nseqnow
            for iseq in range(nseqpre, nseqnow):
                ix = xseq[iseq]
                iy = yseq[iseq]
                if nextx[ix, iy] > 0:
                    jx = nextx[ix, iy] - 1
                    jy = nexty[ix, iy] - 1
                    upnow[jx, jy] += 1
                    if upnow[jx, jy] == upst[jx, jy]:
                        again = True
                        xseq.append(jx)
                        yseq.append(jy)
                        jseq += 1
            nseqpre = nseqnow
            nseqnow = jseq

        print(f"River sequence length: {nseqnow}")

        # Read runoff climatology
        print("calc_outclm: read runoff climatology file")
        if data_type == 'cdf':
            try:
                import netCDF4
                nc = netCDF4.Dataset(runoff_file, 'r')
                # Read runoff data [time, lat, lon]
                roffin_raw = nc.variables[runoff_var][0, :, :]  # First time step
                nc.close()

                # NetCDF might be [lat, lon], need to transpose to [lon, lat] for Fortran order
                # Shape should be (nyin, nxin) which we'll access as roffin[jx-1, jy-1]
                roffin = np.asfortranarray(roffin_raw.T)  # Now shape is (nxin, nyin)
                print(f"Runoff data shape: {roffin.shape} (expected: {nxin} x {nyin})")

            except ImportError:
                raise ImportError("netCDF4 library required for reading NetCDF runoff data")
        else:
            # Binary format
            roffin = FortranBinary.read_direct(runoff_file, (nxin, nyin), 'real', rec=1)

        # Interpolate runoff to river network
        print("calc_outclm: interpolate runoff to river network")
        runoff = np.zeros((nx, ny), dtype=np.float32, order='F')

        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    for i in range(inpn):
                        jx = inpx[ix, iy, i]  # 1-based index
                        jy = inpy[ix, iy, i]  # 1-based index
                        if jx > 0:
                            # Convert to 0-based and check runoff value
                            ro_val = roffin[jx - 1, jy - 1]
                            if ro_val != self.rmis and not np.isnan(ro_val):
                                # runoff in mm/s, inpa in m2, convert to m3/s
                                # mm/s = 0.001 m/s, so: 0.001 m/s * m² = 0.001 m³/s
                                # Therefore: mm/s * m² * 1e-3 = m³/s ✓
                                runoff[ix, iy] += max(ro_val, 0.0) * inpa[ix, iy, i] * 1.e-3

        # Sum runoff from upstream to downstream to get discharge
        print("calc_outclm: calculate discharge")
        rivout = np.zeros((nx, ny), dtype=np.float32, order='F')

        for iseq in range(nseqnow):
            ix = xseq[iseq]
            iy = yseq[iseq]
            rivout[ix, iy] += runoff[ix, iy]
            if nextx[ix, iy] > 0:
                jx = nextx[ix, iy] - 1
                jy = nexty[ix, iy] - 1
                rivout[jx, jy] += rivout[ix, iy]

        # Set missing values
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] == self.imis:
                    rivout[ix, iy] = self.imis

        # Write output
        outclm_file = os.path.join(map_dir, 'outclm.bin')
        FortranBinary.write_direct(outclm_file, rivout, rec=1)

        print(f"Wrote discharge to {outclm_file}")
        print(f"Discharge range: {np.min(rivout[rivout != self.imis]):.2f} to {np.max(rivout[rivout != self.imis]):.2f} m3/s")
        print("CALC_OUTCLM completed successfully!")

    def calc_rivwth(self, map_dir, HC, HP, HO, HMIN, WC, WP, WO, WMIN):
        """
        Calculate river width and depth from discharge
        Equivalent to src_param/calc_rivwth.F90
        """
        print("=" * 60)
        print("CALC_RIVWTH: Calculating river channel parameters")
        print("=" * 60)
        print(f"HEIGHT H=max({HMIN}, {HC}*Qave**{HP} + {HO})")
        print(f"WIDTH  W=max({WMIN}, {WC}*Qave**{WP} + {WO})")

        # Read parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']

        # Read nextxy and discharge
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        rivout = FortranBinary.read_direct(os.path.join(map_dir, 'outclm.bin'), (nx, ny), 'real', rec=1)

        # Calculate width and depth
        rivwth = np.full((nx, ny), self.imis, dtype=np.float32, order='F')
        rivhgt = np.full((nx, ny), self.imis, dtype=np.float32, order='F')
        rivman = np.full((nx, ny), self.imis, dtype=np.float32, order='F')

        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    rivhgt[ix, iy] = max(HMIN, HC * rivout[ix, iy]**HP + HO)
                    rivwth[ix, iy] = max(WMIN, WC * rivout[ix, iy]**WP + WO)
                    rivman[ix, iy] = 0.03  # Manning's n

        # Write output
        FortranBinary.write_direct(os.path.join(map_dir, 'rivwth.bin'), rivwth, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'rivhgt.bin'), rivhgt, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'rivman.bin'), rivman, rec=1)

        # Create infinite height for no-floodplain simulation
        rivhgt_inf = np.full((nx, ny), 100000.0, dtype=np.float32, order='F')
        FortranBinary.write_direct(os.path.join(map_dir, 'rivhgt_inf.bin'), rivhgt_inf, rec=1)

        print("CALC_RIVWTH completed successfully!")

    def set_gwdlr(self, map_dir):
        """
        Merge empirical width with satellite-observed width
        Equivalent to src_param/set_gwdlr.F90
        """
        print("=" * 60)
        print("SET_GWDLR: Merging width data")
        print("=" * 60)

        # Read parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']

        # Read data
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        rivwth = FortranBinary.read_direct(os.path.join(map_dir, 'rivwth.bin'), (nx, ny), 'real', rec=1)
        width = FortranBinary.read_direct(os.path.join(map_dir, 'width.bin'), (nx, ny), 'real', rec=1)

        # Merge widths
        gwdlr = width.copy()

        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    if gwdlr[ix, iy] < 50:
                        gwdlr[ix, iy] = max(gwdlr[ix, iy], rivwth[ix, iy])
                    elif gwdlr[ix, iy] < rivwth[ix, iy] * 0.5:
                        gwdlr[ix, iy] = rivwth[ix, iy] * 0.5
                    else:
                        if gwdlr[ix, iy] > rivwth[ix, iy] * 5.0:
                            gwdlr[ix, iy] = rivwth[ix, iy] * 5.0
                        if gwdlr[ix, iy] > 10000.0:
                            gwdlr[ix, iy] = 10000.0
                else:
                    gwdlr[ix, iy] = self.imis

        # Write output
        FortranBinary.write_direct(os.path.join(map_dir, 'rivwth_gwdlr.bin'), gwdlr, rec=1)

        print("SET_GWDLR completed successfully!")

    def calc_prmwat(self, map_dir, hires_tag='1min'):
        """
        Calculate permanent water area in each unit-catchment
        Equivalent to src_param/calc_prmwat.F90

        Args:
            map_dir: Output directory
            hires_tag: High-resolution map tag (1min, 30sec, 15sec, etc.)
        """
        print("=" * 60)
        print("CALC_PRMWAT: Calculating permanent water area")
        print("=" * 60)
        print(f"High-resolution map: {hires_tag}")

        # Read parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']

        # Read nextxy and ctmare
        print("\nReading river network data...")
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        ctmare = FortranBinary.read_direct(os.path.join(map_dir, 'ctmare.bin'), (nx, ny), 'real', rec=1)

        # Initialize prmwat
        prmwat = np.full((nx, ny), self.imis, dtype=np.float32, order='F')
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    prmwat[ix, iy] = 0.0

        # Read location file
        hires_dir = os.path.join(map_dir, hires_tag)
        location_file = os.path.join(hires_dir, 'location.txt')

        if not os.path.exists(location_file):
            print(f"WARNING: High-resolution data not found: {location_file}")
            print("Skipping permanent water calculation.")
            print("Setting prmwat to zero for all valid grids.")
            # Write zero prmwat
            prmwat_file = os.path.join(map_dir, 'prmwat.bin')
            FortranBinary.write_direct(prmwat_file, prmwat, rec=1)
            return

        print(f"Reading location file: {location_file}")
        with open(location_file, 'r') as f:
            narea = int(f.readline().split()[0])
            f.readline()  # Skip header

            for i in range(narea):
                parts = f.readline().split()
                area = parts[1]
                west0 = float(parts[2])
                east0 = float(parts[3])
                south0 = float(parts[4])
                north0 = float(parts[5])
                nx0 = int(parts[6])
                ny0 = int(parts[7])
                csize = float(parts[8])

                print(f"\nProcessing area: {area} ({nx0}x{ny0}, csize={csize:.6f})")

                # Read high-resolution files
                catmxy_file = os.path.join(hires_dir, f"{area}.catmxy.bin")
                rivwth_file = os.path.join(hires_dir, f"{area}.rivwth.bin")
                grdare_file = os.path.join(hires_dir, f"{area}.grdare.bin")
                flddif_file = os.path.join(hires_dir, f"{area}.flddif.bin")

                if not os.path.exists(catmxy_file):
                    print(f"  Skipping {area} - catmxy file not found")
                    continue

                print(f"  Reading {area}.catmxy.bin...")
                catmXX = FortranBinary.read_direct(catmxy_file, (nx0, ny0), 'int2', rec=1)
                catmYY = FortranBinary.read_direct(catmxy_file, (nx0, ny0), 'int2', rec=2)

                print(f"  Reading {area}.rivwth.bin...")
                rivwth0 = FortranBinary.read_direct(rivwth_file, (nx0, ny0), 'real', rec=1)

                # Read or calculate grdare
                if os.path.exists(grdare_file):
                    print(f"  Reading {area}.grdare.bin...")
                    grdare0 = FortranBinary.read_direct(grdare_file, (nx0, ny0), 'real', rec=1)
                else:
                    print(f"  Calculating grid areas...")
                    grdare0 = np.full((nx0, ny0), self.imis, dtype=np.float32, order='F')
                    for iy0 in range(ny0):
                        clat = north0 - csize * (iy0 + 0.5)
                        carea = rgetarea(0., 10. * csize, clat + 5. * csize, clat - 5. * csize) * 0.01 * 1.e-6  # km2
                        for ix0 in range(nx0):
                            if catmXX[ix0, iy0] != self.imis:
                                grdare0[ix0, iy0] = carea

                print(f"  Reading {area}.flddif.bin...")
                flddif0 = FortranBinary.read_direct(flddif_file, (nx0, ny0), 'real', rec=1)

                # Accumulate permanent water area
                print(f"  Accumulating permanent water areas...")
                count = 0
                for iy0 in range(ny0):
                    for ix0 in range(nx0):
                        iXX = catmXX[ix0, iy0]
                        iYY = catmYY[ix0, iy0]

                        if iXX > 0:  # Valid catchment
                            rivwth_val = rivwth0[ix0, iy0]
                            flddif_val = flddif0[ix0, iy0]

                            # Check if this is permanent water
                            # rivwth > 0 or rivwth == -1 (permanent water marker)
                            # flddif <= 5m (exclude hilltop water)
                            if (rivwth_val > 0 or rivwth_val == -1) and flddif_val <= 5.0:
                                # Convert to 0-based indexing
                                ix = iXX - 1
                                iy = iYY - 1
                                if 0 <= ix < nx and 0 <= iy < ny:
                                    prmwat[ix, iy] += grdare0[ix0, iy0] * 1.e6  # km2 -> m2
                                    count += 1

                print(f"  Added {count} permanent water pixels")

        # Limit by catchment area
        print("\nApplying catchment area limiter...")
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    prmwat[ix, iy] = min(prmwat[ix, iy], ctmare[ix, iy])

        # Write output
        prmwat_file = os.path.join(map_dir, 'prmwat.bin')
        print(f"\nWriting permanent water to: {prmwat_file}")
        FortranBinary.write_direct(prmwat_file, prmwat, rec=1)

        # Statistics
        valid_prmwat = prmwat[prmwat != self.imis]
        if len(valid_prmwat) > 0:
            print(f"\nPermanent water statistics:")
            print(f"  Min: {np.min(valid_prmwat):.2f} m²")
            print(f"  Max: {np.max(valid_prmwat):.2f} m²")
            print(f"  Mean: {np.mean(valid_prmwat):.2f} m²")
            nonzero = valid_prmwat[valid_prmwat > 0]
            if len(nonzero) > 0:
                print(f"  Grids with water: {len(nonzero)} ({100*len(nonzero)/len(valid_prmwat):.1f}%)")

        print("\nCALC_PRMWAT completed successfully!")

    def set_bifparam(self, map_dir, bifori_file, nlev_new=5):
        """
        Set bifurcation parameters from bifori.txt
        Equivalent to src_param/set_bifparam.F90

        Args:
            map_dir: Output directory
            bifori_file: Original bifurcation file path
            nlev_new: Number of bifurcation layers to use (default: 5)
        """
        print("=" * 60)
        print("SET_BIFPARAM: Setting bifurcation parameters")
        print("=" * 60)
        print(f"Bifori file: {bifori_file}")
        print(f"Number of layers: {nlev_new}")

        # Read parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']

        # Read river network data
        print("\nReading river network data...")
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        nexty = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=2)
        rivhgt = FortranBinary.read_direct(os.path.join(map_dir, 'rivhgt.bin'), (nx, ny), 'real', rec=1)
        basin = FortranBinary.read_direct(os.path.join(map_dir, 'basin.bin'), (nx, ny), 'int4', rec=1)

        # Initialize bifurcation arrays
        bifdph = np.full((nx, ny), self.imis, dtype=np.float32, order='F')
        bifwth = np.full((nx, ny), self.imis, dtype=np.float32, order='F')

        # Read bifori.txt
        print(f"\nReading bifurcation file: {bifori_file}")
        if not os.path.exists(bifori_file):
            print(f"WARNING: Bifurcation file not found: {bifori_file}")
            print("Skipping bifurcation parameter generation.")
            return

        with open(bifori_file, 'r') as f:
            # Read header
            header = f.readline().split()
            npth = int(header[0])
            nlev = int(header[1])

            print(f"Original paths: {npth}, Original layers: {nlev}")

            # First pass: count how many paths have data in first nlev_new layers
            paths_data = []
            for ipth in range(npth):
                line = f.readline().split()
                ix = int(line[0]) - 1  # Convert to 0-based
                iy = int(line[1]) - 1
                jx = int(line[2]) - 1
                jy = int(line[3]) - 1
                length = float(line[4])
                elev = float(line[5])
                wth = [float(line[6 + i]) for i in range(nlev)]
                lat = float(line[6 + nlev])
                lon = float(line[7 + nlev])

                # Check if data exists in first nlev_new layers
                has_data = any(w > 0 for w in wth[:nlev_new])
                if has_data:
                    paths_data.append({
                        'ix': ix, 'iy': iy, 'jx': jx, 'jy': jy,
                        'length': length, 'elev': elev,
                        'wth': wth[:nlev_new], 'lat': lat, 'lon': lon
                    })

        npth_new = len(paths_data)
        print(f"New paths (with data in first {nlev_new} layers): {npth_new}")

        # Write bifprm.txt
        bifprm_file = os.path.join(map_dir, 'bifprm.txt')
        print(f"\nWriting bifurcation parameters to: {bifprm_file}")

        with open(bifprm_file, 'w') as f:
            # Write header
            f.write(f"{npth_new:8d}{nlev_new:8d}  npath_new, nlev_new, (ix,iy), (jx,jy), length, elevtn, depth, (width1, width2, ... wodth_nlev), (lat,lon), (basins)\n")

            # Process each path
            for path in paths_data:
                ix, iy = path['ix'], path['iy']
                jx, jy = path['jx'], path['jy']
                length = path['length']
                elev = path['elev']
                wth = path['wth']
                lat, lon = path['lat'], path['lon']

                # Calculate bifurcation depth
                if wth[0] <= 0:
                    dph = -9999.0
                    wth[0] = 0.0
                else:
                    # Depth formula: dph = log10(width) * 2.5 - 4.0
                    dph = np.log10(wth[0]) * 2.5 - 4.0
                    dph = max(0.5, dph)  # Minimum 0.5m

                    # Limit by nearby channel depth
                    dph0 = max(rivhgt[ix, iy], rivhgt[jx, jy])
                    dph = min(dph, dph0)

                    # Store in arrays
                    bifdph[ix, iy] = dph
                    bifwth[ix, iy] = wth[0]

                # Get basin IDs
                ibsn = basin[ix, iy]
                jbsn = basin[jx, jy]
                if ibsn > jbsn:
                    ibsn, jbsn = jbsn, ibsn

                # Write to file (convert back to 1-based indexing)
                line = f"{ix+1:8d}{iy+1:8d}{jx+1:8d}{jy+1:8d}"
                line += f"{length:12.2f}{elev:12.2f}{dph:12.2f}"
                for w in wth:
                    line += f"{w:12.2f}"
                line += f"{lat:10.3f}{lon:10.3f}{ibsn:8d}{jbsn:8d}\n"
                f.write(line)

        # Write bifdph.bin
        bifdph_file = os.path.join(map_dir, 'bifdph.bin')
        print(f"\nWriting bifurcation depth map to: {bifdph_file}")
        FortranBinary.write_direct(bifdph_file, bifdph, rec=1)
        FortranBinary.write_direct(bifdph_file, bifwth, rec=2)

        print("\nSET_BIFPARAM completed successfully!")
        print(f"Generated files:")
        print(f"  - bifprm.txt ({npth_new} paths, {nlev_new} layers)")
        print(f"  - bifdph.bin (2 records: depth and width)")

    def allocate_dams(self, map_dir):
        """
        Allocate dams from a list to the CaMa-Flood river network.
        Equivalent to src_param/allocate_dam.F90
        """
        print("=" * 60)
        print("ALLOCATE_DAMS: Allocating dams on CaMa-Flood river network")
        print("=" * 60)

        dam_input_file = self.config.get('RiverMap_Gen', 'dam_input_file', '../../data/GRanD_allocated.csv')
        if not os.path.exists(dam_input_file):
            print(f"ERROR: Dam input file not found: {dam_input_file}")
            return

        print(f"Reading dam list from: {dam_input_file}")

        # Read CaMa-Flood map data
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']
        gsize = params['gsize']
        west, north, east, south = params['west'], params['north'], params['east'], params['south']

        uparea = FortranBinary.read_direct(os.path.join(map_dir, 'uparea.bin'), (nx, ny), 'real', rec=1)
        ctmare = FortranBinary.read_direct(os.path.join(map_dir, 'ctmare.bin'), (nx, ny), 'real', rec=1)
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        nexty = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=2)

        # Convert to km2
        uparea[uparea > 0] *= 1.e-6
        ctmare[ctmare > 0] *= 1.e-6

        # Detect and read high-resolution map data
        hires_tag = self.config.get('PRMWAT', 'hires_tag', '1min') # Get from PRMWAT section for now
        hires_dir = os.path.join(map_dir, hires_tag)
        location_file = os.path.join(hires_dir, 'location.txt')

        if not os.path.exists(location_file):
            print(f"  ERROR no high resolution data available at {location_file}")
            return

        print(f"  Hires map found: {hires_tag}")

        with open(location_file, 'r') as f:
            f.readline() # skip narea
            f.readline() # skip header
            parts = f.readline().split()
            # IMPORTANT: Use float32 to match Fortran precision
            west2, east2, south2, north2 = [np.float32(p) for p in parts[2:6]]
            nx_hires, ny_hires = int(parts[6]), int(parts[7])
            # Calculate csize using float32 to match Fortran: csize=dble(east2-west2)/dble(nx)
            # In Fortran, this is computed in double but stored as real (float32)
            csize = np.float32((east2 - west2) / nx_hires)

        print(f"  Hires map params: nx={nx_hires}, ny={ny_hires}, csize={csize}")

        # Read arrays in Fortran order: shape (nx, ny), access as [ix, iy], order='F' ensures column-major
        upa_hires = FortranBinary.read_direct(os.path.join(hires_dir, f'{hires_tag}.uparea.bin'), (nx_hires, ny_hires), 'real', rec=1)
        catmx_hires = FortranBinary.read_direct(os.path.join(hires_dir, f'{hires_tag}.catmxy.bin'), (nx_hires, ny_hires), 'int2', rec=1)
        catmy_hires = FortranBinary.read_direct(os.path.join(hires_dir, f'{hires_tag}.catmxy.bin'), (nx_hires, ny_hires), 'int2', rec=2)
        downx_hires = FortranBinary.read_direct(os.path.join(hires_dir, f'{hires_tag}.downxy.bin'), (nx_hires, ny_hires), 'int2', rec=1)
        downy_hires = FortranBinary.read_direct(os.path.join(hires_dir, f'{hires_tag}.downxy.bin'), (nx_hires, ny_hires), 'int2', rec=2)

        # Pre-calculate upstream grid list
        print("  Calculating upstream grid list...")
        nsta = 8
        upstXX = np.full((nx, ny, nsta), -9999, dtype=np.int32, order='F')
        upstYY = np.full((nx, ny, nsta), -9999, dtype=np.int32, order='F')
        maxupa = np.zeros((nx, ny), dtype=np.float32, order='F')

        for ista in range(nsta):
            maxupa[:, :] = 0
            for iy in range(ny):
                for ix in range(nx):
                    if nextx[ix, iy] > 0:
                        jx, jy = nextx[ix, iy] - 1, nexty[ix, iy] - 1
                        is_duplicate = False
                        if ista >= 1:
                            for jsta in range(ista):
                                if ix == upstXX[jx, jy, jsta] and iy == upstYY[jx, jy, jsta]:
                                    is_duplicate = True
                                    break
                        if not is_duplicate:
                            if uparea[ix, iy] > maxupa[jx, jy]:
                                maxupa[jx, jy] = uparea[ix, iy]
                                upstXX[jx, jy, ista] = ix
                                upstYY[jx, jy, ista] = iy

        # Pre-calculate outlet pixel location of each unit catchment
        print("  Calculating outlet pixel locations...")
        outx = np.full((nx, ny), -9999, dtype=np.int32, order='F')
        outy = np.full((nx, ny), -9999, dtype=np.int32, order='F')
        maxupa[:, :] = 0

        for iy in range(ny_hires):
            for ix in range(nx_hires):
                if catmx_hires[ix, iy] > 0:
                    iXX, iYY = catmx_hires[ix, iy] - 1, catmy_hires[ix, iy] - 1
                    if downx_hires[ix, iy] <= -900: # River mouth
                        outx[iXX, iYY] = ix
                        outy[iXX, iYY] = iy
                    else:
                        # Using a simplified nextxy logic for python
                        jx, jy = ix + downx_hires[ix, iy], iy + downy_hires[ix, iy]
                        if 0 <= jx < nx_hires and 0 <= jy < ny_hires:
                            if catmx_hires[jx, jy] != catmx_hires[ix, iy] or catmy_hires[jx, jy] != catmy_hires[ix, iy]:
                                if outx[iXX, iYY] != -9999: # Multiple outlets
                                    if upa_hires[jx, jy] > maxupa[iXX, iYY]:
                                        maxupa[iXX, iYY] = upa_hires[jx, jy]
                                        outx[iXX, iYY] = ix
                                        outy[iXX, iYY] = iy
                                else:
                                    outx[iXX, iYY] = ix
                                    outy[iXX, iYY] = iy

        # Main loop - process dams
        print("  Processing dam list...")
        dam_out_river = os.path.join(map_dir, 'GRanD_river.txt')
        dam_out_small = os.path.join(map_dir, 'GRanD_small.txt')
        dam_out_error = os.path.join(map_dir, 'GRanD_error.txt')

        with open(dam_input_file, 'r') as f_in, \
             open(dam_out_river, 'w') as f_river, \
             open(dam_out_small, 'w') as f_small, \
             open(dam_out_error, 'w') as f_error:

            # Write headers
            header = '        ID       lat       lon   area_CaMa  area_Input       error        diff      ix      iy     cap_mcm  year  damname                         rivname\n'
            f_river.write(header)
            f_small.write(header)
            f_error.write(header)

            f_in.readline() # Skip header

            for line in f_in:
                parts = line.strip().split(',')
                dam_id, lat0, lon0, area0 = int(parts[0]), np.float32(parts[1]), np.float32(parts[2]), np.float32(parts[3])
                dam_name, riv_name = parts[4], parts[5]
                cap_mcm, year = np.float32(parts[6]), int(parts[7])

                if not (west < lon0 < east and south < lat0 < north and cap_mcm >= 0):
                    continue

                # Use float32 arithmetic to match Fortran: ix=int((lon0-west2)/csize)+1
                # Note: Fortran uses 1-indexed, Python uses 0-indexed
                ix = int(np.float32(lon0 - west2) / csize) + 1  # 1-indexed (Fortran style)
                iy = int(np.float32(north2 - lat0) / csize) + 1  # 1-indexed (Fortran style)

                err0, err1, rate0 = np.float32(1.e20), np.float32(1.e20), np.float32(1.e20)
                kx, ky = -1, -1
                area = np.float32(-1.0)
                nn = 3 # Search window size

                # Search for best fit in high-res map
                # IMPORTANT: Use dx outer loop, dy inner loop to match Fortran order
                for dx in range(-nn, nn + 1):
                    for dy in range(-nn, nn + 1):
                        jx_1idx = ix + dx  # 1-indexed for Fortran compatibility
                        jy_1idx = iy + dy  # 1-indexed for Fortran compatibility

                        # Convert to 0-indexed for Python array access
                        jx = jx_1idx - 1
                        jy = jy_1idx - 1

                        if not (0 <= jx < nx_hires and 0 <= jy < ny_hires):
                            continue

                        if upa_hires[jx, jy] > area0 * 0.05:
                            err = (upa_hires[jx, jy] - area0) / area0
                            # Calculate dd using 1-indexed coordinates (Fortran style)
                            dd = np.sqrt(np.float32((jx_1idx - ix)**2 + (jy_1idx - iy)**2))
                            err2 = err + np.float32(0.02) * dd if err > 0 else err - np.float32(0.02) * dd

                            if err2 >= 0:
                                rate = 1 + err2
                            elif -1 < err2 < 0:
                                rate = 1. / (1 + err2)
                                rate = min(rate, np.float32(1000.))
                            else:
                                rate = np.float32(1000.)

                            if rate < rate0:
                                err0, err1, rate0 = err2, err, rate
                                kx, ky = jx, jy  # Store as 0-indexed for array access
                                area = upa_hires[kx, ky]

                # If dam cannot be allocated
                if err0 == 1.e20 or area0 <= 0:
                    # Use :30.30s to truncate names to 30 chars (Fortran a30 behavior)
                    f_error.write(f'{dam_id:10d}{lat0:10.3f}{lon0:10.3f}{-999.:12.1f}{area0:12.1f}{-999.:12.1f}{-999.:12.1f}{-999:8d}{-999:8d}{cap_mcm:12.1f}{year:6d}  {dam_name:30.30s}  {riv_name:30.30s}\n')
                    continue

                # Find best grid for the gauge
                ix0, iy0 = kx, ky
                iXX0, iYY0 = catmx_hires[ix0, iy0] - 1, catmy_hires[ix0, iy0] - 1

                if iXX0 < 0 or iYY0 < 0:
                    f_error.write(f'{dam_id:10d}{lat0:10.3f}{lon0:10.3f}{-999.:12.1f}{area0:12.1f}{-999.:12.1f}{-999.:12.1f}{-999:8d}{-999:8d}{cap_mcm:12.1f}{year:6d}  {dam_name:30.30s}  {riv_name:30.30s}\n')
                    continue

                # Handle small dams
                if area0 < ctmare[iXX0, iYY0] * 0.3:
                    area_cmf, diff, err1 = np.float32(-888.0), np.float32(-888.0), np.float32(-8.0)
                    staX, staY = iXX0, iYY0
                    ifile = f_small
                else:
                    staX, staY = iXX0, iYY0
                    area_cmf = uparea[staX, staY]
                    diff = area_cmf - area0
                    err1 = diff / area0 if area0 != 0 else np.float32(1.e20)
                    snum = 0

                    # Upstream refinement
                    if abs(err1) > 0.2:
                        for ista in range(nsta):
                            jXX, jYY = upstXX[iXX0, iYY0, ista], upstYY[iXX0, iYY0, ista]
                            if jXX < 0: break

                            ox, oy = outx[jXX, jYY], outy[jXX, jYY]

                            # nextxy logic with longitude wrap-around
                            px, py = ox + downx_hires[ox, oy], oy + downy_hires[ox, oy]
                            if px < 0:
                                px += nx_hires
                            elif px >= nx_hires:
                                px -= nx_hires

                            is_found = False
                            while catmx_hires[px, py] - 1 == iXX0 and catmy_hires[px, py] - 1 == iYY0 and downx_hires[px, py] > -900:
                                if px == ix0 and py == iy0:
                                    is_found = True
                                    break
                                # nextxy logic with longitude wrap-around
                                kx_new, ky_new = px + downx_hires[px, py], py + downy_hires[px, py]
                                px, py = kx_new, ky_new
                                if px < 0:
                                    px += nx_hires
                                elif px >= nx_hires:
                                    px -= nx_hires

                            if is_found:
                                if uparea[jXX, jYY] < area0 * 0.1: continue
                                area2 = uparea[jXX, jYY]
                                diff2 = area2 - area0
                                err2 = diff2 / area0 if area0 != 0 else np.float32(1.e20)

                                if abs(err2) < abs(err1):
                                    err1 = err2
                                    snum = 1
                                    staX, staY = jXX, jYY
                                    area_cmf = area2
                                    if abs(err1) < 0.1: break

                    if snum == 0:
                        area_cmf = uparea[iXX0, iYY0]
                    
                    diff = area_cmf - area0
                    err1 = diff / area0 if area0 != 0 else np.float32(1.e20)

                    if err1 > 1:
                        staX, staY = iXX0, iYY0
                        area_cmf, diff, err1 = np.float32(-888.0), np.float32(-888.0), np.float32(-8.0)
                        ifile = f_small
                    else:
                        ifile = f_river

                # Write output (use :30.30s to truncate names to 30 chars, matching Fortran a30)
                ifile.write(f'{dam_id:10d}{lat0:10.3f}{lon0:10.3f}{area_cmf:12.1f}{area0:12.1f}{err1:12.1f}{diff:12.1f}{staX+1:8d}{staY+1:8d}{cap_mcm:12.1f}{year:6d}  {dam_name:30.30s}  {riv_name:30.30s}\n')
