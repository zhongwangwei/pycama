"""
Region processing tools - equivalent to src_region
"""
import numpy as np
import os
from .fortran_io import FortranBinary, read_params_txt, write_params_txt
from .geo_utils import create_lon_lat_arrays, rgetlen, rgetarea


def heap_sort2(nmax, a, b):
    """
    Heap sort implementation that exactly matches Fortran's heap_sort2
    Sorts array a in descending order and carries along array b

    Args:
        nmax: Number of elements
        a: Array to sort (modified in place) - basin sizes
        b: Array to carry along (modified in place) - basin indices

    Note: Uses 1-based indexing to match Fortran exactly
    """
    # Convert to 1-based indexing by prepending dummy element
    a = np.concatenate([[0], a])
    b = np.concatenate([[0], b])

    # Phase 1: Build heap
    i = int(nmax / 2)
    n = i

    while True:
        if n == 0:
            break

        if 2 * n > nmax:
            i = i - 1
            n = i
            continue
        else:
            if 2 * n + 1 > nmax:
                if a[2*n] > a[n]:
                    c = a[n]
                    a[n] = a[2*n]
                    a[2*n] = c
                    d = b[n]
                    b[n] = b[2*n]
                    b[2*n] = d
                    n = 2 * n
                    continue
                else:
                    i = i - 1
                    n = i
                    continue
            else:
                if a[n] >= a[2*n] and a[n] >= a[2*n+1]:
                    i = i - 1
                    n = i
                    continue
                elif a[2*n] > a[2*n+1]:
                    c = a[n]
                    a[n] = a[2*n]
                    a[2*n] = c
                    d = b[n]
                    b[n] = b[2*n]
                    b[2*n] = d
                    n = 2 * n
                    continue
                else:
                    c = a[n]
                    a[n] = a[2*n+1]
                    a[2*n+1] = c
                    d = b[n]
                    b[n] = b[2*n+1]
                    b[2*n+1] = d
                    n = 2 * n + 1
                    continue

    # Phase 2: Sort
    for n in range(1, nmax + 1):
        c = a[1]
        a[1] = a[nmax - n + 1]
        a[nmax - n + 1] = c
        d = b[1]
        b[1] = b[nmax - n + 1]
        b[nmax - n + 1] = d

        i = 1
        mod = 1

        while mod == 1:
            mod = 0
            if 2 * i <= nmax - n:
                if 2 * i + 1 <= nmax - n:
                    if a[2*i] > a[i] and a[2*i] >= a[2*i+1]:
                        c = a[i]
                        a[i] = a[2*i]
                        a[2*i] = c
                        d = b[i]
                        b[i] = b[2*i]
                        b[2*i] = d
                        i = 2 * i
                        mod = 1
                    elif a[2*i+1] > a[i] and a[2*i+1] > a[2*i]:
                        c = a[i]
                        a[i] = a[2*i+1]
                        a[2*i+1] = c
                        d = b[i]
                        b[i] = b[2*i+1]
                        b[2*i+1] = d
                        i = 2 * i + 1
                        mod = 1
                else:
                    if a[2*i] > a[i]:
                        c = a[i]
                        a[i] = a[2*i]
                        a[2*i] = c
                        d = b[i]
                        b[i] = b[2*i]
                        b[2*i] = d
                        i = 2 * i
                        mod = 1

    # Convert back to 0-based indexing
    return a[1:], b[1:]


class RegionProcessor:
    """Handles regional map generation from global maps"""

    def __init__(self, config):
        self.config = config
        self.imis = -9999  # Integer missing value
        self.rmis = 1.e20  # Real missing value

    def cut_domain(self, global_dir, west, east, north, south, output_dir):
        """
        Cut regional domain from global map

        Args:
            global_dir: Path to global map directory
            west, east, north, south: Regional boundaries
            output_dir: Output directory for regional map
        """
        print("=" * 60)
        print("CUT_DOMAIN: Cutting regional domain from global map")
        print("=" * 60)

        # Read global params
        global_params_file = os.path.join(global_dir, 'params.txt')
        if os.path.exists(global_params_file):
            global_params = read_params_txt(global_params_file)
            nx_global = global_params['nx']
            ny_global = global_params['ny']
            nflp = global_params['nflp']
            gsize = global_params['gsize']
            lon_ori = global_params['west']
            lat_ori = global_params['north']
        else:
            # Assume standard global 15min map if params.txt doesn't exist
            print("Note: params.txt not found, using standard global 15min parameters")
            nx_global = 1440
            ny_global = 720
            nflp = 10
            gsize = 0.25  # 15 min = 0.25 degrees
            lon_ori = -180.0
            lat_ori = 90.0

        print(f"Global map size: {nx_global} x {ny_global}")
        print(f"  (Read from: {global_params_file})")
        print(f"Input boundaries: W={west}, E={east}, S={south}, N={north}")

        # Find optimum boundaries aligned with global grid
        west_opt, east_opt = self._find_optimal_lon(west, east, lon_ori, gsize, nx_global)
        north_opt, south_opt = self._find_optimal_lat(north, south, lat_ori, gsize, ny_global)

        print(f"Adjusted boundaries: W={west_opt}, E={east_opt}, S={south_opt}, N={north_opt}")

        # Calculate regional grid size
        mx = int(round((east_opt - west_opt) / gsize))
        my = int(round((north_opt - south_opt) / gsize))
        dx = int(round((west_opt - lon_ori) / gsize))
        dy = int(round((lat_ori - north_opt) / gsize))

        print(f"Regional grid size: {mx} x {my}")
        print(f"Offset from global: dx={dx}, dy={dy}")
        print(f"DEBUG: nx_global={nx_global}, ny_global={ny_global}, mx={mx}, my={my}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Read global maps
        print("\nReading global maps...")
        nextx0 = FortranBinary.read_direct(os.path.join(global_dir, 'nextxy.bin'),
                                           (nx_global, ny_global), 'int4', rec=1)
        nexty0 = FortranBinary.read_direct(os.path.join(global_dir, 'nextxy.bin'),
                                           (nx_global, ny_global), 'int4', rec=2)

        downx0 = FortranBinary.read_direct(os.path.join(global_dir, 'downxy.bin'),
                                           (nx_global, ny_global), 'int4', rec=1)
        downy0 = FortranBinary.read_direct(os.path.join(global_dir, 'downxy.bin'),
                                           (nx_global, ny_global), 'int4', rec=2)

        elevtn0 = FortranBinary.read_direct(os.path.join(global_dir, 'elevtn.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        ctmare0 = FortranBinary.read_direct(os.path.join(global_dir, 'ctmare.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        grdare0 = FortranBinary.read_direct(os.path.join(global_dir, 'grdare.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        uparea0 = FortranBinary.read_direct(os.path.join(global_dir, 'uparea.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        lon0 = FortranBinary.read_direct(os.path.join(global_dir, 'lonlat.bin'),
                                         (nx_global, ny_global), 'real', rec=1)
        lat0 = FortranBinary.read_direct(os.path.join(global_dir, 'lonlat.bin'),
                                         (nx_global, ny_global), 'real', rec=2)

        nxtdst0 = FortranBinary.read_direct(os.path.join(global_dir, 'nxtdst.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        rivlen0 = FortranBinary.read_direct(os.path.join(global_dir, 'rivlen.bin'),
                                            (nx_global, ny_global), 'real', rec=1)

        width0 = FortranBinary.read_direct(os.path.join(global_dir, 'width.bin'),
                                           (nx_global, ny_global), 'real', rec=1)

        # Read floodplain heights
        fldhgt0 = np.zeros((nx_global, ny_global, nflp), dtype=np.float32, order='F')
        for iflp in range(nflp):
            fldhgt0[:, :, iflp] = FortranBinary.read_direct(
                os.path.join(global_dir, 'fldhgt.bin'),
                (nx_global, ny_global), 'real', rec=iflp + 1)

        # Initialize regional arrays
        print("\nCutting domain...")
        nextx = np.full((mx, my), self.imis, dtype=np.int32, order='F')
        nexty = np.full((mx, my), self.imis, dtype=np.int32, order='F')
        # Initialize downx/downy to -9999 to match source global map
        downx = np.full((mx, my), self.imis, dtype=np.int32, order='F')
        downy = np.full((mx, my), self.imis, dtype=np.int32, order='F')
        elevtn = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        fldhgt = np.full((mx, my, nflp), self.imis, dtype=np.float32, order='F')
        ctmare = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        grdare = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        uparea = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        lon = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        lat = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        nxtdst = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        rivlen = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        width = np.full((mx, my), self.imis, dtype=np.float32, order='F')
        lsmask = np.zeros((mx, my), dtype=np.int32, order='F')

        # Cut domain (note: using 0-based indexing in Python)
        for iy in range(my):
            for ix in range(mx):
                jx = ix + dx
                jy = iy + dy

                if nextx0[jx, jy] > 0:
                    nextx[ix, iy] = nextx0[jx, jy] - dx
                    nexty[ix, iy] = nexty0[jx, jy] - dy
                    downx[ix, iy] = downx0[jx, jy]
                    downy[ix, iy] = downy0[jx, jy]

                    kx = nextx[ix, iy]
                    ky = nexty[ix, iy]

                    # Check if downstream is outside domain (convert to 0-based for comparison)
                    if kx < 1 or kx > mx or ky < 1 or ky > my:
                        nextx[ix, iy] = -10
                        nexty[ix, iy] = -10
                        downx[ix, iy] = -1000
                        downy[ix, iy] = -1000

                elif nextx0[jx, jy] != self.imis:  # River mouth
                    nextx[ix, iy] = nextx0[jx, jy]
                    nexty[ix, iy] = nexty0[jx, jy]
                    downx[ix, iy] = downx0[jx, jy]
                    downy[ix, iy] = downy0[jx, jy]

                if nextx[ix, iy] != self.imis:
                    elevtn[ix, iy] = elevtn0[jx, jy]
                    fldhgt[ix, iy, :] = fldhgt0[jx, jy, :]
                    ctmare[ix, iy] = ctmare0[jx, jy]
                    grdare[ix, iy] = grdare0[jx, jy]
                    uparea[ix, iy] = uparea0[jx, jy]
                    lon[ix, iy] = lon0[jx, jy]
                    lat[ix, iy] = lat0[jx, jy]
                    nxtdst[ix, iy] = nxtdst0[jx, jy]
                    rivlen[ix, iy] = rivlen0[jx, jy]
                    width[ix, iy] = width0[jx, jy]
                    lsmask[ix, iy] = 1

        # Write regional maps BEFORE masking edge rivers
        # (Fortran writes nextxy.bin before edge masking, and nextxy_noedge.bin after)
        print("\nWriting regional maps...")
        self._write_regional_maps(output_dir, mx, my, nflp, gsize,
                                 west_opt, east_opt, south_opt, north_opt,
                                 nextx, nexty, downx, downy, elevtn, fldhgt,
                                 ctmare, grdare, uparea, lon, lat,
                                 nxtdst, rivlen, width, lsmask)

        # Mask out-of-domain rivers
        print("\nMasking edge rivers...")
        nextx_noedge, nexty_noedge = self._mask_edge_rivers(nextx, nexty, mx, my)

        # Write nextxy_noedge.bin
        FortranBinary.write_direct(os.path.join(output_dir, 'nextxy_noedge.bin'), nextx_noedge, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'nextxy_noedge.bin'), nexty_noedge, rec=2)

        # Save dimension change info
        dim_change = {
            'global_dir': global_dir,
            'nx_global': nx_global,
            'ny_global': ny_global,
            'gsize': gsize,
            'lon_ori': lon_ori,
            'lon_end': lon_ori + nx_global * gsize,
            'lat_end': lat_ori - ny_global * gsize,
            'lat_ori': lat_ori,
            'mx': mx,
            'my': my,
            'dx': dx,
            'dy': dy,
            'west': west_opt,
            'east': east_opt,
            'south': south_opt,
            'north': north_opt
        }

        with open(os.path.join(output_dir, 'dim_change.txt'), 'w') as f:
            # IMPORTANT: The Fortran code writes global dimensions first, then regional.
            f.write(f"{dim_change['global_dir']}\n")
            f.write(f"{dim_change['nx_global']}\n")
            f.write(f"{dim_change['ny_global']}\n")
            f.write(f"{dim_change['gsize']}\n")
            f.write(f"{dim_change['lon_ori']}\n")
            f.write(f"{dim_change['lon_end']}\n")
            f.write(f"{dim_change['lat_end']}\n")
            f.write(f"{dim_change['lat_ori']}\n")
            f.write("../\n")
            f.write(f"{dim_change['mx']}\n")
            f.write(f"{dim_change['my']}\n")

            f.write(f"{dim_change['dx']}\n")
            f.write(f"{dim_change['dy']}\n")
            f.write(f"{dim_change['west']}\n")
            f.write(f"{dim_change['east']}\n")
            f.write(f"{dim_change['south']}\n")
            f.write(f"{dim_change['north']}\n")

        print("\nCUT_DOMAIN completed successfully!")
        return dim_change

    def _find_optimal_lon(self, west, east, lon_ori, gsize, nx):
        """Find optimal longitude boundaries aligned with grid"""
        west_opt = west
        east_opt = east

        d1 = 1.e20
        d2 = 1.e20

        for ix in range(nx):
            glon = lon_ori + ix * gsize
            if west >= glon - gsize * 0.1 and west < glon + gsize * 1.1:
                if abs(west - glon) < d1:
                    west_opt = glon
                    d1 = abs(west - glon)

            if east > glon - gsize * 0.1 and east <= glon + gsize * 1.1:
                if abs(east - glon - gsize) < d2:
                    east_opt = glon + gsize
                    d2 = abs(east - glon - gsize)

        if abs(west_opt - west) < 0.001:
            west_opt = west
        if abs(east_opt - east) < 0.001:
            east_opt = east

        return west_opt, east_opt

    def _find_optimal_lat(self, north, south, lat_ori, gsize, ny):
        """Find optimal latitude boundaries aligned with grid"""
        north_opt = north
        south_opt = south

        d1 = 1.e20
        d2 = 1.e20

        for iy in range(ny):
            glat = lat_ori - iy * gsize

            if north > glat - gsize * 1.1 and north <= glat + gsize * 0.1:
                if abs(north - glat) < d1:
                    north_opt = glat
                    d1 = abs(north - glat)

            if south >= glat - gsize * 1.1 and south < glat + gsize * 0.1:
                if abs(south - glat + gsize) < d2:
                    south_opt = glat - gsize
                    d2 = abs(south - glat + gsize)

        if abs(north_opt - north) < 0.001:
            north_opt = north
        if abs(south_opt - south) < 0.001:
            south_opt = south

        return north_opt, south_opt

    def _mask_edge_rivers(self, nextx, nexty, mx, my):
        """
        Mask edge rivers following Fortran cut_domain.F90 logic
        Equivalent to lines 387-451 in cut_domain.F90
        """
        check = np.full((mx, my), -9, dtype=np.int32, order='F')

        # Step 1: Initialize check array
        # check = 0 for valid river cells
        # check = 10 for outlets (-9) and out-of-domain (-10)
        n_valid_rivers = 0
        for iy in range(my):
            for ix in range(mx):
                if nextx[ix, iy] != self.imis:
                    check[ix, iy] = 0
                    n_valid_rivers += 1
                if nextx[ix, iy] == -9 or nextx[ix, iy] == -10:
                    check[ix, iy] = 10

        print(f"    Step 1: Found {n_valid_rivers} valid river cells")

        # Step 2: Mark edge rivers (within 2 cells of boundary)
        # Trace downstream from edge cells and mark entire path as 20
        n_edge_rivers = 0
        for iy in range(my):
            for ix in range(mx):
                if nextx[ix, iy] != self.imis:
                    # Check if near boundary (ix<=2 or ix>=mx-1 or iy<=2 or iy>=my-1)
                    # Fortran uses 1-based indexing: ix<=2, ix>=mx-1, iy<=2, iy>=my-1
                    # Python uses 0-based indexing: ix<=1, ix>=mx-2, iy<=1, iy>=my-2
                    if ix <= 1 or ix >= mx - 2 or iy <= 1 or iy >= my - 2:
                        jx = ix
                        jy = iy
                        # Trace downstream while check==0 and nextx>0
                        while check[jx, jy] == 0 and nextx[jx, jy] > 0:
                            check[jx, jy] = 20
                            n_edge_rivers += 1
                            kx = nextx[jx, jy] - 1  # Convert to 0-based
                            ky = nexty[jx, jy] - 1
                            jx = kx
                            jy = ky
                        # Unconditional assignment to match Fortran behavior (cut_domain.F90:411)
                        check[jx, jy] = 20
                        n_edge_rivers += 1

        print(f"    Step 2: Marked {n_edge_rivers} edge river cells")

        # Step 3: For remaining check==0 cells, inherit check value from downstream
        n_inherited = 0
        for iy in range(my):
            for ix in range(mx):
                if nextx[ix, iy] > 0 and check[ix, iy] == 0:
                    # Trace downstream until we hit a marked cell
                    jx = ix
                    jy = iy
                    while check[jx, jy] == 0:
                        kx = nextx[jx, jy] - 1
                        ky = nexty[jx, jy] - 1
                        jx = kx
                        jy = ky
                    icheck = check[jx, jy]

                    # Now go back and mark the entire upstream path
                    jx = ix
                    jy = iy
                    while check[jx, jy] == 0:
                        check[jx, jy] = icheck
                        if icheck == 20:
                            n_inherited += 1
                        kx = nextx[jx, jy] - 1
                        ky = nexty[jx, jy] - 1
                        jx = kx
                        jy = ky

        print(f"    Step 3: Inherited {n_inherited} additional edge river cells")

        # Step 4: Mask out cells marked as 20 (edge rivers)
        nextx_noedge = nextx.copy()
        nexty_noedge = nexty.copy()
        n_masked = 0

        for iy in range(my):
            for ix in range(mx):
                if check[ix, iy] == 20:
                    nextx_noedge[ix, iy] = self.imis
                    nexty_noedge[ix, iy] = self.imis
                    n_masked += 1

        print(f"    Step 4: Masked {n_masked} total edge river cells")
        print(f"    Final: {n_valid_rivers - n_masked} valid river cells remaining")

        return nextx_noedge, nexty_noedge

    def _write_regional_maps(self, output_dir, mx, my, nflp, gsize,
                            west, east, south, north,
                            nextx, nexty, downx, downy, elevtn, fldhgt,
                            ctmare, grdare, uparea, lon, lat,
                            nxtdst, rivlen, width, lsmask):
        """Write all regional map files"""
        # Write params.txt
        params = {
            'nx': mx,
            'ny': my,
            'nflp': nflp,
            'gsize': gsize,
            'west': west,
            'east': east,
            'south': south,
            'north': north
        }
        write_params_txt(os.path.join(output_dir, 'params.txt'), params)

        # Write binary files
        FortranBinary.write_direct(os.path.join(output_dir, 'nextxy.bin'), nextx, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'nextxy.bin'), nexty, rec=2)

        FortranBinary.write_direct(os.path.join(output_dir, 'downxy.bin'), downx, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'downxy.bin'), downy, rec=2)

        FortranBinary.write_direct(os.path.join(output_dir, 'elevtn.bin'), elevtn, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'ctmare.bin'), ctmare, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'grdare.bin'), grdare, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'uparea.bin'), uparea, rec=1)

        FortranBinary.write_direct(os.path.join(output_dir, 'lonlat.bin'), lon, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'lonlat.bin'), lat, rec=2)

        FortranBinary.write_direct(os.path.join(output_dir, 'nxtdst.bin'), nxtdst, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'rivlen.bin'), rivlen, rec=1)
        FortranBinary.write_direct(os.path.join(output_dir, 'width.bin'), width, rec=1)

        # Write floodplain heights
        for iflp in range(nflp):
            FortranBinary.write_direct(os.path.join(output_dir, 'fldhgt.bin'),
                                      fldhgt[:, :, iflp], rec=iflp + 1)

        FortranBinary.write_direct(os.path.join(output_dir, 'lsmask.bin'), lsmask, rec=1)

    def set_map(self, map_dir):
        """
        Set various maps - river sequence, upstream area, basin, etc.
        Equivalent to src_region/set_map.F90
        """
        print("=" * 60)
        print("SET_MAP: Generating river network derived maps")
        print("=" * 60)

        # Read parameters
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']
        gsize = params['gsize']
        west, east, north, south = params['west'], params['east'], params['north'], params['south']

        print(f"Map size: {nx} x {ny}")

        # Read next xy and grid area
        nextx = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=1)
        nexty = FortranBinary.read_direct(os.path.join(map_dir, 'nextxy.bin'), (nx, ny), 'int4', rec=2)
        grdare = FortranBinary.read_direct(os.path.join(map_dir, 'grdare.bin'), (nx, ny), 'real', rec=1)

        # Create lon/lat arrays
        lon = np.array([west + (ix + 0.5) * gsize for ix in range(nx)])
        lat = np.array([north - (iy + 0.5) * gsize for iy in range(ny)])

        # Calculate river sequence
        print("Calculating river sequence...")
        rivseq = self._calc_river_sequence(nextx, nexty, nx, ny)

        # Calculate upstream area and grid
        print("Calculating upstream area...")
        uparea, upgrid = self._calc_upstream_area(nextx, nexty, grdare, rivseq, nx, ny)

        # Calculate distance to next grid (matches Fortran set_map.F90)
        print("Calculating distance to next grid...")
        nxtdst = self._calc_next_distance(nextx, nexty, lon, lat, nx, ny)

        # Calculate basin
        print("Calculating basins...")
        basin, bsncol = self._calc_basin(nextx, nexty, rivseq, upgrid, nx, ny, west, east)

        # Write output files
        print("Writing output files...")
        FortranBinary.write_direct(os.path.join(map_dir, 'rivseq.bin'), rivseq, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'uparea_grid.bin'), uparea, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'upgrid.bin'), upgrid, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'nxtdst_grid.bin'), nxtdst, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'rivlen_grid.bin'), nxtdst, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'basin.bin'), basin, rec=1)
        FortranBinary.write_direct(os.path.join(map_dir, 'bsncol.bin'), bsncol, rec=1)

        print("SET_MAP completed successfully!")

    def _calc_river_sequence(self, nextx, nexty, nx, ny):
        """Calculate river sequence from upstream to downstream"""
        rivseq = np.ones((nx, ny), dtype=np.int32, order='F')

        # Initialize: 0 for downstream points, 1 for upstream heads
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] > 0:
                    jx = nextx[ix, iy] - 1  # Convert to 0-based
                    jy = nexty[ix, iy] - 1
                    rivseq[jx, jy] = 0
                elif nextx[ix, iy] == self.imis:
                    rivseq[ix, iy] = self.imis

        # Propagate sequence numbers downstream
        nseq = 2
        changed = True

        while changed:
            changed = False
            for iy in range(ny):
                for ix in range(nx):
                    if rivseq[ix, iy] == nseq - 1 and nextx[ix, iy] > 0:
                        jx = nextx[ix, iy] - 1
                        jy = nexty[ix, iy] - 1
                        if rivseq[jx, jy] < nseq:
                            rivseq[jx, jy] = nseq
                            changed = True
            nseq += 1

        print(f"  Maximum sequence: {nseq - 1}")
        return rivseq

    def _calc_upstream_area(self, nextx, nexty, grdare, rivseq, nx, ny):
        """Calculate upstream drainage area and number of upstream grids

        This follows the Fortran algorithm: trace from each source downstream,
        accumulating area as we go.
        """
        uparea = np.zeros((nx, ny), dtype=np.float32, order='F')
        upgrid = np.zeros((nx, ny), dtype=np.int32, order='F')

        # Initialize
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis:
                    uparea[ix, iy] = 0
                    upgrid[ix, iy] = 0
                else:
                    uparea[ix, iy] = self.imis
                    upgrid[ix, iy] = self.imis

        # Process each source (rivseq == 1) and trace downstream
        for iy in range(ny):
            for ix in range(nx):
                if rivseq[ix, iy] == 1:
                    # Start from source
                    jx, jy = ix, iy
                    uparea[jx, jy] = grdare[jx, jy]
                    upgrid[jx, jy] = 1
                    upa = uparea[jx, jy]
                    upg = upgrid[jx, jy]

                    # Trace downstream
                    while nextx[jx, jy] > 0:
                        # Move to next cell
                        kx = nextx[jx, jy] - 1  # Convert to 0-based
                        ky = nexty[jx, jy] - 1
                        jx, jy = kx, ky

                        # Check if this cell has been visited before
                        if uparea[jx, jy] == 0:
                            # First time visiting this cell
                            uparea[jx, jy] = upa + grdare[jx, jy]
                            upa = uparea[jx, jy]
                            upgrid[jx, jy] = upg + 1
                            upg = upgrid[jx, jy]
                        else:
                            # Cell already visited (tributary confluence)
                            uparea[jx, jy] = uparea[jx, jy] + upa
                            upgrid[jx, jy] = upgrid[jx, jy] + upg

        return uparea, upgrid

    def _calc_next_distance(self, nextx, nexty, lon, lat, nx, ny):
        """Calculate distance to next downstream grid"""
        nxtdst = np.full((nx, ny), self.imis, dtype=np.float32, order='F')

        dstmth = 25000.0  # Distance to mouth in meters

        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] > 0:
                    jx = nextx[ix, iy] - 1
                    jy = nexty[ix, iy] - 1

                    lon1 = lon[ix]
                    lon2 = lon[jx]
                    lat1 = lat[iy]
                    lat2 = lat[jy]

                    # Handle longitude wrap-around
                    if lon1 < -90 and lon2 > 270:
                        lon2 -= 360
                    if lon1 > 270 and lon2 < -90:
                        lon2 += 360

                    nxtdst[ix, iy] = rgetlen(lon1, lat1, lon2, lat2)  # Now returns meters directly

                elif nextx[ix, iy] != self.imis:
                    nxtdst[ix, iy] = dstmth

        return nxtdst

    def _calc_basin(self, nextx, nexty, rivseq, upgrid, nx, ny, west, east):
        """Calculate basin IDs and colors"""
        basin = np.zeros((nx, ny), dtype=np.int32, order='F')
        nbsn = 0

        # Assign basin IDs
        for iy in range(ny):
            for ix in range(nx):
                if rivseq[ix, iy] == 1:
                    nbsn += 1
                    jx, jy = ix, iy
                    basin[jx, jy] = nbsn

                    # Trace downstream
                    while nextx[jx, jy] > 0:
                        kx = nextx[jx, jy] - 1
                        ky = nexty[jx, jy] - 1

                        # Check bounds
                        if kx < 0 or kx >= nx or ky < 0 or ky >= ny:
                            break

                        if basin[kx, ky] != 0:
                            # Merge with existing basin
                            basin_this = basin[kx, ky]
                            nbsn -= 1
                            jx, jy = ix, iy
                            while basin[jx, jy] != basin_this:
                                basin[jx, jy] = basin_this
                                if nextx[jx, jy] > 0:
                                    kx_next = nextx[jx, jy] - 1
                                    ky_next = nexty[jx, jy] - 1
                                    if kx_next >= 0 and kx_next < nx and ky_next >= 0 and ky_next < ny:
                                        jx, jy = kx_next, ky_next
                                    else:
                                        break
                                else:
                                    break
                            break

                        basin[kx, ky] = nbsn
                        jx, jy = kx, ky

        # Renumber basins by size (largest = 1)
        basin = self._renumber_basins_by_size(basin, nextx, nexty, upgrid, nx, ny, nbsn)

        # Assign colors for visualization
        bsncol = self._assign_basin_colors(basin, nx, ny, west, east)

        print(f"  Number of basins: {nbsn}")
        return basin, bsncol

    def _renumber_basins_by_size(self, basin, nextx, nexty, upgrid, nx, ny, nbsn):
        """Renumber basins in order of size (largest first)"""
        basin_grid = np.zeros(nbsn, dtype=np.int32)
        basin_order = np.arange(1, nbsn + 1, dtype=np.int32)

        # Get basin sizes
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] < 0 and nextx[ix, iy] != self.imis:
                    bid = basin[ix, iy] - 1  # Convert to 0-based
                    if 0 <= bid < nbsn:
                        basin_grid[bid] = upgrid[ix, iy]

        # Sort by size using Fortran's heap_sort2 (sorts in ascending order)
        basin_grid_sorted, basin_order_sorted = heap_sort2(nbsn, basin_grid.copy(), basin_order.copy())

        # Create mapping from old basin ID to new basin ID
        # Fortran reverses the numbering: basin_new(basin_this)=nbsn_max-nbsn+1
        basin_new = np.zeros(nbsn, dtype=np.int32)
        for i in range(nbsn):
            basin_this = basin_order_sorted[i]  # 1-based Fortran index
            basin_new[basin_this - 1] = nbsn - i  # Reverse numbering (largest gets ID=1)

        # Renumber
        basin_out = basin.copy()
        for iy in range(ny):
            for ix in range(nx):
                if nextx[ix, iy] != self.imis and basin[ix, iy] > 0:
                    old_id = basin[ix, iy] - 1
                    if 0 <= old_id < nbsn:
                        basin_out[ix, iy] = basin_new[old_id]
                elif nextx[ix, iy] == self.imis:
                    basin_out[ix, iy] = self.imis

        return basin_out

    def _assign_basin_colors(self, basin, nx, ny, west, east):
        """
        Assign colors to basins for visualization using graph coloring
        Matches Fortran's algorithm: ensures adjacent basins have different colors
        """
        bsncol = np.full((nx, ny), self.imis, dtype=np.int32, order='F')

        # Get number of basins
        nbsn_max = int(np.max(basin[basin > 0]))

        # Calculate bounding box for each basin (bsn_mask)
        # bsn_mask[basin, 0:4] = [min_x, min_y, max_x, max_y]
        bsn_mask = np.zeros((nbsn_max, 4), dtype=np.int32)
        bsn_mask[:, 0] = 999999  # min_x
        bsn_mask[:, 1] = 999999  # min_y
        bsn_mask[:, 2] = -999999  # max_x
        bsn_mask[:, 3] = -999999  # max_y

        for iy in range(ny):
            for ix in range(nx):
                if basin[ix, iy] > 0:
                    nbsn = int(basin[ix, iy]) - 1  # Convert to 0-based
                    bsn_mask[nbsn, 0] = min(bsn_mask[nbsn, 0], ix + 1)  # 1-based
                    bsn_mask[nbsn, 1] = min(bsn_mask[nbsn, 1], iy + 1)
                    bsn_mask[nbsn, 2] = max(bsn_mask[nbsn, 2], ix + 1)
                    bsn_mask[nbsn, 3] = max(bsn_mask[nbsn, 3], iy + 1)

        # Expand bounding boxes by 1 grid
        for nbsn in range(nbsn_max):
            if bsn_mask[nbsn, 0] <= 1 or bsn_mask[nbsn, 2] >= nx:
                # Basin touches edge
                bsn_mask[nbsn, 0] = 1
                bsn_mask[nbsn, 1] = max(bsn_mask[nbsn, 1] - 1, 1)
                bsn_mask[nbsn, 2] = nx
                bsn_mask[nbsn, 3] = min(bsn_mask[nbsn, 3] + 1, ny)
            else:
                bsn_mask[nbsn, 0] = bsn_mask[nbsn, 0] - 1
                bsn_mask[nbsn, 1] = max(bsn_mask[nbsn, 1] - 1, 1)
                bsn_mask[nbsn, 2] = bsn_mask[nbsn, 2] + 1
                bsn_mask[nbsn, 3] = min(bsn_mask[nbsn, 3] + 1, ny)

        # Step 1: Color large basins (grid >= 20)
        for nbsn in range(1, nbsn_max + 1):
            col_used = [0] * 10  # Track which colors are used by neighbors
            grid = 0

            # Count grid cells and check neighbor colors
            for iy in range(bsn_mask[nbsn-1, 1] - 1, bsn_mask[nbsn-1, 3]):
                for ix in range(bsn_mask[nbsn-1, 0] - 1, bsn_mask[nbsn-1, 2]):
                    if basin[ix, iy] == nbsn:
                        grid += 1

                        # Check 4 neighbors
                        neighbors = [
                            (ix, iy + 1),
                            (ix + 1, iy),
                            (ix, iy - 1),
                            (ix - 1, iy)
                        ]

                        for jx, jy in neighbors:
                            # Handle global wrap-around for longitude
                            if east - west == 360:
                                if jx > nx - 1:
                                    jx = 0
                                elif jx < 0:
                                    jx = nx - 1

                            if 0 <= jx < nx and 0 <= jy < ny:
                                if bsncol[jx, jy] > 0:
                                    col_used[bsncol[jx, jy]] = 1

            # Assign color for large basin
            if grid >= 20:
                icol = 2
                while icol < 10 and col_used[icol] == 1:
                    icol += 1
                color_this = icol

                # Apply color to all cells in this basin
                for iy in range(bsn_mask[nbsn-1, 1] - 1, bsn_mask[nbsn-1, 3]):
                    for ix in range(bsn_mask[nbsn-1, 0] - 1, bsn_mask[nbsn-1, 2]):
                        if basin[ix, iy] == nbsn:
                            bsncol[ix, iy] = color_this

        # Step 2: Color small basins (grid < 20)
        for nbsn in range(1, nbsn_max + 1):
            col_used = [0] * 10
            grid = 0

            # Count grid cells and check neighbor colors
            for iy in range(bsn_mask[nbsn-1, 1] - 1, bsn_mask[nbsn-1, 3]):
                for ix in range(bsn_mask[nbsn-1, 0] - 1, bsn_mask[nbsn-1, 2]):
                    if basin[ix, iy] == nbsn:
                        grid += 1

                        # Check 4 neighbors
                        neighbors = [
                            (ix, iy + 1),
                            (ix + 1, iy),
                            (ix, iy - 1),
                            (ix - 1, iy)
                        ]

                        for jx, jy in neighbors:
                            # Handle global wrap-around
                            if east - west == 360:
                                if jx > nx - 1:
                                    jx = 0
                                elif jx < 0:
                                    jx = nx - 1

                            if 0 <= jx < nx and 0 <= jy < ny:
                                if bsncol[jx, jy] > 0:
                                    col_used[bsncol[jx, jy]] = 1

            # Assign color for small basin
            if grid < 20:
                icol = 2
                while icol < 10 and col_used[icol] == 1:
                    icol += 1
                color_this = icol

                # Special rule: single-cell basins use color 1
                if grid == 1:
                    color_this = 1

                # Apply color to all cells in this basin
                for iy in range(bsn_mask[nbsn-1, 1] - 1, bsn_mask[nbsn-1, 3]):
                    for ix in range(bsn_mask[nbsn-1, 0] - 1, bsn_mask[nbsn-1, 2]):
                        if basin[ix, iy] == nbsn:
                            bsncol[ix, iy] = color_this

        return bsncol

    def combine_hires(self, output_dir, global_map_dir, hires_tag='1min'):
        """
        Combine high-resolution data for regional domain
        Equivalent to src_region/combine_hires.F90

        Args:
            output_dir: Regional output directory
            global_map_dir: Global map directory containing high-resolution data
            hires_tag: High-resolution data tag (1min, 30sec, etc.)
        """
        print("=" * 60)
        print(f"COMBINE_HIRES: Processing {hires_tag} high-resolution data")
        print("=" * 60)

        # Read dimension change info
        dim_change_file = os.path.join(output_dir, 'dim_change.txt')
        if not os.path.exists(dim_change_file):
            raise FileNotFoundError(f"dim_change.txt not found: {dim_change_file}")

        with open(dim_change_file, 'r') as f:
            lines = f.readlines()
            global_dir = lines[0].strip()
            nXX = int(lines[1].strip())
            nYY = int(lines[2].strip())
            gsize = float(lines[3].strip())
            lon_ori = float(lines[4].strip())
            lon_end = float(lines[5].strip())
            lat_end = float(lines[6].strip())
            lat_ori = float(lines[7].strip())
            # lines[8] is "../"
            mXX = int(lines[9].strip())
            mYY = int(lines[10].strip())
            dXX = int(lines[11].strip())
            dYY = int(lines[12].strip())
            west = float(lines[13].strip())
            east = float(lines[14].strip())
            south = float(lines[15].strip())
            north = float(lines[16].strip())

        print(f"Regional domain: W={west}, E={east}, S={south}, N={north}")
        print(f"Regional grid: {mXX} x {mYY}")
        print(f"Domain offset: dXX={dXX}, dYY={dYY}")

        # Determine pixel size based on hires_tag
        tag_to_cnum = {
            '1min': 60,
            '30sec': 120,
            '15sec': 240,
            '5sec': 720,
            '3sec': 1200,
            '1sec': 3600
        }

        if hires_tag not in tag_to_cnum:
            raise ValueError(f"Unsupported hires_tag: {hires_tag}")

        cnum = tag_to_cnum[hires_tag]
        csize = 1.0 / cnum

        # Calculate regional hires dimensions
        nx = int((east - west) * cnum)
        ny = int((north - south) * cnum)

        print(f"High-res grid: {nx} x {ny}, pixel size: {csize:.6f} deg")

        # Determine if tiling is needed
        isTile = 0
        if hires_tag != '1min' and (nx > 16000 or ny > 16000):
            isTile = 1
            print("WARNING: Large domain detected, tiled processing not yet implemented")
            print("Proceeding with non-tiled processing...")

        # Process high-resolution data
        hires_dir = os.path.join(global_map_dir, hires_tag)
        out_hdir = os.path.join(output_dir, hires_tag)
        os.makedirs(out_hdir, exist_ok=True)

        # Read global location.txt
        list_loc = os.path.join(hires_dir, 'location.txt')
        if not os.path.exists(list_loc):
            raise FileNotFoundError(f"Global location.txt not found: {list_loc}")

        print(f"\nReading: {list_loc}")

        with open(list_loc, 'r') as f:
            narea = int(f.readline().split()[0])
            f.readline()  # Skip header

            # Initialize output arrays
            catmXX = np.full((nx, ny), -9999, dtype=np.int16, order='F')
            catmYY = np.full((nx, ny), -9999, dtype=np.int16, order='F')
            catmZZ = np.full((nx, ny), -9, dtype=np.int8, order='F')
            flddif = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            grdare = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            elevtn = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            hand = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            uparea = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            rivwth = np.full((nx, ny), -9999, dtype=np.float32, order='F')
            visual = np.full((nx, ny), -9, dtype=np.int8, order='F')
            downx = np.full((nx, ny), -9999, dtype=np.int16, order='F')
            downy = np.full((nx, ny), -9999, dtype=np.int16, order='F')
            flwdir = np.full((nx, ny), -9, dtype=np.int8, order='F')

            isDownXY = False
            isFdir = False

            print(f"\nProcessing {narea} areas...")
            for i in range(narea):
                parts = f.readline().split()
                area = parts[1]
                west0 = float(parts[2])
                east0 = float(parts[3])
                south0 = float(parts[4])
                north0 = float(parts[5])
                nx0 = int(parts[6])
                ny0 = int(parts[7])

                # Check if area overlaps with region
                if west0 > east or east0 < west or north0 < south or south0 > north:
                    print(f"  Skipping {area}: out of domain")
                    continue

                print(f"  Processing {area}: {nx0} x {ny0}")

                # Read input data
                catmxy_file = os.path.join(hires_dir, f'{area}.catmxy.bin')
                if not os.path.exists(catmxy_file):
                    print(f"    WARNING: {catmxy_file} not found, skipping")
                    continue

                catmXX0 = FortranBinary.read_direct(catmxy_file, (nx0, ny0), 'int2', rec=1)
                catmYY0 = FortranBinary.read_direct(catmxy_file, (nx0, ny0), 'int2', rec=2)

                # Read other data files
                catmzz_file = os.path.join(hires_dir, f'{area}.catmzz.bin')
                catmZZ0 = FortranBinary.read_direct(catmzz_file, (nx0, ny0), 'int1', rec=1) if os.path.exists(catmzz_file) else None

                flddif_file = os.path.join(hires_dir, f'{area}.flddif.bin')
                flddif0 = FortranBinary.read_direct(flddif_file, (nx0, ny0), 'real', rec=1) if os.path.exists(flddif_file) else None

                grdare_file = os.path.join(hires_dir, f'{area}.grdare.bin')
                grdare0 = FortranBinary.read_direct(grdare_file, (nx0, ny0), 'real', rec=1) if os.path.exists(grdare_file) else None

                elevtn_file = os.path.join(hires_dir, f'{area}.elevtn.bin')
                elevtn0 = FortranBinary.read_direct(elevtn_file, (nx0, ny0), 'real', rec=1) if os.path.exists(elevtn_file) else None

                hand_file = os.path.join(hires_dir, f'{area}.hand.bin')
                hand0 = FortranBinary.read_direct(hand_file, (nx0, ny0), 'real', rec=1) if os.path.exists(hand_file) else None

                uparea_file = os.path.join(hires_dir, f'{area}.uparea.bin')
                uparea0 = FortranBinary.read_direct(uparea_file, (nx0, ny0), 'real', rec=1) if os.path.exists(uparea_file) else None

                rivwth_file = os.path.join(hires_dir, f'{area}.rivwth.bin')
                rivwth0 = FortranBinary.read_direct(rivwth_file, (nx0, ny0), 'real', rec=1) if os.path.exists(rivwth_file) else None

                visual_file = os.path.join(hires_dir, f'{area}.visual.bin')
                visual0 = FortranBinary.read_direct(visual_file, (nx0, ny0), 'int1', rec=1) if os.path.exists(visual_file) else None

                # Check for downxy and flwdir
                downxy_file = os.path.join(hires_dir, f'{area}.downxy.bin')
                if os.path.exists(downxy_file):
                    downx0 = FortranBinary.read_direct(downxy_file, (nx0, ny0), 'int2', rec=1)
                    downy0 = FortranBinary.read_direct(downxy_file, (nx0, ny0), 'int2', rec=2)
                    isDownXY = True
                else:
                    downx0 = None
                    downy0 = None

                flwdir_file = os.path.join(hires_dir, f'{area}.flwdir.bin')
                if os.path.exists(flwdir_file):
                    flwdir0 = FortranBinary.read_direct(flwdir_file, (nx0, ny0), 'int1', rec=1)
                    isFdir = True
                else:
                    flwdir0 = None

                # Create coordinate arrays
                lon0 = np.array([west0 + (ix0 + 0.5) * csize for ix0 in range(nx0)])
                lat0 = np.array([north0 - (iy0 + 0.5) * csize for iy0 in range(ny0)])

                # Copy data to regional arrays
                for iy0 in range(ny0):
                    for ix0 in range(nx0):
                        if lon0[ix0] > west and lon0[ix0] < east and lat0[iy0] > south and lat0[iy0] < north:
                            ix = int((lon0[ix0] - west) / csize)
                            iy = int((north - lat0[iy0]) / csize)

                            if 0 <= ix < nx and 0 <= iy < ny:
                                if catmXX[ix, iy] == -9999:
                                    if catmXX0[ix0, iy0] > 0:
                                        catmXX[ix, iy] = catmXX0[ix0, iy0] - dXX
                                        catmYY[ix, iy] = catmYY0[ix0, iy0] - dYY

                                        # Check if within regional bounds
                                        if catmXX[ix, iy] < 1 or catmXX[ix, iy] > mXX or catmYY[ix, iy] < 1 or catmYY[ix, iy] > mYY:
                                            catmXX[ix, iy] = -999
                                            catmYY[ix, iy] = -999

                                        if catmZZ0 is not None:
                                            catmZZ[ix, iy] = catmZZ0[ix0, iy0]
                                    else:
                                        catmXX[ix, iy] = catmXX0[ix0, iy0]
                                        catmYY[ix, iy] = catmYY0[ix0, iy0]
                                        if catmZZ0 is not None:
                                            catmZZ[ix, iy] = catmZZ0[ix0, iy0]

                                # Copy other data
                                if downx0 is not None:
                                    downx[ix, iy] = downx0[ix0, iy0]
                                    downy[ix, iy] = downy0[ix0, iy0]
                                if flwdir0 is not None:
                                    flwdir[ix, iy] = flwdir0[ix0, iy0]
                                if visual0 is not None:
                                    visual[ix, iy] = visual0[ix0, iy0]
                                if flddif0 is not None:
                                    flddif[ix, iy] = flddif0[ix0, iy0]
                                if grdare0 is not None:
                                    grdare[ix, iy] = grdare0[ix0, iy0]
                                if hand0 is not None:
                                    hand[ix, iy] = hand0[ix0, iy0]
                                if elevtn0 is not None:
                                    elevtn[ix, iy] = elevtn0[ix0, iy0]
                                if uparea0 is not None:
                                    uparea[ix, iy] = uparea0[ix0, iy0]
                                if rivwth0 is not None:
                                    rivwth[ix, iy] = rivwth0[ix0, iy0]

        # Write output files
        print(f"\nWriting regional high-resolution data to: {out_hdir}")

        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.catmxy.bin'), catmXX, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.catmxy.bin'), catmYY, rec=2)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.catmzz.bin'), catmZZ, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.flddif.bin'), flddif, rec=1)

        if hires_tag != '3sec':
            FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.grdare.bin'), grdare, rec=1)

        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.hand.bin'), hand, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.elevtn.bin'), elevtn, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.uparea.bin'), uparea, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.rivwth.bin'), rivwth, rec=1)
        FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.visual.bin'), visual, rec=1)

        if isDownXY:
            FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.downxy.bin'), downx, rec=1)
            FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.downxy.bin'), downy, rec=2)

        if isFdir:
            FortranBinary.write_direct(os.path.join(out_hdir, f'{hires_tag}.flwdir.bin'), flwdir, rec=1)

        # Write location.txt
        location_file = os.path.join(out_hdir, 'location.txt')
        with open(location_file, 'w') as f:
            f.write(f"{1:6d}{'  narea':>10}\n")
            f.write(f"{'iarea':>6}{'carea':>10}{'west':>10}{'east':>10}{'south':>10}{'north':>10}{'nx':>8}{'ny':>8}{'csize':>20}\n")
            f.write(f"{1:6d}{hires_tag:>10}{west:10.3f}{east:10.3f}{south:10.3f}{north:10.3f}{nx:8d}{ny:8d}{csize:20.15f}\n")

        print(f"Created: {location_file}")
        print("\nCOMBINE_HIRES completed successfully!")

    def cut_bifway(self, global_map_dir, output_dir):
        """
        Cut bifurcation pathways from global map to regional domain
        Equivalent to src_region/cut_bifway.F90

        Args:
            global_map_dir: Global map directory containing bifori.txt
            output_dir: Regional output directory
        """
        print("=" * 60)
        print("CUT_BIFWAY: Cutting bifurcation pathways")
        print("=" * 60)

        # Read parameters
        params = read_params_txt(os.path.join(output_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']
        gsize = params['gsize']
        west, east = params['west'], params['east']
        south, north = params['south'], params['north']

        # Read global parameters (to calculate offset)
        global_params_file = os.path.join(global_map_dir, 'params.txt')
        if os.path.exists(global_params_file):
            global_params = read_params_txt(global_params_file)
            global_nx, global_ny = global_params['nx'], global_params['ny']
            global_west, global_north = global_params['west'], global_params['north']

            # Calculate domain offset
            dXX = int(round((west - global_west) / gsize))
            dYY = int(round((global_north - north) / gsize))
        else:
            # Assume global 15min map
            dXX = int(round((west - (-180.0)) / gsize))
            dYY = int(round((90.0 - north) / gsize))

        print(f"\nRegional domain: {nx} x {ny}")
        print(f"Domain offset: dXX={dXX}, dYY={dYY}")

        # Read nextxy_noedge to check valid grids
        nextx = FortranBinary.read_direct(
            os.path.join(output_dir, 'nextxy_noedge.bin'),
            (nx, ny), 'int4', rec=1
        )

        # Read global bifori.txt
        global_bifori = os.path.join(global_map_dir, 'bifori.txt')
        if not os.path.exists(global_bifori):
            print(f"WARNING: Global bifori.txt not found: {global_bifori}")
            print("Skipping bifurcation pathway cutting.")
            return

        print(f"\nReading global bifori: {global_bifori}")

        # First pass: count paths in region
        mpath = 0
        with open(global_bifori, 'r') as f:
            header = f.readline().split()
            npath = int(header[0])
            nlev = int(header[1])

            for ipath in range(npath):
                parts = f.readline().split()
                iXX = int(parts[0])
                iYY = int(parts[1])
                jXX = int(parts[2])
                jYY = int(parts[3])

                # Convert to regional coordinates
                iXX2 = iXX - dXX
                iYY2 = iYY - dYY
                jXX2 = jXX - dXX
                jYY2 = jYY - dYY

                # Check if both endpoints are in region
                # Note: Fortran uses iXX2<mXX (not <=), so we use < not <=
                if (0 < iXX2 < nx and 0 < iYY2 < ny and
                    0 < jXX2 < nx and 0 < jYY2 < ny):
                    # Convert to 0-based for Python array access
                    if (nextx[iXX2-1, iYY2-1] != self.imis and
                        nextx[jXX2-1, jYY2-1] != self.imis):
                        mpath += 1

        print(f"Global paths: {npath}")
        print(f"Regional paths: {mpath}")

        # Second pass: write regional bifori.txt
        regional_bifori = os.path.join(output_dir, 'bifori.txt')
        print(f"\nWriting regional bifori: {regional_bifori}")

        with open(global_bifori, 'r') as fin, open(regional_bifori, 'w') as fout:
            # Read and skip header
            header = fin.readline().split()
            npath = int(header[0])
            nlev = int(header[1])

            # Write new header
            fout.write(f"{mpath:8d}{nlev:8d}   npath, nlev, (ix,iy), (jx,jy), length, elevtn, (width1, width2, ... wodth_nlev), (lon,lat)\n")

            # Process each path
            for ipath in range(npath):
                line = fin.readline()
                parts = line.split()

                iXX = int(parts[0])
                iYY = int(parts[1])
                jXX = int(parts[2])
                jYY = int(parts[3])
                dst = float(parts[4])
                elv = float(parts[5])
                wth = [float(parts[6 + i]) for i in range(nlev)]
                lon = float(parts[6 + nlev])
                lat = float(parts[7 + nlev])

                # Convert to regional coordinates
                iXX2 = iXX - dXX
                iYY2 = iYY - dYY
                jXX2 = jXX - dXX
                jYY2 = jYY - dYY

                # Check if both endpoints are in region
                # Note: Fortran uses iXX2<mXX (not <=), so we use < not <=
                if (0 < iXX2 < nx and 0 < iYY2 < ny and
                    0 < jXX2 < nx and 0 < jYY2 < ny):
                    # Convert to 0-based for Python array access
                    if (nextx[iXX2-1, iYY2-1] != self.imis and
                        nextx[jXX2-1, jYY2-1] != self.imis):
                        # Write path with regional coordinates
                        fout.write(f"{iXX2:8d}{iYY2:8d}{jXX2:8d}{jYY2:8d}")
                        fout.write(f"{dst:12.2f}{elv:12.2f}")
                        for w in wth:
                            fout.write(f"{w:12.2f}")
                        fout.write(f"{lon:10.3f}{lat:10.3f}\n")

        print(f"\nCUT_BIFWAY completed successfully!")
        print(f"Generated bifori.txt with {mpath} paths")
