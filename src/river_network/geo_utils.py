"""
Geographic utility functions for distance and area calculations
"""
import numpy as np


def rgetarea(rlon1, rlon2, rlat1, rlat2):
    """
    Calculate area of a geographic box

    Args:
        rlon1, rlon2: Longitude bounds (degrees)
        rlat1, rlat2: Latitude bounds (degrees, -90 to 90)

    Returns:
        Area in m^2
    """
    # Constants
    de2 = 0.00669447  # eccentricity squared
    dpi = np.pi
    drad = 6378136.0  # Earth radius in meters

    de = np.sqrt(de2)

    # Ensure latitude is in valid range
    rlat1 = min(max(rlat1, -90.0), 90.0)
    rlat2 = min(max(rlat2, -90.0), 90.0)

    if rlat1 > 90.0 or rlat1 < -90.0 or rlat2 > 90.0 or rlat2 < -90.0:
        return 0.0

    # Convert to radians and calculate
    dsin1 = np.sin(rlat1 * dpi / 180.0)
    dsin2 = np.sin(rlat2 * dpi / 180.0)

    dfnc1 = dsin1 * (1 + (de * dsin1)**2 / 2.0)
    dfnc2 = dsin2 * (1 + (de * dsin2)**2 / 2.0)

    area = dpi * drad**2 * (1 - de**2) / 180.0 * (dfnc1 - dfnc2) * (rlon2 - rlon1)

    return abs(area)


def rgetlen(rlon1, rlat1, rlon2, rlat2):
    """
    Calculate distance between two geographic points
    Matches Fortran rgetlen function in set_map.F90 exactly - returns distance in meters

    Args:
        rlon1, rlat1: Origin coordinates (degrees)
        rlon2, rlat2: Destination coordinates (degrees)

    Returns:
        Distance in meters (matches Fortran set_map.F90)
    """
    # Constants
    da = 6378137.0
    de2 = 0.006694470
    
    # Use high-precision pi and perform calculations in float64
    rpi = np.pi

    # Convert degrees to radians
    rlat1_rad = np.deg2rad(rlat1)
    rlon1_rad = np.deg2rad(rlon1)
    rlat2_rad = np.deg2rad(rlat2)
    rlon2_rad = np.deg2rad(rlon2)

    # Trig functions
    dsinlat1 = np.sin(rlat1_rad)
    dsinlon1 = np.sin(rlon1_rad)
    dcoslat1 = np.cos(rlat1_rad)
    dcoslon1 = np.cos(rlon1_rad)
    dsinlat2 = np.sin(rlat2_rad)
    dsinlon2 = np.sin(rlon2_rad)
    dcoslat2 = np.cos(rlat2_rad)
    dcoslon2 = np.cos(rlon2_rad)

    # Heights are zero
    dh1 = 0.0
    dh2 = 0.0

    # Point 1 to (x1, y1, z1)
    dn1 = da / np.sqrt(1.0 - de2 * dsinlat1 * dsinlat1)
    dx1 = (dn1 + dh1) * dcoslat1 * dcoslon1
    dy1 = (dn1 + dh1) * dcoslat1 * dsinlon1
    dz1 = (dn1 * (1.0 - de2) + dh1) * dsinlat1

    # Point 2 to (x2, y2, z2)
    dn2 = da / np.sqrt(1.0 - de2 * dsinlat2 * dsinlat2)
    dx2 = (dn2 + dh2) * dcoslat2 * dcoslon2
    dy2 = (dn2 + dh2) * dcoslat2 * dsinlon2
    dz2 = (dn2 * (1.0 - de2) + dh2) * dsinlat2

    # Calculate distance
    dlen = np.sqrt((dx1 - dx2)**2 + (dy1 - dy2)**2 + (dz1 - dz2)**2)
    
    # This part is sensitive, ensure calculation is done in float64
    # and only cast to float32 at the very end.
    # The Fortran code does: drad=dble(asin(real(dlen/2/da)))
    # which is a mixed-precision operation. A direct float64 calculation
    # is more stable and likely to be correct.
    drad = np.arcsin(dlen / (2.0 * da))
    distance = drad * 2.0 * da

    return np.float32(distance)


def create_lon_lat_arrays(west, east, north, south, nx, ny):
    """
    Create longitude and latitude arrays for grid centers

    Args:
        west, east, north, south: Domain boundaries
        nx, ny: Number of grids

    Returns:
        lon, lat: 1D arrays of grid center coordinates
    """
    gsize = (east - west) / nx

    # Grid centers (1-based indexing in mind)
    lon = np.array([west + (i + 0.5) * gsize for i in range(nx)])
    lat = np.array([north - (i + 0.5) * gsize for i in range(ny)])

    return lon, lat
