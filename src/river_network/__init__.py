"""
River Network Generation Package
Python implementation of CaMa-Flood river network generation tools
"""
from .config import Config, create_default_config
from .fortran_io import FortranBinary, read_params_txt, write_params_txt
from .geo_utils import rgetarea, rgetlen, create_lon_lat_arrays
from .region_tools import RegionProcessor
from .param_tools import ParamProcessor

__version__ = '1.0.0'
__all__ = [
    'Config',
    'create_default_config',
    'FortranBinary',
    'read_params_txt',
    'write_params_txt',
    'rgetarea',
    'rgetlen',
    'create_lon_lat_arrays',
    'RegionProcessor',
    'ParamProcessor',
]
