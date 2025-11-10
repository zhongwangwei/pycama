"""
Configuration file parser for grid routing initialization
"""
import configparser
import os


class Config:
    """Configuration manager for grid routing initialization"""

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read(config_file)

    def get(self, section, key, default=None):
        """Get configuration value"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def getint(self, section, key, default=None):
        """Get integer value"""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    def getfloat(self, section, key, default=None):
        """Get float value"""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    def getboolean(self, section, key, default=None):
        """Get boolean value"""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    @property
    def global_map_dir(self):
        """Global map directory"""
        return self.get('PATHS', 'global_map_dir', '../../glb_15min_natural/')

    @property
    def output_dir(self):
        """Output directory"""
        return self.get('PATHS', 'output_dir', './output/')

    @property
    def runoff_file(self):
        """Runoff climatology file"""
        return self.get('PATHS', 'runoff_file', '../data/runoff.nc')

    @property
    def runoff_var(self):
        """Runoff variable name in netCDF"""
        return self.get('RUNOFF', 'variable_name', 'ro')

    # Regional domain
    @property
    def west(self):
        return self.getfloat('DOMAIN', 'west', 60.0)

    @property
    def east(self):
        return self.getfloat('DOMAIN', 'east', 150.0)

    @property
    def south(self):
        return self.getfloat('DOMAIN', 'south', 5.0)

    @property
    def north(self):
        return self.getfloat('DOMAIN', 'north', 55.0)

    # Input data
    @property
    def input_gsize(self):
        return self.getfloat('INPUT', 'grid_size', 0.25)

    @property
    def input_west(self):
        return self.getfloat('INPUT', 'west', 60.0)

    @property
    def input_east(self):
        return self.getfloat('INPUT', 'east', 150.0)

    @property
    def input_north(self):
        return self.getfloat('INPUT', 'north', 55.0)

    @property
    def input_south(self):
        return self.getfloat('INPUT', 'south', 5.0)

    @property
    def input_lat_order(self):
        return self.get('INPUT', 'lat_order', 'NtoS')

    # High-resolution data
    @property
    def hires_tag(self):
        return self.get('HIRES', 'tag', '1min')

    # River channel parameters
    @property
    def channel_depth_coef(self):
        return self.getfloat('CHANNEL', 'depth_coef', 0.1)

    @property
    def channel_depth_power(self):
        return self.getfloat('CHANNEL', 'depth_power', 0.5)

    @property
    def channel_depth_offset(self):
        return self.getfloat('CHANNEL', 'depth_offset', 0.0)

    @property
    def channel_depth_min(self):
        return self.getfloat('CHANNEL', 'depth_min', 1.0)

    @property
    def channel_width_coef(self):
        return self.getfloat('CHANNEL', 'width_coef', 2.5)

    @property
    def channel_width_power(self):
        return self.getfloat('CHANNEL', 'width_power', 0.6)

    @property
    def channel_width_offset(self):
        return self.getfloat('CHANNEL', 'width_offset', 0.0)

    @property
    def channel_width_min(self):
        return self.getfloat('CHANNEL', 'width_min', 5.0)

    # Dam allocation
    @property
    def run_dam_allocation(self):
        return self.getboolean('DAM', 'run_dam_allocation', False)

    @property
    def dam_input_file(self):
        return self.get('DAM', 'dam_input_file', '../../data/GRanD_allocated.csv')

    # Dam parameter calculation
    @property
    def run_dam_param_calculation(self):
        return self.getboolean('DAM_PARAMS', 'run_dam_param_calculation', False)

    @property
    def dam_syear(self):
        return self.getint('DAM_PARAMS', 'syear', 1980)

    @property
    def dam_eyear(self):
        return self.getint('DAM_PARAMS', 'eyear', 2019)

    @property
    def dam_dt(self):
        return self.getint('DAM_PARAMS', 'dt', 86400)

    @property
    def dam_min_uparea(self):
        return self.getint('DAM_PARAMS', 'min_uparea', 0)

    @property
    def dam_natsim_dir(self):
        return self.get('DAM_PARAMS', 'natsim_dir', '../../data/GRFR-china_15min/')

    @property
    def dam_grand_river_file(self):
        return self.get('DAM_PARAMS', 'grand_river_file', '../output_python/GRanD_river.txt')

    @property
    def dam_grsad_dir(self):
        return self.get('DAM_PARAMS', 'grsad_dir', '../../data/Dam+Lake/GRSAD/GRSAD_timeseries/')

    @property
    def dam_regeom_dir(self):
        return self.get('DAM_PARAMS', 'regeom_dir', '../../data/Dam+Lake/ReGeom/Area_Strg_Dp/Area_Strg_Dp/')


def create_default_config(filename):
    """Create a default configuration file"""
    config_content = """# Grid Routing Initialization Configuration File
# All paths are relative to the working directory

[PATHS]
# Global map directory (must contain params.txt and binary files)
global_map_dir = ../../glb_15min_natural/

# Output directory for regional map
output_dir = ./output/

# Runoff climatology file (netCDF)
runoff_file = /Users/zhongwangwei/Desktop/Github/CoLM2024_grid_routing_init/data/runoff/runoff_climc_025_China_mm_s.nc

[DOMAIN]
# Regional domain boundaries (degrees)
west = 60.0
east = 150.0
south = 5.0
north = 55.0

[INPUT]
# Input runoff grid specifications
grid_size = 0.25
west = 60.0
east = 150.0
north = 55.0
south = 5.0
lat_order = NtoS  # NtoS or StoN

[HIRES]
# High-resolution data tag (1min, 30sec, 15sec, or 3sec)
tag = 1min

[RUNOFF]
# Runoff data specifications
variable_name = ro

[CHANNEL]
# River channel geometry parameters
# Depth: H = max(depth_min, depth_coef * Q^depth_power + depth_offset)
depth_coef = 0.1
depth_power = 0.50
depth_offset = 0.00
depth_min = 1.0

# Width: W = max(width_min, width_coef * Q^width_power + width_offset)
width_coef = 2.50
width_power = 0.60
width_offset = 0.00
width_min = 5.0

[DAM]
# Dam allocation settings
run_dam_allocation = false
dam_input_file = ../../data/GRanD_allocated.csv

[DAM_PARAMS]
# Dam parameter calculation settings
run_dam_param_calculation = false
syear = 1980
eyear = 2019
dt = 86400
min_uparea = 0
natsim_dir = ../../data/GRFR-china_15min/
grand_river_file = ../output_python/GRanD_river.txt
grsad_dir = ../../data/Dam+Lake/GRSAD/GRSAD_timeseries/
regeom_dir = ../../data/Dam+Lake/ReGeom/Area_Strg_Dp/Area_Strg_Dp/
"""

    with open(filename, 'w') as f:
        f.write(config_content)

    print(f"Default configuration created: {filename}")
