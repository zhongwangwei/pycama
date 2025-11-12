# PyCaMa - Python Implementation of CaMa-Flood

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A Python implementation of the CaMa-Flood (Catchment-based Macro-scale Floodplain) hydrodynamic model, providing an efficient and flexible framework for large-scale river routing and flood inundation simulations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

PyCaMa is a Python reimplementation of the CaMa-Flood model, originally written in Fortran. It simulates river discharge, water level, and flood inundation at continental to global scales using a local inertial equation approach. The model uses a unit-catchment discretization scheme that efficiently represents river networks and floodplain dynamics.

**Key Differences from Original CaMa-Flood:**

- **Language**: Pure Python implementation (original is Fortran 90)
- **Precision**: Float64 throughout for numerical stability
- **Performance**: ~50% slower than Fortran (ongoing optimization)
- **Accuracy**: ~1% difference compared to Fortran version

## Features

### Core Capabilities

- ✅ **River Network Generation**: Extract regional river networks from global maps
- ✅ **NetCDF Initialization**: Convert binary river network data to CF-compliant NetCDF
- ✅ **Flood Routing Simulation**: Local inertial equation with adaptive time stepping
- ✅ **Bifurcation Support**: Simulate river bifurcations and distributaries
- ✅ **Floodplain Dynamics**: Optional floodplain inundation and storage
- ✅ **Flexible Restart**: Support for hourly/daily/monthly/yearly restart frequencies
- ✅ **Multi-format I/O**: NetCDF and binary file support
- ✅ **Runoff Interpolation**: Conservative downscaling from coarse input grids

### Physics Options

- Local inertial approximation for river flow
- Kinematic wave option for faster computation
- Adaptive sub-time stepping for numerical stability
- Manning's roughness parameterization
- Channel-floodplain interaction
- Water surface elevation calculation

### Experimental/Incomplete Features

- ⚠️ **Dam Operation**: Code implemented but not fully tested
- ⚠️ **Debug Tracer**: Runtime cell tracing (not production-ready)
- ⚠️ **Groundwater Delay**: Feature present but requires validation

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- netCDF4-python
- (Optional) xarray for data analysis

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/pycama.git
cd pycama
```

2. **Install dependencies:**

```bash
pip install numpy scipy netCDF4
```

Or using conda:

```bash
conda install numpy scipy netcdf4
```

3. **Verify installation:**

```bash
python src/main.py --help
```

### Optional: Extract Test Data

**Only needed if you want to run the included test case:**

```bash
# Extract the initialization file for the test simulation
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

**Note:** If you're generating your own river network (using `--grid` and `--init` options), you don't need to extract this file.

## Quick Start

### Running the Test Case

A complete test case is included in the repository with a 3-day simulation (1980-01-01 to 1980-01-03).

**First, extract the test data** (see [Optional: Extract Test Data](#optional-extract-test-data) above):

```bash
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

**Then run the simulation:**

```bash
# Run only the model simulation using pre-generated data
python src/main.py nml/namelist-15min.input --run-only
```

**Expected output:**
- River discharge, water depth, and storage in `output/Global15min/model_output/`
- Output files organized by month: `Global15min_198001.nc`

### Basic Usage Patterns

```bash
# Generate regional river network only
python src/main.py nml/namelist.input --grid-only

# Convert to NetCDF (initialization)
python src/main.py nml/namelist.input --init-only

# Run flood simulation
python src/main.py nml/namelist.input --run-only

# Run specific combinations
python src/main.py nml/namelist.input --grid --init
```

## Workflow

PyCaMa consists of three sequential workflows:

```
┌─────────────────────────────────────────────────────────────┐
│  1. RIVER NETWORK GENERATION                                │
│  Input:  Global river map (binary)                          │
│  Output: Regional network → output/{case}/rivermap/         │
│          - nextxy.bin, params.txt, diminfo.txt, etc.        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. INITIALIZATION                                          │
│  Input:  Binary river network files                         │
│  Output: NetCDF file → output/{case}/Initialization/        │
│          - grid_routing_data.nc (CF-compliant)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. MODEL SIMULATION                                        │
│  Input:  NetCDF initialization + forcing data (runoff)      │
│  Output: Time series → output/{case}/model_output/          │
│          - discharge, water depth, flood extent             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

All configuration is done through Fortran-style namelist files (e.g., `nml/namelist-15min.input`).

### Key Configuration Sections

#### 1. Options (Control which workflows to run)

```fortran
&OPTIONS
  run_grid = .false.    ! River network generation
  run_init = .false.    ! Initialization to NetCDF
  run_sim  = .true.     ! Model simulation
/
```

#### 2. Output Settings

```fortran
&OUTPUT
  output_base_dir = 'output/'
  case_name = 'Global15min'  ! All outputs go to output/Global15min/
/
```

#### 3. River Network Generation

```fortran
&RiverMap_Gen
  global_map_dir = '/path/to/global/map/'
  west = -180.0    ! Domain boundaries
  east = 180.0
  south = -90.0
  north = 90.0

  run_inpmat = .true.        ! Generate input matrix
  run_params = .true.        ! Calculate parameters
  run_bifurcation = .true.   ! Process bifurcations
  run_dam = .false.          ! Dam allocation (experimental)
/
```

#### 4. Model Physics

```fortran
&MODEL_RUN
  ! Time settings
  syear = 1980
  smon  = 1
  sday  = 1
  eyear = 1980
  emon  = 1
  eday  = 3

  dt = 3600                  ! Time step [seconds]
  ifrq_inp = 24              ! Forcing frequency [hours]
  ifrq_out = 24              ! Output frequency [hours]

  ! Physics options
  ladpstp  = .true.          ! Adaptive time stepping
  lpthout  = .true.          ! Bifurcation scheme
  ldamout  = .false.         ! Dam operation
  lfplain  = .true.          ! Floodplain scheme
  lkine    = .false.         ! Kinematic wave (vs local inertial)

  ! Forcing data
  linpcdf  = .true.
  crofdir_nc = './data/GRFR_0p25/'
  crofpre_nc = 'RUNOFF_remap_sel_'
  crofsuf_nc = '.nc'
  forcing_file_freq = 'yearly'  ! 'single', 'yearly', 'monthly', 'daily'
/
```

#### 5. Restart Configuration

```fortran
&MODEL_RUN
  lrestart = .false.         ! Start from restart file
  creststo = ''              ! Input restart file

  ! Restart frequency
  ifrq_rst = 1               ! Restart every 1 unit
  cfrq_rst_unit = 'month'    ! Unit: 'hour', 'day', 'month', 'year'

  lrestcdf = .true.          ! NetCDF format (vs binary)

  ! Default restart directory: output/{case_name}/restart/
/
```

## Performance

### Computational Performance

| Metric | PyCaMa | Fortran CaMa-Flood | Ratio |
|--------|--------|-------------------|-------|
| **Speed** | ~50% slower | Baseline | 0.5x |
| **Memory** | Float64 precision | Float32/64 mixed | ~1.3x |
| **Accuracy** | ~1% difference | Baseline | 99% |

**Performance Notes:**

- Python implementation uses float64 throughout for numerical stability
- Adaptive time stepping overhead in Python is higher than Fortran
- Future optimizations planned:
  - Numba JIT compilation for critical loops
  - Sparse matrix operations for large domains
  - Parallel processing for independent river reaches

### Benchmark (Global 15-min resolution, 3-day simulation)

```
Domain: 1440 x 720 grid cells
Active cells: ~250,000
Time steps:  
Wall clock time:  
```

## Known Limitations

### Incomplete Features

1. **Dam Operation** (`ldamout = .true.`)
   - Implementation complete but not validated
   - Dam allocation works, but release calculation needs testing
   - **Status**: Use at your own risk

2. **Debug Tracer** (`src/model_run/trace_debug.py`)
   - Runtime debugging tool for cell-by-cell tracing
   - Not intended for production runs
   - **Status**: Development tool only

3. **Groundwater Delay** (`lgdwdly`)
   - Code present but requires validation
   - **Status**: Experimental

### Known Issues

1. **Numerical Differences**
   - ~1% deviation from Fortran version
   - Mainly due to:
     - Float64 vs mixed precision
     - Different compiler optimizations
     - Rounding in intermediate calculations

2. **Performance**
   - Currently ~50% slower than Fortran
   - Most overhead in adaptive time stepping and I/O
   - Optimization ongoing

3. **Memory Usage**
   - Higher memory footprint due to float64
   - Sequence-based arrays not yet fully optimized

### Platform-Specific Notes

- **macOS**: Tested and working
- **Linux**: Should work (not extensively tested)
- **Windows**: May require path adjustments

## Project Structure

```
pycama/
├── src/
│   ├── main.py                      # Unified entry point
│   ├── river_network/               # Network generation
│   │   ├── workflow.py              # Orchestration
│   │   ├── region_tools.py          # Domain cutting
│   │   ├── param_tools.py           # Parameter calculation
│   │   ├── dam_param_tools.py       # Dam processing
│   │   └── namelist.py              # Config parser
│   ├── initialization/              # NetCDF conversion
│   │   └── grid_routing_init.py
│   └── model_run/                   # Simulation engine
│       ├── runner.py                # Main orchestrator
│       ├── physics.py               # Hydrodynamic equations
│       ├── forcing.py               # Input data handling
│       ├── output.py                # NetCDF output
│       ├── time_control.py          # Time stepping
│       ├── restart.py               # Restart I/O
│       └── dam_operation.py         # Dam release (experimental)
├── nml/
│   └── namelist-15min.input         # Test configuration
├── output/
│   └── Global15min/                 # Test case output
│       ├── rivermap/                # Generated network
│       ├── Initialization/          # NetCDF initialization
│       │   └── grid_routing_data.nc.zip  # Extract this!
│       └── model_output/            # Simulation results
├── data/                            # Input data (not in repo)
├── CLAUDE.md                        # AI assistant guidance
└── README.md                        # This file
```

## Testing

### Running the Test Case

The repository includes a complete test case for global 15-minute resolution:

```bash
# 1. Extract initialization file (REQUIRED)
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..

# 2. Run 3-day simulation
python src/main.py nml/namelist-15min.input --run-only

# 3. Check results
ls output/Global15min/model_output/
# Expected: Global15min_198001.nc
```

### Verifying Results

```python
import netCDF4 as nc

# Open output file
ds = nc.Dataset('output/Global15min/model_output/Global15min_198001.nc')

# Check variables
print(ds.variables.keys())
# Expected: 'rivout', 'rivsto', 'rivdph', 'outflw', etc.

# View discharge time series
discharge = ds.variables['rivout'][:]  # Shape: (time, y, x)
print(f"Discharge range: {discharge.min():.2f} - {discharge.max():.2f} m³/s")
```

### Creating Your Own Case

1. Prepare global river map (or use provided data)
2. Create namelist file with your domain
3. Run network generation: `--grid-only`
4. Run initialization: `--init-only`
5. Prepare forcing data (runoff)
6. Run simulation: `--run-only`

## Contributing

Contributions are welcome! Areas for improvement:

- **Performance optimization**: Numba, Cython, or parallel processing
- **Validation**: More test cases and benchmarks
- **Dam operation**: Testing and validation
- **Documentation**: More examples and tutorials
- **Testing**: Unit tests and continuous integration

Please see [CLAUDE.md](CLAUDE.md) for development guidelines.

## Citation

If you use PyCaMa in your research, please cite:

**Original CaMa-Flood Model:**
```
Yamazaki, D., Kanae, S., Kim, H., & Oki, T. (2011).
A physically based description of floodplain inundation dynamics in a global river routing model.
Water Resources Research, 47(4), W04501. doi:10.1029/2010WR009726
```


## Acknowledgments

- Original CaMa-Flood model by Prof. Dai Yamazaki (University of Tokyo)
- Fortran codebase: http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback:
- **Issues**: [GitHub Issues](https://github.com/yourusername/pycama/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pycama/discussions)

## Version History

- **v0.1.0** (Current) - Initial Python implementation
  - Core functionality working
  - ~1% accuracy vs Fortran
  - ~50% slower performance
  - Dam and tracer features incomplete

---

**Note**: This is a research-grade implementation. Please validate results for your specific application before use in production or publication.
