# Quick Start Guide

This guide will help you run your first PyCaMa simulation in 5 minutes.

## Prerequisites

Make sure you have Python 3.8+ installed:

```bash
python --version
# Should show Python 3.8.0 or higher
```

## Step 1: Install Dependencies

```bash
# Install required packages
pip install numpy scipy netCDF4

# Or using conda
conda install numpy scipy netcdf4
```

## Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/pycama.git
cd pycama
```

## Step 2.5: Extract Test Data

**This step is ONLY needed to run the included test case:**

```bash
# Extract the pre-generated initialization file
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

You should now see `grid_routing_data.nc` in the Initialization folder.

> **Note:** If you plan to generate your own river network from scratch (using the full workflow with `--grid` and `--init`), you can skip this extraction step.

## Step 3: Run Test Simulation

```bash
# Run a 3-day simulation (1980-01-01 to 1980-01-03)
python src/main.py nml/namelist-15min.input --run-only
```

**Expected output:**

```
======================================================================
CaMa-Flood Grid Routing System
River Network Generation | Initialization | Model Run
======================================================================

Reading configuration file: .../nml/namelist-15min.input

Configuration information:
  Case name: Global15min
  ...
  Run model simulation: True

======================================================================
Function 3: Model Run
======================================================================

======================================================================
CaMa-Flood Model Runner Initialization
======================================================================

1. Reading model configuration...
  Simulation period: 1980/01/01 00:00 to 1980/01/03 00:00
  Time step: 3600 seconds
  ...

Starting Model Simulation
...
Simulation Complete
```

## Step 4: Check Results

```bash
# Output is in model_output directory
ls output/Global15min/model_output/

# Should show: Global15min_198001.nc
```

## Step 5: Analyze Results

Use Python to view the results:

```python
import netCDF4 as nc
import numpy as np

# Open output file
ds = nc.Dataset('output/Global15min/model_output/Global15min_198001.nc', 'r')

# List variables
print("Available variables:", list(ds.variables.keys()))
# Output: ['time', 'lat', 'lon', 'rivout', 'rivsto', 'rivdph', ...]

# Get discharge data
discharge = ds.variables['rivout'][:]  # Shape: (time, lat, lon)
print(f"Discharge shape: {discharge.shape}")
print(f"Discharge range: {np.nanmin(discharge):.2f} to {np.nanmax(discharge):.2f} m³/s")

# Get time
time = ds.variables['time'][:]
print(f"Time steps: {len(time)}")

# Close file
ds.close()
```

## What Just Happened?

You just ran a global river routing simulation that:

1. **Loaded** pre-generated river network data (15-minute resolution)
2. **Read** forcing data (runoff) for Jan 1-3, 1980
3. **Simulated** river discharge and water levels
4. **Saved** results to NetCDF file

## Next Steps

### Modify Simulation Period

Edit `nml/namelist-15min.input`:

```fortran
&MODEL_RUN
  syear = 1980
  smon  = 1
  sday  = 1
  eyear = 1980
  emon  = 1
  eday  = 7    ! Change to 7 for a week-long simulation
/
```

Then re-run:

```bash
python src/main.py nml/namelist-15min.input --run-only
```

### Change Output Variables

Edit the `cvarsout` parameter:

```fortran
&MODEL_RUN
  cvarsout = 'outflw,rivout,rivsto,rivdph,fldsto,flddph'
  # Add more variables: storge, sfcelv, fldout, etc.
/
```

### Customize Output Frequency

```fortran
&MODEL_RUN
  ifrq_out = 6   ! Output every 6 hours (default: 24)
/
```

### Enable Restart Files

```fortran
&MODEL_RUN
  ifrq_rst = 1
  cfrq_rst_unit = 'day'  ! Save restart every day
  lrestcdf = .true.      ! NetCDF format
/
```

Restart files will be saved to `output/Global15min/restart/`

## Troubleshooting

### Problem: "grid_routing_data.nc not found"

**Solution:** This file is needed only for the test case. Make sure you extracted the zip file:

```bash
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

**Note:** If you're generating your own river network (not using the test case), you won't need this file. Instead, run the complete workflow:

```bash
python src/main.py nml/your-namelist.input --grid --init --run
```

### Problem: "Forcing file not found"

**Solution:** Check forcing file path in namelist:

```fortran
&MODEL_RUN
  crofdir_nc = './data/GRFR_0p25/'   # Check this path exists
  crofpre_nc = 'RUNOFF_remap_sel_'
/
```

### Problem: "Import error: No module named netCDF4"

**Solution:** Install dependencies:

```bash
pip install netCDF4
```

### Problem: Simulation is slow

**Expected:** PyCaMa is ~50% slower than Fortran version. For the 3-day test:
- Expected time: ~5-15 minutes (depends on CPU)
- Longer for larger domains or longer simulations

**Tips for faster runs:**
- Use kinematic wave: `lkine = .true.` (less accurate)
- Disable adaptive time stepping: `ladpstp = .false.`
- Increase time step: `dt = 7200` (2 hours)

## Common Tasks

### Run Complete Workflow (Network + Init + Simulation)

```bash
# All three steps
python src/main.py nml/namelist.input

# Note: This requires global map data (not included in test case)
```

### Run Specific Combinations

```bash
# Network generation + initialization
python src/main.py nml/namelist.input --grid --init

# Initialization + simulation
python src/main.py nml/namelist.input --init --run
```

### View Namelist Options

```bash
python src/main.py --help
```

## Learning More

- **README.md**: Full documentation
- **CLAUDE.md**: Code architecture for developers
- **nml/namelist-15min.input**: Example configuration with comments

## Getting Help

- Check existing [GitHub Issues](https://github.com/yourusername/pycama/issues)
- Open a new issue with:
  - Error message
  - Your namelist configuration
  - Python version and OS

---

**Congratulations!** You've successfully run your first PyCaMa simulation! 🎉
