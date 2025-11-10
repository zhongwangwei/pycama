#!/usr/bin/env python3
"""
CaMa-Flood Grid Routing System - Main Entry Point
Unified entry script integrating three main functions: river network generation, initialization, and model run

Functional modules:
1. River Network Generation: Generate regional river network files from global map
2. Initialization: Convert river network data to NetCDF format for model use
3. Model Run: Run CaMa-Flood flood simulation

All configuration is done through namelist.input file
"""
import sys
import os
import argparse
import traceback

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'river_network'))
sys.path.insert(0, os.path.join(src_dir, 'initialization'))

# Import modules
from river_network.namelist import Namelist
from river_network.workflow import (
    run_region_generation,
    run_param_generation,
    execute_dam_allocation,
    execute_dam_param_calculation,
    print_header
)
from initialization.grid_routing_init import CoLMGridRoutingInit
from model_run import run_camaflood_model


def print_banner():
    """Print program banner"""
    print("=" * 70)
    print("CaMa-Flood Grid Routing System")
    print("River Network Generation | Initialization | Model Run")
    print("=" * 70)


def run_grid_routing_generation(nml, output_dir, case_name, global_map_dir):
    """
    Function 1: River Network Generation
    Generate regional river network files and parameter files from global map
    """
    print_header("Function 1: River Network Generation")
    
    # Get options from namelist
    run_inpmat = nml.get('RiverMap_Gen', 'run_inpmat', True)
    run_params = nml.get('RiverMap_Gen', 'run_params', True)
    run_prmwat = nml.get('RiverMap_Gen', 'run_prmwat', False)
    run_bifurcation = nml.get('RiverMap_Gen', 'run_bifurcation', False)
    run_dam = nml.get('RiverMap_Gen', 'run_dam', False)
    
    run_dam_allocation = run_dam
    run_dam_param_calculation = run_dam
    
    success = True
    
    # Step 1: Region generation
    if not run_region_generation(nml, global_map_dir, output_dir, case_name, 
                                  run_bifurcation, run_inpmat):
        success = False
    
    # Step 2: Parameter generation
    if run_params and success:
        if not run_param_generation(nml, output_dir, case_name, global_map_dir, 
                                     run_bifurcation, run_prmwat):
            success = False
    
    # Step 3: Dam allocation
    if run_dam_allocation and success:
        if not execute_dam_allocation(nml, output_dir):
            success = False
    
    # Step 4: Dam parameter calculation
    if run_dam_param_calculation and success:
        if not execute_dam_param_calculation(nml, output_dir):
            success = False
    
    return success


def run_initialization(nml, case_dir, case_name):
    """
    Function 2: Initialization
    Convert river network data to NetCDF format for model use
    """
    print_header("Function 2: Initialization")

    # River network files are in the rivermap subdirectory
    rivermap_dir = os.path.join(case_dir, 'rivermap')
    diminfo_file = os.path.join(rivermap_dir, f'diminfo_15min_{case_name}.txt')

    if not os.path.exists(diminfo_file):
        print(f"ERROR: Parameter file not found: {diminfo_file}")
        print("Please run grid routing generation first.")
        return False

    # Get output directory for NetCDF
    # Default: use Initialization subdirectory under the case directory
    default_init_dir = os.path.join(case_dir, 'Initialization')
    init_output_dir = nml.get('INIT', 'output_dir', default_init_dir)
    if not os.path.isabs(init_output_dir):
        init_output_dir = os.path.abspath(init_output_dir)
    os.makedirs(init_output_dir, exist_ok=True)

    print(f"River network directory: {rivermap_dir}")
    print(f"Parameter file: {os.path.basename(diminfo_file)}")
    print(f"Output directory: {init_output_dir}")

    try:
        # Initialize the export class
        init_app = CoLMGridRoutingInit()

        # Set paths manually (bypassing command line parsing)
        init_app.INPUT_DIR = rivermap_dir  # Read from rivermap subdirectory
        init_app.PARAM_FILE = os.path.basename(diminfo_file)
        init_app.OUTPUT_DIR = init_output_dir
        
        # Run initialization steps
        init_app.read_param_file()
        init_app.cmf_rivmap_init()
        
        # Check if bifurcation file exists
        bifprm_file = os.path.join(rivermap_dir, 'bifprm.txt')
        if os.path.exists(bifprm_file):
            init_app.read_bifurcation()
        else:
            print("Warning: bifprm.txt not found, skipping bifurcation reading")
            init_app.NPTHOUT = 0
            init_app.NPTHLEV = 0
        
        init_app.read_inpmat()
        init_app.read_dam_param()
        init_app.cmf_topo_init()
        init_app.export_to_netcdf()
        
        print("\n✓ Initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR in initialization: {e}")
        traceback.print_exc()
        return False


def run_model_simulation(nml, case_dir, case_name):
    """
    Function 3: Model Run
    Run CaMa-Flood flood simulation
    """
    print_header("Function 3: Model Run")

    # Get initialization netCDF file
    init_nc_file = None

    # Check if netCDF map input is enabled
    lmapcdf = nml.get('MODEL_RUN', 'lmapcdf', False)

    if lmapcdf:
        # First, try to use cinitnc from namelist if specified
        cinitnc = nml.get('MODEL_RUN', 'cinitnc', None)
        if cinitnc:
            # Make path absolute if relative
            if not os.path.isabs(cinitnc):
                cinitnc = os.path.abspath(cinitnc)

            if os.path.exists(cinitnc):
                init_nc_file = cinitnc
                print(f"Found initialization file from namelist: {init_nc_file}")
            else:
                print(f"WARNING: cinitnc specified but file not found: {cinitnc}")

        # If not found via cinitnc, try default initialization directory
        if not init_nc_file:
            default_init_dir = os.path.join(case_dir, 'Initialization')
            init_output_dir = nml.get('INIT', 'output_dir', default_init_dir)
            if not os.path.isabs(init_output_dir):
                init_output_dir = os.path.abspath(init_output_dir)

            if os.path.exists(init_output_dir):
                # Look for any netCDF file in the directory
                nc_files = [f for f in os.listdir(init_output_dir) if f.endswith('.nc')]
                if nc_files:
                    init_nc_file = os.path.join(init_output_dir, nc_files[0])
                    print(f"Found initialization file: {init_nc_file}")
                else:
                    print(f"WARNING: No initialization netCDF file found in {init_output_dir}")
    else:
        print("WARNING: lmapcdf is False - NetCDF map input is disabled")
        print("Set lmapcdf = .true. in namelist to enable NetCDF initialization")

    if not init_nc_file:
        print("WARNING: No initialization file found")
        print("Model will try to continue with available data")

    # Run CaMa-Flood model
    try:
        success = run_camaflood_model(nml, init_nc_file)
        if success:
            print("\n✓ Model simulation completed successfully!")
        return success
    except Exception as e:
        print(f"\n✗ ERROR in model simulation: {e}")
        traceback.print_exc()
        return False


def main():
    """Main program"""
    parser = argparse.ArgumentParser(
        description='CaMa-Flood Grid Routing System - Unified Entry Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Function descriptions:
  1. River Network Generation: Generate regional river network files and parameter files from global map
  2. Initialization: Convert river network data to NetCDF format for model use
  3. Model Run: Run CaMa-Flood flood simulation

Usage examples:
  # Run all functions (according to namelist configuration)
  python src/main.py nml/namelist.input

  # Run only river network generation
  python src/main.py nml/namelist.input --grid-only

  # Run only initialization
  python src/main.py nml/namelist.input --init-only

  # Run river network generation and initialization
  python src/main.py nml/namelist.input --grid --init
        """
    )

    parser.add_argument('namelist', nargs='?', default='../nml/namelist.input',
                        help='Namelist file path (default: ../nml/namelist.input)')
    parser.add_argument('--grid-only', action='store_true',
                        help='Run only river network generation')
    parser.add_argument('--init-only', action='store_true',
                        help='Run only initialization')
    parser.add_argument('--run-only', action='store_true',
                        help='Run only model simulation')
    parser.add_argument('--grid', action='store_true',
                        help='Run river network generation')
    parser.add_argument('--init', action='store_true',
                        help='Run initialization')
    parser.add_argument('--run', action='store_true',
                        help='Run model simulation')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Read namelist - resolve relative path
    namelist_path = args.namelist
    if not os.path.isabs(namelist_path):
        # Resolve relative to current script location (camflood/src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        camflood_dir = os.path.dirname(script_dir)  # camflood/
        namelist_path = os.path.join(camflood_dir, namelist_path)
        namelist_path = os.path.normpath(namelist_path)

    print(f"\nReading configuration file: {namelist_path}")
    try:
        nml = Namelist(namelist_path)
    except Exception as e:
        print(f"ERROR: Failed to read configuration file: {e}")
        return 1
    
    # Get output configuration
    output_base_dir = nml.get('OUTPUT', 'output_base_dir', './output')
    case_name = nml.get('OUTPUT', 'case_name', 'default')
    case_dir = os.path.join(output_base_dir, case_name)  # Base case directory
    rivermap_dir = os.path.join(case_dir, 'rivermap')     # River network output directory
    
    # Get global map directory from RiverMap_Gen
    global_map_dir = nml.get('RiverMap_Gen', 'global_map_dir')
    
    # Resolve relative paths (relative to camflood directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    camflood_dir = os.path.dirname(script_dir)  # camflood/
    project_root = os.path.dirname(camflood_dir)  # project root
    
    if not os.path.isabs(global_map_dir):
        global_map_dir = os.path.join(project_root, global_map_dir)
        global_map_dir = os.path.normpath(global_map_dir)
    if not os.path.isabs(case_dir):
        case_dir = os.path.join(camflood_dir, case_dir)
        case_dir = os.path.normpath(case_dir)
        rivermap_dir = os.path.join(case_dir, 'rivermap')
    
    # Determine which functions to run
    run_grid = nml.get('OPTIONS', 'run_grid', True)
    run_init = nml.get('OPTIONS', 'run_init', False)
    run_sim = nml.get('OPTIONS', 'run_sim', False)
    
    # Override with command line arguments
    if args.grid_only:
        run_grid = True
        run_init = False
        run_sim = False
    elif args.init_only:
        run_grid = False
        run_init = True
        run_sim = False
    elif args.run_only:
        run_grid = False
        run_init = False
        run_sim = True
    else:
        # Use explicit flags if provided
        if args.grid:
            run_grid = True
        if args.init:
            run_init = True
        if args.run:
            run_sim = True
    
    print(f"\nConfiguration information:")
    print(f"  Case name: {case_name}")
    print(f"  Global map directory: {global_map_dir}")
    print(f"  Case directory: {case_dir}")
    print(f"  River network output directory: {rivermap_dir}")
    print(f"  Run river network generation: {run_grid}")
    print(f"  Run initialization: {run_init}")
    print(f"  Run model simulation: {run_sim}")

    # Create output directories
    os.makedirs(case_dir, exist_ok=True)
    if run_grid:
        os.makedirs(rivermap_dir, exist_ok=True)

    # Run tasks in sequence
    success = True

    # Step 1: River Network Generation
    if run_grid:
        if not run_grid_routing_generation(nml, rivermap_dir, case_name, global_map_dir):
            success = False

    # Step 2: Initialization
    if run_init and success:
        if not run_initialization(nml, case_dir, case_name):
            success = False

    # Step 3: Model Simulation
    if run_sim and success:
        if not run_model_simulation(nml, case_dir, case_name):
            success = False
    
    # Summary
    print_header("Execution Summary")
    if success:
        print("\n✓ All tasks executed successfully!")
        print(f"\nCase output directory: {case_dir}")

        # List generated files in rivermap directory
        if run_grid and os.path.exists(rivermap_dir):
            files = sorted([f for f in os.listdir(rivermap_dir)
                           if f.endswith('.bin') or f.endswith('.txt')])
            if files:
                print(f"\nRiver network generated files ({len(files)} files) in rivermap/:")
                for f in files[:20]:  # Show first 20 files
                    fpath = os.path.join(rivermap_dir, f)
                    if os.path.isfile(fpath):
                        size_kb = os.path.getsize(fpath) / 1024
                        if size_kb < 1:
                            print(f"  - {f}")
                        else:
                            print(f"  - {f} ({size_kb:.1f} KB)")
                if len(files) > 20:
                    print(f"  ... {len(files) - 20} more files")

        print("\n" + "=" * 70)
        return 0
    else:
        print("\n✗ Some tasks failed, please check the error messages above.")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

