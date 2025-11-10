#!/usr/bin/env python3
"""
Main program for grid routing initialization
Integrates all processing steps controlled by namelist configuration
"""
import sys
import os
import argparse
from .config import Config, create_default_config
from .region_tools import RegionProcessor
from .param_tools import ParamProcessor


class GridRoutingInit:
    """Main controller for grid routing initialization"""

    def __init__(self, config_file):
        """
        Initialize with configuration file

        Args:
            config_file: Path to configuration file
        """
        self.config = Config(config_file)
        self.region_proc = RegionProcessor(self.config)
        self.param_proc = ParamProcessor(self.config)

    def run_all(self):
        """Run complete initialization workflow"""
        print("\n" + "=" * 70)
        print(" Grid Routing Initialization - Python Implementation")
        print(" Equivalent to CaMa-Flood map preprocessing")
        print("=" * 70 + "\n")

        # Step 1: Cut regional domain from global map
        print("\n### Step 1/5: Regional Map Generation ###\n")
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        dim_change = self.region_proc.cut_domain(
            global_dir=self.config.global_map_dir,
            west=self.config.west,
            east=self.config.east,
            north=self.config.north,
            south=self.config.south,
            output_dir=output_dir
        )

        # Step 2: Generate derived maps
        print("\n### Step 2/5: Generate Derived Maps ###\n")
        self.region_proc.set_map(output_dir)

        # Step 3: Generate input matrix
        print("\n### Step 3/5: Generate Input Matrix ###\n")
        diminfo_file = os.path.join(output_dir, f'diminfo_{self.config.hires_tag}_{os.path.basename(output_dir)}.txt')
        inpmat_file = os.path.join(output_dir, f'inpmat_{self.config.hires_tag}_{os.path.basename(output_dir)}.bin')

        # Check if high-resolution data exists
        hires_dir = os.path.join(output_dir, self.config.hires_tag)
        if os.path.exists(os.path.join(hires_dir, 'location.txt')):
            self.param_proc.generate_inpmat(
                map_dir=output_dir,
                hires_tag=self.config.hires_tag,
                gsizein=self.config.input_gsize,
                westin=self.config.input_west,
                eastin=self.config.input_east,
                northin=self.config.input_north,
                southin=self.config.input_south,
                olat=self.config.input_lat_order,
                diminfo_file=diminfo_file,
                inpmat_file=inpmat_file
            )
        else:
            print(f"Warning: High-resolution data '{self.config.hires_tag}' not found. Skipping input matrix generation.")
            print(f"Expected location: {hires_dir}/location.txt")

        # Step 4: Calculate river channel parameters
        print("\n### Step 4/5: Calculate Channel Parameters ###\n")

        # Note: This step requires outclm.bin which should be calculated from runoff data
        # For now, we'll skip this if the file doesn't exist
        outclm_file = os.path.join(output_dir, 'outclm.bin')
        if os.path.exists(outclm_file):
            self.param_proc.calc_rivwth(
                map_dir=output_dir,
                HC=self.config.channel_depth_coef,
                HP=self.config.channel_depth_power,
                HO=self.config.channel_depth_offset,
                HMIN=self.config.channel_depth_min,
                WC=self.config.channel_width_coef,
                WP=self.config.channel_width_power,
                WO=self.config.channel_width_offset,
                WMIN=self.config.channel_width_min
            )

            # Step 5: Merge with satellite width
            print("\n### Step 5/5: Merge Width Data ###\n")
            self.param_proc.set_gwdlr(output_dir)
        else:
            print(f"Warning: outclm.bin not found. Skipping channel parameter calculation.")
            print(f"You need to run calc_outclm first with runoff data.")

        print("\n" + "=" * 70)
        print(" Grid Routing Initialization Completed!")
        print(f" Output directory: {output_dir}")
        print("=" * 70 + "\n")

    def run_region_only(self):
        """Run only regional map generation"""
        print("\n### Running: Regional Map Generation Only ###\n")

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.region_proc.cut_domain(
            global_dir=self.config.global_map_dir,
            west=self.config.west,
            east=self.config.east,
            north=self.config.north,
            south=self.config.south,
            output_dir=output_dir
        )

        self.region_proc.set_map(output_dir)

        print("\nRegional map generation completed!")

    def run_params_only(self):
        """Run only parameter calculation"""
        print("\n### Running: Parameter Calculation Only ###\n")

        output_dir = self.config.output_dir

        # Check if regional map exists
        if not os.path.exists(os.path.join(output_dir, 'params.txt')):
            print(f"Error: Regional map not found in {output_dir}")
            print("Please run regional map generation first.")
            return

        # Generate input matrix
        diminfo_file = os.path.join(output_dir, f'diminfo_{self.config.hires_tag}_{os.path.basename(output_dir)}.txt')
        inpmat_file = os.path.join(output_dir, f'inpmat_{self.config.hires_tag}_{os.path.basename(output_dir)}.bin')

        hires_dir = os.path.join(output_dir, self.config.hires_tag)
        if os.path.exists(os.path.join(hires_dir, 'location.txt')):
            self.param_proc.generate_inpmat(
                map_dir=output_dir,
                hires_tag=self.config.hires_tag,
                gsizein=self.config.input_gsize,
                westin=self.config.input_west,
                eastin=self.config.input_east,
                northin=self.config.input_north,
                southin=self.config.input_south,
                olat=self.config.input_lat_order,
                diminfo_file=diminfo_file,
                inpmat_file=inpmat_file
            )

        # Calculate channel parameters
        outclm_file = os.path.join(output_dir, 'outclm.bin')
        if os.path.exists(outclm_file):
            self.param_proc.calc_rivwth(
                map_dir=output_dir,
                HC=self.config.channel_depth_coef,
                HP=self.config.channel_depth_power,
                HO=self.config.channel_depth_offset,
                HMIN=self.config.channel_depth_min,
                WC=self.config.channel_width_coef,
                WP=self.config.channel_width_power,
                WO=self.config.channel_width_offset,
                WMIN=self.config.channel_width_min
            )

            self.param_proc.set_gwdlr(output_dir)

        print("\nParameter calculation completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Grid Routing Initialization - Python Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default configuration
  python -m river_network.main --create-config config.ini

  # Run complete workflow
  python -m river_network.main config.ini

  # Run only regional map generation
  python -m river_network.main config.ini --region-only

  # Run only parameter calculation
  python -m river_network.main config.ini --params-only
        """
    )

    parser.add_argument('config', nargs='?', help='Configuration file')
    parser.add_argument('--create-config', metavar='FILE',
                       help='Create default configuration file')
    parser.add_argument('--region-only', action='store_true',
                       help='Run only regional map generation')
    parser.add_argument('--params-only', action='store_true',
                       help='Run only parameter calculation')

    args = parser.parse_args()

    # Create default config
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Check if config file is provided
    if not args.config:
        parser.print_help()
        return

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print(f"\nCreate a default configuration with:")
        print(f"  python -m river_network.main --create-config {args.config}")
        return

    # Run initialization
    try:
        init = GridRoutingInit(args.config)

        if args.region_only:
            init.run_region_only()
        elif args.params_only:
            init.run_params_only()
        else:
            init.run_all()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
