#!/usr/bin/env python3
"""
Input validation for river network generation

Validates all paths, dimensions, and configurations before starting processing.
Provides clear error messages for any issues found.
"""
import os
import sys
import numpy as np
from .fortran_io import read_params_txt, FortranBinary


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputValidator:
    """Validates all inputs before processing"""

    def __init__(self, namelist):
        """
        Initialize validator with namelist

        Args:
            namelist: Namelist object with configuration
        """
        self.nml = namelist
        self.errors = []
        self.warnings = []

    def validate_all(self, global_map_dir, output_dir, run_inpmat=False):
        """
        Run all validation checks

        Args:
            global_map_dir: Path to global map directory (will be converted to absolute)
            output_dir: Output directory (will be converted to absolute)
            run_inpmat: Whether inpmat generation is enabled

        Returns:
            tuple: (validated_global_dir, validated_output_dir)

        Raises:
            ValidationError: If any validation fails
        """
        print("=" * 70)
        print("VALIDATING INPUT CONFIGURATION")
        print("=" * 70)

        # 1. Validate and convert paths
        global_map_dir = self._validate_global_map_dir(global_map_dir)
        output_dir = self._validate_output_dir(output_dir)

        # 2. Validate domain boundaries
        self._validate_domain_boundaries()

        # 3. Validate global map files exist
        self._validate_global_map_files(global_map_dir)

        # 4. Validate global map dimensions
        nx_global, ny_global, nflp, gsize = self._validate_global_map_dimensions(global_map_dir)

        # 5. Validate binary file dimensions
        self._validate_binary_file_dimensions(global_map_dir, nx_global, ny_global, nflp)

        # 6. Validate inpmat configuration if needed
        if run_inpmat:
            self._validate_inpmat_config(global_map_dir)

        # Report results
        self._report_validation_results()

        if self.errors:
            raise ValidationError(f"Validation failed with {len(self.errors)} error(s)")

        print("\n✓ All validation checks passed!")
        print(f"  Global map directory: {global_map_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Grid dimensions: {nx_global} x {ny_global}")
        print("=" * 70)
        print()

        return global_map_dir, output_dir

    def _validate_global_map_dir(self, global_map_dir):
        """Validate and convert global map directory to absolute path"""
        if global_map_dir is None:
            self.errors.append("global_map_dir is not specified in namelist")
            return None

        # Convert to absolute path
        if not os.path.isabs(global_map_dir):
            # Try relative to current directory
            abs_path = os.path.abspath(global_map_dir)
            print(f"  Converting relative path to absolute:")
            print(f"    Relative: {global_map_dir}")
            print(f"    Absolute: {abs_path}")
            global_map_dir = abs_path

        # Check existence
        if not os.path.exists(global_map_dir):
            self.errors.append(
                f"Global map directory does not exist: {global_map_dir}\n"
                f"  Please check the 'global_map_dir' path in your namelist"
            )
            return global_map_dir

        # Check it's a directory
        if not os.path.isdir(global_map_dir):
            self.errors.append(
                f"Global map path is not a directory: {global_map_dir}\n"
                f"  Expected a directory containing river network map files"
            )
            return global_map_dir

        print(f"✓ Global map directory exists: {global_map_dir}")
        return global_map_dir

    def _validate_output_dir(self, output_dir):
        """Validate and convert output directory to absolute path"""
        if output_dir is None:
            self.errors.append("Output directory is not specified")
            return None

        # Convert to absolute path
        if not os.path.isabs(output_dir):
            abs_path = os.path.abspath(output_dir)
            print(f"  Converting output path to absolute:")
            print(f"    Relative: {output_dir}")
            print(f"    Absolute: {abs_path}")
            output_dir = abs_path

        # Try to create output directory (including all parents)
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Output directory ready: {output_dir}")
        except Exception as e:
            self.errors.append(
                f"Cannot create output directory: {output_dir}\n"
                f"  Error: {str(e)}\n"
                f"  Please check directory permissions"
            )

        return output_dir

    def _validate_domain_boundaries(self):
        """Validate domain boundary values"""
        west = self.nml.get('RiverMap_Gen', 'west')
        east = self.nml.get('RiverMap_Gen', 'east')
        south = self.nml.get('RiverMap_Gen', 'south')
        north = self.nml.get('RiverMap_Gen', 'north')

        # Check all boundaries are specified
        if west is None:
            self.errors.append("Domain boundary 'west' is not specified in namelist")
        if east is None:
            self.errors.append("Domain boundary 'east' is not specified in namelist")
        if south is None:
            self.errors.append("Domain boundary 'south' is not specified in namelist")
        if north is None:
            self.errors.append("Domain boundary 'north' is not specified in namelist")

        if any(x is None for x in [west, east, south, north]):
            return

        # Check logical consistency
        if west >= east:
            self.errors.append(
                f"Invalid domain: west ({west}°) must be < east ({east}°)"
            )

        if south >= north:
            self.errors.append(
                f"Invalid domain: south ({south}°) must be < north ({north}°)"
            )

        # Check reasonable ranges
        if west < -180 or west > 180:
            self.errors.append(
                f"Invalid west boundary: {west}° (must be -180 to 180)"
            )

        if east < -180 or east > 180:
            self.errors.append(
                f"Invalid east boundary: {east}° (must be -180 to 180)"
            )

        if south < -90 or south > 90:
            self.errors.append(
                f"Invalid south boundary: {south}° (must be -90 to 90)"
            )

        if north < -90 or north > 90:
            self.errors.append(
                f"Invalid north boundary: {north}° (must be -90 to 90)"
            )

        print(f"✓ Domain boundaries valid: [{west}, {east}] x [{south}, {north}]")

    def _validate_global_map_files(self, global_map_dir):
        """Check that all required files exist in global map directory"""
        if global_map_dir is None or not os.path.exists(global_map_dir):
            return

        # Essential files that must exist
        essential_files = [
            'params.txt',
            'nextxy.bin',
            'elevtn.bin',
            'ctmare.bin',
            'grdare.bin',
            'uparea.bin',
            'lonlat.bin',
            'nxtdst.bin',
            'rivlen.bin',
            'width.bin',
            'fldhgt.bin'
        ]

        missing_files = []
        for filename in essential_files:
            filepath = os.path.join(global_map_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            self.errors.append(
                f"Missing essential files in global map directory:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                f"\n  Directory: {global_map_dir}"
            )
        else:
            print(f"✓ All {len(essential_files)} essential files present")

    def _validate_global_map_dimensions(self, global_map_dir):
        """Read and validate dimensions from params.txt"""
        if global_map_dir is None or not os.path.exists(global_map_dir):
            return None, None, None, None

        params_file = os.path.join(global_map_dir, 'params.txt')
        if not os.path.exists(params_file):
            return None, None, None, None

        try:
            params = read_params_txt(params_file)
            nx = params['nx']
            ny = params['ny']
            nflp = params['nflp']
            gsize = params['gsize']

            # Validate dimensions are reasonable
            if nx <= 0 or nx > 100000:
                self.errors.append(f"Invalid nx dimension: {nx} (must be 1-100000)")

            if ny <= 0 or ny > 100000:
                self.errors.append(f"Invalid ny dimension: {ny} (must be 1-100000)")

            if nflp <= 0 or nflp > 100:
                self.errors.append(f"Invalid nflp dimension: {nflp} (must be 1-100)")

            if gsize <= 0 or gsize > 10:
                self.errors.append(f"Invalid grid size: {gsize}° (must be 0-10°)")

            print(f"✓ Global map dimensions: {nx} x {ny}, {nflp} floodplain layers, {gsize}° resolution")

            return nx, ny, nflp, gsize

        except Exception as e:
            self.errors.append(
                f"Error reading params.txt: {str(e)}\n"
                f"  File: {params_file}"
            )
            return None, None, None, None

    def _validate_binary_file_dimensions(self, global_map_dir, nx, ny, nflp):
        """Validate that binary files have correct dimensions"""
        if any(x is None for x in [global_map_dir, nx, ny, nflp]):
            return

        if not os.path.exists(global_map_dir):
            return

        # Expected file sizes (in bytes)
        # int4: 4 bytes per element
        # real: 4 bytes per element
        # Note: Fortran headers are handled by FortranBinary class
        expected_sizes = {
            'nextxy.bin': nx * ny * 4 * 2,   # 2 records (x,y)
            'downxy.bin': nx * ny * 4 * 2,   # 2 records
            'elevtn.bin': nx * ny * 4,       # 1 record
            'ctmare.bin': nx * ny * 4,
            'grdare.bin': nx * ny * 4,
            'uparea.bin': nx * ny * 4,
            'lonlat.bin': nx * ny * 4 * 2,   # 2 records
            'nxtdst.bin': nx * ny * 4,
            'rivlen.bin': nx * ny * 4,
            'width.bin': nx * ny * 4,
            'fldhgt.bin': nx * ny * 4 * nflp,  # nflp records
        }

        dimension_errors = []
        for filename, expected_size in expected_sizes.items():
            filepath = os.path.join(global_map_dir, filename)
            if os.path.exists(filepath):
                actual_size = os.path.getsize(filepath)
                if actual_size != expected_size:
                    dimension_errors.append(
                        f"  {filename}: Expected {expected_size} bytes, got {actual_size} bytes"
                    )

        if dimension_errors:
            self.errors.append(
                f"Binary file dimension mismatches detected:\n" +
                "\n".join(dimension_errors) +
                f"\n  Expected dimensions: {nx} x {ny} x {nflp}\n"
                f"  This may indicate corrupted files or incorrect params.txt"
            )
        else:
            print(f"✓ All binary files have correct dimensions")

    def _validate_inpmat_config(self, global_map_dir):
        """Validate input matrix configuration"""
        if global_map_dir is None or not os.path.exists(global_map_dir):
            return

        # Check high-resolution data
        hires_tag = self.nml.get('RiverMap_Gen', 'hires_tag', '1min')
        hires_dir = os.path.join(global_map_dir, hires_tag)

        if not os.path.exists(hires_dir):
            self.warnings.append(
                f"High-resolution data directory not found: {hires_dir}\n"
                f"  run_inpmat=True requires high-resolution data\n"
                f"  Expected directory: {hires_tag}/"
            )
            return

        location_file = os.path.join(hires_dir, 'location.txt')
        if not os.path.exists(location_file):
            self.warnings.append(
                f"High-resolution location.txt not found: {location_file}\n"
                f"  This file is required for inpmat generation"
            )
        else:
            print(f"✓ High-resolution data found: {hires_tag}/")

        # Validate input domain parameters
        grsizein = self.nml.get('RiverMap_Gen', 'grsizein')
        westin = self.nml.get('RiverMap_Gen', 'westin')
        eastin = self.nml.get('RiverMap_Gen', 'eastin')
        southin = self.nml.get('RiverMap_Gen', 'southin')
        northin = self.nml.get('RiverMap_Gen', 'northin')

        if grsizein is None:
            self.warnings.append("Input grid size 'grsizein' not specified for inpmat")

        if any(x is None for x in [westin, eastin, southin, northin]):
            self.warnings.append("Input domain boundaries not fully specified for inpmat")

    def _report_validation_results(self):
        """Print validation results"""
        print()
        print("-" * 70)
        print("VALIDATION SUMMARY")
        print("-" * 70)

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"\n{i}. {warning}")

        if self.errors:
            print(f"\n❌ {len(self.errors)} ERROR(S):")
            for i, error in enumerate(self.errors, 1):
                print(f"\n{i}. {error}")
        else:
            print("\n✓ No errors found")

        print("-" * 70)


def validate_inputs(namelist, global_map_dir, output_dir, run_inpmat=False):
    """
    Convenience function to validate all inputs

    Args:
        namelist: Namelist object
        global_map_dir: Global map directory path
        output_dir: Output directory path
        run_inpmat: Whether inpmat generation is enabled

    Returns:
        tuple: (validated_global_dir, validated_output_dir)

    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator(namelist)
    return validator.validate_all(global_map_dir, output_dir, run_inpmat)
