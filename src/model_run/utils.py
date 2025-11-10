"""
Utility functions for CaMa-Flood model run

Includes:
- NaN checking and validation
- State variable validation
- Diagnostic tools
"""
import numpy as np
import sys


def check_nan(array, name, raise_error=True, verbose=True):
    """
    Check array for NaN values

    Parameters:
    -----------
    array : ndarray
        Array to check
    name : str
        Name of the array for error messages
    raise_error : bool
        If True, raise ValueError when NaN is found
        If False, just print warning and return False
    verbose : bool
        If True, print diagnostic information

    Returns:
    --------
    bool : True if no NaN found, False if NaN found

    Raises:
    -------
    ValueError : If NaN found and raise_error=True
    """
    if np.any(np.isnan(array)):
        nan_count = np.sum(np.isnan(array))
        nan_indices = np.where(np.isnan(array))[0]

        msg = f"NaN detected in {name}:\n"
        msg += f"  Total NaN values: {nan_count}\n"
        msg += f"  Array shape: {array.shape}\n"
        msg += f"  First few NaN indices: {nan_indices[:min(5, len(nan_indices))]}\n"
        msg += f"  Array min (ignoring NaN): {np.nanmin(array):.6e}\n"
        msg += f"  Array max (ignoring NaN): {np.nanmax(array):.6e}"

        if raise_error:
            raise ValueError(msg)
        else:
            if verbose:
                print(f"WARNING: {msg}", file=sys.stderr)
            return False

    return True


def check_inf(array, name, raise_error=True, verbose=True):
    """
    Check array for Inf values

    Parameters:
    -----------
    array : ndarray
        Array to check
    name : str
        Name of the array
    raise_error : bool
        If True, raise ValueError when Inf is found
    verbose : bool
        If True, print diagnostic information

    Returns:
    --------
    bool : True if no Inf found, False if Inf found
    """
    if np.any(np.isinf(array)):
        inf_count = np.sum(np.isinf(array))
        pos_inf = np.sum(np.isposinf(array))
        neg_inf = np.sum(np.isneginf(array))
        inf_indices = np.where(np.isinf(array))[0]

        msg = f"Inf detected in {name}:\n"
        msg += f"  Total Inf values: {inf_count}\n"
        msg += f"  Positive Inf: {pos_inf}\n"
        msg += f"  Negative Inf: {neg_inf}\n"
        msg += f"  Array shape: {array.shape}\n"
        msg += f"  First few Inf indices: {inf_indices[:min(5, len(inf_indices))]}"

        if raise_error:
            raise ValueError(msg)
        else:
            if verbose:
                print(f"WARNING: {msg}", file=sys.stderr)
            return False

    return True


def check_negative(array, name, raise_error=True, verbose=True):
    """
    Check array for negative values (for variables that should be non-negative)

    Parameters:
    -----------
    array : ndarray
        Array to check
    name : str
        Name of the array
    raise_error : bool
        If True, raise ValueError when negative values found
    verbose : bool
        If True, print diagnostic information

    Returns:
    --------
    bool : True if no negative values, False if negative found
    """
    if np.any(array < 0):
        neg_count = np.sum(array < 0)
        neg_indices = np.where(array < 0)[0]
        min_val = np.min(array)

        msg = f"Negative values detected in {name}:\n"
        msg += f"  Total negative values: {neg_count}\n"
        msg += f"  Minimum value: {min_val:.6e}\n"
        msg += f"  Array shape: {array.shape}\n"
        msg += f"  First few negative indices: {neg_indices[:min(5, len(neg_indices))]}"

        if raise_error:
            raise ValueError(msg)
        else:
            if verbose:
                print(f"WARNING: {msg}", file=sys.stderr)
            return False

    return True


def check_array_valid(array, name, do_check_nan=True, do_check_inf=True,
                      do_check_negative=False, raise_error=True, verbose=True):
    """
    Comprehensive array validation

    Parameters:
    -----------
    array : ndarray
        Array to check
    name : str
        Name of the array
    do_check_nan : bool
        Check for NaN values
    do_check_inf : bool
        Check for Inf values
    do_check_negative : bool
        Check for negative values
    raise_error : bool
        Raise error if validation fails
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    bool : True if all checks pass
    """
    from . import utils  # Import module to access functions

    valid = True

    if do_check_nan:
        valid = check_nan(array, name, raise_error, verbose) and valid

    if do_check_inf:
        valid = check_inf(array, name, raise_error, verbose) and valid

    if do_check_negative:
        valid = check_negative(array, name, raise_error, verbose) and valid

    return valid


def check_state_valid(state, raise_error=True, verbose=True):
    """
    Validate model state variables

    Checks for common issues:
    - NaN values in any variable
    - Inf values in any variable
    - Negative storage (should be non-negative)
    - Negative depth (should be non-negative)

    Parameters:
    -----------
    state : dict
        Dictionary of state variables
    raise_error : bool
        If True, raise error on first validation failure
        If False, check all variables and print warnings
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    bool : True if all checks pass

    Example:
    --------
    >>> state = physics.get_state()
    >>> check_state_valid(state, raise_error=True)
    """
    all_valid = True

    # Variables that should be non-negative
    non_negative_vars = [
        'rivsto', 'fldsto',      # Storage
        'rivdph', 'flddph',      # Depth
        'storge',                 # Total storage
        'fldare', 'fldfrc',      # Floodplain area and fraction
    ]

    # Check each variable in state
    for varname, array in state.items():

        # Check NaN
        valid_nan = check_nan(array, varname, raise_error, verbose)
        all_valid = all_valid and valid_nan

        # Check Inf
        valid_inf = check_inf(array, varname, raise_error, verbose)
        all_valid = all_valid and valid_inf

        # Check negative values for specific variables
        if varname in non_negative_vars:
            valid_neg = check_negative(array, varname, raise_error, verbose)
            all_valid = all_valid and valid_neg

        # If any check failed and raise_error is True, we would have raised already
        # If raise_error is False, continue checking all variables

    return all_valid


def print_state_summary(state, nseqall=None):
    """
    Print summary statistics for state variables

    Parameters:
    -----------
    state : dict
        Dictionary of state variables
    nseqall : int (optional)
        Number of active cells (if None, use full array)
    """
    print("\n" + "=" * 70)
    print("STATE SUMMARY")
    print("=" * 70)

    if nseqall is None:
        nseqall = len(next(iter(state.values())))

    for varname, array in sorted(state.items()):
        data = array[:nseqall] if len(array.shape) == 1 else array

        # Skip empty arrays
        if data.size == 0:
            continue

        # Calculate statistics
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Count special values
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        neg_count = np.sum(data < 0) if min_val < 0 else 0

        print(f"\n{varname}:")
        print(f"  Min:  {min_val:12.4e}    Max:  {max_val:12.4e}")
        print(f"  Mean: {mean_val:12.4e}    Std:  {std_val:12.4e}")

        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN values")
        if inf_count > 0:
            print(f"  WARNING: {inf_count} Inf values")
        if neg_count > 0:
            print(f"  WARNING: {neg_count} negative values")

    print("=" * 70)


def find_problematic_cells(state, nseqall=None, threshold=1e10):
    """
    Find cells with problematic values (very large, NaN, Inf)

    Parameters:
    -----------
    state : dict
        Dictionary of state variables
    nseqall : int (optional)
        Number of active cells
    threshold : float
        Threshold for "very large" values

    Returns:
    --------
    dict : Dictionary mapping cell indices to problematic variables
    """
    if nseqall is None:
        nseqall = len(next(iter(state.values())))

    problematic_cells = {}

    for varname, array in state.items():
        if len(array.shape) != 1:
            continue  # Skip multi-dimensional arrays

        data = array[:nseqall]

        # Find problematic indices
        nan_idx = np.where(np.isnan(data))[0]
        inf_idx = np.where(np.isinf(data))[0]
        large_idx = np.where(np.abs(data) > threshold)[0]

        # Record problematic cells
        for idx in np.unique(np.concatenate([nan_idx, inf_idx, large_idx])):
            if idx not in problematic_cells:
                problematic_cells[idx] = []
            problematic_cells[idx].append({
                'variable': varname,
                'value': data[idx],
                'issue': 'NaN' if idx in nan_idx else 'Inf' if idx in inf_idx else 'Large'
            })

    return problematic_cells


def diagnose_model_state(state, nseqall=None):
    """
    Comprehensive diagnostic of model state

    This function provides a full report on the model state,
    including statistics, validation checks, and problematic cells.

    Parameters:
    -----------
    state : dict
        Dictionary of state variables
    nseqall : int (optional)
        Number of active cells

    Example:
    --------
    >>> state = physics.get_state()
    >>> diagnose_model_state(state, nseqall=100)
    """
    print("\n" + "=" * 70)
    print("MODEL STATE DIAGNOSTICS")
    print("=" * 70)

    # 1. Print summary statistics
    print_state_summary(state, nseqall)

    # 2. Validate state
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    is_valid = check_state_valid(state, raise_error=False, verbose=True)

    if is_valid:
        print("\n✓ All validation checks passed")
    else:
        print("\n✗ Some validation checks failed")

    # 3. Find problematic cells
    print("\n" + "=" * 70)
    print("PROBLEMATIC CELLS")
    print("=" * 70)
    problematic = find_problematic_cells(state, nseqall)

    if len(problematic) == 0:
        print("\n✓ No problematic cells found")
    else:
        print(f"\n✗ Found {len(problematic)} problematic cells:")
        for cell_idx, issues in list(problematic.items())[:10]:  # Show first 10
            print(f"\n  Cell {cell_idx}:")
            for issue in issues:
                print(f"    {issue['variable']}: {issue['value']:.4e} ({issue['issue']})")

        if len(problematic) > 10:
            print(f"\n  ... and {len(problematic) - 10} more cells")

    print("\n" + "=" * 70)


# Convenience function for integration into physics module
def validate_physics_state(physics, step_num=None, raise_error=True, verbose=False):
    """
    Validate physics state (convenience wrapper)

    Parameters:
    -----------
    physics : CaMaPhysics
        Physics instance
    step_num : int (optional)
        Time step number for diagnostic message
    raise_error : bool
        Raise error if validation fails
    verbose : bool
        Print detailed diagnostic information

    Returns:
    --------
    bool : True if valid

    Example:
    --------
    >>> # In physics.py
    >>> if self.ldebug:
    >>>     from .utils import validate_physics_state
    >>>     validate_physics_state(self, step_num=istep, raise_error=True)
    """
    state = physics.get_state()

    if step_num is not None and verbose:
        print(f"\nValidating state at step {step_num}...")

    return check_state_valid(state, raise_error=raise_error, verbose=verbose)
