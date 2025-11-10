"""
Runtime debugging trace module
Records all intermediate variables for specified grid cells during simulation
"""
import numpy as np

class CellTracer:
    """Grid cell tracer"""

    def __init__(self, cell_list=None):
        """
        Initialize tracer

        Parameters:
        -----------
        cell_list : list of int
            Sequence indices of grid cells to trace
        """
        self.enabled = cell_list is not None and len(cell_list) > 0
        self.cells = set(cell_list) if cell_list else set()
        self.trace_data = []
        self.timestep = 0
        self.substep = 0

    def set_timestep(self, timestep, substep=0):
        """Set current timestep"""
        self.timestep = timestep
        self.substep = substep

    def trace(self, iseq, stage, **kwargs):
        """
        Record state of a grid cell

        Parameters:
        -----------
        iseq : int
            Grid cell sequence index
        stage : str
            Computation stage identifier
        **kwargs : dict
            Variables to record
        """
        if not self.enabled or iseq not in self.cells:
            return

        record = {
            'timestep': self.timestep,
            'substep': self.substep,
            'iseq': iseq,
            'stage': stage,
        }
        record.update(kwargs)
        self.trace_data.append(record)

    def save_to_file(self, filename):
        """Save trace data to file"""
        if not self.trace_data:
            return

        with open(filename, 'w') as f:
            f.write("="*100 + "\n")
            f.write("CaMa-Flood Python Version - Intermediate Variable Trace\n")
            f.write("="*100 + "\n")

            # Group by grid cell
            by_cell = {}
            for record in self.trace_data:
                iseq = record['iseq']
                if iseq not in by_cell:
                    by_cell[iseq] = []
                by_cell[iseq].append(record)

            # Output trace data for each grid cell
            for iseq in sorted(by_cell.keys()):
                f.write(f"\n\n{'='*100}\n")
                f.write(f"Complete trace for grid cell {iseq}\n")
                f.write(f"{'='*100}\n")

                for record in by_cell[iseq]:
                    f.write(f"\nTimestep={record['timestep']}, Substep={record['substep']}, Stage={record['stage']}\n")
                    f.write("-"*80 + "\n")

                    for key in sorted(record.keys()):
                        if key not in ['timestep', 'substep', 'iseq', 'stage']:
                            value = record[key]
                            if isinstance(value, (int, np.integer)):
                                f.write(f"  {key:<25s}: {value:20d}\n")
                            elif isinstance(value, (float, np.floating)):
                                f.write(f"  {key:<25s}: {value:25.15e}\n")
                            elif isinstance(value, bool):
                                f.write(f"  {key:<25s}: {str(value):>20s}\n")
                            else:
                                f.write(f"  {key:<25s}: {str(value)}\n")

        print(f"Trace data saved to: {filename}")
        print(f"  Total records: {len(self.trace_data)}")
        print(f"  Traced cells: {sorted(self.cells)}")

    def print_summary(self):
        """Print trace data summary"""
        if not self.trace_data:
            print("No trace data")
            return

        by_cell = {}
        for record in self.trace_data:
            iseq = record['iseq']
            if iseq not in by_cell:
                by_cell[iseq] = []
            by_cell[iseq].append(record)

        print(f"\nTrace data summary:")
        for iseq in sorted(by_cell.keys()):
            stages = {}
            for record in by_cell[iseq]:
                stage = record['stage']
                stages[stage] = stages.get(stage, 0) + 1

            print(f"  Grid cell {iseq}: {len(by_cell[iseq])} records")
            for stage, count in sorted(stages.items()):
                print(f"    {stage}: {count}")

# Global tracer instance
_global_tracer = CellTracer()

def init_tracer(cell_list):
    """Initialize global tracer"""
    global _global_tracer
    _global_tracer = CellTracer(cell_list)
    return _global_tracer

def get_tracer():
    """Get global tracer"""
    return _global_tracer

def trace_outflw_input(iseq, **kwargs):
    """Trace calc_outflw inputs"""
    _global_tracer.trace(iseq, 'outflw_input', **kwargs)

def trace_outflw_river(iseq, **kwargs):
    """Trace river flow calculation"""
    _global_tracer.trace(iseq, 'outflw_river', **kwargs)

def trace_outflw_flood(iseq, **kwargs):
    """Trace floodplain flow calculation"""
    _global_tracer.trace(iseq, 'outflw_flood', **kwargs)

def trace_conservation(iseq, **kwargs):
    """Trace mass conservation correction"""
    _global_tracer.trace(iseq, 'conservation', **kwargs)

def trace_stonxt(iseq, **kwargs):
    """Trace storage update"""
    _global_tracer.trace(iseq, 'stonxt', **kwargs)
