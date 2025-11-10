"""
Diagnostic module for CaMa-Flood model run
Manages average and maximum diagnostic variables for output

Based on CMF_CALC_DIAG_MOD.F90

Implements two levels of diagnostics:
1. Adaptive timestep diagnostics (_adp): accumulated within adaptive sub-steps
2. Output diagnostics (_out): accumulated for output intervals
"""
import numpy as np


class DiagnosticManager:
    """
    Manage diagnostic variables (averages and maximums) for CaMa-Flood

    This class implements a two-level diagnostic system:
    - Level 1: Adaptive timestep diagnostics (reset/accumulate at each main timestep)
    - Level 2: Output diagnostics (reset/accumulate at each output interval)
    """

    def __init__(self, nseqmax, npthout=0, npthlev=1, ldamout=False, lwevap=False):
        """
        Initialize diagnostic manager

        Parameters:
        -----------
        nseqmax : int
            Maximum number of river cells
        npthout : int
            Number of bifurcation channels
        npthlev : int
            Number of bifurcation levels
        ldamout : bool
            Dam operation enabled
        lwevap : bool
            Evaporation enabled
        """
        self.nseqmax = nseqmax
        self.npthout = npthout
        self.npthlev = npthlev
        self.ldamout = ldamout
        self.lwevap = lwevap

        # Initialize adaptive timestep diagnostics
        self._initialize_adp_diagnostics()

        # Initialize output diagnostics
        self._initialize_out_diagnostics()

    def _initialize_adp_diagnostics(self):
        """Initialize adaptive timestep diagnostic variables"""
        # Time accumulator for adaptive timestep
        self.nadd_adp = 0.0  # Accumulated time [seconds]

        # Average variables (accumulated over adaptive timesteps)
        self.d2rivout_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2fldout_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2outflw_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivvel_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2pthout_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2gdwrtn_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2runoff_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rofsub_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)

        # Maximum variables (accumulated over adaptive timesteps)
        self.d2outflw_adp_max = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivdph_adp_max = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2storge_adp_max = np.zeros(self.nseqmax, dtype=np.float64)

        # Optional variables
        if self.ldamout:
            self.d2daminf_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)

        if self.lwevap:
            self.d2wevapex_adp_avg = np.zeros(self.nseqmax, dtype=np.float64)

        # Bifurcation diagnostics
        if self.npthout > 0:
            self.d1pthflw_adp_avg = np.zeros((self.npthout, self.npthlev), dtype=np.float64)
            self.d1pthflwsum_adp_avg = np.zeros(self.npthout, dtype=np.float64)

    def _initialize_out_diagnostics(self):
        """Initialize output diagnostic variables"""
        # Time accumulator for output
        self.nadd_out = 0.0  # Accumulated time [seconds]

        # Average variables (accumulated for output)
        self.d2rivout_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2fldout_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2outflw_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivvel_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2pthout_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2gdwrtn_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2runoff_out_avg = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rofsub_out_avg = np.zeros(self.nseqmax, dtype=np.float64)

        # Maximum variables (accumulated for output)
        self.d2outflw_out_max = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2rivdph_out_max = np.zeros(self.nseqmax, dtype=np.float64)
        self.d2storge_out_max = np.zeros(self.nseqmax, dtype=np.float64)

        # Optional variables
        if self.ldamout:
            self.d2daminf_out_avg = np.zeros(self.nseqmax, dtype=np.float64)

        if self.lwevap:
            self.d2wevapex_out_avg = np.zeros(self.nseqmax, dtype=np.float64)

        # Bifurcation diagnostics
        if self.npthout > 0:
            self.d1pthflw_out_avg = np.zeros((self.npthout, self.npthlev), dtype=np.float64)

    def reset_adp(self):
        """Reset adaptive timestep diagnostic variables"""
        self.nadd_adp = 0.0

        # Reset averages
        self.d2rivout_adp_avg[:] = 0.0
        self.d2fldout_adp_avg[:] = 0.0
        self.d2outflw_adp_avg[:] = 0.0
        self.d2rivvel_adp_avg[:] = 0.0
        self.d2pthout_adp_avg[:] = 0.0
        self.d2gdwrtn_adp_avg[:] = 0.0
        self.d2runoff_adp_avg[:] = 0.0
        self.d2rofsub_adp_avg[:] = 0.0

        # Reset maximums
        self.d2outflw_adp_max[:] = 0.0
        self.d2rivdph_adp_max[:] = 0.0
        self.d2storge_adp_max[:] = 0.0

        # Reset optional variables
        if self.ldamout:
            self.d2daminf_adp_avg[:] = 0.0

        if self.lwevap:
            self.d2wevapex_adp_avg[:] = 0.0

        # Reset bifurcation
        if self.npthout > 0:
            self.d1pthflw_adp_avg[:, :] = 0.0
            self.d1pthflwsum_adp_avg[:] = 0.0

    def accumulate_adp(self, state, dt):
        """
        Accumulate diagnostic variables at adaptive timestep

        Parameters:
        -----------
        state : dict
            Current model state containing diagnostic variables
        dt : float
            Time step [seconds]
        """
        self.nadd_adp += dt

        # Accumulate averages (weighted by time)
        self.d2rivout_adp_avg += state.get('rivout', 0.0) * dt
        self.d2fldout_adp_avg += state.get('fldout', 0.0) * dt
        self.d2outflw_adp_avg += state.get('outflw', 0.0) * dt
        self.d2rivvel_adp_avg += state.get('rivvel', 0.0) * dt
        self.d2pthout_adp_avg += state.get('pthout', 0.0) * dt
        self.d2gdwrtn_adp_avg += state.get('gdwrtn', 0.0) * dt
        self.d2runoff_adp_avg += state.get('runoff', 0.0) * dt
        self.d2rofsub_adp_avg += state.get('rofsub', 0.0) * dt

        # Update maximums
        self.d2outflw_adp_max = np.maximum(self.d2outflw_adp_max,
                                           np.abs(state.get('outflw', 0.0)))
        self.d2rivdph_adp_max = np.maximum(self.d2rivdph_adp_max,
                                           state.get('rivdph', 0.0))
        self.d2storge_adp_max = np.maximum(self.d2storge_adp_max,
                                           state.get('storge', 0.0))

        # Accumulate optional variables
        if self.ldamout and 'daminf' in state:
            self.d2daminf_adp_avg += state['daminf'] * dt

        if self.lwevap and 'wevapex' in state:
            self.d2wevapex_adp_avg += state['wevapex'] * dt

        # Accumulate bifurcation
        if self.npthout > 0 and 'pthflw' in state:
            self.d1pthflw_adp_avg += state['pthflw'] * dt

    def finalize_adp(self):
        """Calculate time-averaged values for adaptive timestep period"""
        if self.nadd_adp > 0:
            # Calculate averages
            self.d2rivout_adp_avg /= self.nadd_adp
            self.d2fldout_adp_avg /= self.nadd_adp
            self.d2outflw_adp_avg /= self.nadd_adp
            self.d2rivvel_adp_avg /= self.nadd_adp
            self.d2pthout_adp_avg /= self.nadd_adp
            self.d2gdwrtn_adp_avg /= self.nadd_adp
            self.d2runoff_adp_avg /= self.nadd_adp
            self.d2rofsub_adp_avg /= self.nadd_adp

            # Optional variables
            if self.ldamout:
                self.d2daminf_adp_avg /= self.nadd_adp

            if self.lwevap:
                self.d2wevapex_adp_avg /= self.nadd_adp

            # Bifurcation
            if self.npthout > 0:
                self.d1pthflw_adp_avg /= self.nadd_adp
                self.d1pthflwsum_adp_avg = np.sum(self.d1pthflw_adp_avg, axis=1)

    def reset_out(self):
        """Reset output diagnostic variables"""
        self.nadd_out = 0.0

        # Reset averages
        self.d2rivout_out_avg[:] = 0.0
        self.d2fldout_out_avg[:] = 0.0
        self.d2outflw_out_avg[:] = 0.0
        self.d2rivvel_out_avg[:] = 0.0
        self.d2pthout_out_avg[:] = 0.0
        self.d2gdwrtn_out_avg[:] = 0.0
        self.d2runoff_out_avg[:] = 0.0
        self.d2rofsub_out_avg[:] = 0.0

        # Reset maximums
        self.d2outflw_out_max[:] = 0.0
        self.d2rivdph_out_max[:] = 0.0
        self.d2storge_out_max[:] = 0.0

        # Reset optional variables
        if self.ldamout:
            self.d2daminf_out_avg[:] = 0.0

        if self.lwevap:
            self.d2wevapex_out_avg[:] = 0.0

        # Reset bifurcation
        if self.npthout > 0:
            self.d1pthflw_out_avg[:, :] = 0.0

    def accumulate_out(self, dt):
        """
        Accumulate output diagnostics from adaptive timestep diagnostics

        This should be called after finalize_adp() to accumulate
        the time-averaged adaptive timestep diagnostics into the
        output diagnostics.

        Parameters:
        -----------
        dt : float
            Main time step [seconds]
        """
        self.nadd_out += dt

        # Accumulate averages from adaptive timestep averages
        self.d2rivout_out_avg += self.d2rivout_adp_avg * dt
        self.d2fldout_out_avg += self.d2fldout_adp_avg * dt
        self.d2outflw_out_avg += self.d2outflw_adp_avg * dt
        self.d2rivvel_out_avg += self.d2rivvel_adp_avg * dt
        self.d2pthout_out_avg += self.d2pthout_adp_avg * dt
        self.d2gdwrtn_out_avg += self.d2gdwrtn_adp_avg * dt
        self.d2runoff_out_avg += self.d2runoff_adp_avg * dt
        self.d2rofsub_out_avg += self.d2rofsub_adp_avg * dt

        # Update maximums from adaptive timestep maximums
        self.d2outflw_out_max = np.maximum(self.d2outflw_out_max,
                                           np.abs(self.d2outflw_adp_max))
        self.d2rivdph_out_max = np.maximum(self.d2rivdph_out_max,
                                           self.d2rivdph_adp_max)
        self.d2storge_out_max = np.maximum(self.d2storge_out_max,
                                           self.d2storge_adp_max)

        # Accumulate optional variables
        if self.ldamout:
            self.d2daminf_out_avg += self.d2daminf_adp_avg * dt

        if self.lwevap:
            self.d2wevapex_out_avg += self.d2wevapex_adp_avg * dt

        # Accumulate bifurcation
        if self.npthout > 0:
            self.d1pthflw_out_avg += self.d1pthflw_adp_avg * dt

    def finalize_out(self):
        """Calculate time-averaged values for output period"""
        if self.nadd_out > 0:
            # Calculate averages
            self.d2rivout_out_avg /= self.nadd_out
            self.d2fldout_out_avg /= self.nadd_out
            self.d2outflw_out_avg /= self.nadd_out
            self.d2rivvel_out_avg /= self.nadd_out
            self.d2pthout_out_avg /= self.nadd_out
            self.d2gdwrtn_out_avg /= self.nadd_out
            self.d2runoff_out_avg /= self.nadd_out
            self.d2rofsub_out_avg /= self.nadd_out

            # Optional variables
            if self.ldamout:
                self.d2daminf_out_avg /= self.nadd_out

            if self.lwevap:
                self.d2wevapex_out_avg /= self.nadd_out

            # Bifurcation
            if self.npthout > 0:
                self.d1pthflw_out_avg /= self.nadd_out

    def get_output_diagnostics(self, nseqall):
        """
        Get output diagnostic variables for writing to file

        Parameters:
        -----------
        nseqall : int
            Number of active river cells

        Returns:
        --------
        dict : Output diagnostic variables
        """
        diag = {
            # Average variables
            'rivout_avg': self.d2rivout_out_avg[:nseqall].copy(),
            'fldout_avg': self.d2fldout_out_avg[:nseqall].copy(),
            'outflw_avg': self.d2outflw_out_avg[:nseqall].copy(),
            'rivvel_avg': self.d2rivvel_out_avg[:nseqall].copy(),
            'pthout_avg': self.d2pthout_out_avg[:nseqall].copy(),
            'runoff_avg': self.d2runoff_out_avg[:nseqall].copy(),

            # Maximum variables
            'outflw_max': self.d2outflw_out_max[:nseqall].copy(),
            'rivdph_max': self.d2rivdph_out_max[:nseqall].copy(),
            'storge_max': self.d2storge_out_max[:nseqall].copy(),
        }

        # Add optional variables
        if self.ldamout:
            diag['daminf_avg'] = self.d2daminf_out_avg[:nseqall].copy()

        if self.lwevap:
            diag['wevapex_avg'] = self.d2wevapex_out_avg[:nseqall].copy()

        if self.npthout > 0:
            diag['pthflw_avg'] = self.d1pthflw_out_avg.copy()

        return diag
