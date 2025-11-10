"""
Time control module for CaMa-Flood model run
Handles time stepping and calendar operations

Based on CMF_CTRL_TIME_MOD.F90
"""
import numpy as np
from datetime import datetime, timedelta


class TimeControl:
    """Time control for CaMa-Flood model integration"""

    def __init__(self, syear, smon, sday, shour, eyear, emon, eday, ehour, dt, lleapyr=True):
        """
        Initialize time control

        Parameters:
        -----------
        syear, smon, sday, shour : int
            Start year, month, day, hour
        eyear, emon, eday, ehour : int
            End year, month, day, hour
        dt : float
            Time step in seconds
        lleapyr : bool
            Consider leap year (True) or skip Feb 29 (False)
        """
        self.dt = dt
        self.lleapyr = lleapyr

        # Start and end time
        self.start_time = datetime(syear, smon, sday, shour)
        self.end_time = datetime(eyear, emon, eday, ehour)

        # Current time
        self.current_time = self.start_time

        # Time step counter
        self.kstep = 0

        # Calculate total number of steps
        total_seconds = (self.end_time - self.start_time).total_seconds()
        self.nsteps = int(total_seconds / dt)

    def time_next(self):
        """
        Advance to next time step
        Similar to CMF_TIME_NEXT in Fortran
        """
        self.kstep += 1
        delta = timedelta(seconds=self.dt)
        self.current_time += delta

        # Skip Feb 29 if lleapyr is False
        if not self.lleapyr and self.current_time.month == 2 and self.current_time.day == 29:
            self.current_time += timedelta(days=1)

    def get_date(self):
        """Get current date in YYYYMMDD format"""
        return int(self.current_time.strftime('%Y%m%d'))

    def get_time(self):
        """Get current time in HHMM format"""
        return int(self.current_time.strftime('%H%M'))

    def get_year(self):
        """Get current year"""
        return self.current_time.year

    def get_month(self):
        """Get current month"""
        return self.current_time.month

    def get_day(self):
        """Get current day"""
        return self.current_time.day

    def get_hour(self):
        """Get current hour"""
        return self.current_time.hour

    def get_minute(self):
        """Get current minute"""
        return self.current_time.minute

    def is_output_time(self, ifrq_out):
        """
        Check if current time is output time

        Parameters:
        -----------
        ifrq_out : int
            Output frequency in hours
        """
        if ifrq_out <= 0:
            return False
        return (self.current_time.hour % ifrq_out == 0 and
                self.current_time.minute == 0)

    def is_restart_time(self, ifrq_rst, unit='hour'):
        """
        Check if current time is restart output time

        Parameters:
        -----------
        ifrq_rst : int
            Restart output frequency (0 = only at end)
        unit : str
            Frequency unit: 'hour', 'day', 'month', 'year'
        """
        if ifrq_rst == 0:
            # Only at the end of simulation
            return self.current_time >= self.end_time

        # Check based on unit
        if unit == 'hour':
            return (self.current_time.hour % ifrq_rst == 0 and
                    self.current_time.minute == 0)
        elif unit == 'day':
            # Daily frequency: check if current day is a multiple of ifrq_rst
            # and at midnight (hour 0)
            return (self.current_time.hour == 0 and
                    self.current_time.minute == 0 and
                    self.current_time.day % ifrq_rst == 0)
        elif unit == 'month':
            # Monthly frequency: check if current month is a multiple of ifrq_rst
            # and on the first day at midnight
            return (self.current_time.day == 1 and
                    self.current_time.hour == 0 and
                    self.current_time.minute == 0 and
                    self.current_time.month % ifrq_rst == 0)
        elif unit == 'year':
            # Yearly frequency: check if current year is a multiple of ifrq_rst
            # and on Jan 1 at midnight
            return (self.current_time.month == 1 and
                    self.current_time.day == 1 and
                    self.current_time.hour == 0 and
                    self.current_time.minute == 0 and
                    (self.current_time.year - self.start_time.year) % ifrq_rst == 0)
        else:
            # Default to hourly if unit not recognized
            return (self.current_time.hour % ifrq_rst == 0 and
                    self.current_time.minute == 0)

    def is_finished(self):
        """Check if simulation is finished"""
        return self.current_time >= self.end_time

    def get_progress(self):
        """Get simulation progress as percentage"""
        elapsed = (self.current_time - self.start_time).total_seconds()
        total = (self.end_time - self.start_time).total_seconds()
        return 100.0 * elapsed / total if total > 0 else 0.0

    def __str__(self):
        """String representation"""
        return f"{self.current_time.strftime('%Y-%m-%d %H:%M:%S')} (step {self.kstep}/{self.nsteps})"
